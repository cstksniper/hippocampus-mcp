# -*- coding: utf-8 -*-
"""5 槽位上下文装配（借鉴 bigeye AttentionFocus）——编排层。

产物是一块可以直接拼进 system prompt 的文本（``text``）+ 结构化槽位
（``blocks``）+ 命中明细（``hits``）：

    block_1 当前查询直接命中  block_2 未决问题/待办
    block_3 时间线位置（最近会话摘要）  block_4 相关实体状态
    block_5 上次结论 / 未完结话题

block_4 有**自己的预算**（``block4_*`` 四个注入值：最多
``block4_max_entities`` 个实体、每个至多 ``block4_max_facts_per_entity``
条事实、全块 ≤ ``block4_max_chars`` 字符），且候选以**查询词实体**优先、
命中实体的前若干条作兜底——**不随 ``top_k`` 膨胀**（zcode D2 主体）。
预算不与全局 ``token_budget`` 抢额度。

``item_max_chars`` 传 0 表示不截断（账目/条文类长句的数字尾巴常被 200
切掉，用户反馈）；它只影响拼进 prompt 的 text，返回的
``hits`` 始终是完整内容。``filter_tags`` 给定时 block_4 **不做查询词
实体扩展**，只在命中集内发生（命中为空则无实体可扩）——按章写作时不
希望别章的实体状态混进上下文。

落点说明：本体是纯编排（分流 + 候选收集 + 预算装填），六个
brain 入口（recall / pinned_brief / trusted_entity_ids / extract_entities /
entity_facts / render_blocks）以**可调用对象**注入，``BLOCK4_*`` 四个上限
由调用方现取传入（住在 common/constants，patch 调用方命名空间即生效）。
四个数据面（entities / intents / summaries / open_topics）是零成本属性
读，原样传值——原实现里它们也都是在一次调用内顺序读，调用期间无写入方
（recall / entity_facts 对它们均只读）。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


def memory_context(
    *,
    query: str = "",
    top_k: int = 5,
    token_budget: int = 1200,
    item_max_chars: int = 200,
    filter_tags: dict[str, Any] | None = None,
    recall: Callable[..., list[dict[str, Any]]],
    pinned_brief: Callable[..., list[dict[str, Any]]],
    trusted_entity_ids: Callable[[], set[str]],
    entities: Mapping[str, Any],
    extract_entities: Callable[[str], list[str]],
    entity_facts: Callable[..., list[str]],
    intents: Sequence[Any],
    summaries: Sequence[Any],
    open_topics: Sequence[Any],
    render_blocks: Callable[[dict[str, list[str]], int, int], str],
    block4_max_entities: int,
    block4_max_facts_per_entity: int,
    block4_max_chars: int,
    block4_hit_entity_topn: int,
) -> dict[str, Any]:
    """装配 5 槽位上下文，可直接拼进 system prompt。

    Args:
        query: 查询语；为空时 block_1 走钉住简报（``pinned_brief``）。
        top_k: 召回条数，透传给 recall（pinned_brief 路径同值）。
        token_budget: 全块 token 预算（粗粒度：中文约 2 字符/token）。
        item_max_chars: 单条文本在上下文里的截断长度，默认 200；
            **传 0 表示不截断**。只影响 ``text``，``hits`` 始终完整。
        filter_tags: 结构化标签过滤，透传给 block_1 的命中检索；
            给定时 block_4 不做查询词实体扩展。
        recall: 召回入口，按 ``recall(query, top_k=..., filter_tags=...)``
            形态调用。
        pinned_brief: 钉住简报入口，按 ``pinned_brief(top_k)`` 形态调用。
        trusted_entity_ids: 受信任实体 id 集合，调用时现取。
        entities: 实体表 ``{小写名: 实体 dict}``（原样传值，调用期内只读）。
        extract_entities: 实体名抽取，按 ``extract_entities(query)`` 调用。
        entity_facts: 实体事实，按
            ``entity_facts(ent["key"], limit=...)`` 形态调用。
        intents / summaries / open_topics: 待办 / 会话摘要 / 未完结话题
            三个时间序列（原样传值，调用期内只读）。
        render_blocks: 块渲染 + 预算装填，按
            ``render_blocks(blocks, token_budget, item_max_chars)`` 调用。
        block4_max_entities / block4_max_facts_per_entity /
            block4_max_chars / block4_hit_entity_topn: block_4 的四道
            独立预算（调用方现取注入）。
    """
    hits = (
        recall(query, top_k=top_k, filter_tags=filter_tags)
        if query else pinned_brief(top_k)
    )

    # block_4：相关实体状态。**已收窄**——
    # 老实现是"从**命中记忆**的实体出发"，于是 top_k 越大牵出的实体越多，
    # 该块退化成噪声墙。现在候选顺序为：
    #   ① 查询词里的实体（最贴题）；
    #   ② 命中记忆的实体，且只取前 block4_hit_entity_topn 条命中（查询没提实体时兜底）。
    # 再受 block4_* 独立预算约束——不与全局 token_budget 抢额度。
    # **只认受信任实体**：这块钱直接拼进 system prompt，短语噪声进来就是纯干扰。
    # 两条路径都过这道门（查询词实体亦然），否则等于把 origin 门控悄悄放开一个口子。
    trusted_ids = trusted_entity_ids()
    by_id = {e["id"]: e for e in entities.values()}
    cand_ids: list[str] = []

    def _push(eid: str) -> None:
        if eid and eid in trusted_ids and eid not in cand_ids:
            cand_ids.append(eid)

    # ① 查询词实体。**按标签收窄时不启用**——那时整块上下文只该在命中集内
    #    扩展（把别章的实体状态拖进"按章写作"的 prompt 会破坏隔离）。
    if query and filter_tags is None:
        for name in extract_entities(query):
            ent = entities.get(name.lower())
            if ent:
                _push(ent["id"])
    # ② 命中记忆的实体（只取前若干条命中）
    for h in hits[:block4_hit_entity_topn]:
        for eid in (h.get("entity_ids") or []):
            _push(eid)

    ent_facts: list[str] = []
    seen_facts: set[str] = set()
    used_chars = 0
    for eid in cand_ids[:block4_max_entities]:
        ent = by_id.get(eid)
        if not ent:
            continue
        facts = [f for f in entity_facts(
            ent["key"], limit=block4_max_facts_per_entity) if f not in seen_facts]
        if not facts:
            continue
        line = f"{ent['name']}: " + " / ".join(facts)
        # 独立预算：超了就**丢弃后续实体整条**（绝不切在行中间——块内事实已按
        # 时间倒序，前几条就是最近的）。第一条无条件保留，否则整块可能全空。
        if ent_facts and used_chars + len(line) > block4_max_chars:
            break
        seen_facts.update(facts)
        ent_facts.append(line)
        used_chars += len(line)

    # 并存口径警示：命中里若出现"疑似同一事实的不同口径"，在块尾点明。
    # 不折叠、不删条目——只让读这块钱的人（或模型）知道"下面这两条是一件事
    # 的两版，别混着用"。旧口径可能是被引用的历史事实，也可能才是对的。
    # 每条命中带日期前缀（2026-11-05）：装配式上下文是宿主最常用的入口，
    # 若这里不带时间，库侧派生的 `occurred_at` 就等于白给——Agent 拿到的
    # 仍是一堆无时间片段，"几周前 / 上个月"这类问题照旧答不了。
    # 只在派生字段存在时加（时间未知的条目不编一个假日期）。
    # 成本约 17 字符/条，块预算由 render_blocks 统一裁剪，不会击穿上限。
    hit_lines = [
        (f"[{h['occurred_at']} {h['weekday']}] {h['content']}"
         if h.get("occurred_at") and h.get("weekday") else h["content"])
        for h in hits
    ]
    conflicted = sorted({
        h["conflict_group"] for h in hits if h.get("conflict_group")
    })
    if conflicted:
        hit_lines.append(
            f"⚠️ 其中 {len(conflicted)} 组疑似同一事实的不同口径"
            f"（并列保留、未折叠）：以权威来源为准；同组内 newer=true 的更近。"
        )

    blocks = {
        "block_1_current_hits": hit_lines,
        "block_2_pending": [
            f"[{i['status']}] {i['text']}" for i in intents if i["status"] == "open"
        ][:5] or ["（无待办）"],
        "block_3_timeline": [s["text"] for s in summaries[-2:]] or ["（无历史会话）"],
        "block_4_entity_state": ent_facts or ["（无相关实体状态）"],
        "block_5_open_topics": [t["text"] for t in open_topics[-3:]] or ["（无未完结话题）"],
    }
    text = render_blocks(blocks, token_budget, item_max_chars)
    return {
        "blocks": blocks,
        "text": text,
        "hits": hits,
        "token_estimate": len(text) // 2,
    }
