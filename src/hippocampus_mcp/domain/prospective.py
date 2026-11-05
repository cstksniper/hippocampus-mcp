# -*- coding: utf-8 -*-
"""O6：前瞻记忆情境触发（domain 纯函数，M2 纪律）。

生物对应：线索触发的前瞻记忆（prospective memory）——「到点了要做某事」
之外，**遇到相关情境时意图自动浮上来**。

现状缺口：``remember_intent`` 登记的待办只在 session_start 的
``block_pending_intents`` 带回（开场一次性、只按列表顺序截 5）；会话中途
``get_memory_context`` 碰到相关话题时，相关待办混在「前 5 条 open」里，
排在后面的永远不会被想起。

机制（全部纯函数，触发集**在登记时派生**、装配时只做交集判定）：

* 登记侧（``derive_intent_triggers``）：从待办文本抽实体名（与检索同一
  套抽取器，词库命中优先）+ 实词集合——``tokens_of``（中文 bigram /
  英文词，去停用词）**再补中文 trigram**。补 trigram 的原因（实测）：
  ``tokens_of`` 的 bigram 是**不重叠**切分，措辞序一变就全灭——
  「写第三章初稿」切出 写第/三章/初稿，「第三章写到哪了」切出
  第三/章写/到哪，明显同题却零交集；trigram「第三章」两侧都在，
  且三字串足够特异、不引入扰民。存进 intent 条目的
  ``trigger_entities`` / ``trigger_topic``——随 brain_intents.json 落盘，
  多会话合并层原样保留额外键。
* 装配侧（``prospective_block2``）：查询的实体名/实词（含 trigram，
  与登记侧同口径）与活跃待办的触发集**相交** → 该待办在 block_2 置前
  回带，行尾加「⚡与当前话题相关」标记；未触发的待办保持原有顺序殿后。
  **cap 5 不膨胀**（block_2 预算不变），触发详情经返回体
  ``prospective_intents`` 键自曝（何时触发、凭什么命中）。

不扰民保证（验收的负对照）：查询为空、待办无触发集、交集为空——三种
情况下 ``prospective_block2`` 的行与原实现**逐字节一致**，且不产生
``prospective_intents`` 键（零痕迹，U5/D4 同款纪律）。

完成语义不打架（验收第三条）：本模块只重排 block_2 的**展示顺序**、
加展示标记，不改 intent 的 ``text`` / ``status``——complete_intent 按
关键词匹配 text、resolve_intent 按 status 过滤，两个通道看到的都是
原条目；触发不产生任何自动完成。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "INTENT_BLOCK_CAP", "TRIGGER_SUFFIX",
    "IntentTriggers", "ProspectivePlan",
    "derive_intent_triggers", "prospective_block2", "zh_trigrams",
]

#: block_2 待办条数上限——与 domain/context_slots.py 的既有 ``[:5]``
#: 一致（抄现状，不新增预算；改上限要两边同步，注释互指）。
INTENT_BLOCK_CAP = 5

#: 触发待办在 block_2 行尾的标记。放在行内让文本槽直接可读；结构化详情
#: 走 ``prospective_intents`` 键，两者同源。
TRIGGER_SUFFIX = "（⚡与当前话题相关）"

_ZH_RUN = re.compile(r"[\u4e00-\u9fa5]{2,}")


def zh_trigrams(text: str) -> set[str]:
    """中文 trigram（连续汉字串内的三字滑窗；不足三字的串不出窗）。

    本模块自备而不改 ``tokens_of``：那是写读两侧"同一事实"判据的分词
    真相源（golden 44 例钉着），触发集没有资格改它的口径——只做**增量**
    （并集），bigram / 英文词口径原样保留。
    """
    out: set[str] = set()
    for run in _ZH_RUN.findall(text or ""):
        for i in range(len(run) - 2):
            out.add(run[i:i + 3])
    return out


@dataclass(frozen=True)
class IntentTriggers:
    """一条待办的触发条件（登记时派生一次，随层落盘）。"""
    entities: tuple[str, ...] = ()      # 实体名（抽取器口径，保留原样）
    topic: frozenset[str] = field(default_factory=frozenset)  # 实词+trigram


def derive_intent_triggers(
    text: str,
    *,
    extract_entities: Callable[[str], Iterable[str]],
    tokens_of: Callable[[str], set[str]],
) -> IntentTriggers:
    """从待办文本派生触发条件（登记侧，纯函数——抽取器注入）。

    实体名保留抽取器原样（大小写/归一由抽取器负责），匹配时两侧 lower；
    实词集合 = ``tokens_of``（检索同一套）∪ ``zh_trigrams``（措辞序
    脆弱性的补丁，见模块 docstring）。
    """
    ents = tuple(dict.fromkeys(
        e for e in (extract_entities(text or "") or []) if e))
    toks = frozenset(tokens_of(text or "")) | frozenset(zh_trigrams(text or ""))
    return IntentTriggers(entities=ents, topic=toks)


@dataclass(frozen=True)
class ProspectivePlan:
    """block_2 的重排结果（装配侧产物）。

    Attributes:
        lines: block_2 的行列表（格式与原实现一致 ``[status] text``，
            触发的行尾带 ``TRIGGER_SUFFIX``）。未触发时与原实现逐字节一致。
        triggered: 触发详情（``[{id, text, matched_on}]``）——空列表 =
            本次没有触发（调用方据此决定是否加 ``prospective_intents`` 键）。
    """
    lines: list[str] = field(default_factory=list)
    triggered: list[dict[str, Any]] = field(default_factory=list)


def _matched_on(
    it: dict[str, Any],
    *,
    query_entities: set[str],
    query_tokens: frozenset[str],
) -> list[str]:
    """一条待办的命中原因清单（实体/实词两路，报告用）。"""
    reasons: list[str] = []
    hits_e = sorted(set(e.lower() for e in (it.get("trigger_entities") or []))
                    & query_entities)
    reasons.extend("entity:" + e for e in hits_e)
    hits_t = sorted(set(it.get("trigger_topic") or ()) & set(query_tokens))
    reasons.extend("topic:" + t for t in hits_t)
    return reasons


def prospective_block2(
    intents: Sequence[Any],
    *,
    query: str,
    extract_entities: Callable[[str], Iterable[str]],
    tokens_of: Callable[[str], set[str]],
    cap: int = INTENT_BLOCK_CAP,
) -> ProspectivePlan:
    """装配 block_2 待办行：触发置前回带，其余保持原序（纯函数）。

    Args:
        intents: brain 的待办序列（原样只读；触发集字段缺失 = 老数据，
            永不触发——行为与无触发集一致）。
        query: 当前查询；空串 = 走原行为（触发机制只在有查询语境时生效）。
        extract_entities / tokens_of: 查询侧的实体抽取与分词（注入）。
        cap: 条数上限（默认 5，与原 ``[:5]`` 一致；先排序后裁——
            触发的待办不会被无关注办挤出 cap）。

    Returns:
        :class:`ProspectivePlan`。``triggered`` 为空时 ``lines`` 与
        context_slots 原实现的 ``[f"[{i['status']}] {i['text']}" ...][:cap]``
        逐字节一致（负对照的锚点）。
    """
    open_intents = [i for i in intents
                    if isinstance(i, dict) and i.get("status") == "open"]

    def _render(items: list[dict[str, Any]], marked: set[int]) -> list[str]:
        lines = [f"[{i['status']}] {i['text']}"
                 + (TRIGGER_SUFFIX if idx in marked else "")
                 for idx, i in enumerate(items)]
        cut = lines[:max(int(cap), 0)]
        return cut or ["（无待办）"]

    if not (query or "").strip():
        return ProspectivePlan(lines=_render(open_intents, set()), triggered=[])

    q_entities = {e.lower() for e in (extract_entities(query) or []) if e}
    # 查询侧与登记侧同口径：tokens_of（bigram/英文）∪ 中文 trigram。
    q_tokens = frozenset(tokens_of(query or "")) | frozenset(
        zh_trigrams(query or ""))
    if not q_entities and not q_tokens:
        return ProspectivePlan(lines=_render(open_intents, set()), triggered=[])

    triggered_items: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for it in open_intents:
        reasons = _matched_on(
            it, query_entities=q_entities, query_tokens=q_tokens)
        if reasons:
            triggered_items.append(it)
            details.append({
                "id": it.get("id", ""),
                "text": it.get("text", ""),
                "matched_on": reasons,
            })
        else:
            rest.append(it)
    if not triggered_items:
        return ProspectivePlan(lines=_render(open_intents, set()), triggered=[])

    ordered = triggered_items + rest
    marked = set(range(len(triggered_items)))
    return ProspectivePlan(lines=_render(ordered, marked), triggered=details)
