# -*- coding: utf-8 -*-
"""``brain.core._knowledge_brief_batch`` 的四段逻辑（从 core 抽取）。

**为什么值得搬**：``_knowledge_brief_batch`` 是 ``action="brief_batch"``
的落点，63 行里限额夹取、候选挑选、素材投影、回显组装四段全是纯逻辑，
却长在 self 面上无法单测。搬走后壳只剩「取数 → 注入 → 编排」一条直线。

**依赖方向**：本模块零 import——``ordered`` / ``pending_summary_ids`` /
``representative_ids`` / ``summary_state`` 四个回调与 ``clip`` 常量
（``TOPIC_EVIDENCE_CLIP``）全由壳调用时现取后注入（domain 同层互引
都不行，topic 包住在 domain 但只准依赖 common；沿用
topic_material 的注入先例）。``defaults``（``_topic_defaults()`` 现取）
同属注入面。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``clamp_batch`` 的 ``int(batch or 10)``——batch 为假值（0/None/""）
   时取 **default 10** 而不是夹取下界 1；负值与超上界才真正被夹。
   per_topic 同理（默认 5，夹 [1,30]；batch 夹 [1,50]）。
2. ``pick_candidates`` 的 want 集合 strip 且丢空串；**显式 topic_ids
   绕过 pending 口径**（连 size<3 的碎主题也要能取——"点名就是我
   要这个"）；topic_ids 为 None/空 ⇒ 按 pending_summary_ids 自动挑
   （与 ``knowledge_map`` 的 pending 同一口径）。
3. 显式点名的返回顺序跟 ``ordered(knowledge)`` 的层序走，**不跟
   请求顺序**（cand 是过滤不是重排）。
4. ``build_items`` 的 ``alive`` = member_ids ∩ info（require_vec=False
   全集）——``size`` 报的是 alive 数，不是 member_ids 长度（死成员
   不计入）；content 先空白归一（``" ".join(str(...).split())``）再
   ``[:clip]``；``truncated`` 严格大于（恰好等于 clip 不算截断）。
5. ``summary_state(t, alive)`` 对 alive 成员算归纳状态。
6. ``assemble_report`` 的 11 键固定序（键序即信号）；``pending``
   截前 50；``done = not pending_all``（空列表 ⇒ True，bool()）。
7. 本方法**不检查 layer_enabled**——门禁在调用方
   （``knowledge_topic`` 开头），层关着时这里会返回空 items 的
   ok=True 回显（该路径实际不可达，记录在案）。
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# 段 1：批量参数夹取（旧壳 3243-3247）
# ======================================================================
def clamp_batch(batch: Any, per_topic: Any) -> tuple[int, int]:
    """批量参数归一：``int(x or default)`` 后夹到 [1,50] / [1,30]。

    怪癖复刻：``batch or 10`` 让 0/None/"" 都取 10（不是下界 1）；
    负值与超上界才真正被夹。非数字字符串会抛 ValueError（与原实现同）。
    """
    n = int(batch or 10)
    n = 1 if n < 1 else (50 if n > 50 else n)
    per = int(per_topic or 5)
    per = 1 if per < 1 else (30 if per > 30 else per)
    return n, per


# ======================================================================
# 段 2：候选主题挑选（旧壳 3252-3260）
# ======================================================================
def pick_candidates(
    knowledge: dict[str, Any],
    topic_ids: list[str] | None,
    n: int,
    *,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    pending_summary_ids: Callable[..., list[str]],
    defaults: Any,
) -> list[dict[str, Any]]:
    """候选主题挑选：显式点名 or pending 口径自动挑，再截前 n 个。

    Args:
        knowledge: 主题层字典（``self.knowledge``，壳直传）。
        topic_ids: 显式点名的主题 id 列表；None/空 ⇒ 自动挑。
        n: 批量上限（已由 :func:`clamp_batch` 夹好）。
        ordered: ``topic_mod.ordered``（层序枚举）。
        pending_summary_ids: ``topic_mod.pending_summary_ids``。
        defaults: ``_topic_defaults()`` 现取快照。

    Returns:
        候选主题列表（层序，截前 n 个）。
    """
    if topic_ids:
        want = {str(x).strip() for x in topic_ids if str(x).strip()}
        cand = [t for t in ordered(knowledge)
                if str(t.get("id")) in want]
    else:
        pend = set(pending_summary_ids(knowledge, defaults=defaults))
        cand = [t for t in ordered(knowledge)
                if str(t.get("id")) in pend]
    return cand[:n]


# ======================================================================
# 段 3：素材条目投影（旧壳 3262-3282）
# ======================================================================
def build_items(
    picked: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    info: dict[str, dict[str, Any]],
    *,
    per: int,
    representative_ids: Callable[..., list[str]],
    summary_state: Callable[..., Any],
    clip: int,
) -> list[dict[str, Any]]:
    """把挑中的主题投影成素材条目（六键固定序）。

    Args:
        picked: 候选主题（:func:`pick_candidates` 的输出）。
        by_id: 有向量记忆索引（壳经 ``_knowledge_items()`` 建）。
        info: 存活全集索引（壳经 ``_knowledge_items(require_vec=False)``
            建）——alive 判定只用这个，死成员不计入 size。
        per: 每主题素材条数（已夹取）。
        representative_ids: ``topic_mod.representative_ids``。
        summary_state: ``topic_mod.summary_state``。
        clip: ``TOPIC_EVIDENCE_CLIP`` 常量（壳调用时现取注入）。

    Returns:
        素材条目列表（与 picked 等长，逐主题六键）。
    """
    out_items: list[dict[str, Any]] = []
    for t in picked:
        tid = str(t.get("id"))
        alive = [m for m in (t.get("member_ids") or []) if m in info]
        reps = representative_ids(t, by_id, per)
        members = []
        for mid in reps:
            it = by_id[mid]
            txt = " ".join(str(it.get("content") or "").split())
            members.append({
                "id": mid,
                "content": txt[:clip],
                "truncated": len(txt) > clip,
            })
        out_items.append({
            "topic_id": tid,
            "title": t.get("title") or "(未命名)",
            "size": len(alive),
            "summary": summary_state(t, alive),
            "members": members,
            "shown": len(members),
        })
    return out_items


# ======================================================================
# 段 4：回显体组装（旧壳 3283-3296）
# ======================================================================
def assemble_report(
    *,
    out_items: list[dict[str, Any]],
    picked_count: int,
    per: int,
    pending_all: list[str],
    auto_built: bool,
) -> dict[str, Any]:
    """组装 brief_batch 回显体（11 键固定序，键序即信号）。

    怪癖复刻：``pending`` 截前 50（全量可能很长）；``done = not
    pending_all``（空列表 ⇒ True）；``auto_built`` 壳直通（知识地图
    自动建标记，本方法不感知建没建）。
    """
    return {
        "ok": True, "enabled": True, "action": "brief_batch",
        "items": out_items,
        "picked": picked_count,
        "per_topic": per,
        "pending_count": len(pending_all),
        "pending": pending_all[:50],
        "done": not pending_all,
        "auto_built": auto_built,
        "note": ("素材按「离质心最近」排序——**不是时间序**，最早那批往往"
                 "正是后来被推翻的。读完用 action=summarize_batch 写回，"
                 "evidence 只填上面列出的 id。"),
    }


# ======================================================================
# 编排入口（壳的唯一调用点）
# ======================================================================
def collect(
    topic_ids: list[str] | None,
    *,
    batch: Any,
    per_topic: Any,
    auto_built: bool,
    knowledge: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    info: dict[str, dict[str, Any]],
    defaults: Any,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    pending_summary_ids: Callable[..., list[str]],
    representative_ids: Callable[..., list[str]],
    summary_state: Callable[..., Any],
    clip: int,
) -> dict[str, Any]:
    """brief_batch 全流程：夹取 → 挑选 → 投影 → 回显（顺序有语义）。

    与原实现的调用次序逐行对齐：clamp 在最前；by_id/info 由壳在建
    本调用前取好（两次 ``_knowledge_items`` 的顺序也不变）；候选
    pend 集合在投影前算一次；回显用的 pending_all 在投影后算一次
    （本路径只读，两次等价，但结构保持两调用）。
    """
    n, per = clamp_batch(batch, per_topic)
    picked = pick_candidates(knowledge, topic_ids, n,
                             ordered=ordered,
                             pending_summary_ids=pending_summary_ids,
                             defaults=defaults)
    out_items = build_items(picked, by_id, info, per=per,
                            representative_ids=representative_ids,
                            summary_state=summary_state, clip=clip)
    pending_all = pending_summary_ids(knowledge, defaults=defaults)
    return assemble_report(out_items=out_items, picked_count=len(picked),
                           per=per, pending_all=pending_all,
                           auto_built=auto_built)
