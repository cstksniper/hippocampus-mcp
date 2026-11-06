"""forget 级联清理中的两段纯逻辑：画像层摘除 + L0 孤儿来源标记。

零 import：只做 dict/list 操作，画像条目与 L0 原始记录都按鸭子类型
（``.get("text")`` / ``.get("id")``）使用。forget 的大脑级编排（墓碑、
物理删除、实体重算、主题层、被取代者恢复）留在 brain 壳里，这里只承载
「给定状态算出清理结果」的可单测部分。

两段均与原实现逐字同构，包括两处刻意保留的怪癖：
  * ``clean_profile``：整槽被摘空时删除该 slot（空 items 的 slot 也会被删）；
  * ``mark_orphan_raws``：raw_log 内同一 id 出现多次、或 source_ids 内同一
    rid 出现多次时，marked 会重复 append（for-if 结构无 break/skip）。
"""
from __future__ import annotations

from typing import Any


def clean_profile(
    profile: dict[str, list[dict[str, Any]]],
    *,
    memory_id: str,
    content: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """摘除画像中由该记忆派生的条目，返回 ``(new_profile, dropped)``。

    命中条件（二者居其一）：
      * 文本副本：画像 ``text`` 等于记忆全文或其前 120 字截断
        （画像 text 本就是记忆内容的截断副本）；
      * 单条证据：``evidence`` 有且仅有一条且精确指向该记忆。
        旧库的 evidence 是整实体的 id 列表（粒度过粗），多证据时只认
        文本匹配，避免误删同实体的其他条目。

    整槽摘空的 slot 从 new_profile 中消失（复刻原实现的 ``del``）。
    dropped 每项记 ``{"slot": ..., "text": ...}``，text 取**未 strip**
    的原始值。
    """
    new_profile: dict[str, list[dict[str, Any]]] = {}
    dropped: list[dict[str, Any]] = []
    content_full = content.strip()
    content_cut = content[:120].strip()
    for slot, items in profile.items():
        kept: list[dict[str, Any]] = []
        for it in items:
            ev = it.get("evidence") or []
            txt = (it.get("text") or "").strip()
            text_hit = txt in (content_full, content_cut)
            evidence_hit = memory_id in ev and len(ev) == 1
            if text_hit or evidence_hit:
                dropped.append({"slot": slot, "text": it.get("text")})
                continue
            kept.append(it)
        if kept:
            new_profile[slot] = kept
    return new_profile, dropped


def mark_orphan_raws(
    raw_log: list[dict[str, Any]],
    *,
    source_ids: list[str] | None,
    alive_refs: set[str],
    reason: str,
) -> list[str]:
    """把不再被任何存活记忆引用的 L0 来源标记 forgotten，返回 marked 列表。

    ``alive_refs`` 是全部存活记忆 source_ids 的并集（调用方构建）。
    标记**就地**写进 raw dict（``forgotten`` / ``forgotten_reason``），
    与原实现一致——壳不需要重赋值 raw_log。

    怪癖复刻：同一 id 在 raw_log 中多次出现、或 source_ids 中重复时，
    marked 会为每次匹配 append 一次。
    """
    marked: list[str] = []
    for rid in source_ids or []:
        if rid in alive_refs:
            continue
        for raw in raw_log:
            if raw.get("id") == rid:
                raw["forgotten"] = True
                raw["forgotten_reason"] = reason or ""
                marked.append(rid)
    return marked
