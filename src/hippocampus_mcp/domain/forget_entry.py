# -*- coding: utf-8 -*-
"""``brain.core.forget`` 的墓碑与受影响键前置纯段（从 core 抽取）。

**背景**：forget 的级联清理辅助已搬进 forget_cleanup，
forget_by_query 已搬进 forget_flow；``forget`` 本体的前置段还剩两块纯
计算——墓碑 dict 构建（含 ``deleted_at`` 时钟注入）与受影响实体键计算
（扫 entities 倒排）。它们决定「删了什么、为什么、牵连谁」，此前没有
独立单测钉住（墓碑键形 / 快照拷贝 / 时钟恰调一次 / 倒排扫描序），回归
只能靠 forget 的端到端套件兜底。

**依赖方向**：本模块只 import 标准库（连 common 都不用——时钟由壳
调用时现取后注入；entities 倒排表按值传入，本模块不碰 brain 状态）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``build_tombstone`` 的 reason 兜底：``reason or ""``——None 与空串
   同形（墓碑里 reason 恒为字符串）。
2. ``build_tombstone`` 的快照拷贝：``entity_ids`` / ``source_ids`` 走
   ``list(x or [])``——墓碑持**副本**，原记录（及其列表）随后被删，
   共享可变对象会让墓碑跟着变；None / 空列表同形为 ``[]``。
3. ``build_tombstone`` 的时钟恰调一次：``deleted_at`` 只在构建时取
   ``now()`` 一次（其余字段全部来自被删记录的既有值）。
4. ``compute_affected_keys`` 的扫描序：按 ``entities`` 的 dict 插入序
   产出（回显 ``entities_touched`` 的顺序即此序，调用方可复现）；
   ``memory_ids`` 缺键 / None / 空列表的实体**不命中**（``or []`` 兜底），
   entities 本体只读不动。
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# 段 1：墓碑 dict 构建（旧壳 2105-2115 行；append 状态写留壳）
# ======================================================================
def build_tombstone(
    *,
    memory_id: str,
    content: str,
    reason: str | None,
    timestamp: float,
    state: str,
    entity_ids: list[str] | None,
    source_ids: list[str] | None,
    now: Callable[[], float],
) -> dict[str, Any]:
    """构建墓碑 dict（八键，键序即 ``tombstones`` 条目形状）。

    墓碑先记是审计前提：原文保住，删除后仍可追溯「删了什么、为什么」。

    Args:
        memory_id: 被删记忆的 id。
        content / timestamp / state: 被删记录的既有值，原样透传。
        reason: 删除原因；``or ""`` 兜底（怪癖 1）。
        entity_ids / source_ids: 被删记录的引用清单；取**快照副本**
            （怪癖 2），None 兜底空列表。
        now: ``brain_core._now`` 本体（壳调用时现取；parity 冻结打在
            它上，两侧 deleted_at 同值）。恰调一次（怪癖 3）。

    Returns:
        八键 dict。append 由壳做——本函数只组装，不碰状态。
    """
    return {
        "memory_id": memory_id,
        "content": content,
        "reason": reason or "",
        "timestamp": timestamp,
        "state": state,
        "entity_ids": list(entity_ids or []),
        "source_ids": list(source_ids or []),
        "deleted_at": now(),
    }


# ======================================================================
# 段 2：受影响实体键计算（旧壳 2117-2120 行；纯读 entities 倒排）
# ======================================================================
def compute_affected_keys(
    entities: dict[str, dict[str, Any]],
    memory_id: str,
) -> list[str]:
    """算出 forget 后需要重算统计的实体键（扫 entities 倒排）。

    Args:
        entities: ``brain.entities`` 倒排表（键 = 实体小写键，值含
            ``memory_ids`` 引用清单）。**只读**——重算/清空是壳里
            后续级联的事。
        memory_id: 被删记忆的 id。

    Returns:
        引用了该记忆的实体键列表，按 entities 的插入序（怪癖 4）。
    """
    return [
        k for k, e in entities.items()
        if memory_id in (e.get("memory_ids") or [])
    ]
