# -*- coding: utf-8 -*-
"""``brain.core.consolidate`` 的四块逻辑（从 core 抽取）。

**为什么值得搬**：``consolidate`` 是周期巩固入口，119 行里沉寂归档循环
（22 行）、状态计数、矛盾投影、回显体组装四块全是纯逻辑，却长在 self 面
的上下文里无法单测。搬走后 ``consolidate`` 只剩「取数 → 投影 → 编排
副作用 → 组装」一条直线。

**依赖方向**：本模块零 import——``DORMANT_DAYS`` / ``PIN_IMPORTANCE`` /
``MemoryState`` 两常量一枚举全由壳**调用时现取后注入**（前两者的测试
monkeypatch 面在 core 命名空间；MemoryState 住在 store 包，domain 看不见，
沿用 stats_views 的注入先例）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``archive_dormant`` 的 ``last`` 取 ``max(last_recalled_at or 0.0,
   timestamp)``——从没被召回过的记忆按写入时间算（两者皆 0 时 age=now，
   大库冷启动照样能归档）。
2. 干跑候选的 ``days_idle`` 是 ``round(..., 1)``；``max_items`` 在干跑
   分支数**候选条数**、在实归档分支数**已归档条数**，两处 break 语义不同。
3. 干跑命中后 ``continue``——同一条不会再被归档（dry_run 下也不会）。
4. ``assemble_report`` 的 ``scope`` 用 ``if filter_tags`` 真值判定：
   空 dict 与 None 一样报 ``null``（"查不到"≠"没设域"的反面）；
   ``max_items`` 用 ``or None``（0 = 不限）。
5. ``dormant_candidates`` 键**只在干跑时存在**，且排在六个固定键之后
   （键序即信号）。
6. ``project_contradictions`` 只留 ``a`` / ``b`` / ``sim`` 三键——
   底层的 ``a_id`` / ``b_id`` / ``newer`` 等不进巩固回显。

**睡眠批增补**：``select_replay_candidates``——daemon 睡眠批
的**重放候选选择**（纯选择，不落任何状态）。共激活语义见函数 docstring；
写入侧（强度变量自增）是 O1 的接缝，住在 ``brain.core``，本模块只出候选。
"""
from __future__ import annotations

from typing import Any, Callable, Iterable


# ======================================================================
# 块 1：沉寂归档（旧壳 2616-2643）
# ======================================================================
def archive_dormant(
    records: list[Any],
    *,
    now: float,
    dormant_days: int,
    pin_importance: float,
    active_state: str,
    archived_state: str,
    allow: set[str] | None,
    dry_run: bool,
    max_items: int,
    archive: Callable[[str, str], None],
) -> tuple[int, list[dict[str, Any]]]:
    """沉寂归档：长期未被调用且低重要性的活跃记忆（钉住的豁免）。

    Args:
        records: 全库存记录快照（壳一次性取好传入，循环内不再取）。
        now: 时间基准（壳经 ``manager._reference_time or _now()`` 现取）。
        dormant_days: 沉寂天数阈值（``DORMANT_DAYS``）。
        pin_importance: 低重要性上界（``PIN_IMPORTANCE``，严格小于）。
        active_state / archived_state: 状态字面值（``MemoryState.*.value``）。
        allow: 标签域白名单；None = 无过滤（**不当空集**）。
        dry_run: 只登记候选，不调 archive。
        max_items: 上限（0 = 不限）。
        archive: 归档回调 ``(rec_id, archived_state)``——仅实归档时调用，
            干跑一次都不调（惰性注入，调用次数与原实现逐次对齐）。

    Returns:
        ``(dormant_archived_count, dormant_candidates)``。
    """
    dormant = 0
    dormant_candidates: list[dict[str, Any]] = []
    for rec in records:
        if rec.state != active_state or rec.pin:
            continue
        # 标签域收窄（无过滤时 allow is None，不拦）
        if allow is not None and rec.id not in allow:
            continue
        last = max(rec.last_recalled_at or 0.0, rec.timestamp)
        if (now - last) / 86400.0 > dormant_days \
                and rec.importance < pin_importance:
            if dry_run:
                # 干跑：只登记候选，不动状态
                dormant_candidates.append({
                    "id": rec.id,
                    "content": rec.content[:80],
                    "days_idle": round((now - last) / 86400.0, 1),
                    "importance": rec.importance,
                })
                if max_items and len(dormant_candidates) >= max_items:
                    break
                continue
            archive(rec.id, archived_state)
            dormant += 1
            if max_items and dormant >= max_items:
                break
    return dormant, dormant_candidates


# ======================================================================
# 块 2：状态计数（旧壳 2668-2671 / 2681-2684）
# ======================================================================
def count_state(records: Iterable[Any], state: str) -> int:
    """库存里处于指定状态的条数（consolidate 回显的 superseded/active 两用）。"""
    return sum(1 for r in records if r.state == state)


# ======================================================================
# 块 3：矛盾候选投影（旧壳 2662-2666）
# ======================================================================
def project_contradictions(pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """矛盾候选只留 a / b / sim 三键（底层 a_id/b_id/newer 不进回显，怪癖 6）。"""
    return [
        {"a": c["a"], "b": c["b"], "sim": c["sim"]}
        for c in pairs
    ]


# ======================================================================
# 块 4：回显体组装（旧壳 2672-2689）
# ======================================================================
def assemble_report(
    *,
    filter_tags: dict[str, Any] | None,
    max_items: int,
    dry_run: bool,
    superseded_total: int,
    dormant: int,
    contradictions: list[dict[str, Any]],
    active_memories: int,
    dormant_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """组装巩固回显体（键序即信号，见模块 docstring 怪癖 4/5）。

    Args:
        dormant_candidates: 干跑候选清单；None = 非干跑（**不加该键**）。

    Returns:
        六键回显体；干跑时尾附 ``dormant_candidates``。
    """
    report: dict[str, Any] = {
        # 自曝：本次巩固覆盖的范围（null = 全库）
        "scope": dict(filter_tags) if filter_tags else None,
        "max_items": max_items or None,
        "dry_run": bool(dry_run),
        "superseded_total": superseded_total,
        "dormant_archived": dormant,
        "contradiction_candidates": contradictions,
        "active_memories": active_memories,
    }
    if dormant_candidates is not None:
        # 干跑到此为止：只列候选。键排在六个固定键**之后**
        report["dormant_candidates"] = dormant_candidates
    return report


# ======================================================================
# 块 5：睡眠重放的候选选择
# ======================================================================
def select_replay_candidates(
    records: Iterable[Any],
    *,
    now: float,
    since: float,
    active_state: str,
    max_items: int,
    min_cluster: int,
) -> list[dict[str, Any]]:
    """O2 睡眠批：选出「最近会话 L1 簇」作为共激活重放候选。

    生物对应（Buzsáki 2015）：SWR 重放的是**近期**经历的序列，不是全库
    倒带——所以窗口过滤（``since`` < timestamp <= ``now``）是硬前提；
    共激活的载体是**实体共现**（同一条经历里一起出现过的实体，重放时
    一起再激活），不是向量相似——那是 O3 扩散重排的战场，这里刻意不用。

    **本函数是纯选择，零状态变更**：不改记录、不写存储；写入侧（检索强度
    变量自增）住在 ``brain.core`` 的接缝方法里，待合流后接线。

    Args:
        records: 全库存记录（L1 事实层即 store 全部记录；L0 是 raw_log、
            L2/L3 是派生层，均不在 store 里）。
        now: 时间基准（壳经 ``manager._reference_time or _now()`` 现取）。
        since: 窗口起点（水位线；``<=0`` = 从头）。
        active_state: 状态字面值（``MemoryState.ACTIVE.value``）。
        max_items: 参与重放的**记录**总数上限（0 = 不限）。计满即停——
            但**首簇必收**（一个巨簇超限时跳过它会让本次睡眠一无所获，
            重放粒度是簇不是记录，截半个簇没有意义）。
        min_cluster: 成簇下限（共激活至少两个成员；无实体/独占实体的
            记录到不了这个下限，自然落选）。

    Returns:
        确定性排序的簇清单，每簇 ``{"ids": [...], "entities": [...],
        "latest": ts}``；``ids`` 按新→旧排（重放叙事顺序），``entities``
        升序去重，``latest`` = 簇内最新写入时刻。簇间按 ``(-latest,
        最小 id)`` 排——同刻按 id 定序，两次调用结果逐位一致（幂等的
        前提：候选选择本身必须可复现）。
    """
    # 窗口过滤：活跃 + 时间在窗内 + 带实体（无实体的记忆没有共激活载体）
    in_window: list[Any] = []
    for rec in records:
        if rec.state != active_state:
            continue
        ents = rec.entity_ids or []
        if not ents:
            continue
        if not (since < rec.timestamp <= now):
            continue
        in_window.append(rec)
    if not in_window:
        return []

    # 实体 -> 记录倒排，再 union-find：共享任一实体的记录连成一个簇。
    parent: dict[int, int] = {}

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_entity: dict[str, list[int]] = {}
    for i, rec in enumerate(in_window):
        parent[i] = i
        for ent in dict.fromkeys(rec.entity_ids):  # 去重保序
            by_entity.setdefault(ent, []).append(i)
    for members in by_entity.values():
        first = _find(members[0])
        for j in members[1:]:
            root_j = _find(j)
            if root_j != first:
                parent[root_j] = first

    clusters: dict[int, list[int]] = {}
    for i in range(len(in_window)):
        clusters.setdefault(_find(i), []).append(i)

    out: list[dict[str, Any]] = []
    for members in clusters.values():
        if len(members) < min_cluster:
            continue
        recs = sorted(
            (in_window[i] for i in members),
            key=lambda r: (-r.timestamp, r.id),
        )
        ents = sorted({e for r in recs for e in (r.entity_ids or [])})
        out.append({
            "ids": [r.id for r in recs],
            "entities": ents,
            "latest": recs[0].timestamp,
        })
    out.sort(key=lambda c: (-c["latest"], c["ids"][-1]))

    # 记录数限幅：首簇必收（见 Args），此后装不下就停。
    if max_items and max_items > 0:
        picked: list[dict[str, Any]] = []
        total = 0
        for idx, cluster in enumerate(out):
            size = len(cluster["ids"])
            if idx > 0 and total + size > max_items:
                break
            picked.append(cluster)
            total += size
        out = picked
    return out
