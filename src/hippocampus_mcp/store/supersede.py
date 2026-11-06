"""写侧取代检测：新记忆含更新标记 + 与旧记忆高度相似 + 共享实词 → 取代最相似的一条。

从 ``brain/core.py`` 的 ``_detect_supersede`` 提取。

三重条件缺一不可：更新标记 + 相似度阈值 + （共同实体 或 高实词重叠），
防止「花生过敏」这类无关长期事实被「现在养猫」误标为已取代。

落点理由：**库存记录上的查询组合 + 状态机写回**——与
``contradictions.py`` 同族（复用 ``store`` 的 ``get`` / ``iter_records`` /
``update`` 与 ``MemoryState``）。domain 受依赖矩阵约束碰不到
``store.memory.MemoryState``，故落 store 侧。

纪律同上：本模块**不 import vector_store / brain**——store 实例由调用
方注入（brain 包装层传 ``self.manager.store``），向量查询 ``vector_of`` 与
判据 ``is_restatement`` 同样是注入的**可调用对象**（不是调用结果）：

* ``vector_of`` 只在「含更新标记」的新记忆上才问，标记过滤在前——不给
  没有更新标记的写入付嵌入查询的代价；
* ``is_restatement`` 不 import 是为了保住测试对 ``brain._is_restatement``
  的 monkeypatch 面，同时也让读侧（``domain.conflict_annotate``）与写侧
  共用同一份判据（各写一套阈值迟早自相矛盾）。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..common.constants import UPDATE_MARKERS
from .memory import MemoryState

__all__ = ["detect_supersede"]


def detect_supersede(
    new_memory_ids: list[str],
    ts: float,
    store: Any,
    vector_of: Callable[[Any], Any | None],
    is_restatement: Callable[..., bool],
    update_markers: tuple[str, ...] = UPDATE_MARKERS,
) -> list[dict[str, str]]:
    """在 ``ts`` 之前入库的 active 记忆里，找被 ``new_memory_ids`` 取代的那些。

    Args:
        new_memory_ids: 本次新写入的记忆 id。
        ts: 本次写入的时间戳；**``>= ts`` 的候选一律跳过**（只取代更旧的，
            同批写入之间互不取代）。
        store: 库存记录存储（需 ``get`` / ``iter_records`` / ``update``）。
        vector_of: 记录 → 向量（``None`` 表示这条没算过向量，跳过）。
        is_restatement: 同一事实判据，签名见
            ``domain.policies.is_restatement``。
        update_markers: 更新标记词表。

    Returns:
        每个取代一条：``old_id`` / ``new_id`` / ``sim`` / ``old`` / ``new``
        （后两个是内容前 60 字，仅供回显）。

    副作用：把被取代的那条改成 ``state=superseded`` 并挂 ``superseded_by``
    ——这是本函数**唯一**的写操作，找不到就不写。

    预存行为（与原实现逐条对齐，不要「顺手修好」）：

    * 只取代**最相似**的那一条（``sim > best_sim`` 严格大于 ⇒ 同分时
      先遇到的胜出），避免误伤无关记忆；
    * 新记忆自己（``rec.id == nid``）、非 active、``timestamp >= ts``
      的一律不作候选；
    * 缺向量的新记忆**整体跳过**（不进内层循环）；缺向量的候选只跳过该候选。
    """
    out: list[dict[str, str]] = []
    for nid in new_memory_ids:
        new_rec = store.get(nid)
        if new_rec is None:
            continue
        if not any(m in new_rec.content.lower() for m in update_markers):
            continue

        q = vector_of(new_rec)
        if q is None:
            continue
        new_ents = set(new_rec.entity_ids or [])
        best, best_sim = None, 0.0
        for rec in store.iter_records():
            if rec.id == nid or rec.state != MemoryState.ACTIVE.value:
                continue
            if rec.timestamp >= ts:
                continue
            v = vector_of(rec)
            if v is None:
                continue
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1e-12
            sim = float(q @ v / denom)
            # 判据统一走 is_restatement（读侧的并存冲突标注用同一个），
            # 两边各写一套阈值迟早会自相矛盾。
            if sim > best_sim and is_restatement(
                sim, new_rec.content, rec.content, new_ents, rec.entity_ids
            ):
                best, best_sim = rec, sim

        if best is not None:
            store.update(
                best.id,
                state=MemoryState.SUPERSEDED.value,
                superseded_by=nid,
            )
            out.append({
                "old_id": best.id, "new_id": nid, "sim": round(best_sim, 3),
                "old": best.content[:60], "new": new_rec.content[:60],
            })
    return out
