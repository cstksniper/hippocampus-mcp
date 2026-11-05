"""矛盾检测：库存记录上的查询组合。

从 ``brain/core.py`` 提取。四个入口一一对应 core 的四个薄包装：

- :func:`ids_matching_tags`  ← ``_ids_matching_tags``（标签白名单）
- :func:`contradiction_buckets` ← ``_contradiction_buckets``（受信任实体分桶）
- :func:`find_contradictions` ← ``_find_contradictions``（候选对主查询）
- :func:`pool_size` ← ``_contradiction_pool_size``（候选分母）

落点理由：这些是**库存记录上的查询组合**——复用 store 的
``pairwise_similarities`` 能力与 ``MemoryState`` 状态机。domain 受依赖
矩阵约束（只准依赖 common）碰不到 ``store.memory.MemoryState``，故与
``tag_ops`` / ``ledger`` 两个 I/O 提取同落 store 侧。

纪律同族：本模块**不 import vector_store**——store 实例由调用方
（brain 包装层传 ``self.manager.store``）注入；可信实体 id  likewise
由包装层现取 ``self._trusted_entity_ids()`` 后传入，本模块不碰 brain。
"""

from __future__ import annotations

from typing import Any

from ..common.constants import (
    CONTRADICTION_BUCKET_MAX,
    CONTRADICTION_REPORT_MAX,
    CONTRADICTION_SIM_CEIL,
    SUPERSEDE_SIM,
)
from .memory import MemoryState

__all__ = [
    "contradiction_buckets",
    "find_contradictions",
    "ids_matching_tags",
    "pool_size",
]


def ids_matching_tags(
    records: Any, filter_tags: dict[str, Any] | None,
) -> set[str] | None:
    """按结构化标签筛出记忆 id 白名单。

    语义：**所有**键都必须匹配（AND）；单个键的值可以是标量或列表
    （列表内取 OR）。比较前统一 ``str()`` 归一——标签经 JSON 往返后
    可能是 ``43`` 也可能是 ``"43"``，不做归一会让过滤静默失效。

    Args:
        records: 候选记录可迭代对象（调用方传 ``store.iter_records()``）。
        filter_tags: 过滤条件。

    Returns:
        None 表示"无过滤"（调用方不要把它当成空集）。
    """
    if not filter_tags:
        return None
    wanted: list[tuple[str, set[str]]] = []
    for key, val in filter_tags.items():
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        wanted.append((str(key), {str(v) for v in vals if v is not None}))

    out: set[str] = set()
    for rec in records:
        tags = rec.tags or {}
        if not tags:
            continue
        hit = True
        for key, vals in wanted:
            if key not in tags or str(tags.get(key)) not in vals:
                hit = False
                break
        if hit:
            out.add(rec.id)
    return out


def contradiction_buckets(
    records: Any, trusted_ids: Any,
) -> dict[str, list[str]]:
    """按受信任实体把活跃记忆分桶（矛盾检测的输入）。

    只认受信任来源的实体：不可信实体（尤其中文高频短语）会把大量互不相关
    的记忆塞进同一个桶，既产生成片的假"矛盾"，又因桶内两两比对是 O(k²)
    而拖慢巩固。抽成独立函数是为了让这条过滤可被单测直接验证。

    Args:
        records: 候选记录可迭代对象。
        trusted_ids: 受信任实体 id 集合（调用方现取，本模块不碰 brain）。
    """
    by_ent: dict[str, list[str]] = {}
    for rec in records:
        if rec.state != MemoryState.ACTIVE.value:
            continue
        for eid in rec.entity_ids or []:
            if eid in trusted_ids:
                by_ent.setdefault(eid, []).append(rec.id)
    return by_ent


def find_contradictions(
    store: Any,
    limit: int = CONTRADICTION_REPORT_MAX,
    source: str = "",
    records: Any = None,
    filter_tags: dict[str, Any] | None = None,
    trusted_ids: Any = None,
) -> list[dict[str, Any]]:
    """筛出"同实体 + 高相似 + 内容不同"的活跃记忆对。

    **只读**：不修改任何记录状态。它只是"提示宿主这里可能有两套口径"，
    裁决权在宿主（只增不删的库里，新旧两版本来就该并存，见 D8）。

    Args:
        store: 存储实例（``pairwise_similarities`` / ``get`` /
            ``iter_records`` 由调用方注入）。
        limit: 最多返回多少对。
        source: 只查某个来源（文件名 / 会话标识）下的记忆。
        records: 候选集，默认全库存活记录。
        filter_tags: 结构化标签过滤，与 ``recall`` 的
            ``filter_tags`` **同一套语义**（全键 AND、单键多值 OR、
            ``str()`` 归一），复用 :func:`ids_matching_tags`。
            作用＝把"整库口径矛盾"收窄成"**某个域内的**口径矛盾"——
            六代口径混在一个库里时，全局 top-N 会被无关域占满。
        trusted_ids: 受信任实体 id 集合（调用方现取）。

    Returns:
        按相似度降序的候选对，每对带 ``a_id``/``b_id``（可直接交给
        :meth:`update_memory` 作废）与 ``newer``（谁的时间戳更新）。
    """
    recs = list(records if records is not None else store.iter_records())
    recs = [
        r for r in recs
        if r is not None and r.state == MemoryState.ACTIVE.value
        and (not source or (r.source or "") == source)
    ]
    # J3：标签域过滤。与 recall 同源（ids_matching_tags），保证"按域查矛盾"
    # 与"按域检索"看到的是同一批条目——两套实现迟早会算出不一样的白名单。
    # None = 无过滤（**不要**当成空集，那是"什么都不查"）。
    if filter_tags:
        allow = ids_matching_tags(store.iter_records(), filter_tags)
        if allow is None:
            allow = set()
        recs = [r for r in recs if r.id in allow]
    by_ent = contradiction_buckets(recs, trusted_ids)
    bucket_cap = max(CONTRADICTION_BUCKET_MAX, int(limit or 0))
    out: list[dict[str, Any]] = []
    for _eid, ids in by_ent.items():
        if len(ids) < 2:
            continue
        for ia, ib, sim in store.pairwise_similarities(
            ids, SUPERSEDE_SIM, CONTRADICTION_SIM_CEIL, limit=bucket_cap,
        ):
            a = store.get(ia)
            b = store.get(ib)
            if a is None or b is None:
                continue
            if a.content == b.content:      # 内容相同 = 重复，不是矛盾
                continue
            out.append({
                "a_id": ia, "b_id": ib,
                "a": a.content[:60], "b": b.content[:60],
                "sim": round(sim, 3),
                "a_source": a.source or "", "b_source": b.source or "",
                "a_timestamp": a.timestamp, "b_timestamp": b.timestamp,
                "newer": "a" if a.timestamp >= b.timestamp else "b",
            })
    out.sort(key=lambda c: -c["sim"])
    return out[:limit] if limit else out


def pool_size(
    store: Any, source: str = "", filter_tags: dict[str, Any] | None = None,
) -> int:
    """矛盾候选的**分母**：过滤后还剩多少条进得了比对。

    存在的理由：``count: 0`` 有两种截然不同的真相——
      ① 候选池非空、但筛不出矛盾对（＝域内口径一致，好消息）
      ② **候选池本身就是空的**（＝过滤条件没命中，和 B2 同型的静默失败）
    不把分母报出来，调用方一律读成 ①，于是"域内没问题"这个结论是假的。

    Args:
        store: 存储实例（``iter_records`` 由调用方注入）。
        source: 只查某个来源（文件名 / 会话标识）下的记忆。
        filter_tags: 结构化标签过滤。
    """
    recs = [
        r for r in store.iter_records()
        if r is not None and r.state == MemoryState.ACTIVE.value
        and (not source or (r.source or "") == source)
    ]
    if filter_tags:
        allow = ids_matching_tags(store.iter_records(), filter_tags)
        if allow is None:
            allow = set()
        recs = [r for r in recs if r.id in allow]
    return len(recs)
