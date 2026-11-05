# -*- coding: utf-8 -*-
"""库级「全貌卡 / 状态卡」投影（从 core 提取）。

两样都是**纯派生、不检索**、O(n) 单遍：

* :func:`project_overview` —— 身份+规模 / 时间跨度 / 结构画像 / 热点实体；
* :func:`project_status` —— 未决 / 健康 / 协作 / 活动四组信号。

为什么放 domain/：只做「既有派生层 → 结构化投影」，`recent_ingest` /
``spark_series`` / ``recent_miss_count`` 三个统计口径本就在 ``common.stats``，
零 I/O、零模型、零配置读数。brain 只留薄包装，把派生层、参考时钟、两个
Config 开关、五个上限与五个**惰性回调**在调用时绑好注入。

🔴 **惰性回调是纪律不是偷懒**：``knowledge_stats`` 只在主题层开关开时调、
``contradictions_count`` 只在状态卡矛盾开关开时调——原实现的调用次数就
是这样（万条库上 0.34s 的量，多调一次就是白烧）。所以这里传的是
**可调用对象**而不是"先算好的值"：算好的值会让开关失效时照样付代价。

🔴 **库名走注入**：``display_name`` 住在 ``library/``，而 arch G1 的矩阵是
``domain → 仅 common``（曾因同层互引吃过亏）。所以库名由包装层
算好传进来。

预存口径按「只挪位置不改行为」保留：

* 实体 Top 只收**有实际关联**的（``memory_ids`` 非空）——启发式实体约 84%
  是短语噪声，不过滤 Top 行会被一堆 (0) 占满；
* 时间跨度在没有任何时间戳时是 ``(None, None)``，不是 0；
* ``contradictions`` 在开关关闭时是 ``{"count": None, "capped": False,
  "limit": 0}`` —— **None 不是 0**（0 会被读成"查过，没有矛盾"）；
* L0 悬空与 ``stats()`` 同口径（受影响的 L1 条数），集合先建一次。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..common.stats import recent_ingest, recent_miss_count, spark_series


def project_overview(
    *,
    records: Sequence[Any],
    entities: Mapping[str, Any],
    knowledge: Any,
    daily_counts: Any,
    ref_time: float | None,
    name: str,
    topic_layer_enabled: bool,
    tag_coverage: Callable[[], dict[str, Any]],
    knowledge_stats: Callable[[], dict[str, Any]],
    recent_days: int,
    top_entities: int,
    top_tag_keys: int,
) -> dict[str, Any]:
    """库级全貌投影（设计文档 §3.1）。纯派生、不检索、O(n) 单遍。"""
    total = len(records)
    ts = [r.timestamp for r in records if r.timestamp]
    first_ts, last_ts = (min(ts), max(ts)) if ts else (None, None)
    tag_cov = tag_coverage()
    kn = knowledge_stats() if topic_layer_enabled else {}
    kinds = kn.get("summary_kinds") or {}
    # 只收**有实际关联**的实体：启发式实体约 84% 是短语噪声（见 ENTITY_ORIGIN_*
    # 说明），不过滤的话 Top 行会被一堆 (0) 占满——那既是噪声又是误导。
    top_ents = sorted(
        (e for e in entities.values()
         if len(e.get("memory_ids") or []) > 0),
        key=lambda e: -len(e.get("memory_ids") or []),
    )[:top_entities]
    return {
        "name": name,
        "memories": total,
        "knowledge_topics": len(knowledge),
        "entities": len(entities),
        "first_write_ts": first_ts,
        "last_write_ts": last_ts,
        "recent_ingest": recent_ingest(recent_days, daily_counts, ref_time),
        "recent_days": recent_days,
        "tag_coverage": tag_cov.get("coverage"),
        "tag_top_keys": [k["key"] for k in (tag_cov.get("top_keys") or [])][:top_tag_keys],
        "knowledge_host": int(kinds.get("host") or 0),
        "knowledge_stale": int(kinds.get("host_stale") or 0),
        "top_entities": [
            {"name": e.get("name") or e.get("key") or "?",
             "memories": len(e.get("memory_ids") or [])}
            for e in top_ents
        ],
    }


def project_status(
    *,
    records: Sequence[Any],
    raw_log: Sequence[Any],
    open_topics: Any,
    intents: Sequence[Any],
    miss_log: Sequence[Any],
    daily_counts: Any,
    ref_time: float | None,
    pending_consolidate: bool,
    contradictions_enabled: bool,
    contradictions_count: Callable[[int], dict[str, Any]],
    active_sessions: Callable[[], Any],
    file_registry_summary: Callable[[], Any],
    contradiction_cap: int,
    miss_days: int,
    spark_days: int,
) -> dict[str, Any]:
    """当前状态投影（设计文档 §3.2）。纯派生、不检索。"""
    series = spark_series(spark_days, daily_counts, ref_time)
    contra = (contradictions_count(contradiction_cap)
              if contradictions_enabled
              else {"count": None, "capped": False, "limit": 0})
    # L0 悬空：与 stats() 同口径（受影响的 L1 条数），集合先建一次——
    # 卡片上的数必须和 brain_stats 对得上（单一真相源），不能各算各的。
    raw_ids = {x.get("id") for x in raw_log}
    dangling = sum(
        1 for r in records
        if any(s not in raw_ids for s in (r.source_ids or []))
    )
    return {
        "open_topics": len(open_topics),
        "open_intents": sum(1 for i in intents if i.get("status") == "open"),
        "pending_consolidate": bool(pending_consolidate),
        "l0_dangling": dangling,
        "contradictions": contra.get("count"),
        "contradictions_capped": bool(contra.get("capped")),
        "recent_miss": recent_miss_count(miss_days, miss_log, ref_time),
        "recent_miss_days": miss_days,
        "active_sessions": len(active_sessions()),
        "spark_days": spark_days,
        "spark_series": series,
        # 文件建档：项目文件与 hippo 的同步态；未启用为 None
        "file_registry": file_registry_summary(),
    }
