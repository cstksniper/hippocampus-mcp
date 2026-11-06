# -*- coding: utf-8 -*-
"""Library-level "overview card / status card" projections (extracted from
core).

Both are **pure derivation, no retrieval**, O(n) single pass:

* :func:`project_overview` -- identity+scale / time span / structural
  profile / hot entities;
* :func:`project_status` -- four signal groups: open intents / health /
  collaboration / activity.

Why in domain/: it only does "existing derived layer → structured
projection"; the three statistical measures `recent_ingest` /
``spark_series`` / ``recent_miss_count`` already live in ``common.stats``.
Zero I/O, zero model, zero config reads. brain keeps only a thin wrapper,
binding and injecting the derived layer, the reference clock, the two
Config switches, the five caps, and the five **lazy callbacks** at call
time.

🔴 **Lazy callbacks are discipline, not laziness**: ``knowledge_stats`` is
called only when the topic-layer switch is on, ``contradictions_count``
only when the status-card contradiction switch is on -- that is exactly how
often the original implementation called them (0.34s-scale on a
ten-thousand-record library; one extra call is pure burn). So what is
passed here is a **callable**, not a "precomputed value": a precomputed
value would pay the cost even when the switch is off.

🔴 **Library name goes through injection**: ``display_name`` lives in
``library/``, and arch G1's matrix is ``domain → common only`` (burned
before by same-tier cross-imports). So the wrapper computes the library
name and passes it in.

Pre-existing criteria preserved under "relocate only, no behavior change":

* entity Top only includes entities with **actual associations**
  (non-empty ``memory_ids``) -- about 84% of heuristic entities are phrase
  noise; unfiltered, the Top rows would be filled with a wall of (0);
* the time span is ``(None, None)`` when there are no timestamps at all,
  not 0;
* ``contradictions`` with the switch off is ``{"count": None, "capped":
  False, "limit": 0}`` -- **None is not 0** (0 would be read as "checked,
  no contradictions");
* L0 dangling uses the same criterion as ``stats()`` (count of affected L1
  entries); the id set is built once.
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
    """Library-level overview projection (design doc §3.1). Pure derivation,
    no retrieval, O(n) single pass."""
    total = len(records)
    ts = [r.timestamp for r in records if r.timestamp]
    first_ts, last_ts = (min(ts), max(ts)) if ts else (None, None)
    tag_cov = tag_coverage()
    kn = knowledge_stats() if topic_layer_enabled else {}
    kinds = kn.get("summary_kinds") or {}
    # Only entities with **actual associations**: about 84% of heuristic
    # entities are phrase noise (see the ENTITY_ORIGIN_* notes); unfiltered,
    # the Top rows would be filled with a wall of (0) -- noise as well as
    # misleading.
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
    """Current status projection (design doc §3.2). Pure derivation, no
    retrieval."""
    series = spark_series(spark_days, daily_counts, ref_time)
    contra = (contradictions_count(contradiction_cap)
              if contradictions_enabled
              else {"count": None, "capped": False, "limit": 0})
    # L0 dangling: same criterion as stats() (count of affected L1 entries);
    # the id set is built once -- the numbers on the card must match
    # brain_stats (single source of truth), not be computed independently.
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
        # File archiving: sync state between project files and hippo; None
        # when not enabled
        "file_registry": file_registry_summary(),
    }
