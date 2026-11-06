"""Knowledge topic-layer operations: member packing / stats self-report.

Two pieces moved wholesale from ``brain.core``. Architectural constraint:
``domain/topic`` is a sibling layer (siblings may not import each other)
⇒ all topic-layer capabilities are injected through :class:`TopicPort`,
fetched live when the shell calls
(``_topic_defaults`` / ``ordered`` / ``pending_summary_ids`` /
``representative_ids`` / ``summary_state`` / ``apply_host_summary`` /
``coverage_of`` / ``by_status`` / ``host_summary_stats`` are callbacks;
``TOPIC_EVIDENCE_CLIP`` is a value).

Semantics preserved verbatim (changing any one is a behavior change; parity
will catch it):

  * knowledge_items: takes only **active**; ``require_vec=False`` also takes
    records with no vector (``vec`` is None) -- "alive" and "has a vector"
    are two different things, and mixing them makes active-without-vector
    members flap; entity names only take **trusted entities** that are in
    the entity table (``e in trusted and e in id_to_name``);
    timestamp/importance go through ``float(x or 0.0)``; tags are
    shallow-copied;
  * knowledge_stats: layer disabled returns only a note; empty-library
    coverage is given by the topic layer; ``last_refresh_took_s`` None is
    not rounded; when the note already carries topic-layer text, do **not**
    overwrite it (``out.get("note") or default text``);
> Note: ``brief_batch`` / ``summarize_batch`` are not in this module -- two
> behaviorally equivalent same-named implementations once coexisted; to
> avoid dual-track drift only a single line was kept, landing in
> ``domain/knowledge_batch.py`` and ``domain/summarize_batch_flow.py``;
> verification is carried by those two modules' targeted tests and the
> compatibility suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TopicPort:
    """Topic-layer capability port (built by the shell, fetched live at call
    time -- architectural constraint: sibling layers do not import each
    other)."""
    defaults_fn: Callable[[], Any]
    ordered: Callable[..., Any]
    pending_summary_ids: Callable[..., Any]
    representative_ids: Callable[..., Any]
    summary_state: Callable[..., Any]
    apply_host_summary: Callable[..., Any]
    coverage_of: Callable[..., Any]
    by_status: Callable[..., Any]
    host_summary_stats: Callable[..., Any]
    evidence_clip: int


def knowledge_items(
    records: Any, *, entities: dict, trusted_ids: Any,
    vector_of: Callable[..., Any], active_state: str,
    require_vec: bool = True,
) -> list[dict[str, Any]]:
    """Pack active L1 records into the topic module's input format.

    Takes only **active**: superseded / archived are positions that no longer
    count. With ``require_vec=False``, records that **have no vector yet**
    are taken too (``vec`` is None) -- "alive" and "has a vector" are two
    different things: clustering cannot compare records without vectors
    (better missing than wrong; they self-report into ``unassigned``), but
    deciding "which members have gone invalid" can only look at active.
    Mixing them makes active-without-vector records flap ⇒ signatures stop
    matching ⇒ false stale everywhere.
    """
    trusted = trusted_ids
    id_to_name = {e.get("id"): k for k, e in entities.items()}
    items: list[dict[str, Any]] = []
    for rec in records:
        if rec.state != active_state:
            continue
        vec = vector_of(rec)
        if vec is None and require_vec:
            continue
        names = [id_to_name[e] for e in (rec.entity_ids or [])
                 if e in trusted and e in id_to_name]
        items.append({
            "id": rec.id,
            "vec": vec,
            "content": rec.content or "",
            "timestamp": float(rec.timestamp or 0.0),
            "importance": float(rec.importance or 0.0),
            "entities": names,
            "tags": dict(rec.tags or {}),
        })
    return items


def knowledge_stats(
    port: TopicPort, *, knowledge: dict, records: Any,
    active_state: str, last_at: Any, last_took: Any, last_dropped: Any,
) -> dict[str, Any]:
    """brain_stats self-report: the topic layer's current state (coverage
    and last refresh time included)."""
    if not port.defaults_fn().layer_enabled:
        return {"enabled": False,
                "note": "the topic layer is disabled (HIPPOCAMPUS_TOPIC_LAYER=0)"}
    active_total = sum(1 for r in records
                       if r.state == active_state)
    out = port.coverage_of(knowledge, active_total)
    out.update({
        "enabled": True,
        "by_status": port.by_status(knowledge),
        # Summarization progress: host = already written back by the host;
        # host_stale = written but members have since changed.
        # With these two cells, "how far along is the curation" needs no
        # guesswork.
        "summary_kinds": port.host_summary_stats(knowledge),
        "last_refresh_at": last_at,
        "last_refresh_took_s": (round(last_took, 3)
                                if last_took is not None else None),
        "members_dropped_last": last_dropped,
        "note": (out.get("note") or
                 "the topic layer is derived from L1; representative sentences are extractive quotes (originals closest to the centroid), not summaries"),
    })
    return out
