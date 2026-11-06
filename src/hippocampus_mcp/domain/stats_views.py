"""Stats self-report projections (read-only): the pin / recall-feedback / literal-channel slots plus a pinned-entries brief.

Extracted from ``brain/core.py`` (``_recall_feedback_stats`` 45 lines /
``_pin_stats`` 50 lines / ``lexical_self_report`` 48 lines / ``_pinned_brief``
43 lines).

Why here: all four are **pure read-only projections** — inputs are store
records (or backend objects) plus a few scalars; outputs are a self-report
dict / entry list. Zero retrieval, zero disk writes, zero models, matching the
domain matrix rule "may only depend on common" (``typing`` is standard
library).

🔴 **No store import**: both ``MemoryState.ACTIVE.value`` and
``PIN_IMPORTANCE`` live in the store package ⇒ they are always injected by the
caller (``active_state`` string / ``pin_threshold`` float). Aligns with
map_view's ``archived_state`` convention.

🔴 **``pinned_brief``'s relevance is a callback**: ``relevance(task,
task_hit_ids)`` is injected by the caller (brain-side it is
``HippoBrain._pin_task_relevance``, which also reads ``_task_entity_ids`` ⇒ it
stays in brain). What is passed is the **callable**, not a call result —
aligned with the original implementation's per-call behavior, and a stand-in
can be injected.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "lexical_self_report",
    "pin_stats",
    "pinned_brief",
    "recall_feedback_stats",
]


def recall_feedback_stats(
    records: list[Any],
    active_state: str = "active",
) -> dict[str, Any]:
    """Recall-feedback self-report: how many records ``last_recalled_at`` actually covers.

    ``recall_count`` records the **candidate set** (who entered the retrieval
    pool); ``last_recalled_at`` records the **result set** (who was actually
    returned to the caller). The difference between the two coverage rates =
    the amount of "retrieved but not adopted".

    Why self-report: ``last_recalled_at`` is exactly consolidate/decay's
    **time basis**, and historically it went long stretches with zero writes
    (measured coverage ~9.5%) — with low coverage, archival conclusions like
    "how long since this was used" become distorted, and **without any error**.

    Convention: **no backfill** (backfilling equals fabricating recall
    history); instead, each stats call self-reports, handing the "how much to
    discount this credibility" judgment back to the host.
    """
    active = [r for r in records if r.state == active_state]
    total = len(active)
    if not total:
        return {
            "active_total": 0,
            "with_last_recalled_at": 0,
            "recall_count_positive": 0,
            "coverage": None,
            "coverage_level": "unknown",
            "note": "no active memories in the library; coverage is meaningless",
        }
    with_last = sum(1 for r in active if (r.last_recalled_at or 0.0) > 0.0)
    counted = sum(1 for r in active if (r.recall_count or 0) > 0)
    cov = with_last / total
    if cov >= 0.5:
        level = "ok"
    elif cov >= 0.2:
        level = "low"
    else:
        level = "very_low"
    return {
        "active_total": total,
        "with_last_recalled_at": with_last,
        "recall_count_positive": counted,
        "coverage": round(cov, 4),
        "coverage_level": level,
        "note": ("last_recalled_at 是 consolidate / 衰减的时间基准；"
                 "coverage_level 非 ok 时，「多久没被用过」这类"
                 "归档结论应当打折看待"),
    }


def pin_stats(
    records: list[Any],
    active_state: str = "active",
    pin_threshold: float = 0.7,
    error: str | None = None,
) -> dict[str, Any]:
    """Pin self-report: how many are pinned, and how many of those were **auto-pinned by lexical rules**.

    Why self-report: ``PIN_IMPORTANCE = 0.7``, and :meth:`_guess_importance`
    grants 0.7 upon hitting 「现在 / 不再 / changed / ！」 ⇒ **auto pin**; and
    pin is **one-way** in this system (the only "exit" is an explicit caller
    action) ⇒ in long-lived libraries pins pile up monotonically (measured: 78%
    in one library), crowding out the usage layer's dynamic slots, all
    **without a single error**.

    The "suspected auto" criterion: ``pin=True`` and importance within
    ``_guess_importance``'s value range (``[PIN_IMPORTANCE, 0.75]``). This is
    **suspected**, not confirmed — an explicit caller-passed 0.7 also lands
    here, hence the ``suspected`` field name.

    Args:
        records: store records (already fetched by the caller; pass ``error``
            when fetching the records themselves failed).
        active_state: the active-state string (``MemoryState.ACTIVE.value``).
        pin_threshold: ``PIN_IMPORTANCE``, fetched and injected by the caller.
        error: the exception type name from a record-fetch failure — a stats
            failure must not take down the other self-report fields.
    """
    if error is not None:
        return {"active_total": 0, "pinned_total": 0, "pinned_ratio": None,
                "suspected_auto_pinned": 0, "level": "unknown",
                "note": f"stats failed (other self-report fields unaffected): {error}"}
    active = [r for r in records if r.state == active_state]
    total = len(active)
    if not total:
        return {"active_total": 0, "pinned_total": 0, "pinned_ratio": None,
                "suspected_auto_pinned": 0, "level": "unknown",
                "note": "no active memories in the library; pin ratio is meaningless"}
    pinned = [r for r in active if getattr(r, "pin", False)]
    auto = [r for r in pinned
            if pin_threshold <= (getattr(r, "importance", 0) or 0) <= 0.75]
    breakdown = {"auto": 0, "manual": 0, "legacy": 0, "unmarked": 0}
    for rec in pinned:
        src = str(getattr(rec, "pin_source", "") or "")
        breakdown[src if src in breakdown else "unmarked"] += 1
    ratio = len(pinned) / total
    quota_warning = ratio >= 0.6
    level = "high" if ratio >= 0.5 else ("ok" if ratio >= 0.2 else "low")
    return {
        "active_total": total,
        "pinned_total": len(pinned),
        "pinned_ratio": round(ratio, 4),
        "suspected_auto_pinned": len(auto),
        "pin_threshold": pin_threshold,
        "pin_source_breakdown": breakdown,
        "quota_warning": quota_warning,
        "level": level,
        "note": (
            "supersede 会自动解 pin；其余退出仍靠调用方显式动作："
            "① report_feedback(useful=False) ② 显式 pin=False "
            "③ 降 importance 触发 sync_pin（store/vector_store.py → "
            "memory.py:175，双向）。level=high 意味着用层动态位被挤占，"
            "「suspected_auto_pinned」那部分是词面规则自动钉的（疑似，非确证）。"
            if level == "high" else
            "pin 占比正常。提醒：除 supersede 自动解 pin 外，"
            "其余解除只能靠调用方显式动作。"),
    }


def pinned_brief(
    records: list[Any],
    relevance: Callable[[str, set[str] | None], dict[str, int]],
    limit: int = 3,
    task: str = "",
    task_hit_ids: set[str] | None = None,
    active_state: str = "active",
) -> list[dict[str, Any]]:
    """The pinned-memories slot.

    Ordering convention: **task relevance first**.

    Pinned entries' importance is generally high and close together (measured:
    book-level iron rule 0.9 vs chapter-level card 0.8); pure importance
    ordering really yields "who got promoted first" — so the iron rules useful
    for this session get crowded out by unrelated session cards. What truly
    discriminates is "does it match this task", hence three tiers:

        ``task_relevance`` = 2  the task's semantic recall hit this entry
                           = 1  intersects the task's **trusted entities**
                           = 0  unrelated

    Within a tier, descending by importance / timestamp; with an empty
    ``task``, fall back to pure importance (same as the old behavior).
    Entities are only trusted from trusted sources: ~84% of heuristic entities
    are phrase noise; letting them into the "relevance" judgment amounts to a
    random reshuffle.

    Args:
        records: store records.
        relevance: relevance callback ``(task, task_hit_ids) -> {memory_id: 0/1/2}``.
        limit: number of entries to return. **Negative = slot off** (returns an
            empty list).
        task: this session's task description, used for task-relevance-first.
        task_hit_ids: memory_ids hit by the task's semantic recall (already
            computed once in ``session_start``; pass it in for reuse, no
            repeated retrieval).
    """
    pinned = [
        r for r in records
        if r.pin and r.state == active_state
    ]
    rel = relevance(task, task_hit_ids)
    pinned.sort(key=lambda r: (-rel.get(r.id, 0), -r.importance, -r.timestamp))
    out: list[dict[str, Any]] = []
    for r in pinned[:max(int(limit), 0)]:
        out.append({
            "memory_id": r.id, "content": r.content, "score": r.importance,
            "similarity": 1.0, "timestamp": r.timestamp, "importance": r.importance,
            "state": r.state, "cognitive_type": r.cognitive_type,
            "source_ids": r.source_ids or [], "entity_ids": r.entity_ids or [],
            "pin": True, "source": r.source or "",
            # Self-reported ordering basis: when the host asks "why did the iron rule drop out of the top three", the reason must be traceable
            "task_relevance": int(rel.get(r.id, 0)),
        })
    return out


def lexical_self_report(
    backend: Any,
    indexed: int,
    path: str,
) -> dict[str, Any]:
    """Runtime self-report of the literal channel (BM25 / FTS5).

    Why this section: sqlite + FTS5's biggest selling point is "**no more
    in-memory BM25 index to build, saving 34.5MB**", yet this optimization had
    no observation entry on the production side — "did it actually save" could
    only be guessed. Every item here self-reports a **measured value**:

    - ``store_backend``: the actual storage backend (json / sqlite)
    - ``fts5_ready``: whether the FTS5 virtual table is really usable on this
      connection
    - ``bm25_indexed``: how many entries the in-memory BM25 currently holds —
      **must be 0 when the FTS5 path is genuinely in effect**
    - ``last_lexical_path``: the path the literal channel actually took on the
      last retrieval (fts5 / memory / off)

    Args:
        backend: the storage backend object (``store._backend``, may be ``None``).
        indexed: the in-memory BM25's current entry count
            (``len(brain._bm25_indexed)``).
        path: the path the last retrieval actually took
            (``brain._last_lexical_path``).
    """
    name = getattr(backend, "name", None) or "unknown"
    fts_ready: bool | None = None
    for meth in ("_fts_ok", "fts5_supported"):
        fn = getattr(backend, meth, None)
        if callable(fn):
            try:
                fts_ready = bool(fn())
            except Exception:  # noqa: BLE001 — self-report must not fail wholesale because a probe failed
                fts_ready = False
            break
    if name == "sqlite" and fts_ready and indexed == 0:
        level, note = "ok", "the literal channel uses FTS5; in-memory BM25 not built (34.5MB saved)"
    elif name == "sqlite" and fts_ready and path == "memory":
        level = "degraded"
        note = (f"FTS5 is available but this query still fell back to in-memory BM25 (built {indexed} records)；"
                "中文 2 字词触发回退是预期行为，频繁出现才需要查")
    elif name == "sqlite" and not fts_ready:
        level = "degraded"
        note = (f"sqlite backend without FTS5 => the literal channel falls back to in-memory BM25 "
                f"(built {indexed} records; the 34.5MB saving is unavailable)")
    elif name == "json":
        level, note = "json", f"json backend: the literal channel can only use in-memory BM25 (built {indexed} records)"
    else:
        level, note = "unknown", f"backend={name}, the literal channel's form cannot be determined"
    return {
        "store_backend": name,
        "fts5_ready": fts_ready,
        "bm25_indexed": indexed,
        "last_lexical_path": path,
        "level": level,
        "note": note,
    }
