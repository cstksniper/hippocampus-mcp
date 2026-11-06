# -*- coding: utf-8 -*-
"""The six computation blocks of ``brain.core.stats`` (extracted from core).

**Why worth moving**: ``stats`` is the self-check entry hosts must call; of
its 183 lines, besides "calling each layer's self-exposure methods", six
blocks of pure computation are tangled in -- state distribution, L0<->L1
dangling references, per-day activity window, source-signature coverage,
shared-library writer projection, forgotten count. They are all pure
dict/list operations, yet untestable because they grew inside self-facing
context; after the move ``stats`` is one straight line of "fetch ->
project -> assemble the echo".

**Dependency direction**: this module imports only the standard library
(domain -> common-only discipline; here not even common is used -- all
records/counts/pids are fetched fresh by the shell and passed in).

**Replicated quirks (touching one is a behavior change; parity will catch
it)**:

1. ``state_counts`` counts via ``dict.get`` rather than ``Counter`` -- both
   count the same, but **key order = first-appearance order** must be kept
   (the echo body is JSON-serialized directly; order is signal).
2. ``dangling_report``'s affected count +1s only records that **truly have
   dangling refs**; one record with several dangling ids counts once (the
   ``if hit`` gate sits outside the update).
3. ``daily_activity``'s totals cover **all days** (not within the echo
   window); ``recent`` takes ``sorted(daily_counts)[-n_days:]`` --
   ``_resolve_daily_days`` always returns >=1 (explicit max(...,1) / env
   validated >0 / default 14), so the ``[-0:]`` whole-table quirk is
   unreachable on this path (on record; do not "fix" it to ``[:n]``).
4. ``source_sig_block``'s referenced double-normalizes source with
   ``str().strip()``: a blank string equals no source; neither counted nor
   blocking the filter.
5. ``writers_block``'s ``is_self`` is decided by **pid equality** (not
   object identity); always False when pid is missing.
6. ``forgotten_count`` uses ``r.get("forgotten")`` rather than
   ``r["forgotten"]`` -- raw entries missing the key are the norm (data
   persisted by old versions); subscript access would KeyError.
"""
from __future__ import annotations

from typing import Any, Iterable


# ======================================================================
# Block 1: state distribution (old shell 3371-3373)
# ======================================================================
def state_counts(records: Iterable[Any]) -> dict[str, int]:
    """Count L1 memories by lifecycle state (key order = first-appearance
    order; see module docstring quirk 1).

    Args:
        records: library record iterator (duck-typed: has a ``.state``
            attribute).
    """
    states: dict[str, int] = {}
    for r in records:
        states[r.state] = states.get(r.state, 0) + 1
    return states


# ======================================================================
# Block 2: L0↔L1 reference integrity (old shell 3379-3386)
# ======================================================================
def dangling_report(
    records: Iterable[Any],
    raw_ids: set[str],
) -> tuple[set[str], int]:
    """The set of source_ids referenced by L1 but absent from L0 + the count
    of affected L1 entries.

    Args:
        records: library record iterator (duck-typed: has ``.source_ids``).
        raw_ids: the set of all raw ids on the L0 layer.

    Returns:
        ``(dangling_ids, affected_records)``. The affected count +1s only
        records that truly have dangling refs; one record with several
        dangling ids counts once (see module docstring quirk 2).
    """
    dangling: set[str] = set()
    dangling_recs = 0
    for _r in records:
        hit = [s for s in (_r.source_ids or []) if s not in raw_ids]
        if hit:
            dangling.update(hit)
            dangling_recs += 1
    return dangling, dangling_recs


# ======================================================================
# Block 3: per-day activity window (old shell 3393-3425's daily_activity keys)
# ======================================================================
def daily_activity(daily_counts: dict[str, dict[str, int]],
                   n_days: int) -> dict[str, Any]:
    """Echo body of per-day write/recall counts (window + all-days totals,
    separately self-exposed).

    Args:
        daily_counts: ``{day: {"recall": n, "ingest": n}}``.
        n_days: echo window length (fetched fresh by the shell via
            ``_resolve_daily_days``, always >=1; passing an explicitly large
            number just shows what exists, no empty slots padded).

    Returns:
        Four keys: ``recent`` / ``totals`` / ``kept_days`` / ``shown_days``.
        totals cover **all days** (see module docstring quirk 3).
    """
    days = sorted(daily_counts)[-n_days:]
    total_recall = sum(d.get("recall", 0) for d in daily_counts.values())
    total_ingest = sum(d.get("ingest", 0) for d in daily_counts.values())
    return {
        "recent": [
            {"day": d,
             "recall": daily_counts[d].get("recall", 0),
             "ingest": daily_counts[d].get("ingest", 0)}
            for d in days
        ],
        "totals": {"recall": total_recall, "ingest": total_ingest},
        "kept_days": len(daily_counts),
        # Self-exposed window: when the host sees "only 7 days", it must be
        # able to tell that is a window, not the inventory
        "shown_days": len(days),
    }


# ======================================================================
# Block 4: source-signature coverage (old shell 3455-3461's source_sig keys)
# ======================================================================
def source_sig_block(source_sigs: dict[str, Any],
                     records: Iterable[Any]) -> dict[str, int]:
    """Source-signature collection coverage: recorded (collected) vs
    referenced (referenced within the library).

    Args:
        source_sigs: ``brain.source_sigs`` (collected source -> signature).
        records: library record iterator (duck-typed: has ``.source``).

    Returns:
        ``{"recorded": n, "referenced": n}``. referenced double-normalizes
        source with ``str().strip()`` (see module docstring quirk 4).
    """
    return {
        "recorded": len(source_sigs),
        "referenced": len({str(r.source or "").strip()
                           for r in records
                           if str(r.source or "").strip()}),
    }


# ======================================================================
# Block 5: shared-library writer projection (old shell 3463-3473's writers keys)
# ======================================================================
def writers_block(active: list[dict[str, Any]], stale_cleaned: int,
                  self_pid: int) -> dict[str, Any]:
    """Make shared-library writers visible: the truth right now (not the
    last persisted snapshot).

    Args:
        active: ``brain.writers_active`` (writers whose heartbeat has not
            expired).
        stale_cleaned: ``brain.writers_stale_cleaned`` (count cleaned last
            time).
        self_pid: this process's pid (the shell passes ``os.getpid()``; a
            stand-in can be injected).

    Returns:
        Four keys ``active`` / ``stale_cleaned`` / ``self_pid`` / ``list``;
        ``is_self`` decided by pid equality (see module docstring quirk 5).
    """
    return {
        "active": len(active),
        "stale_cleaned": stale_cleaned,
        "self_pid": self_pid,
        "list": [
            {"pid": w.get("pid"), "host": w.get("host"),
             "cwd": w.get("cwd"), "saves": w.get("saves"),
             "is_self": w.get("pid") == self_pid}
            for w in active
        ],
    }


# ======================================================================
# Block 6: forgotten raw count (old shell 3477)
# ======================================================================
def forgotten_count(raw_log: Iterable[dict[str, Any]]) -> int:
    """The count of entries marked forgotten in L0 (``.get`` keeps old
    persisted data missing the key from crashing; quirk 6)."""
    return sum(1 for r in raw_log if r.get("forgotten"))
