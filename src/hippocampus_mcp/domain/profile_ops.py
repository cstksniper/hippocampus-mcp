"""Profile-layer operation family: daily counts / profile refresh / staleness pruning.

Three pieces moved wholesale from ``brain.core``. **The state containers do not move**:
``profile`` and ``daily_counts`` are injected by the shell as the live objects (mutated
in place); store reads, the trust criterion, and the record stream are all **fetched at
call time** and injected (``records_fn`` is lazy -- the pinned scan materializes only
after the entity scan); ``base_fn`` is lazy -- ``bump_daily``'s kind/n gate runs
**before** the time base is computed, so with the switch off it is never called
(lazy-injection discipline).

Semantics preserved verbatim (changing any one is a behavior change; parity will catch it):

  * bump_daily: kind accepts only ``"recall"`` / ``"ingest"`` with n>0, otherwise
    **returns directly without touching the table**; the day key is ``%Y-%m-%d`` in
    local time; the slot defaults to ``{"recall":0,"ingest":0}``; beyond ``keep_days``
    the oldest are trimmed in **ascending key order** (keys are ISO dates; ascending =
    oldest); trimming happens **after the append**;
  * prune_stale_profile: only touches the ``stable_facts`` and ``entity:*`` slot
    families (other custom slots are left alone); the criterion applies only to entries
    with **exactly 1 evidence** id; entries whose memory cannot be located are always
    kept (better to miss a prune than prune wrongly); reason has three values,
    memory_missing / memory_inactive / unpinned (unpinned is judged only for
    stable_facts); a slot emptied entirely is popped, non-empty ones get kept written back;
  * update_profile: **prune stale first, then add entries** (prune runs first); all
    three sources admit only active records -- entities take the top entity_max by
    mention_count descending (trusted origins only), skip mention_count<3, build one
    entry per memory taking the 3 most recent, confidence=0.3*mention_count; pinned
    entries are ordered by (importance, timestamp) descending; the O4 graduation source
    (enabled only when recurring_slot is non-empty) = entries that **nobody pinned but
    that keep being recalled** (recall strength >= strength_min) **and have lived long
    enough** (age >= strength_days), taking the top strength_max by (strength,
    timestamp) descending into a separate slot, confidence = min(0.9, BASE + STEP·strength)
    (calibration constants in common.constants; strength 3→0.48, 5→0.60, 10→0.90 --
    self-consistent ordering within the slot, never competing with pinned/entity slots);
    entries dedup on exact text equality; a full slot drops the **lowest-confidence**
    entry (the new entry participates in the comparison; on ties the old one stays);
    confidence is capped at 0.9 and round2; evidence is clipped to the first 3.

Boundary between O4 and the U3 "never auto-promote" red line: U3 governs the project→user
**cross-library** promotion flow (explicit confirmation required before writing the user
library); the graduation source only appends entries to the display layer of the
**in-library** L3 profile -- no records written, no state changed, the user library
never touched. L3 already refreshes automatically with consolidate; the two flows have
independent criteria and isolated channels.
"""
from __future__ import annotations

import time as _time
from typing import Any, Callable

from ..common.constants import PROFILE_STRENGTH_BASE, PROFILE_STRENGTH_STEP


def bump_daily(
    kind: str, n: int, *, daily_counts: dict, base_fn: Callable[[], float],
    keep_days: int,
) -> None:
    """Accumulate one activity count per day (addition 5): ``kind`` is ``"recall"`` / ``"ingest"``.

    **The time base must match consolidate's now** (the shell provides
    ``manager._reference_time or now`` lazily via ``base_fn``): a bare ``time.time()``
    would book all counts onto today under evals whose data timestamps lie in the past.
    """
    if kind not in ("recall", "ingest") or n <= 0:
        return
    base = base_fn()
    day = _time.strftime("%Y-%m-%d", _time.localtime(base))
    slot = daily_counts.setdefault(day, {"recall": 0, "ingest": 0})
    slot[kind] = int(slot.get(kind, 0)) + int(n)
    if len(daily_counts) > keep_days:
        for old in sorted(daily_counts)[
            : len(daily_counts) - keep_days
        ]:
            daily_counts.pop(old, None)


def prune_stale_profile(
    profile: dict, *, store_get: Callable[[str], Any],
    stable_slot: str, active_state: str,
    recurring_slot: str = "",
) -> list[dict[str, str]]:
    """Prune **stale** L3 profile entries (zcode addition 19②).

    Criterion (applies only to entries with exactly 1 evidence id that resolves to that
    memory; anything unresolvable is always kept -- **better to miss a prune than prune
    wrongly**): ``stable_facts`` keeps an entry only while that memory is still pin and
    active; ``entity:*`` keeps it only while the memory is still active; the O4
    graduation slot (``recurring_slot``) follows the same rule as entity:* -- kept only
    while still active (graduation entries were never pinned, so the unpinned criterion
    does **not** apply); other custom slots are untouched.
    Returns the list of pruned entries -- **what the profile lost must be visible**.
    """
    dropped: list[dict[str, str]] = []
    for slot, items in list(profile.items()):
        managed = (
            slot == stable_slot
            or (bool(recurring_slot) and slot == recurring_slot)
            or slot.startswith("entity:")
        )
        if not managed:
            continue
        kept: list[dict[str, Any]] = []
        for it in items:
            ev = it.get("evidence") or []
            reason = ""
            if len(ev) == 1:
                rec = store_get(ev[0])
                if rec is None:
                    reason = "memory_missing"
                elif rec.state != active_state:
                    reason = "memory_inactive"
                elif slot == stable_slot and not rec.pin:
                    # addition 19②'s main complaint: unpinning / importance downgrade out of pinned
                    reason = "unpinned"
            if reason:
                dropped.append({
                    "slot": slot,
                    "text": (it.get("text") or "")[:80],
                    "memory_id": ev[0] if len(ev) == 1 else "",
                    "reason": reason,
                })
                continue
            kept.append(it)
        if kept:
            profile[slot] = kept
        else:
            profile.pop(slot, None)
    return dropped


def update_profile(
    profile: dict, *, prune_fn: Callable[[], list],
    entities: dict, entity_trusted: Callable[[dict], bool],
    store_get: Callable[[str], Any], records_fn: Callable[[], Any],
    now_fn: Callable[[], float], active_state: str,
    slot_max: int, entity_max: int, stable_slot: str,
    recurring_slot: str = "",
    strength_min: float = 0.0, strength_days: float = 0.0,
    strength_max: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Distill the L3 profile: admit only high-confidence entries; conflicts append, never overwrite.

    Returns ``(updates, dropped)``; ``dropped`` is persisted by the shell into
    ``last_profile_pruned`` (stale entries pruned before the refresh; prune runs first).
    """
    dropped = prune_fn()
    updates: list[dict[str, Any]] = []

    def _append(slot: str, text: str, evidence: list[str],
                confidence: float) -> None:
        existing = profile.setdefault(slot, [])
        if any(f["text"] == text for f in existing):
            return
        # Slot capacity cap (coverage report · high 5): when full, drop the lowest-confidence
        # entry (the new entry participates in the comparison too).
        entry = {
            "text": text, "since": now_fn(), "evidence": evidence[:3],
            "confidence": round(min(confidence, 0.9), 2),
        }
        if len(existing) >= slot_max:
            weakest = min(range(len(existing)),
                          key=lambda i: existing[i].get("confidence", 0.0))
            if existing[weakest].get("confidence", 0.0) >= entry["confidence"]:
                return
            existing.pop(weakest)
        existing.append(entry)
        updates.append({"slot": slot, "text": text,
                        "confidence": entry["confidence"]})

    # Entity profiles take only the top few by mention_count; trusted origins only.
    ranked_ents = sorted(
        ((k, e) for k, e in entities.items() if entity_trusted(e)),
        key=lambda kv: -int(kv[1].get("mention_count", 0)),
    )[:entity_max]
    for key, ent in ranked_ents:
        if ent["mention_count"] < 3:
            continue
        # One profile entry per memory; evidence points at the exact memory backing that entry.
        records = [store_get(m) for m in ent["memory_ids"]]
        records = [
            r for r in records
            if r is not None and r.state == active_state
        ]
        records.sort(key=lambda r: -r.timestamp)
        for rec in records[:3]:
            _append(f"entity:{key}", rec.content[:120], [rec.id],
                    0.3 * ent["mention_count"])

    # The record stream materializes once: the pinned and O4 graduation sources share the
    # same snapshot (records_fn is lazy -- called after the entity scan; discipline unchanged).
    recs = list(records_fn())
    # Pinned memories = long-term stable preferences / taboos / hard constraints.
    # Enqueue by importance descending: the slot has a capacity cap, so first-come-first-served
    # would let "insertion order", not "importance", decide what survives.
    pinned = [
        r for r in recs
        if r.pin and r.state == active_state
    ]
    pinned.sort(key=lambda r: (-r.importance, -r.timestamp))
    for rec in pinned:
        _append(stable_slot, rec.content[:120], [rec.id], rec.importance)

    # Access-driven "graduation" source -- L1 entries that nobody pinned but that keep
    # being recalled (recall strength) and have lived long enough (stabilization period).
    # recurring_slot empty = the whole source is off (closing the gate is a zero behavior
    # change); pin entries are not admitted (they already have the stable_facts channel).
    # Separate slot: confidence calibration competes only within its own kind, never
    # squeezed against pinned/entity entries.
    if recurring_slot:
        now = now_fn()
        grads = [
            r for r in recs
            if not r.pin and r.state == active_state
            and float(r.recall_strength or 0.0) >= strength_min
            and (now - r.timestamp) / 86400.0 >= strength_days
        ]
        grads.sort(key=lambda r: (-float(r.recall_strength or 0.0),
                                  -r.timestamp))
        for rec in grads[:max(int(strength_max), 0)]:
            conf = (PROFILE_STRENGTH_BASE
                    + PROFILE_STRENGTH_STEP * float(rec.recall_strength or 0.0))
            _append(recurring_slot, rec.content[:120], [rec.id], conf)
    return updates, dropped
