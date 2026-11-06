# -*- coding: utf-8 -*-
"""The three logic blocks of ``brain.core.refresh_knowledge`` (extracted from
core).

**Why it is worth moving**: ``refresh_knowledge`` is the topic layer's total
refresh entry point; within its 81 lines, the "force/incremental fork
decision", "host summary migration count", and "19-key echo body assembly"
blocks are all pure logic, while the three topic calls build/refresh/migrate
and the self-state writes (``_knowledge_last_*``) stay in the shell. After the
move, the shell is a single straight line of "gate -> fetch data -> inject ->
timing frame".

**Dependency direction**: this module has zero imports -- the eight topic
package symbols ``layer_epoch`` / ``reclip`` / ``build`` / ``refresh`` /
``migrate_host_summaries`` / ``coverage_of`` / ``by_status`` /
``SUMMARY_KIND_HOST`` are all fetched live by the shell at call time and then
injected (same-layer mutual imports are forbidden in domain; topic lives in
domain but G1 allows only common; follows the knowledge_batch /
knowledge_map_view precedent). ``defaults`` / ``max_topics`` / ``now`` / ``t0``
/ ``clock`` are shell-side assembly values (config fetched live, LNN reference
time, timing window).

**Replicated quirks (changing any one is a behavior change; the parity probe
will catch it)**:

1. **force + empty layer takes the incremental branch**: ``if force and
   knowledge`` -- forcing an empty layer does not bump the epoch, does not
   take the old snapshot, and does not call reclip, but reports
   ``rebuilt=True`` with ``new_topics=len(all)``. The "rebuilt" flag is
   **decoupled** from epoch/migration.
2. **The incremental branch's reclip guard**: ``if knowledge`` -- an empty
   layer does not call reclip (calling it on an empty dict would also return
   0, but the call count aligns with the original implementation call by
   call).
3. **reclip mutates members in place, build trims again**: build re-trims
   existing clusters against the same alive set (idempotent), so the reported
   ``members_dropped`` comes only from the reclip pass.
4. **build returns a new dict (shallow copy) while old topic dicts are mutated
   in place** => the force branch's ``old = dict(knowledge)`` snapshot must be
   taken **before** build, so migration reads the truly old layer (collect
   calling plan_rebuild first exists for exactly this).
5. ``now`` = ``manager._reference_time or _now()`` -- the LNN reference time
   takes precedence over the wall clock, computed by the shell and injected
   (store_memory writes _reference_time as the seed timestamp; the probe's
   freeze must come after the seed).
6. ``took`` spans the whole "fetch + build + refresh + migrate" window: both
   ``t0`` and ``clock`` are given by the shell (domain only takes the
   difference); the report carries ``took_s=round(took,3)``, while
   ``_knowledge_last_took`` stores the **unrounded** original value (the shell
   takes it from collect's fourth return value instead of reading the echo
   body -- avoiding double rounding).
7. ``at_cap``'s ``max_topics`` is a **second** live fetch of
   ``_topic_defaults()`` (kept separate from the ``_td`` at the top: when
   tests monkeypatch ``_topic_defaults``, both places must evaluate at call
   time, the M1b discipline).
8. ``count_lost_host``'s ``max(0, ...)`` is defensive: the migrated count
   cannot exceed the old host summary count, but never report a negative;
   only clusters in old with ``kind==host`` and a non-empty summary are
   counted.
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# Block 1: force / incremental fork decision (old shell 2928-2936)
# ======================================================================
def plan_rebuild(
    knowledge: dict[str, Any],
    alive: list[str],
    *,
    force: bool,
    layer_epoch: Callable[[dict[str, Any]], int],
    reclip: Callable[[dict[str, Any], list[str]], int],
) -> tuple[dict[str, Any], dict[str, Any] | None, int | None, int]:
    """Decide whether this refresh round is a "redraw" or "incremental";
    return the three rebuild parameters + the number of members removed.

    Args:
        knowledge: the current topic layer (the shell passes ``self.knowledge``
            straight through).
        alive: the full set of surviving record ids (including those without
            vectors yet).
        force: whether to redraw cluster boundaries.
        layer_epoch: ``topic_mod.layer_epoch`` (reads the layer epoch).
        reclip: ``topic_mod.reclip`` (re-trims members against alive, mutating
            in place).

    Returns:
        ``(base, old, epoch, dropped)`` -- ``base`` is the ledger passed to
        build (a brand-new empty dict when force, the original layer object
        when incremental); ``old`` is a shallow-copy snapshot of the old layer
        (for summary migration) only when force and the layer is non-empty,
        otherwise None; ``epoch`` is the old epoch +1 only on redraw, otherwise
        None; ``dropped`` is the number of member references removed this
        round.

    Quirk replication: force + empty layer takes the incremental branch
    (quirk 1); an empty layer does not call reclip (quirk 2).
    """
    if force and knowledge:
        # Redraw: keep the old layer for summary migration, then build from
        # empty (no reclip -- the new layer has no dead members to trim).
        old = dict(knowledge)
        epoch = layer_epoch(old) + 1
        base: dict[str, Any] = {}
        dropped = 0
    else:
        old = None
        epoch = None
        base = knowledge
        dropped = reclip(knowledge, alive) if knowledge else 0
    return base, old, epoch, dropped


# ======================================================================
# Block 2: host summary migration count (old shell 2947-2952)
# ======================================================================
def count_lost_host(
    old: dict[str, Any],
    migrated: list[dict[str, Any]],
    *,
    summary_kind_host: str,
) -> int:
    """The count of old host summaries not carried over on redraw (self
    -disclosed, never silent).

    Args:
        old: the old layer snapshot (the force branch's ``dict(knowledge)``).
        migrated: the return of ``migrate_host_summaries`` (the list actually
            moved).
        summary_kind_host: the ``topic_mod.SUMMARY_KIND_HOST`` constant.

    Returns:
        ``max(0, old host summary count - moved count)`` (quirk 8: defensive
        lower bound).

    Only clusters with ``summary_kind == host`` and a non-empty summary are
    counted -- a host cluster with an empty summary has no assets to move in
    the first place.
    """
    n_old_host = sum(
        1 for t in old.values()
        if isinstance(t, dict)
        and str(t.get("summary_kind") or "") == summary_kind_host
        and str(t.get("summary") or "").strip())
    return max(0, n_old_host - len(migrated))


# ======================================================================
# Block 3: echo body assembly (old shell 2960-2980)
# ======================================================================
def assemble_report(
    *,
    knowledge: dict[str, Any],
    alive_total: int,
    force: bool,
    before: int,
    old: dict[str, Any] | None,
    migrated: list[dict[str, Any]],
    lost_host: int,
    dropped: int,
    no_vec: int,
    took: float,
    max_topics: int,
    coverage_of: Callable[[dict[str, Any], int], dict[str, Any]],
    by_status: Callable[[dict[str, Any]], dict[str, int]],
    layer_epoch: Callable[[dict[str, Any]], int],
) -> dict[str, Any]:
    """Assemble the refresh echo body: coverage's seven keys first + a 12-key
    update (key order is itself a signal).

    Quirk replication:

    - ``new_topics`` follows the ``force`` flag, not ``old``: force (including
      force on an empty layer) reports the full cluster count, incremental
      reports the incremental difference (quirk 1).
    - ``dropped_old_topics = len(old) if old else 0`` -- force on an empty
      layer reports 0.
    - ``at_cap`` uses ``>=`` (hitting the topic count cap => later records are
      **forcibly assigned** to the nearest cluster and must be visible,
      HIPPOCAMPUS_TOPIC_MAX).
    - ``took_s = round(took, 3)``; the denominator ``alive_total`` is the
      **total active count** (including vectorless ones), not "the count with
      vectors" -- the latter would inflate the coverage ratio.
    """
    out = coverage_of(knowledge, alive_total)
    out.update({
        "enabled": True,
        "rebuilt": bool(force),
        "dropped_old_topics": len(old) if old else 0,
        "migrated_host": migrated,
        "lost_host": lost_host,
        "epoch": layer_epoch(knowledge),
        # force is a clear-and-redraw; subtracting before would go negative --
        # in that case "new" equals everything.
        "new_topics": (len(knowledge) if force
                       else len(knowledge) - before),
        "members_dropped": dropped,
        # The count of surviving records without vectors: part of coverage's
        # unassigned is exactly them. Without reporting it, users only see
        # "coverage dropped" and cannot find the cause.
        "no_vector": no_vec,
        "took_s": round(took, 3),
        "at_cap": len(knowledge) >= max_topics,
        "by_status": by_status(knowledge),
    })
    return out


# ======================================================================
# Orchestration entry (the shell's only call site)
# ======================================================================
def collect(
    items: list[dict[str, Any]],
    info: dict[str, dict[str, Any]],
    knowledge: dict[str, Any],
    *,
    force: bool,
    defaults: Any,
    max_topics: int,
    now: float | None,
    t0: float,
    clock: Callable[[], float],
    layer_epoch: Callable[[dict[str, Any]], int],
    reclip: Callable[[dict[str, Any], list[str]], int],
    build: Callable[..., dict[str, Any]],
    refresh: Callable[..., None],
    migrate_host_summaries: Callable[..., list[dict[str, Any]]],
    coverage_of: Callable[[dict[str, Any], int], dict[str, Any]],
    by_status: Callable[[dict[str, Any]], dict[str, int]],
    summary_kind_host: str,
) -> tuple[dict[str, Any], dict[str, Any], float, int]:
    """The full refresh_knowledge flow: fork -> build -> refresh -> migrate ->
    echo.

    The three-step order carries semantics (line-by-line aligned with the
    original implementation): first re-trim against surviving records -> then
    incrementally absorb the unassigned -> finally recompute derived fields.
    Reversed, deleted members would be baked into the centroids first.

    Args:
        items: surviving records with vectors (``_knowledge_items()``).
        info: the full surviving set ``id -> item`` (including vectorless,
            ``require_vec=False``).
        knowledge: the current topic layer (the shell passes
            ``self.knowledge`` straight through).
        force: whether to redraw cluster boundaries.
        defaults: a live snapshot of ``_topic_defaults()`` (the one at the
            top).
        max_topics: the max_topics from the **second** live fetch of
            ``_topic_defaults()``.
        now: ``manager._reference_time or _now()`` (computed by the shell and
            injected).
        t0: the start of the timing window (the shell stamps it before
            fetching).
        clock: a callable returning the current wall clock (the shell gives
            ``time.time``).
        others: the eight topic package symbols (see the module docstring).

    Returns:
        ``(new_knowledge, report, took, dropped)`` -- the new topic layer (the
        shell assigns it back to ``self.knowledge``), the echo body, the
        unrounded elapsed time, and the number of members removed.
    """
    alive = list(info.keys())
    no_vec = len(info) - len(items)
    before = len(knowledge)
    base, old, epoch, dropped = plan_rebuild(
        knowledge, alive, force=force, layer_epoch=layer_epoch,
        reclip=reclip)
    new_knowledge = build(items, base, alive_ids=alive, epoch=epoch,
                          defaults=defaults)
    refresh(new_knowledge, info, defaults=defaults, now=now)
    # Summary migration: host summaries are host assets; redrawing boundaries
    # must not wash them away. After moving, the signature is **deliberately
    # not recomputed** => the map flags it stale => prompting a re-check.
    migrated: list[dict[str, Any]] = []
    lost_host = 0
    if old:
        migrated = migrate_host_summaries(old, new_knowledge)
        lost_host = count_lost_host(old, migrated,
                                    summary_kind_host=summary_kind_host)
    took = clock() - t0
    report = assemble_report(
        knowledge=new_knowledge, alive_total=len(alive), force=force,
        before=before, old=old, migrated=migrated, lost_host=lost_host,
        dropped=dropped, no_vec=no_vec, took=took, max_topics=max_topics,
        coverage_of=coverage_of, by_status=by_status,
        layer_epoch=layer_epoch)
    return new_knowledge, report, took, dropped
