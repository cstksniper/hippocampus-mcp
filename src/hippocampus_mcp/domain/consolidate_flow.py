# -*- coding: utf-8 -*-
"""The four logic blocks of ``brain.core.consolidate`` (extracted from core).

**Why worth moving**: ``consolidate`` is the periodic consolidation
entry; of its 119 lines, the dormant-archiving loop (22 lines), state
counting, contradiction projection, and report-body assembly are four
blocks of pure logic, yet they live on the self surface and cannot be
unit-tested. After the move, ``consolidate`` is one straight line of
「fetch data → project → orchestrate side effects → assemble」.

**Dependency direction**: this module has zero imports -- the two
constants ``DORMANT_DAYS`` / ``PIN_IMPORTANCE`` and the ``MemoryState``
enum are all **fetched live at call time and injected** by the shell
(the former two's test monkeypatch surface is in the core namespace;
MemoryState lives in the store package, invisible to domain; following
stats_views' injection precedent).

**Replicated quirks (changing any one is a behavior change; parity will
catch it)**:

1. ``archive_dormant``'s ``last`` takes ``max(last_recalled_at or 0.0,
   timestamp)`` -- memories never recalled are counted by their write
   time (when both are 0, age=now, so cold-started large libraries can
   still archive).
2. Dry-run candidates' ``days_idle`` is ``round(..., 1)``; ``max_items``
   counts **candidates** in the dry-run branch and **archived items** in
   the real-archive branch; the two break semantics differ.
3. After a dry-run hit, ``continue`` -- the same record will not be
   archived again (nor under dry_run).
4. ``assemble_report``'s ``scope`` uses an ``if filter_tags`` truthiness
   check: an empty dict reports ``null`` just like None (the converse of
   "not found" ≠ "no domain set");
   ``max_items`` uses ``or None`` (0 = unlimited).
5. The ``dormant_candidates`` key **exists only on dry runs**, and is
   placed after the six fixed keys (key order is itself a signal).
6. ``project_contradictions`` keeps only the ``a`` / ``b`` / ``sim``
   keys -- the underlying ``a_id`` / ``b_id`` / ``newer`` etc. do not
   enter the consolidation report.

**Sleep-batch addendum**: ``select_replay_candidates`` -- the daemon
sleep batch's **replay candidate selection** (pure selection, no state
written). Coactivation semantics in the function docstring; the write
side (strength variable increments) is O1's seam, living in
``brain.core``; this module only produces candidates.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable


# ======================================================================
# Block 1: dormant archiving (old shell 2616-2643)
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
    """Dormant archiving: active memories long uncalled and of low
    importance (pinned ones exempt).

    Args:
        records: snapshot of all store records (the shell fetches it once
            and passes it in; not re-fetched inside the loop).
        now: time baseline (the shell fetches it live via
            ``manager._reference_time or _now()``).
        dormant_days: dormant-days threshold (``DORMANT_DAYS``).
        pin_importance: low-importance upper bound (``PIN_IMPORTANCE``,
            strictly less than).
        active_state / archived_state: state literal values
            (``MemoryState.*.value``).
        allow: tag-domain whitelist; None = no filtering (**not an empty
            set**).
        dry_run: register candidates only; do not call archive.
        max_items: cap (0 = unlimited).
        archive: archive callback ``(rec_id, archived_state)`` -- called
            only on real archiving, never once on a dry run (lazy
            injection; call counts align with the original implementation
            one by one).

    Returns:
        ``(dormant_archived_count, dormant_candidates)``.
    """
    dormant = 0
    dormant_candidates: list[dict[str, Any]] = []
    for rec in records:
        if rec.state != active_state or rec.pin:
            continue
        # Tag-domain narrowing (when unfiltered, allow is None; do not block)
        if allow is not None and rec.id not in allow:
            continue
        last = max(rec.last_recalled_at or 0.0, rec.timestamp)
        if (now - last) / 86400.0 > dormant_days \
                and rec.importance < pin_importance:
            if dry_run:
                # Dry run: register candidates only, do not touch state
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
# Block 2: state counting (old shell 2668-2671 / 2681-2684)
# ======================================================================
def count_state(records: Iterable[Any], state: str) -> int:
    """Number of records in the store in the given state (dual use for consolidate's superseded/active report)."""
    return sum(1 for r in records if r.state == state)


# ======================================================================
# Block 3: contradiction candidate projection (old shell 2662-2666)
# ======================================================================
def project_contradictions(pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Contradiction candidates keep only the a / b / sim keys (the underlying a_id/b_id/newer do not enter the report; quirk 6)."""
    return [
        {"a": c["a"], "b": c["b"], "sim": c["sim"]}
        for c in pairs
    ]


# ======================================================================
# Block 4: report-body assembly (old shell 2672-2689)
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
    """Assemble the consolidation report body (key order is itself a
    signal; see module docstring quirks 4/5).

    Args:
        dormant_candidates: dry-run candidate list; None = not a dry run
            (**key not added**).

    Returns:
        Six-key report body; on dry runs ``dormant_candidates`` is
        appended at the end.
    """
    report: dict[str, Any] = {
        # Self-report: the scope this consolidation covered (null = whole library)
        "scope": dict(filter_tags) if filter_tags else None,
        "max_items": max_items or None,
        "dry_run": bool(dry_run),
        "superseded_total": superseded_total,
        "dormant_archived": dormant,
        "contradiction_candidates": contradictions,
        "active_memories": active_memories,
    }
    if dormant_candidates is not None:
        # A dry run stops here: candidates only. The key goes **after** the six fixed keys
        report["dormant_candidates"] = dormant_candidates
    return report


# ======================================================================
# Block 5: candidate selection for sleep replay
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
    """O2 sleep batch: select 「recent-session L1 clusters」 as
    coactivation replay candidates.

    Biological correspondence (Buzsáki 2015): SWR replays sequences of
    **recent** experience, not a rewind of the whole library -- so the
    window filter (``since`` < timestamp <= ``now``) is a hard premise;
    the carrier of coactivation is **entity co-occurrence** (entities
    that appeared together in one experience record are re-activated
    together on replay), not vector similarity -- that is O3's
    diffusion-rerank battleground, deliberately unused here.

    **This function is pure selection, zero state changes**: no record
    mutation, no storage writes; the write side (retrieval strength
    variable increments) lives in a seam method of ``brain.core``, to be
    wired up after the merge.

    Args:
        records: all store records (the L1 fact layer is the store's
            entire record set; L0 is raw_log and L2/L3 are derived
            layers, none of them in the store).
        now: time baseline (the shell fetches it live via
            ``manager._reference_time or _now()``).
        since: window start (watermark; ``<=0`` = from the beginning).
        active_state: state literal value (``MemoryState.ACTIVE.value``).
        max_items: total cap on **records** participating in replay
            (0 = unlimited). Stop when filled -- but **the first cluster
            is always taken** (skipping a giant cluster that exceeds the
            limit would leave this sleep with nothing; replay granularity
            is the cluster, not the record; half a cluster is
            meaningless).
        min_cluster: clustering lower bound (coactivation needs at least
            two members; records with no entities / sole ownership of
            entities never reach this bound and naturally lose).

    Returns:
        A deterministically ordered cluster list, each cluster
        ``{"ids": [...], "entities": [...], "latest": ts}``; ``ids``
        ordered new→old (replay narrative order), ``entities`` ascending
        and deduplicated, ``latest`` = the newest write time within the
        cluster. Clusters sort by ``(-latest, smallest id)`` -- same
        instant broken by id, so two calls match bit for bit (the
        premise of idempotence: candidate selection itself must be
        reproducible).
    """
    # Window filter: active + timestamp in window + has entities (memories without entities have no coactivation carrier)
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

    # Entity -> record inverted index, then union-find: records sharing any entity are joined into one cluster.
    parent: dict[int, int] = {}

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_entity: dict[str, list[int]] = {}
    for i, rec in enumerate(in_window):
        parent[i] = i
        for ent in dict.fromkeys(rec.entity_ids):  # dedup, order-preserving
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

    # Record-count capping: the first cluster is always taken (see Args); after that, stop when it no longer fits.
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
