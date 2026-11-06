"""Contradiction detection: query compositions over inventory records.

Extracted from ``brain/core.py``. The four entry points map one-to-one to
core's four thin wrappers:

- :func:`ids_matching_tags`  ← ``_ids_matching_tags`` (tag whitelist)
- :func:`contradiction_buckets` ← ``_contradiction_buckets`` (trusted-entity
  bucketing)
- :func:`find_contradictions` ← ``_find_contradictions`` (candidate-pair main
  query)
- :func:`pool_size` ← ``_contradiction_pool_size`` (candidate denominator)

Why they landed here: these are **query compositions over inventory
records** -- they reuse store's ``pairwise_similarities`` capability and the
``MemoryState`` state machine. domain is constrained by the dependency
matrix (may depend only on common) and cannot reach
``store.memory.MemoryState``, so they landed on the store side together with
the two I/O extractions ``tag_ops`` / ``ledger``.

Same-family discipline: this module does **not import vector_store** -- the
store instance is injected by the caller (the brain wrapper passes
``self.manager.store``); trusted entity ids  likewise
are fetched live by the wrapper via ``self._trusted_entity_ids()`` and
passed in; this module never touches brain.
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
    """Filter a whitelist of memory ids by structured tags.

    Semantics: **all** keys must match (AND); a single key's value may be a
    scalar or a list (OR within the list). Values are normalized with
    ``str()`` before comparison -- after a JSON round-trip a tag may be
    ``43`` or ``"43"``; without normalization the filter silently stops
    working.

    Args:
        records: Iterable of candidate records (the caller passes
            ``store.iter_records()``).
        filter_tags: Filter conditions.

    Returns:
        None means "no filtering" (callers must not read it as an empty set).
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
    """Bucket active memories by trusted entity (input to contradiction
    detection).

    Only entities from trusted sources count: untrusted entities (especially
    high-frequency Chinese phrases) cram large numbers of unrelated memories
    into one bucket, producing swathes of false "contradictions" while the
    O(k²) pairwise comparison inside a bucket slows down consolidation. This
    filter is its own function so unit tests can verify it directly.

    Args:
        records: Iterable of candidate records.
        trusted_ids: Set of trusted entity ids (fetched live by the caller;
            this module never touches brain).
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
    """Screen out active memory pairs that are "same entity + high
    similarity + different content".

    **Read-only**: no record state is modified. It only "hints to the host
    that two versions of a position may coexist here"; the verdict belongs
    to the host (in an append-only library, an old and a new version are
    supposed to coexist; see D8).

    Args:
        store: Storage instance (``pairwise_similarities`` / ``get`` /
            ``iter_records`` injected by the caller).
        limit: Maximum number of pairs returned.
        source: Only look at memories from one source (file name / session
            identifier).
        records: Candidate set; defaults to all active records in the
            library.
        filter_tags: Structured tag filter, **the same semantics** as
            ``recall``'s ``filter_tags`` (all keys AND, multi-value OR within
            one key, ``str()`` normalization); reuses
            :func:`ids_matching_tags`.
            Purpose = narrow "whole-library position conflicts" down to
            "conflicts **within one domain**" -- when six generations of
            positions are mixed in one library, the global top-N gets filled
            by unrelated domains.
        trusted_ids: Set of trusted entity ids (fetched live by the caller).

    Returns:
        Candidate pairs sorted by similarity descending, each with
        ``a_id``/``b_id`` (can be handed straight to :meth:`update_memory`
        for superseding) and ``newer`` (whose timestamp is newer).
    """
    recs = list(records if records is not None else store.iter_records())
    recs = [
        r for r in recs
        if r is not None and r.state == MemoryState.ACTIVE.value
        and (not source or (r.source or "") == source)
    ]
    # J3: tag-domain filtering. Same origin as recall (ids_matching_tags), so
    # "query contradictions by domain" and "retrieve by domain" see the same
    # batch of entries -- two implementations would sooner or later compute
    # different whitelists.
    # None = no filter (**do not** read it as an empty set; that means
    # "match nothing").
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
            if a.content == b.content:      # identical content = duplicate, not a contradiction
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
    """The **denominator** of contradiction candidates: how many records
    survive filtering and can enter comparison.

    Reason it exists: ``count: 0`` has two completely different truths --
      ① the candidate pool is non-empty but no contradiction pairs were
      found (= positions agree within the domain, good news)
      ② **the candidate pool itself is empty** (= the filter matched nothing,
      the same silent-failure shape as B2)
    Without reporting the denominator, callers always read it as ①, so the
    conclusion "no problem within the domain" is false.

    Args:
        store: Storage instance (``iter_records`` injected by the caller).
        source: Only look at memories from one source (file name / session
            identifier).
        filter_tags: Structured tag filter.
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
