# -*- coding: utf-8 -*-
"""Shared, side-effect-free orchestration for recall scoring (R1).

Previously this orchestration was inlined in ``HippoBrain.recall``. R1 makes
BOTH callers -- the brain shell and the production MCP recall service --
delegate here, so the evaluation harness and the host see the same scoring,
ordering and filtering.

Dependency direction: ``retrieval`` may import ``common`` and ``domain``,
and both ``brain`` and ``application`` may import ``retrieval``. The pure
scoring blocks stay in ``domain/recall_flow.py``; everything else this
pipeline needs from outer layers is injected through :class:`RecallDeps`,
and the host brain is duck-typed, never imported.

Only one host attribute is written: ``brain._last_lexical_path`` (an
idempotent diagnostic self-report). The three accounting effects --
``stamp_recalled``, ``bump_daily`` and ``log_miss`` -- deliberately stay in
the callers' shells; putting them here would double-count them once both
paths share this function.

Behavioural quirks are preserved verbatim. See ``domain/recall_flow.py`` for
the three pure blocks this module composes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..domain import recall_flow as _recall_flow

# Defaults come from common (allowed for retrieval as well),
# but callers override them to preserve their own monkeypatch surface.
from ..common.constants import (
    ENTITY_BONUS,
    MMR_LITERAL_EXEMPT_TOPN,
    RELEVANCE_HARD_FLOOR,
    _now,
    strip_internal_fields,
)
from ..common.recency import (
    in_time_window,
    iso_date,
    out_of_window,
    weekday_abbr,
)

_UNSET = object()


@dataclass(frozen=True)
class RecallRequest:
    """All business parameters for one recall call."""

    query: str
    top_k: int = 5
    use_lnn: bool = True
    sample: bool = False
    allow_ids: set[str] | None = None
    client: str | None = None
    include_superseded: bool = False
    use_mmr: bool = True
    annotate_conflicts: bool = True
    occurred_after: float = 0.0
    occurred_before: float = 0.0
    use_lexical: bool = True
    #: Value forwarded to ``manager.recall`` as ``query``. By default the
    #: request query is forwarded verbatim (brain.recall behaviour). The
    #: production service sets it to ``query or None`` to keep its old
    #: empty-query semantics without changing the brain side.
    manager_query: Any = _UNSET


@dataclass(frozen=True)
class RecallDeps:
    """Outer-layer symbols injected by the caller (domain stays pure)."""

    fuse: Callable[..., dict[str, float]]
    exp: Callable[[float], float]
    archived_state: str
    superseded_state: str
    hard_floor: float = RELEVANCE_HARD_FLOOR
    entity_bonus: float = ENTITY_BONUS
    literal_topn: int = MMR_LITERAL_EXEMPT_TOPN
    now_fn: Callable[[], float] = _now
    strip_fn: Callable[[dict[str, Any]], Any] = strip_internal_fields
    iso_fn: Callable[[float], str] = iso_date
    weekday_fn: Callable[[float], str] = weekday_abbr
    in_window_fn: Callable[..., bool] = in_time_window
    out_window_fn: Callable[..., bool] = out_of_window


def run_recall_pipeline(
    brain: Any,
    req: RecallRequest,
    deps: RecallDeps,
) -> list[dict[str, Any]]:
    """Compose the pure recall blocks and return the final result items.

    Mirrors the former ``HippoBrain.recall`` body exactly. Returns stripped
    items with ``occurred_at`` / ``weekday``; no accounting side effects.
    """
    mgr = brain.manager
    qtext = req.query or ""
    mquery = req.query if req.manager_query is _UNSET else req.manager_query
    fetch_k = max(req.top_k * 4, 12)
    raw = mgr.recall(
        query=mquery,
        top_k=fetch_k,
        use_lnn=req.use_lnn,
        sample=req.sample,
        allow_ids=req.allow_ids,
        client=req.client,
        occurred_after=req.occurred_after,
        occurred_before=req.occurred_before,
    )
    now = mgr._reference_time or deps.now_fn()

    query_ent_ids = _recall_flow.query_entity_ids(
        brain.extract_entities(qtext), brain.entities,
        brain._trusted_entity_ids())

    bm25_scores: dict[str, float] = {}
    if req.use_lexical and brain.use_bm25:
        fts_hits = brain._fts_hits(qtext, fetch_k * 2)
        if fts_hits is not None:
            bm25_scores = fts_hits
            brain._last_lexical_path = "fts5"
        else:
            brain._sync_bm25()
            if brain._bm25.docs:
                bm25_scores = brain._bm25.query(qtext, top_k=fetch_k * 2)
            brain._last_lexical_path = "memory"
        bm25_ranked = list(bm25_scores.keys())
    else:
        bm25_ranked = []
        brain._last_lexical_path = "off"

    fusion = _recall_flow.build_fusion(
        raw, bm25_ranked, bm25_weight=brain.bm25_weight, fuse=deps.fuse)

    # O8 recency window (fusion-lane fallback): vector candidates were already
    # filtered in manager.recall, this catches the lexical lane's rescues.
    candidate_ids = fusion.candidate_ids
    window_active = req.occurred_after > 0 or req.occurred_before > 0
    if brain._recency_window_seconds > 0 or window_active:
        kept: list[str] = []
        for cid in candidate_ids:
            rec = mgr.store.get(cid)
            if rec is None:
                continue
            if (brain._recency_window_seconds > 0 and not window_active
                    and deps.out_window_fn(rec.timestamp, rec.pin, now,
                                           brain._recency_window_seconds)):
                continue
            if window_active and not deps.in_window_fn(
                    rec.timestamp, req.occurred_after, req.occurred_before):
                continue
            kept.append(cid)
        candidate_ids = kept

    items = _recall_flow.collect_recall_items(
        candidate_ids,
        _recall_flow.RecallCandidateContext(
            allow_ids=req.allow_ids,
            include_superseded=req.include_superseded,
            archived_state=deps.archived_state,
            superseded_state=deps.superseded_state,
            hard_floor=deps.hard_floor,
            entity_bonus=deps.entity_bonus,
            literal_topn=deps.literal_topn,
            w_time=mgr.scorer.w_time,
            lambda_decay=brain.lambda_decay,
            nav_penalty=brain.nav_penalty,
            nav_tag_key=brain.nav_tag_key,
            bm25_scores=bm25_scores,
            query_ent_ids=query_ent_ids,
            vec_set=fusion.vec_set,
            fused=fusion.fused,
            vec_score_by_id=fusion.vec_score_by_id,
            sim_by_id=fusion.sim_by_id,
            get_record=mgr.store.get,
            embed=mgr.vector_of,
            exp=deps.exp,
            now=now,
        ))

    if req.use_mmr and len(items) > req.top_k:
        items = brain._mmr_select(items, req.top_k)
    else:
        items = sorted(items, key=lambda x: -x["score"])[:req.top_k]

    # Conflict annotation must run BEFORE internal fields are stripped (it
    # needs _emb) and AFTER sorting (only compute for what the caller sees).
    if req.annotate_conflicts:
        brain._annotate_conflicts(items)

    for it in items:
        deps.strip_fn(it)
        # Derived rather than replacing ``timestamp``: existing callers still
        # use the float for sorting, range compares and decay.
        it["occurred_at"] = deps.iso_fn(it["timestamp"])
        it["weekday"] = deps.weekday_fn(it["timestamp"])

    return items
