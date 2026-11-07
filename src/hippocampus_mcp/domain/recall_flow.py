# -*- coding: utf-8 -*-
"""The three inlined blocks of ``brain.core.recall`` (extracted from core).

**Why worth moving**: recall's shell mixes three unrelated
responsibilities -- query entity collection (about 6 lines), the vector/
literal two-channel fusion (about 15 lines), and candidate filtering plus
score assembly (about 68 lines). The third block is the bulk: hard-gate
filtering + hit attribution + four-way bonuses + navigation downweighting +
response-body assembly, all pure dict/list operations, yet untestable in
isolation because it reads a dozen facets of ``self.manager``. After the
move, ``recall`` keeps only the things that truly need brain state:
"empty-library short circuit / tag allowlist / fetch two channels of
candidates / call manager / MMR and annotation orchestration / bookkeeping
at three side-effect points".

**Dependency direction**: this module imports only the standard library
(domain → common-only discipline; here not even common is used -- all
constants, callbacks, and ``np.exp`` are fetched by the shell at call time
and injected).

**Replicated quirks (touching any one is a behavior change; the parity
tests will catch it)**:

1. ``literal_ids`` is taken **from the same source** as the old
   implementation: the old shell first computed
   ``bm25_ranked = list(bm25_scores.keys())``, then took the top
   ``MMR_LITERAL_EXEMPT_TOPN``. This module computes directly from the key
   order of ``bm25_scores`` -- with ``use_bm25=False`` both sides are empty,
   and the ``fts5`` branch likewise both follow the key order of
   ``dict(hits)``; bit-for-bit equivalent.
2. The response body carries ``literal`` / ``_literal`` / ``_emb`` -- **three
   keys coexist** (``_literal`` duplicates ``literal``; historical
   redundancy). ``_emb`` feeds :meth:`_mmr_select`'s diversity penalty and
   ``_annotate_conflicts``'s neighbor search; deleting either key changes
   downstream behavior.
3. ``_emb`` is only fetched after a candidate **passes all filters** (the
   old code called ``manager.vector_of`` inside append); extracted into
   :meth:`RecallCandidateContext.embed`, the same laziness is preserved --
   eliminated candidates never touch the embedding model.
4. ``hit_entity`` is the short-circuit form of ``bool(query_ent_ids) and
   bool(intersection)``: when ``query_ent_ids`` is empty, ``rec.entity_ids``
   is **not touched**.
5. ``matched_by`` is collected in a fixed order (vector → bm25 → entity →
   pin) and then emitted via ``sorted()``; the order itself does not affect
   the result, but missing a channel loses attribution.
6. The pinned exemption's exponential decay is **wrapped in an outer
   ``float()``** (``np.exp`` returns ``np.float64``; without the cast a
   numpy scalar would leak into the response body).
7. Navigation downweighting **multiplies the final score** and is
   double-gated: it applies only when ``nav_penalty < 1.0`` and the tag key
   is truthy; with the default ``nav_penalty == 1.0`` the whole block is
   skipped (score untouched).
8. The hard-gate exemption recognizes only the literal channel: a candidate
   is eliminated only when ``sim < hard_floor`` and ``did not in
   bm25_scores`` -- an entry hit by BM25 alone must not be killed even
   though its ``similarity`` is 0.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


# ======================================================================
# Block A: query entities (old shell 1451-1456)
# ======================================================================
def query_entity_ids(
    names: list[str],
    entities: dict[str, Any],
    trusted_ids: set[str],
) -> set[str]:
    """Resolve entity names in the query into a **trusted** set of entity
    ids.

    Args:
        names: output of :meth:`HippoBrain.extract_entities` (entity names in
            four-channel order).
        entities: the entity table (``brain.entities``, keyed by lowercase
            entity name).
        trusted_ids: allowlist of entity ids from trusted sources
            (heuristically produced entities do not participate in bonuses).

    Returns:
        Deduplicated set of entity ids. Empty when the query contains no
        trusted entity -- the caller uses that to short-circuit
        :meth:`collect_recall_items`'s entity bonus.

    Replicated quirk: the lookup key is ``name.lower()``, **not** the name
    itself; entries in the entity table missing ``id`` (``ent["id"]``) raise
    KeyError -- existing behavior, no fallback added.
    """
    out: set[str] = set()
    for name in names:
        ent = entities.get(name.lower())
        if ent and ent["id"] in trusted_ids:
            out.add(ent["id"])
    return out


# ======================================================================
# Block B: two-channel fusion (old shell 1484-1500)
# ======================================================================
@dataclass(frozen=True)
class RecallFusion:
    """Fusion intermediate state of the vector channel and the literal
    channel.

    Attributes:
        vec_ranked: vector-channel ranking (original order, RRF input).
        vec_set: set form of the above, used only for hit attribution.
        fused: RRF fused scores (with the BM25 weight), **only ids that
            participated in the fusion**.
        candidate_ids: union of vector ∪ literal channels (vector first,
            literal appended at the tail).
        sim_by_id: vector similarity (entries fetched by the literal channel
            alone get 0.0).
        vec_score_by_id: raw vector-channel scores (secondary ranking
            signal, weight 0.1).
    """
    vec_ranked: list[str]
    vec_set: set[str]
    fused: dict[str, float]
    candidate_ids: list[str]
    sim_by_id: dict[str, float]
    vec_score_by_id: dict[str, float]


def build_fusion(
    raw: list[Any],
    bm25_ranked: list[str],
    *,
    bm25_weight: float,
    fuse: Callable[[list[list[str]], Any], dict[str, float]],
) -> RecallFusion:
    """Fuse the two channels of candidates into a unified scoring/union view
    (the whole old-shell block 1484-1500).

    Args:
        raw: return value of manager.recall (duck-typed: the three
            attributes ``memory_id`` / ``similarity`` / ``score``).
        bm25_ranked: literal-channel ranking; **an empty list means the
            literal channel is off**, in which case ``fuse`` receives a
            single channel (the weight table degrades to ``[1.0]``).
        bm25_weight: the literal channel's weight in RRF
            (``brain.bm25_weight``).
        fuse: :func:`~hippocampus_mcp.retrieval.bm25.rrf_fuse` itself
            (retrieval is outside domain's allowed dependencies; injected
            per iron rule no. 16).

    Returns:
        :class:`RecallFusion`. With empty ``raw`` and empty ``bm25_ranked``
        everything is empty and the caller's loop naturally yields nothing.

    Replicated quirk: the call shape of ``fuse`` branches on the **truthiness**
    of ``bm25_ranked`` -- with an empty list the weight table has the single
    entry ``[1.0]``, verbatim identical to the old shell (not the
    ``[1.0, w]`` of an empty channel).
    """
    vec_ranked = [r.memory_id for r in raw]
    vec_set = set(vec_ranked)
    if bm25_ranked:
        fused = fuse([vec_ranked, bm25_ranked], weights=[1.0, bm25_weight])
    else:
        fused = fuse([vec_ranked], weights=[1.0])

    candidate_ids = list(vec_ranked)
    for did in bm25_ranked:
        if did not in candidate_ids:
            candidate_ids.append(did)

    sim_by_id = {r.memory_id: r.similarity for r in raw}
    vec_score_by_id = {r.memory_id: r.score for r in raw}
    return RecallFusion(
        vec_ranked=vec_ranked,
        vec_set=vec_set,
        fused=fused,
        candidate_ids=candidate_ids,
        sim_by_id=sim_by_id,
        vec_score_by_id=vec_score_by_id,
    )


# ======================================================================
# Block C: candidate filtering and score assembly (old shell 1502-1570)
# ======================================================================
@dataclass(frozen=True)
class RecallCandidateContext:
    """Every knob and callback needed to traverse candidates (fetched by the
    shell at call time and injected in one go).

    Attributes:
        allow_ids: tag allowlist; ``None`` means no filtering (**not an
            empty set**).
        include_superseded: whether superseded records are admitted.
        archived_state: ``MemoryState.ARCHIVED.value`` -- state is a string,
            injected by value (store.memory is outside domain's dependency
            scope).
        superseded_state: same, ``MemoryState.SUPERSEDED.value``.
        hard_floor: relevance hard gate (``RELEVANCE_HARD_FLOOR``).
        entity_bonus: entity-hit bonus (``ENTITY_BONUS``).
        literal_topn: literal strong-hit exemption rank
            (``MMR_LITERAL_EXEMPT_TOPN``); with ``<= 0`` the exemption set is
            empty.
        w_time: time-decay weight (``manager.scorer.w_time``, used to
            backfill the pinned exemption).
        lambda_decay: decay time constant (``brain.lambda_decay``).
        nav_penalty: navigation-content discount (effective only when
            ``< 1.0``).
        nav_tag_key: navigation tag key (``brain.nav_tag_key``).
        bm25_scores: literal-channel score dict (**key order is the
            ranking**).
        query_ent_ids: query entity id set (Block A's output).
        vec_set: vector-channel id set (for attribution).
        fused: RRF fused scores (Block B's output).
        vec_score_by_id: raw vector-channel scores (secondary signal).
        sim_by_id: vector similarity (0.0 when missing).
        get_record: ``manager.store.get`` itself (duck-typed return value,
            may be None).
        embed: ``manager.vector_of`` itself (**lazy**: called only for
            candidates that pass the filters).
        exp: ``np.exp`` itself (numpy is outside domain's dependency scope).
        now: time base (``manager._reference_time or _now()``, computed at
            call time).
        usage_w: UR usage-recency master gate (0.0 = off, the default).
        usage_tau: UR recency time constant in SECONDS (config carries
            days; callers convert).
        usage_cap: UR log1p(strength) cap (<= 0 disables the item).
    """
    allow_ids: set[str] | None
    include_superseded: bool
    archived_state: str
    superseded_state: str
    hard_floor: float
    entity_bonus: float
    literal_topn: int
    w_time: float
    lambda_decay: float
    nav_penalty: float
    nav_tag_key: str
    bm25_scores: dict[str, float]
    query_ent_ids: set[str]
    vec_set: set[str]
    fused: dict[str, float]
    vec_score_by_id: dict[str, float]
    sim_by_id: dict[str, float]
    get_record: Callable[[str], Any]
    embed: Callable[[Any], Any]
    exp: Callable[[float], float]
    now: float
    # UR usage-recency (defaults keep every pre-UR construction site valid
    # and bit-identical: w=0.0 short-circuits the whole item).
    usage_w: float = 0.0
    usage_tau: float = 0.0
    usage_cap: int = 0


def collect_recall_items(
    candidate_ids: list[str],
    ctx: RecallCandidateContext,
) -> list[dict[str, Any]]:
    """Filter candidates, score, and assemble the response body in the old
    shell's 1502-1570 order.

    Returns:
        Entries that passed all filters (**order is the traversal order of
        candidate_ids**; MMR and the final truncation happen in the shell;
        this does not cut ``top_k``). Each entry carries the two internal
        fields ``_emb`` / ``_literal`` for :meth:`_mmr_select` and
        :meth:`_annotate_conflicts` to consume, cleaned up uniformly by
        ``strip_internal_fields`` at the end -- this function is **not
        responsible for the cleanup**.

    Replicated quirks: see module docstring items 2-8; among them:

    - the hard-gate exemption recognizes only the literal channel (``did not
      in ctx.bm25_scores``);
    - the pinned exemption and the entity bonus are stacked **in this
      order** after the fused score and before navigation downweighting;
    - navigation downweighting is multiplicative, and with ``nav_penalty``
      at 1.0 (default) the whole block is skipped.
    """
    # Literal strong-hit exemption set: key order is the ranking (see module
    # docstring item 1)
    if ctx.literal_topn > 0:
        literal_ids = set(list(ctx.bm25_scores.keys())[:ctx.literal_topn])
    else:
        literal_ids = set()

    items: list[dict[str, Any]] = []
    for did in candidate_ids:
        rec = ctx.get_record(did)
        if rec is None:
            continue
        if ctx.allow_ids is not None and did not in ctx.allow_ids:
            continue
        if rec.state == ctx.archived_state:
            continue
        if rec.state == ctx.superseded_state and not ctx.include_superseded:
            continue
        # D2a write-side isolation: quarantined cards stay out of the recall
        # pool -- this is the **single enforcement point** of the "kept out
        # of the recall pool" promise (brain.recall is the one call path;
        # the scope_recall user-library flow passes through here too).
        # getattr fallback: old record objects lacking the field are treated
        # as not quarantined.
        if getattr(rec, "quarantine", False):
            continue
        sim = ctx.sim_by_id.get(did, 0.0)
        # Relevance hard gate: missed by both vector and BM25 → eliminate
        # (prevents the summary layer from dominating the board)
        if sim < ctx.hard_floor and did not in ctx.bm25_scores:
            continue

        # Hit attribution: one entry per channel of signal, for the response
        # body's matched_by
        hit_entity = bool(ctx.query_ent_ids) and bool(
            ctx.query_ent_ids & set(rec.entity_ids or [])
        )
        srcs = []
        if did in ctx.vec_set:
            srcs.append("vector")
        if did in ctx.bm25_scores:
            srcs.append("bm25")
        if hit_entity:
            srcs.append("entity")
        if rec.pin:
            srcs.append("pin")

        # Primary ranking signal = RRF fused score; vector score as the
        # secondary signal
        score = ctx.fused.get(did, 0.0) + 0.1 * ctx.vec_score_by_id.get(did, 0.0)
        # Pinned memories are exempt from time decay (backfills w_time's
        # maximum possible contribution)
        if rec.pin:
            age = max(ctx.now - rec.timestamp, 0.0)
            score += ctx.w_time * (
                1.0 - float(ctx.exp(-age / ctx.lambda_decay))
            )
        # Entity-hit bonus (bridging the semantic gap: low vector similarity
        # but the entity matches)
        if hit_entity:
            score += ctx.entity_bonus
        # UR usage-recency boost (scoring layer only -- no write-path
        # change): "recently AND frequently recalled", anchored on
        # ``last_recalled_at`` (written only by ``stamp_recalled``, i.e.
        # true result-set membership) times capped retrieval strength.
        # Fail-safe degradation to zero: disabled knob (w <= 0), illegal
        # tau/cap, never-recalled records (last_recalled_at <= 0), zero
        # strength, and negative delta-t (clock skew / pinned reference
        # time -- clamped, same guard as the pinned exemption above, so
        # exp never exceeds 1 and the cap bound holds).
        usage_boost = 0.0
        # getattr fallback for the three UR fields: hand-rolled fake
        # contexts (test SimpleNamespace sites) predate the fields -- same
        # idiom as the quarantine read above ("lacking the field = off").
        usage_w = getattr(ctx, "usage_w", 0.0)
        if usage_w > 0.0:
            last_used = float(rec.last_recalled_at or 0.0)
            strength = max(float(rec.recall_strength or 0.0), 0.0)
            usage_tau = getattr(ctx, "usage_tau", 0.0)
            usage_cap = getattr(ctx, "usage_cap", 0)
            if (last_used > 0.0 and strength > 0.0
                    and usage_tau > 0.0 and usage_cap > 0):
                delta_t = max(ctx.now - last_used, 0.0)
                usage_boost = usage_w * float(
                    ctx.exp(-delta_t / usage_tau)
                ) * math.log1p(min(strength, float(usage_cap)))
        if usage_boost > 0.0:
            score += usage_boost
            # Attribution must not hide the boost (D7 matched_by spirit:
            # "why this entry ranked here" stays explainable).
            srcs.append("usage")
        # Navigation-content downweighting: the default penalty=1.0 is a
        # no-op; it only kicks in when the import side explicitly tagged
        # NAV_TAG_KEY and configured a discount, avoiding default-behavior
        # changes.
        if ctx.nav_penalty < 1.0 and (rec.tags or {}).get(ctx.nav_tag_key):
            score *= ctx.nav_penalty
        items.append({
            "memory_id": rec.id,
            "content": rec.content,
            "score": score,
            "similarity": sim,
            "timestamp": rec.timestamp,
            "importance": rec.importance,
            "state": rec.state,
            "cognitive_type": rec.cognitive_type,
            "source_ids": rec.source_ids or [],
            "entity_ids": rec.entity_ids or [],
            "pin": rec.pin,
            "source": rec.source or "",
            "tags": dict(rec.tags or {}),
            # D7 (ZCode addendum three): **why** this entry was recalled. In
            # multi-channel hybrid retrieval, "high score" does not mean
            # "the retriever considered it relevant" -- it may be BM25
            # literally hitting a stopword, or the entity/pin bonus pushing
            # it up. Attribution is what makes false recalls explainable,
            # and the basis for tuning weights.
            "matched_by": sorted(srcs),
            "literal": did in literal_ids,
            "_emb": ctx.embed(rec),
            "_literal": did in literal_ids,
        })
    return items
