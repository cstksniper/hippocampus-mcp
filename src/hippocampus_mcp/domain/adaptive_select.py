# -*- coding: utf-8 -*-
"""Adaptive result-count selector for the recall pipeline (2026-11-07).

Design doc: ``docs/design/自适应召回条数_设计书_20261107.md`` (repo-only,
Chinese). This module is the Stage-C selector: it receives the *candidate
pool* that already passed every hard gate (relevance floor / tags / states /
quarantine / sensitivity) and decides HOW MANY entries to return — up to the
caller's ``top_k`` — by marginal utility, instead of always filling ``top_k``
the way fixed-mode MMR does.

Why a new module and not a tweak to ``brain._mmr_select``:

1. **Scale mismatch.** The fused ranking score spans ~0.01-0.12 while the
   cosine redundancy penalty spans 0-1; in the fixed selector the redundancy
   term dominates by an order of magnitude. Here relevance is NORMALIZED
   within the query (``r = score / score_top``, in (0, 1]) so both terms
   share a scale. The normalized ``r`` is an INTERNAL stopping signal only —
   it is never emitted as a relevance probability (RRF scores are rank
   fusions, not calibrated probabilities; Cormack et al., SIGIR 2009).
2. **No early stopping in fixed MMR.** ``_mmr_select`` always fills
   ``top_k``; here the loop stops when the best remaining marginal utility
   drops below ``tau_stop`` or the next candidate is not "high relevance"
   enough after ``k_min`` picks (``expand_floor``).
3. **Redundancy must cover vectorless candidates.** Fixed MMR gives
   BM25-only hits (no embedding) zero diversity penalty. The selector adds a
   character n-gram Jaccard channel so exact/near duplicates without vectors
   still collapse.
4. **Conflicts and complementary facts are not redundancy.** Candidates
   sharing a ``conflict_group`` are never deduplicated against each other
   (both accounts must stay visible), and two highly-similar entries with
   *different* numbers are treated as complementary updates, not repeats.

Purity: domain depends only on the standard library. Numpy vectors arrive
pre-attached on the items (``_emb``, as produced by ``collect_recall_items``);
cosine is computed with plain Python here to keep this module importable in
stdlib-only contexts — candidate pools are small (tens), so the O(k*n) dot
products are negligible.

The selector NEVER supersedes, stamps, or writes anything: it is a pure
selection function (same side-effect discipline as the shared pipeline).
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

__all__ = ["select_adaptive", "normalize_relevance", "ngram_jaccard"]

#: Fallback relevance when every candidate score is zero (all-BM25 extreme):
#: use reciprocal rank so ordering information still drives selection.
_FALLBACK_RANK_FLOOR = 0.1

#: Numbers are extracted to tell "same fact re-stated" apart from "same
#: topic, updated figures". Two high-similarity entries with disjoint
#: number sets are complementary, not redundant. CJK numeral RUNS are kept
#: as raw substrings (「三十」 vs 「五十」 compare unequal without any
#: value parsing — cross-notation "30" vs 「三十」 will not match, an
#: accepted v1 limitation; restatements virtually always share notation).
_NUM_RE = re.compile(r"[0-9０-９]+(?:[.,][0-9０-９]+)*|[零〇一二两三四五六七八九十百千万亿]+")

#: When both entries carry numbers and the sets are disjoint, redundancy is
#: capped at this value (they still might partially overlap, so not forced
#: to zero — but they will usually pass the tau gate).
_COMPLEMENTARY_RED_CAP = 0.5

#: Cosine values below this are treated as "no measurable overlap" for the
#: missing-embedding path (a normalized dot of near-orthogonal short texts).
_COS_EPS = 1e-9


def _norm_cos(a: Any, b: Any) -> float:
    """Cosine of two pre-attached embedding vectors (plain Python).

    Either side missing (BM25-only rescue) returns 0.0 — the n-gram channel
    covers that case instead.
    """
    if a is None or b is None:
        return 0.0
    try:
        la = math.sqrt(sum(float(x) * float(x) for x in a))
        lb = math.sqrt(sum(float(x) * float(x) for x in b))
        if la <= _COS_EPS or lb <= _COS_EPS:
            return 0.0
        return sum(float(x) * float(y) for x, y in zip(a, b)) / (la * lb)
    except (TypeError, ValueError):
        return 0.0


def _grams(text: str, n: int = 3) -> set[str]:
    """Character n-gram set (CJK-friendly; whitespace collapsed)."""
    t = re.sub(r"\s+", "", str(text or "").lower())
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Jaccard of character n-gram sets; 0.0 when either side is empty."""
    ga, gb = _grams(a, n), _grams(b, n)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    if inter == 0:
        return 0.0
    return inter / len(ga | gb)


def _numbers(text: str) -> set[str]:
    """Normalized number tokens (comma/dot decimals collapse to one form)."""
    out: set[str] = set()
    for m in _NUM_RE.findall(str(text or "")):
        out.add(m.replace(",", "."))
    return out


def _entity_jaccard(a: list, b: list) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize_relevance(items: list[dict[str, Any]]) -> dict[str, float]:
    """Within-query relevance normalization: ``r = score / score_top``.

    Returns ``{memory_id: r}`` with values in (0, 1]. When every score is
    zero (the all-BM25-only extreme), falls back to reciprocal rank
    (``1 / rank + 1``, floored at ``_FALLBACK_RANK_FLOOR``) so ordering
    still matters. The output is an internal stopping signal — never a
    user-facing relevance probability.
    """
    scores = [max(float(it.get("score") or 0.0), 0.0) for it in items]
    top = max(scores) if scores else 0.0
    out: dict[str, float] = {}
    if top > 0.0:
        for it, s in zip(items, scores):
            r = s / top
            out[str(it.get("memory_id"))] = r if r > 0.0 else _FALLBACK_RANK_FLOOR
        return out
    for rank, it in enumerate(items):
        out[str(it.get("memory_id"))] = max(1.0 / (rank + 2), _FALLBACK_RANK_FLOOR)
    return out


def _pair_redundancy(cand: dict[str, Any], sel: dict[str, Any],
                     cand_num: set[str], sel_num: set[str]) -> float:
    """Redundancy between one candidate and one selected entry.

    Channels (max): embedding cosine, character 3-gram Jaccard, entity
    Jaccard. Guards:

    - same ``conflict_group`` ⇒ 0.0 (both accounts stay visible);
    - both sides carry numbers and the sets are disjoint ⇒ capped at
      ``_COMPLEMENTARY_RED_CAP`` (updated figures, not a restatement);
    - missing embeddings degrade to the n-gram channel (fixed MMR's blind
      spot for BM25-only rescues).
    """
    cg_a, cg_b = cand.get("conflict_group"), sel.get("conflict_group")
    if cg_a and cg_a == cg_b:
        return 0.0
    cos = _norm_cos(cand.get("_emb"), sel.get("_emb"))
    jac = ngram_jaccard(cand.get("content"), sel.get("content"))
    ent = _entity_jaccard(cand.get("entity_ids"), sel.get("entity_ids"))
    red = max(cos, jac, ent if ent > 0.8 else 0.0)
    if (red > 0.0 and cand_num and sel_num
            and not (cand_num & sel_num)):
        red = min(red, _COMPLEMENTARY_RED_CAP)
    return red


def _max_redundancy(cand: dict[str, Any], selected: list[dict[str, Any]]) -> tuple[float, int]:
    cand_num = _numbers(cand.get("content"))
    worst, worst_i = 0.0, -1
    for i, sel in enumerate(selected):
        r = _pair_redundancy(cand, sel, cand_num, _numbers(sel.get("content")))
        if r > worst:
            worst, worst_i = r, i
    return worst, worst_i


def _facet_gain(cand: dict[str, Any], covered: set[str],
                query_keys: set[str]) -> float:
    """Coverage increment for query entities/tags not yet represented."""
    if not query_keys:
        return 0.0
    have = {str(e) for e in (cand.get("entity_ids") or [])}
    have |= {str(k) for k in ((cand.get("tags") or {}).keys())}
    fresh = (have & query_keys) - covered
    return len(fresh) / len(query_keys)


def select_adaptive(
    items: list[dict[str, Any]],
    *,
    top_k: int,
    k_min: int = 2,
    budget_hard_chars: int = 2400,
    expand_floor: float = 0.6,
    tau_stop: float = 0.1,
    dup_floor: float = 0.9,
    lam: float = 0.7,
    beta: float = 0.15,
    gamma: float = 0.1,
    query_ent_ids: set[str] | None = None,
    query_tags: set[str] | None = None,
    sensitive_pred: Callable[[Any], bool] | None = None,
    get_record: Callable[[str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stage-C adaptive selection over a hard-gated candidate pool.

    Args:
        items: candidate pool from ``collect_recall_items`` (internal
            ``_emb`` fields still attached; NOT yet stripped).
        top_k: hard cap on the number of selected entries (the caller's
            ``top_k`` semantics become "maximum", per the design doc).
        k_min: picks before the ``expand_floor`` gate applies (keeps the
            small-result behaviour of the fixed path when the pool is thin).
        budget_hard_chars: total content-character budget (render-side
            convention: Chinese ≈ 2 chars/token, so chars are the currency).
        expand_floor: minimum normalized relevance ``r`` for picks after
            ``k_min`` — the "only expand on high relevance" gate.
        tau_stop: marginal-utility stop threshold.
        dup_floor: HARD collapse gate — a candidate whose max redundancy
            against the selected set reaches this is dropped regardless of
            utility. The soft ``(1-lam)`` penalty alone cannot suppress a
            near-exact duplicate of a highly-relevant pick (0.3 penalty vs
            0.7*r reward), and the design's precondition is "expand only
            when the entries are NOT repetitive". Conflicts and
            complementary-number pairs never reach this gate (their
            redundancy is already floored in ``_pair_redundancy``).
        lam / beta / gamma: relevance / coverage / cost weights.
        query_ent_ids / query_tags: coverage targets for the facet term.
        sensitive_pred: when given, candidates whose record matches are
            dropped BEFORE selection (masked-tier backfill: safe candidates
            fill the freed slots instead of the count silently shrinking).
        get_record: record lookup for ``sensitive_pred`` (candidate dicts do
            not carry the sensitivity flag).

    Returns:
        ``(selected_items, report)``. The report is the ``selection``
        self-report for the tool envelope: counts, budget, stop reason and
        why entries were omitted — never raw scores.

    Purity: no stamping, no writes, no exceptions on degenerate input (an
    empty pool returns an empty selection with ``pool_empty``).
    """
    query_keys: set[str] = {str(e) for e in (query_ent_ids or set())}
    query_keys |= {str(t) for t in (query_tags or set())}

    pool = list(items)
    filtered_sensitive = 0
    if sensitive_pred is not None and get_record is not None:
        kept: list[dict[str, Any]] = []
        for it in pool:
            rec = get_record(str(it.get("memory_id")))
            try:
                hit = bool(rec is not None and sensitive_pred(rec))
            except Exception:  # noqa: BLE001 — a broken predicate must not fail recall
                hit = False
            if hit:
                filtered_sensitive += 1
                continue
            kept.append(it)
        pool = kept

    report: dict[str, Any] = {
        "mode": "adaptive",
        "candidates": len(items),
        "eligible": len(pool),
        "selected": 0,
        "budget_used_chars": 0,
        "budget_hard_chars": int(budget_hard_chars),
        "stop_reason": "pool_empty",
        "omitted": {"low_relevance": 0, "redundant": 0, "budget_or_cap": 0},
        "has_more_high_relevance": False,
        "filtered_sensitive": filtered_sensitive,
    }
    if not pool or top_k <= 0 or budget_hard_chars <= 0:
        if top_k <= 0 or budget_hard_chars <= 0:
            report["stop_reason"] = "budget_or_cap"
        return [], report

    rel = normalize_relevance(pool)
    ordered = sorted(pool, key=lambda it: (-rel.get(str(it.get("memory_id")), 0.0),
                                           str(it.get("memory_id"))))
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    used = 0
    reason = "pool_empty"
    # stop_reason semantics: WHY THE NEXT-BEST CANDIDATE did not make it —
    # the rejection reason of the FIRST (highest-relevance) rejected entry.
    # Later, weaker rejections must not overwrite it.
    stop_locked = False

    for cand in ordered:
        mid = str(cand.get("memory_id"))
        r = rel.get(mid, 0.0)
        cost = max(1, len(str(cand.get("content") or "")) // 2)
        if len(selected) >= top_k:
            reason = "max_k"
            report["omitted"]["budget_or_cap"] += 1
            continue
        if used + cost > budget_hard_chars and selected:
            reason = "budget"
            report["omitted"]["budget_or_cap"] += 1
            continue
        if selected:
            red, _ = _max_redundancy(cand, selected)
        else:
            red = 0.0
        gain = _facet_gain(cand, covered, query_keys)
        util = (lam * r - (1.0 - lam) * red + beta * gain
                - gamma * (cost / max(budget_hard_chars - used, 1)))
        first_picks = len(selected) < max(int(k_min), 1)
        if red >= dup_floor:
            report["omitted"]["redundant"] += 1
            if not stop_locked:
                reason = "low_marginal_utility"
                stop_locked = True
            continue
        if util < tau_stop:
            report["omitted"]["redundant" if red > 0.5 else "low_relevance"] += 1
            if not stop_locked:
                reason = "low_marginal_utility"
                stop_locked = True
            continue
        if not first_picks and r < expand_floor:
            report["omitted"]["low_relevance"] += 1
            if not stop_locked:
                reason = "below_expand_floor"
                stop_locked = True
            continue
        selected.append(cand)
        used += cost
        reason = "pool_exhausted"
        have = {str(e) for e in (cand.get("entity_ids") or [])}
        have |= {str(k) for k in ((cand.get("tags") or {}).keys())}
        covered |= have & query_keys

    # "High relevance left on the table" = an unselected pool entry with
    # r >= expand_floor that was cut by budget or cap (not by redundancy) —
    # the hook surfaces this as a "deep-search available" hint.
    leftovers = [it for it in ordered if it not in selected]
    high_left = 0
    for it in leftovers:
        if rel.get(str(it.get("memory_id")), 0.0) >= expand_floor:
            red, _ = _max_redundancy(it, selected) if selected else (0.0, -1)
            if red < 0.5:
                high_left += 1
    report.update({
        "selected": len(selected),
        "budget_used_chars": used,
        "stop_reason": reason,
        "has_more_high_relevance": high_left > 0,
    })
    return selected, report
