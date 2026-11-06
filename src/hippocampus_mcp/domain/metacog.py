# -*- coding: utf-8 -*-
"""O7: making metacognitive confidence explicit (domain pure functions, M2
discipline).

Biological counterpart: metacognitive monitoring -- the hippocampus does not
only say "what was recalled"; it also says "how confident is this recall"
(knowing that it "cannot remember clearly").

Current gap: retrieval results carry no per-block confidence, so the host
cannot tell "confirmed" from "plausible-sounding" -- the last mile left
unmatched with the anti-hallucination goal (L0 back-reference verifiable).

Confidence formula (**all calibration anchors use existing in-repo
constants**, no separate scale is erected):

    conf = W_SIM   · min(sim / SUPERSEDE_SIM, 1)     # 0.5 -- similarity:
                                                          reaching the
                                                          "same fact" tier
                                                          (0.72) is full
                                                          marks
         + W_STATE · state_score                      # 0.2 -- lifecycle:
                                                          active 1.0 /
                                                          superseded 0.2 /
                                                          archived 0.0
         + W_SRC   · source_resolved_ratio            # 0.2 -- L0
                                                          resolvability:
                                                          the fraction of
                                                          source_ids that
                                                          can be matched in
                                                          raw_log; **no
                                                          provenance = 0**
         + W_PIN   · (1.0 if pin else 0.0)            # 0.1 -- pinned =
                                                          confirmed

Calibration sanity (two of the acceptance spot checks; the formula
guarantees they always hold):

* pinned above ordinary: at equal sim / state / provenance, pin adds +0.1;
* no provenance below provenanced: when ``source_ids`` is empty the W_SRC
  term scores 0; when present and resolvable it scores full.

BM25-only hits (``similarity`` always 0.0) naturally land in the
low-confidence region -- a literal collision the vector channel cannot
confirm; exactly the score "plausible-sounding" deserves.

Self-report carries the receipt discipline: the response body carries a
``confidence`` receipt block (each block's value / basis / the full inputs /
weights / computed_at), answering "how were these confidences computed";
per-hit values sit on ``hits[i]["confidence"]``.
**No new tools, no changed existing keys** -- only new keys are added;
backward compatible.

Dependency direction: imports only common constants (L0).
"""
from __future__ import annotations

from typing import Any, Iterable

from ..common.constants import SUPERSEDE_SIM

__all__ = [
    "W_SIM", "W_STATE", "W_SRC", "W_PIN", "STATE_SCORE", "FIRST_PARTY_SCORE",
    "hit_confidence", "mean_confidence", "confidence_receipt",
]

W_SIM: float = 0.5
W_STATE: float = 0.2
W_SRC: float = 0.2
W_PIN: float = 0.1

#: Lifecycle state scores. superseded is not 0: a superseded old criterion is
#: still "something really said" (it has value as a historical fact to cite);
#: it just cannot be used as the current fact.
STATE_SCORE: dict[str, float] = {
    "active": 1.0,
    "superseded": 0.2,
    "archived": 0.0,
}

#: Base score for each non-empty block: block_2/3/5 come from the first-party
#: layer (intents / summaries / open_topics are books this system kept
#: itself, not retrieved), given a full 1.0 with the basis spelled out --
#: this is what makes the "confirmed" vs "needs verification" distinction
#: meaningful.
FIRST_PARTY_SCORE: float = 1.0


def hit_confidence(
    *,
    similarity: float,
    state: str,
    source_ids: Iterable[str],
    source_resolved: int,
    pin: bool,
) -> float:
    """Confidence of a single hit (0~1, four decimal places). For the input
    criteria see the module docstring.

    Args:
        similarity: vector similarity (BM25-only hits are always 0.0 --
            counted as "unconfirmable").
        state: lifecycle state (unknown states count 0.5; neither
            arbitrarily boosted nor zeroed).
        source_ids: L0 back-reference id list (empty = no provenance; that
            term scores 0).
        source_resolved: how many of them can be matched in the L0 raw_log.
        pin: whether pinned.
    """
    sim_n = max(0.0, min(float(similarity) / SUPERSEDE_SIM, 1.0))
    state_score = STATE_SCORE.get(str(state or ""), 0.5)
    ids = [s for s in (source_ids or []) if s]
    if not ids:
        src_score = 0.0
    else:
        src_score = max(0.0, min(float(source_resolved) / len(ids), 1.0))
    v = (W_SIM * sim_n + W_STATE * state_score
         + W_SRC * src_score + W_PIN * (1.0 if pin else 0.0))
    return round(min(max(v, 0.0), 1.0), 4)


def mean_confidence(values: Iterable[float]) -> float | None:
    """Block-level aggregation of per-hit confidences (arithmetic mean, four
    decimal places; empty set = None).

    Mean rather than max: a block-level "confidence" vouches for the **whole
    block**; if one plausible-sounding hit sneaks in among confirmed hits,
    the block score should drop -- the host can then drill down to per-hit
    ``confidence`` to find out which one.
    """
    vals = [float(v) for v in values]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def confidence_receipt(
    block_values: dict[str, dict[str, Any]],
    *,
    computed_at: float,
) -> dict[str, Any]:
    """K4 receipt-style confidence self-report block (a new key of the
    get_memory_context response body).

    Args:
        block_values: ``{block name: {"value": float|None, "basis": str}}``.
        computed_at: computation time (Unix seconds).

    Returns:
        ``{"blocks": ..., "inputs": [...], "weights": ..., "per_hit_key":
        "confidence", "computed_at": ..., "note": ...}``.
    """
    return {
        "blocks": dict(block_values),
        "inputs": ["similarity", "state", "source_ids_resolvable", "pin"],
        "weights": {"similarity": W_SIM, "state": W_STATE,
                    "source_ids_resolvable": W_SRC, "pin": W_PIN},
        "per_hit_key": "confidence",
        "computed_at": computed_at,
        "note": ("value=None = 该块不可评估（空块 / 派生层无法逐条溯源），"
                 "不是 0 分。block_1 为逐条命中的平均；校准锚点：相似度以"
                 "SUPERSEDE_SIM（同一事实档）为满分线，pinned 高于普通、"
                 "无溯源低于有溯源。"),
    }
