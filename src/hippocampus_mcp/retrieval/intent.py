# -*- coding: utf-8 -*-
"""Intent-aware query routing: zero-dependency lexical classification of
query intent + a weight routing table.

Motivation: queries of different intents have different needs for
「freshness vs similarity vs diversity」 -- factual Lookup wants
similarity to dominate with harsher diversity penalties; temporal
questions want freshness; exploration wants relaxed diversity
penalties. The current single weight combination treats all intents
alike.

Honest boundary: lexical rules are a **cheap first version**, with
limited accuracy on Chinese-writing-style queries; their value is
standing up the 「intent → weights」 route (skeleton + switch +
self-report), so a future classifier swap only touches
:func:`classify_intent`.

Switch: ``HIPPOCAMPUS_INTENT_ROUTING`` (default 0 = off). When off,
manager.recall does not even call classification and behavior is
bit-identical -- a hard guarantee that the evaluation baseline is
unaffected.
"""

from __future__ import annotations

import os

# Construction injection seam: the knob is registered in config
# (intent_routing); the production path injects it at engine import
# time; the resolution order (env first, injected value as fallback) is
# in common/config_bridge.
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

# Intent constants (general = fallback; routing leaves weights untouched; equivalent to being off)
INTENT_TEMPORAL = "temporal"
INTENT_LOOKUP = "lookup"
INTENT_EXPLORE = "explore"
INTENT_GENERAL = "general"

# Lexical trigger words (mainly for Chinese-writing scenarios; the intent
# with the most hits wins; ties/all-zero → general)
_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    INTENT_TEMPORAL: (
        "什么时候", "何时", "哪年", "哪月", "哪天", "哪一年", "多久",
        "最近", "之前", "后来", "先后", "顺序", "时间线", "历程",
    ),
    INTENT_LOOKUP: (
        "是谁", "是什么", "名字", "叫什么", "在哪里", "在哪儿", "哪个",
        "多少", "几个", "数量",
    ),
    INTENT_EXPLORE: (
        "为什么", "为何", "怎么", "如何", "原因", "解释", "分析", "影响",
    ),
}


def intent_routing_enabled() -> bool:
    """Whether intent routing is on (reads env every call, for easy test/runtime switching). Default off."""
    return _resolve(
        "HIPPOCAMPUS_INTENT_ROUTING", "intent_routing", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def classify_intent(query: str) -> str:
    """Classify the query into one of four intents by lexical rules.

    Scoring = hit counts of each intent's trigger words (the same word
    hit multiple times counts once);
    the highest wins; all-zero or tie → :data:`INTENT_GENERAL` (the
    fallback leaves weights untouched).
    """
    if not query:
        return INTENT_GENERAL
    scores: dict[str, int] = {}
    for intent, words in _INTENT_PATTERNS.items():
        scores[intent] = sum(1 for w in words if w in query)
    best = max(scores.values())
    if best <= 0:
        return INTENT_GENERAL
    winners = [k for k, v in scores.items() if v == best]
    return winners[0] if len(winners) == 1 else INTENT_GENERAL


# Routing table: multipliers on ScoringEngine's **current weights** (not
# absolute values -- library/config may have customized the weights;
# multiplier semantics keep routing valid on any baseline).
# general = empty table = same weights as when off.
INTENT_WEIGHT_MULTIPLIERS: dict[str, dict[str, float]] = {
    INTENT_TEMPORAL: {"w_time": 4.0, "w_recall": 0.5},
    INTENT_LOOKUP: {"w_sim": 1.05, "w_recall": 1.5},
    INTENT_EXPLORE: {"w_recall": 0.5},
    INTENT_GENERAL: {},
}
