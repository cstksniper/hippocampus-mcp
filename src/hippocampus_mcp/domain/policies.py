# -*- coding: utf-8 -*-
"""M2: judgment-policy pure functions (the "how to judge" that touches no I/O and no instance state)

Moved verbatim from the brain layer (pure.py / core.py). Guarantees that the
move causes **zero behavior change**:
1. The function bodies in this module are copied verbatim, with only the
   ``cls.`` prefix changed to module-level direct calls;
2. ``tests/golden/policies_golden.json`` is a snapshot of the pre-move behavior
   (44 cases, including threshold boundaries); ``tests/test_policies_golden.py``
   asserts them case by case;
3. The original locations keep same-named delegates (pure.py's
   staticmethod/classmethod, core.py's ``_is_restatement`` wrapper); call
   sites are untouched line by line.

Why worth moving: these judgment cores were buried in the 6299-line core.py /
Mixin, where the policy logic (thresholds, word lists, gate conditions) could
not be unit-tested — testing meant constructing an entire HippoBrain +
embedder. In domain they have zero infrastructure dependencies, and the golden
tests finish in milliseconds.

Dependency direction: this module imports only common constants (L0), matching
the target architecture's §3 layering.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..common.constants import (
    _GUESS_IMPORTANCE_CAP,
    _ORIGIN_RANK,
    _STOPWORDS,
    ENTITY_ORIGIN_MANUAL,
    SUPERSEDE_OVERLAP,
    SUPERSEDE_SIM,
)


def guess_importance(text: str) -> float:
    """Heuristic importance: strong emotion/commitment/negated change → high score.

    🔴 **Correction (derived from reading the code, not speculation)**:
    ``min(score, 0.95)`` is **dead code** — the branches below can only yield
    the four values 0.5 / 0.6 / 0.7 / 0.75, so the cap never takes effect. ⇒
    **Automatic judgment cannot produce an importance >0.75**; the 0.95s in
    the library can only come from ``report_feedback(useful=True)``'s +0.05
    accumulation (9 times) or from callers passing values explicitly. Earlier
    attributions of "a lone sentence got 0.95" to this function were
    **wrong**.

    The cap is kept so that adding higher tiers in the future won't cross
    :data:`_GUESS_IMPORTANCE_CAP` — above 0.95 is the "repeatedly confirmed
    by the user" territory, which lexical rules should not hand out.

    ⚠️ Another known consequence (**whether to adjust is deferred**): 0.7 >=
    ``PIN_IMPORTANCE``(0.7) ⇒ memories hitting 「现在 / 不再 / changed / ！」
    get **auto-pinned**, and pins only go in, never out ⇒ on long libraries
    pins accumulate monotonically (one real library measured 78%).
    """
    t = text.lower()
    score = 0.5
    if any(m in t for m in ("always", "never", "must", "important", "remember", "记住", "永远", "绝不", "重要")):
        score = 0.75
    if any(m in t for m in ("love", "hate", "崩溃", "狂喜", "amazing", "terrible", "！", "!")):
        score = max(score, 0.7)
    if any(m in t for m in ("no longer", "used to", "changed", "现在", "不再")):
        score = max(score, 0.7)
    if len(text) > 200:
        score = max(score, 0.6)
    return min(score, _GUESS_IMPORTANCE_CAP)


def tokens_of(s: str) -> set[str]:
    """Content-word set: Chinese bigrams + English words (stop words removed)."""
    zh = set(re.findall(r"[\u4e00-\u9fa5]{2}", s))
    en = {w for w in re.findall(r"[A-Za-z]{3,}", s.lower()) if w not in _STOPWORDS}
    return zh | en


def overlap_of_sets(ta: set[str], tb: set[str]) -> float:
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def is_restatement(
    sim: float,
    text_a: str,
    text_b: str,
    ents_a: Any = None,
    ents_b: Any = None,
    toks_a: set[str] | None = None,
    toks_b: set[str] | None = None,
) -> bool:
    """Whether two memories are "talking about the same thing" (suspected different versions of the same fact).

    **The criterion lives in exactly one place**: the write side
    (``_detect_supersede`` judging "who supersedes whom") and the read side
    (``_annotate_conflicts`` judging "who coexists conflicting with whom")
    share it.

    Why share one criterion: the write side is limited by timing —
    ``supersede`` disabled (the default for document imports), old data
    predating the mechanism, or pairs missed at the time all leave
    **coexisting old/new versions** behind; with the same criterion, the read
    side can label them retroactively. If each side wrote its own thresholds,
    sooner or later you would get the self-contradiction of "judged the same
    fact at write time, judged different facts at read time".

    Conditions: similarity passes (``sim >= SUPERSEDE_SIM``) **and** (shared
    entities **or** content-word overlap passes
    (``>= SUPERSEDE_OVERLAP``)). A single condition is not enough — under the
    Chinese bigram standard, two passages of the same document easily cross
    the overlap gate; looking only at overlap would misjudge large swaths of
    chronological narrative (coverage report · high-4 measured 28 pairs
    wrongly killed).
    """
    if sim < SUPERSEDE_SIM:
        return False
    if set(ents_a or []) & set(ents_b or []):
        return True
    ta = toks_a if toks_a is not None else tokens_of(text_a)
    tb = toks_b if toks_b is not None else tokens_of(text_b)
    return overlap_of_sets(ta, tb) >= SUPERSEDE_OVERLAP


def merge_entity_origins(
    explicit: list[str] | None,
    extracted: dict[str, str],
    *,
    canonical: Callable[[str], str],
) -> tuple[list[str], dict[str, str]]:
    """Merge explicit entities (manual origin) with extracted entities (their own origins) into names/origins.

    Extracted from inline code in ``brain.core.ingest``. Three preserved
    semantics:

    * Explicitly passed entities **also go through alias canonicalization**
      (``canonical``, the lexicon's ``别名 => 主名``) — without it, a host
      passing an alias would conjure up an alias entity entry, which is
      exactly the source of "one entity split into multiple dossiers";
    * When the same entity (lowercased name) hits multiple times, **upgrade
      only, never downgrade**: take the higher-credibility source per
      ``_ORIGIN_RANK``, so an entity first stored heuristically can still be
      "promoted" when later hit by the lexicon/host;
    * names keep **appearance order without dedup** (``_link_entities``
      dedups internally via ``dict.fromkeys``), matching this function's
      pre-extraction behavior.

    Args:
        explicit: The host's explicitly passed ``entities=[...]`` (may be
            None).
        extracted: The product of ``extract_entities_with_origin(text)``:
            ``{entity name: origin}``.
        canonical: The alias canonicalization function (brain's
            ``canonical_entity``).
    """
    names: list[str] = []
    origins: dict[str, str] = {}
    for e in (explicit or []):
        e = (e or "").strip()
        if not e:
            continue
        # Explicitly passed entities **must also go through alias
        # canonicalization**: without it, a host passing an alias
        # (entities=["子安"]) would conjure up an alias entity entry, which
        # is exactly the source of the "one entity split into multiple
        # dossiers" problem. Doing canonicalization only on the extraction
        # side is not enough.
        e = canonical(e)
        names.append(e)
        k = e.lower()
        prev = origins.get(k)
        if prev is None or _ORIGIN_RANK[ENTITY_ORIGIN_MANUAL] > _ORIGIN_RANK[prev]:
            origins[k] = ENTITY_ORIGIN_MANUAL
    for name, origin in extracted.items():
        names.append(name)
        k = name.lower()
        prev = origins.get(k)
        if prev is None or _ORIGIN_RANK[origin] > _ORIGIN_RANK[prev]:
            origins[k] = origin
    return names, origins
