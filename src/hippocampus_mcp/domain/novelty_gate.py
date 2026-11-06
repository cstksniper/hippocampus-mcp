# -*- coding: utf-8 -*-
"""O5: novelty / prediction-error gated encoding (domain pure functions, M2 discipline).

Biological counterpart: the hippocampus encodes "inputs consistent with existing
schemas" cheaply (corroboration → no strengthening; duplicates are dedup's job),
significantly strengthens "expectation-violating inputs" (contradiction →
importance boost + echo via the existing conflict-annotation vocabulary), and
gives "brand-new stimuli" a small weight boost (novel entities → small boost).

**The criteria are deliberately conservative heuristics** (no LLM embedded in the
production path — a standing constraint of this system):

* Contradiction = the new sentence contains **negation/correction marker words**
  AND overlaps some high-confidence schema target's content words at
  ``SUPERSEDE_OVERLAP`` (0.6, aligned with the "same fact" overlap tier shared
  by write-side supersede and read-side conflict annotation — **the threshold
  lives in exactly one place**, no second set). It catches explicit corrections
  like 「张岱死期 1689」vs「张岱其实死于 1690」; implicit contradictions
  (different numbers without correction words) are left to the read-side
  contradictions tool and dedup echo.
* Corroboration = overlap passes but **no** marker words → expensive encoding
  is unnecessary; importance keeps the keyword heuristic's original value.
* Novel entity = an entity name extracted from the sentence (lowercased) never
  seen in the library → small boost. Among the three verdicts, precedence is
  contradiction > corroboration > novel entity.

**Two boundaries (zero-regression guarantee relative to existing mechanisms)**:

1. The gate only adjusts "encoding strength" — the importance at write time —
   and **does not create pins**: when base < ``PIN_IMPORTANCE``, a boosted value
   is clamped to ``PIN_IMPORTANCE - 0.01``, never newly crossing the auto-pin
   gate (monotonic pin accumulation is a lesson on record in
   ``domain/policies.py``: one real library measured 78% auto pins). The
   0.7-tier profile candidates / promotion candidates / user-level inheritance
   slot likewise must not be newly sent in by the gate.
2. dedup / supersede criteria are **untouched**: not one of the merge
   thresholds or supersede-detection triggers (UPDATE_MARKERS + similarity +
   is_restatement) lives in this module; the module only yields importance and
   conflict-object echo before the write loop, changing no
   "who supersedes whom / who merges into whom" judgment.

Dependency direction: imports only common constants (L0), matching the target
architecture's §3 layering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..common.constants import (
    _GUESS_IMPORTANCE_CAP,
    SUPERSEDE_OVERLAP,
)

__all__ = [
    "CONTRADICTION_MARKERS", "HIGH_CONF_IMPORTANCE", "CONTRADICT_BOOST",
    "NOVEL_BOOST", "SCHEMA_L1_CAP", "NoveltyOutcome", "SchemaTarget",
    "build_schema_targets", "gate_sentence", "summarize_events",
]

# Negation / correction marker words. Deliberately a **different table** from
# store.supersede's UPDATE_MARKERS: that table mixes in time-migration words
# (现在 / 以前 / 目前 / recently — "我现在住北京" is a normal statement, not a
# correction), while this table only takes words meaning "negating a previous
# claim", to suppress false positives.
CONTRADICTION_MARKERS: tuple[str, ...] = (
    "不再", "不是", "其实", "错了", "搞错", "不对", "弄错", "记错",
    "取消", "撤销", "作废", "收回", "纠正", "更正", "改成", "改为",
    "换成", "改成是", "no longer", "not anymore", "actually", "wrong",
    "mistake", "correction", "instead", "changed to", "replaced by",
    " rather than", "my bad",
)

#: High-confidence L1 gate: the same tier (0.7) as the auto-pin gate
#: (``store.memory.PIN_IMPORTANCE``), profile candidates, promotion candidates,
#: and the user-level inheritance slot — a "schema" is built from memories that
#: were repeatedly confirmed; a generic statement with a 0.5 baseline score
#: does not constitute an "expectation", and violating it is not a prediction
#: error.
#: (domain does not import store — the same discipline as
#: ``consolidate_flow`` / ``stats_views``: gate values are injected by the
#: caller; this constant is only a same-value fallback when nothing is
#: injected, keeping the test pin aligned.)
HIGH_CONF_IMPORTANCE: float = 0.7

#: Same-value fallback for the pin gate (the truth is the core-side-injected
#: ``store.memory.PIN_IMPORTANCE``; the two drifting apart = tests catch red).
PIN_IMPORTANCE_DEFAULT: float = 0.7

#: Content-word overlap tier for the contradiction verdict: shares
#: ``SUPERSEDE_OVERLAP`` (0.6) with ``is_restatement``'s no-entity overlap
#: tier — the "same fact" standard lives in exactly one place.
CONTRADICT_OVERLAP: float = SUPERSEDE_OVERLAP

#: Contradiction (prediction error) boost; small weight boost for novel entities.
CONTRADICT_BOOST: float = 0.1
NOVEL_BOOST: float = 0.05

#: Cap on the high-confidence L1 comparison set (the N most recent by time).
#: On large libraries, comparing everything burns CPU for nothing; recent
#: high-confidence memories are the "current schema"; profiles and topic
#: summaries are kept in full (they are small anyway).
SCHEMA_L1_CAP: int = 200


@dataclass(frozen=True)
class SchemaTarget:
    """One schema comparison target (one each from L3 profile / L2T topic summary / high-confidence L1)."""
    ref: str          # back-reference: memory_id / "profile:<slot>" / "topic:<id>"
    kind: str         # "l1" | "l3" | "l2t"
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class NoveltyOutcome:
    """Per-sentence gate verdict (the product of ``gate_sentence``)."""
    verdict: str                    # contradict | corroborate | novel_entity | none
    importance: float               # adjusted (unadjusted = original value)
    base_importance: float
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    novel_entities: list[str] = field(default_factory=list)


def build_schema_targets(
    *,
    profile_texts: Iterable[str],
    topic_texts: Iterable[str],
    high_conf_records: Iterable[tuple[str, str, float]],
    tokens_of: Callable[[str], set[str]],
    cap: int = SCHEMA_L1_CAP,
) -> list[SchemaTarget]:
    """Assemble the schema comparison target set (once before the write loop; core.ingest injects the tokenizer).

    Args:
        profile_texts: L3 profile entry texts (core side takes
            ``profile[slot][i]["text"]``).
        topic_texts: L2T topic summary texts (core side takes non-empty
            ``summary``).
        high_conf_records: ``(memory_id, content, importance, timestamp)``
            4-tuples, **already filtered on the core side** to
            importance >= HIGH_CONF_IMPORTANCE and active, most recent by
            time (nlargest) — this function only applies the cap (domain
            never touches store objects).
        tokens_of: text → content-word set (brain's ``_tokens_of``).
        cap: how many high-confidence L1 records may join the comparison
            (profiles / summaries do not take from this quota).
    """
    targets: list[SchemaTarget] = []
    for t in profile_texts:
        t = (t or "").strip()
        if t:
            targets.append(SchemaTarget(
                ref="", kind="l3", text=t, tokens=frozenset(tokens_of(t))))
    for t in topic_texts:
        t = (t or "").strip()
        if t:
            targets.append(SchemaTarget(
                ref="", kind="l2t", text=t, tokens=frozenset(tokens_of(t))))
    n = 0
    for mid, content, _imp, _ts in high_conf_records:
        if n >= max(int(cap), 0):
            break
        content = (content or "").strip()
        if not content:
            continue
        targets.append(SchemaTarget(
            ref=mid, kind="l1", text=content, tokens=frozenset(tokens_of(content))))
        n += 1
    return targets


def _clamp_boosted(
    base: float, boosted: float, pin_importance: float
) -> float:
    """Boosted importance: **no pin creation**, no crossing the heuristic cap.

    Sentences whose base is already >= the pin gate are already in pin
    territory (the 0.7/0.75 tier); lifting them further does not change pin
    status; sentences with base < the gate are clamped below it — the gate's
    remit is "encoding strength", not "retention policy".
    """
    v = min(float(boosted), float(_GUESS_IMPORTANCE_CAP))
    if float(base) < float(pin_importance):
        v = min(v, float(pin_importance) - 0.01)
    return round(v, 4)


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def gate_sentence(
    text: str,
    *,
    base_importance: float,
    sentence_entities: Iterable[str],
    known_entity_keys: Iterable[str],
    targets: list[SchemaTarget],
    tokens_of: Callable[[str], set[str]],
    markers: tuple[str, ...] = CONTRADICTION_MARKERS,
    overlap_threshold: float = CONTRADICT_OVERLAP,
    contradict_boost: float = CONTRADICT_BOOST,
    novel_boost: float = NOVEL_BOOST,
    pin_importance: float = PIN_IMPORTANCE_DEFAULT,
) -> NoveltyOutcome:
    """Give a novelty verdict and adjusted importance for one sentence (pure function).

    Verdict order (mutually exclusive, one taken): contradiction > corroboration
    > novel entity > none.

    Args:
        text: The sentence's raw text.
        base_importance: Baseline score from the keyword heuristic
            (``guess_importance(sent)``).
        sentence_entities: Entity names extracted from this sentence
            (lowercased; the core side owns normalization).
        known_entity_keys: Key set of the pre-write entity table (lowercased) —
            "novel" is judged against the **pre-write** library; this
            sentence's own entities do not count as known.
        targets: The product of :func:`build_schema_targets`.
        tokens_of: Tokenizer (injected).
        markers / overlap_threshold / the two boosts: defaults are the module
            constants; parameterized only so tests can probe the boundaries.
        pin_importance: The pin gate (core side injects
            ``store.memory.PIN_IMPORTANCE`` — the clamping baseline behind
            "no pin creation"; domain does not import store).
    """
    base = float(base_importance)
    t = (text or "").lower()
    toks = frozenset(tokens_of(text or ""))
    ents = [e.lower() for e in (sentence_entities or []) if e]
    known = {k.lower() for k in (known_entity_keys or [])}

    # ① Contradiction (prediction error): correction markers + "same fact"-tier
    #    overlap with some schema target. Take the target with the largest
    #    overlap as the conflict object (ties go to the earlier one — the
    #    targets order is l3/l2t first, l1 in reverse time order, so earlier
    #    means more "recent/high-level").
    if toks and any(m in t for m in markers):
        best: tuple[float, SchemaTarget] | None = None
        for tg in targets:
            ov = _overlap(toks, tg.tokens)
            if ov >= overlap_threshold and (best is None or ov > best[0]):
                best = (ov, tg)
        if best is not None:
            ov, tg = best
            return NoveltyOutcome(
                verdict="contradict",
                importance=_clamp_boosted(
                    base, base + contradict_boost, pin_importance),
                base_importance=base,
                conflicts=[{
                    "ref": tg.ref or f"{tg.kind}:（无回指）",
                    "kind": tg.kind,
                    "text": tg.text[:200],
                    "overlap": round(ov, 4),
                    "boosted_from": base,
                    "boosted_to": _clamp_boosted(
                        base, base + contradict_boost, pin_importance),
                }],
            )

    # ② Corroboration (schema-consistent): overlap passes, no correction
    #    markers → cheap encoding, not one point added. (True duplicates go
    #    through hit dedup merging, which is not the gate's business.)
    if toks:
        for tg in targets:
            if _overlap(toks, tg.tokens) >= overlap_threshold:
                return NoveltyOutcome(
                    verdict="corroborate", importance=base,
                    base_importance=base)

    # ③ Novel entity: an entity name unseen in the library appears → a small
    #    weight boost for a new stimulus.
    fresh = [e for e in ents if e and e not in known]
    if fresh:
        return NoveltyOutcome(
            verdict="novel_entity",
            importance=_clamp_boosted(base, base + novel_boost, pin_importance),
            base_importance=base,
            novel_entities=fresh,
        )

    return NoveltyOutcome(verdict="none", importance=base, base_importance=base)


#: Cap on conflict objects echoed per ingest (a batch import can hit many
#: targets; the echo body is not an audit log — the read-side contradictions
#: tool is the panorama entry).
CONFLICTS_ECHO_CAP: int = 20


def summarize_events(
    events: list[tuple[str, NoveltyOutcome]],
) -> dict[str, Any] | None:
    """Summarize the sentence events into the ingest response's ``novelty`` key (no gate output = None).

    ``None`` = this write produced no novelty signal (all ``none``) or the gate
    did not run — zero trace in the response body (same discipline as U5/D4);
    the host just checks truthiness.
    """
    if not events:
        return None
    counts = {"contradict": 0, "corroborate": 0, "novel_entity": 0, "none": 0}
    conflicts: list[dict[str, Any]] = []
    fresh_all: list[str] = []
    for _sent, out in events:
        counts[out.verdict] = counts.get(out.verdict, 0) + 1
        if out.conflicts and len(conflicts) < CONFLICTS_ECHO_CAP:
            c = dict(out.conflicts[0])
            c["sentence"] = " ".join(_sent.split())[:200]
            conflicts.append(c)
        for e in out.novel_entities:
            if e not in fresh_all:
                fresh_all.append(e)
    if counts["contradict"] == counts["corroborate"] == counts["novel_entity"] == 0:
        return None
    return {
        "verdicts": counts,
        "conflicts": conflicts,
        "novel_entities": fresh_all[:20],
        "note": ("矛盾→importance 提升（不制造 pin）；印证→不加强"
                 "（重复归 dedup 管）；全新实体→小幅加权。冲突对象回显只"
                 "是提示，裁决走 update_memory / 读侧 contradictions。"),
    }
