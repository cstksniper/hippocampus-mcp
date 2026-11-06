# -*- coding: utf-8 -*-
"""The five inline logic segments of ``brain.core.ingest`` (extracted from core).

**Why worth moving**: ``ingest`` is the host's most common write entry point.
Besides the things that truly need brain state ("write L0 / link entities /
call manager / keep the ledger"), the shell was tangled with five pure-logic
blocks -- auto_tags merging (6 lines), cognitive type and importance
determination (7 lines), sentence splitting and memory_type (11 lines),
kept_separate neighbor annotation (10 lines), response body assembly (33
lines). They are all pure dict/list operations, yet untestable in isolation
because they were scattered across the self-facing context. After the move,
``ingest`` is a single straight line: ledger -> write L0 -> determine ->
split -> write loop -> detect -> neighbors -> echo.

**Dependency direction**: this module imports only the standard library (the
domain -> common-only discipline: here not even common is used -- all
callbacks and constants are fetched by the shell at call time and injected).

**Replicated quirks (touching any one is a behavior change; parity will
catch it)**:

1. ``resolve_tags``'s short-circuit order: the three shapes -- ``auto_tags``
   false, extraction empty, extraction non-empty -- all flow through the
   **same expression** (``merge_tag_dicts(tags, auto)``) -- including
   returning None (not ``{}``) when ``tags=None`` and ``auto=None``.
2. ``guess_basics``'s ``explicit_importance`` only looks at whether the
   argument is None, regardless of the final value; passing ``importance=0``
   still counts as explicitly specified (0 is not None).
3. ``split_sentences``' chunk branch produces an **empty list** for
   all-whitespace text (no L1 write); the sentence branch hands empty strings
   to ``_split_facts`` (its pre-existing behavior).
4. ``annotate_neighbors`` only adds three keys and never touches existing
   ones; the hit count is the return value. With no kept_separate the whole
   segment is skipped (neighbors is not even iterated).
5. ``assemble_ingest_result``'s key order is deliberate (zcode addendum 16):
   ``merged_into`` must come first -- JSON preserves order; order conveys
   priority.
6. The ``truncation`` sub-object's ``truncated`` is always present (not only
   on truncation); ``max_seq_length`` is fetched by the shell and passed in
   (0 = unknown).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ======================================================================
# Segment 1: auto_tags merging (old shell lines 874-878)
# ======================================================================
def resolve_tags(
    tags: dict[str, Any] | None,
    *,
    auto_tags: bool,
    text: str,
    extract: Callable[[str], dict[str, Any] | None],
    merge: Callable[[dict[str, Any] | None, dict[str, Any] | None],
                    dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Merge auto-extracted tags into the explicit tags (explicit wins, only
    fills missing keys).

    Args:
        tags: tags explicitly passed by the caller (may be None).
        auto_tags: switch. When false, **extraction does not even run** (saves
            one regex scan).
        text: body text (the corpus for tag extraction).
        extract: ``brain.tag_extractor.extract`` itself.
        merge: ``store.memory.merge_tag_dicts`` itself (keeps left, i.e.
            explicitly passed values are not overwritten by the auto results).

    Returns:
        The merged tags; when ``auto_tags`` is off, extraction is empty, or both
        sides are empty, the incoming ``tags`` is returned unchanged (None comes
        back as None, see module docstring quirk 1).
    """
    if auto_tags:
        # Explicitly passed values win: what the caller knows beats the regex
        # (merge keeps left)
        auto = extract(text or "")
        if auto:
            tags = merge(tags, auto)
    return tags


# ======================================================================
# Segment 2: cognitive type and importance determination (old shell lines 902-908)
# ======================================================================
@dataclass(frozen=True)
class IngestBasics:
    """Normalized result for cognitive type / importance.

    Attributes:
        cognitive_type: the finally adopted value (caller's explicit value wins).
        importance: the finally adopted value (auto-determined when None).
        explicit_importance: whether the caller explicitly passed importance --
            the write loop uses this to decide whether to run per-sentence
            determination again (the block-level value is for the summary only).
    """
    cognitive_type: str
    importance: float
    explicit_importance: bool


def guess_basics(
    text: str,
    speaker: str,
    cognitive_type: str | None,
    importance: float | None,
    *,
    guess_cognitive: Callable[[str, str], str],
    guess_importance: Callable[[str], float],
) -> IngestBasics:
    """Automatic cognitive type determination + automatic importance
    determination (block-level, used only for the returned summary values).

    Replicated quirk: ``explicit_importance`` only looks at whether the argument
    is None -- passing 0 still counts as explicitly specified (see module
    docstring quirk 2).
    """
    if cognitive_type is None:
        cognitive_type = guess_cognitive(text, speaker)
    explicit_importance = importance is not None
    if importance is None:
        importance = guess_importance(text)
    return IngestBasics(
        cognitive_type=cognitive_type,
        importance=importance,
        explicit_importance=explicit_importance,
    )


# ======================================================================
# Segment 3: sentence splitting and memory_type (old shell lines 919-931)
# ======================================================================
def split_sentences(
    text: str,
    granularity: str,
    *,
    split_facts: Callable[[str], list[str]],
) -> list[str]:
    """Split the original text into L1 candidate sentences by granularity.

    Args:
        text: original body text.
        granularity: ``"chunk"`` = one entry per whole block; anything else
            (including ``"sentence"``) goes through the sentence splitter.
        split_facts: ``brain._split_facts`` itself (quirks like merging short
            sentences into the previous one live there; this function does not
            reimplement them).

    Returns:
        List of candidate sentences. At chunk granularity, all-whitespace text
        returns an **empty list** (no L1 write); at sentence granularity an
        empty string is handed to ``split_facts`` as-is (see module docstring
        quirk 3).
    """
    if granularity == "chunk":
        stripped = text.strip()
        return [stripped] if stripped else []
    return split_facts(text)


def memory_type_for(cognitive_type: str, experience_value: str) -> str:
    """Cognitive type -> memory_type mapping (CognitiveType lives in the store
    package, invisible to domain; injected by value -- a loop invariant,
    computed once).

    Returns:
        ``"episodic"`` (experience types) or ``"semantic"`` (everything else).
    """
    return "episodic" if cognitive_type == experience_value else "semantic"


# ======================================================================
# Segment 4: kept_separate neighbor annotation (old shell lines 970-979)
# ======================================================================
def annotate_neighbors(
    neighbors: list[dict[str, Any]],
    kept_separate: list[dict[str, Any]],
) -> int:
    """C10: annotate the neighbors with the "deliberately not merged because of
    new hard info" markers.

    Even if the host only looks at neighbors, it can tell on the spot: these
    two are similar, but this was not a duplicate write.

    Args:
        neighbors: the write side's echoed top3 near neighbors (**modified in
            place**).
        kept_separate: the "not merged" list produced by the write loop.

    Returns:
        How many entries were actually annotated (for tests and self-reporting).

    Replicated quirk: only three keys are added (``hard_info_kept_separate`` /
    ``missing_hard_info`` / ``new_memory_id``), existing keys are untouched;
    when ``kept_separate`` is empty the whole segment is skipped -- neighbors is
    not even iterated (see module docstring quirk 4).
    """
    if not kept_separate:
        return 0
    ks = {k["memory_id"]: k for k in kept_separate}
    n = 0
    for nb in neighbors:
        hit = ks.get(nb.get("memory_id"))
        if hit is not None:
            nb["hard_info_kept_separate"] = True
            nb["missing_hard_info"] = hit["missing_hard_info"]
            nb["new_memory_id"] = hit["new_memory_id"]
            n += 1
    return n


# ======================================================================
# Segment 5: response body assembly (old shell lines 992-1024)
# ======================================================================
def assemble_ingest_result(
    *,
    merged_into: list[dict[str, Any]],
    raw_id: str,
    memory_ids: list[str],
    entities: list[str],
    cognitive_type: str,
    importance: float,
    superseded: list[Any],
    tags: dict[str, Any] | None,
    neighbors: list[dict[str, Any]],
    kept_separate: list[dict[str, Any]],
    l0_pending: int,
    truncations: list[dict[str, Any]],
    max_seq_length: int,
) -> dict[str, Any]:
    """Assemble ingest's response body (key order is priority, see module
    docstring quirk 5).

    Args:
        merged_into: existing memories this write merged into on a dedup hit.
            **Non-empty = nothing new was created this time**; placing it first
            is zcode addendum 16's decision.
        raw_id: this call's L0 record id.
        memory_ids: list of L1 ids actually written this time.
        entities: normalized entity name list (aliases already merged).
        cognitive_type / importance: the finally adopted values.
        superseded: old memories superseded by this batch (may be empty).
        tags: tags actually attached (including auto_tags extractions); the
            echo is a copy.
        neighbors: near-neighbor echo (segment 4 may have annotated in place).
        kept_separate: near-duplicates not merged (mutually exclusive with
            merged_into).
        l0_pending: current L0 outstanding count (always present).
        truncations: truncation details (empty list = no truncation).
        max_seq_length: embedding model token cap (0 = unknown).

    Returns:
        The echo body. ``tags`` is a copy via ``dict(tags or {})``;
        ``truncation.truncated`` is always present (see module docstring quirk 6).
    """
    return {
        # Write-side echo: existing memories merged into on a dedup hit.
        # **Non-empty = nothing new was created this time**.
        "merged_into": merged_into,
        "raw_id": raw_id,
        "memory_ids": memory_ids,
        "entities": entities,
        "cognitive_type": cognitive_type,
        "importance": importance,
        "superseded": superseded,
        # Tags actually attached by this batch (including auto_tags extractions)
        # -- echoed back, otherwise "with auto_tags on, what exactly got
        # extracted" would require another library query
        "tags": dict(tags or {}),
        "neighbors": neighbors,
        # Dedup guard (scheme B): near-duplicates that hit dedup but were
        # **not merged** because of new numbers/dates. Non-empty == these
        # memories were newly created (mutually exclusive with merged_into).
        "kept_separate": kept_separate,
        # How many new raw entries have not yet reached brain_raw.json. Always
        # present, otherwise the caller cannot tell "definitely no outstanding
        # debt" from "never checked".
        "l0_pending": l0_pending,
        # C13: truncation self-report. **Always present** (not only on
        # truncation), otherwise the caller cannot tell "definitely not
        # truncated" from "never checked". When truncated=True, in items the
        # original text after embedded_chars is invisible to vector retrieval --
        # the card must be split.
        "truncation": {
            "truncated": bool(truncations),
            "max_seq_length": max_seq_length,
            "items": truncations,
        },
    }
