# -*- coding: utf-8 -*-
"""Cross-library retrieval scope: pure functions for value validation and
two-library merging.

Placement: I/O orchestration lives in ``application/recall_service.py``
(running the second stream on the user library's brain injected via the
composition root); this module only does **pure data** criteria (domain may
only depend on common):

* ``validate_scope`` -- scope value validation (illegal values raise
  explicitly... in text form; the error text is maintained at this single
  point, callers take it to build the bare error body);
* ``merge_scope_hits`` -- scope=all's hard-segmented merge (project library
  results all first, user-level all after, top_k is the **total**;
  ``duplicate_of`` is marked only on **exact content** matches across
  libraries -- the two libraries' independent embedding models have
  incomparable scores; similarity is not guessed).

Boundary note: scope is a three-value enum, not generalized to arbitrary
many libraries -- cross-task retrieval like ``libraries="A,B"`` is a later
increment (once the task directory lands, it becomes the source of legal
addressing values).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SCOPE_ALL", "SCOPE_PROJECT", "SCOPE_USER", "VALID_SCOPES",
    "MAX_LIBRARY_SPEC",
    "merge_scope_hits", "merge_multi_library", "parse_library_spec",
    "rrf_merge_streams", "validate_scope",
]

SCOPE_PROJECT = "project"
SCOPE_USER = "user"
SCOPE_ALL = "all"

VALID_SCOPES = (SCOPE_PROJECT, SCOPE_USER, SCOPE_ALL)

#: How many periphery libraries ``libraries=`` may fan out to per call (B1
#: cost cap: one brain per library, 25-33s cold load / marginal memory; too
#: much fan-out = swap thrashing between them). Exceeding it raises
#: explicitly, no silent truncation -- how many libraries the caller wants
#: is explicit intent; cutting either half of it is wrong.
MAX_LIBRARY_SPEC = 4

#: RRF fusion constant (the standard value). rank counts from 1:
#: score = Σ 1/(k + rank).
RRF_K = 60


def validate_scope(scope: str, *, allow_all: bool = True) -> str | None:
    """Validate the scope value. Legal returns ``None``; illegal returns the
    **error text** (does not raise).

    Args:
        scope: the caller's scope (the new argument of
            recall/get_memory_context).
        allow_all: ``False`` = this entry point does not support all (the
            5-slot assembly of get_memory_context has no natural place for a
            "project first, user second" ordering; narrowed; the error text
            spells it out separately).

    Why no exception: the error body's shape is the caller's to decide
    (following recall's filter_tags parse-error precedent -- bare dict
    ``{"error": ..., "input": ...}``, without any self-exposure keys,
    "error bodies carry none").
    """
    s = str(scope or "").strip()
    if s in VALID_SCOPES:
        if s == SCOPE_ALL and not allow_all:
            return ("scope=all is only supported by recall; get_memory_context is a 5-slot "
                    "assembly structure with no place for a project-first, user-second ordering. "
                    "For single-library assembly across libraries, call scope=project / scope=user separately.")
        return None
    return (f"scope accepts only {'/'.join(VALID_SCOPES)}, got {scope!r}. "
            "project=the current library (default); user=the user-level library; all=project library first, "
            "user library second (recall only).")


def merge_scope_hits(project_hits: list[dict[str, Any]],
                     user_hits: list[dict[str, Any]],
                     *, top_k: int) -> list[dict[str, Any]]:
    """scope=all's hard-segmented merge.

    Rules:
    1. **Hard segmentation**: project library results all first, user-level
       all after (each keeps its original order -- scores are incomparable
       across libraries, no fused re-ranking whatsoever);
    2. **top_k is the total**: the two libraries compete for slots, the
       project segment claims them first (the caller's "I want 5 entries"
       semantics unchanged);
    3. **duplicate_of only annotates, never drops** (following the "annotate,
       don't adjudicate" precedent): when a user-level entry's content is
       **exactly equal** to an **already-listed** project entry's, mark
       ``duplicate_of = <project entry memory_id>``. Two deliberate
       narrowings -- compare exact content only (similarity is not guessed),
       and annotate only visible duplicates (when the project twin was cut
       off by the cap, the user-level one standing alone does not count as
       "appearing twice").

    Args:
        project_hits: the project stream's result entries (already in final
            dict form).
        user_hits: the user library stream's result entries.
        top_k: total cap (``<= 0`` returns empty, same semantics as recall's
            top_k).
    """
    if top_k <= 0:
        return []
    kept_project = list(project_hits)[:top_k]
    room = max(int(top_k) - len(kept_project), 0)
    # Visible-duplicate criterion table: listed project entries' content -> memory_id (first)
    project_ids = {}
    for h in kept_project:
        c = h.get("content")
        if c is not None and c not in project_ids:
            project_ids[c] = h.get("memory_id")
    out: list[dict[str, Any]] = [dict(h) for h in kept_project]
    for h in user_hits[:room]:
        entry = dict(h)
        dup = project_ids.get(entry.get("content"))
        if dup is not None:
            entry["duplicate_of"] = dup
        out.append(entry)
    return out


def parse_library_spec(spec: str) -> tuple[list[str], str | None]:
    """Parse ``libraries="甲,乙"`` (B1). Returns ``(names, error_text)``.

    * Empty string / pure whitespace = **unspecified** (``[], None``), not an
      error;
    * comma split (**fullwidth 「，」 carries the same weight** -- the normal
      state of Chinese IMEs; do not make the caller suffer), strip
      whitespace, **empty segments ignored**, duplicate values
      **deduplicated order-preserving** (a caller typo is not an error, but
      dedup keeps the "how many libraries" accounting countable);
    * beyond :data:`MAX_LIBRARY_SPEC` -> error text (no silent truncation).

    The values themselves are **not validated for legality here** (whether a
    library exists is the addressing layer's business; tri-state resolution
    failures will raise on their own); this function only handles syntax.
    """
    raw = str(spec or "").replace("，", ",")
    names: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p and p not in names:
            names.append(p)
    if not names:
        return [], None
    if len(names) > MAX_LIBRARY_SPEC:
        return [], (f"libraries= accepts at most {MAX_LIBRARY_SPEC} libraries per call "
                    f"(got {len(names)}: {names!r}). "
                    "Fan-out cost grows linearly with library count (one brain per library); split multi-library queries into separate calls.")
    return names, None


def rrf_merge_streams(
    streams: list[tuple[str, list[dict[str, Any]]]],
    *, top_k: int, k: int = RRF_K,
) -> list[dict[str, Any]]:
    """RRF rank fusion for periphery library fan-out (decision (2)a: fusion
    only **inside** the periphery set).

    ``score(d) = Σ 1/(k + rank_i(d))`` (rank counts from 1, independent
    ranks per stream) -- rank fusion is independent of each library's score
    system (BM25 IDF is within-library statistics; vector cosine score
    distributions drift with the corpus; comparing scores across libraries
    is meaningless, comparing ranks is not).

    Each entry carries ``library`` (the **first-seen stream's** library
    name; when the same id repeats across streams, first seen wins) and
    ``rrf_score``. Same-score ties break by ascending ``memory_id``
    (deterministic).
    """
    if top_k <= 0:
        return []
    scores: dict[str, float] = {}
    first_seen: dict[str, dict[str, Any]] = {}
    lib_by_id: dict[str, str] = {}
    for lib_name, hits in streams:
        for rank, h in enumerate(hits, start=1):
            mid = str(h.get("memory_id"))
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
            if mid not in first_seen:
                first_seen[mid] = h
                lib_by_id[mid] = lib_name
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[dict[str, Any]] = []
    for mid, score in ranked[:top_k]:
        entry = dict(first_seen[mid])
        entry.setdefault("library", lib_by_id[mid])
        # No rounding: rrf_score is the ranking basis; truncating digits would drift same-score decisions
        entry["rrf_score"] = score
        out.append(entry)
    return out


def merge_multi_library(
    home_hits: list[dict[str, Any]],
    periphery: list[tuple[str, list[dict[str, Any]]]],
    *, top_k: int, home_name: str,
) -> list[dict[str, Any]]:
    """Hierarchical merge (decision (2)a): **the home library segment all
    first (original order); the periphery competes via RRF for the remaining
    slots**.

    Same hard-segmentation spirit as :func:`merge_scope_hits` -- the home /
    task libraries' existing ordering semantics and tests are preserved
    bit-for-bit; RRF is used only inside the periphery set, never re-ranks
    across segments.

    Returns:
        The merged list (each entry carries ``library``: home segment =
        ``home_name``, periphery = each stream's name). ``top_k`` is the
        grand total (the home segment claims first; the periphery competes
        for ``top_k - len(home segment)``).
    """
    if top_k <= 0:
        return []
    kept_home = [dict(h, library=home_name) for h in list(home_hits)[:top_k]]
    room = max(int(top_k) - len(kept_home), 0)
    if room == 0 or not periphery:
        return kept_home
    # Cap each periphery stream to the remaining slots before fusing: a
    # stream's extra entries could not rank in RRF anyway (there are only
    # this many slots); capping early saves one full scoring pass.
    capped = [(name, hits[:room]) for name, hits in periphery]
    return kept_home + rrf_merge_streams(capped, top_k=room)
