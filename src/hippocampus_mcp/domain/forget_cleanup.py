"""Two pieces of pure logic inside the forget cascade cleanup: profile-layer removal + L0 orphan-source marking.

Zero imports: only dict/list operations; profile entries and L0 raw records are used
duck-typed (``.get("text")`` / ``.get("id")``). The brain-level orchestration of forget
(tombstones, physical deletion, entity recomputation, the topic layer, restoring the
superseded) stays in the brain shell; this module only carries the unit-testable part
of "given the state, compute the cleanup result".

Both pieces are verbatim isomorphic to the original implementation, including two quirks kept on purpose:
  * ``clean_profile``: a slot emptied entirely gets deleted (a slot with empty items gets deleted too);
  * ``mark_orphan_raws``: when the same id appears multiple times in raw_log, or the same
    rid appears multiple times in source_ids, marked gets repeated appends (the for-if structure
    has no break/skip).
"""
from __future__ import annotations

from typing import Any


def clean_profile(
    profile: dict[str, list[dict[str, Any]]],
    *,
    memory_id: str,
    content: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Remove the profile entries derived from this memory, returning ``(new_profile, dropped)``.

    Hit conditions (either one suffices):
      * text copy: the profile ``text`` equals the memory's full text or its first-120-char truncation
        (profile text is itself a truncated copy of the memory content);
      * single evidence: ``evidence`` has exactly one entry and it points precisely at this memory.
        In old libraries evidence is the id list of the whole entity (too coarse a granularity);
        when there are multiple evidence entries only the text match is honored, to avoid
        accidentally removing other entries of the same entity.

    A slot emptied entirely disappears from new_profile (replicating the original ``del``).
    Each dropped entry records ``{"slot": ..., "text": ...}``, with text taken as the
    **un-stripped** original value.
    """
    new_profile: dict[str, list[dict[str, Any]]] = {}
    dropped: list[dict[str, Any]] = []
    content_full = content.strip()
    content_cut = content[:120].strip()
    for slot, items in profile.items():
        kept: list[dict[str, Any]] = []
        for it in items:
            ev = it.get("evidence") or []
            txt = (it.get("text") or "").strip()
            text_hit = txt in (content_full, content_cut)
            evidence_hit = memory_id in ev and len(ev) == 1
            if text_hit or evidence_hit:
                dropped.append({"slot": slot, "text": it.get("text")})
                continue
            kept.append(it)
        if kept:
            new_profile[slot] = kept
    return new_profile, dropped


def mark_orphan_raws(
    raw_log: list[dict[str, Any]],
    *,
    source_ids: list[str] | None,
    alive_refs: set[str],
    reason: str,
) -> list[str]:
    """Mark L0 sources no longer referenced by any alive memory as forgotten, returning the marked list.

    ``alive_refs`` is the union of all alive memories' source_ids (built by the caller).
    The marking is written **in place** into the raw dict (``forgotten`` / ``forgotten_reason``),
    same as the original implementation — the shell does not need to reassign raw_log.

    Quirk replication: when the same id appears multiple times in raw_log, or is duplicated
    in source_ids, marked gets one append per match.
    """
    marked: list[str] = []
    for rid in source_ids or []:
        if rid in alive_refs:
            continue
        for raw in raw_log:
            if raw.get("id") == rid:
                raw["forgotten"] = True
                raw["forgotten_reason"] = reason or ""
                marked.append(rid)
    return marked
