"""Locate candidate memories by natural language and delete them (or preview only) — the orchestration layer.

What the host agent usually holds is natural language like "the one about the
March launch was wrong", not a memory_id. The flow is: recall first, then
delete, **preview-only by default** (dry_run=True); execute with dry_run=False
only after confirming the candidates are right.

**Executing deletion (dry_run=False) has two guardrails**:

1. **Vector endorsement required**: only candidates with
   ``similarity >= gate`` are considered. recall lets through BM25 surface-form
   hits that are semantically unrelated; their similarity is **0.0**, and
   deleting them amounts to "wiping the library by keyword".
2. **By default delete only the best match**: with ``allow_multiple=False``,
   even if several candidates pass the gate, only the top-ranked one is
   deleted. "Delete the one that was wrong" is almost always singular in
   meaning; the extra slots are exactly where mis-deletions come from. For a
   genuine bulk delete, pass ``allow_multiple=True`` explicitly.

Neither guardrail is implemented by "raising the similarity gate" — relevance
drifts wholesale with query length (see the measured distribution in the
``FORGET_MIN_SIMILARITY`` comment), and a fixed gate saves neither direction;
what is actually stable is the relative advantage of "the target is always #1".

Skipped candidates are always listed as-is in ``skipped`` with a
``skip_reason``; the host can then switch to precise deletion by memory_id.

Placement note: the body is pure orchestration (dispatch + deletion loop); the
two brain entries recall / forget are injected as **callables** (not
pre-called); ``min_similarity_default`` is fetched fresh by the caller — the
"None takes the default" semantics is preserved as-is, and it follows when the
caller's namespace is patched (consistent with how brain's other constants are
handled).
"""
from __future__ import annotations

from typing import Any, Callable


def forget_by_query(
    *,
    query: str,
    top_k: int = 3,
    dry_run: bool = True,
    min_similarity: float | None = None,
    allow_multiple: bool = False,
    min_similarity_default: float,
    recall: Callable[..., list[dict[str, Any]]],
    forget: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Locate candidate memories by natural language and delete them (or preview only).

    Args:
        query: natural-language description ("the one about the March launch was wrong").
        top_k: how many to recall, passed through to recall.
        dry_run: True previews without deleting (default).
        min_similarity: deletion gate; None takes ``min_similarity_default``.
        allow_multiple: allow deleting several at once; default False (only the
            best-matching one is deleted).
        min_similarity_default: the default gate (fetched fresh by the caller;
            do not use a def default argument).
        recall: the recall entry, called as
            ``recall(query, top_k=..., include_superseded=True)``.
        forget: the single-deletion entry, called as
            ``forget(memory_id, reason="forget_by_query: <query>")``.
    """
    floor = min_similarity_default if min_similarity is None else float(min_similarity)
    hits = recall(query, top_k=top_k, include_superseded=True)
    candidates = [
        {
            "memory_id": h["memory_id"],
            "content": h["content"],
            "score": round(float(h.get("score", 0.0)), 4),
            "similarity": round(float(h.get("similarity", 0.0)), 4),
            "state": h.get("state"),
        }
        for h in hits
    ]
    if not candidates:
        return {"ok": False, "error": f"no memories matched: {query}",
                "query": query, "candidates": []}

    if dry_run:
        return {
            "ok": True, "dry_run": True, "query": query,
            "candidates": candidates,
            "min_similarity": floor,
            "note": (f"When executed: only candidates with similarity >= {floor} are processed; "
                     f"at most 1 deletion unless allow_multiple=True is passed explicitly."),
        }

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for c in candidates:
        if c["similarity"] >= floor:
            eligible.append(c)
        else:
            skipped.append({**c, "skip_reason":
                            f"similarity {c['similarity']} below the threshold {floor}"})

    if not eligible:
        return {
            "ok": False,
            "error": (f"None of the {len(skipped)} candidates reached the deletion threshold {floor}; "
                      f"nothing was deleted. They are most likely BM25 surface-form hits. "
                      f"Rephrase closer to the target's exact wording, or call forget_memory with the "
                      f"memory_id from the dry_run result for a precise delete."),
            "query": query, "dry_run": False,
            "min_similarity": floor, "skipped": skipped, "removed_count": 0,
        }

    if not allow_multiple and len(eligible) > 1:
        for c in eligible[1:]:
            skipped.append({
                **c, "skip_reason":
                "single mode: only the best-matching one is deleted (pass allow_multiple=True for bulk deletion)",
            })
        eligible = eligible[:1]

    removed = []
    for c in eligible:
        report = forget(c["memory_id"], reason=f"forget_by_query: {query}")
        removed.append({
            "memory_id": c["memory_id"],
            "content": c["content"],
            "similarity": c["similarity"],
            "ok": report.get("ok", False),
        })
    out: dict[str, Any] = {
        "ok": True, "dry_run": False, "query": query,
        "min_similarity": floor,
        "removed": removed, "removed_count": len(removed),
    }
    if skipped:
        out["skipped"] = skipped
        out["skipped_count"] = len(skipped)
        out["note"] = (f"{len(skipped)} candidates were not deleted (see skip_reason in skipped); "
                       f"to delete them anyway, call forget_memory with their memory_id.")
    return out
