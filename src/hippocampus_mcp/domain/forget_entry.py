# -*- coding: utf-8 -*-
"""The tombstone and affected-keys pre-stages of ``brain.core.forget`` (pure
segments extracted from core).

**Background**: forget's cascade-cleanup helpers have moved into
forget_cleanup, and forget_by_query into forget_flow; the pre-stage of
``forget`` itself still has two pure computations -- tombstone dict
construction (with the ``deleted_at`` clock injection) and affected entity key
computation (scanning the entities inverted index). They decide "what was
deleted, why, and who was implicated", but previously had no dedicated unit
tests pinning them (tombstone key shape / snapshot copies / clock called
exactly once / index scan order); regression coverage leaned on forget's
end-to-end suite.

**Dependency direction**: this module imports only the standard library (not
even common -- the clock is fetched by the shell at call time and injected;
the entities inverted index is passed in by value, and this module never
touches brain state).

**Replicated quirks (touching any one is a behavior change; parity will
catch it)**:

1. ``build_tombstone``'s reason fallback: ``reason or ""`` -- None and the
   empty string are the same shape (the tombstone's reason is always a
   string).
2. ``build_tombstone``'s snapshot copies: ``entity_ids`` / ``source_ids`` go
   through ``list(x or [])`` -- the tombstone holds **copies**; the original
   record (and its lists) is deleted right after, and shared mutable objects
   would let the tombstone mutate with them; None / empty list both normalize
   to ``[]``.
3. ``build_tombstone``'s clock called exactly once: ``deleted_at`` takes
   ``now()`` exactly once at construction (all other fields come from the
   deleted record's existing values).
4. ``compute_affected_keys``' scan order: produced in ``entities``' dict
   insertion order (the order echoed in ``entities_touched`` is this order,
   reproducible by the caller); entities whose ``memory_ids`` key is missing /
   None / empty **do not hit** (``or []`` fallback), and the entities dict
   itself is read-only and untouched.
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# Segment 1: tombstone dict construction (old shell lines 2105-2115; the
# append state write stays in the shell)
# ======================================================================
def build_tombstone(
    *,
    memory_id: str,
    content: str,
    reason: str | None,
    timestamp: float,
    state: str,
    entity_ids: list[str] | None,
    source_ids: list[str] | None,
    now: Callable[[], float],
) -> dict[str, Any]:
    """Build the tombstone dict (eight keys; the key order is the ``tombstones``
    entry shape).

    Recording the tombstone first is the audit prerequisite: the original text
    is preserved, so "what was deleted and why" stays traceable after deletion.

    Args:
        memory_id: the deleted memory's id.
        content / timestamp / state: the deleted record's existing values,
            passed through as-is.
        reason: deletion reason; ``or ""`` fallback (quirk 1).
        entity_ids / source_ids: the deleted record's reference lists; taken as
            **snapshot copies** (quirk 2), None falls back to an empty list.
        now: the ``brain_core._now`` function itself (fetched by the shell at
            call time; the parity freeze is pinned on it so both sides get the
            same deleted_at). Called exactly once (quirk 3).

    Returns:
        An eight-key dict. The append is done by the shell -- this function
        only assembles, never touching state.
    """
    return {
        "memory_id": memory_id,
        "content": content,
        "reason": reason or "",
        "timestamp": timestamp,
        "state": state,
        "entity_ids": list(entity_ids or []),
        "source_ids": list(source_ids or []),
        "deleted_at": now(),
    }


# ======================================================================
# Segment 2: affected entity key computation (old shell lines 2117-2120; pure
# read of the entities inverted index)
# ======================================================================
def compute_affected_keys(
    entities: dict[str, dict[str, Any]],
    memory_id: str,
) -> list[str]:
    """Compute the entity keys whose stats need recomputing after forget
    (scans the entities inverted index).

    Args:
        entities: the ``brain.entities`` inverted index (keys = lowercased
            entity keys, values carry the ``memory_ids`` reference list).
            **Read-only** -- recomputing/clearing is the shell's subsequent
            cascade business.
        memory_id: the deleted memory's id.

    Returns:
        List of entity keys referencing that memory, in entities' insertion
        order (quirk 4).
    """
    return [
        k for k, e in entities.items()
        if memory_id in (e.get("memory_ids") or [])
    ]
