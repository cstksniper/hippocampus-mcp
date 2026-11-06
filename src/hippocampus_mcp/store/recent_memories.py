# -*- coding: utf-8 -*-
"""Recently written memories (the cheapest implementation of "where did we
leave off", extracted from brain/core.py).

Takes only active records, content clipped to 200 chars -- the
pre-activation package / opening package gets assembled into the prompt.
Optionally filters by ``source`` (the most recent few from one source, for
"what just came in from this collection source").

Why store/: these are **query compositions over inventory records**
(state filter + sort + projection), same family as contradictions;
``MemoryState`` lives in ``store.memory`` which domain cannot reach
(which is exactly why compositions like this live on the store side).

Port discipline:

* The record sequence is passed in by the caller (the semantics of
  ``iter_records()`` are decided by brain);
* The ``active`` state value is injected rather than imported
  (``MemoryState.ACTIVE.value`` is a string constant; once injected, tests
  can substitute it freely);
* Clip length ``clip`` and count ``limit`` are explicit parameters; no
  Config reads.

Pre-existing behavior is preserved under "move only, change nothing":
``limit <= 0`` returns an empty list (**not** Python slice negative-index
semantics -- the original implementation guards with ``if limit > 0``,
so -1 does not mean "drop the last one"; it means the whole slot is off).
"""
from __future__ import annotations

from typing import Any, Iterable


def recent_memories(
    records: Iterable[Any],
    limit: int = 10,
    source: str | None = None,
    *,
    active: str,
    clip: int = 200,
) -> list[dict[str, Any]]:
    """Recently written memories (newest first)."""
    rows = [
        r for r in records
        if r.state == active
        and (source is None or (r.source or "") == source)
    ]
    rows.sort(key=lambda r: -r.timestamp)
    return [
        {
            "memory_id": r.id,
            "content": r.content[:clip],
            "timestamp": r.timestamp,
            "source": r.source or "",
        }
        for r in rows[:limit] if limit > 0
    ]
