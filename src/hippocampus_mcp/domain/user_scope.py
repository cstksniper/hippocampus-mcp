# -*- coding: utf-8 -*-
"""User-level memory inheritance slot: pure selection and rendering.

Placement: I/O orchestration lives in
``brain/session.py::_user_scope_brief`` (a lightweight read path -- no brain
construction, no embedding model loading); this module does only the **pure
data** selection and text rendering, the same approach as
:mod:`stats_views.pinned_brief`:

* ``records`` is duck-typed (uses ``.id / .content / .importance /
  .timestamp / .pin / .state``) -- domain does not import store (arch G1);
  ``active_state`` is injected by the caller (``MemoryState.ACTIVE.value``);
* Selection criteria: **pinned first** (threshold not consulted) -> the
  threshold band (non-pinned and ``importance >= importance_min``) -> within
  each group always ``(-importance, -timestamp)`` descending -> capped
  overall to ``limit``; a pinned entry appears only in the pinned band even
  if its importance also passes the threshold.
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_user_scope", "select_user_scope"]


def select_user_scope(
    records: list[Any],
    *,
    limit: int,
    importance_min: float,
    active_state: str = "active",
) -> list[dict[str, Any]]:
    """Select inheritance-slot entries from the user-level library's stock
    records.

    Args:
        records: stock records (the caller has already fetched them from the
            user library backend and converted them into record objects).
        limit: how many to carry. **``<= 0`` returns an empty list** -- this
            is "off", not slicing: ``records[:limit]`` with ``limit=-1`` means
            "everything except the last one", and negative slicing is wrong
            here (off = literally zero traces).
        importance_min: the threshold band's lower bound
            (``Config.user_scope_importance``, default 0.7). The boundary is
            **inclusive**: ``importance == importance_min`` enters the band.
        active_state: the active state string (injected by the caller).

    Returns:
        A list of projection dicts for the selected entries (``memory_id /
        content / importance / timestamp / pin``), pinned band first,
        threshold band after, each ``(-importance, -timestamp)`` descending,
        capped overall to ``limit``.
    """
    if limit <= 0:
        return []
    # D2a write-side isolation: quarantined cards stay out of the user_scope
    # slot -- fabricated "user preferences" must not reach any session through
    # the inheritance slot, even if written into the user library (the last
    # link of the attack chain, see design doc §4.3).
    # The getattr fallback matches recall_flow: old record objects without the
    # field are treated as not isolated.
    active = [r for r in records
              if r.state == active_state and not getattr(r, "quarantine", False)]
    pinned = [r for r in active if r.pin]
    threshold = [
        r for r in active
        if not r.pin and r.importance >= importance_min
    ]
    pinned.sort(key=lambda r: (-r.importance, -r.timestamp))
    threshold.sort(key=lambda r: (-r.importance, -r.timestamp))
    return [
        {
            "memory_id": r.id,
            "content": r.content,
            "importance": r.importance,
            "timestamp": r.timestamp,
            "pin": bool(r.pin),
        }
        for r in (pinned + threshold)[:limit]
    ]


def render_user_scope(entries: list[dict[str, Any]]) -> list[str]:
    """Inheritance-slot text lines: the same rendering as ``block_task_hits``
    (``content[:200]（id[:8]``).

    The response body key ``block_user_scope``: a text slot the host
    concatenates into its prompt directly, same source as the structured
    ``user_scope.memories``.
    """
    return [
        f"{e['content'][:200]}（{e['memory_id'][:8]}）" for e in entries
    ]
