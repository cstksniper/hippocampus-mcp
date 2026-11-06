"""update_tags' candidate-selection segment: pick the records whose tags
will be updated, per the filter conditions.

Moved wholesale from ``brain.core._select_for_tag_update``. This is a
private helper of update_tags with no direct tool-call surface; the
shell keeps only 「fetch ``store.iter_records()`` → dispatch」.

Zero imports. ``records`` is passed in by the shell (the result of
``store.iter_records()``); records are used duck-typed (must have
``id`` / ``source`` / ``content`` / ``state`` / ``tags`` attributes).

Semantics preserved verbatim (changing any one is a behavior change;
parity will catch it):
  * ``memory_ids`` is normalized into a ``str`` set for **exact id
    matching**; an empty list (falsy) means "unspecified", not "want
    none of them";
  * ``source`` / ``state`` are **equality** matches (the record side
    gets an ``or ""`` fallback in the ``source`` comparison);
    ``source_contains`` / ``contains`` are substrings;
  * ``untagged_only`` is "exclude if it has any tags" (a non-empty
    ``tags`` means tagged);
  * ``missing_keys`` means 「**at least one of the keys is missing**」
    (the fill-missing scenario), not 「all missing」 -- the latter is
    almost always an empty set when filling two keys;
  * the second return item is the **filter self-report**: when the
    filter is written wrong, ``matched=0`` and 「really no matching
    records」 look identical; the self-report is the only way to tell
    them apart. Empty strings/lists are normalized to ``None``.
"""
from __future__ import annotations

from typing import Any


def select_for_tag_update(
    records: Any,
    *,
    source: str = "",
    source_contains: str = "",
    contains: str = "",
    state: str = "",
    untagged_only: bool = False,
    missing_keys: list[str] | None = None,
    memory_ids: list[str] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Pick the records whose tags will be updated, per the filter
    conditions.

    Why the ``contains`` (body-keyword) path exists: a real-library
    measurement found **52% of untagged entries come from one file**
    (``PROJECT_MEMORY_LOG.md``); ``source`` prefix filtering cannot
    subdivide that block at all -- only body keywords can circle out
    「this batch」.
    """
    wanted = {str(i) for i in memory_ids} if memory_ids else None
    mkeys = [str(k) for k in (missing_keys or []) if str(k)]
    out: list[Any] = []
    for rec in records:
        if wanted is not None and rec.id not in wanted:
            continue
        if source and (rec.source or "") != source:
            continue
        if source_contains and source_contains not in (rec.source or ""):
            continue
        if contains and contains not in (rec.content or ""):
            continue
        if state and rec.state != state:
            continue
        tags = rec.tags or {}
        if untagged_only and tags:
            continue
        if mkeys and any(k in tags for k in mkeys):
            continue
        out.append(rec)
    return out, {
        "source": source or None,
        "source_contains": source_contains or None,
        "contains": contains or None,
        "state": state or None,
        "untagged_only": bool(untagged_only),
        "missing_keys": mkeys or None,
        "memory_ids_n": len(wanted) if wanted else None,
    }
