# -*- coding: utf-8 -*-
"""Write credentials (origin): turn "who wrote this, under which task" into **structured fields**.

**The specific problem being solved**: ``MemoryRecord.source`` is a free-form string
(file name / relative path / session identifier), filled in by the host itself. Once
multiple Workers share one library, this channel cannot answer three questions:

1. Was this memory written by the Router or by Worker-3? (``source`` may contain only a path)
2. Which ticket / which task does it belong to? (currently one can only dig through ``raw_log``
   by hand to infer it)
3. Post-hoc attribution: a certain Worker went off the rails — which entries did it pollute?
   (without a subject field this cannot be looked up)

The super-agent (Router + one Worker per ticket) turned an "architectural assumption" into a
hard requirement: **multiple subjects are writing to the same library at the same time**.
Without subject attribution, a bad result can only be rolled back as a whole and cannot
be precisely reclaimed.

Three conventions (same family as ``domain/write_quarantine`` / ``domain/task_directory``):

1. **Write only, do not filter**: origin produced by this module **does not enter the retrieval
   path** (recall does not read it). Let the data land first; wire up filtering only after the
   value distribution has been validated on a real library — change one layer at a time;
   changing the write path and the retrieval path simultaneously is a lesson phase G already
   paid for.
2. **Sparse**: empty values produce no keys; existing cards stay byte-for-byte unchanged
   (the same serialization discipline as ``quarantine`` / ``sensitivity``).
3. **Do not guess**: without a ``task`` source, leave it empty; do not guess the task name
   from the body text. The source of truth for ``task`` should be the host's
   ``session_start(task=...)``; this module only ferries it along.

Dependency direction: this module is pure functions, depends only on the standard library,
and imports no other module of this package (``common`` is L0 — any layer may depend on it,
but it may not depend back on anyone).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ORIGIN_FIELDS",
    "MAX_LEN",
    "MAX_VALUE_CHARS",
    "build_origin",
    "merge_origin",
    "normalize_origin",
]

#: The fixed field set of origin. **Order is semantic priority**: ``session`` always has a value
#: (a per-connection key under the daemon), ``task`` depends on the host passing it explicitly,
#: ``surface`` is the write surface (tool name / CLI / import script).
ORIGIN_FIELDS: tuple[str, ...] = ("session", "task", "surface")

#: Per-field length caps. origin is an **index dimension**, not body text; storing an extra
#: 200 chars of task full text per memory would needlessly bloat the 5.28MB memories.json
#: (and those chars are already in the body text).
MAX_LEN: dict[str, int] = {"session": 64, "task": 200, "surface": 32}

#: Total-length backstop across all fields (prevents the host from stuffing in an over-long
#: JSON string to bypass the per-field caps).
MAX_VALUE_CHARS = 320


def normalize_origin(raw: Any) -> dict[str, str]:
    """Normalize arbitrarily shaped input into the three-key ``{session/task/surface}`` dict.

    Only whitelisted fields are kept (unknown keys are dropped — origin is meant to be
    queried by field; extra keys are neither read nor indexed, and keeping them is just
    noise); values are uniformly ``str().strip()``-ed and truncated per :data:`MAX_LEN`.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key in ORIGIN_FIELDS:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        out[key] = text[: MAX_LEN.get(key, 64)]
    if not out:
        return {}
    # total-length backstop: truncate in field order, preserving session (the first attribution key)
    budget = MAX_VALUE_CHARS
    trimmed: dict[str, str] = {}
    for key in ORIGIN_FIELDS:
        if key not in out:
            continue
        value = out[key]
        if len(value) > budget:
            value = value[:budget]
        if value:
            trimmed[key] = value
            budget -= len(value)
    return trimmed


def build_origin(
    session: str = "", task: str = "", surface: str = ""
) -> dict[str, str]:
    """Build one write credential (empty values produce no keys; the result may be ``{}``).

    ``session`` takes the host connection identifier (``common.session_ctx.session_key()``);
    in the stdio direct-connect form it is always ``__default__`` — that is not a bug, it is
    the normal state of **one process, one session** (the same same-source restriction as the
    LNN split; see the ``common/session_ctx`` docstring).
    """
    return normalize_origin(
        {"session": session, "task": task, "surface": surface}
    )


def merge_origin(
    old: dict[str, str] | None, new: dict[str, str] | None
) -> dict[str, str]:
    """Merge two origin dicts: **existing values win; only missing keys are filled in**.

    Same principle as :func:`store.memory.merge_tag_dicts` — a deduplicating merge means
    the two entries are near-identical in content, and the subject is usually the same too;
    if a genuinely different subject does appear (two Workers independently wrote to the
    same fact), keeping the earlier writer is at least **deterministic**, and the new subject
    has already left a trace in ``raw_log``, so it is not lost.
    """
    merged = dict(normalize_origin(new))
    merged.update(normalize_origin(old))
    return merged
