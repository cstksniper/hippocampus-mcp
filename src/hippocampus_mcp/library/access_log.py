# -*- coding: utf-8 -*-
"""Write-side audit log for the user library (D2a: append and read of access_log.jsonl).

Shape follows the U5 task-directory sidecar precedent: an independent file
living in the user library directory, next to memories.json, **not vector
memories and not in the recall pool**. Write-side events are appended by
the write path (WriteService) -- auditing is a side channel: **append
failure only warns, never blocks the write** (the main path is writing the
memory itself; a broken audit must not take the user's normal writes down
with it).

The read side (drift_audit reports / the future D2b read audit) tolerates
only bad lines: skip and count; never condemn the whole log over one line
of bad JSON -- the "corrupt ≠ empty" lesson applies here in reverse: a
read failure must produce a visible count, but must not escalate into a
denial of service.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..domain.write_quarantine import ACCESS_LOG_FILENAME

logger = logging.getLogger("hippocampus_mcp")

__all__ = ["access_log_path", "append_access_event", "read_access_events"]


def access_log_path(lib_path: str) -> str:
    """User library directory → absolute path of access_log.jsonl."""
    return os.path.join(lib_path or "", ACCESS_LOG_FILENAME)


def append_access_event(lib_path: str, event: dict[str, Any]) -> bool:
    """Append one audit event. **Returns success/failure honestly**, never raises to the caller."""
    if not lib_path:
        return False
    path = access_log_path(lib_path)
    try:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except OSError as e:
        logger.warning("access_log append failed (writes are not blocked): %s: %s",
                       path, e)
        return False


def read_access_events(lib_path: str, limit: int = 50) -> dict[str, Any]:
    """Read the most recent ``limit`` audit events (a **tail** window in ascending time order).

    Bad lines are skipped and counted (``skipped``), never condemning the
    whole; missing file = empty log (``count=0``, not an error -- no
    non-home writes have happened yet).
    """
    path = access_log_path(lib_path)
    out: dict[str, Any] = {"path": path, "count": 0, "skipped": 0,
                           "events": []}
    if limit <= 0 or not os.path.isfile(path):
        return out
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        logger.warning("access_log read failed: %s: %s", path, e)
        out["skipped"] = -1  # -1 = the whole file was unreadable (distinct from per-line skips of bad lines)
        return out
    events: list[dict[str, Any]] = []
    skipped = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(ev, dict):
            events.append(ev)
    out["skipped"] = skipped
    tail = events[-limit:] if limit < len(events) else events
    out["events"] = tail
    out["count"] = len(tail)
    return out
