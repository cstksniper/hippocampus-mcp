# -*- coding: utf-8 -*-
"""Read/write for the task directory sidecar: ``<user 库>/directory.json``.

Three structural constraints:

1. **The sidecar is not memory**: a plain JSON file at the same level as
   ``memories.json`` (same shape as the knowledge-map layer's ``maps.json``
   sidecar). **It never enters the recall pool and is not vector memory** --
   the directory needs exact hits; a semantic-retrieval miss = an addressing
   failure.
2. **Managed structure**: the write path has only the managed channel
   (``library_create`` registers on library creation). This module provides
   the single write primitive ``upsert_entry`` and **exposes no ordinary write
   tool**: if any agent could write the directory raw, changing one task
   address would pollute all future cross-library addressing. Reads are open
   -- it is meant to be read.
3. **A bad file is never silent**: broken JSON / broken structure always
   returns an issue code for the caller to self-report, and **never proceeds
   as an empty directory** (an empty directory = "this task was never
   registered", which is a different thing from "the file is broken"). The
   write path **refuses to overwrite** a bad file (raises), to avoid wiping
   user-hand-written entries.

Atomic write: temp file + ``os.replace`` (same directory, cannot fail even
across drive letters). In-process read-modify-write is serialized by a
module-level lock; **across processes** (multiple stdio services pointing at
one user library) one update can in theory still be lost -- the cost is "a new
library missed registration" (visible, fixable by hand), so no file lock is
introduced.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from . import normpath
from ..domain.task_directory import (DIRECTORY_SCHEMA, normalize_entries,
                                     normalize_entry)

__all__ = [
    "DIRECTORY_NAME",
    "TaskDirectoryError",
    "directory_path",
    "load_directory",
    "save_directory",
    "upsert_entry",
]

#: Sidecar file name, at the same level as ``memories.json`` / ``library.json``.
DIRECTORY_NAME = "directory.json"

#: Read-modify-write serialization (in-process). Protects only the write path --
#: reads take no lock; bad files are the caller's to handle.
_WRITE_LOCK = threading.Lock()


class TaskDirectoryError(ValueError):
    """Task directory write failure (bad file / invalid entry). **Never
    swallowed**: the caller must make noise."""


def directory_path(library_path: str) -> str:
    """Absolute path of ``<user 库>/directory.json``."""
    return os.path.join(normpath(library_path), DIRECTORY_NAME)


def load_directory(library_path: str) -> tuple[dict[str, Any], str | None]:
    """Read the sidecar, returning ``(data, issue)``.

    ``data`` is always ``{"schema": int, "entries": [...]}`` (entries already
    normalized); ``issue`` being ``None`` means it was read (including **file
    does not exist** -- a fresh install is an empty directory, not an error);
    otherwise it is the self-report code ``"task_directory_read_failed"``.

    Counting contract: ``entries`` contains only **valid-after-normalization**
    entries; bad entries are dropped here (ones lacking ``task``/``path``
    cannot be addressed). The dropped count is self-reported via
    ``data["dropped"]``.
    """
    path = directory_path(library_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0}, None)
    except (OSError, ValueError):
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0},
                "task_directory_read_failed")
    if not isinstance(raw, dict):
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0},
                "task_directory_read_failed")
    raw_entries = raw.get("entries")
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list):
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0},
                "task_directory_read_failed")
    entries = normalize_entries(raw_entries)
    try:
        schema = int(raw.get("schema") or DIRECTORY_SCHEMA)
    except (TypeError, ValueError):
        schema = DIRECTORY_SCHEMA
    return ({"schema": schema, "entries": entries,
             "dropped": len(raw_entries) - len(entries)}, None)


def save_directory(library_path: str, data: dict[str, Any]) -> str:
    """Atomically write the sidecar (temp file + ``os.replace``). Returns the
    written path."""
    path = directory_path(library_path)
    payload = {
        "schema": int(data.get("schema") or DIRECTORY_SCHEMA),
        "entries": normalize_entries(data.get("entries") or []),
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError:
        # A leftover temp file would make the next reader think "the directory
        # grew a weird file"; clean up before re-raising.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def upsert_entry(library_path: str, entry: dict[str, Any],
                 *, now: float | None = None) -> dict[str, Any]:
    """Register/update one ``task -> library`` mapping (**the only write
    primitive**).

    The dedup key is the **normalized path** (``normpath``: drive-letter case
    and relative paths all count as the same), not ``library_id`` -- the id is
    for human reading and future dedup; the addressing primary key is the path.
    When the same task name switches library (re-registration after a rebuild
    or relocation), the old entry is updated in place to the new path, avoiding
    the ambiguity of one task name pointing at two libraries (the flip side of
    "don't guess" is: don't manufacture ambiguity).

    Returns:
        ``{"registered": True, "created": bool, "moved": bool, "count": int}``.

    Raises:
        TaskDirectoryError: the sidecar is bad/unreadable, or the entry lacks
            ``task``/``path``. **A bad file is never overwritten** -- it may be
            a user-hand-written directory; wiping it loses it forever.
    """
    item = normalize_entry(entry)
    if item is None:
        raise TaskDirectoryError("task-directory entry lacks task or path; registration refused")
    item["updated_at"] = float(now if now is not None else time.time())
    with _WRITE_LOCK:
        data, issue = load_directory(library_path)
        if issue:
            raise TaskDirectoryError(
                f"{directory_path(library_path)} is unreadable ({issue}): "
                "refusing to overwrite; repair the file or back it up and delete it first")
        entries: list[dict[str, Any]] = list(data.get("entries") or [])
        key = normpath(item["path"])
        created = moved = False
        idx = next((i for i, e in enumerate(entries)
                    if normpath(e["path"]) == key), None)
        if idx is not None:
            entries[idx] = item           # same library re-registered = in-place update
        else:
            old = next((i for i, e in enumerate(entries)
                        if e["task"].casefold() == item["task"].casefold()), None)
            if old is not None:
                entries[old] = item       # same-named task switched library: move in place
                moved = True
            else:
                entries.append(item)
                created = True
        save_directory(library_path, {"schema": data.get("schema"), "entries": entries})
    return {"registered": True, "created": created, "moved": moved,
            "count": len(entries)}
