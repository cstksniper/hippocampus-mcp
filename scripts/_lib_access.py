# -*- coding: utf-8 -*-
"""Unified library-access entry for maintenance scripts (backend-agnostic).

Memory libraries have two persistence backends (json / sqlite). Deciding
"this is a library" by checking whether ``memories.json`` exists does **not**
hold for sqlite libraries — they have no memories.json at all (after a
migration the old file may or may not still be there). Under that old check,
health-check scripts **report FAIL for sqlite libraries**; write-path scripts
are worse: they write a file nobody in the library reads, so the change looks
successful but has no effect.

This module gives script-side tools three things:
  1. :func:`library_kind` — detect the backend from **directory contents**
     (**db first**: after a migration both files exist; checking memories.json
     first misclassifies a sqlite library as json);
  2. :func:`load_records` — read L1 records via the backend (same structure
     for both backends);
  3. :func:`require_library` — a single entry check, replacing the scattered
     ``os.path.isfile(os.path.join(d, "memories.json"))`` in every script.

**json-only write scripts** (``unsupersede`` / ``repair_l0_dangling`` style,
which rewrite memories.json directly) pass ``writable=True`` to
:func:`require_library` and are **explicitly rejected on sqlite libraries**
with migration/export guidance — refusing to act is better than silently
editing a file nobody reads.

Usage (top of each script):
    from _lib_access import require_library, load_records
"""
from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hippocampus_mcp.store.backend import (  # noqa: E402
    DB_FILENAME,
    MEMORIES_FILENAME,
    JsonBackend,
    SqliteBackend,
)

__all__ = [
    "DB_FILENAME",
    "MEMORIES_FILENAME",
    "library_kind",
    "has_library_data",
    "open_backend",
    "load_records",
    "load_header",
    "require_library",
]


def library_kind(directory: str) -> str:
    """Detect the backend from **directory contents**, not env.

    Returns ``"sqlite"`` / ``"json"`` / ``"empty"``.

    IMPORTANT: **db must be checked first** — after a migration, a library can
    contain both ``hippocampus.db`` and a leftover ``memories.json``; checking
    json first misclassifies it as a json library.
    """
    if not directory:
        return "empty"
    if os.path.isfile(os.path.join(directory, DB_FILENAME)):
        return "sqlite"
    if os.path.isfile(os.path.join(directory, MEMORIES_FILENAME)):
        return "json"
    return "empty"


def has_library_data(directory: str) -> bool:
    """Does this directory contain library data (either backend)?"""
    return library_kind(directory) != "empty"


def open_backend(directory: str) -> object:
    """Build the backend from directory contents (**no env reads** — scripts
    must not be influenced by host env vars).

    The caller is responsible for the directory actually containing a library
    (call :func:`require_library` first); otherwise this constructor happily
    creates an empty one.
    """
    if library_kind(directory) == "sqlite":
        return SqliteBackend(directory)
    return JsonBackend(directory)


def load_records(directory: str) -> list[dict]:
    """Read the L1 record list (identical structure for both backends).
    Empty for a library-less directory."""
    if not has_library_data(directory):
        return []
    backend = open_backend(directory)
    data, _err = backend.load_memories()
    if not isinstance(data, dict):
        return []
    return list(data.get("records") or [])


def load_header(directory: str) -> dict:
    """Read the library header (``model_key`` / ``dim`` / ``format``),
    backend-agnostic.

    Purpose: scripts must verify "which model produced the stored vectors"
    before constructing a brain — a mismatch **silently corrupts the library
    header** (brain.save() writes the current session's model_key back), and
    afterwards the retrieval identity no longer matches the vectors' actual
    source. A sqlite library's memories.json cannot provide this at all.
    """
    if not has_library_data(directory):
        return {}
    backend = open_backend(directory)
    data, _err = backend.load_memories()
    if not isinstance(data, dict):
        return {}
    return {k: data.get(k) for k in ("format", "dim", "model_key", "models")}


def require_library(directory: str, action: str = "this operation",
                    writable: bool = False) -> str:
    """Single entry check: exit if not a library; exit if the library is sqlite
    and this script only supports json.

    Replaces the scattered ``os.path.isfile(memories.json)`` checks.
    ``writable=True`` means the script **rewrites memories.json directly**
    (not via brain / store) — those must be rejected on sqlite libraries.
    """
    d = (directory or "").strip()
    if not d:
        print("✗ No library directory given (--dir / --persist-dir).", flush=True)
        raise SystemExit(2)
    kind = library_kind(d)
    if kind == "empty":
        print("✗ Not a memory library directory (neither %s nor %s): %s"
              % (DB_FILENAME, MEMORIES_FILENAME, d), flush=True)
        raise SystemExit(2)
    if writable and kind == "sqlite":
        print(
            "✗ %s only rewrites %s directly, but %s is a **sqlite library** "
            "(data lives in %s).\n"
            "  Editing the json file has no effect — the library never reads it.\n"
            "  Options:\n"
            "    1) change data via MCP tools (recommended, backend-agnostic);\n"
            "    2) export first, edit, import back: python scripts/migrate_store.py"
            " export-json --dir <lib>;\n"
            "    3) port the script onto the store backend (backend.load_memories / "
            "save_memories_full), like the other scripts in this module."
            % (action, MEMORIES_FILENAME, d, DB_FILENAME), flush=True)
        raise SystemExit(2)
    return kind
