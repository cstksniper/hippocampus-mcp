# -*- coding: utf-8 -*-
"""User-level task directory: **pure data** normalization / selection /
rendering.

In one sentence: the user layer has, besides "content" (preferences /
methodology), also "navigation" -- the ``task -> library`` mapping is
cross-project, user-owned knowledge that every per-task agent needs. When
the user says "reference task A", looking up the directory yields library
A's exact address -- **no guessing, no disk scanning, no dependence on a
scan root**.

This module does pure functions only, sharing conventions with
:mod:`domain.user_scope` / :mod:`domain.scope_recall`:

* sidecar read/write (``<user 库>/directory.json``) lives in
  :mod:`library.task_directory` (L1 infra, with file I/O);
* assembly and injection live in ``brain/session.py::_task_directory_brief``;
* domain does **not import library / store** (architectural constraint:
  domain may only depend on common); ``path`` here is just a string --
  normalized comparison is the read/write side's business.

Four conventions (design §3.2/§3.3/§3.6/§7, each landed here):

1. **No vector memory, not in the recall pool**: the directory needs exact
   hits; semantic retrieval misses, and a miss = addressing failure,
   squarely against the "never silent" red line => this is deterministic
   selection/ordering, not similarity.
2. **``path`` is required** (absolute path): ``lib_`` id matching goes
   through the scan pool; when a library is not under the scan root, id
   resolution fails, while path addressing can be "patched in place";
   ``library_id`` is only for human reading and future dedup, **not the
   addressing primary key**.
3. **``limit <= 0`` means "off", not a slice**: ``entries[:limit]`` on a
   negative means "everything except the last N", which is not the same as
   "off = zero trace" (same lesson as U1/D4).
4. **No guessing**: if the directory lacks A, report "A is not in the
   directory" -- this module provides only exact lookup :func:`find_task`;
   fuzzy matching / auto-claiming are **deliberately not done**.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DIRECTORY_SCHEMA",
    "VALID_ACCESS",
    "find_task",
    "normalize_entries",
    "normalize_entry",
    "render_task_directory",
    "resolve_user_access",
    "resolve_workspace",
    "select_task_directory",
]

#: sidecar schema version (§3.3). On upgrade, follow "read old, write new":
#: old fields may always be omitted.
DIRECTORY_SCHEMA = 1

#: Fixed projection of entry fields (key-order sensitive: the field order of
#: the persisted JSON is decided here).
_ENTRY_KEYS = ("task", "name", "library_id", "path", "note", "updated_at",
               "workspace_root", "access")

#: D2b read-side access, three levels (design draft §4.2): full (everything
#: visible) / masked (default, sensitive:* excluded) / none (the user
#: library is invisible as a whole). Empty string = undeclared -> treated as
#: masked (the default level for registered workspaces); unregistered (the
#: directory has no such workspace) -> none (fail-closed, Q-8's leaning).
VALID_ACCESS = ("full", "masked", "none")

#: Truncation length for ``note`` at render time: the directory is an
#: **index**, not body text (§3.4); the note is only a one-sentence hint of
#: "what is this task for", same spirit as ``block_task_hits``'s 200 chars.
_NOTE_CAP = 40


def normalize_entry(raw: Any) -> dict[str, Any] | None:
    """Normalize one raw entry into the fixed projection; **None when
    ``task`` or ``path`` is missing**.

    ``path`` being required is a hard constraint (§3.3): ids fail to resolve
    after a library moves / deployment changes, while paths can be "patched
    in place" -- the directory exists to be **exactly reachable**; rather
    than treat an unreachable entry as a valid index, drop the bad entry
    (the caller counts it in ``dropped`` and self-exposes).
    """
    if not isinstance(raw, dict):
        return None
    task = str(raw.get("task") or "").strip()
    path = str(raw.get("path") or "").strip()
    if not task or not path:
        return None
    try:
        updated = float(raw.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        updated = 0.0
    return {
        "task": task,
        # name falls back to task when missing: ``library_create`` registers
        # under the same name at creation, and users hand-adding entries need
        # not write it twice.
        "name": str(raw.get("name") or "").strip() or task,
        "library_id": str(raw.get("library_id") or "").strip(),
        "path": path,
        "note": str(raw.get("note") or "").strip(),
        "updated_at": updated,
        # C1: the workspace root this entry binds (optional). When non-empty,
        # HELLO reverse lookup may set that workspace's connection default
        # library to ``path`` -- the resolution basis of "bind the library,
        # not the installation".
        # Empty string = unbound (default for old entries; behavior
        # bit-identical to U5).
        "workspace_root": str(raw.get("workspace_root") or "").strip(),
        # D2b: read-side access level (optional). Illegal values all become
        # an empty string (= masked) -- a mis-written access must not become
        # privilege escalation (fail-closed direction: rather wrongly lower
        # than wrongly raise).
        "access": (str(raw.get("access") or "").strip()
                   if str(raw.get("access") or "").strip() in VALID_ACCESS
                   else ""),
    }


def normalize_entries(raw_entries: Any) -> list[dict[str, Any]]:
    """Batch normalization (bad entries dropped; non-lists always become an
    empty list -- a broken sidecar must not explode here)."""
    if not isinstance(raw_entries, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_entries:
        entry = normalize_entry(raw)
        if entry is not None:
            out.append(entry)
    return out


def select_task_directory(entries: list[Any], *, limit: int) -> list[dict[str, Any]]:
    """Select the entries to inject: **most recently updated first**, capped
    to ``limit``.

    Args:
        entries: entries (raw dicts or already-normalized both fine; they
            are normalized once more internally).
        limit: how many to carry. **``<= 0`` returns an empty list** -- this
            is "off", not a slice (a negative slice is wrong here, same as
            U1/D4).

    Ordering: ``(-updated_at, task.casefold())`` descending/ascending combo.
    The second key is a **deterministic tiebreak**: a batch of
    ``library_create`` calls often lands in the same second; sorting by time
    alone would let the injection order drift with dict iteration order,
    irreproducible for tests and human review.
    """
    if limit <= 0:
        return []
    pool = normalize_entries(entries)
    pool.sort(key=lambda e: (-e["updated_at"], e["task"].casefold()))
    return pool[:limit]


def render_task_directory(entries: list[dict[str, Any]]) -> list[str]:
    """Render into compact index lines: ``任务 → 库名（路径）`` (with a note
    appended when present).

    Same discipline as the ``block_*`` family: one entry per line, no
    wrapping, directly pasteable into a prompt; paths are given verbatim
    (**no relativization/abbreviation** -- addressing needs an absolute path
    that can be fed straight into ``library_use`` or ``library=``;
    abbreviated, one would have to guess again).
    """
    lines: list[str] = []
    for e in entries:
        line = f"{e['task']} → {e['name']}（{e['path']}）"
        note = str(e.get("note") or "")
        if note:
            line += f" — {note[:_NOTE_CAP]}"
        lines.append(line)
    return lines


def find_task(entries: list[Any], task: str) -> dict[str, Any] | None:
    """Exact lookup by task name (``casefold`` normalized); None when absent.

    **No fuzzy matching** (§7): guessing the library wrong means reading and
    writing someone else's library; the user explicitly required "if A is
    not in the directory, say so".
    """
    q = str(task or "").strip().casefold()
    if not q:
        return None
    for entry in normalize_entries(entries):
        if entry["task"].casefold() == q:
            return entry
    return None


def resolve_workspace(entries: list[Any], cwd: str) -> tuple[dict[str, Any] | None, str]:
    """Reverse-look-up the home library by the connection's workspace path
    (C1: exact match on ``workspace_root``).

    Args:
        entries: directory entries (raw or normalized both fine; normalized
            once more internally).
        cwd: the connection's workspace path (launcher HELLO's ``cwd``).

    Returns:
        ``(entry | None, status)``. status has four values: ``bound`` (exactly
        one hit; entry is that entry) / ``unbound`` (no hit) / ``ambiguous``
        (two or more hits -- **the first is not taken**; the error state is
        left for the caller to self-expose: an ambiguous home library is more
        dangerous than none) / ``no_cwd`` (cwd empty; nothing to look up).

    Matching is **exact equality** after ``os.path.normcase`` normalization
    (v1 decision: no prefix matching; the monorepo subdirectory scenario is
    left for an observation period before deciding -- design draft §3.2).
    domain only decides; where the directory is read from and how a hit
    takes over the connection's default library are read/write-side (server)
    matters.
    """
    q = os.path.normcase(str(cwd or "").strip())
    if not q:
        return None, "no_cwd"
    hits = [e for e in normalize_entries(entries)
            if e["workspace_root"] and os.path.normcase(e["workspace_root"]) == q]
    if not hits:
        return None, "unbound"
    if len(hits) > 1:
        return None, "ambiguous"
    return hits[0], "bound"


def resolve_user_access(
    *, home_is_user: bool, binding_status: str, entry_access: str,
    escape_open: bool,
) -> str:
    """D2b read-side access-level decision (single source of truth, shared by
    the composition root and tests).

    Args:
        home_is_user: whether this connection's home library is the user
            library (**counts as the full basis only in the no-workspace-
            declaration form** -- narrowed 2026-10-31, see below).
        binding_status: C1 binding status (bound/unbound/ambiguous/no_cwd/
            no_user_library/directory_unreadable/lookup_error/undeclared).
        entry_access: the hit entry's ``access`` declaration (empty =
            undeclared).
        escape_open: escape hatch (HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1) --
            read/write fully open during the ops window, same convention as
            write-side isolation.

    Returns:
        ``full`` / ``masked`` / ``none``.

    Narrowing (decided 2026-10-31, migration doc §六): the old version's
    first check ``home_is_user -> full`` had a hole under user-level
    installation -- an unbound workspace falling back to the declared
    library (= the user library) would be judged "home" and get full
    access. New semantics (symmetric: full = authorized to treat the user
    library as one's own home):

    * ``bound``: the entry declaration wins -- only a declared ``full`` is
      full (privilege escalation must be explicit; checking the household
      registry does not rely on inference), default masked, declared none
      means none;
    * ``unbound``/``ambiguous``/``no_user_library``/``directory_unreadable``/
      ``lookup_error`` -> **none** (fail-closed, Q-8's leaning; the hole of
      falling back to "home" closes here -- even if the home library happens
      to be the user library);
    * ``no_cwd``/``undeclared`` (direct connection with no workspace
      declaration): home=user means **full** -- this is not a fallback but
      A1's everyday-host explicitly single-library deployment form (no cwd
      at all, so no "strange project fallback" attack surface); the rest is
      masked (D2b's existing ruling: the direct connection's default level;
      the U1 inheritance slot of a single-library deployment must not be
      killed wholesale).
    """
    if escape_open:
        return "full"
    if binding_status == "bound":
        return entry_access or "masked"
    if binding_status in ("unbound", "ambiguous", "no_user_library",
                          "directory_unreadable", "lookup_error"):
        return "none"
    if home_is_user:
        return "full"
    return "masked"
