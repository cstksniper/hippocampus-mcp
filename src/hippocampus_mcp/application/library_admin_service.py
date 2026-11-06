# -*- coding: utf-8 -*-
"""LibraryAdminService: cross-library management use cases (5 tools).

Tool surface (the other half of the original "Maintain 11"): library_list /
library_info / library_whoami / library_use / library_create.

**Why not fold into MaintainService / why not inherit ``BaseService``**: the
common trait of these 5 tools is that they are **not bound to a single
brain**--

* ``library_info`` deliberately does not construct a HippoBrain (it only
  reads catalog metadata, no need to wait 25-33 seconds for the embedding
  model), while the ``BaseService`` construction contract is "one resolved
  target library + its brain". If they went through the same factory,
  constructing the service would immediately construct the brain -- exactly
  what this design intent wants to avoid;
* ``library_use`` / ``library_create`` / ``library_list`` operate on the
  **library collection and session binding**, not on the contents of one
  library.

Hence its own class, with all dependencies **constructor-injected** via
:class:`LibraryPorts`: storage (registry / binding table / connection-level
default library) stays in the adapter layer (server.py); use cases only
declare what they need. This aligns with the boundary already written in
base.py -- "services do not touch library discovery / session stickiness,
that connection-state stuff"; here it is made concrete in an interface
rather than only written in a docstring.

The orchestration logic corresponds line for line with the server.py tool
bodies (minus ``json.dumps``); only relocated, behavior unchanged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from ..library import (LibraryInfo, LibraryResolutionError, default_roots,
                       library_id_for, normpath, resolve_library,
                       write_manifest)
# Borrowed-private note: the library-name fallback inference has only this one
# implementation in the library package (server.py originally used it as
# ``_library._guess_name`` too). Rather than duplicating the naming logic in
# the service layer (two copies would drift sooner or later), keep this
# explicit borrow; when it is promoted to a public API there is exactly one
# place to change.
from ..library import _guess_name
#: The ``permission`` key of ``whoami`` uses the same decision
#: function as the write guard -- the self-reported criterion must share a
#: source with the enforcement criterion, otherwise "what reports itself and
#: what blocks are not the same thing" all over again.
from ..domain.write_boundary import classify_write_permission

logger = logging.getLogger("hippocampus_mcp")


@dataclass(frozen=True)
class LibraryPorts:
    """Inventory of external capabilities for cross-library use cases (the
    composition root fills it with server.py's connection-state functions).

    Every entry is **state or I/O living in the adapter layer**: the
    library-list TTL cache, connection-level default library, session
    binding table, brain registry, ``os.makedirs`` / ``os.listdir``. What
    the use case receives is a "capability", not "the state itself" -- a
    counterexample: passing the ``_library_binding`` dict straight in would
    amount to letting the use-case layer hold connection state.
    """

    #: ``libraries(force)``: read-only scan with a TTL cache (zero libraries builds nothing)
    libraries: Callable[[bool], list[Any]]
    #: Default library path in the current call context (connection level > process level)
    default_path: Callable[[], str]
    #: ``resolve_library_path(name_or_none)``: explicit > sticky > default; raises when ambiguous
    resolve_path: Callable[[str | None], str]
    #: Library path bound to the current session (returns the default library when unbound)
    binding_get: Callable[[], str]
    #: ``set_library_binding(path, client)``: returns the effective session key
    binding_set: Callable[[str, str | None], str]
    #: ``_brain_for(path)``: **only** for ``create`` (initializes storage after creation)
    brain_for: Callable[[str], Any]
    #: Resident brain registry stats (the ``resident`` section of ``whoami``)
    registry_stats: Callable[[], dict[str, Any]]
    #: Boolean version of ``os.makedirs(path, exist_ok=True)``: the adapter layer already speaks up on failure
    ensure_dir: Callable[[str], bool]
    #: Layer file names in the directory -> byte sizes (returns ``{"_error": ...}`` when unreadable)
    layer_files: Callable[[str], dict[str, Any]]
    #: Background-thread prefetch of a library's brain (failure does not affect binding)
    prefetch: Callable[[str], None]
    #: Register the newly created library into the user-level task directory
    #: (``<user library>/directory.json``, the **only automatic hook** of the
    #: creation flow).
    #: Input is an entry dict (task/name/library_id/path/note); returns
    #: ``{"registered": bool, "reason"?: str, ...}``; failures are
    #: self-reported by the caller and are not raised into the creation flow.
    task_directory_register: Callable[[dict[str, Any]], dict[str, Any]]


class LibraryAdminService:
    """Library management use cases: list / inspect / self-report / switch /
    create (discovery and manifest writing live in the library package)."""

    def __init__(self, ports: LibraryPorts) -> None:
        self.ports = ports

    # ---------------------------------------------------------------- Internal

    def _default_info(self) -> Any:
        """LibraryInfo of the default library -- **it must be able to
        self-report even when it is not under any scan root**.

        Equivalent to server's ``default_library_info()`` (goes through the
        same default_path port, so under the daemon it is counted per
        connection; behavior unchanged).
        """
        path = self.ports.default_path()
        for i in self.ports.libraries(False):
            if i.path == path:
                return i
        return LibraryInfo(path=path, library_id=library_id_for(path),
                           name=_guess_name(path), explicit=False)

    # ---------------------------------------------------------------- List

    def list_libraries(self, *, include_backups: bool) -> dict[str, Any]:
        """List discovered libraries. **Read-only** -- the only exception is
        the zero-library first start creating the default directory.

        Backup directories are not listed by default (they are read-only
        snapshots and must not become write targets; enable
        ``include_backups`` only for disk-usage inventories);
        ``auto_created`` is the self-report of "a directory was really
        created this time" -- previously, with zero libraries, list returned
        empty and the user could not see what to do next.
        """
        ports = self.ports
        infos = ports.libraries(True)
        rows: list[dict[str, Any]] = []

        def _collect(infos_: object) -> None:
            for i in infos_:
                if i.backup and not include_backups:
                    continue
                d = i.to_dict()
                d["is_default"] = (i.path == ports.default_path())
                d["is_bound"] = (i.path == ports.binding_get())
                rows.append(d)

        _collect(infos)

        # Zero-library first start auto-creates default.
        auto_created = None
        _cur_default = ports.default_path()
        if not rows and _cur_default:
            if ports.ensure_dir(_cur_default):
                auto_created = _cur_default
                rows.clear()
                _collect(ports.libraries(True))

        return {
            "libraries": rows,
            "count": len(rows),
            "default_path": _cur_default,
            "bound_path": ports.binding_get(),
            "roots": default_roots(),
            "auto_created": auto_created,
        }

    # ---------------------------------------------------------------- Inspect

    def describe(self, *, library: str) -> dict[str, Any]:
        """Details of one library (path/id/name/aliases/layer files).
        **Deliberately does not construct a brain**.

        When ``resolve_path`` raises (unknown library name / pointed at a
        backup directory) it is not swallowed -- "unsure which library you
        mean" must be visible to the caller, **never silently falling back
        to the default library**.
        """
        ports = self.ports
        path = ports.resolve_path(library or None)
        info = None
        for i in ports.libraries(False):
            if i.path == path:
                info = i
                break
        if info is None:
            info = self._default_info() if path == ports.default_path() else None
        if info is None:
            info = LibraryInfo(path=path, library_id=library_id_for(path),
                               name=_guess_name(path))
        out = info.to_dict()
        out["is_default"] = (info.path == ports.default_path())
        out["is_bound"] = (info.path == ports.binding_get())
        out["layer_files"] = ports.layer_files(path)
        return out

    # ---------------------------------------------------------------- Self-report

    def whoami(self) -> dict[str, Any]:
        """Which library this call lands in + who can be written -- the
        first anti-cross-library self-report (including the ``source``
        basis).

        Three keys:

        * ``home`` -- the **writable home library** of this connection (the
          connection-level default library; writes are always pinned here);
        * ``current`` -- the **addressing target** (after ``library_use`` it
          may be another library; reads land here);
        * ``permission`` -- what a write to ``current`` would do now:
          ``read-write`` (home library) / ``managed-write`` (the user
          managed channel) / ``read-only`` (blocked by the guard).

        Criterion discipline: ``permission`` calls the **guard's own** pure
        function directly (``domain.write_boundary.classify_write_permission``)
        and does not re-implement a judgment here -- self-report and
        enforcement must share a source, otherwise what the agent sees and
        what actually executes are two different things. The escape-hatch
        env is a **process-level ops switch** and is not reflected in the
        self-report: when it is on, the self-report still says
        ``read-only`` (conservative; it will not mislead the agent into
        cross-library writes).

        ``library`` is a legacy key kept with the same value as ``current``
        (tool surface compatibility).
        """
        ports = self.ports
        bound = ports.binding_get()
        home_path = ports.default_path()
        is_bound = bound != home_path
        info = None
        for i in ports.libraries(False):
            if i.path == bound:
                info = i
                break
        if info is None:
            info = self._default_info()
        home = self._default_info()
        current = {"id": info.library_id, "name": info.name, "path": info.path}
        return {
            "library": dict(current),
            "home": {"id": home.library_id, "name": home.name,
                     "path": home.path},
            "current": current,
            "permission": classify_write_permission(
                target_path=info.path, home_path=home_path,
                target_name=info.name),
            "source": "binding" if is_bound else "default",
            "default_path": home_path,
            "resident": ports.registry_stats(),
        }

    # ---------------------------------------------------------------- Switch

    def use(self, *, library: str, client: str) -> dict[str, Any]:
        """Session-level switch: tools in this session now target that
        library automatically. **Does not auto-create** (a typo simply
        fails).

        After binding, prefetch in the background (otherwise the first call
        pays the 25-33 second load itself) -- do not change the order: the
        binding lands first, and a prefetch failure does not affect the
        binding itself.
        """
        ports = self.ports
        info = resolve_library(library, libs=ports.libraries(True),
                               default_path=ports.default_path())
        if info is None:
            raise LibraryResolutionError("library_use requires a library name")
        if info.backup:
            raise LibraryResolutionError(
                f"{info.name} is a backup/snapshot directory; it is read-only and cannot be switched to.")
        key = ports.binding_set(info.path, client or None)
        ports.prefetch(info.path)
        return {
            "status": "bound",
            "library": {"id": info.library_id, "name": info.name,
                        "path": info.path},
            "session_key": key,
            "note": "Read tools in this session now target this library automatically; writes remain pinned to the home library "
                    "(see library_whoami home). Pass library= for a one-call override.",
        }

    # ---------------------------------------------------------------- Create

    def create(self, *, name: str, path: str, aliases: str, use: bool,
               client: str, workspace_root: str = "") -> dict[str, Any]:
        """Create a new memory library (writes the manifest + initializes
        storage, optionally switching over immediately).

        Creation is an **explicit action**: only this path writes the
        manifest; the routine scan never writes to disk. Duplicate names and
        "a same-named path exists but is not a directory" both raise
        immediately -- the price of ambiguous addressing is writing to the
        wrong library.

        After a successful creation the library is **also registered into
        the user-level task directory** (the only automatic hook of that
        sidecar; part of the managed channel). A registration failure **does
        not block creation**, but it must speak up --
        ``directory_registered: false`` + ``directory_reason`` + one
        warning: a silently missed registration makes the new library
        "invisible from birth" (all later cross-library addressing cannot
        find it, while the user believes it was registered).
        """
        ports = self.ports
        nm = (name or "").strip()
        if not nm:
            raise LibraryResolutionError("Library name cannot be empty")
        for i in ports.libraries(True):
            if not i.backup and i.name.lower() == nm.lower():
                raise LibraryResolutionError(
                    f"Library name {nm} already exists ({i.path}). Duplicate names make addressing ambiguous; choose another.")
        target = (path or "").strip()
        if not target:
            target = os.path.join(default_roots()[0], "hippo_" + nm)
        target = normpath(os.path.expanduser(target))
        if os.path.exists(target) and not os.path.isdir(target):
            raise LibraryResolutionError(f"{target} already exists and is not a directory")
        alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
        info = write_manifest(target, nm, alias_list)
        # Initialize storage right after creation: an empty directory does not automatically become a "ready library" before the next scan.
        ports.brain_for(target).save()
        out: dict[str, Any] = {
            "status": "created",
            "library": {"id": info.library_id, "name": info.name,
                        "path": info.path},
            "aliases": alias_list,
        }
        # Creation registers into the user-level task directory. Registration
        # is a giveaway and never blocks creation; but a failure must
        # self-report -- "invisible from birth" is harder to debug than a
        # failed creation.
        reg: dict[str, Any] = {"registered": False, "reason": "register_error"}
        try:
            entry = {"task": info.name, "name": info.name,
                     "library_id": info.library_id, "path": info.path,
                     "note": ""}
            if (workspace_root or "").strip():
                # C2 managed binding (Q-B/Q-6 re-review landed): creating a
                # library binds the workspace -- the third step of onboarding
                # "ask once, create once, bind once" becomes one step here.
                entry["workspace_root"] = workspace_root.strip()
            reg = ports.task_directory_register(entry)
        except Exception as exc:  # noqa: BLE001 -- a registration failure must not blow up creation
            reg = {"registered": False, "reason": f"register_error: {exc}"}
        out["directory_registered"] = bool(reg.get("registered"))
        if not out["directory_registered"]:
            out["directory_reason"] = str(reg.get("reason") or "unknown")
            logger.warning(
                "Task directory registration failed (library %s was created but not added to the user-level task directory: %s). "
                "Cross-library lookup cannot find it by task name; add it to directory.json manually or check the user library.",
                info.name, out["directory_reason"])
        if use:
            out["session_key"] = ports.binding_set(info.path, client or None)
        # Field feedback 2026-10-31: the write boundary is pinned to the
        # **connection-level** home library, so on the creation connection
        # ingest cannot write into the new library -- without a hint the
        # agent only finds out from a write error, paying an extra round of
        # trial and error. "Create library + write the first memory"
        # naturally splits across two sessions; this spells that out at
        # creation time. The note is a giveaway: failing to get the home
        # name never blows up creation.
        try:
            out["note"] = (
                f"Write boundary note: writes on this connection remain pinned to the original home library"
                f" ({self._default_info().name}). To write the first memory into "
                f"{info.name}, start a new session (the new connection HELLO will bind by "
                "workspace_root to the new library). Read side can use library_use to switch immediately.")
        except Exception:  # noqa: BLE001 -- the note is a giveaway, never blows up creation
            out["note"] = (
                f"Write boundary note: writes on this connection remain pinned to the original home library; to write the first memory into "
                f"{info.name}, start a new session (the new connection HELLO will bind by "
                "workspace_root to the new library).")
        return out
