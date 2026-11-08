# -*- coding: utf-8 -*-
"""Zero-step project onboarding: ``hippocampus init --project <dir>``.

Usability-closure WS-3 (2026-11-08, plan:
``docs/design/Agent易用性收口_执行计划书_20261108.md`` §3). One command that
makes a NEW project memory-ready with no manual steps:

1. create the library (manifest + initialized storage) under the default
   library root, named after the project directory (``--name`` overrides;
   collisions get a ``-2``..``-9`` suffix instead of an error);
2. register the ``workspace_root -> library`` binding in the user-level task
   directory (the same sidecar the daemon's HELLO reverse-lookup reads —
   after this, a NEW session in the project lands on the new library);
3. write the memory-discipline template into the project's ``AGENTS.md``
   (idempotent marker block; default ON per decision P-3, ``--no-agents``
   skips). Existing AGENTS.md content is preserved: markers present → the
   block is replaced in place; no markers → the block is appended.

Idempotent: re-running on a bound project reports ``already_bound`` and only
refreshes the AGENTS block. Nothing here touches any OTHER library, and no
server/daemon needs to be running — everything is done through the library
package's own primitives (``write_manifest`` / ``upsert_entry``), the exact
code paths ``library_create`` uses.

Privacy/scope: only the new library's directory and the user-level sidecar
are written. The user library is created on first use if missing (mirroring
what ``library_create("user")`` would have produced).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .brain import HippoBrain
from .library import (default_roots, library_id_for, normpath,
                      scan_libraries, write_manifest)
from .library.task_directory import (DIRECTORY_SCHEMA, load_directory,
                                     save_directory, upsert_entry)

__all__ = ["run_init", "write_agents_snippet"]

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_BEGIN_MARK = "<!-- hippocampus:begin"
_END_MARK = "<!-- hippocampus:end -->"


def _user_lib_default() -> str:
    """Same location the CLI/library code uses for the user-level library."""
    return os.path.join(os.path.expanduser("~"), ".hippocampus_mcp", "user")


def _load_template() -> str:
    return (_TEMPLATES_DIR / "agents_snippet.md").read_text(encoding="utf-8")


def write_agents_snippet(project: str) -> str:
    """Ensure ``<project>/AGENTS.md`` carries the marker block.

    Returns one of ``created`` / ``appended`` / ``replaced`` / ``unchanged``.
    Content OUTSIDE the markers is never touched.
    """
    block = _load_template().strip()
    path = Path(project) / "AGENTS.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if _BEGIN_MARK in text and _END_MARK in text:
            pre = text.split(_BEGIN_MARK, 1)[0]
            post = text.split(_END_MARK, 1)[1]
            new = pre + block + "\n" + post
            if new == text:
                return "unchanged"
            path.write_text(new, encoding="utf-8", newline="\n")
            return "replaced"
        appended = text.rstrip() + "\n\n" + block + "\n"
        path.write_text(appended, encoding="utf-8", newline="\n")
        return "appended"
    path.write_text(block + "\n", encoding="utf-8", newline="\n")
    return "created"


def _pick_name(name: str, root: str, project: str) -> str:
    """Library name: explicit > directory basename; on a name collision with
    a DIFFERENT library, append ``-2``..``-9`` (never an error for a mere
    naming coincidence — the binding is by path, not by name)."""
    existing: dict[str, str] = {}
    for info in scan_libraries([root]):
        if getattr(info, "backup", False):
            continue
        existing[(info.name or "").lower()] = normpath(info.path)
    base = (name or "").strip() or Path(project).name or "project"
    cand = base
    for n in range(2, 10):
        if cand.lower() not in existing or \
                existing[cand.lower()] == normpath(project):
            return cand
        cand = f"{base}-{n}"
    raise RuntimeError(
        f"library name {base!r} and suffixes -2..-9 are all taken under "
        f"{root}; pass --name explicitly")


def _ensure_user_lib(user_lib: str) -> str:
    """Create the user-level library on first use (mirrors what
    ``library_create("user")`` would have produced) and guarantee the task
    directory sidecar exists with the ``user`` self-entry registered."""
    manifest = os.path.join(user_lib, "hippocampus.json")
    if not os.path.exists(manifest):
        write_manifest(user_lib, "user")
        HippoBrain(persist_dir=user_lib).save()
    if not os.path.exists(os.path.join(user_lib, "directory.json")):
        save_directory(user_lib, {"schema": DIRECTORY_SCHEMA, "entries": []})
        upsert_entry(user_lib, {
            "task": "user", "name": "user",
            "library_id": library_id_for(user_lib), "path": user_lib,
            "note": "self-registered by hippocampus init"})
    return user_lib


def run_init(project: str, *, name: str = "", root: str = "",
             user_lib: str = "", agents: bool = True,
             create_storage: bool = True) -> dict[str, Any]:
    """Execute the onboarding and return a machine-readable report.

    Args:
        project: workspace root to bind (must exist and be a directory).
        name: library name override (default: project directory basename).
        root: library root override (default: first configured root).
        user_lib: user-level library override (default: ``~/.hippocampus_mcp/user``).
        agents: write the AGENTS.md discipline block (decision P-3: default ON).
        create_storage: initialize the library storage immediately (tests may
            pass False to exercise the naming/binding logic without loading
            the embedding stack).

    Idempotent: if the project's workspace_root is ALREADY bound (any entry
    in the sidecar with the same normalized workspace path), creation and
    registration are skipped and ``status`` reads ``already_bound``; the
    AGENTS block is still refreshed.
    """
    project = normpath(os.path.abspath(os.path.expanduser(project)))
    if not os.path.isdir(project):
        raise NotADirectoryError(f"project directory not found: {project}")
    root = os.path.expanduser(root) if root else default_roots()[0]
    os.makedirs(root, exist_ok=True)
    user_lib = os.path.expanduser(user_lib) if user_lib else _user_lib_default()

    # Already bound? (any entry whose workspace_root normalizes to project)
    bound_entry = None
    if os.path.isdir(user_lib):
        data, _issue = load_directory(user_lib)
        want = normpath(project)
        for e in data.get("entries") or []:
            if normpath(str(e.get("workspace_root") or "")) == want:
                bound_entry = e
                break

    report: dict[str, Any] = {"project": project, "agents": "skipped"}
    if bound_entry is not None:
        report.update({
            "status": "already_bound",
            "name": bound_entry.get("task") or "",
            "target": bound_entry.get("path") or "",
        })
    else:
        os.makedirs(user_lib, exist_ok=True)
        _ensure_user_lib(user_lib)
        final = _pick_name(name, root, project)
        target = normpath(os.path.join(root, "hippo_" + final))
        manifest = os.path.join(target, "library.json")  # MANIFEST_NAME
        if os.path.exists(manifest):
            # Same name AND same path => a previous init already created it;
            # treat as re-init instead of failing.
            report["notes"] = ["library directory already initialized"]
        else:
            write_manifest(target, final)
            if create_storage:
                HippoBrain(persist_dir=target).save()
        reg = upsert_entry(user_lib, {
            "task": final, "name": final,
            "library_id": library_id_for(target), "path": target,
            "workspace_root": project,
            "note": "hippocampus init"})
        report.update({
            "status": "created",
            "name": final,
            "target": target,
            "registered": {"created": reg.get("created"),
                           "moved": reg.get("moved"),
                           "count": reg.get("count")},
        })

    if agents:
        report["agents"] = write_agents_snippet(project)
    report["next"] = ("restart the host session; verify with "
                      "`hippocampus doctor --host zcode --project <dir>` and "
                      "`library_whoami` inside the new session")
    return report
