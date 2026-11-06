# -*- coding: utf-8 -*-
"""Multi-library addressing zone (the implementation body of server.py's
multi-library addressing block).

Implemented here: workspace guide file (``GUIDE_FILENAME`` /
``_find_workspace_guide``), default library resolution
(``_resolve_default_lib_path``), session key (``session_key``), TTL-cached
library listing scan (``libraries``), three-level library resolution
(``resolve_library_path``: **explicit > sticky > default**), session-level
sticky binding (table + lock + the ``set/get/clear_library_binding`` trio),
library self-disclosure (``_library_self``), envelope assembly
(``_with_library``), and cross-library port I/O helpers
(``_ensure_default_dir`` / ``_layer_files`` / ``_prefetch_library``).

🔴 State and constants are **deliberately not moved** -- they stay in server.py
(the tests' monkeypatch surface): ``_DEFAULT_LIB_PATH`` (assignment patch),
``_conn_default_lib`` / ``_conn_lib_lock`` (clear), ``_library_cache`` /
``_library_cache_at`` / ``_LIBRARY_CACHE_TTL`` (set to None then rescan),
``_registry`` / ``_registry_lock`` / ``_registry_get`` (patch),
``_PERSIST_DIR``. Every read and write of these symbols in this module goes
through :func:`_srv` and is **fetched live** (lesson: snapshotting them into a
new module = monkeypatches silently stop working); calls to server-side
functions (``default_library_path`` / ``_registry_get``) are likewise fetched
live -- only then do test replacements of ``server._brain_for`` /
``server._registry_get`` take effect all the way into addressing and prefetch.
The sticky binding table (``_library_binding`` / ``_library_binding_lock``)
moves with this module: no direct patch surface anywhere in the repo (externals
operate only through the set/get/clear APIs).

server.py re-exports all public symbols of this module unchanged under the
same import names, so existing call sites (the tool bodies' three-level
resolution / library self-disclosure first key / envelope assembly) and direct
test calls (``S.resolve_library_path`` / ``S._find_workspace_guide`` /
``S.GUIDE_FILENAME`` ...) need zero changes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from hippocampus_mcp import library as _library
from hippocampus_mcp.common import session_ctx as _session_ctx
from hippocampus_mcp.library import LibraryResolutionError  # noqa: F401 (the resolve signature must raise it)

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """The server module itself (fetched live; see the module docstring's monkeypatch surface notes)."""
    from hippocampus_mcp import server as _s

    return _s


# Workspace guide file name: placing this one file in a home directory points the default library elsewhere.
GUIDE_FILENAME = ".hippocampus.json"


def _find_workspace_guide(start: str | None = None) -> str | None:
    """Walk upward from ``start`` (default cwd) looking for the workspace guide
    file (at most 5 levels).

    The guide file mechanism is a **zero-config convenience**: if the file does
    not exist it has no effect at all and changes no existing behavior -- the
    main channel is always explicit addressing by the caller.
    """
    d = os.path.abspath(start or os.getcwd())
    for _ in range(5):
        cand = os.path.join(d, GUIDE_FILENAME)
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def resolve_default_lib_ex(
    persist_dir: str | None = None,
    start: str | None = None,
) -> tuple[str, dict[str, str] | None]:
    """The **full version** of default library path resolution:
    ``(path, fallback detail)``.

    Default library path: guide file > env ``HIPPOCAMPUS_PERSIST_DIR``.

    ``persist_dir`` / ``start`` are **connection-level** inputs: one daemon
    process serves N connections, each of which may belong to a different
    workspace; they cannot share the copy computed at process startup. When not
    passed, module-level env / cwd is used, **bit-for-bit identical** to before
    the rework.
    🔴 Exception: ``persist_dir`` explicitly passed while ``start`` is empty =>
    **do not consult the guide file**, use the declared value directly -- the
    process cwd does not belong to this connection (cross-workspace library
    cross-contamination hole, see the code comment below); the process-level
    default (both None) is still "guide file > env".

    🔴 The guide file **only accepts values that resolve unambiguously to a
    library**. If unresolvable, log a warning and fall back to the env default --
    that is how a "convenience" should behave (it must not be able to keep the
    service from starting), and it must be **loud**, not silent.

    The fallback detail is non-``None`` only when "declared but not used":
    ``declared_raw`` (the raw declared name) / ``reason`` (``unresolvable`` =
    unresolvable | ``read_error`` = file broken or not JSON) / ``detail``
    (exception or explanation text). Server logs are invisible to the agent
    (real-world feedback 2026-10-31); these three keys are the source of truth
    for session_start's top-level warning block -- see
    ``server._inject_fallback_warning``.

    ``_PERSIST_DIR`` is fetched live via :func:`_srv` (server namespace).
    """
    s = _srv()
    base = _library.normpath(os.path.expanduser(
        persist_dir if persist_dir is not None else s._PERSIST_DIR))
    # 🔴 The guide file belongs to the "cwd semantics" only. For a
    # connection-level **explicit declaration** (persist_dir passed) whose
    # connection carries no cwd, never use the process cwd to find the guide
    # file: the daemon serves N workspaces and the process cwd belongs to the
    # **first** one; using it to refine someone else's connection declaration =
    # cross-workspace library cross-contamination (f9_4e group C exposed exactly
    # this hole when a .hippocampus.json sat at this repo's root); clients
    # without a handshake line (tests / manual direct connections, the f9_4e red
    # line) must also be completely unaffected by the workspace the host process
    # happens to sit in. With a cwd, refine as before (the launcher sending cwd
    # exists for exactly this, see t_b5d); the process-level default (both None)
    # remains "guide file > env".
    if (persist_dir or "").strip() and not (start or "").strip():
        return base, None
    guide = _find_workspace_guide(start)
    if not guide:
        return base, None
    want = ""
    try:
        with open(guide, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        want = str((cfg or {}).get("library") or "").strip()
        if not want:
            return base, None
        info = _library.resolve_library(want, default_path=base)
        if info is None:
            logger.warning(
                "Workspace guide file %s library %r resolved to empty; falling back to env default library",
                guide, want)
            return base, {"declared_raw": want, "reason": "unresolvable",
                          "detail": "resolve_library returned None"}
        logger.info("Workspace guide file %s set default library: %s (%s)", guide, info.name, info.path)
        return info.path, None
    except _library.LibraryResolutionError as e:
        # Declared but unresolvable: be loud, and do **not** use it as the default library (cross-library use is far more dangerous than falling back)
        logger.warning("Workspace guide file %s library was unresolvable; falling back to env default library: %s", guide, e)
        return base, {"declared_raw": want, "reason": "unresolvable",
                      "detail": str(e)}
    except Exception as e:      # file broken / not JSON: likewise fall back loudly
        logger.warning("Workspace guide file %s read failed; falling back to env default library: %s", guide, e)
        return base, {"declared_raw": want, "reason": "read_error",
                      "detail": str(e)}


def _resolve_default_lib_path(persist_dir: str | None = None,
                              start: str | None = None) -> str:
    """Default library path (thin shell): guide file > env. The fallback detail
    is discarded; signature/return value stay **bit-for-bit identical** to
    before the split -- existing test stubs patch the old signature and must not
    break; for fallback detail use :func:`resolve_default_lib_ex`.
    """
    path, _fallback = resolve_default_lib_ex(persist_dir, start)
    return path


# Session-level sticky binding: session key -> library path.
# The key space matches the repo-wide session identity convention exactly:
# **explicit client > ctx.session > __default__**.
# Under direct stdio there is only the __default__ path (one process = one session);
# once the daemon binds ctx sessions, per-connection routing happens
# automatically; nothing to change here.
# The table and lock move with the addressing zone (no direct patch surface
# anywhere in the repo; externals only go through the set/get/clear APIs).
_library_binding: dict[str, str] = {}
_library_binding_lock = threading.RLock()


def session_key(client: str | None = None) -> str:
    """The session key for this call (shares one namespace with all other
    session-identity consumers in the repo).

    An explicit ``client`` signature has the highest priority (the caller
    reports its own identity); without a signature, the context key from
    :mod:`session_ctx` is read -- **always ``__default__`` under direct stdio**
    (one process serves one session, behavior identical to before the rework);
    after daemonization, each connection binds before starting its session, so
    the N sessions inside one daemon process each land on their own key and
    ``library_use()`` stickiness does not cross.

    session_ctx is imported directly in this module, not via server.
    """
    c = (client or "").strip()
    if c:
        return c
    return _session_ctx.current_session()


def libraries(force: bool = False) -> list[Any]:
    """The discovered library list (with TTL cache). **Read-only**: creates and
    modifies no files.

    The cache trio (``_library_cache`` / ``_library_cache_at`` /
    ``_LIBRARY_CACHE_TTL``) stays in server.py (monkeypatch surface), fetched
    live via :func:`_srv` -- once tests set it to ``None``, this function
    rescans immediately.
    """
    s = _srv()
    now = time.time()
    if (not force and s._library_cache is not None
            and now - s._library_cache_at < s._LIBRARY_CACHE_TTL):
        return s._library_cache
    try:
        found = _library.scan_libraries()
    except Exception as exc:  # noqa: BLE001 -- a scan failure must not take the tool down
        logger.warning("Library scan failed: %s", exc)
        found = []
    s._library_cache = found
    s._library_cache_at = now
    return found


def resolve_library_path(library: str | None = None,
                         client: str | None = None) -> str:
    """Decide which library this call lands in (returns the normalized absolute
    path).

    Priority: **explicit library > session-level sticky > default library**.

    Raises:
        LibraryResolutionError: explicitly specified but unresolvable
            (including ambiguous duplicate names).
            **Never silently fall back to the default library** -- that is the
            gateway to cross-library contamination.

    The library list/default library are fetched live via :func:`_srv`
    (``s.libraries()`` / ``s.default_library_path()`` -- the server namespace's
    patch surface stays effective end to end); the sticky binding table moves
    with this module (no direct patch surface; externals only go through the
    set/get/clear APIs).
    """
    s = _srv()
    q = (library or "").strip()
    if q:
        info = _library.resolve_library(
            q, libs=s.libraries(), default_path=s.default_library_path(client))
        if info is None:  # theoretically unreachable (with non-empty q it either hits or raises)
            raise LibraryResolutionError(f"Library {q} could not be resolved")
        if info.backup:
            raise LibraryResolutionError(
                f"{info.name} is a backup/snapshot directory; it is read-only and cannot be a write target."
                f"Use a real library name or library_id.")
        return info.path
    with _library_binding_lock:
        bound = _library_binding.get(session_key(client))
    if bound:
        return bound
    return s.default_library_path(client)


def set_library_binding(path: str, client: str | None = None) -> str:
    """Session-level sticky binding (where ``library_use`` lands). Returns the effective key."""
    key = session_key(client)
    with _library_binding_lock:
        _library_binding[key] = _library.normpath(path)
    return key


def get_library_binding(client: str | None = None) -> str:
    """The library path bound to the current session (returns the default library path if unbound)."""
    with _library_binding_lock:
        return _library_binding.get(session_key(client)) or _srv().default_library_path(client)


def clear_library_binding(client: str | None = None) -> bool:
    """Unbind (return to the default library). Returns whether a binding was actually removed."""
    key = session_key(client)
    with _library_binding_lock:
        return _library_binding.pop(key, None) is not None


def _library_self(library: str | None = None) -> dict[str, str]:
    """Library self-disclosure: the target library of this call
    (``{id, name, path}``).

    It is the **first key** of every write response body. Cross-library
    contamination is scary because a write to the wrong library looks identical
    in the response body to a write to the right one; this line puts the
    difference right up front.

    Resolution and listing are fetched live via :func:`_srv`
    (``s.resolve_library_path`` / ``s.libraries()``).
    """
    s = _srv()
    path = s.resolve_library_path(library)
    name = ""
    lid = ""
    for i in s.libraries():
        if i.path == path:
            name, lid = i.name, i.library_id
            break
    if not lid:
        lid = _library.library_id_for(path)
        name = _library._guess_name(path)
    return {"id": lid, "name": name, "path": path}


def _with_library(payload: Any, library: str | None = None) -> Any:
    """Insert the ``library`` first key into write response bodies. **Only dict
    payloads get the key**.

    Non-dict returns (error-branch strings, lists, etc.) are returned as-is --
    no key added, so the error path's structure is not broken.

    Tools routed through the application service layer no longer go through
    this function -- the envelope is produced by the service layer itself
    (``BaseService._envelope``, the same bit-for-bit rules as this function).
    The three hard constraints of envelope assembly and error-body rules (insert
    the key only for dicts, return non-dict error branches as-is, use identical
    bit-for-bit serialization parameters on both sides) must hold identically at
    both assembly points, this function and ``BaseService._envelope``;
    ``_map_fields_error`` (envelope assembly for map_set fields parse errors) is
    one of this function's active users.

    Library self-disclosure is fetched live via :func:`_srv`
    (``s._library_self``).
    """
    if isinstance(payload, dict):
        return {"library": _srv()._library_self(library), **payload}
    return payload


# ------------------------------------------------------------------
# Cross-library use-case port implementations (connection state and I/O stay
# in the adapter layer; use cases only receive capabilities)
# (the three I/O helpers move with the addressing zone; the registry is
# fetched live via _srv())
# ------------------------------------------------------------------
def _ensure_default_dir(path: str) -> bool:
    """Boolean version of ``os.makedirs(path, exist_ok=True)``: on failure, be
    **loud** and return False.

    The zero-library first-start path is the only branch where ``library_list``
    creates anything; a failed directory creation must not take the listing tool
    down, but it must not be silent either -- "a default library was created
    automatically" is a self-disclosure people will take at face value.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logger.warning("Zero-library first-start failed to create default (use unaffected): %s", e)
        return False


def _layer_files(path: str) -> dict[str, Any]:
    """Layer file names in the library directory -> byte sizes (``brain_*`` /
    ``memories.json`` / manifest).

    When the directory is unreadable, put the reason into ``_error``:
    self-disclosure beats availability -- rather return a structure carrying the
    error than have the tool die entirely on a stat failure.
    """
    layers: dict[str, Any] = {}
    try:
        for fn in sorted(os.listdir(path)):
            if fn.startswith("brain_") or fn in ("memories.json",
                                                 _library.MANIFEST_NAME):
                fp = os.path.join(path, fn)
                if os.path.isfile(fp):
                    layers[fn] = os.path.getsize(fp)
    except OSError as e:
        layers["_error"] = str(e)
    return layers


def _prefetch_library(path: str) -> None:
    """Prefetch a library's brain in a background thread (avoids the first call
    after a library switch paying the 25-33 second load itself).

    The registry is fetched live via :func:`_srv` -- after tests replace
    ``server._registry_get``, the prefetch gets the stand-in.
    """
    def _prefetch() -> None:
        try:
            _srv()._registry_get().get(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Library prefetch failed (later lazy loading unaffected): %s", e)

    threading.Thread(target=_prefetch, name="hippo-lib-prefetch",
                     daemon=True).start()
