# -*- coding: utf-8 -*-
"""Library catalog and discovery layer (**read-only**: never modifies any
existing data files).

Why this layer exists: the first thing multi-library support needs is not
"switching" but "**knowing which libraries exist**". Before a library name
can be resolved, calls like ``library_use("写作")`` can only guess, and a
wrong guess writes memories into someone else's library -- an accident
harder to discover than data loss (no error on either side).

Three design premises (all from field measurements; do not change on
intuition):

1. **Paths must be normalized**: measurements show the same workspace can
   yield both ``d:\\`` and ``D:\\`` drive letters; without normalization the
   same library is seen as two, leading to "two brains pointing at one
   store". The whole module uses :func:`normpath` exclusively.
2. **Backups are read-only**: ``.bak-*`` / ``.snap-*`` / ``corrupt``
   directories **must** be flagged ``backup`` and excluded from name
   resolution -- otherwise ``library_use("写作")`` could pick a historical
   snapshot, and all subsequent writes would land in the snapshot while the
   real library got nothing.
3. **Legacy libraries are lazily backfilled, never written to disk**:
   existing libraries have no ``library.json``; their names are inferred
   in **memory** from the directory name (stripping the ``hippo_`` prefix),
   **without persisting any file** -- the hard constraint of this layer is
   "never modify any existing data files". The manifest is only written
   when ``library_create`` builds a new library.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Construction injection seam: the knob is registered in config
# (library_roots) and fed in by engine at import time in production; for the
# parse order (env first, injected value as fallback) see common/config_bridge.
from ..common.config_bridge import apply_config, resolve as _resolve

#: Library manifest file name (lives inside the library directory)
MANIFEST_NAME = "library.json"

# Decide "is this directory a memory library": hitting any layer file counts.
# Layer files rather than "has subdirectories", because an empty library is
# still a library (the zero-library first start creates it).
_LAYER_MARKERS = (
    "memories.json",
    "brain_meta.json",
    "brain_raw.json",
    MANIFEST_NAME,
    # ⚠️ Note: newer libraries may only have hippocampus.db (sqlite backend)
    # and never wrote a json layer. Missing it ⇒ a pure sqlite library never
    # enters the library catalog ⇒ library_use / the workspace guide file
    # both fail to resolve it (measured: all three spellings raise
    # LibraryResolutionError; the host's PERSIST_DIR bypasses the catalog,
    # which is why it never surfaced).
    "hippocampus.db",
)

# Backup / snapshot / corrupted-quarantine directories: **read-only**,
# excluded from automatic selection
_BACKUP_PREFIXES = (".bak", ".snap", ".corrupt")
_BACKUP_TOKENS = ("corrupt",)

# Directory-name prefixes stripped when lazily backfilling legacy library
# names (data/hippo_写作 → 写作)
_NAME_PREFIXES = ("hippo_",)


class LibraryResolutionError(ValueError):
    """Library addressing failure.

    **Must be raised explicitly, never silently falling back to the default
    library** -- the hard rule of multi-library correctness. "The library you
    named was not found" and "use the default library" are two entirely
    different things; conflating them is cross-library contamination.
    """


def normpath(p: str) -> str:
    """Path normalization: casing + absolutization (measured: drive-letter
    casing is inconsistent)."""
    return os.path.normcase(os.path.abspath(p))


def library_id_for(path: str) -> str:
    """Library id: derived from the **normalized absolute path**.

    Derived from the path rather than the name so the id stays stable: after
    a rename, or a copy into ``.bak``, the id still shares the same origin,
    and identical ids in two places can be recognized as ``duplicate_of``.
    """
    return "lib_" + hashlib.sha1(normpath(path).encode("utf-8")).hexdigest()[:12]


def _looks_like_library(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for marker in _LAYER_MARKERS:
        if os.path.exists(os.path.join(path, marker)):
            return True
    return False


def _is_backup(dirname: str) -> bool:
    """Backup/snapshot detection.

    ⚠️ **Checking the prefix alone is not enough**: the actual backup
    directory name is ``hippo_写作.bak-<timestamp>`` -- ``.bak`` sits in the
    **middle**, not at the start (the backup script names it
    `<original>.bak-<timestamp>`). Testing only ``startswith(".bak")`` would
    classify all such backups as real libraries, directly violating "backups
    are excluded from automatic selection". Hence the "contains" test.
    """
    low = os.path.basename(dirname.rstrip("\\/")).lower()
    for tok in (*_BACKUP_PREFIXES, *_BACKUP_TOKENS):
        if tok in low:
            return True
    return low.startswith("bak-") or low.startswith("snap-")


def _guess_name(path: str) -> str:
    """Legacy-library lazy backfill: strip the ``hippo_`` prefix and leading
    dots from the directory name."""
    base = os.path.basename(path.rstrip("\\/")) or path
    low = base.lower()
    for pref in _NAME_PREFIXES:
        if low.startswith(pref):
            base = base[len(pref):]
            break
    return base.lstrip(".") or base or "unnamed"


def display_name(path: str) -> str:
    """Display name of a library: library.json's name → directory-name lazy
    backfill.

    When the manifest cannot be read, it is inferred from path by the
    ``_guess_name`` rules -- the authoritative identity always lives in the
    ``library`` key of the session_start response; this is just one label on
    a card, and falling back to the directory name when inference fails is
    nothing to be ashamed of.

    Same criterion as ``_info_for(path).name`` but **deliberately not
    reused**: that one ``normpath``s first and then infers the name, while
    this function infers from the raw path the caller provided (brain's
    persist_dir is already a normalized absolute path, so the two agree for
    those inputs, but for edge shapes -- e.g. un-normalized paths -- this
    implementation's behavior is authoritative).
    """
    try:
        fp = os.path.join(path or "", MANIFEST_NAME)
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        name = str((data or {}).get("name") or "").strip()
        if name:
            return name
    except Exception:  # noqa: BLE001 -- a missing/bad file is both treated as "absent"
        pass
    return _guess_name(path or "") or "unnamed"


@dataclass
class LibraryInfo:
    """A discovered memory library.

    Attributes:
        path: normalized absolute path (**unique identity**).
        library_id: ``lib_<first 12 hex of sha1>``, derived from path ⇒
            stable.
        name: human-readable name (manifest first, otherwise lazy backfill).
        aliases: alias list, participates in name resolution.
        backup: True = backup/snapshot/corrupted directory, **read-only and
            excluded from automatic selection**.
        explicit: True = the directory has ``library.json`` (False = lazy
            backfill, in-memory only).
    """

    path: str
    library_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    created_at: str = ""
    embedder_model: str = ""
    backup: bool = False
    explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_id": self.library_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "path": self.path,
            "created_at": self.created_at,
            "embedder_model": self.embedder_model,
            "backup": self.backup,
            "explicit": self.explicit,
        }


def _read_manifest(path: str) -> dict[str, Any]:
    """Read ``library.json``. Missing/corrupt are **not errors** (legacy
    libraries simply lack it)."""
    fp = os.path.join(path, MANIFEST_NAME)
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 -- a missing file is the norm; a bad file is treated as "absent"
        return {}
    return data if isinstance(data, dict) else {}


def _info_for(path: str) -> LibraryInfo:
    path = normpath(path)
    manifest = _read_manifest(path)
    base = os.path.basename(path.rstrip("\\/"))
    name = str(manifest.get("name") or "").strip() or _guess_name(path)
    aliases = manifest.get("aliases")
    aliases = [str(a).strip() for a in aliases if str(a).strip()] if isinstance(aliases, list) else []
    return LibraryInfo(
        path=path,
        library_id=str(manifest.get("library_id") or "").strip() or library_id_for(path),
        name=name,
        aliases=aliases,
        created_at=str(manifest.get("created_at") or ""),
        embedder_model=str(manifest.get("embedder_model") or ""),
        backup=_is_backup(base),
        explicit=bool(manifest),
    )


# ----------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------
def _repo_root() -> str:
    """Repo root = three dirname levels up from
    ``src/hippocampus_mcp/library``.

    ⚠️ The three dirnames correspond one to one with this module's depth
    inside the ``src/hippocampus_mcp/library/`` package; if the module moves
    a level, these must change in lockstep, otherwise ``<repo>/data`` gets
    computed as ``src/data`` and one library root is silently missed by the
    scan.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))


def default_roots() -> list[str]:
    """Default scan roots: env ``HIPPOCAMPUS_LIBRARY_ROOTS`` (separated by
    ``os.pathsep``) > ``~/.hippocampus_mcp`` + ``<repo>/data``.

    Both roots must be scanned: the former is the MCP default persist_dir
    (older deployments live there), the latter is where this project lays
    libraries out as directories.
    """
    env = _resolve("HIPPOCAMPUS_LIBRARY_ROOTS", "library_roots", "").strip()
    if env:
        return [x.strip() for x in env.split(os.pathsep) if x.strip()]
    roots = [
        os.path.join(os.path.expanduser("~"), ".hippocampus_mcp"),
        os.path.join(_repo_root(), "data"),
    ]
    return roots


def scan_libraries(roots: list[str] | None = None) -> list[LibraryInfo]:
    """Scan directories to discover libraries. **Read-only**: creates and
    modifies nothing.

    Each root first checks whether it is itself a library (older deployments
    put the library directly under ``~/.hippocampus_mcp``), then scans its
    **direct** subdirectories (no recursion: libraries do not nest).
    """
    found: dict[str, LibraryInfo] = {}
    for root in (roots if roots is not None else default_roots()):
        root = normpath(os.path.expanduser(root))
        if not os.path.isdir(root):
            continue
        candidates: list[str] = []
        if _looks_like_library(root):
            candidates.append(root)
        try:
            for entry in os.listdir(root):
                sub = os.path.join(root, entry)
                if os.path.isdir(sub):
                    candidates.append(sub)
        except OSError:  # noqa: BLE001 -- skip unreadable directories; the whole scan must not fail
            continue
        for cand in candidates:
            if not _looks_like_library(cand):
                continue
            key = normpath(cand)
            if key not in found:
                found[key] = _info_for(key)
    return sorted(found.values(), key=lambda i: (i.backup, i.name))


# ----------------------------------------------------------------------
# Addressing: three-state resolution (id → name/alias → path)
# ----------------------------------------------------------------------
def _match_by_id(query: str, libs: list[LibraryInfo]) -> LibraryInfo | None:
    q = query.strip().lower()
    for info in libs:
        if info.library_id.lower() == q:
            return info
    return None


def _match_by_name(query: str, libs: list[LibraryInfo]) -> tuple[LibraryInfo | None, list[LibraryInfo]]:
    """Exact match by name/alias. **Never guess on ambiguity** -- return the
    candidates so the caller can fail explicitly."""
    q = query.strip()
    ql = q.lower()
    hits = [i for i in libs if not i.backup and (i.name.lower() == ql or ql in [a.lower() for a in i.aliases])]
    if len(hits) == 1:
        return hits[0], hits
    if len(hits) > 1:
        return None, hits
    return None, []


def _match_by_path(query: str, libs: list[LibraryInfo]) -> LibraryInfo | None:
    key = normpath(os.path.expanduser(query))
    for info in libs:
        if info.path == key:
            return info
    # The path exists and looks like a library but is not in the scan result
    # (e.g. no scan root covers it) → synthesize an entry in place
    if _looks_like_library(key):
        return _info_for(key)
    return None


def resolve_library(
    query: str | None,
    roots: list[str] | None = None,
    libs: list[LibraryInfo] | None = None,
    default_path: str | None = None,
) -> LibraryInfo | None:
    """Three-state resolution: ``lib_`` id → name/alias → path.

    Args:
        query: library identifier given by the caller. **Empty returns
            None** (= "not specified"; the caller decides whether to use the
            sticky/default library) -- "not specified" and "specified but
            nonexistent" must be handled separately.
        default_path: default library path. It must stay resolvable even
            when absent from the scan result (older deployments' persist_dir
            often lies under no scan root).

    Raises:
        LibraryResolutionError: specified but unresolvable (including
            duplicate-name ambiguity). **Never silently falls back**.
    """
    if query is None or not str(query).strip():
        return None
    q = str(query).strip()
    pool = list(libs) if libs is not None else scan_libraries(roots)
    if default_path:
        dkey = normpath(default_path)
        if not any(i.path == dkey for i in pool) and _looks_like_library(dkey):
            pool.append(_info_for(dkey))

    if q.lower().startswith("lib_"):
        hit = _match_by_id(q, pool)
        if hit is not None:
            return hit
        raise LibraryResolutionError(
            f"No memory library with id {q}. Use library_list to see available libraries."
        )

    hit, ambiguous = _match_by_name(q, pool)
    if hit is not None:
        return hit

    # When the name only hits **backup** directories, the message must be
    # precise. Saying "not found" is wrong: the library is there, just
    # read-only. Seeing "not found", ops would assume a misconfigured path
    # and go tweak config -- effort wasted.
    ql = q.lower()
    bak_hits = [
        i for i in pool if i.backup
        and (i.name.lower() == ql or ql in [a.lower() for a in i.aliases])
    ]
    if bak_hits:
        raise LibraryResolutionError(
            f"{q} is a backup/snapshot directory and is read-only; it cannot be a write target ({len(bak_hits)} match(es): "
            f"{', '.join(i.path for i in bak_hits[:3])}). Use a real library name or library_id."
        )

    if ambiguous:
        names = ", ".join(f"{i.name}({i.library_id})" for i in ambiguous)
        raise LibraryResolutionError(
            f"Library name {q} is ambiguous; matched {len(ambiguous)} libraries: {names}."
            f"Use a library_id (lib_...) or full path instead."
        )

    hit = _match_by_path(q, pool)
    if hit is not None:
        return hit

    # Reaching here = "specified but unresolvable"; must raise. Offer
    # near-name suggestions (edit distance ≤ 2).
    suggestions = _suggest(q, pool)
    tail = f"Did you mean: {suggestions}?" if suggestions else "Use library_list to see available libraries."
    raise LibraryResolutionError(f"Memory library {q} not found. {tail}")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _suggest(query: str, libs: list[LibraryInfo], max_dist: int = 2) -> str:
    ql = query.strip().lower()
    out = []
    for info in libs:
        if info.backup:
            continue
        cands = [info.name.lower()] + [a.lower() for a in info.aliases]
        if any(_levenshtein(ql, c) <= max_dist for c in cands):
            out.append(f"{info.name}({info.library_id})")
    return "、".join(out[:3])


# ----------------------------------------------------------------------
# Library creation: not called from the scan path, reserved for library_create
# ----------------------------------------------------------------------
def write_manifest(path: str, name: str, aliases: list[str] | None = None,
                   embedder_model: str = "") -> LibraryInfo:
    """Write a new library manifest. **Called only from library_create**;
    the scan path never writes to disk."""
    path = normpath(path)
    os.makedirs(path, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = {
        "library_id": library_id_for(path),
        "name": name,
        "aliases": list(aliases or []),
        "created_at": now,
        "embedder_model": embedder_model,
    }
    with open(os.path.join(path, MANIFEST_NAME), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return _info_for(path)
