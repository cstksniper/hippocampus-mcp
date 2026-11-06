"""Project file registry: staleness detection between hippo and the project disk.

Requirement: "register all project files, recording when each file was
last synced to hippo; if a file was updated after that, we know its
information is stale". This module implements it as **signature
comparison** rather than wall-clock comparison (rationale in the
common/constants constant-block comment: when git checkout / robocopy
mass-reset mtime without changing content, wall-clock comparison would
misjudge the whole repository as stale).

Sidecar JSON (``<persist_dir>/file_registry.json``)::

    {
      "version": 1,
      "root": "/absolute/path/to/project",  # given explicitly at init, never guessed
      "ignores": [...],                   # snapshot of the effective ignore rules
      "last_checked": 1710000000.0,
      "scan_complete": true,
      "files": {
        "src/main.py": {
          "size": 1234, "mtime_ns": 1710000000000000000,
          "hash": "b3a1...",              # blake2b-128, see _hash_file
          "first_seen": ..., "last_synced": ..., "last_checked": ...,
          "status": "synced"              # synced | stale | missing
        }
      }
    }

Hard disciplines:
1. **opt-in**: without the sidecar there is no behavior at all (no scan,
   no report, no card line) -- enabling means an explicit ``init``,
   ruling out "guess a project root and scan the wrong directory".
2. **hash arbitration**: the stale verdict is not "mtime changed counts
   as changed". git checkout / robocopy / touch reset mtime without
   changing content; when the (mtime, size) signature mismatches, **read
   the content hash once** to confirm: same hash => just a touch, update
   the signature and stay synced. Conversely, mtime-preserving tampering
   (``touch -r`` backdating) is out of scope -- same nature as C1a
   source_sig; the hot path does not pay full reads for adversarial
   scenarios.
3. **budget fallback**: scanning has dual caps on file count + wall
   clock; over the limit mark ``scan_complete=false`` and self-report
   "scan incomplete" on the status card -- never swallow silently
   (consistent with the project-wide "self-report" discipline).
4. **failures do not blow up the hot path**: all disk-operation
   exceptions are handled by the caller; the only things this module
   raises itself are "usage errors" (e.g. duplicate init, sync of a
   nonexistent path).
5. **read-modify-write under lock**: the daemon's periodic scan and CLI
   ``sync`` / ``init`` will collide; bare tmp+rename is **silent loss of
   updates**. Always use :func:`update`; failing to get the lock raises
   :class:`SidecarBusy`, never a silent write.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tempfile
import time

from ..common import sidecar_lock as sl
from ..common.constants import SOURCE_SIG_FILE
from .constants import (
    ENV_FILE_REGISTRY_BUDGET_S,
    ENV_FILE_REGISTRY_IGNORE,
    ENV_FILE_REGISTRY_MAX_FILES,
    FILE_REGISTRY_BUDGET_S,
    FILE_REGISTRY_DEFAULT_IGNORES,
    FILE_REGISTRY_MAX_FILES,
)
from ..common.sidecar_lock import SidecarBusy, SidecarUnreadable

# Construction injection seam: the knobs are registered in config; the
# production path injects them at engine import time; the resolution order
# (env first, injected value as fallback) is in common/config_bridge.
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

REGISTRY_VERSION = 1
IGNORE_FILE_NAME = ".hippoignore"
HASH_MAX_BYTES = 64 * 1024 * 1024   # larger than this, no hashing (data files):
                                    # a stat signature mismatch is conservatively
                                    # judged stale -- better to over-report


# ── Read/write ──────────────────────────────────────────────────

def registry_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "file_registry.json")


def load(persist_dir: str) -> dict | None:
    """Read the sidecar; returns None for missing/corrupt (corrupt is treated as not enabled; never blows up the caller)."""
    p = registry_path(persist_dir)
    try:
        with open(p, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(reg, dict) or "files" not in reg:
        return None
    return reg


def _write_atomic(persist_dir: str, reg: dict) -> None:
    """Atomic write (tmp + rename): a scan can be interrupted by a
    concurrent session; a half-written JSON must not remain.

    **No locking here** -- called only by :func:`update` (already holding
    the lock) or :func:`save` (takes the lock itself).
    """
    p = registry_path(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".file_registry_", dir=persist_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save(persist_dir: str, reg: dict, wait: float = 0.0) -> None:
    """Persist to disk (atomic write + sidecar lock). **Does not prevent lost updates**; rationale same as map/store.save."""
    lock = sl.lock_path(registry_path(persist_dir))
    tok = sl.acquire(lock, wait=wait)
    if tok is None:
        raise SidecarBusy(lock, sl.holder(lock))
    try:
        _write_atomic(persist_dir, reg)
    finally:
        sl.release(lock, tok)


def update(persist_dir: str, fn, create: bool = False, wait: float = 0.0,
           reset_unreadable: bool = False):
    """「Read → modify → write」 under lock; ``fn(reg) -> (save, result)``.

    Same contract as :func:`map/store.update` (including ``create`` / no
    fn call without a sidecar / 「unreadable ≠ nonexistent」 / failing to
    get the lock raises :class:`SidecarBusy`). **This is the only safe
    write posture for the registry sidecar**: when the periodic scan
    (possibly seconds long) collides with CLI ``sync``, the later writer
    does not erase the earlier one.
    """
    lock = sl.lock_path(registry_path(persist_dir))
    tok = sl.acquire(lock, wait=wait)
    if tok is None:
        raise SidecarBusy(lock, sl.holder(lock))
    try:
        reg = load(persist_dir)
        if reg is None:
            p = registry_path(persist_dir)
            exists = os.path.isfile(p)
            if exists and not reset_unreadable:
                raise sl.SidecarUnreadable(p)
            if exists:
                sl.quarantine(p)
            if not create:
                return None, False
            reg = {"version": REGISTRY_VERSION, "root": "",
                   "ignores": [], "last_checked": 0.0,
                   "scan_complete": False, "files": {}}
        save_flag, result = fn(reg)
        if save_flag:
            _write_atomic(persist_dir, reg)
        return result, bool(save_flag)
    finally:
        sl.release(lock, tok)


def init(persist_dir: str, root: str, force: bool = False) -> dict:
    """Registry init: the project root is given explicitly (never guessed --
    a wrong guess means scanning the wrong directory).

    Refuses when one already exists (unless force): a repeated init would
    lose the entire registry history; such an operation must be explicit.

    「Check existence → create」 under lock: when two sessions init
    concurrently, the later one must see the earlier one's result and take
    the "already registered" branch, instead of each writing its own copy
    and erasing the other's history.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError("the project root does not exist: %s" % root)

    def _do(reg):
        if (reg.get("root") or reg.get("files")) and not force:
            raise FileExistsError(
                "already registered (root=%s, %d files); re-init would lose history -- pass force to confirm"
                % (reg.get("root"), len(reg.get("files") or {})))
        reg.clear()
        reg.update({
            "version": REGISTRY_VERSION,
            "root": root,
            "ignores": effective_ignores(root),
            "last_checked": 0.0,
            "scan_complete": False,
            "files": {},
        })
        return True, reg

    # force also means "allow quarantining an unreadable old sidecar and
    # rebuilding" -- rebuilding is an explicitly destructive operation
    # anyway; unify the two refusals ("read a corrupt file" and "already
    # registered") onto force.
    reg, _saved = update(persist_dir, _do, create=True,
                         reset_unreadable=force)
    return reg


# ── Ignore rules ────────────────────────────────────────────────

def effective_ignores(root: str) -> list[str]:
    """Default ignores + .hippoignore under the root + extra env globs
    (snapshotted into the sidecar).

    .hippoignore uses a subset of gitignore syntax: ``#`` comments, blank
    lines ignored, ``*`` globs (fnmatch), a trailing ``/`` matches
    directories only, everything else matches by path segment (any parent
    directory or the full relative path hitting means ignored).
    """
    patterns = list(FILE_REGISTRY_DEFAULT_IGNORES)
    p = os.path.join(root, IGNORE_FILE_NAME)
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    except OSError:
        pass
    extra = _resolve(ENV_FILE_REGISTRY_IGNORE, "file_registry_ignore", "")
    for pat in extra.split(","):
        pat = pat.strip()
        if pat:
            patterns.append(pat)
    return patterns


def _match_pattern(rel: str, is_dir: bool, pat: str) -> bool:
    """Single-pattern match (subset syntax; see the effective_ignores comment)."""
    if pat.endswith("/"):
        if not is_dir:
            return False
        pat = pat[:-1]
    rel_n = rel.replace(os.sep, "/")
    parts = rel_n.split("/")
    # directory name / path segment: any parent directory or its own basename hitting
    if not any(ch in pat for ch in "*?["):
        if pat in parts:
            return True
    # glob: full relative path or basename
    if fnmatch.fnmatch(rel_n, pat):
        return True
    if fnmatch.fnmatch(parts[-1], pat):
        return True
    return False


def is_ignored(rel: str, is_dir: bool, patterns: list[str]) -> bool:
    """**Reference implementation** (per-pattern fnmatch). The semantic
    baseline, for tests/CLI cross-checking; the scan hot path uses
    :class:`_IgnoreIndex` (the same semantics, compiled; see its docstring).
    """
    return any(_match_pattern(rel, is_dir, p) for p in patterns)


class _IgnoreIndex:
    """Compiled form of the ignore rules (compiled once per scan; moving
    26 patterns × 20k files of repeated matching out of the loop).

    The motivation is measured (NTFS, ~20k files):
    **hot scan 21.6s, of which 81% is is_ignored** -- per file, each of 26
    patterns gets a ``str.replace/split`` + two ``fnmatch`` calls, and
    fnmatch on Windows carries ``normcase`` (Win32 LCMapStringEx) every
    time. Pure-Python pattern matching is CPU-bound by nature; no language
    change needed: precompiling solves it.

    Semantics are **character-for-character aligned** with
    :func:`is_ignored` (probe cross-checked against all 362 paths of the
    repo):
    - patterns without glob characters → path-segment set (any parent
      directory or the basename hitting counts),
      plus whole-path equality (fnmatch degenerates to equality for
      glob-free patterns);
    - patterns with globs → precompiled regex, matched against the full
      relative path or the basename;
    - trailing ``/`` matches directories only;
    - case folded on Windows (normcase; historical behavior unchanged).
    """

    __slots__ = ("_plain", "_dir_plain", "_globs")

    def __init__(self, patterns: list[str]):
        plain: set[str] = set()
        dir_plain: set[str] = set()
        globs: list[tuple[re.Pattern, bool]] = []
        for raw in patterns:
            pat = raw
            dir_only = pat.endswith("/")
            if dir_only:
                pat = pat[:-1]
            if not pat:
                continue
            npat = os.path.normcase(pat)
            if any(ch in pat for ch in "*?["):
                globs.append((re.compile(fnmatch.translate(npat)), dir_only))
            else:
                (dir_plain if dir_only else plain).add(npat)
        self._plain = plain
        self._dir_plain = dir_plain
        self._globs = globs

    def is_ignored(self, rel: str, is_dir: bool) -> bool:
        # Order matters: split by "/" first, then normcase. On Windows
        # normcase turns "/" into "\\" and lowercases -- normcase first and
        # the parent-directory segments can never be split again (measured
        # lesson of 40k false-negative paths: must be the same order as the
        # reference implementation's replace(sep,"/")). The reference
        # implementation's net semantics: **any** path segment equal after
        # normcase hits (a parent hit ignores the whole branch), or the
        # whole path equal after normcase; globs go through the full path
        # or the basename.
        rel_f = rel.replace(os.sep, "/") if os.sep != "/" else rel
        segs = rel_f.split("/")
        rel_n = os.path.normcase(rel_f)
        for s in segs:
            ns = os.path.normcase(s)
            if ns in self._plain:
                return True
            if is_dir and ns in self._dir_plain:
                return True
        if rel_n in self._plain:
            return True
        if is_dir and rel_n in self._dir_plain:
            return True
        base = os.path.normcase(segs[-1])
        for rx, dir_only in self._globs:
            if dir_only and not is_dir:
                continue
            if rx.match(rel_n) or rx.match(base):
                return True
        return False


def scan_limits() -> tuple[int, float]:
    """Effective values of the dual budget (env-overridable; shared by
    brain's automatic patrol and the CLI).

    Defaults: 20k files / 1s. A quant project far above this scale (100k+
    files) needs no code change:
    ``HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES`` /
    ``HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S`` override directly; the
    over-limit degradation is unchanged (scan_complete=false self-report,
    no missing verdicts).
    """
    def _int_env(name: str, default: int, lo: int, hi: int) -> int:
        try:
            raw = _resolve(name, "file_registry_max_files", str(default))
            v = int(float(raw or default))
        except ValueError:
            v = default
        return max(lo, min(hi, v))

    def _float_env(name: str, default: float, lo: float, hi: float) -> float:
        try:
            raw = _resolve(name, "file_registry_budget_s", str(default))
            v = float(raw or default)
        except ValueError:
            v = default
        return max(lo, min(hi, v))

    return (_int_env(ENV_FILE_REGISTRY_MAX_FILES, FILE_REGISTRY_MAX_FILES, 1, 10 ** 7),
            _float_env(ENV_FILE_REGISTRY_BUDGET_S, FILE_REGISTRY_BUDGET_S, 0.05, 3600.0))


# ── Scanning ────────────────────────────────────────────────────

def _sig(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def _hash_file(path: str) -> str | None:
    """blake2b-128 content hash; returns None if too large/unreadable (the
    caller judges stale conservatively).

    Called only when the stat signature mismatches (arbitration for touch /
    git checkout); files whose signature is unchanged are never read --
    the rsync --checksum idea of 「quick check + hash arbitration」;
    routine scans read zero content.
    """
    try:
        if os.path.getsize(path) > HASH_MAX_BYTES:
            return None
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def scan(reg: dict, max_files: int, budget_s: float) -> dict:
    """Re-scan the project root under a budget, updating each file's
    status (mutates reg in place and returns it).

    status semantics:
      synced  -- hippo's information reflects the current disk content
      stale   -- content changed and hippo has not synced (or the file is
                 too large to hash / signature mismatch, judged stale
                 conservatively); a new file that has never been synced
                 enters as synced right away ("seen" is not "stale";
                 only modified counts)
      missing -- registered but not on disk (deleted/moved ⇒ equally
                 worth knowing)

    Verdict in two paths (hash arbitration; see module docstring
    discipline 2):
      stat signature (size, mtime_ns) unchanged ⇒ synced, zero reads;
      signature mismatch ⇒ read content hash: same ⇒ pure touch, update
      the signature, still synced;
      different (or too large to hash) ⇒ stale.

    Performance (measured on NTFS, ~20k files):
    the old code hot-scanned 20k files in 21.6s -- 81% spent in
    is_ignored's per-pattern fnmatch (each carrying normcase/LCMapStringEx
    on Windows), the rest one os.stat per file (≈0.29ms, the NTFS
    filter-driver price). Now: ignore rules compiled into
    :class:`_IgnoreIndex` (compile once, set + regex matching) + the
    scandir recursion reuses the enumeration cache via
    ``DirEntry.stat()`` (on Windows FindFirstFile already brought back
    size/mtime, zero extra syscalls; measured 5.59s → 0.16s).
    """
    root = reg.get("root") or ""
    if not root or not os.path.isdir(root):
        reg["scan_complete"] = False
        return reg
    # Re-read ignore rules each time: changes to .hippoignore / env must take effect (the init-time snapshot goes stale)
    patterns = effective_ignores(root)
    reg["ignores"] = patterns
    idx = _IgnoreIndex(patterns)
    now = time.time()
    files: dict = reg.setdefault("files", {})
    seen: set[str] = set()
    scanned = 0
    complete = True
    deadline = now + budget_s

    def _walk(dirpath: str, rel_dir: str) -> bool:
        """Recursively scan one level (scandir: gives DirEntry; stat
        reuses the enumeration cache). Returns False = dual budget
        exhausted, stop recursing (equivalent to breaking out of os.walk
        early)."""
        nonlocal scanned
        try:
            it = os.scandir(dirpath)
        except OSError:
            return True  # unreadable directory (permissions/vanishing): skip this branch, does not count as budget exhaustion
        with it:
            for entry in it:
                # Per-file dual-budget check (not only after big
                # directories): when the wall clock is over, at most one
                # extra file is scanned; a whole directory is not eaten
                if scanned >= max_files or time.time() > deadline:
                    return False
                name = entry.name
                rel = "%s/%s" % (rel_dir, name) if rel_dir else name
                try:
                    is_link = entry.is_symlink()
                    if entry.is_dir(follow_symlinks=False):
                        # Pruning: ignored directories are skipped whole (.git has hundreds of thousands of objects).
                        if idx.is_ignored(rel, True):
                            continue
                        if not _walk(entry.path, rel):
                            return False
                        continue
                    if is_link and entry.is_dir():
                        # symlink→directory: same semantics as
                        # os.walk(followlinks=False) -- no recursion (cycle
                        # guard) and no registration; character-identical
                        # to the old behavior.
                        continue
                except OSError:
                    continue
                if idx.is_ignored(rel, False):
                    continue
                scanned += 1
                seen.add(rel)
                try:
                    st = entry.stat()  # follows symlinks by default = os.stat semantics
                except OSError:
                    st = None
                ent = files.get(rel)
                if st is None:
                    if ent is not None:
                        ent["status"] = "missing"
                        ent["last_checked"] = now
                    continue
                size, mtime_ns = st.st_size, st.st_mtime_ns
                if ent is None:
                    files[rel] = {
                        "size": size, "mtime_ns": mtime_ns,
                        "hash": _hash_file(entry.path),
                        "first_seen": now, "last_synced": now,
                        "last_checked": now, "status": "synced",
                    }
                    continue
                ent["last_checked"] = now
                if (size, mtime_ns) == (ent.get("size"), ent.get("mtime_ns")):
                    # Signature unchanged ⇒ content did not change again.
                    # But stale is the 「hippo has not synced」 state
                    # machine, **clearable only by mark_synced**
                    # (re-ingested); a scan must not self-heal it --
                    # otherwise after one fix, the next scan quietly turns
                    # green, the 「stale」 signal lives at most one scan
                    # interval, i.e. it is nothing.
                    if ent.get("status") == "missing":
                        ent["status"] = "synced"  # came back with the signature unchanged (deleted then restored)
                    continue
                # Signature mismatch: hash arbitration (touch / git checkout do not false-positive)
                digest = _hash_file(entry.path)
                ent["size"], ent["mtime_ns"] = size, mtime_ns
                if digest is not None and digest == ent.get("hash"):
                    ent["status"] = "synced"   # pure touch: refresh the signature, not stale
                else:
                    if digest is not None:
                        ent["hash"] = digest
                    ent["status"] = "stale"
        return True

    if not _walk(root, ""):
        complete = False
    # Registered but not seen in this scan (regions cut off by a break must
    # not be judged missing -- that is exactly the false positive
    # scan_complete=false guards against)
    if complete:
        for rel, ent in files.items():
            if rel not in seen and ent.get("status") != "missing":
                ent["status"] = "missing"
                ent["last_checked"] = now
    reg["last_checked"] = now
    reg["scan_complete"] = complete
    reg["scanned"] = scanned
    return reg


def load_source_sigs(persist_dir: str) -> dict | None:
    """Read the C1a source-signature layer (brain's ``source_sigs``,
    backfilled at ingest).

    Missing / unreadable ⇒ None. This is **missing evidence**, which is
    **not the same thing** as 「the layer exists but this file was never
    recorded」 (= fact: never ingested) -- the caller must handle the two
    separately (see :func:`mark_synced`'s ``source_sigs`` parameter).

    With the sqlite backend this file does not exist (the layer lives in
    the library) ⇒ also returns None; the CLI self-reports 「无法校验」
    (cannot verify) instead of silently letting things through on trust.

    Known boundary: what is read is the **on-disk** layer; an ingest the
    daemon has not yet persisted is invisible ⇒ verification can be one
    beat older than brain's in-memory layer. The direction is
    conservative (older ⇒ more likely to reject, never lets through what
    should not pass), and CLI and daemon are two processes anyway; a
    consistent snapshot would need an extra cross-process protocol, not
    worth it.
    """
    p = os.path.join(persist_dir, SOURCE_SIG_FILE)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def source_sig_index(sigs: dict | None) -> dict[str, dict]:
    """Source-signature layer → **absolute path** index; for the same path
    with multiple entries, keep the one with the latest ``sig_at``.

    Consistent with the layer-merge rule
    (``_LAYER_MERGE_RULES["sourcesig"]``: same key, take the newer)
    -- when both processes ingested the same file, the later-verified one
    is the disk's current state.

    Pure function: brain passes the in-memory layer, the CLI passes the
    on-disk read; they share one path normalization, avoiding writing it
    twice and drifting apart.
    """
    out: dict[str, dict] = {}
    for e in (sigs or {}).values():
        if not isinstance(e, dict):
            continue
        path = e.get("path")
        if not path:
            continue
        key = os.path.normcase(os.path.abspath(str(path)))
        prev = out.get(key)
        if prev is None or float(e.get("sig_at") or 0.0) >= float(
                prev.get("sig_at") or 0.0):
            out[key] = e
    return out


def _sha1_file(path: str) -> str | None:
    """sha1 content hash (same algorithm as brain's ``_file_sig``, so the
    values can match).

    Only a fallback read when (mtime_ns, size) mismatches, and sync is an
    explicit command with a bounded number of paths -- the hot scan path
    does not touch it (that one is blake2b; see :func:`_hash_file`).
    """
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sync_verify(full: str, disk: tuple[int, int],
                 idx: dict[str, dict]) -> str | None:
    """Criterion for 「hippo really ingested the current content」: None =
    pass, otherwise a rejection reason code.

    * ``not_ingested``: the signature layer has no such file ⇒ cannot
      prove it, reject
    * ``sig_stale``   : recorded, but the recorded version is not the
      current disk version (changed after ingest)

    Two pass conditions: (mtime_ns, size) both equal ⇒ the file did not
    change after that ingest; mismatched but **sha1 equal** ⇒ touch / git
    checkout, the content is still the ingested version (same spirit as
    :func:`scan`'s hash arbitration: a signature mismatch does not
    directly mean the content changed).
    """
    sig = idx.get(os.path.normcase(os.path.abspath(full)))
    if not sig:
        return "not_ingested"
    size, mtime_ns = disk
    try:
        if (int(sig.get("mtime_ns") or 0) == int(mtime_ns)
                and int(sig.get("size") or -1) == int(size)):
            return None
    except (TypeError, ValueError):
        pass
    want = str(sig.get("sha1") or "")
    if want:
        got = _sha1_file(full)
        if got and got == want:
            return None
    return "sig_stale"


def mark_synced(reg: dict, rel_paths: list[str],
                source_sigs: dict | None = None
                ) -> tuple[list[str], list[str], list[dict]]:
    """Mark the given paths as 「hippo reflects the current disk content」
    (update registry signature + hash).

    Returns ``(ok, not_found, rejected)``:

    * ``ok``        marked (registered paths relative to the project root)
    * ``not_found`` not registered, or the file is gone from disk (also
                    marked missing)
    * ``rejected``  **verification failed; not one was marked**
                    (``{"path","reason"}``; reason codes in
                    :func:`_sync_verify`)

    **Trust mode vs this function's default**: without ``source_sigs`` ⇒
    old behavior (synced because you say so). With it ⇒ each path is
    verified against the source-signature layer for 「hippo really
    ingested the current content」; files never ingested **cannot be
    marked synced** -- otherwise the stale signal of 「edit the file, one
    sync line, and it turns green」 is hollowed out (added after design
    review).

    Verification is **evidence-based judgment, not content transfer**:
    passing only means the signature layer recognizes this version of the
    content; how the content enters memory remains the host/agent's
    action (no automatic re-import).
    """
    now = time.time()
    ok: list[str] = []
    miss: list[str] = []
    rejected: list[dict] = []
    files = reg.get("files") or {}
    idx = source_sig_index(source_sigs) if source_sigs is not None else None
    for raw in rel_paths:
        rel = str(raw).replace(os.sep, "/").lstrip("./")
        ent = files.get(rel)
        if ent is None:
            # Tolerate callers passing absolute paths or various spellings of registered paths
            cand = [k for k in files if k.endswith("/" + rel) or rel.endswith("/" + k)]
            if not cand:
                miss.append(rel)
                continue
            rel = cand[0]
            ent = files[rel]
        full = os.path.join(reg["root"], rel)
        sig = _sig(full)
        if sig is None:
            ent["status"] = "missing"
            ent["last_checked"] = now
            miss.append(rel)
            continue
        if idx is not None:
            why = _sync_verify(full, sig, idx)
            if why:
                rejected.append({"path": rel, "reason": why})
                continue
        size, mtime_ns = sig
        ent["size"], ent["mtime_ns"] = size, mtime_ns
        ent["hash"] = _hash_file(full)
        ent["last_synced"] = now
        ent["last_checked"] = now
        ent["status"] = "synced"
        ok.append(rel)
    return ok, miss, rejected


# ── Summary ─────────────────────────────────────────────────────

def _stale_days(ent: dict, now: float) -> int:
    """How stale: counted from 「the change happening」 (disk mtime) or
    「the last sync」, whichever is earlier.

    Uses max(last_synced, mtime): a file synced 30 days ago and modified
    2 days ago ⇒ stale for 2 days, not 30 -- 「how long since the last
    sync」 overestimates; 「how long ago it changed」 is what the agent
    wants to know. mtime in the future (clock drift) clamps to 0.
    """
    ref = max(float(ent.get("last_synced") or 0.0),
              float(ent.get("mtime_ns") or 0) / 1e9)
    return max(0, int((now - ref) // 86400))


def summarize(reg: dict, now: float | None = None) -> dict:
    """Compact summary for the status card / session-end checklist
    (O(number of registered files), pure memory)."""
    now = time.time() if now is None else now
    files = reg.get("files") or {}
    stale, missing, oldest = [], [], 0
    for rel, ent in files.items():
        status = ent.get("status")
        if status == "stale":
            stale.append(rel)
            oldest = max(oldest, _stale_days(ent, now))
        elif status == "missing":
            missing.append(rel)
    return {
        "root": reg.get("root"),
        "total": len(files),
        "stale": len(stale),
        "missing": len(missing),
        "oldest_stale_days": oldest if stale else 0,
        "stale_paths": sorted(stale)[:20],
        "missing_paths": sorted(missing)[:20],
        "scan_complete": bool(reg.get("scan_complete")),
        "last_checked": float(reg.get("last_checked") or 0.0),
    }
