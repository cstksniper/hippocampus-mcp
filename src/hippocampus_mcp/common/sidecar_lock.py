"""Short critical-section lock for sidecar files.

**The specific disease this solves**: ``maps.json`` / ``file_registry.json``
used to be 「read → modify in memory → tmp+rename write」. tmp+rename only
guarantees **no half file left behind**; it cannot guarantee **no lost
update**:

    daemon session scan:  load(v1) ──────────────── save(v1+scan result)
    CLI maps.py set:            load(v1) ── save(v1+new entry)
                                            ↑ the later write erases the earlier one's changes entirely

Both are legitimate writes, neither errors out, and one process's work is
wasted. This is nastier than a crash -- a crash leaves traces; this is a
**silent** lost update.

**Why not filelock**: the main library's cross-process protection depends
on ``filelock``, which is an optional dependency (this machine's 3.13 does
not have it; see the degradation note in vector_store). Sidecars are
opt-in auxiliary data; they should not become entirely unavailable because
one third-party package is missing, nor should "missing dependency" become
"silently unlocked". So this module uses **stdlib-self-sufficient atomic
publish + takeover**, the same paradigm as ``launcher.acquire_spawn_lock``
but implemented independently (importing launcher would drag the daemon
startup chain into a CLI that only wants to write the map).

**Three design constraints**:

1. **Non-blocking first, no queueing**. Sidecar writes are millisecond-level
   critical sections; if the lock cannot be taken, say 「busy」 honestly and
   let the caller decide whether to wait or skip. Never block indefinitely --
   the holder may be an already-killed session, and blocking would jam
   everyone (launcher was burned by blocking mutual exclusion in the double
   daemon incident).
2. **Takeover after crashes**. The lock carries ``{pid, ts, token}``: if
   the holder process is gone, or ``ttl`` seconds have passed, judge it
   stale, clean up, and retry once. TTL is the last-resort fallback (pids
   may be reused), not the primary criterion.
3. **Only delete your own lock**. Release verifies the token and does
   nothing on mismatch -- otherwise a slow process waking up would delete
   someone else's lock, back to double holding.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid

#: Maximum sidecar lock hold time (seconds). Measured critical sections: a
#: full maps.json write takes tens of ms; a full file_registry scan is
#: bounded by the dual budget (default 5s) ⇒ 60s leaves ample margin.
#: Exceeding it means the holder is presumed dead (fallback for pid reuse).
SIDECAR_LOCK_TTL = 60.0

#: Grace period for a lock file that "just appeared, contents not yet
#: written". The normal path publishes atomically via os.link, so contents
#: are guaranteed complete; only the degraded path (filesystem without
#: hardlink support) can produce half-written contents, and this grace
#: keeps us from grabbing too early then.
SIDECAR_LOCK_GRACE = 2.0

#: Polling interval (``wait`` mode).
_POLL_S = 0.02

#: Reentrancy sentinel: returned by acquire when the same thread already
#: holds the lock; release does nothing for it.
REENTRANT = object()


class SidecarBusy(RuntimeError):
    """The sidecar lock is held by another process/thread and the caller is unwilling to wait.

    **This is explicit**: the caller must decide 「retry / skip this round /
    report to the user」; treating it as a successful write is not allowed
    (a silent lost update is exactly what this module exists to eliminate).
    """

    def __init__(self, path: str, holder: dict | None = None):
        self.path = path
        self.holder = holder or {}
        who = ""
        if self.holder:
            who = " (pid=%s, held for %.1fs)" % (
                self.holder.get("pid", "?"),
                max(0.0, time.time() - float(self.holder.get("ts") or 0)))
        super().__init__("The sidecar is being written by another process%s: %s" % (who, path))


class SidecarUnreadable(RuntimeError):
    """The sidecar file **exists but cannot be read** (corrupt JSON / exclusively held / permissions).

    🔴 Must be distinguished from "does not exist": ``load`` returns None
    for both, and if the write path takes that as a cue to start an empty
    sidecar (``create=True``) and write the full content back, one
    transient read failure would **wipe the whole map library**. A lost
    update can be recomputed by the next sweep; a wiped library is
    irreversible. So this raises explicitly and lets a human decide whether
    to repair the file or explicitly rebuild.
    """

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        self.reason = reason
        super().__init__(
            "侧车文件存在但读不出来%s：%s（不会当作不存在而覆盖——"
            "确需重建请显式 force）" % (("（%s）" % reason) if reason else "",
                                       path))


_tls = threading.local()


def lock_path(target: str) -> str:
    """The lock file for a sidecar file: ``maps.json`` ⇒ ``.maps.json.lock``.

    Placed in the same directory (same volume) -- ``os.link`` cannot cross
    volumes.
    """
    full = os.path.abspath(target)
    d, base = os.path.split(full)
    return os.path.join(d, "." + base + ".lock")


def _held() -> dict[str, str]:
    paths = getattr(_tls, "paths", None)
    if paths is None:
        _tls.paths = paths = {}
    return paths


def _read_lock(path: str) -> dict | None:
    """Read the lock contents. Unreadable / malformed ⇒ None (**does not mean nobody holds it**, see the staleness judgment)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(info, dict) and isinstance(info.get("ts"), (int, float)):
        return info
    return None


def pid_alive(pid: int) -> bool:
    """Read-only liveness probe, same semantics as ``launcher._pid_alive``.

    🔴 On Windows you **must not** use ``os.kill(pid, 0)`` -- there os.kill
    routes any non-CTRL_* signal to TerminateProcess, turning a liveness
    probe into a kill. So we go through
    OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION).
    """
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)
            if h:
                k32.CloseHandle(h)
                return True
            return False
        except Exception:  # noqa: BLE001 -- if we cannot probe, assume alive; better to wait for TTL
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_stale(path: str, ttl: float = SIDECAR_LOCK_TTL,
             grace: float = SIDECAR_LOCK_GRACE) -> bool:
    """Whether the lock can be taken over.

    🔴 **Unreadable contents ≠ nobody holds it**: in the window where the
    other side just created the file but has not finished writing the JSON,
    judging "takeover possible" would double-hold the lock. File exists but
    contents incomplete ⇒ give a grace period based on mtime.
    """
    info = _read_lock(path)
    if info:
        pid = info.get("pid")
        # 🔴 pid == ourselves ⇒ **never judge stale**, ahead of the TTL
        # check: it can only be another thread of the same process holding
        # the lock (a crashed process's pid cannot equal the current
        # process's). If stale were judged here on timeout while the other
        # thread is still writing, this side would delete the lock ⇒
        # double holding. The cost -- waiting out a stuck thread in the
        # same process -- is acceptable.
        if isinstance(pid, int) and pid == os.getpid():
            return False
        ts = float(info.get("ts") or 0.0)
        if ts and time.time() - ts > ttl:
            return True
        if isinstance(pid, int) and pid > 0:
            return not pid_alive(pid)
        return False
    try:
        age = time.time() - os.stat(path).st_mtime
    except OSError:
        return True                       # File doesn't exist ⇒ genuinely nobody holds it
    return age > grace


def acquire(path: str, wait: float = 0.0,
            ttl: float = SIDECAR_LOCK_TTL,
            grace: float = SIDECAR_LOCK_GRACE):
    """Try to take the lock. On success returns the token (or
    :data:`REENTRANT`); returns ``None`` when the lock cannot be taken.

    With ``wait`` > 0, waits at most that many seconds (short polling);
    default is no waiting.
    """
    key = os.path.abspath(path)
    held = _held()
    if key in held:
        return REENTRANT                  # Same-thread reentry: the lock is already ours

    try:
        os.makedirs(os.path.dirname(key) or ".", exist_ok=True)
    except OSError:
        return None

    token = "%d:%s" % (os.getpid(), uuid.uuid4().hex[:12])
    deadline = time.time() + max(0.0, float(wait))
    while True:
        # Staleness judgment + cleanup (remove only after confirming stale; never rm unconditionally)
        if is_stale(key, ttl=ttl, grace=grace):
            try:
                os.remove(key)
            except OSError:
                pass
        if _publish(key, token):
            held[key] = token
            return token
        if time.time() >= deadline:
            return None
        time.sleep(_POLL_S)


def _publish(lock: str, token: str) -> bool:
    """Atomically publish the lock file.

    🔴 The lock file must be "complete the moment it appears": write a
    temporary file + fsync first, then attach it to the lock path with
    ``os.link`` -- hardlink creation is atomic and necessarily fails when
    the target already exists, so the mutual-exclusion semantics are fully
    preserved and the half-written window disappears entirely.
    """
    tmp = "%s.tmp.%d.%s" % (lock, os.getpid(), token.split(":")[-1])
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time(),
                       "token": token}, f)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        _unlink(tmp)
        return False
    try:
        os.link(tmp, lock)
        return True
    except FileExistsError:
        return False                      # Normal contention, lost to the other side
    except OSError:
        return _publish_legacy(lock, token)   # Hardlinks unsupported ⇒ degrade
    finally:
        _unlink(tmp)


def _publish_legacy(lock: str, token: str) -> bool:
    """Degraded path when ``os.link`` is unavailable (e.g. some network filesystems).

    Still ``O_CREAT|O_EXCL``, so **the half-written window remains** --
    covered by :func:`is_stale`'s ``grace``: the other side gets grace
    seconds to finish writing the contents.
    """
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time(),
                       "token": token}, f)
            f.flush()
            os.fsync(f.fileno())
        return True
    except OSError:
        _unlink(lock)
        return False


def release(path: str, token) -> bool:
    """Release the lock. **Only deletes its own**: does nothing when the token in the contents does not match.

    Returns whether it was actually removed (``REENTRANT`` returns False,
    meaning an outer layer still holds it).
    """
    if token is REENTRANT or token is None:
        return False
    key = os.path.abspath(path)
    _held().pop(key, None)
    info = _read_lock(key)
    if not info or info.get("token") != token:
        return False                      # Already taken over / not ours
    return _unlink(key)


def _unlink(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def holder(path: str) -> dict | None:
    """Info about the current lock holder (for diagnostics; not used in judgments)."""
    return _read_lock(path)


def quarantine(path: str) -> str:
    """Rename an unreadable sidecar aside for quarantine (``.corrupt.<ts>``), returning the new path.

    **Never deletes outright**: a corrupt file may still hold hundreds of
    manually recorded states; delete them and they are gone for good. After
    the rename the write path can safely rebuild, and a human can still
    rescue data from it by hand.
    """
    dst = "%s.corrupt.%d" % (path, int(time.time()))
    try:
        os.replace(path, dst)
        return dst
    except OSError:
        return ""
