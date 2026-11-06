# -*- coding: utf-8 -*-
"""launcher: an extremely thin launcher (**pure standard library**, imports no torch / model).

Why it exists: today every MCP subprocess loads its own copy of the embedding
model, a **fixed 755 MB overhead per process** (independent of memory item
count). N sessions = N copies. The launcher does only three things, then turns
into a pipe:

    1. Compute the daemon key for "this library + this model"
    2. Find the daemon: connect if alive; spawn one if absent
    3. Turn itself into a **bidirectional byte pump** between stdio and the daemon socket

Design constraints (fixed before any change; do not let them drift):

- **Pure standard library**. The launcher itself must not import heavy libraries,
  or it would eat the memory it just saved.
- **Disabled by default**. Only ``HIPPOCAMPUS_LAUNCHER=1`` takes the daemon path;
  with 0 / unset it falls back to "run the server directly", **bit-for-bit
  identical** to today's behavior.
- **If the daemon cannot be started, degrade silently** -- never leave the host
  unable to connect to MCP. Availability beats memory savings.
- **Multiple sessions share one daemon** (whenever the key matches). An early
  comment here said "one daemon per connection, no reuse" -- that was wrong:
  ``ensure_daemon`` reuses on a registry hit, and reuse is the entire point of
  "N models -> 1". Session isolation comes from **one thread per connection +
  an independent ``asyncio.run``** inside the daemon (``session_ctx``'s
  ``conn-N``), not from "one process per session".

  🔴 Reuse introduced cross-library contamination (fixed): the daemon's default
  library used to be a **process-level** constant, so a later workspace would
  write into the earlier workspace's library. Now, upon connecting, the launcher
  first sends an out-of-band handshake line (``session_ctx.HELLO_PREFIX``)
  declaring its library, and the daemon remembers it per connection.
  Before changing this path, read ``tests/test_f9_4e_conn_library.py``.
"""

# Exemption registry: this file's direct env reads are startup-topology plumbing
# (launcher switch / readiness timeout / announce passthrough); the closed list
# is in config.py's "root-level exemption list".
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any

try:  # pure stdlib, zero cost; but the launcher must never fail to start over an import error
    from hippocampus_mcp.common.session_ctx import HELLO_PREFIX
except Exception:  # noqa: BLE001 -- fall back to the literal if unavailable (same value as session_ctx)
    HELLO_PREFIX = "HIPPO-HELLO "

# Launcher switch. Off by default -- memory savings are a benefit, being unable
# to connect is an incident; protect the latter first.
ENV_LAUNCHER = "HIPPOCAMPUS_LAUNCHER"
# daemon registry directory (default: hippocampus_mcp under the system temp dir)
ENV_RUNTIME_DIR = "HIPPOCAMPUS_RUNTIME_DIR"
# Memory library directory. The launcher does not interpret its semantics, but
# **writability** must be probed before startup: a library that cannot be
# written means every memory write is silently lost -- not "degraded
# availability" but "apparent availability".
ENV_PERSIST = "HIPPOCAMPUS_PERSIST_DIR"
# Upper limit (seconds) for waiting until the daemon is ready. Model loading
# takes minutes; be generous but bounded.
DAEMON_READY_TIMEOUT = float(os.environ.get("HIPPOCAMPUS_DAEMON_READY_TIMEOUT", "180"))
# Chunk size for a single read of the byte pump
RELAY_CHUNK = 65536


def runtime_dir() -> str:
    """Where the daemon registry lives."""
    d = os.environ.get(ENV_RUNTIME_DIR, "").strip()
    if not d:
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "hippocampus_mcp")
    return d


def launcher_enabled() -> bool:
    return os.environ.get(ENV_LAUNCHER, "0").lower() in ("1", "true", "yes", "on")


def daemon_key(library: str = "", model: str = "", version: str = "") -> str:
    """Daemon identity = library path + model + code version. Any one differing
    means a different daemon.

    The model must be part of the key: the vectors stored in the library were
    computed by one specific model; change the model and you must change the
    process, otherwise the weights in the daemon and the vectors on disk are not
    the same thing and recall silently degrades.
    """
    try:
        lib = os.path.normcase(os.path.abspath(library)) if library else ""
    except Exception:  # noqa: BLE001 -- a path error must not keep the launcher from starting
        lib = str(library or "")
    raw = "|".join([lib, model or "", version or ""])
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


def registry_path(key: str) -> str:
    return os.path.join(runtime_dir(), f"daemon-{key}.json")


def read_registry(key: str) -> dict[str, Any] | None:
    """Read the daemon registry entry. Unreadable / not a dict / no port -> None."""
    try:
        with open(registry_path(key), "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:  # noqa: BLE001 -- unreadable means "no daemon"
        return None
    if not isinstance(info, dict):
        return None
    if not isinstance(info.get("port"), int) or not isinstance(info.get("pid"), int):
        return None
    return info


def write_registry(key: str, info: dict[str, Any]) -> bool:
    d = runtime_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return False
    path = registry_path(key)
    # Include the pid in the tmp name: when two daemons announce concurrently
    # (e.g. the host kills one and a second immediately takes its place, or one
    # enters after the singleflight lock is taken over as stale), sharing one
    # `.tmp` makes one of them hit OSError at the open/replace stage => announce
    # returns False => the daemon exits outright without printing a banner --
    # dying more stealthily than "spawning an extra one" (verified by negative
    # experiment).
    tmp = path + f".{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def stop_request_path(key: str) -> str:
    """Path of the cooperative stop marker for one daemon."""
    return os.path.join(runtime_dir(), f"daemon-{key}.stop")


#: Environment keys that change daemon behaviour.  A mismatch means the
#: running process was started under a different configuration.
_CONFIG_ENV_KEYS = (
    "HIPPOCAMPUS_DAEMON_VERSION",
    "HIPPOCAMPUS_EMB_MODEL",
    "HIPPOCAMPUS_TOOL_PROFILE",
    "HIPPOCAMPUS_STORE_BACKEND",
    "HIPPOCAMPUS_PERSIST_DIR",
)

#: Identity fields whose drift means the daemon is serving stale code/config.
#: ``cwd`` and ``library_stamp`` are recorded for diagnostics only: one daemon
#: legitimately serves several workspaces, so they must never trigger a warning.
STALE_IDENTITY_KEYS = ("code_stamp", "config_stamp")

_STALE_WARNED: set[str] = set()


def runtime_identity(library: str = "") -> dict[str, Any]:
    """Snapshot of the code/config a daemon was started with.

    ``daemon_key`` only changes when ``HIPPOCAMPUS_DAEMON_VERSION`` is set, so
    without this stamp a source edit leaves an old daemon in place forever.
    The stamp uses file metadata (not content) to stay cheap on the hot path.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    newest, count, total = 0, 0, 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            try:
                st = os.stat(os.path.join(base, name))
            except OSError:
                continue
            newest = max(newest, st.st_mtime_ns)
            count += 1
            total += st.st_size
    code = hashlib.sha1(
        f"{newest}|{count}|{total}".encode("ascii")
    ).hexdigest()[:12]
    config = hashlib.sha1(
        "|".join(f"{k}={os.environ.get(k, '')}" for k in _CONFIG_ENV_KEYS)
        .encode("utf-8", "replace")
    ).hexdigest()[:12]
    lib_stamp = 0
    if library:
        try:
            lib_stamp = os.stat(
                os.path.join(library, "directory.json")).st_mtime_ns
        except OSError:
            lib_stamp = 0
    return {"cwd": os.getcwd(), "code_stamp": code,
            "config_stamp": config, "library_stamp": lib_stamp}


def warn_daemon_stale(info: dict[str, Any]) -> list[str]:
    """Warn once when a reusable daemon runs older code/config.

    Never blocks reuse: refusing to connect on a cwd/config difference would
    break the multi-host, single-daemon topology.  The operator decides via
    ``hippocampus-mcp restart``.
    """
    saved = info.get("identity")
    if not isinstance(saved, dict) or not saved:
        return []
    now = runtime_identity(info.get("library") or "")
    diffs = [k for k in STALE_IDENTITY_KEYS
             if saved.get(k) and now.get(k) and saved.get(k) != now.get(k)]
    if not diffs:
        return []
    tag = f"{info.get('pid')}|{','.join(diffs)}"
    if tag in _STALE_WARNED:
        return diffs
    _STALE_WARNED.add(tag)
    print(f"[launcher] daemon pid={info.get('pid')} is stale ({', '.join(diffs)}); "
          f"it still serves the previous code/config. Run `hippocampus-mcp restart` "
          f"to load the current one.", file=sys.stderr, flush=True)
    return diffs



def _pid_alive(pid: int) -> bool:
    """Read-only liveness probe (same semantics as brain.pid_alive; implemented
    independently here to stay pure standard library)."""
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
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def try_connect(port: int, timeout: float = 1.0) -> socket.socket | None:
    """Probe whether the daemon port is reachable (connectable counts as alive).

    🔴 **The timeout must be cleared after connecting**: ``create_connection(timeout=...)``
    leaves the timeout on the socket, and this socket is then handed to
    :func:`relay` as a **long-lived connection**. If the timeout stays: whenever
    the daemon takes longer than ``timeout`` to handle one message (tool calls
    routinely take seconds to tens of seconds, e.g. loading the embedding model
    on the spot), ``sock.recv()`` raises ``socket.timeout``, which ``_pump``'s
    ``except Exception: pass`` **silently swallows**, killing the whole
    daemon->stdout return pump on the spot -- the host sees "the request went
    out, the response never comes back", **with no error at all**.

    This is the second bug found in real testing (two launcher subprocesses
    connected to the same daemon): the handshake and ``tools/list`` both
    returned within the timeout and looked fine; the first ``tools/call``
    exceeded the timeout because the model had to load first, and the return
    path died there.
    """
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    except OSError:
        return None
    try:
        s.settimeout(None)          # a long-lived connection once handed to relay; no more timeout
    except OSError:
        pass
    return s


def _daemon_creationflags() -> tuple[int, int]:
    """Windows subprocess creation flags: ``(base_flags, breakaway_flag)``.

    ⚠️ **Revision: the premise originally written down was wrong; the code stays
    but is no longer treated as the root fix.**

    The original belief was "the host's Job has ``KILL_ON_JOB_CLOSE`` and kills
    the daemon along with the host during cleanup", based on reading the
    observed ``LimitFlags=0x2400`` as ``KILL_ON_JOB_CLOSE=True``. **The bitmask
    was misread**: the real values in winnt.h are ``BREAKAWAY_OK=0x200`` /
    ``SILENT_BREAKAWAY_OK=0x400`` / ``KILL_ON_JOB_CLOSE=0x800`` -- by the correct
    values, ``0x2400 = JOB_MEMORY | SILENT_BREAKAWAY_OK``, a nearly opposite
    meaning (verified by behavioral experiment).

    The flag is still kept, for two reasons:
    1. It is a **probe**: whether breakaway succeeds directly reflects the host
       Job policy, so the failure warning is valuable in itself;
    2. If the host ever relaxes its Job policy (e.g. allows ``BREAKAWAY_OK`` on
       the inner Job), this takes effect automatically.

    But **do not expect it to fix the root cause**. The real cause: the daemon
    was terminated **externally** (0 orphan reclaims, 0 crash traces in the
    logs), triggered by the host's dispose after roughly 5 minutes idle; the
    launcher itself never kills the daemon (no atexit / terminate).

    It is a separate function so tests can directly assert that flag bit still
    exists (T1 in ``tests/test_f9_6_process_tree.py``), preventing later removal.
    """
    base = 0
    breakaway = 0
    if os.name == "nt":
        base = getattr(subprocess, "DETACHED_PROCESS", 0) | \
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    return base, breakaway


def spawn_daemon(key: str, library: str = "", model: str = "",
                 extra_env: dict[str, str] | None = None,
                 cwd: str = "") -> dict[str, Any] | None:
    """Spawn the daemon and wait for it to write its registry entry. Returns
    None on failure (the caller handles the fallback).

    ``cwd``: the daemon's working directory; empty means the caller's cwd.
    🔴 **Must be given explicitly in the prewarm scenario**: when ``library`` is
    empty the daemon locates the library by **its own cwd**; a wrong cwd spawns
    a daemon pointed at the wrong place -- it still starts, still writes the
    registry, still gets reused by launchers, except the **library is wrong**,
    and nothing reports an error.
    """
    env = dict(os.environ)
    run_cwd = cwd or os.getcwd()
    env["HIPPOCAMPUS_DAEMON"] = "1"
    # 🔴 Output must be **unbuffered**: the daemon dies when the host force-kills
    # it, the bus kills it, or a native crash occurs; nothing in the block buffer
    # survives -- the incident scene shows "the daemon vanished silently, not one
    # line of traceback". This matters even more than "persisting stderr to
    # disk": written to disk but never flushed is like sealing a letter in an
    # envelope and burning it.
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["HIPPOCAMPUS_DAEMON_KEY"] = key
    env[ENV_RUNTIME_DIR] = runtime_dir()
    if library:
        env["HIPPOCAMPUS_DAEMON_LIBRARY"] = library
    if model:
        env["HIPPOCAMPUS_DAEMON_MODEL"] = model
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "hippocampus_mcp.daemon"]
    base_flags, breakaway_flag = _daemon_creationflags()
    creationflags = base_flags | breakaway_flag
    # The daemon's stdout/stderr must go to a file, not DEVNULL: it is the only
    # long-lived heavy process; a crash traceback that enters a black hole can
    # never be attributed (hit during the first real-host verification).
    log_path = os.path.join(runtime_dir(), f"daemon-{key}.log")
    try:
        if os.path.exists(log_path) and os.path.getsize(log_path) > 2 * 1024 * 1024:
            os.replace(log_path, log_path + ".1")
    except OSError:
        pass
    log_fh, log_used = _open_daemon_log(log_path)
    if log_fh is not None:
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_fh.write(
                f"\n=== spawn {stamp} key={key} launcher_pid={os.getpid()} "
                f"cwd={run_cwd} log={os.path.basename(log_used)} ===\n".encode())
            log_fh.flush()
        except OSError:
            pass
    proc = None
    try:
        try:
            proc = subprocess.Popen(
                cmd, env=env, cwd=run_cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_fh if log_fh is not None else subprocess.DEVNULL,
                stderr=log_fh if log_fh is not None else subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except OSError as exc:
            proc = _spawn_without_breakaway(
                cmd, env, log_fh, base_flags, breakaway_flag, exc, run_cwd)
    except Exception:  # noqa: BLE001 -- must degrade if spawn fails; never leave the host unable to connect
        proc = None
    finally:
        # The child process holds its own handle copy; close the parent's copy when done
        if log_fh is not None:
            log_fh.close()
    if proc is None:
        return None
    deadline = time.time() + DAEMON_READY_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            return None                       # spawn failed: do not just keep waiting
        info = read_registry(key)
        if info and info.get("pid") == proc.pid:
            s = try_connect(int(info["port"]))
            if s is not None:
                s.close()
                return info
        time.sleep(0.2)
    return None


# ── spawn singleflight ────────────────────────────────────────────────
# Origin: two launchers 5 seconds apart each discover "no daemon" and each
# spawn => **two daemons race-writing the same registry**, and the first one
# dies silently before it even initializes its brain. That round self-healed
# into a single daemon via idle reclamation, at the cost of two ~1GB processes
# side by side and the registry overwritten by the later writer (who serves
# whom is random).
SPAWN_LOCK_TTL = 120.0      # max lock hold time (seconds); past it, the holder is presumed dead
#: Grace period (seconds) for "just created, content incomplete". See
#: :func:`_spawn_lock_stale` -- during this window an incomplete lock file must
#: not be treated as stale, or you would squash the neighbor that is mid-creation.
SPAWN_LOCK_CREATE_GRACE = 5.0
SPAWN_WAIT_TIMEOUT = 60.0   # when the lock is taken, max wait following another's spawn (seconds)
#: How long at most (seconds) to wait when "the registry has a live pid but the
#: port is not serving yet".
#:
#: During this window it is **not** "no daemon" but "that one is not ready yet":
#: ``Daemon.bind()`` listens before ``announce()``, so on a normal start this
#: window is short; it grows for exactly one reason -- that process is shutting
#: down (socket closed, pid still alive). After waiting the full window without
#: readiness, give up spawning and let the caller fall back to the inline
#: server: **rather keep this session a bit heavier than stack up a second
#: ~1GB daemon racing on the same registry**.
READY_WAIT_WHEN_ALIVE = SPAWN_WAIT_TIMEOUT


def _spawn_without_breakaway(cmd, env, log_fh, base_flags: int,
                             breakaway_flag: int, exc: OSError,
                             cwd: str = ""):
    """When the spawn with BREAKAWAY fails, retry with the old flags.

    🔴 **The fallback must be shouted out**. A silent fallback hides the fact
    that "the daemon remains in the host process tree" -- the next host cleanup
    kills it again, and the scene still looks like "15 daemons, not one line of
    traceback". Availability first is right, but **diagnostic information must
    not be degraded along with it**.
    """
    if not breakaway_flag:
        return None
    msg = (f"[launcher] ⚠ daemon spawn with CREATE_BREAKAWAY_FROM_JOB failed"
           f" ({type(exc).__name__}: {exc}) -- daemon remains in the host process tree; "
           f"it will be terminated with the host session. Retrying without this flag works, "
           f"but does not protect against host cleanup."
           f"To fully detach, use :mod:`hippocampus_mcp.prewarm` to change the creator to "
           f"a login-startup or scheduled task.")
    print(msg, file=sys.stderr, flush=True)
    try:
        return subprocess.Popen(
            cmd, env=env, cwd=cwd or os.getcwd(),
            stdin=subprocess.DEVNULL,
            stdout=log_fh if log_fh is not None else subprocess.DEVNULL,
            stderr=log_fh if log_fh is not None else subprocess.DEVNULL,
            creationflags=base_flags,
        )
    except OSError:
        return None


def spawn_lock_path(key: str) -> str:
    return os.path.join(runtime_dir(), f"daemon-{key}.spawnlock")


def _read_spawn_lock(key: str) -> dict[str, Any] | None:
    """Read the lock content ``{pid, ts}``. Unreadable / wrong format => treat as unlocked."""
    try:
        with open(spawn_lock_path(key), "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(info, dict) and isinstance(info.get("ts"), (int, float)):
        return info
    return None


def _spawn_lock_stale(key: str, info: dict[str, Any] | None) -> bool:
    """Whether the lock can be preempted.

    Deliberately **no** blocking mutex: the holder is another independent
    process that may be killed together with its host session (34280 died
    without leaving a single word). Any blocking lock would then block every
    later launcher behind it => always "time-limited + takeable".

    🔴 **Unreadable content != nobody holds the lock**. The old implementation's
    hole was right here: when the other side had just created an **empty file**
    with ``O_CREAT|O_EXCL`` and not yet written the JSON, our ``json.load``
    failed => None => judged "preemptible" => rm the lock that was not written
    yet, then grab it ourselves -- two launchers both believing they hold the
    lock, two ~1GB daemons side by side (the double-daemon incident had exactly
    this shape).

    Guarding it: when the file exists but its content is incomplete, give it a
    "finish-writing grace period" based on mtime.
    """
    if info:
        ts = float(info.get("ts", 0))
        if ts and time.time() - ts > SPAWN_LOCK_TTL:
            return True
        pid = info.get("pid")
        if (isinstance(pid, int) and pid > 0 and pid != os.getpid()
                and not _pid_alive(pid)):
            return True
        return False
    try:
        age = time.time() - os.stat(spawn_lock_path(key)).st_mtime
    except OSError:
        return True                       # file does not exist => genuinely unlocked
    return age > SPAWN_LOCK_CREATE_GRACE


# In-process serialization lock for lock acquisition.
#
# 🔴 Why it is still needed (os.link is already atomic): only the **publish**
# step is atomic, not the "judge stale -> delete stale lock" step. When 8
# threads in one process start in lockstep, all see "no lock" => all judge it
# preemptible; the first one's link succeeds, and the slower one then runs
# os.remove(lock), **deleting the freshly published new lock**, then links
# successfully itself => two winners. This is exactly why test_f9_6 T6
# occasionally got wins=[True,...,True] on Linux, the same shape as
# "two ~1GB daemons side by side".
#
# Scope note: this lock is effective **within this process only**; cross-process
# mutual exclusion is still guaranteed by os.link's atomicity (others publish
# complete JSON, so a later stale check reads fresh content and yields). The
# narrower cross-process "check-delete" window remains; fully closing it would
# require an atomic claim protocol.
_SPAWN_LOCK_MUTEX = threading.Lock()


def acquire_spawn_lock(key: str) -> bool:
    """Acquire the spawn lock. Returns True when acquired.

    🔴 **The lock file must be "complete the moment it appears"**. The old
    implementation created an **empty file** with ``O_CREAT|O_EXCL`` and only
    then wrote the JSON into it; the window in between is exactly the
    double-daemon cause described in :func:`_spawn_lock_stale`'s comment (the
    old code comment itself admitted this window; now it is closed).

    Fix: **write the complete content to a temp file + fsync first, then attach
    it to the lock path atomically with ``os.link``**. Hard-link creation is
    atomic and necessarily fails when the target exists -- mutual exclusion
    semantics fully preserved, the half-baked window gone entirely.

    On top of that, :data:`_SPAWN_LOCK_MUTEX` serializes the whole "judge stale
    -> clear lock -> write temp file -> atomic publish" sequence into an
    in-process critical section, closing the TOCTOU race described above.
    """
    try:
        os.makedirs(runtime_dir(), exist_ok=True)
    except OSError:
        return False
    lock = spawn_lock_path(key)
    with _SPAWN_LOCK_MUTEX:
        if not _spawn_lock_stale(key, _read_spawn_lock(key)):
            return False
        try:
            os.remove(lock)               # cleared only after confirming staleness
        except OSError:
            pass
        tmp = f"{lock}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "ts": time.time()}, f)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False
        try:
            os.link(tmp, lock)            # atomic publish: target exists => failure => yield
            return True
        except FileExistsError:
            return False                  # lost a normal race to someone else
        except OSError:
            return _acquire_lock_legacy(lock)  # filesystem without hardlink support => fall back
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _acquire_lock_legacy(lock: str) -> bool:
    """Fallback path when ``os.link`` is unavailable (e.g. a filesystem without
    hardlink support).

    Still the old ``O_CREAT|O_EXCL`` semantics, **the half-baked window
    remains**, backstopped by :func:`_spawn_lock_stale`'s grace period -- the
    other side gets :data:`SPAWN_LOCK_CREATE_GRACE` seconds to finish writing
    the content.
    """
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
        return True
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        return False


def _open_daemon_log(log_path: str):
    """Open the daemon log and return ``(handle, actual path)``.

    🔴 **On failure, never quietly fall back to DEVNULL** -- that throws all of
    this daemon's output (including crash tracebacks) into a black hole, showing
    up afterwards as "the daemon died silently, zero clues". Both earlier
    "silent deaths" are under this suspicion.

    **Measured on Windows**: when two processes concurrently ``open(path, "ab")``
    the same file, one of them raises ``PermissionError [Errno 13]`` (probe
    ``_probe_out/_append_concurrency_probe_20260929.py``). Worse, the daemon
    **holds the inherited handle long-term** => a **later** launcher is quite
    likely to hit it -- the same time window as the old double-daemon
    race-writing.

    Handling order: retry a few times (transient conflict) -> if still failing,
    fall back to a **pid-suffixed alternate file** (this daemon's output still
    survives, just not in the main file) -> only if even the alternate cannot be
    opened, hand over to DEVNULL; both degradations print a line to the
    launcher's own stderr so inspections can see them.
    """
    last_err: BaseException | None = None
    for _ in range(3):
        try:
            return open(log_path, "ab"), log_path
        except OSError as exc:
            last_err = exc
            time.sleep(0.05)
    alt = log_path[:-4] + f"-{os.getpid()}.log" if log_path.endswith(".log") \
        else f"{log_path}.{os.getpid()}"
    try:
        fh = open(alt, "ab")
        print(f"[launcher] daemon main log is busy ({type(last_err).__name__}); "
              f"falling back to {alt}", file=sys.stderr)
        return fh, alt
    except OSError:
        print(f"[launcher] cannot open daemon log ({type(last_err).__name__}); "
              f"crash output will be lost", file=sys.stderr)
        return None, ""


def release_spawn_lock(key: str) -> None:
    """Release the lock: **remove only our own**.

    A blanket ``os.remove`` would also delete a new lock someone else just
    acquired -- another path to the double-daemon incident, and the S4 test
    (which only checks "no residue after use") cannot distinguish this at all.
    """
    path = spawn_lock_path(key)
    info = _read_spawn_lock(key)
    if isinstance(info, dict) and info.get("pid") == os.getpid():
        try:
            os.remove(path)
        except OSError:
            pass
        return
    print(f"[launcher] ⚠ spawn lock is no longer owned by this process (content={info}); not removing {path}",
          file=sys.stderr, flush=True)


def _registry_usable(info: dict[str, Any] | None) -> bool:
    """Whether a registry entry is connectable (pid alive + port handshakes)."""
    if not info:
        return False
    pid = info.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False
    s = try_connect(int(info["port"]), timeout=2.0)
    if s is None:
        return False
    s.close()
    warn_daemon_stale(info)
    return True


def _registry_has_live_pid(info: dict[str, Any] | None) -> bool:
    """Is the pid recorded in the registry still alive (regardless of whether
    the port is serving)?

    🔴 This check fixes a **semantic error**, not a race: the old code only asked
    "is the port connectable", and treated a failure as "no daemon", then spawned
    another one. But "not connectable" has two sources -- nobody there, or
    **someone there but not ready yet**; the old code counted the latter as the
    former.
    """
    if not info:
        return False
    pid = info.get("pid")
    return isinstance(pid, int) and pid > 0 and _pid_alive(pid)


def _wait_until_ready(key: str, info: dict[str, Any] | None,
                      limit: float) -> dict[str, Any] | None:
    """Wait for a "alive but not serving yet" daemon to become ready.

    The moment the pid dies or the registry entry is cleared (the normal exit
    path clears it), return None immediately -- the caller spawns its own, not
    one second more. If ``limit`` elapses without readiness, also return None:
    by then it is either shutting down or hung; spawning another in either case
    only makes things worse.
    """
    end = time.time() + max(0.0, float(limit))
    while time.time() < end:
        time.sleep(0.2)
        fresh = read_registry(key)
        if _registry_usable(fresh):
            return fresh
        if not _registry_has_live_pid(fresh):
            return None
    return None


def ensure_daemon(key: str, library: str = "", model: str = "") -> dict[str, Any] | None:
    """Find or spawn the daemon -- **at any moment only one launcher actually
    spawns**.

    Four paths:
      ① Registry usable -> return directly (most common, zero wait)
      ② Got the spawn lock and the registry has **no live pid** -> double-check,
         then spawn for real (someone may have finished starting while we held
         the lock)
      ③ Got the spawn lock but the registry pid is **still alive** (port not
         serving) -> wait for it to become ready, **never start another**; only
         if it dies does this round spawn
      ④ Could not get the lock -> follow and wait (if the lock holder dies or
         the window times out, back to ② to try ourselves)
    """
    deadline = time.time() + DAEMON_READY_TIMEOUT
    while time.time() < deadline:
        info = read_registry(key)
        if _registry_usable(info):
            return info
        if acquire_spawn_lock(key):
            try:
                info = read_registry(key)          # double-check
                if _registry_usable(info):
                    return info
                # ③ Someone is alive but not serving => wait for it, **not**
                #    start another. Under 4-way concurrency this branch
                #    measurably stacked a second daemon 1/4 of the time (two
                #    banners with different pids, the registry overwritten by
                #    the later arrival, who serves whom is random).
                if _registry_has_live_pid(info):
                    ready = _wait_until_ready(
                        key, info,
                        min(READY_WAIT_WHEN_ALIVE,
                            max(0.0, deadline - time.time())))
                    if ready is not None:
                        return ready
                    if _registry_has_live_pid(read_registry(key)):
                        return None               # alive but not serving: prefer degrading to inline
                return spawn_daemon(key, library=library, model=model)
            finally:
                release_spawn_lock(key)
        # ④ Someone else is spawning: follow, watching both the registry and the lock (to avoid waiting on a corpse)
        wait_until = min(time.time() + SPAWN_WAIT_TIMEOUT, deadline)
        while time.time() < wait_until:
            time.sleep(0.2)
            info = read_registry(key)
            if _registry_usable(info):
                return info
            # The registry entry is still there but the pid is dead -- equally
            # "following someone who no longer exists": break out early instead
            # of burning the full SPAWN_WAIT_TIMEOUT before our own spawn.
            if not _registry_has_live_pid(info) and _spawn_lock_stale(
                    key, _read_spawn_lock(key)):
                break                              # lock holder gone and daemon not up => do it ourselves
    return None


#: Debug switch for the relayed byte stream (``HIPPOCAMPUS_LAUNCHER_DEBUG=1``).
#: When on, every relayed segment's direction and length go to stderr -- for
#: diagnosing silent hangs like "the request went out but the response never
#: returns", this is the only way to see the bidirectional flow.
_RELAY_DEBUG = os.environ.get("HIPPOCAMPUS_LAUNCHER_DEBUG", "") == "1"


def _name(obj: Any) -> str:
    if hasattr(obj, "recv"):
        return "sock"
    if obj is getattr(sys, "stdin", None):
        return "stdin"
    return "stdout"


def _pump(src, dst, chunk: int = RELAY_CHUNK) -> None:
    """One-way byte pump. Stops when either end closes, never raises.

    🔴 **Must use ``read1`` rather than ``read``**: ``BufferedReader.read(n)``'s
    semantics are "return only when n bytes are read or EOF is hit", which for
    pipes/stdio means **the host is blocked forever after writing one JSON-RPC
    line** -- MCP over stdio frames are usually only a few hundred bytes, far
    below 64KB, so the host's initialize never gets a response, showing up as
    "tool count 0 / service unavailable", **with no error at all**.
    ``read1(n)`` does exactly one underlying read and returns whatever is
    available -- that is the interface a byte pump should use.

    This pitfall was found in real testing (two launcher subprocesses connected
    to the same library): the daemon was up and the registry written, yet
    session 1's handshake could not get a single byte within 180 seconds.
    """
    try:
        while True:
            if hasattr(src, "read1"):
                data = src.read1(chunk)      # return whatever is available, never wait to fill
            elif hasattr(src, "read"):
                data = src.read(chunk)
            else:
                data = src.recv(chunk)
            if not data:
                break
            if _RELAY_DEBUG:
                print(f"[relay] {_name(src)}->{_name(dst)} {len(data)}B",
                      file=sys.stderr, flush=True)
            dst.write(data) if hasattr(dst, "write") else dst.sendall(data)
            if hasattr(dst, "flush"):
                dst.flush()
    except Exception:  # noqa: BLE001 -- a broken pipe is the normal way this ends
        pass
    try:
        if hasattr(dst, "close"):
            dst.close()
    except Exception:  # noqa: BLE001
        pass


def _send_hello(sock: socket.socket) -> bool:
    """Before relaying any MCP traffic, tell the daemon "which library this
    connection belongs to".

    🔴 Without this step, libraries get crossed: the daemon's default library is
    a **process-level** constant (computed at server module import time from
    whatever launcher's env/cwd came first). When two workspaces reuse the same
    daemon (same key -- which is exactly the point of saving memory), the later
    workspace writes into the earlier one's library, with no error.

    It must be sent **before** all client bytes so the daemon can consume it
    cleanly with a single peek.
    """
    payload = {
        "library": os.environ.get("HIPPOCAMPUS_PERSIST_DIR", ""),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }
    line = HELLO_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n"
    try:
        sock.sendall(line.encode("utf-8"))
        return True
    except OSError:
        # Failing to send must not keep the session from starting: the daemon falls back to the process-level default library (old behavior)
        return False


# ══════════════════════════════════════════════════════════════════════
# stdio <-> daemon relay: **line framing + self-healing on disconnect**
# ══════════════════════════════════════════════════════════════════════
# This section is a rewrite forced by a real-host incident. The old
# implementation was two **bare byte pumps**: ``_pump(stdin, sock)`` (main
# thread) + ``_pump(sock, stdout)`` (another thread). When the daemon was
# reclaimed by the host, the two directions behaved **asymmetrically**, and the
# incident unfolded in two steps:
#
#   ① The return pump got EOF => exited => reached ``_pump``'s finally =>
#      **close(stdout)**. The host read EOF, judged the MCP server offline,
#      and logged ``mcp.server.closed``.
#   ② The outbound pump was blocked in ``stdin.read1()`` and knew nothing of the
#      peer's death => the launcher **process stayed alive** holding a dead pipe
#      => zombie state. The host neither reconnected nor restarted, because it
#      had already pronounced this connection dead.
#
# Every hippo call in later sessions returned ``Connection closed`` in 22~33 ms.
# Note: **not one of them reached the network layer** -- we closed the
# connection ourselves.
#
# With line framing, three things became true for the first time:
#   · **Atomic**: a whole JSON-RPC message either fully reaches the daemon or is
#     not sent at all; no half frames.
#   · **Countable**: in-flight request ids are recorded per JSON-RPC semantics.
#   · **Reconnectable**: breaks happen at frame boundaries, so a fresh socket can
#     pick up where things stopped, **with the host never needing to know a
#     disconnect happened**.
#
# The daemon side cooperates naturally: ``bridge.py``'s ``sock_reader`` already
# splits frames on ``\n``, and for illegal frames it "skips and shouts" rather
# than killing the connection (see bridge.py lines 111-119).
#
# Self-check list (read ``tests/test_f9_5_relay_reconnect.py`` before changing
# this):
#   1. **The return thread must never close(stdout)** -- that pronounces the
#      death sentence on the host's behalf;
#   2. If self-healing fails, **fail explicitly** (close stdout + non-zero exit),
#      never stay in a zombie state;
#   3. After reconnecting, the **MCP session must be rebuilt on the new daemon**
#      (replay initialize), otherwise later tools/call are rejected with "not
#      initialized";
#   4. Pending requests must get an **explicit settlement** (synthesized error),
#      never left hanging.

#: Self-healing switch. ``HIPPOCAMPUS_LAUNCHER_RECONNECT=0`` reverts to the old
#: one-way byte pump behavior, only for A/B comparison and as the **negative
#: control** in tests. Do not turn it off in production.
ENV_RECONNECT = "HIPPOCAMPUS_LAUNCHER_RECONNECT"
#: Diagnostic output switch (writes stderr, never stdout).
ENV_DIAG = "HIPPOCAMPUS_LAUNCHER_DIAG"
#: Max reconnect attempts per disconnection.
RECONNECT_MAX_ATTEMPTS = 3
#: Total repair budget (seconds) per disconnection. A cold daemon start loads
#: the embedding model (~13-35s measured); be generous, but keep it below the
#: host's typical tool-call timeout.
RECONNECT_DEADLINE = 90.0
#: Backoff base (seconds) after a failed reconnect, scaled linearly by attempt.
RECONNECT_BACKOFF = 0.5
#: Upper limit (seconds) for waiting for the response when replaying initialize
#: after a reconnect.
HANDSHAKE_REPLAY_TIMEOUT = 60.0

#: The message used to account for pending requests. **It must state clearly
#: that "the result is unknown"** -- never imply success or failure; callers
#: (especially of write operations) rely entirely on this sentence to decide
#: whether to retry.
ERR_DAEMON_LOST = (
    "HippoMCP daemon connection was interrupted and could not recover immediately: the result of this request is UNKNOWN"
    " (it may have executed, or may never have reached the daemon). Idempotent operations may be safely retried; "
    "for non-idempotent writes (ingest / update_memory / forget), read back before retrying."
)

#: Filled in by main() before running -- _Relay relies on it to know "which key to reconnect to".
_SESSION_PARAMS: dict[str, str] = {"key": "", "library": "", "model": ""}


def _diag(msg: str) -> None:
    """Diagnostics may only go to stderr.

    stdout is the **JSON-RPC-only channel**; dirtying a single byte ruins the
    whole MCP stdio session (``server.py``'s ``_ThreadAwareStdout`` guard exists
    for the same reason).
    """
    if os.environ.get(ENV_DIAG, "1") == "0":
        return
    try:
        print(f"[launcher] {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 -- diagnostics must never affect the main path
        pass


def reconnect_enabled() -> bool:
    return os.environ.get(ENV_RECONNECT, "1") != "0"


def _id_key(rid: Any) -> str:
    """Compress a JSON-RPC id (str / int / None) into a stable dict key."""
    return json.dumps(rid, sort_keys=True, default=str)


def _request_key(line: bytes) -> str | None:
    """Is this frame a **request that needs a response**? If so, return its key.

    Notifications (method without id) need no response and therefore stay out of
    the pending ledger -- recording them would have ``_fail_inflight``
    synthesize an error nobody will ever claim.
    """
    try:
        obj = json.loads(line.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(obj, dict):
        return None
    if "method" not in obj or "id" not in obj:
        return None
    return _id_key(obj["id"])


class _RelayLost(Exception):
    """The daemon connection is unrecoverable -- this stdio must be terminated, never left in a zombie state."""


class _Relay:
    """Line-framed relay between stdio and the daemon socket.

    Thread model (different from the old double pump):
      · The main thread only does stdin -> daemon (**line by line**)
      · Each daemon connection gets a return thread doing daemon -> stdout
        (**line by line**)
      · The return thread does **not own** stdout's lifetime; stdout closes only
        when the host closes stdin or self-healing is explicitly abandoned.
    """

    def __init__(self, key: str, library: str = "", model: str = "") -> None:
        self.key = key
        self.library = library
        self.model = model
        self._sock: socket.socket | None = None
        self._gen = 0
        self._sock_lock = threading.Lock()
        self._stdout_lock = threading.Lock()
        self._reseat_lock = threading.Lock()
        self._inflight: dict[str, str] = {}     # id_key -> original request frame
        self._inflight_lock = threading.Lock()
        self._handshake: list[bytes] = []       # cached initialize/initialized
        self._initialized = False
        self._sock_dead = threading.Event()
        self._stopped = threading.Event()
        self._stdout = None
        self._stdin = None

    # ── Main loop ──────────────────────────────────────────────────────
    def run(self, sock: socket.socket) -> int:
        """Returns 0 = host closed normally; 4 = self-healing failed, disconnected explicitly."""
        try:
            self._stdout = sys.stdout.buffer
            self._stdin = sys.stdin.buffer
        except Exception:  # noqa: BLE001
            return 2
        # 🔴 Must declare which library this connection belongs to **before**
        # relaying any MCP bytes, or libraries get crossed (see :func:`_send_hello`'s
        # comment). The reconnect path is handled by ``_connect_fresh`` separately;
        # only the first connection is handled here.
        _send_hello(sock)
        if not self._install(sock):
            return 3
        try:
            self._pump_stdin()
        except _RelayLost:
            self._abandon()
            return 4
        finally:
            self._stopped.set()
            self._drop_sock()
        return 0

    def _pump_stdin(self) -> None:
        """Main thread: host stdio -> daemon, one line at a time.

        🔴 Using ``readline`` instead of ``read``/``read1`` is deliberate: what is
        wanted here is exactly the "return at end of line" semantics.
        ``BufferedReader.read(n)`` waits until n bytes are read or EOF
        (``read1`` was introduced to fix exactly that; see :func:`_pump`'s
        comment), while ``readline`` naturally returns per line and has no such
        problem. Do not "optimize" it back to ``read1`` + a hand-rolled buffer --
        that is reinventing BufferedReader.
        """
        while True:
            if self._stopped.is_set():
                return
            line = self._stdin.readline()
            if not line:
                return                              # host closed stdin: normal shutdown
            line = line.strip()
            if not line:
                continue
            self._deliver(line)

    # ── host → daemon ───────────────────────────────────────────────
    def _deliver(self, line: bytes) -> None:
        """Deliver one whole frame. If it cannot be sent, self-heal (reconnect + retransmit this frame)."""
        self._remember_handshake(line)
        # Registration must happen **before** sending: the response may come back
        # the instant sendall returns; registering after sending would miss this
        # frame's settlement.
        self._register_inflight(line)

        # 🔴 Check ``_sock_dead`` before sending: the return thread usually learns
        # the peer died earlier than the send path. Without this step, sendall
        # **successfully writes into the kernel buffer** (the first write to a
        # closed peer often does not error), so we believe it went out and the
        # host hangs until timeout -- far worse than self-healing immediately.
        if not self._sock_dead.is_set() and self._send(line):
            return

        # 🔴 **Registration is deliberately not undone here**. This frame has only
        # two destinies: retransmitted successfully by ``_reseat``, or canceled by
        # ``_replay_pending`` when retransmission fails. Canceling midway makes it
        # vanish from the ledger -- ``_fail_inflight`` never sees it, the host
        # waits forever for this response and hangs until timeout (an accounting
        # bug hit in self-testing; the first version did exactly this: the log
        # showed "registered pending id=13 in-flight=2", yet only 1 was settled).
        if self._reseat(pending=line):
            return
        n = self._fail_inflight(ERR_DAEMON_LOST)
        _diag(f"Recovery failed; returned error for {n} pending requests and terminated this stdio session")
        raise _RelayLost("daemon connection is unrecoverable")

    def _send(self, line: bytes) -> bool:
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall(line + b"\n")
            return True
        except OSError:
            self._sock_dead.set()
            return False

    # ── daemon → host ───────────────────────────────────────────────
    def _pump_sock(self, sock: socket.socket, gen: int) -> None:
        """Return path: split frames on ``\\n`` and write whole lines to stdout.

        🔴 **This function must never close(stdout)**. The old implementation's
        return pump closed stdout in its finally block; the host read EOF and
        judged the server offline -- the first domino of the incident. As long as
        the host has not closed stdin, this MCP connection is still valid and we
        must keep this stdout.
        """
        buf = b""
        try:
            while True:
                try:
                    data = sock.recv(RELAY_CHUNK)
                except OSError:
                    break
                if not data:
                    break
                with self._sock_lock:
                    stale = self._gen != gen or self._sock is not sock
                if stale:
                    return                       # superseded connection of an older generation: exit quietly
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    self._settle_inflight(raw.strip())
                    self._write_stdout(raw + b"\n")
        finally:
            if buf.strip():
                # Broke mid-frame: this frame can never be complete, so it must be
                # dropped. Say so loudly -- a silently dropped frame will be read
                # as "the daemon did not respond" in some future incident.
                _diag(f"Discarded partial frame {len(buf)}B (connection broke at a non-frame boundary)")
            with self._sock_lock:
                current = self._gen == gen and self._sock is sock
            if current and not self._stopped.is_set():
                self._sock_dead.set()
                if reconnect_enabled():
                    # Proactive prewarm: rather than spawn the daemon only when the
                    # next request arrives (cold start ~13-35s, likely to blow
                    # through the host's call timeout), refill it in the background now.
                    threading.Thread(
                        target=self._reseat, kwargs={"pending": b""},
                        daemon=True).start()

    # ── Reconnect ────────────────────────────────────────────────────
    def _reseat(self, pending: bytes = b"") -> bool:
        """Swap in a new daemon connection; on success, retransmit ``pending``.

        Holds ``_reseat_lock`` throughout: the return thread's background prewarm
        and the main thread's request delivery can arrive here concurrently;
        without serialization, two launchers each spawn a daemon (exactly that
        ~1GB x2 double-daemon incident).
        """
        if not reconnect_enabled():
            return False
        if not self._reseat_lock.acquire(timeout=RECONNECT_DEADLINE):
            return False
        try:
            deadline = time.time() + RECONNECT_DEADLINE
            attempt = 0
            while time.time() < deadline:
                with self._sock_lock:
                    alive = not self._sock_dead.is_set()
                if alive:
                    return self._replay_pending(pending)   # someone else already fixed it
                attempt += 1
                if attempt > RECONNECT_MAX_ATTEMPTS:
                    break
                _diag(f"daemon connection dropped; reconnect attempt {attempt}")
                s = self._connect_fresh()
                if s is None:
                    time.sleep(RECONNECT_BACKOFF * attempt)
                    continue
                if not self._install(s, replay_handshake=True):
                    continue
                _diag("Reconnected; MCP session rebuilt on new daemon")
                return self._replay_pending(pending)
            return False
        finally:
            self._reseat_lock.release()

    def _replay_pending(self, pending: bytes) -> bool:
        """Retransmit the frame that failed to send before the reconnect.

        🔴 This frame was already registered earlier; do **not** register it again
        here, and cancel it only once sending is confirmed failed -- ``_deliver``
        is the sole entry point to the ledger, otherwise the same id gets
        overwritten/canceled from two places and nobody can say whether it was
        ever sent.
        """
        if not pending:
            return True
        if self._send(pending):
            return True
        self._rollback_inflight(pending)
        return False

    def _connect_fresh(self) -> socket.socket | None:
        info = ensure_daemon(self.key, library=self.library, model=self.model)
        if info is None:
            return None
        try:
            s = try_connect(int(info["port"]), timeout=5.0)
        except Exception:  # noqa: BLE001
            return None
        if s is None:
            return None
        if not _send_hello(s):
            try:
                s.close()
            except OSError:
                pass
            return None
        return s

    def _install(self, sock: socket.socket, replay_handshake: bool = False) -> bool:
        """Take over a new connection (and start its return thread)."""
        if replay_handshake and not self._replay_handshake(sock):
            _diag("Failed to rebuild MCP session after reconnect; trying another connection")
            try:
                sock.close()
            except OSError:
                pass
            return False
        with self._sock_lock:
            old = self._sock
            self._sock = sock
            self._gen += 1
            gen = self._gen
        if old is not None:
            try:
                old.close()
            except OSError:
                pass
        self._sock_dead.clear()
        threading.Thread(
            target=self._pump_sock, args=(sock, gen), daemon=True).start()
        return True

    def _replay_handshake(self, sock: socket.socket) -> bool:
        """Redo the host's handshake on the new daemon.

        🔴 **This step cannot be skipped**: on the new daemon this is a brand new
        MCP session; without replay, the host's subsequent ``tools/list`` /
        ``tools/call`` are rejected with "initialize not completed". The host
        itself considers that history done (it initialized long ago) and will
        never redo it => only the launcher can fill it in on the host's behalf.

        The initialize response must be **consumed here**, not forwarded to the
        host -- otherwise the same id would get a second response.
        """
        if not self._initialized or not self._handshake:
            return True
        for raw in self._handshake:
            if _RELAY_DEBUG:
                print(f"[relay] replaying handshake frame {raw[:70]!r}",
                      file=sys.stderr, flush=True)
            try:
                sock.sendall(raw + b"\n")
            except OSError:
                return False
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001
                continue
            if obj.get("method") == "initialize" and "id" in obj:
                if not self._drink(sock):
                    if _RELAY_DEBUG:
                        print("[relay] no response to replayed handshake",
                              file=sys.stderr, flush=True)
                    return False
        return True

    @staticmethod
    def _drink(sock: socket.socket) -> bool:
        """Read one whole line and discard it."""
        try:
            sock.settimeout(HANDSHAKE_REPLAY_TIMEOUT)
            buf = b""
            while b"\n" not in buf:
                data = sock.recv(RELAY_CHUNK)
                if not data:
                    return False
                buf += data
            return True
        except OSError:
            return False
        finally:
            try:
                sock.settimeout(None)
            except OSError:
                pass

    def _drop_sock(self) -> None:
        with self._sock_lock:
            sock = self._sock
            self._sock = None
            self._gen += 1
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # ── Handshake history / in-flight ledger ─────────────────────────
    def _remember_handshake(self, line: bytes) -> None:
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        if not isinstance(obj, dict):
            return
        method = obj.get("method")
        if method == "initialize":
            # Keep only the newest initialize: the host redoes the handshake when it restarts the session.
            self._handshake = [line.strip()]
            self._initialized = False
        elif method == "notifications/initialized":
            if self._handshake:
                self._handshake.append(line.strip())
            self._initialized = True

    def _register_inflight(self, line: bytes) -> str | None:
        """Register a pending request. Only a **request** (has both method and id) needs a response."""
        k = _request_key(line)
        if k is None:
            return None
        with self._inflight_lock:
            self._inflight[k] = line.decode("utf-8", "replace")
            if _RELAY_DEBUG:
                print(f"[relay] -> registered pending id={k} in-flight={len(self._inflight)}",
                      file=sys.stderr, flush=True)
        return k

    def _rollback_inflight(self, line: bytes) -> None:
        """Cancel the registration only after **confirming this frame ultimately failed to send**."""
        k = _request_key(line)
        if k is None:
            return
        with self._inflight_lock:
            existed = self._inflight.pop(k, None)
        if _RELAY_DEBUG and existed is not None:
            print(f"[relay] xx canceled pending id={k}", file=sys.stderr, flush=True)

    def _settle_inflight(self, line: bytes) -> None:
        """A response arrived; settle the pending entry."""
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        if not isinstance(obj, dict) or "id" not in obj:
            return
        if "method" in obj:
            # A request initiated by the daemon (sampling / elicitation etc.),
            # not a response to a host request -- must not be used for settlement.
            return
        with self._inflight_lock:
            removed = self._inflight.pop(_id_key(obj["id"]), None)
        if _RELAY_DEBUG and removed is not None:
            print(f"[relay] <- completed id={_id_key(obj['id'])}",
                  file=sys.stderr, flush=True)

    def _fail_inflight(self, reason: str) -> int:
        """Give every pending request an **explicit settlement**.

        Why an error instead of leaving them there: leaving them means every
        future host request hangs until timeout and the user sees a context-free
        "tool call failed". Why not replay automatically: we do not know whether
        this frame was already executed by the daemon -- replaying ``ingest``
        would silently write duplicates. Telling the caller "the result is
        unknown" honestly and letting the **semantics-aware upper layer** decide
        whether to retry is the proper duty.
        """
        with self._inflight_lock:
            items = list(self._inflight.items())
            self._inflight.clear()
        for k, raw in items:
            try:
                rid = json.loads(k)
            except Exception:  # noqa: BLE001
                rid = raw
            resp = {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32000, "message": reason},
            }
            self._write_stdout(
                (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        return len(items)

    def _write_stdout(self, data: bytes) -> None:
        # The return thread and the main thread (synthesized error) may write
        # concurrently => must serialize + write whole frames.
        with self._stdout_lock:
            try:
                self._stdout.write(data)
                self._stdout.flush()
            except Exception:  # noqa: BLE001
                self._stopped.set()

    def _abandon(self) -> None:
        """If self-healing is impossible, give up explicitly -- close stdout +
        exit non-zero.

        The worst part of the old behavior was not dying, it was **not dying**:
        the launcher process alive, stdout closed, the host believing the
        connection open, so every call fired into the void and returned a
        content-free ``Connection closed`` in 22~33ms. Either work, or
        disconnect explicitly.
        """
        self._stopped.set()
        self._drop_sock()
        try:
            self._stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._stdout.close()
        except Exception:  # noqa: BLE001
            pass


def relay(sock: socket.socket) -> int:
    """Bridge this daemon connection into the host's stdio until the session ends.

    Return codes: 0 = host closed stdin (normal); 2/3 = could not start; 4 =
    self-healing failed, disconnected.
    """
    return _Relay(
        _SESSION_PARAMS.get("key", ""),
        _SESSION_PARAMS.get("library", ""),
        _SESSION_PARAMS.get("model", ""),
    ).run(sock)


def run_server_inline(argv: list[str] | None = None) -> int:
    """Fallback path: run the original server directly (identical to today).

    The server entry takes zero arguments (``entry.run_main`` does not read the
    command line), so argv has nowhere to go: the parameter exists only to keep
    the fallback branch's call site unchanged.
    """
    from .server import main as _server_main   # lazy: the heavy libraries are imported only here
    return _server_main() or 0


def preflight_persist_dir() -> int | None:
    """A fail-fast probe of the library directory before startup.

    The one place **deliberately opposite** to the overall "degrade to inline on
    any failure" principle: the degradation philosophy protects "the launcher's
    own mechanism broke, yet the service must still start"; an unwritable
    library directory is **the configuration itself being broken** -- the
    service still starts, but every write is silently lost and the user only
    discovers it when "nothing can be remembered". Rather than hang until the
    host times out and get silently marked failed, exit now with a clear stderr
    message that states the fix path directly.

    Returns None = continue; returns int = exit with that code.
    """
    lib = os.environ.get(ENV_PERSIST, "").strip()
    if not lib:
        # Unset is not an error (manually running the launcher for debugging is a
        # legitimate scenario), but it must self-report: the default library at
        # this moment is probably not the one the user thinks -- cross-library
        # use never errors.
        print(f"[launcher] ⚠ {ENV_PERSIST} is not set; this run will use the default library. "
              f"If launched by a host, the host probably did not read its config. "
              f"Run `python -m hippocampus_mcp.cli doctor --host <host>` for a verdict.",
              file=sys.stderr, flush=True)
        return None
    try:
        os.makedirs(lib, exist_ok=True)
        probe = os.path.join(lib, ".hippo-write-probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as exc:
        print(f"[launcher] ✗ Library directory is not writable; refusing to start: {lib}\n"
              f"  Reason: {type(exc).__name__}: {exc}\n"
              f"  Why: rather than let the host time out and silently fail every later write, "
              f"disconnect now.\n"
              f"  Fix: change {ENV_PERSIST} in the host config env to a writable directory, "
              f"or check disk/permissions and restart the session.",
              file=sys.stderr, flush=True)
        return 2
    return None


def main(argv: list[str] | None = None) -> int:
    """Launcher entry point.

    Any failure falls back to ``run_server_inline`` -- availability > memory
    savings. The exception is an unwritable library directory: that is broken
    configuration, fail fast (see preflight_persist_dir).
    """
    fail = preflight_persist_dir()
    if fail:
        return fail
    if not launcher_enabled():
        return run_server_inline(argv)
    try:
        library = os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", "")
        model = os.environ.get("HIPPOCAMPUS_EMB_MODEL", "")
        version = os.environ.get("HIPPOCAMPUS_DAEMON_VERSION", "")
        key = daemon_key(library, model, version)
        # "Slow" and "broken" must be distinguishable: this stretch can wait up to
        # DAEMON_READY_TIMEOUT (default 180s; the first start loads the embedding
        # model), and if stderr says nothing meanwhile, host and user can only
        # conclude "hung", then treat it as "broken" (kill the process / change
        # config). Flush a status line first so the wait is recognizable as a wait.
        _diag(f"Connecting to shared daemon (first start loads the embedding model; tens of seconds is normal; "
              f"after {DAEMON_READY_TIMEOUT:.0f}s without readiness, falls back to direct server)")
        info = ensure_daemon(key, library=library, model=model)
        if info is None:
            _diag("Daemon failed to start; falling back to direct server (this session gets its own model; "
                  f"logs under {runtime_dir()}/daemon-*.log)")
            return run_server_inline(argv)
        sock = try_connect(int(info["port"]), timeout=5.0)
        if sock is None:
            _diag("Daemon registry is ready but its port is unreachable; falling back to direct server")
            return run_server_inline(argv)
        # _Relay relies on this to know "which key to reconnect to after a break"
        _SESSION_PARAMS["key"] = key
        _SESSION_PARAMS["library"] = library
        _SESSION_PARAMS["model"] = model
        rc = relay(sock)
        if rc == 4:
            _diag("stdio disconnected deliberately; surface an explicit failure to the host instead of leaving a zombie connection")
        return rc
    except Exception:  # noqa: BLE001 -- the launcher must never, for any reason, leave the host unable to connect
        return run_server_inline(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
