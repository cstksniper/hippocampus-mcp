# -*- coding: utf-8 -*-
"""hippocampusd: the resident daemon. **One global copy of the model** lives here.

Division of labor:
- :mod:`hippocampus_mcp.launcher` is a thin pure-stdlib shell that only finds/starts
  the daemon and forwards bytes
- This module is the resident process, **holding all HippoBrain instances**; one
  process may serve multiple libraries at once

Why this saves memory: of the fixed 755 MB single-process overhead, the bulk is
embedding model weights, which is independent of the number of memory records.
N sessions each spawning a process = N copies of the weights; after daemonizing
there is 1.

Connection semantics: each TCP connection is an independent MCP ``ServerSession``,
so ``ctx.session`` naturally routes by connection — this is the channel for
automatic session addressing.
**Never reuse a connection**, otherwise the per-connection hidden state gets glued
back together.

Usage::

    HIPPOCAMPUS_DAEMON=1 python -m hippocampus_mcp.daemon

Status: implemented and verified on local loopback; **the default is still direct
connection** (see the switch in launcher).
Changing the default requires one attended verification (host stdio semantics /
hot upgrade / multi-library), so it is not changed in unattended automation.
"""

# Exemption registration: this file's direct env reads are process-topology
# plumbing (socket/key / sentinel thresholds / diagnostics); the closed list is
# in config.py's "root-layer exemption list".
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from typing import Any

from .launcher import (read_registry, registry_path, runtime_dir,
                       runtime_identity, stop_request_path, write_registry)
from .engine import model_lifecycle
from .library.registry import DEFAULT_KEY


#: Heartbeat interval (seconds). 30s is a tradeoff: too dense ⟹ wasted disk
#: writes; too sparse ⟹ imprecise death-time localization.
#: ``0`` disables it (use it in tests that don't want polling).
HEARTBEAT_INTERVAL = float(
    os.environ.get("HIPPOCAMPUS_DAEMON_HEARTBEAT", "30") or 0)
#: Write one log line every N heartbeat rounds (to keep 30s-per-line from
#: flooding the log).
HEARTBEAT_LOG_EVERY = 10

#: How long to idle before unloading the embedding model (seconds).
#: **Default 30 minutes**: enough to cover getting up for a glass of water,
#: without letting a continuous conversation get bogged down by repeated
#: reloads; measured reload takes only 2.5–3.7s.
#: ``0`` disables it (keeps the old "loaded as long as the process lives"
#: behavior).
DEFAULT_MODEL_UNLOAD = 1800.0
ENV_MODEL_UNLOAD = "HIPPOCAMPUS_MODEL_IDLE_UNLOAD"
#: Poll interval of the unload sentinel (seconds). Every 30s; the error is
#: meaningless against a 30-minute threshold.
MODEL_UNLOAD_POLL = 30.0

#: Cooperative stop marker poll / timed-flush tick (seconds).
MAINTENANCE_POLL = 5.0
#: Time-driven flush window.  Step-based autosave (L1 every 10 writes, L0
#: every 20) leaves an unbounded in-memory tail when writes are sparse.
#: Default 15 min; ``0`` disables the timer (the stop marker still works).
DEFAULT_SAVE_IDLE = 900.0
ENV_SAVE_IDLE = "HIPPOCAMPUS_DAEMON_SAVE_IDLE"

# ----------------------------------------------------------------------
# Idle "sleep window" — the host-side knob for offline replay-style
# consolidation. Same pattern as model unload: decided by the daemon process
# topology, direct env reads bypassing config (rationale: root-layer entry
# points are startup topology plumbing). The three semantic knobs on the
# brain side (replay window / rate limit / cluster minimum) live in
# config.py and are fetched at call time by brain.core.sleep_batch.
# ----------------------------------------------------------------------
#: How many seconds of idle before sleeping is allowed. **Default 0 = off**
#: (don't change production behavior until retrieval-strength wiring is
#: done; flip the default for separate verification once wiring completes).
#: The criterion is "the brain is genuinely idle": no active connections
#: **and** ``model_lifecycle.idle_seconds()`` >= this value.
DEFAULT_SLEEP_IDLE = 0.0
ENV_SLEEP_IDLE = "HIPPOCAMPUS_SLEEP_IDLE"
#: Fallback time budget for a single sleep batch (seconds), passed to
#: ``brain.sleep_batch``.
DEFAULT_SLEEP_BUDGET = 120.0
ENV_SLEEP_BUDGET = "HIPPOCAMPUS_SLEEP_BUDGET_S"
#: Poll interval of the sleep sentinel (seconds). Same rationale as
#: MODEL_UNLOAD_POLL: 30s granularity is meaningless for minute-scale
#: activity like sleeping.
SLEEP_POLL = 30.0
from .library.registry import BrainRegistry  # multi-library registry sunk into a shared module

# How many bytes to read from each connection at once
CONN_CHUNK = 65536
# Pending-queue cap: messages pushed by a single connection must not pile
# up without bound
CONN_QUEUE = 256


# ======================================================================
# Connection handling
# ======================================================================
# The library registry has sunk into .registry (server-side direct stdio
# also needs multi-library, sharing the same copy)
class Connection:
    """One client connection.

    The current implementation is a **byte pump + session registration**: before
    handing the connection to the real MCP server, the identity of "this
    connection = one session" is attached here, for ``ctx.session`` to route by.
    The real MCP session runs in ``handle()`` (taken over by the server side).
    """

    def __init__(self, sock: socket.socket, addr, conn_id: int) -> None:
        self.sock = sock
        self.addr = addr
        self.conn_id = conn_id
        self.started_at = time.time()
        self.closed = False

    def session_key(self) -> str:
        """Automatic channel: one key per connection, naturally routed by session."""
        return f"conn-{self.conn_id}"


def _pump(src, dst) -> None:
    """One-way byte pump (used by the echo diagnostics channel).

    Same discipline as ``launcher._pump``: **use ``read1``, not ``read``** —
    ``read(n)`` only returns after reading n bytes or EOF, which blocks forever
    on pipes.
    """
    try:
        while True:
            if hasattr(src, "recv"):
                data = src.recv(CONN_CHUNK)
            elif hasattr(src, "read1"):
                data = src.read1(CONN_CHUNK)
            else:
                data = src.read(CONN_CHUNK)
            if not data:
                break
            if hasattr(dst, "sendall"):
                dst.sendall(data)
            else:
                dst.write(data)
    except Exception:  # noqa: BLE001
        pass


def serve_connection(conn: Connection, handler, on_exit=None) -> None:
    """Hand one connection to the handler (bytes -> bytes).

    A handler exception must not take the daemon down — log one line, then
    close the connection normally.
    ``on_exit``: connection teardown callback (orphan cleanup uses it to
    decrement the active-connection count).
    """
    try:
        handler(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"[daemon] connection {conn.conn_id} handling failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    finally:
        conn.closed = True
        try:
            conn.sock.close()
        except OSError:
            pass
        if on_exit is not None:
            try:
                on_exit(conn)
            except Exception:  # noqa: BLE001 -- a counting failure must not wreck teardown
                pass


class Daemon:
    """TCP daemon: listens on 127.0.0.1, one thread per connection.

    Orphan cleanup: after all launchers die, the daemon must not spin forever
    (it holds ~1GB of model). When ``idle_exit_seconds`` > 0, once the active
    connection count reaches zero and stays there past that many seconds ⇒
    automatically save all brains + clear the registry + exit.
    Any new connection resets the idle timer. ``0``/negative disables it
    (keeps the old behavior; used by tests).
    """

    def __init__(self, registry: BrainRegistry, handler=None,
                 idle_exit_seconds: float = 0.0, key: str = "",
                 heartbeat_seconds: float = HEARTBEAT_INTERVAL) -> None:
        self.registry = registry
        self.handler = handler
        self.key = key
        self.idle_exit_seconds = float(idle_exit_seconds or 0)
        self.heartbeat_seconds = float(heartbeat_seconds or 0)
        # Idle-unload the model (note: **not** idle-exit the process).
        # The difference from idle_exit_seconds must be spelled out: the former
        # merely returns the ~400MB embedding model to the system; the process
        # and memory libraries remain, the next call auto-reloads (measured
        # 2.5–3.7s) with results identical to before unloading (cosine
        # similarity 1.000000); the latter takes the process down too.
        # Default 1800s (30 minutes); 0/negative disables.
        self.model_unload_seconds = float(
            os.environ.get(ENV_MODEL_UNLOAD, str(DEFAULT_MODEL_UNLOAD))
            or 0)
        # O2 sleep window (read the same way as model unload: garbage values
        # blow up immediately, no silent fallback — a mistyped sleep env should
        # surface at startup, not silently do nothing).
        self.sleep_idle_seconds = float(
            os.environ.get(ENV_SLEEP_IDLE, str(DEFAULT_SLEEP_IDLE))
            or 0)
        self.sleep_budget_s = float(
            os.environ.get(ENV_SLEEP_BUDGET, str(DEFAULT_SLEEP_BUDGET))
            or 0)
        self.sock: socket.socket | None = None
        self.port = 0
        self._conn_seq = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._active_conns = 0
        self._last_conn_at = time.time()
        self._stopped = False
        self._heartbeat_round = 0
        self.save_idle_seconds = float(
            os.environ.get(ENV_SAVE_IDLE, str(DEFAULT_SAVE_IDLE)) or 0)
        self._last_flush_at = time.time()
        # Teardown-complete signal + idle-reclaim thread handle (see join_idle:
        # the thread must be **non-daemon**, otherwise teardown gets cut off by
        # interpreter shutdown)
        self._finished = threading.Event()
        self._idle_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._model_unload_thread: threading.Thread | None = None
        self._maintenance_thread: threading.Thread | None = None
        # O2 sleep batch: cancel signal (set as soon as a new connection
        # arrives) + single-flight flag + thread handle
        self._sleep_cancel = threading.Event()
        self._sleep_running = False
        self._sleep_thread: threading.Thread | None = None

    # -- Lifecycle -------------------------------------------------------
    def bind(self) -> int:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        return self.port

    def announce(self, key: str, library: str = "", model: str = "") -> bool:
        """Write "where I am" into the registry so the launcher can find us."""
        self.key = key
        return write_registry(key, {
            "pid": os.getpid(),
            "port": self.port,
            "key": key,
            "library": library,
            "model": model,
            "started_at": time.time(),
            "alive_at": time.time(),
            "runtime_dir": runtime_dir(),
            "identity": runtime_identity(library),
        })

    def _idle_watch(self) -> None:
        """Orphan cleanup sentinel: on idle timeout ⇒ exit voluntarily (saving happens in stop())."""
        interval = min(5.0, max(0.2, self.idle_exit_seconds / 4 or 5.0))
        while not self._stop.wait(interval):
            if self._active_conns > 0:
                continue
            idle = time.time() - self._last_conn_at
            if idle < self.idle_exit_seconds:
                continue
            print(
                f"[daemon] orphan cleanup: no active connections for {idle:.0f}s"
                f" (threshold {self.idle_exit_seconds:.0f}s); saving and exiting",
                file=sys.stderr,
            )
            self.stop()
            return

    # -- Heartbeat --------------------------------------------------------
    def _touch_registry(self) -> bool:
        """Refresh "I'm still alive" into the registry.

        Why this is needed: in real hosts there was a day where **15 daemons ran
        and not one exited cleanly** (0 "orphan cleanup" entries in the logs).
        To tell "it exited voluntarily" from "it was killed", the only way is to
        leave a continuously refreshed timestamp that is **not on the crash
        path**:

          · The last ``alive_at`` ≈ the moment of death (the real daemon's
            dated sepulchre)
          · Cross-checking it against the host's ``mcp.server.closed`` proves
            the reclamation came from the **host process tree**, not our own
            idle sentinel — once that is established, there is grounding for
            adding ``CREATE_BREAKAWAY_FROM_JOB`` to ``spawn_daemon``.

        🔴 **Stop writing once the registry is no longer mine**: that means a
        later daemon overwrote me and I am an orphan (an "invisible orphan");
        continuing to write would scramble the living daemon's data.
        """
        if not self.key:
            return False
        try:
            info = read_registry(self.key)
        except Exception:  # noqa: BLE001
            return False
        if not info or info.get("pid") != os.getpid():
            return False
        info["alive_at"] = time.time()
        info["conns"] = self._active_conns
        try:
            return bool(write_registry(self.key, info))
        except Exception:  # noqa: BLE001
            return False

    def _heartbeat_watch(self) -> None:
        interval = max(1.0, self.heartbeat_seconds)
        while not self._stop.wait(interval):
            try:
                self._touch_registry()
            except Exception:  # noqa: BLE001 -- the heartbeat must never make the main flow fail
                pass
            self._heartbeat_round += 1
            if self._heartbeat_round % HEARTBEAT_LOG_EVERY == 0:
                print(
                    f"[daemon] heartbeat pid={os.getpid()} "
                    f"conns={self._active_conns} t={time.time():.3f}",
                    file=sys.stderr,
                )

    def _maintenance_watch(self) -> None:
        """Housekeeping that must NOT depend on ``_active_conns``.

        The idle-exit sentinel requires ``conns == 0``, which real hosts never
        reach (the launcher holds the socket for the whole session).  Parking
        either the stop marker or the timed flush there would silently disable
        both.  This thread is independent by design.
        """
        while not self._stop.wait(MAINTENANCE_POLL):
            try:
                if self.key and os.path.exists(stop_request_path(self.key)):
                    print("[daemon] cooperative stop requested; saving and exiting",
                          file=sys.stderr, flush=True)
                    self.stop()
                    return
            except Exception as exc:  # noqa: BLE001
                print(f"[daemon] stop marker check failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            if self.save_idle_seconds <= 0:
                continue
            if time.time() - self._last_flush_at < self.save_idle_seconds:
                continue
            self._last_flush_at = time.time()
            self._flush_dirty_brains()

    def _flush_dirty_brains(self) -> None:
        """Persist resident libraries that still hold in-memory changes.

        ``save_ledger_view`` is the single source for the dirty judgement;
        for a brand-new library with no ledger entry yet it reports ``unknown``,
        so fall back to the store's dirty markers in that case.
        """
        for key in self.registry.keys():
            if self._stop.is_set():
                return
            brain = self.registry.get(None if key == DEFAULT_KEY else key)
            if brain is None:
                continue
            try:
                view = brain.save_ledger_view()
            except Exception:  # noqa: BLE001 -- one bad library must not stop others
                continue
            dirty = bool(view.get("l1_dirty")) or bool(view.get("pending_layers"))
            if not dirty and view.get("level") == "unknown":
                store = getattr(getattr(brain, "manager", None), "store", None)
                dirty = bool(getattr(store, "_dirty_ids", None)
                             or getattr(store, "_deleted_ids", None))
            if not dirty:
                continue
            try:
                brain.save()
                print(f"[daemon] idle flush pid={os.getpid()} library={key} "
                      f"interval_s={view.get('interval_s')}",
                      file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[daemon] idle flush failed library={key}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def _model_unload_watch(self) -> None:
        """Idle-unload sentinel: past the threshold with nobody using it ⇒ unload the embedding model.

        Why watch "last model use" instead of "active connection count"
        --------------------------------------------------------------
        The host-side launcher holds the socket until the session ends, and
        zombie launchers never let go ⇒ ``_active_conns`` may stay non-zero
        indefinitely. This is most likely why "the banner said
        ``idle_exit=600s`` yet it never exited once". Using
        :func:`model_lifecycle.idle_seconds` to see real usage is the only way
        not to be fooled by an idle connection.
        """
        while not self._stop.wait(MODEL_UNLOAD_POLL):
            try:
                idle = model_lifecycle.idle_seconds()
                if idle < self.model_unload_seconds:
                    continue
                res = model_lifecycle.release_models(reason="idle-timeout")
                # busy (someone is using it) is not an error: the next round
                # naturally retries. Log only when it **actually unloaded** or
                # something went wrong, to keep the log from filling up with
                # "busy every time".
                if res.get("ok"):
                    print(
                        f"[daemon] idle model unload pid={os.getpid()} "
                        f"idle={idle:.0f}s freed={res.get('reclaimed_mb')}MB "
                        f"({res.get('before_mb')}→{res.get('after_mb')}MB) "
                        f"pools={res.get('pools')} conns={self._active_conns}",
                        file=sys.stderr, flush=True,
                    )
                elif res.get("reason") != "busy":
                    print(f"[daemon] model unload failed: {res}",
                          file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001 -- a sentinel must never crash the daemon
                print(f"[daemon] unload sentinel failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)

    # -- O2 sleep window --------------------------------------------------
    def _sleep_watch(self) -> None:
        """Sleep sentinel: during idle, run one "sleep batch" (offline replay-style
        consolidation) per resident library.

        The idle criterion is the **AND** of two conditions, each irreplaceable:

        - ``_active_conns == 0``: currently serving clients ⇒ don't sleep. But
          it **cannot stand alone** as the criterion — the host-side launcher
          holds the socket until the session ends, and zombie launchers never
          let go (the same pathology as the model-unload sentinel);
        - ``model_lifecycle.idle_seconds() >= sleep_idle_seconds``: the brain
          is genuinely idle. ``idle_seconds()==0`` means "never used" ⇒ don't
          sleep either, which is **deliberate**: a daemon that has served no
          session has no "recent session" to replay — sleep has nothing to
          chew on.

        Interruption contract: when a new connection arrives, ``serve_forever``
        sets ``_sleep_cancel`` immediately; the sleep batch
        (``brain.sleep_batch``) notices at **unit boundaries** such as
        clusters/phases and yields as soon as it can — online requests and
        sleeping live on different threads, requests need not wait for sleep
        to finish, and sleep merely stops promptly to reduce CPU / lock
        contention. Single-flight is guaranteed by ``_sleep_running`` (the
        re-entrancy guard inside the brain's ``sleep_batch`` is the second
        line of defense).
        """
        while not self._stop.wait(SLEEP_POLL):
            if self.sleep_idle_seconds <= 0:
                continue
            if self._active_conns > 0 or self._sleep_running:
                continue
            try:
                idle = model_lifecycle.idle_seconds()
            except Exception:  # noqa: BLE001 -- on timer failure, treat as "don't sleep"
                continue
            if idle < self.sleep_idle_seconds:
                continue
            self._sleep_cancel.clear()
            self._sleep_running = True
            try:
                self._run_sleep_once(idle)
            except Exception as exc:  # noqa: BLE001 -- a sentinel must never crash the daemon
                print(f"[daemon] sleep batch failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            finally:
                self._sleep_running = False

    def _run_sleep_once(self, idle: float) -> None:
        """Run one sleep batch per **resident** library.

        Why iterate only ``registry.keys()`` and use no other entry point:
        constructing a brain takes 25–33s plus model loading, while sleep
        happens precisely after the model has been unloaded — constructing a
        brain for sleep's sake inverts the whole semantics. keys() only
        returns resident libraries; if a concurrent get() between keys() and
        get() swaps that library out, get() would reconstruct it — during
        idle there are no connections and nobody triggers a swap, so that
        window is negligible, and even if hit it is only wasted work (same
        key = same store), not a correctness problem.
        """
        for key in self.registry.keys():
            if self._stop.is_set() or self._sleep_cancel.is_set():
                return
            # 🔴 keys() returns **registry keys** (DEFAULT_KEY or a normalized
            # absolute path), while get() takes **library paths** — DEFAULT_KEY
            # is not a path; feeding it straight to get() would abspath it into
            # "<cwd>/__default__", building two brains for one logical library
            # (caught in practice by the C2e targeted test). Feeding a path key
            # back is an identity (normcase ∘ abspath is idempotent); only the
            # default key needs mapping back to None.
            brain = self.registry.get(None if key == DEFAULT_KEY else key)
            if brain is None or not hasattr(brain, "sleep_batch"):
                continue
            t0 = time.time()
            try:
                report = brain.sleep_batch(
                    cancel=self._sleep_cancel, budget_s=self.sleep_budget_s)
            except Exception as exc:  # noqa: BLE001 -- one library failing must not block other libraries' sleep
                print(f"[daemon] sleep batch failed for library={key}: "
                      f"{type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                continue
            print(
                f"[daemon] sleep batch pid={os.getpid()} library={key} "
                f"idle={idle:.0f}s took={time.time() - t0:.1f}s "
                f"clusters={report.get('replay_clusters')} "
                f"records={report.get('replay_records')} "
                f"strengthened={report.get('strengthened')} "
                f"consolidated={report.get('consolidated')} "
                f"dormant={report.get('dormant_archived')} "
                f"interrupted={report.get('interrupted')}",
                file=sys.stderr, flush=True,
            )

    def serve_forever(self) -> None:
        if self.sock is None:
            raise RuntimeError("bind() must run first")
        if self.model_unload_seconds > 0:
            # daemon=True: this thread holds nothing that needs teardown
            if self._model_unload_thread is None:
                self._model_unload_thread = threading.Thread(
                    target=self._model_unload_watch, name="model-unload",
                    daemon=True)
                self._model_unload_thread.start()
        if self.sleep_idle_seconds > 0:
            # O2 sleep sentinel. daemon=True: holds nothing needing teardown
            # (the sleep batch being cut off by interpreter shutdown is
            # equivalent to being cut off by cancel — every step is idempotent).
            if self._sleep_thread is None:
                self._sleep_thread = threading.Thread(
                    target=self._sleep_watch, name="sleep-watch",
                    daemon=True)
                self._sleep_thread.start()
        if self.idle_exit_seconds > 0:
            # daemon=False: **do not** let interpreter shutdown cut off the
            # thread mid-teardown
            self._idle_thread = threading.Thread(
                target=self._idle_watch, name="idle-watch", daemon=False)
            self._idle_thread.start()
        if self.heartbeat_seconds > 0:
            # daemon=True: the heartbeat holds nothing needing teardown and may
            # be cut off at interpreter shutdown
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_watch, name="heartbeat", daemon=True)
            self._heartbeat_thread.start()
        # Independent of heartbeat_seconds: the stop marker must work even
        # when heartbeats are disabled (tests and embedded deployments).
        if self._maintenance_thread is None:
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_watch, name="maintenance", daemon=True)
            self._maintenance_thread.start()
        while not self._stop.is_set():
            try:
                client, addr = self.sock.accept()
            except OSError:
                break
            if self._stop.is_set():
                # ``stop()``'s self-connection wake-up packet: must not go to
                # the handler, and must not count into ``_active_conns``
                # (otherwise the idle timer would be reset by this fake
                # connection). Close it here and break, so the main thread can
                # reach the finally teardown.
                try:
                    client.close()
                except OSError:
                    pass
                break
            self._conn_seq += 1
            conn = Connection(client, addr, self._conn_seq)
            self._active_conns += 1
            self._last_conn_at = time.time()
            # O2: wake the sleep batch — the new connection is accepted
            # immediately (thread already started), and the sleep batch notices
            # cancel at the next unit boundary and yields, reducing CPU / lock
            # contention with online requests.
            self._sleep_cancel.set()
            t = threading.Thread(
                target=serve_connection,
                args=(conn, self.handler or _default_handler),
                kwargs={"on_exit": self._on_conn_exit},
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def _on_conn_exit(self, conn: Connection) -> None:
        self._active_conns = max(0, self._active_conns - 1)
        self._last_conn_at = time.time()

    def _wake_accept(self) -> None:
        """Wake the main thread blocked in ``accept()`` (cross-platform; see below).

        🔴 On Linux, **close(listening fd) from another thread does not
        interrupt a call already blocked in accept()**: the syscall still holds
        a reference to the file structure, ``close()`` only drops the fd
        reference, the socket keeps LISTENing and accept keeps waiting. Two
        consequences:

        - ``serve_forever`` never returns ⇒ after the idle sentinel calls
          ``stop()`` the process stays alive (``batch2_finalize_daemon``'s
          D1/D2 ``returncode=None`` on CI);
        - the port still accepts connections ⇒ ``f9_4_launcher_daemon``'s L4
          "cannot connect after stop" goes red.

        Windows' ``closesocket`` makes accept raise OSError, so this machine
        never exposed the bug. The only cross-platform reliable way is to
        **connect to yourself once**: accept gets a connection and returns
        naturally, the main loop sees ``_stop`` set and exits; only after the
        fd's last reference is released does the listening port really close.
        """
        sock = self.sock
        if sock is None:
            return
        try:
            port = sock.getsockname()[1]
        except OSError:
            return
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                pass
        except OSError:
            # Port already closed / nobody listening either way: that means
            # accept could return on its own, and the wake-up was a wasted trip.
            pass

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        # O2: halt any running sleep batch (each step idempotent; safe to stop
        # at any layer)
        self._sleep_cancel.set()
        # 🔴 Order: wake first, then close. After close, getsockname() is
        # invalid; and on Linux close alone cannot wake accept at all (see
        # _wake_accept).
        self._wake_accept()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        # Teardown persistence: **must not refuse to exit just because a save
        # failed**, but the failure must be shouted out
        errors = self.registry.save_all()
        if errors:
            print(f"[daemon] save before exit failed: {json.dumps(errors, ensure_ascii=False)}",
                  file=sys.stderr)
        # Clear the registry: this daemon has exited; don't let a launcher hit
        # a dead entry and wait an extra round of probing (the launcher already
        # validates pid+port liveness; this just cleans up the scene)
        if self.key:
            try:
                os.remove(stop_request_path(self.key))
            except OSError:
                pass
            path = registry_path(self.key)
            try:
                os.remove(path)
                print(f"[daemon] registry removed: {path}", file=sys.stderr)
            except OSError as e:
                # If it can't be removed, shout: a leftover entry makes the next
                # launcher wait an extra round of liveness probing, which looks
                # exactly like "the daemon is still alive".
                print(f"[daemon] failed to remove registry: {path} -> {type(e).__name__}: {e}",
                      file=sys.stderr)
        self._finished.set()

    def join_idle(self, timeout: float = 20.0) -> None:
        """Wait for the idle-reclaim thread to finish its teardown.

        🔴 Why this step must exist (a real bug hit in practice):
        the idle thread is a **daemon thread**; it calls ``stop()`` → closes
        the socket → the main thread's ``accept()`` raises OSError →
        ``serve_forever`` returns → the interpreter enters shutdown and
        **directly cuts off the daemon thread still running teardown**.
        Consequences:

        - The log says "orphan cleanup: … save then exit", but ``save_all()``
          and **registry clearing** never complete ⇒ the registry keeps a dead
          entry and the next launcher hits it and waits an extra round of
          liveness probing (indistinguishable from "the daemon is still
          alive");
        - ``test_batch2_finalize_daemon``'s D1 was killed by exactly this (not
          flaky).

        Fix, both together: make the idle thread **non-daemon** (the
        interpreter waits for it before shutting down) + an explicit join here.
        """
        t = self._idle_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=timeout)
        # Fallback: if the thread somehow failed to tear down on its own (e.g.
        # stuck), do it once here — idempotent (when _stopped is already set,
        # stop() returns immediately without saving again).
        if not self._finished.is_set():
            self.stop()


def _default_handler(conn: Connection) -> None:
    """Placeholder handler: echo back whatever is received (handy for local
    loopback verification and for diagnostics).

    Replaced once the server side takes over the real MCP session; this
    deliberately does **not** pretend to handle JSON-RPC.
    """
    _pump(conn.sock, conn.sock)


def _make_handler():
    """Pick a handler. **The default must be a real MCP session**, not an echo pump.

    An echo pump leaves the host "connected yet never receiving a valid
    response", so the tool count silently becomes 0 — much harder to diagnose
    than "cannot connect" (in practice 41% tool loss happened exactly like
    this, with no clue).
    So if the bridge is unavailable, **raise explicitly** and let the launcher
    fall back to the inline server; only an explicit
    ``HIPPOCAMPUS_DAEMON_ECHO=1`` may degrade to echo (for local diagnostics).
    """
    if os.environ.get("HIPPOCAMPUS_DAEMON_ECHO", "").strip() == "1":
        print("[daemon] HIPPOCAMPUS_DAEMON_ECHO=1 -> echo pump (local diagnostics only)",
              file=sys.stderr)
        return _default_handler
    try:
        from .bridge import bridge_available, make_mcp_handler  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"MCP session bridge unavailable (missing anyio / mcp): {type(exc).__name__}: {exc}"
        ) from exc
    if not bridge_available():
        raise RuntimeError("MCP session bridge self-check failed: anyio / mcp unavailable")
    return make_mcp_handler()


def _install_crash_diagnostics() -> str:
    """Leave a trace for **native-level** death (previously the only observability gap).

    Background: real hosts saw daemons **die silently** — no traceback in the
    log, no WER record, the registry not cleaned up (unlike normal orphan
    cleanup), and the process never even got to initialize a brain.
    Python-level exceptions already go to stderr (the launcher persists stderr
    to ``daemon-<key>.log``), so what is added here is the part where **even
    Python never gets a chance to act**: ``SIGSEGV`` / ``SIGABRT`` /
    ``SIGFPE`` / ``SIGILL`` and other fatal signals.

    One diagnostic criterion (kept in this docstring for later; use directly
    when reviewing silent deaths):

    - faulthandler installed yet **not a single word written** ⇒ the process
      was terminated externally (Windows ``TerminateProcess`` triggers no
      handler; Python's ``os.kill(pid, SIGTERM)`` is also TerminateProcess on
      Windows);
    - faulthandler installed and a dump appeared ⇒ a native-level crash; keep
      investigating via the stack.

    Why ``chain=True`` must not become ``False``: ``False`` **no longer calls
    the original handler** after dumping, meaning the process **keeps living**
    after receiving ``SIGTERM`` — the host believes it killed the daemon while
    the daemon still holds ~1GB there. ``chain=True`` is "terminate as usual
    after the dump"; semantics unchanged.

    Escape switch: ``HIPPOCAMPUS_DAEMON_FAULTHANDLER=0`` disables.
    """
    if os.environ.get("HIPPOCAMPUS_DAEMON_FAULTHANDLER", "1").strip() in ("0", "false", "no", "off"):
        return "off"
    try:
        import faulthandler
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:  # noqa: BLE001 -- diagnostics must not let their own failure drag down the daemon
        return "unavailable"
    # enable() itself installs handlers for **fatal signals** like
    # SIGSEGV / SIGFPE / SIGABRT / SIGILL — i.e. the core native-crash tracing
    # capability is already there. register() only adds "dump for arbitrary
    # signals too"; in practice both mainstream interpreters (CPython 3.11 and
    # 3.13) ship a **stripped-down built-in** faulthandler with
    # hasattr(faulthandler, "register") == False, so this path is usually
    # unavailable; record that absence explicitly so a later reviewer doesn't
    # mistake it for registered.
    if not hasattr(faulthandler, "register"):
        return "enable(no-register)"
    installed = ["enable"]
    for signame in ("SIGTERM", "SIGBREAK"):
        signum = getattr(__import__("signal"), signame, None)
        if signum is None:
            continue
        try:
            faulthandler.register(signum, file=sys.stderr, all_threads=True, chain=True)
            installed.append(signame)
        except Exception:  # noqa: BLE001 -- some platforms lack these signals; just ignore
            pass
    return "+".join(installed)


def main(argv: list[str] | None = None) -> int:
    key = os.environ.get("HIPPOCAMPUS_DAEMON_KEY", "") or "default"
    library = os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", "")
    model = os.environ.get("HIPPOCAMPUS_DAEMON_MODEL", "")
    # Orphan cleanup: once all launchers die, exit automatically on idle
    # timeout. Default 600s; <=0 disables (keeps the old "never auto-exit"
    # behavior; used by tests).
    try:
        idle_exit = float(os.environ.get("HIPPOCAMPUS_DAEMON_IDLE_EXIT", "600") or 600)
    except ValueError:
        idle_exit = 600.0
    registry = BrainRegistry(factory=_default_factory)
    d = Daemon(registry, handler=_make_handler(),
               idle_exit_seconds=idle_exit, key=key)
    diag = _install_crash_diagnostics()
    d.bind()
    if not d.announce(key, library=library, model=model):
        print(f"[daemon] failed to write registry: {registry_path(key)}", file=sys.stderr)
        return 1
    print(f"[daemon] key={key} port={d.port} pid={os.getpid()} "
          f"idle_exit={idle_exit:.0f}s "
          f"sleep={'off' if d.sleep_idle_seconds <= 0 else f'{d.sleep_idle_seconds:.0f}s'} "
          f"faulthandler={diag}", file=sys.stderr)
    try:
        d.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        d.join_idle()      # first wait for the idle-reclaim thread's teardown (save + registry clearing)
        d.stop()           # idempotent fallback
    return 0


def _default_factory(library: str | None = None):
    """Construct the brain lazily (the heavy imports happen here)."""
    from .brain import HippoBrain
    return HippoBrain(persist_dir=library or None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
