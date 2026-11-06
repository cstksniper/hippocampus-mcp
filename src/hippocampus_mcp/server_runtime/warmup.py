# -*- coding: utf-8 -*-
"""MCP server warmup section (implementation body of server.py's warmup block).

Implemented here: the background warmup thread body (``_warmup_body`` /
``start_background_warmup`` / ``_warmup_library`` / ``maybe_warmup_library``),
tool-side waiting (``_await_warmup``), the zero-wait readiness probe
(``warmup_status``), the diagnostic watchdog (``_start_warmup_watchdog``),
import-chain preloading (``_preload_import_chain`` / ``_preload_list``),
the ``STORE_ONLY_TOOLS`` tier table, the two warmup exception classes, and
the torch single-thread fallback (``_prepare_torch``).

🔴 State and constants are **deliberately not moved** -- they stay in
server.py because that is the tests' monkeypatch surface (``_WARMUP_MODE`` /
``_warmup_ready`` / ``_warmup_started`` / ``_warmup_error`` /
``_warmup_fatal`` / ``_warmup_phase`` / ``_store_ready`` /
``_WARMUP_WAIT_TIMEOUT`` / ``_WARMUP_WAIT_POLICY`` / ``_PRELOAD*`` /
``_warmup_libs`` / ``_PERSIST_DIR`` ...; see tests/test_warmup.py and
tests/test_i3_warmup_per_library.py). This module reads/writes all of those
symbols via :func:`_srv`, **fetching them live** (lesson: snapshotting them
into a new module makes monkeypatching silently ineffective); calls to
server-side functions (``get_brain`` / ``_registry_get`` /
``default_library_path`` ...) are likewise fetched live -- only that way
does test replacement of ``server.HippoBrain`` / ``server._registry_get``
reach all the way into the warmup thread.

server.py re-exports this module's public symbols verbatim by import name
(``_await_warmup`` / ``warmup_status`` / ``STORE_ONLY_TOOLS`` /
``WarmupPendingError`` / ``WarmupFailedError`` / ``start_background_warmup``
/ ``maybe_warmup_library`` / ``_prepare_torch`` / ``_preload_import_chain``
/ ``_start_warmup_watchdog``); existing call sites (the 46 tool bodies'
literal ``await _await_warmup()``) and test references
(``srv.STORE_ONLY_TOOLS`` / ``isinstance(e, srv.WarmupPendingError)`` ...)
need zero changes. Test replacement of ``srv._await_warmup = <stub>``
works the same way -- the tool call sites share the name with the server
module, so name resolution goes through the server module's namespace.
"""

# env direct-read exemption: this file's env direct reads are third-party
# library thread env and warm early reads (before Config consumes them);
# the closed list is in config.py under 「根层豁免清单」 (root-layer
# exemption list).
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from typing import Any

from hippocampus_mcp import library as _library

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """The server module itself (fetched live; see the module docstring's monkeypatch-surface notes)."""
    from hippocampus_mcp import server as _s

    return _s


class WarmupPendingError(RuntimeError):
    """Warmup did not finish within its budget.

    Raised to the MCP layer it becomes an **explicit tool error** (rather
    than a client timeout): the caller can retry or inform the user; this
    separates "not ready yet" from "broken".
    """


class WarmupFailedError(RuntimeError):
    """An exception was raised during warmup. The message carries the original cause, so a configuration problem is not misread as a performance problem."""


# torch is not imported at the top level (see server.py's header comment),
# but the "single thread" setting must be in effect before torch is truly
# used. This function is the unified fallback: whichever path first
# touches torch (auto's warmup thread / off's first tool call / sync's
# main thread), the setting is already in place.
# This flag is owned by this module (no test patch surface; decoupled from
# the server namespace).
_torch_ready = False


def _prepare_torch() -> None:
    """Import torch for the first time and pin it to a single thread
    (**only when there really is a torch consumer**).

    On Windows, torch/OpenMP inference inside an anyio worker thread
    occasionally hits an access violation; pinning to one thread avoids it
    (embedding throughput is not this service's bottleneck).

    ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` must be set in the
    environment **before** ``import torch``, so this lives in this
    function rather than main().

    The default production configuration (LNN=numpy + embedding auto/onnx)
    runs **zero torch**; we no longer import unconditionally here -- only
    the explicit fallback paths (``HIPPOCAMPUS_LNN_BACKEND=torch`` /
    ``HIPPOCAMPUS_EMB_BACKEND=torch``) pay this memory. The ST fallback
    path (auto without artifacts) has ``embeddings._load_model`` set the
    thread limits when it actually loads ST.
    """
    global _torch_ready
    if _torch_ready:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    needs_torch = (
        (os.environ.get("HIPPOCAMPUS_LNN_BACKEND") or "").strip().lower() == "torch"
        or (os.environ.get("HIPPOCAMPUS_EMB_BACKEND") or "").strip().lower() == "torch"
    )
    if needs_torch:
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:  # noqa: BLE001 an import failure is not raised here; construction later gives a clearer error
            pass
    _torch_ready = True


def _warmup_body() -> None:
    """Background thread body: moves heavy initialization (memory library /
    embedding model / spaCy) out of the handshake path.

    Per-phase timing logs are **operationally essential**: once warmup
    hangs (e.g. model path unreachable, or an attempt to download the
    spaCy model online), the host side sees "the tool was called but never
    responds"; without phase logs you can only guess.
    """
    t_all = time.time()
    s = _srv()

    def _phase(name: str, fn: Any) -> Any:
        t0 = time.time()
        s._warmup_phase = name          # the timeout message must be able to say which step hung
        logger.info("warmup phase start: %s", name)
        try:
            return fn()
        finally:
            logger.info("warmup phase done: %s (%.1fs)", name, time.time() - t0)

    try:
        try:
            brain = _phase("load memory library / construct HippoBrain", s.get_brain)
        except BaseException:
            # The core layer did not come up: no later tool can succeed; mark as fatal.
            s._warmup_fatal = True
            raise
        # Store ready: counting / fetch by id / list sources / stats only
        # need it (about 5 seconds)
        s._store_ready.set()
        # 🔴 **Do not warm up spaCy**. spaCy's thinc.compat does
        # `import torch` (measured: +160MB resident), while the NER path is
        # lazy anyway (`_get_nlp` only truly loads on the first English
        # extraction, falling back to heuristics on failure). Forcing a
        # warmup load = every session process permanently pays ~240MB extra
        # (spaCy 78 + torch 160), pure waste for Chinese libraries. The
        # cost: the first English entity extraction pays an extra ~2-5s
        # load delay.
        emb = brain.manager.embedder
        # One more split: model construction vs actually computing one
        # vector. The two hang for completely different reasons (the former
        # usually model path/missing files or slow DLL loading; the latter
        # usually inference-thread environment issues).
        _phase("construct embedding model", emb._ensure_model)
        _phase("first inference", lambda: emb.embed("warmup"))
        # The backend self-description must be visible **at startup**. The
        # common thread of both accidents -- spaCy failing to install, the
        # embedder silently falling back to MiniLM -- is "the result changed
        # and nobody knew", and the log is the only place the host side
        # will actually look (brain_stats needs a human to call it).
        # ⚠️ **Do not** call stats() here: it walks the whole library
        # computing entity_origins and other self-report items; if it
        # raises, the outer try records it as a warmup failure -- ruining
        # warmup for one log line is not worth it. The ledger cell only
        # reads the last in-memory entry; the cost is negligible.
        try:
            _last_save = (brain.save_ledger_view().get("last_save_at")
                          or "never saved")
        except Exception:  # noqa: BLE001 -- a missing log field does not affect warmup success
            _last_save = "unknown"
        logger.info(
            "Background warmup done: L1=%d entities=%d (total %.1fs)"
            " | NER=%s | embedding backend=%s (%sD, model source=%s)"
            " | ledger: last save=%s",
            brain.manager.store.count(), len(brain.entities), time.time() - t_all,
            brain.ner_backend_lazy,  # side-effect-free version: do not pull spaCy/torch into the process via warmup logging
            brain.manager.embedder_backend,
            getattr(brain.manager, "embedding_dim", "?"),
            getattr(brain.manager.embedder, "model_source", "n/a"),
            _last_save,
        )
    except BaseException as e:  # noqa: BLE001 warmup failure must not take the service down either
        s._warmup_error = f"{type(e).__name__}: {e} (phase: {s._warmup_phase})"
        logger.warning("background warmup failed (first call will initialize): %s", s._warmup_error)
    finally:
        # Release even on failure, otherwise every tool would hang until
        # timeout.
        # _store_ready must also be set: if get_brain() itself raised, it
        # never reached set(); store-tier tools would wait out the full
        # timeout before falling back to lazy initialization (which would
        # fail too, but at least not play dead).
        s._store_ready.set()
        s._warmup_ready.set()


def start_background_warmup() -> threading.Thread | None:
    """Start the background warmup thread (auto mode only). Returns immediately, does not block the handshake."""
    s = _srv()
    if s._WARMUP_MODE != "auto":
        return None
    # main()'s warmup covers the process-level default library; register it
    # in the set so the first connection does not start a second warmup
    # thread for the same library (double model loading).
    with s._warmup_libs_lock:
        s._warmup_libs.add(_library.normpath(s.default_library_path()))
    t = threading.Thread(target=_warmup_body, name="hippo-warmup", daemon=True)
    t.start()
    s._warmup_started = True
    logger.info("warmup moved to background thread (mode=auto; timeout %.0fs)", s._WARMUP_WAIT_TIMEOUT)
    return t


def _warmup_library(path: str) -> None:
    """Warm up a **specified library** in the background (build brain +
    load model + run one inference).

    Difference from main()'s ``_warmup_body`` run: that one can only cover
    the process-level default library; this one covers a **connection
    having declared a different library**. Exceptions are swallowed
    throughout -- warmup failure only means the first call pays the load
    time itself; it must not fail connection setup or any tool call.
    """
    try:
        brain = _srv()._registry_get().get(path)
        emb = brain.manager.embedder
        emb._ensure_model()
        emb.embed("warmup")
        logger.info("library warmup done: %s (L1=%d records)", path,
                    brain.manager.store.count())
    except BaseException as e:  # noqa: BLE001 -- warmup failure must never take the service down
        logger.warning("library warmup failed (first call will initialize): %s -- %s",
                       type(e).__name__, e)


def maybe_warmup_library(path: str) -> None:
    """After a connection declares its own default library, run one
    background warmup for it (deduplicated per library).

    Triggers **only in auto mode** (off / sync are the user's explicit
    choice; do not override them); one thread per library at most;
    failures do not raise.
    """
    s = _srv()
    if s._WARMUP_MODE != "auto" or not path:
        return
    with s._warmup_libs_lock:
        if path in s._warmup_libs:
            return
        s._warmup_libs.add(path)
    threading.Thread(target=_warmup_library, args=(path,),
                     name="hippo-lib-warmup", daemon=True).start()


# Preload tiers (HIPPOCAMPUS_PRELOAD):
#   off / 0   no preload -- **will hang**. Measured: PRELOAD=off still
#             unfinished after 150s.
#   1 (default)  scipy.special + scipy.interpolate (warm 0.9s / cold 6.6s)
#   other comma-separated module names also work.
#
# Why this set **cannot be narrowed further** (measured in a standalone
# process):
#   · torch has been switched to on-demand import (see server.py's header
#     comment) -- importing it after the event loop has started works
#     **fine**; measured: warmup finished at 28.6s with L1=9359, recall
#     hit 3 records. So torch no longer occupies handshake time.
#   · but scipy does not: preloading only scipy.special still hangs (the
#     hang point moves to _fitpack); not preloading at all hangs for
#     sure. So the optimization must be found elsewhere; do not compress
#     this set further.
#   · footnote: preloading only scipy.interpolate also passes (saves ~2s
#     of cold start), but the asymmetry of "special alone hangs,
#     interpolate alone does not" shows the hang point is sensitive to
#     load order; the default tier keeps both for safety.
#
# Measured data (warm state, probe_handshake auto):
#              default tier                preload 0.9s → tools visible 5.6s / first recall 29.2s ✅
#              sklearn                     3.6s → tools 10.3s / recall 22.4s
#              full = sentence_transformers 13.7s → tools 18.7s ⚠️ / recall 9.3s
#              `full` can push the handshake past the registration window;
#              use it only when confirmed locally that the host will
#              register afterwards.
#
# The tier values themselves (``_PRELOAD``) and the budgets
# (``_PRELOAD_BUDGET`` / ``_HANDSHAKE_BASE_COLD``) stay in server.py (no
# patch surface but part of the header config block); this module fetches
# them live via _srv().
# Modules measured to hang on LoadLibrary. Safe to load only **before the
# event loop starts**. The pattern is "loading any scipy C extension after
# the event loop is up hangs", so this set must cover the scipy C
# extensions later imports will touch; scipy.interpolate pulls in
# linalg/sparse/special.
_PRELOAD_DEFAULT = ("scipy.special", "scipy.interpolate")
_PRELOAD_FULL = ("sentence_transformers",)


def _preload_list() -> tuple[str, ...]:
    """Parse ``HIPPOCAMPUS_PRELOAD`` into the tuple of modules to preload."""
    preload = _srv()._PRELOAD
    low = preload.lower()
    if low in ("0", "false", "no", "off", ""):
        return ()
    if low == "full":
        return _PRELOAD_FULL
    if low in ("1", "true", "yes", "on"):
        return _PRELOAD_DEFAULT
    return tuple(m.strip() for m in preload.split(",") if m.strip())


def _preload_import_chain() -> None:
    """Preload the embedding model's import chain before the event loop
    starts (best-effort; failures do not block startup).

    **Why this must happen before the loop starts** (all measured in
    standalone processes):
    the same `import sentence_transformers` (which pulls in torch /
    transformers / sklearn / scipy):
      · done **before** the event loop starts → completes normally, and
        constructing the model afterwards takes only **0.7s**
      · done after `server.run()` is **up** → hangs on the `LoadLibrary`
        of scipy's C extensions (`_ufuncs.pyd`; clear it and
        `_fitpack.pyd` takes its place...); three thread-stack dumps at
        30/60/120s all stopped at the **same line**, never returning
        (not slow, but immobile)
    the same load completes normally in 12-16s in a plain script, so this
    is a process-level phenomenon of this machine's environment.

    Costs and trade-offs: see the ``_PRELOAD`` notes (server.py header
    config block).
    """
    s = _srv()
    mods = _preload_list()
    if not mods:
        return
    t_all = time.time()
    for name in mods:
        t0 = time.time()
        try:
            __import__(name)
            logger.info("import-chain preload: %s (%.2fs)", name, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            logger.warning("import-chain preload failed: %s (%s); startup unaffected", name, e)
    used = time.time() - t_all
    logger.info("import-chain preload total %.2fs (tier=%s)", used, s._PRELOAD)
    if used > s._PRELOAD_BUDGET:
        logger.warning(
            "Preload took %.1fs over its %.1fs budget: the handshake will be delayed to roughly %.1fs, "
            "risking the host tool-registration window (about 15s). To shorten, reduce "
            "the HIPPOCAMPUS_PRELOAD preload set.",
            used, s._PRELOAD_BUDGET, used + s._HANDSHAKE_BASE_COLD,
        )


def _start_warmup_watchdog() -> None:
    """Check at 30/60/120s steps whether warmup has finished; if not, dump
    all thread stacks.

    Independent of the warmup mode: hangs can happen in the background
    thread (auto) or in the main thread (off mode's lazy initialization);
    watching only auto would miss half the scenes.
    """
    s = _srv()

    def _dump() -> None:
        import traceback

        deadline = 30.0
        while not s._warmup_ready.is_set():
            if not s._warmup_ready.wait(min(deadline, s._WARMUP_WAIT_TIMEOUT)):
                if s._warmup_ready.is_set():
                    break
                frames = sys._current_frames()
                for tid, frame in frames.items():
                    tname = next(
                        (x.name for x in threading.enumerate() if x.ident == tid),
                        str(tid),
                    )
                    logger.warning(
                        "DIAG[%.0fs] warmup not finished; thread %s stack:\n%s",
                        deadline, tname,
                        "".join(traceback.format_stack(frame)[-14:]),
                    )
                deadline = min(deadline * 2, s._WARMUP_WAIT_TIMEOUT)
                if deadline >= s._WARMUP_WAIT_TIMEOUT:
                    break

    threading.Thread(target=_dump, name="hippo-diag", daemon=True).start()


# Tools that only need 「reading the store」 (no embedding model): waiting
# for _store_ready is enough for them to work.
# Measured: loading a 9359-record store takes about 5 seconds, while the
# BGE model takes 25-100s to load -- making these tools wait for the
# model too drags second-scale operations into minute-scale (and the
# first call is the one most easily judged a timeout by the host).
# Tools not listed are treated as embed-tier (conservative = safe;
# affects latency only, not correctness).
STORE_ONLY_TOOLS = frozenset({
    "get_memory", "forget_memory",
    "list_memories", "list_sources", "get_stats",
    "save_state", "load_state", "save_brain", "reset_context",
    "complete_intent", "report_feedback",
    "entity_profile", "entity_list", "lexicon_suggest",
    "lexicon_alias_suggest", "contradictions",
    "brain_stats",
    # topic-layer read-only derived tables (when empty they auto-derive
    # once, using the already-loaded vectors; no need to wait for the
    # embedding model)
    "knowledge_map",
    # source-drift audit reads only the store + source signature layer
    # (file mtime/sha1); it never touches vectors ⇒ store tier suffices;
    # no need to wait 25-33 seconds of model warmup for it.
    "drift_audit",
    # the four map-layer tools only read/write the maps.json sidecar +
    # source-of-truth files; no store, no vectors ⇒ store tier (real
    # latency is only spent waiting for the store to load).
    "map_set", "map_get", "map_list", "map_verify",
    # ⚠️ knowledge_topic is **deliberately not in this table**: the topic
    # ids it queries depend on the topic layer having been built (the
    # first call on an empty library does one refresh); letting it through
    # at store tier would present as 「没有这个主题」 (no such topic) --
    # far harder to debug than waiting a few dozen seconds. Writing the
    # derivation itself does not use vectors, but to avoid that silent
    # failure it is better to wait conservatively at embed tier.
    # library-management tools only need directory metadata, not the
    # embedding model (which takes 25-33 seconds)
    "library_list", "library_info", "library_whoami", "library_use",
})


async def _await_warmup(tier: str | None = None) -> None:
    """The first statement of every MCP tool entry: wait for the backend to
    be ready to the required tier, then proceed.

    Args:
        tier: ``"store"`` only uses the store (read/list/stats/delete by
              id); can start within seconds.
              ``"embed"`` also needs the embedding model (retrieval /
              writes / consolidation); must wait for the model.
              **Leave empty to auto-detect from the caller** (see below);
              tools just write `await _await_warmup()`, and new tools need
              not remember which tier to pick.

    Auto-detection looks the calling function's name up in
    ``STORE_ONLY_TOOLS``: the tool itself decides whether it is
    vector-related; call sites need not repeat it. Why not pass the tier
    explicitly at every call site: 30 call sites is too large a change
    surface, and external acceptance scripts count coverage by the
    literal `await _await_warmup()` -- keeping the zero-argument form
    keeps both sides consistent.

    - mode=off / background warmup never started (e.g. direct import from
      a script) → return immediately;
    - target tier already ready → return immediately;
    - not ready → wait on a thread pool (**does not occupy the event
      loop**; other MCP messages proceed normally).
      The wait has a **budget** (``_WARMUP_WAIT_TIMEOUT``, default 90s,
      below the host's 120s timeout); over budget, ``_WARMUP_WAIT_POLICY``
      decides: raise (default) raises :class:`WarmupPendingError`;
      release lets it through to lazy initialization.

    Why the timeout default is **raise rather than release**: releasing
    lets the caller keep running with "backend still not ready" (most
    paths then touch the embedding model right away) and wait tens of
    seconds more -- the host times out first, and the user sees "call
    timed out" instead of "still warming up". Raising is a **parseable,
    retryable** state. Also, when warmup **failed** (as opposed to being
    slow), :class:`WarmupFailedError` is raised immediately with the
    original cause, so "misconfigured" cannot masquerade as a
    "performance problem".

    State and mode constants are fetched live via :func:`_srv` (the
    server namespace's monkeypatch surface).
    """
    if tier is None:
        try:
            caller = sys._getframe(1).f_code.co_name
        except ValueError:  # pragma: no cover
            caller = ""
        tier = "store" if caller in STORE_ONLY_TOOLS else "embed"
    s = _srv()
    event = s._store_ready if tier == "store" else s._warmup_ready
    if s._WARMUP_MODE != "auto" or not s._warmup_started:
        return
    if event.is_set():
        # Even when ready, check whether this is "released after failure" --
        # a fatal failure must be called out on the spot, otherwise the
        # caller proceeds and hits a more obscure error next.
        if s._warmup_fatal and s._warmup_error:
            raise WarmupFailedError(
                f"Backend initialization failed: {s._warmup_error}. This is not a timeout issue; check the memory library"
                f" directory, model path, and dependencies (HIPPOCAMPUS_WARMUP_DIAG=1 captures thread stacks)."
            )
        return
    started = time.time()
    logger.info("background warmup in progress (tier=%s); this call waits for completion...", tier)
    await asyncio.to_thread(event.wait, s._WARMUP_WAIT_TIMEOUT)
    if event.is_set():
        return
    waited = time.time() - started
    if s._warmup_error:
        raise WarmupFailedError(
            f"Backend warmup failed: {s._warmup_error}. This is not a timeout issue; check the model path/"
            f"dependencies (HIPPOCAMPUS_WARMUP_DIAG=1 captures phase timing and thread stacks)."
        )
    msg = (
        f"Backend is still warming up (waited {waited:.0f}s, phase: {s._warmup_phase or 'starting'}); "
        f"this call was not executed; retry shortly. This is not a data or parameter problem. "
        f"If it persists, restart with HIPPOCAMPUS_WARMUP_DIAG=1 to capture the state."
    )
    if s._WARMUP_WAIT_POLICY == "raise":
        logger.warning("background warmup exceeded budget (tier=%s, %.0fs): %s", tier, waited, msg)
        raise WarmupPendingError(msg)
    logger.warning(
        "background warmup timed out (tier=%s, %.0fs); policy=release deferred to lazy initialization",
        tier, waited,
    )


def warmup_status() -> dict[str, Any]:
    """Backend readiness self-description (**zero wait**, returns
    immediately).

    Reason to exist: during warmup, "slow" and "broken" look identical to
    the caller (both are no response), so we need a ready probe that can
    be **queried proactively** -- asking first and then deciding whether
    to call beats repeated trial-and-error calls. (Lazy-loading services
    exposing a ready endpoint is standard practice; here it hangs off the
    ``brain_stats`` return instead of adding a tool.)

    Threads within one process share the same state, so whatever is seen
    during any tool call is the same fact: ``ready=false`` with a
    non-empty ``phase`` = warming up; non-empty ``error`` = warmup
    already failed (vector-requiring tools raise immediately then,
    without waiting out the budget).

    All fields are fetched live via :func:`_srv` -- test assignments to
    the server namespace (``srv._warmup_error = ...`` etc.) are visible
    to this probe immediately.
    """
    s = _srv()
    return {
        "mode": s._WARMUP_MODE,
        "ready": s._warmup_ready.is_set(),
        "store_ready": s._store_ready.is_set(),
        "started": s._warmup_started,
        "phase": s._warmup_phase,
        "error": s._warmup_error,
        # fatal=True → the core layer did not come up; vector-requiring
        # tools error on the spot instead of being let through;
        # fatal=False with a non-empty error → degraded but usable
        # (e.g. only spaCy missing)
        "fatal": s._warmup_fatal,
        "wait_timeout": s._WARMUP_WAIT_TIMEOUT,
        "wait_policy": s._WARMUP_WAIT_POLICY,
        "persist_dir": s._PERSIST_DIR,
    }
