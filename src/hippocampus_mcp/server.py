# -*- coding: utf-8 -*-
"""MCP server entry point.

Starts a stdio server conforming to the Model Context Protocol,
providing memory augmentation for AI agents.

Exposed MCP tools:
    - store_memory:    store a new memory (text → embedding → LNN → vector store)
    - recall:          recall relevant memories (LNN query + scoring + ranking)
    - get_memory:      fetch a single memory by ID
    - update_memory:   update memory fields
    - list_memories:   list all memories
    - reset_context:   reset the LNN context (new conversation)
    - get_stats:       get system status
    - save_state:      persist all state
    - load_state:      restore state from disk

How to start:
    python -m hippocampus_mcp.server

    or specify it in the MCP client configuration:
    {
        "command": "python",
        "args": ["-m", "hippocampus_mcp.server"],
        "env": {"HIPPOCAMPUS_PERSIST_DIR": "/path/to/data"}
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

# Library discovery layer and multi-library registry (**both pure stdlib**,
# torch-free, so top-level imports are safe and stay off the MCP handshake path).
from hippocampus_mcp import library as _library
from hippocampus_mcp.library import LibraryResolutionError  # noqa: F401 (thrown by tool signatures)
from hippocampus_mcp.library.registry import BrainRegistry, DEFAULT_KEY
# Read/write primitives for the task-directory sidecar (library layer, pure stdlib).
# Placed here rather than the composition root: the root has a "top-level import
# whitelist" (logging/os/typing/application only), and the registration point is just
# invoked via _LIB_PORTS's live reference ``_task_directory_register`` — same technique
# as the home-library write-boundary predicate.
from hippocampus_mcp.library import task_directory as _task_directory
# session_key moved with the multi-library addressing area to server_runtime/routing.py
# (the session_ctx it uses moved along), making the alias here a dead import; removed —
# f9_4e/f9_5a now always do ``from hippocampus_mcp.common import session_ctx`` directly
# (no ``S._session_ctx`` references remain repo-wide, no patch-surface loss).
from .config import load_config
# Gating/dispatch/header/entry live in profiles.py / dispatch.py /
# stdout_guard.py / entry.py. State stays in this module's namespace via
# re-exports (patch surface unchanged); all four modules fetch live through
# _srv() — _dispatch_map fetches tool functions via vars(server)
# (snapshot stubs would swallow newly registered tools; target test N1 pins this);
# _gated_tool's _original_tool is assigned before the server.tool replacement and
# fetched at call time; main keeps its shell and calls entry.run_main()
# (zero import changes in cli/launcher).
from .server_runtime import entry as _entry
from .server_runtime import profiles as _profiles  # noqa: F401 (factory host)
from .server_runtime import stdout_guard as _sg
from .server_runtime.dispatch import (_dispatch_map,  # noqa: F401 (re-export)
                                      _hippo_extra, _hippo_help)
from .server_runtime.profiles import (_COMMON_TOOLS,  # noqa: F401 (re-export)
                                      _COMMON_TOOLS_DEFAULT,
                                      _PROFILE_GROUPS, _TOOL_GROUPS,
                                      _TOOL_PROFILE, make_gated_tool)
from .server_runtime.profiles import _tool_enabled  # noqa: F401 (re-export)
from .server_runtime.stdout_guard import (_ThreadAwareStdout,  # noqa: F401 (re-export)
                                          configure_logging, ensure_paths)

# The sys.path fallback (libs append + src insert) lives in
# server_runtime/stdout_guard.py; this file's path is passed in (the old ``__file__`` semantics).
_sg.ensure_paths(__file__)

try:  # mcp >= 1.0 (new SDK)
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:  # pragma: no cover - legacy SDK
    try:
        from mcp.server import MCPServer  # type: ignore
    except ImportError:
        from mcp.server.lowlevel import Server as MCPServer  # type: ignore

# Heavy dependencies (``brain`` → ``manager`` → ``torch``, 5.7s even warm on this
# machine) are **deliberately not imported here**: every extra second at the top
# level delays the MCP ``initialize`` response by one second, eating directly into
# the host's tool-registration window (about 15s). Instead they are imported on
# demand inside ``get_brain()`` — in auto mode the first import happens on the
# warmup thread (event loop already running); in practice that path does not
# trigger the scipy-style LoadLibrary hang (see ``_preload_import_chain``'s notes
# and the diag script for comparison).
if TYPE_CHECKING:  # static checks only, never executed (annotations are string-ified by the future import)
    from hippocampus_mcp.brain import HippoBrain
    from hippocampus_mcp.engine.manager import MemoryManager

from hippocampus_mcp.engine.embeddings import ENV_MODEL  # noqa: F401 (_EMB_MODEL goes through config; kept as re-export to avoid breakage outside the package)
# Outside the package, take ConcurrentWriteError from ``hippocampus_mcp.store``;
# do not import it from the server symbol.

# Parameter parsing helpers and key validation live in common/params.py and
# application/tags.py — the tool layer and the use-case layer share one
# implementation; **alias imports** here keep the call-site names unchanged;
# the service layer never imports this module (dependency-direction discipline).
# update_tags's four-segment parsing goes through parse_tagops_args (strict
# per-segment parsing, stop at the first error, message text byte-identical);
# session_start still uses ``parse_tags_json``. Strict and lenient versions each
# keep their own name to avoid shadowing each other.
from .application import (BootstrapService, LibraryAdminService,
                          LibraryPorts, MaintainService, OrganizeService,
                          RecallService, WriteService)
from .common.params import parse_str_list_lenient as _parse_str_list
from .common.params import parse_tagops_args as _parse_tagops_args
from .common.params import parse_tags_json as _parse_tags_json
# The warmup area (background thread / tool-side waiting / readiness probes /
# watchdog / preload / tier table / exception classes / torch fallback) moved
# wholesale to server_runtime/warmup.py. **State and mode constants are
# deliberately kept in this module** (the monkeypatch surface tests target on
# this namespace: _WARMUP_MODE / _warmup_ready / _warmup_started / _warmup_error /
# _warmup_fatal / _warmup_phase / _store_ready / _WARMUP_WAIT_* / _warmup_libs …
# all take effect via warmup._srv() live fetches, see that module's docstring).
# Re-exported here under the import names — the 46 tool bodies' await
# _await_warmup() literals and existing references like srv.STORE_ONLY_TOOLS /
# srv.WarmupPendingError change zero.
from .server_runtime.warmup import (STORE_ONLY_TOOLS,  # noqa: F401 (re-export)
                                     WarmupFailedError, WarmupPendingError,
                                     _await_warmup, _preload_import_chain,
                                     _prepare_torch, _start_warmup_watchdog,
                                     maybe_warmup_library,
                                     start_background_warmup, warmup_status)

# Multi-library addressing (guide file / default-library resolution / session
# key / library listing / three-tier precedence / sticky binding / whoami
# self-report / envelope assembly / port I/O helpers) and the composition root
# (_LIB_PORTS + 6 use-case factories + _build_brain) moved to
# server_runtime/routing.py + composition.py. **State and the patch surface are
# deliberately kept in this module** (_DEFAULT_LIB_PATH / _conn_* /
# _library_cache* / the _registry trio / _brain_for / get_brain / _brain /
# _PERSIST_DIR / config constants); both new modules fetch live via _srv() (see
# their docstrings). Re-exported here under the import names — tool-body call
# sites, m4_batch1/4 hasattr checks, and f10/f9_4e/workspace_guide direct calls
# change zero.
from .server_runtime.routing import (GUIDE_FILENAME,  # noqa: F401 (re-export)
                                     _ensure_default_dir,
                                     _find_workspace_guide, _layer_files,
                                     _library_binding,
                                     _library_binding_lock, _library_self,
                                     _prefetch_library,
                                     _resolve_default_lib_path, _with_library,
                                     clear_library_binding,
                                     resolve_default_lib_ex,
                                     get_library_binding, libraries,
                                     resolve_library_path, session_key,
                                     set_library_binding)
from .server_runtime.composition import (_LIB_PORTS,  # noqa: F401 (re-export)
                                         _bootstrap_service, _build_brain,
                                         _maintain_service, _organize_service,
                                         _recall_service, _write_service)
# The **pure predicates** of the home-library write boundary (write-class tool
# list / normalized comparison / three-valued permission / rejection text) live
# in domain/write_boundary.py — placed there so they can be unit-tested with zero
# I/O and boundary tweaks never touch the registry or disk.
# Why not import it straight into composition.py: the composition root has the
# S4.2 "no cross-layer top-level imports" assertion (whitelist is only
# logging/os/typing/application); adding it would trip that pin. **This module
# imports first, the composition root fetches via _srv()** — same technique as
# ``s._library`` from the U2 merge, so the monkeypatch surface penetrates all the
# way down to the guard.
from .domain.write_boundary import (CrossLibraryWriteDenied,  # noqa: F401 (re-export)
                                    READ_ONLY, USER_LIBRARY_NAME,
                                    WRITE_CLASS_TOOLS, boundary_error_message,
                                    caller_tool_name, classify_write_permission,
                                    library_display, same_library)
# Same technique for D2a write-side quarantine pure predicates: import the domain
# truth into this module first, composition root fetches via _srv() (the S4.2
# whitelist takes no domain imports; monkeypatch penetration kept).
from .domain.write_quarantine import should_quarantine  # noqa: F401 (re-export)
from .common.session_task import bind_task, clear_task, current_task
from .common.write_origin import build_origin
# Pure predicates for the C1 workspace-binding reverse lookup
# (domain/task_directory.resolve_workspace)
# + the D2b read-side tier decision (resolve_user_access).
from .domain.task_directory import (normalize_entries, resolve_user_access,
                                    resolve_workspace)
# B1 libraries= syntax parsing (composition root fetches via _srv(), same technique as the S4.2 whitelist).
from .domain.scope_recall import parse_library_spec  # noqa: F401 (re-export)
# Note: _library_admin_service stays in this module (see the note in its docstring);
# it is not among the composition re-export names.

# ------------------------------------------------------------------
# Logging configuration (handler/level assembly in stdout_guard.configure_logging,
# byte-identical; the _ThreadAwareStdout class moved too, re-exported by import name)
# ------------------------------------------------------------------
logger = _sg.configure_logging()


# ------------------------------------------------------------------
# Global configuration (the sole memory manager is lazily created in get_brain() below)
# ------------------------------------------------------------------
_PERSIST_DIR = load_config().persist_dir

# Benchmark-optimal configuration (LoCoMo tuning result)
# Note: embedding dim and neuron count are decided by the embedding model /
# defaults (HIPPOCAMPUS_EMB_DIM / HIPPOCAMPUS_N_NEURONS no longer take effect;
# kept only for backward compatibility with old config files).
_LAMBDA_DECAY = load_config().lambda_decay  # 90 days
_MU_IMPORTANCE = load_config().mu_importance  # 180 days
_W_SIM = load_config().w_sim
_W_TIME = load_config().w_time
_W_RECALL = load_config().w_recall
_CRL_MODEL_PATH = load_config().crl_model
_EMB_MODEL = load_config().emb_model  # e.g. BAAI/bge-base-zh-v1.5 (empty = follow the library's own model)
# Auto-persist to disk every 10 writes (coverage report, addendum 4).
# **On by default in the MCP scenario**: the host process may be recycled at any
# time (measured: ingest 2 items → process exit → on-disk count regresses), while
# writes are sparse (conversation-level); amortized the cost is about +0.15s per
# write (on large libraries a full save is still seconds-level and does not block
# the session).
# The library layer (HippoBrain/MemoryManager) defaults to False — benchmarks and
# bulk imports should not be interrupted by disk writes; ``ingest_batch`` also
# temporarily turns it off within a batch.
_AUTO_SAVE = load_config().auto_save
# Auto-backfill missing vectors at startup. When stored vectors do not match the
# current model (e.g. first start after a model swap), backfill takes minutes and
# blocks synchronously, so the MCP client would wait until timeout — for large
# libraries run scripts/reembed.py offline first, then start with HIPPOCAMPUS_AUTO_BACKFILL=0.
_AUTO_BACKFILL = load_config().auto_backfill
# Recall discount for navigational content (INDEX / TOC / table headers); 1.0 = no demotion (addendum eight).
# Only after the import side tags those memories with the ``_nav`` tag via
# ``import_md.py --nav-files`` is there anything to demote — both ends must be on;
# missing either end silently disables it (the importer prints the value it uses).
_NAV_PENALTY = load_config().nav_penalty
# Fact-update (supersede) detection switch (turn off for bulk document imports)
_SUPERSEDE = load_config().supersede

# ------------------------------------------------------------------
# Warmup policy (addendum 10: the handshake must not be blocked by warmup)
# ------------------------------------------------------------------
# Background: warmup = load all library memories + embedding model + spaCy,
# measured 30-36 seconds. It used to run synchronously **before** ``server.run()``,
# which blocked the MCP handshake too — the host finishes tool registration about
# 15 seconds after session start, so this service's tools never got registered at
# all, showing up as "connection succeeds, but the session has no tools"
# (observed on ZCode).
#
# Three modes (env HIPPOCAMPUS_WARMUP):
#   auto (default)  start a background warmup thread, answer the handshake
#                   immediately; if the first tool call arrives before warmup
#                   finishes, wait on a worker thread (does not occupy the event
#                   loop, does not block other MCP messages).
#                   ➜ the host can register the tools; worst case the wait moves
#                   to the first call, and only once.
#   off             no warmup at all, handshake answers immediately; lazy init on
#                   first tool call (about 25-30 seconds).
#   sync            old behavior again: handshake only after warmup. For offline
#                   scripts / benchmarks only, **never give this to a host**.
#
# Note: under ``auto``, tool calls must still guarantee "execute only after warmup
# completes", otherwise the first call would construct HippoBrain concurrently
# with the warmup thread — see the singleton lock in ``get_brain``.
_WARMUP_MODE = load_config().warmup.strip().lower()
if _WARMUP_MODE not in ("auto", "off", "sync"):
    logger.warning("Unknown HIPPOCAMPUS_WARMUP=%r; falling back to auto", _WARMUP_MODE)
    _WARMUP_MODE = "auto"
# Upper bound for how long a tool call waits on background warmup.
#
# **This value must be smaller than the host's call timeout** (ZCode's is 120s),
# otherwise "waiting" and "hung" look identical on the host side: the caller times
# out first and gets a timeout error instead of "the backend is still warming up".
# The old 300s default was the direct cause of "concurrent first calls during cold
# start hitting the 120s timeout (retry works)".
_WARMUP_WAIT_TIMEOUT = load_config().warmup_timeout

# What to do once the budget is exceeded:
#   raise (default)  raise WarmupPendingError. The host gets a **parseable,
#                    explicit status** (how long it waited, which phase it is stuck
#                    on, that it may retry) instead of a timeout. Concurrent first
#                    calls also dedupe naturally: every waiter waits on the same
#                    Event, so no double loading (thundering herd).
#   release          old behavior: let it through, lazy init takes over (slow, but
#                    "it will definitely run"). Only use when you know model loading
#                    really is that slow and the host timeout is long enough.
_WARMUP_WAIT_POLICY = load_config().warmup_wait.strip().lower()
if _WARMUP_WAIT_POLICY not in ("raise", "release"):
    logger.warning("Unknown HIPPOCAMPUS_WARMUP_WAIT=%r; falling back to raise", _WARMUP_WAIT_POLICY)
    _WARMUP_WAIT_POLICY = "raise"

# Process start timestamp: self-reports the **real** elapsed time from spawn to
# just before server.run(). To tell "slow on our side" from "host aggregation
# timing", an estimate via _HANDSHAKE_BASE_COLD does not count as evidence.
_PROCESS_T0 = time.time()

# The window (seconds) the host allots to "handshake + tool registration"; ~15s
# measured on ZCode. If the time before server.run() exceeds it, this service's
# tools may be missed by registration entirely.
_HANDSHAKE_WINDOW = load_config().handshake_window

_warmup_ready = threading.Event()   # fully ready (embedding model included)
_store_ready = threading.Event()    # library store ready (no embedding model; usually 20-60s earlier)
_warmup_started = False  # tools only need to wait if main() actually started background warmup
# The **reason** warmup failed (empty = no failure). A failure must be explainable
# immediately rather than letting the caller wait out the full timeout only to hear
# "timed out" — that would disguise a "configuration problem" as a "performance problem".
_warmup_error: str = ""
_warmup_phase: str = ""            # name of the current/last phase entered (included in timeout messages)
# Whether the failure is **fatal** (the core layer never came up). Telling fatal
# from non-fatal matters:
#   · fatal (constructing HippoBrain itself fails) → no later tool can ever
#     succeed, must error out on the spot;
#   · non-fatal (spaCy load failed / first inference failed) → the core still
#     works, should be let through, keeping only the reason in the readiness
#     probe. Blanket errors would misread "degraded but usable" as "service down".
_warmup_fatal = False
# Diagnostic switch: when warmup stubbornly does not finish, dump every thread's
# Python stack at 30s/60s/120s … intervals.
# Background: model construction on a **background thread** measured an order of
# magnitude slower than on the main thread (29s → >250s), and "stuck" vs "slow"
# look identical from the host (tool calls unresponsive) — without stacks you can
# only guess. Off by default; set HIPPOCAMPUS_WARMUP_DIAG=1 and restart to capture
# the scene when trouble hits.
_WARMUP_DIAG = load_config().warmup_diag

# The warmup exception classes (WarmupPendingError / WarmupFailedError), the torch
# single-thread fallback (_prepare_torch), and all warmup logic live in
# server_runtime/warmup.py (re-exported via the top imports, namespace unchanged).
# The old "singleton construction lock _INIT_LOCK" retired with the multi-library
# registry (per-library locks); no references remain repo-wide and it was removed.

# Deliberately kept: _brain_class is the class-resolution point of ``_build_brain``
# (``globals().get("HippoBrain")`` semantics; tests rely on it for the
# ``server.HippoBrain = Fake`` replacement); get_manager is the direct-call
# surface of test_warmup / test_recall_consistency / f10;
# _PRELOAD / _PRELOAD_BUDGET / _HANDSHAKE_BASE_COLD are fetched live by warmup.py
# via _srv() (tier semantics and measured numbers in warmup.py comments).


def _brain_class() -> Any:
    """Resolve the ``HippoBrain`` class: **module attribute first**, real import second.

    Why attribute first: this module no longer imports ``brain`` at the top level
    (that would push torch's ~5.7s onto the MCP handshake path). But tests and
    diagnostic scripts need to swap the implementation (``server.HippoBrain = Fake``;
    see the concurrent-singleton case in ``scripts/test_warmup.py``), so the
    "module attribute first" semantics are kept. The real import happens only when
    no replacement is present (afterwards ``sys.modules`` caches it; repeat calls
    cost almost nothing).
    """
    cls = globals().get("HippoBrain")
    if cls is not None:
        return cls
    from hippocampus_mcp.brain import HippoBrain as _cls

    return _cls


def get_manager() -> MemoryManager:
    """Get the global MemoryManager singleton.

    **Since v0.4 the v0.3 use layer shares the same instance** (i.e.
    ``get_brain().manager``). Previously this function and ``get_brain()`` each
    ``new``ed a MemoryManager that both pointed at the same ``memories.json``:
    memories written by ingest were invisible to recall, and a ``save`` on either
    side wholesale-overwrote the other (measured: records deleting each other,
    data loss). Now the whole process has one in-memory state and one store.

    Configuration comes from environment variables uniformly (see ``get_brain``);
    the old path no longer holds its own parameters.
    """
    return _brain_for(None).manager


# The preload tier value itself (``HIPPOCAMPUS_PRELOAD`` parse result) and the
# budget: no patch surface of their own but part of the header config area;
# warmup.py's _preload_list / _preload_import_chain fetch them live via _srv()
# (tier semantics, why they cannot shrink further, and measured numbers in
# warmup.py comments).
_PRELOAD = load_config().preload.strip()

# In-repo default upper bound (seconds) for acceptable preload time: beyond it the
# handshake may miss the host's registration window.
# Used to log an explicit warning instead of making ops compute it themselves.
_PRELOAD_BUDGET = load_config().preload_budget

# The fixed overhead of "spawn → preload start" (interpreter boot + top-level
# imports); ~5.7s measured on cold start (import mcp is 1.6s of that). torch is
# not a top-level import (loaded in the warmup area), otherwise this would be 11.4s.
# Only used to convert "preload time" into "total handshake time" in warnings so
# ops can judge whether the window is exceeded.
_HANDSHAKE_BASE_COLD = 5.7


# ------------------------------------------------------------------
# Create the MCP server and register tools
# ------------------------------------------------------------------
#: One-shot global instructions (resident for the session; **do not** stuff
#: content that would need repeating on every call into it).
#: Why it lives here: every `session_start` response pays tokens, whereas notes
#: like "how to recover after compaction" are **one-time** for the host — moving
#: them into instructions pays once. The response's
#: `receipt.resume_hint` keeps just a pointer to this tool (keys unchanged,
#: shape probes stay green).
_INSTRUCTIONS = (
    "This service is a memory layer for AI agents. "
    "Write with ingest, read with get_memory_context, raw retrieval with recall, "
    "open sessions with session_start and close them with session_end. "
    "If tools such as remember_intent / consolidate are not in this session's tool list, "
    "call hippo_help for the catalog, then invoke via hippo_extra(action=\"name\", args=\"{...}\") -- no capability is lost. "
    "If the host context was compacted: do not copy memory content; keep only the identity fields from the session_start "
    "receipt (session_id / assembled_at / blocks / memory_active / "
    "knowledge_epoch) and call session_start once more with session_id to rebuild the opening pack;"
    "If memory_active or knowledge_epoch differs from before going offline, the library was changed by another session; trust this delivery."
)


def _create_server() -> Any:
    """Absorb constructor-parameter differences across mcp SDK versions."""
    try:
        return MCPServer(
            name="hippocampus-mcp",
            version="0.1.0",
            description="Agent memory layer: four-layer memory + entity index + session lifecycle",
            instructions=_INSTRUCTIONS,
        )
    except TypeError:
        # FastMCP (mcp >= 1.0): accepts only name/instructions
        return MCPServer(name="hippocampus-mcp", instructions=_INSTRUCTIONS)


server = _create_server()


# ------------------------------------------------------------------
# Tool profile gating (state/decisions/wrapping in profiles.py, fetched live via _srv())
# ------------------------------------------------------------------
# Profile semantics (plan D), the _TOOL_GROUPS group table, the common set and
# HIPPOCAMPUS_COMMON_TOOLS override, the _tool_enabled decision, and the
# _gated_tool wrapper factory are fully documented in
# server_runtime/profiles.py; state stays in this module's namespace via
# re-exports (import block above), so the tests' patch surface on this namespace
# is unchanged. _original_tool must be assigned before the server.tool
# replacement (_gated_tool fetches it at call time).
_original_tool = server.tool
_gated_tool = make_gated_tool()
server.tool = _gated_tool


@server.tool(structured_output=False)
async def store_memory(
    content: str,
    importance: float = 0.5,
    memory_type: str = "general", library: str = "",
    idem_key: str = "",
) -> str:
    """Bare single-fact write WITHOUT provenance; prefer ingest (L0 + entities + dedup). Args: content, importance 0-1, memory_type, idem_key."""
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.store_memory(
        content=content, importance=importance, memory_type=memory_type,
        # For wiring see WriteService.store_memory: client decides which lane the
        # LNN hidden state lands on (with a daemon, N connections cannot all crowd
        # onto __default__).
        client=session_key(),
        # P0-1 write origin: who wrote from which write surface (written only,
        # never filtered; excluded from retrieval).
        origin=build_origin(session=session_key(), task=_origin_task(), surface="store_memory"),
        idem_key=idem_key), ensure_ascii=False)


@server.tool(structured_output=False)
async def recall(
    query: str = "",
    top_k: int = 5,
    use_lnn: bool = True,
    sample: bool = False,
    filter_tags: str = "", library: str = "",
    scope: str = "project", libraries: str = "",
    occurred_after: str = "", occurred_before: str = "",
) -> str:
    """Hybrid retrieval: vector + BM25 (RRF fusion) + MMR + entity bonus, hard relevance floor; shared scoring pipeline (R1). HIPPOCAMPUS_RECALL_FUSION=0 restores the legacy additive path.
    
    Call BEFORE asserting anything about the past; no hit is not proof of absence.
    
    Args:
        query: natural sentence (Chinese OK). top_k: max entries (default 5).
        filter_tags: JSON object, narrows BEFORE retrieval, e.g. {"chapter": 43}.
        occurred_after / occurred_before: 'YYYY-MM-DD', inclusive window.
        scope: project (default) | user | all; libraries='a,b' fans out to <=4 named libraries (mutually exclusive with scope=user/all).
    
    score is a fused RRF ranking point, NOT cosine - judge closeness by 'similarity'. Full text: get_memory; budgeted view: get_memory_context."""
    await _await_warmup()
    # filter_tags parsing, key-name whitelist validation, candidate narrowing,
    # and miss/count bookkeeping all live in RecallService.recall (parsing comes
    # after library resolution, order preserved); the adapter layer passes
    # client=session_key() (read-side wiring).
    svc = _recall_service(library, scope, libraries)
    return json.dumps(svc.recall(
        query=query, top_k=top_k, use_lnn=use_lnn, sample=sample,
        filter_tags=filter_tags, client=session_key(), scope=scope,
        libraries=libraries,
        occurred_after=occurred_after, occurred_before=occurred_before),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def get_memory(memory_id: str, library: str = "",
                     include_history: bool = False) -> str:
    """Fetch one memory by exact ID.
Returns the stored record, including state, source ids, entity ids, tags, and
provenance metadata when present.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.get_memory(memory_id=memory_id,
                                     include_history=include_history,
                                     client=session_key()),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def update_memory(
    memory_id: str,
    content: str = "",
    importance: float = -1.0,
    memory_type: str = "",
    state: str = "",
    superseded_by: str = "",
    clear_quarantine: bool = False, library: str = "",
) -> str:
    """Update only the provided fields of one memory (locate the id via recall). To retire an outdated fact: write the new version via ingest, then state='superseded' + superseded_by=<new id> (the result echoes this chain). clear_quarantine is a home-session-managed promotion.
    
    Args: memory_id; content; importance (-1 keep); memory_type; state active|superseded|archived; superseded_by."""
    await _await_warmup()
    svc = _write_service(library)
    # Validation (MemoryState legality / superseded_by existence) and field
    # assembly live in WriteService.update_memory; this tool is just
    # "call → serialize".
    return json.dumps(svc.update_memory(
        memory_id=memory_id, content=content, importance=importance,
        memory_type=memory_type, state=state, superseded_by=superseded_by,
        clear_quarantine=clear_quarantine),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def update_tags(
    tags: str = "",
    clear_keys: str = "",
    source: str = "",
    source_contains: str = "",
    contains: str = "",
    state: str = "",
    untagged_only: bool = False,
    missing_keys: str = "",
    memory_ids: str = "",
    mode: str = "merge",
    dry_run: bool = True,
    limit: int = 0,
    confirm: bool = False,
    undo: str = "", library: str = "",
) -> str:
    """Batch-update tags on existing memories.
Executes an explicit dry-run plan by default. Because tags are a flat mapping and
there is no automatic transaction, operations above the configured threshold
require explicit confirmation. The response includes a full before/after ledger
and an ``op_id`` for reversible undo.

Args:
    tags: JSON object of tag key/value updates.
    clear_keys: JSON array of tag keys to remove.
    missing_keys: JSON array of keys that must be removed if present.
    memory_ids: JSON array restricting the operation to specific memories.
    confirm: Required confirmation when the operation exceeds the safety threshold.
    dry_run: Preview the plan without writing; default true.
    undo: Undo a previous operation by ledger ``op_id`` or ``"last"``.
    untagged_only: Restrict the operation to records with no tags.
    """
    await _await_warmup()
    # Four-segment JSON parsing goes through the shared helper (behavior
    # byte-identical): its error body **does not** carry the library first key,
    # and the current code already runs it before library resolution (syntax
    # error first, library untouched) — preserving order preserves byte-identical
    # behavior. The helper uses the strict variant internally; bad JSON is an
    # explicit error, never conflated with the lenient variant used by
    # knowledge_topic.
    parsed, arg_err = _parse_tagops_args(tags, clear_keys, missing_keys, memory_ids)
    if arg_err:
        return json.dumps({"ok": False, "error": arg_err}, ensure_ascii=False)

    svc = _write_service(library)
    return json.dumps(svc.update_tags(
        tags=parsed["tags"], clear_keys=parsed["clear_keys"],
        source=source, source_contains=source_contains, contains=contains,
        state=state, untagged_only=untagged_only, missing_keys=parsed["missing_keys"],
        memory_ids=parsed["memory_ids"], mode=mode, dry_run=dry_run, limit=limit,
        confirm=confirm, undo=undo), ensure_ascii=False)


@server.tool(structured_output=False)
async def forget_memory(memory_id: str, reason: str = "", library: str = "") -> str:
    """Delete one memory by id: cascade (entity index, L3 derivatives, tombstone). DESTRUCTIVE - confirm the id via recall first; prefer state='superseded' for corrections.
    
    Args: memory_id, reason."""
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.forget_memory(memory_id=memory_id, reason=reason),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def forget_by_query(query: str, top_k: int = 3, dry_run: bool = True,
                          min_similarity: float = -1.0,
                          allow_multiple: bool = False, library: str = "") -> str:
    """Find and optionally delete memories matching a query.
Preview is the default: it returns ranked candidates without deleting. Pass
``dry_run=false`` to delete the selected candidates. This is useful for “that
whole topic is wrong,” but review the preview before deleting.

Args:
    query: Query text used to locate candidates.
    top_k: Maximum candidates to inspect; default 3.
    dry_run: When true, return candidates only; default true.
    confirm: Confirmation required for a non-dry-run destructive operation.
    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.forget_by_query(
        query=query, top_k=top_k, dry_run=dry_run,
        min_similarity=min_similarity, allow_multiple=allow_multiple),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def list_memories(
    limit: int = 50, source: str = "", state: str = "", library: str = ""
) -> str:
    """List raw entries with metadata (source/state filters); audit path - recall is the search entry. Args: limit, source, state."""
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.list_memories(
        limit=limit, source=source, state=state, client=session_key()),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def list_sources(library: str = "") -> str:
    """List known source labels and record counts.
Useful before bulk import or source-scoped cleanup. The response is metadata
only; use ``list_memories(source=...)`` to inspect records from one source.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.list_sources(client=session_key()),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def reset_context(library: str = "") -> str:
    """Reset the LNN temporal context for the current library.
This clears hidden state and recent-context windows. It does not delete
memories or alter stored records.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # Note: the response body **does not** carry the library first key (byte-identical with the current tool).
    return json.dumps(svc.reset_context(), ensure_ascii=False)


@server.tool(structured_output=False)
async def get_stats(library: str = "") -> str:
    """Return basic service/library statistics.
For richer self-reporting, use ``brain_stats``.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    # Note: the stats response body **does not** carry the library first key (byte-identical with the current tool).
    return json.dumps(svc.get_stats(), ensure_ascii=False)


@server.tool(structured_output=False)
async def save_state(library: str = "") -> str:
    """Persist the current library state.
Normally automatic, but useful after large in-memory operations or before
external inspection. Returns a save receipt/report.
    """
    await _await_warmup()
    svc = _write_service(library)
    # The ConcurrentWriteError conflict envelope is assembled inside
    # WriteService.save_state (hint points to load_state / retry_save); the tool
    # only serializes.
    return json.dumps(svc.save_state(), ensure_ascii=False)


@server.tool(structured_output=False)
async def retry_save(library: str = "") -> str:
    """Retry the most recent failed save.
Use after fixing disk, permission, or concurrent-write issues. The response
reports whether the retry succeeded and which layers were written.
    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.retry_save(), ensure_ascii=False)


@server.tool(structured_output=False)
async def load_state(library: str = "") -> str:
    """Reload persisted state from disk.
Use after another process modified the library or after recovering from a failed
save. Avoid reloading while another writer is mid-operation.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # Both the load failure (load_issue) and the success path are assembled in the
    # service; the tool only serializes. The response body **does not** carry the
    # library first key (byte-identical with the current tool).
    return json.dumps(svc.load_state(), ensure_ascii=False)


# ------------------------------------------------------------------
# v0.3 use-layer tools: session lifecycle + 5-slot context + consolidation + feedback
# ------------------------------------------------------------------
# ==================================================================
# Multi-library: in-process registry + call-side addressing (main channel)
# ==================================================================
# Default library = env HIPPOCAMPUS_PERSIST_DIR. Empty (unset) falls back to it,
# with behavior **byte-identical** to pre-multi-library — compatibility is an
# acceptance item.
_DEFAULT_LIB_PATH = _resolve_default_lib_path()

# Connection-level default library: session key -> library path (fixes daemon-reuse
# cross-library contamination).
# Why it must exist: one daemon process serves N connections, and
# ``_DEFAULT_LIB_PATH`` is a **process-level** constant (computed at module import
# from that launcher's env/cwd at the time). When a second workspace reuses the
# same daemon, without this connection-level override its writes would land in the
# first workspace's library — silent cross-contamination.
# Written by the launcher's out-of-band handshake line (``session_ctx.HELLO_PREFIX``)
# at connection setup.
_conn_default_lib: dict[str, str] = {}
_conn_lib_lock = threading.RLock()

# C1 workspace binding: session key -> reverse-lookup result (for session_start self-report).
# State machine values + declared: bound (directory workspace_root exact hit,
# library = the hit library) / unbound (directory exists but not this workspace) /
# ambiguous (multiple hits — **never take the first**; an ambiguous home library is
# more dangerous than none, fall back to the declared library and self-report) /
# no_cwd / no_user_library / directory_unreadable (a broken sidecar ≠ empty, U5
# lesson) / lookup_error.
# declared = the library declared by the launcher HELLO (env semantics) — when the
# reverse lookup hits it differs from library, which is the visible evidence that
# "bind the library, not the install" overrides the env declaration.
_conn_workspace_binding: dict[str, dict[str, Any]] = {}


def _resolve_workspace_binding(cwd: str) -> tuple[str, dict[str, Any]]:
    """Reverse-look up the task directory by the connection's workspace path to
    pick the home library (C1, design doc §3.2).

    The reverse lookup is an **enhancement**, not a precondition: any failure
    (user library unreachable / broken sidecar / unexpected exception) returns an
    empty path plus a status with reason and never blocks connection setup — the
    same discipline as the bridge handshake falling back to the process default
    library on failure. The criteria (exact normcase equality, ambiguity left
    unadjudicated) live in domain/task_directory.resolve_workspace; this function
    only does discovery and I/O.

    Returns:
        ``(path, info)``. Non-empty path = hit, i.e. home library; info always
        carries status (self-report raw material).
    """
    info: dict[str, Any] = {"cwd": cwd or "", "status": "no_cwd",
                            "library": "", "task": ""}
    if not (cwd or "").strip():
        return "", info
    try:
        user_lib = None
        for cand in libraries(False):
            if (cand.name or "").strip() == USER_LIBRARY_NAME:
                user_lib = cand
                break
        if user_lib is None:
            info["status"] = "no_user_library"
            return "", info
        data, issue = _task_directory.load_directory(user_lib.path)
        if issue:
            info["status"] = "directory_unreadable"
            info["reason"] = issue
            return "", info
        entry, status = resolve_workspace(
            normalize_entries((data or {}).get("entries")), cwd)
        info["status"] = status
        if entry is not None:
            info["library"] = entry["path"]
            info["task"] = entry["task"]
            # D2b: the hit entry's read-side tier declaration travels with the binding record ("" = undeclared → masked).
            info["access"] = entry.get("access", "")
            return entry["path"], info
        return "", info
    except Exception as exc:  # noqa: BLE001 — the reverse lookup never blocks connection setup
        logger.warning("workspace binding reverse lookup failed (falling back to the declared library): %r", exc)
        info["status"] = "lookup_error"
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return "", info

# **Per-library** deduplicated warmup registration.
# Background: main()'s warmup runs during process startup, when the contextvar is
# empty ⇒ session_key() is always __default__ ⇒ it uses the **process-level**
# _DEFAULT_LIB_PATH (= the first launcher's library). When the daemon is reused by
# a second workspace, that library would **never be warmed up** and its first call
# would pay the 25–33s load itself (hitting the host's tool timeout — same class of
# problem). Hence each connection triggers one extra warmup when it declares its
# library. A set, not a counter: N connections declaring the same library warm it up once.
_warmup_libs: set[str] = set()
_warmup_libs_lock = threading.Lock()


def set_connection_default_library(persist_dir: str = "", cwd: str = "",
                                   session: str | None = None) -> str:
    """Declare the default library for the **current connection** (one per
    connection after daemonization).

    Since C1 there is an extra **workspace-binding reverse lookup**: an entry in
    the task directory (user library sidecar) whose ``workspace_root`` exactly
    matches this connection's cwd hands its ``path`` the connection default
    library — the resolution basis of "bind the library, not the install" (design
    doc §3.2). Precedence: **directory reverse-lookup hit > launcher declaration
    (env semantics) > process-level default**. That a reverse-lookup hit overrides
    the env declaration was an explicit ruling: under a user-level install the
    launcher always carries the env declaration, so if the declaration won,
    binding would never trigger; explicitness is backstopped by session_start's
    ``workspace_binding`` self-report (declared and library visible side by side).
    Miss/ambiguity/any failure falls back to the declared library with the status
    recorded — **a miss never lands on the user library** (design doc §3.2 red
    line); status is left for the opening pack to self-report.
    """
    path, fallback = resolve_default_lib_ex(persist_dir or None, cwd or None)
    declared = path
    key = session_key(session)
    hit, binding = _resolve_workspace_binding(cwd or "")
    binding["declared"] = declared
    if fallback:
        # Field feedback 2026-10-31: declared-library resolution failures used to
        # land only in server logs (invisible to the agent), and declared was even
        # rewritten to the fallback path, losing the original declared name. Three
        # keys go into binding (visible via the task_directory slot's workspace
        # passthrough), and the warning text additionally surfaces to the
        # session_start response via _inject_fallback_warning.
        binding["declared_raw"] = fallback["declared_raw"]
        binding["fallback_reason"] = fallback["reason"]
        binding["warning"] = _fallback_warning_text(fallback, declared)
    if hit:
        path = hit
        logger.info("workspace binding active: cwd=%s -> home library=%s (task=%s, overriding the declared library %s)",
                    binding.get("cwd"), path, binding.get("task"), declared)
    with _conn_lib_lock:
        _conn_default_lib[key] = path
        _conn_workspace_binding[key] = binding
    # I3: when the daemon is reused by multiple workspaces, main()'s process-level
    # warmup never covers this library. Run one extra warmup right after the
    # connection declares it (per-library dedup; failure logs a single warning line).
    maybe_warmup_library(path)
    return path


def default_library_path(client: str | None = None) -> str:
    """Default library for this call's context: connection-level > process-level."""
    with _conn_lib_lock:
        cd = _conn_default_lib.get(session_key(client))
    if cd:
        return cd
    return _DEFAULT_LIB_PATH


def _workspace_binding_brief(client: str | None = None) -> dict[str, Any]:
    """This connection's workspace binding status (for session_start's
    task_directory slot self-report).

    Direct clients without HELLO (tests/troubleshooting) have no binding record
    → ``undeclared`` — not an error, just "this connection never went through the
    declaration channel".
    """
    with _conn_lib_lock:
        rec = _conn_workspace_binding.get(session_key(client))
    if not rec:
        return {"status": "undeclared", "cwd": "", "library": "",
                "task": "", "declared": "",
                "note": "Direct connection without a HELLO declaration: "
                        "normal for probes and single-library setups, not "
                        "an error. The library is matched by cwd."}
    return dict(rec)


def _origin_task() -> str:
    """Value of the write-origin ``origin.task``: session declaration >
    workspace-binding mapping > empty.

    Both sources are **explicitly registered by the host**, never guessed from
    content (``common/write_origin`` rule 3):

    - session declaration ``session_start(task=...)``: what this session is doing;
      dynamic;
    - the task directory's "task → library" mapping (``_conn_workspace_binding``'s
      task): which work item this library belongs to; stable.

    A super-agent's Ticket is closer to the latter, but when the host can declare
    the task explicitly the dynamic value is more accurate, hence the former wins.
    Defined in the lower half of the file; the three write tools above resolve it
    by name at runtime, independent of definition order.

    Keys always go through ``session_key()`` (**without client**) — the three
    write tools have no ``client`` parameter; mixing in ``session_key(client)``
    would misalign whenever the host passes client, showing up as the task being
    silently unreadable (C1 binding shares this namespace; the two cannot diverge).
    """
    declared = current_task()
    if declared:
        return declared
    with _conn_lib_lock:
        rec = _conn_workspace_binding.get(session_key()) or {}
    return str(rec.get("task") or "").strip()


def _user_access_tier(info: dict[str, str] | None,
                      client: str | None = None) -> str:
    """D2b: this connection's read-side tier for the user library (returns the
    empty string for non-user-library targets).

    The single source of truth for the criteria is
    domain/task_directory.resolve_user_access; this only does discovery (home
    library comparison, binding record, escape hatch). The tier resolves per
    client — under a multi-library daemon every connection has its own binding.
    """
    info = info or {}
    if (info.get("name") or "").strip() != USER_LIBRARY_NAME:
        return ""
    home = default_library_path(client)
    home_is_user = same_library(home, info.get("path", ""))
    with _conn_lib_lock:
        rec = _conn_workspace_binding.get(session_key(client)) or {}
    return resolve_user_access(
        home_is_user=home_is_user,
        binding_status=str(rec.get("status") or "undeclared"),
        entry_access=str(rec.get("access") or ""),
        escape_open=bool(load_config().cross_library_write))


def _fallback_warning_text(fallback: dict[str, str], declared: str) -> str:
    """Agent-visible warning text for declared-library resolution fallback
    (change two, 2026-10-31).

    Wording aligned with the D2b reality: in the unbound fallback form, writes go
    to review quarantine, not full direct writes — say "goes to review", not
    "would write the wrong library". The fallback library name takes the path's
    tail segment for display (under user-level access the env default library's
    directory name is `user`).
    """
    raw = fallback.get("declared_raw") or "(empty)"
    tail = os.path.basename((declared or "").rstrip("\\/")) or declared or "(unknown)"
    return (f"The workspace-declared library {raw!r} is unregistered or failed to resolve ({fallback.get('reason')}); "
            f"fell back to the default library {tail!r}: reads land there and writes go to the review queue (not direct writes). "
            f"Register the library via library_create, or check the library field of the "
            f"workspace .hippocampus.json.")


def _inject_session_disclosures(payload: str, client: str | None) -> str:
    """All session_start injections (the extraction point for the "tool body
    ≤20 lines" discipline).

    Order: first the C1 workspace-binding self-report (inside the task_directory
    slot), then the declared-library fallback ``⚠`` (top level of the response) —
    the latter rewrites the response structure, so it goes last to avoid being
    overwritten by the former.
    """
    return _inject_fallback_warning(
        _inject_workspace_disclosure(payload, client), client)


def _inject_fallback_warning(payload: str, client: str | None) -> str:
    """Field feedback 2026-10-31: when this connection has had a declared-library
    fallback, session_start's response gets a ``⚠`` warning block at the **top
    level**.

    **Decoupled from the task_directory slot** (injected even when the slot is
    off): a mistaken fallback costs more than saved tokens — this is a
    **deliberate exception** to the "off = zero traces" rule; the injection
    precondition is "this connection really did fall back" (binding carries a
    warning), not a resident key. The self-report is a freebie: any exception
    returns the payload as-is and never blows up the opening pack.
    """
    try:
        with _conn_lib_lock:
            rec = _conn_workspace_binding.get(session_key(client)) or {}
        warning = rec.get("warning")
        if not warning:
            return payload
        obj = json.loads(payload) if isinstance(payload, str) else payload
        if isinstance(obj, dict) and "⚠" not in obj:
            return json.dumps({"⚠": str(warning), **obj}, ensure_ascii=False)
    except Exception:  # noqa: BLE001 — the self-report never blows up the opening pack
        pass
    return payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False)


def _inject_workspace_disclosure(payload: str, client: str | None) -> str:
    """C1: inject the ``workspace`` binding self-report when the task_directory
    slot is ready.

    Slot off / not ready = zero traces (no key added; same rule as U5's
    "off = zero traces"). The self-report is a freebie: any exception returns the
    payload as-is and never blows up the opening pack. **Being a separate function
    is required by the M6 S1.2 "tool body ≤20 lines" discipline** — the tool body
    keeps only the one-line call.
    """
    try:
        obj = json.loads(payload) if isinstance(payload, str) else payload
        if isinstance(obj, dict):
            td = obj.get("task_directory")
            if isinstance(td, dict) and td.get("ready"):
                td["workspace"] = _workspace_binding_brief(client)
                return json.dumps(obj, ensure_ascii=False)
    except Exception:  # noqa: BLE001 — the self-report never blows up the opening pack
        pass
    return payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False)


# Library list cache: scanning walks the disk; rescanning on every call would slow the list-type tools.
_library_cache: list[Any] | None = None
_library_cache_at = 0.0
_LIBRARY_CACHE_TTL = load_config().library_cache_ttl

_registry: BrainRegistry | None = None
_registry_lock = threading.RLock()


def _registry_get() -> BrainRegistry:
    """In-process multi-library registry (lazy). The injected factory lets tests plug in fake brains."""
    global _registry
    if _registry is not None:
        return _registry
    with _registry_lock:
        if _registry is None:
            _registry = BrainRegistry(factory=_build_brain)
    return _registry


def _brain_for(library: str | None = None,
               client: str | None = None) -> HippoBrain:
    """Get the brain for this call's target library (multi-library entry).

    **Each library path is constructed only once**: two brains pointing at the
    same ``memories.json`` are explicitly forbidden in v0.4 (writes invisible to
    each other + wholesale overwrite loses data).
    Per-library locks ⇒ library A loading (25–33s) never blocks library B calls.
    """
    return _registry_get().get(resolve_library_path(library, client))


def _manager_for(library: str | None = None,
                 client: str | None = None) -> MemoryManager:
    """Get this call's target-library MemoryManager (same in-memory state as the brain)."""
    return _brain_for(library, client).manager

def _map_fields_error(message: str, library: str | None = None) -> str:
    """Error body for map_set fields parse failures (produced by this shared
    helper; serialization byte-identical).

    Both fields errors (bad JSON / non-JSON-object) are currently assembled via
    ``json.dumps(_with_library({"status": "error", "error": ...}, library),
    ensure_ascii=False)`` — the error body **does** carry the library first key
    (the only divergence between the two order-preserving tools map_set and
    update_tags, pinned by a probe case). Extracting the helper only saves tool
    body lines; the dict key order (status→error) and the serialization arguments
    are preserved as-is.

    Note the ``json.loads`` parse of fields itself is **deliberately not
    extracted**: the tool-layer case (test_m4_batch3_tool_face T4) pins the source
    literal "json.loads(fields) before _organize_service(library)" (order
    preservation: bad library name + bad fields must raise the library error
    first — envelope assembly IS library resolution); removing the literal would
    turn that assertion red.
    """
    return json.dumps(_with_library(
        {"status": "error", "error": message}, library), ensure_ascii=False)


def _task_directory_register(entry: dict[str, Any]) -> dict[str, Any]:
    """Register an entry into the user-level task directory
    (``<user library>/directory.json``).

    This is one of the task directory's **managed channels** (registration) and
    the only automatic hook: ``library_create`` calls it right after creating a
    library. Three semantics:

    * **The target library is fixed by role**: the user-level library with the
      conventional name ``user`` — not "the current addressing target", nor the
      caller's home library. If the user library is missing/unreachable ⇒ return
      ``registered: false`` + ``user_library_not_found`` (**do not create** the
      user library: library creation is an explicit action and must not be
      quietly triggered by a registration).
    * **Failures do not raise**: registration is a freebie; raising would turn a
      successful library creation into a failure (the library is already on disk,
      yet the agent gets an error). The reason is returned as-is for the use-case
      layer to turn into ``directory_registered: false`` + warning.
    * **Never silent**: broken sidecar / write failures both carry a reason;
      never pretend registration succeeded.

    The read path (``brain/session.py::_task_directory_brief``) does not go
    through here — it takes the lightweight path reading the sidecar directly,
    and no symbol from this function appears there (reads need no permission).
    """
    try:
        info = _library.resolve_library("user", libs=libraries())
    except LibraryResolutionError as exc:
        return {"registered": False, "reason": "user_library_not_found",
                "detail": str(exc)}
    if info is None:  # a non-empty "user" always resolves; None = truly unreachable (same defense as above)
        return {"registered": False, "reason": "user_library_not_found"}
    try:
        return _task_directory.upsert_entry(info.path, entry)
    except Exception as exc:  # noqa: BLE001 — broken files/write failures must self-report
        logger.warning("task-directory write failed (%s): %r", _task_directory.directory_path(
            info.path), exc)
        return {"registered": False, "reason": "task_directory_write_failed",
                "detail": str(exc)}


def _library_admin_service() -> LibraryAdminService:
    """Use-case service for the 5 cross-library admin tools
    (list/info/whoami/use/create).

    **Deliberately no brain passed**: these 5 tools never construct a brain for
    some library. The only one needing a brain, ``library_create``, fetches it
    after creation via the port's ``brain_for`` (and only for the **newly created**
    library); the other four never touch the registry.

    Note: this factory **stays in server.py** (the other five live in
    ``server_runtime/composition.py``) — test_m4_batch4_tool_face T4 pins this
    function's body in server.py source via ``ast.get_source_segment`` (the
    source-level contract "no brain passed, no registry touch"), kept per the
    "adapt code to tests, never edit tests" discipline (same case as map_set's
    json.loads); the ports object is the same instance fetched via re-export.
    """
    return LibraryAdminService(ports=_LIB_PORTS)


_brain: HippoBrain | None = None


def get_brain() -> HippoBrain:
    """Entry point getting the HippoBrain singleton.

    **Kept for compatibility** (warmup, old scripts, and tests all use it): it
    equals ``_brain_for(None)``, i.e. the **default library**. For multi-library
    use :func:`_brain_for`. The module-level ``_brain`` is now only a **shortcut
    cache for the default library**; the real holder is :class:`BrainRegistry`
    (key = normalized library path).
    """
    global _brain
    if _brain is not None:
        return _brain
    brain = _brain_for(None)
    _brain = brain
    return brain


@server.tool(structured_output=False)
async def ingest(
    text: str,
    speaker: str = "user",
    importance: float = -1.0,
    cognitive_type: str = "",
    entities: str = "",
    source: str = "",
    tags: str = "",
    auto_tags: bool = False, library: str = "",
    idem_key: str = "",
) -> str:
    """RECOMMENDED write entry: raw text -> L0 archive + extracted L1 facts + entity links; dedup, supersede detection, idem_key retries. Use for "remember this".
    
    Args:
        text: verbatim. speaker: user|assistant|role. importance: -1 = auto.
        cognitive_type: experience|world|opinion (blank = auto). entities: comma list.
        source: origin label. tags: JSON object. idem_key: same key on retry = no duplicate."""
    await _await_warmup()
    svc = _write_service(library)
    # tags parsing (error body carries the library first key), sentinel
    # normalization of entities/importance/cognitive_type, save_warning injection,
    # and logging all live in WriteService.ingest.
    return json.dumps(svc.ingest(
        text=text, speaker=speaker, importance=importance,
        cognitive_type=cognitive_type, entities=entities, source=source,
        tags=tags, auto_tags=auto_tags,
        origin=build_origin(session=session_key(), task=_origin_task(), surface="ingest"),
        idem_key=idem_key), ensure_ascii=False)


@server.tool(structured_output=False)
async def ingest_batch(items: str, source: str = "", tags: str = "",
                       granularity: str = "sentence",
                       detect_supersede: bool = False, save: bool = False,
                       auto_tags: bool = False, library: str = "") -> str:
    """Ingest multiple texts in one call.
Each item is processed independently; one invalid item does not block the rest.
The response reports per-item success, new L1 IDs, merges, truncations, and
errors. Prefer this over repeated single ``ingest`` calls for bulk imports.
    """
    await _await_warmup()
    svc = _write_service(library)
    # items/tags parsing (both error bodies carry the library first key),
    # per-item isolation, and whole-batch persistence live in
    # WriteService.ingest_batch; this tool is just "call → serialize".
    return json.dumps(svc.ingest_batch(
        items=items, source=source, tags=tags, granularity=granularity,
        detect_supersede=detect_supersede, save=save, auto_tags=auto_tags,
        # P0-1: batch takes only the write origin, not an idempotency key (bulk-
        # import replay detection is the source-file signature source_drift, not
        # a request key).
        origin=build_origin(session=session_key(), task=_origin_task(), surface="ingest_batch")),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def get_memory_context(query: str = "", top_k: int = 5, token_budget: int = 1200,
                             item_max_chars: int = 200, filter_tags: str = "", library: str = "",
                             scope: str = "project") -> str:
    """Five-slot assembled context (hits / open todos / timeline / entity state / unfinished topics) under a char budget. For "what should I know right now"; point lookups use recall.
    
    Args:
        query: empty = pinned brief. top_k: hit count. token_budget: whole-block chars (coarse). item_max_chars: per-line clip, 0 = off (numeric tails survive).
        filter_tags: JSON object. scope: project | user only (no all)."""
    await _await_warmup()
    # filter_tags parsing (error body carries no envelope) and the J2 echo live
    # in RecallService.get_memory_context; parsing comes after library resolution,
    # order preserved.
    svc = _recall_service(library, scope)
    return json.dumps(svc.get_memory_context(
        query=query, top_k=top_k, token_budget=token_budget,
        item_max_chars=item_max_chars, filter_tags=filter_tags, scope=scope,
        client=session_key()),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def session_start(task: str = "", recent_limit: int = 10,
                        working_on: str = "", task_top_k: int = 5,
                        pin_limit: int = 0, filter_tags: str = "",
                        client: str = "", library: str = "",
                        user_scope: int = 0, task_directory: int = 0) -> str:
    """Open a session; return the pre-activation package (pins, profile, open topics, intents, task recall, receipt). Call at session start or after compaction - on compaction keep ONLY the receipt identity and call again. Check active_sessions before editing files.
    
    Args:
        task: drives pre-recall and write attribution. working_on: module/file.
        recent_limit / task_top_k / pin_limit / user_scope / task_directory: 0 = env default, negative = off."""
    await _await_warmup()
    return await _session_start_payload(
        task=task, recent_limit=recent_limit, working_on=working_on,
        task_top_k=task_top_k, pin_limit=pin_limit,
        filter_tags=filter_tags, client=client, library=library,
        user_scope=user_scope, task_directory=task_directory)


async def _session_start_payload(*, task: str, recent_limit: int,
                                 working_on: str, task_top_k: int,
                                 pin_limit: int, filter_tags: str,
                                 client: str, library: str,
                                 user_scope: int,
                                 task_directory: int) -> str:
    """Assemble the session_start payload after tool-face parsing/gating."""
    # Register the task declared for this connection; later writes use it for
    # origin.task. An empty task does not clear it; session_end does.
    bind_task(task)
    # Library resolution stays first: a bad library + bad JSON must report the
    # library error first, and that error body has no library envelope.
    svc = _bootstrap_service(library)
    parsed_tags, tag_err = _parse_tags_json(filter_tags)
    if tag_err:
        return json.dumps({"error": f"filter_tags parse failed: {tag_err}",
                           "input": filter_tags[:200]}, ensure_ascii=False)
    payload = json.dumps(svc.session_start(
        task=task, recent_limit=recent_limit, working_on=working_on,
        task_top_k=task_top_k, pin_limit=pin_limit, filter_tags=parsed_tags,
        client=client, user_scope=user_scope,
        task_directory=task_directory,
        user_access=_user_access_tier(svc.library_self, client)),
        ensure_ascii=False)
    return _inject_session_disclosures(payload, client)


@server.tool(structured_output=False)
async def export_memories(since_ts: float = 0.0, source: str = "", state: str = "",
                          limit: int = 0, library: str = "") -> str:
    """Export memories without vectors for backup or migration.
Supports incremental export: pass the largest ``timestamp`` from the previous
export. The boundary is inclusive, so deduplicate by ID after export.

Args:
    since_ts: Only records with timestamp >= this Unix time; 0 means all.
    source: Exact source filter.
    state: ``active``, ``superseded``, ``archived``, or empty for all.
    limit: Maximum records; 0 means unlimited.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # Note: the response body **does not** carry the library first key (byte-identical with the current tool).
    return json.dumps(svc.export_memories(since_ts=since_ts, source=source,
                                          state=state, limit=limit,
                                          client=session_key()),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def session_end(summary: str = "", open_topics: str | list[str] = "",
                      close_topics: str | list[str] = "",
                      session_id: str = "",
                      client: str = "", library: str = "") -> str:
    """Close THIS session: L2 summary, profile update, consolidate trigger, forced save.
    
    Args:
        summary: one line. open_topics: unfinished topics (array or comma string).
        close_topics: topics to close (exact text). client: caller id.
    Skip entirely when this session never touched a hippocampus tool."""
    await _await_warmup()
    # P1: clear the task registration at session end so the next session does not
    # wrongly inherit the previous origin.task.
    clear_task()
    svc = _bootstrap_service(library)

    # Normalization of the two open_topics / close_topics forms (JSON array / comma string) stays in the adapter layer.
    def _norm_list(v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return [str(t).strip() for t in v if str(t).strip()]

    return json.dumps(svc.session_end(
        summary=summary, open_topics=_norm_list(open_topics),
        close_topics=_norm_list(close_topics), session_id=session_id,
        client=client), ensure_ascii=False)


@server.tool(structured_output=False)
async def remember_intent(text: str, due: str = "", library: str = "") -> str:
    """Register a todo/intent for future sessions.
The intent is returned by later ``session_start`` calls and can trigger proactive
recall when a related topic appears. Use for user requests that span sessions.

Args:
    text: What should be done.
    due: Optional due date/deadline.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.remember_intent(text=text, due=due), ensure_ascii=False)


@server.tool(structured_output=False)
async def complete_intent(keyword: str, library: str = "") -> str:
    """Mark a todo/intent complete by keyword.
Completion is keyword-based and reports whether a matching open intent was
found. Completed intents no longer participate in proactive recall.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.complete_intent(keyword=keyword), ensure_ascii=False)


@server.tool(structured_output=False)
async def report_feedback(memory_id: str, useful: bool = True, library: str = "") -> str:
    """Report whether a recalled memory was useful.
Useful feedback increases future ranking priority; not-useful feedback lowers
visibility without deleting the memory. Use ``forget_memory`` for genuine
correction/deletion.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.report_feedback(memory_id=memory_id, useful=useful),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def consolidate(filter_tags: str = "", max_items: int = 0,
                      dry_run: bool = False, library: str = "") -> str:
    """Run maintenance/consolidation over the library.
Can mark superseded facts, archive dormant records, and list contradiction
candidates. Dry-run is supported and recommended before a real write.

Args:
    filter_tags: JSON object string narrowing the consolidation scope.
    max_items: Maximum items to process; 0 uses the configured limit.
    dry_run: Report the plan without writing; default false.
    """
    await _await_warmup()
    # filter_tags parsing (error body carries no envelope), key validation, and
    # the J2 soft-degrade echo all live in OrganizeService.consolidate; parsing
    # comes after library resolution, order preserved.
    svc = _organize_service(library)
    return json.dumps(svc.consolidate(
        filter_tags=filter_tags, max_items=max_items, dry_run=dry_run),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def contradictions(limit: int = 10, source: str = "",
                         filter_tags: str = "", library: str = "",
                         plan: bool = False, confirm: str = "") -> str:
    """List candidate pairs where one fact may have two versions.
This is annotation, not adjudication. Review candidates and resolve through
``update_memory`` (usually supersession) or ``forget_memory``.

Args:
    limit: Maximum candidate pairs; default 10.
    source: Optional source filter.
    filter_tags: JSON object string narrowing the candidate pool.
    plan: Also return a reviewable supersede_plan and a need_human bucket.
    confirm: Host-approved plan array (JSON string) to execute in one call.
            The service never adjudicates; it writes only what was sent.
    """
    await _await_warmup()
    # filter_tags parsing (error body carries no envelope) and the candidate_pool
    # echo live in RecallService.contradictions; parsing comes after library
    # resolution, order preserved.
    svc = _recall_service(library)
    if confirm:
        return json.dumps(svc.apply_supersede_plan(confirm), ensure_ascii=False)
    return json.dumps(svc.contradictions(
        limit=limit, source=source, filter_tags=filter_tags, plan=plan),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def drift_audit(source: str = "", limit: int = 50,
                      include_ok: bool = False, library: str = "") -> str:
    """Audit memories whose source files may have drifted.
For file-backed sources, compares recorded signatures (size/mtime/hash) with the
current files and reports missing, changed, or unbaselined sources. Also reports
user-library quarantine and access-log summaries when applicable.

Args:
    source: Specific source to audit; empty audits known sources.
    limit: Maximum detailed rows; default 50.
    filter_tags: JSON object string narrowing the audited records.
    """
    # Zero-argument form: the tier is inferred by looking the function name up in
    # STORE_ONLY_TOOLS (drift_audit is already in the table; it only reads the
    # store + source-signature layer, no embedding model needed). The dispatcher's
    # two explicit-argument spots are structural exceptions.
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.drift_audit(
        source=source, limit=limit, include_ok=include_ok,
        client=session_key()),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def entity_profile(entity: str, limit: int = 20, order: str = "asc",
                         include_superseded: bool = False, library: str = "") -> str:
    """Return the memory profile for one entity.
Includes linked memories, aliases when known, and mention statistics. Useful for
“what do we know about X?” before writing or updating project facts.

Args:
    entity: Entity name or alias.
    limit: Maximum linked memories; default 20.
    order: ``asc`` or ``desc`` chronological order.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.entity_profile(
        entity=entity, limit=limit, order=order,
        include_superseded=include_superseded), ensure_ascii=False)


@server.tool(structured_output=False)
async def entity_list(prefix: str = "", limit: int = 50, min_mentions: int = 1,
                      trusted_only: bool = False, library: str = "") -> str:
    """List known entities with optional prefix/count filters.
Use this to discover project vocabulary and inspect entity coverage.

Args:
    prefix: Optional case-insensitive prefix.
    limit: Maximum entities; default 50.
    min_mentions: Minimum linked-memory count; default 1.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.entity_list(
        prefix=prefix, limit=limit, min_mentions=min_mentions,
        trusted_only=trusted_only), ensure_ascii=False)


# ------------------------------------------------------------------
# Information map layer: maps answer "what is the state now", memories answer
# "what happened". State writes go only through map_set (keyed updates; rewriting
# a key replaces the entry wholesale); exact lookups go through map_get (semantic
# top-k for state = trading exactness for a guess).
# The four tools form the standalone profile group "map" ⇒ zero added tokens on
# the common surface; reachable under full and hippo_extra (same precedent as
# drift_audit in the admin group).
# ------------------------------------------------------------------

@server.tool(structured_output=False)
async def map_set(name: str, key: str, status: str = "", fields: str = "{}",
                  desc: str = "", truth_source: str = "", note: str = "",
                  updated_by: str = "", library: str = "") -> str:
    """Create/update one key in a named information map.
Information maps store current state (e.g. “where is X deployed?”) separate from
historical memories. Writing the same key replaces the prior entry; provenance
and audit fields are retained.

Args:
    name: Map name.
    key: Entry key.
    status: Current status/value.
    fields: JSON object of additional structured fields.
    desc: Human-readable description; used for relevance matching.
    truth_source: File/path/system that is authoritative for this entry.
    note: Optional note.
    updated_by: Who/what made this update.
    library: Optional library override.
    """
    await _await_warmup()
    # The JSON protocol parsing of fields stays in the adapter layer: currently it
    # runs before _brain_for (syntax error first, library untouched), and the error
    # body carries the library first key (the two order-preserving tools' only
    # divergence). The json.loads literal is deliberately kept in this function —
    # cases pin order preservation on it, see _map_fields_error.
    fdict: dict | None = None
    if (fields or "").strip():
        try:
            fdict = json.loads(fields)
        except json.JSONDecodeError as exc:
            return _map_fields_error("fields is not valid JSON: %s" % exc, library)
        if not isinstance(fdict, dict):
            return _map_fields_error("fields must be a JSON object", library)
    svc = _organize_service(library)
    # The ValueError (illegal name/key) error body of brain.map_set is assembled in the service, also carrying the library first key.
    return json.dumps(svc.map_set(
        name=name, key=key, status=status, fields=fdict, desc=desc,
        truth_source=truth_source, note=note, updated_by=updated_by),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def map_get(name: str, key: str, library: str = "") -> str:
    """Fetch one information-map entry by exact map/key.
Returns ``found=false`` with a reason when the key is absent. Prefer this over
semantic recall when you need current state, not historical discussion.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # Note: the response body **does not** carry the library first key (byte-identical with the current tool).
    return json.dumps(svc.map_get(name=name, key=key), ensure_ascii=False)


@server.tool(structured_output=False)
async def map_list(name: str = "", library: str = "") -> str:
    """List information maps or all entries in one map.
Without ``name``, returns map names, counts, freshness, and drift summaries.
With ``name``, returns the entries in that map.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # Note: the response body **does not** carry the library first key (byte-identical with the current tool).
    return json.dumps(svc.map_list(name=name), ensure_ascii=False)


@server.tool(structured_output=False)
async def map_verify(name: str = "", library: str = "") -> str:
    """Verify information-map entries against their truth sources.
For entries with a ``truth_source``, re-reads and compares the recorded source
signature, reporting stale or unverifiable entries. This is the map-layer
counterpart to source-drift auditing.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # The envelope (library first key) is assembled in the service — byte-identical with the current _with_library.
    return json.dumps(svc.map_verify(name=name), ensure_ascii=False)


@server.tool(structured_output=False)
async def knowledge_map(limit: int = 20, include_archived: bool = False,
                        library: str = "") -> str:
    """List derived knowledge topics in the library.
Topics are clusters of related L1 records with representative excerpts. Use this
to answer “what does this library know about?” and to locate a topic before
reading or writing its summary.

Args:
    limit: Maximum topics; default 20.
    include_archived: Include archived topics; default false.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _recall_service(library)
    # The envelope (library first key) is assembled in the service — this tool is one of the few read-side tools with F5 self-report.
    return json.dumps(svc.knowledge_map(
        limit=limit, include_archived=include_archived), ensure_ascii=False)


@server.tool(structured_output=False)
async def knowledge_topic(topic_id: str = "", action: str = "brief",
                          summary: str = "", evidence: str = "",
                          limit: int = 30, topics: str = "",
                          items: str = "", batch: int = 10,
                          per_topic: int = 5, fras: str = "",
                          library: str = "") -> str:
    """Read material for, or write/undo, one knowledge topic.
``action=brief`` returns cluster material. ``action=summarize`` writes a
host-generated summary back. ``action=revert`` undoes a host summary and returns
to extractive representative sentences. Evidence must be members of the topic.

Args:
    topic_id: Topic ID from ``knowledge_map``.
    action: ``brief`` / ``summarize`` / ``revert``.
    summary: Host-written summary for ``summarize``.
    evidence: JSON array of member memory IDs supporting the summary.
    fras: Optional four-field structured summary as a JSON object.
    library: Optional library override.
    """
    await _await_warmup()
    # JSON parsing of items / fras (explicit error on failure, error body carries
    # no envelope) and the lenient parsing of evidence/topics all live in
    # RecallService.knowledge_topic; parsing comes after library resolution, same
    # order as the current code.
    svc = _recall_service(library)
    return json.dumps(svc.knowledge_topic(
        topic_id=topic_id, action=action, summary=summary,
        evidence=evidence, limit=limit, topics=topics, items=items,
        batch=batch, per_topic=per_topic, fras=fras), ensure_ascii=False)


@server.tool(structured_output=False)
async def lexicon_add(terms: str, relink: bool = True, library: str = "") -> str:
    """Add trusted entity names/aliases to the library lexicon.
The lexicon improves extraction, entity linking, and recall for project jargon,
model hashes, ticket numbers, and other names generic embedders miss. By default
relinks existing memories afterward.

Args:
    terms: JSON array or comma-separated list. Use ``alias => canonical`` for aliases.
    relink: Re-run entity linking after adding terms; default true.
    library: Optional library override.
    """
    await _await_warmup()
    # The lenient parsing of terms (accepts JSON arrays and comma strings, never
    # errors) and the envelope both live in OrganizeService.lexicon_add; parsing
    # comes after library resolution, order preserved.
    svc = _organize_service(library)
    return json.dumps(svc.lexicon_add(terms=terms, relink=relink),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def lexicon_suggest(top_n: int = 30, min_docs: int = 4, min_files: int = 2,
                          min_len: int = 3, max_len: int = 4, exclude: str = "", library: str = "") -> str:
    """Suggest high-frequency unregistered names for the lexicon.
Read-only candidate generation. Review suggestions before adding them; do not
automatically accept every frequent token.
    """
    await _await_warmup()
    # The lenient parsing of exclude lives in the service (currently after library resolution, order preserved);
    # Note: the response body **does not** carry the library first key (byte-identical with the current tool).
    svc = _organize_service(library)
    return json.dumps(svc.lexicon_suggest(
        top_n=top_n, min_docs=min_docs, min_files=min_files,
        min_len=min_len, max_len=max_len, exclude=exclude),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def lexicon_alias_suggest(top_n: int = 30, min_confidence: float = 0.35,
                                min_support: int = 2, min_lift: float = 2.0,
                                min_containment: float = 0.5,
                                allow_stat_only: bool = False,
                                exclude: str = "", cue_words: str = "",
                                with_paren: bool = True, library: str = "") -> str:
    """Suggest possible aliases for the same entity.
Returns candidate groups with evidence; it does not merge automatically. Review
evidence, then register confirmed aliases through ``lexicon_add``.
    """
    await _await_warmup()
    # The lenient parsing of exclude / cue_words lives in the service (currently
    # after library resolution, order preserved); Note: the response body
    # **does not** carry the library first key (byte-identical with the current tool).
    svc = _organize_service(library)
    return json.dumps(svc.lexicon_alias_suggest(
        top_n=top_n, min_confidence=min_confidence, min_support=min_support,
        min_lift=min_lift, min_containment=min_containment,
        allow_stat_only=allow_stat_only, exclude=exclude,
        cue_words=cue_words, with_paren=with_paren),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def relink_entities(library: str = "") -> str:
    """Re-run entity extraction/linking across the library.
Use after a lexicon changes or when entity linking may be stale. The response
reports how many memories were relinked and any errors.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # The envelope (library first key) is assembled in the service — byte-identical with the current _with_library.
    return json.dumps(svc.relink_entities(), ensure_ascii=False)


@server.tool(structured_output=False)
async def miss_log(limit: int = 20, clear: bool = False, library: str = "") -> str:
    """Inspect recent retrieval misses.
A miss is a query whose top result did not pass the relevance floor. This is a
diagnostic signal for vocabulary gaps, retrieval failures, or missing project
knowledge—not an audit log of all queries.

Args:
    limit: Maximum recent misses; default 20.
    clear: Clear the miss log after reporting; default false.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # Like save_brain, this tool **does not** carry the library first key (byte-identical with the current tool).
    return json.dumps(svc.miss_log(limit=limit, clear=clear),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def brain_stats(daily_days: int = 0, library: str = "") -> str:
    """Return detailed library/runtime self-report.
Includes layer counts, entity stats, model/backend identity, save ledger,
knowledge coverage, and other health signals. Prefer this over ``get_stats``
for diagnosis.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # The warmup section is produced by an injected probe (MaintainService.brain_stats); the tool only serializes.
    return json.dumps(svc.brain_stats(daily_days=daily_days),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def save_brain(library: str = "") -> str:
    """Force a save of the current brain/library.
Useful before process exit, model swaps, or risky maintenance. Returns a save
report; failures are not silently swallowed.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # The envelope (including the ConcurrentWriteError conflict body) is assembled in the service; the tool only serializes.
    return json.dumps(svc.save_brain(), ensure_ascii=False)


# ------------------------------------------------------------------
# F4' · Multi-library: library_* tools (36 → 41)
# ------------------------------------------------------------------
@server.tool(structured_output=False)
async def library_list(include_backups: bool = False) -> str:
    """List discoverable memory libraries.
By default backup/snapshot directories are omitted. Use this before
``library_use`` or ``library_create`` to avoid guessing names or crossing
projects.

Args:
    include_backups: Include backup/snapshot directories; default false.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # Scanning / zero-library auto default-directory creation / row assembly all
    # live in the service (ports keep connection state and disk I/O in the adapter
    # layer); the tool only serializes. The response body **does not** carry the
    # library first key (byte-identical).
    return json.dumps(svc.list_libraries(include_backups=include_backups),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def library_info(library: str = "") -> str:
    """Describe one memory library.
Returns identity, path, layer files, sizes, model binding, and related metadata.
Empty library means the default/current library.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # Deliberately no brain constructed for it (directory metadata only); layer file sizes are read from disk via the port.
    return json.dumps(svc.describe(library=library), ensure_ascii=False)


@server.tool(structured_output=False)
async def library_whoami() -> str:
    """Which library this session reads/writes (binding self-report). Call when the target library is uncertain; near-zero cost."""
    await _await_warmup()
    svc = _library_admin_service()
    return json.dumps(svc.whoami(), ensure_ascii=False)


@server.tool(structured_output=False)
async def library_use(library: str, client: str = "") -> str:
    """Switch this session's sticky library.
After a successful switch, subsequent tools use this library without needing
``library=`` on every call. An unresolvable name fails loudly; it never silently
falls back to another library.

Args:
    library: Library name, alias, ``lib_*`` ID, or path.
    client: Caller identity; required on shared connections.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # Unresolvable / target is a backup ⇒ the service raises LibraryResolutionError (never swallowed, never falls back to the default library).
    return json.dumps(svc.use(library=library, client=client),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def library_create(name: str, path: str = "",
                         aliases: str = "", use: bool = False,
                         client: str = "", workspace_root: str = "") -> str:
    """Create a new memory library and optionally switch to it.
Creation is explicit: only this operation writes a manifest/initial storage.
Duplicate names and ambiguous paths fail rather than guessing.

Args:
    name: Library name.
    path: Optional explicit path; omitted uses the configured library root.
    aliases: Optional comma-separated aliases.
    use: Switch the current session to the new library after creation; default true.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # Duplicate name / empty name / target exists and is not a directory ⇒ the service raises LibraryResolutionError.
    return json.dumps(svc.create(name=name, path=path, aliases=aliases,
                                 use=use, client=client,
                                     workspace_root=workspace_root), ensure_ascii=False)


# ------------------------------------------------------------------
# Dispatcher (progressive disclosure): implementation lives in dispatch.py, the
# registration block stays in this module
# ------------------------------------------------------------------
# _dispatch_map (fetches tool functions via vars(server)) / _hippo_help /
# _hippo_extra: see server_runtime/dispatch.py (re-exported via the imports
# above); the name list is fetched live from _TOOL_GROUPS (the old
# _ALL_TOOL_NAMES snapshot moved along and became a live derivation), taking
# effect immediately after patching. Registered only when _TOOL_PROFILE != "full",
# for two reasons (original text moved to dispatch.py): under full the dispatcher
# is dead weight; add_tool's programmatic registration keeps "declared tool surface = 46".

if _TOOL_PROFILE != "full":
    server._tool_manager.add_tool(
        _hippo_help,
        name="hippo_help",
        description=(
            "List extension actions hidden in this session but callable via hippo_extra, with their arguments "
            "(export / lexicon / entity graph / ops / state access, etc.). "
            "When unsure whether a memory capability exists or you cannot find the tool you need, call this first."
        ),
        structured_output=False,
    )
    server._tool_manager.add_tool(
        _hippo_extra,
        name="hippo_extra",
        description=(
            "Call a tool not directly exposed in this session by action name (export / lexicon / entity graph / ops, etc.). "
            'Call hippo_help first for the action list and argument names, then invoke via hippo_extra(action="...", '
            'args=\'{"argument": value}\')'
        ),
        structured_output=False,
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> None:
    """Start the MCP stdio server (startup sequence in entry.run_main)."""
    _entry.run_main()


def __getattr__(name: str) -> Any:
    """Resolve heavy symbols on demand (PEP 562).

    This module deliberately does not import ``brain`` / ``manager`` at the top
    level (that would push torch's ~5.7s onto the MCP handshake path; see the
    import notes in the file header). To keep existing references like
    ``server.HippoBrain`` working, the real import happens **on first attribute
    read** — not reading costs nothing, and ``get_brain()`` goes through
    ``_brain_class()``.
    """
    if name == "HippoBrain":
        from hippocampus_mcp.brain import HippoBrain

        return HippoBrain
    if name == "MemoryManager":
        from hippocampus_mcp.engine.manager import MemoryManager

        return MemoryManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
