# -*- coding: utf-8 -*-
"""Memory manager (MemoryManager).

Orchestrates the LNN dynamic memory core, vector store, scoring engine and
sampler into a unified memory service. Exposes store / recall / get / update /
delete / save / load / reset and related operations; it is the bridge between
the MCP tool layer and the underlying components.

Architecture mapping (hippocampus metaphor):
    - MemoryCore  -> CA3: temporal integration, generates query vector h(t)
    - VectorStore -> CA1/cortex: long-term memory storage
    - ScoringEngine -> hippocampal recall mechanism: similarity x time decay x importance
    - Sampler     -> exploratory recall: distance-weighted sampling avoids local optima
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

from ..config import LNN_CTX_MAX_DEFAULT, load_config  # noqa: F401 (LNN_CTX_MAX_DEFAULT re-exported)
from ..common.recency import (
    in_time_window,
    iso_date,
    out_of_window,
    weekday_abbr,
)
from ..store import mem_history as _mem_history
from ..store.idem import IdemIndex, normalize_key


def _stderr_print(*args, **kwargs) -> None:
    """Diagnostic output always goes to stderr.

    **Under MCP stdio mode stdout is the JSON-RPC protocol channel**: anything
    written to stdout gets parsed by the client as a protocol message, which
    yields "Invalid JSON" errors and may drop the connection outright. This
    module's warnings (vector backfill, training-weight load failures, metadata
    fallback, etc.) all happen on the service startup path, hence stderr.
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)

logger = logging.getLogger(__name__)

from .embeddings import (
    ENV_ALLOW_MODEL_SWITCH,
    backend_kind,
    create_embedder,
    embedder_pool_stats,
    embedding_coverage,
)
from ..lnn import MemoryCore, StateCheckpoint
from ..retrieval import DistanceWeightedSampler, ScoringEngine
# C5 intent routing (default off) and C6 retrieval-time decay (default off) -- see retrieval/intent.py
from ..retrieval.intent import (
    INTENT_WEIGHT_MULTIPLIERS,
    classify_intent,
    intent_routing_enabled,
)


def _recall_decay_enabled() -> bool:
    """Whether C6 retrieval-time memory decay is enabled (default off; reads env
    every call so tests can flip it).

    M1: now goes through the config single read point (whitelist semantics,
    see config._BOOL_SEMANTICS B).
    """
    return load_config().recall_decay


def _recall_decay_tau() -> float:
    """C6 decay time constant tau (seconds). Defaults to 90 days, same basis as lambda_decay."""
    tau = load_config().recall_decay_tau
    return tau if tau > 0 else 7776000.0
from ..store import (
    CognitiveType,
    MemoryRecord,
    MemoryState,
    MemoryType,
    VectorStore,
    merge_id_lists,
    merge_tag_dicts,
)


def _seed_store_config() -> None:
    """Production injection point for the store/lnn/retrieval knobs.

    The G1 dependency matrix only allows engine/brain to import config; the
    16 consolidated store/lnn/retrieval knobs get the string transport values
    from ``load_config()`` poured into their injection seams (``apply_config``)
    at import time by the **sole production constructor** (this module).
    Env direct reads at consumption points are demoted to a "direct-connection
    fallback" (env first, injected value as fallback) -- tests that use
    store/retrieval directly without going through this module keep a
    zero-change surface; parsing (float/whitelist) stays at the consumption
    point, zero drift. See the common/config_bridge module docstring for details.
    """
    from ..lnn import memory_core as _lnn_mod
    from ..retrieval import intent as _intent_mod
    from ..store import backend as _backend_mod
    from ..store import vector_store as _vs_mod

    c = load_config()
    _vs_mod.apply_config(
        snapshot_min_hours=c.snapshot_min_hours,
        multi_writer=c.multi_writer,
        save_lock_timeout=c.save_lock_timeout,
        matrix_dtype=c.matrix_dtype,
        vectors_mmap=c.vectors_mmap,
        allow_stale_save=c.allow_stale_save,
    )
    _backend_mod.apply_config(store_backend=c.store_backend, fts5=c.fts5)
    _lnn_mod.apply_config(lnn_backend=c.lnn_backend)
    _intent_mod.apply_config(intent_routing=c.intent_routing)


_seed_store_config()


class EmbedderModelMismatchError(RuntimeError):
    """The explicitly configured embedding model does not match the vectors
    already in the library, and switching was not allowed.

    This is the hard block against "**the service silently using the wrong
    model**" class of incident: rather than recompute the whole library with
    the wrong model (thousands of entries take tens of seconds to minutes), or
    worse -- quietly retrieve with another model's vectors (hit rate drops to
    zero with nothing visible) -- fail explicitly and spell out the remedies.

    The "library's model" is decided by :meth:`VectorStore.preferred_model_key`:
    the persisted ``model_key`` first, then the non-``__legacy__`` slot with
    the highest coverage.

    To deliberately compare retrieval quality of different models on the same
    data, set ``HIPPOCAMPUS_ALLOW_MODEL_SWITCH=1`` to explicitly allow it
    (equivalent to the old "backfill" behavior).
    """


class RecallResult:
    """Wrapper for a single recall result."""

    def __init__(
        self,
        memory_id: str,
        content: str,
        score: float,
        similarity: float,
        timestamp: float,
        importance: float,
        recall_count: int,
        memory_type: str,
    ) -> None:
        self.memory_id = memory_id
        self.content = content
        self.score = score
        self.similarity = similarity
        self.timestamp = timestamp
        self.importance = importance
        self.recall_count = recall_count
        self.memory_type = memory_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": round(self.score, 4),
            "similarity": round(self.similarity, 4),
            "timestamp": self.timestamp,
            # Time-derived fields (2026-11-05): same contract as brain.recall's
            # response body. The production path (MCP tools) goes through
            # manager.recall, not recall_flow; if these were not added here the
            # host would get results **without** dates -- the two paths' field
            # surfaces would diverge, which is exactly how "looks good in the
            # benchmark, missing on the real host" bugs are born.
            "occurred_at": iso_date(self.timestamp),
            "weekday": weekday_abbr(self.timestamp),
            "importance": self.importance,
            "recall_count": self.recall_count,
            "memory_type": self.memory_type,
        }


# ==================================================================
# Before dedup-merging, check "would we lose hard information"
# ==================================================================

# Arabic numerals (incl. decimals, thousands separators). No percent sign --
# in "0.1—0.4%" the % is a unit, not information.
_RE_ARABIC = re.compile(r"\d+(?:[.,]\d+)*")
# Chinese numeral runs: **only sequences of length >= 2 are collected**. Single
# characters ("一" "二") are overwhelmingly particles/classifiers in Chinese
# ("一样" "再一次"); collecting them would make "almost every entry carry hard
# info" and dedup effectively useless -- that is equivalent to turning dedup
# off by the back door, the biggest misuse risk of this scheme.
_RE_CN_NUM = re.compile(r"[零〇一二三四五六七八九十百千万亿两]{2,}")
# Dates: 2026-09-21 / 2026年9月21日 / 9/21 / 八月中 / 己丑年
_RE_DATE = re.compile(
    r"\d{2,4}\s*[-/年.]\s*\d{1,2}(?:\s*[-/月.]\s*\d{1,4}\s*日?)?"
    r"|[〇零一二三四五六七八九十]{1,3}\s*月(?:\s*[〇零一二三四五六七八九十]{1,3}\s*日)?"
    r"|[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]\s*年?"
)


def extract_hard_info(text: str) -> set[str]:
    """Extract the **decidable hard information** in the text: numbers and dates.

    Why only these two categories: on merge ``content`` keeps the **longer**
    one and the shorter side is dropped wholesale, and the most painful loss is
    precisely "the number the old text did not have" -- a measured case is
    「燕行日程·约二千余里」vanishing completely after being swallowed by the
    0.85 threshold.

    These two categories are **decidable**: whether there is a new number is
    something a machine can compute accurately; "whether it is semantically a
    new fact" is not machine-decidable, that belongs to supersede / neighbors.
    Prefer false negatives (a few fewer duplicates built) over silently losing
    numbers -- lost numbers are unrecoverable.
    """
    if not text:
        return set()
    out: set[str] = set()
    for m in _RE_ARABIC.finditer(text):
        out.add(m.group(0))
    for m in _RE_CN_NUM.finditer(text):
        out.add(m.group(0))
    for m in _RE_DATE.finditer(text):
        out.add(m.group(0).strip())
    return out


# ==================================================================
# LNN temporal hidden-state partitioning
# ==================================================================

# Fallback route when there is no client info. All unnamed calls share it --
# equivalent to the pre-refactor behavior.
DEFAULT_LNN_KEY = "__default__"

# Max number of context routes kept at once (LRU). Each is ~131 KB
# (hx + window), 8 of them ~= 1 MB, a negligible cost; over the limit the
# **least recently used** route is evicted, never one in active use.
# M1: the constant moved up to config.py (L0 base); re-exported here so
# out-of-package references do not break.


def lnn_ctx_max_from_env() -> int:
    """Read the LNN route cap (through the config single read point; >0 keeps the
    current semantics)."""
    v = load_config().lnn_ctx_max
    return v if v > 0 else LNN_CTX_MAX_DEFAULT


class LnnContext:
    """One LNN temporal context (one per client).

    Packs the four kinds of **process-level runtime state**: CfC hidden state
    ``hx``, last step output ``last_output``, sliding window ``window``, step
    count ``step_count``, plus the time base ``reference_time``. These are all
    the things that get cross-contaminated under shared connections -- memories
    / entities / profile / summary / open intents / pin slots are meant to be
    shared across sessions and are not part of it.
    """

    __slots__ = ("hx", "last_output", "window", "step_count",
                 "reference_time", "window_maxlen")

    def __init__(self, window_maxlen: int | None = None) -> None:
        self.hx = None
        self.last_output = None
        self.window_maxlen = window_maxlen
        self.window = deque(maxlen=window_maxlen)
        self.step_count = 0
        self.reference_time: float | None = None

    def snapshot(self) -> dict:
        """Lightweight snapshot for self-reporting (no tensor copies, shapes only)."""
        return {
            "step_count": self.step_count,
            "window_size": len(self.window),
            "has_state": self.hx is not None,
            "reference_time": self.reference_time,
        }

class MemoryManager:
    """Memory manager: orchestrates LNN + vector store + scoring + sampling.

    Args:
        embedding_dim: embedding dimension, default 384 (**only effective for
            the hash backend**; with sentence_transformer the dimension is
            decided by the model and gets overridden by the actual model).
            With the hash backend it must be set to 32 manually.
        n_neurons: CfC neuron count, default 512 (must be > output_dim + 2).
        window_size: LNN sliding window, default 5.
        persist_dir: persistence directory path. None means no persistence.
        lambda_decay: time decay constant (seconds). Default 7776000
            (90 days, LoCoMo-tuned).
        mu_importance: importance decay constant (seconds). Default 15552000
            (180 days, LoCoMo-tuned).
        embedder_backend: embedding backend, "sentence_transformer" (default)
            or "hash".
        w_sim: additive-formula similarity weight, default 0.95.
        w_time: additive-formula time-decay weight, default 0.03.
        w_recall: additive-formula recall-frequency weight, default -0.02
            (negative = diversity penalty).
        auto_save: whether to persist automatically after each store, default False.
        backend: storage backend. Currently only "numpy" (VectorStore) is
            supported. Any other value raises **explicitly** instead of quietly
            switching to a data-losing path.
            (An experimental ChromaDB-based implementation once existed; testing
            found 6 semantic differences that would silently corrupt data, and
            it was removed from the codebase; the reasons are kept in the error
            message below for reference when the backend is reimplemented.)
        dedup_threshold: dedup-merge threshold, default 0.85. Set 1.0 to disable dedup.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        n_neurons: int = 512,
        window_size: int = 5,
        persist_dir: str | None = None,
        lambda_decay: float = 7776000.0,
        mu_importance: float = 15552000.0,
        embedder_backend: str = "sentence_transformer",
        embedder_model: str = "",
        w_sim: float = 0.95,
        w_time: float = 0.03,
        w_recall: float = -0.02,
        auto_save: bool = False,
        backend: str = "numpy",
        dedup_threshold: float = 0.85,
        recency_window_days: int = 0,
        w_strength: float = 0.03,
        mem_history_ring: int = 200,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.embedder_backend = embedder_backend
        self.auto_save = auto_save
        self.backend = backend
        self.dedup_threshold = dedup_threshold
        # O8 retrieval-side staleness window (days): age > window and not pinned
        # -> excluded from candidates. **Default 0 = off** -- archived old
        # memories must remain reachable by similar queries; output length is
        # controlled by top_k truncation, not by time-based death sentences.
        # **Zero state changes** -- no deletion, no archiving, no state change,
        # output gating only. The MCP recall tool reaches this method directly
        # via RecallService (not through brain.recall), so the window must take
        # effect at this layer for full coverage.
        self._recency_window_seconds = max(int(recency_window_days), 0) * 86400.0
        # Rewrite-history ledger (mem_history.jsonl, sidecar): ring injected at
        # construction (discipline: store-side functions only take explicit
        # arguments); line count/seq scanned lazily -- one disk read on first
        # write; pure in-memory brain (persist_dir=None) bypasses it entirely.
        self.mem_history_ring = max(int(mem_history_ring), 0)
        self._mem_history_lines: int | None = None
        self._mem_history_last_seq: int = 0

        # Embedder: default is sentence-transformers semantic embedding.
        # Always goes through create_embedder(), which validates the embedder
        # protocol (embed/embed_query/embed_batch/model_key) -- a missing method
        # errors at initialization instead of raising AttributeError at recall
        # time (coverage report D5).
        #
        # Model-name resolution no longer branches here: the (possibly empty)
        # name is handed to the embedder as-is, resolve_embedder_model() decides
        # in one place "argument > env > default" and records the **source** on
        # embedder.model_source. "Some entry point forgot to set env and it
        # silently ran as MiniLM" is exactly how it used to happen; with
        # centralized resolution there is no second place to get it wrong.
        if embedder_backend == "sentence_transformer":
            self.embedder = create_embedder(
                "sentence_transformer", model_name=embedder_model or ""
            )
            # Actual dimension is decided by the model; sync it
            self.embedding_dim = self.embedder.dim
        else:
            self.embedder = create_embedder(embedder_backend, dim=embedding_dim)

        # Constructor arguments kept for later: when the library's model differs
        # from this process's and "follow the data" is needed, the LNN core must
        # be rebuilt with the new dimension (see _follow_store_model).
        self._ctor_n_neurons = n_neurons
        self._ctor_window_size = window_size
        # Report of the most recent "follow-the-data model switch" (None = never
        # happened), for upper-layer logs/test assertions
        self.model_follow: dict[str, Any] | None = None

        # LNN dynamic memory core
        # AutoNCP requires output_dim < n_neurons - 2
        actual_n_neurons = max(n_neurons, self.embedding_dim + 3)
        self.memory_core = MemoryCore(
            input_dim=self.embedding_dim,
            n_neurons=actual_n_neurons,
            output_dim=self.embedding_dim,
            window_size=window_size,
        )

        # Vector store: the only supported backend today is numpy (VectorStore).
        # model_key identifies the currently active embedding model:
        # switching model = switching vector slot, no document re-import needed
        self._model_key = self.embedder.model_key
        if backend != "numpy":
            raise ValueError(
                f"Storage backend {backend!r} is not wired up. The only supported backend is 'numpy'.\n"
                f"An experimental ChromaDB-based implementation (backend='chroma') once existed; testing showed the following defects, "
                f"so wiring it back in would silently corrupt data -- the implementation was removed and this refusal kept:\n"
                f"  1. after switching embedding models, ChromaDB vectors from the old model get tagged with the current model name; "
                f"missing_ids() returns empty -> never backfilled -> retrieval uses wrong vectors (silent pollution);\n"
                f"  2. save/load round-trips lose source_ids / entity_ids -> the L0 audit chain and entity index break;\n"
                f"  3. update() does not sync pin state (related fixes do not cover that path);\n"
                f"  4. no last_load_issue / corruption quarantine (related fixes do not cover that path);\n"
                f"  5. 构造即自动加载，与其他后端\"显式 load()\"的时序不一致。\n"
                f"Once all five are fixed with dual-backend equivalence regression, this restriction can be lifted."
            )
        store_path = None
        if persist_dir:
            import os
            os.makedirs(persist_dir, exist_ok=True)
            store_path = os.path.join(persist_dir, "memories.json")
        self.store = VectorStore(
            dim=self.embedding_dim,
            persist_path=store_path,
            model_key=self._model_key,
        )

        # Scoring engine: additive formula (best LoCoMo benchmark configuration)
        self.scorer = ScoringEngine(
            lambda_decay=lambda_decay,
            mu_importance=mu_importance,
            mode="additive",
            w_sim=w_sim,
            w_time=w_time,
            w_recall=w_recall,
            w_strength=w_strength,
        )
        self.sampler = DistanceWeightedSampler(beta=1.0, epsilon=0.1)

        # Persistence
        self.persist_dir = persist_dir
        self._checkpoint: StateCheckpoint | None = None
        if persist_dir:
            import os
            # H7: primary format is npz (zero torch I/O); old libraries'
            # lnn_state.pt is read by StateCheckpoint for compatibility
            # (lazy torch), so startup needs no migration.
            ckpt_path = os.path.join(persist_dir, "lnn_state.npz")
            self._checkpoint = StateCheckpoint(ckpt_path)

        # Steps processed in the current session (process-level total, used by
        # auto_save and checkpoint; after partitioning each route keeps its own
        # step_count, see LnnContext)
        self._step_count: int = 0

        # "Flush L0 before persisting L1" hook (R2 / L0 invariant, semantics in
        # :meth:`save`). Why the hook lives on manager instead of teaching
        # engine about L0: the four entry points auto_save / save_state /
        # retry_save / session_end all funnel into save(), so one hook here
        # covers every path -- patching per path means patching the same bug
        # four times.
        self._pre_persist: Callable[[], None] | None = None

        # LNN hidden-state partition containers. All unnamed calls land on
        # DEFAULT_LNN_KEY, so **without wiring the behavior is bit-for-bit
        # identical to the pre-refactor one**.
        self._lnn_contexts: dict[str, LnnContext] = {}
        self._lnn_ctx_lru: list[str] = []
        self._lnn_ctx_max = lnn_ctx_max_from_env()

        # Reference time: use the newest timestamp of stored messages, not
        # wall-clock (LoCoMo benchmark found time.time() diverges hugely from
        # the data timestamps, which nullifies time decay)
        self._reference_time: float | None = None

        # Auto-save health (zcode feedback: when a save is rejected for
        # concurrency the host has no idea -- it believes ingest is already
        # persisted and only discovers the lost batch after a restart).
        # See the auto-save branch in store_memory(): failure does not block
        # the write, but it must leave a trace.
        self.last_save_error: str | None = None
        # C5 self-report: intent classification of the most recent recall
        # (always None when routing is off)
        self.last_recall_intent: str | None = None
        self.autosave_failures: int = 0
        # Whether the last store_memory hit dedup (and merged into which existing
        # memory), see store_memory
        self.last_store_merge_of: str | None = None
        # C10: when dedup was hit but **not merged because of new hard info**,
        # the near-duplicate existing memory. Mutually exclusive with
        # last_store_merge_of: this field non-empty == this call really created.
        self.last_store_near_miss: dict[str, Any] | None = None
        # C13: truncation self-report for this write's content (embedding model
        # token hard cap). When truncated is True, the original text after
        # embedded_chars is invisible to vector retrieval.
        self.last_truncation: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # LNN context routing
    # ------------------------------------------------------------------
    def lnn_key(self, client: str | None) -> str:
        """Resolve which route this call uses. Three levels: explicit client >
        default route.

        The second level (``ctx.session``: one ServerSession per TCP connection
        under the daemon) is resolved by the server layer at wiring time and
        passed in as ``client``; the manager only trusts explicit values --
        the protocol layer has no access to the session, so no guessing here.
        """
        c = (client or "").strip()
        return c or DEFAULT_LNN_KEY

    def _lnn_ctx(self, client: str | None) -> LnnContext:
        """Get (or create) one route's context and move it to the LRU tail
        (most recently used)."""
        key = self.lnn_key(client)
        ctx = self._lnn_contexts.get(key)
        if ctx is None:
            ctx = LnnContext(
                getattr(getattr(self.memory_core, "window", None), "maxlen", None)
            )
            self._lnn_contexts[key] = ctx
        if key in self._lnn_ctx_lru:
            self._lnn_ctx_lru.remove(key)
        self._lnn_ctx_lru.append(key)
        # LRU eviction: only evict the **least recently used** route, and never
        # one currently in flight
        while len(self._lnn_contexts) > self._lnn_ctx_max and len(self._lnn_ctx_lru) > 1:
            victim = self._lnn_ctx_lru.pop(0)
            self._lnn_contexts.pop(victim, None)
        return ctx

    @contextmanager
    def _lnn_route(self, client: str | None):
        """Install one route's hidden state into memory_core and restore it
        verbatim on exit.

        ``MemoryCore`` is a **singleton** (there is only one set of weights,
        which is meant to be shared); only hidden state is per-route runtime
        data -- so what happens here is "swap state", not "swap model". The
        context manager guarantees the state is returned even on exception
        paths.
        """
        ctx = self._lnn_ctx(client)
        core = self.memory_core
        saved = (core.hx, core.last_output, core.window, self._reference_time)
        core.hx = ctx.hx
        if ctx.last_output is not None:
            core.last_output = ctx.last_output
        core.window = ctx.window
        self._reference_time = ctx.reference_time
        try:
            yield ctx
        finally:
            ctx.hx = core.hx
            ctx.last_output = core.last_output
            ctx.window = core.window
            ctx.reference_time = self._reference_time
            core.hx, core.last_output, core.window, self._reference_time = saved
            self._touch_global_reference_time()

    def _touch_global_reference_time(self) -> None:
        """Align the process-level ``_reference_time`` to the max across routes.

        consolidate / save / load read the process-level copy; after
        partitioning it must still mean "newest timestamp ever seen", otherwise
        old memories get permanently exempt from archiving (historical incident,
        see architecture doc section 8.4).
        """
        vals = [c.reference_time for c in self._lnn_contexts.values()
                if c.reference_time is not None]
        if vals:
            mx = max(vals)
            if self._reference_time is None or mx > self._reference_time:
                self._reference_time = mx

    def lnn_context_report(self) -> dict:
        """Self-report: how many context routes exist and each one's step count."""
        return {
            "count": len(self._lnn_contexts),
            "max": self._lnn_ctx_max,
            "keys": {k: self._lnn_contexts[k].snapshot()
                     for k in self._lnn_ctx_lru},
        }

    # ---------------------------------------------------------------- idempotency index
    def _idem_index(self) -> IdemIndex:
        """Get the idempotency inverted index (rebuilt from in-memory records on
        first use, **no extra disk**).

        The index is derived from ``MemoryRecord.idem``: the first keyed write
        after a process restart scans the records once. A linear scan over ten
        thousand entries is ~ms-scale, and it only happens on libraries that
        actually use idempotency keys -- without this feature it is **zero
        cost, zero memory**.
        """
        idx = getattr(self, "_idem", None)
        if idx is None:
            idx = IdemIndex()
            # M3 port discipline: use iter_records, never touch _records
            idx.rebuild(self.store.iter_records())
            self._idem = idx
        return idx

    def _remember_idem(self, idem_key: str, memory_id: str) -> None:
        """Register the idempotency mapping after a successful write (empty key /
        empty id are always ignored)."""
        if idem_key and memory_id:
            self._idem_index().remember(idem_key, memory_id)

    def idem_lookup(self, key: str) -> str | None:
        """Idempotency-key lookup: returns the existing memory id; miss / dangling
        both return ``None``.

        Used by ``brain.ingest`` / ``WriteService`` to decide **before writing**
        (a pre-write decision avoids leaving even an L0 entry). Dangling
        receipts (the record was truly deleted by ``forget()``) are pruned here
        automatically -- returning an id that no longer exists in the library is
        worse than a duplicate write: the host would believe the memory is
        still there.
        """
        if not key:
            return None
        idx = self._idem_index()
        hit = idx.lookup(key)
        if not hit:
            return None
        if self.store.get(hit) is not None:
            return hit
        idx.forget(key)
        return None

    def store_memory(
        self,
        content: str,
        importance: float = 0.5,
        memory_type: str = "general",
        timestamp: float | None = None,
        cognitive_type: str = "world",
        source_ids: list[str] | None = None,
        entity_ids: list[str] | None = None,
        state: str = "active",
        source: str = "",
        tags: dict[str, Any] | None = None,
        client: str | None = None,
        quarantine: bool = False,
        sensitivity: str = "",
        origin: dict[str, str] | None = None,
        idem_key: str = "",
    ) -> str:
        """Store a new memory.

        Flow: text -> embedding -> LNN step (updates temporal state) -> write
        to the vector store.

        Args:
            content: memory text content.
            importance: importance [0, 1], default 0.5.
            memory_type: memory type (episodic/semantic/procedural/general).
            timestamp: custom timestamp; None uses the current time.
            cognitive_type: cognitive type (experience/world/opinion), added in v0.3.
            source_ids: list of ids pointing back to the L0 raw layer, added in v0.3.
            entity_ids: list of associated entity ids, added in v0.3.
            state: lifecycle state (active/superseded/archived), added in v0.3.
            source: source label (file name/relative path), passed through to
                recall hit results so "which document this memory came from" is
                traceable (coverage report, high 3).
            tags: structured tags, e.g. ``{"章": 43, "年": "己丑"}``, for
                recall/get_memory_context conditional filtering. On a dedup-hit
                merge they are **unioned** (same principle as
                source_ids/entity_ids): the merged-in content may carry tags for
                different chapters; dropping them would make this hybrid record
                unfindable by chapter forever.
            quarantine: D2a write-side quarantine flag (default False = status
                quo). True = pending review: stays out of the recall pool and
                the user_scope slot, and is promoted through the managed channel
                after drift_audit review. The single source of truth for the
                criterion is domain/write_quarantine.should_quarantine; this
                layer only stores the flag and makes no judgment.
            sensitivity: D1 privacy label (default empty string = unlabeled).
                "personal" / "sensitive:<category>"; stored and echoed with the
                card only, not consumed at this layer.

        Returns:
            The memory's unique ID. **Note**: on a dedup hit (similarity >=
            ``dedup_threshold`` against some active memory) nothing new is
            created; the write merges into that memory and its id is returned --
            in that case ``source_ids`` / ``entity_ids`` are unioned,
            ``content`` keeps the longer one, ``timestamp`` the newest,
            ``importance`` the max, and type fields are replaced by the new
            value only when the original was the default.
        """
        if timestamp is None:
            timestamp = time.time()

        # Update the reference time (used for scoring at recall)
        # Recorded per-route; the process-level global still takes the max
        # across routes (_touch_global_reference_time), so consolidate / save
        # keep the same contract as before the refactor.
        _ctx = self._lnn_ctx(client)
        if _ctx.reference_time is None or timestamp > _ctx.reference_time:
            _ctx.reference_time = timestamp
            self._touch_global_reference_time()

        # Whether this call hit dedup, and merged into which existing memory.
        # The return value keeps the old "id only" signature; when the host
        # (ingest) needs to distinguish "created / merged", it reads this
        # attribute -- in dual-write scenarios "I thought I wrote a new entry
        # but it merged into an old one" must be visible.
        self.last_store_merge_of: str | None = None
        self.last_store_near_miss = None

        # P0-2 idempotency: a host-declared replay of "the same request" =>
        # no embedding, no write, no creation. The lookup is deliberately placed
        # **before** embedding: what a replay should skip is precisely the most
        # expensive step.
        self.last_idem_hit: str | None = None
        if idem_key:
            _hit = self.idem_lookup(idem_key)
            if _hit:
                self.last_idem_hit = _hit
                return _hit

        # Generate embedding
        emb = self.embedder.embed(content)

        # C13: truncation self-report. Embedding models have a token hard cap
        # (BGE / MiniLM = 512); the excess is silently dropped and contributes
        # nothing to the vector: the write succeeds, the record reads back, stats
        # look normal -- **only retrieval is quietly degraded**. Here the
        # "how many characters actually entered the embedding" is computed
        # explicitly and echoed back to the caller by ingest / store_memory.
        self.last_truncation = embedding_coverage(self.embedder, content)

        # Phase 5 step 5: memory dedup -- merge instead of creating when
        # similarity > threshold
        if self.dedup_threshold < 1.0 and self.store.count() > 0:
            candidates = self.store.search(emb, top_k=1)
            if candidates:
                existing, sim = candidates[0]
                # Merge only into **still valid** memories. Superseded /
                # archived records are semantically not "the same current fact";
                # swallowing the new write would make it invisible from birth.
                # That case should create new, then let the upper layer's fact
                # update detection decide the supersede relation.
                # ---- merge information-loss guard--------------------------
                # Merging keeps only the "longer" one -- the shorter new text is
                # **dropped wholesale**. If the new text brings numbers/dates the
                # old text lacked, that drop is unrecoverable information loss,
                # so no merge: create separately, and leave the near-duplicate
                # notice in last_store_near_miss.
                _is_dup = (
                    sim >= self.dedup_threshold
                    and existing.state == MemoryState.ACTIVE.value
                    # D2a: quarantined writes are **never merged** -- merging
                    # would fold unreviewed content into a visible card (the
                    # quarantine flag is dropped on merge), i.e. bypassing
                    # review. Quarantined cards are always created separately
                    # and re-reviewed by drift_audit.
                    and not quarantine
                )
                _missing_hard: set[str] = (
                    extract_hard_info(content) - extract_hard_info(existing.content)
                ) if _is_dup else set()
                if _is_dup and _missing_hard:
                    self.last_store_near_miss = {
                        "memory_id": existing.id,
                        "similarity": round(float(sim), 4),
                        "missing_hard_info": sorted(_missing_hard)[:12],
                        "note": (
                            "the new text contains numbers/dates absent from the old text -> not merged per C10; "
                            "created as a separate record (merging would lose that information)"
                        ),
                        "content": existing.content[:200],
                    }
                if _is_dup and not _missing_hard:
                    merged_content = (
                        content if len(content) > len(existing.content)
                        else existing.content
                    )
                    # The swallowed text must be captured **before** store.update --
                    # update mutates existing in place, so reading
                    # existing.content afterwards yields the surviving text and
                    # "who was swallowed" can no longer be computed.
                    discarded_content = (
                        existing.content
                        if merged_content == content else content
                    )
                    merged_importance = max(importance, existing.importance)
                    if merged_content != existing.content:
                        merged_emb = self.embedder.embed(merged_content)
                    else:
                        # Content stays the **old text** -> must keep the old
                        # text's vector. Previously this assigned emb (the new
                        # text's vector), producing "content unchanged but vector
                        # swapped": the record and its vector decoupled, the write
                        # succeeds, the record reads back, stats look normal --
                        # only retrieval quietly degrades. Measured consequence:
                        # two cards with different content persisted
                        # element-identical vectors (cos=1.000000), one of which
                        # had only 0.8908 against its own text.
                        merged_emb = self.vector_of(existing)
                        if merged_emb is None:
                            # The old record has no vector under the current
                            # model slot (should not happen); fall back to this
                            # call's vector -- better than persisting a
                            # vectorless record.
                            merged_emb = emb
                    self.store.update(
                        existing.id,
                        content=merged_content,
                        embedding=merged_emb,
                        timestamp=max(timestamp, existing.timestamp),
                        importance=merged_importance,
                        # Defect D3: the four fields below used to be **dropped
                        # entirely**. Consequences:
                        #   (1) this call's L0 raw text lost its source_ids
                        #       back-reference and became a permanent orphan
                        #       (the "verbatim text can be cross-checked" audit
                        #       chain silently broke, and forget's cascade
                        #       cleanup did not cover it)
                        #   (2) entities mentioned in this call lost their link
                        #       to the memory, breaking the entity inverted index
                        # So a merge must carry sources and indexes over
                        # (ordered union, not overwrite).
                        source_ids=merge_id_lists(existing.source_ids, source_ids),
                        entity_ids=merge_id_lists(existing.entity_ids, entity_ids),
                        # Type takes the "more specific" side: the new value is
                        # adopted only when the original is still the default,
                        # so a later coarse classification cannot wash away an
                        # already-decided specific type.
                        memory_type=(
                            existing.memory_type
                            if existing.memory_type != MemoryType.GENERAL.value
                            else memory_type
                        ),
                        cognitive_type=(
                            existing.cognitive_type
                            if existing.cognitive_type != CognitiveType.WORLD.value
                            else cognitive_type
                        ),
                        # The source label must not be swallowed either: the
                        # merged-in content may come from another file; dropping
                        # source would make this hybrid record's file lineage
                        # untraceable. Single-value fields take "the pre-existing
                        # one, or the new one if empty".
                        source=existing.source or source,
                        # Tags are unioned too: the merged-in content may carry
                        # other chapter/year tags; overwriting would make this
                        # hybrid record unfindable under "filter by chapter".
                        tags=merge_tag_dicts(existing.tags, tags),
                        # O1: merge keeps the max of both sides' retrieval
                        # strength -- the new trajectory is always 0, so this is
                        # equivalent to keeping the old trace's consolidation
                        # gains; a merge never loses strength.
                        recall_strength=max(
                            float(existing.recall_strength or 0.0), 0.0),
                    )
                    # Merge keeps the prior value: the swallowed side's text
                    # enters the ledger (both directions are recorded --
                    # "new longer" swallows the old text, "old longer" swallows
                    # the new text; when both texts are identical nothing is
                    # recorded since nothing was lost). after holds the
                    # surviving text.
                    if discarded_content != merged_content:
                        self._mem_history_append({
                            "kind": "merge",
                            "memory_id": existing.id,
                            "before": {"content": discarded_content},
                            "after": {"content": merged_content},
                        })
                    self.last_store_merge_of = existing.id
                    # Register the idempotency key as well: the merge path also
                    # counts as "this request got a receipt", otherwise a replay
                    # would redo the full match (same result but a wasted
                    # embedding).
                    self._remember_idem(idem_key, existing.id)
                    return existing.id

        # LNN step: update hidden state (temporal memory)
        # What advances is **this route's** hidden state. Under a shared
        # connection without partitioning, A's write would directly overwrite
        # B's retrieval starting point -- no exception, no log, just degraded
        # temporal scores.
        # H7: MemoryCore takes numpy (the backend handles dtype/shape
        # internally); the process is torch-free.
        emb_input = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        with self._lnn_route(client) as _rctx:
            self.memory_core.step(emb_input)
            _rctx.step_count += 1
        self._step_count += 1

        # Write to the vector store (vector attached under the current model slot)
        record = MemoryRecord(
            id="",
            content=content,
            timestamp=timestamp,
            embeddings={self._model_key: emb.tolist()},
            importance=importance,
            recall_count=0,
            memory_type=memory_type,
            cognitive_type=cognitive_type,
            source_ids=list(source_ids) if source_ids else [],
            entity_ids=list(entity_ids) if entity_ids else [],
            state=state,
            source=source or "",
            tags=dict(tags or {}),
            quarantine=bool(quarantine),
            sensitivity=str(sensitivity or ""),
            origin=dict(origin or {}),
            idem=normalize_key(idem_key),
        )
        memory_id = self.store.add(record)
        self._remember_idem(idem_key, memory_id)

        # Auto-persist (optional)
        if self.auto_save and self._step_count % 10 == 0:
            try:
                self.save()
                self.last_save_error = None
            except Exception as e:
                # Does not block the main flow -- but **must never be silent**:
                # when the concurrent-write guard rejects the save, this batch
                # of memories exists only in memory. The host sees ingest
                # succeed, believes it is persisted, and loses everything on
                # restart with no trace to inspect (zcode feedback).
                # Record it here; get_stats / save_state expose it explicitly.
                self.last_save_error = f"{type(e).__name__}: {e}"
                self.autosave_failures += 1
                logger.warning(
                    "auto-save failed (the batch of memories is still in memory; call save_state or "
                    "retry_save explicitly to retry): %s", self.last_save_error,
                )

        return memory_id

    def recall(
        self,
        query: str | None = None,
        top_k: int = 5,
        use_lnn: bool = True,
        sample: bool = False,
        allow_ids: set[str] | None = None,
        client: str | None = None,
        occurred_after: float = 0.0,
        occurred_before: float = 0.0,
    ) -> list[RecallResult]:
        """Recall memories relevant to the query.

        Flow:
            1. Generate the query vector (LNN's h(t) or plain text embedding).
            2. Retrieve top_k * 3 candidates from the vector store.
            3. Rank comprehensively with the scoring engine.
            4. Optional: distance-weighted sampling adds exploration.

        Args:
            query: query text. None uses the LNN's current state as the
                implicit query.
            top_k: number of memories to return.
            use_lnn: whether to blend the LNN into the query vector (False uses
                the plain text embedding).
            sample: use sampling (True) or greedy ranking (False).
            allow_ids: search only within this id whitelist (tag filtering
                narrows the candidate set **before** retrieval). None = no
                limit. Empty set = return empty immediately.
        """
        if self.store.count() == 0:
            return []
        if allow_ids is not None and not allow_ids:
            return []

        # This retrieval reads **this route's** temporal state (same key as the
        # write side)
        window_active = occurred_after > 0 or occurred_before > 0
        with self._lnn_route(client) as _rctx:
            # Generate the query vector
            if query is not None:
                # Query side: BGE and similar models need an instruction prefix
                # (embed_query equals embed for non-BGE models)
                query_emb = self.embedder.embed_query(query)
                if use_lnn:
                    # Hybrid: weighted fusion of the LNN query vector and the
                    # text embedding
                    lnn_query = self.memory_core.get_query()
                    lnn_np = np.asarray(lnn_query, dtype=np.float64)
                    # Weighting: 0.6 * text embedding + 0.4 * LNN state
                    query_vec = 0.6 * query_emb + 0.4 * lnn_np
                    norm = np.linalg.norm(query_vec)
                    if norm > 1e-12:
                        query_vec = query_vec / norm
                else:
                    query_vec = query_emb
            else:
                # No explicit query: use the LNN's current state
                lnn_query = self.memory_core.get_query()
                query_vec = np.asarray(lnn_query, dtype=np.float64)

            # Retrieve candidates (fetch extra for scoring)
            # Under tag filtering the cap switches to the whitelist size: a
            # whitelist may hold only a few dozen entries; sticking with the
            # full-library count just wastes compute on masked rows.
            pool_size = self.store.count() if allow_ids is None else len(allow_ids)
            # When a time window is active the candidate pool is enlarged 10x:
            # filtering happens **after** retrieval, so with a pool of only
            # top_k*3 the in-window entries may not fill top_k -- that would be
            # "filtering too hard", not "the library has nothing". Fetch more
            # first, then trim by time.
            candidate_k = min(max(top_k * (10 if window_active else 3), top_k),
                              pool_size)
            candidates = self.store.search(query_vec, top_k=candidate_k, allow_ids=allow_ids)

            if not candidates:
                return []

            # Prepare scoring inputs
            # Use the reference time (newest timestamp of stored messages), not
            # wall-clock (key LoCoMo fix: time.time() diverges hugely from the
            # data timestamps, which nullifies time decay)
            current_time = self._reference_time if self._reference_time else time.time()

        # O8 retrieval-side staleness window: age > window and not pinned ->
        # excluded from candidates. Applied before scoring -- blocked entries
        # are not scored, not ranked, and get no recall_count bump (zero state
        # writes for blocked entries).
        if self._recency_window_seconds > 0 or window_active:
            candidates = [
                (r, s) for r, s in candidates
                # O8: automatic system-level blocking (timestamp<=0 conservatively
                # kept, see the recency notes). **Yields when the user gave an
                # explicit time window**: O8 is a "default policy", the time
                # window is an "explicit intent"; when they conflict, explicit
                # wins -- otherwise "find that incident from three years ago"
                # would be blocked first by the 180-day window, rendering the
                # time window useless.
                if (self._recency_window_seconds <= 0 or window_active
                    or not out_of_window(r.timestamp, r.pin, current_time,
                                         self._recency_window_seconds))
                # User's explicit time window (timestamp<=0 excluded, see
                # in_time_window for the rationale)
                and (not window_active
                     or in_time_window(r.timestamp, occurred_after,
                                       occurred_before))
            ]
            if not candidates:
                return []
        similarities = np.array([sim for _, sim in candidates])
        timestamps = np.array([r.timestamp for r, _ in candidates])
        recall_counts = np.array([r.recall_count for r, _ in candidates])
        # O1 retrieval strength (a result-set-semantics accumulator, see
        # reinforce/stamp_recalled)
        strengths = np.array([r.recall_strength for r, _ in candidates])

        # Comprehensive scoring
        # C5 (default off, HIPPOCAMPUS_INTENT_ROUTING=1 to enable): routes
        # weight multipliers by query intent; when off, not even the classifier
        # runs, bit-for-bit unchanged behavior.
        intent = None
        multipliers = None
        if intent_routing_enabled() and query:
            intent = classify_intent(query)
            multipliers = INTENT_WEIGHT_MULTIPLIERS.get(intent) or None
        scores = self.scorer.score(
            similarities, timestamps, recall_counts, current_time,
            weight_multipliers=multipliers,
            strengths=strengths,
        )
        # C6 (default off, HIPPOCAMPUS_RECALL_DECAY=1 to enable): retrieval-time
        # memory decay -- multiplicatively re-weights the composite score by
        # exp(-age/tau). The additive formula's time term is too lightly
        # weighted (~0.03); changing weights is noise (proven by the C2 scan),
        # while multiplicative re-weighting can lift "recently happened" items
        # out of the ranking depths as a whole. When off, no multiplication,
        # bit-for-bit unchanged.
        if _recall_decay_enabled():
            age = np.maximum(current_time - timestamps, 0.0)
            scores = scores * np.exp(-age / _recall_decay_tau())
        self.last_recall_intent = intent

        if sample:
            # Distance-weighted sampling
            chosen_idx = self.sampler.sample(similarities, k=top_k)
            ranked = [(i, scores[i]) for i in chosen_idx]
            ranked.sort(key=lambda x: -x[1])
        else:
            # Greedy ranking
            ranked = self.scorer.rank(scores, top_k=top_k)

        # Build results
        results: list[RecallResult] = []
        for idx, score in ranked:
            record, sim = candidates[idx]
            results.append(
                RecallResult(
                    memory_id=record.id,
                    content=record.content,
                    score=score,
                    similarity=sim,
                    timestamp=record.timestamp,
                    importance=record.importance,
                    recall_count=record.recall_count,
                    memory_type=record.memory_type,
                )
            )

            # Bump the recall count.
            #
            # Note **last_recalled_at is NOT written here**: this method returns
            # the **candidate set** (brain.recall fetches candidates with
            # fetch_k = max(top_k*4, 12), then applies hard gates / MMR / tag
            # filtering); stamping timestamps here would also count "fetched and
            # glanced at but never returned to the host" records as "recalled" --
            # dormant archiving (consolidate uses last_recalled_at as its time
            # base) gets diluted accordingly: every query would extend the life
            # of a dozen records. The real "was recalled" timestamp is written
            # by **brain.recall on the final results** (see the end of that
            # method).
            self.store.update(record.id, recall_count=record.recall_count + 1)

        return results

    def stamp_recalled(self, memory_ids) -> int:
        """Stamp "last recalled" timestamps on entries that were truly returned
        to the host.

        Why a separate method instead of doing it in :meth:`search`:
        ``search`` returns the **candidate set** (the upper layer still applies
        hard gates / MMR / tag filtering); stamping at the candidate stage means
        "every query extends the life of a dozen memories that never showed
        up", diluting dormant archiving (``consolidate`` uses
        ``last_recalled_at`` as its time base).

        The time base must match ``consolidate``'s ``now`` (``_reference_time
        or wall-clock``): LoCoMo benchmark data timestamps may be from 2023; if
        a bare ``time.time()`` were written, old memories would be judged
        "just recalled" -> **permanently exempt from dormant archiving**.

        Returns:
            Number of entries actually stamped successfully.
        """
        stamp = self._reference_time if self._reference_time else time.time()
        n = 0
        for mid in memory_ids:
            if not mid:
                continue
            rec = self.store.get(mid)
            if rec is None:
                continue
            # O1: recall is consolidation -- a true recall (entered the result
            # set) bumps retrieval strength by +1 while stamping the timestamp.
            # Division of labor with recall_count (candidate set, diversity
            # penalty): see the MemoryRecord.recall_strength comment.
            new_strength = max(float(rec.recall_strength or 0.0), 0.0) + 1.0
            if self.store.update(mid, last_recalled_at=stamp,
                                 recall_strength=new_strength):
                n += 1
        return n

    def reinforce(self, memory_ids, amount: float = 1.0) -> int:
        """Only accumulates retrieval strength, never touches last_recalled_at
        (the seam for sleep replay).

        Division-of-labor discipline (an extension of constraints 4/5):
        ``stamp_recalled`` = true recall (entered the result set) -- stamps the
        timestamp + strength +1; this method = **non-recall reinforcement**
        (the sleep batch's replay co-activation) -- strength only. A replay is
        not "recalled by the host"; if it stamped last_recalled_at, dormant
        archiving would be diluted by replays into immortality.
        """
        n = 0
        for mid in memory_ids:
            if not mid:
                continue
            rec = self.store.get(mid)
            if rec is None:
                continue
            new_strength = max(float(rec.recall_strength or 0.0), 0.0) + float(amount)
            if self.store.update(mid, recall_strength=new_strength):
                n += 1
        return n

    # ------------------------------------------------------------------
    # Rewrite-history ledger (mem_history.jsonl)
    # ------------------------------------------------------------------
    def _mem_history_append(self, entry: dict[str, Any]) -> None:
        """Append one rewrite-history entry (pure bypass -- any failure never
        blocks the main write path).

        Line count/seq lazily initialized (one disk scan on first write); the
        pure in-memory brain gives up immediately. Write failures log a warning
        via store.mem_history ("what it was before the change" must be visible
        to ops); this method neither duplicates the record nor raises.
        """
        if not self.persist_dir:
            return
        if self._mem_history_lines is None:
            self._mem_history_lines, self._mem_history_last_seq = (
                _mem_history.scan_history(self.persist_dir))
        _op_id, self._mem_history_last_seq, self._mem_history_lines = (
            _mem_history.append_history(
                self.persist_dir, entry,
                self._mem_history_last_seq, self._mem_history_lines,
                ring=self.mem_history_ring, trim_at=self.mem_history_ring))

    def memory_history(self, memory_id: str | None = None,
                       limit: int = 20) -> list[dict[str, Any]]:
        """Read rewrite history (newest first). ``memory_id`` filters to one
        entry; the pure in-memory brain returns empty. The ledger is the audit
        surface: forget's cascade does **not** clean it (tombstones block
        resurrection, not history), and deleted records' history entries age out
        through the ring naturally."""
        entries = _mem_history.read_history(self.persist_dir)
        if memory_id:
            entries = [e for e in entries if e.get("memory_id") == memory_id]
        entries.reverse()
        return entries[:max(int(limit), 0)]

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Get a single memory by ID."""
        record = self.store.get(memory_id)
        if record is None:
            return None
        return record.to_dict()

    def update_memory(
        self, memory_id: str, **fields: Any
    ) -> bool:
        """Update memory fields (content / importance / memory_type etc.).

        If content is updated, the embedding is regenerated and the LNN state
        updated.

        content/importance changes first record the old values into
        ``mem_history.jsonl`` (sidecar ledger, pure bypass -- write failure only
        warns and never blocks the main path). A state-**only** change is not
        recorded: the supersede/archive path keeps the old record itself (the
        state machine already provides the audit trail), so recording it is just
        noise; when state changes together with content/importance the old
        values are recorded together. embedding is not recorded (recomputable
        from content).
        """
        record = self.store.get(memory_id)
        if record is None:
            return False

        # Rewrite keeps the prior value: before = full three-field snapshot
        # (audit value -- the state at rewrite time must also be examinable),
        # after = the new values of the changed fields. See the record/do-not-
        # record boundary in the method docstring.
        changed = [k for k in ("content", "importance") if k in fields]
        if changed:
            self._mem_history_append({
                "kind": "update",
                "memory_id": memory_id,
                "before": {k: getattr(record, k)
                           for k in ("content", "importance", "state")},
                "after": {k: fields[k] for k in changed},
            })

        # If content was updated, re-embed
        if "content" in fields:
            new_content = fields["content"]
            new_emb = self.embedder.embed(new_content)
            fields["embedding"] = new_emb.tolist()

        return self.store.update(memory_id, **fields)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete the given memory (also removes idempotency receipts pointing
        to it, preventing dangling ids)."""
        ok = self.store.delete(memory_id)
        # No need to prune if the index was never built: on rebuild this record
        # would not be there anyway (zero-cost bypass)
        if ok and getattr(self, "_idem", None) is not None:
            self._idem.forget_memory(memory_id)
        return ok

    def list_memories(
        self,
        limit: int = 100,
        source: str | None = None,
        state: str | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List memories (without embedding vectors).

        Args:
            limit: maximum number of entries to return.
            source: only entries from this source (exact match; what coverage
                report, high 3 asked for: "list which entries a file imported").
                Pass ``""`` to filter for old records "without a source".
            state: only the given state (active/superseded/archived).
            offset: skip the first N entries (paging for auditing large
                libraries together with limit).
        """
        results = []
        for record in self.store.iter_records():
            if source is not None and (record.source or "") != source:
                continue
            if state is not None and record.state != state:
                continue
            if offset > 0:
                offset -= 1
                continue
            if len(results) >= limit:
                break
            results.append(record.to_dict())
        return results

    def sources(self) -> dict[str, int]:
        """Count entries per source: ``{source: count}``.

        Import scripts use it for the "skip already-imported source files"
        idempotency check (coverage report, medium 9). Old records without a
        source are grouped under the empty-string key.
        """
        out: dict[str, int] = {}
        for record in self.store.iter_records():
            key = record.source or ""
            out[key] = out.get(key, 0) + 1
        return out

    def reset_context(self, client: str | None = None,
                      clear_all: bool = False) -> list[str]:
        """Reset the LNN hidden state (start a new conversation context).

        By default only **this route** is cleared. In a shared process,
        "clearing everyone" is the most typical cross-contamination incident --
        A starts a new session and B's temporal context zeroes out with it.
        Clearing everything requires explicit ``clear_all=True`` (and the return
        value lists the cleared routes).

        Returns:
            List of cleared keys.
        """
        if clear_all:
            keys = list(self._lnn_contexts)
            for c in self._lnn_contexts.values():
                c.hx = None
                c.last_output = None
                c.window.clear()
                c.step_count = 0
                c.reference_time = None
            self._lnn_contexts.clear()
            self._lnn_ctx_lru.clear()
            self.memory_core.reset_state()
            self._step_count = 0
            self._reference_time = None
            return keys

        key = self.lnn_key(client)
        ctx = self._lnn_contexts.pop(key, None)
        if key in self._lnn_ctx_lru:
            self._lnn_ctx_lru.remove(key)
        # If the cleared route is the one currently mounted on the core, the
        # core must be zeroed too, otherwise the next reader gets residue.
        self.memory_core.reset_state()
        self._step_count = 0
        self._reference_time = None
        return [key] if ctx is not None else []

    # ------------------------------------------------------------------
    # Multi-model vector management (model switch without re-import)
    # ------------------------------------------------------------------
    @property
    def model_key(self) -> str:
        """Identifier of the currently active embedding model."""
        return self._model_key

    def vector_of(self, record: MemoryRecord) -> np.ndarray | None:
        """Get the record's vector under the currently active model; returns None
        if never computed (needs backfilling)."""
        vec = record.vector_for(self._model_key)
        if vec is None:
            return None
        return np.asarray(vec, dtype=np.float64)

    def ensure_vectors(
        self, batch_size: int = 32, verbose: bool = True
    ) -> dict[str, int]:
        """Backfill missing vectors for the current model (no document re-import,
        no LLM calls).

        The first load after switching embedding models takes this path: text
        and structure are preserved as-is, only the vectors are recomputed with
        the new model. Even ten-thousand-entry libraries need only minutes.

        Returns:
            {"missing": number missing, "filled": number actually backfilled}
        """
        missing = self.store.missing_ids(self._model_key)
        if not missing:
            return {"missing": 0, "filled": 0}

        if verbose and len(missing) >= 50:
            # Backfilling a large library is a minutes-scale operation, and on
            # the load() path it is **synchronously blocking**: the MCP service
            # startup warmup stalls here and the client may time out and
            # disconnect. So say it up front, so people know "it is not hung,
            # it is computing".
            _stderr_print(
                f"[MemoryManager] model {self._model_key} is missing {len(missing)} vectors; "
                f"backfilling now (synchronous and blocking -- do not interrupt; for large libraries run "
                f"scripts/reembed.py offline first).",
                flush=True,
            )

        filled = 0
        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            texts = []
            ids = []
            for mid in chunk:
                rec = self.store.get(mid)
                if rec is None or not rec.content:
                    continue
                ids.append(mid)
                texts.append(rec.content)
            if not texts:
                continue
            vecs = self.embedder.embed_batch(texts)
            for mid, vec in zip(ids, vecs):
                self.store.set_vector(mid, self._model_key, vec)
                filled += 1
            if verbose:
                _stderr_print(f"  backfilled vectors {filled}/{len(missing)}", flush=True)

        if filled:
            # Persist only the vector store; leave the LNN checkpoint alone
            # (avoid overwriting the state awaiting restoration mid-load)
            self.store.save()
        return {"missing": len(missing), "filled": filled}

    def switch_model(self, model_key: str, dim: int) -> dict[str, int]:
        """Switch the active embedding model (for offline experiments comparing
        retrieval quality across models).

        Args:
            model_key: target model identifier.
            dim: target model dimension.

        Returns:
            ensure_vectors' result (missing vectors backfilled automatically).
        """
        self.store.set_model(model_key, dim)
        self._model_key = model_key
        self.embedding_dim = dim
        return self.ensure_vectors()

    def get_stats(self) -> dict[str, Any]:
        """Return current system state statistics."""
        return {
            "total_memories": self.store.count(),
            "lnn_steps": self._step_count,
            "embedding_dim": self.embedding_dim,
            "embed_model": self._model_key,
            "vectors": self.store.model_stats(),
            "lnn_has_state": self.memory_core.hx is not None,
            "window_size": len(self.memory_core.window),
            # Self-report: how many context routes exist, each one's step count.
            # Under a shared connection this number should be > 1; a constant 1
            # means the caller never passed client (wiring incomplete).
            "lnn_contexts": self.lnn_context_report(),
            # H7 self-report: CfC's actual runtime (numpy = torch-free;
            # torch(ncps) = fallback path). Reports the **actual** value, not the
            # configured one -- same discipline as embedder_runtime.
            "lnn_runtime": self.memory_core.runtime,
            "lnn_backend": self.memory_core.backend,
            "scoring_mode": self.scorer.mode,
            "w_sim": self.scorer.w_sim,
            "w_time": self.scorer.w_time,
            "w_recall": self.scorer.w_recall,
            "lambda_decay_days": round(self.scorer.lambda_decay / 86400, 1),
            "mu_importance_days": round(self.scorer.mu_importance / 86400, 1),
            # Auto-save health: non-None means a whole batch exists **only in
            # memory** and is lost on restart
            "last_save_error": self.last_save_error,
            "autosave_failures": self.autosave_failures,
            # B2-4 self-report: actual inference backend. **Must report the
            # actual value, not the configured one** -- with onnx configured but
            # artifacts missing it silently falls back to torch; looking only at
            # config never reveals that.
            "embedder_runtime": self.embedder_runtime_backend(),
            "embedder_model_source": getattr(
                self.embedder, "model_source", None
            ),
            "onnx_attached": self.embedder_runtime_backend() == "onnx",
            # F6 self-report: whether this library's embedder reuses the shared
            # in-process heavy object (False while the pool has stock = sharing
            # not effective, the second library is still pointlessly loading a
            # full set of weights).
            "embedder_shared": bool(
                getattr(self.embedder, "shared_from_pool", False)
            ),
            "embedder_pool": embedder_pool_stats(),
            # C5/C6 self-report (B2-4 paradigm: switch states must be visible
            # from stats)
            "intent_routing_enabled": intent_routing_enabled(),
            "recall_decay_enabled": _recall_decay_enabled(),
            "recall_decay_tau_days": round(_recall_decay_tau() / 86400, 1),
            "last_recall_intent": self.last_recall_intent,
        }

    def embedder_runtime_backend(self) -> str:
        """Return the **actual** inference backend name: ``onnx`` / ``torch`` /
        ``hash`` / ...

        RED FLAG -- why reporting only the configured value is not enough:
        ``HIPPOCAMPUS_EMB_BACKEND=onnx`` silently falls back to torch when
        artifacts are missing (or assembly-stage failures get swallowed); "I
        thought I was on ONNX but was not" is exactly the wiring defect found on
        a real library. What is reported here is the **mount outcome**. There
        are two onnx forms: the ST wrapper with ``_onnx`` attached (ST-side
        path), or the embedder itself being an ``OnnxEmbedder`` (the follow
        direct-construction path) -- both count as onnx.
        """
        e = self.embedder
        if getattr(e, "_onnx", None) is not None or type(e).__name__ == "OnnxEmbedder":
            return "onnx"
        name = type(e).__name__
        if name == "SentenceTransformerEmbedder":
            return "torch"
        return (self.embedder_backend or name).lower()

    def load_trained_weights(self, model_path: str) -> bool:
        """Load trained CfC weights (CRL retrieval-guided training).

        H7: supports two sources -- ``.npz`` (the export of
        ``scripts/export_lnn_weights.py``, works with both numpy/torch cores,
        **needs no torch**) and legacy ``.pt`` (lazy torch fallback, keeps the
        old benchmark chain usable). Recommended: run the export once offline so
        runtime stays on the torch-free path.

        Args:
            model_path: path to a .npz or .pt file.

        Returns:
            True on successful load.
        """
        try:
            if model_path.endswith(".npz"):
                with np.load(model_path, allow_pickle=False) as z:
                    arrays = {
                        k: np.asarray(z[k]) for k in z.files if k != "meta_json"
                    }
                # The npz may carry mask keys; load_weights only accepts
                # weight/bias, extra keys are ignored
                if arrays:
                    self.memory_core.load_weights(arrays)
                    return True
                return False
            # legacy .pt: lazy torch (this dependency is paid only when a .pt is
            # explicitly passed)
            import torch

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            # Keys saved by the CRL training script carry a "cfc." prefix; strip
            # it before loading
            cfc_state = {}
            for k, v in checkpoint.get("model_state", checkpoint).items():
                if k.startswith("cfc."):
                    cfc_state[k[4:]] = v  # strip the "cfc." prefix
                else:
                    cfc_state[k] = v
            if cfc_state:
                self.memory_core.load_weights(cfc_state)
                return True
            return False
        except Exception as e:
            _stderr_print(f"[MemoryManager] failed to load training weights: {e}")
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @property
    def meta_path(self) -> str | None:
        """Scalar metadata sidecar file path (same directory as the checkpoint)."""
        if self._checkpoint is None:
            return None
        return self._checkpoint.path + ".meta.json"

    def save(self) -> None:
        """Persist all state: vector store + LNN hidden state + scalar metadata.

        The checkpoint uses **backed-up** writes (``save_with_backup``): the
        previous code used the non-backing-up ``save_atomic``, which left the
        "fall back to .bak" branch in ``StateCheckpoint.load`` with no backup
        ever available -- the advertised "atomic write + backup" only delivered
        the atomic write (defect N2).

        KNOWN TRAP (fixed): the core is a **singleton**; when ``_lnn_route``
        exits it restores the pre-entry snapshot onto the core => a plain
        ``get_state()`` here would persist a **stale** state (in a new process
        hx is always empty; the old torch path was hit equally -- weights were
        random on every restart anyway, which is why nobody noticed). The live
        state lives in the default route's ctx; take it from there. Other named
        routes are not persisted (historical boundary unchanged: the checkpoint
        carries only the default session's temporal state). After H7 froze the
        weights, persisting hx gained real meaning for the first time.

        Pre-persist hook (``_pre_persist``): flush the pending L0 **before**
        touching L1, guaranteeing "every L1 entry's source_ids on disk can be
        found in the on-disk L0". The order is deliberate: L0 first, L1 second
        -- killed mid-way, at worst you get "raw exists but unreferenced"
        (harmless); L1 first, L0 second leaves "references pointing at a
        nonexistent raw" (unrecoverable -- measured at nearly 10% of entries in
        one library, and still trickling in after the fix shipped).
        """
        # Contract boundary: **a hook raising = pre-persist failure, but it must
        # never block this persist**. Blocking means "not even L1 gets saved",
        # worse than "L1 present, L0 missing"; the error is written back into
        # last_save_error at the end -- otherwise the trailing `= None` would
        # silently swallow the L0 failure, and "silent failure" is exactly why
        # this incident stayed hidden until an external project found it.
        pre_error: str | None = None
        hook = self._pre_persist
        if hook is not None:
            try:
                hook()
            except Exception as e:  # noqa: BLE001 -- see the contract above
                pre_error = f"L0 前置落盘失败：{type(e).__name__}: {e}"
                logger.warning("[save] %s", pre_error)
        self.store.save()
        if self._checkpoint is not None:
            state = self.memory_core.get_state()
            dctx = self._lnn_contexts.get(DEFAULT_LNN_KEY)
            if dctx is not None:
                if dctx.hx is not None:
                    state["hx"] = MemoryCore._state_array(dctx.hx)
                if dctx.last_output is not None:
                    state["last_output"] = MemoryCore._state_array(dctx.last_output)
                state["window"] = [
                    MemoryCore._state_array(t) for t in dctx.window
                ]
            state["step_count"] = self._step_count
            state["reference_time"] = self._reference_time
            self._checkpoint.save_with_backup(state)
            self._save_meta()
        # Persist succeeded -> clear earlier failure traces (keep the cumulative
        # counter for audit); but a **failed pre-persist L0 flush is not
        # cleared**: that is an unpaid debt, kept visible on purpose.
        self.last_save_error = pre_error

    def set_pre_persist_hook(self, hook: Callable[[], None] | None) -> None:
        """Register the "flush L0 before persisting" hook (call sites and
        semantics in :meth:`save`).

        Why a hook instead of manager holding brain directly: the engine layer
        does not know about L0 (it only knows store / checkpoint). The hook
        keeps the "layer ordering" **business invariant** in brain; engine just
        calls it at the right moment.
        """
        self._pre_persist = hook

    def retry_save(self) -> dict[str, Any]:
        """Retry a save after a concurrency conflict: **take the disk's newest
        version first, then re-add only the increments the disk lacks**.

        Why not simply "save() again": ``disk_stamp`` already proved another
        process modified the disk; overwriting directly from the in-memory
        snapshot = clobbering someone else's fresh data with a stale version.
        ``retry_save`` does:

            1. Snapshot all records in this process's memory (they are not yet
               persisted at this point);
            2. Re-``load()`` the disk's newest version (disk wins);
            3. Re-add the ids "missing on disk" -- those are this process's true
               increments;
            4. ``save()`` again (the stamp is refreshed by then, a safe write).

        Semantic boundary (no automatic merge, additive only):
            - ids missing on disk -> re-added (increment, safe);
            - present on both sides but differing -> **disk version kept**, the
              local edits recorded in ``dropped_local_edits`` for human review
              (no field-level merging, avoiding a semi-automatic merge nobody
              can explain);
            - records deleted by this process but still on disk -> they
              "revive", recorded in ``disk_only`` (ids newly added by the other
              process also land here; this process cannot tell the two apart).

        Returns:
            Statistics dict with added / dropped_local_edits / disk_only etc.

        Raises:
            ConcurrentWriteError: written again during the retry (the conflict
                has not stopped; stop the other writer first instead of
                retrying).
        """
        before = {rec.id: rec for rec in self.store.iter_records()}

        # 2) Reload with disk as the source of truth (no vector backfill:
        #    backfilling is heavy, the caller triggers it explicitly)
        self.load(auto_backfill=False)
        disk = {rec.id: rec for rec in self.store.iter_records()}

        # 3) Re-add only the increments the disk lacks
        added: list[str] = []
        for mid, rec in before.items():
            if mid not in disk:
                self.store.add(rec)
                added.append(mid)

        # Present on both sides but differing -> keep the disk version, log the
        # locally dropped edits
        dropped: list[str] = []
        for mid, rec in before.items():
            other = disk.get(mid)
            if other is None:
                continue
            if rec.to_dict(with_embeddings=False) != other.to_dict(with_embeddings=False):
                dropped.append(mid)

        disk_only = [mid for mid in disk if mid not in before]

        # 4) The stamp is refreshed; this is now a safe write
        self.save()
        return {
            "added": added,
            "added_count": len(added),
            "dropped_local_edits": dropped,
            "dropped_local_edits_count": len(dropped),
            "disk_only": disk_only,
            "disk_only_count": len(disk_only),
            "total": self.store.count(),
        }

    def _save_meta(self) -> None:
        """Persist the scalar metadata (``reference_time`` / ``step_count``) to a
        separate JSON.

        A ``.pt`` file holds tensors; any corruption makes the whole state
        unreadable, while ``reference_time`` is the key scoring scalar -- losing
        it makes ``recall()`` fall back to ``time.time()`` and time decay
        **silently distorts** (the same trap the LoCoMo benchmark hit). Keeping
        a plain-text JSON copy means it can be salvaged even if the ``.pt`` is
        completely lost.
        """
        path = self.meta_path
        if path is None:
            return
        payload = {
            "reference_time": self._reference_time,
            "step_count": self._step_count,
        }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as e:
            _stderr_print(f"[MemoryManager] metadata save failed: {e}")

    def _load_meta(self) -> dict[str, Any]:
        """Read the sidecar metadata JSON; returns an empty dict when missing or
        corrupt."""
        path = self.meta_path
        if path is None or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            _stderr_print(f"[MemoryManager] metadata read failed (ignored): {e}")
            return {}

    def _follow_store_model(self, model_key: str) -> None:
        """Swap the embedder to the model the library actually uses (the executor
        of "the model follows the data")."""
        old_key, old_dim = self._model_key, self.embedding_dim
        backend = self.embedder_backend
        if backend == "sentence_transformer":
            probe = create_embedder(
                "sentence_transformer", model_name=model_key
            )
            # H7: when ONNX is possible, **construct OnnxEmbedder directly** --
            # bypassing the ST wrapper, the embedder itself is a torch-free
            # implementation. probe only lends its local model directory
            # resolution (pure path logic; construction imports no torch/ST;
            # lazy loading happens on first encode).
            direct = None
            if backend_kind() != "torch":
                try:
                    from .onnx_embedder import find_onnx_artifacts, get_shared_onnx

                    model_dir = probe._find_local_model()
                    if model_dir and find_onnx_artifacts(model_dir) is not None:
                        # F6: use the shared pool -- if another library with the
                        # same model is already running, this hands back the very
                        # same session/tokenizer (saving ~400MB of weights and
                        # one load) instead of loading another copy.
                        direct = get_shared_onnx(
                            model_dir=model_dir, model_name=model_key
                        )
                except Exception as e:  # noqa: BLE001 - direct construction failed, fall back to the wrapper path
                    _stderr_print(f"[MemoryManager] direct ONNX construction failed; falling back to the ST wrapper: {e}")
            if direct is not None:
                self.embedder = direct
            else:
                # H2: the model name is only **finalized at this moment** -- at
                # construction it may merely be the fallback default (e.g.
                # DEFAULT_MODEL_NAME=all-MiniLM-L6-v2) while the library actually
                # uses another model. ONNX assembly must run again here,
                # otherwise under the default path ONNX never takes effect and
                # never errors ("I thought I was saving memory but was not").
                self.embedder = probe
                attach = getattr(self.embedder, "_attach_onnx", None)
                if callable(attach):
                    try:
                        attach(from_store=True)
                    except Exception as e:  # noqa: BLE001 - assembly failure must not keep the library from starting
                        _stderr_print(
                            f"[MemoryManager] ONNX assembly failed; keeping the torch backend: {e}"
                        )
        else:
            self.embedder = create_embedder(backend, dim=old_dim)
        self.embedding_dim = self.embedder.dim
        self._model_key = self.embedder.model_key
        # The LNN core's input/output dimensions follow the embedding dimension
        # (AutoNCP requires output_dim < n-2)
        self.memory_core = MemoryCore(
            input_dim=self.embedding_dim,
            n_neurons=max(self._ctor_n_neurons, self.embedding_dim + 3),
            output_dim=self.embedding_dim,
            window_size=self._ctor_window_size,
        )
        self.store.set_model(self._model_key, self.embedding_dim)
        self.model_follow = {
            "from": old_key,
            "to": self._model_key,
            "from_dim": old_dim,
            "to_dim": self.embedding_dim,
        }
        _stderr_print(
            f"[MemoryManager] embedding model follows the library's data: {old_key} -> {self._model_key} "
            f"(vectors for this model already exist in the library; no recomputation needed)."
        )

    def _sync_embedder_with_store(self) -> None:
        """After loading, verify the "effective model" matches the "library's
        vectors" (guards against silently using the wrong model).

        Three cases:

        * **Match** (the current model has coverage in the library, or the
          library is empty) -> do nothing, normal path;
        * **Mismatch, and the model was explicitly specified** -> raise
          :class:`EmbedderModelMismatchError`. When config and data disagree,
          do not guess; to compare models deliberately set
          ``HIPPOCAMPUS_ALLOW_MODEL_SWITCH=1``;
        * **Mismatch, but the model is a fallback default** -> **follow the
          library's data** and switch (this is the real cure for "the service
          runs as MiniLM silently when env is unset": a default value can no
          longer quietly override a library with a definite model).
        """
        self.model_follow = None
        if self.embedder_backend != "sentence_transformer":
            return
        preferred = self.store.preferred_model_key()
        if not preferred or preferred == self._model_key:
            return
        # The library's model has coverage and is not the current model -> conflict
        source = getattr(self.embedder, "model_source", "explicit")
        allowed = load_config().allow_model_switch
        if source == "default" and not allowed:
            self._follow_store_model(preferred)
            return
        if allowed:
            _stderr_print(
                f"[MemoryManager] {ENV_ALLOW_MODEL_SWITCH} enabled: the library was built with "
                f"{preferred}; running with {self._model_key} this time and backfilling missing vectors."
            )
            return
        raise EmbedderModelMismatchError(
            f"embedding model mismatch with the library's data: the effective model is {self._model_key!r} "
            f"(source: {source}), but the vectors in {self.store.persist_path!r} belong to "
            f"{preferred!r}.\n"
            f"Using the wrong model silently destroys retrieval (or triggers a full-library recompute), so loading is refused. Remedies:\n"
            f"  1) make them consistent (recommended): set HIPPOCAMPUS_EMB_MODEL={preferred};\n"
            f"  2) to compare other models on this data deliberately: set "
            f"{ENV_ALLOW_MODEL_SWITCH}=1 (all vectors will be backfilled for the new model)."
        )

    def load(self, auto_backfill: bool = True) -> None:
        """Restore all state from disk.

        Args:
            auto_backfill: after loading, automatically backfill missing vectors
                for the current embedding model. Old libraries (built before a
                model switch, or with the early single-vector format) need no
                re-import; the first load re-embeds automatically; set False to
                control it manually.
        """
        self.store.load()
        # After loading **align the model before touching any vector**: whatever
        # model the library says it runs with is the model used (or an explicit
        # error); never force the current model onto it (see the method docs).
        self._sync_embedder_with_store()
        if self._checkpoint is not None and (
            # Main file missing but .bak still present (last save crashed between
            # replace steps) still counts as recoverable state
            self._checkpoint.exists()
            or os.path.isfile(self._checkpoint.bak_path)
        ):
            state = self._checkpoint.load()
            self.memory_core.load_state(state)
            self._step_count = state.get("step_count", 0)
            self._reference_time = state.get("reference_time")

        # When the .pt is unusable (corrupt/missing), the sidecar JSON recovers
        # the scalars as a fallback
        meta = self._load_meta()
        if self._reference_time is None and meta.get("reference_time") is not None:
            self._reference_time = meta["reference_time"]
            _stderr_print("[MemoryManager] reference_time restored from metadata JSON (.pt unavailable)")
        if not self._step_count and meta.get("step_count"):
            self._step_count = int(meta["step_count"])

        if auto_backfill:
            stats = self.ensure_vectors()
            if stats["filled"]:
                _stderr_print(
                    f"[MemoryManager] backfilled {self._model_key} vectors for model "
                    f"{stats['filled']}"
                )
