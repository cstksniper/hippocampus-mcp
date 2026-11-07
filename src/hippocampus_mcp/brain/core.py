# -*- coding: utf-8 -*-
"""HippoBrain — the orchestration layer of the agent memory foundation (v0.3).

Wraps MemoryManager (vector + LNN + scoring) into a complete "memory brain":

    L0 raw layer       raw_log    raw conversation stream, append-only; L1's source_ids point back at it
    L1 fact layer      memories   extracted facts + state machine (active/superseded/archived) + cognitive types
    L2 summary layer   summaries  session summaries / topic aggregation
    L3 profile layer   profile    long-term stable facts and preferences (low confidence not admitted; conflicts only flagged, never overwritten)
    index layer        entities   entity table + entity↔memory inverted index (precursor of the P0 entity graph)
    usage layer        session_*  5-slot context assembly (modeled on bigeye AttentionFocus)

Design points drawn from research:
    - fact superseded → mark superseded, never delete (bigeye relation_store)
    - importance >= 0.7 → pin, exempt from time decay (bigeye: >=7 pinned)
    - similarity hard gate 0.30 → keep the summary layer from "being similar to everything" and dominating recall
    - MMR dedup (λ=0.7) → suppress recall noise
"""

from __future__ import annotations

import atexit as _atexit
import heapq  # noqa: F401  (novelty gate: take the top cap of the high-confidence L1 comparison set, O(N) without a full sort)
# json is no longer called directly by this module (the only call site has
# moved into domain.ingest_batch_flow), but the import **must stay here**:
# tests/test_l0_autosave.py monkeypatches ``brain.json.dump`` to inject an
# OSError simulating a full disk — the B in ``B.json.dump`` is this module.
# Remove this import and the "persist-failure must not interrupt" guard
# test loses its injection point.
import json  # noqa: F401
import logging
import os
import re
import sys
import threading
import time
import zlib

import numpy as np

try:  # only fetch the hostname; leave empty on failure — startup must not hinge on the hostname
    import socket as _socket
except ImportError:  # pragma: no cover
    _socket = None  # type: ignore[assignment]
from typing import Any

import numpy as np

from ..config import load_config
from ..common.recency import (
    in_time_window,
    iso_date,
    out_of_window,
    weekday_abbr,
)
from ..domain import policies as _policies
from ..domain import novelty_gate as _novelty_gate
from ..domain import prospective as _prospective
from ..domain import forget_cleanup as _forget_cleanup
from ..domain import ingest_flow as _ingest_flow
from ..domain import knowledge_batch as _knowledge_batch
from ..domain import knowledge_map_view as _knowledge_map_view
from ..domain import topic_material as _topic_material
from ..domain import tag_update as _tag_update
from ..domain import consolidate_flow as _consolidate_flow
from ..domain import ingest_batch_flow as _batch_flow
from ..domain import ingest_pipeline as _ingest_pipeline
from ..domain import recall_flow as _recall_flow
from ..domain import refresh_knowledge_flow as _refresh_flow
from ..domain import stats_projection as _stats_projection
from ..domain import stats_report as _stats_report
from ..domain import miss_log_flow as _miss_log_flow
from ..domain import file_registry_flow as _fr_flow
from ..domain import summarize_batch_flow as _summarize_batch_flow
from ..domain import pure_checks as _pure_checks
from ..domain import tag_select as _tag_select
from ..domain import drift_ops as _drift_ops
from ..domain import entity_link as _entity_link
from ..domain import supersede_views as _supersede_views
from ..domain import map_ops as _map_ops
from ..domain import map_drift as _map_drift
from ..domain import knowledge_ops as _knowledge_ops
from ..domain import profile_ops as _profile_ops
from ..store import layer_gc as _layer_gc
from ..domain import ingest_entry as _ingest_entry
from ..domain import forget_entry as _forget_entry

from ..domain import aliases as alias_mod
from ..engine.embeddings import ENV_MODEL, embedder_pool_stats
from ..engine.manager import MemoryManager
from ..retrieval import recall_pipeline as _recall_pipeline
from ..retrieval.bm25 import BM25, rrf_fuse
from ..store.memory import (
    PIN_IMPORTANCE,  # noqa: F401  (the sole threshold definition lives in store/memory.py)
    CognitiveType,
    MemoryState,
    merge_id_lists,
    merge_tag_dicts,
)
from ..store.vector_store import multi_writer_enabled
from ..store.backend import get_backend
# The two-part orchestration of layer saving (the ledger begin/end/fail
# trio + the derived layers' "read-merge-write") lives on the store side;
# core keeps only two thin shells as the adapter layer — writing back the
# brain bookkeeping state (fingerprint stamps / sequence numbers / l0
# backlog) also happens in the shells; store returning means the write to
# disk succeeded.
from ..store.layer_save import LayerSavePorts, LayerSaveResult, LayerLedgerPorts
from ..store.layer_save import run_layer_save, save_layers_core
# I/O for the tag-change ledger: the JSONL ring's read/scan/append/trim is
# pure persistence infrastructure and belongs to the store side; only three
# thin wrappers remain below as the adapter layer — the ring knobs are
# fetched from the core namespace at call time, preserving the surface that
# lets tests monkeypatch ``core.TAG_OPS_RING``.
from ..store.tag_ops import append_ops, read_ops, scan_ops
# I/O for the save ledger: same rationale as above — the ledger line count /
# last-entry state semantics stay in the wrapper layer; the ring knobs
# (LEDGER_RING / LEDGER_TRIM_AT) are fetched from the core namespace at call
# time, preserving test_save_ledger's monkeypatch surface on
# ``core.LEDGER_RING`` (snapshotting the knobs into the store module would
# silently break the L5 ring-size assertions).
from ..store.ledger import (
    append as ledger_append,
    ledger_path,
    read_tail,
    reconcile_on_boot,
    trim as ledger_trim,
)
# Contradiction detection: query compositions over the in-store records.
# The store instance and trusted entity ids are injected by the wrapper at
# call time (self.manager.store / self._trusted_entity_ids()) — this module
# does not import vector_store and does not touch brain, same discipline as
# tag_ops / ledger.
from ..store.supersede import detect_supersede as _detect_supersede_fn
from ..store.contradictions import (
    contradiction_buckets, find_contradictions, ids_matching_tags, pool_size,
)
# Source-file signatures and drift audit: query compositions over the
# in-store records + file I/O. The store instance (iter_records), the
# source_sigs layer and the sha1 cache are injected by the wrapper; the two
# Config knobs (source_root / sig_enabled) are fetched from the core
# namespace at call time (store must not import config, same discipline as
# tag_ops / ledger).
from ..store.source_sig import (
    drift_audit, file_sig, record_source_sig as _record_source_sig,
    resolve_source_path,
)
# Tag-change undo: the undo_tag_ops method body. Ledger reading/appending,
# summarization, coverage and persistence are injected by the wrapper; the
# two gate constants (UPDATE_TAGS_SAMPLE_MAX / UPDATE_TAGS_CONFIRM_ABOVE)
# are fetched from the core namespace at call time — tests monkeypatch
# core.UPDATE_TAGS_CONFIRM_ABOVE; snapshotting them into the store module
# would silently break the gate assertions (same discipline as tag_ops /
# ledger / source_sig).
from ..store.tag_ops_undo import undo_tag_ops as _undo_tag_ops
# Layer fingerprint projection: read-only projection, writes nothing.
# ``_LAYER_ATTRS``, the fingerprint functions and the disk-stamp comparison
# are bound and injected by the wrapper; the ``layer_fp`` switch is fetched
# from the core namespace at call time (store must not import config, same
# discipline as tag_ops / ledger / source_sig).
from ..store.layer_view import (
    classify_layer_quadrants,
    compute_all_layer_fps,
    ledger_view,
    pending_layer_names,
)
from ..domain.tagging import TagExtractor, load_extractor, merge_tag_values
from ..map import file_registry as fr_mod
from ..map import store as map_mod
from ..map import auto as map_auto_mod
from ..map import mention as map_mention_mod
from ..common.sidecar_lock import SidecarBusy, SidecarUnreadable
from ..domain import topic as topic_mod
# Task profile projection: pure derivation, no retrieval, no model calls.
# The topic-layer switch is fetched by the wrapper from ``_topic_defaults()``
# at call time (discipline: making it an import-time snapshot would silently
# break HIPPOCAMPUS_TOPIC_LAYER=0); the five-tier limits are fetched from
# the core namespace at call time (the arch-frozen constant names live in
# core); the reference clock goes through manager._reference_time.
from ..domain.task_profile import Limits as _TaskProfileLimits
from ..domain.task_profile import task_profile as _task_profile
# Library-level overview card / status card projections: pure derivation,
# no retrieval. The derivation layer and reference clock are bound by the
# wrapper; the two Config switches (topic layer / status-card contradictions)
# and the five limits are fetched at call time; the five **lazy callbacks**
# (tag_coverage / knowledge_stats / contradictions_count /
# active_sessions / file_registry_summary) are passed as callables — the
# ones whose switches are off are never called at all (~0.34s per call on a
# 10k-memory library; burning them for nothing is burning them for nothing).
from ..domain.conflict_annotate import annotate_conflicts as _annotate_conflicts_fn
from ..domain.project_view import project_overview as _project_overview
from ..domain.project_view import project_status as _project_status
# Two slots of the opening package: recent memories are a query composition
# over in-store records (state filter + sort + projection, same family as
# contradictions) ⇒ store; the knowledge-map slot is a derived projection ⇒
# domain, but the topic conventions (ordered / archived state /
# summary_state) go through injection rather than import (architecture
# constraint: domain → common only); store.count() is likewise injected to
# preserve laziness.
from ..store.recent_memories import recent_memories as _recent_memories
from ..domain.knowledge_brief import knowledge_brief as _knowledge_brief
from ..domain.map_view import DetailLimits as _MapDetailLimits
from ..domain.map_view import IndexLimits as _MapIndexLimits
from ..domain.map_view import map_detail as _map_detail_fn
from ..domain.map_view import map_index as _map_index_fn
# Four statistics self-report projections: pin / recall feedback / lexical
# channel self-report + pinned-entries brief, all read-only.
# ``MemoryState.ACTIVE.value`` / ``PIN_IMPORTANCE`` live in the store package
# ⇒ fetched and injected at call time (architecture constraint: domain →
# common only); ``_pinned_brief``'s relevance goes through a callback
# (``_pin_task_relevance`` also reads ``_task_entity_ids`` ⇒ stays in brain).
from ..domain.stats_views import lexical_self_report as _lexical_self_report_fn
from ..domain.stats_views import pin_stats as _pin_stats_fn
from ..domain.stats_views import pinned_brief as _pinned_brief_fn
from ..domain.stats_views import recall_feedback_stats as _recall_feedback_stats_fn
# Orchestration for delete-by-natural-language: the body is triage + the
# deletion loop; the two brain entries recall / forget are injected as
# callables; the default gate FORGET_MIN_SIMILARITY is fetched at call time
# (patching this namespace takes effect immediately).
from ..domain.forget_flow import forget_by_query as _forget_by_query_fn
# 5-slot context assembly: the body is triage + candidate collection +
# budget loading; the six entries recall / _pinned_brief /
# _trusted_entity_ids / extract_entities / entity_facts / _render_blocks are
# injected as callables, and the zero-cost attribute reads entities /
# intents / summaries / open_topics are passed through as-is; the four
# BLOCK4_* limits are fetched from this namespace at call time (values from
# common/constants; patching takes effect immediately).
from ..domain.context_slots import memory_context as _memory_context_fn
# Library display name (extracted as a free function): it reads library.json
# (I/O), so it belongs to the library package — common is the bottom layer
# and must not depend on library; brain→library is legal. Three time-series
# statistics (recent_ingest / spark_series / recent_miss_count) moved into
# domain/project_view.py along with the overview/status cards and now have
# **no call sites left in core** ⇒ the dead imports were removed; they still
# live in common/stats.py for domain and the renderers to use.
from ..library import display_name

logger = logging.getLogger(__name__)


from .pure import _PureMixin
from .entity import _EntityLexiconMixin
from .session import _SessionMixin, _topic_defaults
from ..common.constants import *  # noqa: F401,F403  see common/constants for the constants table
# Map layer constants (see map/constants.py)
from ..map.constants import (
    MAP_DETAIL_MAX_MAPS,
    MAP_DETAIL_MAX_ROWS,
    MAP_ENTRY_KEY,
    MAP_INDEX_MAX,
    MAP_NAME_KEY,
    MAP_STALE_DAYS,
    MAP_STATUS_KEY,
    MAP_WRITE_WAIT_S,
)
# Profile graduation-slot constants live in domain/constants (a shared
# foundation is not a dumping ground); they are imported **explicitly**
# rather than relying on common's import *, to preserve the brain package's
# re-export name baseline and this module's monkeypatch surface (tests patch
# the core namespace).
from ..domain.constants import (
    PROFILE_SLOT_RECURRING,
    PROFILE_STRENGTH_DAYS,
    PROFILE_STRENGTH_MAX,
    PROFILE_STRENGTH_MIN,
)

# M1b: the three ring values are now supplied by Config (still an import-time
# snapshot, semantics unchanged); the topic defaults package is the opposite —
# the original implementation read env at call time; for the call-time
# assembly see session._topic_defaults().
_cfg_import = load_config()
TAG_OPS_RING, TAG_OPS_TRIM_AT = _cfg_import.tag_ops_ring, _cfg_import.tag_ops_ring * 2
LEDGER_RING, LEDGER_TRIM_AT = _cfg_import.save_ledger_ring, _cfg_import.save_ledger_ring * 2
L0_AUTOSAVE_EVERY = _cfg_import.l0_autosave_every


def _seed_library_map_config() -> None:
    """Production injection point for the library/map/domain(tagging) knobs.

    Same pattern as ``engine.manager._seed_store_config`` (that one handles
    store/lnn/retrieval; by dependency discipline engine must not import
    library, so these three are poured in at import time by this module —
    brain, which every production path passes through). The consumers' direct
    env reads are demoted to "direct-connect fallback"; the surfaces tests
    hit when connecting directly without going through brain are unchanged.
    For the resolution order and the zero-drift argument see
    common/config_bridge.
    """
    from ..domain import tagging as _tagging_mod
    from ..library import apply_config as _lib_apply_config
    from ..library import registry as _lib_registry_mod
    from ..map import file_registry as _fr_mod

    _lib_registry_mod.apply_config(max_resident_libs=_cfg_import.max_resident_libs)
    _lib_apply_config(library_roots=_cfg_import.library_roots)
    _fr_mod.apply_config(
        file_registry_ignore=_cfg_import.file_registry_ignore,
        file_registry_max_files=_cfg_import.file_registry_max_files,
        file_registry_budget_s=_cfg_import.file_registry_budget_s,
    )
    _tagging_mod.apply_config(tag_schema=_cfg_import.tag_schema)


_seed_library_map_config()


# The four inputs for access-driven graduation. Extracted into a module-level
# function **only** so the ``HippoBrain._update_profile`` shell can stay
# within its line budget; the switch and thresholds are still **fetched at
# call time** (monkeypatching this module's namespace takes effect). With the
# gate off it returns an empty dict — on the domain side the default
# ``recurring_slot=""`` means "no graduation entries collected".
def _profile_promotion_kwargs() -> dict[str, Any]:
    if not load_config().profile_promotion:
        return {}
    return {
        "recurring_slot": PROFILE_SLOT_RECURRING,
        "strength_min": PROFILE_STRENGTH_MIN,
        "strength_days": PROFILE_STRENGTH_DAYS,
        "strength_max": PROFILE_STRENGTH_MAX,
    }


class HippoBrain(_SessionMixin, _EntityLexiconMixin, _PureMixin):
    """Agent memory foundation: L0-L3 + index layer + usage layer + consolidation task.

    Args:
        persist_dir: Persistence directory. None means purely in-memory (lost on restart).
        embedder_backend: Embedding backend, default sentence_transformer.
        auto_consolidate_every: Run consolidation once per this many ingested items; 0 disables.
        auto_save: Auto-persist every 10 writes (passed through to MemoryManager).
        mu_importance: Importance decay constant (seconds), passed through to the scoring engine.
        w_sim / w_time / w_recall: Additive scoring weights, passed through to the scoring engine.
        supersede_enabled: Whether to enable "same fact updated" detection. **Turn it off
            when bulk-importing documents** (the md is already the current truth;
            historical narrative would be massively misjudged).
        entity_lexicon_path: Path to the project proper-noun lexicon file (one name per line).
            Defaults to ``HIPPOCAMPUS_ENTITY_LEXICON``, then to
            ``<persist_dir>/entity_lexicon.txt`` (placing it in the memory library directory
            activates it automatically).
            **Aliases can be written inline**: ``别名 => 主名`` (``->`` / ``→`` equivalent);
            both the extraction and query sides normalize to the canonical name
            (see :meth:`canonical_entity`).
        tag_schema: Extraction config for structured tags (JSON file path). Defaults to
            ``HIPPOCAMPUS_TAG_SCHEMA``, then to
            ``<persist_dir>/tag_schema.json`` (placing it in the memory library directory
            activates it automatically).
            The config can rename keys, add structural suffixes, add custom regexes — see
            ``hippocampus_mcp.domain.tagging``. Without a config, built-in generic rules apply.
        auto_backfill: Whether to backfill vectors missing for the current model automatically
            at load time. Backfilling a large library takes minutes and **synchronously
            blocks** the startup path, dragging down the MCP client; in that case set False
            and run ``scripts/reembed.py`` offline first.

    Note (v0.4): the ``manager`` held by this class is the process's **only** memory manager.
    Both server entries (the legacy tools / the v0.3 usage-layer tools) must reuse it, not
    each create their own — one instance per path would leave both pointing at the same
    ``memories.json``, causing writes invisible to each other + full overwrite on save
    (observed data loss).
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        embedder_backend: str = "sentence_transformer",
        auto_consolidate_every: int = 0,
        lambda_decay: float = 7776000.0,
        use_bm25: bool = True,
        bm25_weight: float = 1.0,
        embedder_model: str = "",
        auto_save: bool = False,
        mu_importance: float = 15552000.0,
        w_sim: float = 0.95,
        w_time: float = 0.03,
        w_recall: float = -0.02,
        supersede_enabled: bool = True,
        entity_lexicon_path: str | None = None,
        tag_schema: str | None = None,
        auto_backfill: bool = True,
        nav_penalty: float = 1.0,
        nav_tag_key: str = NAV_TAG_KEY,
        recency_window_days: int = 0,
        w_strength: float = 0.03,
        mem_history_ring: int = 200,
        usage_recency_w: float = 0.0,
        usage_recency_tau_days: float = 0.0,
        usage_recency_cap_strength: int = 0,
        adaptive_enabled: bool = False,
        adaptive_expand_floor: float = 0.6,
        adaptive_tau_stop: float = 0.1,
        adaptive_lambda: float = 0.7,
        adaptive_beta: float = 0.15,
        adaptive_gamma: float = 0.1,
        adaptive_max_k: int = 20,
    ) -> None:
        self.persist_dir = persist_dir
        # Persistence backend (json multi-file / sqlite single DB). Only the
        # read/write layer is swapped; the in-memory layer objects and the
        # business logic stay untouched. See the module notes in
        # store/backend.py.
        self._backend = get_backend(persist_dir) if persist_dir else None
        # Recall discount for navigational content (INDEX / tables of
        # contents / table headers) (appendix eight). 1.0 = no demotion; the
        # default stays unchanged — it is a mechanism that only takes effect
        # when **explicitly configured**, and must not quietly alter existing
        # ranking.
        self.nav_penalty = float(nav_penalty)
        self.nav_tag_key = str(nav_tag_key or NAV_TAG_KEY)
        # Retrieval-side staleness window (days): age > window and not pinned
        # → excluded from the result set. **Zero state changes** — no
        # deletion, no archiving, no state change; it only blocks output on
        # the retrieval side. The vector path lives in MemoryManager
        # (covering the MCP recall tool's direct path); this only passes the
        # knob down + backs up the fusion path.
        # **Default 0 = off**: old memories in the library must remain
        # findable by similar queries — return the top-ranked few rather than
        # forgetting things just because they lack "recency". Configure
        # explicitly if an expiry window is wanted.
        self.recency_window_days = max(int(recency_window_days), 0)
        self._recency_window_seconds = self.recency_window_days * 86400.0
        # Retrieval strength weight: +w_strength·log1p(strength) — a
        # consolidation trace for successful recall (testing effect).
        # 0 = off (bit-identical to the old scoring). Passed through to manager.
        self.w_strength = w_strength
        # Rewrite-history ledger ring (passed through to manager; ledger I/O
        # lives in store/mem_history, hooked into manager.update_memory /
        # store_memory's merge branch, invisible to brain).
        self.mem_history_ring = mem_history_ring
        self.auto_consolidate_every = auto_consolidate_every
        self.lambda_decay = lambda_decay
        # UR usage-recency knobs (scoring-layer only; the composition root
        # masks the numerics by the usage_recency bool). w = 0.0 (default)
        # keeps the pipeline bit-identical to pre-UR. tau is carried in
        # days (config symmetry) and converted to seconds at the
        # RecallDeps construction site.
        self.usage_recency_w = usage_recency_w
        self.usage_recency_tau_days = usage_recency_tau_days
        self.usage_recency_cap_strength = usage_recency_cap_strength
        # Adaptive result-count selector knobs (2026-11-07). Default off =
        # bit-identical to the pre-adaptive pipeline. The composition root
        # masks the numerics by the adaptive_enabled bool (numeric envs
        # alone must not enable the selector — B-type allow-list promise).
        # brain.recall treats the caller's top_k as a hard CAP clamped by
        # adaptive_max_k.
        self.adaptive_enabled = bool(adaptive_enabled)
        self.adaptive_expand_floor = float(adaptive_expand_floor)
        self.adaptive_tau_stop = float(adaptive_tau_stop)
        self.adaptive_lambda = float(adaptive_lambda)
        self.adaptive_beta = float(adaptive_beta)
        self.adaptive_gamma = float(adaptive_gamma)
        self.adaptive_max_k = int(adaptive_max_k)
        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self._bm25 = BM25()
        # BM25 incremental index: id -> content fingerprint (crc32), used to identify additions/changes/deletions
        self._bm25_indexed: dict[str, int] = {}
        # "Which path the lexical channel took on the last retrieval" — H4's
        # core benefit is **no longer building an in-memory BM25 index**
        # (saving 34.5MB), but the switch that turns it on (backend / FTS5
        # availability) has had no observability on the production side,
        # leaving step 4 of the on-call checklist with no way to verify.
        self._last_lexical_path: str = "none"
        # Fact-update detection switch. Should be off when bulk-importing
        # documents: the md file itself is already the deduplicated current
        # truth, while _detect_supersede was designed for "fact updates in
        # conversation" and would massively false-positive on historical
        # narrative ("X 改成 Y" is the norm in such documents)
        # (coverage report · high 4).
        self.supersede_enabled = supersede_enabled
        # Switch for auto-backfilling missing vectors. Backfilling a large
        # library on the startup path **synchronously blocks**, causing MCP
        # client timeouts; for large libraries run an offline reembed first,
        # then start.
        self.auto_backfill = auto_backfill

        # Embedding model resolution (coverage report · blocker 1):
        # when not explicitly specified, read HIPPOCAMPUS_EMB_MODEL (constant
        # in embeddings.ENV_MODEL; do not copy the literal here again).
        # Previously only server.py read this env; import_md.py and the
        # verification snippets in the docs defaulted straight to
        # all-MiniLM-L6-v2, so "everything imported into the library was
        # MiniLM vectors while the service queried with BGE" — dimension
        # mismatch, retrieval bound to fail. Even so, this is not the only
        # line of defense: when env is also absent the embedder falls back to
        # its default, and manager.load() aligns "the fallback value vs the
        # model actually in the library" (see
        # MemoryManager._sync_embedder_with_store) — so "forgot to set env"
        # is no longer fatal.
        if not embedder_model:
            embedder_model = load_config().emb_model
        # Proper-noun lexicon (coverage report · high 6): explicit arg > env > <memory library>/entity_lexicon.txt
        self._explicit_lexicon_path = entity_lexicon_path
        self.entity_lexicon_path = self._resolve_lexicon_path(
            entity_lexicon_path, persist_dir
        )
        self.entity_lexicon, self.entity_lexicon_aliases = self._read_lexicon(
            self.entity_lexicon_path
        )
        # Lexicon file mtime snapshot (for hot-awareness, see :meth:`_reload_lexicon_if_changed`)
        self._lexicon_mtime = self._lexicon_mtime_of(self.entity_lexicon_path)
        # Reverse-index cache of alias → canonical name (built lazily; set to None on lexicon reload to invalidate)
        self._alias_reverse: dict[str, list[str]] | None = None
        # Structured tag extractor (generalized): configurable, with no corpus-specific structural words hardcoded.
        self._explicit_tag_schema = tag_schema
        self.tag_extractor = load_extractor(tag_schema, persist_dir)

        self.manager = MemoryManager(
            persist_dir=persist_dir,
            embedder_backend=embedder_backend,
            embedder_model=embedder_model,
            lambda_decay=lambda_decay,
            mu_importance=mu_importance,
            w_sim=w_sim,
            w_time=w_time,
            w_recall=w_recall,
            auto_save=auto_save,
            recency_window_days=recency_window_days,
            w_strength=w_strength,
            mem_history_ring=mem_history_ring,
            save_ledger_hook=self._store_autosave_ledger,
        )

        # L0 raw layer
        self.raw_log: list[dict[str, Any]] = []
        # L2 summary layer
        self.summaries: list[dict[str, Any]] = []
        # Topic layer (L2T): a derived layer of L1 — a skeleton that condenses
        # flat facts into "knowledge points". For the algorithm and hard
        # constraints see the module notes in src/hippocampus_mcp/topic.py
        # (why it stays out of the retrieval path, why it is a derived layer,
        # why it is only built during consolidation / explicit calls).
        self.knowledge: dict[str, dict[str, Any]] = {}
        # Topic-layer self-report: last refresh time / duration / dropped count ("how long since it was tidied" must be visible)
        self._knowledge_last_at = None
        self._knowledge_last_took = None
        self._knowledge_last_dropped = 0
        # L3 profile layer
        self.profile: dict[str, list[dict[str, Any]]] = {}
        # What the most recent profile refresh **pruned** (appendix 19②). The
        # profile previously only grew and never pruned; "the profile lost two
        # entries" is something the host should see, so it is surfaced in the
        # return body.
        self.last_profile_pruned: list[dict[str, str]] = []
        # Index layer: entity table name -> entity dict
        self.entities: dict[str, dict[str, Any]] = {}
        # Session state / open topics / intent queue
        self.sessions: dict[str, dict[str, Any]] = {}
        self.intents: list[dict[str, Any]] = []
        self.open_topics: list[dict[str, Any]] = []
        # Session identity is split per client. Unsigned calls land on the
        # __default__ lane; behavior is bit-identical to before the rework
        # (see the current_session_id property).
        self._current_sessions: dict[str, str] = {}
        # For C-1 self-report: which old session the current session_start auto-closed (None if none).
        self._superseded_session: str | None = None
        # B (verification feedback 2026-11-01): the layer breakdown of the most
        # recent save (written/skipped/failed), echoed in the session_end
        # return body.
        self.last_save_layers: dict[str, str] = {}
        # While other clients are still in session, session_end only does this
        # lane's bookkeeping + persistence, **deferring the global wrap-up
        # (profile extraction + consolidation) to the last one to leave**.
        # Otherwise, under the daemon, A's session_end would scan the whole
        # library / touch derived layers, disturbing B's in-progress session.
        self._pending_consolidate = False
        # Sleep-batch state. The watermark is **memory-only** — after a daemon
        # restart the first round recomputes it from the replay window;
        # whether to persist it will be decided together with the storage
        # fields when the O1 strength variables get wired up (a per-record
        # stamp survives interruption better than a global watermark; do both
        # at that point).
        # _sleep_lock is a re-entry guard: the sleep batch and online writes
        # belong to different threads; running two sleep batches concurrently
        # against the same library has no benefit.
        self._last_sleep_at: float = 0.0
        self._sleep_lock = threading.Lock()
        # For each derived layer, how many entries the last "merge back from disk" brought in; surfaced by stats()
        self.last_layer_merge: dict[str, int] = {}
        # Writer visibility. _save_seq is the number of persists (part of the heartbeat).
        self._created_at = time.time()
        self._save_seq = 0
        # V1-a: autosave ledger rows use their own seq space (kind-tagged
        # rows), so layer-save rows keep their historical contiguous seqs and
        # boot reconcile matches begin/fail pairs by (seq, kind).
        self._autosave_seq = 0
        # F14: the ledger's last row and line count. The count only feeds ring
        # buffer trigger timing and is not meant to be exact — each process
        # counts its own; an overestimate just trims a bit earlier and does
        # not affect correctness.
        self._ledger_last: dict[str, Any] | None = None
        self._ledger_lines = 0
        # F14-5: the last few ledger rows (for a fail to look back at the begin with the same seq) + the boot reconciliation result.
        self._ledger_tail: list[dict[str, Any]] = []
        self._boot_reconcile: dict[str, Any] | None = None
        # F14-5: staging for the begin/end rows' layer fingerprints (begin is computed before writing to disk, end is filled in inside core).
        self._begin_layer_fps: dict[str, str] = {}
        self._end_layer_fps: dict[str, str] = {}
        self.writers_active: list[dict] = []
        self.writers_stale_cleaned = 0
        # Stamps (mtime_ns + size) of each layer file as of this process's
        # last persist. Unchanged stamp = nobody else touched it = no need to
        # read and merge from disk.
        self._layer_stamp: dict[str, tuple[int, int] | None] = {}
        # F14-1 / G0: each layer's content fingerprint (layer version number)
        # as of this process's **last persist**. "In-memory fingerprint !=
        # written fingerprint" means the layer has changes not yet persisted —
        # exactly the "which layer changed but was not written" question slice
        # 1 could not answer, and also the criterion for G0's derived-layer
        # dirty set (write only dirty layers).
        self._layer_written_fp: dict[str, str] = {}
        # Layers skipped by the last save because "content unchanged" (surfacing the dirty-set effect)
        self._layers_skipped: list[str] = []
        # Tombstones: record "what was deleted and why" so it stays traceable after physical deletion (for audit)
        self.tombstones: list[dict[str, Any]] = []
        # Derived-layer deletion ledger. Structure {f"{layer filename}::{key}":
        # deletion time}. Deleting an entry in memory does not mean "it won't
        # come back" — other processes still hold an old snapshot, and once
        # they persist, the entry gets merged back in. Only by recording "when
        # this key was deleted" can the merge block the late-arriving old
        # version.
        self._layer_del: dict[str, float] = {}
        # Self-report: how many entries this merge blocked via this ledger (0 = blocked none, possibly because there was nothing to block)
        self._layer_del_blocked = 0
        # Self-report: how many entries the ledger slimming dropped in the last save (>0 means some tombstones expired)
        self._layer_del_gc_last = 0
        # C1a: the source-file signature layer {source tag: {path, mtime_ns,
        # size, sha1, sig_at, first_seen}}. The source of truth for
        # drift_audit; leave it untouched while the content facts are
        # unchanged (prevents write amplification).
        self.source_sigs: dict[str, dict[str, Any]] = {}
        # In-memory cache of (path, mtime_ns, size) → sha1 (batch imports from
        # the same source compute it once). Pure performance cache: losing it
        # across a process restart is fine; the layer files are the source of
        # truth.
        self._sig_cache: dict[tuple[str, int, int], str] = {}
        # S2: in-memory state of the update_tags change ledger (line count +
        # last seq). The disk (tag_ops.jsonl) is the source of truth; this
        # exists only to allocate seqs and decide trimming.
        self._tag_ops_lines = 0
        self._tag_ops_last_seq = 0
        # Miss logging: automatically record when a query recalls nothing, to
        # feed attribution analysis. The test is "does the highest similarity
        # reach the hard gate", not "is the result empty" — entries rescued by
        # the BM25-only path have similarity exactly 0.0, and looking only at
        # empty results would miss them.
        self.miss_log: list[dict[str, Any]] = []
        self.miss_log_threshold: float = MISS_LOG_THRESHOLD
        self.miss_log_enabled: bool = True
        # Per-day activity counts (appendix five): ``{"YYYY-MM-DD": {"recall": n, "ingest": n}}``.
        # Parallel-phase evaluation wants "how many recalls should have
        # happened vs how many did"; previously one could only dig through the
        # total of ``recall_count``, unable to see which days saw use or
        # whether usage was balanced.
        self.daily_counts: dict[str, dict[str, int]] = {}

        self._ingest_since_consolidate = 0
        # Batch write depth (>0 means inside ingest_batch): during batches **do
        # not trigger** auto-consolidation per item; run it once at the end —
        # otherwise a 500-item batch would trigger consolidate dozens of times.
        self._batch_depth = 0
        # F11: number of raw entries written but not yet persisted to
        # brain_raw.json. Always surfaced (report even 0) — "confirmed no
        # backlog" and "never checked" must be distinguishable.
        self._l0_pending = 0
        self._l0_autosave_every = L0_AUTOSAVE_EVERY
        # R2 / L0 invariant: before each L1 persist, first flush the pending
        # L0 down (body in :meth:`_flush_l0_before_persist`). Hooked onto the
        # manager so one place covers every persist entry point: auto_save /
        # save_state / retry_save / session_end.
        self.manager.set_pre_persist_hook(self._flush_l0_before_persist)
        self._nlp = None
        self._nlp_tried = False
        # The reason spaCy is unavailable. An empty string means "available" or
        # "not tried yet". Why record it instead of silently setting None:
        # failing to install spaCy makes English entity extraction fall back
        # to heuristics, quietly degrading recall without the user ever
        # knowing — exactly the "silently wrong default" class of pit this
        # project has repeatedly hit (compare: embedder silently falling back
        # to MiniLM).
        self._nlp_error = ""

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.load()
            # F14: read the ledger's last row — the processes changed hands
            # but the books remain (that is the point of a ledger).
            self._ledger_last, self._ledger_lines, self._ledger_tail = (
                self._ledger_read_tail())
            # F14-5: boot reconciliation — if the last save did not finish /
            # disk and ledger disagree, report it on the spot instead of
            # waiting to infer it afterwards from l0_dangling.
            self._reconcile_ledger_on_boot()
            # S2: read the tag-change ledger (line count + last seq) —
            # processes changed hands, and undo still needs to find the op the
            # previous process recorded.
            self._tag_ops_lines, self._tag_ops_last_seq = (
                self._tag_ops_scan())
            # **Register presence at startup**. Only registering at persist
            # time is not enough — when two processes connect to one library
            # and neither has written anything yet, writers/ is empty, so "how
            # many hands are on this library" would answer 0 while the true
            # answer is 2.
            self._touch_writer()
            try:
                _atexit.register(self._unregister_writer)
            except Exception:  # noqa: BLE001 — unregister failure must not affect the main flow
                pass






    def reload_tag_schema(self, schema: str | None = None) -> dict[str, Any]:
        """Reload the tag extraction config (config changes take effect without restarting the service). Returns the effective rules' self-description."""
        if schema:
            self._explicit_tag_schema = schema
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        self.tag_extractor = load_extractor(self._explicit_tag_schema, persist_dir)
        return self.tag_extractor.describe()

    # Separator for lexicon alias lines. **The direction is "alias → canonical
    # name"**, matching the caller's intuition (read as "子安 refers to 顾言").
    _ALIAS_SEP = re.compile(r"\s*(?:=>|->|→|\u21d2)\s*")





    # ==================================================================
    # Entity extraction (proper-noun lexicon → heuristics; spaCy English only)
    # ==================================================================
    #: Switch to re-admit torch during the sentinel period (blocked by default; set to 1 when eval scripts need real torch)
    NER_ALLOW_TORCH_ENV = "HIPPOCAMPUS_NER_ALLOW_TORCH"

    @staticmethod
    def _block_torch_reflow() -> bool:
        """Temporarily block the torch reflow while spaCy loads.

        The top of ``thinc/compat.py`` is ``try: import torch … except
        ImportError`` — as long as torch is installed in the environment, the
        first English NER pulls it into the process (~160MB resident) and
        never releases it (confirmed by measurement; it stays resident without
        falling back).

        The approach: before ``import spacy``, place a **None sentinel** in
        ``sys.modules``; ``import torch`` then raises ImportError, landing
        exactly in thinc's fallback branch (``has_torch=False``).
        en_core_web_sm is a thinc numpy ops pipeline and does not need torch.
        Returns True when the sentinel was installed; **the caller must remove
        it**. No sentinel when torch is already present (eval scripts import
        torch first) or when ``HIPPOCAMPUS_NER_ALLOW_TORCH=1``.
        """
        if "torch" in sys.modules:
            return False
        if load_config().ner_allow_torch:
            return False
        sys.modules["torch"] = None
        return True

    @staticmethod
    def _unblock_torch_reflow(installed: bool) -> None:
        """Remove the sentinel. Only delete it if it is still there (nobody swapped it); real torch imports are unaffected afterwards."""
        if installed and sys.modules.get("torch") is None:
            del sys.modules["torch"]

    def _get_nlp(self):
        if self._nlp_tried:
            return self._nlp
        self._nlp_tried = True
        _blocked = self._block_torch_reflow()
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load("en_core_web_sm")
            self._nlp_error = ""
        except Exception as ex:  # noqa: BLE001
            self._nlp = None
            # The failure reason **must be recorded**: ner_backend / brain_stats below report it.
            self._nlp_error = f"{type(ex).__name__}: {ex}"[:200]
        finally:
            self._unblock_torch_reflow(_blocked)
        return self._nlp

    @property
    def ner_backend(self) -> str:
        """The backend actually used for English entity extraction.

        When spaCy cannot be installed (missing package / missing
        ``en_core_web_sm`` model), it falls back to pure heuristics —
        **recall quality changes but no error is raised**. This property turns
        "which backend is actually in use" into a verifiable fact instead of a
        guess. The Chinese path skips spaCy anyway and is unaffected.

        ⚠️ **Access has a side effect**: it really loads spaCy (~92MB
        resident). Its thinc.compat reflow that pulls in torch has been
        intercepted by the :meth:`_block_torch_reflow` sentinel
        ）, so the process still carries zero torch after loading. For
        scenarios that only want the self-report without triggering the load,
        use :attr:`ner_backend_lazy`.
        """
        if self._get_nlp() is not None:
            return "spacy"
        if self._nlp_error:
            return f"heuristic-only（spaCy 不可用：{self._nlp_error}）"
        return "heuristic-only"

    @property
    def ner_backend_lazy(self) -> str:
        """The **side-effect-free** version of ``ner_backend``.

        🔴 Why it exists: ``_build_brain``'s startup log goes through
        ``stats()`` → ``ner_backend``, which would force spaCy (→ thinc →
        **torch**, measured at +160MB-class resident) into the process at the
        moment the service starts — a self-report must not have a loading side
        effect. Report truthfully when tried/loaded; when not yet tried, say
        plainly "not loaded" — neither lying nor paying the cost.
        """
        if self._nlp is not None:
            return "spacy"
        if self._nlp_tried:
            if self._nlp_error:
                return f"heuristic-only（spaCy 不可用：{self._nlp_error}）"
            return "heuristic-only"
        return "lazy（未加载——首次英文实体抽取时按需加载，失败退 heuristic-only）"



    def extract_entities(self, text: str) -> list[str]:
        """Extract entity names (returns original casing; callers lower it themselves).

        Three sources, ordered by reliability:
            1. **Project proper-noun lexicon** (lexicon): one name per line,
               linked on hit — in document scenarios more accurate than any
               heuristic (the recommended approach of coverage report · high 6);
            2. Heuristic rules (Chinese sentence patterns + 《》 title brackets +
               quotes + capitalized English words);
            3. spaCy (**English text only**; Chinese text skips it —
               en_core_web_sm cannot catch Chinese).

        Use :meth:`extract_entities_with_origin` when you need to know which
        path each entity came from — origin decides whether it can take part
        in retrieval boosts / profile / contradiction detection.
        """
        return list(self.extract_entities_with_origin(text))

    def extract_entities_with_origin(self, text: str) -> dict[str, str]:
        """Same as :meth:`extract_entities`, but returns ``{entity name: origin}``.

        Implemented in
        :func:`hippocampus_mcp.domain.entity_link.extract_entities_with_origin`;
        this method only does the binding: preprocessing/CJK ratio/lexicon/
        NER/title & heuristic extraction/cleaning/alias normalization are all
        **fetched at call time** and injected — for the four-path ordering,
        dedup merge and trust semantics see the new module's docstring.
        """
        return _entity_link.extract_entities_with_origin(
            text,
            preprocess=self._preprocess_for_entities,
            cjk_ratio=self._cjk_ratio,
            cjk_skip_ratio=_CJK_SKIP_SPACY_RATIO,
            lexicon=self.entity_lexicon,
            get_nlp=self._get_nlp,
            title_entities=self._title_entities,
            heuristic_entities=self._heuristic_entities,
            clean_name=self._clean_entity_name,
            canonicalize_map=self._canonicalize_map,
            origin_rank=_ORIGIN_RANK,
            origin_lexicon=ENTITY_ORIGIN_LEXICON,
            origin_spacy=ENTITY_ORIGIN_SPACY,
            origin_title=ENTITY_ORIGIN_TITLE,
            origin_heuristic=ENTITY_ORIGIN_HEURISTIC,
        )


    _NOISE_CHARS = ("的", "是", "在", "了", "和", "与", "我", "你", "他", "她", "它",
                    "主要", "做了", "我们", "这个", "那个", "一只", "一个", "一条",
                    "叫", "有", "也", "就", "都", "还", "不", "要", "给", "对")

    # Short ASCII-only strings containing digits (H05 / B03 / ch11 / p12) are
    # almost always document numbering, not entities — the junk entities
    # called out by coverage report · high 6.
    _CODE_LIKE = re.compile(r"^(?=.*\d)[A-Za-z0-9_\-]{2,8}$")


    # Conversation-specific patterns: in Chinese, sentence frames like
    # "我叫X/X公司/养了一只X" already bracket proper names; a high-confidence
    # signal in both document and conversation scenarios, kept enabled for
    # Chinese.
    _STATEMENT_PATTERNS = (
        r"我叫([\u4e00-\u9fa5A-Za-z0-9]{1,8})",
        r"我是([\u4e00-\u9fa5]{2,4})(?:，|,|。)",
        r"叫([\u4e00-\u9fa5]{1,6})的",
        r"养(?:了|的|着)?(?:一只|一条|一个)?([\u4e00-\u9fa5]{2,4})",
        r"([\u4e00-\u9fa5A-Za-z]{2,8})(?:品类|公司|品牌|项目|平台|团队)",
        r"在([\u4e00-\u9fa5]{2,4})(?:做|工作|生活|开|读)",
    )

    # `《》` is Chinese's only unambiguous "work title" marker; kept in every scenario and given its own title trust tier.
    _TITLE_RE = re.compile(r"《([\u4e00-\u9fa5A-Za-z0-9 ]{2,12})》")

    # Quote patterns: **enabled for English text only**. In Chinese, quotes
    # mark emphasis/quotation/dialogue, not proper names — measured on a
    # Chinese novel document library: 67% (560 of 832) Chinese candidates came
    # from ASCII double quotes, catching phrases like 「这笔账没法记」; real
    # proper names are handled by the lexicon and 《》, not by quotes.
    _QUOTE_PATTERNS = (
        r"「([\u4e00-\u9fa5A-Za-z0-9 ]{2,12})」",
        r'"([\u4e00-\u9fa5A-Za-z0-9 ]{2,12})"',
    )



    def _link_entities(
        self,
        names: list[str],
        memory_id: str,
        ts: float,
        origins: dict[str, str] | None = None,
    ) -> list[str]:
        """Write entities into the entity table and maintain the inverted index; returns the list of entity ids.

        Implemented in :func:`hippocampus_mcp.domain.entity_link.link_entities`;
        this method only does the binding: the entity table itself (mutated in
        place, container not moved), the id generator and the origin constants
        are **fetched at call time** and injected; for dedup/upgrade/
        derived-quantity semantics see the new module's docstring.
        """
        return _entity_link.link_entities(
            names, memory_id, ts, origins,
            entities=self.entities, nid=_nid,
            origin_rank=_ORIGIN_RANK,
            origin_heuristic=ENTITY_ORIGIN_HEURISTIC,
        )












    # ==================================================================
    # Source-file signatures and drift audit (borrowing the predecessor's entry-to-source binding + verify four-way classification)
    # ==================================================================
    def _resolve_source_path(self, source: str) -> str | None:
        """Resolve a source tag into an absolute path on disk; return None if unresolvable.

        Resolution order: absolute path → memory library directory →
        ``HIPPOCAMPUS_SOURCE_ROOT`` → process CWD. **Unresolvable is the norm,
        not an error**: source also carries session ids and free-text origins
        ("群聊" / "口头"), which by nature have no corresponding file —
        recording a fake signature for them would only muddy the unbaselined
        pool.
        """
        return resolve_source_path(source, self.persist_dir,
                                   load_config().source_root)

    def _file_sig(self, path: str) -> dict[str, Any] | None:
        """Compute a file's {mtime_ns, size, sha1}; with a (path, mtime, size) cache.

        The cache key includes mtime/size: a changed file (mtime changed)
        naturally recomputes; unchanged, repeated ingests of the same file
        within one process hash only once. Returns None on OSError (file being
        deleted / no permission) — the caller treats it as "unresolvable",
        keeping the write path from blowing up.
        """
        return file_sig(path, self._sig_cache)

    def record_source_sig(self, source: str) -> dict[str, Any] | None:
        """Write side: record a source-file signature for a source tag (called by ingest).

        **Leave the layer untouched while the content facts are unchanged**:
        the signature entry carries sig_at; re-stamping on every ingest would
        change the layer fingerprint every time ⇒ write to disk on every save
        (a 500-item batch import = 500 disk writes). When unchanged, not even
        the dict is touched and the layer stays clean.

        Returns:
            The currently effective signature entry (newly recorded or
            existing); None when the path is unresolvable / the switch is off.
        """
        return _record_source_sig(
            source, self.source_sigs, self.persist_dir,
            load_config().source_root, self._sig_cache,
            sig_enabled=load_config().source_sig)

    def source_drift(self, source: str = "", limit: int = 50,
                     include_ok: bool = False) -> dict[str, Any]:
        """C1b: source-file drift audit — the server-side answer to "file changed but memories not updated".

        Borrowing the predecessor's verify four-way classification, grounded in
        the memory library's actual shape as:

        ============== ======================================================
        Category       Meaning
        ============== ======================================================
        ``stale``      Signed, file still present, but any of mtime/size/sha1
                       differs ⇒ memories under this source may be outdated
                       (the file is the newer truth)
        ``missing``    Signed, file gone ⇒ memories remain, source is lost
                       (imported_but_deleted); the chain back to the original
                       wording is broken
        ``unbaselined``File resolvable, but **never signed** (stock imported
                       before this feature shipped, or written during
                       ``HIPPOCAMPUS_SOURCE_SIG=0``) ⇒ not "no drift" but
                       "cannot judge" — its own category, not folded into ok
        ``orphan``     Signed, but no memory in the library references this
                       source ⇒ the signature entry itself expired (memories
                       forget-cleaned / re-imported under different tags)
        ``unresolvable``source is not a file (session id / free text) ⇒
                       informational, not part of drift judgment
        ``ok``         Signed, file present, content facts consistent
        ============== ======================================================

        Every row carries its **denominator** (per-category counts + total
        sources + referenced memory count): ``count:0`` has two possible
        truths (truly no drift vs never collected at all); without the
        denominator callers read it as the former — same discipline as
        contradictions' ``candidate_pool``.

        What it **does not do**: ``never_ingested`` (files under some
        directory that were never imported) requires knowing what the "watched
        directory" is; that is an input for project-side scripts
        (``export_memories(source=)`` can already compute it) and this tool
        does not assume one.

        Args:
            source: Audit only this one source (empty = whole library).
            limit: Max rows returned per category (0 = counts only).
            include_ok: Whether to also list ``ok`` sources (default lists only problematic ones).
        """
        # The record stream / signature layer / sha1 cache are injected by this layer; the two knobs are fetched at call time.
        return drift_audit(
            self.manager.store.iter_records(), self.source_sigs,
            source=source, limit=limit, include_ok=include_ok,
            persist_dir=self.persist_dir, source_root=load_config().source_root,
            sig_enabled=load_config().source_sig, sig_cache=self._sig_cache)


    # ==================================================================
    # Write side: L0 raw + L1 extraction
    # ==================================================================
    def ingest(
        self,
        text: str,
        speaker: str = "user",
        timestamp: float | None = None,
        importance: float | None = None,
        cognitive_type: str | None = None,
        entities: list[str] | None = None,
        session_id: str | None = None,
        source: str = "",
        detect_supersede: bool | None = None,
        granularity: str = "sentence",
        tags: dict[str, Any] | None = None,
        auto_tags: bool = False,
        client: str | None = None,
        quarantine: bool = False,
        sensitivity: str = "",
        origin: dict[str, str] | None = None,
        idem_key: str = "",
    ) -> dict[str, Any]:
        """Write one raw message and extract it into L1 memories.

        This is the host agent's most common write entry: the raw text goes
        into L0 (verbatim), the extracted sentences/facts go into L1 (with
        source_ids pointing back at L0).

        Args:
            source: Source tag (file name / relative path / session
                identifier). Written into both L0 and L1 and returned together
                in recall hits, so "which document this memory came from" can
                be traced back (coverage report · high 3).
            detect_supersede: Whether to run fact-update detection this time.
                None = follow the instance switch ``self.supersede_enabled``.
                Pass False per item for bulk imports.
            granularity: ``"sentence"`` (default, split into one L1 per
                sentence) or ``"chunk"`` (the whole passage as one L1). For
                document imports the chunk granularity can markedly reduce
                entry count and downstream noise (the configurable item of
                coverage report · medium 7).
            tags: Structured tags, e.g. ``{"章": 43, "年": "己丑"}``. Written
                into every L1 of this batch, letting recall/get_memory_context
                filter by condition — in strongly constrained writing projects
                the high-frequency "query by chapter/year" scenario becomes a
                query instead of a guess. On dedup-merge hits the union is
                taken (see :func:`merge_tag_dicts`).
                Aliases can be written inline (lexicon ``别名 => 主名``); both
                entity sides normalize.
            auto_tags: Let the server auto-extract tags from the **text body**
                (default off).
                **This is the only write-side switch that can move tag
                coverage**: backfilling afterwards can only dredge the existing
                text body (measured ceiling 17.1% on a 9359-entry Chinese novel
                library), and write-time extraction is equally bounded by the
                physical limit "the marker must be in the text" — so it cannot
                conjure coverage out of thin air, but it lets **newly written**
                entries carry tags from the start, and the host need not
                reimplement extraction. Explicitly passed ``tags`` win; the
                automatic result only fills missing keys
                (:func:`merge_tag_dicts`).

        Returns:
            Beyond the existing fields, two **write-side echoes**:
                - ``merged_into``: the existing memory hit by dedup that this
                  content was merged into (no new entry created);
                - ``neighbors``: the top3 existing memories most similar to
                  this one, with state / superseded_by / source, for spotting
                  duplicate conventions and already-superseded old versions
                  on the spot.
        """
        # Semantics of ts None (only None reads the clock; 0.0 is explicit too)
        # → ingest_entry.resolve_timestamp (SessionC batch C1).
        ts = _ingest_entry.resolve_timestamp(timestamp, now=_now)
        # P0-2 idempotency fronting (must come **before** raw_log.append): a
        # replay of the same idem_key returns the existing id directly,
        # writing neither L0 nor L1. Placing it later would leave a verbatim
        # orphan raw text — "no new memory" yet an extra audit record, the
        # hardest kind of inconsistency to debug.
        if idem_key:
            _hit = self.manager.idem_lookup(idem_key)
            if _hit:
                # The return body lives in domain/ingest_entry.py::duplicate_body
                # (the S1 structural guard pins the shell at ≤170 lines; only
                # the "look up → return" two lines stay here).
                return _ingest_entry.duplicate_body(_hit)
        # C1a: write-side source-file signature recording (drift_audit's
        # source of truth starts accumulating here). Placed first: even if
        # sentence splitting / extraction throws later, "this source was read"
        # already holds. Sources whose path cannot be resolved (session ids
        # etc.) are not recorded — see record_source_sig.
        self.record_source_sig(source)
        # auto_tags merge (body in domain/ingest_pipeline.py):
        # the two dependencies, extract and merge, are fetched at call time
        # and injected.
        tags = _ingest_pipeline.resolve_tags(
            tags, auto_tags=auto_tags, text=text,
            extract=self.tag_extractor.extract,
            merge=merge_tag_dicts)
        # Gate one · legacy-side guard: a write carrying the 「地图」+「条目」
        # reserved tags = declaring a map entry. Rejected when an **active**
        # entry with the same key already exists — "old state competing with
        # new state" is exactly the failure shape the design review rejected
        # for "treat it as mere memory"; ingest must not become a backdoor
        # around map_set / update_tags (legacy relies entirely on agent
        # discipline; this is the step that turns discipline into mechanism).
        # Archived entries with the same name are not blocked: those are dead
        # entries, and rebuilding is a legitimate move.
        self._guard_map_entry_dup(tags)
        raw_id = _nid("raw")
        # L0 assembly → ingest_entry.build_raw_entry (batch C1; the append
        # stays in the shell; session_id short-circuit / source fallback are
        # replicated in the body).
        self.raw_log.append(_ingest_entry.build_raw_entry(
            raw_id=raw_id, text=text, speaker=speaker, timestamp=ts,
            session_id=session_id, source=source,
            resolve_session=self.current_session_of, client=client))
        # F11: L0 is the only layer without automatic persistence (L1 has
        # manager.auto_save). Bookkeeping happens here; the persist timing is
        # handed to _maybe_autosave_l0: nothing persists mid-batch —
        # ingest_batch's tail-end persists once uniformly, avoiding 500
        # rewrites for a 500-item batch.
        self._l0_pending += 1

        # Automatic cognitive-type + importance determination:
        # the block-level importance is only used for the summary value
        # returned to the caller; the two determiners are injected.
        _basics = _ingest_pipeline.guess_basics(
            text, speaker, cognitive_type, importance,
            guess_cognitive=self._guess_cognitive_type,
            guess_importance=self._guess_importance)
        cognitive_type = _basics.cognitive_type
        importance = _basics.importance
        explicit_importance = _basics.explicit_importance

        # Entity origin grading: explicitly passed by the host is the most
        # trusted (manual); extracted ones are labeled by their path. origins
        # are keyed by lowercase name, aligned with _link_entities.
        # The merge body is in domain/policies.py (including alias
        # normalization and rank upgrades).
        names, origins = _policies.merge_entity_origins(
            entities, self.extract_entities_with_origin(text),
            canonical=self.canonical_entity,
        )
        entity_ids = self._link_entities(names, "", ts, origins)

        # Novelty / prediction-error gating. Before per-sentence judgment,
        # compare against the existing schema (L3 profile / L2T knowledge-point
        # summaries / high-confidence L1): confirming → no boost (duplicates
        # are dedup's job); contradiction → importance raised + the conflicting
        # object echoed back; brand-new entity → small boost. Construction body
        # in :meth:`_novelty_gate_guess` (keeps this shell thin — batch19 S1
        # structural guard); explicit importance bypasses the whole thing;
        # dedup/supersede criteria untouched; pure judgment in
        # domain/novelty_gate.py.
        novelty_events: list[Any] = []
        gate_guess = self._novelty_gate_guess(novelty_events, explicit_importance)

        # Sentence split → each sentence becomes one L1 memory (too-short
        # sentences merge into the previous one); chunk granularity keeps the
        # whole passage as one, unsplit.
        # The splitting body is in domain/ingest_pipeline.py::split_sentences.
        sentences = _ingest_pipeline.split_sentences(
            text, granularity, split_facts=self._split_facts)
        # memory_type is a loop invariant: CognitiveType lives in the store
        # package, invisible to domain; this layer computes it once and passes
        # it into the write loop.
        memory_type = _ingest_pipeline.memory_type_for(
            cognitive_type, CognitiveType.EXPERIENCE.value)
        # The sentence write-loop body is in domain/ingest_flow.py:
        # four dependencies injected — L1 port + entity linking + union merge
        # + importance judgment.
        _sw = _ingest_flow.write_sentences(
            _ingest_flow.SentenceWritePorts(
                manager=self.manager,
                link_entities=self._link_entities,
                merge_ids=merge_id_lists,
                guess_importance=gate_guess,
                # D2a: for ingest via the user exemption channel from a
                # non-home workspace, the whole batch of L1 entries carries
                # the quarantine mark (the criterion is computed in the
                # composition-root factory and passed in; brain does not
                # judge).
                quarantine=quarantine,
                sensitivity=sensitivity,
                origin=origin,
                idem_key=idem_key,
            ),
            sentences=sentences,
            importance=importance,
            explicit_importance=explicit_importance,
            memory_type=memory_type,
            cognitive_type=cognitive_type,
            timestamp=ts,
            raw_id=raw_id,
            entity_ids=entity_ids,
            source=source or "",
            tags=tags,
            names=names,
            origins=origins,
        )
        memory_ids = _sw.memory_ids
        merged_into = _sw.merged_into
        kept_separate = _sw.kept_separate
        truncations = _sw.truncations

        # Fact-update detection: whether the new memory supersedes the old
        # same fact. Tri-state normalization (None follows the instance
        # switch) → ingest_entry.resolve_supersede_switch (batch C1).
        do_detect = _ingest_entry.resolve_supersede_switch(
            detect_supersede, instance_enabled=self.supersede_enabled)
        superseded = self._detect_supersede(memory_ids, ts) if do_detect else []

        # Write-side echo of "existing memories most similar to this one"
        # (zcode requirement A3): when double-writing, the old convention is
        # visible on the spot so one can immediately decide whether to
        # explicitly supersede, instead of waiting for contradictions to
        # accumulate until consolidate.
        neighbors = self._near_duplicates(memory_ids, top_k=3)

        # C10: mark the old neighbors that were deliberately not merged
        # because they carry new hard information. Even looking only at
        # neighbors, the host can see: these two are similar but not duplicate
        # writes. The neighbor annotation body is in
        # domain/ingest_pipeline.py::annotate_neighbors.
        _ingest_pipeline.annotate_neighbors(neighbors, kept_separate)

        # Per-day write counting (appendix five): count **entries**, not
        # "calls" — ingest_batch can persist dozens of entries in one call;
        # counting calls would understate the scale of bulk imports.
        self.bump_daily("ingest", max(len(memory_ids), 1))
        # Accumulate this write's memory ids into the current session; session_end hangs them onto the L2 summary.
        self._note_session_memory(client, memory_ids)

        self._ingest_since_consolidate += 1
        if self._batch_depth == 0:
            self._maybe_auto_consolidate()
            self._maybe_autosave_l0()

        # The field order is intentional: **merged_into comes first** (adjusted
        # after user feedback). When dedup hits, "this content was not stored
        # separately" is what the host most needs to see first — it used to be
        # second to last, pushed to the end by raw_id / memory_ids, effectively
        # invisible when the host reads only the first few fields (measured: a
        # new figure got swallowed by the 0.85 threshold unnoticed, recoverable
        # only by rewording to lower similarity). JSON preserves order, so
        # order conveys priority here.
        # The result assembly body is in
        # domain/ingest_pipeline.py::assemble_ingest_result.
        result = _ingest_pipeline.assemble_ingest_result(
            merged_into=merged_into,
            raw_id=raw_id,
            memory_ids=memory_ids,
            entities=names,
            cognitive_type=cognitive_type,
            importance=importance,
            superseded=superseded,
            tags=tags,
            neighbors=neighbors,
            kept_separate=kept_separate,
            l0_pending=self._l0_pending,
            truncations=truncations,
            max_seq_length=self._embed_max_seq_length(),
        )
        # O5 echo (zero-trace discipline): add no key when the gate did not
        # run or produced no novelty signals. The key goes last — do not
        # disturb the existing key order (key order is priority; see the
        # ingest docstring).
        _novelty = _novelty_gate.summarize_events(novelty_events)
        if _novelty is not None:
            result["novelty"] = _novelty
        return result

    def _novelty_gate_guess(self, events: list[Any],
                            explicit_importance: bool) -> Callable[[str], float]:
        """O5: build the gated version of the per-sentence judgment (judgment core in ``domain/novelty_gate.py``).

        Builds the target set from the existing schema (L3 profile texts / L2T
        knowledge-point summaries / the newest ``SCHEMA_L1_CAP`` high-confidence
        L1 entries). When the schema is empty (new library), the gate is off
        (``HIPPOCAMPUS_NOVELTY_GATE``, type A default on), or the host
        explicitly passed importance (the host decides), the keyword judge is
        returned as-is — a **zero-cost bypass**. nlargest instead of a full
        sort: ingest_batch goes through here per item; sorting is
        O(N log N) per item while taking the top cap is O(N), so bulk imports
        on a large library do not pay an extra order of magnitude for this.

        Args:
            events: Out-parameter — each per-sentence judgment appends
                ``(sentence, NoveltyOutcome)`` to this list (ingest's tail-end
                folds them into the ``novelty`` echo).
            explicit_importance: Whether the host explicitly passed importance (true = bypass).
        """
        base_guess = self._guess_importance
        if explicit_importance or not load_config().novelty_gate:
            return base_guess
        targets = _novelty_gate.build_schema_targets(
            profile_texts=[
                it.get("text", "")
                for _items in self.profile.values() for it in _items],
            topic_texts=[
                t.get("summary", "") for t in self.knowledge.values()],
            high_conf_records=heapq.nlargest(
                _novelty_gate.SCHEMA_L1_CAP,
                ((r.id, r.content, r.importance, r.timestamp)
                 for r in self.manager.store.iter_records()
                 if r.state == MemoryState.ACTIVE.value
                 and r.importance >= _novelty_gate.HIGH_CONF_IMPORTANCE),
                key=lambda t: (t[3], t[0])),
            tokens_of=self._tokens_of,
        )
        if not targets:
            return base_guess
        known = frozenset(self.entities.keys())

        def _gated_guess(sent: str) -> float:
            out = _novelty_gate.gate_sentence(
                sent, base_importance=base_guess(sent),
                sentence_entities=self.extract_entities(sent),
                known_entity_keys=known, targets=targets,
                tokens_of=self._tokens_of,
                pin_importance=PIN_IMPORTANCE)
            events.append((sent, out))
            return out.importance

        return _gated_guess

    def _embed_max_seq_length(self) -> int:
        """The embedding model's hard token limit (for self-report). Returns 0 when unavailable = unknown.

        BGE / MiniLM families are 512. Text beyond it is silently dropped at
        encode time, contributing nothing to the vector — the write succeeds,
        the read-back is in the library, stats look normal; only retrieval
        quietly falls short.
        """
        try:
            emb = getattr(self.manager, "embedder", None)
            if emb is None:
                return 0
            limit = getattr(emb, "max_seq_length", 0)
            return int(limit) if limit else 0
        except Exception:  # the self-report must not in turn break the write
            return 0

    def _maybe_auto_consolidate(self) -> bool:
        """Run one consolidation when the threshold is reached (``auto_consolidate_every`` 0 means disabled).

        Extracted so :meth:`ingest_batch` can judge once **uniformly at the
        end** instead of being triggered dozens of times mid-batch.
        """
        if (self.auto_consolidate_every > 0
                and self._ingest_since_consolidate >= self.auto_consolidate_every):
            self.consolidate()
            self._ingest_since_consolidate = 0
            return True
        return False

    def _flush_l0_before_persist(self) -> None:
        """Flush the pending L0 down **before** L1 persists (R2; manager's pre_persist hook).

        Why not rely on ``_maybe_autosave_l0``: the two have **different
        units**. L1 persists every 10 **memories** (``manager.auto_save``); L0
        persists every 20 **ingest calls** (``_l0_autosave_every``), and one
        ingest splits into N L1 entries while ``_l0_pending`` only +1. So L1
        may have persisted twice while the L0 count is still 1 — if the host
        recycles the process at that moment, the on-disk L1 batch's
        ``source_ids`` point at a raw that exists only in memory. Measured: a
        ~7k-entry library had nearly 10% dangling (many still pinned), and new
        ones kept appearing after the fix shipped.

        Why flush before ``save()``: ordering decides the **direction of
        damage**. L0 first, then L1: a kill at worst leaves "raw with no
        referent" orphans (harmless, recoverable); L1 first, then L0, leaves
        "references pointing at nonexistent raw" (the original text
        permanently unrecoverable).

        No flushing mid-batch: ``ingest_batch`` persists in pairs at its
        tail-end, and each mid-batch persist would rewrite the entire L0 layer
        (a 500-item batch = 500 full overwrites).

        Known duplicate (deliberately not optimized): paths through
        ``brain.save()`` (save_state / session_end / update_tags) will, after
        this hook finishes writing L0, have ``_save_layers`` write the same
        content again per its own fingerprint bookkeeping. Explicit saves are
        not a hot path (auto-save only calls ``manager.save()`` and never
        touches ``_save_layers``), and changing ``_layer_written_fp``
        accounting from here would mean touching the ledger semantics — not
        worth it.
        """
        if self._batch_depth > 0 or self._l0_pending <= 0 or not self.persist_dir:
            return
        if not self._save_l0_only():
            # Raise so manager.save() records it in last_save_error: the
            # **silent failure** is exactly why this incident went unnoticed
            # until an external project found it. Does not block the L1
            # persist (see manager.save).
            raise RuntimeError(self.manager.last_save_error or "L0 flush failed")

    def _maybe_autosave_l0(self) -> bool:
        """F11: when accumulated new raw reaches the threshold, persist **only L0** once. Returns whether it persisted.

        Why L0 gets its own auto-persist: L1 has ``manager.auto_save`` (every
        10 steps), but L0 (``brain_raw.json``) is only persisted by explicit
        ``save()`` / ``save_state`` / ``session_end`` ⇒ **the host recycling
        the process** yields "L1 present, L0 absent", with L1's ``source_ids``
        pointing at nonexistent raw (a genuinely observed incident). This is
        not an optimization; it patches an asymmetry.
        """
        every = self._l0_autosave_every
        if every < 0:
            return False          # explicitly disabled: only surface the backlog, no auto write
        if self._batch_depth > 0:
            return False          # no persist mid-batch: ingest_batch's tail-end persists once uniformly
        if every > 0 and self._l0_pending < every:
            return False
        return self._save_l0_only()

    def _save_l0_only(self) -> bool:
        """Persist only ``brain_raw.json`` (L0). Returns True on success; False + self-report on failure.

        Implemented in :func:`hippocampus_mcp.store.layer_gc.save_l0_only`;
        this method only does binding + side-effect bookkeeping: the persist
        core lives in store (exceptions propagate up), while
        ``last_save_error`` bookkeeping / logger self-report /
        ``_l0_pending`` reset stay in the shell.
        """
        if not self.persist_dir:
            return False
        # V1-a: the L0 autosave used to be invisible to the save ledger (the
        # explicit path goes through _save_layers/run_layer_save, this one did
        # not) — the 2026-11-06 dangling audit traced 699 lost-source memories
        # to exactly this blind spot. begin carries the raw-layer fingerprint
        # so a kill mid-flush gets content-level loss judgment on reboot.
        row_layers = {"brain_raw.json": len(self.raw_log)}
        row_fps = {"brain_raw.json": self._layer_fingerprint(self.raw_log)}
        self._ledger_row("begin", "l0_autosave",
                         layers=row_layers, layer_fps=row_fps)
        try:
            self.raw_log = _layer_gc.save_l0_only(
                persist_dir=self.persist_dir, raw_log=self.raw_log,
                multi_writer_fn=multi_writer_enabled,
                read_raw_fn=self._read_raw_log_from_disk,
                merge_raw_fn=self._merge_raw_log,
                backend=self._backend, get_backend_fn=get_backend,
                layer_stamp=self._layer_stamp)
            self._l0_pending = 0
            self._ledger_row("end", "l0_autosave",
                             layers=row_layers, layer_fps=row_fps)
            return True
        except Exception as e:  # noqa: BLE001 — see docstring: do not raise, but self-report
            self._ledger_row("fail", "l0_autosave", layers=row_layers,
                             layer_fps=row_fps,
                             error=f"{type(e).__name__}: {e}")
            self.manager.last_save_error = (
                f"L0 自动落盘失败：{type(e).__name__}: {e}")
            logger.warning("[l0_autosave] %s", self.manager.last_save_error)
            return False

    def ingest_batch(
        self,
        items: list[dict[str, Any]],
        source: str = "",
        tags: dict[str, Any] | None = None,
        granularity: str = "sentence",
        detect_supersede: bool | None = None,
        save: bool = False,
        auto_tags: bool = False,
        client: str | None = None,
        quarantine: bool = False,
        sensitivity: str = "",
        origin: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Batch-write multiple messages (zcode D5)

        Per-item semantics are identical to :meth:`ingest`; the only
        differences are the three things "batch" changes:

        - **Per-item error isolation**: an exception in one item is only
          recorded in ``errors`` without aborting the batch. In a bulk
          scenario "item 37 is malformed" should not waste the first 36
          writes.
        - **No per-item auto-consolidation**: the in-batch counter is
          suppressed and :meth:`_maybe_auto_consolidate` judges once uniformly
          at the end.
        - **No per-item persistence**: ``manager.auto_save`` is disabled
          in-batch (persistence is a **full overwrite**; per-item save costs
          O(N × library size)); one save at the end as needed.

        Args:
            items: One dict per item; the fields are the same-named subset of
                :meth:`ingest` — ``text`` (**required**, non-empty string),
                ``speaker`` / ``timestamp`` / ``importance`` /
                ``cognitive_type`` / ``entities`` / ``source`` / ``tags`` /
                ``granularity`` / ``detect_supersede``.
                **Item-level fields override batch-level defaults** (no need
                to write per item when the whole batch shares a source).
                A negative ``importance`` is treated as None (auto-determined,
                consistent with the MCP side); ``entities`` accepts a list or
                a comma-separated string; ``tags`` accepts a dict or a JSON
                object string.
            source / tags / granularity / detect_supersede: Batch-level defaults.
            auto_tags: Whether the batch extracts tags from the text per item
                (see :meth:`ingest`); an item's own ``auto_tags`` wins.
            save: Whether to persist explicitly once at the end. If
                ``manager.auto_save`` was originally true, one save still
                happens at the end even with ``save=False`` (keeping the
                "auto-save on means it persists" contract, just from N times
                to 1).

        Returns:
            ``count`` succeeded · ``failed`` failed · ``total_input`` input
            count · ``raw_ids`` / ``memory_ids`` (whole batch expanded) ·
            ``entities`` (dedup union) · ``merged_into`` (each carries
            ``index`` pointing back to the input) · ``superseded`` (count) ·
            ``items`` (per-item summaries) · ``errors``
            (``[{index, error, preview}]``) · ``saved`` / ``save_warning``.
        """
        _batch_flow.check_capacity(items, max_items=INGEST_BATCH_MAX)

        rows: list[_batch_flow.CollectedRow] = []
        errors: list[dict[str, Any]] = []

        prev_auto_save = self.manager.auto_save
        self.manager.auto_save = False
        self._batch_depth += 1
        try:
            for idx, it in enumerate(items):
                # normalize_item must be inside the try: a malformed item-level
                # tags makes json.loads raise ValueError, which belongs to
                # "this item is broken", not "the whole batch is broken".
                plan: Any = None
                try:
                    plan = _batch_flow.normalize_item(
                        idx, it,
                        default_tags=tags, default_source=source,
                        default_granularity=granularity,
                        default_detect_supersede=detect_supersede,
                        default_client=client, default_auto_tags=auto_tags)
                    if isinstance(plan, _batch_flow.ItemError):
                        errors.append({"index": plan.index, "error": plan.error,
                                       "preview": plan.preview})
                        continue
                    # D2a: quarantine is a batch-level **policy** (the service
                    # layer's verdict), not open to item-level override —
                    # injected after normalize, so even if an item smuggles in
                    # a same-named key it gets overwritten.
                    payload = dict(plan.payload)
                    payload["quarantine"] = quarantine
                    payload["sensitivity"] = sensitivity
                    # P0-1: write credentials are shared by the whole batch
                    # (batch-level policy, same as the quarantine mark; a
                    # same-named key in an item gets overwritten too). Batches
                    # take no idempotency keys — the replay criterion for
                    # document imports is the source-file signature
                    # (source_drift), not a request key.
                    payload["origin"] = origin
                    res = self.ingest(**payload)
                except Exception as e:  # noqa: BLE001 — per-item isolation; one item must not destroy the batch
                    # plan may be None (normalize_item itself blew up,
                    # typically a JSON parse failure of item-level tags) — the
                    # preview must still be that text, not degrade to str of
                    # the whole dict (the convention caught by parity).
                    errors.append({
                        "index": idx, "error": f"{type(e).__name__}: {e}",
                        "preview": _batch_flow.error_preview(it, plan)})
                    continue
                rows.append(_batch_flow.collect_item(
                    res, index=idx, source=plan.source_str))
        finally:
            self._batch_depth -= 1
            self.manager.auto_save = prev_auto_save

        # One uniform consolidation at the end (suppressed in-batch)
        if rows:
            self._maybe_auto_consolidate()

        out = _batch_flow.assemble_output(
            rows, errors=errors, total_input=len(items),
            max_seq_length=self._embed_max_seq_length(),
            l0_pending=self._l0_pending)

        if rows and (save or prev_auto_save):
            try:
                self.manager.save()
                # F11: L1 persisted but L0 not persisted means L1's source_ids
                # point at nonexistent raw. Patch it here; failure does not
                # abort (same conventions as above) and goes into a warning.
                # R2: manager.save()'s pre_persist hook has already flushed L0
                # in pairs **before** L1 (pending zeroed) ⇒ flushing again here
                # would rewrite the whole layer.
                l0_ok = True if self._l0_pending <= 0 else self._save_l0_only()
                if l0_ok:
                    self.manager.last_save_error = None
                else:
                    out["save_warning"] = (
                        f"L0 未落盘，{self._l0_pending} 条 raw 仅在内存中："
                        f"{self.manager.last_save_error or '未知原因'}。"
                        f"请调用 save_state。"
                    )
                out["saved"] = True
            except Exception as e:  # noqa: BLE001
                # Same conventions as store_memory's autosave: no abort, but never silent.
                out["save_warning"] = (
                    f"落盘失败，本批记忆仅在内存中：{type(e).__name__}: {e}。"
                    f"请调用 save_state；若报 conflict 则先 load_state 或 retry_save。"
                )
                self.manager.last_save_error = f"{type(e).__name__}: {e}"
                self.manager.autosave_failures += 1
                logger.warning("[ingest_batch] %s", out["save_warning"])
        return out

    def _near_duplicates(
        self,
        memory_ids: list[str],
        top_k: int = 3,
        min_sim: float = INGEST_NEIGHBOR_MIN_SIM,
    ) -> list[dict[str, Any]]:
        """Find the **existing** memories most similar to "the memory batch just written".

        Only vector similarity is consulted (no contradiction judgment
        anymore — that is consolidate's job), but ``state`` / ``superseded_by``
        / ``source`` are returned alongside: superseded old versions and
        paraphrases from other files are exactly what the host should see on
        the spot.

        Implemented in :func:`hippocampus_mcp.domain.pure_checks.near_duplicates`;
        this method only does binding: store / vector query are **fetched at
        call time** and injected; the ``min_sim`` default keeps its def-time
        binding as before.
        """
        return _pure_checks.near_duplicates(
            memory_ids,
            store=self.manager.store,
            vector_of=self.manager.vector_of,
            top_k=top_k,
            min_sim=min_sim,
        )




    def _detect_supersede(self, new_memory_ids: list[str], ts: float) -> list[dict[str, str]]:
        """Detect "the same fact got updated": new memory carries an update marker + is highly similar to an old memory + shares content words
        → supersede only the most similar one (avoid collateral damage to unrelated memories).

        All three conditions are required: update marker + similarity
        threshold + (shared entity or high content-word overlap), so unrelated
        long-term facts like "花生过敏" are not mis-marked as superseded by
        "现在养猫".

        The content-word overlap gate was raised from 0.3 to
        ``SUPERSEDE_OVERLAP`` (0.6): under the Chinese bigram convention, two
        passages in the same document easily exceed 0.3, so a document's
        historical narrative ("X 降位" / "改成 Y" is the norm in such
        documents) gets massively misjudged as fact updates — coverage report
        · high 4 measured 28 false kills, including real facts like
        "张岱死期 1689".

        Implemented in :func:`hippocampus_mcp.store.supersede.detect_supersede`;
        this method only does binding: store / vector query / update-marker
        lexicon / the ``_is_restatement`` criterion are all **fetched at call
        time** and injected — keeping the criterion unimported preserves the
        monkeypatch surface and lets the read-side annotation reuse the same
        criterion.
        """
        return _detect_supersede_fn(
            new_memory_ids, ts,
            store=self.manager.store,
            vector_of=self.manager.vector_of,
            is_restatement=self._is_restatement,
            update_markers=UPDATE_MARKERS,
        )




    @classmethod
    def _is_restatement(
        cls,
        sim: float,
        text_a: str,
        text_b: str,
        ents_a: Any = None,
        ents_b: Any = None,
        toks_a: set[str] | None = None,
        toks_b: set[str] | None = None,
    ) -> bool:
        """Whether two memories are "talking about the same thing" (suspected different phrasings of the same fact).

        **The criterion lives in exactly one place**: the write side
        (:meth:`_detect_supersede` judges "who supersedes whom") and the read
        side (:meth:`_annotate_conflicts` judges "who coexists in conflict
        with whom") share it.

        Why share one criterion: the write side is limited by timing —
        ``supersede`` off (the document-import default), old data predating
        the mechanism, or pairs that did not match back then, all leave
        **coexisting old and new conventions**; with the same criterion the
        read side can retro-mark them. If the two sides each keep their own
        thresholds, "judged the same fact at write time but a different fact
        at read time" self-contradiction is only a matter of time.

        Conditions: similarity reaches the bar (``sim >= SUPERSEDE_SIM``)
        **and** (shared entity **or** content-word overlap reaches the bar
        (``>= SUPERSEDE_OVERLAP``). A single condition is not enough — under
        the Chinese bigram convention two passages in one document easily
        clear the overlap gate; relying on overlap alone would massively
        misjudge historical narrative (coverage report · high 4 measured 28
        false kills).

        M2: the judgment core has moved to ``domain.policies.is_restatement``
        (the threshold constants ``SUPERSEDE_SIM`` / ``SUPERSEDE_OVERLAP``
        moved with it); this is a same-name delegate — the write/read call
        sites are untouched line-for-line, and behavior is pinned by
        ``tests/test_policies_golden.py`` (44 cases).
        """
        return _policies.is_restatement(
            sim, text_a, text_b, ents_a, ents_b, toks_a, toks_b)

    # ==================================================================
    # Read side: recall + 5-slot context
    # ==================================================================
    def _sync_bm25(self) -> None:
        """Sync the BM25 index to the currently alive records (incremental; coverage report · low 14).

        Maintains an ``id -> content fingerprint`` table and compares entry by
        entry: the index is only touched when something **really was added/
        changed/removed**; unchanged documents are not re-tokenized.
        Previously every recall re-ran ``index()`` over all alive records —
        imperceptible at 8k entries, a major chunk of query latency at scale.
        """
        current: dict[str, tuple[int, str]] = {}
        for r in self.manager.store.iter_records():
            if r.state == MemoryState.ARCHIVED.value:
                continue
            current[r.id] = (zlib.crc32(r.content.encode("utf-8")), r.content)

        for did in list(self._bm25_indexed):
            if did not in current:
                self._bm25.remove_doc(did)
                del self._bm25_indexed[did]
        for did, (fp, content) in current.items():
            if self._bm25_indexed.get(did) == fp:
                continue
            self._bm25.add_doc(did, content)
            self._bm25_indexed[did] = fp

    def _fts_hits(self, query: str, top_n: int) -> dict[str, float] | None:
        """Lexical retrieval via SQLite FTS5.

        Returns:
            ``None`` = this path is unusable (json backend / FTS5 off / terms
            too short); the caller must fall back to in-memory BM25.
            ``dict`` = the index result; **even when empty** it is the
            authoritative answer — do not fall back again (falling back =
            building the in-memory index for nothing).

        Failures are always swallowed with a fallback: the full-text index is
        an **accelerator**; it breaking must not take recall down with it.
        """
        backend = getattr(self.manager.store, "_backend", None)
        if backend is None or getattr(backend, "name", "") != "sqlite":
            return None
        try:
            hits = backend.fts5_search(query, top_n)
        except Exception as e:      # noqa: BLE001
            logger.debug("FTS5 检索不可用，回退内存 BM25: %s", e)
            return None
        if hits is None:
            return None
        # The same id should appear only once in theory; dict conversion dedups as a side effect
        return dict(hits)

    def lexical_self_report(self) -> dict[str, Any]:
        """Runtime self-report of the lexical channel (BM25 / FTS5).

        Why this section exists: sqlite + FTS5's biggest selling point is
        "**no more in-memory BM25 index**, saving 34.5MB", yet production
        previously had **zero observability** for it — step 4 of the on-call
        checklist wrote `_bm25_indexed=0` as the acceptance condition, but
        `get_stats` had no such field at all, so "did we actually save it"
        could only be guessed. Per B2-4's rule, everything here is a
        **measured value**:

        - ``store_backend``: the actual storage backend (json / sqlite)
        - ``fts5_ready``: whether the FTS5 virtual table is really usable on this connection
        - ``bm25_indexed``: how many entries the in-memory BM25 currently has — **this must be 0 when H4 truly takes effect**
        - ``last_lexical_path``: which path the lexical channel actually took on the last retrieval (fts5 / memory / off)
        """
        # The body is in domain/stats_views.py: the assembly layer only
        # fetches store._backend (double-getattr defense) and the two brain
        # instance attributes.
        backend = getattr(getattr(self.manager, "store", None), "_backend", None)
        return _lexical_self_report_fn(
            backend,
            len(self._bm25_indexed),
            self._last_lexical_path,
        )

    def _ids_matching_tags(self, filter_tags: dict[str, Any] | None) -> set[str] | None:
        """Filter memory ids into a whitelist by structured tags.

        Semantics: **all** keys must match (AND); a single key's value may be
        a scalar or a list (OR within the list). Comparison normalizes through
        ``str()`` first — after a JSON round-trip a tag may be ``43`` or
        ``"43"``; without normalization the filter silently fails.

        Returns:
            None means "no filter" (callers must not treat it as an empty set).
        """
        return ids_matching_tags(self.manager.store.iter_records(), filter_tags)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        include_superseded: bool = False,
        use_mmr: bool = True,
        log_miss: bool = True,
        filter_tags: dict[str, Any] | None = None,
        annotate_conflicts: bool = True,
        client: str | None = None,
        occurred_after: float = 0.0,
        occurred_before: float = 0.0,
        use_lnn: bool = True,
        sample: bool = False,
    ) -> list[dict[str, Any]]:
        """Recall memories: hard relevance floor, MMR, pin decay, entity bonus.

        Args:
            log_miss: Log a miss when the top similarity is below threshold.
                Internal callers may pass False to avoid noise.
            annotate_conflicts: Mark suspected conflicting statements in the
                result set (annotation only: no folding, no reordering).
            filter_tags: Structured tag filter, e.g. ``{"chapter": 43}``.
                Narrow the candidate pool BEFORE retrieval; post-filtering
                would miss items ranked outside top_k. The lexical lane obeys
                the same whitelist.
            use_lnn: Forwarded to ``manager.recall``. Defaults to True, the
                value previously hard-coded here.
            sample: Forwarded to ``manager.recall``. Defaults to False.

        R1 (2026-11-06): the scoring orchestration now lives in the shared,
        side-effect-free ``retrieval/recall_pipeline``. This shell keeps only
        library guards and the three accounting effects, so the production
        MCP tool can call the very same pipeline. The two new parameters
        default to the previous behaviour exactly.
        """
        if self.manager.store.count() == 0:
            return []

        # Adaptive mode (default off): the caller's top_k becomes a hard
        # CAP clamped by adaptive_max_k, never a quota.
        adaptive_on = bool(self.adaptive_enabled)
        eff_k = min(int(top_k), self.adaptive_max_k) if adaptive_on \
            else int(top_k)

        allow_ids = self._ids_matching_tags(filter_tags)
        if allow_ids is not None and not allow_ids:
            # No tag matches is a real signal, not "library is empty".
            if log_miss and query and self.miss_log_enabled:
                self.log_miss(query, 0.0, top_k=top_k,
                              path="brain.recall:tag_empty", client=client)
            return []

        items = _recall_pipeline.run_recall_pipeline(
            self,
            _recall_pipeline.RecallRequest(
                query=query, top_k=eff_k, use_lnn=use_lnn, sample=sample,
                allow_ids=allow_ids, client=client,
                include_superseded=include_superseded, use_mmr=use_mmr,
                annotate_conflicts=annotate_conflicts,
                occurred_after=occurred_after,
                occurred_before=occurred_before,
                adaptive=adaptive_on,
            ),
            _recall_pipeline.RecallDeps(
                fuse=rrf_fuse, exp=np.exp,
                archived_state=MemoryState.ARCHIVED.value,
                superseded_state=MemoryState.SUPERSEDED.value,
                # Explicit overrides keep the core-namespace monkeypatch
                # surface alive (see the constants import comment above).
                hard_floor=RELEVANCE_HARD_FLOOR,
                entity_bonus=ENTITY_BONUS,
                literal_topn=MMR_LITERAL_EXEMPT_TOPN,
                now_fn=_now,
                strip_fn=strip_internal_fields,
                iso_fn=iso_date, weekday_fn=weekday_abbr,
                in_window_fn=in_time_window,
                out_window_fn=out_of_window,
                # UR usage-recency: constructor-injected (composition root
                # masks by the usage_recency bool); tau converts from
                # config days to pipeline seconds here. Both RecallDeps
                # construction sites (this one and the RecallService one)
                # must stay in lockstep or the two paths diverge when the
                # knob is on (R1's parity promise).
                usage_w=self.usage_recency_w,
                usage_tau=self.usage_recency_tau_days * 86400.0,
                usage_cap=self.usage_recency_cap_strength,
                # Adaptive selector: same lockstep discipline as UR — the
                # numerics ride the constructor (masked by the composition
                # root); both construction sites must match. The branch
                # flag rides the Request above.
                adaptive_k_min=2,
                adaptive_expand_floor=self.adaptive_expand_floor,
                adaptive_tau_stop=self.adaptive_tau_stop,
                adaptive_lambda=self.adaptive_lambda,
                adaptive_beta=self.adaptive_beta,
                adaptive_gamma=self.adaptive_gamma,
            ),
        )

        # Stamp only the items actually returned to the caller, never the
        # candidate pool: the floor / MMR / tag filters drop most of it, and
        # stamping candidates would keep stale memories alive forever.
        # Use the same time base as consolidate so LoCoMo-style historical
        # timestamps do not look "just recalled".
        self.manager.stamp_recalled([it["memory_id"] for it in items])
        self.bump_daily("recall")

        # A miss is judged by top similarity, not by an empty result: an item
        # rescued by the lexical lane alone has similarity 0.0.
        if log_miss and query and self.miss_log_enabled:
            top_sim = max((it["similarity"] for it in items), default=0.0)
            if top_sim < self.miss_log_threshold:
                self.log_miss(query, top_sim, top_k=top_k, path="brain.recall",
                              client=client)
        return items

    def log_miss(
        self,
        query: str,
        top_similarity: float,
        top_k: int = 0,
        path: str = "",
        client: str | None = None,
    ) -> dict[str, Any]:
        """Record one miss (for attribution analysis).

        In the parallel-phase three-column records, "queries that missed" no
        longer need manual notes: the trace carries the query, the top
        similarity at the time, the entry path and the session id, ready to
        feed evaluations like the 80% gate.
        """
        entry = {
            "ts": _now(),
            "query": (query or "")[:300],
            "top_similarity": round(float(top_similarity), 4),
            "threshold": self.miss_log_threshold,
            "top_k": top_k,
            "path": path,
            "session_id": self.current_session_of(client),
        }
        # Append + cap trimming (the deletion-must-leave-a-tombstone
        # discipline) is in domain/miss_log_flow.py:
        # max_n is fetched from this namespace; the tombstone callback is
        # injected as a bare reference.
        return _miss_log_flow.append_miss_entry(
            miss_log=self.miss_log, entry=entry, max_n=MISS_LOG_MAX,
            note_deletions=self.note_layer_deletion_items,
            ledger_name="brain_misslog.json")

    def clear_miss_log(self) -> int:
        """Clear the miss log (the landing point of the MCP tool ``miss_log(clear=True)``).

        Clearing is a **true deletion**: leave a tombstone for every entry
        first, then clear. Previously plain ``miss_log.clear()`` only worked
        when "I am the last process to write" — as soon as another process's
        stale snapshot persisted, the whole batch got merged back and the
        "cleared" the user saw was false.

        Returns:
            How many entries were cleared.
        """
        # Tombstones for the whole batch before clearing are in domain/miss_log_flow.py
        return _miss_log_flow.clear_miss_log(
            miss_log=self.miss_log,
            note_deletions=self.note_layer_deletion_items,
            ledger_name="brain_misslog.json")


    def bump_daily(self, kind: str, n: int = 1) -> None:
        """Accumulate one daily activity count (appendix five): ``kind`` is ``"recall"`` / ``"ingest"``.

        Implemented in :func:`hippocampus_mcp.domain.profile_ops.bump_daily`;
        this method only does binding: the ``daily_counts`` container is
        injected into the body (mutated in place); ``base_fn`` is lazy — the
        kind/n gate runs before the time-base computation, so with the switch
        off it is never called.
        """
        _profile_ops.bump_daily(
            kind, n, daily_counts=self.daily_counts,
            base_fn=lambda: self.manager._reference_time or _now(),
            keep_days=DAILY_COUNT_KEEP_DAYS)


    def _annotate_conflicts(self, items: list[dict[str, Any]]) -> int:
        """Annotate **suspected different phrasings of the same fact** within the result set; returns the number of pairs.

        **Annotation only**: paired entries get ``conflict_group`` (same
        number per group) / ``conflict_with`` (the other side's id list) /
        ``newer`` (the newest entry in the group). Scores, order and counts
        are never touched — the old convention may be a cited historical fact,
        or it may be the correct one; which one governs is decided by the
        authoritative source (docs); the retriever does not adjudicate for the
        user.

        The trigger is "the same semantics": the criterion comes from
        :meth:`_is_restatement`, fully consistent with the write-side
        ``supersede`` detection. What the write side cannot reach (mechanism
        off / historical data / pairs that did not match at the time) gets
        retro-marked here — this is the general backstop for the universal
        "coexisting old and new conventions" problem.

        Cost: O(n²) pairing × O(1) set intersection (content-word sets are
        precomputed, no re-tokenization). n is bounded by
        ``CONFLICT_ANNOTATE_MAX``; beyond it the whole thing is skipped
        (**affects only annotation, not the retrieval result itself**).

        Implemented in
        :func:`hippocampus_mcp.domain.conflict_annotate.annotate_conflicts`;
        this method only does binding: tokenization ``_tokens_of``, the
        criterion ``_is_restatement`` and the cap ``CONFLICT_ANNOTATE_MAX``
        are all **fetched at call time** and injected — keeping the criterion
        unimported preserves the monkeypatch surface and lets the write-side
        supersede detection reuse the same criterion.
        """
        return _annotate_conflicts_fn(
            items,
            tokens_of=self._tokens_of,
            is_restatement=self._is_restatement,
            max_items=CONFLICT_ANNOTATE_MAX,
        )

    # ==================================================================
    # Usage layer: session lifecycle + 5-slot context
    # ==================================================================
    def recent_memories(
        self, limit: int = 10, source: str | None = None
    ) -> list[dict[str, Any]]:
        """Recently written memories (newest first), the cheapest implementation of "where did we leave off".

        Only active entries; content truncated to 200 chars — the
        pre-activation package gets spliced into the prompt.
        """
        return _recent_memories(
            self.manager.store.iter_records(), limit, source,
            active=MemoryState.ACTIVE.value,
        )

    def export_memories(
        self,
        since: float = 0.0,
        source: str | None = None,
        state: str | None = None,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        """Incrementally export memories by timestamp / source / state (no vectors).

        Use it to move a slice of the memory library elsewhere (e.g.
        "everything added since the last export") or to audit by source.
        ``since`` is a Unix second-level timestamp with **>=** boundary
        semantics.

        Why not strictly greater: multiple memories may share one timestamp
        (batch writes with explicit timestamps). Strictly-greater would skip
        the whole group of "other same-ts entries beyond the previous batch's
        max ts" — silent data loss; including equality returns at most one
        extra row and the caller dedups by id.
        """
        rows = []
        for r in self.manager.store.iter_records():
            if r.timestamp < since:
                continue
            if source is not None and (r.source or "") != source:
                continue
            if state is not None and r.state != state:
                continue
            rows.append(r.to_dict())
        rows.sort(key=lambda r: r["timestamp"])
        return rows[:limit] if limit > 0 else rows





    def get_memory_context(self, query: str = "", top_k: int = 5,
                           token_budget: int = 1200,
                           item_max_chars: int = 200,
                           filter_tags: dict[str, Any] | None = None) -> dict[str, Any]:
        """Assemble the 5-slot context (modeled on bigeye AttentionFocus), ready to splice into a system prompt.

        block_1 direct hits for the current query  block_2 open questions / open intents
        block_3 timeline position (recent session summaries)  block_4 related entity states
        block_5 last conclusions / unfinished topics

        block_4 has **its own budget** (``BLOCK4_*``: at most
        ``BLOCK4_MAX_ENTITIES`` entities, at most
        ``BLOCK4_MAX_FACTS_PER_ENTITY`` facts each, whole block ≤
        ``BLOCK4_MAX_CHARS`` chars), and candidates prioritize **query-term
        entities** with the hit entities' top few as fallback — it **does not
        inflate with ``top_k``** (the main body of zcode D2).

        Args:
            item_max_chars: Truncation length for each text in the context,
                default 200.
                **0 means no truncation** — the numeric tails of ledger/
                clause-style long sentences are often cut off at 200; such
                scenarios should open it up (user feedback).
                Note it only affects the text spliced into the prompt; the
                returned ``hits`` always carry full content.
            filter_tags: Structured tag filter, forwarded to block_1's hit
                retrieval. When given, block_4 **does no query-term entity
                expansion** and only draws from the hit set (empty hits leave
                no entities to expand); this is the desired behavior: when
                writing by chapter you do not want other chapters' entity
                states mixed into the context.
        """
        # The body is in domain/context_slots.py: the six brain entries are
        # injected as callables (recall / _pinned_brief / _trusted_entity_ids /
        # extract_entities / entity_facts / _render_blocks); the four BLOCK4_*
        # limits are fetched from this namespace at call time.
        return _memory_context_fn(
            query=query, top_k=top_k, token_budget=token_budget,
            item_max_chars=item_max_chars, filter_tags=filter_tags,
            recall=self.recall,
            pinned_brief=self._pinned_brief,
            trusted_entity_ids=self._trusted_entity_ids,
            entities=self.entities,
            extract_entities=self.extract_entities,
            entity_facts=self.entity_facts,
            intents=self.intents,
            summaries=self.summaries,
            open_topics=self.open_topics,
            render_blocks=self._render_blocks,
            block4_max_entities=BLOCK4_MAX_ENTITIES,
            block4_max_facts_per_entity=BLOCK4_MAX_FACTS_PER_ENTITY,
            block4_max_chars=BLOCK4_MAX_CHARS,
            block4_hit_entity_topn=BLOCK4_HIT_ENTITY_TOPN,
        )



    # ==================================================================
    # L3 profile (conservative: only recurring / high-importance entries; conflicts only flagged, never overwritten)
    # ==================================================================
    def _update_profile(self) -> list[dict[str, Any]]:
        """Distill the L3 profile: only high-confidence entries; conflicts appended, never overwritten.

        Implemented in :func:`hippocampus_mcp.domain.profile_ops.update_profile`;
        this method only does binding: the ``profile`` container is injected
        into the body (not moved); the prepended prune's result lands in
        ``last_profile_pruned``; ``records_fn`` is lazy (the pinned scan
        materializes only after the entity scan).

        O4: the third source "access-driven graduation" is decided by
        :func:`_profile_promotion_kwargs` (switch and thresholds fetched at
        call time; closing the gate only stops new additions — stale pruning
        stays unconditional).
        """
        updates, dropped = _profile_ops.update_profile(
            self.profile, prune_fn=self.prune_stale_profile,
            entities=self.entities, entity_trusted=self._entity_trusted,
            store_get=self.manager.store.get,
            records_fn=lambda: self.manager.store.iter_records(),
            now_fn=_now, active_state=MemoryState.ACTIVE.value,
            slot_max=PROFILE_SLOT_MAX, entity_max=PROFILE_ENTITY_MAX,
            stable_slot=PROFILE_SLOT_STABLE, **_profile_promotion_kwargs())
        self.last_profile_pruned = dropped
        return updates

    def prune_stale_profile(self) -> list[dict[str, str]]:
        """Prune **stale** L3 profile entries.

        Implemented in
        :func:`hippocampus_mcp.domain.profile_ops.prune_stale_profile`;
        this method only does binding: the ``profile`` container is injected
        into the body; for the "rather miss than mis-prune" criterion and the
        three-state reason see the new module's docstring.
        O4: the graduation slot is managed **unconditionally** (independent of
        the ``profile_promotion`` gate — the switch only stops new additions
        and does not block stale pruning).
        """
        return _profile_ops.prune_stale_profile(
            self.profile, store_get=self.manager.store.get,
            stable_slot=PROFILE_SLOT_STABLE,
            active_state=MemoryState.ACTIVE.value,
            recurring_slot=PROFILE_SLOT_RECURRING)

    def pinned_total(self) -> int:
        """Total count of currently pinned and active entries — compare with what the pin slot actually brought back; the difference is what "got blocked out by pin_limit" (the self-report convention of appendix 19①)."""
        return sum(
            1 for r in self.manager.store.iter_records()
            if r.pin and r.state == MemoryState.ACTIVE.value
        )


    def _knowledge_brief(self, limit: int = 8) -> list[str]:
        """The opening package's "knowledge map" slot: a few one-line knowledge points (title + scale + one sentence).

        🔴 **Deliberately no auto-tidying here**. The opening package is on the
        hot path (once per session), and the first tidying is a whole-library
        clustering — measured at only 2-3s on a real large library. Stuffing
        those seconds into every opening buys only "in case the host wants to
        see the map". So this reads only the layer already on disk; when empty
        it says plainly "calling knowledge_map will tidy once", leaving the
        one-time cost to an explicit user call.

        The one-liner prefers the **host's synthesis** (that is the
        upper-layer knowledge) and only falls back to an extractive
        representative sentence without it. For the conventions and the reason
        behind "never auto-tidy here" see :mod:`..domain.knowledge_brief`.
        """
        return _knowledge_brief(
            self.knowledge, limit,
            topic_layer_enabled=_topic_defaults().layer_enabled,
            # Topic conventions injected (architecture constraint: domain may depend on common only)
            ordered_topics=topic_mod.ordered,
            archived_status=topic_mod.STATUS_ARCHIVED,
            summary_state=topic_mod.summary_state,
            # Ask store.count only on the "has material but never tidied" branch
            store_count=self.manager.store.count,
            clip=KNOWLEDGE_BLOCK_CLIP,
        )

    def _profile_brief(self, limit: int = 5) -> list[str]:
        out = []
        for slot, items in self.profile.items():
            for it in items[-1:]:
                out.append(f"{slot.replace('entity:', '')}: {it['text'][:80]}")
            if len(out) >= limit:
                break
        return out or ["（画像尚在积累）"]

    def _pinned_brief(self, limit: int = 3, task: str = "",
                      task_hit_ids: set[str] | None = None) -> list[dict[str, Any]]:
        """Pinned-memory slot.

        Ranking convention: **task relevance first**.

        Pinned entries' importance is generally high and close together
        (measured: book-level iron rules 0.9 vs chapter-level cards 0.8);
        ranking purely by importance really ranks "who got promoted first" —
        so the iron rules this session needs get crowded out by unrelated
        session cards. What actually discriminates is "does it match this
        task", hence three tiers:

            ``task_relevance`` = 2  the task's semantic recall hit this entry
                       = 1  intersects the task's **trusted entities**
                       = 0  unrelated

        Within a tier, sort by importance / timestamp descending; with
        ``task`` empty, fall back to pure importance (same as the old
        behavior). Only trusted sources count as entities: ~84% of heuristic
        entities are phrase noise, and letting them join "relevance" judgment
        equals a random reshuffle.

        Args:
            limit: How many to return. **Negative = turn the slot off** (empty list).
            task: This session's task description, for task-relevance-first ranking.
            task_hit_ids: memory_ids hit by the task's semantic recall (already
                computed once in ``session_start``; pass it in to reuse, no
                duplicate retrieval).
        """
        # The body is in domain/stats_views.py: relevance goes through a
        # callback (_pin_task_relevance also reads _task_entity_ids ⇒ stays in
        # brain).
        return _pinned_brief_fn(
            self.manager.store.iter_records(),
            relevance=self._pin_task_relevance,
            limit=limit, task=task, task_hit_ids=task_hit_ids,
            active_state=MemoryState.ACTIVE.value,
        )

    def _pin_task_relevance(self, task: str,
                            task_hit_ids: set[str] | None = None) -> dict[str, int]:
        """Relevance (0/1/2) of a pinned entry vs the task; for the criterion see :meth:`_pinned_brief`."""
        rel: dict[str, int] = {}
        if not (task or "").strip():
            return rel
        for mid in (task_hit_ids or ()):
            rel[mid] = 2
        ent_ids = self._task_entity_ids(task)
        if not ent_ids:
            return rel
        for r in self.manager.store.iter_records():
            if r.id in rel:
                continue
            if ent_ids.intersection(r.entity_ids or ()):
                rel[r.id] = 1
        return rel











    def relink_entities(self, only_ids: list[str] | None = None) -> dict[str, int]:
        """Re-run entity extraction over existing memories and **attach the missing links** (a required companion to lexicon growth).

        Implemented in :func:`hippocampus_mcp.domain.drift_ops.relink_entities`;
        this method only does binding: extract/link/merge/write-back are all
        **fetched at call time** and injected (``merge_id_lists`` lives in the
        store layer; domain does not import it, it goes through a callback);
        ``archived_state`` is passed by value (an unpatched constant).
        """
        return _drift_ops.relink_entities(
            self.manager.store.iter_records(), only_ids=only_ids,
            extract_fn=self.extract_entities_with_origin,
            link_fn=self._link_entities,
            merge_fn=merge_id_lists,
            update_fn=self.manager.store.update,
            archived_state=MemoryState.ARCHIVED.value,
        )


    # ==================================================================
    # Alias discovery (design tradeoff: alias "discovery" still rests entirely on humans)
    # ==================================================================
    def _stop_grams(
        self, exclude: list[str] | None = None
    ) -> tuple[set[str], set[str], dict[str, int]]:
        """Three stop-word tiers → ``(raw word set, expanded substring set, per-tier counts)``.

        Implemented in :func:`hippocampus_mcp.domain.drift_ops.stop_grams`;
        this method only does binding: the lexicon path and reader callback
        and the built-in stop-word values are **fetched at call time** and
        injected; ``lexicon_suggest`` and :meth:`discover_aliases` share the
        same noise definition — the criterion lives in exactly one place.
        """
        return _drift_ops.stop_grams(
            self.entity_lexicon_path, self._load_lexicon,
            builtin_stop=_DEFAULT_LEXICON_STOP, exclude=exclude)




    # ==================================================================
    # Intents / open intents (simplified WorkMemory)
    # ==================================================================
    def track_intent(self, text: str, due: str = "") -> str:
        entry: dict[str, Any] = {
            "id": _nid("it"), "text": text, "due": due,
            "status": "open", "created_at": _now(),
        }
        # Prospective-memory contextual triggering. At registration the
        # trigger conditions (entity names + content-word set) are **derived**
        # from the open-intent text and persisted with brain_intents.json —
        # later, when get_memory_context's query intersects a trigger set, that
        # open intent is brought back front-and-center in block_2
        # (domain/prospective.py). Switch off = no derivation (old data
        # missing the fields likewise never triggers, same as having no
        # trigger set).
        if load_config().prospective_trigger:
            tri = _prospective.derive_intent_triggers(
                text, extract_entities=self.extract_entities,
                tokens_of=self._tokens_of)
            entry["trigger_entities"] = list(tri.entities)
            entry["trigger_topic"] = sorted(tri.topic)
        self.intents.append(entry)
        return f"已登记待办：{text}"

    def resolve_intent(self, keyword: str) -> str:
        for it in self.intents:
            if it["status"] == "open" and keyword.lower() in it["text"].lower():
                it["status"] = "done"
                return f"已完成待办：{it['text']}"
        return f"未找到匹配的待办：{keyword}"

    # ==================================================================
    # Feedback loop
    # ==================================================================
    def report_feedback(self, memory_id: str, useful: bool = True) -> bool:
        """Memory actually adopted → raise importance; useless → mark it without demoting (failed experience is valuable too)."""
        rec = self.manager.store.get(memory_id)
        if rec is None:
            return False
        if useful:
            new_imp = min(rec.importance + 0.05, 1.0)
            self.manager.store.update(memory_id, importance=new_imp,
                                      pin=new_imp >= PIN_IMPORTANCE)
        else:
            # No demotion of importance, only of "recency priority": clear the pin to let it decay naturally
            self.manager.store.update(memory_id, pin=False)
        return True

    # ==================================================================
    # Memory deletion (cascade cleanup) and correction
    # ==================================================================
    def forget(self, memory_id: str, reason: str = "") -> dict[str, Any]:
        """Delete one memory and cascade-clean its traces in every derived structure.

        Key difference from ``manager.delete_memory``: the latter only
        physically removes from L1, leaving residual copies in the derived
        layers (entity inverted index / profile / superseded entries'
        back-references / L0 source), so "already deleted wrong information"
        still enters context through the profile or entity states. forget
        cleans this whole chain and leaves a tombstone for traceability.

        Cascade scope:
          1. Entity inverted index: remove references and recompute stats;
             when emptied, remove the whole entity; also drop the vanished
             entity from other memories' entity_ids (reverse dangling).
          2. Profile layer: remove entries whose evidence points at this
             memory or whose text copy equals its content.
          3. Superseded entries: if a memory was superseded by this one,
             restore it to active (the supersede rationale no longer holds).
          4. L0 source: if no other alive memory references the same raw
             message, mark it forgotten.

        Args:
            memory_id: The id of the memory to delete.
            reason: Deletion reason, written into the tombstone for audit.

        Returns:
            Cleanup report per cascade layer.
        """
        rec = self.manager.store.get(memory_id)
        if rec is None:
            return {"ok": False, "error": f"记忆不存在: {memory_id}"}

        # Tombstone first (keeps the original text so "what was deleted and
        # why" stays traceable after deletion).
        # Construction body → forget_entry.build_tombstone (batch C2; the
        # append stays in the shell; clock injection and snapshot copying are
        # replicated in the body).
        self.tombstones.append(_forget_entry.build_tombstone(
            memory_id=memory_id, content=rec.content, reason=reason,
            timestamp=rec.timestamp, state=rec.state,
            entity_ids=rec.entity_ids, source_ids=rec.source_ids, now=_now))
        # Affected entity keys (scan the entities inverted index) → forget_entry.compute_affected_keys.
        affected_keys = _forget_entry.compute_affected_keys(
            self.entities, memory_id)

        # 1) Physical deletion (the recomputations below take "alive memories" as the base and must come after this)
        self.manager.delete_memory(memory_id)
        # 1b) Topic layer: remove member references. It is a derived copy of
        #     L1; **deleting only L1 equals a fake deletion** — the "knowledge
        #     point" would keep referencing a memory that no longer exists.
        knowledge_dropped = self._forget_from_knowledge(memory_id)

        # 2) Entity inverted index: recompute + remove empty entities + reverse-clean other memories' entity_ids
        self._recompute_entity_stats(affected_keys)
        empty_keys = [
            k for k in affected_keys
            if k in self.entities and not self.entities[k]["memory_ids"]
        ]
        dead_entity_ids = {self.entities[k]["id"] for k in empty_keys}
        for k in empty_keys:
            del self.entities[k]
        relinked = 0
        if dead_entity_ids:
            for r in self.manager.store.iter_records():
                if not r.entity_ids:
                    continue
                kept = [e for e in r.entity_ids if e not in dead_entity_ids]
                if len(kept) != len(r.entity_ids):
                    self.manager.store.update(r.id, entity_ids=kept)
                    relinked += 1

        # 3) Profile layer: drop derived entries pointing at this memory
        self.profile, dropped_profile = _forget_cleanup.clean_profile(
            self.profile, memory_id=memory_id, content=rec.content,
        )

        # 4) Memories superseded by this one: restore active and clear the back-reference (the supersede rationale no longer holds)
        restored = []
        for r in self.manager.store.iter_records():
            if r.superseded_by == memory_id:
                self.manager.store.update(
                    r.id, superseded_by=None, state=MemoryState.ACTIVE.value
                )
                restored.append(r.id)

        # 5) L0 source: mark when no longer referenced by any alive memory (the original text is kept and remains auditable)
        alive_refs = {
            sid for r in self.manager.store.iter_records()
            for sid in (r.source_ids or [])
        }
        orphan_raws = _forget_cleanup.mark_orphan_raws(
            self.raw_log, source_ids=rec.source_ids,
            alive_refs=alive_refs, reason=reason,
        )

        self.save()
        return {
            "ok": True,
            "memory_id": memory_id,
            "content": rec.content,
            "reason": reason or "",
            "entities_touched": affected_keys,
            "entities_removed": empty_keys,
            "memories_relinked": relinked,
            "profile_dropped": dropped_profile,
            "restored_superseded": restored,
            "knowledge_members_dropped": knowledge_dropped,
            "raw_marked": orphan_raws,
            "remaining_l1": self.manager.store.count(),
        }

    def forget_by_query(self, query: str, top_k: int = 3,
                        dry_run: bool = True,
                        min_similarity: float | None = None,
                        allow_multiple: bool = False) -> dict[str, Any]:
        """Locate candidate memories by natural language and delete them (or only preview).

        What the host agent gets is usually natural language like
        "那条关于三月上线的记错了", not a memory_id. This method recalls first
        and then deletes, **preview-only by default** (dry_run=True); execute
        with dry_run=False after confirming the candidates, to avoid wrong
        deletions.

        **Executing deletion (dry_run=False) has two guardrails**:

        1. **Vector endorsement required**: only candidates with
           ``similarity >= min_similarity`` are considered. recall lets
           through BM25 lexical hits that are semantically unrelated; their
           similarity is **0.0**, and deleting them equals "wiping the library
           by keyword".
        2. **By default delete only the best match**: with
           ``allow_multiple=False``, even if several candidates pass the gate,
           only the top-ranked one is deleted. "Delete that wrong entry" is
           almost always singular in meaning; the extra slots are exactly
           where wrong deletions come from. To really batch-delete, pass
           ``allow_multiple=True`` explicitly.

        Neither guardrail is implemented by "raising the similarity gate" —
        relevance drifts wholesale with query length (see the measured
        distribution in the ``FORGET_MIN_SIMILARITY`` comment); a fixed-value
        gate saves neither direction. What is genuinely stable is the relative
        advantage of "the target is always #1".

        Skipped candidates are listed verbatim in ``skipped`` with a
        ``skip_reason``; the host can switch to precise deletion by memory_id
        accordingly.

        Args:
            min_similarity: The gate; None takes FORGET_MIN_SIMILARITY.
            allow_multiple: Allow deleting multiple at once; default False.
        """
        # The body is in domain/forget_flow.py: recall / forget are injected
        # as callables (never pre-invoked; with dry_run nothing is deleted at
        # all); the default gate FORGET_MIN_SIMILARITY is fetched at call time
        # — patching this namespace takes effect immediately.
        return _forget_by_query_fn(
            query=query,
            top_k=top_k,
            dry_run=dry_run,
            min_similarity=min_similarity,
            min_similarity_default=FORGET_MIN_SIMILARITY,
            allow_multiple=allow_multiple,
            recall=self.recall,
            forget=self.forget,
        )

    def list_superseded(
        self, limit: int = 200, source: str | None = None
    ) -> list[dict[str, Any]]:
        """List memories marked superseded (already superseded), including "superseded by whom".

        Implemented in
        :func:`hippocampus_mcp.domain.supersede_views.list_superseded`;
        this method only does binding: the store read and the superseded-state
        constant are **fetched at call time** and injected; the ``limit<=0``
        slicing quirk is kept as-is (see the new module's docstring).
        """
        return _supersede_views.list_superseded(
            self.manager.store.iter_records(),
            get=self.manager.store.get,
            superseded_state=MemoryState.SUPERSEDED.value,
            limit=limit, source=source)

    def un_supersede(
        self,
        memory_ids: list[str] | None = None,
        all: bool = False,  # noqa: A002 - aligned with the CLI semantics; keep this name
        source: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Restore memories wrongly marked "superseded" to active (clear superseded_by).

        Implemented in
        :func:`hippocampus_mcp.domain.supersede_views.un_supersede`;
        this method only does binding: the store write-back and state
        constants are **fetched at call time** and injected; ``save`` is a
        lazy callback (persist only with targets; preview writes and persists
        nothing).
        """
        return _supersede_views.un_supersede(
            self.manager.store.iter_records(),
            update=self.manager.store.update, save=self.save,
            superseded_state=MemoryState.SUPERSEDED.value,
            active_state=MemoryState.ACTIVE.value,
            memory_ids=memory_ids, all=all, source=source, dry_run=dry_run)

    # ==================================================================
    # Consolidation task (sleep pruning): dormant archival / profile refresh / contradiction list
    # ==================================================================
    def _contradiction_buckets(self, records) -> dict[str, list[str]]:
        """Bucket active memories by trusted entity (the input to contradiction detection).

        Only entities from trusted sources count: untrusted entities
        (especially high-frequency Chinese phrases) would stuff many unrelated
        memories into one bucket, producing sheets of false "contradictions"
        and slowing consolidation because within-bucket pairwise comparison is
        O(k²). Extracted into its own method so this filter can be verified
        directly by a unit test.
        """
        return contradiction_buckets(records, self._trusted_entity_ids())

    # ==================================================================
    # Bulk tag maintenance: stock backfill / bulk key rename / bulk key removal
    # ==================================================================
    def _select_for_tag_update(
        self,
        source: str = "",
        source_contains: str = "",
        contains: str = "",
        state: str = "",
        untagged_only: bool = False,
        missing_keys: list[str] | None = None,
        memory_ids: list[str] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Pick records whose tags need changing, by filter conditions.

        Implemented in
        :func:`hippocampus_mcp.domain.tag_select.select_for_tag_update`;
        this method only does binding: ``store.iter_records()`` is fetched at
        call time and passed in — for the filtering semantics and the
        filter-condition self-report see the new module's docstring.
        """
        return _tag_select.select_for_tag_update(
            self.manager.store.iter_records(),
            source=source, source_contains=source_contains,
            contains=contains, state=state, untagged_only=untagged_only,
            missing_keys=missing_keys, memory_ids=memory_ids,
        )

    def tag_coverage(self) -> dict[str, Any]:
        """Library-level tag coverage and key distribution (the acceptance criterion).

        Implemented in
        :func:`hippocampus_mcp.domain.supersede_views.tag_coverage`;
        this method only does binding: ``store.iter_records()`` is fetched at
        call time and passed in; for the coverage convention and top-key
        truncation see the new module's docstring.
        """
        return _supersede_views.tag_coverage(
            self.manager.store.iter_records())

    # ------------------------------------------------------------------
    # S2: the update_tags change ledger (provably undoable)
    # ------------------------------------------------------------------
    def _tag_ops_read(self) -> list[dict[str, Any]]:
        """Read all ledger entries (bad lines skipped — the ledger itself must not make undo unusable)."""
        return read_ops(self.persist_dir)

    def _tag_ops_scan(self) -> tuple[int, int]:
        """Scan the ledger once at startup: returns (line count, last seq)."""
        return scan_ops(self.persist_dir)

    def _tag_ops_append(self, entry: dict[str, Any]) -> str | None:
        """Append one ledger entry (ring-buffer trim). Returns the op id; None when there is no persist capability.

        The state triple (last_seq / lines) lands in brain instance
        attributes; the ring knobs are fetched from the core namespace **at
        call time** — tests monkeypatch ``core.TAG_OPS_RING``, and
        snapshotting them into the store module would silently break S2's
        ring-size assertions.
        """
        op_id, seq, n = append_ops(
            self.persist_dir, entry,
            self._tag_ops_last_seq, self._tag_ops_lines,
            TAG_OPS_RING, TAG_OPS_TRIM_AT,
        )
        self._tag_ops_last_seq = seq
        self._tag_ops_lines = n
        return op_id

    def undo_tag_ops(self, target: str = "last", dry_run: bool = True,
                     confirm: bool = False) -> dict[str, Any]:
        """Undo one update_tags execution.

        **Fail-closed rollback**: only entries whose current value still
        equals the recorded after are changed back to before. Entries changed
        in the meantime by a third party (another update_tags / update_memory)
        are always skipped and surfaced in ``conflicts`` — overwriting a later
        modification is worse than not rolling back.

        The undo itself is also logged (op="undo"), so a mistaken undo can be
        undone again (symmetric recovery).

        Args:
            target: op id (``tagops_3``) or ``"last"`` (the most recent).
            dry_run: Default True — produce the rollback plan only.
            confirm: Must be true when the rollback count exceeds
                :data:`UPDATE_TAGS_CONFIRM_ABOVE` (a rollback can cause damage
                just as large as a write; the gates are peers).
        """
        return _undo_tag_ops(
            self._tag_ops_read(), target, dry_run, confirm,
            store=self.manager.store,
            append_op=self._tag_ops_append,
            brief=self._tag_ops_brief,
            coverage=self.tag_coverage,
            save=self.save,
            sample_max=UPDATE_TAGS_SAMPLE_MAX,
            confirm_above=UPDATE_TAGS_CONFIRM_ABOVE,
        )


    def update_tags(
        self,
        tags: dict[str, Any] | None = None,
        clear_keys: list[str] | None = None,
        source: str = "",
        source_contains: str = "",
        contains: str = "",
        state: str = "",
        untagged_only: bool = False,
        missing_keys: list[str] | None = None,
        memory_ids: list[str] | None = None,
        mode: str = "merge",
        dry_run: bool = True,
        limit: int = 0,
        confirm: bool = False,
        undo: str = "",
    ) -> dict[str, Any]:
        """Bulk-maintain tags on existing memories (with undo).

        Entries ingested without tags / tag keys renamed later both need
        backfill. **This tool is the only path for that** — ``update_memory``
        does not accept ``tags``, so tags are frozen after the write.

        Three gates (all added after stepping in the same kind of pit):

        * **Default ``dry_run=True``**: produce the plan first. Bulk tag edits
          have no transaction; a mistake does not raise, it only shows up
          later as「查不到东西」during filtering;
        * **Above :data:`UPDATE_TAGS_CONFIRM_ABOVE` entries, ``confirm=True``
          is required**: on a real library one run touched 2,010 entries — one
          wrong character in a filter condition means 2,010 junk tags;
        * **Unchanged values are not counted in ``to_change``**: so "run it
          again" yields ``to_change=0``. Idempotency is the precondition for
          whether a bulk tool can be re-run (the idempotency stated in the
          original request is not achievable through merge semantics alone —
          under merge, on the second run every key already exists, but the
          **values may differ**).

        Execution records a **full before/after image ledger**
        (``tag_ops.jsonl``); ``undo=<op_id|"last">`` rolls back fail-closed.
        The ledger is a ring buffer (keeps the last 50 by default), and the
        undo itself is also logged — a mistaken undo can be undone again.

        Args:
            tags: The tags to write (dict). ``mode="merge"`` overwrites per
                key; ``mode="replace"`` replaces wholesale.
            clear_keys: Keys to remove (executed after the write).
            source / source_contains / contains / state / untagged_only /
                missing_keys / memory_ids: Filter conditions, all optional
                (omitted = whole library).
            mode: ``merge`` (default) / ``replace``.
            dry_run: Default True — plan only, no writes.
            limit: Max entries to process (0 = unlimited). For small trial runs first.
            confirm: Explicit confirmation when the affected count exceeds the threshold.
            undo: Undo mode. Given an op id (``tagops_3``) or ``"last"``, this
                method delegates to :meth:`undo_tag_ops`, ignoring tags /
                clear_keys.

        Returns:
            The plan or the execution result. ``sample`` gives only the first
            few entries' **before/after**.
            Since S2 the server records a **full before/after image ledger**
            (``tag_ops.jsonl``): after execution the return body carries
            ``op_id``; ``undo=tagops_N`` (or ``"last"``) rolls back
            fail-closed — only entries whose current value still equals the
            recorded ``after`` are changed back to ``before``; entries changed
            meanwhile are skipped and surfaced, never overwriting later
            modifications.
        """
        # S2: undo takes precedence over all validation — it does not depend on tags/clear_keys.
        if str(undo or "").strip():
            return self.undo_tag_ops(target=str(undo).strip(),
                                     dry_run=dry_run, confirm=confirm)
        mode, new_tags, ckeys, err = _tag_update.normalize_request(
            mode, tags, clear_keys)
        if err is not None:
            return err

        recs, filt = self._select_for_tag_update(
            source=source, source_contains=source_contains,
            contains=contains, state=state, untagged_only=untagged_only,
            missing_keys=missing_keys, memory_ids=memory_ids,
        )
        matched = len(recs)
        capped = False
        if limit and limit > 0 and len(recs) > limit:
            recs = recs[:limit]
            capped = True

        plan = _tag_update.build_plan(
            recs, mode=mode, new_tags=new_tags, ckeys=ckeys)
        to_change = len(plan)
        payload, proceed = _tag_update.build_payload(
            dry_run=dry_run, confirm=confirm, mode=mode,
            new_tags=new_tags, ckeys=ckeys, filt=filt,
            matched=matched, considered=len(recs), capped=capped,
            to_change=to_change, plan=plan,
            sample_max=UPDATE_TAGS_SAMPLE_MAX,
            confirm_above=UPDATE_TAGS_CONFIRM_ABOVE,
        )
        if not proceed:
            return payload

        result = _tag_update.apply_plan(
            plan,
            _tag_update.ApplyPorts(
                store=self.manager.store,
                map_name_key=MAP_NAME_KEY,
                now=_now,
            ),
        )
        payload["applied"] = True
        payload["changed"] = result.changed
        if result.failed:
            payload["failed"] = result.failed[:UPDATE_TAGS_SAMPLE_MAX]
            payload["failed_n"] = len(result.failed)

        # S2: record the full before/after image ledger. **Bookkeep before
        # persisting** — the in-memory tags are already changed; "what it was
        # before" must stay on disk, otherwise from the moment save fails the
        # undo basis is gone (same order as save_ledger's begin preceding the
        # write). Only entries **actually changed successfully** are recorded.
        payload["op_id"] = self._tag_ops_append({
            "op": "update_tags",
            "ts": _now(),
            "filter": filt,
            "mode": mode,
            "tags": new_tags,
            "clear_keys": ckeys,
            "to_change": to_change,
            "applied": result.changed,
            "changes": result.applied_changes,
        })

        # Persistence: the memory write side only auto-saves every 10 steps in
        # store_memory; bulk tag edits do not go through that path. Without an
        # explicit save, a daemon idle exit loses everything, **without any
        # error** (it stays visible in memory) — the topic-layer write-back
        # hit the same pit.
        try:
            self.save()
            payload["saved"] = True
        except Exception as exc:  # noqa: BLE001
            payload["saved"] = False
            payload["save_error"] = f"{type(exc).__name__}: {exc}"
            payload["hint"] = "标签已在内存生效但未落盘，请调 save_state 重试"
        payload["coverage_after"] = self.tag_coverage()
        return payload

    def consolidate(
        self,
        filter_tags: dict[str, Any] | None = None,
        max_items: int = 0,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Periodic consolidation: archive dormant memories, refresh the L3 profile, list contradiction candidates.

        Two notes:
          - **Redundant merging** is the write side's job (``store_memory``'s
            cosine dedup); no re-merging here. The old ``merged`` field was an
            unimplemented dead value and has been removed.
          - **Profile refresh** used to fire only in ``session_end``, which
            meant "re-running consolidation after deleting a memory could not
            self-heal". Now every consolidation refreshes
            (``_update_profile`` is itself idempotent; repeated calls produce
            no duplicate entries).

        Three optional parameters; **default behavior is bit-identical to
        before the rework** (``session_end`` / ``_maybe_auto_consolidate``
        both call it with no args internally):

        Args:
            filter_tags: Narrow the **dormant-archival and
                contradiction-listing** candidates to a tag domain, same
                semantics as ``recall`` / ``contradictions`` (reuses
                :meth:`_ids_matching_tags`: all-keys AND, multi-value OR
                within a key, str normalization).
                Default = whole library; the return body self-reports "no
                domain set this time" via ``scope: null``.
                ⚠️ **Not made required**: this method is auto-invoked inside
                ``session_end``; making it required would break every existing
                caller; nor does it "warn when defaulted" — that would spam a
                meaningless warning at every session wrap-up.
            max_items: How many entries this dormant archival may process at
                most (0 = unlimited). For batched archival on large libraries,
                avoiding too many sample changes in one consolidation.
            dry_run: Dry run only — list the candidates that **would** be
                archived and the contradiction candidates, with no state
                changes (no archiving, no profile / topic-layer refresh, no
                persisting).
        """
        now = self.manager._reference_time or _now()
        # Tag-domain whitelist. None = no filter (**do not** treat it as an empty set — that means "nothing found")
        allow = self._ids_matching_tags(filter_tags)

        records = list(self.manager.store.iter_records())
        # Dormant archival: long uncalled and low importance (the body is in
        # domain/consolidate_flow.py; the two constants + two state literals +
        # the archive callback are fetched at call time and injected; archive
        # is called only on real archival).
        dormant, dormant_candidates = _consolidate_flow.archive_dormant(
            records,
            now=now,
            dormant_days=DORMANT_DAYS,
            pin_importance=PIN_IMPORTANCE,
            active_state=MemoryState.ACTIVE.value,
            archived_state=MemoryState.ARCHIVED.value,
            allow=allow,
            dry_run=dry_run,
            max_items=max_items,
            archive=lambda rid, st: self.manager.store.update(rid, state=st),
        )

        # Contradiction candidates: same entity, high similarity but different
        # content and both active (prompt the host to adjudicate; no auto
        # changes). The implementation is extracted into
        # :meth:`_find_contradictions`, shared with the on-demand tool
        # :meth:`contradictions` — two implementations would sooner or later
        # compute different answers.
        #
        # **Performance (measured; this block was the single root cause of the
        # whole session_end timeout)**: the old implementation was a
        # **pair-by-pair Python loop** within buckets, each pair doing 2
        # list→numpy conversions (manager.vector_of returns a Python list) +
        # 2 norms + 1 dot product. Chinese novel library: 293 buckets /
        # 2,815,359 total pairs → **206.11s**, guaranteed to blow the host's
        # 120s timeout (a single run measured 294.9s), and retries timed out
        # the same way (no cache, no short-circuit). Now
        # store.pairwise_similarities (blocked matrix multiply) → **0.34s,
        # 610×**, numerically identical to the pairwise version (16 sampled
        # pairs, difference <1e-9).
        #
        # Also: the old implementation built dicts for **all** hit pairs
        # (measured 12,812) appended to a list, only to take [:5] at the end —
        # 12,807 dicts built for nothing. Now per-bucket top-N then global
        # top-5, plus the "different content" check the docstring always
        # promised but the code **never implemented**.
        #
        # filter_tags passes through, narrowing contradiction listing to the
        # same tag domain.
        # The contradiction projection body is in
        # domain/consolidate_flow.py::project_contradictions.
        contradictions = _consolidate_flow.project_contradictions(
            self._find_contradictions(CONTRADICTION_REPORT_MAX,
                                      filter_tags=filter_tags)
        )

        # The state-counting body is in domain/consolidate_flow.py::count_state.
        # 🔴 Must **re-iterate the store** (not the records snapshot above) —
        # dormant archival just flipped some entries to archived; the original
        # implementation's two counts reflected the **post-archival** state;
        # passing the snapshot would over-count active_memories by a few
        # entries (behavior drift).
        superseded_total = _consolidate_flow.count_state(
            self.manager.store.iter_records(), MemoryState.SUPERSEDED.value)
        report = _consolidate_flow.assemble_report(
            filter_tags=filter_tags,
            max_items=max_items,
            dry_run=dry_run,
            superseded_total=superseded_total,
            dormant=dormant,
            contradictions=contradictions,
            active_memories=_consolidate_flow.count_state(
                self.manager.store.iter_records(),
                MemoryState.ACTIVE.value),
            dormant_candidates=dormant_candidates if dry_run else None,
        )
        if dry_run:
            # Dry run stops here: list candidates only, do not touch profile / topic layer / memory states
            return report
        # Profile refresh (idempotent): lets post-delete/edit states self-heal via a consolidation re-run
        profile_updates = self._update_profile()
        # Topic layer: refresh only during consolidation (**not on the write
        # path** — the 6.7s perf_write baseline is a hard constraint, and
        # re-ordering topics on every write buys zero: the map is for humans,
        # it does not need to be real-time).
        knowledge = self.refresh_knowledge()
        report["profile_updates"] = len(profile_updates)
        # Appendix 19②: how many stale profile entries this consolidation pruned (unpinned / archived / deleted)
        report["profile_pruned"] = len(self.last_profile_pruned)
        report["knowledge"] = knowledge
        return report

    # ------------------------------------------------------------------
    # daemon idle "sleep window" — offline replay-style consolidation
    # ------------------------------------------------------------------
    def sleep_batch(
        self,
        *,
        cancel: "threading.Event | None" = None,
        budget_s: float = 120.0,
    ) -> dict[str, Any]:
        """O2 sleep batch: offline consolidation run during daemon idle (SWR replay-style, Buzsáki 2015).

        Division of labor with :meth:`consolidate`: consolidate is "periodic
        maintenance" (archive dormant, refresh derived layers, list
        contradictions); the sleep batch is "offline learning" — first replay
        recent sessions' L1 clusters for co-activation strengthening
        (biological analogue: hippocampal sharp-wave ripple replay, the engine
        of systems consolidation), then reuse consolidate for the
        consolidation trio.

        Four hard requirements (TODO O2's acceptance surface) and how each is
        met:

        - **Idempotent**: replay-candidate selection is a pure function
          (``domain.consolidate_flow.select_replay_candidates``, zero state
          changes); the consolidation trio is itself idempotent (see the
          :meth:`consolidate` docstring). The watermark ``_last_sleep_at``
          advances only on a **clean finish** — after an interruption the next
          round reselects the same window, and the replay side (once wired)
          must tolerate candidates being selected twice; this is written into
          the seam method's contract.
        - **Interruptible**: once ``cancel`` is set, check at every **unit
          boundary** (between clusters, between phases) and exit promptly.
          consolidate is a single unit: once entered it runs to completion
          (every internal step is rate-limited, seconds to minutes); check
          cancel before calling. The return body self-reports via
          ``interrupted``.
        - **Does not block online writes**: no cross-unit locks held at any
          point (the re-entry guard Lock is held only for the instant of
          ``acquire(blocking=False)``); no proactive save — persisting is left
          to the daemon's ``stop()→save_all()`` and the online write path's
          existing save moments; the sleep batch grabbing the save-file lock
          is precisely the "fighting with online writes" cautionary tale.
        - **No large model**: replay does not even use stored vectors — entity
          co-occurrence is pure data; the consolidation trio is all in-store
          computation too. The sleep can run after the model has been unloaded
          (and precisely for that reason the sleep batch never touches the
          model lifecycle clock and does not postpone idle unloading). The
          parts that need an LLM (e.g. cluster title synthesis) are left to
          the existing host-facing consolidation flow.

        Args:
            cancel: Interruption signal (the daemon sets it when a new
                connection arrives). None = not interruptible (direct script
                calls).
            budget_s: Time-budget backstop (seconds), preventing one phase
                from running away on a large library. 0/negative = unlimited.
                Over-budget is treated as interrupted
                (``interrupted="budget"``).

        Returns:
            Report body: ``replay_clusters`` (replay candidate cluster count)
            / ``replay_records`` (participating record count) /
            ``strengthened`` (entries actually strengthened this batch,
            accumulated by :meth:`_sleep_replay_cluster` via
            ``manager.reinforce``) / ``consolidated``
            (whether the consolidation trio ran) / ``dormant_archived`` etc.
            passed through from the consolidate report / ``interrupted``
            (False | "cancel" | "budget") / ``covered_pending`` (whether it
            covered the consolidation owed by session_end).
        """
        # Re-entry guard: grab the lock non-blocking; failure means another
        # thread is sleeping (in theory a backstop to the daemon single-flight
        # guard; also useful when a script double-runs).
        if not self._sleep_lock.acquire(blocking=False):
            return {"skipped": "already_running", "interrupted": False}
        try:
            return self._sleep_batch_locked(cancel=cancel, budget_s=budget_s)
        finally:
            self._sleep_lock.release()

    def _sleep_batch_locked(
        self,
        *,
        cancel: "threading.Event | None",
        budget_s: float,
    ) -> dict[str, Any]:
        now = self.manager._reference_time or _now()
        deadline = (time.monotonic() + budget_s) if budget_s and budget_s > 0 else None

        def _interrupted() -> str:
            if cancel is not None and cancel.is_set():
                return "cancel"
            if deadline is not None and time.monotonic() >= deadline:
                return "budget"
            return ""

        report: dict[str, Any] = {
            "replay_clusters": 0, "replay_records": 0, "strengthened": 0,
            "consolidated": False, "interrupted": False,
        }
        # ---- Phase R: replay candidate selection + co-activation strengthening ----------------------
        if _interrupted():
            report["interrupted"] = _interrupted()
            return report
        cfg = load_config()
        since = self._last_sleep_at or (now - cfg.sleep_replay_days * 86400.0)
        clusters = _consolidate_flow.select_replay_candidates(
            self.manager.store.iter_records(),
            now=now,
            since=since,
            active_state=MemoryState.ACTIVE.value,
            max_items=cfg.sleep_max_replay,
            min_cluster=cfg.sleep_min_cluster,
        )
        report["replay_clusters"] = len(clusters)
        report["replay_records"] = sum(len(c["ids"]) for c in clusters)
        for cluster in clusters:
            flag = _interrupted()
            if flag:
                # Interrupted at a cluster boundary: already-strengthened
                # clusters stay (idempotence semantics in the seam method);
                # unprocessed ones wait for the next round; the watermark does
                # not move ⇒ the next round reselects the same window.
                report["interrupted"] = flag
                return report
            report["strengthened"] += self._sleep_replay_cluster(cluster)

        # ---- Phase C: consolidation trio (dormant archival + profile + topic layer) --------------
        flag = _interrupted()
        if flag:
            report["interrupted"] = flag
            return report
        report["consolidated"] = True
        con = self.consolidate()
        had_pending = self._pending_consolidate
        self._pending_consolidate = False
        report["covered_pending"] = bool(had_pending)
        report["dormant_archived"] = con.get("dormant_archived", 0)
        report["profile_updates"] = con.get("profile_updates", 0)
        report["profile_pruned"] = con.get("profile_pruned", 0)

        # ---- Wrap-up: advance the watermark only on a clean finish ------------------------------
        report["interrupted"] = _interrupted() or False
        if not report["interrupted"]:
            self._last_sleep_at = now
        return report

    def _sleep_replay_cluster(self, cluster: dict[str, Any]) -> int:
        """O1 seam (wired): co-activation strengthening for one replay cluster; returns the number actually strengthened.

        Implementation = ``manager.reinforce(cluster["ids"], amount=1.0)`` —
        replay is the **offline form** of "entering the result set": only the
        retrieval strength accumulates, never touching ``recall_count``
        (candidate-set semantics, which constraint 4/5 dormant archival
        depends on) or ``last_recalled_at`` (result-set timestamp; replay is
        not a real recall, and changing it would pollute dormant archival's
        time base).

        Two idempotence contracts (written in the sleep-batch docstring's
        idempotence clause):
        1. Must tolerate the same window being selected repeatedly (after an
           interruption the watermark does not advance ⇒ reselection next
           round; strength +1 again is biologically consistent — repeated
           co-activation is repeated consolidation);
        2. See above: touch only the strength variable.
        """
        ids = [mid for mid in (cluster.get("ids") or []) if mid]
        if not ids:
            return 0
        return self.manager.reinforce(ids, amount=1.0)

    # ------------------------------------------------------------------
    # Contradiction candidates (shared by consolidate and the on-demand tool)
    # ------------------------------------------------------------------
    def _find_contradictions(
        self,
        limit: int = CONTRADICTION_REPORT_MAX,
        source: str = "",
        records: list[Any] | None = None,
        filter_tags: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Screen out active memory pairs that are "same entity + high similarity + different content".

        **Read-only**: no record state is modified. It only "hints to the host
        that two conventions may coexist here"; the adjudication belongs to
        the host (in an append-only library, the old and new versions are
        supposed to coexist; see D8).

        Args:
            limit: Max pairs to return.
            source: Only memories under one source (file name / session identifier).
            records: The candidate set; defaults to all alive records in the library.
            filter_tags: Structured tag filter with **the same semantics** as
                ``recall``'s ``filter_tags`` (all-keys AND, multi-value OR
                within a key, ``str()`` normalization), reusing
                :meth:`_ids_matching_tags`.
                Effect = narrowing "whole-library convention contradictions"
                down to "contradictions **within one domain**" — when six
                generations of conventions share one library, the global
                top-N gets filled by unrelated domains.

        Returns:
            Candidate pairs sorted by similarity descending; each carries
            ``a_id``/``b_id`` (can be handed straight to
            :meth:`update_memory` for superseding) and ``newer`` (whose
            timestamp is newer).
        """
        return find_contradictions(
            self.manager.store, limit, source=source, records=records,
            filter_tags=filter_tags, trusted_ids=self._trusted_entity_ids(),
        )

    def _contradiction_pool_size(
        self, source: str = "", filter_tags: dict[str, Any] | None = None,
    ) -> int:
        """The **denominator** of contradiction candidates: how many entries remain comparable after filtering.

        Why it exists: ``count: 0`` has two very different truths —
          ① the candidate pool is non-empty but no contradiction pairs screen
            out (= conventions consistent within the domain; good news)
          ② **the candidate pool itself is empty** (= the filter matched
            nothing, the same silent-failure shape as B2)
        Without the denominator reported, callers always read ①, so the "no
        problems in this domain" conclusion is false.
        """
        return pool_size(self.manager.store, source=source, filter_tags=filter_tags)

    def contradictions(
        self, limit: int = CONTRADICTION_REPORT_MAX, source: str = "",
        filter_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """On-demand listing of candidates suspected to be "one fact, two conventions" in the library.

        Complementary to :meth:`ingest`'s write-side echo (``neighbors``):
        that one is "who does **the just-written entry** resemble", this one
        is "which coexisting conventions does the **whole library** have right
        now". Read-only, no state changes.

        Args:
            limit: Max pairs to return (default :data:`CONTRADICTION_REPORT_MAX`).
            source: Only one source (file name / session identifier). Empty = whole library.
            filter_tags: Narrow candidates to a domain by structured tags. Empty = whole library.

        Returns:
            ``candidates`` / ``count`` / ``sim_range`` / ``note``.
            Each pair carries ids and timestamps; use :meth:`update_memory` to
            mark the old one ``superseded`` — **not** delete it; deleting
            would erase the trace that "the convention was once changed".
        """
        cands = self._find_contradictions(limit=limit, source=source,
                                          filter_tags=filter_tags)
        return {
            "candidates": cands,
            "count": len(cands),
            "limit": limit,
            "source": source,
            # J3: surface "did the narrowing actually happen this time". When
            # no tags match, the whitelist is an empty set and the result is
            # naturally 0 pairs — without this number, callers would conclude
            # "no contradictions in the domain" when the truth is "the filter
            # matched no entries" (the same silent-failure shape as B2).
            "filter_tags": dict(filter_tags or {}),
            "candidate_pool": self._contradiction_pool_size(
                source=source, filter_tags=filter_tags),
            "sim_range": [SUPERSEDE_SIM, CONTRADICTION_SIM_CEIL],
            "note": (
                "只标注不裁决：同一事实的新旧口径都要留在库里，由宿主决定取哪条。"
                "确认旧的那条作废时，用 update_memory(id, state='superseded', "
                "superseded_by=<新id>)，不要删。"
            ),
        }

    # ==================================================================
    # Statistics / persistence
    # ==================================================================
    def _recall_feedback_stats(self) -> dict[str, Any]:
        """C12 self-report: how many entries recall feedback (``last_recalled_at``) actually covers.

        ``recall_count`` records the **candidate set** (who entered the
        retrieval pool); ``last_recalled_at`` records the **result set** (who
        was actually returned to the caller). The coverage gap between them =
        the amount of "retrieved but not adopted".

        Why self-report: ``last_recalled_at`` is exactly the **time base** for
        consolidate / decay, and in historical data it went unwritten for a
        long time (measured coverage ~9.5%) — at low coverage, archival
        conclusions like "how long since this was used" become distorted,
        **without any error**.

        Convention: **no backfill** (backfilling would fabricate recall
        history); instead each stats run self-reports, handing the "how much
        to discount this credibility" judgment back to the host.
        """
        # The body is in domain/stats_views.py: a pure read-only projection.
        return _recall_feedback_stats_fn(
            self.manager.store.iter_records(),
            active_state=MemoryState.ACTIVE.value,
        )

    def _pin_stats(self) -> dict[str, Any]:
        """I4 self-report: how many pins there are, and how many were **auto-pinned by lexical rules**.

        Why self-report: ``PIN_IMPORTANCE = 0.7``, and
        :meth:`_guess_importance` gives 0.7 on hitting
        「现在 / 不再 / changed / ！」 ⇒ **auto pin**; while pin is **one-way**
        in the system (the only "exit" is an explicit caller action) ⇒ in a
        long-lived library pins pile up monotonically (measured 78% on one
        library), crowding out the usage layer's dynamic slots, **all without
        any error**.

        The "suspected automatic" criterion: ``pin=True`` and importance
        within ``_guess_importance``'s value range
        (``[PIN_IMPORTANCE, 0.75]``). This is **suspected**, not confirmed —
        a caller explicitly passing 0.7 also lands here, hence the
        ``suspected`` in the field name.
        """
        # The body is in domain/stats_views.py. PIN_IMPORTANCE is fetched at
        # call time (same semantics as the original implementation's
        # in-function try import: fall back to 0.7 on import failure);
        # a failure fetching the records themselves must also self-report
        # unknown so the other stats items do not die.
        try:
            from ..store.memory import PIN_IMPORTANCE
        except Exception:      # pragma: no cover — an import failure must not take stats down
            PIN_IMPORTANCE = 0.7
        try:
            records = self.manager.store.iter_records()
        except Exception as e:  # noqa: BLE001
            return _pin_stats_fn([], active_state=MemoryState.ACTIVE.value,
                                 error=type(e).__name__)
        return _pin_stats_fn(
            records,
            active_state=MemoryState.ACTIVE.value,
            pin_threshold=PIN_IMPORTANCE,
        )

    # ==================================================================
    # Topic layer (L2T): a derived layer of L1 — the "knowledge point" skeleton
    #
    # The algorithm, thresholds and hard constraints all live in the module
    # notes of src/hippocampus_mcp/topic.py. Only brain-side orchestration
    # (data fetching / trigger timing / self-report) sits here; pure
    # computation does not.
    # ==================================================================

    def _knowledge_items(self, require_vec: bool = True) -> list[dict[str, Any]]:
        """Pack the active L1 records into the topic module's input.

        Implemented in
        :func:`hippocampus_mcp.domain.knowledge_ops.knowledge_items`;
        this method only does binding: entity table / trust list / vector
        query are **fetched at call time** and injected; for require_vec's
        semantics see the new module's docstring.
        """
        return _knowledge_ops.knowledge_items(
            self.manager.store.iter_records(), entities=self.entities,
            trusted_ids=self._trusted_entity_ids(),
            vector_of=self.manager.vector_of,
            active_state=MemoryState.ACTIVE.value, require_vec=require_vec)

    def _reclip_knowledge(self) -> int:
        """Re-clip topic membership against the currently alive active records (derived-layer discipline)."""
        if not self.knowledge:
            return 0
        alive = [r.id for r in self.manager.store.iter_records()
                 if r.state == MemoryState.ACTIVE.value]
        return topic_mod.reclip(self.knowledge, alive)

    def _forget_from_knowledge(self, memory_id: str) -> int:
        """forget cascade: remove this memory from every topic's membership.

        Topics whose members are emptied are marked ``archived`` rather than
        deleted outright — that would be a **true deletion**, subject to the
        deletion-must-leave-a-tombstone convention (the union only adds; a
        deletion would be merged back by someone else's stale snapshot).
        """
        if not self.knowledge:
            return 0
        dropped = 0
        for t in self.knowledge.values():
            before = list(t.get("member_ids") or [])
            after = [m for m in before if m != memory_id]
            if len(after) != len(before):
                dropped += len(before) - len(after)
                t["member_ids"] = after
                t["size"] = len(after)
                if not after:
                    t["status"] = topic_mod.STATUS_ARCHIVED
        return dropped

    def refresh_knowledge(self, force: bool = False) -> dict[str, Any]:
        """Rebuild / incrementally refresh the topic layer. **Runs only on explicit calls and during consolidate**.

        Three steps (order carries meaning): re-clip against alive records
        first → then incrementally absorb unassigned ones → finally recompute
        derived fields. Reversed, deleted members would be baked into the
        centroids first.

        Args:
            force: **Redraw cluster boundaries** (when changing parameters, or
                when the old structure has grown crooked). Two differences
                from incremental: the whole layer's epoch +1 (merge overwrites
                by whole generation, so old clusters really do get cleared),
                and old host summaries migrate by member overlap (see
                :func:`topic.migrate_host_summaries`).
        """
        _td = _topic_defaults()
        if not _td.layer_enabled:
            return {"enabled": False,
                    "note": "主题层已关闭（HIPPOCAMPUS_TOPIC_LAYER=0）"}
        t0 = time.time()
        items = self._knowledge_items()
        # The alive superset (including entries without vectors for now):
        # "stale" judgment only recognizes active, not vector presence (see
        # _knowledge_items' docstring). Titles/entities/tags also need these
        # records' text, so they are passed to refresh too — otherwise topics
        # would show "no members".
        info = {str(it["id"]): it for it in self._knowledge_items(require_vec=False)}
        self.knowledge, out, took, dropped = _refresh_flow.collect(
            items, info, self.knowledge, force=force,
            defaults=_td, max_topics=_topic_defaults().max_topics,
            now=self.manager._reference_time or _now(),
            t0=t0, clock=time.time,
            layer_epoch=topic_mod.layer_epoch, reclip=topic_mod.reclip,
            build=topic_mod.build, refresh=topic_mod.refresh,
            migrate_host_summaries=topic_mod.migrate_host_summaries,
            coverage_of=topic_mod.coverage_of, by_status=topic_mod.by_status,
            summary_kind_host=topic_mod.SUMMARY_KIND_HOST)
        self._knowledge_last_at = _now()
        self._knowledge_last_took = took
        self._knowledge_last_dropped = dropped
        return out

    def knowledge_stats(self) -> dict[str, Any]:
        """brain_stats self-report: the topic layer's current state (coverage and last refresh time included).

        Implemented in
        :func:`hippocampus_mcp.domain.knowledge_ops.knowledge_stats`;
        this method only does binding: the topic layer's capabilities are
        injected via :class:`TopicPort` (same layer, no cross-references); the
        last-refresh tri-state is fetched at call time.
        """
        return _knowledge_ops.knowledge_stats(
            _knowledge_ops.TopicPort(
                defaults_fn=_topic_defaults, ordered=topic_mod.ordered,
                pending_summary_ids=topic_mod.pending_summary_ids,
                representative_ids=topic_mod.representative_ids,
                summary_state=topic_mod.summary_state,
                apply_host_summary=topic_mod.apply_host_summary,
                coverage_of=topic_mod.coverage_of,
                by_status=topic_mod.by_status,
                host_summary_stats=topic_mod.host_summary_stats,
                evidence_clip=topic_mod.TOPIC_EVIDENCE_CLIP),
            knowledge=self.knowledge,
            records=self.manager.store.iter_records(),
            active_state=MemoryState.ACTIVE.value,
            last_at=self._knowledge_last_at,
            last_took=self._knowledge_last_took,
            last_dropped=self._knowledge_last_dropped)

    def knowledge_map(self, limit: int = 20,
                      include_archived: bool = False) -> dict[str, Any]:
        """The library's knowledge structure: topic list + one sentence per topic + scale + open items.

        **Read-only**. It answers "what knowledge points does this library
        have" — a question that could not be asked before: L1 is flat, L2 has
        only session-level summaries that fall dormant right after being
        written, and L3 is hot entries, not knowledge.
        """
        _td = _topic_defaults()
        if not _td.layer_enabled:
            return {"enabled": False, "topics": [],
                    "note": "主题层已关闭（HIPPOCAMPUS_TOPIC_LAYER=0）"}
        # Never tidied in this library → build once automatically on first
        # call. Deliberately **not** on every call: the map is for humans, it
        # does not need to be real-time; afterwards it refreshes incrementally
        # only during consolidate.
        auto_built = False
        if (not self.knowledge and self.manager.store.count()):
            self.refresh_knowledge()
            auto_built = True
        # The three blocks' bodies are in domain/knowledge_map_view.py; the
        # topic package's five symbols are fetched at call time and injected
        # (domain same layer, cross-references forbidden). stats / open_* are
        # fetched by value; playbook is a lazy callback (called only when
        # pending is non-empty, call counts aligned with the original).
        return _knowledge_map_view.collect(
            self.knowledge,
            include_archived=include_archived,
            limit=limit,
            defaults=_td,
            ordered=topic_mod.ordered,
            status_archived=topic_mod.STATUS_ARCHIVED,
            summary_kind_host=topic_mod.SUMMARY_KIND_HOST,
            signature_of=topic_mod.signature_of,
            pending_summary_ids=topic_mod.pending_summary_ids,
            stats=self.knowledge_stats(),
            open_topics=[t.get("text") for t in self.open_topics[-5:]],
            open_intents=[i.get("text") for i in self.intents
                          if i.get("status") == "open"][:5],
            playbook=lambda n, m: self._knowledge_playbook(n, m),
            auto_built=auto_built,
        )


    def knowledge_topic(self, topic_id: str = "", action: str = "brief",
                        summary: str = "",
                        evidence: list[str] | None = None,
                        limit: int = 30,
                        topics: list[str] | None = None,
                        items: list[dict[str, Any]] | None = None,
                        per_topic: int = 5,
                        batch: int = 10,
                        fras: dict[str, Any] | None = None) -> dict[str, Any]:
        """The topic layer's read/write channel: produce material (brief) / write back a synthesis (summarize) / undo (revert).

        This is the real entrance of "upper-layer knowledge". The first
        version could only produce representative sentences — **excerpts** of
        the original texts closest to the centroid, answering "what is this
        topic roughly about"; the "synthesis" the user wants writes N facts
        into one conclusion, a step that requires intelligence, and "no LLM
        embedded in the production path" is this project's boundary (TODO J
        section D1). So this is a **write-back channel**: this method produces
        the material → the host synthesizes → this method persists it.

        After persisting, ``summary_kind="host"``, and the discipline from
        then on:
          * refresh (consolidation) **only fills extractive** summaries, never
            overwriting the host synthesis;
          * layer merge (multi-process union) prefers **host** over timestamps;
          * when the member set changes, only mark ``stale``, never rewrite
            content — this module cannot judge whether content is right, it
            only surfaces the fact that "the members the synthesis was based
            on have changed".

        S3: ``summarize`` can carry a lightweight **FRAS four-field structure**
        (``fras``): ``f`` category (used directly as the title, fixing
        "multiple clusters with the same name") / ``r`` related entities and
        topics / ``a`` citable conclusions (when only fras is passed, it can
        replace summary) / ``s`` constraints that cannot be inferred
        (admission: not derivable + mistake prevention; when there are none,
        write "无" explicitly). All four keys are required; a structurally
        invalid entry rejects the whole item. Free-text write-back is still
        supported (just do not pass fras).

        Args:
            topic_id: Topic id (from ``knowledge_map``).
            action: ``brief`` produce material / ``summarize`` write back /
                ``revert`` undo / ``brief_batch`` / ``summarize_batch`` batch /
                ``rebuild`` redraw cluster boundaries (epoch +1) /
                ``refresh`` incremental refresh.
            summary: The synthesis text (required for summarize, ≤2000 chars;
                can be omitted when fras is passed).
            evidence: Sources (L1 memory ids, **must be members of this
                topic**), ≤20.
            limit: Number of material entries (1–200, default 30).
            fras: Optional four-field structure, see above. Each item of
                ``summarize_batch`` may also carry its own ``fras`` key.
        """
        if not _topic_defaults().layer_enabled:
            return {"ok": False, "enabled": False,
                    "error": "主题层已关闭（HIPPOCAMPUS_TOPIC_LAYER=0）"}
        act = str(action or "brief").strip().lower()
        if act not in ("brief", "summarize", "revert",
                       "brief_batch", "summarize_batch",
                       "rebuild", "refresh"):
            return {"ok": False, "enabled": True,
                    "error": ("action 只支持 brief / summarize / revert / "
                              "brief_batch / summarize_batch / rebuild / "
                              "refresh，收到 " + str(action))}
        auto = False
        if not self.knowledge and self.manager.store.count():
            self.refresh_knowledge()
            auto = True
        # Batch actions go first: they have no concept of "a single topic_id";
        # forcing them through the single path would produce a fake "no such
        # topic" error.
        if act == "brief_batch":
            return self._knowledge_brief_batch(
                topics, batch=batch, per_topic=per_topic, auto_built=auto)
        if act == "summarize_batch":
            return self._knowledge_summarize_batch(items)
        # "Redraw" is also a layer-level operation, likewise without a single
        # topic_id. The only difference between rebuild and refresh is force:
        # force=True bumps the whole layer's epoch +1, and layer merge
        # overwrites by whole generation ⇒ old clusters are **really** cleared
        # (a key union cannot do that; see the _merge_topic_layer comment).
        # Use rebuild only when changing parameters or when the old structure
        # has grown crooked.
        if act in ("rebuild", "refresh"):
            out = self.refresh_knowledge(force=(act == "rebuild"))
            if not out.get("enabled"):
                return {"ok": False, "enabled": False,
                        "action": act, "error": out.get("note")}
            self._persist_knowledge()
            out.update({"ok": True, "action": act, "saved": True,
                        "note": ("已重画簇边界并落盘；旧宿主归纳按成员重叠迁移"
                                 "（标 ⚠️ 待复核），迁移不了的在 lost_host 里"
                                 "自曝——不会静默消失"
                                 if act == "rebuild" else
                                 "已增量刷新并落盘（簇边界不变，不动宿主归纳）")})
            return out

        tid = str(topic_id or "").strip()
        t = self.knowledge.get(tid)
        if t is None:
            return {"ok": False, "enabled": True, "action": act,
                    "error": "没有这个主题：" + (str(topic_id) or "(空)"),
                    "hint": "先调 knowledge_map 列一下 id",
                    "sample_ids": [x.get("id") for x in
                                   topic_mod.ordered(self.knowledge)[:5]],
                    "total": len(self.knowledge)}
        # Alive members = the active superset (including those without vectors
        # for now); material uses only those with vectors (ranking needs
        # cosine). The two are kept separate: mixing them would make the
        # "member went stale" judgment jitter along with vector backfill.
        by_id = {str(it["id"]): it for it in self._knowledge_items()}
        info = {str(it["id"]): it for it in self._knowledge_items(require_vec=False)}
        alive = [m for m in (t.get("member_ids") or []) if m in info]
        no_vec = sum(1 for m in alive if m not in by_id)

        if act == "summarize":
            ok, err = topic_mod.apply_host_summary(
                t, summary, evidence, alive,
                now=self.manager._reference_time or _now(), fras=fras)
            if not ok:
                return {"ok": False, "enabled": True, "action": act,
                        "topic_id": tid, "error": err}
            self._persist_knowledge()
            return {"ok": True, "enabled": True, "action": act,
                    "topic_id": tid, "written": True, "saved": True,
                    "summary": topic_mod.summary_state(t, alive),
                    "note": ("已写回并落盘。host 归纳不会被 refresh / 层合并"
                             "覆盖；要改就再写一次，要退回抽取式就 action=revert")}

        if act == "revert":
            ok, err = topic_mod.revert_host_summary(t)
            if not ok:
                return {"ok": False, "enabled": True, "action": act,
                        "topic_id": tid, "error": err}
            self._persist_knowledge()
            return {"ok": True, "enabled": True, "action": act,
                    "topic_id": tid, "reverted": True, "saved": True,
                    "summary": topic_mod.summary_state(t, alive),
                    "note": "已撤销并落盘；下次 refresh 会填回抽取式代表句"}

        # ---- brief: produce material ----
        n = _topic_material.clamp_limit(limit)
        picked = topic_mod.representative_ids(t, by_id, n)
        members = _topic_material.build_members(
            by_id, picked, clip=topic_mod.TOPIC_EVIDENCE_CLIP)
        hint = _topic_material.brief_hint(
            has_alive=bool(alive), has_vec=bool(by_id),
            has_members=bool(members))
        return _topic_material.build_brief_payload(
            tid=tid, topic=t, alive_n=len(alive), members=members,
            # no_vector: members that are alive but have no vector yet — they
            # still belong to this topic (usable as evidence), they just
            # cannot enter the material ordering. Without surfacing it, the
            # host would think this topic contains only this much.
            no_vec=no_vec, auto=auto, hint=hint,
            summary=topic_mod.summary_state(t, alive))

    # ------------------------------------------------------------------
    # Batch: bring "tidying a library" from 400 round-trips down to ~40
    # ------------------------------------------------------------------
    def _persist_knowledge(self) -> None:
        """Persist the topic layer to disk (with the F14 ledger).

        Why it must be done explicitly: ``refresh_knowledge`` and write-backs
        only touch memory; persisting must be called by hand. Without this
        step, whatever the host painstakingly synthesized is gone the moment
        the daemon exits — **without any error** (it stays visible in memory
        until the restart reveals the loss).
        """
        try:
            self._save_layers()
        except Exception as exc:  # noqa: BLE001 — a persist failure must not swallow the synthesis itself
            logger.warning("[knowledge] 主题层落盘失败：%s", exc)

    def _knowledge_brief_batch(self, topic_ids: list[str] | None,
                               batch: int = 10, per_topic: int = 5,
                               auto_built: bool = False) -> dict[str, Any]:
        """Fetch synthesis material for N topics in one call (``action="brief_batch"``).

        Empty ``topic_ids`` ⇒ **auto-pick the ones most needing synthesis**
        (same convention as ``knowledge_map``'s ``pending``: not archived /
        ≥3 members / no synthesis yet or already stale). The host need not
        compute which ones to do, nor redo ones already written.
        """
        # The four blocks' bodies are in domain/knowledge_batch.py; the topic
        # package's four callbacks + one constant + defaults are fetched at
        # call time and injected (domain same layer, cross-references
        # forbidden). The order of the two _knowledge_items calls (by_id /
        # info) matches the original implementation (with vectors first, then
        # the alive superset). Note this method **does not check
        # layer_enabled** — the gate is at the top of knowledge_topic;
        # behavior stays as before.
        return _knowledge_batch.collect(
            topic_ids,
            batch=batch,
            per_topic=per_topic,
            auto_built=auto_built,
            knowledge=self.knowledge,
            by_id={str(it["id"]): it for it in self._knowledge_items()},
            info={str(it["id"]): it
                  for it in self._knowledge_items(require_vec=False)},
            defaults=_topic_defaults(),
            ordered=topic_mod.ordered,
            pending_summary_ids=topic_mod.pending_summary_ids,
            representative_ids=topic_mod.representative_ids,
            summary_state=topic_mod.summary_state,
            clip=topic_mod.TOPIC_EVIDENCE_CLIP,
        )

    def _knowledge_summarize_batch(
            self, items: list[dict[str, Any]] | None) -> dict[str, Any]:
        """Write back N syntheses in one call (``action="summarize_batch"``).

        **Per-item independence**: an item failing validation returns only
        that item, without affecting others in the batch — otherwise one
        mistyped id in a slip of the hand would lose dozens of syntheses at
        once and the host would have to synthesize everything again.
        """
        # The five blocks' bodies are in domain/summarize_batch_flow.py; the
        # topic package's two callbacks (apply_host_summary /
        # pending_summary_ids) + defaults are fetched at call time and
        # injected (domain same layer, cross-references forbidden). Call-count
        # semantics fully preserved: the alive superset is fetched per item
        # (one require_vec=False call per entry that passes the topic lookup),
        # pending exactly twice (once for pending_count and once for note),
        # persist only when written is non-empty.
        return _summarize_batch_flow.collect(
            items,
            knowledge=self.knowledge,
            items_of=self._knowledge_items,
            apply_summary=topic_mod.apply_host_summary,
            now=self.manager._reference_time or _now(),
            defaults=_topic_defaults(),
            pending_summary_ids=topic_mod.pending_summary_ids,
            persist=self._persist_knowledge,
        )

    def stats(self, daily_days: int = 0) -> dict[str, Any]:
        # Lexicon hot-awareness: after the host adds entity_lexicon.txt, no
        # daemon restart is needed (see :meth:`_reload_lexicon_if_changed`).
        # Placed in stats() because it is the self-check entry every host must
        # call, and it **adds no MCP tool** (the tool surface is injected into
        # the host context per session via tools/list; one more tool is a
        # permanent token cost, and changing the tool surface means syncing
        # the hardcoded tool counts in several places across this repo — grep
        # for the `== 42` style assertions).
        self._reload_lexicon_if_changed()
        # State distribution + L0↔L1 dangling references + daily activity
        # window (body in domain/stats_projection.py; records passed through
        # duck-typed).
        states = _stats_projection.state_counts(
            self.manager.store.iter_records())
        model_stats = self.manager.store.model_stats()
        missing = len(self.manager.store.missing_ids())
        # L0↔L1 referential integrity **self-report** (zcode appendix five):
        # L1's source_ids pointing at nothing used to be discoverable only by
        # digging through files. Computed on every stats run here (O(n)) so
        # the drift is visible on the spot — a non-0 tells the host to check
        # the save chain.
        raw_ids = {r.get("id") for r in self.raw_log}
        dangling, dangling_recs = _stats_projection.dangling_report(
            self.manager.store.iter_records(), raw_ids)
        # Tag-coverage self-report (the acceptance criterion): "how many
        # entries carry domain tags" used to require exporting the whole
        # library and counting by hand. Without this number, the host would
        # only go by feel — "seems like most have them" — the real library
        # measured 41.8%, over half with no tags at all.
        tag_cov = self.tag_coverage()
        # Daily activity (appendix five): the parallel-phase
        # "should-have-recall rate" numerator/denominator previously required
        # digging through the recall_count total; which days saw use and
        # whether usage was balanced was invisible.
        n_days = self._resolve_daily_days(daily_days)
        daily_block = _stats_projection.daily_activity(
            self.daily_counts, n_days)
        # writers must be the truth of **this moment**, not a snapshot from
        # the last persist. This number answers "how many hands are on this
        # library right now" — answering with a stale value is worse than not
        # answering: the user would take themselves for the sole writer.
        # _touch_writer() also refreshes this process's own heartbeat: the
        # very act of coming to stats proves this process is alive.
        try:
            self._touch_writer()
        except Exception:  # noqa: BLE001 — a self-report failure must never drag stats down with it
            pass
        # The return-body assembly (45 keys in fixed order + per-key rationale
        # comments) is in domain/stats_report.py: the shell only computes
        # values; the kwargs evaluation order is aligned key-by-key with the
        # old dict literal; the six projections stay in the shell, each
        # exactly once. test_f6 F3 anchors on the source containing the
        # "embedder_shared" literal, so that key's computation stays in the
        # shell.
        return _stats_report.assemble_report(
            l0_raw=len(self.raw_log),
            l1_memories=self.manager.store.count(),
            tags=tag_cov,
            l0_dangling_source_ids=len(dangling),
            l0_dangling_memories=dangling_recs,
            daily_activity=daily_block,
            l1_by_state=states,
            recall_feedback=self._recall_feedback_stats(),
            pin_stats=self._pin_stats(),
            l2_summaries=len(self.summaries),
            topic_layer=self.knowledge_stats(),
            l3_profile_slots=len(self.profile),
            entities=len(self.entities),
            sessions=len(self.sessions),
            open_intents=sum(1 for i in self.intents if i["status"] == "open"),
            layer_merge_last=dict(self.last_layer_merge),
            layer_del_enabled=load_config().layer_del,
            layer_del_tombstones_n=len(self._layer_del),
            layer_del_blocked_last=self._layer_del_blocked,
            layer_del_gc_last=self._layer_del_gc_last,
            source_sig_enabled=load_config().source_sig,
            source_sig_block=_stats_projection.source_sig_block(
                self.source_sigs, self.manager.store.iter_records()),
            writers=_stats_projection.writers_block(
                self.writers_active, self.writers_stale_cleaned,
                os.getpid()),
            open_topics=len(self.open_topics),
            pinned=sum(1 for r in self.manager.store.iter_records() if r.pin),
            tombstones=len(self.tombstones),
            raw_forgotten=_stats_projection.forgotten_count(self.raw_log),
            embed_model=self.manager.model_key,
            embedder_backend=self.manager.embedder_backend,
            embedder_runtime=self.manager.embedder_runtime_backend(),
            onnx_attached=(
                self.manager.embedder_runtime_backend() == "onnx"),
            lnn_runtime=getattr(self.manager.memory_core, "runtime", None),
            pending_consolidate=self._pending_consolidate,
            embedding_dim=getattr(self.manager, "embedding_dim", None),
            embedder_model_source=(
                getattr(self.manager.embedder, "model_source", None)
                or "n/a（该后端不经模型名解析）"),
            # F6 self-report (the key-name anchor; test_f6 F3 greps the source
            # for "embedder_shared"): whether this library's embedder reuses
            # the process-wide shared heavy object.
            embedder_shared=bool(
                getattr(self.manager.embedder, "shared_from_pool", False)),
            embedder_pool=embedder_pool_stats(),
            ner_backend=self.ner_backend_lazy,
            lexical=self.lexical_self_report(),
            model_stats=model_stats,
            vectors_missing=missing,
            supersede_enabled=self.supersede_enabled,
            entity_lexicon_size=len(self.entity_lexicon),
            entity_aliases=len(self.entity_lexicon_aliases),
            tag_schema=self.tag_extractor.describe(),
            entity_origins=self.entity_origin_stats(),
            sources=len([s for s in self.manager.sources() if s]),
            save_ledger=self.save_ledger_view(),
            l0_pending=self._l0_pending,
        )

    # ------------------------------------------------------------------
    # Project overview and status cards
    # Stance: two new slots of the session_start pre-activation package — the
    # compact projection of "what this library is overall and what shape it
    # is in right now" that the agent consults at every start of work. Pure
    # derivation, no retrieval, no clustering (hot-path discipline: the 2-3s
    # first knowledge tidying never enters every session opening).
    # Only signals that "cannot be derived but would hurt if wrong" go in.
    # ------------------------------------------------------------------

    def contradictions_count(self, limit: int = STATUS_CONTRADICTION_CAP) -> dict[str, Any]:
        """Contradiction count for the status card (with a budget cap; the same signal shape as Stale).

        A whole-library pair scan takes ~0.34s on a 10k-entry library
        (measured via vector_store.pairwise_similarities); the hot path
        (every session opening) must not pay the full cost, hence the cap: at
        limit, ``capped=True`` and the card reports「≥N」instead of a false
        precision. ``limit<=0`` = disabled.

        Returns:
            ``{count, capped, limit}``. count is the actual pair count after
            truncation; when capped=True the true value is ≥ limit.
        """
        if limit <= 0:
            return {"count": 0, "capped": False, "limit": 0}
        pairs = self._find_contradictions(limit=limit)
        return {
            "count": len(pairs),
            "capped": len(pairs) >= limit,
            "limit": limit,
        }

    def project_overview(self) -> dict[str, Any]:
        """Library-level overview projection (design doc §3.1). Pure derivation, no retrieval, single O(n) pass.

        Five pieces of orientation: identity + scale / time span / structural
        profile / hot entities. All come from existing derived layers
        (tag_coverage / knowledge_stats / entities); the only newly derived
        quantity is the min/max of record timestamps. For the conventions see
        :mod:`..domain.project_view`.
        """
        return _project_overview(
            records=list(self.manager.store.iter_records()),
            entities=self.entities,
            knowledge=self.knowledge,
            daily_counts=self.daily_counts,
            ref_time=self.manager._reference_time,
            # display_name lives in library/ (arch G1: domain may depend on common only)
            name=display_name(self.persist_dir),
            topic_layer_enabled=_topic_defaults().layer_enabled,
            # Lazy callbacks: never called when the switch is off
            tag_coverage=self.tag_coverage,
            knowledge_stats=self.knowledge_stats,
            recent_days=OVERVIEW_RECENT_DAYS,
            top_entities=OVERVIEW_TOP_ENTITIES,
            top_tag_keys=OVERVIEW_TOP_TAG_KEYS,
        )

    def project_status(self) -> dict[str, Any]:
        """Current-status projection (design doc §3.2). Pure derivation, no retrieval.

        Four signal groups: pending (open topics / open intents /
        consolidation) / health (L0 dangling / contradictions / miss gap) /
        collaboration (active sessions) / activity (sparkline series). The
        contradiction count carries a budget cap and can be disabled via env
        (~0.34s on a 10k-entry library; see contradictions_count).
        """
        return _project_status(
            records=list(self.manager.store.iter_records()),
            raw_log=self.raw_log,
            open_topics=self.open_topics,
            intents=self.intents,
            miss_log=self.miss_log,
            daily_counts=self.daily_counts,
            ref_time=self.manager._reference_time,
            pending_consolidate=self._pending_consolidate,
            contradictions_enabled=load_config().status_contradictions,
            # Lazy callbacks: never called when the switch is off
            contradictions_count=self.contradictions_count,
            active_sessions=self.active_sessions,
            file_registry_summary=self.file_registry_summary,
            contradiction_cap=STATUS_CONTRADICTION_CAP,
            miss_days=STATUS_MISS_DAYS,
            spark_days=STATUS_SPARKLINE_DAYS,
        )

    # ==================================================================
    # Map index rows + file registry
    # maps.json sidecar maps merged into the index + full text of task-relevant L1 maps
    # ==================================================================

    # ── Gate one · legacy-side guard ─────────────────────────────────

    def find_map_entry_record(self, name: str, key: str):
        """Find the **alive** entry memory by the (map, entry) reserved tags; None when absent.

        Entries archived (by consolidate's dormant cleanup) do not count —
        dead entries must not block rebuilding. O(entries within the
        registry), called only on the write path carrying map tags (see
        _guard_map_entry_dup).
        """
        name, key = str(name or "").strip(), str(key or "").strip()
        if not name or not key:
            return None
        for rec in self.manager.store.iter_records():
            if rec.state == MemoryState.ARCHIVED.value:
                continue
            tags = rec.tags or {}
            if (str(tags.get(MAP_NAME_KEY) or "").strip() == name
                    and str(tags.get(MAP_ENTRY_KEY) or "").strip() == key):
                return rec
        return None

    def _guard_map_entry_dup(self, tags: dict[str, Any] | None) -> None:
        """ingest-side same-key duplicate check (gate one). Misuse raises ValueError — ingest_batch
        quarantines it into errors without aborting the batch (same philosophy
        as map/store's guardrails).
        """
        if not tags:
            return
        name = str(tags.get(MAP_NAME_KEY) or "").strip()
        key = str(tags.get(MAP_ENTRY_KEY) or "").strip()
        if not name or not key:
            return
        dup = self.find_map_entry_record(name, key)
        if dup is None:
            return
        cur = str((dup.tags or {}).get(MAP_STATUS_KEY) or "（无状态）")
        raise ValueError(
            "地图条目「%s / %s」已存在（记忆 %s，当前状态：%s）——地图是"
            "「现在是什么」的按键状态，同键只能有一条。改状态请走 "
            "update_tags（legacy）或 map_set（registry 侧车，推荐），"
            "不要用 ingest 写第二条" % (name, key, dup.id[:8], cur))

    def map_index(self) -> dict[str, Any]:
        """Map index (L0 delivery): which maps exist in the library, how many entries each has, how long since updated.

        Two sources merged:
        - **legacy**: memories carrying the reserved tag keys (``{"地图": "合约", "条目":
          "CN2601", "状态": "验证集"}``), one entry per key;
        - **registry**: the ``maps.json`` sidecar (the formal map layer
          written by map_set).

        Same-named maps are added (entry counts summed, newest taken, source
        marked ``both``) — old data is not lost, new data goes the new way.
        Reports only name + counts + freshness + drift count + conflict
        count, no body text (zero body text is the balance point between
        "necessary information" and "must not be too long"); the body arrives
        via L1 hits into the package / L2 ``map_get`` exact queries. Pure
        O(n) reads of existing layers, no retrieval, no clustering.

        Design-review tightening:
        - **Archived entries stay out of the index** (consolidate archives
          dormant map-entry memories; still counting them would inflate index
          rows, and archived entries are unreachable by recall);
        - **Dual-source same-key conflict detection**: the same (map, entry
          key) with inconsistent state between legacy memories and the
          registry sidecar ⇒ ``conflicts`` count + an index-row「冲突N」.
          Summing only solved count merging, not "two states competing" —
          that is exactly the core veto proposition of design §2, and before
          the fix it could live silently inside both maps.
        """
        return _map_index_fn(
            self.manager.store.iter_records(),
            map_mod.load(self.persist_dir),
            list_maps=map_mod.list_maps,
            now=_now,
            archived_state=MemoryState.ARCHIVED.value,
            limits=_MapIndexLimits(
                max_maps=MAP_INDEX_MAX, stale_days=MAP_STALE_DAYS,
                name_key=MAP_NAME_KEY, entry_key=MAP_ENTRY_KEY,
                status_key=MAP_STATUS_KEY,
            ),
        )

    # ==================================================================
    # Map layer (maps.json sidecar + full text of task-relevant L1 maps)
    # Design doc §5-L1 / §6: task tokens/anchor entities ∩ map desc/entry
    # names ⇒ that map's full text enters the opening package (cap 20 rows
    # per map + an "N more entries" self-report); no hit, no delivery
    # — the counterexample constraint is guaranteed naturally by matching; no
    # manual "who sees which map" configuration is needed.
    # ==================================================================
    def map_detail(self, task: str,
                   tp: dict[str, Any] | None) -> dict[str, Any]:
        """Full-text projection of task-relevant maps (L1 delivery). Reads the sidecar only; no retrieval, no model calls.

        token = the task text + anchor entity names in tp (the same
        :func:`task_tokens` as the phase-two task_profile; **no
        re-retrieval**). Hit criterion (§6): a map's desc contains any token,
        or any of its entry keys contains a token.
        """
        return _map_detail_fn(
            task, tp,
            map_mod.load(self.persist_dir),
            list_maps=map_mod.list_maps,
            task_tokens_fn=task_tokens,
            limits=_MapDetailLimits(
                max_maps=MAP_DETAIL_MAX_MAPS, max_rows=MAP_DETAIL_MAX_ROWS,
            ),
        )

    # ── Map tool backends (server.py's four map_* tools are thin wrappers, same precedent as drift_audit) ──

    def map_source_root(self) -> str | None:
        """Resolution root for truth_source relative paths: the file registry's project root (opt-in).

        Returns None when the registry is not enabled — resolve_source falls
        back to the process cwd.
        """
        reg = fr_mod.load(self.persist_dir)
        return (reg or {}).get("root") or None

    def map_set(self, name: str, key: str, status: str = "",
                fields: dict | None = None, desc: str = "",
                truth_source: str = "", note: str = "",
                updated_by: str = "") -> dict[str, Any]:
        """Gate one's write entrance: upsert by key (a same-key rewrite replaces wholesale; old state does not compete).

        Implemented in :func:`hippocampus_mcp.domain.map_ops.map_set`. This
        method only does binding: the map package's capabilities are injected
        via construction (``store``/``domain`` must not import ``map``); the
        busy/unreadable exception classes are passed by value.
        """
        return _map_ops.map_set(
            name, key, status=status, fields=fields, desc=desc,
            truth_source=truth_source, note=note, updated_by=updated_by,
            persist_dir=self.persist_dir,
            set_entry=map_mod.set_entry, update_fn=map_mod.update,
            root_fn=self.map_source_root, busy_exc=map_mod.SidecarBusy,
            unreadable_exc=SidecarUnreadable, wait=MAP_WRITE_WAIT_S)

    def map_get(self, name: str, key: str) -> dict[str, Any]:
        """L2 exact query: a single entry. **Misses are reported explicitly** (no silent empties)."""
        data = map_mod.load(self.persist_dir)
        if data is None:
            return {"found": False, "reason": "no_maps", "map": name,
                    "key": key,
                    "hint": "本地图不存在（未启用地图层）——先 map_set 建图"}
        ent = map_mod.get_entry(data, name, key)
        if ent is not None:
            return {"found": True, "map": name, "key": key, "entry": ent}
        m = (data.get("maps") or {}).get(str(name))
        if m is None:
            return {"found": False, "reason": "no_map", "map": name,
                    "key": key,
                    "hint": "没有这张图；现有图：%s"
                            % ("、".join(map_mod.map_names(data)) or "（空）")}
        return {"found": False, "reason": "no_key", "map": name, "key": key,
                "hint": "该图有 %d 个条目；要全列表走 map_list(name=...)"
                        % len(m.get("entries") or {})}

    def map_list(self, name: str = "") -> dict[str, Any]:
        """List maps (name empty) or list one map's entries (with cap + self-report).

        Implemented in :func:`hippocampus_mcp.domain.map_ops.map_list`. This
        method only does binding: ``load`` is fetched at call time and its
        data passed in; for the quirk of the entry-row 100-char truncation
        being separate from the map-count cap see the new module's docstring.
        """
        return _map_ops.map_list(
            map_mod.load(self.persist_dir), name,
            list_maps_fn=map_mod.list_maps, map_names_fn=map_mod.map_names,
            max_maps=MAP_INDEX_MAX)

    def map_verify(self, name: str = "") -> dict[str, Any]:
        """Gate two: re-read truth_source and compare signatures (hash arbitration; a touch does not false-positive).

        Implemented in :func:`hippocampus_mcp.domain.map_ops.map_verify`.
        This method only does binding: the map package's capabilities are
        injected via construction; drift is written inside the lock, and the
        busy/unreadable echo shape is deliberately different from map_set's
        (see the new module's docstring).
        """
        return _map_ops.map_verify(
            name, persist_dir=self.persist_dir,
            verify_fn=map_mod.verify, update_fn=map_mod.update,
            list_maps_fn=map_mod.list_maps,
            root_fn=self.map_source_root, busy_exc=map_mod.SidecarBusy,
            unreadable_exc=SidecarUnreadable, wait=MAP_WRITE_WAIT_S)

    def map_auto_verify(self, force: bool = False) -> dict[str, Any]:
        """Auto verify on signals — value-level validation / directory reconciliation.

        Implemented in :func:`hippocampus_mcp.domain.map_drift.auto_verify`;
        this method only does binding: the map package's capabilities are
        injected via construction; the three Config knobs are fetched at call
        time and passed by value; for the interval gating / budget /
        best-effort persist runtime discipline see the new module's docstring.
        """
        return _map_drift.auto_verify(
            force, persist_dir=self.persist_dir, now_fn=_now,
            enabled=load_config().map_auto_verify,
            interval_h=load_config().map_auto_interval_h,
            budget_s=load_config().map_auto_budget_s,
            registry_load=fr_mod.load, auto_run=map_auto_mod.run,
            auto_needs_registry=map_auto_mod.needs_registry,
            update_fn=map_mod.update, busy_exc=SidecarBusy)

    def map_mention_drift(self, force: bool = False) -> dict[str, Any]:
        """Gate three: after an update, being mentioned as「another state」⇒ the entry is suspected stale.

        Implemented in :func:`hippocampus_mcp.domain.map_drift.mention_drift`;
        this method only does binding: the map package's capabilities are
        injected via construction; the record snapshot goes through the
        ``records_fn`` lazy callback (materialized only past the gate); for
        the triple criterion and the weak-signal-never-overrides-strong-facts
        discipline see the new module's docstring.
        """
        return _map_drift.mention_drift(
            force, persist_dir=self.persist_dir, now_fn=_now,
            enabled=load_config().map_mention,
            interval_h=load_config().map_mention_interval_h,
            budget_s=load_config().map_mention_budget_s,
            max_records=load_config().map_mention_max_records,
            needs_run_fn=map_mention_mod.needs_run,
            run_fn=map_mention_mod.run,
            records_fn=lambda: list(self.manager.store.iter_records()),
            update_fn=map_mod.update, busy_exc=SidecarBusy)

    def file_registry_summary(self) -> dict[str, Any] | None:
        """File registry summary. **Returns None when not enabled (no sidecar)** — opt-in.

        When enabled and the interval since the last scan has elapsed (default
        6h, env-adjustable), rescan once under budget: session_start is the
        hot path of every session opening, and even a full stat cannot be
        unbounded — beyond it, scan_complete=false self-reports「scan
        incomplete」. A scan failure does not blow up (the previous summary is
        returned, last_checked does not advance — the next opening retries).

        Design-review tightening: after scanning, reconcile against the
        **source signature layer** (:meth:`_fr_reconcile_source_sigs`) — for
        re-ingested files, ``source_sigs`` holds a signature of the current
        content version, so Hippo already reflects the disk content and the
        registry must not keep reporting stale. Otherwise one file would get
        two contradictory answers: drift_audit (recorded the new signature ⇒
        self-healed) vs file_registry (no manual sync ⇒ flagged red), and the
        manual sync step is the easiest to forget.
        """
        if load_config().file_registry_off:
            return None
        now = _now()
        interval_h = load_config().file_registry_interval_h
        limits = fr_mod.scan_limits()
        # Orchestration in domain/file_registry_flow.py:
        # fr_mod's four entries are fetched at call time and injected
        # (patching core.fr_mod takes effect immediately); the reconcile
        # callback is a bare reference (reads self.source_sigs, brain state).
        return _fr_flow.summary_flow(
            persist_dir=self.persist_dir,
            now=now, interval_h=interval_h, limits=limits,
            load=fr_mod.load, scan=fr_mod.scan, update=fr_mod.update,
            summarize=fr_mod.summarize,
            reconcile=self._fr_reconcile_source_sigs,
        )

    def _fr_reconcile_source_sigs(self, reg: dict[str, Any]) -> int:
        """Registry scan × source-signature-layer reconciliation: hippo has ingested the current content version ⇒ clear stale.

        The criterion is (mtime_ns, size) double equality — the same threat
        model as the registry layer (adversarial wall-clock manipulation was
        never in the defense scope; see file_registry docstring discipline 2).
        Returns the number of entries cleared (0 = nothing to clear, the
        norm).
        """
        root = reg.get("root") or ""
        if not root or not self.source_sigs:
            return 0
        # The index construction is a pure function of the fr module (the
        # CLI's verified sync uses it too) — both sides share one convention,
        # so a change to normalization/take-newest in one place does not
        # slowly drift the other into a second set of facts.
        sig_by_path = fr_mod.source_sig_index(self.source_sigs)
        # The stale-clearing loop is in domain/file_registry_flow.py (D-3):
        # path normalization is injected via a callback (os stays out of
        # domain); last_synced is fetched per entry.
        return _fr_flow.reconcile_stale_entries(
            reg=reg, sig_by_path=sig_by_path,
            path_of=lambda rel: os.path.normcase(
                os.path.join(root, str(rel))),
            now=_now,
        )

    # ==================================================================
    # Task-level "problem profile"
    # ==================================================================
    def task_profile(self, task: str,
                     task_hits: list[dict[str, Any]]) -> dict[str, Any]:
        """Structural projection of what the task touches (design doc §6). Pure derivation, no retrieval, no model calls.

        The anchor is ``task_hits``, **already computed** by session_start
        (the pre-recall ran once; this never re-retrieves — the same hot-path
        discipline as the overview card). For the four blocks' conventions and
        the empty-task "empty block ⟺ structurally empty" dual-track
        discipline see :mod:`..domain.task_profile`.
        """
        return _task_profile(
            task, task_hits,
            knowledge=self.knowledge,
            entities=self.entities,
            intents=self.intents,
            miss_log=self.miss_log,
            ref_time=self.manager._reference_time,
            # Switch and five-tier limits always fetched at call time (the in-package constants table is frozen in core)
            topic_layer_enabled=_topic_defaults().layer_enabled,
            limits=_TaskProfileLimits(
                topic_max=TASK_TOPIC_MAX, entity_max=TASK_ENTITY_MAX,
                intent_max=TASK_INTENT_MAX, miss_max=TASK_MISS_MAX,
                miss_days=TASK_MISS_DAYS,
            ),
            # Topic conventions injected (arch G1: domain may depend on common only, not even its own layer)
            ordered_topics=topic_mod.ordered,
            archived_status=topic_mod.STATUS_ARCHIVED,
            now=_now,
        )

    def _read_raw_log_from_disk(self) -> list[dict]:
        """Read the L0 stream from disk (for the pre-write union). When unreadable, treat as empty — never block saving."""
        if not self.persist_dir:
            return []
        # G1: go through the backend — under sqlite, L0 lives in the layers
        # table with no separate file to read; a direct open would silently
        # return empty and a multi-writer merge would wipe someone else's raw
        # (the same class of incident).
        data = self._read_layer_from_disk("brain_raw.json")
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Derived layers' "read-merge-write"
    # ------------------------------------------------------------------

    # Layer file -> memory attribute
    _LAYER_ATTRS: dict[str, str] = {
        "brain_raw.json": "raw_log",
        "brain_entities.json": "entities",
        "brain_summaries.json": "summaries",
        "brain_profile.json": "profile",
        "brain_sessions.json": "sessions",
        "brain_intents.json": "intents",
        "brain_topics.json": "open_topics",
        "brain_tombstones.json": "tombstones",
        "brain_misslog.json": "miss_log",
        "brain_counters.json": "daily_counts",
        "brain_knowledge.json": "knowledge",
        # The deletion ledger. It is **not** an 11th business dataset; it is
        # the tombstone set of "which keys were deleted", read by other layers
        # during merge to block revivals.
        LAYER_DEL_FILE: "_layer_del",
        # The source-file signature layer (drift_audit's source of truth).
        SOURCE_SIG_FILE: "source_sigs",
    }

    # Layer file -> merge convention
    _LAYER_MERGE_RULES: dict[str, str] = {
        "brain_raw.json": "raw",
        "brain_summaries.json": "list",
        # Open intents cannot use the generic list rule: it would let an old
        # snapshot undo a completed status; see :meth:`_merge_intent_layer`.
        "brain_intents.json": "intent",
        # Open topics cannot use the generic list either: unbounded
        # append-only would silently bloat (same reason as miss_log); use
        # list_cap to trim the oldest, capped at OPEN_TOPICS_MAX.
        "brain_topics.json": "list_cap",
        "brain_tombstones.json": "list",
        "brain_misslog.json": "list_cap",
        "brain_entities.json": "entity",
        "brain_profile.json": "dict",
        "brain_sessions.json": "session",
        "brain_counters.json": "counter",
        "brain_knowledge.json": "topic",
        # The deletion ledger itself uses "take max per key": both sides are
        # deletions, and taking the later one is the only way not to weaken
        # the other side's deletion (the dict rule's "local wins" is wrong
        # here).
        LAYER_DEL_FILE: "del_ledger",
        # C1a: for the signature layer, take the newer sig_at per key — when
        # two processes both ingest the same file, the later-verified one
        # reflects the disk's current state; with the same sha1 the facts do
        # not differ and either will do.
        SOURCE_SIG_FILE: "sourcesig",
    }

    # The **forward order** of open-intent statuses: may only move right,
    # never back. In a multi-process shared library, if "whoever persisted
    # last decides" still applied, a process that just marked an intent done
    # could have it undone by one save from another process holding only a
    # stale snapshot — with no error on either side.
    _INTENT_STATUS_RANK: dict[str, int] = {"open": 0, "done": 1}








    def _read_layer_from_disk(self, name: str):
        """Read one layer from disk (unreadable / corrupt -> None, treated as "no such layer on disk").

        G1: disk reads go through :attr:`_backend` (json files / sqlite's
        layers table).
        """
        if not self.persist_dir:
            return None
        backend = self._backend or get_backend(self.persist_dir)
        return backend.read_layer(name)


    def note_layer_deletion(self, layer: str, keys) -> int:
        """Record "these keys were deleted from the **layer**" so later merges can block late-arriving old versions.

        Returns:
            How many were newly recorded (0 = the ledger is off / no valid keys given).
        """
        if not load_config().layer_del or not keys:
            return 0
        now = _now()
        n = 0
        for k in keys:
            if not k:
                continue
            self._layer_del[self._del_ledger_key(layer, str(k))] = now
            n += 1
        return n

    def note_layer_deletion_items(self, layer: str, items) -> int:
        """list-layer specific: leave tombstones for a batch of deleted entries via :meth:`_layer_key`."""
        if not load_config().layer_del:
            return 0
        keys = [self._layer_key(it) for it in (items or []) if isinstance(it, dict)]
        return self.note_layer_deletion(layer, keys)



    def _apply_layer_deletions(self, name: str, merged):
        """After merging: block keys the ledger has marked as deleted.

        Implemented in
        :func:`hippocampus_mcp.store.layer_gc.apply_layer_deletions`;
        this method only does binding + count write-back
        (``_layer_del_blocked`` is accumulated in the shell); for the pass
        conditions and the two interception shapes see the new module's
        docstring.
        """
        merged, blocked = _layer_gc.apply_layer_deletions(
            name, merged, enabled=load_config().layer_del,
            layer_del=self._layer_del, layer_key_fn=self._layer_key,
            entry_ts_fn=self._entry_ts)
        self._layer_del_blocked += blocked
        return merged

    def _gc_layer_del(self) -> int:
        """Ledger slimming: drop over-age entries; still over quota, drop the oldest. Returns how many were dropped.

        Implemented in :func:`hippocampus_mcp.store.layer_gc.gc_layer_del`;
        this method only does binding + write-back (the module returns the new
        ledger and the shell rebinds ``self._layer_del`` — the same rebind
        semantics as the original implementation).
        """
        new_ledger, dropped = _layer_gc.gc_layer_del(
            self._layer_del, now_fn=_now, ttl=LAYER_DEL_TTL,
            max_keys=LAYER_DEL_MAX_KEYS)
        self._layer_del = new_ledger
        return dropped

    def _merge_layer(self, name: str, disk, mine):
        """Merge per the layer's convention (disk as base + local additions), then intercept via the deletion ledger.

        Implemented in :func:`hippocampus_mcp.store.layer_gc.merge_layer`;
        this method only does binding: the rules table and the ten-way merge
        callbacks are fetched at call time and injected (helpers in
        brain/pure.py).
        """
        return _layer_gc.merge_layer(
            name, disk, mine, rules=self._LAYER_MERGE_RULES,
            fns={
                "del_ledger": self._merge_del_ledger_layer,
                "raw": self._merge_raw_log,
                "list": self._merge_list_layer,
                "list_cap": self._merge_list_layer,
                "entity": self._merge_entity_layer,
                "session": self._merge_session_layer,
                "intent": self._merge_intent_layer,
                "counter": self._merge_counter_layer,
                "topic": self._merge_topic_layer,
                "sourcesig": self._merge_sourcesig_layer,
                "dict": self._merge_dict_layer,
            },
            apply_del_fn=self._apply_layer_deletions,
            miss_log_max=MISS_LOG_MAX,
            # topics' per-layer cap (patching core's OPEN_TOPICS_MAX takes effect immediately).
            list_caps={"brain_topics.json": OPEN_TOPICS_MAX})



    def _disk_only_count(self, name: str, disk, mine) -> int:
        """Entries present on disk but missing from this process's memory (= how many this merge rescued)."""
        if isinstance(disk, dict) and isinstance(mine, dict):
            return len(set(disk) - set(mine))
        if isinstance(disk, list) and isinstance(mine, list):
            mk = {self._layer_key(x) for x in mine if isinstance(x, dict)}
            dk = {self._layer_key(x) for x in disk if isinstance(x, dict)}
            return len(dk - mk)
        return 0


    def _layer_changed_by_other(self, name: str, path: str) -> bool:
        """Whether this layer file was touched by **someone else** since this process's last persist.

        Untouched means no need to read and merge from disk — the in-memory
        copy is the newest. A single writer per library is the norm; without
        this check the perf case (N=1000) went from 30s to 192s: every save
        re-read and re-merged all 10 layer files, pure waste.

        First persist (no stamp) always returns True: this library may have
        been serving other processes for a long time.
        """
        if name not in self._layer_stamp:
            return True
        # G1: sqlite has no layer files; the stamp is supplied by the backend itself (updated_at + length)
        return self._layer_stamp.get(name) != (
            self._backend.layer_stamp(name) if self._backend
            else self._stamp_of(path)
        )

    def _reclip_derived_layers(self) -> None:
        """After the union, re-clip the derived layers against the **current truth**.

        Implemented in
        :func:`hippocampus_mcp.store.layer_gc.reclip_derived_layers`;
        this method only does binding: the three containers are injected into
        the body (emptied entities popped in place) + three brain capability
        callbacks fetched at call time; for the "rather under-clip than
        mis-clip" count gate see the new module's docstring.
        """
        _layer_gc.reclip_derived_layers(
            self.entities, self.profile, self.knowledge,
            count_fn=self.manager.store.count,
            recompute_entity_stats_fn=self._recompute_entity_stats,
            prune_fn=self.prune_stale_profile,
            reclip_knowledge_fn=self._reclip_knowledge)



    # ------------------------------------------------------------------
    # F14 save ledger: one save = begin + end rows; the exception path records fail
    # ------------------------------------------------------------------
    def _layer_count(self, attr: str) -> int:
        """The layer's current entry count (the count field of the ledger's begin/end rows)."""
        try:
            return len(getattr(self, attr))
        except TypeError:
            return -1

    def _ledger_path(self) -> str | None:
        """The ledger file path; None when there is no persist_dir (in-memory-only brain)."""
        return ledger_path(self.persist_dir)

    def _ledger_read_tail(self) -> tuple[dict[str, Any] | None, int, list]:
        """Read the ledger's last row + total line count + a few tail rows (for boot reconciliation).

        Bad lines are skipped line by line; a bad file is quarantined by
        convention as .corrupt-<ts>; **startup is never blocked** (design §8
        risk 9: a missing / corrupt ledger reports unknown and continues).
        """
        return read_tail(self.persist_dir)

    def _ledger_append(self, entry: dict[str, Any]) -> None:
        """Append one row. **Append-only is a hard requirement**: a full rewrite would turn the ledger itself into a new data-loss point.

        The last row and line count update only on a successful persist (on
        write failure, "what it was before" must not point at a record that
        never made it to disk); trimming at trim_at happens on the store side;
        ring / trim_at are fetched from the core namespace **at call time** —
        tests monkeypatch ``core.LEDGER_RING``, and snapshotting would break
        L5.
        """
        ok, n = ledger_append(
            self.persist_dir, entry,
            self._ledger_lines, LEDGER_TRIM_AT, LEDGER_RING,
        )
        if ok:
            self._ledger_last = entry
            self._ledger_lines = n

    def _ledger_trim(self) -> None:
        """Ring buffer: keep only the most recent LEDGER_RING rows, bounding the ledger's size.

        When the store side returns None (no persist_dir / read failure /
        write failure), the line count keeps its previous value — a brief
        mismatch between the ledger's line count and the disk reality is
        allowed.
        """
        n = ledger_trim(self.persist_dir, LEDGER_RING)
        if n is not None:
            self._ledger_lines = n

    def _ledger_row(self, phase: str, kind: str, *, layers: dict | None = None,
                    layer_fps: dict | None = None,
                    error: str | None = None,
                    extra: dict[str, Any] | None = None) -> None:
        """V1-a: one ledger row for an autosave persist (outside _save_layers).

        Same row conventions as ``store.layer_save.run_layer_save``
        (seq / at_wall / at_mono / pid / layers / layer_fps), plus a ``kind``
        field telling which previously-unledgered path wrote it
        (``l0_autosave`` / ``store_autosave``). ``seq`` comes from a dedicated
        autosave counter — layer rows keep their historical contiguous seqs,
        and boot reconcile matches begin/fail pairs by ``(seq, kind)`` so the
        two spaces cannot cross. Appending stays best-effort like every ledger
        write: failures inside ``_ledger_append`` never break the persist
        itself.
        """
        self._autosave_seq += 1
        entry: dict[str, Any] = {
            "seq": self._autosave_seq,
            "phase": phase,
            "kind": kind,
            "at_wall": time.time(),
            "at_mono": time.monotonic(),
            "pid": os.getpid(),
        }
        if layers is not None:
            entry["layers"] = layers
        if layer_fps is not None:
            entry["layer_fps"] = layer_fps
        if error is not None:
            entry["error"] = error
        if extra:
            entry.update(extra)
        self._ledger_append(entry)

    def _store_autosave_ledger(self, phase: str, info: dict[str, Any]) -> None:
        """V1-a hook target for the manager's every-10-steps store autosave.

        The explicit save path is ledgered by ``_save_layers``; this closes
        the same visibility gap for the store-only autosave. Rows carry no
        layer fingerprints (the store is not a layer) — boot reconciliation
        reports them as ``layer_fps_absent`` and still flags an unpaired
        ``begin`` after a kill.
        """
        self._ledger_row(phase, "store_autosave", extra=dict(info))

    def save_ledger_view(self) -> dict[str, Any]:
        """F14: the ledger projection (``brain_stats.save_ledger``).

        The template copies ``recall_feedback``: number + level + plain-language
        note. Empty library / never persisted reports ``null`` + ``unknown``,
        **never 0** — 0 would be read as "just saved", a lie worse than
        unknown.
        """
        # Arguments given in the original implementation's evaluation order:
        # quadrant → l1_dirty → pending. The two ``time.monotonic()`` calls
        # (at_mono fallback / interval computation) go through the ``now``
        # callback, aligned call-for-call with the original (with a fixed
        # injected clock, interval_s is reproducible).
        return ledger_view(
            self._ledger_last, self._ledger_lines, self._boot_reconcile,
            self.classify_layers(),
            bool(
                getattr(getattr(self.manager, "store", None), "_dirty_ids", None)
                or getattr(getattr(self.manager, "store", None), "_deleted_ids", None)
            ),
            self._pending_layer_names(),
            now=time.monotonic,
            stamp_of=lambda ts: time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(ts)),
        )

    def _save_layers(self) -> None:
        """F14: open the books for this save; the actual disk write goes to :meth:`_save_layers_core`.

        Each of the three phases has its reason: ``begin`` must come **before**
        the write (a crash at least leaves "what we meant to write"), ``end``
        records the actual result, ``fail`` records the failure reason —
        without them, "meant to write but never did" would never have happened
        (design §8 risk 8). The orchestration body is in
        :mod:`hippocampus_mcp.store.layer_save`.
        """
        try:
            run_layer_save(LayerLedgerPorts(
                persist_dir=self.persist_dir,
                save_seq=self._save_seq,
                layer_attrs=self._LAYER_ATTRS,
                now_wall=lambda: self.manager._reference_time or time.time(),
                now_mono=time.monotonic,
                get_pid=os.getpid,
                compute_all_fps=self._compute_all_layer_fps,
                layer_count=self._layer_count,
                ledger_append=self._ledger_append,
                set_begin_fps=lambda fps: setattr(self, "_begin_layer_fps", fps),
                core_fn=self._save_layers_core,
            ))
        except BaseException:
            # B (verification feedback 2026-11-01): the failure-side breakdown
            # — when the disk write raises, every layer is marked failed
            # (partial writes are unknowable; report honestly and strictly),
            # paired with session_end's saved=false + save_error, so upper
            # layers no longer read only the aggregate boolean.
            self.last_save_layers = {n: "failed" for n in self._LAYER_ATTRS}
            raise


    def _pending_layer_names(self) -> list[str]:
        """F14 slice 2/3: the layer names that **have in-memory changes not yet written** right now.

        The "memory side" dimension of the four quadrants comes from here:
        compare the current in-memory fingerprint with the last-persisted
        fingerprint. With no persist record (new library), returns empty —
        not "no changes" but "cannot judge"; :meth:`save_ledger_view`'s
        ``pending_note`` makes that explicit.
        """
        return pending_layer_names(
            self._LAYER_ATTRS, self._layer_written_fp,
            lambda attr: self._layer_fingerprint(getattr(self, attr)),
        )

    def _compute_all_layer_fps(self) -> dict[str, str]:
        """All layers' content fingerprints (the ``layer_fps`` field of the ledger's begin/end rows).

        With the fingerprint switch off, returns an empty dict — the ledger
        rows carry no ``layer_fps``, and boot reconciliation degrades honestly
        on that basis (no guessing).
        """
        return compute_all_layer_fps(
            self._LAYER_ATTRS,
            lambda attr: self._layer_fingerprint(getattr(self, attr)),
            # The knob is fetched at call time: monkeypatching ``core.load_config`` still works
            load_config().layer_fp,
        )

    def classify_layers(self) -> dict[str, str]:
        """F14-3 four-quadrant classification (convention 2): each layer's current standing.

        Memory side (**primary criterion**) = current in-memory fingerprint vs
        this process's last-persisted fingerprint; disk side (**corroborating
        only**) = whether the layer file stamp is still there (mtime never
        convicts alone — when a touch does not change content, the fingerprint
        is unchanged and it still counts as clean, only recorded as an
        external_write). This process never wrote the layer → ``unknown``
        ("not built yet", not "changed but not written").
        """
        return classify_layer_quadrants(
            self._LAYER_ATTRS, self._layer_written_fp, self._layer_stamp,
            self.persist_dir,
            lambda attr: self._layer_fingerprint(getattr(self, attr)),
            self._layer_changed_by_other,
        )

    def _reconcile_ledger_on_boot(self) -> None:
        """Boot reconciliation: report「which layers had changes that never got written」(lost_since_last_save).

        Implemented in
        :func:`hippocampus_mcp.store.layer_gc.reconcile_ledger_on_boot`
        (pure parsing delegated to store.ledger.reconcile_on_boot).
        Three decidable cases: last row is an ``end`` with fingerprints →
        ``disk_mismatch``; last row is a ``begin`` → layers whose disk differs
        from the begin fingerprint are reported lost; last row is a ``fail`` →
        look back through the tail for the begin and judge as above. **Known
        limitation**: a process killed with changes that never initiated a
        save (no begin) ⇒ the ledger cannot know. Two-clock discipline: no new
        time fields. This method only does binding + log self-report.
        """
        self._boot_reconcile = None
        last = self._ledger_last
        if not isinstance(last, dict):
            return
        attrs = self._LAYER_ATTRS
        rec = _layer_gc.reconcile_ledger_on_boot(
            last, self._ledger_tail, list(attrs),
            lambda name: self._layer_fingerprint(getattr(self, attrs[name])),
        )
        self._boot_reconcile = rec
        lost = rec.get("lost_since_last_save")
        mismatch = rec.get("disk_mismatch")
        if lost:
            logger.warning(
                "启动对账：上次保存(seq=%s, phase=%s)没走完，%d 个层声明要写但磁盘"
                "内容不符（lost_since_last_save）：%s",
                last.get("seq"), last.get("phase"), len(lost), rec["note"],
            )
        elif mismatch:
            logger.warning(
                "启动对账：%d 个层的磁盘内容与账本末条(end seq=%s)不符：%s",
                len(mismatch), last.get("seq"), "、".join(mismatch),
            )

    def _save_layers_core(self) -> LayerSaveResult:
        """Write only the layers **this file is responsible for** (L0 / L2 / L3 etc.; never touches manager's L1 + vectors).

        **Under multi-process, every derived layer must share L1's semantics**:
        L1 (``memories.json``) is "conflict means merge" (additive), while the
        layers this file owns used to be **wholesale overwrites** — with two
        processes sharing one library, whoever persisted last made their
        version the whole library's, silently wiping the other's entity
        attachments / session registrations / tombstones (a measured
        incident). No exceptions, no logs.

        So this does **read-merge-write**: before persisting, read the disk
        version, take the union per the layer's convention, then write.
        Single-process (``multi_writer_enabled()`` false) behavior is exactly
        as before the change.

        The merged-in count is recorded in :attr:`last_layer_merge` and
        surfaced by ``stats()`` — "how many of someone else's entries I
        rescued" must be a visible number, not an archaeology project. The
        orchestration body is in :mod:`hippocampus_mcp.store.layer_save`;
        this shell only assembles the injection surface and writes the store
        segment's output back into brain bookkeeping state — fingerprint
        stamps / sequence numbers update only on a successful write (failure
        raises, never reaching the write-back).
        """
        result = save_layers_core(LayerSavePorts(
            persist_dir=self.persist_dir,
            layer_attrs=self._LAYER_ATTRS,
            get_layer=lambda attr: getattr(self, attr),
            set_layer=lambda attr, value: setattr(self, attr, value),
            layer_fingerprint=self._layer_fingerprint,
            layer_changed_by_other=self._layer_changed_by_other,
            read_layer_from_disk=self._read_layer_from_disk,
            read_raw_log_from_disk=self._read_raw_log_from_disk,
            merge_raw_log=self._merge_raw_log,
            merge_del_ledger_layer=self._merge_del_ledger_layer,
            merge_layer=self._merge_layer,
            disk_only_count=self._disk_only_count,
            reclip_derived_layers=self._reclip_derived_layers,
            gc_layer_del=self._gc_layer_del,
            reset_del_blocked=lambda: setattr(self, "_layer_del_blocked", 0),
            backend=self._backend,
            begin_layer_fps=self._begin_layer_fps,
            written_fp=self._layer_written_fp,
            layer_fp_on=lambda: load_config().layer_fp,
        ))
        self.last_save_layers = self._save_layer_status(
            result.persisted, result.layers_skipped)
        if not result.persisted:
            return result
        self.last_layer_merge = result.last_layer_merge
        self._layers_skipped = result.layers_skipped
        self._end_layer_fps = result.end_layer_fps
        self._layer_written_fp.update(result.written_fp)
        self._layer_stamp.update(result.layer_stamp)
        if result.del_gc_last is not None:
            self._layer_del_gc_last = result.del_gc_last
        # F11: after a full persist, L0 no longer has a backlog
        self._l0_pending = 0
        # Register presence on persist (a crashed exit still leaves a record; TTL-based cleanup is the backstop)
        self._save_seq += 1
        self._touch_writer()
        return result

    def _save_layer_status(self, persisted: bool,
                           skipped: list[str]) -> dict[str, str]:
        """B (verification feedback 2026-11-01): save's layer breakdown (written/skipped).

        persisted=False = no disk-write action happened; report all as skipped
        honestly (not failed — nothing errored, there was just nothing to
        write). The failed side lives in ``_save_layers``' exception hook
        (every layer marked failed, paired with saved=false + save_error).
        saved:true once papered "computed in memory, never written" over as
        silent success (which is how A's bug stayed hidden); this breakdown
        makes each layer's real disposition visible and is echoed in the
        session_end return body.
        """
        if not persisted:
            return {n: "skipped" for n in self._LAYER_ATTRS}
        dropped = set(skipped or [])
        return {n: ("skipped" if n in dropped else "written")
                for n in self._LAYER_ATTRS}

    def save(self) -> None:
        """Persist: vector store + LNN (manager) + the remaining layers (this file).

        G3: under the sqlite backend these two steps are wrapped in **one
        transaction** — L1 and the 10 derived layers live and die together,
        never leaving the interleaved state of "memories written, entity layer
        not" (the JSON scheme cannot supply that primitive at the filesystem
        level; the 10 layer files can only be ``os.replace``d one by one). The
        json backend's transaction is a no-op, behavior exactly as before the
        change.
        """
        backend = self._backend
        if backend is None:
            self.manager.save()
            self._save_layers()
            return
        with backend.transaction():
            self.manager.save()
            self._save_layers()

    def retry_save(self) -> dict[str, Any]:
        """Retry saving after a concurrency conflict: **L1 and L0 are patched up together**.

        Why not just ``manager.retry_save()``: it saves only **L1**
        (memories.json + vectors + LNN); the trailing ``manager.save()`` never
        writes ``brain_raw.json``. So "the batch this process just ingested"
        would end up "L1 on disk, L0 still in memory"; once the process exits
        the raw text is lost and L1's ``source_ids`` dangle — the first hole
        of appendix five's "L0 bookkeeping drift". This patches in L0's
        persist (``_save_layers`` unions with disk internally and will not
        overwrite raw written by other processes).

        Returns:
            ``manager.retry_save()``'s original stats (added /
            dropped_local_edits / disk_only), plus ``raw_log_written`` saying
            whether L0 truly persisted.
        """
        result = self.manager.retry_save()
        result["raw_log_written"] = bool(self.persist_dir)
        self._save_layers()
        return result

    def load(self, auto_backfill: bool | None = None) -> None:
        if not self.persist_dir:
            return
        self.manager.load(
            auto_backfill=self.auto_backfill if auto_backfill is None else auto_backfill
        )
        # Storage corruption is never silent: the bad file is already
        # quarantined; shout the reason here (defect N1)
        issue = self.manager.store.last_load_issue
        if issue:
            logger.error(
                "记忆存储加载异常（内存状态未清空，坏文件已隔离）: %s", issue
            )
        # 🔴 This used to hand-write a second "layer file -> attribute"
        # mapping, **duplicating** the class attribute _LAYER_ATTRS. Adding the
        # deletion-ledger layer stepped on it: the new layer was added to
        # _LAYER_ATTRS but not to this copy ⇒ the new layer was **never
        # loaded**, memory stayed empty, multi-process interception failed
        # wholesale, and the only symptom was one boot-reconciliation line
        # "brain_layerdel.json 磁盘内容与账本不符" (easily dismissed as an
        # unrelated dirty ledger).
        # Lesson: as long as one correspondence has a second hand-written
        # copy, adding a layer will miss one of them, and the missed copy
        # raises no error. Now _LAYER_ATTRS is reused directly.
        mapping = self._LAYER_ATTRS
        backend = self._backend or get_backend(self.persist_dir)
        for name, attr in mapping.items():
            if not backend.has_layer(name):
                continue
            disk_data = backend.read_layer(name)
            if disk_data is None:
                continue
            # L0 must **not** be replaced in memory directly from disk: this
            # process may still hold raw not yet persisted (ingest does not
            # persist by default), and a replacement loses it — exactly the
            # second hole of appendix five's "L0 bookkeeping drift"
            # (`load_state` is the conflict self-rescue action the guide
            # recommends; one call used to lose a batch).
            # In multi-process mode, union by id; single process keeps the
            # original semantics (direct replacement).
            if attr == "raw_log" and multi_writer_enabled():
                setattr(self, attr,
                        self._merge_raw_log(disk_data, self.raw_log))
            else:
                setattr(self, attr, disk_data)

        # 实体统计自愈：mention_count 旧实现是累加值（虚高），
        # memory_ids 也可能含已删记忆的悬挂引用 —— 以当前存活记忆重算一次。
        dropped = self._recompute_entity_stats()
        if dropped:
            logger.info("实体索引自愈：摘除 %d 条悬挂引用", dropped)
