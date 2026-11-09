# -*- coding: utf-8 -*-
"""Single source of truth for configuration — the repo's only env read
point, producing an immutable Config.

## Why this file exists

Design motivation: env reads used to be scattered across **23 files /
101 places**: 18 module-level import-time snapshots in server.py, 4
in-function reads each in manager/core/pure, 4 in domain/topic.py, plus
another 18 ``_env_*`` module-level evaluations in constants.
The consequence was "changing one knob requires grepping half the repo"
and "a test patches one copy while another stays unchanged".

This file collects them into one **explicit field list**:

* :class:`Config` — frozen dataclass, each field = one env knob, defaults
  checked one by one against pre-migration values (pinned by
  ``tests/test_config_parity.py``);
* :func:`load_config` — the **sole** ``os.environ`` read point.
  Deliberately **not cached**: tests and evaluation scripts rely on
  "change env → call again" to observe behavior changes (e.g.
  ``test_c1_source_drift.py`` D6 sets ``HIPPOCAMPUS_SOURCE_SIG=0``);
  caching would turn such switches into silent no-ops.

## Boundaries (scope-creep prevention)

* **Only collects the scattered points in server/manager/core/pure**
  (32 places). The ``_env_*`` module-level evaluations in
  ``common/constants.py`` and the parameterized scattered points in
  ``domain/topic.py`` are absorbed by the second batch (see the
  corresponding groups below). config depends on common/constants'
  defaults table (the lowest base layer) and never depends back on any
  upper layer.
* **Names and semantics of the 101 envs stay untouched**: Config
  relocates the read points, it does not redesign the knobs. Not a
  single user environment variable needs to change.

Dependency direction: ``config`` → ``common.constants`` (for default
value constants), no other dependencies.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

#: Default for the LNN context-branch ceiling (config is the lowest layer
#: and must not depend back on engine; manager imports from here)
LNN_CTX_MAX_DEFAULT = 8

# ---------------------------------------------------------------------------
# Typed env reads. **Four coexisting bool semantics are the status quo**
# (faithfully reproducing the existing scattered points), reproduced one
# by one, not unified — unification would be behavior drift:
#   A deny-list default true: "false/0/no" off, everything else including
#       unset on (server.py _AUTO_SAVE/_AUTO_BACKFILL/_SUPERSEDE)
#   B allow-list          : only "1/true/yes/on" on, everything else
#       including unset off (manager _recall_decay_enabled,
#       core NER_ALLOW_TORCH_ENV)
#   C deny-list default false: "false/0/no" off, otherwise... unset off,
#       other values on (server.py _WARMUP_DIAG, get default "0")
#   D non-empty and non-0 : empty/unset/"0" off, the rest (including
#       "false"!) on (manager ENV_ALLOW_MODEL_SWITCH)
# ---------------------------------------------------------------------------


def _flag_deny_true(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("false", "0", "no")


def _flag_allow_only(name: str, default: bool = False,
                     with_on: bool = True) -> bool:
    """Allow-list flag. ``with_on=False`` reproduces core.py NER's narrow
    list (("1","true","yes") without "on" — one word short of manager
    recall_decay's wide list; the second parity crack caught live during
    M1; reproduced field by field, not unified)."""
    words = ("1", "true", "yes", "on") if with_on else ("1", "true", "yes")
    return os.environ.get(name, "").strip().lower() in words


def _flag_deny_false(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "0")
    return raw.strip().lower() not in ("false", "0", "no")


def _flag_nonempty(name: str, default: bool = False) -> bool:
    return (os.environ.get(name) or "").strip() not in ("", "0")


def _flag_tristate(name: str, default: bool) -> bool:
    """Type E: both lists are recognized, **any other value falls back to
    default** (unlike type A, which judges it on).

    This matters: whatever ``HIPPOCAMPUS_MAP_MENTION=maybe`` produced
    before the migration, it must still produce after — behavior on dirty
    values is part of the behavior.
    """
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


def _positive_float(name: str, default: float) -> float:
    """Value must be positive, otherwise fall back to the default (the
    original ``v > 0`` gate of ``constants.writer_ttl``).

    Why keep this gate instead of treating it as ordinary misconfiguration
    fallback: for knobs like TTL, ``0`` means "expire immediately,
    invisible to everyone" — configuring it would only create a batch of
    ghost writers, worse than not taking effect. The status quo chose to
    block it, and this absorption copies that.
    """
    v = _float(name, default)
    return v if v > 0 else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Whole-repo config snapshot (immutable). Field = env knob, default =
    the measured pre-migration value.

    Every :func:`load_config` call re-reads env — that is a feature, not
    an omission: tests and evaluation depend on the "change env, takes
    effect immediately" semantics (see the module docstring).
    """

    # --- Startup and library location (server.py module level) ---
    log_level: str = "INFO"
    persist_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".hippocampus_mcp"))

    # --- Retrieval tuning (server.py module level, LoCoMo tuning results) ---
    lambda_decay: float = 7776000.0        # 90 days
    mu_importance: float = 15552000.0      # 180 days
    w_sim: float = 0.95
    w_time: float = 0.03
    w_recall: float = -0.02
    # Retrieval strength weight: +w·log1p(recall_strength) — successful
    # recall of consolidation traces (testing effect). 0 = off
    # (bit-for-bit identical to the old scoring).
    w_strength: float = 0.03
    # Rewrite-history ledger ring: mem_history.jsonl keeps the latest N
    # entries (the before journal of update/merge, a sidecar audit
    # surface; shape mirrors tag_ops). For the slicing semantics of 0 see
    # the store/mem_history module docstring (the [-0:] full-volume
    # quirk; callers never pass 0).
    mem_history_ring: int = 200
    nav_penalty: float = 1.0
    recall_decay: bool = False
    recall_decay_tau: float = 7776000.0    # same basis as lambda_decay (90 days)
    library_cache_ttl: float = 30.0

    # --- Model and embedding (server.py / core.py / embeddings.ENV_MODEL) ---
    emb_model: str = ""
    crl_model: str = ""
    allow_model_switch: bool = False

    # --- Write policy (server.py module level) ---
    auto_save: bool = True
    auto_backfill: bool = True
    supersede: bool = True

    # --- Entity and block limits (pure.py in-function) ---
    entity_lexicon: str = ""
    daily_activity_days: str = ""
    knowledge_block_limit: str = ""
    pin_block_limit: str = ""

    # --- LNN (manager.py in-function) ---
    lnn_ctx_max: int = LNN_CTX_MAX_DEFAULT

    # --- Drift audit (core.py in-function) ---
    source_root: str = ""
    ner_allow_torch: bool = False

    # --- Warmup and handshake (server.py module level, launcher cooperation protocol) ---
    warmup: str = "auto"
    warmup_timeout: float = 90.0
    warmup_wait: str = "raise"
    warmup_diag: bool = False
    handshake_window: float = 15.0
    preload: str = "1"
    preload_budget: float = 6.0

    # --- Tool surface (server.py module level, profile gating) ---
    tool_profile: str = "common"
    common_tools: str = ""

    # ------------------------------------------------------------------
    # The 28 knobs absorbed in the second batch
    # ------------------------------------------------------------------
    # Four sources:
    #   * common/constants' **module-level evaluations** (3 rings +
    #     writer TTL + three layer switches)
    #   * brain/core.py in-function scattered points (status-card
    #     contradiction signal + three inspection chains)
    #   * brain/session.py's 5 pre-activation slot switches (tri-state
    #     since long ago)
    #   * domain/topic.py's 6 (domain may not import config ⇒ parameterized
    #     injection)
    # Defaults are always written as **literals** rather than imported
    # constants: config is an L0 pure leaf (dependency graph
    # ``_ALLOWED["config"] == {"config"}``) and must not depend back on
    # common or domain — the equivalence of both sides' identical values
    # is pinned by ``test_config_parity`` P7.

    # Ring buffers and write discipline (formerly constants module-level evaluations)
    tag_ops_ring: int = 50
    save_ledger_ring: int = 200
    l0_autosave_every: int = 20
    #: Writer heartbeat expiry window. **Positive-value gate**: any parsed
    #: value <= 0 falls back to the default (status quo semantics, see the
    #: original ``if v > 0`` branch of ``constants.writer_ttl`` — it is not
    #: misconfiguration fallback; meaningless values like "expire in 0
    #: seconds" must be blocked)
    writer_ttl: float = 3600.0
    layer_fp: bool = True
    layer_del: bool = True
    source_sig: bool = True

    # Inspection chains and status card (formerly brain/core.py in-function scattered points)
    status_contradictions: bool = True
    map_auto_verify: bool = True
    map_auto_interval_h: float = 6.0
    map_auto_budget_s: float = 1.0
    map_mention: bool = True
    map_mention_interval_h: float = 6.0
    map_mention_budget_s: float = 1.0
    map_mention_max_records: int = 2000
    file_registry_off: bool = False
    file_registry_interval_h: int = 6

    # session_start pre-activation slots (formerly the env branch of
    # _block_switch; the tri-state semantics itself stays in session.py,
    # Config only provides the answer for **param==0**)
    overview_block: bool = True
    status_block: bool = True
    task_profile_block: bool = True
    map_index_block: bool = True
    map_detail_block: bool = True

    # Topic layer (formerly domain/topic.py; injected into the pure function by the caller)
    topic_layer: bool = True
    topic_join_sim: float = 0.55
    topic_max: int = 1000
    topic_max_members: int = 60
    topic_summary_topn: int = 3
    topic_low_info: bool = True

    # ------------------------------------------------------------------
    # User-level memory inheritance slots (new fields have no
    # pre-migration semantics; types: count tri-state + threshold-tier
    # floor)
    # ------------------------------------------------------------------
    #: User-level inheritance slot count. For the tri-state table see
    #: ``brain/pure.resolve_user_scope_limit``: when param==0 this field
    #: is consulted — **positive = count, negative = off**, 0 / empty /
    #: invalid falls back to the default 3 (note the difference from
    #: pin_limit: a negative env value for pin falls to the default,
    #: while a negative env value for user_scope is an explicit off).
    user_scope_limit: int = 3
    #: User-level task directory slot count. The tri-state table lives in
    #: ``brain/pure.resolve_task_directory_limit`` (isomorphic with
    #: user_scope): when param==0 this field is consulted — **positive =
    #: count, negative = off**, 0 / empty / invalid falls back to
    #: default 8.
    task_directory_limit: int = 8
    #: Threshold-tier floor: non-pinned entries enter the inheritance slot
    #: only when importance >= this value.
    user_scope_importance: float = 0.7
    #: Promotion candidate stability period: becomes a candidate only
    #: after being reproduced in >= N distinct sessions.
    promote_min_sessions: int = 2
    # Recall-side staleness window (days): a memory with age > window and
    # not pinned → excluded from the recall / get_memory_context result
    # set. **Zero state change**: no deletion, no archiving, no state
    # change, only output gating.
    # **Default 0 = off**: archived old memories must remain findable by
    # similar queries; time alone must not unilaterally kill them. Output
    # length is controlled by top_k / limit; lower-ranked items are
    # naturally truncated.
    # To "see only recent", explicitly set a number of days (or pass
    # occurred_after/before at query time).
    # Time basis = _reference_time or wall clock (constraint 4).
    recency_window_days: int = 0
    #: Sensitive word list file path: one word per line, ``#`` comment
    #: lines and blank lines skipped; empty = no deployed word list
    #: (**the pattern-level built-in fallback remains** — the second gate
    #: of the "any check fails → no write" discipline does not depend on
    #: the word list existing).
    sensitive_words: str = ""
    #: Escape hatch: **False (default) = home-library write boundary
    #: enforced** — write tools raise explicitly when their target is
    #: neither the home library bound for this connection nor the user
    #: managed channel; True = globally open up cross-library writes (for
    #: ops library relocation / bulk migration).
    #: Semantic type B (**allow-list**, same as recall_decay) — an escape
    #: hatch must be "explicitly stated before it opens": empty strings
    #: and dirty values (``maybe``) all fall back to the default False.
    #: Type C (deny-list) cannot be used: its *"everything that isn't
    #: false is true"* would let an empty env or a typoed value silently
    #: open the boundary.
    cross_library_write: bool = False

    # ------------------------------------------------------------------
    # The 4 knobs absorbed from the engine/brain direct-read surface
    # ------------------------------------------------------------------
    # String transport, semantic parsing stays at the call site (original
    # expressions copied verbatim, zero semantic drift):
    #   * onnx_no_arena's original form `not in ("1", "true")` — case
    #     sensitive, without yes/on; the generic bool dispatch would
    #     drift, so it is absorbed as str;
    #   * emb_model / entity_lexicon fields already exist; this batch
    #     only kills the second read points in
    #     embeddings.resolve_embedder_model and brain/entity.
    # For the absorption of the store/lnn/library/map/retrieval/domain
    # constructor-injection surface + root-level entries, see the
    # corresponding groups below.
    onnx_backend: str = "auto"        # torch/onnx/auto; invalid values fall back to auto
    onnx_no_arena: str = ""           # arena off only for "1"/"true" (on by default)
    shared_embedder: str = ""         # shared pool off only for 0/false/off/no (on by default)
    emb_cache: str = ""               # explicit embedding cache directory
    # Auto-select the "application default" embedding model by system
    # language (Chinese → BGE-zh, others → MiniLM).
    # String transport, semantic parsing in
    # embeddings.auto_model_selection_enabled (same rule as
    # shared_embedder copied verbatim: 0/false/off/no turns it off,
    # unset = on).
    # **Only affects the "nothing configured" path**: an explicit
    # HIPPOCAMPUS_EMB_MODEL always wins.
    emb_model_auto: str = ""
    # Disposition policy for over-long content (beyond the embedding
    # model's token ceiling).
    # warn (default, = self-report only) / split (split into multiple
    # cards) / reject (refuse the write).
    # **Allow-list**: only an explicit split|reject changes behavior,
    # everything else is warn (same rule: a typo equals not written; the
    # switch is never silently opened).
    # Semantic parsing in domain.long_content.parse_policy (pure
    # function, unit-testable).
    long_content: str = ""
    # **Per-entry character cap on a single memory's content** in the
    # recall response body (truncation guard).
    # 0 = no truncation (status quo; one recall of a long memory can
    # exceed ten thousand characters). String transport, semantic parsing
    # in application/recall_service (`_content_cap`; non-numeric/negative
    # treated as 0 = no truncation; a typo equals not written).
    # **Truncation must self-report**: truncated entries carry
    # content_truncated=true + content_chars, never a silent cut.
    recall_content_max_chars: str = ""

    # ------------------------------------------------------------------
    # Absorption of the 16-knob constructor-injection surface of
    # store/lnn/library/map/retrieval/domain.
    # All are **string transport** (rule: semantic parsing stays at the
    # consumer, zero drift). The consumers
    # (store/lnn/library/map/retrieval/domain) are bound by the G1 matrix
    # not to import config — the sole production constructor **engine**
    # (at manager import time) injects via apply_config; the consumers'
    # direct env reads are demoted to "direct-read fallback" (env first,
    # injected value as fallback — the surface where tests set os.environ
    # directly is preserved as-is; on the production path, when env is
    # unset the default is routed single-point through here).
    # All defaults = the verbatim pre-migration env fallback values.
    # ------------------------------------------------------------------
    snapshot_min_hours: str = "6"       # HIPPOCAMPUS_SNAPSHOT_MIN_HOURS
    multi_writer: str = "1"             # HIPPOCAMPUS_MULTI_WRITER
    save_lock_timeout: str = "60"       # HIPPOCAMPUS_SAVE_LOCK_TIMEOUT
    matrix_dtype: str = "fp32"          # HIPPOCAMPUS_MATRIX_DTYPE
    vectors_mmap: str = "0"             # HIPPOCAMPUS_VECTORS_MMAP
    allow_stale_save: str = ""          # HIPPOCAMPUS_ALLOW_STALE_SAVE
    store_backend: str = "json"         # HIPPOCAMPUS_STORE_BACKEND
    fts5: str = ""                      # HIPPOCAMPUS_FTS5
    lnn_backend: str = ""               # HIPPOCAMPUS_LNN_BACKEND
    library_roots: str = ""             # HIPPOCAMPUS_LIBRARY_ROOTS
    #: Default residency 2→3 (measured capacity trade-off); the field lands as 3.
    max_resident_libs: str = "3"        # HIPPOCAMPUS_MAX_RESIDENT_LIBS
    file_registry_ignore: str = ""      # HIPPOCAMPUS_FILE_REGISTRY_IGNORE
    file_registry_max_files: str = ""   # HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES
    file_registry_budget_s: str = ""    # HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S
    intent_routing: str = "0"           # HIPPOCAMPUS_INTENT_ROUTING
    tag_schema: str = ""                # HIPPOCAMPUS_TAG_SCHEMA

    # ------------------------------------------------------------------
    # Brain-side knobs for the daemon idle "sleep window".
    # **The master switch is not in config**: ``HIPPOCAMPUS_SLEEP_IDLE``
    # (idle seconds before sleeping, 0 = off) is decided by the daemon
    # process topology and, like model-unload, is read directly from env
    # in daemon.py (rule: root-level entries are startup-topology
    # plumbing).
    # Only the three semantic knobs of the "sleep batch content" are
    # absorbed here, read live by brain.core.sleep_batch at call time
    # (same precedent as source_root / status_contradictions).
    # Defaults do not change production behavior: the sleep batch itself
    # is off by default (SLEEP_IDLE=0); these knobs are only read after
    # the switch is on.
    # ------------------------------------------------------------------
    #: Replay window (days): only L1 memories "written within the window"
    #: are replayed for co-activation reinforcement.
    #: Aligned with the biological fact — SWR replays **recent**
    #: experience, not a whole-library rewind.
    sleep_replay_days: float = 2.0
    #: Max L1 records participating in replay per sleep batch (cap for
    #: large libraries; 0 = unlimited).
    sleep_max_replay: int = 200
    #: Clustering floor: memories with fewer entity co-occurrences than
    #: this do not form a replay cluster (co-activation needs at least
    #: two members to matter; memories without entities do not
    #: participate).
    sleep_min_cluster: int = 2

    # ------------------------------------------------------------------
    # Two write-side presentation-line knobs (encoding gating /
    # prospective triggering)
    # ------------------------------------------------------------------
    #: Novelty/prediction-error gated encoding. True (default) = at
    #: ingest sentence adjudication, compare against existing schemas
    #: (L3 profile / L2T knowledge-point summaries / high-confidence L1):
    #: confirmation → no boost (duplicates are dedup's business),
    #: contradiction → importance raised + conflicting object echoed
    #: back, brand-new entity → small weight boost. **Gating only adjusts
    #: encoding strength (importance), it never creates a pin**: when
    #: base < 0.7 the boosted value is clamped to 0.69 and never newly
    #: crosses PIN_IMPORTANCE — pin entry/exit authority is left to the
    #: user/feedback channel (pins accumulating monotonically is a lesson
    #: on record in policies.py).
    #: With an explicitly passed importance the gate is bypassed entirely
    #: (the host decides).
    #: Semantic type A (deny-list default true): 0/false/no/off explicitly
    #: off, dirty values fall back to default on.
    novelty_gate: bool = True
    #: Prospective memory contextual triggering. True (default) = at
    #: remember_intent registration, trigger conditions are derived from
    #: the text (trigger_entities / trigger_topic); when the query
    #: entities/content words of get_memory_context intersect the trigger
    #: set of an active open intent, that intent is carried back at the
    #: front of block_2 (cap 5, no inflation). Turning it off shuts
    #: **both ends**: registration no longer derives trigger sets and
    #: assembly no longer triggers carry-back (existing open intents that
    #: already carry trigger sets also never trigger). Semantic type A as
    #: above.
    prospective_trigger: bool = True
    #: R1 (2026-11-06): the production MCP ``recall`` tool shares
    #: the evaluation scoring pipeline (BM25/RRF, MMR, entity bonus,
    #: hard relevance floor, retired/quarantine filtering). True by
    #: default; ``HIPPOCAMPUS_RECALL_FUSION=0`` restores the legacy
    #: additive ``manager.recall`` path. Semantic type A (deny-list):
    #: only an explicit 0/false/no turns it off; a typo stays enabled.
    recall_fusion: bool = True
    #: R1-B4 (2026-11-08) MMR dimension-scale fix: the shared pipeline's
    #: MMR selection multiplies the fused-domain relevance by
    #: ``common.constants.MMR_REL_SCALE`` (calibration anchor in the
    #: constant's note) so the relevance term and the native 0-1 cosine
    #: diversity term compete on the same order of magnitude. Default ON --
    #: this is a defect fix, not a new scoring item: measured on a real
    #: library the bug costs -34% mean top-5 similarity with 1.73 sim=0
    #: literal placeholders per query, and the golden fixtures cannot
    #: discriminate it (their candidate pools never exceed top_k, so the
    #: MMR branch never runs there). Semantic type A (deny-list, same as
    #: recall_fusion): only an explicit 0/false/no turns it off; a typo
    #: stays enabled. The constant itself is deliberately NOT an env knob
    #: (calibration surface, not a tuning knob). Rollback chain:
    #: ``HIPPOCAMPUS_MMR_SCALE_FIX=0`` -> raw-score MMR;
    #: ``RECALL_FUSION=0`` -> legacy additive path. Design doc:
    #: docs/design/R1-B4_MMR量纲缺陷_拍板材料_20261108.md (repo-only).
    mmr_scale_fix: bool = True
    #: V2 (2026-11-07) recall query fanout: for natural-sentence queries
    #: (>= 10 chars) derive a keyword variant and run the shared pipeline
    #: twice, merging the hit lists with RRF. Default OFF — recall behaviour
    #: must stay reproducible for the pinned evaluation baselines until this
    #: is validated on the dev library and promoted by decision. Semantic
    #: type B (allow-list): only an explicit 1/true enables it; a typo
    #: stays off. Scoring formulas untouched (R1's territory).
    recall_query_fanout: bool = False
    #: UR (2026-11-07) usage-recency scoring: boost = w * exp(-dt/tau) *
    #: log1p(min(strength, cap)), anchored on ``last_recalled_at`` (true
    #: result-set membership) times capped retrieval strength. Default OFF,
    #: semantic type B (allow-list): only an explicit 1/true enables it.
    #: The numeric sub-knobs are MASKED by this bool at the composition
    #: root -- setting only the numeric envs must NOT enable the item.
    #: Calibration: the production fusion-path total score spans
    #: 0.0068-0.1191 (tests/fixtures/r1_recall_golden.json), so w=0.005
    #: caps the boost at ~0.015 (~13% of a top score) -- a tie-breaker,
    #: not a dominant term. (The 0.30 hard floor applies to vector
    #: SIMILARITY, not to the total score.)
    usage_recency: bool = False
    usage_recency_w: float = 0.005
    usage_recency_tau_days: float = 7.0
    usage_recency_cap_strength: int = 20
    #: Adaptive result-count selection (2026-11-07): Stage-C selector over
    #: the recall pipeline's hard-gated candidate pool. Replaces the fixed
    #: "always fill top_k" MMR truncation with marginal-utility selection
    #: (normalized relevance + redundancy + coverage - token cost), so
    #: high-relevance low-redundancy pools return MORE entries and noisy
    #: pools return FEWER -- top_k becomes a hard CAP, never a quota.
    #: Default ON since 2026-11-08 (user decision after the serial AR/UR
    #: trial: 120-question LongMemEval, AR +1.67pp Hit@5, 4 rescued / 2
    #: lost, no-regression signal; R4 full-500 review pending, rollback =
    #: this bool or HIPPOCAMPUS_ADAPTIVE_RECALL=0). Numeric sub-knobs are
    #: MASKED by this bool at the composition root. Design doc:
    #: docs/design/自适应召回条数_设计书_20261107.md (repo-only).
    adaptive_recall: bool = True
    #: Stop expanding after k_min picks unless normalized relevance
    #: (score / score_top, internal signal only) reaches this floor.
    adaptive_expand_floor: float = 0.6
    #: Marginal-utility stop threshold (normalized units).
    adaptive_tau_stop: float = 0.1
    #: Relevance weight in the utility formula (fixed MMR uses 0.7 too,
    #: but against the RAW score -- here it acts on the normalized r).
    adaptive_lambda: float = 0.7
    #: Facet-coverage bonus weight (new query entities/tags per pick).
    adaptive_beta: float = 0.15
    #: Token-cost penalty weight.
    adaptive_gamma: float = 0.1
    #: Global ceiling on the caller's top_k when adaptive is on (top_k
    #: stays a cap; this bounds worst-case expansion).
    adaptive_max_k: int = 20
    #: Access-driven profile promotion. True (default) = when consolidate
    #: refreshes the profile, L1 entries that "no one pinned but keep
    #: being recalled (retrieval strength >= 3) and have lived long
    #: enough (>= 7 days)" graduate into the dedicated L3 slot
    #: ``recurring_facts`` (appended at the presentation layer; no ledger
    #: writes, no state changes, never touches the user library — for
    #: the boundary with the U3 cross-library promotion flow see the
    #: domain/profile_ops module docstring). Turning it off only stops
    #: **new** graduate entries: invalidation pruning of existing entries
    #: (after archive/delete) is still managed by prune, leaving no
    #: zombies. Threshold constants in common/constants
    #: (PROFILE_STRENGTH_*). Semantic type A (deny-list default true):
    #: 0/false/no/off explicitly off, dirty values fall back to default
    #: on.
    profile_promotion: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Dict view (for debugging/self-reporting; not part of behavior)."""
        return dict(self.__dict__)


#: env name → Config field name. One mapping table, guarding against
#: "changed the env, forgot the field".
ENV_MAP: dict[str, str] = {
    "HIPPOCAMPUS_LOG_LEVEL": "log_level",
    "HIPPOCAMPUS_PERSIST_DIR": "persist_dir",
    "HIPPOCAMPUS_LAMBDA_DECAY": "lambda_decay",
    "HIPPOCAMPUS_MU_IMPORTANCE": "mu_importance",
    "HIPPOCAMPUS_W_SIM": "w_sim",
    "HIPPOCAMPUS_W_TIME": "w_time",
    "HIPPOCAMPUS_W_RECALL": "w_recall",
    # --- Retrieval strength weight (0 = off) ---
    "HIPPOCAMPUS_W_STRENGTH": "w_strength",
    # --- Rewrite-history ledger ring ---
    "HIPPOCAMPUS_MEM_HISTORY_RING": "mem_history_ring",
    "HIPPOCAMPUS_NAV_PENALTY": "nav_penalty",
    "HIPPOCAMPUS_RECALL_DECAY": "recall_decay",
    "HIPPOCAMPUS_RECALL_DECAY_TAU": "recall_decay_tau",
    "HIPPOCAMPUS_LIBRARY_CACHE_TTL": "library_cache_ttl",
    "HIPPOCAMPUS_EMB_MODEL": "emb_model",
    "HIPPOCAMPUS_CRL_MODEL": "crl_model",
    "HIPPOCAMPUS_ALLOW_MODEL_SWITCH": "allow_model_switch",
    "HIPPOCAMPUS_AUTO_SAVE": "auto_save",
    "HIPPOCAMPUS_AUTO_BACKFILL": "auto_backfill",
    "HIPPOCAMPUS_SUPERSEDE": "supersede",
    "HIPPOCAMPUS_ENTITY_LEXICON": "entity_lexicon",
    "HIPPOCAMPUS_DAILY_ACTIVITY_DAYS": "daily_activity_days",
    "HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT": "knowledge_block_limit",
    "HIPPOCAMPUS_PIN_BLOCK_LIMIT": "pin_block_limit",
    "HIPPOCAMPUS_LNN_CTX_MAX": "lnn_ctx_max",
    "HIPPOCAMPUS_SOURCE_ROOT": "source_root",
    "HIPPOCAMPUS_NER_ALLOW_TORCH": "ner_allow_torch",
    "HIPPOCAMPUS_WARMUP": "warmup",
    "HIPPOCAMPUS_WARMUP_TIMEOUT": "warmup_timeout",
    "HIPPOCAMPUS_WARMUP_WAIT": "warmup_wait",
    "HIPPOCAMPUS_WARMUP_DIAG": "warmup_diag",
    "HIPPOCAMPUS_HANDSHAKE_WINDOW": "handshake_window",
    "HIPPOCAMPUS_PRELOAD": "preload",
    "HIPPOCAMPUS_PRELOAD_BUDGET": "preload_budget",
    "HIPPOCAMPUS_TOOL_PROFILE": "tool_profile",
    "HIPPOCAMPUS_COMMON_TOOLS": "common_tools",

    # --- Recall-side staleness window (days; 0 = off) ---
    "HIPPOCAMPUS_RECENCY_WINDOW_DAYS": "recency_window_days",

    # --- Second batch, 28 absorbed ---
    "HIPPOCAMPUS_TAG_OPS_RING": "tag_ops_ring",
    "HIPPOCAMPUS_SAVE_LEDGER_RING": "save_ledger_ring",
    "HIPPOCAMPUS_L0_AUTOSAVE_EVERY": "l0_autosave_every",
    "HIPPOCAMPUS_WRITER_TTL": "writer_ttl",
    "HIPPOCAMPUS_LAYER_FP": "layer_fp",
    "HIPPOCAMPUS_LAYER_DEL_LEDGER": "layer_del",
    "HIPPOCAMPUS_SOURCE_SIG": "source_sig",
    "HIPPOCAMPUS_STATUS_CONTRADICTIONS": "status_contradictions",
    "HIPPOCAMPUS_MAP_AUTO_VERIFY": "map_auto_verify",
    "HIPPOCAMPUS_MAP_AUTO_INTERVAL": "map_auto_interval_h",
    "HIPPOCAMPUS_MAP_AUTO_BUDGET_S": "map_auto_budget_s",
    "HIPPOCAMPUS_MAP_MENTION": "map_mention",
    "HIPPOCAMPUS_MAP_MENTION_INTERVAL": "map_mention_interval_h",
    "HIPPOCAMPUS_MAP_MENTION_BUDGET_S": "map_mention_budget_s",
    "HIPPOCAMPUS_MAP_MENTION_MAX_RECORDS": "map_mention_max_records",
    "HIPPOCAMPUS_FILE_REGISTRY_OFF": "file_registry_off",
    "HIPPOCAMPUS_FILE_REGISTRY_INTERVAL": "file_registry_interval_h",
    "HIPPOCAMPUS_OVERVIEW_BLOCK": "overview_block",
    "HIPPOCAMPUS_STATUS_BLOCK": "status_block",
    "HIPPOCAMPUS_TASK_PROFILE_BLOCK": "task_profile_block",
    "HIPPOCAMPUS_MAP_INDEX_BLOCK": "map_index_block",
    "HIPPOCAMPUS_MAP_DETAIL_BLOCK": "map_detail_block",
    "HIPPOCAMPUS_TOPIC_LAYER": "topic_layer",
    "HIPPOCAMPUS_TOPIC_JOIN_SIM": "topic_join_sim",
    "HIPPOCAMPUS_TOPIC_MAX": "topic_max",
    "HIPPOCAMPUS_TOPIC_MAX_MEMBERS": "topic_max_members",
    "HIPPOCAMPUS_TOPIC_SUMMARY_TOPN": "topic_summary_topn",
    "HIPPOCAMPUS_TOPIC_LOW_INFO_FILTER": "topic_low_info",

    # --- User-level memory inheritance slots ---
    "HIPPOCAMPUS_USER_SCOPE_LIMIT": "user_scope_limit",
    "HIPPOCAMPUS_USER_SCOPE_IMPORTANCE": "user_scope_importance",

    # --- Promotion flow ---
    "HIPPOCAMPUS_PROMOTE_MIN_SESSIONS": "promote_min_sessions",
    "HIPPOCAMPUS_SENSITIVE_WORDS": "sensitive_words",

    # --- Home-library write boundary ---
    "HIPPOCAMPUS_CROSS_LIBRARY_WRITE": "cross_library_write",

    # --- User-level task directory ---
    "HIPPOCAMPUS_TASK_DIRECTORY_LIMIT": "task_directory_limit",

    # --- The 4 engine/brain direct-read surface knobs ---
    "HIPPOCAMPUS_EMB_BACKEND": "onnx_backend",
    "HIPPOCAMPUS_ONNX_NO_ARENA": "onnx_no_arena",
    "HIPPOCAMPUS_SHARED_EMBEDDER": "shared_embedder",
    "HIPPOCAMPUS_EMB_CACHE": "emb_cache",

    # --- Auto-select default embedding model by system language ---
    "HIPPOCAMPUS_EMB_MODEL_AUTO": "emb_model_auto",

    # --- Over-long content disposition (warn/split/reject, default warn) ---
    "HIPPOCAMPUS_LONG_CONTENT": "long_content",

    # --- recall per-entry content truncation guard (0 = no truncation) ---
    "HIPPOCAMPUS_RECALL_CONTENT_MAX_CHARS": "recall_content_max_chars",

    # --- R1: production recall shares the evaluation scoring pipeline ---
    "HIPPOCAMPUS_RECALL_FUSION": "recall_fusion",
    # --- R1-B4: MMR relevance/penalty dimension-scale fix (default on) ---
    "HIPPOCAMPUS_MMR_SCALE_FIX": "mmr_scale_fix",
    # --- V2: recall query fanout (keyword variant + RRF merge; default off) ---
    "HIPPOCAMPUS_RECALL_QUERY_FANOUT": "recall_query_fanout",

    # --- UR: usage-recency scoring (bool + numeric sub-knobs; default off) ---
    "HIPPOCAMPUS_USAGE_RECENCY": "usage_recency",
    "HIPPOCAMPUS_USAGE_RECENCY_W": "usage_recency_w",
    "HIPPOCAMPUS_USAGE_RECENCY_TAU_DAYS": "usage_recency_tau_days",
    "HIPPOCAMPUS_USAGE_RECENCY_CAP": "usage_recency_cap_strength",

    # --- Adaptive result-count selection (bool + numeric sub-knobs;
    # --- default ON since 2026-11-08; explicit 0 disables everything) ---
    "HIPPOCAMPUS_ADAPTIVE_RECALL": "adaptive_recall",
    "HIPPOCAMPUS_ADAPTIVE_EXPAND_FLOOR": "adaptive_expand_floor",
    "HIPPOCAMPUS_ADAPTIVE_TAU_STOP": "adaptive_tau_stop",
    "HIPPOCAMPUS_ADAPTIVE_LAMBDA": "adaptive_lambda",
    "HIPPOCAMPUS_ADAPTIVE_BETA": "adaptive_beta",
    "HIPPOCAMPUS_ADAPTIVE_GAMMA": "adaptive_gamma",
    "HIPPOCAMPUS_ADAPTIVE_MAX_K": "adaptive_max_k",

    # --- Sleep-window brain-side knobs (master switch in daemon) ---
    "HIPPOCAMPUS_SLEEP_REPLAY_DAYS": "sleep_replay_days",
    "HIPPOCAMPUS_SLEEP_MAX_REPLAY": "sleep_max_replay",
    "HIPPOCAMPUS_SLEEP_MIN_CLUSTER": "sleep_min_cluster",

    # --- Write-side presentation line (encoding gating / prospective triggering) ---
    "HIPPOCAMPUS_NOVELTY_GATE": "novelty_gate",
    "HIPPOCAMPUS_PROSPECTIVE_TRIGGER": "prospective_trigger",

    # --- Access-driven profile promotion ---
    "HIPPOCAMPUS_PROFILE_PROMOTION": "profile_promotion",

    # --- The 16-knob store/lnn/library/map/retrieval/domain constructor
    #     injection surface (injection seams at consumers: each module's apply_config) ---
    "HIPPOCAMPUS_SNAPSHOT_MIN_HOURS": "snapshot_min_hours",
    "HIPPOCAMPUS_MULTI_WRITER": "multi_writer",
    "HIPPOCAMPUS_SAVE_LOCK_TIMEOUT": "save_lock_timeout",
    "HIPPOCAMPUS_MATRIX_DTYPE": "matrix_dtype",
    "HIPPOCAMPUS_VECTORS_MMAP": "vectors_mmap",
    "HIPPOCAMPUS_ALLOW_STALE_SAVE": "allow_stale_save",
    "HIPPOCAMPUS_STORE_BACKEND": "store_backend",
    "HIPPOCAMPUS_FTS5": "fts5",
    "HIPPOCAMPUS_LNN_BACKEND": "lnn_backend",
    "HIPPOCAMPUS_LIBRARY_ROOTS": "library_roots",
    "HIPPOCAMPUS_MAX_RESIDENT_LIBS": "max_resident_libs",
    "HIPPOCAMPUS_FILE_REGISTRY_IGNORE": "file_registry_ignore",
    "HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES": "file_registry_max_files",
    "HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S": "file_registry_budget_s",
    "HIPPOCAMPUS_INTENT_ROUTING": "intent_routing",
    "HIPPOCAMPUS_TAG_SCHEMA": "tag_schema",
}


# ── Root-level entry env direct-read exemption list (closed) ──────────
# The env direct reads of root-level entries (launcher/daemon/cli/warmup/
# bridge/__init__) are **deliberately not absorbed** into load_config
# fields — they are **startup-topology plumbing**: mostly "process
# decisions made before Config loads" (launcher switches / daemon
# socket·key / self-exit timeout / third-party library thread env);
# absorbing them into config would loop the load order. The list is
# **closed**: the G7 guard in tests/test_m1c_batch2_config.py pins the
# read counts per file; a new direct read must first register its
# rationale here, otherwise the test goes red.
#
# | File | Reads | List and exemption rationale |
# |---|---|---|
# | launcher.py | 13 | DAEMON_READY_TIMEOUT (module snapshot, a process decision waiting for daemon readiness); RUNTIME_DIR (runtime directory); the LAUNCHER switch (whether to use the daemon topology); `dict(os.environ)` (spawn environment copy, not a knob); LAUNCHER_DEBUG; DIAG; RECONNECT; PERSIST_DIR / EMB_MODEL / DAEMON_LIBRARY / DAEMON_VERSION (**announce pass-through to the daemon** — launcher predates config and is only a messenger); PERSIST_DIR **also gets one pre-start writability precheck** (`preflight_persist_dir`: an unwritable library directory means every write is silently lost — that is broken configuration and must fail fast rather than fall back to direct connection) |
# | daemon.py | 11 | DAEMON_HEARTBEAT (module snapshot); MODEL_UNLOAD / SLEEP_IDLE / SLEEP_BUDGET (sentinel thresholds, decided by process topology — the sleep master switch stays in daemon under the "root-level entries are startup-topology plumbing" rule); DAEMON_ECHO / FAULTHANDLER (diagnostics, effective before Config); DAEMON_KEY / LIBRARY / MODEL / IDLE_EXIT (socket/key/self-exit, process topology) |
# | cli.py | 16 | APPDATA (system env, not this project); PERSIST_DIR / EMB_MODEL / EMB_CACHE / TOOL_PROFILE (**display pass-through** for human-facing output/introspection commands — the CLI reads for humans, values match config, no behavior branching); `dict(os.environ)` (setup post-write verification copies the spawn environment per the just-written entry for a real handshake, not a knob); HF_ENDPOINT (model prefetch uses the China mirror, `setdefault`'s return value is reused directly, **never read a second time**); DAEMON_LIBRARY / EMB_MODEL / DAEMON_VERSION / PERSIST_DIR once more each (the context subcommand assembles the HELLO handshake from the **real host's perspective** — the one-shot client must see the raw env, not config-parsed values, otherwise injected content would diverge from the library the host actually bound) |
# | warmup.py | 4 | OMP/MKL_NUM_THREADS (third-party library env, setdefault warmup thread caps); LNN_BACKEND / EMB_BACKEND (early reads for the warm decision — happen before Config is consumed) |
# | bridge.py | 1 | BRIDGE_DEBUG (module snapshot, debug surface) |
# | __init__.py | 1 | OPENBLAS_NUM_THREADS (third-party library env, setdefault — must be fixed before numpy import) |

#
# R2 (2026-11-06): new root-level env reads, all startup-topology plumbing
# or CLI pass-through (registered here before the guard count was updated):
# | launcher.py | +1 | ``runtime_identity`` scans _CONFIG_ENV_KEYS to build
#   ``config_stamp``; it runs before config loading and never branches.
# | daemon.py | +1 | ``HIPPOCAMPUS_DAEMON_SAVE_IDLE`` -- timed-flush window,
#   same sentinel class as DAEMON_HEARTBEAT / MODEL_UNLOAD.
# | cli.py | +5 | New ``stop`` / ``restart`` commands recompute the daemon
#   key exactly as launcher.main() does (DAEMON_LIBRARY / EMB_MODEL /
#   DAEMON_VERSION) and pass LIBRARY / MODEL through on restart. Same
#   rationale as the existing ``context`` HELLO pass-through.

#: bool field → semantic type (A/B/C/D, see the comment above). The four
#: pre-migration scattered points each wrote their own rules; at
#: absorption each field's original semantics must be remembered,
#: otherwise boundary values like "HIPPOCAMPUS_RECALL_DECAY=maybe"
#: change from False to True.
#: Type B (allow-list) further splits into wide/narrow lists:
#: recall_decay includes "on", ner_allow_torch does not (as recorded in
#: core.py). _BOOL_NARROW lists the narrow-list fields.
_BOOL_SEMANTICS: dict[str, str] = {
    # A deny-list default true (server.py module level)
    "auto_save": "A", "auto_backfill": "A", "supersede": "A",
    # The two write-side presentation-line knobs are also type A: only
    # explicit 0/false/no/off turns them off, dirty values fall back to
    # default on (encoding gating and prospective triggering are both
    # "on by default, off only when explicitly stated" mechanisms).
    "novelty_gate": "A", "prospective_trigger": "A",
    # R1 production recall fusion (default on, explicit deny only).
    "recall_fusion": "A",
    # R1-B4 MMR dimension-scale fix (default on, explicit deny only --
    # a defect fix, same reasoning as recall_fusion; the golden fixtures
    # cannot discriminate it, so the "pinned baselines" argument for type B
    # does not bind here).
    "mmr_scale_fix": "A",
    # V2 query fanout (default off, explicit allow only — evaluation
    # reproducibility first; promoted by decision after dev-library validation).
    "recall_query_fanout": "B",
    # UR usage-recency (default off, explicit allow only — same rationale
    # as query fanout: pinned evaluation baselines first). The numeric
    # sub-knobs (usage_recency_w / _tau_days / _cap_strength) are coerced
    # by default type and need no entry here; the composition root masks
    # them by ``usage_recency`` so numeric envs alone stay off.
    "usage_recency": "B",
    # Adaptive result-count selection — type A since 2026-11-08 (default
    # ON after the serial AR/UR trial, off only when explicitly stated;
    # same semantics class as profile_promotion). Was type B while the
    # trial ran with pinned baselines. Numeric sub-knobs masked by the
    # bool at the composition root (explicit 0 disables the selector and
    # zeroes every sub-knob).
    "adaptive_recall": "A",
    # Profile promotion is also type A (a consolidation-period mechanism
    # on by default, off only when explicitly stated).
    "profile_promotion": "A",
    # B allow-list (manager/core in-function)
    #   The escape hatch (cross_library_write) is also allow-list:
    #   **only an explicit truthy value opens it**, empty string/dirty
    #   value (maybe) falls to the default False — the boundary must not
    #   be silently disabled by a typoed env.
    "recall_decay": "B", "ner_allow_torch": "B", "cross_library_write": "B",
    # C deny-list default false (server.py _WARMUP_DIAG)
    "warmup_diag": "C",
    # D non-empty and non-0 (manager ENV_ALLOW_MODEL_SWITCH)
    "allow_model_switch": "D",
    # E tri-state (formerly ``common/constants._env_flag``)**:
    #   0/false/no/off off, 1/true/yes/on on, other dirty values fall
    #   back to the field default**. The only difference from type A
    #   shows on inputs like ``X=maybe``: A judges on, E judges default —
    #   the parity probe measured this column live; it is the sole
    #   criterion distinguishing the two semantics.
    "layer_fp": "E", "layer_del": "E", "source_sig": "E",
    "status_contradictions": "E", "map_auto_verify": "E", "map_mention": "E",
    "file_registry_off": "E",
    "overview_block": "E", "status_block": "E", "task_profile_block": "E",
    "map_index_block": "E", "map_detail_block": "E",
    "topic_layer": "E", "topic_low_info": "E",
}
_BOOL_NARROW: set[str] = {"ner_allow_torch"}

#: Special value shapes (bypassing the generic type dispatch). What is
#: registered here is the **semantics**, not the parameter.
#: For ``writer_ttl``'s positive-value gate see the Config field comment.
_POSITIVE_ONLY: set[str] = {"writer_ttl"}


def load_config() -> Config:
    """Read env and produce an immutable config. **Re-read on every call**
    (no caching; rationale in the module docstring)."""
    c = Config()
    values: dict[str, Any] = {}
    for env, attr in ENV_MAP.items():
        default = getattr(c, attr)
        if isinstance(default, bool):
            sem = _BOOL_SEMANTICS.get(attr, "A")
            if sem == "A":
                values[attr] = _flag_deny_true(env, default)
            elif sem == "B":
                values[attr] = _flag_allow_only(
                    env, default, with_on=attr not in _BOOL_NARROW)
            elif sem == "C":
                values[attr] = _flag_deny_false(env, default)
            elif sem == "D":
                values[attr] = _flag_nonempty(env, default)
            else:
                values[attr] = _flag_tristate(env, default)
        elif isinstance(default, int) and not isinstance(default, bool):
            values[attr] = _int(env, default)
        elif attr in _POSITIVE_ONLY:
            values[attr] = _positive_float(env, default)
        elif isinstance(default, float):
            values[attr] = _float(env, default)
        else:
            values[attr] = _str(env, default)
    # recall_decay_tau's status quo semantics: empty string = default
    # 90 days (not 0.0); an explicit 0 is judged "off" by the caller.
    # _float's empty-string fallback already guarantees this.
    return Config(**values)
