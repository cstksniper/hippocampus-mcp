# -*- coding: utf-8 -*-
"""Assembly of the ``brain.core.stats`` echo body (extracted from core).

**Why this is worth moving**: ``stats`` is the self-inspection entry point every host
must call; besides two side-effect calls (lexicon heat sensing / writer heartbeat) and
the six projections already extracted, the remaining return dict is ~120 lines,
45 keys, each key carrying a design-rationale comment — pure "key-order-sensitive echo
body assembly", exactly the same shape as knowledge_map_view's assemble_report.

**Dependency direction**: this module has zero imports — all values are computed by the
shell and passed in: brain-side capabilities such as ``tag_coverage`` /
``_recall_feedback_stats`` / ``_pin_stats`` / ``knowledge_stats`` /
``lexical_self_report`` / ``save_ledger_view`` / ``entity_origin_stats`` /
``tag_extractor.describe`` are injected by value; the two ``load_config`` switches
(layer_del / source_sig) and the six projections (state_counts / dangling_report
/ daily_activity / source_sig_block / writers_block / forgotten_count) are also
fetched fresh by the shell (invariant: each of the six calls happens exactly once in
the shell).

**Replicated quirks (changing any one is a behavior change; parity will catch it)**:

1. Fixed order of the 45 keys (key order is signal — the JSON echo preserves order, and
   host and tests consume it in order);
2. ``layer_del`` is a nested sub-dict: the four scalars are joined in place at the
   assembly site, key order enabled / tombstones_n / blocked_last / gc_last;
3. ``source_sig`` is enabled first + ``**block`` (key order = enabled /
   recorded / referenced; order is signal);
4. ``vectors_by_model`` projects ``v["vectors"]`` from the injected model_stats;
5. The fallback copy of ``embedder_model_source``, 「n/a（该后端不经模型名解析）」,
   is a contract value (the computation stays in the shell: test_runtime_selfreport C1
   pins by **behavior** that unknown is not lied about; test_f6 F3 pins the
   shared-embedder key by the source containing the ``"embedder_shared"`` literal —
   that key's computation likewise stays in the shell, with the comment anchors moved
   along with it).
"""
from __future__ import annotations

from typing import Any


def assemble_report(
    *, l0_raw: int, l1_memories: int, tags: dict[str, Any],
    l0_dangling_source_ids: int, l0_dangling_memories: int,
    daily_activity: dict[str, Any], l1_by_state: dict[str, Any],
    recall_feedback: dict[str, Any], pin_stats: dict[str, Any],
    l2_summaries: int, topic_layer: dict[str, Any], l3_profile_slots: int,
    entities: int, sessions: int, open_intents: int,
    layer_merge_last: dict[str, Any], layer_del_enabled: bool,
    layer_del_tombstones_n: int, layer_del_blocked_last: int,
    layer_del_gc_last: Any, source_sig_enabled: bool,
    source_sig_block: dict[str, Any], writers: dict[str, Any],
    open_topics: int, pinned: int, tombstones: int, raw_forgotten: int,
    embed_model: str, embedder_backend: str, embedder_runtime: str,
    onnx_attached: bool, lnn_runtime: Any, pending_consolidate: bool,
    embedding_dim: Any, embedder_model_source: str, embedder_shared: bool,
    embedder_pool: dict[str, Any], ner_backend: str,
    lexical: dict[str, Any], model_stats: dict[str, Any],
    vectors_missing: int, supersede_enabled: bool, entity_lexicon_size: int,
    entity_aliases: int, tag_schema: dict[str, Any],
    entity_origins: dict[str, Any], sources: int,
    save_ledger: dict[str, Any], l0_pending: int,
) -> dict[str, Any]:
    """Assemble the stats echo body (fixed order of 45 keys; key order is signal).

    The shell passes arguments in the evaluation order of the old dict literal (kwargs
    evaluate left to right); this function only assembles: the four scalars are joined
    into the nested ``layer_del`` dict, enabled + ``**block`` into ``source_sig``,
    ``vectors_by_model`` is projected from model_stats, and the rest land key by key
    as-is — the rationale comment above each key moved here with the key, not a word
    deleted.
    """
    return {
        "L0_raw": l0_raw,
        "L1_memories": l1_memories,
        # Tag coverage / key distribution: the self-report prerequisite of B1 "finding
        # things by domain"
        "tags": tags,
        # Number of ids referenced by L1 yet absent from L0 / the number of affected L1
        # entries. Normally 0.
        "l0_dangling_source_ids": l0_dangling_source_ids,
        "l0_dangling_memories": l0_dangling_memories,
        "daily_activity": daily_activity,
        "L1_by_state": l1_by_state,
        # Recall-feedback coverage. **Self-report rather than backfill** — when coverage
        # is low, archival conclusions like "how long since last used" become distorted;
        # this makes it visible to the host.
        "recall_feedback": recall_feedback,
        # pins only enter, never leave; once they pile up past a certain ratio they will
        # crowd out the usage layer's dynamic slots
        "pin_stats": pin_stats,
        "L2_summaries": l2_summaries,
        "topic_layer": topic_layer,
        "L3_profile_slots": l3_profile_slots,
        "entities": entities,
        "sessions": sessions,
        "open_intents": open_intents,
        # Number of entries merged back from disk at the last save (per layer).
        # Non-zero means another writer really is touching this library.
        "layer_merge_last": layer_merge_last,
        # Deletion-ledger self-report.
        # tombstones_n = how many tombstones still stand on this ledger; blocked_last =
        # how many late stale entries the last save intercepted (>0 = someone really is
        # reviving things from an old snapshot).
        # "Did interception take effect" must have a number; it cannot rest on
        # "I think it should have worked".
        "layer_del": {
            "enabled": layer_del_enabled,
            "tombstones_n": layer_del_tombstones_n,
            "blocked_last": layer_del_blocked_last,
            "gc_last": layer_del_gc_last,
        },
        # Source-signature layer self-report. recorded = number of sources whose signature
        # has been collected; referenced = deduplicated count of memories in the library
        # carrying a source tag. The difference between the two is the pool of
        # unbaselined/orphan in drift_audit — the "collection coverage"
        #must be visible, otherwise "no drift" and "not collected" cannot be told apart.
        "source_sig": {
            # key order = the old implementation's enabled/recorded/referenced (order is signal)
            "enabled": source_sig_enabled,
            **source_sig_block,
        },
        # Making co-library writers visible. active>1 means someone really is contending
        # with you for this library.
        "writers": writers,
        "open_topics": open_topics,
        "pinned": pinned,
        "tombstones": tombstones,
        "raw_forgotten": raw_forgotten,
        "embed_model": embed_model,
        # ---- Runtime self-description: lets the user verify "what is actually in effect"
        # instead of guessing ----
        # Embedder backend (sentence_transformer / hash / …). The hash backend is the
        # zero-dependency quick-verification path with weak semantic capability — running
        # benchmarks should not silently land on it.
        "embedder_backend": embedder_backend,
        # The **actual** inference backend (onnx / torch / hash). Difference from the line
        # above: the line above is "what is configured", this one is "what is actually
        # running". When artifacts are missing, ONNX silently falls back to torch; looking
        # only at the config will never reveal it (verified on a real library).
        "embedder_runtime": embedder_runtime,
        # onnx comes in two forms (an ST wrapper holding _onnx / the embedder itself being
        # a directly constructed OnnxEmbedder); the manager's runtime is the single arbiter;
        # lnn_runtime self-reports the CfC actual runtime (numpy = zero torch).
        "onnx_attached": onnx_attached,
        "lnn_runtime": lnn_runtime,
        # Session-split self-report: True while some sessions have not been closed out
        # (global consolidation deferred)
        "pending_consolidate": pending_consolidate,
        # Actual vector dimension (decided by the model, not a constructor parameter)
        "embedding_dim": embedding_dim,
        # The **source** of the model name: parameter / env / default. This is the only
        # detector for incidents like "silently falling back to MiniLM" — when the source
        # is default and the user believed a Chinese model was configured, one glance here
        # reveals it.
        # When it cannot be obtained, the backend does not resolve via a model name (e.g.
        # hash); in that case "source" is meaningless — write that out explicitly, and do
        # not let unknown be misread as "resolution failed".
        "embedder_model_source": embedder_model_source,
        # Embedder-sharing self-report: whether this library's embedder reuses the shared
        # in-process heavy object. With two libraries on the same model the second library
        # should be True; False means it loaded its own copy of the weights.
        "embedder_shared": embedder_shared,
        # Current state of the process-level shared pool: how many copies of onnx_size /
        # torch_size each. With two libraries on the same model this should be 1; seeing 2
        # means the second library loaded another copy of the weights.
        "embedder_pool": embedder_pool,
        # English entity-extraction backend. When spaCy cannot be installed, this carries
        # the failure reason instead of showing only "heuristic-only" and making people
        # think it is by design. Uses the **side-effect-free** version — the startup log
        # goes through stats(); if this used ner_backend it would force
        # spaCy→thinc→torch (+160MB resident) into the process at startup.
        "ner_backend": ner_backend,
        # ---- Literal-channel self-report (the acceptance point of the literal-channel
        # optimization) ----
        # The "sqlite+FTS5 ⇒ no in-memory BM25 ⇒ saves 34.5MB" optimization must have a
        # production-side observation entry point, otherwise "did it actually save" can
        # only be guessed. This section provides the backend, FTS5 availability, in-memory
        # index entry count, and the path taken last time, all at once.
        "lexical": lexical,
        "vectors_by_model": {
            k: v["vectors"] for k, v in model_stats.items()
        },
        "vectors_missing": vectors_missing,
        # Hard coverage self-check rule: it must be 0,
        # otherwise the in-library vectors do not match the current model and retrieval
        # will be distorted.
        "supersede_enabled": supersede_enabled,
        "entity_lexicon_size": entity_lexicon_size,
        # Number of alias entries: at 0, "multiple names for the same entity" will not be
        # merged (both read-side merging and query-side normalization depend on it).
        # Hosts verifying "are my configured aliases actually in effect" look here.
        "entity_aliases": entity_aliases,
        # Self-description of the effective tag-extraction rules: the key names are the
        # **caller contract**; they must be exposed for both ends to cross-check
        # (mismatched key names = filtering silently fails, with no error).
        "tag_schema": tag_schema,
        # Entity-origin distribution: only trusted (manual/lexicon/spacy) origins
        # participate in profile and retrieval boosts; heuristic is the candidate pool
        # (the main noise source for Chinese documents, ~84%). See ENTITY_ORIGIN_*.
        "entity_origins": entity_origins,
        "sources": sources,
        # Save-ledger projection — turns "when was the last save, how long since persisted
        # to disk" into one line.
        "save_ledger": save_ledger,
        # Number of outstanding raw entries. Invisible means nonexistent — this pit
        # dragged on for two months precisely because "L1 present, L0 absent" was
        # nowhere visible
        "l0_pending": l0_pending,
    }
