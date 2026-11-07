# -*- coding: utf-8 -*-
"""WriteService: the single write-path source of truth (9 tools).

Tool surface: store_memory / update_memory /
update_tags / forget_memory / forget_by_query / ingest / ingest_batch /
retry_save / save_state.

Every method here is **orchestration logic moved verbatim from the server.py
tool bodies** (minus ``_await_warmup`` / brain acquisition / ``json.dumps``,
with ``_with_library`` replaced by the base class's ``_envelope``) — relocation
only, no behavior change; behavioral equivalence is guaranteed by the full
regression gate. When brain method bodies are later lifted into this class,
this class's API stays put.

The "who parses the parameters" split (fixed by the **envelope shape of the
current error paths**, to keep byte-level identical outputs):

* Error bodies carrying a ``library`` first key (tags / items parsing of
  ingest / ingest_batch) → parsed inside the service (the envelope lives on
  the service side; adapters don't touch it);
* Error bodies without an envelope (the four update_tags parses, session_start's
  filter_tags parse) → stay in the adapter and must come **before** the factory
  (library parsing) — the current code already does this; reordering would make
  the "bad JSON + bad library name" error body drift.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..common.params import parse_tags_json
from ..domain.long_content import (
    POLICY_WARN,
    char_limit,
    plan_long_content,
    parse_policy,
)
from ..domain.sensitive import classify_sensitivity, match_sensitive
from ..domain.write_quarantine import build_access_event, sensitivity_label
from ..library.access_log import append_access_event
from ..store import ConcurrentWriteError
from ..store.memory import MemoryState
from .base import BaseService

logger = logging.getLogger("hippocampus_mcp")


class WriteService(BaseService):
    """Write use cases: validate → policy → persist → ledger → echo (policy and persistence details still live in brain/manager; to be lifted later)."""

    # ---------------------------------------------------------------- Bare single writes

    # ---------------------------------------------------------------- D2a quarantine

    def _user_lib_label(self, content: str) -> str:
        """D1 labeling when the target is the user library (the user library auto-assigns personal by default)."""
        return sensitivity_label(classify_sensitivity(content or ""))

    def _log_access(self, *, kind: str, tool: str, quarantined: bool,
                    sensitivity: str, content: str,
                    memory_id: str | None = None) -> None:
        """Append a write-side audit event (side path: failures only warn, never affect the main path)."""
        if not self.access_log_dir:
            return
        ev = build_access_event(
            kind=kind, tool=tool,
            home="home" if self.is_home_write else "other",
            target_name=(self.library_self or {}).get("name", ""),
            target_path=(self.library_self or {}).get("path", ""),
            quarantined=quarantined, sensitivity=sensitivity,
            content=content)
        if memory_id:
            ev["memory_id"] = memory_id
        append_access_event(self.access_log_dir, ev)

    def _apply_long_content_policy(self, *, content: str, importance: float,
                                   memory_type: str, client: str,
                                   policy: str,
                                   quarantine: bool = False,
                                   sensitivity: str = "") -> dict[str, Any] | None:
        """Handle over-long content per policy. **Returning None = no intervention**; the original write path proceeds.

        Called only when the user explicitly enabled ``split`` / ``reject``
        (``warn`` skips this whole section, preserving the byte-for-byte
        pre-change behavior).
        """
        mgr = self.manager
        embedder = getattr(mgr, "embedder", None)
        cov_fn = getattr(embedder, "embedding_coverage", None)
        if not callable(cov_fn):
            return None
        try:
            limit = int(getattr(embedder, "max_seq_length", 0) or 0)
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0:
            # Cannot get the limit (hash backend / missing metadata) ⇒ nothing
            # to judge against; don't intervene.
            return None
        # Character count not exceeding the token limit ⇒ no language can
        # exceed the window (Chinese: 1 char ≈ 1 token, English: 1 token ≈ 4
        # chars). Keeps short cards at zero extra overhead.
        if len(content or "") <= limit:
            return None
        try:
            cov = cov_fn(content or "") or {}
        except Exception:  # self-reporting must never break the write itself (same stance as the truncation self-report)
            return None

        plan = plan_long_content(
            content or "", limit, policy,
            truncated=bool(cov.get("truncated")),
            split_facts=self.split_facts)
        if plan.action == "write":
            return None

        if plan.action == "reject":
            return self._envelope({
                "status": "rejected_long_content",
                "memory_id": None,
                "error": plan.reason,
                "long_content_policy": policy,
                "content_chars": len(content or ""),
                "char_limit": char_limit(limit),
                "content_preview": (content or "")[:100],
            })

        ids: list[str] = []
        for piece in plan.pieces:
            ids.append(mgr.store_memory(
                content=piece, importance=importance,
                memory_type=memory_type, client=client,
                # D2a: split products inherit the quarantine flag — splitting
                # must not become a quarantine bypass.
                quarantine=quarantine, sensitivity=sensitivity))
        out: dict[str, Any] = {
            "status": "stored_split",
            "memory_id": ids[0] if ids else None,  # back-compat with the old single-item echo field
            "memory_ids": ids,
            "piece_count": len(ids),
            "long_content_policy": policy,
            "long_content_reason": plan.reason,
            "content_chars": len(content or ""),
            "char_limit": char_limit(limit),
            "content_preview": (content or "")[:100],
        }
        if plan.note:
            out["long_content_note"] = plan.note
        if quarantine:
            # Same stance as the single-item path: the quarantine self-report
            # is never missing (splits also enter the review queue).
            out["quarantined"] = True
            out["quarantine_note"] = (
                "All cards produced by this split entered the user-library review queue: they are excluded from recall and "
                "the user_scope slot. Review them in drift_audit, then promote or delete.")
        if sensitivity:
            # D1: self-report the label (split products match the single-item
            # path; home-library writes carry it too).
            out["sensitivity"] = sensitivity
        return self._envelope(out)

    def store_memory(self, *, content: str, importance: float,
                     memory_type: str, client: str,
                     origin: dict[str, str] | None = None,
                     idem_key: str = "") -> dict[str, Any]:
        """Bare write of a single fact (no L0 provenance, no entities; long content goes through :meth:`ingest`).

        Sensitive-filter red line: when the target library is named ``user``
        (the user-level library's convention name), a **dedicated sensitive
        filter** applies (closing the direct-write bypass) — a hit means the
        write is rejected, and the response self-reports the hit words and
        positions (never silent). **Existing-library behavior unchanged**:
        non-user libraries see zero branching and zero I/O. The word list is
        injected by the composition root (application does not read env); a
        mode-level built-in fallback lives in domain (effective even when the
        word list is absent). The factory layer's home-write guard composes
        **orthogonally** with this filter — the guard must not open another
        write path that bypasses this filter.
        """
        if (self.library_self or {}).get("name") == "user":
            hits = match_sensitive(content or "", words=self.sensitive_words)
            if hits:
                return self._envelope({
                    "status": "blocked_sensitive",
                    "memory_id": None,
                    "hits": hits,
                    "content_preview": (content or "")[:100],
                    "note": ("The user-level library rejects sensitive content. See "
                             "hits for details; rewrite and resend, or use the two-step manual promotion flow."),
                })
        # D2a/D1: target = user library ⇒ label (the user library auto-assigns
        # personal by default, better safe than sorry); non-home workspace
        # (factory-injected quarantine criterion true) ⇒ quarantine flag
        # + audit event. Non-user library targets: zero branching, zero overhead.
        quarantined = bool(self.user_write_quarantine)
        label = ""
        if (self.library_self or {}).get("name") == "user":
            label = self._user_lib_label(content)
        # Over-long content policy. **warn (default) skips this whole
        # section** — the current behavior is "write first, self-report
        # after"; computing coverage up front would pay for the tokenizer one
        # extra time, and there is no reason to change a default path everyone
        # depends on. Only an explicit split/reject takes the branch below
        # (whitelist semantics: a mistyped value is the same as unset).
        policy = parse_policy(self.long_content_policy)
        if policy != POLICY_WARN:
            decided = self._apply_long_content_policy(
                content=content, importance=importance,
                memory_type=memory_type, client=client, policy=policy,
                quarantine=quarantined, sensitivity=label)
            if decided is not None:
                if quarantined or label:
                    self._log_access(kind="store_memory", tool="store_memory",
                                     quarantined=quarantined, sensitivity=label,
                                     content=content)
                return decided

        mgr = self.manager
        memory_id = mgr.store_memory(
            content=content,
            importance=importance,
            memory_type=memory_type,
            # client wiring (a lesson measured on daemon multi-session): without
            # client, the LNN hidden state always lands on the __default__ lane
            # — irrelevant under stdio (one process, one session), but under
            # the daemon N connections all crowd onto one lane, and A's writes
            # directly rewrite B's retrieval starting point, with no exception
            # and no log — only degraded temporal scores.
            client=client,
            quarantine=quarantined,
            sensitivity=label,
            origin=origin,
            idem_key=idem_key,
        )
        # P0-2 idempotent hit: the manager returned the existing id before
        # embedding (skipping the most expensive step), so last_truncation is
        # the **previous call's** leftover value — never use it to build the
        # response, otherwise the host sees a truncation warning unrelated to
        # this call. Take the minimal-receipt path here.
        if getattr(mgr, "last_idem_hit", None):
            return self._envelope({
                "status": "duplicate",
                "idempotent": True,
                "memory_id": memory_id,
                "content_preview": content[:100],
                "note": ("Replay of the same idem_key: nothing new was created or embedded; the existing "
                         "memory id was returned. If this was actually new content, the host reused the key."),
            })
        if quarantined or label:
            self._log_access(kind="store_memory", tool="store_memory",
                             quarantined=quarantined, sensitivity=label,
                             content=content, memory_id=memory_id)
        # Truncation self-report. The embedding model has a token **hard limit**
        # (BGE / MiniLM = 512); the excess is silently dropped at encode time
        # and contributes nothing to the vector — the write succeeds, the
        # record is in the library, brain_stats looks normal, and **only recall
        # quietly falls short**. This puts "how many characters actually
        # entered the embedding" up front in the response, so the host sees it
        # immediately instead of discovering it when recall misses.
        tcov = getattr(mgr, "last_truncation", None) or {}
        truncated = bool(tcov.get("truncated"))
        out: dict[str, Any] = {
            "status": "stored",
            "memory_id": memory_id,
            "truncated": truncated,
            "max_seq_length": tcov.get("max_seq_length", 0),
            "embedded_chars": tcov.get("embedded_chars", len(content)),
            "content_chars": tcov.get("total_chars", len(content)),
            "content_preview": content[:100],
        }
        if truncated:
            out["truncation_warning"] = (
                f"This content is {out['content_chars']} characters, above the embedding model token limit of "
                f"{out['max_seq_length']}; only the first {out['embedded_chars']} characters entered the vector -- "
                f"the remainder was written but is not retrievable. Split long content into cards under 500 characters."
            )
        if quarantined:
            # Decision ①: the write response self-reports the quarantined state
            # (never silent) — "written successfully yet invisible to every
            # downstream channel" must be visible on the spot, otherwise the
            # host believes the memory took effect.
            out["quarantined"] = True
            out["quarantine_note"] = (
                "This item entered the user-library review queue (non-home direct write through the exemption channel): "
                "it is excluded from recall and the user_scope slot. Have the user review it in drift_audit, "
                "then promote or delete through the managed channel.")
        if label:
            # D1: self-report the label (home-library writes carry it too — the
            # host should know what label a card received).
            out["sensitivity"] = label
        return self._envelope(out)

    # ---------------------------------------------------------------- Field updates

    def update_memory(self, *, memory_id: str, content: str,
                      importance: float, memory_type: str, state: str,
                      superseded_by: str,
                      clear_quarantine: bool = False) -> dict[str, Any]:
        """Update only the provided fields, leaving the rest unchanged (the "change to X" channel).

        ``clear_quarantine`` (D2a managed promotion): promote a pending-review
        card in the user library. **The privilege key is not in the agent's
        hands**: open only to home-library sessions (or when the escape hatch
        is open — the factory counts both cases into ``is_home_write``); a
        request from a non-home session = explicit error, never silently
        ignored. The promotion action itself also goes into access_log (who
        promoted what stays auditable).
        """
        mgr = self.manager
        if clear_quarantine:
            if not self.is_home_write:
                return self._envelope({
                    "status": "error",
                    "memory_id": memory_id,
                    "error": ("clear_quarantine is allowed only from a home-library session: promotion "
                              "is a managed action. Operate from a home workspace for the user library (daily assistant "
                              "host), or enable the escape-hatch env for operations."),
                })
            rec = mgr.store.get(memory_id)
            if rec is None:
                return self._envelope({"status": "not_found",
                                       "memory_id": memory_id})
            if not getattr(rec, "quarantine", False):
                return self._envelope({
                    "status": "noop", "memory_id": memory_id,
                    "note": "This memory is not in the review queue (quarantine=False); no promotion needed."})
            mgr.update_memory(memory_id, quarantine=False)
            self._log_access(kind="quarantine_clear", tool="update_memory",
                             quarantined=False, sensitivity="",
                             content=str(getattr(rec, "content", "")),
                             memory_id=memory_id)
            return self._envelope({
                "status": "quarantine_cleared", "memory_id": memory_id,
                "note": "Promoted: this memory is restored to recall and the user_scope slot. "
                        "The promotion was recorded in access_log."})
        fields: dict[str, Any] = {}
        warnings: list[str] = []
        if content:
            fields["content"] = content
        if importance >= 0:
            fields["importance"] = importance
        if memory_type:
            fields["memory_type"] = memory_type

        if state:
            allowed = {s.value for s in MemoryState}
            st = state.strip().lower()
            if st not in allowed:
                return self._envelope(
                    {"status": "error", "memory_id": memory_id,
                     "error": f"Invalid state: {state!r}",
                     "allowed_states": sorted(allowed)})
            fields["state"] = st
            if st == MemoryState.SUPERSEDED.value and not superseded_by:
                warnings.append(
                    "Marked superseded without superseded_by; later traceability to the replacing record is lost"
                )

        if superseded_by:
            if mgr.store.get(superseded_by) is None:
                return self._envelope(
                    {"status": "error", "memory_id": memory_id,
                     "error": f"superseded_by points to a missing memory: {superseded_by}"})
            fields["superseded_by"] = superseded_by
            if not state:
                # Only superseded_by given without state: complete it by
                # semantics rather than leaving the self-contradictory state of
                # "a pointer exists but the state is still active".
                fields["state"] = MemoryState.SUPERSEDED.value
                warnings.append("state was omitted; inferred as superseded from the superseded_by semantics")

        if not fields:
            # Note: this error body does **not** carry a library first key
            # (bit-identical with the current tool).
            return {"error": "no fields to update"}

        success = mgr.update_memory(memory_id, **fields)
        payload: dict[str, Any] = {
            "status": "updated" if success else "not_found",
            "memory_id": memory_id,
            "updated_fields": sorted(fields),
        }
        if warnings:
            payload["warnings"] = warnings
        # Correction-chain hint (naturalization P3: result-envelope teaching,
        # zero standing cost — the resident docstring was slimmed on the
        # promise this note carries the routing). Only on success AND when
        # the update looks like a fact correction (content touched) or a
        # supersede action — pure importance/tag tweaks stay quiet.
        if success and ("content" in fields or "state" in fields):
            payload.setdefault("note",
                               "Correcting a fact: write the new version "
                               "(ingest), then retire this one traceably with "
                               "state='superseded' and superseded_by=<new id>.")
        return self._envelope(payload)

    # ---------------------------------------------------------------- Bulk tags

    def update_tags(self, *, tags: dict[str, Any] | None, clear_keys: list[str] | None,
                    source: str, source_contains: str, contains: str, state: str,
                    untagged_only: bool, missing_keys: list[str] | None,
                    memory_ids: list[str] | None, mode: str, dry_run: bool,
                    limit: int, confirm: bool, undo: str) -> dict[str, Any]:
        """Backfill tags on existing records (dry_run by default; undoable, fail-closed).

        JSON parsing of the inputs happens in the adapter layer (its error body
        carries no envelope; see the module docstring); this receives parsed
        values. dry_run / confirm gates / idempotency / ledger semantics all
        live in ``brain.update_tags`` (to be lifted together with the method
        body later).
        """
        return self._envelope(self.brain.update_tags(
            tags=tags, clear_keys=clear_keys,
            source=source, source_contains=source_contains, contains=contains,
            state=state, untagged_only=untagged_only, missing_keys=missing_keys,
            memory_ids=memory_ids, mode=mode, dry_run=dry_run, limit=limit,
            confirm=confirm, undo=undo,
        ))

    # ---------------------------------------------------------------- Deletion

    def forget_memory(self, *, memory_id: str, reason: str) -> dict[str, Any]:
        """Cascade delete by ID (clear entity inverted index / L3 derivatives / tombstones / restore superseded records)."""
        return self._envelope(self.brain.forget(memory_id, reason=reason))

    def forget_by_query(self, *, query: str, top_k: int, dry_run: bool,
                        min_similarity: float, allow_multiple: bool) -> dict[str, Any]:
        """Locate suspected wrong memories by natural language and delete them (preview by default, single delete by default)."""
        return self._envelope(self.brain.forget_by_query(
            query, top_k=top_k, dry_run=dry_run, allow_multiple=allow_multiple,
            min_similarity=None if min_similarity < 0 else min_similarity,
        ))

    # ---------------------------------------------------------------- Ingest

    def ingest(self, *, text: str, speaker: str, importance: float,
               cognitive_type: str, entities: str, source: str,
               tags: str, auto_tags: bool,
               origin: dict[str, str] | None = None,
               idem_key: str = "") -> dict[str, Any]:
        """Recommended write entry: raw text into L0, extracted sentences into L1, entity index maintained.

        Split against the tool layer: ``tags`` parse errors carry a library
        first key, so parsing happens inside the service (keeping byte-level
        identical outputs); ``entities`` / ``importance`` / ``cognitive_type``
        sentinel normalization has no error path and sits here with the
        orchestration.
        """
        logger.info("[ingest] start: %s", text[:40])
        parsed_tags, tag_err = parse_tags_json(tags)
        if tag_err:
            return self._envelope(
                {"error": f"tags parse failed: {tag_err}", "input": tags[:200]})
        # D2a/D1: target = user library ⇒ label the whole batch's L1 records
        # (classify the raw text once, shared within the batch, better safe
        # than sorry); non-home workspace ⇒ the whole batch's L1 records carry
        # the quarantine flag + audit event.
        quarantined = bool(self.user_write_quarantine)
        label = ""
        if (self.library_self or {}).get("name") == "user":
            label = self._user_lib_label(text)
        result = self.brain.ingest(
            text=text,
            speaker=speaker,
            importance=None if importance < 0 else importance,
            cognitive_type=cognitive_type or None,
            entities=[e.strip() for e in entities.split(",") if e.strip()] or None,
            source=source,
            tags=parsed_tags,
            auto_tags=auto_tags,
            quarantine=quarantined,
            sensitivity=label,
            origin=origin,
            idem_key=idem_key,
        )
        if quarantined or label:
            self._log_access(kind="ingest", tool="ingest",
                             quarantined=quarantined, sensitivity=label,
                             content=text)
            if quarantined:
                result["quarantined"] = True
                result["quarantine_note"] = (
                    "All %d L1 records in this batch entered the user-library review queue (non-home direct "
                    "write through the exemption channel): they are excluded from recall and the user_scope slot. "
                    "Review them in drift_audit, then promote or delete through the managed channel."
                    % len(result.get("memory_ids", [])))
            if label:
                # D1: self-report the label (home-library writes carry it too —
                # the host should know what label a card received).
                result["sensitivity"] = label
        logger.info("[ingest] done: %d memories", len(result.get("memory_ids", [])))
        # When auto-save is rejected by concurrency protection, warn on the
        # spot: seeing "written", the host believes it persisted, when in fact
        # this batch lives only in memory and is lost on process restart
        # (previously completely silent here — zcode feedback).
        save_err = self.manager.last_save_error
        if save_err:
            result["save_warning"] = (
                f"Auto-save failed; this batch remains memory-only: {save_err}。"
                f"Call save_state; if it reports conflict, run load_state or retry_save first."
            )
            logger.warning("[ingest] %s", result["save_warning"])
        return self._envelope(result)

    def ingest_batch(self, *, items: Any, source: str, tags: str,
                     granularity: str, detect_supersede: bool, save: bool,
                     auto_tags: bool,
                     origin: dict[str, str] | None = None) -> dict[str, Any]:
        """Batch write (per-item error isolation + a single persist at batch end).

        ``items`` JSON parse errors carry a library first key, so parsing
        happens inside the service (same rationale as :meth:`ingest`);
        ``items`` is annotated ``Any``: the tool signature says ``str``, but
        historically hosts passing arrays directly also reached this (the
        defensive branch is kept).
        """
        parsed_tags, tag_err = parse_tags_json(tags)
        if tag_err:
            return self._envelope(
                {"error": f"tags parse failed: {tag_err}", "input": tags[:200]})
        if isinstance(items, str):
            s = items.strip()
            if not s:
                return self._envelope(
                    {"error": "items is empty; pass a JSON array string such as [{\"text\":\"...\"}]"})
            try:
                parsed_items = json.loads(s)
            except Exception as e:  # noqa: BLE001
                # A parse failure must raise explicitly — silently treating it
                # as empty would make the host believe the whole batch was
                # written.
                return self._envelope(
                    {"error": f"items JSON parse failed: {e}", "input": s[:200]})
        else:
            parsed_items = items
        try:
            # D2a/D1: same stance as single-item ingest (labels classify the
            # merged raw text of all items in the batch once; quarantine is a
            # batch-level policy, see the injection-point comment in
            # brain.ingest_batch).
            quarantined = bool(self.user_write_quarantine)
            label = ""
            if (self.library_self or {}).get("name") == "user":
                label = self._user_lib_label(json.dumps(
                    parsed_items, ensure_ascii=False)[:2000])
            result = self.brain.ingest_batch(
                items=parsed_items,
                source=source,
                tags=parsed_tags,
                granularity=granularity,
                detect_supersede=detect_supersede,
                auto_tags=auto_tags,
                save=save,
                quarantine=quarantined,
                sensitivity=label,
                origin=origin,
            )
        except ValueError as e:
            # Non-array / over the per-batch cap: return an error directly;
            # don't let the host believe the write succeeded.
            return self._envelope({"error": str(e)})
        if quarantined or label:
            self._log_access(kind="ingest_batch", tool="ingest_batch",
                             quarantined=quarantined, sensitivity=label,
                             content=json.dumps(parsed_items,
                                                ensure_ascii=False)[:2000])
            if quarantined:
                result["quarantined"] = True
                result["quarantine_note"] = (
                    "All items in this batch entered the user-library review queue: they are excluded from recall and "
                    "the user_scope slot. Review them in drift_audit, then "
                    "promote or delete through the managed channel.")
        if label:
            # D1: self-report the label (home-library writes carry it too — the
            # host should know what label a card received).
            result["sensitivity"] = label
        logger.info("[ingest_batch] done: %d/%d items, %d failed",
                    result["count"], result["total_input"], result["failed"])
        return self._envelope(result)

    # ---------------------------------------------------------------- Persistence

    def save_state(self) -> dict[str, Any]:
        """Full persistence (vector store + LNN + L0–L3 + entity index)."""
        try:
            self.brain.save()
        except ConcurrentWriteError as e:
            return self._envelope(
                {"status": "conflict", "detail": str(e),
                 "hint": "Another process rewrote the same memory library. Run load_state to get the latest state, then redo this change."})
        return self._envelope(
            {"status": "saved", "persist_dir": self.persist_dir})

    def retry_save(self) -> dict[str, Any]:
        """Retry after a concurrency conflict: take the latest on-disk version and fill in only "the increments the disk doesn't have"."""
        try:
            result = self.brain.retry_save()
        except ConcurrentWriteError as e:
            return self._envelope(
                {"status": "conflict", "detail": str(e),
                 "hint": "Conflict persists: stop the other writer (close the other MCP session), then retry."})
        return self._envelope({"status": "retried", **result})
