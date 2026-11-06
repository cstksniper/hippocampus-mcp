# -*- coding: utf-8 -*-
"""OrganizeService: library-curation use cases (10 tools).

Tool surface: consolidate /
relink_entities / lexicon_add / lexicon_suggest / lexicon_alias_suggest /
map_set / map_get / map_list / map_verify / reset_context.

Every method in this class is **orchestration logic moved verbatim from the
server.py tool bodies** (minus ``_await_warmup`` / brain fetching /
``json.dumps``) — relocated without behavior change; behavioral equivalence is
guaranteed by the full regression gate. Brain method bodies will later be
pulled up into this class, at which point this class's API stays put.

The "who parses the arguments" split (fixed per **the current order of parsing
vs. library resolution in the code**, preserving byte-level behavior):

* The parsing of 9 tools (consolidate's filter_tags, the lexicon trio's
  terms/exclude/cue_words) currently runs **after** library resolution, so it
  moves into the service along with the orchestration — the factory call is
  the library resolution, and parsing inside the method body naturally happens
  after it, keeping the error order unchanged;
* ``map_set``'s ``fields`` is the sole exception: current code parses fields
  first and only then calls ``_brain_for(library)``, and the parse-failure
  error body **carries** the library envelope. Preserving the order (bad
  fields + bad library name ⇒ report the fields error first, without touching
  the library) requires the protocol parsing to stay in the adapter layer; this
  method receives the already-parsed ``fields: dict | None`` — the same
  structural cut as update_tags' four-segment JSON parsing;
* The lexicon trio's terms/exclude/cue_words use lenient parsing
  (``common.params.parse_str_list_lenient``, never errors) and move into the
  service with the orchestration.

Envelopes (the library self-report first key) **keep the current shape per
tool**, not unified:

* **With** envelope: consolidate / relink_entities / lexicon_add / map_set /
  map_verify (currently via ``_with_library``; map_set's fields parse error
  and brain.map_set's ValueError error bodies also carry it — the adapter
  layer and the service assemble with ``_with_library`` / ``_envelope``
  respectively, bit-for-bit identical);
* **Without**: lexicon_suggest / lexicon_alias_suggest / map_get /
  map_list / reset_context (currently a plain ``json.dumps``).
"""
from __future__ import annotations

import json
from typing import Any

from ..common.params import (filter_applied, parse_str_list_lenient,
                             parse_tags_json)
from ..domain.promote_candidates import collect_promote_candidates
from ..domain.sensitive import match_sensitive
from .base import BaseService
from .tags import check_tag_keys


class OrganizeService(BaseService):
    """Library-curation use cases: consolidation / entity relinking / lexicon / info maps / LNN context reset (policy and algorithms temporarily live in brain, to be pulled up later)."""

    # ---------------------------------------------------------------- consolidation

    def consolidate(self, *, filter_tags: str, max_items: int,
                    dry_run: bool) -> dict[str, Any]:
        """Dormancy pruning: mark superseded facts, archive dormant ones, list contradiction candidates (dry_run lists without changing).

        ``filter_tags`` parsing runs after library resolution (moves with the
        service, order preserved); the parse-error body **does not** carry the
        library first key (bit-for-bit identical to the current tool); the
        success path goes through ``_envelope``. Soft-degradation echo: the
        stripped keys must be visible.

        The return body gains a trailing ``promote_candidates`` key (extending
        the return body without changing existing semantics; **never promotes
        automatically** — writes go through the manual two-step of
        ``store_memory(library="user")``). Criteria and annotation are in
        :func:`domain.promote_candidates.collect_promote_candidates`;
        lexicon content / stable-period thresholds are injected by the
        composition root (application does not read env).
        ``brain.consolidate`` (the session_end internal path) does **not**
        carry this key — the candidate list is a gift on the MCP tool surface,
        not part of the session-end hot path.
        """
        brain = self.brain
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            return {"error": f"filter_tags parse failed: {tag_err}",
                    "input": filter_tags[:200]}
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        report = brain.consolidate(filter_tags=parsed_tags, max_items=max_items,
                                   dry_run=dry_run)
        # Promotion candidates. The entity table is assembled fresh at the
        # service layer (the per-library I/O stays in the service layer;
        # domain only receives the inverted dict).
        try:
            entities = brain.entities or {}
            report["promote_candidates"] = collect_promote_candidates(
                list(brain.manager.store.iter_records()),
                summaries=brain.summaries,
                entity_members={name: meta.get("memory_ids") or []
                                for name, meta in entities.items()},
                min_sessions=self.promote_min_sessions,
                sensitive_fn=lambda text: match_sensitive(
                    text, words=self.sensitive_words),
            )
        except Exception as e:  # noqa: BLE001 — the candidate list is a gift; it must not blow up consolidation
            report["promote_candidates"] = []
            report["promote_candidates_error"] = f"{type(e).__name__}: {e}"
        # Soft degradation must be visible — otherwise the stripped keys are silently swallowed.
        if key_info and key_info.get("degraded"):
            report["filter_applied"] = filter_applied(parsed_tags, None, key_info)
        return self._envelope(report)

    # ---------------------------------------------------------------- entity relinking

    def relink_entities(self) -> dict[str, Any]:
        """Re-run entity extraction over the whole library and re-mount (the wrap-up action after lexicon expansion). **With** envelope."""
        return self._envelope(self.brain.relink_entities())

    # ---------------------------------------------------------------- lexicon trio

    def lexicon_add(self, *, terms: str,
                    relink: bool) -> dict[str, Any]:
        """Append proper nouns to the lexicon and hot-reload (including alias lines). **With** envelope.

        ``terms`` uses lenient parsing (accepts both a JSON array and a comma
        string) — currently parsed right after ``_brain_for``; it moves into
        the service with the orchestration, order preserved.
        """
        return self._envelope(self.brain.lexicon_add(
            parse_str_list_lenient(terms), relink=relink))

    def lexicon_suggest(self, *, top_n: int, min_docs: int, min_files: int,
                        min_len: int, max_len: int,
                        exclude: str) -> dict[str, Any]:
        """High-frequency unregistered proper-noun candidates (display only, nothing persisted). **Without** envelope."""
        return self.brain.lexicon_suggest(
            top_n=top_n, min_docs=min_docs, min_files=min_files,
            min_len=min_len, max_len=max_len,
            exclude=parse_str_list_lenient(exclude))

    def lexicon_alias_suggest(self, *, top_n: int, min_confidence: float,
                              min_support: int, min_lift: float,
                              min_containment: float, allow_stat_only: bool,
                              exclude: str, cue_words: str,
                              with_paren: bool) -> dict[str, Any]:
        """Candidates for "the same entity under multiple names" (candidates + evidence only, no automatic merging). **Without** envelope."""
        return self.brain.discover_aliases(
            top_n=top_n, min_confidence=min_confidence, min_support=min_support,
            min_lift=min_lift, min_containment=min_containment,
            allow_stat_only=allow_stat_only,
            exclude=parse_str_list_lenient(exclude),
            cue_words=parse_str_list_lenient(cue_words),
            with_paren=with_paren,
        )

    # ---------------------------------------------------------------- info maps

    def map_set(self, *, name: str, key: str, status: str,
                fields: dict | None, desc: str, truth_source: str, note: str,
                updated_by: str) -> dict[str, Any]:
        """Change a map entry's current state (rewriting the same key replaces it wholesale). **With** envelope.

        The JSON protocol parsing of ``fields`` happens in the **adapter layer**
        (order preserved: current code parses fields before calling
        ``_brain_for``, and the parse error also carries the envelope — see the
        module docstring); this method only receives the parsed dict.
        brain.map_set's ValueError (invalid name/key) error body likewise
        carries the envelope, bit-for-bit identical to the current behavior.
        """
        try:
            out = self.brain.map_set(name, key, status=status, fields=fields,
                                     desc=desc, truth_source=truth_source,
                                     note=note, updated_by=updated_by)
        except ValueError as exc:
            return self._envelope({"status": "error", "error": str(exc)})
        return self._envelope(out)

    def map_get(self, *, name: str, key: str) -> dict[str, Any]:
        """Look up exactly what an entry is now (miss: found=false + reason). **Without** envelope."""
        return self.brain.map_get(name, key)

    def map_list(self, *, name: str) -> dict[str, Any]:
        """List maps (name + entry count + freshness + drift count) or all entries of one map. **Without** envelope."""
        return self.brain.map_list(name=name)

    def map_verify(self, *, name: str) -> dict[str, Any]:
        """Verify map entries against truth-source files (re-read and compare signatures; zero-read causes no false alarm). **With** envelope."""
        return self._envelope(self.brain.map_verify(name=name))

    # ---------------------------------------------------------------- LNN context

    def reset_context(self) -> dict[str, Any]:
        """Reset the LNN temporal context (clears hidden state and the sliding window; deletes no memories). **Without** envelope."""
        self.manager.reset_context()
        return {"status": "context_reset", "message": "LNN temporal context cleared"}
