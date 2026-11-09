# -*- coding: utf-8 -*-
"""RecallService: read-side use cases (11 tools).

Tool surface: recall / get_memory /
get_memory_context / list_memories / list_sources / contradictions /
drift_audit / knowledge_map / knowledge_topic / entity_profile /
entity_list.

Every method of this class is **orchestration logic moved verbatim from
the server.py tool bodies** (minus ``_await_warmup`` / brain fetching /
``json.dumps``) — relocation only, no behavior change; behavioral
equivalence is guaranteed by the full regression gate. When brain method
bodies are lifted into this class later, this class's API stays put.

The "who parses the arguments" split (decided by the **order of parsing
vs library resolution in the current code**, to keep byte-level parity):

* All 11 tools parse JSON **after** library resolution (current code runs
  ``_brain_for(library)`` first, then parses), so parsing moves into the
  service along with the orchestration — the factory call is the library
  resolution, and parsing inside the method body naturally comes after
  it; the error order is unchanged;
* Parse-failure error bodies **carry no** ``library`` first key (recall /
  get_memory_context / contradictions' filter_tags, knowledge_topic's
  items/fras); inside the service they return as bare dicts, bit-for-bit
  identical to the current tools;
* knowledge_topic's ``evidence`` / ``topics`` go through lenient parsing
  (``common.params.parse_str_list_lenient``, never raises) and move into
  the service with the orchestration.

Envelopes (library self-report first key) **keep the current shape per
tool**, no unification — this layer is a behavior-preserving migration;
"adding a library first key to the read-side tools" is a product decision,
not a migration side effect:

* **With** envelope: knowledge_map / knowledge_topic (currently via
  ``_with_library``);
* **Without**: the other 9 (currently straight ``json.dumps``, including
  all error paths).
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
from typing import Any

from ..common.constants import (ENTITY_BONUS, MMR_LITERAL_EXEMPT_TOPN,
                                RELEVANCE_HARD_FLOOR, _now,
                                strip_internal_fields)
from ..common.params import (filter_applied, parse_str_list_lenient,
                             parse_tags_json)
from ..common.recency import (in_time_window, iso_date, out_of_window,
                              parse_date_bound, weekday_abbr)
from ..retrieval import recall_pipeline as _recall_pipeline
from ..retrieval import query_variants
from ..retrieval.bm25 import rrf_fuse
from ..store.memory import MemoryState
from ..domain import metacog as _metacog
from ..domain.context_slots import BLOCK_PLACEHOLDERS
from ..domain import prospective as _prospective
from ..domain.scope_recall import (SCOPE_ALL, SCOPE_PROJECT, SCOPE_USER,
                                   merge_multi_library, merge_scope_hits,
                                   parse_library_spec, validate_scope)
from ..domain.write_quarantine import build_access_event
from ..library import resolve_library, scan_libraries
from ..library.access_log import append_access_event, read_access_events
from ..library.task_directory import load_directory
from .base import BaseService
from .tags import check_tag_keys

logger = logging.getLogger("hippocampus_mcp")

#: D2b: the read-denial message for none-tier access to the user library
#: (single source of truth, shared by multiple tools).
USER_READ_DENIED = (
    "This workspace has none-level read access to the user library (not registered/bound; fail-closed)."
    "Register the task directory from a session in this workspace first (library_create with workspace_root), "
    "or operate from a home workspace (daily assistant host) session.")


#: R1: the MCP ``recall`` envelope exposes exactly these 14 per-item
#: keys. The shared pipeline returns 17 (it also carries
#: ``source_ids`` / ``entity_ids`` / ``pin`` / ``matched_by`` /
#: ``literal``) and lacks ``recall_count`` / ``memory_type``;
#: :meth:`_to_mcp_item` adapts both ways so the host sees the same
#: field surface as the legacy ``RecallResult.to_dict()`` path.
_MCP_RECALL_ITEM_ORDER = (
    "memory_id", "content", "score", "similarity", "timestamp",
    "occurred_at", "weekday", "importance", "recall_count",
    "memory_type", "source", "state", "cognitive_type", "tags",
)


class RecallService(BaseService):
    """Read use cases: retrieval / single item / listing / contradictions / drift / entities / knowledge map (policies and algorithms stay in brain/manager for now, to be lifted later)."""

    # ---------------------------------------------------------------- semantic retrieval

    def recall(self, *, query: str, top_k: int, use_lnn: bool, sample: bool,
               filter_tags: str, client: str,
               scope: str = SCOPE_PROJECT,
               libraries: str = "",
               occurred_after: str = "",
               occurred_before: str = "") -> dict[str, Any]:
        """Vector + BM25(RRF) + MMR + entity bonus + hard floor (superseded/archived filtered by default).

        ``client`` is passed by the adapter layer as ``session_key()`` (reads
        also ride this session's LNN state — the same wiring as
        store_memory).

        The three scope values (enumerated, not generalized — deliberately
        narrow):

        * ``project`` (default): single-library flow; the response carries no
          scope/libraries keys (simplest possible shape);
        * ``user``: the whole flow runs on the user-level library brain
          (injected by the composition root); on success the body gains
          ``scope`` + ``libraries`` self-report; bookkeeping (miss/
          bump_daily) goes to the user library by intent, stamp_recalled is
          stamped as usual;
        * ``all``: project flow first in full, user flow behind it (hard
          segmentation; scores are not comparable across libraries and are
          not fused), ``top_k`` is the **total**; each library stamps its
          own; miss/ bump_daily go to the project library (the query context
          is the project's); user-level entries whose **exact content**
          matches across libraries are marked ``duplicate_of`` (annotation
          only, nothing dropped); on success the body gains ``scope`` +
          ``libraries`` (order = the order the libraries were actually hit).

        Error bodies (invalid scope / filter_tags syntax error / tag-key
        validation failure) are always bare dicts, carrying **no**
        self-report keys.
        """
        err = validate_scope(scope)
        if err:
            return {"error": err, "input": scope}
        # B1 mutual exclusion (decision ②a): libraries= is the explicit
        # multi-library channel; combined with the two-library sugar
        # scope=user/all it has no predictable semantics → explicit error,
        # no silent pick of one.
        spec_names: list[str] = []
        if (libraries or "").strip():
            if scope != SCOPE_PROJECT:
                return {"error": (
                    f"libraries= and scope={scope} are mutually exclusive: "
                    "scope=user/all is sugar for the project+user pair, while "
                    "libraries= is the explicit multi-library channel. Their "
                    "combination has no predictable semantics. Use only "
                    "libraries= for multi-library, or only scope for the paired "
                    "scopes."), "input": libraries}
            spec_names, spec_err = parse_library_spec(libraries)
            if spec_err:
                return {"error": spec_err, "input": libraries}
            if not spec_names:
                return {"error": f"libraries= contains only whitespace: {libraries!r}",
                        "input": libraries}
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            # Note: this error body carries **no** library first key
            # (bit-for-bit identical to the current tool).
            return {"error": f"filter_tags parse failed: {tag_err}",
                    "input": filter_tags[:200]}
        # Time-window parsing (2026-11-05): a malformed date is an **explicit
        # error**, never silently treated as "unbounded". Silently dropping
        # the argument makes 「结果看起来像库里真没有」 (the results look like
        # the library truly has nothing) when in fact the condition never
        # took effect.
        try:
            win_after = parse_date_bound(occurred_after)
            win_before = parse_date_bound(occurred_before, end_of_day=True)
        except ValueError as e:
            return {"error": f"Invalid time-window argument: {e}",
                    "input": f"{occurred_after!r} / {occurred_before!r}"}
        if win_after > 0 and win_before > 0 and win_after > win_before:
            return {"error": (f"Inverted time-window bounds: occurred_after={occurred_after} "
                              f"is later than occurred_before={occurred_before}"),
                    "input": f"{occurred_after!r} / {occurred_before!r}"}
        tw = (win_after, win_before)
        if scope == SCOPE_PROJECT:
            if spec_names:
                return self._recall_multi(
                    query=query, top_k=top_k, use_lnn=use_lnn, sample=sample,
                    parsed_tags=parsed_tags, client=client, time_window=tw,
                    spec_names=spec_names)
            # D2b: when the target is the user library (library="user"),
            # execute per access tier.
            tier = self._ua(client)
            if tier == "none":
                return {"error": USER_READ_DENIED, "input": scope}
            body = self._recall_one(
                self.brain, query=query, top_k=top_k, use_lnn=use_lnn,
                sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                record_miss=True, exclude_sensitive=(tier == "masked"))
            if tier:
                self._last_route_query = query
                self._log_route(client, scope="single",
                                probed=[self.library_self.get("name", "")],
                                hits={self.library_self.get("name", ""):
                                      int(body.get("count", 0))})
            return body
        user_brain = self.user_brain
        if user_brain is None:
            # Normally unreachable: with the library missing, the factory
            # resolution has already raised LibraryResolutionError (an
            # explicit recall that cannot find its library must error).
            # Defensive self-report; never silently fall back to project.
            return {"error": "scope=user/all requires the user-level library "
                             "(conventional name user); it was not injected for "
                             "this call.", "input": scope}
        # D2b: the access-tier gate for the user-library flow (scope=user/all
        # targets the user library).
        tier = self._ua(client)
        if scope == SCOPE_USER:
            if tier == "none":
                return {"error": USER_READ_DENIED, "input": scope}
            body = self._recall_one(
                user_brain, query=query, top_k=top_k, use_lnn=use_lnn,
                sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                record_miss=True, exclude_sensitive=(tier == "masked"))
            if "error" in body:
                return body
            user_name = (self.user_library_self or {}).get("name")
            body["memories"] = [dict(h, library=user_name)
                                for h in body.get("memories") or []]
            body["scope"] = SCOPE_USER
            body["libraries"] = [self.user_library_self]
            self._last_route_query = query
            self._log_route(client, scope=SCOPE_USER, probed=["user"],
                            hits={"user": int(body.get("count", 0))})
            return body
        # scope=all: project flow (full bookkeeping) + user flow (stamp only;
        # miss/daily counts go to the project library by query intent) →
        # hard-segmented merge via the domain pure function.
        # D2b: tier gate — none degrades to project-only with a self-report;
        # masked filters sensitive entries from the user flow.
        tier = self._ua(client)
        proj_body = self._recall_one(
            self.brain, query=query, top_k=top_k, use_lnn=use_lnn,
            sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
            record_miss=True)
        if "error" in proj_body:
            return proj_body
        if tier == "none":
            out = dict(proj_body)
            out["scope"] = SCOPE_ALL
            out["user_access"] = "none"
            out["libraries"] = [self.library_self]
            self._last_route_query = query
            self._log_route(client, scope=SCOPE_ALL,
                            probed=["project", "user(denied)"],
                            hits={"project": int(proj_body.get("count", 0)),
                                  "user": 0})
            return out
        user_body = self._recall_one(
            user_brain, query=query, top_k=top_k, use_lnn=use_lnn,
            sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
            record_miss=False, exclude_sensitive=(tier == "masked"))
        if "error" in user_body:
            return user_body
        proj_name = self.library_self.get("name")
        user_name = (self.user_library_self or {}).get("name")
        proj_hits = [dict(h, library=proj_name)
                     for h in proj_body.get("memories") or []]
        user_hits = [dict(h, library=user_name)
                     for h in user_body.get("memories") or []]
        merged = merge_scope_hits(proj_hits, user_hits, top_k=top_k)
        out = {
            "count": len(merged),
            "memories": merged,
            # The echoed tag domain follows the project library (query
            # context; the user library's own filtering is already reflected
            # in each entry's tags — filtering is per library).
            "filter_applied": proj_body.get("filter_applied"),
            "scope": SCOPE_ALL,
            "libraries": [self.library_self, self.user_library_self],
        }
        if tier == "masked":
            out["filtered_sensitive"] = int(
                user_body.get("filtered_sensitive", 0))
        self._last_route_query = query
        self._log_route(client, scope=SCOPE_ALL,
                        probed=["project", "user"],
                        hits={"project": int(proj_body.get("count", 0)),
                              "user": int(user_body.get("count", 0))})
        return out

    def _log_route(self, client: str, *, scope: str, probed: list[str],
                   hits: dict[str, int]) -> None:
        """B3: route events go into the user library's access_log —
        「谁在什么时候跨库查了什么、各库命中几条」 (who queried what across
        libraries when, and how many hits each library got). Only a
        fingerprint of the raw query is kept (build_access_event policy);
        audit is a side path and fails silently."""
        lib = (self.user_library_self or {}).get("path")
        if not lib:
            return
        try:
            ev = build_access_event(
                kind="recall_route", tool="recall", home="route",
                target_name="user", target_path=lib, quarantined=False,
                content=str(getattr(self, "_last_route_query", "") or ""))
            ev["route"] = {"scope": scope, "probed": probed, "hits": hits}
            append_access_event(lib, ev)
        except Exception:  # noqa: BLE001 -- audit side path
            pass

    def _recall_multi(self, *, query: str, top_k: int, use_lnn: bool,
                      sample: bool, parsed_tags: dict[str, Any] | None,
                      client: str,
                      spec_names: list[str] | None = None,
                      time_window: tuple[float, float] = (0.0, 0.0)
                      ) -> dict[str, Any]:
        """B1 hierarchical multi-library (decision ②a): home-library segment all first + periphery libraries compete for the remaining slots via RRF.

        The home-library segment keeps the current ordering semantics and
        bookkeeping policy (record_miss=True); each periphery flow runs
        record_miss=False (query intent belongs to the home library, same as
        the user-library flow of scope=all), and each flow passes its own
        library's tag whitelist (``_recall_one``, per-library policy). The
        envelope self-reports ``mode`` + ``libraries`` (the order the
        libraries were actually hit) + out-of-directory values as
        ``unregistered``.
        """
        tw = time_window
        home_body = self._recall_one(
            self.brain, query=query, top_k=top_k, use_lnn=use_lnn,
            sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
            record_miss=True)
        if "error" in home_body:
            return home_body
        home_name = self.library_self.get("name")
        home_hits = [dict(h, library=home_name)
                     for h in home_body.get("memories") or []]
        periphery: list[tuple[str, list[dict[str, Any]]]] = []
        probed: list[dict[str, str]] = [self.library_self]
        tier = self._ua(client)
        for info, brain in self.extra_libraries:
            # D2b: when a periphery flow includes the user library, the tier
            # applies the same way — none errors the whole call (the denied
            # library was explicitly named), masked filters sensitive
            # entries from that flow.
            if (info.get("name") or "").strip() == "user" and tier != "full":
                if tier == "none":
                    return {"error": USER_READ_DENIED,
                            "input": ",".join(spec_names or [])}
                body = self._recall_one(
                    brain, query=query, top_k=top_k, use_lnn=use_lnn,
                    sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                    record_miss=False, exclude_sensitive=True)
            else:
                body = self._recall_one(
                    brain, query=query, top_k=top_k, use_lnn=use_lnn,
                    sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                    record_miss=False)
            if "error" in body:
                return body
            name = info.get("name")
            periphery.append((name, [dict(h, library=name)
                                     for h in body.get("memories") or []]))
            probed.append(info)
        merged = merge_multi_library(home_hits, periphery, top_k=top_k,
                                     home_name=home_name)
        out: dict[str, Any] = {
            "count": len(merged),
            "memories": merged,
            # The echoed tag domain follows the home library (query context;
            # each library's filtering is already reflected in entry tags).
            "filter_applied": home_body.get("filter_applied"),
            "mode": "multi_library",
            "libraries": probed,
        }
        if self.unregistered_names:
            out["unregistered"] = list(self.unregistered_names)
        self._last_route_query = query
        # Route events are recorded only for calls that probed the user
        # library (access_log is the user library's read audit).
        probed_names = [i.get("name", "") for i in probed]
        if "user" in probed_names:
            self._log_route(client, scope="multi", probed=probed_names,
                            hits={n: len(h) for n, h in periphery})
        return out

    def _recall_one(self, brain: Any, *, query: str, top_k: int,
                    use_lnn: bool, sample: bool,
                    parsed_tags: dict[str, Any] | None, client: str,
                    record_miss: bool,
                    exclude_sensitive: bool = False,
                    time_window: tuple[float, float] = (0.0, 0.0)
                    ) -> dict[str, Any]:
        """Single-library retrieval flow (recall's single-library execution body).

        Parameterizing ``brain`` makes the project path and the user-library
        path the same flow; ``record_miss=False`` is used only by the
        scope=all user-library flow: stamp_recalled is stamped as usual
        (in-library governance data, stamped on both sides), while miss/
        daily counts are not recorded (query intent belongs to the project
        library). ``filter_tags`` JSON syntax parsing happens once in the
        caller; the key-name whitelist validation is done **per library**
        (each library has its own schema).
        """
        mgr = brain.manager
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        allow_ids = brain._ids_matching_tags(parsed_tags)
        if allow_ids is not None and not allow_ids:
            # The filter matched nothing — matched=0 lets the caller see
            # immediately that "key names/values do not line up", instead of
            # misreading it as "the library truly has nothing".
            return {"count": 0, "memories": [],
                    "filter_applied": filter_applied(parsed_tags, 0, key_info)}
        # R1 (2026-11-06): both branches stay verbatim so the rollout switch
        # doubles as an A/B baseline. The shared branch calls the very same
        # side-effect-free pipeline as ``brain.recall``; disabling it restores
        # the legacy additive ``manager.recall`` path bit-for-bit.
        # Adaptive mode state (fusion lane only): eff_k clamps the caller's
        # top_k to the adaptive ceiling; _selection_reports carries the
        # selector self-report up to the response body.
        adaptive_on = bool(self.adaptive_enabled)
        eff_k = min(int(top_k), self.adaptive_max_k) if adaptive_on \
            else int(top_k)
        _selection_reports: list[dict[str, Any]] = []
        if self.recall_fusion_enabled:
            deps = _recall_pipeline.RecallDeps(
                fuse=rrf_fuse, exp=np.exp,
                archived_state=MemoryState.ARCHIVED.value,
                superseded_state=MemoryState.SUPERSEDED.value,
                hard_floor=RELEVANCE_HARD_FLOOR,
                entity_bonus=ENTITY_BONUS,
                literal_topn=MMR_LITERAL_EXEMPT_TOPN,
                # R1-B4: MMR dimension-scale fix (composition-root injected;
                # both RecallDeps construction sites must stay in lockstep).
                mmr_scale_fix=self.mmr_scale_fix,
                now_fn=_now,
                strip_fn=strip_internal_fields,
                iso_fn=iso_date, weekday_fn=weekday_abbr,
                in_window_fn=in_time_window,
                out_window_fn=out_of_window,
                # UR usage-recency: injected by the composition root
                # (masked by the usage_recency bool); tau converts from
                # config days to pipeline seconds here.
                usage_w=self.usage_recency_w,
                usage_tau=self.usage_recency_tau_days * 86400.0,
                usage_cap=self.usage_recency_cap_strength,
                # Adaptive result-count selector (2026-11-07): same lockstep
                # discipline as UR — the composition root masks the
                # numerics by adaptive_enabled; both RecallDeps construction
                # sites must match. The branch flag rides the Request; the
                # caller's top_k becomes a hard CAP (adaptive_max_k), never
                # a quota.
                adaptive_k_min=2,
                adaptive_expand_floor=self.adaptive_expand_floor,
                adaptive_tau_stop=self.adaptive_tau_stop,
                adaptive_lambda=self.adaptive_lambda,
                adaptive_beta=self.adaptive_beta,
                adaptive_gamma=self.adaptive_gamma,
                # Masked tier PRE-selection: safe candidates backfill the
                # slots that sensitive records would have burned (the fixed
                # path keeps its post-truncation filter below).
                sensitive_pred=(
                    (lambda rec: str(getattr(rec, "sensitivity", "") or "")
                     .startswith("sensitive:"))
                    if exclude_sensitive else None),
            )

            def _run(q: str) -> list[dict[str, Any]]:
                rep: dict[str, Any] = {}
                out = _recall_pipeline.run_recall_pipeline(
                    brain,
                    _recall_pipeline.RecallRequest(
                        query=q, top_k=eff_k, use_lnn=use_lnn, sample=sample,
                        allow_ids=allow_ids, client=client,
                        include_superseded=False, use_mmr=True,
                        # Conflict annotation would add three fields and break the
                        # 14-key MCP envelope. Scoring, ordering and filtering are
                        # still identical to brain.recall; only annotation differs.
                        # (The adaptive branch annotates the POOL internally for
                        # dedup decisions; _to_mcp_item drops the fields again.)
                        annotate_conflicts=False,
                        occurred_after=time_window[0],
                        occurred_before=time_window[1],
                        # The lexical lane is mandatory: the 0.30 hard floor kills
                        # every candidate that is neither semantic nor a BM25 hit.
                        use_lexical=True,
                        manager_query=q or None,
                        adaptive=self.adaptive_enabled,
                    ),
                    deps,
                    report_out=rep,
                )
                if self.adaptive_enabled and rep:
                    _selection_reports.append(rep)
                return out

            # V2 (2026-11-07): query fanout — for natural-sentence queries a
            # keyword variant runs the SAME pipeline and the two hit lists are
            # merged with RRF. Opt-in (default off, evaluation
            # reproducibility); scoring formulas are untouched (R1's
            # territory) — fusion only reorders candidates, and every item
            # keeps its own similarity from its source run.
            # NOTE (adaptive interaction): under adaptive each variant run
            # already selects at the cap, so fanout+adaptive fuses two
            # SELECTED lists — candidate-level fusion is a recorded v1
            # limitation (design doc §6.C4), not a regression of fixed mode.
            variant = ""
            if self.recall_query_fanout_enabled:
                variant = query_variants.build_variant(
                    query or "",
                    entity_names=[
                        e.get("name", "") for e in (brain.entities or {}).values()
                        if isinstance(e, dict)
                    ],
                )
            if variant:
                items = query_variants.fuse_variant_items(
                    _run(query or ""), _run(variant), eff_k)
            else:
                items = _run(query or "")
            memories = [self._to_mcp_item(it, mgr) for it in items]
            # Stamp only what the host actually sees (same rule as
            # brain.recall); stamping the candidate pool keeps stale rows alive.
            stamp_ids = [it["memory_id"] for it in items]
            top_sim = max((float(it["similarity"])
                           for it in items), default=0.0)
        else:
            results = mgr.recall(
                query=query if query else None,
                top_k=top_k,
                use_lnn=use_lnn,
                sample=sample,
                allow_ids=allow_ids,
                client=client,
                occurred_after=time_window[0],
                occurred_before=time_window[1],
            )
            memories = []
            top_sim = 0.0
            quarantined_hidden = 0
            for r in results:
                d = r.to_dict()
                rec = mgr.store.get(r.memory_id)
                if rec is not None and getattr(rec, "quarantine", False):
                    quarantined_hidden += 1
                    continue
                if rec is not None:
                    d["source"] = rec.source or ""
                    d["state"] = rec.state
                    d["cognitive_type"] = rec.cognitive_type
                    d["tags"] = dict(rec.tags or {})
                top_sim = max(top_sim, float(r.similarity))
                memories.append(d)
            stamp_ids = [r.memory_id for r in results]
        memories = self._cap_contents(memories)
        # D2b masked tier: sensitive:* cards never leave the flow (count is
        # self-reported, never silent).
        filtered_sensitive = 0
        if exclude_sensitive:
            kept: list[dict[str, Any]] = []
            for d in memories:
                rec = mgr.store.get(d.get("memory_id"))
                if rec is not None and str(
                        getattr(rec, "sensitivity", "") or ""
                ).startswith("sensitive:"):
                    filtered_sensitive += 1
                    continue
                kept.append(d)
            memories = kept
        # Stamp the 「最后一次被召回」 (last recalled) timestamp: the shared
        # layer is deliberately side-effect free, so bookkeeping stays inside
        # the caller's shell (same discipline as brain.recall, preventing
        # double counting across the two paths).
        # With fusion on, only the final returned entries are stamped; with
        # it off, the legacy policy of stamping all candidates is kept.
        mgr.stamp_recalled(stamp_ids)
        if record_miss:
            # Miss logging + daily recall counting: MCP recall is the host's
            # main retrieval entry; the shared layer does no bookkeeping, so
            # it must be added here, otherwise
            # miss_log / the daily stats are all zeros.
            if query:
                if brain.miss_log_enabled and top_sim < brain.miss_log_threshold:
                    brain.log_miss(query, top_sim, top_k=top_k,
                                   path="mcp.recall")
            brain.bump_daily("recall")
        out = {
            "count": len(memories),
            "memories": memories,
            # Echo this call's tag domain and post-filter candidate count
            # (matched=None = no filter, whole library).
            "filter_applied": filter_applied(
                parsed_tags, len(allow_ids) if allow_ids is not None else None,
                key_info),
        }
        if exclude_sensitive:
            # D2b masked tier: self-report the filtered count (the key is
            # present even at 0 — tier behavior stays assertable and is not
            # confused with "no filtering happened").
            out["filtered_sensitive"] = filtered_sensitive
        if self.adaptive_enabled and _selection_reports:
            # Adaptive selection self-report (design doc §3.3.5): counts,
            # budget and WHY entries were omitted — never raw scores.
            # Fanout merges two runs: keep the primary query's report.
            out["selection"] = _selection_reports[0]
        if self.recall_fusion_enabled and memories and top_sim < RELEVANCE_HARD_FLOOR:
            # Score-trap guard (naturalization review P3-3): the top hits
            # came through the BM25/entity lane only — surface the "RRF
            # scores are ranking points, not cosine similarities" warning
            # exactly when it can mislead, zero standing cost otherwise.
            out["note"] = (
                "Top results matched via keyword/entity lanes only "
                "(vector similarity below the %.2f relevance floor); "
                "'score' is an RRF ranking point, not a cosine similarity "
                "— use 'similarity' for semantic closeness."
                % RELEVANCE_HARD_FLOOR)
        if time_window[0] > 0 or time_window[1] > 0:
            # Time-window self-report: **especially important when count=0**
            # — otherwise 「窗口内确实没有」 (truly nothing in the window) and
            # 「检索没找到」 (retrieval found nothing) look identical on the
            # response surface, and the caller will read the former as the
            # latter.
            out["time_window"] = {
                "after": iso_date(time_window[0]) if time_window[0] > 0 else None,
                "before": iso_date(time_window[1]) if time_window[1] > 0 else None,
            }
        return out

    @staticmethod
    def _to_mcp_item(it: dict[str, Any], mgr: Any) -> dict[str, Any]:
        """Adapt a shared-pipeline item to the 14-key MCP envelope.

        Cuts the five pipeline-only attribution fields, adds the two
        ``RecallResult`` fields read back from the stored record,
        and rounds both floats to the legacy precision.
        """
        rec = mgr.store.get(it["memory_id"])
        d = dict(it)
        d["score"] = round(float(d["score"]), 4)
        d["similarity"] = round(float(d["similarity"]), 4)
        d["recall_count"] = int(getattr(rec, "recall_count", 0) or 0)
        d["memory_type"] = str(getattr(rec, "memory_type", "") or "general")
        return {k: d[k] for k in _MCP_RECALL_ITEM_ORDER}

    def _cap_contents(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Truncation guard: clamp each hit memory's ``content`` to the character cap.

        Why it exists: ``recall`` originally **bounded only the entry count**
        (``top_k``) while a single ``content`` could be arbitrarily long —
        one long memory could push a single recall into the tens of thousands
        of characters, and the caller (the host's context) had no way to
        anticipate it.

        Why not truncate silently: the host cannot see what was cut and would
        believe "this is all the library has". So a truncated entry must
        carry ``content_truncated=true`` + ``content_chars`` (original
        length); for the full text, use ``get_memory(memory_id)``.

        ``0`` = no truncation (**default**, = zero change from the status
        quo); when the cap is longer than the text, no keys are added (the
        response surface is byte-identical to the off state). The cap is read
        now by the composition root from
        ``HIPPOCAMPUS_RECALL_CONTENT_MAX_CHARS`` and injected (application
        does not read config).
        """
        cap = int(self.recall_content_max_chars or 0)
        if cap <= 0:
            return memories
        out: list[dict[str, Any]] = []
        for d in memories:
            text = d.get("content")
            if not isinstance(text, str) or len(text) <= cap:
                out.append(d)
                continue
            d = dict(d)
            d["content"] = text[:cap]
            d["content_truncated"] = True
            d["content_chars"] = len(text)
            out.append(d)
        return out

    # ---------------------------------------------------------------- single item and listing

    def get_memory(self, *, memory_id: str,
                   include_history: bool = False,
                   client: str = "") -> dict[str, Any]:
        """Fetch one entry's full information by ID (explicit error when absent, never a silent empty).

        With ``include_history=True`` a ``history`` key is attached —
        the pre-rewrite values of this record from mem_history.jsonl
        (update/merge, newest first,
        cap 20). The key is absent by default (zero change to the response
        surface); the ledger is an audit surface and stays out of retrieval.

        D2b: when the target is the user library, per tier — the masked tier
        errors explicitly on sensitive:* cards (a single-item read has no
        "filter out" to speak of; handing over half of it is a leak); the
        none tier refuses entirely.
        """
        tier = self._ua(client)
        if tier == "none":
            return {"error": USER_READ_DENIED, "memory_id": memory_id}
        if tier == "masked":
            rec = self.manager.store.get(memory_id)
            if rec is not None and str(
                    getattr(rec, "sensitivity", "") or ""
            ).startswith("sensitive:"):
                return {"error": (
                            "This memory is sensitive and this workspace has "
                            "masked read access (sensitive records are unreadable)."),
                        "memory_id": memory_id}
        record = self.manager.get_memory(memory_id)
        if record is None:
            return {"error": "memory not found", "memory_id": memory_id}
        if include_history:
            record["history"] = self.manager.memory_history(
                memory_id=memory_id, limit=20)
        return record

    def list_memories(self, *, limit: int, source: str,
                      state: str, client: str = "") -> dict[str, Any]:
        """List metadata by source/state (no embedding vectors).

        D2b masked tier: sensitive:* rows are filtered out and counted (the
        listing is a content surface, same policy as recall).
        """
        memories = self.manager.list_memories(
            limit=limit, source=source or None, state=state or None
        )
        tier = self._ua(client)
        if tier == "masked":
            kept = []
            filtered = 0
            for row in memories:
                rec = self.manager.store.get(row.get("memory_id")
                                             or row.get("id"))
                if rec is not None and str(
                        getattr(rec, "sensitivity", "") or ""
                ).startswith("sensitive:"):
                    filtered += 1
                    continue
                kept.append(row)
            memories = kept
            return {"count": len(memories), "memories": memories,
                    "filtered_sensitive": filtered}
        return {"count": len(memories), "memories": memories}

    def list_sources(self, client: str = "") -> dict[str, Any]:
        """Source labels and their entry counts (the basis of import idempotency).

        D2b: a source is the **index** behind L0 back-references (it can
        locate the original text directly), so the masked tier refuses
        outright (fail-closed: filtering the content surface while leaving
        the back-reference surface open equals no filtering).
        """
        if self._ua(client) in ("masked", "none"):
            return {"error": USER_READ_DENIED}
        raw = self.manager.sources()
        labeled = {("(no source)" if k == "" else k): v for k, v in raw.items() if k != ""}
        return {
            "count": len(labeled),
            "sources": dict(sorted(labeled.items(), key=lambda kv: -kv[1])),
            "unlabeled": raw.get("", 0),
        }

    # ---------------------------------------------------------------- assembled context

    def get_memory_context(self, *, query: str, top_k: int,
                           token_budget: int, item_max_chars: int,
                           filter_tags: str,
                           scope: str = SCOPE_PROJECT,
                           client: str = "") -> dict[str, Any]:
        """5-slot assembly (current hits / pending open intents / timeline / entity state / last session's unfinished topics).

        A ``filter_tags`` parse-error body carries no envelope (same as
        :meth:`recall`); the difference from recall: it does not return
        count=0 early — empty candidates are still assembled (each slot
        self-reports as empty), and ``filter_applied.matched`` is echoed as
        usual.

        scope: **only project / user are supported** (5-slot assembly has no
        natural place for "project first, user behind", deliberately
        narrow). scope=user moves the whole assembly onto the user-library
        brain; on success the body gains ``scope`` + ``libraries``
        self-report; the project default path adds no self-report keys.
        scope=all is an explicit error (the error body carries no
        self-report keys).
        """
        err = validate_scope(scope, allow_all=False)
        if err:
            return {"error": err, "input": scope}
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            return {"error": f"filter_tags parse failed: {tag_err}",
                    "input": filter_tags[:200]}
        brain = self.brain
        if scope == SCOPE_USER:
            if self.user_brain is None:
                return {"error": "scope=user requires the user-level library "
                                 "(conventional name user); it was not injected "
                                 "for this call.", "input": scope}
            # D2b: the assembly surface's cross-library channel opens only at
            # the full tier — 5-slot assembly has no natural place for
            # 「滤一半」 (filtering half), same narrowing rationale as
            # scope=all being recall-only.
            tier = self._ua(client)
            if tier == "none":
                return {"error": USER_READ_DENIED, "input": scope}
            if tier == "masked":
                return {"error": (
                            "This workspace has masked read access to the user "
                            "library; assembled context (get_memory_context) does "
                            "not support sensitive-level filtering. Use "
                            "recall(scope=\"user\") (which self-reports "
                            "filtered_sensitive) or assemble from a "
                            "home-workspace session."), "input": scope}
            brain = self.user_brain
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        ctx = brain.get_memory_context(query=query, top_k=top_k,
                                       token_budget=token_budget,
                                       item_max_chars=item_max_chars,
                                       filter_tags=parsed_tags)
        # Prospective-memory contextual triggering. When the query's
        # entities/content words intersect an active open intent's trigger
        # set, that intent is brought back at the front of block_2 (cap 5, no
        # inflation) and the trigger details are self-reported under the new
        # key ``prospective_intents``. Done as service-layer post-processing
        # rather than in the brain wrapper — brain.get_memory_context stays
        # a thin shell (parity and AST-level tests pin its structure), and
        # context_slots is untouched.
        # The switch is read now by the composition root and injected
        # (application does not import config).
        if self.prospective_trigger:
            self._apply_prospective(ctx, brain, query=query,
                                    token_budget=token_budget,
                                    item_max_chars=item_max_chars)
        # Metamemory confidence made visible. Each hit gets a ``confidence``
        # field, and the whole block set gets a receipt-style ``confidence``
        # self-report block (inheriting the session_start receipt discipline:
        # it answers "what state is this assembly's certainty in and how was
        # it computed"). Only new keys are added, backward compatible; no new
        # tool. Formulas and calibration anchors: domain/metacog.py.
        self._apply_confidence(ctx, brain)
        # Echo the tag domain + post-filter candidate count (same policy as
        # recall / contradictions;
        # matched=None = no filter, 0 = the filter matched nothing).
        allow = brain._ids_matching_tags(parsed_tags)
        ctx["filter_applied"] = filter_applied(
            parsed_tags, len(allow) if allow is not None else None, key_info)
        # Recent-miss visibility (naturalization review P1-4 data source):
        # how many retrieval misses this library logged in the last 24h.
        # Additive key only — the hook turns a high count into a one-line
        # "verify conclusions independently" hint; the field itself never
        # blocks or reorders anything.
        try:
            cutoff = _now() - 86400.0
            ctx["recent_miss_24h"] = sum(
                1 for e in (getattr(brain, "miss_log", None) or [])
                if float(e.get("ts") or 0.0) >= cutoff)
        except (TypeError, ValueError):
            ctx["recent_miss_24h"] = 0
        if scope == SCOPE_USER:
            ctx["scope"] = SCOPE_USER
            ctx["libraries"] = [self.user_library_self]
        return ctx

    # -------------------------------------------- prospective trigger / confidence

    @staticmethod
    def _apply_prospective(ctx: dict[str, Any], brain: Any, *,
                           query: str, token_budget: int,
                           item_max_chars: int) -> None:
        """block_2 contextual-trigger bring-back (mutates ``ctx`` in place; when not triggered, zero trace and no keys added).

        ``text`` is re-rendered after the reordering (blocks and text are two
        tracks of one source; once blocks change, text must be re-rendered in
        sync, and ``token_estimate`` refreshed along with it). The open-intent
        entries themselves are read-only —
        text / status are not modified, so complete_intent's completion
        semantics are unaffected.
        """
        if not (query or "").strip():
            return
        plan = _prospective.prospective_block2(
            getattr(brain, "intents", None) or [],
            query=query,
            extract_entities=brain.extract_entities,
            tokens_of=brain._tokens_of)
        if not plan.triggered:
            return
        blocks = ctx.get("blocks")
        if not isinstance(blocks, dict):
            return
        blocks["block_2_pending"] = plan.lines
        ctx["text"] = brain._render_blocks(blocks, token_budget, item_max_chars)
        ctx["token_estimate"] = len(ctx["text"]) // 2
        ctx["prospective_intents"] = plan.triggered

    @staticmethod
    def _apply_confidence(ctx: dict[str, Any], brain: Any) -> None:
        """Per-hit + per-block confidence (mutates ``ctx`` in place; only adds keys, never changes existing ones).

        block_1 = the mean over per-hit confidences; block_2/3/5 come from
        the first-party layer and get the full score;
        block_4 (entity state) is derived from L1 and entity_facts returns
        text only, never ids, so per-item traceability is impossible —
        honestly report None and say so in basis (receipt discipline:
        not assessable ≠ a score of 0).
        """
        raw_ids = {r.get("id")
                   for r in (getattr(brain, "raw_log", None) or [])
                   if isinstance(r, dict) and r.get("id")}
        vals: list[float] = []
        for h in ctx.get("hits") or []:
            src = h.get("source_ids") or []
            vals.append(_metacog.hit_confidence(
                similarity=float(h.get("similarity") or 0.0),
                state=str(h.get("state") or ""),
                source_ids=src,
                source_resolved=sum(1 for s in src if s in raw_ids),
                pin=bool(h.get("pin"))))
            h["confidence"] = vals[-1]
        blocks = ctx.get("blocks") or {}

        def _first_party(lines: Any, basis: str) -> dict[str, Any]:
            """Confidence entry for a first-party block: non-empty gets the full score; placeholder/empty = None (not a score of 0).

            basis is coupled with value: when None it reports empty_* rather
            than slapping on a fake "full score" label.
            """
            lines = lines or []
            if not lines or (len(lines) == 1 and str(lines[0]) in BLOCK_PLACEHOLDERS):
                return {"value": None, "basis": "empty_" + basis}
            return {"value": _metacog.FIRST_PARTY_SCORE, "basis": basis}

        ctx["confidence"] = _metacog.confidence_receipt(
            {
                "block_1_current_hits": {
                    "value": _metacog.mean_confidence(vals),
                    "basis": (f"mean_of_{len(vals)}_hits"
                              if vals else "empty")},
                "block_2_pending": _first_party(
                    blocks.get("block_2_pending"), "first_party_intents"),
                "block_3_timeline": _first_party(
                    blocks.get("block_3_timeline"), "first_party_summaries"),
                "block_4_entity_state": {
                    "value": None,
                    "basis": "derived_from_L1_no_direct_trace"},
                "block_5_open_topics": _first_party(
                    blocks.get("block_5_open_topics"), "first_party_open_topics"),
            },
            computed_at=time.time())

    # ---------------------------------------------------------------- contradictions and drift

    def contradictions(self, *, limit: int, source: str,
                       filter_tags: str, plan: bool = False) -> dict[str, Any]:
        """Candidate pairs of 「同一事实、两种口径」 (one fact, two accounts) (annotation only, no adjudication; after confirmation go through update_memory)."""
        brain = self.brain
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            return {"error": f"filter_tags parse failed: {tag_err}",
                    "input": filter_tags[:200]}
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        result = brain.contradictions(limit=limit, source=source,
                                      filter_tags=parsed_tags)
        # Echo the tag domain + post-filter candidate count. candidate_pool
        # is "how many entries entered the comparison after narrowing",
        # same policy as recall / get_memory_context's matched.
        result["filter_applied"] = filter_applied(
            parsed_tags, result.get("candidate_pool"), key_info)
        if plan:
            result.update(self._supersede_plan(result.get("candidates") or []))
        return result

    def _supersede_plan(self, candidates: list) -> dict:
        """Project a reviewable plan; never decide which side is correct.

        Pairs whose timestamps are indistinguishable go to ``need_human``
        instead of being guessed from an arbitrary tie-break.
        """
        plan, need_human = [], []
        for c in candidates:
            a_ts = float(c.get("a_timestamp") or 0)
            b_ts = float(c.get("b_timestamp") or 0)
            if abs(a_ts - b_ts) < 1.0:
                need_human.append({k: c.get(k) for k in
                                   ("a_id", "b_id", "a", "b", "sim")})
                continue
            if c.get("newer") == "a":
                old_id, new_id = c.get("b_id"), c.get("a_id")
                old_text, new_text = c.get("b"), c.get("a")
            else:
                old_id, new_id = c.get("a_id"), c.get("b_id")
                old_text, new_text = c.get("a"), c.get("b")
            plan.append({"old_id": old_id, "new_id": new_id,
                         "old": old_text, "new": new_text,
                         "sim": c.get("sim"), "newer": c.get("newer")})
        return {"supersede_plan": plan, "need_human": need_human,
                "plan_count": len(plan), "need_human_count": len(need_human),
                "note": "Review the array, then pass the approved entries back "
                        "as the `confirm` JSON argument. The service never "
                        "adjudicates; it only writes what the host approved."}

    def apply_supersede_plan(self, plan_json: str) -> dict:
        """Execute host-approved supersede pairs in one call.

        Supersession is a state change, not a deletion: the old card stays
        auditable.  Pinned cards are unpinned (batch B) as part of the update.
        """
        try:
            plan = json.loads(plan_json)
        except (ValueError, TypeError) as exc:
            return {"error": "confirm is not valid JSON: %s" % exc,
                    "input": str(plan_json)[:200]}
        if not isinstance(plan, list):
            return {"error": "confirm must be a JSON array of {old_id, new_id}"}
        from ..store.memory import MemoryState
        store = self.brain.manager.store
        applied, rejected = [], []
        for i, item in enumerate(plan):
            if not isinstance(item, dict) or not item.get("old_id") or not item.get("new_id"):
                rejected.append({"index": i, "reason": "missing old_id/new_id"})
                continue
            old_id, new_id = str(item["old_id"]), str(item["new_id"])
            if old_id == new_id:
                rejected.append({"index": i, "reason": "old_id == new_id"})
                continue
            old = store.get(old_id)
            new = store.get(new_id)
            if old is None or new is None:
                rejected.append({"index": i, "old_id": old_id,
                                 "reason": "id not found"})
                continue
            if old.state != MemoryState.ACTIVE.value:
                rejected.append({"index": i, "old_id": old_id,
                                 "reason": "old state=%s" % old.state})
                continue
            ok = store.update(old_id, state=MemoryState.SUPERSEDED.value,
                              superseded_by=new_id, pin=False)
            if ok:
                applied.append({"old_id": old_id, "new_id": new_id})
            else:
                rejected.append({"index": i, "old_id": old_id,
                                 "reason": "update returned False"})
        return {"applied": applied, "rejected": rejected,
                "applied_count": len(applied), "rejected_count": len(rejected),
                "note": "Only host-approved pairs were written."}

    def drift_audit(self, *, source: str, limit: int,
                    include_ok: bool, client: str = "") -> dict[str, Any]:
        """Source-file drift audit: signature vs on-disk reality, six report classes each with its denominator.

        D2b (landing the Q-14 leaning): when the target is the user library,
        the audit surface is **callable from the home workspace only** — the
        audit emits L2/L3 aggregate content (profile/topics), which under the
        masked tier would amount to a surface bypassing content filtering.
        Quarantine review belongs in the home workspace to begin with (the
        privilege-escalation key is not in the agent's hands).

        Since D2a, also carries a **quarantine-queue review section**
        (design draft §4.3, decision ①): this library's quarantine cards are
        listed one by one for the user to review (promotion goes through
        ``update_memory``'s ``clear_quarantine``, deletion through
        ``forget_memory``); the user library also gets an access_log
        tail window — "who touched the user library, and when" is auditable
        here.
        Audit is a side path: any failure in this section does not block the
        source-drift report itself.
        """
        # The tier gate applies only to user-library targets (the resolver
        # returns "" for non-user targets) —
        # use `in` to test masked/none, not != "full" (the empty-string tier
        # would be wrongly refused).
        _tier = self._ua(client)
        if _tier in ("masked", "none"):
            return {"error": (
                "User-library audit (drift_audit) is callable only from a home "
                f"workspace (current access level: {_tier}); audits expose L2/L3 "
                "aggregate content, so masked/none levels are always refused. "
                "Review quarantined items from a home workspace (daily assistant "
                "host) session.")}
        out = self.brain.source_drift(source=source, limit=limit,
                                      include_ok=include_ok)
        try:
            quarantined = []
            for rec in self.manager.store.iter_records():
                if getattr(rec, "quarantine", False):
                    quarantined.append({
                        "memory_id": rec.id,
                        "content_preview": (rec.content or "")[:120],
                        "sensitivity": getattr(rec, "sensitivity", ""),
                        "timestamp": rec.timestamp,
                        "source_ids": list(rec.source_ids or []),
                    })
            quarantined.sort(key=lambda r: -r["timestamp"])
            out["quarantine"] = {
                "count": len(quarantined),
                "items": quarantined[:limit],
                "hint": ("Promote: update_memory(memory_id=..., "
                         "clear_quarantine=True) (home-library session only); "
                         "delete: forget_memory(memory_id=...)."),
            }
            # B2 directory health: lost library paths/bound roots and zombie
            # entries (the descriptive audit hook,
            # design draft §2.2/§3.2 — directory notes rot and paths drift;
            # this must be visible).
            try:
                ulib = resolve_library("user", libs=scan_libraries())
                ddata, dissue = load_directory(ulib.path)
                health: dict[str, Any] = {"ready": dissue is None,
                                          "entries": 0,
                                          "path_missing": [],
                                          "workspace_missing": []}
                if dissue is None:
                    ents = ddata.get("entries") or []
                    health["entries"] = len(ents)
                    for e in ents:
                        if not isinstance(e, dict):
                            continue
                        if not os.path.isdir(str(e.get("path") or "")):
                            health["path_missing"].append(e.get("task"))
                        wr = str(e.get("workspace_root") or "")
                        if wr and not os.path.isdir(wr):
                            health["workspace_missing"].append(e.get("task"))
                out["directory_health"] = health
            except Exception as e2:  # noqa: BLE001 -- audit side path
                out["directory_health"] = {
                    "ready": False, "reason": f"{type(e2).__name__}: {e2}"}
            if (self.library_self or {}).get("name") == "user":
                out["access_log"] = read_access_events(self.persist_dir,
                                                       limit=limit)
        except Exception as e:  # noqa: BLE001 -- audit side path must never blow up the main report
            out["quarantine_error"] = f"{type(e).__name__}: {e}"
        return out

    # ---------------------------------------------------------------- entity profiles

    def entity_profile(self, *, entity: str, limit: int, order: str,
                       include_superseded: bool) -> dict[str, Any]:
        """One entity's full profile: every memory it took part in, pulled in time order."""
        return self.brain.entity_profile(entity, limit=limit, order=order,
                                         include_superseded=include_superseded)

    def entity_list(self, *, prefix: str, limit: int, min_mentions: int,
                    trusted_only: bool) -> dict[str, Any]:
        """Which entities exist (descending by mention count; trusted-source filtering to cut noise)."""
        return self.brain.entity_list(prefix=prefix, limit=limit,
                                      min_mentions=min_mentions,
                                      trusted_only=trusted_only)

    # ---------------------------------------------------------------- knowledge map

    def knowledge_map(self, *, limit: int,
                      include_archived: bool) -> dict[str, Any]:
        """Knowledge-structure overview (topics + representative sentences + playbook). **With** the library envelope."""
        return self._envelope(self.brain.knowledge_map(
            limit=limit, include_archived=include_archived))

    def knowledge_topic(self, *, topic_id: str, action: str, summary: str,
                        evidence: str, limit: int, topics: str, items: str,
                        batch: int, per_topic: int, fras: str) -> dict[str, Any]:
        """Single knowledge-point read/write channel: brief / summarize / revert / rebuild / refresh.

        Parse-failure error bodies for ``items`` / ``fras`` carry **no**
        library first key —
        silently treating them as empty would make the host believe the
        write-back succeeded when in fact nothing entered the library.
        The success path goes through ``_envelope`` (same rule as the
        adapter layer's ``_with_library``).
        """
        # items is the JSON string of the 「归纳条目数组」 (summary-entry
        # array) ([{"topic_id":..,"summary":..
        # ,"evidence":[..]}]). Parse failure must be an **explicit error**.
        parsed_items: Any = None
        raw_items = (items or "").strip()
        if raw_items:
            try:
                parsed_items = json.loads(raw_items)
            except json.JSONDecodeError as exc:
                return {"ok": False, "action": action,
                        "error": f"items is not valid JSON: {exc}"}
            if not isinstance(parsed_items, list):
                return {"ok": False, "action": action,
                        "error": "items must be an array (each item contains topic_id/summary/evidence)"}
        # fras same as above: parse failure is an explicit error (silently
        # treating it as "not provided" would make the host believe the
        # structure was written).
        parsed_fras: Any = None
        raw_fras = (fras or "").strip()
        if raw_fras:
            try:
                parsed_fras = json.loads(raw_fras)
            except json.JSONDecodeError as exc:
                return {"ok": False, "action": action,
                        "error": f"fras is not valid JSON: {exc}"}
            if not isinstance(parsed_fras, dict):
                return {"ok": False, "action": action,
                        "error": "fras must be a JSON object (f/r/a/s keys)"}
        return self._envelope(self.brain.knowledge_topic(
            topic_id=topic_id, action=action, summary=summary,
            evidence=parse_str_list_lenient(evidence), limit=limit,
            topics=parse_str_list_lenient(topics) or None,
            items=parsed_items, batch=batch, per_topic=per_topic,
            fras=parsed_fras))
