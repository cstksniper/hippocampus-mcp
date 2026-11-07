# -*- coding: utf-8 -*-
"""B2-3 P3: session and writer cluster

This cluster was moved verbatim from ``brain.py``.

Why this cluster can move: the **writers of ``_current_sessions`` and of
the writer registry are all inside this cluster** (``_touch_writer`` /
``_unregister_writer`` / ``session_*``) — the same cohesion criterion as
P2: the writers of an attribute travel as one cluster, so moving the whole
cluster never splits one attribute's writers across two files.

⚠️ ``current_session_id`` is a **property** whose getter and setter share
the **same name**; that only works when both live in one class — the mover
treats it as multiple nodes and moves them together; moving just one would
silently break the property (readable, not writable).

``session_start`` / ``session_end`` are direct-call entries of server.py;
after moving into the mixin they resolve via MRO, and the
``brain.session_start(...)`` call site stays unchanged.
"""

from __future__ import annotations

from ..common.constants import *  # noqa: F401,F403  (constants table: see common/constants.__all__)
from ..common.constants import logger
import difflib
import re
from ..common.constants import logger
from ..common.present import (
    render_map_detail_block,
    render_map_index_block,
    render_overview_block,
    render_status_block,
    render_task_profile_block,
)
from ..config import load_config
# Map-layer constants (see map/constants.py)
from ..map.constants import MAP_STALE_DAYS
# B/C/D session constants live in domain/constants (single consumer, kept out
# of the shared base; G4 guard).
from ..domain.constants import (ENV_SESSION_IDLE_RECYCLE,  # noqa: F401
                                ENV_TOPIC_SIM_DEDUP, SESSION_IDLE_RECYCLE,
                                TOPIC_SIM_DEDUP)
import json
import os
import time
from typing import Any
from ..store.memory import MemoryRecord, MemoryState
from ..domain import topic as topic_mod
from ..domain.user_scope import render_user_scope, select_user_scope
# Pure data layer of the task-directory slot (selection/cap/rendering). domain
# depends only on the standard library; the I/O lives in library.task_directory
# (the line below).
from ..domain.task_directory import (render_task_directory,
                                     select_task_directory)
# Lightweight read path for the user-level inheritance slot (brain → library +
# store is an allowed direction, same as core.py's display_name). Read-only:
# no brain construction, no embedding model load. The task-directory sidecar
# (``<user 库>/directory.json``) takes the same lightweight path — it reads a
# file, not a brain.
from ..library import (LibraryResolutionError, resolve_library,
                       scan_libraries)
from ..library import task_directory as task_directory_mod
from ..store.backend import get_backend
try:
    import socket as _socket
except ImportError:  # pragma: no cover
    _socket = None


def _topic_sim(a: str, b: str) -> float:
    """D: topic similarity (0~1), normalized dual metric + hard-identifier penalty.

    Normalization = Latin lowercased + Chinese/Western punctuation and
    separators (，、；：/|()[] etc.) flattened — for the validator's real
    pair 「TA/I/J/…按月续传」 vs 「其余品种（TA、I、…）分期补齐」, raw
    difflib gives only 0.469 (the 0.85 threshold was effectively dead; the
    first version failed exactly here), while normalization + the dual
    metric gives 0.63; unrelated pairs score ≤0.21. The threshold defaults
    to 0.60, leaving margin on both sides.

    Hard-identifier penalty: strings whose token sets of digits/Latin/
    Chinese numerals (43, TA, the "五" of 卷五) differ are **treated as
    different topics** — single-character differences like "第43章校对" vs
    "第44章校对" or "卷五" vs "卷六" reach 0.6~0.85 under bigrams, so
    without the penalty they would be wrongly blocked (T2/T3 in this very
    round were live offenders). The penalty halves the score, pushing such
    pairs below the threshold; paraphrases that keep the hard identifiers
    (the validator's real pair has identical token sets) are unaffected.
    """
    def _norm(s: str) -> str:
        s = re.sub(r"[，、；：/|,;:（）()「」【】\[\]]", " ", s.lower())
        return re.sub(r"\s+", "", s)

    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    ba = {na[i:i + 2] for i in range(len(na) - 1)} if len(na) > 1 else {na}
    bb = {nb[i:i + 2] for i in range(len(nb) - 1)} if len(nb) > 1 else {nb}
    overlap = len(ba & bb) / min(len(ba), len(bb)) if ba and bb else 0.0
    score = max(ratio, overlap)
    tokens = re.compile(r"[a-z0-9]+|[一二三四五六七八九十百千万零两甲乙丙丁戊]+")
    ta = set(tokens.findall(na))
    tb = set(tokens.findall(nb))
    if ta != tb and (ta or tb):
        # Hard-identifier token sets differ (order-insensitive: "43、44" and
        # "44、43" are synonymous) ⇒ halve the score.
        score *= 0.5
    return score


def _slot_switch(param: int, cfg_value: bool) -> bool:
    """Three-state slot switch: ``param>0`` on / ``<0`` off / ``==0`` use the config value.

    Semantics moved verbatim from the retired ``common/constants._block_switch``
    (same family as the 「0 = 用 env」 convention of ``pin_limit`` /
    ``knowledge_limit``).

    Why it moved here instead of staying in constants: its second operand is
    **no longer an env name** but a Config field already computed by the
    composition root, and constants must not depend on config (dependency
    graph ``_ALLOWED["common"] == {"common"}``). The three-state form itself
    is a **tool-surface UI protocol** (the caller uses 0 to mean "don't
    decide for me"), so it belongs to the session layer, not the constants
    layer.
    """
    if param > 0:
        return True
    if param < 0:
        return False
    return cfg_value


def _topic_defaults() -> topic_mod.TopicDefaults:
    """The defaults package of domain/topic: assembled from Config at call time (domain must not read env).

    **Why a function and not a module-level constant**: the original
    implementation of topic's six knobs read env at call time
    (``topic_layer_enabled()`` did ``os.environ.get`` every call), and both
    tests and the daemon pin behavior via 「设 env → 调 brain」 (set env →
    call brain); making it an import-time snapshot would silently break
    settings like ``HIPPOCAMPUS_TOPIC_LAYER=0`` (measured during the
    migration: the four assertions of test_knowledge_topic_layer A8/B8 and
    test_topic_low_info L6 all went red). The three ring items
    (``core.TAG_OPS_RING`` etc.) deliberately keep the snapshot the other
    way — their original implementation was constants import-time
    evaluation to begin with.

    **Why it lives in session.py**: core.py hit its line ceiling (the
    line-count guard pins ≤6299), and this is the place in the brain package
    that already imports both ``topic_mod`` and ``load_config``.
    """
    c = load_config()
    return topic_mod.TopicDefaults(
        join_sim=c.topic_join_sim, max_topics=c.topic_max,
        max_members=c.topic_max_members, summary_topn=c.topic_summary_topn,
        layer_enabled=c.topic_layer, low_info=c.topic_low_info)


def _health_snapshot(brain) -> dict:
    """Compact, opt-out health bill shown at every session_end.

    Reuses projections that already exist; no new scan pipeline and no new
    tool.  Governance debt was invisible between manual review batches, so it
    is surfaced here as five numbers rather than a prose reminder.
    """
    import os as _os
    out: dict = {}
    try:
        ledger = brain.save_ledger_view()
        out["unsaved_seconds"] = ledger.get("interval_s")
        out["unsaved_layers"] = len(ledger.get("pending_layers") or [])
        out["l1_dirty"] = bool(ledger.get("l1_dirty"))
        out["save_level"] = ledger.get("level")
    except Exception as exc:  # noqa: BLE001 -- health is a gift, not a failure
        out["save_level"] = "unknown:%s" % type(exc).__name__
    try:
        pin = brain._pin_stats()
        out["pinned_ratio"] = pin.get("pinned_ratio")
        out["pin_quota_warning"] = bool(pin.get("quota_warning"))
    except Exception:  # noqa: BLE001
        pass
    try:
        kinds = brain.knowledge_stats().get("summary_kinds") or {}
        out["knowledge_stale"] = int(kinds.get("host_stale") or 0)
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..domain import stats_projection
        raw_ids = {r.get("id") for r in brain.raw_log}
        dangling, recs = stats_projection.dangling_report(
            brain.manager.store.iter_records(), raw_ids)
        out["l0_dangling_ids"] = len(dangling)
        out["l0_dangling_memories"] = recs
    except Exception:  # noqa: BLE001
        pass
    try:
        out["open_topics"] = len(brain.open_topics)
    except Exception:  # noqa: BLE001
        pass
    out["enabled"] = _os.environ.get(
        "HIPPOCAMPUS_HEALTH_REPORT", "1").strip().lower() not in (
            "0", "false", "no", "off")
    return out


#: L0 health-line threshold for ``session_start`` (dangling memory count).
#: 0 = report if there is even one.
L0_HEALTH_DANGLING_THRESHOLD = 0


def _l0_health_line(brain) -> dict | None:
    """Opening-pack L0 health bill — **only when it is actually off**.

    A field nobody reads is not observability: the dangling refs in a
    7000-record library were found by an external project, not by us.
    Returns ``None`` when healthy so the default opening pack stays
    byte-identical (its key order is pinned by golden tests); set
    ``HIPPOCAMPUS_L0_HEALTH=0`` to silence it entirely.

    Two signals, both shown only when something is **actually wrong**:
      * ``dangling_memories > 0``: L1 references to raw entries that are
        missing on disk (unrecoverable);
      * ``l0_pending`` reached the autosave threshold but was never
        persisted: that path is blocked or switched off (pending below the
        threshold is a normal intermediate state; reporting it is noise).
    """
    import os as _os
    if _os.environ.get("HIPPOCAMPUS_L0_HEALTH", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    pending = int(getattr(brain, "_l0_pending", 0) or 0)
    every = int(getattr(brain, "_l0_autosave_every", 0) or 0)
    dangling_ids = 0
    dangling_memories = 0
    try:
        from ..domain import stats_projection
        raw_ids = {r.get("id") for r in brain.raw_log}
        dangling, dangling_memories = stats_projection.dangling_report(
            brain.manager.store.iter_records(), raw_ids)
        dangling_ids = len(dangling)
    except Exception:  # noqa: BLE001 -- a failed health check must not block the opening
        dangling_memories = 0
    pending_stuck = bool(pending) and (
        every <= 0 or pending >= every)
    if dangling_memories <= L0_HEALTH_DANGLING_THRESHOLD and not pending_stuck:
        return None
    notes = []
    if dangling_memories > 0:
        notes.append(
            f"{dangling_memories} memories cite {dangling_ids} raw entries that "
            f"are missing from this library's L0 (source text unrecoverable)")
    if pending_stuck:
        notes.append(
            f"{pending} raw entries are memory-only (autosave threshold "
            f"{every} not reached or failed)")
    return {
        "level": "warn",
        "l0_pending": pending,
        "dangling_ids": dangling_ids,
        "dangling_memories": dangling_memories,
        "note": ("; ".join(notes) + ". Run scripts/audit_l0_dangling.py "
                 "--persist-dir <library> for the breakdown."),
    }


#: Cap on the topic ids named in a session-end nudge. A library with hundreds
#: of pending clusters must not turn the closing receipt into a dump.
KNOWLEDGE_TODO_MAX_IDS = 5


def _knowledge_todo(brain, session_memory_ids: list[str]) -> dict | None:
    """Name the knowledge topics a session should tidy, with one runnable step.

    The service deliberately does not summarise on its own (its production
    path embeds no LLM), but "leave it to the host's initiative" turned out to
    mean "never happens": real-library runs ended with only 3 summaries
    applied. This is a nudge, not an execution: it reuses the layer already
    refreshed by ``consolidate`` and never triggers first-time clustering on
    the hot path. Returns ``None`` when nothing is pending, so a healthy
    library keeps the exact same response shape and key order. Disable with
    ``HIPPOCAMPUS_KNOWLEDGE_TODO=0``.
    """
    import os as _os
    if _os.environ.get("HIPPOCAMPUS_KNOWLEDGE_TODO", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    try:
        td = _topic_defaults()
        if not td.layer_enabled:
            return None
        topics = getattr(brain, "knowledge", None) or {}
        if not topics:
            return None
        pending = topic_mod.pending_summary_ids(topics, defaults=td)
        if not pending:
            return None
        stats = topic_mod.host_summary_stats(topics)
        written = {str(m) for m in (session_memory_ids or [])}
        touched = [tid for tid in pending
                   if written & {str(m) for m in
                                 (topics.get(tid) or {}).get("member_ids") or []}]
        return {
            "pending_count": len(pending),
            "from_this_session": len(touched),
            "topic_ids": (touched or pending)[:KNOWLEDGE_TODO_MAX_IDS],
            "host_summaries": int(stats.get("host", 0)),
            "host_stale": int(stats.get("host_stale", 0)),
            "action": ("""hippo_extra(action="knowledge_topic", args='{"action":"brief_batch","batch":5}')"""),
            "then": ("""Write one conclusion per topic, then hippo_extra(action="knowledge_topic", args='{"action":"summarize_batch","items":"[...]"}')"""),
            "done_when": "pending_count == 0",
            "note": ("Summarising intelligence stays on the host; this "
                     "service only prepares material, validates and stores."),
        }
    except Exception:  # noqa: BLE001 - a nudge must never break session_end
        return None


class _SessionMixin:
    """B2-3 P3: session and writer cluster"""

    # ------------------------------------------------------------------
    # Session identity (signature attribution) routing
    # ------------------------------------------------------------------

    @property
    def current_session_id(self) -> str | None:
        """Current session id of the default route.

        On a shared connection, the server-generated sess_xxx identifies the
        **process**, not the caller, and is guaranteed to point at the wrong
        party. After the rework, each client gets its own route and this
        property exposes only the default route, keeping behavior bit-for-bit
        identical for unwired callers.
        """
        return self._current_sessions.get(DEFAULT_SESSION_KEY)



    @current_session_id.setter
    def current_session_id(self, value: str | None) -> None:
        if value is None:
            self._current_sessions.pop(DEFAULT_SESSION_KEY, None)
        else:
            self._current_sessions[DEFAULT_SESSION_KEY] = value



    def current_session_of(self, client: str | None = None) -> str | None:
        """Current session id of a given client (None if absent)."""
        return self._current_sessions.get(_session_key(client))



    def end_session_of(self, client: str | None = None) -> str | None:
        """Detach a given client's current session (returns the detached id). Touches only this route."""
        return self._current_sessions.pop(_session_key(client), None)



    def session_routes(self) -> dict[str, str]:
        """All currently active session routes (for self-check / self-report)."""
        return dict(self._current_sessions)



    def _note_session_memory(self, client: str | None,
                             memory_ids: list[str]) -> None:
        """Accumulate the memory ids written by this round of ingests onto the current session.

        session_end's L2 summary used to be pure free text, with no way to
        trace "which memories the last conclusion came from". Here, after the
        L1 write, the new ids are recorded on the current client's session,
        and at session_end they are attached to that summary as a whole so
        conclusions can point back to their source memories.

        Only **writes** are recorded, not retrieval references — recall's
        response already carries memory_id, so the host side can trace on its
        own; counting internal indirect recalls (assembled context, pinned
        slot) would only inject noise. When there is no active session
        (legacy hosts never call session_start), skip silently; behavior
        unchanged.
        """
        sid = self.current_session_of(client)
        if not sid or not memory_ids:
            return
        sess = self.sessions.get(sid)
        if sess is None:
            return
        bucket = sess.setdefault("memory_ids", [])
        for mid in memory_ids:
            if mid not in bucket:
                bucket.append(mid)




    def session_start(self, task: str = "", recent_limit: int = 10,
                      working_on: str = "", task_top_k: int = 5,
                      pin_limit: int = 0,
                      knowledge_limit: int = 0,
                      filter_tags: dict[str, Any] | None = None,
                      client: str | None = None,
                      overview: int = 0, status: int = 0,
                      task_profile: int = 0,
                      map_index: int = 0,
                      map_detail: int = 0,
                      user_scope: int = 0,
                      task_directory: int = 0,
                      user_access: str = "full") -> dict[str, Any]:
        """Start a session: return the pre-activation package (profile + recent summaries + open topics + open intents + recent memories).

        Args:
            task: description of the task for this session. **When non-empty,
                one recall is run with it** — previously task was only logged
                into the session record and never took part in retrieval, so
                the host announced its task at opening and the server answered
                with a pile of "recently written memories" unrelated to the
                task at hand.
            working_on: declare "what I am changing this time" (file / module /
                chapter, free text). Once recorded in the session registry,
                other sessions can call :meth:`active_sessions` to check
                whether someone is working in the same place before they start
                (this project has accumulated plenty of collision scars).
                **Registration only, not a lock** — no write is blocked.
            recent_limit: how many recent memories the pre-activation package
                carries; 0 disables it.
            pin_limit: how many entries the pinned slot carries (zcode
                addendum 19①). **0 = follow env
                ``HIPPOCAMPUS_PIN_BLOCK_LIMIT`` / default 3; negative =
                disabled**. Pinned entries are ordered with "relevant to
                task" first, then by importance within the same tier.
            task_top_k: how many entries the task pre-recall carries, default
                5; 0 disables it (falls back to the old behavior).
            knowledge_limit: how many knowledge points the knowledge-map slot
                carries. **0 = use env ``HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT``**,
                8 when neither is set; negative = omit the slot.
            filter_tags: structured tag filter, e.g. ``{"章": 43}``.
                Narrows the **task pre-recall** (``task_hits``) to one tag
                domain, with the same semantics as :meth:`recall`'s
                ``filter_tags`` (all keys AND, multi-value single key OR,
                ``str()`` normalization). **Applies only to the task
                pre-recall**; the recent / pinned / knowledge slots are fixed
                views and are unaffected. Empty = no domain restriction. When
                no tag matches, ``task_hits`` is empty (recall logs it as a
                miss).
            overview: project-overview card slot.
                **0 = follow env ``HIPPOCAMPUS_OVERVIEW_BLOCK`` (enabled when
                neither is set); positive = on; negative = off**. Same
                three-state semantics as pin_limit.
            status: current-status card slot. Three-state semantics same as
                overview; env is ``HIPPOCAMPUS_STATUS_BLOCK``.
            map_index: map index-line slot. Three-state semantics same as
                overview; env is ``HIPPOCAMPUS_MAP_INDEX_BLOCK``.
            map_detail: related-map full-text slot (L1 delivery). Three-state
                semantics same as overview; env is
                ``HIPPOCAMPUS_MAP_DETAIL_BLOCK``.
                **Content can only appear when task is non-empty** — the
                match criterion is task tokens/anchor entities ∩ map
                desc/entry names; no match means nothing is delivered (the
                counter-example constraint is guaranteed by matching itself).
            user_scope: user-level inheritance slot: how many preference-type
                memories of the **user-level library** (conventional name
                ``user``) to carry. **0 = follow env
                ``HIPPOCAMPUS_USER_SCOPE_LIMIT`` (3 when neither is set);
                negative = off** (same three-state parameter form as
                pin_limit; a negative env value also means off, see
                :meth:`resolve_user_scope_limit`). Selection: pinned first
                (threshold ignored) → threshold tier (importance ≥
                ``Config.user_scope_importance``, default 0.7) → within-group
                ``(-importance, -timestamp)`` descending → cap. A missing or
                not-yet-created user library **must not blow up the opening**:
                the structured ``user_scope`` key self-reports
                ``ready=false`` + ``reason``.
            task_directory: user-level task-directory slot: how many
                ``任务 → 库`` (task → library) mappings to carry (the
                user-level library's ``directory.json`` sidecar).
                **0 = follow env ``HIPPOCAMPUS_TASK_DIRECTORY_LIMIT``
                (8 when neither is set); negative = off** (three-state same as
                user_scope; when off, zero return keys and zero sidecar I/O).
                Capped by ``updated_at`` descending; older tasks beyond the
                cap are self-reported via ``task_directory.total``. A missing
                user library, a lost one, or a broken sidecar all **must not
                blow up the opening**: ``ready=false`` + ``reason``
                self-report (same form).

        Returns:
            ``block_task_hits`` is the text slot for prompt assembly;
            ``task_hits`` is the **full entries** of the same batch of hits
            (with ``memory_id`` / ``matched_by`` / ``tags``) for host
            back-reference and debugging — "was the entry I wanted actually
            fetched" is answered by this field.
            ``block_project_overview`` / ``block_project_status`` are compact
            projections of the library-level overview and current status
            (pure derivation, no retrieval); ``project_overview`` /
            ``project_status`` are their structured forms (for programmatic
            consumption and future UI rendering).
            ``block_task_profile`` is the task-level 「问题概况」 (problem
            profile): the knowledge points / entities the task touches, the
            matching open intents, and recent miss gaps in that domain,
            anchored on the already-computed ``task_hits`` (no re-retrieval);
            ``task_profile`` is its structured form. With an empty task the
            slot is an empty string.
            ``block_map_index`` is the map index line: which maps exist in
            the library, how many entries each has, how long since each was
            updated (⚠️ marks overdue). **Names and freshness only, zero
            content** — map entries are memories carrying the reserved tag
            keys (``地图``/``条目``); content goes through the L1 task hits /
            L2 ``recall(filter_tags=...)``.
            ``block_map_detail`` is L1 delivery: the sidecar maps hit by the
            task (maps.json, written via map_set) enter the package whole,
            capped at 20 lines per map plus a 「等 N 条」 (N more)
            self-report; ``map_detail`` is its structured form.
        """
        sid = _nid("sess")
        now = _now()
        key = _session_key(client)
        # C-1 (validation feedback 2026-11-01): the same client opening again
        # ⇒ the old session is terminated automatically. Previously old
        # sessions hung on active_sessions forever with ended_at=None (a
        # leak; the user library was observed accumulating zombies), and
        # their stale state also took part in the on-disk merge. Termination
        # over reuse: reuse would entangle the write attribution of two
        # conversations. The compaction-recovery scenario (the tool docstring
        # explicitly says it will call session_start again) benefits
        # naturally.
        old_sid = self._current_sessions.get(key)
        if (old_sid and old_sid in self.sessions
                and not self.sessions[old_sid].get("ended_at")):
            self.sessions[old_sid]["ended_at"] = now
            self.sessions[old_sid]["superseded_by"] = sid
            self._superseded_session = old_sid
        # Registered on this route only. On a shared connection, other
        # clients' currents are unaffected.
        self._current_sessions[key] = sid
        self.sessions[sid] = {
            "id": sid,
            "client": _session_key(client),
            "task": task,
            "working_on": working_on or "",
            "started_at": now,
            "last_seen_at": now,
            "ended_at": None,
            "summary": "",
            "open_topics": [],
            # Accumulate the memory ids this session's ingests wrote;
            # session_end attaches them to the L2 summary so "which memories
            # the conclusion came from" is traceable. Sessions loaded from
            # old libraries lack this field; reads fall back to .get.
            "memory_ids": [],
        }
        # task takes part in recall. Done after the session registration so
        # that recall's miss_log / current_session_id both attribute to this
        # session (useful for attribution).
        task_hits: list[dict[str, Any]] = []
        if task and task_top_k > 0:
            task_hits = self.recall(task, top_k=task_top_k, log_miss=True,
                                    client=client, filter_tags=filter_tags)
        # Pinned slot (addendum 19①): configurable count + task relevance
        # first. The task hit set reuses the recall result above; no
        # duplicate retrieval.
        pin_n = self.resolve_pin_limit(pin_limit)
        kn_n = self.resolve_knowledge_limit(knowledge_limit)
        us_n = self.resolve_user_scope_limit(user_scope)
        # Three-state count for the task-directory slot (same family as
        # user_scope).
        # 0 = follow env HIPPOCAMPUS_TASK_DIRECTORY_LIMIT (8 when neither is set); negative = off.
        td_n = self.resolve_task_directory_limit(task_directory)
        # M1b: take the config snapshot exactly once per opening. The five
        # groups of slot switches + the topic-layer master switch all read it,
        # avoiding a dozen load_config() calls on the same hot path (each one
        # a full round of env parsing).
        cfg = load_config()
        # Summarization-progress self-report: how many knowledge points
        # already have a host summary, how many have changed supporting
        # members. Laid out at opening so the host knows 「梳理到什么程度」
        # (how far the grooming has gotten) without an extra call.
        kn_stats = (topic_mod.host_summary_stats(self.knowledge)
                    if (kn_n > 0 and cfg.topic_layer)
                    else {})
        pinned_block = self._pinned_brief(
            pin_n, task=task,
            task_hit_ids={h["memory_id"] for h in task_hits} if task_hits else None,
        )
        recent_block = self.recent_memories(limit=recent_limit)
        knowledge_block = self._knowledge_brief(kn_n)
        # Project-overview card + status card: library-level directed info.
        # Pure derivation, no retrieval — same discipline as the knowledge-map
        # slot: read only the existing derived layers; the 2-3 s of first-time
        # grooming must never enter the hot path.
        ov_on = _slot_switch(overview, cfg.overview_block)
        st_on = _slot_switch(status, cfg.status_block)
        ov = self.project_overview() if ov_on else None
        st = self.project_status() if st_on else None
        # Task-level 「问题概况」 (problem profile). The anchor is the
        # task_hits already computed above — no re-retrieval. With an empty
        # task all four parts are empty and the renderer returns an empty
        # string, so the gate adds bool(task): no task, no "task profile" to
        # speak of.
        tp_on = _slot_switch(task_profile, cfg.task_profile_block) \
            and bool(task)
        tp = self.task_profile(task, task_hits) if tp_on else None
        # Map index line (L0 delivery). Names + counts + freshness only,
        # zero content — a library without maps renders as an empty string,
        # zero cost in non-user tokens.
        mi_on = _slot_switch(map_index, cfg.map_index_block)
        # Code switches map auto-verify on; **must run before the index line
        # is computed** — otherwise this round's index line reports last
        # round's drift/pending-verify counts. Interval gating (default 6h) +
        # budget + best-effort persistence all live in the method; libraries
        # without an auto recipe do zero disk reads and zero disk writes. No
        # exception may blow up the opening.
        if mi_on:
            try:
                self.map_auto_verify()
            except Exception:  # noqa: BLE001 -- patrol is a gift; the opening must not fail
                pass
            # Gate three: entries mentioned with 「另一个状态」 (a different
            # status) after their update ⇒ suspected stale. Also must run
            # before the index line is computed (otherwise this round's index
            # line reports last round's numbers). Pure in-memory string
            # matching, no file reads; interval gating/budget/locks live in
            # the method.
            try:
                self.map_mention_drift()
            except Exception:  # noqa: BLE001 -- same as above, must not break the opening
                pass
        mi = self.map_index() if mi_on else None
        # L1 full text of related maps. The anchor reuses task_profile's
        # entity names (no re-retrieval); with an empty task, same gate as
        # task_profile — no task, no 「任务相关的地图」 (task-related maps)
        # to speak of.
        md_on = _slot_switch(map_detail, cfg.map_detail_block) \
            and bool(task)
        md = self.map_detail(task, tp) if md_on else None
        # User-level inheritance slot. **Off (us_n <= 0) = zero trace**:
        # the three return keys and two receipt keys all disappear, and no
        # user-library I/O happens either (the literal implementation of
        # "injection can be switched off") — so resolve/scan below are never
        # even entered.
        us_info: dict[str, Any] | None = None
        us_active: int | None = None
        us_block: list[str] = []
        if us_n > 0:
            try:
                us_info, us_active = self._user_scope_brief(
                    us_n, cfg.user_scope_importance,
                    access=str(user_access or "full"))
                us_block = render_user_scope(us_info["memories"])
            except Exception as e:  # noqa: BLE001 -- the inheritance slot is a gift; it must not blow up the opening
                logger.warning("user-scope slot assembly failed (opening unaffected): %r", e)
                us_info = {"ready": False, "library": None, "limit": us_n,
                           "count": 0, "memories": [],
                           "reason": "user_scope_error"}
                us_active = None
        # User-level task-directory slot (任务 → 库, task → library
        # navigation).
        # **Off (td_n <= 0) = zero trace**: no user-library resolution, no
        # sidecar reads, zero return keys (same 「注入可关」 (injection can
        # be switched off) semantics as user_scope).
        td_info: dict[str, Any] | None = None
        td_block: list[str] = []
        if td_n > 0:
            try:
                td_info = self._task_directory_brief(td_n)
                td_block = render_task_directory(td_info["entries"])
            except Exception as e:  # noqa: BLE001 -- the directory is a gift; it must not blow up the opening
                logger.warning("task-directory slot assembly failed (opening unaffected): %r", e)
                td_info = {"ready": False, "library": None, "limit": td_n,
                           "count": 0, "total": 0, "entries": [],
                           "reason": "task_directory_error"}
        out = {
            "session_id": sid,
            "query": task,
            "working_on": working_on or "",
            # Task-related memories come first: this answers "for the thing
            # you asked me to do, what material do I have on hand"
            "block_task_hits": [
                f"{h['content'][:200]}（{h['memory_id'][:8]}）" for h in task_hits
            ],
            "task_hits": task_hits,
            # Echo the tag domain actually used by the task pre-recall
            # (null = not set, whole-library retrieval).
            "filter_tags": dict(filter_tags) if filter_tags else None,
            "block_profile": self._profile_brief(),
            "block_recent_summaries": [s["text"] for s in self.summaries[-3:]],
            "block_open_topics": [t["text"] for t in self.open_topics[-5:]],
            "block_pending_intents": [
                f"[{i['status']}] {i['text']}" for i in self.intents if i["status"] == "open"
            ][:5],
            "block_pinned_memories": pinned_block,
            # Self-report the actual count: the host learns from it "how many
            # pinned entries did not make it in"
            "pin_limit": pin_n,
            "pinned_total": self.pinned_total(),
            # The cheapest implementation of "where we left off last time":
            # the N most recently written entries, with source for
            # back-reference
            "block_recent_memories": recent_block,
            # Knowledge-map slot: which knowledge points this library has.
            # Listed, not retrieved — the lesson that summary-type content in
            # retrieval hurts (V9f: F1 .311 → .200) applies here too.
            "block_knowledge": knowledge_block,
            "knowledge_stats": kn_stats,
            "knowledge_limit": kn_n,
            # Project-overview card + status card: the compact projection of
            # 「这个库整体是什么、现在什么状况」 (what this library is as a
            # whole and what state it is in now) for the agent at every start
            # of work. The text slots can be pasted straight into the
            # prompt; the structured dicts are for programmatic consumption
            # and for rendering real infographics in a future human UI
            # (design doc §3.3, dual track). When off, the slots are empty
            # strings — the keys are always present, the host just checks
            # truthiness, no try/except needed.
            "block_project_overview": render_overview_block(ov) if ov else "",
            "block_project_status": render_status_block(st) if st else "",
            "project_overview": ov,
            "project_status": st,
            # Phase two: task-level 「问题概况」 (problem profile) — task_hits
            # are point hits; this is the structure the task touches (related
            # knowledge points/entities/matching open intents/recent miss
            # gaps in that domain). Same dual track as the two cards: text
            # slots paste into the prompt, dicts for host back-reference.
            "block_task_profile": render_task_profile_block(tp) if tp else "",
            "task_profile": tp,
            # Map index line. Zero content — for a map's content, L1 relies
            # on task hits (block_map_detail) and
            # L2 on the map_get exact query.
            "block_map_index": render_map_index_block(mi) if mi else "",
            "map_index": mi,
            # Full text of the sidecar maps hit by the task (L1). Same dual
            # track as the index line: text slot pastes into the prompt, dict
            # for host back-reference. No hit = empty string, zero cost in
            # non-user tokens.
            "block_map_detail": render_map_detail_block(md) if md else "",
            "map_detail": md,
            # While opening a new session, lay out 「还有谁在作业」 (who else
            # is working) — saves one call
            "active_sessions": self.active_sessions(),
            # **Delivery receipt** (borrowed from the predecessor design).
            # After the host's context compaction, the handoff **may carry
            # only this receipt's identity information** (session_id /
            # assembly time / per-slot counts / library epoch); copying the
            # opening pack's body into the compacted handoff is not allowed —
            # copied prose cannot prove "the system picture in my head is
            # still valid"; re-calling this tool once with the receipt is the
            # proper recovery path. knowledge_epoch / memory_active are the
            # criteria for "is the picture stale": if the library was written
            # by other sessions while the host was offline, these two numbers
            # will not match.
            "receipt": {
                "session_id": sid,
                "assembled_at": now,
                "blocks": {
                    "task_hits": len(task_hits),
                    "profile": len(self.profile),
                    "recent_summaries": len(self.summaries[-3:]),
                    "open_topics": len(self.open_topics[-5:]),
                    "pending_intents": sum(
                        1 for i in self.intents if i["status"] == "open"),
                    "pinned": len(pinned_block),
                    "recent_memories": len(recent_block),
                    "knowledge": len(knowledge_block),
                    "overview": 1 if ov else 0,
                    "status": 1 if st else 0,
                    "task_profile": 1 if tp else 0,
                    "map_index": 1 if mi else 0,
                    "map_detail": 1 if md else 0,
                },
                "memory_active": sum(
                    1 for r in self.manager.store.iter_records()
                    if r.state != MemoryState.ARCHIVED.value),
                "knowledge_epoch": (topic_mod.layer_epoch(self.knowledge)
                                    if cfg.topic_layer else None),
                "knowledge_topics": (len(self.knowledge)
                                     if cfg.topic_layer else None),
                # 🔴 The recovery protocol's **body** has moved to the server's
                # instructions (one-time delivery); only a pointer remains
                # here: session_start pays tokens on every return, while this
                # explanation is one-time for the host. The key keeps the same
                # semantics of "pointing at this tool + mentioning
                # session_id" (tests/test_s4_session_receipt.py R6a/R6b/R6c).
                "resume_hint": (
                    "After context compaction, call session_start again with session_id to rebuild the opening pack"
                    "(full recovery protocol in the session instructions)."),
            },
        }
        l0_health = _l0_health_line(self)
        if l0_health:
            # Appears only when there is **actually a debt/dangling refs**:
            # zero trace when healthy, key order verbatim identical to the
            # baseline (existing golden assertions depend on this).
            out["l0_health"] = l0_health
        if us_n > 0:
            # D4: the three return keys + two receipt keys appear only when
            # "on" (zero trace when off). Placed at the end of the body —
            # byte-identical key order with the baseline in the off state is
            # a hard constraint; the placement of newly added keys in the on
            # state is unconstrained, and tail placement disturbs the
            # existing key order the least.
            out["block_user_scope"] = us_block
            out["user_scope"] = us_info
            out["user_scope_limit"] = us_n
            out["receipt"]["blocks"]["user_scope"] = len(us_block)
            out["receipt"]["user_scope_active"] = us_active
        if td_n > 0:
            # The three return keys + two receipt keys appear only when "on"
            # (zero trace when off). Also placed at the tail: byte-identical
            # key order with the U1 baseline in the off state remains a hard
            # constraint.
            out["block_task_directory"] = td_block
            out["task_directory"] = td_info
            out["task_directory_limit"] = td_n
            out["receipt"]["blocks"]["task_directory"] = len(td_block)
            # Total directory entries (before the cap): the host uses it to
            # judge "how many old tasks have not entered the pack".
            out["receipt"]["task_directory_total"] = td_info.get("total", 0)
        if getattr(self, "_superseded_session", None):
            # C-1 self-report (zero trace: no key when no termination
            # happened).
            out["superseded_session"] = self._superseded_session
            self._superseded_session = None
        return out



    def _user_scope_brief(self, limit: int,
                          importance_min: float,
                          access: str = "full"
                          ) -> tuple[dict[str, Any], int | None]:
        """Lightweight read path of the user-level inheritance slot (U1/D2; U5's directory.json sidecar will reuse the same path — **do not add brain construction here**).

        Path: ``resolve_library("user")`` → ``get_backend(path).load_memories()``
        → ``MemoryRecord.from_dict`` → :func:`domain.user_scope.select_user_scope`.
        **No MemoryManager construction, no embedding model load** (zero
        25–33 s first load, zero 1.2 GB residency; under the sqlite env,
        get_backend opening a manifest-only user library for the first time
        cleanly initializes hippocampus.db — the U0 library-creation report
        accepted that behavior).

        D2b read-side tier (``access``): ``none`` → ``ready=false`` +
        ``reason="user_scope_access_none"``, **zero user-library I/O** (not
        even resolve is entered — a tier rejection must have no side-channel
        reads or writes); ``masked`` → sensitive:* cards filtered out before
        entering the selector; ``full`` (default) unchanged bit for bit.

        Failures never blow up the opening (the inheritance flow is a gift);
        structured self-report:
        - user library unresolvable → ``ready=false`` +
          ``reason="user_library_not_found"``
          (a deployment loss covering ``HIPPOCAMPUS_LIBRARY_ROOTS`` must be
          **visible**);
        - library present but unreadable/records broken → ``ready=false`` +
          ``reason="user_library_read_failed"``;
        - unexpected exceptions are caught by the caller's safety net
          (``reason="user_scope_error"``).

        Returns:
            ``(info, active)``. info has the fixed shape ``{ready, library,
            limit, count, memories[, reason]}`` (``library = {id, name,
            path}``, None when resolution failed); active = the count of
            **non-archived** entries in the user library (the same generation
            criterion as the receipt's ``memory_active``), None when the
            library is unavailable.
        """
        def _not_ready(reason: str,
                       lib: dict[str, Any] | None = None
                       ) -> tuple[dict[str, Any], int | None]:
            return ({"ready": False, "library": lib, "limit": limit,
                     "count": 0, "memories": [], "reason": reason}, None)

        # D2b: the none tier does zero user-library I/O (resolve is not even
        # entered) — a rejection must have no side channel.
        if access == "none":
            return _not_ready("user_scope_access_none")
        try:
            lib = resolve_library("user", libs=scan_libraries())
        except LibraryResolutionError:
            return _not_ready("user_library_not_found")
        if lib is None:  # "user" is non-empty so resolution always runs; None = truly missing (same defense as above)
            return _not_ready("user_library_not_found")
        meta = {"id": lib.library_id, "name": lib.name, "path": lib.path}
        try:
            data, _issue = get_backend(lib.path).load_memories()
            records = [MemoryRecord.from_dict(r)
                       for r in (data or {}).get("records") or []]
        except Exception:  # noqa: BLE001 -- a broken library file must not blow up the opening; self-report the read failure
            return _not_ready("user_library_read_failed", meta)
        if access == "masked":
            # D2b masked: sensitive:* cards stay out of the inheritance slot
            # (forged preferences must not be injected through the slot even
            # if their pending review lapsed — same policy as recall's
            # exclude_sensitive).
            records = [r for r in records
                       if not str(getattr(r, "sensitivity", "") or "")
                       .startswith("sensitive:")]
        entries = select_user_scope(
            records, limit=limit, importance_min=importance_min,
            active_state=MemoryState.ACTIVE.value)
        active = sum(
            1 for r in records if r.state != MemoryState.ARCHIVED.value)
        return ({"ready": True, "library": meta, "limit": limit,
                 "count": len(entries), "memories": entries}, active)



    def _task_directory_brief(self, limit: int) -> dict[str, Any]:
        """Lightweight read path of the user-level task-directory slot.

        Path: ``resolve_library("user")`` → ``library.task_directory
        .load_directory(user.path)`` → :func:`domain.task_directory
        .select_task_directory`. **No brain construction, no embedding model
        load** — the same path as :meth:`_user_scope_brief` (the directory is
        a sidecar file, not a memory; it should certainly not cost 25–33 s or
        1.2 GB).

        Failures never blow up the opening (the directory is a navigation
        gift); structured self-report:

        - user library unresolvable → ``reason="user_library_not_found"``
          (a deployment loss covering ``HIPPOCAMPUS_LIBRARY_ROOTS`` must be
          **visible**);
        - sidecar broken/unreadable → ``reason="task_directory_read_failed"``
          (**not treated as an empty directory**: an empty directory means
          「没登记过」 (never registered); a broken file is a different
          matter);
        - no entries in the directory is **not a failure**: ``ready=true`` +
          ``count=0``.

        Returns:
            ``{ready, library, limit, count, total, entries[, reason]}`` —
            ``library = {id, name, path}`` (None when resolution failed);
            ``total`` is the valid entry count **before the cap**
            (``count < total`` means 「还有没进包的老任务」 (old tasks that
            did not make it into the pack)).
        """
        def _not_ready(reason: str,
                       lib: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"ready": False, "library": lib, "limit": limit,
                    "count": 0, "total": 0, "entries": [], "reason": reason}

        try:
            lib = resolve_library("user", libs=scan_libraries())
        except LibraryResolutionError:
            return _not_ready("user_library_not_found")
        if lib is None:  # "user" is non-empty so resolution always runs; None = truly missing (same defense as above)
            return _not_ready("user_library_not_found")
        meta = {"id": lib.library_id, "name": lib.name, "path": lib.path}
        data, issue = task_directory_mod.load_directory(lib.path)
        if issue:
            return _not_ready(issue, meta)
        entries = select_task_directory(data.get("entries") or [], limit=limit)
        return {"ready": True, "library": meta, "limit": limit,
                "count": len(entries),
                "total": len(data.get("entries") or []),
                "entries": entries}

    def active_sessions(self, ttl: float = SESSION_TTL_SECONDS,
                        include_self: bool = True,
                        client: str | None = None) -> list[dict[str, Any]]:
        """List sessions still at work (B1) — "look before touching files".

        Args:
            ttl: survival window (seconds). Judged by ``last_seen_at``
                (falling back to ``started_at`` when absent); beyond it the
                session counts as expired and is no longer listed. Default
                :data:`SESSION_TTL_SECONDS` (2 hours).
            include_self: whether to include the current session, included by
                default (handy for confirming one's own registration
                succeeded).

        This is a **visibility** tool, not a lock: it blocks no writes and
        arbitrates nothing.
        """
        now = _now()
        out: list[dict[str, Any]] = []
        try:
            recycle_after = float(os.environ.get(
                ENV_SESSION_IDLE_RECYCLE, "") or SESSION_IDLE_RECYCLE)
        except ValueError:
            recycle_after = SESSION_IDLE_RECYCLE
        for s in self.sessions.values():
            if s.get("ended_at"):
                continue
            seen = s.get("last_seen_at") or s.get("started_at") or 0.0
            age = now - seen
            if recycle_after > 0 and age > recycle_after:
                # C-3 (validation feedback 2026-11-01): sessions that look
                # 「活跃」 (active) but exceeded the idle timeout are closed
                # automatically — state only, no data touched (marked
                # recycled for attribution), and the current pointer is
                # removed. Previously they were shown but never recycled, and
                # zombie sessions kept piling up.
                s["ended_at"] = now
                s["recycled"] = True
                if self._current_sessions.get(s.get("client")) == s.get("id"):
                    self._current_sessions.pop(s.get("client"), None)
                continue
            if age > ttl:
                continue
            if not include_self and s.get("id") == self.current_session_of(client):
                continue
            out.append({
                "session_id": s.get("id"),
                # Self-report the signature so the host sees at a glance
                # "whose session this is"
                "client": s.get("client") or DEFAULT_SESSION_KEY,
                "task": s.get("task") or "",
                "working_on": s.get("working_on") or "",
                "started_at": s.get("started_at"),
                "idle_seconds": round(age, 1),
                "is_self": s.get("id") == self.current_session_of(client),
            })
        out.sort(key=lambda x: x["idle_seconds"])
        return out



    def touch_session(self, session_id: str = "",
                      client: str | None = None) -> bool:
        """Refresh a session's survival time (heartbeat). Returns False if it does not exist."""
        sid = session_id or self.current_session_of(client)
        if not sid or sid not in self.sessions:
            return False
        s = self.sessions[sid]
        if s.get("ended_at"):
            return False
        s["last_seen_at"] = _now()
        return True




    def session_end(
        self,
        summary: str = "",
        open_topics: list[str] | None = None,
        client: str | None = None,
        close_topics: list[str] | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        """End a session: write the L2 summary, register open topics, conservatively update the L3 profile, trigger consolidation.

        Field feedback 2026-10-31: open_topics used to be pure append —
        duplicate registrations were not blocked, closed topics lingered
        forever, and the list grew without bound. Now close first, open
        after: ``close_topics`` removes by whitespace-stripped **exact**
        match (fuzzy similarity could close the wrong topic, so not done)
        and **raises a deletion tombstone** (the every-deletion-gets-a-
        tombstone discipline — without a tombstone the persisted merge would
        revive closed topics from disk; the P0 of the 2026-11-01 validation
        report); appends go through exact + similarity dedup first
        (threshold ``ENV_TOPIC_SIM_DEDUP``; similar hits are echoed in
        ``dedup_hits`` so the caller can decide whether to close the old
        entry). The global list also has a persist-side cap
        (``OPEN_TOPICS_MAX``, trims oldest); when reached,
        ``topics_cap_reached`` is returned.

        C-2 (validation feedback 2026-11-01): ``session_id`` exact close —
        only sessions of **this client** may be closed (no closing on behalf
        of others; a mismatch is an error with zero side effects), and when
        the target is not this route's current, current stays untouched.
        """
        if session_id:
            rec = self.sessions.get(session_id)
            if rec is None or rec.get("ended_at"):
                return {"error": f"session_id {session_id!r} does not exist or is already closed; "
                                 "no change was made"}
            if rec.get("client") != _session_key(client):
                return {"error": "session_id does not belong to this client (collecting on behalf of others is not allowed); "
                                 "no change was made"}
            sid = session_id
            # Remove the pointer only when the target is exactly this route's
            # current; closing a historical session leaves current alone.
            if self._current_sessions.get(_session_key(client)) == session_id:
                self._current_sessions.pop(_session_key(client), None)
        else:
            # Remove only this route's current; other clients' sessions
            # untouched.
            sid = self.end_session_of(client)
        session_mem_ids: list[str] = []
        if sid and sid in self.sessions:
            self.sessions[sid]["ended_at"] = _now()
            self.sessions[sid]["summary"] = summary
            self.sessions[sid]["open_topics"] = open_topics or []
            # Pull out the memory ids this session's ingests accumulated;
            # attached to the L2 summary below.
            session_mem_ids = list(self.sessions[sid].get("memory_ids") or [])
        if summary:
            self.summaries.append(
                {"id": _nid("sum"), "text": summary, "timestamp": _now(),
                 "session_id": sid, "memory_ids": session_mem_ids}
            )
        def _norm_topic(t: Any) -> str:
            return str(t or "").strip()

        closed = 0
        removed: list[dict[str, Any]] = []
        for t in close_topics or []:
            key = _norm_topic(t)
            if not key:
                continue
            before = len(self.open_topics)
            kept = [x for x in self.open_topics
                    if _norm_topic(x.get("text")) != key]
            if len(kept) < before:
                removed.extend(x for x in self.open_topics
                               if _norm_topic(x.get("text")) == key)
                self.open_topics = kept
                closed += before - len(kept)
        if removed:
            # Every deletion gets a tombstone (same discipline as miss_log):
            # without one, save's 「磁盘打底并集」 (disk-first union) would
            # merge the just-closed topics back from disk exactly as they
            # were — closed in memory, still open on disk, and the body would
            # still say saved:true (the P0 of the 2026-11-01 validation
            # report). Once the tombstone is persisted with save, other
            # processes' merges are also blocked by _apply_layer_deletions,
            # closing the cross-process revival path too.
            self.note_layer_deletion_items("brain_topics.json", removed)
        try:
            sim = float(os.environ.get(ENV_TOPIC_SIM_DEDUP, "")
                        or TOPIC_SIM_DEDUP)
        except ValueError:
            sim = TOPIC_SIM_DEDUP
        sim = max(0.0, min(1.0, sim))
        deduped = 0
        dedup_hits: list[dict[str, Any]] = []
        for t in open_topics or []:
            key = _norm_topic(t)
            if not key:
                continue
            # D (validation feedback 2026-11-01): exact hits are skipped as
            # before; similarity dedup is added on top — 「可见的模糊」
            # (visible fuzziness): a hit must be echoed in dedup_hits (with
            # the old entry's text), and the escape hatch = close the old
            # entry then register again. No forced-registration flag.
            dup = next((x for x in self.open_topics
                        if _norm_topic(x.get("text")) == key), None)
            if dup is None and sim > 0:
                for x in self.open_topics:
                    ratio = _topic_sim(key, _norm_topic(x.get("text")))
                    if ratio >= sim:
                        dup = x
                        dedup_hits.append({"input": key,
                                           "matched": x.get("text"),
                                           "sim": round(ratio, 3)})
                        break
            if dup is not None:
                deduped += 1
                continue
            self.open_topics.append({"text": t, "timestamp": _now(),
                                     "session_id": sid})

        # **Global finalization is triggered by the last one to leave**. When
        # active clients exist beyond this route's session (registered via
        # session_start/touch), profile extraction + whole-library
        # consolidation are deferred — they scan the whole library and touch
        # derived layers, which under the daemon would stomp on B's in-flight
        # session. No route ever registered a session (legacy hosts never
        # call session_start) ⇒ others=0, behavior bit-for-bit identical to
        # before the rework. save() is **not** deferred (the persistence
        # guarantee of addendum 4).
        others_active = len(self._current_sessions)
        profile_updates: list[dict[str, Any]] = []
        pruned: list[dict[str, Any]] = []
        report: dict[str, Any] | None = None
        if others_active > 0:
            self._pending_consolidate = True
        else:
            had_pending = self._pending_consolidate
            profile_updates = self._update_profile()
            # consolidate() refreshes the profile again internally, and by
            # then the pruned list has been cleared — keep a copy here so
            # session_end returns "everything this finalization pruned in
            # total".
            pruned = list(self.last_profile_pruned)
            report = self.consolidate()
            if had_pending:
                report = dict(report or {})
                report["covered_pending"] = True
            self._pending_consolidate = False

        # Persist to disk (coverage report · addendum 4): semantically the
        # agent treats session_end as "the wrap-up", yet it previously
        # **persisted nothing at all** — measured: host script ingests 2
        # entries → session_end → process exits → on-disk count rolls back,
        # 2 entries evaporate (only save_brain could salvage it). This now
        # force-saves at the end, and **failure must not fail session_end**:
        # under multiprocessing, concurrent write conflicts are routine;
        # throwing the exception at the host only makes it believe the
        # session never ended, so it retries → conflict again. Hence the
        # exception is swallowed and self-reported in the response.
        saved = True
        save_error = ""
        try:
            self.save()
        except Exception as exc:  # noqa: BLE001 -- see above: must be swallowed
            saved = False
            save_error = f"{type(exc).__name__}: {exc}"

        # Closing maintenance checklist. The companion to gate one (update at
        # write time) — embeds 「状态改了你同步图了吗」 (the status changed;
        # did you sync the map?) into the tail of the agent's existing
        # workflow, adding no new action. Lists only things **actually
        # wrong** (stale files / overdue maps); when nothing is wrong it is
        # an empty list, zero cost in non-user tokens. Both signals are pure
        # derivation; their failure does not affect the finalization itself.
        reminders: list[str] = []
        try:
            fr = self.file_registry_summary()
            if fr and (fr.get("stale") or fr.get("missing")):
                bits = []
                if fr.get("stale"):
                    bits.append("%d file-registry updates not synced (oldest %d days ago)"
                                % (fr["stale"], fr["oldest_stale_days"]))
                if fr.get("missing"):
                    bits.append("%d registered files have disappeared" % fr["missing"])
                reminders.append(
                    "File registry: %s. After re-ingesting the relevant content, run "
                    "scripts/file_registry.py sync <path> to mark it synced"
                    % " · ".join(bits))
            # Maps have two sources; reminders split by source give the right
            # action (legacy→update_tags, registry→map_set); registry maps
            # additionally get the gate-two drift reminder.
            mi = self.map_index()
            legacy_stale = [m["name"] for m in mi["maps"]
                            if m.get("stale")
                            and m.get("source") != "registry"]
            registry_stale = [m["name"] for m in mi["maps"]
                              if m.get("stale")
                              and m.get("source") in ("registry", "both")]
            drifted = [(m["name"], m["drifted"]) for m in mi["maps"]
                       if m.get("drifted")]
            if legacy_stale:
                reminders.append(
                    "Map(s) %s not updated for over %d days -- when a status changes, sync via "
                    "update_tags (one entry per write; do not write a second entry)"
                    % ("、".join(legacy_stale), MAP_STALE_DAYS))
            if registry_stale:
                reminders.append(
                    "Map(s) %s not updated for over %d days -- update per key via "
                    "map_set(name, key, status) (one entry per write; "
                    "do not write a second entry)"
                    % ("、".join(registry_stale), MAP_STALE_DAYS))
            if drifted:
                bits = "、".join("%s(%d)" % (n, c) for n, c in drifted)
                reminders.append(
                    "Map(s) %s have entries whose source of truth changed (drift detected by map_verify) -- "
                    "re-verify, then refresh with map_set to clear" % bits)
            # Gate three: after the update, some memory mentioned 「另一个状态」
            # (a different status). This is the weakest of the three signals
            # (someone said so ≠ the truth is so), so the action is written as
            # two steps: look at the evidence first, then decide ack
            # (unchanged) or map_set (changed). Maps that cannot be judged
            # (status lexicon too thin) do not nag here — that is a
            # configuration problem, and nagging means making people read an
            # unactionable hint every day (same discipline as gate four's
            # 「标太勤等于没标」 (flagging too often equals not flagging)).
            mentioned = [(m["name"], m["mentioned"]) for m in mi["maps"]
                         if m.get("mentioned")]
            if mentioned:
                reminders.append(
                    "Map(s) %s have %d entries mentioned with a different status after their last update (possibly stale, "
                    "evidence in entry mention.quote) -- if verified unchanged use scripts/maps.py ack "
                    "<map> <entry>; if truly changed use map_set"
                    % ("、".join("%s(%d)" % (n, c) for n, c in mentioned),
                       sum(c for _, c in mentioned)))
            # Phase C C1: auto-verify's two kinds of 「说不清」 (cannot tell).
            # Neither is drift — one is an entry value that fails to parse
            # (dynamic value/bad format), the other a map source that cannot
            # be read. Name the problem but do not prescribe: the fix depends
            # on the specifics (add extract_args / rework the auto recipe /
            # switch the map to manual maintenance), and the machine does not
            # make that call for the human.
            unresolved = [(m["name"], m["failed"]) for m in mi["maps"]
                          if m.get("failed")]
            if unresolved:
                reminders.append(
                    "Map(s) %s have %d entries auto-verify could not decide (dynamic values / unrecognized format) "
                    "-- see the entry note for reasons; to verify automatically, add an auto recipe; "
                    "otherwise maintain manually via map_set"
                    % ("、".join("%s(%d)" % (n, c) for n, c in unresolved),
                       sum(c for _, c in unresolved)))
            broken_auto = [(m["name"], m["auto_error"]) for m in mi["maps"]
                           if m.get("auto_error")]
            if broken_auto:
                # The wording does not hardcode 「读不到真相源」 (cannot read
                # the source of truth): C2's failure surface also includes
                # 「没建档/对账目录不存在/扫描不完整」 (not registered /
                # reconciliation directory missing / scan incomplete), which
                # likewise must not be treated as verified.
                reminders.append(
                    "Map %r: auto-verify could not complete -- %s"
                    % ("、".join(n for n, _ in broken_auto),
                       broken_auto[0][1]))
            # The three signal classes of directory reconciliation. Reported
            # separately from auto-verify's drift because the actions differ:
            # a missing file needs confirmation it was truly deleted, a
            # misplaced one needs its actual location found, an unregistered
            # one needs an entry added — collapsing them into one
            # 「有 N 条漂移」 (N drifts) says nothing.
            dir_miss = [(m["name"], m["dir_missing"]) for m in mi["maps"]
                        if m.get("dir_missing")]
            if dir_miss:
                reminders.append(
                    "地图%s有条目在数据目录里找不到（已删或改了名）——"
                    "确认真删就 map_set 改状态；整体搬过家就把 auto.root "
                    "指对再核对"
                    % "、".join("%s(%d)" % (n, c) for n, c in dir_miss))
            dir_conf = [(m["name"], m["dir_conflict"]) for m in mi["maps"]
                        if m.get("dir_conflict")]
            if dir_conf:
                reminders.append(
                    "地图%s有条目实际所在子目录与 status 不符——"
                    "map_get 看 drift.observed 是实际位置，改状态或搬文件"
                    % "、".join("%s(%d)" % (n, c) for n, c in dir_conf))
            unmatched = [(m["name"], m["unmatched"]) for m in mi["maps"]
                         if m.get("unmatched")]
            if unmatched:
                reminders.append(
                    "地图%s有目录里存在但没登记的键——核对后 map_set 补条目"
                    "（对账只报不填：世界多了东西，得有人告诉地图）"
                    % "、".join("%s(%d)" % (n, c) for n, c in unmatched))
            # Budget exceeded / some sources not fully verified: equally
            # 「这轮没核完」 (not finished this round), must not count as all clean
            truncated = [m["name"] for m in mi["maps"]
                         if m.get("auto_incomplete")
                         and not m.get("auto_error")]
            if truncated:
                reminders.append(
                    "地图「%s」的自动 verify 没核完（预算超限）——"
                    "下轮巡检会补，急的话 scripts/maps.py verify --auto"
                    % "、".join(truncated))
            # Review tightening: same-key conflicts across the two sources
            # (legacy old value vs registry new value). The list form of the
            # index line's 冲突N (conflict count) — without a prescribed
            # action the two statuses keep silently competing forever (the
            # leftover form of design §2's verify-or-not proposition).
            conflicted = [m["name"] for m in mi["maps"] if m.get("conflicts")]
            if conflicted:
                reminders.append(
                    "地图「%s」有条目在 legacy 记忆与 registry 侧车里状态"
                    "不一致——以 registry 为准：把 legacy 那条 update_tags "
                    "改成一致，或 scripts/maps.py import-legacy 迁移后退役 "
                    "legacy" % "、".join(conflicted))
        except Exception:  # noqa: BLE001 -- reminders are a gift; they must not break the finalization
            pass

        out = {
            "session_id": sid,
            # Memory ids written by this session and already attached to the
            # L2 summary (back-reference for "which memories the conclusion
            # came from").
            "memory_ids": session_mem_ids,
            "summaries": len(self.summaries),
            "open_topics": len(self.open_topics),
            # Field feedback 2026-10-31: topics closed / skipped by dedup in
            # this finalization (a dedup hit is not an error, but the agent
            # must see it, otherwise it looks like input being silently
            # swallowed).
            "topics_closed": closed,
            "topics_deduped": deduped,
            "profile_updates": profile_updates,
            "profile_pruned": pruned,
            "consolidate": report,
            # Self-report: active sessions exist beyond this route ⇒ global
            # finalization was deferred (honest echo)
            "finalized": report is not None,
            "deferred_consolidate": others_active > 0,
            "active_clients_remaining": others_active,
            # Closing maintenance checklist (empty list = nothing to sync).
            # One entry each for stale files / overdue maps, with the next
            # action — no new tools, just a reminder.
            "maintenance_reminders": reminders,
            # Persistence self-report (addendum 4): when saved=false the host
            # should explicitly call save_brain /
            # retry_save instead of believing it is already stored
            "saved": saved,
            "save_error": save_error,
        }
        if dedup_hits:
            # D: echo of the similarity-dedup hits (visible fuzziness — who
            # was blocked, which entry was matched).
            out["dedup_hits"] = dedup_hits
        if len(self.open_topics) >= OPEN_TOPICS_MAX:
            # D: flag when the cap is reached; do not let the registration
            # side silently rely on the merge side's "trim oldest" fallback.
            out["topics_cap_reached"] = True
        # B (validation feedback 2026-11-01): per-layer detail. saved still
        # means "no exception thrown"; failure details live in the failed
        # layers + save_error; skipped is a legitimate optimization, echoed
        # as-is.
        out["saved_layers"] = dict(getattr(self, "last_save_layers", {}) or {})
        # R2: passive health bill; opt out with HIPPOCAMPUS_HEALTH_REPORT=0
        import os as _os
        if _os.environ.get("HIPPOCAMPUS_HEALTH_REPORT", "1").strip().lower() not in (
                "0", "false", "no", "off"):
            out["health"] = _health_snapshot(self)
        # A4: name the topics this session touched and hand the host one
        # copy-paste command; absent when nothing is pending (zero noise).
        knowledge_todo = _knowledge_todo(self, session_mem_ids)
        if knowledge_todo:
            out["knowledge_todo"] = knowledge_todo
        return out




    def _touch_writer(self) -> None:
        """Write 「我是谁、我在哪、我改过几次」 (who I am, where I am, how many times I have written) into the library, and clear expired registrations.

        With a shared connection / multiple processes on one library, "how
        many processes are actually touching this library" used to be pure
        guesswork. The C11 data loss could only be forensically reconstructed
        precisely because no party left a name.

        Crashes / hard kills never unregister, so every writer sweeps the
        directory opportunistically while persisting its own state, clearing
        registrations whose **process is gone** or whose **heartbeat has
        expired** — cleanup is opportunistic, no dedicated thread.
        """
        if not self.persist_dir:
            return
        wdir = os.path.join(self.persist_dir, WRITER_DIRNAME)
        try:
            os.makedirs(wdir, exist_ok=True)
        except OSError:
            return
        try:
            host = _socket.gethostname() if _socket is not None else ""
        except Exception:  # noqa: BLE001 -- failing to get the hostname does not affect leaving the name
            host = ""
        info = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "host": host,
            "cwd": os.getcwd(),
            "started_at": self._created_at,
            "updated_at": time.time(),
            "saves": self._save_seq,
        }
        path = os.path.join(wdir, f"{os.getpid()}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except OSError:
            return
        self.writers_active, self.writers_stale_cleaned = scan_writers(
            wdir, load_config().writer_ttl)



    def _unregister_writer(self) -> None:
        """Erase this process's writer registration on exit.

        Normal exits must unregister, otherwise other scans will see a pid
        that no longer exists — :func:`scan_writers` does fall back to
        pid-liveness checks, but that is **after the fact**; a voluntary exit
        is cleaner. Crashes / SIGKILL never reach here; in those cases only
        the pid check remains.

        Multiple Brain instances with the same pid delete the same file; the
        second delete finding nothing is expected, so return silently.
        """
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, WRITER_DIRNAME,
                            f"{os.getpid()}.json")
        try:
            os.remove(path)
        except OSError:
            return
