# -*- coding: utf-8 -*-
"""MaintainService: library-state maintenance use cases (5 tools).

Tool surface (the **library-state maintenance** half of the original
"Maintain 11"):
export_memories / miss_log / brain_stats / save_brain / load_state.

.. note:: session_end went into the services with BootstrapService (it is
   the session lifecycle wrap-up; semantically it belongs to the
   opening/closing group). The other 5 library_* tools **do not enter this
   class** -- they are not bound to a single brain (``library_info``
   deliberately avoids constructing a HippoBrain so as not to wait 25-33
   seconds for the embedding model); see ``library_admin_service``.

The "who parses arguments" split: this class's 5 tools have **no JSON
protocol parsing** (the only shape conversion is ``export_memories``'s
``source or None`` / ``state or None`` empty-string normalization), so there
is no ordering question of "parsing before or after library resolution" --
orchestration moves into the service wholesale; the factory call is the
library resolution, and the error ordering is unchanged.

Envelopes (the library self-report first key) **keep the current shape per
tool**:

* **With** envelope: only ``save_brain`` (currently via ``_with_library``,
  including the conflict branch);
* **Without**: export_memories / miss_log / brain_stats / load_state
  (currently a direct ``json.dumps``).

The ``warmup`` section of ``brain_stats`` is deliberately produced by an
**injected probe** (the composition root passes ``server.warmup_status``):
it is **process-level facility state** (how far the backend warm-up got),
not a library fact -- returning the two mixed in one dict is a product
decision (one fewer round trip), but the provenance must be visible, hence
the constructor injection marks it explicitly (same class as
``persist_dir``: an environment value, not a domain value).
"""
from __future__ import annotations

from typing import Any, Callable

from ..store import ConcurrentWriteError
from .base import BaseService


class MaintainService(BaseService):
    """Maintenance use cases: export / miss logging / stats / persist /
    reload (policy and algorithms temporarily live in brain; to be lifted up
    later)."""

    def __init__(self, *, brain: Any, library_self: dict[str, str],
                 persist_dir: str = "",
                 probe: Callable[[], dict[str, Any]] | None = None,
                 **base_kwargs: Any) -> None:
        """``probe`` = zero-wait readiness probe (composition root injects
        ``server.warmup_status``).

        Defaults to ``None`` rather than being required: this service ships
        with 5 tools from day one, and only brain_stats actually needs the
        probe. Optional so unit tests can construct minimally (new it
        directly without a probe); the cost is that without a probe the
        ``warmup`` section of ``brain_stats`` degrades to ``{}`` -- the
        composition root always injects, so this degraded path is never
        exercised in production.

        ``**base_kwargs`` are passed through to the base class (D2b's
        ``user_access_resolver`` and other optional injections arrive this
        way; this service does not consume them itself).
        """
        super().__init__(brain=brain, library_self=library_self,
                         persist_dir=persist_dir, **base_kwargs)
        self.probe = probe

    # ---------------------------------------------------------------- export

    def export_memories(self, *, since_ts: float, source: str, state: str,
                        limit: int, client: str = "") -> dict[str, Any]:
        """Incremental export by time/source/state (no vectors; can be
        persisted to disk elsewhere as-is). **No** envelope.

        The boundary is "inclusive of equals" (with several records sharing a
        timestamp, the last one of that batch repeats once; dedupe by id) --
        this semantics is written in the tool docstring; the service side
        only orchestrates and neither restates nor rewrites it.

        D2b: refused by tier when the target is the user library (design
        §4.2: export is a bulk channel; for the masked tier "either refuse or
        restrict to non-sensitive" we chose refuse -- the ledger does not
        balance for per-line filtering in bulk, while the leak surface stays
        the same size).
        """
        tier = self._ua(client)
        if tier in ("masked", "none"):
            return {"error": (
                f"User-library export is refused from this workspace (read level {tier}）：导出是"
                "批量通道，masked/none 档一律拒绝。请在 home 工作区（日常"
                "助手宿主）会话里导出。")}
        rows = self.brain.export_memories(since=since_ts, source=source or None,
                                          state=state or None, limit=limit)
        return {"count": len(rows), "memories": rows}

    # ---------------------------------------------------------------- miss log

    def miss_log(self, *, limit: int, clear: bool) -> dict[str, Any]:
        """View/clear the trail of queries that failed to recall. **No**
        envelope.

        ``clear`` goes through ``brain.clear_miss_log()`` rather than wiping
        the list directly: clearing is a deletion, so a tombstone must be
        laid first, or another process's old snapshot merges it back the
        moment it persists.
        """
        brain = self.brain
        if clear:
            return {"cleared": brain.clear_miss_log()}
        rows = sorted(brain.miss_log, key=lambda r: -r.get("ts", 0.0))
        return {
            "count": len(rows),
            "threshold": brain.miss_log_threshold,
            "enabled": brain.miss_log_enabled,
            "entries": rows[:limit] if limit > 0 else rows,
        }

    # ---------------------------------------------------------------- stats

    def brain_stats(self, *, daily_days: int) -> dict[str, Any]:
        """Size stats for the four layers + index layer, doubling as a
        backend readiness probe. **No** envelope.

        The ``warmup`` section comes from the injected probe: this tool
        depends only on inventory-level data (answers in seconds), so it can
        also answer "is the backend ready" -- when ``ready=false``, the other
        vector-dependent tools return "retry later".
        """
        out = self.brain.stats(daily_days=daily_days)
        out["warmup"] = self.probe() if self.probe else {}
        return out

    # ---------------------------------------------------------------- persist / reload

    def save_brain(self) -> dict[str, Any]:
        """Persist the whole memory base (equivalent to save_state). **With**
        envelope.

        The ConcurrentWriteError conflict body shares word-for-word wording
        with WriteService.save_state (the hint points to load_state) -- the
        two tools are two entry points of the same semantics to begin with.
        """
        try:
            self.brain.save()
        except ConcurrentWriteError as e:
            return self._envelope(
                {"status": "conflict", "detail": str(e),
                 "hint": "Another process rewrote the same memory library. Call load_state for the latest state, then redo this change."})
        return self._envelope({"status": "saved", "persist_dir": self.persist_dir})

    def load_state(self) -> dict[str, Any]:
        """Restore the full state from disk, overwriting memory. **No**
        envelope.

        A failed load is not silent: when ``store.last_load_issue`` has a
        value, the quarantined file and the reason are returned to the caller
        along with it (because "loadsucceeded" and "loaded but lost a layer"
        must not look the same in the response body).
        """
        brain = self.brain
        brain.load()
        issue = brain.manager.store.last_load_issue
        payload: dict[str, Any] = {"status": "loaded", "stats": brain.stats()}
        if issue:
            payload["load_issue"] = issue
        return payload
