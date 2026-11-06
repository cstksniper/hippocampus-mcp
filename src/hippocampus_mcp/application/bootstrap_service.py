# -*- coding: utf-8 -*-
"""BootstrapService: opening package and session lifecycle (5 tools).

Tool surface: session_start / session_end /
remember_intent / complete_intent / report_feedback + get_stats.

On why get_stats belongs here: of the 46-tool five-way classification it is
the only one not claimed by the first four groups, and semantically it fits
-- "check system state at session open". MaintainService's brain_stats is
the maintenance-side statistics; the two do not overlap.

Methods correspond verbatim to the server.py tool bodies (minus warmup /
brain fetching / ``json.dumps``; ``_with_library`` → base-class
``_envelope``): relocation only, behavior unchanged, equivalence guaranteed
by the regression gate; bodies like ``brain.session_start`` will be folded
into this class later.
"""
from __future__ import annotations

from typing import Any

from ..common.params import filter_applied
from .base import BaseService
from .tags import check_tag_keys


class BootstrapService(BaseService):
    """Opening use cases: pre-activation package (receipt / overview /
    profile / map index) + open intents + feedback loop."""

    # ---------------------------------------------------------------- Opening package

    def session_start(self, *, task: str, recent_limit: int, working_on: str,
                      task_top_k: int, pin_limit: int,
                      filter_tags: dict[str, Any] | None,
                      client: str,
                      user_scope: int = 0,
                      task_directory: int = 0,
                      user_access: str = "full") -> dict[str, Any]:
        """First step of every session: fetch the pre-activation package
        (after compaction, rebuild via session_id).

        ``filter_tags`` is parsed by the adapter layer before being passed in
        (its JSON syntax error body carries no envelope, see the split
        explanation in the write_service module docstring); the tag-key
        allowlist check needs this library's schema, which is a use-case
        responsibility, done here.

        ``user_scope`` is passed through to ``brain.session_start``:
        the user-level inheritance slot count, three states
        (0 = per env / positive = count / negative = off).

        ``task_directory`` is passed through the same way: the user-level
        task directory slot count, three states (same family as
        ``user_scope``; 0 = per env / positive = count / negative = off).
        Assembly and the self-report shape live entirely on the brain side
        (``_task_directory_brief``); the use-case layer only ferries
        parameters -- the same discipline as ``user_scope``.

        ``user_access`` (D2b) is also passed through: this connection's
        read-side tier for the user library (computed by the composition
        root and injected); the inheritance slot filters/refuses by tier;
        the default full = status quo.
        """
        parsed_tags, key_info = check_tag_keys(self.brain, filter_tags)
        if key_info and key_info.get("error"):
            # Note: this error body does **not** carry the library first key
            # (bit-for-bit identical to the current tool).
            return key_info
        started = self.brain.session_start(
            task=task, recent_limit=recent_limit,
            working_on=working_on, task_top_k=task_top_k,
            pin_limit=pin_limit, filter_tags=parsed_tags,
            client=client or None, user_scope=user_scope,
            task_directory=task_directory, user_access=user_access)
        # Soft degradation must be visible -- otherwise the stripped keys
        # were silently swallowed.
        if key_info and key_info.get("degraded"):
            started["filter_applied"] = filter_applied(parsed_tags, None, key_info)
        return self._envelope(started)

    def session_end(self, *, summary: str, open_topics: list[str],
                    client: str, close_topics: list[str] | None = None,
                    session_id: str = "") -> dict[str, Any]:
        """Session wrap-up: write the summary, register open topics, update
        the profile, trigger consolidation."""
        return self._envelope(self.brain.session_end(
            summary=summary, open_topics=open_topics,
            close_topics=close_topics, client=client or None,
            session_id=session_id))

    # ---------------------------------------------------------------- Open intents

    def remember_intent(self, *, text: str, due: str) -> dict[str, Any]:
        """Register an open intent (reminded once in the next session, not
        nagged repeatedly)."""
        return self._envelope({"message": self.brain.track_intent(text, due)})

    def complete_intent(self, *, keyword: str) -> dict[str, Any]:
        """Mark an open intent complete (matched by keyword)."""
        return self._envelope({"message": self.brain.resolve_intent(keyword)})

    # ---------------------------------------------------------------- Feedback and state

    def report_feedback(self, *, memory_id: str, useful: bool) -> dict[str, Any]:
        """Useful raises importance; not useful only lowers priority, never
        the weight."""
        ok = self.brain.report_feedback(memory_id, useful)
        return self._envelope(
            {"status": "ok" if ok else "not_found", "memory_id": memory_id})

    def get_stats(self) -> dict[str, Any]:
        """System state statistics. Does **not** carry the library first key
        (bit-for-bit identical to the current tool)."""
        return self.manager.get_stats()
