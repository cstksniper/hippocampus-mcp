# -*- coding: utf-8 -*-
"""Base class for use-case services: composition-root-injected runtime +
library self-exposure + a unified envelope.

The injection surface (deliberately minimal; widened as needed):

* ``brain``     -- the target library's HippoBrain instance (``Any``: this
  package does not import brain; see package docstring discipline 1). It
  is still a "per-library facade"; method bodies will move into the
  individual Services one by one, and this injected object will slim down
  step by step into a pure port composition.
* ``library_self`` -- the library self-exposure ``{id, name, path}``,
  computed by the composition root from the adapter layer's resolution and
  passed in (services never touch library discovery / session stickiness,
  that connection-state machinery).
* ``persist_dir``  -- a config value read once by the composition root
  (echoed back by save_state). Not the whole Config: inject exactly what a
  use case needs, YAGNI.

``manager`` is not a second injection -- it is ``brain.manager`` (server's
``_manager_for`` has always been that equation); deriving it here only
saves one layer of typing.
"""
from __future__ import annotations

from typing import Any


class BaseService:
    """Base class of the use-case services. Every method returns a **dict
    awaiting serialization** (the envelope already assembled)."""

    def __init__(self, *, brain: Any, library_self: dict[str, str],
                 persist_dir: str = "",
                 user_brain: Any = None,
                 user_library_self: dict[str, str] | None = None,
                 sensitive_words: tuple[str, ...] = (),
                 promote_min_sessions: int = 2,
                 long_content_policy: str = "",
                 prospective_trigger: bool = True,
                 split_facts: Any = None,
                 recall_content_max_chars: int = 0,
                 recall_fusion_enabled: bool = True,
                 recall_query_fanout_enabled: bool = False,
                 user_write_quarantine: bool = False,
                 is_home_write: bool = True,
                 access_log_dir: str = "",
                 extra_libraries: list[tuple[dict[str, str], Any]] | None = None,
                 unregistered_names: list[str] | None = None,
                 user_access_resolver: Any = None) -> None:
        self.brain = brain
        self.manager = brain.manager
        self.library_self = library_self
        self.persist_dir = persist_dir
        # The user library's second brain for cross-library retrieval. Only
        # the ``_recall_service(scope=user/all)`` factory injects it (the
        # composition root resolves the library under the reserved name
        # user and builds it via the registry -- an explicit retrieval
        # intent pays the first-visit load cost); other services do not
        # pass it, None = this service has no cross-library capability
        # (if unused, do not touch it).
        self.user_brain = user_brain
        self.user_library_self = user_library_self
        # The sensitive double gate + promotion stability period.
        # application must not import config (dependency discipline) ⇒
        # injected by the composition root (server factories), read fresh
        # on every call: the word list **content** (the service layer reads
        # the file) and min_sessions -- consistent with load_config's
        # no-cache semantics. Services not using these two knobs are
        # unaffected.
        self.sensitive_words = sensitive_words
        self.promote_min_sessions = promote_min_sessions
        # Long-content handling policy (warn/split/reject; default warn =
        # status quo). application does not read config ⇒ injected by the
        # composition root, read live (same as sensitive_words).
        self.long_content_policy = long_content_policy or ""
        # Prospective-memory contextual trigger switch (the block_2
        # carried back by get_memory_context). As above, injected by the
        # composition root, read live; default True = active (the
        # registration-side trigger-set derivation lives in
        # brain.track_intent, the brain side reads config directly).
        # Only RecallService uses this knob; other services are unaffected.
        self.prospective_trigger = bool(prospective_trigger)
        # Per-record recall content character cap (truncation guard).
        # 0 = no truncation (status quo). application does not read config
        # ⇒ injected by the composition root, read live (same as
        # long_content_policy); only RecallService.recall uses it, all
        # other methods unchanged.
        self.recall_content_max_chars = int(recall_content_max_chars or 0)
        # R1: production recall shares the evaluation scoring pipeline.
        # application must not read config -> the composition root
        # injects load_config().recall_fusion on every call.
        self.recall_fusion_enabled = bool(recall_fusion_enabled)
        # V2: query fanout (keyword variant + RRF merge) for natural-sentence
        # queries. Default off — injected by the composition root the same
        # way; only RecallService._recall_one consumes it.
        self.recall_query_fanout_enabled = bool(recall_query_fanout_enabled)
        # The sentence splitter (the brain._split_facts body itself, with
        # its quirks such as "short sentence merged into the previous
        # one"). domain must not write a second copy of the splitting
        # truth ⇒ it can only be injected; when None, hard-split by length
        # and disclose.
        self.split_facts = split_facts
        # D2a write-side isolation trio (computed and injected fresh by the
        # composition-root factories; application makes no judgment):
        # user_write_quarantine -- this write's target = the user library
        #   and not the home library and the escape hatch closed (the
        #   result of domain/write_quarantine.should_quarantine):
        #   store/ingest cards are persisted with the quarantine flag +
        #   an audit event appended;
        # is_home_write -- the target is the home library (or the escape
        #   hatch is open): update_memory's clear_quarantine is open only
        #   for it (the elevation key is not in the agent's hands);
        # access_log_dir -- which library directory audit events land in
        #   (has a value only when the target = the user library).
        self.user_write_quarantine = bool(user_write_quarantine)
        self.is_home_write = bool(is_home_write)
        self.access_log_dir = access_log_dir or ""
        # B1 multi-library retrieval (only the _recall_service factory
        # injects it; None for the other services):
        # extra_libraries -- the periphery libraries' [(library_self,
        #   brain)] (already resolved by the composition root; resolution
        #   failure errors out explicitly in the factory, the service layer
        #   does no addressing);
        # unregistered_names -- names among the libraries= values **not in
        #   the task directory** (soft-check rule: resolvable by three-state
        #   addressing means usable; out-of-directory values are disclosed
        #   by the envelope, not blocked).
        self.extra_libraries = list(extra_libraries or [])
        self.unregistered_names = list(unregistered_names or [])
        # D2b read-side tier (injected only by the recall/maintain
        # factories): callable(client) → "full"/"masked"/"none"; a
        # non-user-library target returns "" (no tier context).
        # The decision truth lives in server._user_access_tier
        # (the connection-level binding state is over there).
        self.user_access_resolver = user_access_resolver

    def _ua(self, client: str | None) -> str:
        """The current call's user-library read-side tier. No resolver =
        non-user-library context (""); a resolver call failure = eligibility
        cannot be proven → fail-closed to none."""
        if self.user_access_resolver is None:
            return ""
        try:
            return str(self.user_access_resolver(client) or "")
        except Exception:  # noqa: BLE001 -- the elevation direction never auto-passes
            return "none"

    def _envelope(self, payload: Any) -> Any:
        """Insert the ``library`` first key into write-class return bodies
        (library self-exposure). **Only dicts get the key**.

        Non-dict returns (error-branch strings, lists, etc.) come back
        as-is -- no key added, so the error path's structure is not broken.
        Bit-for-bit identical to server.py's ``_with_library``
        (response assembly belongs to the use-case layer; tools only
        serialize).
        """
        if isinstance(payload, dict):
            return {"library": self.library_self, **payload}
        return payload
