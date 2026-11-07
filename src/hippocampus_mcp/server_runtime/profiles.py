# -*- coding: utf-8 -*-
"""Tool profile gating (the implementation body of server.py's profile
gating block).

Implemented here: profile state (``_TOOL_PROFILE`` / ``_TOOL_GROUPS`` /
``_PROFILE_GROUPS`` / ``_COMMON_TOOLS_DEFAULT`` / ``_COMMON_TOOLS`` and the
``HIPPOCAMPUS_COMMON_TOOLS`` env override), the decision function
``_tool_enabled``, and the profile-aware wrapper factory for
``server.tool``, :func:`make_gated_tool`.

🔴 Live-fetch discipline: ``_tool_enabled`` / ``make_gated_tool`` products
always read gating state via ``s._X`` (the re-exported bindings in the
server namespace) -- tests doing
``monkeypatch.setattr(server, "_TOOL_PROFILE", ...)" /
``server._TOOL_GROUPS = {...}" keep working as usual (lesson: snapshotting
state into a new module = monkeypatch silently stops working).
``_original_tool`` is assigned by server.py **before** replacing
``server.tool`` (the live-fetch source); the wrapper factory reads it only
**at call time**.

Profile semantics:
  Default common: the tool-surface token tax is a hard constraint -- the
  full surface's 46 tools ≈15.8k tokens permanently resident in the host
  context, while front-line observation shows daily sessions use only 6~9
  of them. Without explicit configuration, only the 10 entries of
  _COMMON_TOOLS are registered (+ the hippo_help/hippo_extra two
  dispatchers, 12 visible in total); the hidden 36 are **not one lost** --
  still callable via hippo_extra dispatch.
  To get all 46 back: add ``HIPPOCAMPUS_TOOL_PROFILE=full`` to the env in
  .mcp.json.
  Optional profiles (take effect only when set explicitly; unset/empty/
  pure-whitespace all fall to default common):
    common  -- the 10 daily high-frequency tools: read+write+session
               open/close+cross-library guard; the rest via hippo_extra
    core    -- retrieval/session/library base only (pure conversation)
    writer  -- core + write/update/forget/ingest
    lexicon -- core + lexicon config (low frequency)
    entity  -- core + entity graph
    admin   -- core + ops/export
    map     -- core + information map layer (map_set/get/list/verify)
    minimal -- core only
    full    -- all 46 (the historical default behavior)
  Unknown profile name → full-surface fallback: better to over-expose than
  to mis-disable; the cost of a typo is merely not saving tokens.
"""
from __future__ import annotations

from typing import Any

from hippocampus_mcp.config import load_config

#: Default common (see the module docstring).
_TOOL_PROFILE = (load_config().tool_profile or "").strip().lower() or "common"

# tool → its profile group (unlisted tools are handled as "conservatively
# kept", see _tool_enabled)
_TOOL_GROUPS: dict[str, str] = {
    # core
    "recall": "core", "get_memory_context": "core", "get_memory": "core",
    "list_memories": "core", "session_start": "core", "session_end": "core",
    "library_use": "core", "library_whoami": "core", "library_list": "core",
    "library_create": "core", "brain_stats": "core", "save_brain": "core",
    "knowledge_map": "core", "knowledge_topic": "core",
    # writer
    "store_memory": "writer", "update_memory": "writer", "forget_memory": "writer",
    "forget_by_query": "writer", "ingest": "writer", "ingest_batch": "writer",
    "list_sources": "writer", "update_tags": "writer",
    # lexicon
    "lexicon_add": "lexicon", "lexicon_suggest": "lexicon",
    "lexicon_alias_suggest": "lexicon",
    # entity
    "entity_profile": "entity", "entity_list": "entity", "relink_entities": "entity",
    "contradictions": "entity",
    # admin
    "export_memories": "admin", "reset_context": "admin", "miss_log": "admin",
    "consolidate": "admin", "report_feedback": "admin", "remember_intent": "admin",
    "complete_intent": "admin", "library_info": "admin",
    "get_stats": "admin", "save_state": "admin", "load_state": "admin",
    "retry_save": "admin",
    # Source drift audit (read-only). Placed in the admin group ⇒ zero token
    # increase for the common profile; reachable from both the full profile
    # and hippo_extra.
    "drift_audit": "admin",
    # Information map layer (state register). Placed in its own map group
    # ⇒ zero token increase for the common surface; reachable from full and
    # hippo_extra; setting HIPPOCAMPUS_TOOL_PROFILE=map explicitly gives the
    # narrow core+map surface.
    "map_set": "map", "map_get": "map",
    "map_list": "map", "map_verify": "map",
}
# each profile = {core} ∪ its own group; full is None (everything)
_PROFILE_GROUPS: dict[str, set[str] | None] = {
    "core": {"core"}, "minimal": {"core"},
    "writer": {"core", "writer"}, "lexicon": {"core", "lexicon"},
    "entity": {"core", "entity"}, "admin": {"core", "admin"},
    "map": {"core", "map"},
    "full": None,  # None = everything
}

# The common set (front-line observation: zcode daily usage never exceeds 8
# tools) -- to add/remove/change, edit this line only.
# It is **not** derived from {core} ∪ {writer}; it is an independently
# maintained "daily high-frequency" roster.
# Tools it excludes are not unavailable: the hidden portion is dispatched
# back in via hippo_extra.
_COMMON_TOOLS_DEFAULT = frozenset({
    # -- read/write backbone, 8 tools (the front-line observed daily surface) --
    "recall", "store_memory", "ingest", "session_start",
    "get_memory_context", "update_memory", "forget_memory", "list_memories",
    # -- cross-library guard --
    # The workspace guide file outranks HIPPOCAMPUS_PERSIST_DIR; on ambiguous
    # resolution it only warns and falls back to the default library
    # (to the user = silently writing into the wrong library). So "which
    # library am I in" must be askable at zero cost, otherwise the
    # "confirm the library when a session opens" discipline gets broken under
    # the common profile.
    # The cost is ~200 tokens; what it buys is never silently writing into
    # the wrong library.
    "library_whoami",
    # -- the maintenance half-loop --
    # The original roster was set by the "front-line observed daily 8" and
    # only covered the opening session_start, missing the other half of the
    # loop: without session_end ⇒ sessions never wrap up -- no L2 summary,
    # no L3 profile update, no consolidation trigger, and **no error the
    # whole time** (a silent absence).
    # If a maintenance tool is not on the resident surface, the agent simply
    # fails to think of calling it; this one must be visible.
    "session_end",
})
_COMMON_TOOLS: frozenset[str] = _COMMON_TOOLS_DEFAULT
# No code changes during the trial period: HIPPOCAMPUS_COMMON_TOOLS="recall,store_memory,..." overrides the common set directly.
_env_common = load_config().common_tools.strip()
if _env_common:
    _COMMON_TOOLS = frozenset(x.strip() for x in _env_common.split(",") if x.strip())


def _srv() -> Any:
    """The server module itself (the live-fetch entry; see the live-fetch
    discipline in the module docstring)."""
    from hippocampus_mcp import server as _s

    return _s


def _tool_enabled(name: str) -> bool:
    """Whether this tool registers as an MCP tool under the current profile.

    Gating state is fetched live via :func:`_srv` -- patching
    ``server._TOOL_PROFILE`` / ``server._COMMON_TOOLS`` /
    ``server._TOOL_GROUPS`` takes effect immediately.
    """
    s = _srv()
    if s._TOOL_PROFILE == "full":
        return True
    if s._TOOL_PROFILE == "common":
        return name in s._COMMON_TOOLS
    groups = s._PROFILE_GROUPS.get(s._TOOL_PROFILE)
    if groups is None:
        return True  # unknown profile → full fallback, avoiding mis-disabling
    grp = s._TOOL_GROUPS.get(name)
    if grp is None:
        return True  # unclassified tool → conservatively kept
    return grp in groups


def make_gated_tool() -> Any:
    """Build the profile-aware wrapper for ``server.tool`` (call before
    assigning server.tool).

    Tools switched off are not registered (the function definition remains,
    callable via internal imports and hippo_extra dispatch). FastMCP runtime
    dispatch goes through ``_tool_manager.call_tool`` and never calls
    ``self.tool`` again at runtime, so the wrapper is safe.

    🔴 ``_original_tool`` is fetched live from ``server._original_tool`` at
    **call time** (server.py must assign it before replacing
    ``server.tool``) -- tests can intercept registration by replacing
    ``server._original_tool``.
    """
    def _gated_tool(structured_output=False, **kwargs):
        s = _srv()
        original = s._original_tool
        if callable(structured_output):  # compat with bare server.tool (no parens) usage
            return original(structured_output, **kwargs)

        def deco(fn):
            if not s._tool_enabled(fn.__name__):
                return fn  # skip MCP registration; the definition remains, callable via internal imports
            # P3-4 telemetry: time every REGISTERED call (dispatcher calls
            # note themselves in server_runtime.dispatch). Pass-through
            # wrapper; functools.wraps keeps schema introspection intact.
            from . import telemetry as _telemetry
            return original(structured_output=structured_output, **kwargs)(
                _telemetry.timed(fn.__name__, fn))

        return deco

    return _gated_tool
