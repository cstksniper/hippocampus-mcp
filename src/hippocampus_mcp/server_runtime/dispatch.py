# -*- coding: utf-8 -*-
"""Dispatcher (the progressive-disclosure scheduling shell; the implementation body of server.py's tool dispatch block).

Implemented here: ``_dispatch_map`` (action → underlying tool function),
``_hippo_help``, ``_hippo_extra``. **The registration block stays in
server.py** (``server._tool_manager.add_tool`` reads the server instance).

🔴 Fetch-live discipline: the tool name list is fetched live from
``s._TOOL_GROUPS`` (a patch changes it), and tool functions are fetched live
from ``vars(s)`` (read from the module namespace in real time) — **a snapshot
stand-in would swallow newly registered tools** (the negative-control
tests/test_m6_batch5_shell_split.py N1 pins down this difference).
What gating turns off is only the MCP registration; the module-level function
objects remain (``_gated_tool``'s deco returns fn), so a callable is always
obtainable here.

Design background (server.py's original comment, migrated along): changing
"46 tools' schemas entering the context every round" to "the common 9 resident
+ the rest dispatched by name" costs low-frequency operations one extra
hippo_help query and moves parameters to JSON. Registered only when
_TOOL_PROFILE != "full", for two reasons:
  1. Under full, all 46 tools are already directly exposed; the dispatcher is
     redundant overhead (about +400 token);
  2. Registering programmatically via _tool_manager.add_tool, rather than
     adding another decorator, keeps the fact "declared tool surface = 46"
     unchanged — tool-count assertions are hardcoded at 46.
"""
from __future__ import annotations

import json
from typing import Any


def _srv() -> Any:
    """The server module itself (fetch-live entry; see the fetch-live discipline in the module docstring)."""
    from hippocampus_mcp import server as _s

    return _s


def _dispatch_map() -> dict[str, Any]:
    """action → underlying tool function (both the name list and the function objects are fetched live).

    The list is fetched live from ``s._TOOL_GROUPS`` (after patching
    ``server._TOOL_GROUPS``, this function's list changes with it); the
    functions are fetched live from ``vars(s)`` (tools newly registered into
    the server namespace are visible immediately) — a snapshot stand-in on
    either side would swallow newly registered tools.
    """
    s = _srv()
    out: dict[str, Any] = {}
    for name in s._TOOL_GROUPS:
        fn = vars(s).get(name)
        if callable(fn):
            out[name] = fn
    return out


async def _hippo_help() -> str:
    """List the actions hidden under the current profile that can be called via hippo_extra, with their parameters."""
    # This is one of the **only places allowed** to pass a tier explicitly: one
    # dispatcher entry represents N tools, so the tier cannot be inferred from
    # this function's own name (this tool only lists a catalog; the store tier
    # suffices).
    # See tests/test_warmup.py W5.2 (the exception list is hardcoded there;
    # after this function was split out of server.py, server.py's top level no
    # longer has argument-bearing calls, so the exception list idles naturally
    # without going red).
    s = _srv()
    await s._await_warmup("store")
    import inspect

    hidden = [n for n in s._TOOL_GROUPS if not s._tool_enabled(n)]
    if not hidden:
        return "The current profile already exposes every tool; hippo_extra is unnecessary."
    lines = [f"This session has {len(hidden)} extension actions available through hippo_extra:", ""]
    for name in hidden:
        fn = vars(s).get(name)
        if not callable(fn):
            continue
        doc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
        brief = doc[0].strip() if doc else "(no description)"
        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):  # pragma: no cover
            params = []
        lines.append(f"- {name}({', '.join(params)}): {brief}")
    lines.append("")
    lines.append('Call: hippo_extra(action="name", args=\'{"argument": value}\')')
    return "\n".join(lines)


async def _hippo_extra(action: str, args: str = "{}") -> str:
    """Call tools not directly exposed in this session by action name (export / lexicon / entity graph / ops, etc.).

    Call hippo_help first for the action list and parameter names, then call
    via this tool; parameters are passed as a JSON object.
    """
    import inspect

    s = _srv()
    fn = s._dispatch_map().get(action)
    if fn is None:
        return (f"Unknown action {action!r}. Available actions:"
                f"{', '.join(sorted(s._dispatch_map()))}")
    # As above, a structural exception of the dispatcher: **the target tier
    # depends on the action** and can only be computed at runtime. The
    # zero-argument convention does not apply here (inferring from
    # _hippo_extra's own name would only ever yield one fixed tier).
    await s._await_warmup("store" if action in s.STORE_ONLY_TOOLS else "embed")
    text = (args or "").strip()
    kwargs: dict[str, Any] = {}
    if text:
        try:
            kwargs = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"args is not valid JSON: {exc}"
        if not isinstance(kwargs, dict):
            return "args must be a JSON object"
    try:
        allowed = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        allowed = None
    if allowed is not None:
        bad = [k for k in kwargs if k not in allowed]
        if bad:
            return f"{action} accepts no arguments {bad}; available arguments: {', '.join(sorted(allowed))}"
    try:
        return await fn(**kwargs)
    except TypeError as exc:
        return f"Call {action} failed (bad arguments): {exc}"
