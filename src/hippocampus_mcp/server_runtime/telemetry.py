# -*- coding: utf-8 -*-
"""Per-tool call telemetry (naturalization P3-4, 2026-11-07).

Motivation: every trimming/rollout decision (tool-surface reduction, the
adaptive-recall and usage-recency default-on decisions) kept stalling on
"no per-tool call distribution". The 2026-10-31 audit was a one-off manual
analysis; this module makes the data STANDING.

Design (deliberately boring):

- One log line per call, JSON-ish body, logger ``hippocampus.telemetry``:
  ``[telemetry] {"tool": "...", "dur_ms": 12, "ok": true}``. The daemon log
  is already the operational sink (bridge/launcher lines live there); a
  separate metrics file would add a second write path with no reader yet.
- An in-memory :data:`COUNTERS` aggregate for in-process introspection
  (tests; future brain_stats exposure if a reader shows up).
- BOTH call surfaces are covered: direct MCP calls (wrapped at profile
  registration in ``profiles.make_gated_tool``) and dispatcher calls
  (``server_runtime.dispatch._hippo_extra`` notes the action itself).
  Telemetry never raises and never blocks the call: the timing wrapper is
  pass-through by construction, and :func:`note` swallows everything.

Privacy: tool NAME + duration + ok-flag only. No arguments, no content,
no session ids — counts must be safe to leave in logs.
"""
from __future__ import annotations

import functools
import json
import logging
import threading
import time
from collections import Counter
from typing import Any, Callable

logger = logging.getLogger("hippocampus.telemetry")

#: tool name -> call count (in-process aggregate; daemon lifetime scope).
COUNTERS: Counter[str] = Counter()
_LOCK = threading.Lock()


def note(tool: str, dur_ms: float, ok: bool = True) -> None:
    """Record one call. Never raises (telemetry must not break tools)."""
    try:
        with _LOCK:
            COUNTERS[tool] += 1
        logger.info("[telemetry] %s",
                    json.dumps({"tool": tool, "dur_ms": round(dur_ms, 1),
                                "ok": bool(ok)}, ensure_ascii=False))
    except Exception:  # noqa: BLE001 — telemetry must never break a tool
        pass


def timed(tool: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Pass-through timing wrapper (functools.wraps keeps the FastMCP
    schema introspection working: __name__/__doc__/__wrapped__ intact)."""
    @functools.wraps(fn)
    async def _inner(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        ok = True
        try:
            return await fn(*args, **kwargs)
        except Exception:
            ok = False
            raise
        finally:
            note(tool, (time.perf_counter() - t0) * 1000.0, ok=ok)
    return _inner


def snapshot() -> dict[str, int]:
    """Copy of the in-process counters (tests / future introspection)."""
    with _LOCK:
        return dict(COUNTERS)
