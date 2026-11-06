# -*- coding: utf-8 -*-
"""Time-series statistics helpers (pure-function layer extracted from brain/core.py).

All four functions are **pure computation**: state (daily_counts / miss_log / the reference
time) is passed in explicitly by the caller; inside the functions nothing touches brain
or disk. They were originally private methods of HippoBrain reading attributes like
``self.manager._reference_time``; after extraction the parameters became explicit, and
``project_overview`` / ``project_status`` (and the future same-origin logic in
BootstrapService) call them directly.

Why they live in ``common/``: the current callers are in brain (L3); once moved into
application (L3) later, the ownership stays unchanged; common is the L0 foundation
usable by both layers. This module imports only ``_now`` from ``common/constants``
and the stdlib time module, violating no dependency edge.

The reference-time semantics are verbatim identical to the brain side:
``reference_time or _now()`` — when the manager injected a reference time
(tests / replay / consolidation replay) use it, otherwise use the wall clock.
That ``or`` is part of the behavior: a caller passing ``None`` and passing nothing
at all are two different things, so no default value is provided — the caller is
forced to state its intent explicitly.
"""
from __future__ import annotations

import time
from typing import Any

from .constants import _now


def day_key(base: float, offset_days: int) -> str:
    """The ``YYYY-MM-D`` key for the day offset_days before base (same convention as _bump_daily)."""
    return time.strftime("%Y-%m-%d", time.localtime(base - offset_days * 86400))


def _daily_ingest(daily_counts: dict[str, dict[str, int]],
                  base: float, offset_days: int) -> int:
    """Single-day ingest count: a missing day-entry, missing field, or falsy value all count as 0."""
    return int(daily_counts.get(day_key(base, offset_days), {}).get("ingest", 0) or 0)


def recent_ingest(days: int,
                  daily_counts: dict[str, dict[str, int]],
                  reference_time: float | None) -> int:
    """Total writes over the last days days (sum of daily_counts.ingest; a missing day-entry counts as 0)."""
    base = reference_time or _now()
    return sum(_daily_ingest(daily_counts, base, i) for i in range(days))


def spark_series(days: int,
                 daily_counts: dict[str, dict[str, int]],
                 reference_time: float | None) -> list[int]:
    """Write series over the last days days (old→new), for sparkline rendering and structured returns."""
    base = reference_time or _now()
    return [_daily_ingest(daily_counts, base, i) for i in range(days - 1, -1, -1)]


def recent_miss_count(days: int,
                      miss_log: list[dict[str, Any]],
                      reference_time: float | None) -> int:
    """Recall miss count over the last days days (miss_log filtered by time, including the cutoff day itself).

    This is the **knowledge-gap signal**: queried but not present in the library — the
    FRAS S-field discipline. It cannot be derived from counts (the entry counts can look
    normal while gaps are everywhere), but without it the same pitfalls get stepped on again.
    """
    cutoff = (reference_time or _now()) - days * 86400
    return sum(
        1 for m in miss_log
        if float(m.get("ts") or 0.0) >= cutoff
    )
