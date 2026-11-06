# -*- coding: utf-8 -*-
"""Pure staleness-window judgment for the retrieval side (common/L0: imported by both
engine and brain; zero in-package dependencies).

Design decision: **staleness forgetting never deletes actively** -- out-of-window
memories are not deleted, not re-stated, not archived; they are only kept out of
retrieval output. Judgment rules:

- ``window_seconds <= 0`` = off, never expires (the switch semantics live inside the
  judgment so callers need not check);
- ``pinned`` (importance>=0.7 auto-pins) is exempt -- pin's existing semantics is already
  "exempt from time decay" (``store/memory.sync_pin``); the window extends that same
  semantics;
- ``timestamp <= 0`` means "time unknown": staleness cannot be judged, so it is not
  blocked -- this keeps zero-timestamp legacy data left by migration/import paths from
  being silently swallowed library-wide by a default window;
- "out of window" takes strict greater-than: age == window is on the not-stale side.

The time base is passed in by the caller (constraint 4: ``_reference_time or time.time()``);
this module never reads the wall clock.
"""
from __future__ import annotations

import time

# Weekday abbreviation table. **Do not use ``time.strftime("%a")``** -- it reads the
# process locale; under a Chinese locale it emits "周五"/"星期六", and downstream places
# that match on English silently mis-match.
# (Same family of locale trap: ``datetime.strptime(..., "%a")`` raises ValueError
# outright when parsing LongMemEval's ``2023/05/20 (Sat)``; see the eval scripts' parsing
# functions.)
_WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def iso_date(timestamp: float) -> str | None:
    """Epoch seconds -> ``YYYY-MM-DD`` (UTC base); time unknown (``<=0``) returns None.

    Why UTC everywhere: external data's session dates look like ``2023/05/20 (Sat) 02:21``
    and **carry no timezone of their own**. The parser persists by UTC and the renderer
    reads back by UTC; the two ends agree, and the date does not drift a day depending on
    which timezone the machine running the eval is in.

    Returns None rather than ``"1970-01-01"``: a zero timestamp in the library means
    "time unknown" (legacy data from migration/import paths); rendering it as 1970 would
    be read as "actually the year 1970".
    """
    if timestamp <= 0:
        return None
    return time.strftime("%Y-%m-%d", time.gmtime(timestamp))


def weekday_abbr(timestamp: float) -> str | None:
    """Epoch seconds -> ``Mon``..``Sun`` (UTC base); time unknown returns None.

    Reason to exist: take "what weekday is this date" out of the model's hands. An LLM
    asked for the weekday of a given date will get it wrong, and this conversion is
    precisely the precondition for relative references like "last Friday".
    """
    if timestamp <= 0:
        return None
    return _WEEKDAY_ABBR[time.gmtime(timestamp).tm_wday]


def out_of_window(timestamp: float, pinned: bool, now: float,
                  window_seconds: float) -> bool:
    """age > window and not pinned -> stale (kept out of retrieval output).

    True = should be filtered; False = keep. **Pure function: zero I/O, zero state, zero writes**.
    """
    if window_seconds <= 0:
        return False
    if pinned:
        return False
    if timestamp <= 0:
        return False
    return (now - timestamp) > window_seconds


def parse_date_bound(text: str, *, end_of_day: bool = False) -> float:
    """``YYYY-MM-DD`` -> epoch seconds (UTC). Empty string returns ``0.0``, meaning that
    bound is unlimited.

    ``end_of_day=True`` takes that day's 23:59:59, for the before bound -- the user's
    mental model is "up to and including that day"; taking that day's 00:00 would exclude
    everything from that day.

    Invalid input **raises ValueError** rather than silently meaning "unlimited": a time
    window is a filter the user gave explicitly; failing silently makes results look like
    "the library truly has none" when in fact the parameter never took effect.
    This is a failure mode this project has stepped on repeatedly (see AGENTS.md's
    "silently wrong library" entry).
    """
    s = (text or "").strip()
    if not s:
        return 0.0
    from datetime import datetime, timezone
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt.timestamp()
    raise ValueError(
        f"unparseable date format: {text!r} (expected YYYY-MM-DD)")


def in_time_window(timestamp: float, after: float, before: float) -> bool:
    """`after <= timestamp <= before` test; ``<=0`` on either bound means that bound is unlimited.

    Deliberately the **opposite policy** of :func:`out_of_window`: there, ``timestamp<=0``
    (time unknown) is **kept** -- when the system blocks automatically it prefers to miss
    a block, fearing it would swallow zero-timestamp legacy data brought in by migration
    library-wide; here it is **excluded** -- the user gave an explicit interval, and an
    unknown time cannot prove the entry falls inside it. The two differ in both direction
    and rationale; do not "fix" one by referencing the other.
    """
    if after <= 0 and before <= 0:
        return True
    if timestamp <= 0:
        return False
    if after > 0 and timestamp < after:
        return False
    if before > 0 and timestamp > before:
        return False
    return True

