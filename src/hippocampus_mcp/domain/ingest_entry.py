# -*- coding: utf-8 -*-
"""M5 SessionC batch C1: the remaining inline pure segments of
``brain.core.ingest`` (extracted from core).

**Background**: ``ingest``'s five large pure-logic blocks have already
moved into ingest_flow / ingest_pipeline; the shell is a straight line of
"bookkeep -> write L0 -> decide -> split sentences -> write loop -> detect
-> neighbors -> echo". This batch also extracts the **three small pure
computations still left on that line** -- each sits on a semantic quirk
previously not pinned by its own unit test (clock None semantics /
session_id short-circuit / supersede tri-state); regression previously had
only whole-ingest end-to-end parity as its safety net; after the move each
quirk has its own assertion surface.

**Why the shell does not keep slimming**: ``ingest`` is called directly by
nearly every write-side suite (recon: 31 test files, 111 occurrences of
``.ingest(``); the remaining lines are almost all "state-write side
effects + composition points" -- ``raw_log.append`` / ``record_source_sig``
/ ``_guard_map_entry_dup`` / ``bump_daily`` are state writes, and the call
assembly of the five domain modules must stay in the shell as composition
points because "domain modules must not reference each other".

**Dependency direction**: this module imports only the standard library
(domain -> common-only discipline; here not even common is used -- the two
dependencies, clock and session table, are fetched fresh by the shell at
call time and injected).

**Replicated quirks (touching one is a behavior change; parity will catch
it)**:

1. ``resolve_timestamp``'s None semantics: **only ``timestamp is None``
   takes the clock** -- passing 0.0 is also an explicit value (epoch
   timestamps of historical replay really exist and must not be swallowed
   by "fill in the current time"); negatives likewise.
2. ``build_raw_entry``'s session_id short-circuit: when the explicitly
   passed session_id is truthy, the session table is **not consulted**
   (``resolve_session`` is not called); only a falsy one (None / empty
   string) falls back to ``current_session_of(client)`` -- when the
   fallback returns None there is **no further rescue**: session_id stays
   None.
3. ``build_raw_entry``'s source fallback: ``source or ""`` -- None and the
   empty string are the same shape (source is always a string in L0
   entries).
4. ``resolve_supersede_switch``'s tri-state: ``None`` = follow the instance
   switch ``supersede_enabled``; True/False taken as-is -- during batch
   import passing False per entry must not be flipped up by the instance
   switch.
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# Segment 1: None semantics of the inbound timestamp (old shell line 876)
# ======================================================================
def resolve_timestamp(
    timestamp: float | None,
    *,
    now: Callable[[], float],
) -> float:
    """Inbound timestamp normalization: only ``None`` takes the clock; the
    rest (including 0.0) is adopted explicitly.

    Args:
        timestamp: the caller's explicitly passed timestamp (may be None).
        now: ``brain_core._now`` itself (fetched fresh by the shell at call
            time -- monkeypatches and parity freezes both target it;
            injection keeps this layer swappable).

    Returns:
        The explicit timestamp or the clock's current value.
    """
    return timestamp if timestamp is not None else now()


# ======================================================================
# Segment 2: L0 raw record assembly (old shell lines 894-904; the append
# state write stays in the shell)
# ======================================================================
def build_raw_entry(
    *,
    raw_id: str,
    text: str,
    speaker: str,
    timestamp: float,
    session_id: str | None,
    source: str | None,
    resolve_session: Callable[[str | None], str | None],
    client: str | None,
) -> dict[str, Any]:
    """Assemble the L0 raw record dict (six keys; the key order is the
    ``raw_log`` entry shape).

    Args:
        raw_id: ``_nid("raw")`` product (uuid has side effects; the shell
            generates it and passes it in).
        text / speaker / timestamp: passed through verbatim, no copying, no
            transformation.
        session_id: explicit session id; when truthy the session table is
            **not consulted** (quirk 2).
        source: source label; ``or ""`` fallback (quirk 3).
        resolve_session: ``brain.current_session_of`` itself.
        client: the client identifier the session belongs to (passed
            verbatim to resolve_session).

    Returns:
        The six-key dict. The append is done by the shell -- this function
        only assembles, touching no state.
    """
    return {
        "id": raw_id,
        "text": text,
        "speaker": speaker,
        "timestamp": timestamp,
        "session_id": session_id or resolve_session(client),
        "source": source or "",
    }


# ======================================================================
# Segment 3: supersede switch tri-state normalization (old shell lines
# 966-969; the detection call stays in the shell)
# ======================================================================
def resolve_supersede_switch(
    detect_supersede: bool | None,
    *,
    instance_enabled: bool,
) -> bool:
    """Tri-state switch normalization for fact-update detection (quirk 4).

    Args:
        detect_supersede: this call's tri-state switch (None = follow the
            instance).
        instance_enabled: the ``brain.supersede_enabled`` instance switch.

    Returns:
        Whether to run supersede detection for this write.
    """
    return instance_enabled if detect_supersede is None else detect_supersede


# ======================================================================
# Segment 4: idempotent replay return body (old shell lines 925-934; the
# idem lookup's state read stays in the shell)
# ======================================================================
def duplicate_body(memory_id: str) -> dict[str, Any]:
    """The return body when the same ``idem_key`` replays.

    **Why extracted separately**: ``brain.core.ingest`` has an S1 structural
    guard (``tests/test_m5_batch19_ingest_flow.py``) pinning the shell at
    <=170 lines (body counted without docstrings). These 10 lines are pure
    dict construction with no state writes -- the last whole block of pure
    computation left in the shell; after the move the shell keeps only the
    two lines "look up -> return".

    Key names, order, and wording preserved verbatim
    (``tests/test_p0_origin_idem.py`` and the parity suite assert them
    as-is; changing a single character is a behavior change).

    Args:
        memory_id: the hit existing memory id (the return value of
            ``manager.idem_lookup``).

    Returns:
        The standard return body of an idempotent replay.
    """
    return {
        "status": "duplicate",
        "idempotent": True,
        "memory_id": memory_id,
        "memory_ids": [memory_id],
        "raw_id": None,
        "note": ("同一 idem_key 的重放：本次未写 L0 / L1，直接回既有"
                 "记忆 id。若这次其实是**新内容**，说明宿主复用了键"
                 "——请换键重发（幂等键应按请求唯一，不按内容）。"),
    }
