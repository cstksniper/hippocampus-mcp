# -*- coding: utf-8 -*-
"""L1 rewrite history ledger (mem_history.jsonl, shape mirrors tag_ops.py verbatim).

Positioning: an **append-only log** sidecar; it takes no part in layer
merging, is not searched, does not enter the 5 slots / profile / topic
layers, and does not enter the default ``to_dict`` (zero change to the
return surface). It records the prior values of two kinds of rewrites:

- ``kind="update"``: ``update_memory`` changing content/importance
  (state-only changes are not recorded -- the old record itself is kept on
  the supersede/archive paths, and the state machine already handles the
  audit trail);
- ``kind="merge"``: the text of the side swallowed by a dedup merge (both
  "new long, old short" and "old long, new short"; identical contents are
  not recorded -- nothing was lost).

Biological counterpart (reconsolidation): after a rewrite the old trace is
"inaccessible but traceable" -- the ledger is the "traceable" part;
"accessible old versions" are handled by the supersede state machine, not
this ledger's job.

Why store/ and not brain: pure persistence plumbing (JSONL read/write +
ring trimming), zero domain logic; isomorphic with tag_ops/save_ledger
(the repo's third ledger, shapes converging). This module **imports no
Config/constants knobs**, honoring only explicit parameters (ring/trim_at
are passed in by the caller); the only imported constant is the file name
(no monkeypatch precedent in the repo; same exception as TAG_OPS_FILE).

State semantics (verbatim identical to tag_ops):
- ``last_seq`` is incremented before persisting ⇒ ids are not reused even
  when the write fails;
- ``lines`` increments only after a successful write; on trim failure
  (OSError) lines keeps its original value;
- ledger write failures **must not be swallowed** (logger.warning): phase
  one has no undo, and "what it was before the change" becoming forever
  unanswerable is something ops must see;
- ⚠️ with ``ring=0`` the slice is **everything** (Python's ``[-0:]``
  semantics) -- same quirk as tag_ops, deliberately kept; the caller
  (config default 200) never passes 0.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..common.constants import MEM_HISTORY_FILE

logger = logging.getLogger(__name__)


def history_path(persist_dir: str | None) -> str | None:
    """Ledger file path; returns None when there is no persist dir (pure in-memory brain)."""
    return (os.path.join(persist_dir, MEM_HISTORY_FILE)
            if persist_dir else None)


def read_history(persist_dir: str | None) -> list[dict[str, Any]]:
    """Read all ledger entries (bad lines skipped -- the ledger must not break auditing)."""
    path = history_path(persist_dir)
    if not path or not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Half line (crash mid-append): skip outright
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def scan_history(persist_dir: str | None) -> tuple[int, int]:
    """Scan the ledger once at startup / before first write: returns (line count, last seq)."""
    entries = read_history(persist_dir)
    last = 0
    for rec in entries:
        try:
            last = max(last, int(rec.get("seq") or 0))
        except (TypeError, ValueError):
            continue
    return len(entries), last


def append_history(persist_dir: str | None,
                   entry: dict[str, Any],
                   last_seq: int,
                   lines: int,
                   ring: int,
                   trim_at: int) -> tuple[str | None, int, int]:
    """Append one ledger entry (ring buffer trimming).

    Returns:
        ``(op_id, last_seq, lines)`` -- the state triple, which the caller
        uses to update itself. op_id None = not persisted (no directory or
        write failure); on write failure last_seq **has already been
        incremented** (prevents id reuse, same as tag_ops).
    """
    path = history_path(persist_dir)
    if not path:
        return None, last_seq, lines
    last_seq += 1
    entry["seq"] = last_seq
    entry["op_id"] = f"memhist_{last_seq}"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        lines += 1
        if lines > trim_at:
            lines = trim_history(persist_dir, ring, lines)
    except OSError as exc:
        logger.warning("[mem_history] ledger append failed: %s", exc)
        return None, last_seq, lines
    return entry["op_id"], last_seq, lines


def trim_history(persist_dir: str | None, ring: int, lines: int) -> int:
    """Trim down to the ring size (keeping the newest). Returns the line count after trimming.

    Returns ``lines`` unchanged when the count does not exceed the ring,
    there is no directory, or trimming fails (OSError). The ``ring=0``
    slice quirk is documented in the module docstring (same as tag_ops,
    deliberately kept).
    """
    path = history_path(persist_dir)
    if not path:
        return lines
    entries = read_history(persist_dir)
    if len(entries) <= ring:
        return lines
    keep = entries[-ring:]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in keep:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return len(keep)
    except OSError as exc:
        logger.warning("[mem_history] ledger trim failed: %s", exc)
        return lines
