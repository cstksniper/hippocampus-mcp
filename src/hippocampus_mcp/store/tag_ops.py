# -*- coding: utf-8 -*-
"""update_tags change ledger (tag_ops.jsonl, extracted from brain/core.py).

Positioning (inheriting the comment on TAG_OPS_FILE in common/constants.py):
it is an **append-only stream** (like save_ledger.jsonl), not part of layer
merging; a ring buffer prevents unbounded growth. Each **execution** records a
full before/after snapshot; undo rolls back only when "current value ==
recorded after" (fail-closed), so a ledger write failure **must not be
swallowed** — losing the undo basis must be made known to the host.

Why store/ and not brain: this is pure persistence infrastructure (JSONL
read/write + ring trimming) with zero domain logic; brain keeps only three thin
wrappers (read/scan/append) as the adapter layer, and knobs such as the ring
size are still passed in by brain at call time — the ``core.TAG_OPS_RING``
name that tests monkeypatch must keep being fetched at the call site, so this
module **imports no Config/constants knobs** and only accepts explicit
arguments (the sole exception is the filename TAG_OPS_FILE, which has no
monkeypatch precedent anywhere in the repo).

State semantics (verbatim identical to brain's original implementation):
- ``last_seq`` is incremented before persisting ⇒ a failed write never reuses
  an id ("what it was before the change" must not point at a record that could
  be overwritten);
- ``lines`` is incremented only after a successful persist; trimming triggers
  only when ``lines > trim_at``, and on trim failure (OSError) lines keeps its
  original value — the ledger's line count may transiently diverge from disk
  reality, but never runs away beyond trim_at for long.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..common.constants import TAG_OPS_FILE

logger = logging.getLogger(__name__)


def ops_path(persist_dir: str | None) -> str | None:
    """Ledger file path; None when there is no persist dir (pure in-memory brain)."""
    return (os.path.join(persist_dir, TAG_OPS_FILE)
            if persist_dir else None)


def read_ops(persist_dir: str | None) -> list[dict[str, Any]]:
    """Read all ledger entries (bad lines skipped — the ledger itself must not make undo unusable)."""
    path = ops_path(persist_dir)
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
                    continue  # half line (crash mid-append): skip directly
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def scan_ops(persist_dir: str | None) -> tuple[int, int]:
    """Scan the ledger once at startup: return (line count, last seq)."""
    ops = read_ops(persist_dir)
    last = 0
    for rec in ops:
        try:
            last = max(last, int(rec.get("seq") or 0))
        except (TypeError, ValueError):
            continue
    return len(ops), last


def append_ops(persist_dir: str | None,
               entry: dict[str, Any],
               last_seq: int,
               lines: int,
               ring: int,
               trim_at: int) -> tuple[str | None, int, int]:
    """Append one ledger entry (ring buffer trimming).

    Args:
        persist_dir: persist directory; None/empty = pure in-memory, skip
            persisting outright.
        entry: entry content; the ``seq`` / ``op_id`` keys are overwritten by
            this function.
        last_seq / lines: ledger state held by the caller (brain's instance
            attributes).
        ring: ring size (keep the latest ring entries).
        trim_at: trimming triggers only when the line count exceeds this.

    Returns:
        ``(op_id, last_seq, lines)`` — the state triple, which the caller uses
        to update itself. op_id None = not persisted (no directory or write
        failure); note that last_seq **has already been incremented** even on
        write failure (preventing id reuse).
    """
    path = ops_path(persist_dir)
    if not path:
        return None, last_seq, lines
    last_seq += 1
    entry["seq"] = last_seq
    entry["op_id"] = f"tagops_{last_seq}"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        lines += 1
        if lines > trim_at:
            lines = trim_ops(persist_dir, ring, lines)
    except OSError as exc:
        # A ledger write failure must not be swallowed: losing the undo basis
        # must be made known to the host (the in-memory tags are already
        # changed, but "what they were before the change" is now untraceable).
        logger.warning("[tag_ops] ledger append failed: %s", exc)
        return None, last_seq, lines
    return entry["op_id"], last_seq, lines


def trim_ops(persist_dir: str | None, ring: int, lines: int) -> int:
    """Trim to the ring size (keeping the newest). Returns the post-trim line count.

    Returns ``lines`` unchanged when the line count is within the ring, when
    there is no directory, or when trimming fails (OSError).
    ⚠️ With ``ring=0``, ``ops[-0:]`` is the **whole** list (Python slice
    semantics) — this is the original implementation's behavior, kept
    deliberately; the caller (config default 200) never passes 0.
    """
    path = ops_path(persist_dir)
    if not path:
        return lines
    ops = read_ops(persist_dir)
    if len(ops) <= ring:
        return lines
    keep = ops[-ring:]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in keep:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return len(keep)
    except OSError as exc:
        logger.warning("[tag_ops] ledger trim failed: %s", exc)
        return lines
