# -*- coding: utf-8 -*-
"""Save ledger (save_ledger.jsonl, extracted from brain/core.py).

Positioning (inheriting the LEDGER_FILENAME comment in common/constants.py):
one save = ``begin`` + ``end`` rows; the exception path records ``fail``.
**Append-only is a hard requirement** -- a full rewrite would turn the ledger
itself into a new data-loss point. A ring buffer prevents unbounded growth.

Why store/ and not brain: this is a pure persistence facility (JSONL I/O +
ring trimming + corrupt-file quarantine + boot reconciliation decision) with
zero domain logic; brain keeps only thin wrappers (path / read tail / append /
trim / reconcile) as the adapter layer. The state semantics of the ledger line
count (``lines``) and last entry (``last``) are written back to instance
attributes by the wrapper layer; this module returns pure results only:

- :func:`append` returns ``(persisted or not, line count after the append)`` --
  on write failure the line count stays put and the last entry is not updated
  ("what it was before the change" must not point at a record that never got
  written);
- :func:`trim` returns the new line count or ``None`` (``None`` = the caller
  keeps its current value: the three "cannot count" cases -- no persist_dir /
  read failure / write failure).

Knob surface (same as tag_ops): tests monkeypatch ``core.LEDGER_RING`` to
control the ring size (test_save_ledger L5) => this module's functions accept
only an explicit ``ring`` argument and **import no Config/constants knobs**;
the wrapper layer fetches them from the core namespace at call time
(``LEDGER_TRIM_AT`` has no monkeypatch precedent but is fetched along the same
path -- one less special case). The only exceptions are the file name
``LEDGER_FILENAME`` and the tail length ``_LEDGER_TAIL_KEEP`` -- no
monkeypatch precedent anywhere in the repo, imported from common.constants.

The ring=0 Python slicing quirk (``lines[-0:]`` = everything) is preserved
as-is under the "move it, don't change it" discipline: at ring=0 a trim is a
verbatim rewrite (line count unchanged).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Sequence

from ..common.constants import LEDGER_FILENAME, _LEDGER_TAIL_KEEP


def ledger_path(persist_dir: str | None) -> str | None:
    """Ledger file path; returns None without a persist_dir (pure in-memory
    brain)."""
    if not persist_dir:
        return None
    return os.path.join(persist_dir, LEDGER_FILENAME)


def read_tail(persist_dir: str | None) -> tuple[dict[str, Any] | None, int, list]:
    """Read the ledger: last entry + total line count + the trailing few rows
    (for boot reconciliation).

    Bad lines are skipped line by line; a bad file is quarantined by convention
    as ``.corrupt-<ts>``; startup is **never blocked** (design section 8, risk
    9: a missing/corrupt ledger reports unknown and continues). The tail keeps
    only the most recent ``_LEDGER_TAIL_KEEP`` rows: a fail row needs the
    same-seq begin from the look-back to know "what the save declared it would
    write"; a full read is unnecessary.
    """
    path = ledger_path(persist_dir)
    if not path or not os.path.exists(path):
        return None, 0, []
    last: dict[str, Any] | None = None
    tail: list[dict[str, Any]] = []
    n = 0
    bad = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    last = json.loads(line)
                    tail.append(last)
                    if len(tail) > _LEDGER_TAIL_KEEP:
                        tail.pop(0)
                except ValueError:
                    bad = True      # skip the bad line, but still count it
    except OSError:
        return None, 0, []
    if bad:
        try:
            os.replace(path, path + ".corrupt-" + str(int(time.time())))
        except OSError:
            pass
    return last, n, tail


def append(persist_dir: str | None, entry: dict[str, Any], lines: int,
           trim_at: int, ring: int) -> tuple[bool, int]:
    """Append one row (trimming to the ring when trim_at is reached).

    Returns ``(persisted or not, line count after the append)``: on write
    failure (OSError) the line count stays put and the caller must not update
    the last entry; if the trim's read/write fails the line count keeps the
    post-append value (a brief divergence between the ledger line count and
    disk reality is tolerated, but it must never run away beyond trim_at).
    """
    path = ledger_path(persist_dir)
    if not path:
        return False, lines
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
    except OSError:
        return False, lines
    lines += 1
    if lines >= trim_at:
        trimmed = trim(persist_dir, ring)
        if trimmed is not None:
            lines = trimmed
    return True, lines


def trim(persist_dir: str | None, ring: int) -> int | None:
    """Ring buffer: keep only the most recent ring rows, bounding the ledger's
    size.

    Returns the new line count; ``None`` = the caller keeps its current value
    (no persist_dir / read failure / write failure). When the line count is
    within the ring, returns a recount from disk reality (the self-healing
    path).
    """
    path = ledger_path(persist_dir)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if len(lines) <= ring:
        return len(lines)
    keep = lines[-ring:]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(keep) + "\n")
        os.replace(tmp, path)
    except OSError:
        return None
    return len(keep)


def reconcile_on_boot(
    last: dict[str, Any],
    tail: list,
    layer_names: Sequence[str],
    layer_fp: Callable[[str], str],
) -> dict[str, Any]:
    """Boot reconciliation: report "which layer changed but was never written
    down" (lost_since_last_save).

    After load, memory == disk, so "memory fingerprint vs ledger fingerprint"
    is the same as "disk vs ledger". Three decidable cases (quadrant
    decision):

    * Last entry ``end`` with layer fingerprints -> layers whose disk content
      differs from the ledger go into ``disk_mismatch`` (an external tool
      changed the content / write corruption; a multi-writer process's normal
      persist appends its own end, so no false positives);
    * Last entry ``begin`` (no same-seq end/fail) -> the previous save **did
      not finish**: layers whose disk fingerprint differs from the begin
      fingerprint => declared a write but the disk no longer holds that
      content => reported as lost;
    * Last entry ``fail`` -> look back through the tail for the same-seq
      begin, decide as in the previous case, and carry the error out.

    A row without ``layer_fps`` (old format / fingerprints off) ->
    ``layer_fps_absent``, no content judgment. ``layer_fp`` fetches the current
    fingerprint by layer name and is called **lazily** (computed only when the
    row carries that layer's fingerprint) -- injected by the wrapper layer from
    the brain instance (this module does not import brain). Dual-clock
    discipline: no new time fields added; lost's time semantics reference the
    ledger row's at_wall/at_mono. **Known limitation**: a process killed with
    uncommitted changes that never even started a save (no begin) => the ledger
    cannot know -- that is F11's territory.

    Returns:
        The reconciliation report dict: ``phase`` / ``seq`` / ``at_wall`` /
        ``error`` passed through as-is + the decision results + a
        human-readable note. Warning logs and the ``_boot_reconcile`` state
        write-back are the wrapper layer's responsibility (this module has no
        logging side effects).
    """
    phase = last.get("phase")
    seq = last.get("seq")
    rec: dict[str, Any] = {
        "phase": phase, "seq": seq, "at_wall": last.get("at_wall"),
        "lost_since_last_save": None, "disk_mismatch": [],
        "error": last.get("error"), "layer_fps_absent": False, "note": "",
    }
    mismatch: list[str] = []
    lost: dict[str, Any] | None = None

    if phase == "end":
        end_fps = last.get("layer_fps")
        if not isinstance(end_fps, dict):
            rec["layer_fps_absent"] = True
            rec["note"] = "The last ledger entry has no layer fingerprints (old format or fingerprinting off); content reconciliation skipped."
        else:
            for name in layer_names:
                fp = end_fps.get(name)
                if fp is not None and layer_fp(name) != fp:
                    mismatch.append(name)
            rec["disk_mismatch"] = mismatch
            rec["note"] = (
                f"Layers whose disk content mismatches the last ledger entry (end seq={seq}): {', '.join(mismatch)} "
                "-- changed by an external tool or corrupted; the ledger has no matching end." if mismatch else
                "Disk content matches the last ledger entry; no unexplained changes.")
    elif phase in ("begin", "fail"):
        begin = last if phase == "begin" else None
        if phase == "fail":
            for row in reversed(tail):
                if (row.get("phase") == "begin" and row.get("seq") == seq
                        and (row.get("kind") or "") == (last.get("kind") or "")):
                    # kind-aware: V1-a autosave rows carry their own seq space
                    # (kind-tagged), so a same-seq begin from the OTHER row
                    # family must not be mistaken for this fail's begin. Old
                    # rows have no kind key -> both sides normalize to "" and
                    # match exactly as before.
                    begin = row
                    break
        begin_fps = begin.get("layer_fps") if isinstance(begin, dict) else None
        if not isinstance(begin_fps, dict):
            rec["layer_fps_absent"] = True
            rec["note"] = (f"The last ledger entry is {phase} (the previous save did not finish), "
                           "and the row carries no layer fingerprints, so the lost layers cannot be identified.")
        else:
            lost = {}
            counts = begin.get("layers") if isinstance(begin, dict) else {}
            for name in layer_names:
                fp = begin_fps.get(name)
                if fp is not None and layer_fp(name) != fp:
                    lost[name] = (counts or {}).get(name)
            rec["lost_since_last_save"] = lost
            if lost:
                detail = ", ".join(f"{k}({v} records)" for k, v in lost.items())
                rec["note"] = (f"The previous save (seq={seq}) did not finish; the following layers declared writes "
                               f"but the disk no longer holds that content: {detail}.")
            else:
                rec["note"] = (f"The previous save (seq={seq}) did not finish, but every layer's disk content matches "
                               "the begin declaration -- the write likely completed and only the end row was lost.")
    else:
        rec["note"] = f"The last ledger entry has unknown phase={phase}; reconciliation skipped."

    return rec
