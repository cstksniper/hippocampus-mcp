# -*- coding: utf-8 -*-
"""Layer fingerprint projection (extracted from brain/core.py).

These functions only **read** in-memory state and project it into outward
conclusions; they write nothing:

- :func:`pending_layer_names` -- the layers that at this moment "have
  in-memory changes not yet persisted" (the memory-side criterion of the
  quadrant view);
- :func:`compute_all_layer_fps` -- content fingerprints of all layers (the
  ``layer_fps`` field of the ledger begin/end lines);
- :func:`classify_layer_quadrants` -- F14-3 per-layer quadrant assignment;
- :func:`ledger_view` -- F14 ledger projection (``brain_stats.save_ledger``).

Why store/ rather than brain: the criteria are all the single pair
"layer file / layer attribute", zero domain logic; brain keeps only thin
wrappers (binding ``_LAYER_ATTRS``, the fingerprint function and the disk
stamp comparison on its side as the adapter). Same domain as
:mod:`hippocampus_mcp.store.ledger`.

Port discipline:

* **No Config knobs are imported**. The fingerprint switch ``layer_fp`` is
  fetched live from the ``core`` namespace by the wrapper at call time and
  passed in as a parameter (required for the monkeypatch surface);
* The fingerprint computation itself is injected as the
  ``fingerprint_of(attr)`` callback -- brain passes
  ``lambda a: self._layer_fingerprint(getattr(self, a))``, a runtime lookup,
  so test patches of ``_layer_fingerprint`` still apply;
* The clock is injected too (``now`` / ``stamp_of``); the ledger projection's
  ``interval_s`` relies entirely on ``time.monotonic()`` differences --
  subtracting wall clocks goes negative on NTP rollbacks.

Pre-existing quirks are preserved as-is under "move only, change nothing":

* When ``_layer_written_fp`` is empty (new library), :func:`pending_layer_names`
  returns an empty list -- not "no changes", but "cannot tell"; this is said
  plainly by ``pending_note``;
* In :func:`classify_layer_quadrants`, ``name in stamped`` short-circuits
  first; **only** layers present in the stamp table ever ask
  ``changed_by_other``; when ``persist_dir`` is None,
  ``os.path.join("", name)`` yields a relative path (hence almost always
  False);
* When ``last`` is not a dict, the ledger projection returns the whole field
  set with ``level=unknown`` and **never reports 0** -- 0 would be read as
  "just saved", a lie worse than unknown.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Collection, Mapping, Sequence


def pending_layer_names(
    layer_attrs: Mapping[str, str],
    written_fp: Mapping[str, str],
    fingerprint_of: Callable[[str], str],
) -> list[str]:
    """Layer names that at this moment **have in-memory changes not yet
    persisted**.

    Returns empty when ``written_fp`` is empty (this process has never
    persisted) -- not "no changes", but "cannot tell". A layer never written
    does not count as "changed" either; it counts as "not yet built".
    """
    if not written_fp:
        return []
    out: list[str] = []
    for name, attr in layer_attrs.items():
        if name not in written_fp:
            continue      # never persisted: not "changed", but "not yet built"
        if fingerprint_of(attr) != written_fp[name]:
            out.append(name)
    return out


def compute_all_layer_fps(
    layer_attrs: Mapping[str, str],
    fingerprint_of: Callable[[str], str],
    enabled: bool,
) -> dict[str, str]:
    """Content fingerprints of all layers (the ``layer_fps`` field of the
    ledger begin/end lines).

    Returns an empty dict when the fingerprint switch is off -- the ledger
    line then carries no ``layer_fps``, and boot reconciliation degrades
    honestly on that basis (no guessing).
    """
    if not enabled:
        return {}
    return {name: fingerprint_of(attr) for name, attr in layer_attrs.items()}


def classify_layer_quadrants(
    layer_attrs: Mapping[str, str],
    written_fp: Mapping[str, str],
    stamped: Collection[str],
    persist_dir: str | None,
    fingerprint_of: Callable[[str], str],
    changed_by_other: Callable[[str, str], bool],
) -> dict[str, str]:
    """F14-3 quadrant classification (wording 2): current assignment per
    layer.

    Memory side (**primary criterion**) = current in-memory fingerprint vs
    this process's last persisted fingerprint;
    disk side (**corroborating evidence only**) = whether the layer file
    stamp is still there (mtime never convicts alone -- touch without a
    content change keeps the fingerprint unchanged and still counts as
    clean, merely recorded as an external_write).
    This process never wrote the layer → ``unknown`` ("not yet built", not
    "changed but unsaved").
    """
    out: dict[str, str] = {}
    for name, attr in layer_attrs.items():
        written = written_fp.get(name)
        if written is None:
            out[name] = "unknown"
            continue
        dirty = fingerprint_of(attr) != written
        # The ``name in stamped`` short-circuit is pre-existing behavior: no
        # need to ask the disk about layers absent from the stamp table.
        changed = name in stamped and changed_by_other(
            name, os.path.join(persist_dir or "", name))
        if dirty:
            out[name] = "conflict" if changed else "unsaved"
        else:
            out[name] = "external_write" if changed else "clean"
    return out


def ledger_view(
    last: Any,
    lines: int,
    boot: Any,
    quadrant: Mapping[str, str],
    l1_dirty: bool,
    pending: Sequence[str],
    *,
    now: Callable[[], float],
    stamp_of: Callable[[float], str],
) -> dict[str, Any]:
    """F14: ledger projection (``brain_stats.save_ledger``).

    Pattern copied from ``recall_feedback``: numbers + level + plain-language
    note. Empty library / never persisted reports ``null`` + ``unknown`` --
    **never 0**.
    """
    if not isinstance(last, dict):
        return {
            "seq": 0,
            "last_save_at": None,
            "interval_s": None,
            "saves_in_ledger": 0,
            "last_phase": None,
            "last_layers": None,
            "pending_layers": None,
            "layers_quadrant": {},
            "l1_dirty": False,
            "boot_reconcile": None,
            "external_touched": [],
            "level": "unknown",
            "note": "No save records yet (new library, or this process has never persisted). "
                    "Reported as unknown rather than 0 -- 0 would read as having just saved.",
        }
    at_wall = last.get("at_wall")
    at_mono = float(last.get("at_mono") or now())
    # Intervals always use the monotonic clock: subtracting wall clocks goes
    # negative on NTP rollbacks.
    interval = max(0.0, now() - at_mono)
    phase = last.get("phase")
    merged = last.get("merged_from_disk") or {}
    stamp = stamp_of(at_wall) if at_wall else "unknown"
    if phase != "end":
        level = "warn"
        note = (f"The last ledger entry is {phase}, not end: the previous save was **interrupted** "
                f"(crash / the host reclaimed the process). This is the only place it is visible after the fact.")
    else:
        hh = int(interval // 3600)
        mm = int((interval % 3600) // 60)
        if interval <= 3600:
            level = "ok"
        elif interval <= 21600:
            level = "warn"
        else:
            level = "stale"
        note = (f"Last save {stamp}, {hh}h{mm}m ago; 0 saves since. "
                f"In-memory changes made during this window are lost if the host reclaims the process.")
    # F14-5: boot reconcile with non-empty lost ⇒ the heaviest level (data
    # really did not get written); disk/ledger mismatch ranks second.
    if isinstance(boot, dict):
        if boot.get("lost_since_last_save"):
            level = "critical"
            note = f"Boot reconcile: {boot['note']}"
        elif boot.get("disk_mismatch"):
            if level == "ok":
                level = "warn"
            note = f"Boot reconcile: {boot['note']}"
    if any(q in ("unsaved", "conflict") for q in quadrant.values()) and level == "ok":
        level = "warn"
    return {
        "seq": last.get("seq"),
        "last_save_at": stamp,
        "interval_s": round(interval, 1),
        "saves_in_ledger": lines,
        "last_phase": phase,
        "last_duration_ms": last.get("duration_ms"),
        "last_layers": last.get("layers"),
        # F14-1 landed: no longer None here. The quadrant view's "memory
        # side" dimension = current in-memory fingerprint vs last persisted
        # fingerprint; a mismatch means "changed but not yet written".
        "pending_layers": list(pending) if pending else [],
        "pending_note": (
            (f"{len(pending)} layers have unsaved changes: {', '.join(pending)}. "
             "If the host reclaims the process now, these changes are gone.")
            if pending else
            "No layer currently holds unwritten changes (memory matches the last persisted content)."
        ),
        # G0: how many layers the last save actually skipped (effect of the
        # derived-layer dirty set; always empty means it never took effect)
        "last_layers_skipped": last.get("layers_skipped") or [],
        # F14-3: per-layer quadrant assignment (unsaved=changed, not written /
        # conflict=changed and someone else touched it too /
        # clean / external_write / unknown=this process never wrote it)
        "layers_quadrant": dict(quadrant),
        # L1 has its own dirty flag and merge chain; kept out of the layer
        # quadrants, reported as a one-line fact
        "l1_dirty": l1_dirty,
        # F14-5: boot reconcile result (None = no ledger to reconcile against)
        "boot_reconcile": boot,
        "external_touched": sorted(merged.keys()),
        "level": level,
        "note": note,
    }
