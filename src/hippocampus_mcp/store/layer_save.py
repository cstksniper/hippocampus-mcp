# -*- coding: utf-8 -*-
"""Ledger orchestration and disk-write orchestration for layer saving (the landing spot for two core methods).

The original ``brain.core._save_layers`` / ``_save_layers_core`` were the last
big orchestrations inside the god class: the former books the current save
(begin / end / fail phases), the latter does the "read-merge-write" for derived
layers. Why they moved to store: throughout, they only deal with **layer file
I/O + ledger + layer state** — persistence infrastructure, not cognitive logic;
and the underlying facilities were already in this package (``ledger.py``'s
ledger-row I/O, ``backend``'s layer read/write, ``vector_store``'s multi-writer
switch).

**Injection-surface discipline**: brain-side capabilities are injected via
ports as callables/values; store never imports brain. Switch-like inputs
(``load_config().layer_fp``, ``multi_writer_enabled``) are evaluated **at call
time**, preserving the test monkeypatch surface over the ``core.load_config``
namespace. The core side keeps only two thin shells (``_save_layers`` /
``_save_layers_core``); the shells write back brain's bookkeeping state — the
original implementation updated fingerprints/sequence numbers only after the
disk write succeeded and booked nothing on failure; the shell's "write back
only after store returns" preserves that discipline by construction.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ..common.constants import LAYER_DEL_FILE
from .backend import get_backend
from .ledger import ledger_path
from .vector_store import multi_writer_enabled


# ----------------------------------------------------------------------
# Disk-write phase (original _save_layers_core)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LayerSavePorts:
    """Brain-side layer I/O capabilities and layer state handles.

    Every field is either a value (``layer_attrs`` / ``begin_layer_fps`` /
    ``written_fp`` / ``backend``) or a **callable** (evaluated at call time,
    not the call's result — so tests that monkeypatch brain instances/class
    attributes still take effect).
    """

    persist_dir: str | None
    layer_attrs: dict[str, str]
    # ---- Layer value read/write (attr-name convention; values match _LAYER_ATTRS) ----
    get_layer: Callable[[str], Any]
    set_layer: Callable[[str, Any], None]
    # ---- Merge family (implementations live in brain; pure functions in brain/pure.py) ----
    layer_fingerprint: Callable[[Any], str]
    layer_changed_by_other: Callable[[str, str], bool]
    read_layer_from_disk: Callable[[str], Any]
    read_raw_log_from_disk: Callable[[], list]
    merge_raw_log: Callable[..., Any]
    merge_del_ledger_layer: Callable[..., Any]
    merge_layer: Callable[..., Any]
    disk_only_count: Callable[..., int]
    reclip_derived_layers: Callable[[], None]
    gc_layer_del: Callable[[], int]
    reset_del_blocked: Callable[[], None]
    # ---- Backend ----
    backend: Any | None
    # ---- Ledger fingerprints (read-only references: the core phase only reads them) ----
    begin_layer_fps: dict[str, str]
    written_fp: dict[str, str]
    # ---- Switches (fetched at call time) ----
    layer_fp_on: Callable[[], bool]


@dataclass
class LayerSaveResult:
    """Output of the disk-write phase. The shell writes back brain's bookkeeping state from it.

    ``persisted=False`` means a pure in-memory brain (no persist_dir) — the
    original implementation **touches no state at all** in that case (does not
    clear ``last_layer_merge``, does not advance sequence numbers), and the
    shell accordingly skips all write-backs.
    """

    persisted: bool = False
    last_layer_merge: dict[str, int] = field(default_factory=dict)
    layers_skipped: list[str] = field(default_factory=list)
    end_layer_fps: dict[str, str] = field(default_factory=dict)
    written_fp: dict[str, str] = field(default_factory=dict)
    layer_stamp: dict[str, Any] = field(default_factory=dict)
    del_gc_last: int | None = None


def save_layers_core(ports: LayerSavePorts) -> LayerSaveResult:
    """Write only the layers brain owns, such as L0 / L2 / L3 (does not touch manager's L1 + vectors).

    **Under multiple processes every derived layer must share L1's semantics**:
    L1 (``memories.json``) uses "merge on conflict" (additive), while brain's
    layers were previously **overwritten wholesale** — with two processes
    sharing one library, whoever persists last defines the whole library's
    version, and the other side's entity mounts / session registrations /
    tombstones get silently wiped (measured incident). No exceptions, no logs.

    So this does **read-merge-write**: before persisting, read the disk version,
    take the union per the layer's convention, then write. In single-process
    mode (``multi_writer_enabled()`` false) behavior is exactly as before the
    change.

    The number of merged-in entries is recorded in the return value's
    ``last_layer_merge`` and self-reported in ``stats()`` — "how many of the
    other side's entries my merge rescued" must be a visible number, not
    something dug up by archaeology.
    """
    if not ports.persist_dir:
        return LayerSaveResult(persisted=False)
    result = LayerSaveResult(persisted=True)
    if multi_writer_enabled():
        ports.set_layer(ports.layer_attrs["brain_raw.json"],
                        ports.merge_raw_log(ports.read_raw_log_from_disk(),
                                            ports.get_layer(
                                                ports.layer_attrs[
                                                    "brain_raw.json"])))
        # The deletion ledger must be merged **first**. It is the basis other
        # layers use for interception; merging it last would mean that in this
        # save miss_log finishes merging against the old ledger before the new
        # tombstones arrive, exactly missing the "just deleted by the other
        # side" batch (wrong order makes the whole thing pointless).
        ports.reset_del_blocked()
        _del_path = os.path.join(ports.persist_dir, LAYER_DEL_FILE)
        if ports.layer_changed_by_other(LAYER_DEL_FILE, _del_path):
            _disk_del = ports.read_layer_from_disk(LAYER_DEL_FILE)
            if isinstance(_disk_del, dict):
                ports.set_layer(ports.layer_attrs[LAYER_DEL_FILE],
                                ports.merge_del_ledger_layer(
                                    _disk_del,
                                    ports.get_layer(
                                        ports.layer_attrs[LAYER_DEL_FILE])))
        for name, attr in ports.layer_attrs.items():
            if name == LAYER_DEL_FILE:
                continue      # already merged separately above
            if name == "brain_raw.json":
                continue
            _path = os.path.join(ports.persist_dir, name)
            if not ports.layer_changed_by_other(name, _path):
                continue      # nobody touched it: the in-memory copy is the latest
            disk = ports.read_layer_from_disk(name)
            if disk is None:
                continue          # layer absent on disk: first upgrade of an old library, write per local state
            mine = ports.get_layer(attr)
            merged = ports.merge_layer(name, disk, mine)
            # What gets reported is "entries that exist on disk, not in my
            # memory, and were rescued by this merge".
            # Not len(merged) - len(disk): when the local copy is empty that is 0,
            # which is precisely the case that **most deserves an alarm** (the
            # wholesale overwrite happens at exactly this moment).
            rescued = ports.disk_only_count(name, disk, mine)
            if rescued > 0:
                result.last_layer_merge[name] = rescued
            # Merge into memory too: otherwise this process's subsequent
            # operations keep judging by "my own copy" and still see the
            # incomplete view until the next save.
            ports.set_layer(attr, merged)
        # The union is purely additive: after merging, reclip against currently
        # alive memories, otherwise the stale disk version resurrects what this
        # process already deleted.
        ports.reclip_derived_layers()
        # Ledger slimming. Placed after merging — tombstones just merged in must
        # take effect this very save; they must not be GC'd before ever being used.
        result.del_gc_last = ports.gc_layer_del()
    # G0 / F14-1: write only layers whose **content changed** (the derived-layer
    # dirty set).
    # Criterion = content fingerprint unchanged **and** disk untouched by others.
    # Both conditions are required: fingerprint alone → if someone else deleted
    # the disk file we would still skip and the layer is lost; disk stamp alone →
    # when this process changed content back to the same fingerprint we would
    # waste a write.
    files: dict[str, Any] = {}
    new_fp: dict[str, str] = {}
    # Whether begin fingerprints can be reused: begin is computed before core;
    # if no multi-process merging happened in between (merging changes layer
    # content via set_layer below), the layer values are unchanged and one
    # fingerprint computation can be saved.
    reuse_begin_fps = not multi_writer_enabled() and bool(ports.begin_layer_fps)
    for name, attr in ports.layer_attrs.items():
        value = ports.get_layer(attr)
        if ports.layer_fp_on():
            if reuse_begin_fps:
                fp = ports.begin_layer_fps[name]
            else:
                fp = ports.layer_fingerprint(value)
            if fp == ports.written_fp.get(name) and not ports.layer_changed_by_other(
                name, os.path.join(ports.persist_dir, name)
            ):
                result.layers_skipped.append(name)
                # For a skipped layer, the fingerprint at this moment is the disk reality (unchanged fingerprint and untouched)
                result.end_layer_fps[name] = fp
                continue
            new_fp[name] = fp
        files[name] = value
    # Merge the actually-written layers' fingerprints into the end fingerprint
    # set (skipped layers were settled inside the loop)
    result.end_layer_fps.update(new_fp)
    # Layers persist within **one transaction**. The json backend still does
    # per-file replace (behavior unchanged); the sqlite backend commits with a
    # single BEGIN IMMEDIATE and rolls the whole batch back on mid-way failure —
    # exactly the primitive the JSON scheme cannot patch in at the filesystem
    # layer (the first reason SQLite was chosen).
    backend = ports.backend or get_backend(ports.persist_dir)
    if files:
        backend.write_layers(files)
    for name in files:
        result.layer_stamp[name] = backend.layer_stamp(name)
    # Fingerprints are updated only after the write **actually succeeded**:
    # booking "written" on a failed write would make the next save skip a layer
    # that never made it to disk (same discipline as clearing the dirty set only
    # after a successful write).
    result.written_fp = new_fp
    # F14-5: the end row's layer fingerprints were settled layer by layer inside
    # the loop (skipped layers = pre-write current value; written layers =
    # new_fp). On write failure end_layer_fps stays empty ⇒ the end row carries
    # no fingerprints, so startup reconciliation never mistakes a half-finished
    # fingerprint for the truth.
    return result


# ----------------------------------------------------------------------
# Ledger phase (original _save_layers)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LayerLedgerPorts:
    """Injection surface for the ledger's three-phase orchestration.

    ``core_fn`` is brain's ``_save_layers_core`` (tests may replace it wholesale
    with a faulting function — the original implementation's
    ``self._save_layers_core()`` looked it up through the instance; semantics
    unchanged).
    """

    persist_dir: str | None
    save_seq: int
    layer_attrs: dict[str, str]
    now_wall: Callable[[], float]
    now_mono: Callable[[], float]
    get_pid: Callable[[], int]
    compute_all_fps: Callable[[], dict[str, str]]
    layer_count: Callable[[str], int]
    ledger_append: Callable[[dict[str, Any]], None]
    set_begin_fps: Callable[[dict[str, str]], None]
    core_fn: Callable[[], LayerSaveResult | None]


def run_layer_save(ports: LayerLedgerPorts) -> None:
    """Book the ledger for this save; the actual disk write is delegated to :func:`save_layers_core`.

    Each phase has its reason: ``begin`` must run **before** the disk write (on
    crash, at least "what we intended to write" survives), ``end`` records the
    actual result, ``fail`` records the failure reason — without recording, the
    event "once intended to write but never succeeded" would never have happened
    (design §8 risk 8).
    """
    path = ledger_path(ports.persist_dir)
    seq = ports.save_seq + 1
    now_wall = ports.now_wall()
    # F14-5: begin carries layer fingerprints ("what I declare I am about to
    # write"). Compute once and stash for reuse in core (only when this save did
    # not go through multi-process merging — merging changes layer content
    # before the fingerprint loop, in which case they must be recomputed).
    ports.set_begin_fps({})
    if path:
        fps_all = ports.compute_all_fps()
        begin_entry = {
            "seq": seq,
            "phase": "begin",
            "at_wall": now_wall,
            "at_mono": ports.now_mono(),
            "pid": ports.get_pid(),
            "layers": {n: ports.layer_count(a)
                       for n, a in ports.layer_attrs.items()},
        }
        if fps_all:
            begin_entry["layer_fps"] = fps_all
            ports.set_begin_fps(fps_all)
        ports.ledger_append(begin_entry)
    t0 = ports.now_mono()
    try:
        result = ports.core_fn()
    except BaseException as e:  # noqa: BLE001 — failures get booked too, then re-raised as usual
        if path:
            ports.ledger_append({
                "seq": seq,
                "phase": "fail",
                "at_wall": ports.now_wall(),
                "at_mono": ports.now_mono(),
                "pid": ports.get_pid(),
                "duration_ms": int((ports.now_mono() - t0) * 1000),
                "error": f"{type(e).__name__}: {e}",
            })
        raise
    result = result or LayerSaveResult(persisted=False)
    if path:
        end_entry = {
            "seq": seq,
            "phase": "end",
            "at_wall": ports.now_wall(),
            "at_mono": ports.now_mono(),
            "pid": ports.get_pid(),
            "duration_ms": int((ports.now_mono() - t0) * 1000),
            "layers": {n: ports.layer_count(a)
                       for n, a in ports.layer_attrs.items()},
            # Layers merged back from disk = an external writer existed during
            # this window (bottom-right of the four quadrants).
            # Slice 1 only records the fact, no judgment (judgment waits for
            # F14-1's layer version numbers).
            "merged_from_disk": dict(result.last_layer_merge or {}),
            # Layers skipped this time because "content unchanged" (the actual
            # effect of the derived-layer dirty set).
            # Makes "how many needless rewrites were saved" visible, and guards
            # against a "should have written but was skipped" going unnoticed.
            "layers_skipped": list(result.layers_skipped or []),
        }
        # F14-5: end carries layer fingerprints (the post-write look; core has
        # already filled them in — written layers take new_fp, skipped layers
        # inherit their begin values).
        if result.end_layer_fps:
            end_entry["layer_fps"] = dict(result.end_layer_fps)
        ports.ledger_append(end_entry)
