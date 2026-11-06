"""Layer GC family: L0 auto-persist / deletion-ledger gating and slimming /
layer merge routing / derived-layer reclip / boot reconciliation.

Six pieces moved wholesale from ``brain.core`` (I/O-heavy work belongs in
store; division of labor with ``store/layer_save.py`` -- that side owns
"full-save orchestration", this side owns "per-layer add/delete plus slimming
edge cases").

**State containers not moved**: ``layer_del`` / ``layer_stamp`` are injected as
the object itself by the shell or handled under the "return the new value, the
shell writes it back" contract (``_gc_layer_del``'s original implementation
**rebinds a new dict** anyway; the module returns ``(new dict, how many were
dropped)`` and the shell writes the attribute back -- semantically identical);
``raw_log`` works the same way (after the multi-writer union the shell writes
``self.raw_log`` back).

**Side effects stay in the shell**: ``_save_l0_only``'s try/except,
``manager.last_save_error`` bookkeeping and logger self-report, and the
``_l0_pending`` reset all stay in the shell (the same judgment as "final
persist stays in the shell" -- an injection port would be longer than the
original code); this module only does the persist core and raises exceptions
upward.

Semantics preserved verbatim (touching any one is a behavior change; parity
will catch it):

  * save_l0_only: under multi-writer processes L0 is append-only; must read the
    disk and union before writing (a whole-file overwrite would silently erase
    raw written by other processes); stamp the layer after writing;
  * apply_layer_deletions: switch off or empty ledger passes straight through;
    the pass condition is an entry timestamp **newer than** the deletion moment
    (a hit after deletion = a new trace, not resurrection); list and dict forms
    are gated separately; the blocked count is accumulated by the shell in
    ``_layer_del_blocked``;
  * gc_layer_del: first drop over-age entries by TTL (non-numeric timestamps
    dropped as well); if still over the cap, keep the newest max_keys entries
    by timestamp descending; returns how many were dropped;
  * merge_layer: route by layer rule (unknown layer = dict rule); the
    ``del_ledger`` layer does not participate in gating (it has no deletion
    semantics of its own); after merging, everything passes the deletion
    ledger;
  * reclip_derived_layers: an empty L1 view (count==0) returns directly --
    rather under-clip than mis-clip; entities recomputed then empty shells
    removed; profile goes through prune; topic layers go through reclip;
  * reconcile_ledger_on_boot: pure parsing (last entry's end/begin/fail
    three-state decision of lost/mismatch); logging stays in the shell.
"""
from __future__ import annotations

from typing import Any, Callable

from .ledger import reconcile_on_boot


def save_l0_only(
    *, persist_dir: str, raw_log: list, multi_writer_fn: Callable[[], bool],
    read_raw_fn: Callable[[], list], merge_raw_fn: Callable[..., list],
    backend: Any, get_backend_fn: Callable[[str], Any],
    layer_stamp: dict,
) -> list:
    """L0 persist core: multi-writer union -> backend write -> layer stamp.
    Returns the (possibly unioned) new raw_log; exceptions propagate upward
    (bookkeeping and self-reporting live in the shell)."""
    if multi_writer_fn():
        # Under multi-writer processes L0 is append-only: must union with the
        # disk, a whole-file overwrite would silently erase raw written by
        # other processes (same class of incident)
        raw_log = merge_raw_fn(read_raw_fn(), raw_log)
    # G1: go through the backend (json writes the single file; sqlite updates
    # only this one row in the layers table)
    backend = backend or get_backend_fn(persist_dir)
    backend.write_layers({"brain_raw.json": raw_log})
    layer_stamp["brain_raw.json"] = backend.layer_stamp("brain_raw.json")
    return raw_log


def apply_layer_deletions(
    name: str, merged: Any, *, enabled: bool, layer_del: dict,
    layer_key_fn: Callable[[dict], str], entry_ts_fn: Callable[[dict], float],
) -> tuple[Any, int]:
    """After merging: block keys the ledger has marked deleted. Returns
    ``(merge result, blocked count)``."""
    blocked = 0
    if not enabled or not layer_del:
        return merged, blocked
    prefix = name + "::"
    if isinstance(merged, list):
        out = []
        for it in merged:
            if isinstance(it, dict):
                dead = layer_del.get(prefix + layer_key_fn(it))
                if dead is not None and entry_ts_fn(it) <= float(dead):
                    blocked += 1
                    continue
            out.append(it)
        return out, blocked
    if isinstance(merged, dict):
        out = {}
        for k, v in merged.items():
            dead = layer_del.get(prefix + str(k))
            if dead is not None and (
                    not isinstance(v, dict) or entry_ts_fn(v) <= float(dead)):
                blocked += 1
                continue
            out[k] = v
        return out, blocked
    return merged, blocked


def gc_layer_del(
    layer_del: dict, *, now_fn: Callable[[], float], ttl: float,
    max_keys: int,
) -> tuple[dict, int]:
    """Ledger slimming: drop over-age entries; if still over the cap, drop the
    oldest. Returns ``(new ledger, how many were dropped)``.

    WARNING -- known trade-off: a dropped tombstone could in theory let an
    **older** snapshot resurrect the corresponding entry. 30 days far exceeds
    the reasonable lifetime of any in-memory snapshot; the risk is accepted and
    documented here.
    """
    if not layer_del:
        return layer_del, 0
    before = len(layer_del)
    now = now_fn()
    if ttl > 0:
        layer_del = {
            k: v for k, v in layer_del.items()
            if isinstance(v, (int, float)) and now - float(v) <= ttl
        }
    if max_keys and len(layer_del) > max_keys:
        keep = sorted(layer_del.items(), key=lambda kv: kv[1],
                      reverse=True)[:max_keys]
        layer_del = dict(keep)
    return layer_del, before - len(layer_del)


def merge_layer(
    name: str, disk: Any, mine: Any, *, rules: dict, fns: dict,
    apply_del_fn: Callable[[str, Any], Any], miss_log_max: int,
    list_caps: dict[str, int] | None = None,
) -> Any:
    """Merge by layer contract (disk as the base + local additions), then gate
    through the deletion ledger.

    ``list_caps``: per-layer cap table for list_cap rules (missing entries fall
    back to ``miss_log_max`` -- historical behavior bit-for-bit unchanged).
    brain_topics goes through here passing ``OPEN_TOPICS_MAX``.
    """
    rule = rules.get(name, "dict")
    if rule == "del_ledger":
        # The ledger itself does not participate in gating: its keys look like
        # "layer::key", so other layers' prefixes never hit it; conversely it
        # cannot block itself (it has no deletion semantics of its own).
        return fns["del_ledger"](disk, mine)
    if rule == "raw":
        merged = fns["raw"](disk if isinstance(disk, list) else [], mine)
    elif rule == "list":
        merged = fns["list"](disk, mine)
    elif rule == "list_cap":
        merged = fns["list_cap"](
            disk, mine, cap=(list_caps or {}).get(name, miss_log_max))
    elif rule == "entity":
        merged = fns["entity"](disk, mine)
    elif rule == "session":
        merged = fns["session"](disk, mine)
    elif rule == "intent":
        merged = fns["intent"](disk, mine)
    elif rule == "counter":
        merged = fns["counter"](disk, mine)
    elif rule == "topic":
        merged = fns["topic"](disk, mine)
    elif rule == "sourcesig":
        merged = fns["sourcesig"](disk, mine)
    else:
        merged = fns["dict"](disk, mine)
    # After merging (a union only ever adds), block already-deleted keys via
    # the deletion ledger.
    return apply_del_fn(name, merged)


def reclip_derived_layers(
    entities: dict, profile: dict, knowledge: dict, *, count_fn: Callable[[], int],
    recompute_entity_stats_fn: Callable[[], Any], prune_fn: Callable[[], Any],
    reclip_knowledge_fn: Callable[[], Any],
) -> None:
    """After the union, re-clip the derived layers against the **current
    truth**.

    Entities / profile are derived copies of L1 in the first place and need no
    extra deletion log: after merging, just re-clip them against the currently
    alive memories. An empty L1 view (count==0) returns directly -- rather
    under-clip than mis-clip (same origin as prune_stale_profile).
    """
    if count_fn() == 0:
        return
    if isinstance(entities, dict) and entities:
        recompute_entity_stats_fn()
        for k in [k for k, e in list(entities.items())
                  if not (e.get("memory_ids") or [])]:
            entities.pop(k, None)
    if isinstance(profile, dict) and profile:
        prune_fn()
    # Topic layers are derived copies of L1 too: the union would merge back
    # members from already-deleted memories.
    if isinstance(knowledge, dict) and knowledge:
        reclip_knowledge_fn()


def reconcile_ledger_on_boot(
    last: Any, tail: list, attrs: list,
    fingerprint_fn: Callable[[str], str],
) -> dict:
    """Pure parsing for boot reconciliation: the last entry's end/begin/fail
    three-state decision of lost/mismatch.

    A row without ``layer_fps`` (old format / fingerprints off) ->
    ``layer_fps_absent``, no content judgment. Dual-clock discipline: no new
    time fields added; lost's time semantics reference the ledger row's
    at_wall/at_mono.
    """
    return reconcile_on_boot(
        last, tail, attrs,
        lambda name: fingerprint_fn(name),
    )
