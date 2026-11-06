# -*- coding: utf-8 -*-
"""Orchestration of the ``brain.core.file_registry_summary`` family
(extracted from core).

**Why worth moving**: the file-archiving summary is a stretch of
orchestration on session_start's hot path -- the six steps "config gate →
sidecar existence → interval gating (re-checked under lock) → rescan with
budget → reconcile → summarize with fallback" are all flow control; the
value computation (scan / summarize / index) already lives in
``map/file_registry.py``. After the move the shell is down to three
straight lines: "read config / fetch the fr_mod entry points / inject the
reconcile callback".

**Dependency direction**: this module has zero imports -- the four
``fr_mod`` entry points (load / scan / update / summarize) are fetched from
that namespace by the shell at call time and injected (patching
``core.fr_mod`` takes effect); the reconcile callback (reads
``self.source_sigs``, brain state) is injected as a bare reference; the
``now`` instant and ``interval_h`` / ``limits`` are computed by the shell
and passed in; path normalization is injected via the ``path_of`` callback
(os stays out of domain).

**Replicated quirks (touching any one is a behavior change; the parity
tests will catch it)**:

1. Interval gating is **re-checked under lock** (``_do`` runs inside
   ``update``'s sidecar lock) -- two sessions opening at once must not each
   scan once;
2. The gate's truth is ``interval_h > 0``: 0 = scan every time (throttling
   off); skip only when ``now - last_checked < interval_h*3600`` (equal to
   the interval means rescan);
3. Scan failure neither crashes nor writes: ``except → (False, reg)`` --
   report the previous summary; ``last_checked`` does not advance (the next
   opening retries);
4. ``update`` itself failing (lock busy-wait) → fall back to ``load`` and
   report the old summary -- the hot path never crashes over a file-archiving
   failure;
5. No sidecar returns ``None`` (opt-in); the ``(result, saved)`` two-tuple
   unpacking semantics of ``update`` are preserved;
6. The reconcile criterion (mtime_ns, size) is **double equality**: the
   size-missing sentinels are **staggered, -1 / -2** -- when neither side
   has a record they can never compare equal, preventing "neither has a
   record" from being misjudged as "already synced";
7. ``last_synced`` fetches ``now()`` **per entry** (lazy callback; the call
   count aligns with the original implementation one by one -- under a
   frozen clock both sides get the same value).
"""
from __future__ import annotations

from typing import Any, Callable


def summary_flow(
    *,
    persist_dir: str,
    now: float,
    interval_h: float,
    limits: tuple[int, float],
    load: Callable[..., dict[str, Any] | None],
    scan: Callable[..., dict[str, Any]],
    update: Callable[..., tuple],
    summarize: Callable[..., dict[str, Any] | None],
    reconcile: Callable[[dict[str, Any]], int],
) -> dict[str, Any] | None:
    """File-archiving summary orchestration: existence → gated rescan under
    lock → failure fallback → summarize.

    Args:
        persist_dir: the brain's persistence directory (where the sidecar
            ``file_registry.json`` lives).
        now: the current instant (fetched by the shell's ``_now()``; shared
            by gating / last_checked / summarize).
        interval_h: rescan interval in hours (read from config by the shell;
            **0 = throttling off**).
        limits: scan budget ``(max_files, budget_s)`` (fetched by the
            shell's ``scan_limits()``).
        load / scan / update / summarize: the four ``fr_mod`` entry points
            (fetched and injected by the shell).
        reconcile: reconcile callback ``(reg) -> number of entries cleared``
            (brain-side ``_fr_reconcile_source_sigs``, reads
            ``self.source_sigs``).

    Returns:
        Summary dict; ``None`` when not enabled / no sidecar / no reg.
    """
    if load(persist_dir) is None:
        return None

    def _do(reg):
        # Interval gating re-checked under lock: two sessions opening at once
        # must not each scan once
        if interval_h > 0 and now - float(
                reg.get("last_checked") or 0.0) < interval_h * 3600:
            return False, reg
        try:
            scan(reg, limits[0], limits[1])
            reg["last_checked"] = now
            reconcile(reg)
        except Exception:  # noqa: BLE001 -- scan failed: do not write, report the old summary
            return False, reg
        return True, reg

    try:
        reg, saved = update(persist_dir, _do, wait=0.0)
    except Exception:  # noqa: BLE001 -- the hot path never crashes over a failed archiving scan
        reg = load(persist_dir)
    if reg is None:
        return None
    return summarize(reg, now)


def reconcile_stale_entries(
    *,
    reg: dict[str, Any],
    sig_by_path: dict[str, dict[str, Any]],
    path_of: Callable[[str], str],
    now: Callable[[], float],
) -> int:
    """Reconcile stale entries against the source-signature layer: hippo has
    consumed the current content version ⇒ clear stale.

    The criterion is (mtime_ns, size) double equality -- the same threat
    model as the archiving layer (adversarial wall-clock comparison was
    never within the defense scope). **Mutates reg in place** (same object
    semantics as the original implementation); returns the number of entries
    cleared (0 = nothing to clear, the common case).

    Args:
        reg: the archiving registry (``files`` is ``{rel: entry}``).
        sig_by_path: absolute-path index produced by
            ``fr_mod.source_sig_index`` (built by the shell; both sides share
            one criterion -- normalization / taking the newest exists only
            here).
        path_of: ``rel → index key`` normalization callback (``normcase(
            join(root, str(rel)))``; os stays out of domain).
        now: time source for ``last_synced`` (**fetched per entry**, lazy
            callback).
    """
    n = 0
    for rel, ent in (reg.get("files") or {}).items():
        if ent.get("status") != "stale":
            continue
        sig = sig_by_path.get(path_of(str(rel)))
        if not sig:
            continue
        try:
            if (int(sig.get("mtime_ns") or 0)
                    == int(ent.get("mtime_ns") or 0)
                    and int(sig.get("size") or -1)
                    == int(ent.get("size") or -2)):
                ent["status"] = "synced"
                ent["last_synced"] = now()
                n += 1
        except (TypeError, ValueError):
            continue
    return n
