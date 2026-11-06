"""Map auto-patrol family: auto verify + mention drift.

Moved wholesale from ``brain.core`` (division of labor with
:meth:`map_verify`'s signature-level comparison: C1/C2 read the source
of truth, mention reads memory text -- weak signals write a separate
``ent["mention"]`` without overwriting ``ent["drift"]``). G1: all map
capabilities are injected by the shell, **fetched live at call time**
(the ``auto_run`` / ``auto_needs_registry`` / ``needs_run_fn`` /
``run_fn`` / ``registry_load`` / ``update_fn`` callbacks, the
``busy_exc`` exception class passed by value, the three config knobs
fetched live on the shell and passed by value); ``records_fn`` is a lazy
callback -- mention's record snapshot is materialized only inside the
lock and after the interval gate (one snapshot guards against
concurrent ingest's "list changed during iteration").

Semantics preserved verbatim (changing any one is a behavior change;
parity will catch it):

  * three runtime disciplines shared by both: interval gate (default 6h,
    **re-judged inside the lock** -- the gate re-reads the latest clock
    in the same critical section as run, preventing two sessions from
    each running once and overwriting each other; ``force`` or
    interval<=0 skips the gate), best-effort persist to disk
    (SidecarBusy→skipped:busy, OSError→save_failed, other exceptions→
    error self-report; patrol can regenerate and never blows up the
    opening), sidecar absent (update returns None) → enabled=False not
    enabled;
  * auto: one disk read serves two purposes (registry root is C1's
    resolved root, files is C2's reconciliation source); a false
    ``needs_registry`` passes no registry (zero duplicate IO); **only
    libraries that really have an auto recipe advance
    ``auto_checked_at``** (``bool(res.get("maps"))`` returned as the
    save criterion -- recipe-less libraries write zero bytes);
  * mention: a false ``needs_run`` returns skipped:no_maps directly (no
    gating, no clock advance); advancing ``mention_checked_at`` is
    unconditional (deliberately different from auto's conditional
    advance).
"""
from __future__ import annotations

from typing import Any, Callable


def auto_verify(
    force: bool = False, *, persist_dir: str, now_fn: Callable[[], float],
    enabled: bool, interval_h: float, budget_s: float,
    registry_load: Callable[[str], Any], auto_run: Callable[..., Any],
    auto_needs_registry: Callable[[dict], bool],
    update_fn: Callable[..., Any], busy_exc: type,
) -> dict[str, Any]:
    """Phase C: signal-driven auto verify -- C1 value-level / C2 directory reconciliation (map-level ``auto`` recipe)."""
    if not enabled:
        return {"enabled": False, "reason": "env 关闭"}

    def _do(data):
        now = now_fn()
        if not force and interval_h > 0:
            last = float(data.get("auto_checked_at") or 0.0)
            elapsed = now - last
            if elapsed < interval_h * 3600:
                return False, {"skipped": "interval", "last_run": last,
                               "next_in_s": int(interval_h * 3600 -
                                                elapsed)}
        # One disk read serves two purposes: the registry root is C1's
        # resolved root; the registry files are C2's reconciliation data
        # source (C2's zero duplicate IO depends entirely on it). Without
        # a dir_reconcile recipe, nothing beyond files is read -- and
        # there is no second read either.
        reg = registry_load(persist_dir)
        res = auto_run(
            data, registry_root=(reg or {}).get("root") or None,
            budget_s=budget_s,
            force=force, now=now,
            registry=(reg if (reg and auto_needs_registry(data))
                      else None))
        # Only libraries that really have an auto recipe advance the patrol clock (recipe-less libraries write zero bytes)
        data["auto_checked_at"] = now
        return bool(res.get("maps")), res

    try:
        res, saved = update_fn(persist_dir, _do)
    except busy_exc:
        return {"enabled": True, "skipped": "busy"}
    except OSError:
        return {"enabled": True, "save_failed": True}
    except Exception as exc:  # noqa: BLE001 -- auto verify must not blow up the opening
        return {"enabled": True,
                "error": "%s: %s" % (type(exc).__name__, exc)}
    if res is None:
        return {"enabled": False, "reason": "未启用地图层"}
    res["enabled"] = True
    return res


def mention_drift(
    force: bool = False, *, persist_dir: str, now_fn: Callable[[], float],
    enabled: bool, interval_h: float, budget_s: float,
    max_records: int, needs_run_fn: Callable[[dict], bool],
    run_fn: Callable[..., Any], records_fn: Callable[[], list],
    update_fn: Callable[..., Any], busy_exc: type,
) -> dict[str, Any]:
    """Gate three: mentioned as 「another state」 after an update ⇒ the entry is possibly stale (writes ``ent["mention"]``)."""
    if not enabled:
        return {"enabled": False, "reason": "env 关闭"}

    def _do(data):
        now = now_fn()
        if not needs_run_fn(data):
            return False, {"skipped": "no_maps"}
        if not force and interval_h > 0:
            last = float(data.get("mention_checked_at") or 0.0)
            elapsed = now - last
            if elapsed < interval_h * 3600:
                return False, {"skipped": "interval", "last_run": last,
                               "next_in_s": int(interval_h * 3600 -
                                                elapsed)}
        res = run_fn(
            # Snapshot once: if another thread in the same process is
            # mid-ingest, iterating the live list directly hits "list
            # changed size during iteration" (references copied; the
            # record bodies are not duplicated, negligible cost)
            data, records_fn(), now=now,
            budget_s=budget_s,
            max_records=max_records)
        data["mention_checked_at"] = now
        return True, res

    try:
        res, saved = update_fn(persist_dir, _do)
    except busy_exc:
        return {"enabled": True, "skipped": "busy"}
    except OSError:
        return {"enabled": True, "save_failed": True}
    except Exception as exc:  # noqa: BLE001 -- patrol must not blow up the opening
        return {"enabled": True,
                "error": "%s: %s" % (type(exc).__name__, exc)}
    if res is None:
        return {"enabled": False, "reason": "未启用地图层"}
    res["enabled"] = True
    return res
