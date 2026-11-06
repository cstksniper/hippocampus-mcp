"""Map tool backend: the decision core of map_set / map_list / map_verify.

Three pieces moved wholesale from ``brain.core``. ``map_index`` was already a
thin shell on the baseline (``domain/map_view.py``'s ``_map_index_fn``) and
was not touched in this batch.

**Injection shape under the G1 constraint**: neither ``store`` nor ``domain``
may import the ``map`` package, so all map-package capabilities are injected
by the shell **fetched at call time** — ``load`` is called in the shell and its
``data`` passed in; ``set_entry`` / ``update`` / ``verify`` / ``list_maps`` /
``map_names`` are callbacks; ``busy_exc`` / ``unreadable_exc`` are exception
classes passed as values (``except <expression>:`` is legal in Python,
evaluated at runtime); ``wait`` / ``max_maps`` are passed as values.
``root_fn`` is lazy (parses the truth_source root; called only inside the
lock).

Semantics preserved verbatim (changing any one is a behavior change; parity
will catch it):

  * map_set: read-modify-write inside the lock (``create=True``); failing to
    get the lock honestly returns ``busy`` with a message (never fakes
    success); a corrupt sidecar returns ``error`` with the bad file left in
    place; the echo carries ``maps_total`` (map count before the update) and
    ``map_entries`` (this map's entry count);
    ``desc`` / ``truth_source`` / ``note`` empty strings are normalized to None
    before entering set_entry;
  * map_list: data None (not enabled) → ``enabled=False``; empty name → the
    map list truncated to ``max_maps`` with a ``hidden`` self-report; a single
    map's entries sorted by ``updated_at`` descending (default 0.0), rows
    truncated at the **literal 100** (not a constant — deliberately separate
    from the map-count cap), with a ``hidden`` self-report; a map desc missing
    falls back to the map name;
  * map_verify: verify inside the lock (writes drift) + backfills the maps
    list; busy/unreadable echoes are shaped **deliberately differently** from
    map_set (``checked:0`` + ``drifted:[]``); data None → ``enabled=False``;
    on a hit, ``enabled=True`` is added.
"""
from __future__ import annotations

from typing import Any, Callable


def map_set(
    name: str, key: str, status: str = "", fields: dict | None = None,
    desc: str = "", truth_source: str = "", note: str = "",
    updated_by: str = "", *, persist_dir: str,
    set_entry: Callable[..., Any], update_fn: Callable[..., Any],
    root_fn: Callable[[], str | None], busy_exc: type,
    unreadable_exc: type, wait: float,
) -> dict[str, Any]:
    """Gate one's write entry: upsert by key (rewriting the same key replaces wholesale; old states do not compete).

    Read-modify-write inside the lock: ``set_entry`` must act on the **latest**
    sidecar — otherwise two sessions calling map_set on different keys
    concurrently means the later write wipes the earlier one's entry wholesale
    (a silently lost update). Failing to get the lock honestly reports
    ``busy``; it never fakes success.
    """

    def _do(data):
        m, ent, notes = set_entry(
            data, name, key, status=status, fields=fields,
            desc=desc or None, truth_source=truth_source or None,
            note=note or None, updated_by=updated_by,
            root=root_fn())
        return True, (m, ent, notes, len(data.get("maps") or {}))

    try:
        got, saved = update_fn(persist_dir, _do, create=True, wait=wait)
    except busy_exc as exc:
        return {"status": "busy", "map": name, "key": key,
                "hint": "另一个进程正在写地图（%s）；这次没写进去，"
                        "稍后重试即可" % exc}
    except unreadable_exc as exc:
        return {"status": "error", "map": name, "key": key,
                "hint": "%s; the corrupt file was kept in place -- repair it or rebuild explicitly first" % exc}
    m, ent, notes, maps_total = got
    return {"status": "ok", "map": name, "key": key, "entry": ent,
            "notes": notes, "maps_total": maps_total,
            "map_entries": len(m.get("entries") or {})}


def map_list(
    data: dict | None, name: str = "", *, list_maps_fn: Callable[..., Any],
    map_names_fn: Callable[[dict], list], max_maps: int,
) -> dict[str, Any]:
    """List maps (empty name) or list one map's entries (with cap + self-report)."""
    if data is None:
        return {"maps": [], "maps_total": 0, "enabled": False}
    if not name:
        items = list_maps_fn(data)
        return {"enabled": True, "maps": items[:max_maps],
                "maps_total": len(items),
                "hidden": max(0, len(items) - max_maps)}
    m = (data.get("maps") or {}).get(str(name))
    if m is None:
        return {"enabled": True, "found": False, "map": name,
                "hint": "no such map; existing maps: %s"
                        % ("、".join(map_names_fn(data)) or "(empty)")}
    ents = m.get("entries") or {}
    rows = []
    for k in sorted(ents, key=lambda k: -float(
            (ents[k] or {}).get("updated_at") or 0.0)):
        e = ents[k] or {}
        rows.append({"key": k, "status": e.get("status") or "",
                     "drift": e.get("drift"),
                     "note": e.get("note") or "",
                     "fields": e.get("fields") or {},
                     "truth_source": e.get("truth_source") or "",
                     "updated_at": e.get("updated_at"),
                     "updated_by": e.get("updated_by") or ""})
    return {"enabled": True, "found": True, "map": name,
            "desc": m.get("desc") or name, "entries": rows[:100],
            "entries_total": len(rows),
            "hidden": max(0, len(rows) - 100)}


def map_verify(
    name: str = "", *, persist_dir: str, verify_fn: Callable[..., Any],
    update_fn: Callable[..., Any], list_maps_fn: Callable[..., Any],
    root_fn: Callable[[], str | None], busy_exc: type,
    unreadable_exc: type, wait: float,
) -> dict[str, Any]:
    """Gate two: re-read the truth_source and compare signatures (hash arbitration; a touch causes no false alarm).

    Read-modify-write inside the lock: verify writes ``drift``; if it read a
    snapshot outside the lock and wrote the whole thing back, it would wipe out
    everyone else's ``map_set`` made in the meantime.
    """

    def _do(data):
        r = verify_fn(data, name or None,
                      root=root_fn())
        r["maps"] = list_maps_fn(data)
        return True, r

    try:
        res, saved = update_fn(persist_dir, _do,
                               wait=wait)
    except busy_exc as exc:
        return {"enabled": True, "status": "busy", "checked": 0,
                "drifted": [], "hint": str(exc)}
    except unreadable_exc as exc:
        return {"enabled": True, "checked": 0, "drifted": [],
                "error": str(exc)}
    if res is None:
        return {"enabled": False, "checked": 0, "drifted": [],
                "hint": "the map layer is not enabled"}
    res["enabled"] = True
    return res
