"""地图工具后端：map_set / map_list / map_verify 的判定核。

从 ``brain.core`` 整搬三件。``map_index`` 在基线上已是薄壳
（``domain/map_view.py`` 的 ``_map_index_fn``），本批未动。

**G1 约束的注入形态**：``store`` 与 ``domain`` 都不许 import ``map`` 包，
所以 map 包能力全部由壳**调用时现取**注入——``load`` 在壳上调好把
``data`` 传进来，``set_entry`` / ``update`` / ``verify`` / ``list_maps`` /
``map_names`` 是回调，``busy_exc`` / ``unreadable_exc`` 是异常类按值传
（``except <表达式>:`` 在 Python 里合法，运行时求值），``wait`` /
``max_maps`` 按值传。``root_fn`` 惰性（truth_source 解析根，锁内才调）。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * map_set：锁内读-改-写（``create=True``）；抢不到锁如实回 ``busy``
    带文案（不假装成功）；坏侧车回 ``error`` 且坏文件保留原处；回显带
    ``maps_total``（update 前的图数）与 ``map_entries``（本图条目数）；
    ``desc`` / ``truth_source`` / ``note`` 空串归一成 None 才进 set_entry；
  * map_list：data 为 None（未启用）→ ``enabled=False``；name 空 → 图
    列表截 ``max_maps`` 带 ``hidden`` 自曝；单图条目按 ``updated_at``
    降序（缺省 0.0），行截 **字面量 100**（不是常量——刻意与图数上限
    分开），``hidden`` 自曝；图 desc 缺省回落图名；
  * map_verify：锁内 verify（写 drift）+ 回填 maps 列表；busy/unreadable
    的回显形状与 map_set **刻意不同**（``checked:0`` + ``drifted:[]``）；
    data 为 None → ``enabled=False``；命中时补 ``enabled=True``。
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
    """闸一写入口：按键 upsert（同键再写整体替换，旧状态不竞争）。

    锁内读-改-写：``set_entry`` 必须作用在**最新的**侧车上——否则两个
    会话同时 map_set 不同键，后写的会把前一个的条目整个抹掉（静默丢
    更新）。抢不到锁如实报 ``busy``，不假装成功。
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
    """列地图（name 空）或列一张图的条目（带 cap + 自曝）。"""
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
    """闸二：重读 truth_source 比对签名（哈希仲裁，touch 不误报）。

    锁内读-改-写：verify 会写 ``drift``，若在锁外读快照再整量写回，
    会把这期间别人的 ``map_set`` 一起抹掉。
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
