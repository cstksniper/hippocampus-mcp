"""地图索引（L0）与任务相关图全文（L1）的读侧投影。

两个入口都不检索、不引模型、不写盘——纯读侧投影：

- :func:`map_index` 合并两个来源（legacy 保留标签记忆 +
  ``maps.json`` 侧车），只报名 + 条数 + 新鲜度 + 漂移/冲突计数，
  不带正文（零正文是「必要信息」与「不能太长」的平衡点）；
- :func:`map_detail` 用任务 token 命中图的 desc / 条目键，把整图
  按 cap 递送（L1）。

**为什么这些能力在 domain**：它们是纯函数投影（输入 records +
registry，输出 dict），不碰 I/O 也不碰模型。但**读什么**由调用方注入：

- ``records`` / ``registry``：调用方从 store 与 map 侧车取好再传进来
  ——domain 不准 import store 也不准 import map（arch 矩阵：domain
  只准依赖 common），「读哪」是装配层的事；
- ``list_maps``：map 包的行投影函数本体，回调注入；
- ``limits``：所有阈值 / 保留标签键（``地图`` / ``条目`` / ``状态``）
  由调用方从 ``map.constants`` 现取——常量放在 map 包，domain 看不见，
  硬编码一份迟早和 map 包各说各话；
- ``now`` / ``task_tokens_fn``：时钟与分词器，替身可注（测试钉
  时间窗、词表行为时不必等真时钟）。

行为口径（与原 brain 方法逐字对齐，parity 探针取证）：

- 归档记忆不进索引（consolidate 会归档沉寂条目，继续计数会让索引行虚高）；
- 双源同键不同状态 ⇒ ``conflicts`` 计数（只进必退同纪律：任一侧改成一致即消失）；
- 冲突关系不合并计数：drifted（已确认变了）与 mentioned（有人这么说）
  分开累加、分开渲染——两种可信度；
- 组号 / newer 之类本模块不涉及（那是并存口径，见 conflict_annotate）；
- 索引按条数倒序、同条数按名；详情按命中键数倒序、⚠️ 行置顶。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..common.constants import _now, task_tokens


@dataclass(frozen=True)
class IndexLimits:
    """索引行的截断与保留标签键（全部由调用方现取注入）。"""

    max_maps: int          # MAP_INDEX_MAX：索引行最多带几张图
    stale_days: int        # MAP_STALE_DAYS：⚠️ 陈旧阈值
    name_key: str          # MAP_NAME_KEY：这张记忆属于哪张地图
    entry_key: str         # MAP_ENTRY_KEY：条目键
    status_key: str        # MAP_STATUS_KEY：条目当前状态


@dataclass(frozen=True)
class DetailLimits:
    """L1 递送的两档 cap（由调用方现取注入）。"""

    max_maps: int          # MAP_DETAIL_MAX_MAPS：一次最多进包几张图
    max_rows: int          # MAP_DETAIL_MAX_ROWS：每图行数 cap


def map_index(
    records: Iterable[Any],
    registry: dict[str, Any] | None,
    *,
    list_maps: Callable[[dict[str, Any]], list[dict[str, Any]]],
    now: Callable[[], float] | None = None,
    archived_state: str = "",
    limits: IndexLimits,
) -> dict[str, Any]:
    """地图索引（L0 递送）：库里有哪几地图、各多少条、多久没更新。

    两个来源合并：

    - **legacy**：带保留标签键的记忆（``{"地图": "合约", "条目":
      "CN2601", "状态": "验证集"}``），一键一条；
    - **registry**：``maps.json`` 侧车（map_set 写入的正式地图层）。

    同名图相加（entries 数相加、取最新、来源标 ``both``）——老数据不丢，
    新数据走新路。纯 O(n) 读既有层，不检索不聚类。

    设计评审后收紧的两点：

    - **归档条目不进索引**（consolidate 会把沉寂的地图条目记忆归档，
      继续计条数会让索引行虚高，且归档条目 recall 不可达）；
    - **双源同键冲突检测**：同一个（图, 条目键）在 legacy 记忆与
      registry 侧车里状态不一致 ⇒ ``conflicts`` 计数 + 索引行「冲突N」。
      相加只解决了计数合并，没解决「两条状态竞争」——修复前它可以
      无声无息地活在 both 图里。
    """
    maps: dict[str, dict[str, Any]] = {}
    legacy_keys: dict[str, dict[str, str]] = {}
    registry_keys: dict[str, dict[str, str]] = {}
    # legacy：标签命名空间
    for rec in records:
        if rec.state == archived_state:
            continue
        tags = rec.tags or {}
        name = tags.get(limits.name_key)
        if not name:
            continue
        m = maps.setdefault(str(name), {"entries": 0, "newest": 0.0,
                                        "drifted": 0, "legacy": False,
                                        "registry": False, "failed": 0,
                                        "unmatched": 0, "dir_missing": 0,
                                        "dir_conflict": 0, "mentioned": 0})
        m["entries"] += 1
        m["legacy"] = True
        m["newest"] = max(m["newest"], float(rec.timestamp or 0.0))
        k = str(tags.get(limits.entry_key) or "").strip()
        if k:
            legacy_keys.setdefault(str(name), {})[k] = \
                str(tags.get(limits.status_key) or "").strip()
    # registry：maps.json 侧车
    data = registry
    if data:
        for item in list_maps(data):
            m = maps.setdefault(item["name"],
                                {"entries": 0, "newest": 0.0,
                                 "drifted": 0, "legacy": False,
                                 "registry": False, "failed": 0,
                                 "unmatched": 0, "dir_missing": 0,
                                 "dir_conflict": 0, "mentioned": 0})
            m["entries"] += item["entries"]
            m["drifted"] += item["drifted"]
            m["failed"] += item.get("failed") or 0
            m["unmatched"] += item.get("unmatched") or 0
            m["dir_missing"] += item.get("dir_missing") or 0
            m["dir_conflict"] += item.get("dir_conflict") or 0
            # 提及信号：更新后被提到另一个状态的条目数。
            # 与 drifted 分开累加、分开渲染，绝不合并成一个数——
            # 「真相变过」与「有人这么说」是两种可信度。
            m["mentioned"] += item.get("mentioned") or 0
            if item.get("mention_state"):
                m["mention_state"] = item["mention_state"]
            if item.get("auto_error"):
                m["auto_error"] = item["auto_error"]
            if item.get("auto_incomplete"):
                m["auto_incomplete"] = True
            m["registry"] = True
            m["newest"] = max(m["newest"], item["newest_ts"])
            for k, e in ((data["maps"].get(item["name"]) or {})
                         .get("entries") or {}).items():
                registry_keys.setdefault(item["name"], {})[str(k)] = \
                    str((e or {}).get("status") or "").strip()
    now_ts = (now or _now)()
    out = []
    for name, m in maps.items():
        age = int((now_ts - m["newest"]) // 86400) if m["newest"] else None
        # 双源同键不同状态 = 冲突（legacy 旧值 vs registry 新值）。
        # 只进必退同纪律：任一侧改成一致即消失。
        lk = legacy_keys.get(name) or {}
        rk = registry_keys.get(name) or {}
        conflict_keys = sorted(k for k in set(lk) & set(rk)
                               if lk[k] != rk[k])
        out.append({
            "name": name, "entries": m["entries"],
            "drifted": m["drifted"],
            "failed": m["failed"],
            "unmatched": m["unmatched"],
            "dir_missing": m["dir_missing"],
            "dir_conflict": m["dir_conflict"],
            "mentioned": m["mentioned"],
            "mention_state": m.get("mention_state") or {},
            "conflicts": len(conflict_keys),
            "conflict_keys": conflict_keys[:5],
            "auto_error": m.get("auto_error") or "",
            "auto_incomplete": bool(m.get("auto_incomplete")),
            "source": ("both" if m["legacy"] and m["registry"]
                       else "registry" if m["registry"] else "legacy"),
            "newest_ts": m["newest"], "age_days": age,
            "stale": bool(age is not None and age > limits.stale_days),
        })
    out.sort(key=lambda x: (-x["entries"], x["name"]))
    return {
        "maps": out[:limits.max_maps],
        "maps_total": len(out),
        "hidden": max(0, len(out) - limits.max_maps),
        "stale_days_threshold": limits.stale_days,
    }


def map_detail(
    task: str,
    tp: dict[str, Any] | None,
    registry: dict[str, Any] | None,
    *,
    list_maps: Callable[[dict[str, Any]], list[dict[str, Any]]],
    task_tokens_fn: Callable[[str, list[str] | None], list[str]] = task_tokens,
    limits: DetailLimits,
) -> dict[str, Any]:
    """任务相关地图的全文投影（L1 递送）。纯读侧车，不检索不引模型。

    token = 任务正文 + tp 里的锚点实体名（与任务概况卡 task_profile
    同一个 :func:`task_tokens`；**不重新检索**）。命中判据：
    某张图的 desc 含任一 token，或其任一条目键含任一 token。
    """
    if not (task or "").strip():
        return {"task": "", "maps": [], "maps_total": 0, "hidden": 0,
                "row_cap": limits.max_rows}
    data = registry
    if not data:
        return {"task": task, "maps": [], "maps_total": 0, "hidden": 0,
                "row_cap": limits.max_rows}
    extras = [e.get("name") for e in (tp or {}).get("entities", [])
              if e.get("name")]
    tokens = [t for t in task_tokens_fn(task, extras) if t]
    if not tokens:
        return {"task": task, "maps": [], "maps_total": 0, "hidden": 0,
                "row_cap": limits.max_rows}
    matched: list[dict[str, Any]] = []
    for item in list_maps(data):
        m = (data.get("maps") or {}).get(item["name"]) or {}
        desc = str(m.get("desc") or item["name"]).lower()
        entries = m.get("entries") or {}
        hit_desc = any(t in desc for t in tokens)
        hit_keys = [k for k in entries
                    if any(t in str(k).lower() for t in tokens)]
        if not (hit_desc or hit_keys):
            continue
        rows = []
        for k in sorted(entries,
                        key=lambda k: -float(
                            (entries[k] or {}).get("updated_at") or 0.0)):
            e = entries[k] or {}
            row = "%s=%s" % (k, e.get("status") or "?")
            extra_bits = []
            for fk, fv in (e.get("fields") or {}).items():
                extra_bits.append("%s:%s" % (fk, fv))
            if e.get("note"):
                extra_bits.append(str(e["note"]))
            if extra_bits:
                row += "（%s）" % "；".join(extra_bits)
            if e.get("drift") or e.get("mention"):
                # 提及漂移也进 ⚠️ 前缀。它的方向
                # 与 drift 相反（「可能过期」而不是「已确认变了」），
                # 但递送动作一样是「这条要看一眼」，证据在 mention 里。
                row = "⚠️" + row
            rows.append(row[:80])
        rows.sort(key=lambda r: r.startswith("⚠️"), reverse=True)
        matched.append({
            "name": item["name"], "desc": m.get("desc") or item["name"],
            "entries": item["entries"], "drifted": item["drifted"],
            "age_days": item["age_days"],
            "matched_by": "desc" if hit_desc else "key",
            "matched_keys": hit_keys[:5],
            "rows": rows[:limits.max_rows],
            "hidden_rows": max(0, len(rows) - limits.max_rows),
        })
    matched.sort(key=lambda x: (-len(x["matched_keys"]),
                                -x["entries"], x["name"]))
    return {
        "task": task,
        "maps": matched[:limits.max_maps],
        "maps_total": len(matched),
        "hidden": max(0, len(matched) - limits.max_maps),
        "row_cap": limits.max_rows,
    }
