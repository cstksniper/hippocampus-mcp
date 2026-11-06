# -*- coding: utf-8 -*-
"""``brain.core.refresh_knowledge`` 的三块逻辑（从 core 抽取）。

**为什么值得搬**：``refresh_knowledge`` 是主题层总刷新入口，81 行里
「force/增量分叉决策」「宿主归纳迁移清点」「19 键回显体组装」三块全是
纯逻辑，剩下的 build/refresh/migrate 三个 topic 调用与 self 状态写
（``_knowledge_last_*``）归壳。搬走后壳只剩「门禁 → 取数 → 注入 →
计时帧」一条直线。

**依赖方向**：本模块零 import——``layer_epoch`` / ``reclip`` / ``build``
/ ``refresh`` / ``migrate_host_summaries`` / ``coverage_of`` /
``by_status`` / ``SUMMARY_KIND_HOST`` 八个 topic 包符号全由壳调用时现取
后注入（domain 同层禁互引，topic 住在 domain 但 G1 只准依赖 common；
沿用 knowledge_batch / knowledge_map_view 先例）。
``defaults`` / ``max_topics`` / ``now`` / ``t0`` / ``clock`` 是壳侧
装配值（配置现取、LNN 参考时间、计时窗）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. **force + 空层走增量分支**：``if force and knowledge``——对空层 force
   不世代 +1、不取 old 快照、不调 reclip，但报告 ``rebuilt=True`` 且
   ``new_topics=len(全部)``。「重建」旗标与世代/迁移**解耦**。
2. **增量分支的 reclip 守卫**：``if knowledge``——空层不调 reclip
   （空 dict 调了也返回 0，但调用次数与原实现逐次对齐）。
3. **reclip 原地改成员，build 再裁一遍**：build 对已有簇按同一 alive
   集合二次裁剪（幂等），所以报的 ``members_dropped`` 只来自 reclip
   那一次。
4. **build 返回新 dict（浅拷贝），旧 topic dict 被原地改** ⇒ force
   分支的 ``old = dict(knowledge)`` 快照必须在 build **之前**取，迁移
   读到的才是真旧层（collect 内 plan_rebuild 先行即为此）。
5. ``now`` = ``manager._reference_time or _now()``——LNN 参考时间优先
   于墙钟，由壳算好注入（store_memory 会把 _reference_time 写成 seed
   时间戳，探针冻结必须在 seed 之后）。
6. ``took`` 跨「取数 + build + refresh + migrate」整窗：``t0`` 与
   ``clock`` 都由壳给（域内只取差），报告 ``took_s=round(took,3)``，
   而 ``_knowledge_last_took`` 存**未取整**的原值（壳从 collect 第四
   返回值拿，不读回显体——避免二次取整）。
7. ``at_cap`` 的 ``max_topics`` 是**第二次** ``_topic_defaults()``
   现取（与开头的 ``_td`` 分离：测试猴子补丁 ``_topic_defaults`` 时
   两处都必须是调用时求值，M1b 纪律）。
8. ``count_lost_host`` 的 ``max(0, ...)`` 是防御性的：迁移数不可能超
   过旧宿主归纳数，但别报负数；只数 old 里 ``kind==host`` 且 summary
   非空的簇。
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# 块 1：force / 增量分叉决策（旧壳 2928-2936）
# ======================================================================
def plan_rebuild(
    knowledge: dict[str, Any],
    alive: list[str],
    *,
    force: bool,
    layer_epoch: Callable[[dict[str, Any]], int],
    reclip: Callable[[dict[str, Any], list[str]], int],
) -> tuple[dict[str, Any], dict[str, Any] | None, int | None, int]:
    """决定这一轮刷新是「重画」还是「增量」，返回重建三参 + 摘掉的成员数。

    Args:
        knowledge: 当前主题层（壳直传 ``self.knowledge``）。
        alive: 存活记录 id 全集（含暂无向量的）。
        force: 是否重画簇边界。
        layer_epoch: ``topic_mod.layer_epoch``（层世代读取）。
        reclip: ``topic_mod.reclip``（按 alive 重裁成员，原地改）。

    Returns:
        ``(base, old, epoch, dropped)``——``base`` 是传给 build 的底册
        （force 为全新空 dict，增量为原层对象）；``old`` 仅 force 且
        层非空时为旧层浅拷贝快照（供归纳迁移），否则 None；``epoch``
        仅重画时为旧世代 +1，否则 None；``dropped`` 是本轮摘掉的
        成员引用条数。

    怪癖复刻：force + 空层走增量分支（怪癖 1）；空层不调 reclip
    （怪癖 2）。
    """
    if force and knowledge:
        # 重画：旧层留着做归纳迁移，然后从空开始建（不 reclip——
        # 新层里没有需要裁掉的死成员）。
        old = dict(knowledge)
        epoch = layer_epoch(old) + 1
        base: dict[str, Any] = {}
        dropped = 0
    else:
        old = None
        epoch = None
        base = knowledge
        dropped = reclip(knowledge, alive) if knowledge else 0
    return base, old, epoch, dropped


# ======================================================================
# 块 2：宿主归纳迁移清点（旧壳 2947-2952）
# ======================================================================
def count_lost_host(
    old: dict[str, Any],
    migrated: list[dict[str, Any]],
    *,
    summary_kind_host: str,
) -> int:
    """重画时没搬过去的旧宿主归纳条数（自曝，不静默）。

    Args:
        old: 旧层快照（force 分支的 ``dict(knowledge)``）。
        migrated: ``migrate_host_summaries`` 的返回（搬成的列表）。
        summary_kind_host: ``topic_mod.SUMMARY_KIND_HOST`` 常量。

    Returns:
        ``max(0, 旧宿主归纳数 - 搬成数)``（怪癖 8：防御性下界）。

    只数 ``summary_kind == host`` 且 summary 非空的簇——空 summary 的
    host 簇本来就没有可搬的资产。
    """
    n_old_host = sum(
        1 for t in old.values()
        if isinstance(t, dict)
        and str(t.get("summary_kind") or "") == summary_kind_host
        and str(t.get("summary") or "").strip())
    return max(0, n_old_host - len(migrated))


# ======================================================================
# 块 3：回显体组装（旧壳 2960-2980）
# ======================================================================
def assemble_report(
    *,
    knowledge: dict[str, Any],
    alive_total: int,
    force: bool,
    before: int,
    old: dict[str, Any] | None,
    migrated: list[dict[str, Any]],
    lost_host: int,
    dropped: int,
    no_vec: int,
    took: float,
    max_topics: int,
    coverage_of: Callable[[dict[str, Any], int], dict[str, Any]],
    by_status: Callable[[dict[str, Any]], dict[str, int]],
    layer_epoch: Callable[[dict[str, Any]], int],
) -> dict[str, Any]:
    """组装刷新回显体：coverage 七键在前 + 12 键更新（键序即信号）。

    怪癖复刻：

    - ``new_topics`` 跟 ``force`` 旗标不跟 ``old``：force（含空层
      force）报全部簇数，增量报增量差（怪癖 1）。
    - ``dropped_old_topics = len(old) if old else 0``——空层 force 报 0。
    - ``at_cap`` 用 ``>=``（撞到主题数上限 ⇒ 之后的记录是被**强制归入**
      最相近的簇，必须看得见，HIPPOCAMPUS_TOPIC_MAX）。
    - ``took_s = round(took, 3)``；分母 ``alive_total`` 是 **active
      总数**（含无向量），不是「有向量的条数」——后者会让覆盖率虚高。
    """
    out = coverage_of(knowledge, alive_total)
    out.update({
        "enabled": True,
        "rebuilt": bool(force),
        "dropped_old_topics": len(old) if old else 0,
        "migrated_host": migrated,
        "lost_host": lost_host,
        "epoch": layer_epoch(knowledge),
        # force 是清空重画，减 before 会算出负数——那时"新增"就等于全部。
        "new_topics": (len(knowledge) if force
                       else len(knowledge) - before),
        "members_dropped": dropped,
        # 无向量的存活记录数：覆盖率里的 unassigned 有一部分就是它们。
        # 不报出来，用户只会看到"覆盖率掉了"而找不到原因。
        "no_vector": no_vec,
        "took_s": round(took, 3),
        "at_cap": len(knowledge) >= max_topics,
        "by_status": by_status(knowledge),
    })
    return out


# ======================================================================
# 编排入口（壳的唯一调用点）
# ======================================================================
def collect(
    items: list[dict[str, Any]],
    info: dict[str, dict[str, Any]],
    knowledge: dict[str, Any],
    *,
    force: bool,
    defaults: Any,
    max_topics: int,
    now: float | None,
    t0: float,
    clock: Callable[[], float],
    layer_epoch: Callable[[dict[str, Any]], int],
    reclip: Callable[[dict[str, Any], list[str]], int],
    build: Callable[..., dict[str, Any]],
    refresh: Callable[..., None],
    migrate_host_summaries: Callable[..., list[dict[str, Any]]],
    coverage_of: Callable[[dict[str, Any], int], dict[str, Any]],
    by_status: Callable[[dict[str, Any]], dict[str, int]],
    summary_kind_host: str,
) -> tuple[dict[str, Any], dict[str, Any], float, int]:
    """refresh_knowledge 全流程：分叉 → build → refresh → 迁移 → 回显。

    三步顺序有语义（与原实现逐行对齐）：先按存活记录重裁 → 再增量归入
    未分配的 → 最后重算派生字段。反过来会先把已删的成员算进质心里。

    Args:
        items: 有向量的存活记录（``_knowledge_items()``）。
        info: 存活全集 ``id -> item``（含无向量，``require_vec=False``）。
        knowledge: 当前主题层（壳直传 ``self.knowledge``）。
        force: 是否重画簇边界。
        defaults: ``_topic_defaults()`` 现取快照（开头那次）。
        max_topics: **第二次** ``_topic_defaults()`` 现取的 max_topics。
        now: ``manager._reference_time or _now()``（壳算好注入）。
        t0: 计时窗起点（壳在取数前打戳）。
        clock: 取当前墙钟的 callable（壳给 ``time.time``）。
        其余: 八个 topic 包符号（见模块 docstring）。

    Returns:
        ``(new_knowledge, report, took, dropped)``——新主题层（壳负责
        赋回 ``self.knowledge``）、回显体、未取整耗时、摘掉的成员数。
    """
    alive = list(info.keys())
    no_vec = len(info) - len(items)
    before = len(knowledge)
    base, old, epoch, dropped = plan_rebuild(
        knowledge, alive, force=force, layer_epoch=layer_epoch,
        reclip=reclip)
    new_knowledge = build(items, base, alive_ids=alive, epoch=epoch,
                          defaults=defaults)
    refresh(new_knowledge, info, defaults=defaults, now=now)
    # 归纳迁移：host 归纳是宿主资产，重画边界不该把它冲掉。
    # 搬过去后**故意不重算签名** ⇒ 地图上标 stale ⇒ 提示复核。
    migrated: list[dict[str, Any]] = []
    lost_host = 0
    if old:
        migrated = migrate_host_summaries(old, new_knowledge)
        lost_host = count_lost_host(old, migrated,
                                    summary_kind_host=summary_kind_host)
    took = clock() - t0
    report = assemble_report(
        knowledge=new_knowledge, alive_total=len(alive), force=force,
        before=before, old=old, migrated=migrated, lost_host=lost_host,
        dropped=dropped, no_vec=no_vec, took=took, max_topics=max_topics,
        coverage_of=coverage_of, by_status=by_status,
        layer_epoch=layer_epoch)
    return new_knowledge, report, took, dropped
