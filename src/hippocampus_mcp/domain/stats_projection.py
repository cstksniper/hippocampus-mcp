# -*- coding: utf-8 -*-
"""``brain.core.stats`` 的六块计算逻辑（从 core 抽取）。

**为什么值得搬**：``stats`` 是宿主必调的自查入口，183 行里除「调用各层
自曝方法」外，还缠着六块纯计算——状态分布、L0↔L1 悬挂引用、按日活动
窗口、源签名覆盖率、共库写者投影、遗忘计数。它们全是纯 dict/list 运算，
却因为长在 self 面的上下文里而无法单测；搬走后 ``stats`` 只剩
「取数 → 投影 → 组装回显」一条直线。

**依赖方向**：本模块只 import 标准库（domain → 仅 common 的纪律这里连
common 都没用上——所有记录/计数/pid 均由壳现取后传入）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``state_counts`` 用 ``dict.get`` 递惯而非 ``Counter``——两者计数结果
   相同，但**键序 = 首次出现序**必须保留（回显体直接 JSON 化，顺序即信号）。
2. ``dangling_report`` 的受影响条数只对**真有悬挂**的记录 +1；一条记录
   多个悬挂 id 只计一次（``if hit`` 门在 update 之外）。
3. ``daily_activity`` 的 totals 统计**全量天数**（不是回显窗口内）；
   ``recent`` 取 ``sorted(daily_counts)[-n_days:]``——``_resolve_daily_days``
   恒返回 ≥1（显式 max(...,1) / env 校验 >0 / 默认 14），故 ``[-0:]``
   整表怪癖在本路径不可达（记录在案，别"修"成 ``[:n]``）。
4. ``source_sig_block`` 的 referenced 对 source 做 ``str().strip()`` 双重
   归一：空白串等价于没 source，不计入也不挡筛。
5. ``writers_block`` 的 ``is_self`` 按 **pid 相等**判定（不是对象身份），
   pid 缺失时恒 False。
6. ``forgotten_count`` 用 ``r.get("forgotten")`` 而非 ``r["forgotten"]``——
   raw 条目缺键是常态（老版本落的盘），下标访问会 KeyError。
"""
from __future__ import annotations

from typing import Any, Iterable


# ======================================================================
# 块 1：状态分布（旧壳 3371-3373）
# ======================================================================
def state_counts(records: Iterable[Any]) -> dict[str, int]:
    """L1 记忆按生命周期状态计数（键序 = 首次出现序，见模块 docstring 怪癖 1）。

    Args:
        records: 库存记录迭代器（鸭子类型：有 ``.state`` 属性）。
    """
    states: dict[str, int] = {}
    for r in records:
        states[r.state] = states.get(r.state, 0) + 1
    return states


# ======================================================================
# 块 2：L0↔L1 引用完整性（旧壳 3379-3386）
# ======================================================================
def dangling_report(
    records: Iterable[Any],
    raw_ids: set[str],
) -> tuple[set[str], int]:
    """被 L1 引用却不在 L0 里的 source_id 集合 + 受影响的 L1 条数。

    Args:
        records: 库存记录迭代器（鸭子类型：有 ``.source_ids``）。
        raw_ids: L0 层全部 raw id 的集合。

    Returns:
        ``(dangling_ids, affected_records)``。受影响条数只对真有悬挂的
        记录 +1，一条记录多个悬挂 id 只计一次（见模块 docstring 怪癖 2）。
    """
    dangling: set[str] = set()
    dangling_recs = 0
    for _r in records:
        hit = [s for s in (_r.source_ids or []) if s not in raw_ids]
        if hit:
            dangling.update(hit)
            dangling_recs += 1
    return dangling, dangling_recs


# ======================================================================
# 块 3：按日活动窗口（旧壳 3393-3425 的 daily_activity 键）
# ======================================================================
def daily_activity(daily_counts: dict[str, dict[str, int]],
                   n_days: int) -> dict[str, Any]:
    """按日写入/召回计数的回显体（窗口 + 全量 totals 分离自曝）。

    Args:
        daily_counts: ``{day: {"recall": n, "ingest": n}}``。
        n_days: 回显窗口条数（由壳经 ``_resolve_daily_days`` 现取，
            恒 ≥1；显式传大数只看存量，不补空位）。

    Returns:
        ``recent`` / ``totals`` / ``kept_days`` / ``shown_days`` 四键。
        totals 统计**全量天数**（见模块 docstring 怪癖 3）。
    """
    days = sorted(daily_counts)[-n_days:]
    total_recall = sum(d.get("recall", 0) for d in daily_counts.values())
    total_ingest = sum(d.get("ingest", 0) for d in daily_counts.values())
    return {
        "recent": [
            {"day": d,
             "recall": daily_counts[d].get("recall", 0),
             "ingest": daily_counts[d].get("ingest", 0)}
            for d in days
        ],
        "totals": {"recall": total_recall, "ingest": total_ingest},
        "kept_days": len(daily_counts),
        # 自曝窗口：宿主看到"只有 7 天"时要能知道是窗口而不是存量
        "shown_days": len(days),
    }


# ======================================================================
# 块 4：源签名覆盖率（旧壳 3455-3461 的 source_sig 键）
# ======================================================================
def source_sig_block(source_sigs: dict[str, Any],
                     records: Iterable[Any]) -> dict[str, int]:
    """源签名采集覆盖率：recorded（采集过）vs referenced（库内被引用）。

    Args:
        source_sigs: ``brain.source_sigs``（采集过的 source → 签名）。
        records: 库存记录迭代器（鸭子类型：有 ``.source``）。

    Returns:
        ``{"recorded": n, "referenced": n}``。referenced 对 source 做
        ``str().strip()`` 双重归一（见模块 docstring 怪癖 4）。
    """
    return {
        "recorded": len(source_sigs),
        "referenced": len({str(r.source or "").strip()
                           for r in records
                           if str(r.source or "").strip()}),
    }


# ======================================================================
# 块 5：共库写者投影（旧壳 3463-3473 的 writers 键）
# ======================================================================
def writers_block(active: list[dict[str, Any]], stale_cleaned: int,
                  self_pid: int) -> dict[str, Any]:
    """共库写者可见化：此刻的真相（不是上次落盘的快照）。

    Args:
        active: ``brain.writers_active``（心跳未过期的写者列表）。
        stale_cleaned: ``brain.writers_stale_cleaned``（上次清理数）。
        self_pid: 本进程 pid（壳传 ``os.getpid()``，替身可注入）。

    Returns:
        ``active`` / ``stale_cleaned`` / ``self_pid`` / ``list`` 四键；
        ``is_self`` 按 pid 相等判定（见模块 docstring 怪癖 5）。
    """
    return {
        "active": len(active),
        "stale_cleaned": stale_cleaned,
        "self_pid": self_pid,
        "list": [
            {"pid": w.get("pid"), "host": w.get("host"),
             "cwd": w.get("cwd"), "saves": w.get("saves"),
             "is_self": w.get("pid") == self_pid}
            for w in active
        ],
    }


# ======================================================================
# 块 6：遗忘 raw 计数（旧壳 3477）
# ======================================================================
def forgotten_count(raw_log: Iterable[dict[str, Any]]) -> int:
    """L0 里被标记 forgotten 的条数（``.get`` 保老盘缺键不崩，怪癖 6）。"""
    return sum(1 for r in raw_log if r.get("forgotten"))
