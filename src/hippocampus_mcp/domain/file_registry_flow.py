# -*- coding: utf-8 -*-
"""``brain.core.file_registry_summary`` 族编排
（从 core 抽取）。

**为什么值得搬**：建档摘要是 session_start 热路径上的一段编排——
「config 门 → 侧车存在性 → 间隔门控（锁内重判）→ 带预算重扫 → 对账 →
汇总回退」六步全是流程控制，值计算（扫描/汇总/索引）本来就在
``map/file_registry.py``。搬走后壳只剩「读 config / 现取 fr_mod 入口 /
注入对账回调」三行直线。

**依赖方向**：本模块零 import——``fr_mod`` 的四个入口（load / scan /
update / summarize）由壳在调用时从本命名空间现取注入（patch
``core.fr_mod`` 即生效）；对账回调（读 ``self.source_sigs``，brain 态）
裸引用注入；``now`` 时刻与 ``interval_h`` / ``limits`` 由壳算好传入；
路径归一化经 ``path_of`` 回调注入（os 不进 domain）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. 间隔门控在**锁内重判**（``_do`` 在 ``update`` 的侧车锁内跑）——
   两个会话同时开场不该各扫一遍；
2. 门控真值 ``interval_h > 0``：0 = 每次都扫（关闭节流）；
   ``now - last_checked < interval_h*3600`` 才跳过（等于间隔就重扫）；
3. 扫描失败不炸不写：``except → (False, reg)``——报上一次的摘要，
   ``last_checked`` 不前进（下次开场会再试）；
4. ``update`` 本身失败（锁忙等）→ 回退 ``load`` 报旧摘要——热路径
   绝不因建档失败而炸；
5. 无侧车返回 ``None``（opt-in）；``update`` 的 ``(result, saved)``
   二元解包语义保留；
6. 对账判据 (mtime_ns, size) **双等**：size 缺省哨兵 **-1 / -2 错开**
   ——两侧都没记录时永不相等，防「都没记录」被误判成「已同步」；
7. ``last_synced`` **逐条现取** ``now()``（惰性回调，调用次数与
   原实现逐次对齐——冻结钟下两侧同值）。
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
    """建档摘要编排：存在性 → 锁内门控重扫 → 失败回退 → 汇总。

    Args:
        persist_dir: 脑持久目录（侧车 ``file_registry.json`` 所在）。
        now: 本时刻（壳 ``_now()`` 现取，门控 / last_checked / 汇总共用）。
        interval_h: 重扫间隔小时数（壳读 config；**0 = 关闭节流**）。
        limits: 扫描预算 ``(max_files, budget_s)``（壳 ``scan_limits()``
            现取）。
        load / scan / update / summarize: ``fr_mod`` 四入口（壳现取注入）。
        reconcile: 对账回调 ``(reg) -> 清掉条数``（brain 侧
            ``_fr_reconcile_source_sigs``，读 ``self.source_sigs``）。

    Returns:
        汇总 dict；未启用 / 无侧车 / 无 reg 时 ``None``。
    """
    if load(persist_dir) is None:
        return None

    def _do(reg):
        # 间隔门控在锁内重判：两个会话同时开场不该各扫一遍
        if interval_h > 0 and now - float(
                reg.get("last_checked") or 0.0) < interval_h * 3600:
            return False, reg
        try:
            scan(reg, limits[0], limits[1])
            reg["last_checked"] = now
            reconcile(reg)
        except Exception:  # noqa: BLE001 —— 扫描失败：不写，报旧摘要
            return False, reg
        return True, reg

    try:
        reg, saved = update(persist_dir, _do, wait=0.0)
    except Exception:  # noqa: BLE001 —— 热路径不因建档扫描失败而炸
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
    """stale 条目 × 源签名层对账：hippo 吃过当前内容版本 ⇒ 清 stale。

    判据是 (mtime_ns, size) 双等——与建档层同 threat model（墙钟比对
    的对抗性场景本来就不在防御范围内）。**原地改 reg**（与原实现同
    对象语义），返回清掉的条数（0 = 没得可清，常态）。

    Args:
        reg: 建档注册表（``files`` 为 ``{rel: entry}``）。
        sig_by_path: ``fr_mod.source_sig_index`` 产出的绝对路径索引
            （壳构造，两边同口径——归一化/取最新只此一份）。
        path_of: ``rel → 索引键`` 归一化回调（``normcase(join(root,
            str(rel)))``，os 不进 domain）。
        now: ``last_synced`` 时间源（**逐条现取**，惰性回调）。
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
