"""地图自动巡检族：auto verify + mention drift。

从 ``brain.core`` 整搬两件（与 :meth:`map_verify` 的签名级比对分工：
C1/C2 读真相源，mention 读记忆正文——弱信号写独立的 ``ent["mention"]``
不覆盖 ``ent["drift"]``）。G1：map 包能力全部由壳**调用时现取**注入
（``auto_run`` / ``auto_needs_registry`` / ``needs_run_fn`` / ``run_fn``
/ ``registry_load`` / ``update_fn`` 回调，``busy_exc`` 异常类按值传，
config 三旋钮在壳上调用时现取按值传）；``records_fn`` 是惰性回调——
mention 的记录快照只在锁内、过了间隔门控之后才物化（快照一份防
并发 ingest 的 "list changed during iteration"）。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * 三条运行时纪律两件共有：间隔门控（默认 6h，**锁内重判**——门控与
    run 同临界区重读最新时钟，防两会话各跑一遍互相覆盖；``force`` 或
    interval<=0 跳过门控）、best-effort 落盘（SidecarBusy→skipped:busy、
    OSError→save_failed、其余异常→error 自曝，巡检可再生不炸开场）、
    侧车不存在（update 返回 None）→ enabled=False 未启用；
  * auto：一次读盘两用（建档 root 是 C1 解析根、files 是 C2 对账源）；
    ``needs_registry`` 为假不传 registry（零重复 IO）；**只有真有 auto
    配方的库才推进 ``auto_checked_at``**（返回 ``bool(res.get("maps"))``
    作为是否保存的判据——没配方的库零写盘）；
  * mention：``needs_run`` 为假直接 skipped:no_maps（不门控不推进时钟）；
    推进 ``mention_checked_at`` 无条件（与 auto 的条件推进刻意不同）。
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
    """C 期：按信号自动 verify——C1 值级 / C2 目录对账（图级 ``auto`` 配方）。"""
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
        # 一次读盘两用：建档 root 是 C1 的解析根，建档 files 是 C2 的
        # 对账数据源（C2 零重复 IO 全靠它）。没有 dir_reconcile 配方时
        # 不读 files 之外的东西——也没有第二次读。
        reg = registry_load(persist_dir)
        res = auto_run(
            data, registry_root=(reg or {}).get("root") or None,
            budget_s=budget_s,
            force=force, now=now,
            registry=(reg if (reg and auto_needs_registry(data))
                      else None))
        # 只有真有 auto 配方的库才推进巡检时钟（没配方的库零写盘）
        data["auto_checked_at"] = now
        return bool(res.get("maps")), res

    try:
        res, saved = update_fn(persist_dir, _do)
    except busy_exc:
        return {"enabled": True, "skipped": "busy"}
    except OSError:
        return {"enabled": True, "save_failed": True}
    except Exception as exc:  # noqa: BLE001 —— 自动 verify 不炸开场
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
    """闸三：更新后被提到「另一个状态」⇒ 条可疑过期（写 ``ent["mention"]``）。"""
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
            # 快照一份：同进程若另一线程正在 ingest，直接迭代活列表会
            # 撞 "list changed size during iteration"（references 拷贝，
            # 不复制记录本体，代价可忽略）
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
    except Exception as exc:  # noqa: BLE001 —— 巡检不炸开场
        return {"enabled": True,
                "error": "%s: %s" % (type(exc).__name__, exc)}
    if res is None:
        return {"enabled": False, "reason": "未启用地图层"}
    res["enabled"] = True
    return res
