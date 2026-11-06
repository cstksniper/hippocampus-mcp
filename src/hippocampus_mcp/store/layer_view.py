# -*- coding: utf-8 -*-
"""层指纹投影（从 brain/core.py 提取）。

这批函数只**读**内存状态并投影成对外结论，不写任何东西：

- :func:`pending_layer_names` —— 此刻「内存有改动但还没写下去」的层
  （四象限的内存侧判据）；
- :func:`compute_all_layer_fps` —— 全部层的内容指纹（账本 begin/end
  行的 ``layer_fps`` 字段）；
- :func:`classify_layer_quadrants` —— F14-3 四象限逐层归属；
- :func:`ledger_view` —— F14 账本投影（``brain_stats.save_ledger``）。

为什么放 store/ 而不是 brain：判据全是「层文件 / 层属性」这一对关系，
零领域逻辑；brain 只留薄包装（把 ``_LAYER_ATTRS``、指纹函数、磁盘戳
比较在本侧绑好当适配层）。与 :mod:`hippocampus_mcp.store.ledger` 同域。

端口纪律：

* **不 import 任何 Config 旋钮**。指纹开关 ``layer_fp`` 由包装层在调用
  时从 ``core`` 命名空间现取后作参数传入（monkeypatch 面必需）；
* 指纹计算本身以 ``fingerprint_of(attr)`` 回调注入 —— brain 侧传的是
  ``lambda a: self._layer_fingerprint(getattr(self, a))``，运行期查找，
  测试 patch ``_layer_fingerprint`` 仍然生效；
* 时钟同样注入（``now`` / ``stamp_of``），账本投影的 ``interval_s`` 全靠
  ``time.monotonic()`` 差值，wall clock 相减遇 NTP 回拨会得负数。

预存怪癖按「只挪位置不改行为」原样保留：

* ``_layer_written_fp`` 为空（新库）时 :func:`pending_layer_names` 返回
  空列表 —— 不是"没有改动"，是"无从判断"，这一点由 ``pending_note`` 说清；
* :func:`classify_layer_quadrants` 里 ``name in stamped`` 先短路，**只有**
  戳表里有的层才去问 ``changed_by_other``；``persist_dir`` 为 None 时
  ``os.path.join("", name)`` 得到相对路径（因此几乎恒为 False）；
* 账本投影在 ``last`` 不是 dict 时返回 ``level=unknown`` 的整套字段，
  **绝不报 0** —— 0 会被读成「刚保存过」，那是比 unknown 更糟的谎话。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Collection, Mapping, Sequence


def pending_layer_names(
    layer_attrs: Mapping[str, str],
    written_fp: Mapping[str, str],
    fingerprint_of: Callable[[str], str],
) -> list[str]:
    """此刻**内存里有改动、但还没写下去**的层名。

    ``written_fp`` 为空（本进程从没落过盘）时返回空 —— 不是"没有改动"，
    是"无从判断"。从没写过的层也不算"改动"，算"还没建"。
    """
    if not written_fp:
        return []
    out: list[str] = []
    for name, attr in layer_attrs.items():
        if name not in written_fp:
            continue      # 从没写过这层：不算"改动"，算"还没建"
        if fingerprint_of(attr) != written_fp[name]:
            out.append(name)
    return out


def compute_all_layer_fps(
    layer_attrs: Mapping[str, str],
    fingerprint_of: Callable[[str], str],
    enabled: bool,
) -> dict[str, str]:
    """全部层的内容指纹（账本 begin/end 行的 ``layer_fps`` 字段）。

    指纹开关关闭时返回空 dict —— 账本行里就不带 ``layer_fps``，
    启动对账据此诚实降级（不猜）。
    """
    if not enabled:
        return {}
    return {name: fingerprint_of(attr) for name, attr in layer_attrs.items()}


def classify_layer_quadrants(
    layer_attrs: Mapping[str, str],
    written_fp: Mapping[str, str],
    stamped: Collection[str],
    persist_dir: str | None,
    fingerprint_of: Callable[[str], str],
    changed_by_other: Callable[[str, str], bool],
) -> dict[str, str]:
    """F14-3 四象限判定（口径 2）：逐层给出当前归属。

    内存侧（**主判据**）＝当前内存指纹 vs 本进程上次落盘指纹；
    磁盘侧（**仅辅证**）＝层文件戳还在不在（mtime 永不单独定罪 ——
    touch 不改内容时指纹不变，仍算 clean，只是记一笔 external_write）。
    本进程从没写过该层 → ``unknown``（是"还没建"，不是"改了没写"）。
    """
    out: dict[str, str] = {}
    for name, attr in layer_attrs.items():
        written = written_fp.get(name)
        if written is None:
            out[name] = "unknown"
            continue
        dirty = fingerprint_of(attr) != written
        # ``name in stamped`` 先短路是预存行为：戳表里没有就不必去问磁盘。
        changed = name in stamped and changed_by_other(
            name, os.path.join(persist_dir or "", name))
        if dirty:
            out[name] = "conflict" if changed else "unsaved"
        else:
            out[name] = "external_write" if changed else "clean"
    return out


def ledger_view(
    last: Any,
    lines: int,
    boot: Any,
    quadrant: Mapping[str, str],
    l1_dirty: bool,
    pending: Sequence[str],
    *,
    now: Callable[[], float],
    stamp_of: Callable[[float], str],
) -> dict[str, Any]:
    """F14：账本投影（``brain_stats.save_ledger``）。

    范式照抄 ``recall_feedback``：数字 + level + 人话 note。
    空库 / 从未落盘报 ``null`` + ``unknown``，**绝不报 0**。
    """
    if not isinstance(last, dict):
        return {
            "seq": 0,
            "last_save_at": None,
            "interval_s": None,
            "saves_in_ledger": 0,
            "last_phase": None,
            "last_layers": None,
            "pending_layers": None,
            "layers_quadrant": {},
            "l1_dirty": False,
            "boot_reconcile": None,
            "external_touched": [],
            "level": "unknown",
            "note": "No save records yet (new library, or this process has never persisted). "
                    "Reported as unknown rather than 0 -- 0 would read as having just saved.",
        }
    at_wall = last.get("at_wall")
    at_mono = float(last.get("at_mono") or now())
    # 间隔一律用单调钟：wall clock 相减遇到 NTP 回拨会得负数。
    interval = max(0.0, now() - at_mono)
    phase = last.get("phase")
    merged = last.get("merged_from_disk") or {}
    stamp = stamp_of(at_wall) if at_wall else "unknown"
    if phase != "end":
        level = "warn"
        note = (f"The last ledger entry is {phase}, not end: the previous save was **interrupted** "
                f"(crash / the host reclaimed the process). This is the only place it is visible after the fact.")
    else:
        hh = int(interval // 3600)
        mm = int((interval % 3600) // 60)
        if interval <= 3600:
            level = "ok"
        elif interval <= 21600:
            level = "warn"
        else:
            level = "stale"
        note = (f"Last save {stamp}, {hh}h{mm}m ago; 0 saves since. "
                f"In-memory changes made during this window are lost if the host reclaims the process.")
    # F14-5：启动对账结果非空 lost ⇒ 这是最重的一级（数据确实没写下去）；
    # 磁盘与账本不符次之。
    if isinstance(boot, dict):
        if boot.get("lost_since_last_save"):
            level = "critical"
            note = f"Boot reconcile: {boot['note']}"
        elif boot.get("disk_mismatch"):
            if level == "ok":
                level = "warn"
            note = f"Boot reconcile: {boot['note']}"
    if any(q in ("unsaved", "conflict") for q in quadrant.values()) and level == "ok":
        level = "warn"
    return {
        "seq": last.get("seq"),
        "last_save_at": stamp,
        "interval_s": round(interval, 1),
        "saves_in_ledger": lines,
        "last_phase": phase,
        "last_duration_ms": last.get("duration_ms"),
        "last_layers": last.get("layers"),
        # F14-1 已落地：这里不再是 None。四象限的"内存侧"维度＝
        # 当前内存指纹 vs 上次落盘指纹，不一致即"改了还没写下去"。
        "pending_layers": list(pending) if pending else [],
        "pending_note": (
            (f"{len(pending)} layers have unsaved changes: {', '.join(pending)}. "
             "If the host reclaims the process now, these changes are gone.")
            if pending else
            "No layer currently holds unwritten changes (memory matches the last persisted content)."
        ),
        # G0：上次 save 实际跳过了几个层（派生层脏集的效果，恒为空说明没生效）
        "last_layers_skipped": last.get("layers_skipped") or [],
        # F14-3：四象限逐层归属（unsaved=改了没写 / conflict=改了且别人也动了 /
        # clean / external_write / unknown=本进程还没写过）
        "layers_quadrant": dict(quadrant),
        # L1 有自己的脏标记与合并链路，不混进层四象限，只给一行事实
        "l1_dirty": l1_dirty,
        # F14-5：启动对账结果（None = 无账本可对）
        "boot_reconcile": boot,
        "external_touched": sorted(merged.keys()),
        "level": level,
        "note": note,
    }
