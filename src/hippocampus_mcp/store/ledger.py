# -*- coding: utf-8 -*-
"""保存账本（save_ledger.jsonl，从 brain/core.py 提取）。

定位（承 common/constants.py 里 LEDGER_FILENAME 的注释）：一次保存 =
``begin`` + ``end`` 两行，异常路径记 ``fail``。**追加式是硬要求**——
全量重写会让账本自己变成新的丢数据点。ring buffer 防无限增长。

为什么放 store/ 而不是 brain：这是纯持久化设施（JSONL 读写 + 环裁剪 +
坏文件隔离 + 启动对账判定），零领域逻辑；brain 只留薄包装（路径/读尾/
追加/裁剪/对账）当适配层。账本行数（``lines``）与末条（``last``）的状态
语义由包装层落回实例属性，本模块只返回纯结果：

- :func:`append` 返回 ``(是否落盘, 追加后行数)``——写失败时行数不动、
  末条不更新（「改之前是什么」不能指向一条没写成功的记录）；
- :func:`trim` 返回新行数或 ``None``（``None`` = 调用方保持原值：
  无 persist_dir / 读失败 / 写失败三种「没法数」的情形）。

旋钮面（同 tag_ops）：``core.LEDGER_RING`` 被测试 monkeypatch
控环大小（test_save_ledger L5）⇒ 本模块函数只认显式 ``ring`` 参数，
**不 import 任何 Config/常量旋钮**；包装层在调用时从 core 命名空间现取
（``LEDGER_TRIM_AT`` 虽无 monkeypatch 先例，同路径一并现取，少一处
特例）。唯一例外是文件名 ``LEDGER_FILENAME`` 与尾部长 ``_LEDGER_TAIL_KEEP``
——全仓无 monkeypatch 先例，从 common.constants 导入。

ring=0 的 Python 切片怪癖（``lines[-0:]`` = 全量）按「只挪位置不改
行为」纪律原样保留：ring=0 时裁剪是逐字重写（行数不变）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Sequence

from ..common.constants import LEDGER_FILENAME, _LEDGER_TAIL_KEEP


def ledger_path(persist_dir: str | None) -> str | None:
    """账本文件路径；无 persist_dir（纯内存 brain）返回 None。"""
    if not persist_dir:
        return None
    return os.path.join(persist_dir, LEDGER_FILENAME)


def read_tail(persist_dir: str | None) -> tuple[dict[str, Any] | None, int, list]:
    """读账本：最后一条 + 总行数 + 尾部若干行（启动对账用）。

    坏行逐行跳过、坏文件按惯例隔离为 ``.corrupt-<ts>``，
    **绝不阻塞启动**（设计 §8 风险 9：账本缺失/损坏报 unknown 并继续）。
    尾部只留最近 ``_LEDGER_TAIL_KEEP`` 行：fail 行要回看同 seq 的 begin
    才知道「当时声明要写什么」，全量读没必要。
    """
    path = ledger_path(persist_dir)
    if not path or not os.path.exists(path):
        return None, 0, []
    last: dict[str, Any] | None = None
    tail: list[dict[str, Any]] = []
    n = 0
    bad = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    last = json.loads(line)
                    tail.append(last)
                    if len(tail) > _LEDGER_TAIL_KEEP:
                        tail.pop(0)
                except ValueError:
                    bad = True      # 坏行跳过，但行数照记
    except OSError:
        return None, 0, []
    if bad:
        try:
            os.replace(path, path + ".corrupt-" + str(int(time.time())))
        except OSError:
            pass
    return last, n, tail


def append(persist_dir: str | None, entry: dict[str, Any], lines: int,
           trim_at: int, ring: int) -> tuple[bool, int]:
    """追加一行（达到 trim_at 时顺手裁剪到 ring）。

    返回 ``(是否落盘, 追加后行数)``：写失败（OSError）时行数不动，
    调用方不应更新末条；裁剪读失败/写失败时行数保持追加后的值
    （账本行数与磁盘实况允许短暂不一致，但绝不超过 trim_at 长期失控）。
    """
    path = ledger_path(persist_dir)
    if not path:
        return False, lines
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
    except OSError:
        return False, lines
    lines += 1
    if lines >= trim_at:
        trimmed = trim(persist_dir, ring)
        if trimmed is not None:
            lines = trimmed
    return True, lines


def trim(persist_dir: str | None, ring: int) -> int | None:
    """ring buffer：只保留最近 ring 条，保证账本体积有界。

    返回新的行数；``None`` = 调用方保持原值（无 persist_dir / 读失败 /
    写失败）。行数不超 ring 时按磁盘实况重数后返回（自愈路径）。
    """
    path = ledger_path(persist_dir)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if len(lines) <= ring:
        return len(lines)
    keep = lines[-ring:]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(keep) + "\n")
        os.replace(tmp, path)
    except OSError:
        return None
    return len(keep)


def reconcile_on_boot(
    last: dict[str, Any],
    tail: list,
    layer_names: Sequence[str],
    layer_fp: Callable[[str], str],
) -> dict[str, Any]:
    """启动对账：报「哪层有改动却没写下去」（lost_since_last_save）。

    load 之后内存 == 磁盘，所以「内存指纹 vs 账本指纹」即「磁盘 vs 账本」。
    三种可判情形（四象限判定）：

    * 末条 ``end`` 且带层指纹 → 磁盘与账本不符的层进 ``disk_mismatch``
      （外部工具改了内容 / 写坏；多写进程正常落盘会 append 自己的 end，
      不会误报）；
    * 末条 ``begin``（无同 seq 的 end/fail）→ 上次保存**没走完**：
      磁盘指纹 ≠ begin 指纹的层 ⇒ 声明要写但磁盘上不是那份 ⇒ 报 lost；
    * 末条 ``fail`` → 回看尾部找同 seq 的 begin，按上一条判，error 带出。

    行里没有 ``layer_fps``（旧格式/指纹关）→ ``layer_fps_absent``，不判内容。
    ``layer_fp`` 按层名取当前指纹，**惰性调用**（只在行里带了该层指纹时
    才算）——由包装层从 brain 实例注入（本模块不 import brain）。
    双钟纪律：不新增时间字段，lost 的时间语义引用账本行的 at_wall/at_mono。
    **已知局限**：进程带着改动被杀且从未发起 save（无 begin）⇒ 账本无从
    知晓——那是 F11 的领域。

    Returns:
        对账报告 dict：``phase`` / ``seq`` / ``at_wall`` / ``error`` 原样
        带出 + 判定结果 + 人话 note。告警日志与 ``_boot_reconcile`` 状态
        落回由包装层负责（本模块无日志副作用）。
    """
    phase = last.get("phase")
    seq = last.get("seq")
    rec: dict[str, Any] = {
        "phase": phase, "seq": seq, "at_wall": last.get("at_wall"),
        "lost_since_last_save": None, "disk_mismatch": [],
        "error": last.get("error"), "layer_fps_absent": False, "note": "",
    }
    mismatch: list[str] = []
    lost: dict[str, Any] | None = None

    if phase == "end":
        end_fps = last.get("layer_fps")
        if not isinstance(end_fps, dict):
            rec["layer_fps_absent"] = True
            rec["note"] = "The last ledger entry has no layer fingerprints (old format or fingerprinting off); content reconciliation skipped."
        else:
            for name in layer_names:
                fp = end_fps.get(name)
                if fp is not None and layer_fp(name) != fp:
                    mismatch.append(name)
            rec["disk_mismatch"] = mismatch
            rec["note"] = (
                f"Layers whose disk content mismatches the last ledger entry (end seq={seq}): {', '.join(mismatch)} "
                "-- changed by an external tool or corrupted; the ledger has no matching end." if mismatch else
                "Disk content matches the last ledger entry; no unexplained changes.")
    elif phase in ("begin", "fail"):
        begin = last if phase == "begin" else None
        if phase == "fail":
            for row in reversed(tail):
                if row.get("phase") == "begin" and row.get("seq") == seq:
                    begin = row
                    break
        begin_fps = begin.get("layer_fps") if isinstance(begin, dict) else None
        if not isinstance(begin_fps, dict):
            rec["layer_fps_absent"] = True
            rec["note"] = (f"The last ledger entry is {phase} (the previous save did not finish), "
                           "and the row carries no layer fingerprints, so the lost layers cannot be identified.")
        else:
            lost = {}
            counts = begin.get("layers") if isinstance(begin, dict) else {}
            for name in layer_names:
                fp = begin_fps.get(name)
                if fp is not None and layer_fp(name) != fp:
                    lost[name] = (counts or {}).get(name)
            rec["lost_since_last_save"] = lost
            if lost:
                detail = ", ".join(f"{k}({v} records)" for k, v in lost.items())
                rec["note"] = (f"The previous save (seq={seq}) did not finish; the following layers declared writes "
                               f"but the disk no longer holds that content: {detail}.")
            else:
                rec["note"] = (f"The previous save (seq={seq}) did not finish, but every layer's disk content matches "
                               "the begin declaration -- the write likely completed and only the end row was lost.")
    else:
        rec["note"] = f"The last ledger entry has unknown phase={phase}; reconciliation skipped."

    return rec
