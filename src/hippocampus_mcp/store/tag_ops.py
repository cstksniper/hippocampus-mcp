# -*- coding: utf-8 -*-
"""update_tags 变更账本（tag_ops.jsonl，从 brain/core.py 提取）。

定位（承 common/constants.py 里 TAG_OPS_FILE 的注释）：它是 **append-only
流水**（同 save_ledger.jsonl），不参与层合并；ring buffer 防无限增长。
每次**执行**记一条全量前后影像，undo 按"当前值 == 记录的 after"才回滚
（fail-closed），所以账本写失败**不能吞**——撤销依据丢了必须让宿主知道。

为什么放 store/ 而不是 brain：这是纯持久化设施（JSONL 读写 + 环裁剪），
零领域逻辑；brain 只留三个薄包装（读/扫/追加）当适配层，ring 大小等
旋钮仍由 brain 侧调用时传入——``core.TAG_OPS_RING`` 这个被测试
monkeypatch 的名字必须继续在调用点现取，故本模块**不 import 任何
Config/常量旋钮**，只认显式参数（唯一例外是文件名 TAG_OPS_FILE，
全仓无 monkeypatch 先例）。

状态语义（与 brain 侧原实现逐字一致）：
- ``last_seq`` 先自增再落盘 ⇒ 写失败也不复用 id（"改之前是什么"
  不能指向一条会被覆盖的记录）；
- ``lines`` 落盘成功才 +1；裁剪只在 ``lines > trim_at`` 时触发，
  且裁剪失败（OSError）时 lines 保持原值——账本行数与磁盘实况
  允许短暂不一致，但绝不超过 trim_at 长期失控。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..common.constants import TAG_OPS_FILE

logger = logging.getLogger(__name__)


def ops_path(persist_dir: str | None) -> str | None:
    """账本文件路径；无持久化目录（纯内存 brain）返回 None。"""
    return (os.path.join(persist_dir, TAG_OPS_FILE)
            if persist_dir else None)


def read_ops(persist_dir: str | None) -> list[dict[str, Any]]:
    """读全部账本条目（坏行跳过——账本自己不该让 undo 用不了）。"""
    path = ops_path(persist_dir)
    if not path or not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 半行（崩溃在append中间）直接跳过
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def scan_ops(persist_dir: str | None) -> tuple[int, int]:
    """启动时扫一遍账本：返回 (行数, 末条 seq)。"""
    ops = read_ops(persist_dir)
    last = 0
    for rec in ops:
        try:
            last = max(last, int(rec.get("seq") or 0))
        except (TypeError, ValueError):
            continue
    return len(ops), last


def append_ops(persist_dir: str | None,
               entry: dict[str, Any],
               last_seq: int,
               lines: int,
               ring: int,
               trim_at: int) -> tuple[str | None, int, int]:
    """追加一条账本（ring buffer 裁剪）。

    Args:
        persist_dir: 持久化目录；None/空 = 纯内存，直接放弃落盘。
        entry: 条目内容；``seq`` / ``op_id`` 两键被本函数覆写。
        last_seq / lines: 调用方持有的账本状态（brain 的实例属性）。
        ring: 环大小（保留最新的 ring 条）。
        trim_at: 行数超过此值才触发裁剪。

    Returns:
        ``(op_id, last_seq, lines)``——状态三元组，调用方据此更新自身。
        op_id 为 None = 没落盘（无目录或写失败）；注意 last_seq 在写
        失败时**也已自增**（防 id 复用）。
    """
    path = ops_path(persist_dir)
    if not path:
        return None, last_seq, lines
    last_seq += 1
    entry["seq"] = last_seq
    entry["op_id"] = f"tagops_{last_seq}"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        lines += 1
        if lines > trim_at:
            lines = trim_ops(persist_dir, ring, lines)
    except OSError as exc:
        # 账本写失败不能吞：撤销依据丢了必须让宿主知道（内存里的标签
        # 已经改了，但"改之前是什么"从此无处可查）。
        logger.warning("[tag_ops] 账本追加失败：%s", exc)
        return None, last_seq, lines
    return entry["op_id"], last_seq, lines


def trim_ops(persist_dir: str | None, ring: int, lines: int) -> int:
    """裁剪到 ring 大小（保留最新的）。返回裁剪后的行数。

    行数不超 ring、无目录、裁剪失败（OSError）时原样返回 ``lines``。
    ⚠️ ``ring=0`` 时 ``ops[-0:]`` 是**全量**（Python 切片语义）——
    这是原实现的行为，刻意保留，调用方（config 默认 200）不会传 0。
    """
    path = ops_path(persist_dir)
    if not path:
        return lines
    ops = read_ops(persist_dir)
    if len(ops) <= ring:
        return lines
    keep = ops[-ring:]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in keep:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return len(keep)
    except OSError as exc:
        logger.warning("[tag_ops] 账本裁剪失败：%s", exc)
        return lines
