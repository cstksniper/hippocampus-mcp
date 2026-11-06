# -*- coding: utf-8 -*-
"""L1 改写历史账本（mem_history.jsonl，形态逐字镜像 tag_ops.py）。

定位：**append-only 流水** sidecar，不参与层合并、不进检索、不进 5 槽位/
画像/主题层、不进默认 ``to_dict``（返回面零变化）。记录两类改写的前值：

- ``kind="update"``：``update_memory`` 改 content/importance（state 单独
  变更不记——supersede/归档路径的旧记录本身保留，状态机已承担留痕）；
- ``kind="merge"``：dedup 合并被吞一方的文本（含"新长旧短"与"旧长新短"
  两态；内容完全相同不记——没有任何东西丢失）。

生物学对应（再巩固）：改写后旧痕迹"不可访问但可追溯"——账本就是"可追溯"；
"可访问旧版"由 supersede 状态机承担，不在本账本职责内。

为什么放 store/ 而不是 brain：纯持久化设施（JSONL 读写 + 环裁剪），零领域
逻辑；与 tag_ops/save_ledger 同构（仓库第三套账本，形态收敛）。本模块
**不 import 任何 Config/常量旋钮**，只认显式参数（ring/trim_at 由调用方
传入）；唯一 import 的常量是文件名（全仓无 monkeypatch 先例，同
TAG_OPS_FILE 例外）。

状态语义（与 tag_ops 逐字一致）：
- ``last_seq`` 先自增再落盘 ⇒ 写失败也不复用 id；
- ``lines`` 落盘成功才 +1；裁剪失败（OSError）时 lines 保持原值；
- 账本写失败**不能吞**（logger.warning）：一期虽无 undo，"改之前是什么"
  从此无处可查这件事本身必须让运维看见；
- ⚠️ ``ring=0`` 时切片是**全量**（Python ``[-0:]`` 语义）——与 tag_ops
  同款怪癖，刻意保留，调用方（config 默认 200）不会传 0。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..common.constants import MEM_HISTORY_FILE

logger = logging.getLogger(__name__)


def history_path(persist_dir: str | None) -> str | None:
    """账本文件路径；无持久化目录（纯内存 brain）返回 None。"""
    return (os.path.join(persist_dir, MEM_HISTORY_FILE)
            if persist_dir else None)


def read_history(persist_dir: str | None) -> list[dict[str, Any]]:
    """读全部账本条目（坏行跳过——账本不该让审计用不了）。"""
    path = history_path(persist_dir)
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
                    continue  # 半行（崩溃在 append 中间）直接跳过
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def scan_history(persist_dir: str | None) -> tuple[int, int]:
    """启动/首写前扫一遍账本：返回 (行数, 末条 seq)。"""
    entries = read_history(persist_dir)
    last = 0
    for rec in entries:
        try:
            last = max(last, int(rec.get("seq") or 0))
        except (TypeError, ValueError):
            continue
    return len(entries), last


def append_history(persist_dir: str | None,
                   entry: dict[str, Any],
                   last_seq: int,
                   lines: int,
                   ring: int,
                   trim_at: int) -> tuple[str | None, int, int]:
    """追加一条账本（ring buffer 裁剪）。

    Returns:
        ``(op_id, last_seq, lines)``——状态三元组，调用方据此更新自身。
        op_id 为 None = 没落盘（无目录或写失败）；写失败时 last_seq
        **也已自增**（防 id 复用，同 tag_ops）。
    """
    path = history_path(persist_dir)
    if not path:
        return None, last_seq, lines
    last_seq += 1
    entry["seq"] = last_seq
    entry["op_id"] = f"memhist_{last_seq}"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        lines += 1
        if lines > trim_at:
            lines = trim_history(persist_dir, ring, lines)
    except OSError as exc:
        logger.warning("[mem_history] ledger append failed: %s", exc)
        return None, last_seq, lines
    return entry["op_id"], last_seq, lines


def trim_history(persist_dir: str | None, ring: int, lines: int) -> int:
    """裁剪到 ring 大小（保留最新的）。返回裁剪后的行数。

    行数不超 ring、无目录、裁剪失败（OSError）时原样返回 ``lines``。
    ``ring=0`` 的切片怪癖见模块 docstring（与 tag_ops 同款，刻意保留）。
    """
    path = history_path(persist_dir)
    if not path:
        return lines
    entries = read_history(persist_dir)
    if len(entries) <= ring:
        return lines
    keep = entries[-ring:]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in keep:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return len(keep)
    except OSError as exc:
        logger.warning("[mem_history] ledger trim failed: %s", exc)
        return lines
