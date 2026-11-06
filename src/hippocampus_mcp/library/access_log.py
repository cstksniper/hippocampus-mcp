# -*- coding: utf-8 -*-
"""user 库写侧审计日志（D2a：access_log.jsonl 的追加与读取）。

形态承 U5 任务目录 sidecar 先例：住 user 库目录下、与 memories.json
同级的独立文件，**不是向量记忆、不进召回池**。写侧事件由写路径
（WriteService）追加——审计是旁路：**追加失败只 warning 不阻断写入**
（主路径是写记忆本身；审计挂了不能把用户的正常写也挂了）。

读侧（drift_audit 报告 / 将来 D2b 的读审计）只容忍坏行：跳过并计数，
绝不因一行坏 JSON 把整个日志判死——「坏≠空」教训在这里的反面适用：
读失败要有可见的计数，但不升级成拒绝服务。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..domain.write_quarantine import ACCESS_LOG_FILENAME

logger = logging.getLogger("hippocampus_mcp")

__all__ = ["access_log_path", "append_access_event", "read_access_events"]


def access_log_path(lib_path: str) -> str:
    """user 库目录 → access_log.jsonl 的绝对路径。"""
    return os.path.join(lib_path or "", ACCESS_LOG_FILENAME)


def append_access_event(lib_path: str, event: dict[str, Any]) -> bool:
    """追加一条审计事件。**成功与否如实返回**，绝不抛到调用方。"""
    if not lib_path:
        return False
    path = access_log_path(lib_path)
    try:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except OSError as e:
        logger.warning("access_log append failed (writes are not blocked): %s: %s",
                       path, e)
        return False


def read_access_events(lib_path: str, limit: int = 50) -> dict[str, Any]:
    """读最近 ``limit`` 条审计事件（时间升序的**尾部**窗口）。

    坏行跳过并计数（``skipped``），不整体判死；文件不存在 =
    空日志（``count=0``，不是错误——还没发生过非家写入）。
    """
    path = access_log_path(lib_path)
    out: dict[str, Any] = {"path": path, "count": 0, "skipped": 0,
                           "events": []}
    if limit <= 0 or not os.path.isfile(path):
        return out
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        logger.warning("access_log read failed: %s: %s", path, e)
        out["skipped"] = -1  # -1 = 整文件读不出来（与坏行的逐行跳过区分）
        return out
    events: list[dict[str, Any]] = []
    skipped = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(ev, dict):
            events.append(ev)
    out["skipped"] = skipped
    tail = events[-limit:] if limit < len(events) else events
    out["events"] = tail
    out["count"] = len(tail)
    return out
