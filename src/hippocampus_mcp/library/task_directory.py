# -*- coding: utf-8 -*-
"""任务目录 sidecar 的读写：``<user 库>/directory.json``。

三条形态约束：

1. **sidecar 不是记忆**：与 ``memories.json`` 同级的普通 JSON 文件
   （同信息地图层 ``maps.json`` 侧车的形态）。**不进召回池、不做向量记忆**
   ——目录要的是精确命中，语义检索漏检＝寻址失败。
2. **受管结构**：写路径只有受管通道（``library_create`` 建库即登记）。
   本模块提供 ``upsert_entry`` 这一个写原语，**不开放普通写工具**：
   目录若被任意 agent 裸写，改掉一个任务地址就会污染此后所有跨库寻址。
   读路径任意——它本来就该被读。
3. **坏文件不静默**：JSON 坏/结构坏一律返回 issue 码交给调用方自曝，**绝不
   当空目录继续**（空目录＝「这个任务没登记过」，与「文件坏了」是两回事）。
   写路径遇到坏文件则**拒绝覆盖**（抛错），避免把用户手写的条目冲掉。

原子写：临时文件 + ``os.replace``（同目录，跨盘符也不会失败）。进程内用
模块级锁串行化读改写；**跨进程**（多个 stdio 服务同指一个 user 库）理论上
仍可能丢一次更新——代价是「新库漏登记」（可见、可手工补），故不引文件锁。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from . import normpath
from ..domain.task_directory import (DIRECTORY_SCHEMA, normalize_entries,
                                     normalize_entry)

__all__ = [
    "DIRECTORY_NAME",
    "TaskDirectoryError",
    "directory_path",
    "load_directory",
    "save_directory",
    "upsert_entry",
]

#: 与 ``memories.json`` / ``library.json`` 同级的侧车文件名。
DIRECTORY_NAME = "directory.json"

#: 读改写串行化（进程内）。只保护写路径——读不持锁，坏文件由调用方兜。
_WRITE_LOCK = threading.Lock()


class TaskDirectoryError(ValueError):
    """任务目录写失败（坏文件 / 条目非法）。**不吞**：调用方要出声。"""


def directory_path(library_path: str) -> str:
    """``<user 库>/directory.json`` 的绝对路径。"""
    return os.path.join(normpath(library_path), DIRECTORY_NAME)


def load_directory(library_path: str) -> tuple[dict[str, Any], str | None]:
    """读 sidecar，返回 ``(data, issue)``。

    ``data`` 恒为 ``{"schema": int, "entries": [...]}``（条目已归一）；
    ``issue`` 为 ``None`` 表示读到了（含**文件不存在**——新装即空目录，
    不是错误），否则是自曝码 ``"task_directory_read_failed"``。

    统计口径：``entries`` 只含**归一后有效**的条目，坏条目在这里被丢弃
    （缺 ``task``/``path`` 的没法寻址）。丢弃条数由 ``data["dropped"]`` 自曝。
    """
    path = directory_path(library_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0}, None)
    except (OSError, ValueError):
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0},
                "task_directory_read_failed")
    if not isinstance(raw, dict):
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0},
                "task_directory_read_failed")
    raw_entries = raw.get("entries")
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list):
        return ({"schema": DIRECTORY_SCHEMA, "entries": [], "dropped": 0},
                "task_directory_read_failed")
    entries = normalize_entries(raw_entries)
    try:
        schema = int(raw.get("schema") or DIRECTORY_SCHEMA)
    except (TypeError, ValueError):
        schema = DIRECTORY_SCHEMA
    return ({"schema": schema, "entries": entries,
             "dropped": len(raw_entries) - len(entries)}, None)


def save_directory(library_path: str, data: dict[str, Any]) -> str:
    """原子写 sidecar（临时文件 + ``os.replace``）。返回写到的路径。"""
    path = directory_path(library_path)
    payload = {
        "schema": int(data.get("schema") or DIRECTORY_SCHEMA),
        "entries": normalize_entries(data.get("entries") or []),
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError:
        # 临时文件残留会让下次判断「目录里多了个怪文件」，清掉再抛。
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def upsert_entry(library_path: str, entry: dict[str, Any],
                 *, now: float | None = None) -> dict[str, Any]:
    """登记/更新一条 ``任务 → 库`` 映射（**唯一的写原语**）。

    去重键是**归一后的路径**（``normpath``：盘符大小写/相对路径都算同一个），
    不是 ``library_id``——id 只作人读与将来去重，寻址主键是路径。
    同一任务名换了库（重建/迁址后重新登记）时，旧条目原地改成新路径，
    避免一个任务名指向两个库这种歧义（「不猜」的反面是别制造歧义）。

    Returns:
        ``{"registered": True, "created": bool, "moved": bool, "count": int}``。

    Raises:
        TaskDirectoryError: sidecar 坏/不可读，或条目缺 ``task``/``path``。
            **不覆盖坏文件**——那可能是用户手写的目录，冲掉就找不回来了。
    """
    item = normalize_entry(entry)
    if item is None:
        raise TaskDirectoryError("任务目录条目缺 task 或 path，拒绝登记")
    item["updated_at"] = float(now if now is not None else time.time())
    with _WRITE_LOCK:
        data, issue = load_directory(library_path)
        if issue:
            raise TaskDirectoryError(
                f"{directory_path(library_path)} 不可读（{issue}）："
                "拒绝覆盖，请先修好或备份后删除该文件")
        entries: list[dict[str, Any]] = list(data.get("entries") or [])
        key = normpath(item["path"])
        created = moved = False
        idx = next((i for i, e in enumerate(entries)
                    if normpath(e["path"]) == key), None)
        if idx is not None:
            entries[idx] = item           # 同一个库重新登记 = 原地更新
        else:
            old = next((i for i, e in enumerate(entries)
                        if e["task"].casefold() == item["task"].casefold()), None)
            if old is not None:
                entries[old] = item       # 同名任务换库：原地搬过去
                moved = True
            else:
                entries.append(item)
                created = True
        save_directory(library_path, {"schema": data.get("schema"), "entries": entries})
    return {"registered": True, "created": created, "moved": moved,
            "count": len(entries)}
