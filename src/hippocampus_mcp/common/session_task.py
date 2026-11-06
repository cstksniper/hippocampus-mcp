# -*- coding: utf-8 -*-
"""会话任务登记：把 ``session_start(task=...)`` 的任务描述按连接记住。

**为什么不是 ContextVar**：``common.session_ctx.bind_session`` 由 bridge 在
**连接建立时**调一次，之后该连接的所有工具调用继承那个 context。但
``session_start`` 只是一次工具调用，MCP 低层 server 可能为每个请求起新
task：在请求 A 里 ``ContextVar.set()`` 的值不保证出现在请求 B 的 context
里（子 task 复制的是**创建时**父 context 的快照）。赌这个等于埋一个
「有时候 origin.task 是空的」的静默失败——正是本项目反复踩到的形态。

于是按 session key 存进程内字典（与 ``server._conn_default_lib`` /
``_conn_workspace_binding`` 同一族、同一把锁的粒度），读写都显式带 key。
代价是进程内状态、重启即空——但任务描述本来就是会话级临时上下文，
**不落盘、不跨进程**是特性不是缺陷。

**两条口径**：

1. **空值不清**：``session_start(task="")`` 表示「这次没声明任务」，不是
   「清除任务」。compaction 恢复按文档只保留 receipt 身份信息再调一次
   ``session_start``，那次很可能不带 task；若空值清掉，之后所有写入的
   ``origin.task`` 会静默变空。清除只由 ``session_end`` 显式做。
2. **不猜**：只搬运宿主显式给的 task，不从正文 / 库名推任务名。兜底来源
   （任务目录的「任务 → 库」映射）由 server 层的 ``_origin_task()`` 合成，
   本模块不碰。

依赖方向：L0，只依赖同层 ``session_ctx`` / ``write_origin`` 与标准库。
"""

from __future__ import annotations

import threading

from .session_ctx import DEFAULT_SESSION_KEY, current_session
from .write_origin import MAX_LEN

__all__ = ["bind_task", "current_task", "clear_task", "reset_tasks"]

_tasks: dict[str, str] = {}
_tasks_lock = threading.RLock()

# 与 write_origin.MAX_LEN["task"] 同源——origin 是索引维度不是正文，
# 一条记忆多存 200 字任务全文只是让 memories.json 无谓膨胀。
_MAX_TASK_CHARS = MAX_LEN["task"]


def _key(key: str = "") -> str:
    """空 key 一律取当前连接（默认用法），与写工具的 ``session_key()`` 对齐。"""
    return (key or "").strip() or current_session() or DEFAULT_SESSION_KEY


def bind_task(task: str, key: str = "") -> None:
    """登记本连接的任务描述；**空值不动**（见模块 docstring 口径 1）。"""
    text = str(task or "").strip()[:_MAX_TASK_CHARS]
    if not text:
        return
    with _tasks_lock:
        _tasks[_key(key)] = text


def current_task(key: str = "") -> str:
    """本连接当前登记的任务描述；没登记返回空串。"""
    with _tasks_lock:
        return _tasks.get(_key(key), "")


def clear_task(key: str = "") -> None:
    """清除本连接的任务登记（``session_end`` 收尾用）。"""
    with _tasks_lock:
        _tasks.pop(_key(key), None)


def reset_tasks() -> None:
    """清空全部登记（测试隔离用；生产不调）。"""
    with _tasks_lock:
        _tasks.clear()
