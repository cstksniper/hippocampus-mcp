# -*- coding: utf-8 -*-
"""Session task registration: remember ``session_start(task=...)``'s task
description per connection.

**Why not a ContextVar**: ``common.session_ctx.bind_session`` is called
once by the bridge **at connection setup**, after which all tool calls on
that connection inherit that context. But ``session_start`` is just one
tool call, and the MCP low-level server may spawn a new task per request:
a value ``ContextVar.set()`` in request A is not guaranteed to appear in
request B's context (a child task copies the parent context's snapshot
**as of creation**). Betting on this plants a silent failure of the form
"sometimes origin.task is empty" -- exactly the shape this project has
repeatedly stepped on.

So an in-process dict keyed by session key (same family and same lock
granularity as ``server._conn_default_lib`` / ``_conn_workspace_binding``);
both reads and writes carry the key explicitly. The cost is in-process
state, empty after restart -- but the task description is inherently
session-scoped temporary context; **not persisting to disk, not crossing
processes** is a feature, not a defect.

**Two conventions**:

1. **Empty does not clear**: ``session_start(task="")`` means "no task
   declared this time", not "clear the task". Compaction recovery
   re-calls ``session_start`` keeping only receipt identity info per the
   docs, and that call very likely carries no task; if empty cleared, all
   later writes' ``origin.task`` would silently become empty. Clearing is
   done only explicitly by ``session_end``.
2. **No guessing**: only carry the task the host explicitly gave; do not
   infer a task name from body text / library names. The fallback source
   (the task directory's ``task -> library`` mapping) is synthesized by
   the server layer's ``_origin_task()``; this module does not touch it.

Dependency direction: L0, depending only on same-layer ``session_ctx`` /
``write_origin`` and the standard library.
"""

from __future__ import annotations

import threading

from .session_ctx import DEFAULT_SESSION_KEY, current_session
from .write_origin import MAX_LEN

__all__ = ["bind_task", "current_task", "clear_task", "reset_tasks"]

_tasks: dict[str, str] = {}
_tasks_lock = threading.RLock()

# Same source as write_origin.MAX_LEN["task"] -- origin is an index
# dimension, not body text; storing 200 more chars of task text per memory
# only bloats memories.json needlessly.
_MAX_TASK_CHARS = MAX_LEN["task"]


def _key(key: str = "") -> str:
    """An empty key always means the current connection (default usage), aligned with the write tools' ``session_key()``."""
    return (key or "").strip() or current_session() or DEFAULT_SESSION_KEY


def bind_task(task: str, key: str = "") -> None:
    """Register this connection's task description; **empty does not touch**
    (see module docstring convention 1)."""
    text = str(task or "").strip()[:_MAX_TASK_CHARS]
    if not text:
        return
    with _tasks_lock:
        _tasks[_key(key)] = text


def current_task(key: str = "") -> str:
    """This connection's currently registered task description; empty string when none."""
    with _tasks_lock:
        return _tasks.get(_key(key), "")


def clear_task(key: str = "") -> None:
    """Clear this connection's task registration (used by ``session_end`` at wrap-up)."""
    with _tasks_lock:
        _tasks.pop(_key(key), None)


def reset_tasks() -> None:
    """Clear all registrations (for test isolation; never called in production)."""
    with _tasks_lock:
        _tasks.clear()
