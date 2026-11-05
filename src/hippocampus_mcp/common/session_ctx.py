"""daemon 化之后的会话分路。

**为什么必须存在**：daemon 把「一个进程服务 N 个会话」变成现实，于是
``library_use()`` 的会话级粘性（``server.set_library_binding``）必须按
连接分路——否则会话 A 切到「甲库」，会话 B 发起的工具调用也会跟着落甲库，
**两边都不报错**（这正是记忆里的「静默失败」原型）。

**为什么用 contextvars 而不是全局变量**：daemon 每条连接跑一个线程，
线程里再跑 anyio task 组。contextvars 在同一 task 内透传、跨连接天然隔离；
全局变量做不到「这条连接一个 key」。

**为什么不是 stdio 的问题**：直连 stdio 下一进程只服务一个会话，
``current_session()`` 恒为 ``__default__``，行为与改造前完全一致。

**明确不做什么**：这里只管「当前上下文属于哪条连接」，不解析库、
不做绑定——那些在 ``server.py`` 里，本模块对 server 是零依赖。
"""

from __future__ import annotations

from contextvars import ContextVar

__all__ = [
    "DEFAULT_SESSION_KEY",
    "HELLO_PREFIX",
    "bind_session",
    "current_session",
    "session_scope",
]

#: launcher → daemon 的**带外握手行**前缀（修 daemon 复用串库）。
#:
#: 背景：daemon 进程的默认库是**进程级**常量（server 模块导入时按 env
#: ``HIPPOCAMPUS_PERSIST_DIR`` + cwd 引导文件算出）。两个工作区若复用同一个
#: daemon（key 相同 —— 而这正是"模型 N→1"省内存的目的），第二个工作区的
#: 写入会落到**先起那个 launcher 的库**里：不报错、不告警，就是串库。
#:
#: 于是让 launcher 在转发任何 MCP 流量之前先发一行纯文本声明自己的库，
#: daemon 按**连接**记住它。这一行不是 JSON-RPC，daemon 必须把它吃掉，
#: 不能留给 MCP 解析（否则 MCP 会报协议错误）。
#:
#: 放在本模块是因为 launcher 与 bridge 都能零成本引用它（本模块纯标准库、
#: 无重依赖），且它是**单一真相源**——两边各写一份字符串迟早会漂移。
HELLO_PREFIX = "HIPPO-HELLO "

#: 未分路时的会话名。与 ``server.DEFAULT_KEY`` 必须一致。
DEFAULT_SESSION_KEY = "__default__"

_current_session: ContextVar[str] = ContextVar(
    "hippocampus_session", default=DEFAULT_SESSION_KEY)


def bind_session(key: str) -> None:
    """把**当前上下文**绑到会话 key。

    由 daemon 的每条连接在拉起 JSON-RPC 会话之前调用。
    空 key 一律落到 :data:`DEFAULT_SESSION_KEY`——「没指定」不能退化成
    「绑到上一路的残留值」，那正是串会话的形态。
    """
    _current_session.set((key or "").strip() or DEFAULT_SESSION_KEY)


def current_session() -> str:
    """当前上下文的会话 key（无绑定则 :data:`DEFAULT_SESSION_KEY`）。"""
    return _current_session.get() or DEFAULT_SESSION_KEY


class session_scope:
    """给一段代码临时绑会话，退出后恢复（测试与桥接都会用）。

    用法::

        with session_scope("conn-7"):
            ...  # 这段里的工具调用都走 conn-7 的库绑定
    """

    def __init__(self, key: str) -> None:
        self.key = (key or "").strip() or DEFAULT_SESSION_KEY
        self._token = None

    def __enter__(self) -> "session_scope":
        self._token = _current_session.set(self.key)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _current_session.reset(self._token)
            self._token = None
