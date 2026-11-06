"""daemon 的 TCP 连接 ↔ MCP 会话桥。

**这段代码补的是 daemon 缺的那一环。** daemon 之前只有 ``_default_handler``
（一个 echo 泵：收到什么原样发回去），而 ``server.py`` 里 grep
``HIPPOCAMPUS_DAEMON`` 零命中——「真正的 MCP 会话由 server 侧接手」只是一句
注释。本模块把这两头接上。

**这段要补的，正是 daemon 早期注释里写明的那句话**：

> daemon 接手的是字节泵，还没把 FastMCP ``Server.run`` 挂到 socket 流上
> （需一层 anyio 内存流适配器）

所以做法就是：把每条 TCP 连接当成一个 MCP 会话，用 anyio 内存对象流当适配层，
一头泵 socket 字节流，一头喂给低层 ``Server.run``。

**线协议**：MCP over stdio 是「换行分隔的 JSON-RPC 消息」，直接搬到 TCP 上，
一行一条。所以本侧只需对字节流做**行帧**，不必实现长度前缀。

**为什么不直接 FastMCP.run()**：它的 ``transport`` 只有
``stdio`` / ``sse`` / ``streamable-http``，没有「给我一对流」。要用它的 stream
就得下到 ``self._mcp_server``（低层 Server），那正是 ``run_stdio_async``
自己用的写法。

**一个进程共享一份模型**就落在这里：``server_obj`` 是模块级单例，
多少个连接进来都是同一个对象、同一个 brain、同一份 torch 权重。
"""

# 豁免登记：模块快照调试面，封闭清单见 config.py「根层豁免清单」。

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from typing import Any

from .common.session_ctx import HELLO_PREFIX   # 纯标准库：不破坏本模块的惰性加载

__all__ = ["make_mcp_handler", "mcp_connection_handler"]

#: 单次 recv 的字节上限。JSON-RPC 帧通常 <64KB，取 64KB 足够且不会卡住软 wait。
INPUT_CHUNK = 65536
#: 桥的调试开关（``HIPPOCAMPUS_BRIDGE_DEBUG=1``）：把每一帧的进出方向打在
#: stderr 上。排查「请求进来了、响应永远不回来」这类**无异常挂起**时，
#: 这是唯一能分清"卡在读 / 卡在写 / 卡在工具里"的手段。
BRIDGE_DEBUG = os.environ.get("HIPPOCAMPUS_BRIDGE_DEBUG", "") == "1"

_server_obj: Any = None


def _get_server() -> Any:
    """惰性取 server 单例。

    **必须惰性**：daemon 要为「可用性 > 省内存」保持启动极快，而 import
    ``server`` 会拉起<｜hy_place▁holder▁no▁813｜> torch。顶层 import 会毁掉这一条。
    """
    global _server_obj
    if _server_obj is None:
        from hippocampus_mcp import server as _mod  # noqa: PLC0415
        _server_obj = _mod.server
    return _server_obj


def _low_level(obj: Any) -> Any:
    """从高层 server 实例取低层 Server（本来就传低层 Server 时原样返回）。

    🔴 属性名在 mcp 2.x 换了：

    * 1.x ``FastMCP`` → ``_mcp_server``；
    * 2.x ``MCPServer`` → ``_lowlevel_server``。

    2.x 的高层类只有 ``run(transport=...)``，既没有
    ``create_initialization_options`` 也不接受一对流；不停到低层就会抛
    ``AttributeError`` 让整条连接断掉。按顺序探测两个名字，两个版本都能用。
    """
    for attr in ("_mcp_server", "_lowlevel_server"):
        inner = getattr(obj, attr, None)
        if inner is not None:
            return inner
    return obj


async def _serve_socket_session(server_obj: Any, sock: socket.socket,
                                key: str) -> None:
    """在一对 anyio 内存流上架起一次完整 MCP 会话。"""
    import anyio  # noqa: PLC0415
    from anyio import BrokenResourceError, ClosedResourceError, EndOfStream  # noqa: PLC0415
    from mcp.shared.message import SessionMessage  # noqa: PLC0415
    import mcp.types as types  # noqa: PLC0415
    from pydantic import TypeAdapter  # noqa: PLC0415

    from .common import session_ctx  # noqa: PLC0415

    # 🔴 mcp 1.x 的 ``JSONRPCMessage`` 是 pydantic RootModel（有
    # ``model_validate_json``）；mcp 2.x 把它改成了 PEP 604 union
    # （``types.UnionType``），**没有**这个方法。旧写法在 2.x 下抛
    # AttributeError，被下面的「坏帧」分支当成非法 JSON 静默丢掉 ⇒
    # 客户端一帧响应都收不到，TaskGroup 随即崩掉关连接（CI 上
    # ``f9_4b`` 的 ``EOFError: daemon 关闭了连接`` 就是它）。
    # ``TypeAdapter`` 对 RootModel 与 union 都能解析，且两种结果都能被
    # ``SessionMessage`` 接受（1.29 / 2.3 实测）。
    jsonrpc_type = types.JSONRPCMessage
    parse_jsonrpc = (jsonrpc_type.model_validate_json
                     if hasattr(jsonrpc_type, "model_validate_json")
                     else TypeAdapter(jsonrpc_type).validate_json)

    # 这一条连接就是一个会话：上下文 key 一旦绑定，
    # 本线程本 task 内的所有工具调用都自动落在这个 key 上。
    session_ctx.bind_session(key)

    low = _low_level(server_obj)
    loop = asyncio.get_running_loop()
    sock.setblocking(False)

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    async def sock_reader() -> None:
        """socket 字节流 → 行帧 → SessionMessage → 读流。"""
        buf = b""
        try:
            while True:
                try:
                    data = await loop.sock_recv(sock, INPUT_CHUNK)
                except (ConnectionResetError, BrokenResourceError, OSError):
                    return
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if BRIDGE_DEBUG:
                        print(f"[bridge {key}] <- {len(line)}B",
                              file=sys.stderr, flush=True)
                    try:
                        msg = parse_jsonrpc(line)
                    except Exception as exc:  # noqa: BLE001
                        # 坏帧不能带走整条连接：跳过这一条，等下一帧。
                        # 但要吼出来——静默丢包和静默串库是一类事故。
                        print(
                            f"[bridge] discarded invalid JSON-RPC frame: {type(exc).__name__}: {exc}",
                            file=sys.stderr)
                        continue
                    try:
                        await read_stream_writer.send(SessionMessage(msg))
                    except (ClosedResourceError, BrokenResourceError):
                        return
        finally:
            try:
                read_stream_writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def sock_writer() -> None:
        """写流 → SessionMessage → JSON → socket。"""
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    payload = session_message.message.model_dump_json(
                        by_alias=True, exclude_none=True)
                    if BRIDGE_DEBUG:
                        print(f"[bridge {key}] -> {len(payload)}B",
                              file=sys.stderr, flush=True)
                    try:
                        await loop.sock_sendall(
                            sock, (payload + "\n").encode("utf-8"))
                    except (ConnectionResetError, BrokenResourceError, OSError):
                        return
        except (ClosedResourceError, EndOfStream):
            return

    async with anyio.create_task_group() as tg:
        tg.start_soon(sock_reader)
        tg.start_soon(sock_writer)
        await low.run(read_stream, write_stream,
                      low.create_initialization_options())


def _read_hello(sock: Any, timeout: float = 1.5) -> dict | None:
    """吃掉 launcher 的带外握手行（返回解析结果；没有就 None）。

    🔴 三条不能破的纪律：

    1. **先 ``MSG_PEEK`` 再决定读不读**。这条通道必须兼容「直接连 daemon 的
       客户端」（测试、手工排障）：它们不发握手行，硬读就会把 MCP 的第一帧
       吃掉，表现为协议错误。
    2. **逐字节读到 ``\\n`` 为止**。loopback 上 launcher 的握手行与随后的
       ``initialize`` 完全可能被合并进同一个 TCP 段；一次 ``recv(4096)`` 会
       连客户端的第一帧一起吃掉，而这里没有"把多读的还回去"的机制。
    3. **用完必须还原 timeout**。``settimeout(t)`` 之后不还原，残留的超时会
       让后续 ``recv`` 抛 ``socket.timeout``，而它被 ``except Exception: pass``
       吞掉 ⇒ 回程泵静默死亡（实测踩过一次，别踩第二次）。
    """
    prefix = HELLO_PREFIX.encode("utf-8")
    try:
        old = sock.gettimeout()
    except Exception:  # noqa: BLE001
        old = None
    try:
        try:
            sock.settimeout(timeout)
            head = sock.recv(len(prefix), socket.MSG_PEEK)
        except Exception:  # noqa: BLE001 —— 超时/对端已关都算"没有握手行"
            return None
        if not head or not head.startswith(prefix):
            return None
        buf = b""
        while not buf.endswith(b"\n") and len(buf) < 8192:
            try:
                ch = sock.recv(1)
            except Exception:  # noqa: BLE001
                break
            if not ch:
                break
            buf += ch
    finally:
        try:
            sock.settimeout(old)   # 还原成原来的阻塞模式
        except Exception:  # noqa: BLE001
            pass
    try:
        obj = json.loads(buf.decode("utf-8", "replace")[len(prefix):].strip())
    except Exception:  # noqa: BLE001 —— 握手行坏了不该让连接起不来
        return None
    return obj if isinstance(obj, dict) else None


def mcp_connection_handler(conn: Any) -> None:
    """daemon 的 handler 形态：同步函数，进去跑完一条连接的整个生命周期。

    每个 conn 一个 ``asyncio.run``（独立 event loop）+ daemon 每连接一个线程，
    于是 N 个会话彼此隔离；而 ``_get_server()`` 是进程级单例，
    模型与 brain 始终只有一份。
    """
    key = getattr(conn, "session_key", None)
    key = key() if callable(key) else (key or "")
    sock = conn.sock
    # 带外握手：这条连接属于哪个库，由 launcher 声明。
    # 拿不到就走进程级默认库 —— 与改造前逐位一致，绝不让连接起不来。
    hello = _read_hello(sock)
    if hello:
        try:
            from .server import set_connection_default_library  # noqa: PLC0415
            path = set_connection_default_library(
                str(hello.get("library") or ""),
                str(hello.get("cwd") or ""),
                session=key,
            )
            print(f"[bridge] session {key} default library={path} (from launcher HELLO)",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 —— 握手失败不能让会话起不来
            print(f"[bridge] session {key} HELLO handling failed (falling back to process default library): "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        asyncio.run(_serve_socket_session(_get_server(), sock, key))
    except RuntimeError as exc:
        # 典型：宿主把 socket 断了，而会话还在收尾
        print(f"[bridge] session {key} ended: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def make_mcp_handler() -> Any:
    """给 ``Daemon(registry, handler=...)`` 用的 factory。

    保持函数形态是为了让 daemon 侧不必在本模块之前加载 heavy import。
    """
    return mcp_connection_handler


def _self_check_imports() -> bool:
    """桥能不能用，取决于 anyio / mcp 在不在。缺失时**必须**明确报错，
    不能让 daemon 静默退化成 echo 泵——那会让宿主连上却永远收不到响应。
    """
    try:
        import anyio  # noqa: F401
        from mcp.shared.message import SessionMessage  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def bridge_available() -> bool:
    """桥验证可用 -> True。daemon 据此决定「起真 MCP」还是「明确报错退出」。"""
    return _self_check_imports()


#: socket 仅用于类型标注，保留引用以免 lint 判为未使用
_ = socket
