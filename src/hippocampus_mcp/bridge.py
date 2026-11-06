"""Bridge between the daemon's TCP connections and MCP sessions.

**This code fills the missing piece of the daemon.** The daemon previously only
had ``_default_handler`` (an echo pump: whatever it receives is sent back
verbatim), and grepping ``HIPPOCAMPUS_DAEMON`` in ``server.py`` hit nothing --
"a real MCP session is handled by the server side" was only a comment. This
module wires the two ends together.

**What this fills in is exactly the sentence written in the daemon's early
comment**:

> The daemon takes over the byte pump and has not yet attached FastMCP
> ``Server.run`` to the socket stream (an anyio memory stream adapter is needed)

So the approach: treat each TCP connection as one MCP session, use anyio memory
object streams as the adapter layer, pumping the socket byte stream on one end
and feeding the low-level ``Server.run`` on the other.

**Wire protocol**: MCP over stdio is "newline-delimited JSON-RPC messages",
moved directly onto TCP, one per line. So this side only needs to **frame by
lines** on the byte stream; no length prefix is needed.

**Why not FastMCP.run() directly**: its ``transport`` only accepts
``stdio`` / ``sse`` / ``streamable-http``; there is no "give me a pair of
streams". To use its stream you must drop down to ``self._mcp_server`` (the
low-level Server), which is exactly what ``run_stdio_async`` itself does.

**One process shares one model** lands here: ``server_obj`` is a module-level
singleton; no matter how many connections come in, it is the same object, the
same brain, the same torch weights.
"""

# Exemption registry: module snapshot debug surface; see the closed list in config.py's "root-level exemption list".

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from typing import Any

from .common.session_ctx import HELLO_PREFIX   # pure stdlib: does not break this module's lazy loading

__all__ = ["make_mcp_handler", "mcp_connection_handler"]

#: Byte cap for a single recv. JSON-RPC frames are usually <64KB, so 64KB is
#: enough and does not stall the soft wait.
INPUT_CHUNK = 65536
#: The bridge's debug switch (``HIPPOCAMPUS_BRIDGE_DEBUG=1``): prints each
#: frame's in/out direction to stderr. When diagnosing **hangs without
#: exceptions** like "the request came in, the response never comes back",
#: this is the only way to tell "stuck on read / stuck on write / stuck in a
#: tool" apart.
BRIDGE_DEBUG = os.environ.get("HIPPOCAMPUS_BRIDGE_DEBUG", "") == "1"

_server_obj: Any = None


def _get_server() -> Any:
    """Get the server singleton lazily.

    **Must be lazy**: the daemon needs a very fast startup for "availability >
    memory savings", while importing ``server`` pulls in torch. A top-level
    import would ruin that.
    """
    global _server_obj
    if _server_obj is None:
        from hippocampus_mcp import server as _mod  # noqa: PLC0415
        _server_obj = _mod.server
    return _server_obj


def _low_level(obj: Any) -> Any:
    """Get the low-level Server from a high-level server instance (returned
    unchanged when a low-level Server was passed in the first place).

    🔴 The attribute name changed in mcp 2.x:

    * 1.x ``FastMCP`` -> ``_mcp_server``;
    * 2.x ``MCPServer`` -> ``_lowlevel_server``.

    The 2.x high-level class only has ``run(transport=...)``; it has neither
    ``create_initialization_options`` nor does it accept a pair of streams; not
    stopping at the low level raises ``AttributeError`` and takes down the whole
    connection. Probe both names in order so both versions work.
    """
    for attr in ("_mcp_server", "_lowlevel_server"):
        inner = getattr(obj, attr, None)
        if inner is not None:
            return inner
    return obj


async def _serve_socket_session(server_obj: Any, sock: socket.socket,
                                key: str) -> None:
    """Set up one complete MCP session over a pair of anyio memory streams."""
    import anyio  # noqa: PLC0415
    from anyio import BrokenResourceError, ClosedResourceError, EndOfStream  # noqa: PLC0415
    from mcp.shared.message import SessionMessage  # noqa: PLC0415
    import mcp.types as types  # noqa: PLC0415
    from pydantic import TypeAdapter  # noqa: PLC0415

    from .common import session_ctx  # noqa: PLC0415

    # 🔴 mcp 1.x's ``JSONRPCMessage`` is a pydantic RootModel (it has
    # ``model_validate_json``); mcp 2.x turned it into a PEP 604 union
    # (``types.UnionType``), which **does not** have that method. The old way
    # raises AttributeError under 2.x, and the "bad frame" branch below then
    # silently drops it as illegal JSON => the client receives not a single
    # response frame and the TaskGroup crashes and closes the connection (CI's
    # ``f9_4b`` ``EOFError: the daemon closed the connection`` was exactly this).
    # ``TypeAdapter`` parses both RootModel and union, and both results are
    # accepted by ``SessionMessage`` (verified on 1.29 / 2.3).
    jsonrpc_type = types.JSONRPCMessage
    parse_jsonrpc = (jsonrpc_type.model_validate_json
                     if hasattr(jsonrpc_type, "model_validate_json")
                     else TypeAdapter(jsonrpc_type).validate_json)

    # This one connection is one session: once the context key is bound,
    # every tool call in this thread and task automatically lands on that key.
    session_ctx.bind_session(key)

    low = _low_level(server_obj)
    loop = asyncio.get_running_loop()
    sock.setblocking(False)

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    async def sock_reader() -> None:
        """Socket byte stream -> line framing -> SessionMessage -> read stream."""
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
                        # A bad frame must not take down the whole connection: skip
                        # this one and wait for the next. But shout it out -- silent
                        # frame drops and silent cross-library use are the same class
                        # of incident.
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
        """Write stream -> SessionMessage -> JSON -> socket."""
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
    """Consume the launcher's out-of-band handshake line (returns the parsed
    result; None if absent).

    🔴 Three disciplines that must not be broken:

    1. **``MSG_PEEK`` first, then decide whether to read**. This channel must
       stay compatible with "clients connecting straight to the daemon" (tests,
       manual debugging): they send no handshake line, and a hard read would eat
       MCP's first frame, showing up as a protocol error.
    2. **Read byte by byte up to ``\\n``**. On loopback, the launcher's handshake
       line and the following ``initialize`` may well be coalesced into one TCP
       segment; a single ``recv(4096)`` would swallow the client's first frame
       too, and there is no mechanism here to "give back the over-read bytes".
    3. **Restore the timeout when done**. Not restoring after ``settimeout(t)``
       leaves a residual timeout that makes later ``recv`` calls raise
       ``socket.timeout``, which then gets swallowed by
       ``except Exception: pass`` => the return pump dies silently (stepped on
       once in real testing; do not step on it a second time).
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
        except Exception:  # noqa: BLE001 -- timeout/peer closed both count as "no handshake line"
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
            sock.settimeout(old)   # restore the original blocking mode
        except Exception:  # noqa: BLE001
            pass
    try:
        obj = json.loads(buf.decode("utf-8", "replace")[len(prefix):].strip())
    except Exception:  # noqa: BLE001 -- a broken handshake line must not keep the connection from starting
        return None
    return obj if isinstance(obj, dict) else None


def mcp_connection_handler(conn: Any) -> None:
    """The daemon's handler shape: a synchronous function that runs the whole
    lifecycle of one connection.

    One ``asyncio.run`` (independent event loop) per conn + one thread per
    connection inside the daemon, so N sessions are isolated from each other;
    and ``_get_server()`` is a process-level singleton, so there is always
    exactly one model and one brain.
    """
    key = getattr(conn, "session_key", None)
    key = key() if callable(key) else (key or "")
    sock = conn.sock
    # Out-of-band handshake: which library this connection belongs to, declared
    # by the launcher. If unavailable, use the process-level default library --
    # bit-for-bit identical to before the rework; never block the connection
    # from starting.
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
        except Exception as exc:  # noqa: BLE001 -- a failed handshake must not keep the session from starting
            print(f"[bridge] session {key} HELLO handling failed (falling back to process default library): "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        asyncio.run(_serve_socket_session(_get_server(), sock, key))
    except RuntimeError as exc:
        # Typical: the host tore down the socket while the session was still winding down
        print(f"[bridge] session {key} ended: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def make_mcp_handler() -> Any:
    """Factory for ``Daemon(registry, handler=...)``.

    Staying a plain function means the daemon side does not have to load this
    module's heavy imports ahead of time.
    """
    return mcp_connection_handler


def _self_check_imports() -> bool:
    """Whether the bridge is usable depends on anyio / mcp being present. When
    missing, **must** report the error explicitly -- never let the daemon
    silently degrade into an echo pump, which would leave the host connected yet
    forever without a response.
    """
    try:
        import anyio  # noqa: F401
        from mcp.shared.message import SessionMessage  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def bridge_available() -> bool:
    """True if the bridge verifies as usable. The daemon decides on this basis
    between "start real MCP" and "exit with an explicit error"."""
    return _self_check_imports()


#: socket is only used for type annotations; keep the reference so lint does not flag it as unused
_ = socket
