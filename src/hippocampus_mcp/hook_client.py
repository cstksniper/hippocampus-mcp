# -*- coding: utf-8 -*-
"""hook_client：宿主 hook 进程用的**只读**轻客户端（纯标准库）。

背景见 ``docs/design/UserPromptSubmit自动注入_20261105.md``：把读侧最常用的
「现在该想起什么」从 agent 主动调用改为宿主 hook 确定性注入（机制替代纪律，
10-31 审计证明"靠 agent 自觉 recall"会漏）。

三条设计约束（改动前先读设计文档 §三/§四）：

- **纯标准库**。hook 进程每次 prompt 都要跑，唯一允许的包内依赖是
  :mod:`hippocampus_mcp.launcher`（它也是纯标准库；``hippocampus_mcp/__init__``
  只设 env 不拖重库，已核）。重库一进来，每条 prompt 都多背几百 MB。
- **只读 best-effort**。daemon 不在就静默放弃，**绝不从这里拉起 daemon**
  （模型冷载 13-35s 必然顶穿 hook 超时；daemon 的拉起是 MCP 会话 launcher
  的职责，有单飞锁）。任何异常都返回空串，hook 故障绝不能卡会话。
- **每条 prompt 一条新连接**：连接 → HELLO → MCP 握手 → 一次 tools/call → 关。
  daemon 每连接一线程，成本毫秒级；不复用长连接是为了让 hook 进程无常驻状态。

与 MCP 会话的库绑定同源：launcher 怎么发 HELLO（``_send_hello``），这里就怎么发
——``library`` 传声明库、``cwd`` 传工作区，daemon 侧按 C1 反查定家库。这样 hook
的装配与同工作区的 MCP 会话落在同一个库，且敏感档判定口径一致。
"""

from __future__ import annotations

import json
import os
import socket
import time

from .launcher import HELLO_PREFIX, daemon_key, read_registry, try_connect

# hook 进程 env 里没有 HIPPOCAMPUS_*（那是 MCP server 的 per-server env），
# 找 daemon 注册表需要的值由调用方传入；这里的默认值只覆盖"与本仓默认
# 安装一致"的场景（cli context / 测试）。ZCode 用户级 hook 薄壳脚本里
# 有自己的镜像常量，改配置时两处要一起看（设计文档 §八 漂移风险）。
DEFAULT_KEY_LIBRARY = ""                      # 与 launcher：HIPPOCAMPUS_DAEMON_LIBRARY 未设同值
DEFAULT_EMB_MODEL = "BAAI/bge-base-zh-v1.5"   # 与 cli.MODEL_DEFAULT、用户级条目 env 同值

#: 整条交换（连接 → 拿到响应）的墙钟预算。daemon 热态单查几十毫秒；
#: daemon 正在冷启动时 warmup 会阻塞 tools/call，这里到点就放弃——
#: 宁可这一轮不注入，也不让 hook 挂住 prompt。
EXCHANGE_DEADLINE_S = 6.0

#: query 截断。嵌入模型 512 token 硬上限（写侧撞过的坑），query 同样受限。
QUERY_MAX_CHARS = 512

# get_memory_context 返回的 5 槽位键（domain/context_slots.py 装配）。
# hook 只取三段：命中 + 待办 + 未完结话题——时间线/实体槽是开场包的材料，
# 每轮重复注入是浪费（设计文档 D1；"待办解决上次没收口"含 block_2）。
BLOCK_HITS = "block_1_current_hits"
BLOCK_PENDING = "block_2_pending"
BLOCK_TOPICS = "block_5_open_topics"

# 装配侧的占位行：整段只有占位行 = 该段为空，不进注入。
_PLACEHOLDERS = {"(No todos)", "(No session history)", "(No related entity state)", "(No open topics)"}

_INJECTION_HEADER = "[HippoMCP auto-injected memory related to the current input (fetched by hook; for reference only)]:"
_SECTION_HEADERS = ((BLOCK_HITS, "◆ Relevant hits"), (BLOCK_PENDING, "◆ Open todos"),
                    (BLOCK_TOPICS, "◆ Unfinished topics"))


def format_injection(blocks: dict, *, item_max_chars: int = 150,
                     token_budget: int = 600) -> str:
    """把装配块渲染成注入文本。空结果返回 ``""``（调用方据此零输出）。

    独立成纯函数是为了单测可以直接盯：占位段跳过、逐行截断、预算逐行硬检查。
    预算是**软首行**：第一条内容行即使超预算也保留（否则会出现"只有标题
    没有内容"的注入），其后一律硬截并打标记。
    """
    lines: list[str] = [_INJECTION_HEADER]
    used = len(lines[0])
    truncated = False
    for key, header in _SECTION_HEADERS:
        raw = [str(x) for x in (blocks.get(key) or [])]
        body = [x for x in raw if x and x not in _PLACEHOLDERS]
        if not body:
            continue
        if len(lines) > 1 and used + len(header) + 1 > token_budget:
            truncated = True
            break
        lines.append(header)
        used += len(header) + 1
        for x in body:
            line = "- " + (x[:item_max_chars] if item_max_chars > 0 else x)
            if len(lines) > 2 and used + len(line) + 1 > token_budget:
                truncated = True
                break
            lines.append(line)
            used += len(line) + 1
        if truncated:
            break
    if len(lines) == 1:
        return ""
    if truncated:
        lines.append("(Truncated to budget)")
    return "\n".join(lines)


class _LineReader:
    """行分帧读取（带墙钟预算）。

    🔴 逐次 ``recv`` 自攒缓冲，别用 ``makefile``：socket 超时要按剩余预算
    动态收紧，文件对象会把它藏进内部缓冲里失真。
    """

    def __init__(self, sock: socket.socket, deadline: float) -> None:
        self._sock = sock
        self._deadline = deadline
        self._buf = b""

    def readline(self) -> str:
        while b"\n" not in self._buf:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("hook_client exchange exceeded budget")
            self._sock.settimeout(remaining)
            data = self._sock.recv(65536)
            if not data:
                raise ConnectionError("daemon closed the connection early")
            self._buf += data
        raw, self._buf = self._buf.split(b"\n", 1)
        return raw.strip().decode("utf-8", "replace")


def _send_json(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def _read_response(reader: _LineReader, want_id: int) -> dict:
    """读到指定 id 的响应帧为止（服务器主动通知一律跳过）。"""
    while True:
        obj = json.loads(reader.readline())
        if isinstance(obj, dict) and obj.get("id") == want_id:
            return obj


def _exchange(sock: socket.socket, *, query: str, top_k: int, token_budget: int,
              item_max_chars: int, hello_library: str, cwd: str,
              deadline: float) -> dict:
    """HELLO → initialize → initialized → tools/call，返回装配 dict。

    握手三步是 F8（launcher.py:984-986 注释）推断 + 实测确认的：daemon 内部
    是标准 MCP low-level server，未 initialize 的 tools/call 会被拒。
    """
    # HELLO 必须先于一切 MCP 字节（launcher._send_hello 同款纪律：不发会串库）
    payload = {"library": hello_library, "cwd": cwd, "pid": os.getpid()}
    sock.sendall((HELLO_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
                 .encode("utf-8"))

    reader = _LineReader(sock, deadline)
    _send_json(sock, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "hippocampus-hook",
                                                "version": "1"}}})
    _read_response(reader, 1)
    _send_json(sock, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    _send_json(sock, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": "get_memory_context",
                                 "arguments": {"query": query[:QUERY_MAX_CHARS],
                                               "top_k": top_k,
                                               "token_budget": token_budget,
                                               "item_max_chars": item_max_chars,
                                               "scope": "project"}}})
    resp = _read_response(reader, 2)
    if resp.get("error"):
        raise RuntimeError(f"tools/call was rejected: {resp['error']}")
    result = resp.get("result") or {}
    # structured_output=False：结果在 content[0].text 里，是一层 JSON 字符串
    content = (result.get("content") or [{}])[0]
    inner = json.loads(content.get("text") or "{}")
    if not isinstance(inner, dict):
        raise RuntimeError("get_memory_context returned an unexpected structure")
    return inner


def fetch_context(prompt: str, cwd: str, *, top_k: int = 3,
                  token_budget: int = 600, item_max_chars: int = 150,
                  key_library: str = DEFAULT_KEY_LIBRARY,
                  emb_model: str = DEFAULT_EMB_MODEL,
                  daemon_version: str = "",
                  hello_library: str = "",
                  deadline_s: float = EXCHANGE_DEADLINE_S,
                  trace: list | None = None) -> str:
    """按 prompt 查记忆，返回注入文本；**任何失败都返回 ``""``**。

    Args:
        prompt: 用户提示词原文（内部截到 :data:`QUERY_MAX_CHARS`）。
        cwd: 工作区根目录——daemon 按 C1 反查定家库的依据，必须与
            同工作区 MCP 会话的 cwd 同源（hook 里取 ``ZCODE_PROJECT_DIR``
            或进程 cwd）。
        key_library / emb_model / daemon_version: daemon 注册表 key 三要素，
            必须与目标 daemon 被拉起时的 env 一致（见模块 docstring 常量说明）。
        hello_library: HELLO 声明库（launcher 的 ``HIPPOCAMPUS_PERSIST_DIR``
            同款；用户级形态传用户库路径，空则声明为无）。
        trace: 传 list 则逐步追加人读诊断（cli context 用；hook 传 None 零开销）。
    """
    def _note(msg: str) -> None:
        if trace is not None:
            trace.append(msg)

    try:
        if not prompt or not prompt.strip():
            _note("prompt is empty")
            return ""
        key = daemon_key(key_library, emb_model, daemon_version)
        info = read_registry(key)
        if not info:
            _note(f"registry entry not found (key={key}); daemon is not running, hook will not start it")
            return ""
        deadline = time.monotonic() + deadline_s
        sock = try_connect(int(info["port"]), timeout=2.0)
        if sock is None:
            _note(f"daemon port is unreachable (port={info.get('port')})")
            return ""
        try:
            inner = _exchange(sock, query=prompt, top_k=top_k,
                              token_budget=token_budget,
                              item_max_chars=item_max_chars,
                              hello_library=hello_library, cwd=cwd,
                              deadline=deadline)
        finally:
            try:
                sock.close()
            except OSError:
                pass
        text = format_injection(inner.get("blocks") or {},
                                item_max_chars=item_max_chars,
                                token_budget=token_budget)
        _note("context assembled" if text else "context is empty (no hits and no pending items)")
        return text
    except Exception as exc:  # noqa: BLE001 —— fail-open：hook 故障绝不卡会话
        _note(f"failed ({type(exc).__name__}: {exc})")
        return ""
