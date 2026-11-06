# -*- coding: utf-8 -*-
"""hook_client: a **read-only** lightweight client for host hook processes
(pure standard library).

Background: ``docs/design/UserPromptSubmit自动注入_20261105.md`` -- moving
the read side's most frequent "what should be recalled now" from agent
initiated calls to deterministic injection by a host hook (mechanism instead
of discipline; the 10-31 audit proved "relying on the agent to recall on its
own" leaks).

Three design constraints (read design doc §三/§四 before changing):

- **Pure standard library**. The hook process runs on every prompt; the only
  allowed in-package dependency is :mod:`hippocampus_mcp.launcher` (also
  pure standard library; ``hippocampus_mcp/__init__`` only sets env vars and
  drags in no heavy libraries -- verified). One heavy import and every
  prompt carries hundreds of MB.
- **Read-only best-effort**. If the daemon is absent, give up silently;
  **never start the daemon from here** (a 13-35s model cold load would blow
  straight through the hook timeout; starting the daemon is the MCP session
  launcher's job and it has a single-flight lock). Any exception returns an
  empty string; a hook failure must never stall the session.
- **One new connection per prompt**: connect → HELLO → MCP handshake → one
  tools/call → close. The daemon runs one thread per connection at
  millisecond cost; long connections are not reused so the hook process
  keeps no resident state.

Same origin as the MCP session's library binding: however launcher sends
HELLO (``_send_hello``), this sends it the same way -- ``library`` carries
the declared library, ``cwd`` the workspace; the daemon side resolves the
home library via C1 reverse lookup. That way the hook's assembly lands in
the same library as the MCP sessions of the same workspace, with an
identical sensitivity-tier criterion.
"""

from __future__ import annotations

import json
import os
import socket
import time

from .launcher import HELLO_PREFIX, daemon_key, read_registry, try_connect

# The hook process's env has no HIPPOCAMPUS_* (that is the MCP server's
# per-server env); the values needed to find the daemon registry are passed
# in by the caller. The defaults here only cover the "matches this repo's
# default install" scenarios (cli context / tests). The ZCode user-level
# hook thin-shell script carries its own mirror constants; when changing
# config, look at both places together (design doc §八 drift risk).
DEFAULT_KEY_LIBRARY = ""                      # matches launcher: HIPPOCAMPUS_DAEMON_LIBRARY unset
DEFAULT_EMB_MODEL = "BAAI/bge-base-zh-v1.5"   # matches cli.MODEL_DEFAULT and the user-level entry env

#: Wall-clock budget for the whole exchange (connect → response received).
#: A warm daemon answers a single query in tens of milliseconds; while the
#: daemon is cold-starting, warmup blocks tools/call -- give up when the
#: budget runs out. Better to skip this round's injection than let the hook
#: hang the prompt.
EXCHANGE_DEADLINE_S = 6.0

#: Query truncation. The embedding model has a 512 token hard cap (a pit hit
#: on the write side); the query is bound by the same limit.
QUERY_MAX_CHARS = 512

# The 5 slot keys returned by get_memory_context (assembled by
# domain/context_slots.py). The hook takes only three sections: hits +
# pending + open topics -- the timeline/entity slots are opening-package
# material; re-injecting them every round is waste (design doc D1; "pending
# left unclosed from last time" includes block_2).
BLOCK_HITS = "block_1_current_hits"
BLOCK_PENDING = "block_2_pending"
BLOCK_TOPICS = "block_5_open_topics"

# Placeholder lines from the assembly side: a section holding only
# placeholders = that section is empty, excluded from injection.
_PLACEHOLDERS = {"(No todos)", "(No session history)", "(No related entity state)", "(No open topics)"}

_INJECTION_HEADER = "[HippoMCP auto-injected memory related to the current input (fetched by hook; for reference only)]:"
_SECTION_HEADERS = ((BLOCK_HITS, "◆ Relevant hits"), (BLOCK_PENDING, "◆ Open todos"),
                    (BLOCK_TOPICS, "◆ Unfinished topics"))


def format_injection(blocks: dict, *, item_max_chars: int = 150,
                     token_budget: int = 600) -> str:
    """Render the assembled blocks into injection text. Empty results return
    ``""`` (the caller emits zero output accordingly).

    Kept as a pure function so unit tests can stare directly at it:
    placeholder sections skipped, per-line truncation, budget hard-checked
    line by line. The budget is a **soft first line**: the first content line
    is kept even if it exceeds the budget (otherwise you get an injection
    with "headers but no content"); everything after is hard-truncated and
    flagged.
    """
    sections = []
    for key, header in _SECTION_HEADERS:
        raw = [str(x) for x in (blocks.get(key) or [])]
        body = [x for x in raw if x and x not in _PLACEHOLDERS]
        if body:
            sections.append((key, header, body))
    stats = {k: {"kept": 0, "total": len(b), "dropped": 0}
             for k, _h, b in sections}
    lines: list[str] = [_INJECTION_HEADER]
    used = len(lines[0])
    truncated = False
    clipped = 0
    for key, header, body in sections:
        if len(lines) > 1 and used + len(header) + 1 > token_budget:
            truncated = True
            break
        lines.append(header)
        used += len(header) + 1
        for x in body:
            if item_max_chars > 0 and len(x) > item_max_chars:
                clipped += 1
            line = "- " + (x[:item_max_chars] if item_max_chars > 0 else x)
            if len(lines) > 2 and used + len(line) + 1 > token_budget:
                truncated = True
                break
            lines.append(line)
            used += len(line) + 1
            stats[key]["kept"] += 1
        if truncated:
            break
    for entry in stats.values():
        entry["dropped"] = entry["total"] - entry["kept"]
    if len(lines) == 1:
        return ""
    if clipped:
        lines.append("(%d item(s) clipped to %d chars; pass "
                     "item_max_chars=0 to keep full text)" % (clipped, item_max_chars))
    if truncated:
        parts = ["%s %d/%d kept" % (k, v["kept"], v["total"])
                 for k, v in stats.items()]
        lines.append("(Slot budget: " + ", ".join(parts) + ")")
        # Keep the legacy marker as the LAST line: existing callers and tests
        # detect truncation by endswith("(Truncated to budget)").
        lines.append("(Truncated to budget)")
    return "\n".join(lines)


class _LineReader:
    """Line-framed reading (with a wall-clock budget).

    🔴 Accumulate the buffer across individual ``recv`` calls; do not use
    ``makefile``: the socket timeout must tighten dynamically against the
    remaining budget, and a file object would hide it inside internal
    buffering, distorting it.
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
    """Read until the response frame with the given id (server-initiated
    notifications are always skipped)."""
    while True:
        obj = json.loads(reader.readline())
        if isinstance(obj, dict) and obj.get("id") == want_id:
            return obj


def _exchange(sock: socket.socket, *, query: str, top_k: int, token_budget: int,
              item_max_chars: int, hello_library: str, cwd: str,
              deadline: float) -> dict:
    """HELLO → initialize → initialized → tools/call; returns the assembled
    dict.

    The three handshake steps were inferred from F8 (comment at
    launcher.py:984-986) and confirmed by measurement: inside the daemon is
    a standard MCP low-level server; a tools/call without initialize is
    rejected.
    """
    # HELLO must precede any MCP bytes (same discipline as
    # launcher._send_hello: skipping it cross-contaminates libraries)
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
    # structured_output=False: the result lives in content[0].text as one
    # more JSON string
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
    """Query memory by prompt and return injection text; **any failure
    returns ``""``**.

    Args:
        prompt: the user's prompt verbatim (truncated internally to
            :data:`QUERY_MAX_CHARS`).
        cwd: workspace root -- the basis for the daemon's C1 reverse lookup
            of the home library; must share origin with the cwd of the MCP
            sessions of the same workspace (in the hook, take
            ``ZCODE_PROJECT_DIR`` or the process cwd).
        key_library / emb_model / daemon_version: the three elements of the
            daemon registry key; must match the env the target daemon was
            started with (see the module docstring's constant notes).
        hello_library: library declared in HELLO (same as launcher's
            ``HIPPOCAMPUS_PERSIST_DIR``; the user-level form passes the user
            library path; empty declares none).
        trace: when given a list, human-readable diagnostics are appended
            step by step (used by cli context; the hook passes None for zero
            overhead).
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
    except Exception as exc:  # noqa: BLE001 -- fail-open: a hook failure must never stall the session
        _note(f"failed ({type(exc).__name__}: {exc})")
        return ""
