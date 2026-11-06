# -*- coding: utf-8 -*-
"""派发器（渐进式披露的调度壳；server.py 工具派发区块的实现本体）。

这里实现：``_dispatch_map``（action → 底层工具函数）、``_hippo_help``、
``_hippo_extra``。**注册块留在 server.py**（``server._tool_manager.add_tool``
读 server 实例）。

🔴 现取纪律：工具名单现取自 ``s._TOOL_GROUPS``（patch 即变）、工具函数
现取自 ``vars(s)``（模块命名空间实时读）——**快照替身会吞新注册的工具**
（负对照 tests/test_m6_batch5_shell_split.py N1 钉死这一差异）。
被门控关掉的只是 MCP 注册，模块级函数对象仍在（``_gated_tool`` 的 deco
return fn），所以这里总能取到可调用对象。

设计背景（server.py 原注释，随迁）：把"46 个工具的 schema 每轮都进上下文"
改成"常用 9 个常驻 + 其余按名字派发"，代价是低频操作多一次 hippo_help
查询、参数改走 JSON。仅在 _TOOL_PROFILE != "full" 时注册，两个理由：
  1. full 下 46 个工具都已直接暴露，派发器是多余负担（约 +400 token）；
  2. 用 _tool_manager.add_tool 程序化注册、而非再加装饰器，可保持
     "已声明工具面 = 46"这一事实不变——工具数断言按 46 写死。
"""
from __future__ import annotations

import json
from typing import Any


def _srv() -> Any:
    """server 模块本体（现取入口，见模块 docstring 的现取纪律）。"""
    from hippocampus_mcp import server as _s

    return _s


def _dispatch_map() -> dict[str, Any]:
    """action → 底层工具函数（名单与函数对象全部现取）。

    名单现取 ``s._TOOL_GROUPS``（patch ``server._TOOL_GROUPS`` 后本函数
    名单随之变），函数现取 ``vars(s)``（新注册进 server 命名空间的工具
    立即可见）——任何一端用快照替身都会吞掉新注册的工具。
    """
    s = _srv()
    out: dict[str, Any] = {}
    for name in s._TOOL_GROUPS:
        fn = vars(s).get(name)
        if callable(fn):
            out[name] = fn
    return out


async def _hippo_help() -> str:
    """列出当前剖面下被隐藏、但可通过 hippo_extra 调用的 action 及其参数。"""
    # 这里是**唯一允许**显式传层级的地方之一：派发器一个入口代表 N 个工具，
    # 层级无法按自身函数名推断（本工具只列举目录，store 级就够）。
    # 详见 tests/test_warmup.py 的 W5.2（例外名单写死在那儿；本函数拆出
    # server.py 后，server.py 顶层不再有带参调用，例外名单自然空转、不翻红）。
    s = _srv()
    await s._await_warmup("store")
    import inspect

    hidden = [n for n in s._TOOL_GROUPS if not s._tool_enabled(n)]
    if not hidden:
        return "The current profile already exposes every tool; hippo_extra is unnecessary."
    lines = [f"This session has {len(hidden)} extension actions available through hippo_extra:", ""]
    for name in hidden:
        fn = vars(s).get(name)
        if not callable(fn):
            continue
        doc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
        brief = doc[0].strip() if doc else "(no description)"
        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):  # pragma: no cover
            params = []
        lines.append(f"- {name}({', '.join(params)}): {brief}")
    lines.append("")
    lines.append('Call: hippo_extra(action="name", args=\'{"argument": value}\')')
    return "\n".join(lines)


async def _hippo_extra(action: str, args: str = "{}") -> str:
    """按 action 名调用本会话未直接暴露的工具（导出 / 词库 / 实体图 / 运维等）。

    先调 hippo_help 拿 action 列表与参数名，再用本工具调用；参数以 JSON 对象传入。
    """
    import inspect

    s = _srv()
    fn = s._dispatch_map().get(action)
    if fn is None:
        return (f"Unknown action {action!r}. Available actions:"
                f"{', '.join(sorted(s._dispatch_map()))}")
    # 同上，派发器的结构性例外：**目标层级取决于 action**，只能运行时算。
    # 零参数约定在这里不适用（按 _hippo_extra 自身名字推断只会得到一个固定层级）。
    await s._await_warmup("store" if action in s.STORE_ONLY_TOOLS else "embed")
    text = (args or "").strip()
    kwargs: dict[str, Any] = {}
    if text:
        try:
            kwargs = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"args is not valid JSON: {exc}"
        if not isinstance(kwargs, dict):
            return "args must be a JSON object"
    try:
        allowed = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        allowed = None
    if allowed is not None:
        bad = [k for k in kwargs if k not in allowed]
        if bad:
            return f"{action} accepts no arguments {bad}; available arguments: {', '.join(sorted(allowed))}"
    try:
        return await fn(**kwargs)
    except TypeError as exc:
        return f"Call {action} failed (bad arguments): {exc}"
