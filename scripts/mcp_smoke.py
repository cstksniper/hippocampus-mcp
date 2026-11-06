# -*- coding: utf-8 -*-
"""端到端 MCP 冒烟：通过真实 stdio 协议调用 HippoBrain 工具。

验证链路：MCP 客户端 → stdio → server.py (FastMCP) → HippoBrain → 持久化

运行：
    python scripts/mcp_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

# 与 scripts/ 下其他工具一致：按 __file__ 相对定位仓库根，不写死绝对路径
# （公开层纪律——clone 到任何目录都能跑）。
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def main() -> None:
    env = os.environ.copy()
    env["HIPPOCAMPUS_PERSIST_DIR"] = os.path.join(PROJECT_DIR, "data", "demo_brain_mcp")
    env["PYTHONPATH"] = os.path.join(PROJECT_DIR, "src")
    env["PYTHONFAULTHANDLER"] = "1"
    # 默认剖面是 common（只挂 10 条）。冒烟要覆盖全工具面，显式放开。
    env["HIPPOCAMPUS_TOOL_PROFILE"] = "full"
    # 模型已缓存，离线模式避免首次 embed 时的 HF 网络检查挂死
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hippocampus_mcp.server"],
        env=env,
        # 🔴 必须换 cwd：工作区引导文件 <cwd>/.hippocampus.json 的优先级**高于**
        # HIPPOCAMPUS_PERSIST_DIR。在仓库根跑时会被重定向到默认库，
        # 于是「以为在测 demo 库，其实在往开发库写」——而整个过程只打一条 info。
        cwd=tempfile.mkdtemp(prefix="hp_smoke_cwd_"),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(name: str, args: dict) -> dict:
                """调工具并**断言没被降级成错误**。

                FastMCP 会把异常压成一行 ``ToolError`` 文本，仍返回正常
                content 列表 —— 不查错误标记就会把失败读成成功
                （实测：common 剖面下 ``brain_stats`` 报
                ``Unknown tool`` 但脚本照样打印 ✅ OK）。
                """
                r = await session.call_tool(name, args)
                # SDK 字段名是 is_error（协议 JSON 里才叫 isError）；双取一次，
                # 兜住不同 mcp SDK 版本的命名差异（实测 2.3.0
                # 只有 is_error，直接写 r.isError 会 AttributeError 崩在第一次调用）。
                if getattr(r, "is_error", False) or getattr(r, "isError", False):
                    raise RuntimeError(f"{name} failed: {r.content[0].text[:300]}")
                return json.loads(r.content[0].text) if r.content else {}

            tools = await session.list_tools()
            print(f"[smoke] {len(tools.tools)} tools registered", flush=True)
            if len(tools.tools) < 40:
                raise RuntimeError(
                    f"tool surface has only {len(tools.tools)} tools; "
                    "the full profile should expose 46"
                )

            # 0. 轻量调用探测（不触发嵌入/NER）
            stats = await call("brain_stats", {})
            print("[stats]", json.dumps(stats, ensure_ascii=False)[:150], flush=True)

            # 1. ingest
            await call("ingest", {
                "text": "我叫林晚，在杭州做独立游戏开发，正在做一款解谜游戏。",
                "speaker": "user",
            })
            print("[ingest] ok")

            # 2. get_memory_context
            ctx = await call("get_memory_context", {
                "query": "用户是做什么工作的",
                "top_k": 3,
            })
            print("[context]", ctx["text"][:200].replace("\n", " | "))

            # 3. session_start / session_end
            start = await call("session_start", {"task": "聊游戏开发"})
            _lib = (start or {}).get("library") or {}
            print("[session_start]", json.dumps(start, ensure_ascii=False)[:100])
            _want = os.path.normcase(os.path.abspath(env["HIPPOCAMPUS_PERSIST_DIR"]))
            _got = os.path.normcase(os.path.abspath(_lib.get("path") or ""))
            if _got and _got != _want:
                print(f"[smoke] ⚠️ actually wrote to {_lib.get('name')} @ {_got}, "
                      f"mismatch with HIPPOCAMPUS_PERSIST_DIR={_want}")
            await call("session_end", {
                "summary": "了解用户：林晚，杭州独立游戏开发者",
                "open_topics": "解谜游戏的美术风格",
            })
            print("[session_end] ok")

            # 4. consolidate + 复查 stats
            await call("consolidate", {})
            print("[consolidate] ok")
            stats = await call("brain_stats", {})
            print("[stats]", json.dumps(stats, ensure_ascii=False)[:200])

    print("\n✅ MCP stdio end-to-end OK")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as e:
        import traceback

        traceback.print_exc()
        print(f"[smoke] failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
