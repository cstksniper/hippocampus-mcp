"""prewarm —— 把 daemon 的创建者从「宿主进程树」换成「用户会话级」。

## 为什么需要这个模块

实测结论：

- daemon 的创建者是**宿主 spawn 的 launcher** ⇒ daemon 落在宿主的**进程树**里
- 宿主空闲约 5 分钟后回收会话，daemon 被一起终止
  —— 日志里 **0 条退出记录、0 条崩溃痕迹**，看起来像"无声消失"
- ``CREATE_BREAKAWAY_FROM_JOB`` 治不了它：已用行为实验证伪，
  daemon 本来就不在宿主 Job 里（``0x2400 = JOB_MEMORY|SILENT_BREAKAWAY_OK``），
  那是条伪需求

**治本解是换创建者**。让 daemon 由「登录启动 / 计划任务」拉起，它的父进程就是
explorer 或 Schedule 服务，宿主回收会话时够不着它。

launcher 侧**一行都不用改**：它本来就先查 registry，有活的就直接复用
（``ensure_daemon`` 的路径 ①）。所以预热只是"提前把 registry 填好"。

## 用法

::

    # 预热（幂等：已有可用 daemon 就跳过）
    python -m hippocampus_mcp.prewarm --cwd D:\\work\\proj

    # 带显式库 / 模型（必须与宿主给 launcher 的环境变量一致）
    python -m hippocampus_mcp.prewarm --cwd D:\\work\\proj \\
        --library D:\\data\\hippo_proj --model BAAI/bge-base-zh-v1.5

    # 只查询，不启动
    python -m hippocampus_mcp.prewarm --status --cwd D:\\work\\proj

    # 机器可读（计划任务 / 包装脚本用）
    python -m hippocampus_mcp.prewarm --cwd ... --json

退出码：0＝已有或已起好；1＝起不来；2＝参数错误。

## 🔴 正确性前提（错了不报错，只会白起一个）

1. **key 必须一致**：``key = daemon_key(library, model, version)``，与 launcher
   从环境变量算出来的必须同一个。三者任一不同 ⇒ 预热起的是**另一个** daemon，
   宿主会话里的 launcher 会另起一个真的，预热这个变成没人用的孤儿。
2. **cwd 必须正确**：``library`` 为空时 daemon 按**自己的 cwd** 定位库。
   cwd 错了会起来一个指向别处的 daemon —— 照样能起、照样写登记，只是库不对。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .launcher import (
    _registry_usable,
    daemon_key,
    read_registry,
    runtime_dir,
    spawn_daemon,
)

__all__ = ["prewarm", "state", "main"]


def state(key: str) -> dict[str, Any]:
    """该 key 当前有没有可用 daemon。只读，不启动任何东西。"""
    info = read_registry(key)
    return {
        "key": key,
        "running": bool(_registry_usable(info)),
        "info": info,
        "registry": os.path.join(runtime_dir(), f"daemon-{key}.json"),
    }


def prewarm(cwd: str = "", library: str = "", model: str = "",
            version: str = "") -> dict[str, Any]:
    """拉起常驻 daemon（若尚未有可用的）。**幂等**。

    返回 dict，至少含 ``ok`` / ``action`` / ``key``。
    ``action`` ∈ ``skip``（已有，复用）| ``spawned``（本次拉起）| ``failed``。
    """
    key = daemon_key(library, model, version)
    before = read_registry(key)
    if _registry_usable(before):
        return {
            "ok": True,
            "action": "skip",
            "key": key,
            "pid": before.get("pid"),      # type: ignore[union-attr]
            "port": before.get("port"),    # type: ignore[union-attr]
            "detail": "已有可用 daemon，复用它（未新起进程）",
        }

    info = spawn_daemon(key, library=library, model=model, cwd=cwd)
    if not info:
        return {
            "ok": False,
            "action": "failed",
            "key": key,
            "cwd": cwd or os.getcwd(),
            "detail": "spawn_daemon 返回 None；详见 "
                      f"{os.path.join(runtime_dir(), f'daemon-{key}.log')}",
        }
    return {
        "ok": True,
        "action": "spawned",
        "key": key,
        "pid": info.get("pid"),
        "port": info.get("port"),
        "cwd": cwd or os.getcwd(),
        "detail": "本次拉起（创建者＝本进程，应已脱离宿主进程树）",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hippocampus_mcp.prewarm",
        description="预热常驻 daemon：让它的创建者不是宿主进程树。")
    p.add_argument("--cwd", default="",
                   help="daemon 的工作目录。library 为空时它决定库定位，"
                        "预热场景应显式给出；默认取当前目录")
    p.add_argument("--library", default="",
                   help="库路径（对应 HIPPOCAMPUS_DAEMON_LIBRARY）")
    p.add_argument("--model", default="",
                   help="嵌入模型（对应 HIPPOCAMPUS_EMB_MODEL）")
    p.add_argument("--version", default="",
                   help="代码版本（对应 HIPPOCAMPUS_DAEMON_VERSION）")
    p.add_argument("--status", action="store_true",
                   help="只查询不启动")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="输出 JSON（便于计划任务/包装脚本解析）")
    args = p.parse_args(argv)

    key = daemon_key(args.library, args.model, args.version)
    run_cwd = args.cwd or os.getcwd()

    if not os.path.isdir(run_cwd):
        print(f"[prewarm] ✗ --cwd 不是目录: {run_cwd}", file=sys.stderr)
        return 2

    if args.status:
        res: dict[str, Any] = state(key)
        if args.as_json:
            print(json.dumps(res, ensure_ascii=False, indent=1))
        else:
            print(f"key={key}  running={res['running']}")
            print(f"registry={res['registry']}")
            if res["info"]:
                print(f"  pid={res['info'].get('pid')} "
                      f"port={res['info'].get('port')}")
        return 0

    res = prewarm(cwd=args.cwd, library=args.library,
                  model=args.model, version=args.version)
    if args.as_json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        mark = "✓" if res["ok"] else "✗"
        print(f"[prewarm] {mark} action={res['action']} key={key}")
        print(f"          {res['detail']}")
        if res["ok"]:
            print(f"          pid={res.get('pid')} port={res.get('port')}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
