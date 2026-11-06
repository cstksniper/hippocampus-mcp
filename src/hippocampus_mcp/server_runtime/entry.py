# -*- coding: utf-8 -*-
"""MCP stdio 启动序列（server.py 的 ``main()`` 启动编排本体；server.py 只留薄壳）。

这里实现 main() 的全部启动编排——文件日志挂接、stdout 护栏安装、导入链
预载、诊断看门狗、三档预热（sync/off/auto）、启动耗时自曝、
``server.run()``。

server.py 留 ``def main(): _entry.run_main()`` 壳（cli.py / launcher.py 的
``from .server import main`` 零改动）。

🔴 现取纪律：server 实例与全部启动依赖（``_PERSIST_DIR`` / ``logger`` /
``_preload_import_chain`` / ``_WARMUP_DIAG`` / ``_WARMUP_MODE`` /状态常量
/ ``start_background_warmup`` / ``_PROCESS_T0`` / ``_HANDSHAKE_WINDOW`` /
``server``）一律经 :func:`_srv` **调用时现取**——测试替换
``server.server``（假 FastMCP）或 patch 任一状态量，启动序列拿到的是替身。
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from hippocampus_mcp.server_runtime.stdout_guard import _ThreadAwareStdout

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """server 模块本体（现取入口，见模块 docstring）。"""
    from hippocampus_mcp import server as _s

    return _s


def run_main() -> None:
    """启动 MCP stdio 服务端（server.py ``main()`` 薄壳的实现本体）。"""
    s = _srv()
    # 调试用文件日志（stdio 模式下 stderr 可能被截断，文件日志能完整还原崩溃现场）
    os.makedirs(s._PERSIST_DIR, exist_ok=True)
    _fh = logging.FileHandler(
        os.path.join(s._PERSIST_DIR, "mcp_server_debug.log"), encoding="utf-8"
    )
    _fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    s.logger.addHandler(_fh)
    for _name in ("mcpserver", "FastMCP", "mcp"):
        logging.getLogger(_name).addHandler(_fh)
        logging.getLogger(_name).setLevel(logging.DEBUG)

    # 注意：这里**不再** import torch。torch 由 ``_prepare_torch()`` 在首次真正
    # 需要时导入（见 ``get_brain``），否则它的 ~5.7s 会直接顶在 MCP 握手路径上。
    # ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` 也随之下沉到那个函数里
    # （必须赶在 ``import torch`` 之前设进环境）。

    # stdout 护栏：后台线程里的第三方输出改道 stderr，避免污染 JSON-RPC
    if not isinstance(sys.stdout, _ThreadAwareStdout):
        sys.stdout = _ThreadAwareStdout(sys.stdout)

    # 导入链预载：必须在 server.run() 之前，否则会卡死（见 _preload_import_chain）
    s._preload_import_chain()

    # 诊断看门狗：与预热模式无关——卡死可以发生在主线程（off 模式的惰性初始化）
    # 或后台线程（auto），只看 auto 会漏掉一半现场。
    if s._WARMUP_DIAG:
        s._start_warmup_watchdog()

    if s._WARMUP_MODE == "sync":
        # 旧行为：预热完才应答握手。**不要给 MCP 宿主用**——宿主会话通常在
        # 10-15 秒内完成工具注册，预热期间的工具会被整个漏掉。
        try:
            brain = s.get_brain()
            brain._get_nlp()
            brain.manager.embedder.embed("warmup")
            s.logger.info(
                "warmup done: L1=%d entities=%d",
                brain.manager.store.count(), len(brain.entities),
            )
        except Exception as e:  # 预热失败不阻断服务（首次调用会重试）
            s.logger.warning("warmup failed (will retry on first call): %s", e)
    elif s._WARMUP_MODE == "off":
        s.logger.info("HIPPOCAMPUS_WARMUP=off: skipping warmup; the first tool call will initialize")
    else:
        s.start_background_warmup()

    # 启动耗时自曝：只报数，不阻断。卡死与慢在宿主侧长得一模一样，
    # 没有这一行就只能靠猜 —— 猜错方向的代价是整套接入纪律继续空转。
    _boot_s = time.time() - s._PROCESS_T0
    if _boot_s > s._HANDSHAKE_WINDOW:
        s.logger.warning(
            "startup took %.1fs, over the host tool-registration window (about %.0fs) -- "
            "this service's tools may have been missed at registration entirely; the durable fix is launcher + resident daemon.",
            _boot_s, s._HANDSHAKE_WINDOW)
    else:
        s.logger.info("startup took %.1fs (within the %.0fs window)",
                      _boot_s, s._HANDSHAKE_WINDOW)

    s.server.run()
