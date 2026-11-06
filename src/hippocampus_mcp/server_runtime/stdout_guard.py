# -*- coding: utf-8 -*-
"""server.py 头部基础设施（sys.path 兜底 / 日志配置 / stdout 护栏）。

这里实现：``sys.path`` 兜底（libs 目录 append + src 目录 insert）、
日志配置（stderr 流 handler + 级别）、stdio 传输的 stdout 护栏类
``_ThreadAwareStdout``。

server.py 侧按 import 名原样再导出 ``_ThreadAwareStdout``（tests/test_warmup.py
直调实例化），``logger`` 由 :func:`configure_logging` 返回。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any


def ensure_paths(server_file: str) -> None:
    """``sys.path`` 兜底（调用方传 server.py 的 ``__file__``）。

    libs 目录只作兜底（append 不遮蔽正常安装的包；libs 内是 py3.10 编译的
    老包，放前面会遮蔽 site-packages 导致 "无法导入 sentence_transformers"）。
    """
    _LIBS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(server_file)))),
        "libs",
    )
    if _LIBS_DIR not in sys.path:
        sys.path.append(_LIBS_DIR)

    # 确保 src 目录在路径中
    _SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(server_file)))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)


def configure_logging() -> logging.Logger:
    """日志配置：stderr 流 handler + 配置级别。"""
    logger = logging.getLogger("hippocampus_mcp")
    if not logger.handlers:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(_handler)
        from hippocampus_mcp.config import load_config

        _log_level = load_config().log_level.upper()
        logger.setLevel(getattr(logging, _log_level, logging.INFO))
    return logger


class _ThreadAwareStdout:
    """stdio 传输下的 stdout 护栏。

    MCP stdio 模式里 fd 1 **只能**承载 JSON-RPC。但第三方库（transformers /
    tqdm / spaCy）在首次加载时可能 print 到标准输出——原先预热在握手前完成，
    顶多难看；现在预热跑到后台线程，正好与客户端的 tools/list 并发，
    一旦插进协议流，宿主就会解析失败（表现为"连上了但没工具"）。

    这里把**非主线程**写往 stdout 的内容改道 stderr，主线程（＝写协议的正主）
    不受影响。注意不能用 ``contextlib.redirect_stdout``：它改的是全局
    ``sys.stdout``，会把主线程的协议输出也一起带到 stderr。
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    @staticmethod
    def _sink(real: Any) -> Any:
        # 主线程 → 真 stdout（协议流）；其他线程 → stderr。
        # 注意不能返回 sys.stdout：安装后它就指向本代理，会无限递归。
        return real if threading.current_thread() is threading.main_thread() else sys.stderr

    def write(self, s: str) -> int:
        return self._sink(self._real).write(s)

    def flush(self) -> None:
        self._sink(self._real).flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)
