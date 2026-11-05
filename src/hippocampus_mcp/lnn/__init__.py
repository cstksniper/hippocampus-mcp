# -*- coding: utf-8 -*-
"""LNN 动态记忆核 (Liquid Neural Network Dynamic Memory Core).

默认为**纯 numpy** 的 CfC 实现（见 ``numpy_cfc`` /
``memory_core``），生产进程不再 import torch；``HIPPOCAMPUS_LNN_BACKEND=torch``
可回退 ncps.torch 实现（惰性导入）。

``Projector`` 依赖 torch，改为惰性导出——顶层 ``from .projector import
Projector`` 会把 torch 拖进每一个 ``import hippocampus_mcp.lnn`` 的进程，
正是这里要拆掉的那条隐式依赖链。
"""

from .checkpoint import StateCheckpoint
from .memory_core import MemoryCore

__all__ = ["MemoryCore", "Projector", "StateCheckpoint"]


def __getattr__(name: str):
    """PEP 562 惰性导出：Projector（及其 torch 依赖）只在真正用到时加载。"""
    if name == "Projector":
        from .projector import Projector

        return Projector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
