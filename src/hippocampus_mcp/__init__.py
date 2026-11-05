# -*- coding: utf-8 -*-
""" hippocampus_mcp —— 智能体的记忆底座 (LNN 增强).

v0.3: 四层记忆 (L0 原始 / L1 事实 / L2 摘要 / L3 画像) + 实体索引层 + 用层.
"""

# 豁免登记：第三方库线程 env（numpy 导入前必须定），清单见 config.py。

import os as _os

# 内存账单：
# numpy 捆绑的 scipy-openblas 默认按核数预留每线程缓冲——32 核机器实测
# import 即提交 754.7MB 私有虚拟内存（未触碰不计 RSS，但吃 commit 上限，
# 多库驻留/daemon/页面文件紧张场景受害）。限 4 线程后 commit 111.8MB，
# 且小算子（9841×768 gemv）延迟反优于全核（1.74ms vs 1.95ms）。
# setdefault：显式设置的 env 优先；必须在任何 numpy import 之前生效，
# 所以放在包 __init__（python -m hippocampus_mcp.server 必经此处）。
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

__version__ = "0.1.0"


def __getattr__(name: str):
    """惰性导出 ``HippoBrain``（PEP 562）。

    这里原本是顶层的 ``from .brain import HippoBrain``，后果是**任何**
    ``import hippocampus_mcp.xxx`` 都会连带加载 brain → manager → torch
    （本机热态约 5.7s）。MCP 的 ``initialize`` 就在这条路径上，多花的时间会
    直接吃掉宿主的工具注册窗口，所以改成按需解析。

    ``from hippocampus_mcp import HippoBrain`` 与属性访问都仍然可用。
    """
    if name == "HippoBrain":
        from .brain import HippoBrain

        return HippoBrain
    if name == "TagExtractor":
        # 标签抽取器同样惰性导出：它只依赖标准库，但放在这里能统一出口，
        # 宿主/脚本不必记住它住在 tagging 模块。
        from .domain.tagging import TagExtractor

        return TagExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
