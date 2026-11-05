# -*- coding: utf-8 -*-
"""pytest 公共配置：把 `src/` 与仓库根注入 sys.path。

tests/ 下的用例自带相对路径的 `sys.path.insert(..., "src")`，所以
`python tests/test_xxx.py` 单跑就能工作。但 pytest 的执行上下文不同
（rootdir / import mode 都可能变），这里再显式兜一次，保证两种跑法行为一致：

    python tests/test_forget.py      # 单跑
    python scripts/run_all_tests.py  # 零依赖跑全套
    pytest tests/                    # 装了 pytest 的环境

本文件只用标准库，不 import pytest —— 所以零依赖 runner 也不会因为它而失败。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
