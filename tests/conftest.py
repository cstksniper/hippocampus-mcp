# -*- coding: utf-8 -*-
"""pytest shared configuration: inject `src/` and the repo root into sys.path.

Test cases under tests/ already carry their own relative-path `sys.path.insert(..., "src")`,
so `python tests/test_xxx.py` works standalone. But pytest's execution context differs
(rootdir / import mode may both change), so this file adds an explicit safety net to keep both run styles consistent:

    python tests/test_forget.py      # standalone run
    python scripts/run_all_tests.py  # full suite with zero dependencies
    pytest tests/                    # environments with pytest installed

This file uses only the standard library and does not import pytest — so the zero-dependency runner never fails because of it.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
