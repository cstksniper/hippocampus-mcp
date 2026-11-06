# -*- coding: utf-8 -*-
"""brain 子包：记忆编排层。

``from hippocampus_mcp.brain import HippoBrain`` 是包外脚本与测试的既定契约
（60+ 个文件在用），所以本模块必须保持**搬迁前的命名空间等价**：

* ``..common.constants`` 的 ``import *`` 走它自己的 ``__all__``，包含
  ``_now`` / ``_nid`` / ``_env_int`` 这类下划线符号 —— ``import *`` 默认
  不导出下划线名字，靠 ``__all__`` 才能带出来，漏了就是只在某些代码路径上
  炸的 NameError；
* ``.core`` 的 ``import *`` 补齐 ``HippoBrain`` 及其余公共符号 —— 与原
  ``brain.py`` 不定义 ``__all__`` 时的导出行为一致。

两个 ``import *`` 都带 noqa：这是刻意的再导出（re-export），不是多余的导入。

⚠️ **可调常量的真身在 ``brain.core``**：``import *`` 只是把名字复制进包命名空间
的一份快照。``hippocampus_mcp.brain.UPDATE_TAGS_CONFIRM_ABOVE = 2`` 这种写法只改
包副本，``core`` 里的函数读的是 ``core`` 自己的全局 —— 模块级全局的 patch 只对
**定义并使用它的那个模块**生效。要调参请改 ``hippocampus_mcp.brain.core.XXX``，
读也请从 ``.core`` 读。两个测试里原有的 patch 已按此修正（j1_update_tags /
s2_tag_ops_undo），这是包化带来的唯一语义边界。

模块职责：
* ``core``    —— ``HippoBrain`` 主类（原 ``brain.py``，约 6300 行）
* ``pure``    —— 无状态 mixin（原 ``brain_pure.py``）
* ``entity``  —— 实体词典 mixin（原 ``brain_entity.py``）
* ``session`` —— 会话续接 mixin（原 ``brain_session.py``）
"""

from ..common.constants import *  # noqa: F401,F403
from .core import *  # noqa: F401,F403
