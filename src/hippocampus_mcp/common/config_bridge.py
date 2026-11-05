"""构造注入缝：受限层与 config 之间的字符串运输桥。

架构依赖矩阵只许 engine/brain import config；store/lnn/retrieval 等消费层
的旋钮已全部登记进 ``config``（ENV_MAP），但消费点拿值的形态是——**唯一
生产构造方 engine**（manager import 时）调用各模块的 ``apply_config`` 把
``load_config()`` 的字段值灌进本模块的注入槽。

解析序＝**env 先、注入值回退**（:func:`resolve`）：

  * env 已设（含空串）→ env 原文——「每次现读、测试可切」的契约原样，
    测试直接设 ``os.environ`` 的全部既有面零改动；
  * env 未设 → 注入值（生产默认值单点走 config）；
  * 都没有 → 各旋钮原 env 兜底值。

config 值本身就是 env 的读取结果，所以两条路径在任何时序下取值一致；
解析（float/int/白名单）仍留在各消费点（统一口径：字符串运输零漂移）。
"""

from __future__ import annotations

import os

_INJECTED: dict[str, str] = {}


def apply_config(**values: str) -> None:
    """构造方注入 config 字符串运输值（同名键后者覆盖）。"""
    _INJECTED.update(values)


def resolve(env_name: str, key: str, env_default: str) -> str:
    """旋钮取值：env 先（含空串原文），env 未设才落注入值/原默认。"""
    injected = _INJECTED.get(key)
    return os.environ.get(
        env_name, env_default if injected is None else injected)
