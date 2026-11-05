# -*- coding: utf-8 -*-
"""``filter_tags`` 键名白名单校验。

它要调 ``domain.tagging``
（纯域校验），server.py 与用例服务都要用——application 层不准
import server，故落址本模块；server.py 以别名 import 保留既有调用点。

返回 ``(effective_tags, info)``。调用方约定：

* ``info is None``            —— 无信息（未声明白名单或键全合法），直接用
  ``effective_tags``（与传入的 ``parsed_tags`` **逐位相同**）。
* ``info`` 含 ``error``       —— 拦截，直接把 ``info`` 当错误体返回。
* ``info`` 含 ``degraded``    —— 已软降级（剥掉非法键），用 ``effective_tags``
  继续跑，并把 ``info`` 交给 :func:`~hippocampus_mcp.common.params.filter_applied`
  回显。

为什么放在解析**之后**：解析失败（不是合法 JSON）与键名非法是两种错，
前者不需要查 schema，先判更省事，且错误文案更准。
"""
from __future__ import annotations

from typing import Any

from ..domain.tagging import validate_filter_keys


def check_tag_keys(brain: Any, parsed_tags: dict[str, Any] | None
                   ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """校验 ``filter_tags`` 的键名（brain 仅用于取 ``tag_extractor``）。"""
    if not parsed_tags:
        return parsed_tags, None
    return validate_filter_keys(parsed_tags, brain.tag_extractor)
