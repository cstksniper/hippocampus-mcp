# -*- coding: utf-8 -*-
"""时间序列统计助手（从 brain/core.py 提取的纯函数层）。

四个函数都是**纯计算**：状态（daily_counts / miss_log / 参考时刻）
由调用方显式传入，函数内不碰 brain、不碰磁盘。原先它们是 HippoBrain
的私有方法，读 ``self.manager._reference_time`` 等属性；提取后参数
显式化，``project_overview`` / ``project_status``（以及未来
BootstrapService 里的同源逻辑）直接调用。

为何放 ``common/``：现调用方在 brain（L3），未来搬进 application
（L3）后归属不变；common 是 L0 底座两层都能用。本模块只 import
``common/constants`` 的 ``_now`` 与标准库 time，不违反任何依赖边。

参考时刻语义与 brain 侧逐字一致：``reference_time or _now()``——
manager 注入了参考时间（测试/回放/巩固回放）就用它，否则用墙钟。
这个 ``or`` 是行为的一部分：调用方传 ``None`` 与不传是两回事，
故不做默认值，逼调用方显式表态。
"""
from __future__ import annotations

import time
from typing import Any

from .constants import _now


def day_key(base: float, offset_days: int) -> str:
    """base 前 offset_days 天的 ``YYYY-MM-D`` 键（与 _bump_daily 同口径）。"""
    return time.strftime("%Y-%m-%d", time.localtime(base - offset_days * 86400))


def _daily_ingest(daily_counts: dict[str, dict[str, int]],
                  base: float, offset_days: int) -> int:
    """单日 ingest 计数：缺日记、缺字段、值 falsy 都按 0。"""
    return int(daily_counts.get(day_key(base, offset_days), {}).get("ingest", 0) or 0)


def recent_ingest(days: int,
                  daily_counts: dict[str, dict[str, int]],
                  reference_time: float | None) -> int:
    """近 days 天的写入总数（daily_counts.ingest 求和；缺日记 0）。"""
    base = reference_time or _now()
    return sum(_daily_ingest(daily_counts, base, i) for i in range(days))


def spark_series(days: int,
                 daily_counts: dict[str, dict[str, int]],
                 reference_time: float | None) -> list[int]:
    """近 days 天的写入序列（旧→新），供 sparkline 渲染与结构化返回。"""
    base = reference_time or _now()
    return [_daily_ingest(daily_counts, base, i) for i in range(days - 1, -1, -1)]


def recent_miss_count(days: int,
                      miss_log: list[dict[str, Any]],
                      reference_time: float | None) -> int:
    """近 days 天的召回未命中次数（miss_log 按时间过滤，含 cutoff 当天）。

    这是**知识缺口信号**：查了但库里没有——FRAS 的 S 字段纪律，
    它推不出（条数正常也可能到处是缺口），但不知道它就会重复踩坑。
    """
    cutoff = (reference_time or _now()) - days * 86400
    return sum(
        1 for m in miss_log
        if float(m.get("ts") or 0.0) >= cutoff
    )
