# -*- coding: utf-8 -*-
"""检索端过时窗口的纯判定（common/L0：engine 与 brain 都要引，零包内依赖）。

设计决策：**过时遗忘不做主动删除**——超窗记忆不删、不改状态、
不归档，只在检索时挡在输出之外。判定规则：

- ``window_seconds <= 0`` ＝ 关，永不过期（开关语义收在判定里，调用方免判）；
- ``pinned``（importance>=0.7 自动 pin）豁免——pin 的既有语义就是
  「豁免时间衰减」（``store/memory.sync_pin``），窗口是同一语义的延伸；
- ``timestamp <= 0`` 视为「时间未知」：判不了"过时"，不挡——防止迁移/导入
  路径留下的零时间戳旧数据被默认窗口整库静默吞掉；
- 「超过」取严格大于：age == window 的边界不算过时。

时间基准由调用方传入（约束 4：``_reference_time or time.time()``），
本模块不取墙钟。
"""
from __future__ import annotations

import time

# 星期缩写表。**不要用 ``time.strftime("%a")``** —— 它读进程 locale，
# 中文 locale 下会吐 "周五"/"星期六"，下游按英文匹配的地方会静默失配。
# （同类的 locale 坑：``datetime.strptime(..., "%a")`` 解析 LongMemEval 的
# ``2023/05/20 (Sat)`` 时会直接抛 ValueError，见 eval 脚本的解析函数。）
_WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def iso_date(timestamp: float) -> str | None:
    """epoch 秒 → ``YYYY-MM-DD``（UTC 基准）；时间未知（``<=0``）返回 None。

    为什么统一走 UTC：外部数据的会话日期形如 ``2023/05/20 (Sat) 02:21``，
    **本身不带时区**。解析端按 UTC 落库、渲染端按 UTC 读回，两端自洽，
    日期不会因为跑评测的机器在哪个时区而漂一天。

    返回 None 而不是 ``"1970-01-01"``：零时间戳在库里表示「时间未知」
    （迁移/导入路径的旧数据），渲染成 1970 会被读成"真的是 1970 年"。
    """
    if timestamp <= 0:
        return None
    return time.strftime("%Y-%m-%d", time.gmtime(timestamp))


def weekday_abbr(timestamp: float) -> str | None:
    """epoch 秒 → ``Mon``..``Sun``（UTC 基准）；时间未知返回 None。

    存在的理由：把「算星期几」从模型手里拿走。LLM 给一个日期问它是周几
    是会算错的，而这个换算正是「上周五」这类相对定位的前提。
    """
    if timestamp <= 0:
        return None
    return _WEEKDAY_ABBR[time.gmtime(timestamp).tm_wday]


def out_of_window(timestamp: float, pinned: bool, now: float,
                  window_seconds: float) -> bool:
    """age > 窗口且未钉住 → 过时（挡在检索输出之外）。

    返回 True = 应被过滤；False = 保留。**纯函数：零 I/O、零状态、零写入**。
    """
    if window_seconds <= 0:
        return False
    if pinned:
        return False
    if timestamp <= 0:
        return False
    return (now - timestamp) > window_seconds


def parse_date_bound(text: str, *, end_of_day: bool = False) -> float:
    """``YYYY-MM-DD`` → epoch 秒（UTC）。空串返回 ``0.0`` 表示该端不限。

    ``end_of_day=True`` 取当天 23:59:59，用于 before 上界——用户的直觉是
    「到那天为止」，若取当天 00:00 会把当天全部内容排除在外。

    非法输入**抛 ValueError**，不静默当"不限"：时间窗是用户显式给的筛选
    条件，静默失效会让结果看起来像"库里真没有"，而实际是参数没生效。
    这是本项目反复踩过的失效模式（见 AGENTS.md 的"静默放错库"一条）。
    """
    s = (text or "").strip()
    if not s:
        return 0.0
    from datetime import datetime, timezone
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt.timestamp()
    raise ValueError(
        f"unparseable date format: {text!r} (expected YYYY-MM-DD)")


def in_time_window(timestamp: float, after: float, before: float) -> bool:
    """`after <= timestamp <= before` 判定；两端 ``<=0`` 表示该端不限。

    与 :func:`out_of_window` 的**策略刻意相反**：那边对 ``timestamp<=0``
    （时间未知）是**保留**——系统自动挡时宁可漏挡，怕把迁移进来的零时间戳
    旧数据整库吞掉；这里是**排除**——用户明确给了区间，时间未知就证明不了
    它落在区间内。两者判定方向不同，理由也不同，别互相参照改。
    """
    if after <= 0 and before <= 0:
        return True
    if timestamp <= 0:
        return False
    if after > 0 and timestamp < after:
        return False
    if before > 0 and timestamp > before:
        return False
    return True

