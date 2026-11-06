# -*- coding: utf-8 -*-
"""最近写入记忆（"上次干到哪"的最便宜实现，从 brain/core.py 提取）。

只取 active，内容截断到 200 字——预激活包/开场包是要拼进 prompt 的。
可选按 ``source`` 过滤（同一来源的最近几条，用于"这个采集源刚才进了什么"）。

为什么放 store/：库存记录的**查询组合**（state 过滤 + 排序 + 投影），与
contradictions 同族；``MemoryState`` 住在 ``store.memory``，domain
碰不到（这正是把这类组合放 store 侧的理由）。

端口纪律：

* 记录序列由调用方传入（``iter_records()`` 的口径由 brain 决定）；
* ``active`` 状态值注入而不是 import（``MemoryState.ACTIVE.value`` 是字符串
  常量，注入后测试可自由替换）；
* 截断长度 ``clip`` 与条数 ``limit`` 都是显式参数，不读 Config。

预存行为按「只挪位置不改行为」保留：``limit <= 0`` 时返回空列表
（**不是** Python 切片的负索引语义——原实现用 ``if limit > 0`` 守卫，
所以 -1 不是"去掉最后一条"，而是整个槽关闭）。
"""
from __future__ import annotations

from typing import Any, Iterable


def recent_memories(
    records: Iterable[Any],
    limit: int = 10,
    source: str | None = None,
    *,
    active: str,
    clip: int = 200,
) -> list[dict[str, Any]]:
    """最近写入的记忆（时间倒序）。"""
    rows = [
        r for r in records
        if r.state == active
        and (source is None or (r.source or "") == source)
    ]
    rows.sort(key=lambda r: -r.timestamp)
    return [
        {
            "memory_id": r.id,
            "content": r.content[:clip],
            "timestamp": r.timestamp,
            "source": r.source or "",
        }
        for r in rows[:limit] if limit > 0
    ]
