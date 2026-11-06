# -*- coding: utf-8 -*-
"""用户级记忆继承槽：纯选择与渲染。

落点：I/O 编排在
``brain/session.py::_user_scope_brief``（轻量读取路径——不构 brain、
不载嵌入模型），本模块只做**纯数据**的选择与文本渲染，与
:mod:`stats_views.pinned_brief` 同口径：

* ``records`` 是鸭子类型（用到 ``.id / .content / .importance /
  .timestamp / .pin / .state``）——domain 不 import store（arch G1），
  ``active_state`` 由调用方注入（``MemoryState.ACTIVE.value``）；
* 选择判据：**pinned 优先**（不看阈值）→ 阈值档
  （非 pinned 且 ``importance >= importance_min``）→ 组内一律
  ``(-importance, -timestamp)`` 降序 → 整体 cap 到 ``limit``；
  pinned 条目即使 importance 也过阈值，只在 pinned 档出现一次。
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_user_scope", "select_user_scope"]


def select_user_scope(
    records: list[Any],
    *,
    limit: int,
    importance_min: float,
    active_state: str = "active",
) -> list[dict[str, Any]]:
    """从用户级库的库存记录里选出继承槽条目。

    Args:
        records: 库存记录（调用方已从用户库后端取好并转成记录对象）。
        limit: 带几条。**``<= 0` 返回空列表**——这是"关闭"，不是切片：
            ``records[:limit]`` 在 ``limit=-1`` 时是"倒数第 1 条之外全要"，
            负值切片在这里是错的（关＝零痕迹的字面义）。
        importance_min: 阈值档下限（``Config.user_scope_importance``，
            默认 0.7）。边界**含等于**：``importance == importance_min`` 入档。
        active_state: 活跃状态字符串（调用方注入）。

    Returns:
        入选条目的投影 dict 列表（``memory_id / content / importance /
        timestamp / pin``），pinned 档在前、阈值档在后，各自
        ``(-importance, -timestamp)`` 降序，总量 cap 到 ``limit``。
    """
    if limit <= 0:
        return []
    # D2a 写侧隔离：待审卡不进 user_scope 槽——伪造"用户偏好"即使写进了
    # 用户库，也不能经继承槽注入任何会话（攻击链的最后一环，见设计稿 §4.3）。
    # getattr 兜底同 recall_flow：旧记录对象无该字段按未隔离处理。
    active = [r for r in records
              if r.state == active_state and not getattr(r, "quarantine", False)]
    pinned = [r for r in active if r.pin]
    threshold = [
        r for r in active
        if not r.pin and r.importance >= importance_min
    ]
    pinned.sort(key=lambda r: (-r.importance, -r.timestamp))
    threshold.sort(key=lambda r: (-r.importance, -r.timestamp))
    return [
        {
            "memory_id": r.id,
            "content": r.content,
            "importance": r.importance,
            "timestamp": r.timestamp,
            "pin": bool(r.pin),
        }
        for r in (pinned + threshold)[:limit]
    ]


def render_user_scope(entries: list[dict[str, Any]]) -> list[str]:
    """继承槽文本行：与 ``block_task_hits`` 同款渲染（``content[:200]（id[:8]``）。

    返回体键 ``block_user_scope``：宿主直接拼 prompt 的文本槽，
    与结构化 ``user_scope.memories`` 同源。
    """
    return [
        f"{e['content'][:200]}（{e['memory_id'][:8]}）" for e in entries
    ]
