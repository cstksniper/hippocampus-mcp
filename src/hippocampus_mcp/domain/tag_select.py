"""update_tags 的候选筛选段：按过滤条件挑出待改标签的记录。

从 ``brain.core._select_for_tag_update`` 整搬。这是
update_tags 的私有 helper，没有工具直调面；壳只留「取
``store.iter_records()`` → 派发」。

零 import。``records`` 由壳传入（``store.iter_records()`` 的
结果），记录按鸭子类型使用（要有 ``id`` / ``source`` / ``content`` /
``state`` / ``tags`` 属性）。

逐字保留的语义（动一个就是行为变化，parity 会抓）：
  * ``memory_ids`` 归一成 ``str`` 集合后做**精确 id 匹配**；空列表
    （falsy）视作"未指定"而不是"一个都不要"；
  * ``source`` / ``state`` 是**全等**匹配（``source`` 对比时对记录侧
    做 ``or ""`` 兜底）；``source_contains`` / ``contains`` 是子串；
  * ``untagged_only`` 是"有任何标签就排除"（``tags`` 非空即 tagged）；
  * ``missing_keys`` 的语义是「**至少缺其中一个键**」（补缺场景），
    不是「全部都缺」——后者在补两个键时几乎恒为空集；
  * 返回的第二项是**过滤条件自曝**：过滤条件写错时 ``matched=0`` 和
    「真没有符合条件的记录」长得一模一样，自曝是唯一的辨析手段。
    空串/空列表归一成 ``None``。
"""
from __future__ import annotations

from typing import Any


def select_for_tag_update(
    records: Any,
    *,
    source: str = "",
    source_contains: str = "",
    contains: str = "",
    state: str = "",
    untagged_only: bool = False,
    missing_keys: list[str] | None = None,
    memory_ids: list[str] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """按过滤条件挑出待改标签的记录。

    之所以要 ``contains``（正文关键词）这一路：真库实测 **52% 的无标签
    条目来自同一个文件**（``PROJECT_MEMORY_LOG.md``），``source`` 前缀
    过滤对这块完全无法细分，只能靠正文关键词把「这一批」圈出来。
    """
    wanted = {str(i) for i in memory_ids} if memory_ids else None
    mkeys = [str(k) for k in (missing_keys or []) if str(k)]
    out: list[Any] = []
    for rec in records:
        if wanted is not None and rec.id not in wanted:
            continue
        if source and (rec.source or "") != source:
            continue
        if source_contains and source_contains not in (rec.source or ""):
            continue
        if contains and contains not in (rec.content or ""):
            continue
        if state and rec.state != state:
            continue
        tags = rec.tags or {}
        if untagged_only and tags:
            continue
        if mkeys and any(k in tags for k in mkeys):
            continue
        out.append(rec)
    return out, {
        "source": source or None,
        "source_contains": source_contains or None,
        "contains": contains or None,
        "state": state or None,
        "untagged_only": bool(untagged_only),
        "missing_keys": mkeys or None,
        "memory_ids_n": len(wanted) if wanted else None,
    }
