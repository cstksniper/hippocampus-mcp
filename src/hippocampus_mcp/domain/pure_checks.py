"""写入侧纯检测：近邻重复检索（``brain.core._near_duplicates`` 的判定核）。

本模块的定位是「判据/检测类纯逻辑」的落点之一。同族其余判据已在更早
各有归属，**不要**往这里搬（防撞车）：

  * ``is_restatement`` 判定核 → ``domain/policies.py``；
  * 写侧取代检测 → ``store/supersede.py``；
  * 读侧冲突标注 → ``domain/conflict_annotate.py``；
  * 矛盾候选筛选 → ``store/contradictions.py``。

零 import（需要的一切由壳注入）。``store`` / ``vector_of`` 按鸭子
类型使用：前者要有 ``get(mid)`` 与 ``search(q, top_k=...)``（后者产出
``(record, sim)`` 对），后者把记录映成查询向量（可为 ``None`` 表示
"没有向量"）。

逐字保留的怪癖（动一个就是行为变化，parity 会抓）：
  * 候选池请求宽度是 ``top_k + len(own) + 2``——多要一点，抵消
    "自己的这批" 占掉的席位，保证既有记忆还能凑满 ``top_k``；
  * 相似度过滤是严格小于（``sim < min_sim`` 才跳过，恰好等于门槛
    **算**命中）；
  * 同一条既有记忆被多条新记忆命中时，**保留相似度最高的那次**
    （先到先得、更高才替换）；
  * ``similarity`` 四舍五入到 3 位小数、``content`` 截前 200 字、
    ``source`` 缺省兜底空串；
  * ``top_k > 0`` 才截断；``top_k <= 0`` 返回全部（负数=不限制）。
"""
from __future__ import annotations

from typing import Any


def near_duplicates(
    memory_ids: list[str],
    *,
    store: Any,
    vector_of: Any,
    top_k: int,
    min_sim: float,
) -> list[dict[str, Any]]:
    """找出与"刚写入的这批记忆"最像的**既有**记忆。

    只看向量相似度（不再做矛盾判定——那是 consolidate 的活），但会把
    ``state`` / ``superseded_by`` / ``source`` 一并给出：已作废的旧版本、
    来自别的文件的同义表述，都是宿主当场该看到的。
    """
    own = set(memory_ids)
    found: dict[str, dict[str, Any]] = {}
    for mid in memory_ids:
        rec = store.get(mid)
        if rec is None:
            continue
        q = vector_of(rec)
        if q is None:
            continue
        for other, sim in store.search(q, top_k=top_k + len(own) + 2):
            if other.id in own or sim < min_sim:
                continue
            prev = found.get(other.id)
            if prev is None or sim > prev["similarity"]:
                found[other.id] = {
                    "memory_id": other.id,
                    "similarity": round(float(sim), 3),
                    "state": other.state,
                    "superseded_by": other.superseded_by,
                    "source": other.source or "",
                    "content": other.content[:200],
                }
    rows = sorted(found.values(), key=lambda r: -r["similarity"])
    return rows[:top_k] if top_k > 0 else rows
