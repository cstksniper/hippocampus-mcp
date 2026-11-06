"""统计自曝投影（只读）：pin / 召回反馈 / 字面通道 三个槽 + 钉住条目简报。

从 ``brain/core.py`` 提取（``_recall_feedback_stats`` 45 行 /
``_pin_stats`` 50 行 / ``lexical_self_report`` 48 行 / ``_pinned_brief`` 43 行）。

落点理由：四个都是**纯只读投影**——输入是库存记录（或后端对象）与几个
标量，输出是一个自曝 dict / 条目列表。零检索、零写盘、零模型，符合
domain「只准依赖 common」的矩阵（``typing`` 是标准库）。

🔴 **不 import store**：``MemoryState.ACTIVE.value`` 与 ``PIN_IMPORTANCE``
都在 store 包 ⇒ 一律由调用方注入（``active_state`` 字符串 /
``pin_threshold`` 浮点）。对齐 map_view 的 ``archived_state`` 口径。

🔴 **``pinned_brief`` 的相关度是回调**：``relevance(task, task_hit_ids)``
由调用方注入（brain 侧是 ``HippoBrain._pin_task_relevance``，它还要读
``_task_entity_ids`` ⇒ 留在 brain）。传的是**可调用对象**不是调用结果——
与原实现逐次对齐，且替身可注。
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "lexical_self_report",
    "pin_stats",
    "pinned_brief",
    "recall_feedback_stats",
]


def recall_feedback_stats(
    records: list[Any],
    active_state: str = "active",
) -> dict[str, Any]:
    """召回反馈自曝：``last_recalled_at`` 实际覆盖了多少条。

    ``recall_count`` 记的是**候选集**（谁进过检索池），
    ``last_recalled_at`` 记的是**结果集**（谁真的返回给了调用方）。
    两者覆盖率的差 = "检索到了但没被采用"的量。

    为什么要自曝：``last_recalled_at`` 正是 consolidate / 衰减的**时间基准**，
    而历史数据里它曾长期零写入（实测覆盖率约 9.5%）—— 覆盖率低时，
    "这条多久没被用过"这类归档结论会失真，且**不会报错**。

    口径：**不回填**（回填等于编造召回历史），
    改为每次统计自曝，把"可信度打几折"的判断交回给宿主。
    """
    active = [r for r in records if r.state == active_state]
    total = len(active)
    if not total:
        return {
            "active_total": 0,
            "with_last_recalled_at": 0,
            "recall_count_positive": 0,
            "coverage": None,
            "coverage_level": "unknown",
            "note": "no active memories in the library; coverage is meaningless",
        }
    with_last = sum(1 for r in active if (r.last_recalled_at or 0.0) > 0.0)
    counted = sum(1 for r in active if (r.recall_count or 0) > 0)
    cov = with_last / total
    if cov >= 0.5:
        level = "ok"
    elif cov >= 0.2:
        level = "low"
    else:
        level = "very_low"
    return {
        "active_total": total,
        "with_last_recalled_at": with_last,
        "recall_count_positive": counted,
        "coverage": round(cov, 4),
        "coverage_level": level,
        "note": ("last_recalled_at 是 consolidate / 衰减的时间基准；"
                 "coverage_level 非 ok 时，「多久没被用过」这类"
                 "归档结论应当打折看待"),
    }


def pin_stats(
    records: list[Any],
    active_state: str = "active",
    pin_threshold: float = 0.7,
    error: str | None = None,
) -> dict[str, Any]:
    """钉住（pin）自曝：到底钉了多少、其中多少是**词面规则自动钉**的。

    为什么要自曝：``PIN_IMPORTANCE = 0.7``，而 :meth:`_guess_importance`
    命中「现在 / 不再 / changed / ！」即给 0.7 ⇒ **自动 pin**；而 pin
    在系统内**只进不退**（唯一"退出"是调用方显式动作）⇒ 长库里 pin 会
    单调堆积（实测某库 78%），挤掉用层的动态位，且**全程无报错**。

    "疑似自动"的判据：``pin=True`` 且 importance 落在 ``_guess_importance``
    的值域内（``[PIN_IMPORTANCE, 0.75]``）。这是**疑似**不是确证 ——
    调用方显式传 0.7 也会落进来，所以字段名带 ``suspected``。

    Args:
        records: 库存记录（调用方已取好；取记录本身失败时传 ``error``）。
        active_state: 活跃状态字符串（``MemoryState.ACTIVE.value``）。
        pin_threshold: ``PIN_IMPORTANCE``，由调用方现取注入。
        error: 取记录失败的异常类型名——统计失败不该让其它自曝项挂掉。
    """
    if error is not None:
        return {"active_total": 0, "pinned_total": 0, "pinned_ratio": None,
                "suspected_auto_pinned": 0, "level": "unknown",
                "note": f"stats failed (other self-report fields unaffected): {error}"}
    active = [r for r in records if r.state == active_state]
    total = len(active)
    if not total:
        return {"active_total": 0, "pinned_total": 0, "pinned_ratio": None,
                "suspected_auto_pinned": 0, "level": "unknown",
                "note": "no active memories in the library; pin ratio is meaningless"}
    pinned = [r for r in active if getattr(r, "pin", False)]
    auto = [r for r in pinned
            if pin_threshold <= (getattr(r, "importance", 0) or 0) <= 0.75]
    breakdown = {"auto": 0, "manual": 0, "legacy": 0, "unmarked": 0}
    for rec in pinned:
        src = str(getattr(rec, "pin_source", "") or "")
        breakdown[src if src in breakdown else "unmarked"] += 1
    ratio = len(pinned) / total
    quota_warning = ratio >= 0.6
    level = "high" if ratio >= 0.5 else ("ok" if ratio >= 0.2 else "low")
    return {
        "active_total": total,
        "pinned_total": len(pinned),
        "pinned_ratio": round(ratio, 4),
        "suspected_auto_pinned": len(auto),
        "pin_threshold": pin_threshold,
        "pin_source_breakdown": breakdown,
        "quota_warning": quota_warning,
        "level": level,
        "note": (
            "supersede 会自动解 pin；其余退出仍靠调用方显式动作："
            "① report_feedback(useful=False) ② 显式 pin=False "
            "③ 降 importance 触发 sync_pin（store/vector_store.py → "
            "memory.py:175，双向）。level=high 意味着用层动态位被挤占，"
            "「suspected_auto_pinned」那部分是词面规则自动钉的（疑似，非确证）。"
            if level == "high" else
            "pin 占比正常。提醒：除 supersede 自动解 pin 外，"
            "其余解除只能靠调用方显式动作。"),
    }


def pinned_brief(
    records: list[Any],
    relevance: Callable[[str, set[str] | None], dict[str, int]],
    limit: int = 3,
    task: str = "",
    task_hit_ids: set[str] | None = None,
    active_state: str = "active",
) -> list[dict[str, Any]]:
    """钉住记忆槽。

    排序口径：**任务相关优先**。

    钉住条目的 importance 普遍偏高且彼此接近（实测：书级铁律 0.9 vs
    章节级卡 0.8），纯 importance 排序排出来的其实是"谁先被提权"——
    于是本次会话用得上的铁律被无关的会话卡挤掉。真正有区分度的是
    "跟这次 task 对不对得上"，故分三档：

        ``task_relevance`` = 2  task 语义召回命中了这条
                           = 1  与 task 里的**受信任实体**相交
                           = 0  无关

    同档内再按 importance / timestamp 降序；``task`` 为空时退回纯 importance
    （与旧行为一致）。实体只认受信任来源：启发式实体约 84% 是短语噪声，
    让它们参与"相关性"判定等于随机重排。

    Args:
        records: 库存记录。
        relevance: 相关度回调 ``(task, task_hit_ids) -> {memory_id: 0/1/2}``。
        limit: 返回条数。**负数 = 关闭该槽**（返回空列表）。
        task: 本次会话任务描述，用于任务相关优先。
        task_hit_ids: task 语义召回命中的 memory_id（``session_start`` 里
            已经算过一次，传进来复用，不重复检索）。
    """
    pinned = [
        r for r in records
        if r.pin and r.state == active_state
    ]
    rel = relevance(task, task_hit_ids)
    pinned.sort(key=lambda r: (-rel.get(r.id, 0), -r.importance, -r.timestamp))
    out: list[dict[str, Any]] = []
    for r in pinned[:max(int(limit), 0)]:
        out.append({
            "memory_id": r.id, "content": r.content, "score": r.importance,
            "similarity": 1.0, "timestamp": r.timestamp, "importance": r.importance,
            "state": r.state, "cognitive_type": r.cognitive_type,
            "source_ids": r.source_ids or [], "entity_ids": r.entity_ids or [],
            "pin": True, "source": r.source or "",
            # 自曝排序依据：宿主发现"铁律怎么掉出前三了"时要能查到原因
            "task_relevance": int(rel.get(r.id, 0)),
        })
    return out


def lexical_self_report(
    backend: Any,
    indexed: int,
    path: str,
) -> dict[str, Any]:
    """字面通道（BM25 / FTS5）的运行期自曝。

    为什么要这一节：sqlite + FTS5 最大的卖点是「**不用再建内存 BM25 索引**，
    省 34.5MB」，而这个优化此前在生产侧没有观测入口，「到底省没省」
    只能靠猜。这里把每一项都自曝**实测值**：

    - ``store_backend``：实际存储后端（json / sqlite）
    - ``fts5_ready``：FTS5 虚表在本连接里是否真的可用
    - ``bm25_indexed``：内存 BM25 现有几条 —— **FTS5 路真正生效时这里必须是 0**
    - ``last_lexical_path``：上次检索的字面通道实际走的路（fts5 / memory / off）

    Args:
        backend: 存储后端对象（``store._backend``，可为 ``None``）。
        indexed: 内存 BM25 现有条数（``len(brain._bm25_indexed)``）。
        path: 上次检索实际走的路（``brain._last_lexical_path``）。
    """
    name = getattr(backend, "name", None) or "unknown"
    fts_ready: bool | None = None
    for meth in ("_fts_ok", "fts5_supported"):
        fn = getattr(backend, meth, None)
        if callable(fn):
            try:
                fts_ready = bool(fn())
            except Exception:  # noqa: BLE001 —— 自曝不能因为探测失败就整体失败
                fts_ready = False
            break
    if name == "sqlite" and fts_ready and indexed == 0:
        level, note = "ok", "the literal channel uses FTS5; in-memory BM25 not built (34.5MB saved)"
    elif name == "sqlite" and fts_ready and path == "memory":
        level = "degraded"
        note = (f"FTS5 is available but this query still fell back to in-memory BM25 (built {indexed} records)；"
                "中文 2 字词触发回退是预期行为，频繁出现才需要查")
    elif name == "sqlite" and not fts_ready:
        level = "degraded"
        note = (f"sqlite backend without FTS5 => the literal channel falls back to in-memory BM25 "
                f"(built {indexed} records; the 34.5MB saving is unavailable)")
    elif name == "json":
        level, note = "json", f"json backend: the literal channel can only use in-memory BM25 (built {indexed} records)"
    else:
        level, note = "unknown", f"backend={name}, the literal channel's form cannot be determined"
    return {
        "store_backend": name,
        "fts5_ready": fts_ready,
        "bm25_indexed": indexed,
        "last_lexical_path": path,
        "level": level,
        "note": note,
    }
