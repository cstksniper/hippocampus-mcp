"""按自然语言定位候选记忆并删除（或仅预览）——编排层。

宿主 agent 拿到的通常是"那条关于三月上线的记错了"这类自然语言，而不是
memory_id。流程是：先召回再删，**默认只预览**（dry_run=True），确认候选
无误后再以 dry_run=False 执行。

**执行删除（dry_run=False）有两道护栏**：

1. **必须有向量背书**：只考虑 ``similarity >= 门槛`` 的候选。recall 会放行
   BM25 词面命中但语义无关的条目，它们的 similarity 是 **0.0**，删它们等于
   "按关键词删库"。
2. **默认只删最匹配的一条**：``allow_multiple=False`` 时，即便有若干条过了
   门槛，也只删排序第一的那条。"删掉那条记错的"在语义上几乎总是单数，多出来
   的槽位正是误删的来源。确实要批量删请显式传 ``allow_multiple=True``。

两道都不是靠"调高相似度门槛"实现的——相关性随查询长度整体漂移（见
``FORGET_MIN_SIMILARITY`` 注释里的实测分布），定值门槛救不了这两个方向中的
任意一个；真正稳定的是"靶子恒为 #1"这个相对优势。

被跳过的候选一律原样列进 ``skipped`` 并带 ``skip_reason``，宿主可据此改用
memory_id 精确删除。

落点说明：本体是纯编排（分流 + 删除循环），recall / forget 两个
brain 入口以**可调用对象**注入（不预调用）；``min_similarity_default`` 由
调用方现取传入——``None`` 取默认值的语义原样保留，且调用方命名空间被 patch
时跟随（与 brain 侧其它常量的处理一致）。
"""
from __future__ import annotations

from typing import Any, Callable


def forget_by_query(
    *,
    query: str,
    top_k: int = 3,
    dry_run: bool = True,
    min_similarity: float | None = None,
    allow_multiple: bool = False,
    min_similarity_default: float,
    recall: Callable[..., list[dict[str, Any]]],
    forget: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """按自然语言定位候选记忆并删除（或仅预览）。

    Args:
        query: 自然语言描述（"那条关于三月上线的记错了"）。
        top_k: 召回条数，透传给 recall。
        dry_run: True 只预览不删除（默认）。
        min_similarity: 删除门槛；None 取 ``min_similarity_default``。
        allow_multiple: 允许多条一起删，默认 False（只删最匹配的一条）。
        min_similarity_default: 默认门槛（调用方现取，勿用 def 默认参数）。
        recall: 召回入口，按 ``recall(query, top_k=..., include_superseded=True)``
            形态调用。
        forget: 单条删除入口，按
            ``forget(memory_id, reason="forget_by_query: <query>")`` 形态调用。
    """
    floor = min_similarity_default if min_similarity is None else float(min_similarity)
    hits = recall(query, top_k=top_k, include_superseded=True)
    candidates = [
        {
            "memory_id": h["memory_id"],
            "content": h["content"],
            "score": round(float(h.get("score", 0.0)), 4),
            "similarity": round(float(h.get("similarity", 0.0)), 4),
            "state": h.get("state"),
        }
        for h in hits
    ]
    if not candidates:
        return {"ok": False, "error": f"no memories matched: {query}",
                "query": query, "candidates": []}

    if dry_run:
        return {
            "ok": True, "dry_run": True, "query": query,
            "candidates": candidates,
            "min_similarity": floor,
            "note": (f"When executed: only candidates with similarity >= {floor} are processed; "
                     f"at most 1 deletion unless allow_multiple=True is passed explicitly."),
        }

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for c in candidates:
        if c["similarity"] >= floor:
            eligible.append(c)
        else:
            skipped.append({**c, "skip_reason":
                            f"similarity {c['similarity']} below the threshold {floor}"})

    if not eligible:
        return {
            "ok": False,
            "error": (f"None of the {len(skipped)} candidates reached the deletion threshold {floor}; "
                      f"nothing was deleted. They are most likely BM25 surface-form hits. "
                      f"Rephrase closer to the target's exact wording, or call forget_memory with the "
                      f"memory_id from the dry_run result for a precise delete."),
            "query": query, "dry_run": False,
            "min_similarity": floor, "skipped": skipped, "removed_count": 0,
        }

    if not allow_multiple and len(eligible) > 1:
        for c in eligible[1:]:
            skipped.append({
                **c, "skip_reason":
                "single mode: only the best-matching one is deleted (pass allow_multiple=True for bulk deletion)",
            })
        eligible = eligible[:1]

    removed = []
    for c in eligible:
        report = forget(c["memory_id"], reason=f"forget_by_query: {query}")
        removed.append({
            "memory_id": c["memory_id"],
            "content": c["content"],
            "similarity": c["similarity"],
            "ok": report.get("ok", False),
        })
    out: dict[str, Any] = {
        "ok": True, "dry_run": False, "query": query,
        "min_similarity": floor,
        "removed": removed, "removed_count": len(removed),
    }
    if skipped:
        out["skipped"] = skipped
        out["skipped_count"] = len(skipped)
        out["note"] = (f"{len(skipped)} candidates were not deleted (see skip_reason in skipped); "
                       f"to delete them anyway, call forget_memory with their memory_id.")
    return out
