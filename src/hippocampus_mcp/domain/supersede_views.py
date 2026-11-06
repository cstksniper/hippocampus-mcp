"""并存口径读侧视图族：取代列表 / 误杀恢复 / 标签覆盖。

从 ``brain.core`` 整搬三件。与 ``store/supersede.py``（写侧取代
检测状态机）同名不同物：这里是**读侧/修复侧视图**。零 import，
store 能力与状态常量由壳**调用时现取**注入。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * list_superseded：只列 ``state == superseded``；``source`` 非 None
    才过滤（全等，记录侧 ``or ""`` 兜底）；``superseded_by_content``
    查不到时兜底空串；按 timestamp 降序后 ``[:limit]``——**limit<=0
    没有特殊语义**（``[:0]`` 空、``[:-1]`` 真截尾），切片怪癖原样保留；
  * un_supersede：三路范围一个都不给 ⇒ 返回 error 而不是动库；
    ``memory_ids`` 假值（含空列表）视作"未指定"；预览 ``content[:80]``；
    恢复 = ``state→active`` + ``superseded_by=None``，**有目标才 save**
    （惰性回调，预览零写零存）；
  * tag_coverage：``tags`` 假值视作无标签；键分布 top 15（按计数降序、
    同分按键名升序）；``coverage`` 空**库**返回 ``0.0`` 而不是除零。
"""
from __future__ import annotations

from typing import Any, Callable, Iterable


def list_superseded(
    records: Iterable[Any],
    *,
    get: Callable[[str], Any],
    superseded_state: str,
    limit: int,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """列出被标记为 superseded（已被取代）的记忆，含"被谁取代"。

    批量导入被误杀时用它做审计：覆盖报告·高 4 实测 28 对，绝大多数是
    把两条各自的真事实判成了更新关系。
    """
    out: list[dict[str, Any]] = []
    for r in records:
        if r.state != superseded_state:
            continue
        if source is not None and (r.source or "") != source:
            continue
        new = get(r.superseded_by) if r.superseded_by else None
        out.append({
            "memory_id": r.id,
            "content": r.content,
            "source": r.source or "",
            "superseded_by": r.superseded_by,
            "superseded_by_content": new.content if new is not None else "",
            "timestamp": r.timestamp,
        })
    out.sort(key=lambda x: -x["timestamp"])
    return out[:limit]


def un_supersede(
    records: Iterable[Any],
    *,
    update: Callable[..., Any],
    save: Callable[[], Any],
    superseded_state: str,
    active_state: str,
    memory_ids: list[str] | None = None,
    all: bool = False,  # noqa: A002 - 与 CLI 语义对齐，保留此名
    source: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """把误判为"已被取代"的记忆恢复为 active（解除 superseded_by）。

    误杀修复用（覆盖报告·高 4 的附带项）：``_detect_supersede`` 曾因实词
    重叠门槛过松，把 28 对无关/独立事实判成更新关系。恢复后它们重新参与
    默认召回，原取代者不受影响。

    必须显式指定范围（``memory_ids`` / ``all=True`` / ``source``），
    避免一条命令把全库状态改掉。默认 ``dry_run=True`` 只预览。
    """
    if not memory_ids and not all and source is None:
        return {
            "ok": False,
            "error": "a scope must be given: one of memory_ids / all=True / source",
        }

    targets = []
    for r in records:
        if r.state != superseded_state:
            continue
        if memory_ids and r.id not in memory_ids:
            continue
        if source is not None and (r.source or "") != source:
            continue
        targets.append(r)

    preview = [
        {"memory_id": r.id, "content": r.content[:80], "source": r.source or ""}
        for r in targets
    ]
    if dry_run:
        return {"ok": True, "dry_run": True, "count": len(targets),
                "candidates": preview}

    for r in targets:
        update(r.id, state=active_state, superseded_by=None)
    if targets:
        save()
    return {"ok": True, "dry_run": False, "restored": len(targets),
            "memory_ids": [r.id for r in targets]}


def tag_coverage(records: Iterable[Any]) -> dict[str, Any]:
    """库级标签覆盖率与键分布（验收判据）。

    「有多少条带了域标签」此前只能靠导出整个库自己数——没有自曝，宿主就
    只能凭感觉说"好像大部分都有"。键分布只回 top 键，避免大库返回体爆炸。
    """
    total = 0
    tagged = 0
    key_counts: dict[str, int] = {}
    for rec in records:
        total += 1
        tags = rec.tags or {}
        if tags:
            tagged += 1
        for k in tags:
            key_counts[str(k)] = key_counts.get(str(k), 0) + 1
    top = sorted(key_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    return {
        "total": total,
        "tagged": tagged,
        "untagged": total - tagged,
        "coverage": round(tagged / total, 4) if total else 0.0,
        "distinct_keys": len(key_counts),
        "top_keys": [{"key": k, "count": v} for k, v in top],
    }
