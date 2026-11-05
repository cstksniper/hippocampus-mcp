# -*- coding: utf-8 -*-
"""``brain.core.knowledge_map`` 的三块逻辑（从 core 抽取）。

**为什么值得搬**：``knowledge_map`` 是知识地图入口，75 行里行投影循环
（22 行）、pending/excluded 双口径、回显体组装三块全是纯逻辑。搬走后壳
只剩「门禁 → 自动建 → 注入 → 编排」一条直线。

**依赖方向**：本模块零 import——``ordered`` / ``STATUS_ARCHIVED`` /
``SUMMARY_KIND_HOST`` / ``signature_of`` / ``pending_summary_ids`` 五个
topic 包符号全由壳调用时现取后注入（domain 同层禁互引，topic 住在
domain 但只准依赖 common；沿用 knowledge_batch 先例）。
``stats`` / ``open_topics`` / ``open_intents`` / ``playbook`` 是 brain
侧能力，按值或按惰性回调注入。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``project_rows`` 的 ``limit`` 用真值判定：``limit=0`` = 不限（全部
   行）；``limit`` 计数的是**已入行的条数**（archived 被 continue 掉的不占
   额度），且 break 检查在 append **之后**（第 limit 行会入行再断）。
2. ``summary_stale`` 是三段 and 的 bool：``kind==HOST and signature and
   signature != signature_of(member_ids)``——host 归纳但签名为空 ⇒ 不标
   stale（宁可漏标也不误标，改写权在宿主）。
3. ``entities`` / ``tags`` 各截前 3；``span`` 拷贝一份（不与主题层同对象）；
   ``size`` / ``recall_hits`` 走 ``int(x or 0)``；``title`` 空则
   ``(未命名)``。
4. ``pending_counts`` 的 excluded 口径是同一函数 ``exclude_low_info=False``
   （假簇自曝差集）；``excluded_n = max(0, 差)``——正常 excluded ≥ pending，
   max(0) 是防御性的（文案钉）。
5. ``assemble_report`` 的 13 键固定序（键序即信号）；``playbook`` 只在
   ``pending`` 非空时经回调现取（空 pending 不调——惰性注入，调用次数
   与原实现逐次对齐）；``note`` 是操作说明书文案（刻意在返回体不在
   docstring，省 tools/list 的 token 税）。
6. 本方法**不检查 layer_enabled**——门禁在壳里；自动建（空层 + 有记录）
   也在壳里，注入的 knowledge 永远是**当前层快照**。
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# 块 1：主题行投影（旧壳 3021-3049）
# ======================================================================
def project_rows(
    knowledge: dict[str, Any],
    *,
    include_archived: bool,
    limit: int,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    status_archived: str,
    summary_kind_host: str,
    signature_of: Callable[[list[str]], str],
) -> list[dict[str, Any]]:
    """把主题层投影成地图行（14 键固定序，含 stale 自曝）。

    Args:
        knowledge: 主题层字典（壳直传当前层快照）。
        include_archived: 是否含已归档主题。
        limit: 行数上限；**0 = 不限**（真值判定，见怪癖 1）。
        ordered: ``topic_mod.ordered``（层序枚举）。
        status_archived: ``topic_mod.STATUS_ARCHIVED`` 常量。
        summary_kind_host: ``topic_mod.SUMMARY_KIND_HOST`` 常量。
        signature_of: ``topic_mod.signature_of``（成员签名函数）。

    Returns:
        地图行列表（层序，最多 limit 行）。
    """
    rows: list[dict[str, Any]] = []
    for t in ordered(knowledge):
        if not include_archived and t.get("status") == status_archived:
            continue
        rows.append({
            "id": t.get("id"),
            "title": t.get("title") or "(未命名)",
            "status": t.get("status"),
            "size": int(t.get("size") or 0),
            "summary": t.get("summary") or "",
            "summary_kind": t.get("summary_kind") or "",
            # 归纳所依据的成员集合已经变了 ⇒ 这条上层结论可能不再成立。
            # 不改写、只标注（改写权在宿主），但必须看得见。
            "summary_stale": bool(
                t.get("summary_kind") == summary_kind_host
                and t.get("summary_signature")
                and t.get("summary_signature")
                != signature_of(t.get("member_ids") or [])),
            "entities": list(t.get("entities") or [])[:3],
            "tags": list(t.get("tags") or [])[:3],
            "span": list(t.get("span") or []),
            "updated_at": t.get("updated_at"),
            "recall_hits": int(t.get("recall_hits") or 0),
            # S3：这条归纳带没带四字段结构（fras）。机器要按域处理归纳
            # （矛盾检测 / 域卡）时，没有结构就没有抓手——先让覆盖率可见。
            "has_fras": bool(t.get("summary_fras")),
        })
        if limit and len(rows) >= limit:
            break
    return rows


# ======================================================================
# 块 2：pending / excluded 双口径（旧壳 3050-3056）
# ======================================================================
def pending_counts(
    knowledge: dict[str, Any],
    *,
    defaults: Any,
    pending_summary_ids: Callable[..., list[str]],
) -> tuple[list[str], int]:
    """待归纳清单 + 假簇自曝差集（同一函数两个口径算出来的差）。

    Args:
        knowledge: 主题层字典。
        defaults: ``_topic_defaults()`` 现取快照。
        pending_summary_ids: ``topic_mod.pending_summary_ids``。

    Returns:
        ``(pending, excluded_low_info_n)``。
    """
    pending = pending_summary_ids(knowledge, defaults=defaults)
    # 假簇自曝：pending 里被低信息量判据挡掉的那批，不报出来
    # 宿主会以为"都归纳完了"。与 pending 同一口径算出来的差集。
    excluded = pending_summary_ids(knowledge, exclude_low_info=False,
                                   defaults=defaults)
    excluded_n = max(0, len(excluded) - len(pending))
    return pending, excluded_n


# ======================================================================
# 块 3：回显体组装（旧壳 3057-3078）
# ======================================================================
def assemble_report(
    *,
    rows: list[dict[str, Any]],
    total: int,
    stats: dict[str, Any],
    open_topics: list[Any],
    open_intents: list[Any],
    auto_built: bool,
    pending: list[str],
    excluded_low_info_n: int,
    playbook: Callable[[int, int], Any],
) -> dict[str, Any]:
    """组装知识地图回显体（13 键固定序，键序即信号）。

    怪癖复刻：``playbook`` 经回调现取且**只在 pending 非空时调一次**
    （空 pending ⇒ None 且不调——惰性注入，调用次数对齐）。
    """
    return {
        "enabled": True,
        "topics": rows,
        "shown": len(rows),
        "total": total,
        "stats": stats,
        "open_topics": open_topics,
        "open_intents": open_intents,
        "auto_built": auto_built,
        "note": ("代表句是抽取式摘录（离质心最近的几条原文），不是归纳；"
                 "真正的上层结论要宿主总结后写回"),
        "pending_excluded_low_info": excluded_low_info_n,
        # ── 「整理这个库」的操作说明书 ────────────────────────────
        # 刻意放在**返回体**而不是 docstring：docstring 每会话随 tools/list
        # 注入一次，写进去就是长期 token 税；返回体只有真要整理时才付，
        # 而且能按当前库的实际状态动态生成（几条待归纳、撞没撞顶）。
        "pending": pending,
        "pending_count": len(pending),
        "playbook": (playbook(len(pending), len(rows)) if pending else None),
    }


# ======================================================================
# 编排入口（壳的唯一调用点）
# ======================================================================
def collect(
    knowledge: dict[str, Any],
    *,
    include_archived: bool,
    limit: int,
    defaults: Any,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    status_archived: str,
    summary_kind_host: str,
    signature_of: Callable[[list[str]], str],
    pending_summary_ids: Callable[..., list[str]],
    stats: dict[str, Any],
    open_topics: list[Any],
    open_intents: list[Any],
    playbook: Callable[[int, int], Any],
    auto_built: bool,
) -> dict[str, Any]:
    """knowledge_map 全流程：投影 → 双口径 pending → 回显（顺序有语义）。

    与原实现的调用次序逐行对齐：行循环在最前；pending/excluded 两份
    清单随后（excluded 在 pending 之后）；stats / open_* / playbook 在
    回显组装时按值/按回调现取。
    """
    rows = project_rows(knowledge, include_archived=include_archived,
                        limit=limit, ordered=ordered,
                        status_archived=status_archived,
                        summary_kind_host=summary_kind_host,
                        signature_of=signature_of)
    pending, excluded_n = pending_counts(
        knowledge, defaults=defaults,
        pending_summary_ids=pending_summary_ids)
    return assemble_report(rows=rows, total=len(knowledge), stats=stats,
                           open_topics=open_topics,
                           open_intents=open_intents,
                           auto_built=auto_built, pending=pending,
                           excluded_low_info_n=excluded_n,
                           playbook=playbook)
