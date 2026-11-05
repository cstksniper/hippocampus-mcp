"""知识主题层操作族：成员打包 / 统计自曝。

从 ``brain.core`` 整搬两件。架构约束：``domain/topic`` 是同层（同层互引都
不行）⇒ topic 层能力全部经 :class:`TopicPort` **由壳调用时现取注入**
（``_topic_defaults`` / ``ordered`` / ``pending_summary_ids`` /
``representative_ids`` / ``summary_state`` / ``apply_host_summary`` /
``coverage_of`` / ``by_status`` / ``host_summary_stats`` 是回调，
``TOPIC_EVIDENCE_CLIP`` 是值）。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * knowledge_items：只收 **active**；``require_vec=False`` 连无向量记录
    一起收（``vec`` 为 None）——「存活」与「有向量」是两件事，混用会让
    active-无向量成员反复抖动；实体名只取**受信任实体**且在实体表里的
    （``e in trusted and e in id_to_name``）；timestamp/importance 走
    ``float(x or 0.0)``；tags 浅拷贝；
  * knowledge_stats：层关闭只回 note；空库 coverage 由 topic 层给；
    ``last_refresh_took_s`` None 不 round；note 有 topic 层文案时
    **不覆盖**（``out.get("note") or 默认文案``）；
> 注：``brief_batch`` / ``summarize_batch`` 两件不在本模块——曾经并存过
> 两套行为等价的同名实现，为避免双轨漂移只保留单一线，落在
> ``domain/knowledge_batch.py`` 与 ``domain/summarize_batch_flow.py``；
> 验证由那两个模块的目标测试与兼容套件承担。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TopicPort:
    """topic 层能力端口（壳构造，调用时现取——架构约束：同层不互引）。"""
    defaults_fn: Callable[[], Any]
    ordered: Callable[..., Any]
    pending_summary_ids: Callable[..., Any]
    representative_ids: Callable[..., Any]
    summary_state: Callable[..., Any]
    apply_host_summary: Callable[..., Any]
    coverage_of: Callable[..., Any]
    by_status: Callable[..., Any]
    host_summary_stats: Callable[..., Any]
    evidence_clip: int


def knowledge_items(
    records: Any, *, entities: dict, trusted_ids: Any,
    vector_of: Callable[..., Any], active_state: str,
    require_vec: bool = True,
) -> list[dict[str, Any]]:
    """把 active 的 L1 记录打包成 topic 模块的输入。

    只收 **active**：superseded / archived 是已经不算数的口径。
    ``require_vec=False`` 时连**还没算向量**的记录一起收（``vec`` 为
    None）——「存活」和「有向量」是两件事：聚类比不了没有向量的记录
    （宁缺勿滥，进 ``unassigned`` 自曝），但判「哪些成员已经失效」只能
    看 active。混用会让 active-无向量 的记录反复抖动 ⇒ 签名对不上 ⇒
    满屏假 stale。
    """
    trusted = trusted_ids
    id_to_name = {e.get("id"): k for k, e in entities.items()}
    items: list[dict[str, Any]] = []
    for rec in records:
        if rec.state != active_state:
            continue
        vec = vector_of(rec)
        if vec is None and require_vec:
            continue
        names = [id_to_name[e] for e in (rec.entity_ids or [])
                 if e in trusted and e in id_to_name]
        items.append({
            "id": rec.id,
            "vec": vec,
            "content": rec.content or "",
            "timestamp": float(rec.timestamp or 0.0),
            "importance": float(rec.importance or 0.0),
            "entities": names,
            "tags": dict(rec.tags or {}),
        })
    return items


def knowledge_stats(
    port: TopicPort, *, knowledge: dict, records: Any,
    active_state: str, last_at: Any, last_took: Any, last_dropped: Any,
) -> dict[str, Any]:
    """brain_stats 自曝：主题层现在什么状态（含覆盖率与上次刷新时点）。"""
    if not port.defaults_fn().layer_enabled:
        return {"enabled": False,
                "note": "主题层已关闭（HIPPOCAMPUS_TOPIC_LAYER=0）"}
    active_total = sum(1 for r in records
                       if r.state == active_state)
    out = port.coverage_of(knowledge, active_total)
    out.update({
        "enabled": True,
        "by_status": port.by_status(knowledge),
        # 归纳进度：host = 宿主已写回的；host_stale = 写了但成员已变。
        # 有这两格，「梳理到什么程度」就不用靠感觉。
        "summary_kinds": port.host_summary_stats(knowledge),
        "last_refresh_at": last_at,
        "last_refresh_took_s": (round(last_took, 3)
                                if last_took is not None else None),
        "members_dropped_last": last_dropped,
        "note": (out.get("note") or
                 "主题层是 L1 的派生层；代表句是抽取式摘录（离质心最近的几条原文），不是归纳"),
    })
    return out
