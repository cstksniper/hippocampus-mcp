# -*- coding: utf-8 -*-
"""任务级「问题概况」投影（从 brain/core.py 提取）。

纯派生、不检索、不引模型。锚点是 session_start **已经算好的**
``task_hits``（预召回跑过一次，这里绝不重新检索——热路径纪律与全貌卡
同族）：

* **知识点**：主题 ``member_ids`` 与锚点记忆 id 求交，按交集大小排序。
  同一个 id 空间，精确，无阈值可调；
* **实体**：实体 ``memory_ids`` 与锚点求交，同频按 mention_count；
* **待办**：``status=open`` 且文本含任一词 token 的 intent（与
  ``resolve_intent`` 的 keyword-in-text 同族语义）；
* **miss 缺口**：近 ``miss_days`` 天且 query 含任一词 token 的留痕——
  「这个领域最近反复缺什么」，比状态卡的 7 天窗口长。

token = 任务正文切词 + 锚点标签值 + 锚点实体名（后两者来自锚点本身，
不是新检索）。**task 为空（或纯空白）时整体返回空概况**——"问题概况"
presupposes 一个有待解决的问题，与 session_start 的 ``bool(task)`` gate
同一口径（双轨不分裂：块空 ⟺ 结构化空）。

为什么放 domain/：这里只做「派生层数据 → 结构化投影」，切词
（``task_tokens``）本就在 common，零 I/O、零模型、零配置读数。brain 只留
薄包装，把四个派生层、参考时钟、主题层开关与五个上限常量在调用时绑好
注入。

🔴 **主题口径走注入，不 import** ``domain.topic``：arch G1 的依赖矩阵是
``domain → 仅 common``，**同层互引都不在允许关系里**（``store`` / ``map``
那种显式同层白名单是特批）。所以 ``ordered`` 与归档状态值由包装层传进来
——顺带保住了 monkeypatch 面（测试 patch ``topic_mod.STATUS_ARCHIVED``
仍然生效）。

端口纪律：

* **不 import Config 也不读 env**。主题层开关 ``topic_layer_enabled`` 由
  包装层从 ``_topic_defaults()`` 现取后传入（教训：做成 import-time
  快照会让 ``HIPPOCAMPUS_TOPIC_LAYER=0`` 静默失效）；
* 五个上限（``TASK_TOPIC_MAX`` / ``TASK_ENTITY_MAX`` / ``TASK_INTENT_MAX``
  / ``TASK_MISS_MAX`` / ``TASK_MISS_DAYS``）打包成 :class:`Limits` 由包装层
  **调用时**从 core 命名空间现取（arch 冻结常量名在 core，monkeypatch 面）；
* 时钟 ``ref_time`` 为 None 时才回落 ``_now()``（``manager._reference_time``
  是测试注入的参考钟，与本项目双钟纪律同源）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..common.constants import _now, task_tokens


@dataclass(frozen=True)
class Limits:
    """概况卡的四档截断 + miss 时间窗（全部由调用方现取注入）。"""

    topic_max: int
    entity_max: int
    intent_max: int
    miss_max: int
    miss_days: int


def task_profile(
    task: str,
    task_hits: list[dict[str, Any]] | None,
    *,
    knowledge: Any,
    entities: Mapping[str, Any],
    intents: Sequence[Any],
    miss_log: Sequence[Any],
    ref_time: float | None,
    topic_layer_enabled: bool,
    limits: Limits,
    ordered_topics: Callable[[Any], list[dict[str, Any]]],
    archived_status: str,
    now: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """任务触及的结构投影（设计文档 §6）。"""
    if not (task or "").strip():
        return {
            "task": "", "topics": [], "topics_total": 0,
            "entities": [], "entities_total": 0,
            "intents": [], "intents_total": 0,
            "misses": [], "misses_total": 0,
            "miss_days": limits.miss_days, "anchor_hits": 0, "tokens": [],
        }
    hits = [h for h in (task_hits or []) if h.get("memory_id")]
    anchor = {h["memory_id"] for h in hits}

    # ── 知识点：member_ids ∩ 锚点 ──
    topics: list[dict[str, Any]] = []
    if topic_layer_enabled:
        for t in ordered_topics(knowledge):
            if str(t.get("status") or "") == archived_status:
                continue
            members = t.get("member_ids") or []
            inter = anchor & set(members)
            if not inter:
                continue
            topics.append({
                "topic_id": str(t.get("id") or ""),
                "title": str(t.get("title") or "(untitled)"),
                "size": len(members),
                "hits": len(inter),
            })
    topics.sort(key=lambda x: (-x["hits"], x["title"]))

    # ── 实体：memory_ids ∩ 锚点 ──
    ent_rows: list[dict[str, Any]] = []
    for ent in entities.values():
        mem = ent.get("memory_ids") or []
        inter = anchor & set(mem)
        if not inter:
            continue
        ent_rows.append({
            "name": str(ent.get("name") or ent.get("key") or ""),
            "hits": len(inter),
            "mentions": len(mem),
        })
    ent_rows.sort(key=lambda x: (-x["hits"], -x["mentions"], x["name"]))

    # ── 匹配 token：任务词 + 锚点标签值 + 锚点实体名 ──
    tag_values: list[str] = []
    for h in hits:
        for v in (h.get("tags") or {}).values():
            vals = v if isinstance(v, (list, tuple)) else [v]
            tag_values.extend(str(x) for x in vals if x is not None)
    tokens = task_tokens(task, tag_values + [e["name"] for e in ent_rows])

    # ── 待办：开放 + 文本含 token ──
    intent_rows: list[dict[str, Any]] = []
    for it in intents:
        if str(it.get("status") or "") != "open":
            continue
        text = str(it.get("text") or "")
        low = text.lower()
        if any(tok in low for tok in tokens):
            intent_rows.append({
                "id": str(it.get("id") or ""),
                "text": text,
                "due": str(it.get("due") or ""),
            })

    # ── miss 缺口：时间窗内 + query 含 token ──
    cutoff = (ref_time if ref_time is not None
              else (now or _now)()) - limits.miss_days * 86400
    miss_rows: list[dict[str, Any]] = []
    for m in miss_log:
        if float(m.get("ts") or 0.0) < cutoff:
            continue
        q = str(m.get("query") or "")
        if not any(tok in q.lower() for tok in tokens):
            continue
        miss_rows.append({
            "query": q,
            "ts": float(m.get("ts") or 0.0),
            "top_similarity": float(m.get("top_similarity") or 0.0),
        })
    miss_rows.sort(key=lambda x: -x["ts"])

    return {
        "task": task or "",
        "topics": topics[:limits.topic_max],
        "topics_total": len(topics),
        "entities": ent_rows[:limits.entity_max],
        "entities_total": len(ent_rows),
        "intents": intent_rows[:limits.intent_max],
        "intents_total": len(intent_rows),
        "misses": miss_rows[:limits.miss_max],
        "misses_total": len(miss_rows),
        "miss_days": limits.miss_days,
        "anchor_hits": len(anchor),
        "tokens": tokens[:12],  # 截断只为回显可读，匹配用的是全量
    }
