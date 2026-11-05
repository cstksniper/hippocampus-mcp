# -*- coding: utf-8 -*-
"""提升候选判据：稳定期限 + 去个案化 + 敏感标注。

提升流决策语义的纯函数落点（domain 只准依赖 common；本模块
**连 sensitive 都不 import**——同层禁互引，敏感匹配经 ``sensitive_fn``
回调由服务层注入，词表内容与模式都在服务层装配）：

* 稳定期限：记录在 ≥ ``min_sessions`` 个**不同会话**中出现。
  "出现"的口径：该记录 id 在某条 L2 摘要的 ``memory_ids`` 回指里
  （ingest 落的账），**或**该记录所属实体名出现在摘要正文里
  "记忆/摘要提到同一实体或主题"）。会话 id 以 ``summaries`` 条目的
  ``session_id`` 为准（空 id 的摘要不计——没有会话归属就没法数"不同"）。
* 去个案化：**只标注不硬拦**——候选带 ``project_entities``
  （该记录命中的本项目实体名），改写由人。
* 敏感：``sensitive_fn(content)`` 命中 → ``blocked=True`` +
  ``sensitive_hits`` 详情（可改写后重提）；带 ``敏感=true`` 标签的记录
  **永不进候选**（服务层与判据层双重尊重）。
* 写入绝不自动：本函数只产候选清单；写入走
  ``store_memory(library="user")`` 的人工两步（改写 + 显式调用），
  source 由人按约定填 ``promoted:<源库>:<memory_id>``。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

__all__ = ["collect_promote_candidates"]

#: "打了这个结构化标签的记录永不进提升候选"（约定键）。
SENSITIVE_TAG_KEY = "敏感"


def collect_promote_candidates(
    records: Iterable[Any],
    *,
    summaries: Iterable[dict[str, Any]],
    entity_members: dict[str, Iterable[str]],
    min_sessions: int,
    sensitive_fn: Callable[[str], list[dict[str, Any]]],
    active_state: str = "active",
) -> list[dict[str, Any]]:
    """从项目库记录里收集"值得提升为用户级记忆"的候选。

    Args:
        records: 库存记录（鸭子类型：``.id/.content/.state/.tags/
            .importance/.timestamp``）。
        summaries: L2 摘要（``{session_id, text, memory_ids}``；其它键
            忽略）。``session_id`` 为空的条目不参与会话统计。
        entity_members: 本项目实体倒排 ``{实体名: [memory_id, ...]}``，
            由服务层从 brain 实体表现取组装（实体表是每库一份的
            I/O，domain 不碰）。
        min_sessions: 稳定期限阈值（Config 注入）。``<=1`` 时判据
            恒真（每条 active 记录都是候选）——部署自己拍这个值。
        sensitive_fn: ``(content) -> hits``，服务层闭包（内置
            :func:`domain.sensitive.match_sensitive` + env 词表内容）。
        active_state: 活跃状态字符串（调用方注入，承 U1 同款口径）。

    Returns:
        候选清单，``(-sessions_seen, -importance, -timestamp)`` 降序。
        每条：``memory_id / content / importance / timestamp /
        sessions（有序去重）/ sessions_seen / project_entities /
        sensitive_hits / blocked``。
    """
    # 会话统计视图（一次算好，per-record 复用）：
    #   ids_sessions[id] = {session_id, ...}  按 memory_ids 回指
    #   text_by_session = [(session_id, text), ...]  按实体名正文匹配
    ids_sessions: dict[str, set[str]] = {}
    session_texts: list[tuple[str, str]] = []
    for s in summaries or ():
        sid = str(s.get("session_id") or "").strip()
        if not sid:
            continue
        for mid in s.get("memory_ids") or ():
            ids_sessions.setdefault(mid, set()).add(sid)
        session_texts.append((sid, str(s.get("text") or "")))

    members: dict[str, list[str]] = {
        name: list(ids or ()) for name, ids in (entity_members or {}).items()
    }
    out: list[dict[str, Any]] = []
    for r in records or ():
        if r.state != active_state:
            continue
        tags = r.tags or {}
        if isinstance(tags, dict) and tags.get(SENSITIVE_TAG_KEY):
            continue  # 打了敏感标记的永不进候选
        sessions = set(ids_sessions.get(r.id, ()))
        ents = [name for name, ids in members.items() if r.id in ids]
        if ents:
            ent_set = set(ents)
            for sid, text in session_texts:
                if any(name in text for name in ent_set):
                    sessions.add(sid)
        if len(sessions) < max(int(min_sessions), 0):
            continue
        hits = sensitive_fn(r.content or "")
        out.append({
            "memory_id": r.id,
            "content": r.content,
            "importance": r.importance,
            "timestamp": r.timestamp,
            "sessions": sorted(sessions),
            "sessions_seen": len(sessions),
            "project_entities": ents,
            "sensitive_hits": hits,
            "blocked": bool(hits),
        })
    out.sort(key=lambda c: (-c["sessions_seen"], -c["importance"],
                            -c["timestamp"]))
    return out
