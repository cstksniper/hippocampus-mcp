"""knowledge_topic 单主题 brief 分支的四段纯逻辑：限额夹取 / 素材投影 /
提示语 / 回显体。

与 ``knowledge_brief``（开场包的一行式知识地图槽）的区别：那个
是「整个层缩成几行」，这个是「单个主题展开成素材列表」——两者都服务
brief 类读取，但粒度与调用方不同。

零 import（domain 同层互引都不行——clip 常量与 summary_state 回调
由壳注入，不直连 domain/topic）。文案与字段口径与 brain 壳内原实现
逐字一致，包括两处刻意保留的怪癖：
  * ``clamp_limit``：``int(limit or default)``——limit 为假值（0/None/""）
    时走 default 而不是夹取下界；
  * ``build_members``：``truncated`` 用严格大于（恰好等于 clip 不算截断）。
"""
from __future__ import annotations

from typing import Any


def clamp_limit(limit: Any, *, default: int = 30, lo: int = 1,
                hi: int = 200) -> int:
    """素材条数归一：``int(limit or default)`` 后夹到 [lo, hi]。

    怪癖复刻：``limit or default`` 让 0/None/"" 都取 default（不是 lo）；
    负值与超上界才真正被夹。非数字字符串会抛 ValueError（与原实现同）。
    """
    n = int(limit or default)
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


def build_members(by_id: dict[str, dict[str, Any]], picked: list[str],
                  *, clip: int) -> list[dict[str, Any]]:
    """把 representative_ids 选中的记忆投影成素材条目。

    内容先做空白归一（``" ".join(str(...).split())``）再按 clip 截断；
    ``truncated`` 是严格大于（恰好 clip 长度不算截断）；entities 只带前
    3 个；timestamp 原样透传（缺键为 None）。
    """
    members: list[dict[str, Any]] = []
    for mid in picked:
        it = by_id[mid]
        txt = " ".join(str(it.get("content") or "").split())
        members.append({
            "id": mid,
            "content": txt[:clip],
            "truncated": len(txt) > clip,
            "timestamp": it.get("timestamp"),
            "entities": list(it.get("entities") or [])[:3],
        })
    return members


def brief_hint(*, has_alive: bool, has_vec: bool,
               has_members: bool) -> str:
    """brief 的 note 四路选择（按优先级从上到下）。"""
    if not has_alive:
        return "none of this topic's members are currently active; check knowledge_map first"
    if not has_vec:
        return ("the library has no vectorized memories yet (the embedding model may not be ready), "
                "so no material can be produced for now; summarize write-back is unaffected")
    if not has_members:
        return "none of this topic's surviving members have vectors yet, so no material can be produced for now"
    return ("Material is ordered by closeness to the centroid (most representative of this topic). "
            "After reading, write the summary back with action=summarize and attach sources via evidence -- "
            "sources must be the member ids listed above; fabricated references are rejected")


def build_brief_payload(*, tid: str, topic: dict[str, Any],
                        alive_n: int, members: list[dict[str, Any]],
                        no_vec: int, auto: bool, hint: str,
                        summary: Any) -> dict[str, Any]:
    """brief 回显体组装（十一键）。topic 投影：title 缺省 "(未命名)"，
    entities/tags 只带前 3 个，size/member_total 用传入的存活计数，
    shown 由 members 长度自算。"""
    return {
        "ok": True, "enabled": True, "action": "brief",
        "topic_id": tid,
        "topic": {
            "id": topic.get("id"),
            "title": topic.get("title") or "(未命名)",
            "status": topic.get("status"),
            "size": alive_n,
            "entities": list(topic.get("entities") or [])[:3],
            "tags": list(topic.get("tags") or [])[:3],
            "span": list(topic.get("span") or []),
        },
        "summary": summary,
        "members": members,
        "member_total": alive_n,
        "no_vector": no_vec,
        "shown": len(members),
        "auto_built": auto,
        "note": hint,
    }
