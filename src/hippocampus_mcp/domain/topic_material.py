"""Four pure-logic pieces of the knowledge_topic single-topic brief branch:
limit clamping / material projection / hint / echo body.

Difference from ``knowledge_brief`` (the one-line knowledge map slot of the
opening pack): that one is "the whole layer compressed into a few lines",
this one is "a single topic expanded into a material list" -- both serve
brief-style reads, but with different granularity and callers.

Zero imports (not even domain same-layer cross-references -- the clip
constant and summary_state callback are injected by the shell, not wired
straight to domain/topic). The wording and field conventions are verbatim
identical to the original in-brain-shell implementation, including two
deliberately kept quirks:
  * ``clamp_limit``: ``int(limit or default)`` -- a falsy limit (0/None/"")
    takes default rather than the clamp lower bound;
  * ``build_members``: ``truncated`` uses strictly greater-than (exactly
    equal to clip does not count as truncated).
"""
from __future__ import annotations

from typing import Any


def clamp_limit(limit: Any, *, default: int = 30, lo: int = 1,
                hi: int = 200) -> int:
    """Material count normalization: ``int(limit or default)`` then clamped
    to [lo, hi].

    Quirk replicated: ``limit or default`` makes 0/None/"" all take default
    (not lo); only negatives and over-the-cap values are truly clamped.
    Non-numeric strings raise ValueError (same as the original).
    """
    n = int(limit or default)
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


def build_members(by_id: dict[str, dict[str, Any]], picked: list[str],
                  *, clip: int) -> list[dict[str, Any]]:
    """Project the memories picked by representative_ids into material
    entries.

    Content is first whitespace-normalized (``" ".join(str(...).split())``)
    then truncated to clip; ``truncated`` is strictly greater-than (exactly
    clip length does not count as truncated); entities carry only the first
    3; timestamp passed through verbatim (None when the key is missing).
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
    """The brief note's four-way selection (top to bottom by priority)."""
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
    """Brief echo body assembly (eleven keys). topic projection: title
    defaults to "(未命名)", entities/tags carry only the first 3,
    size/member_total use the passed-in alive counts, shown is computed
    from len(members)."""
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
