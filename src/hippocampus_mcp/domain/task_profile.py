# -*- coding: utf-8 -*-
"""Task-level "question profile" projection (extracted from brain/core.py).

Pure derivation; no retrieval, no models. The anchor is the ``task_hits``
session_start **already computed** (pre-recall ran once; this never
re-searches -- the same hot-path discipline family as the overview card):

* **Knowledge points**: intersect topic ``member_ids`` with anchor memory
  ids, sorted by intersection size. Same id space, exact, no tunable
  threshold;
* **Entities**: intersect entity ``memory_ids`` with the anchor, ties broken
  by mention_count;
* **Open intents**: intents with ``status=open`` whose text contains any word
  token (same semantic family as ``resolve_intent``'s keyword-in-text);
* **Miss gaps**: traces within the last ``miss_days`` days whose query
  contains any word token -- "what this domain has repeatedly lacked
  recently"; longer than the status card's 7-day window.

token = task text word-split + anchor tag values + anchor entity names (the
latter two come from the anchors themselves, not a new retrieval). **An empty
(or whitespace-only) task returns an entirely empty profile** -- a "question
profile" presupposes a question to be solved, the same rule as session_start's
``bool(task)`` gate (the two tracks never split: block-empty <=> structurally
empty).

Why domain/: this only does "derived-layer data -> structured projection";
word splitting (``task_tokens``) already lives in common; zero I/O, zero
models, zero config reads. brain keeps only a thin wrapper that binds the four
derived layers, the reference clock, the topic-layer switch, and the five cap
constants at call time and injects them.

🔴 **The topic semantics go through injection, not import** of
``domain.topic``: arch G1's dependency matrix is ``domain -> common only``,
and **same-layer mutual imports are not among the allowed relations** (the
explicit same-layer whitelists like ``store`` / ``map`` are specially
approved). So ``ordered`` and the archived status value are passed in by the
wrapper -- which incidentally preserves the monkeypatch surface (tests
patching ``topic_mod.STATUS_ARCHIVED`` keep working).

Port discipline:

* **No Config import and no env reads**. The topic-layer switch
  ``topic_layer_enabled`` is fetched live by the wrapper from
  ``_topic_defaults()`` and passed in (lesson: making it an import-time
  snapshot would let ``HIPPOCAMPUS_TOPIC_LAYER=0`` silently stop working);
* The five caps (``TASK_TOPIC_MAX`` / ``TASK_ENTITY_MAX`` /
  ``TASK_INTENT_MAX`` / ``TASK_MISS_MAX`` / ``TASK_MISS_DAYS``) are packed
  into :class:`Limits`, fetched live by the wrapper from the core namespace
  **at call time** (arch freezes the constant names in core; monkeypatch
  surface);
* The clock falls back to ``_now()`` only when ``ref_time`` is None
  (``manager._reference_time`` is the test-injected reference clock, same
  origin as this project's dual-clock discipline).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..common.constants import _now, task_tokens


@dataclass(frozen=True)
class Limits:
    """The profile card's four truncation caps + the miss time window (all
    injected, fetched live by the caller)."""

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
    """Structural projection of what the task touches (design doc §6)."""
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

    # ── Knowledge points: member_ids ∩ anchor ──
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

    # ── Entities: memory_ids ∩ anchor ──
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

    # ── Matching tokens: task words + anchor tag values + anchor entity names ──
    tag_values: list[str] = []
    for h in hits:
        for v in (h.get("tags") or {}).values():
            vals = v if isinstance(v, (list, tuple)) else [v]
            tag_values.extend(str(x) for x in vals if x is not None)
    tokens = task_tokens(task, tag_values + [e["name"] for e in ent_rows])

    # ── Open intents: open + text contains a token ──
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

    # ── Miss gaps: within the time window + query contains a token ──
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
        "tokens": tokens[:12],  # truncation is for echo readability only; matching uses the full set
    }
