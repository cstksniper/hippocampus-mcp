# -*- coding: utf-8 -*-
"""Promotion candidate criteria: stability tenure + de-casing + sensitive
flagging.

Pure-function home for the promotion-flow decision semantics (domain may
only depend on common; this module **does not even import sensitive** --
same-layer cross-imports are forbidden; sensitive matching is injected
by the service layer via the ``sensitive_fn`` callback; the lexicon
content and patterns are assembled in the service layer):

* Stability tenure: the record appears in >= ``min_sessions``
  **different sessions**. The "appearance" criterion: the record id is
  back-referenced in some L2 summary's ``memory_ids`` (the ledger
  written at ingest), **or** a name of an entity the record belongs to
  appears in the summary text ("memory/summary mentions the same entity
  or topic"). Session ids follow the ``summaries`` entries'
  ``session_id`` (summaries with an empty id do not count -- no session
  attribution, no way to count "different").
* De-casing: **flag only, never hard-block** -- candidates carry
  ``project_entities`` (this project's entity names the record hit);
  rewriting is a human's job.
* sensitive: a ``sensitive_fn(content)`` hit → ``blocked=True`` +
  ``sensitive_hits`` details (can be rewritten and re-submitted);
  records tagged ``sensitive=true`` **never enter candidates**
  (respected doubly, at the service layer and the criteria layer).
* Writes are never automatic: this function only produces a candidate
  list; writing goes through the manual two steps of
  ``store_memory(library="user")`` (rewrite + explicit call), with
  source filled by a human per convention as
  ``promoted:<source library>:<memory_id>``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

__all__ = ["collect_promote_candidates"]

#: "Records carrying this structured tag never enter promotion candidates" (conventional key).
SENSITIVE_TAG_KEY = "sensitive"


def collect_promote_candidates(
    records: Iterable[Any],
    *,
    summaries: Iterable[dict[str, Any]],
    entity_members: dict[str, Iterable[str]],
    min_sessions: int,
    sensitive_fn: Callable[[str], list[dict[str, Any]]],
    active_state: str = "active",
) -> list[dict[str, Any]]:
    """Collect candidates "worth promoting to user-level memory" from
    project-library records.

    Args:
        records: store records (duck-typed: ``.id/.content/.state/.tags/
            .importance/.timestamp``).
        summaries: L2 summaries (``{session_id, text, memory_ids}``;
            other keys ignored). Entries with an empty ``session_id`` do
            not participate in session counting.
        entity_members: this project's entity inverted index
            ``{entity name: [memory_id, ...]}``, assembled by the service
            layer from brain's entity profiles (the entity table is
            per-library I/O; domain does not touch it).
        min_sessions: stability-tenure threshold (injected via Config).
            When ``<=1`` the criterion is always true (every active
            record is a candidate) -- the deployment picks this value
            itself.
        sensitive_fn: ``(content) -> hits``, a service-layer closure
            (built-in :func:`domain.sensitive.match_sensitive` + env
            lexicon content).
        active_state: active-state string (injected by the caller,
            inheriting U1's same convention).

    Returns:
        Candidate list, descending by
        ``(-sessions_seen, -importance, -timestamp)``.
        Each: ``memory_id / content / importance / timestamp /
        sessions (ordered, deduplicated) / sessions_seen /
        project_entities / sensitive_hits / blocked``.
    """
    # Session-counting views (computed once, reused per record):
    #   ids_sessions[id] = {session_id, ...}  via memory_ids back-references
    #   text_by_session = [(session_id, text), ...]  via entity-name text matching
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
            continue  # records flagged sensitive never enter candidates
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
