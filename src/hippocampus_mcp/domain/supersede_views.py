"""Read-side view family for coexisting versions: supersede list / false-kill recovery / tag coverage.

Three pieces moved wholesale from ``brain.core``. Same names as
``store/supersede.py`` (the write-side supersede-detection state machine) but
different things: these are **read-side / repair-side views**. Zero imports;
store capabilities and state constants are injected by the shell, **fetched
live at call time**.

Semantics preserved verbatim (changing any one is a behavior change; parity
will catch it):

  * list_superseded: lists only ``state == superseded``; filters on
    ``source`` only when it is not None (exact equality, with the record
    side's ``or ""`` fallback); falls back to an empty string when
    ``superseded_by_content`` cannot be found; sorts by timestamp descending
    then ``[:limit]`` — **limit<=0 has no special semantics** (``[:0]`` is
    empty, ``[:-1]`` truly truncates the tail); the slicing quirks are kept
    as-is;
  * un_supersede: none of the three scoping options given ⇒ return error
    instead of touching the library; falsy ``memory_ids`` (including an
    empty list) counts as "unspecified"; previews ``content[:80]``; restore
    = ``state→active`` + ``superseded_by=None``, **saves only when there
    are targets** (lazy callback; previews do zero writes and zero saves);
  * tag_coverage: falsy ``tags`` counts as no tags; key distribution top 15
    (count descending, ties by key name ascending); ``coverage`` returns
    ``0.0`` for an empty **library** instead of dividing by zero.
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
    """List memories marked superseded (already superseded), including "superseded by whom".

    Use it for audits when batch imports were falsely killed: coverage report
    · high-4 measured 28 pairs, the vast majority being two records' own
    genuine facts judged into an update relation.
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
    all: bool = False,  # noqa: A002 - aligned with the CLI semantics; keep this name
    source: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Restore memories misjudged as "already superseded" to active (clearing superseded_by).

    For false-kill repair (the companion item of coverage report · high-4):
    ``_detect_supersede`` once judged 28 pairs of unrelated/independent facts
    into update relations because the content-word overlap gate was too
    loose. After restoration they rejoin default recall; the original
    superseder is unaffected.

    A scope must be given explicitly (``memory_ids`` / ``all=True`` /
    ``source``), to avoid one command flipping the whole library's state.
    Default ``dry_run=True`` previews only.
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
    """Library-level tag coverage and key distribution (acceptance criterion).

    "How many records carry domain tags" previously could only be counted by
    exporting the whole library yourself — without self-reporting, the host
    could only go on a feeling of "seems like most have them". The key
    distribution returns only the top keys, keeping large libraries from
    blowing up the response body.
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
