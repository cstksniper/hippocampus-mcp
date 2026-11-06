"""Write-side pure checks: near-duplicate retrieval (the decision core of
``brain.core._near_duplicates``).

This module's position is one of the landing spots for "criterion/check-class
pure logic". The remaining criteria of the same family already have their own
homes from earlier -- do **not** move them here (to avoid collisions):

  * the ``is_restatement`` decision core -> ``domain/policies.py``;
  * write-side supersede detection -> ``store/supersede.py``;
  * read-side conflict annotation -> ``domain/conflict_annotate.py``;
  * contradiction candidate filtering -> ``store/contradictions.py``.

Zero imports (everything needed is injected by the shell). ``store`` /
``vector_of`` are used duck-typed: the former needs ``get(mid)`` and
``search(q, top_k=...)`` (the latter yields ``(record, sim)`` pairs), the
latter maps a record to a query vector (may be ``None`` meaning "no vector").

Verbatim-preserved quirks (changing any one is a behavior change; the parity
probe will catch it):
  * the candidate pool request width is ``top_k + len(own) + 2`` -- ask for a
    bit more to offset the seats taken by "this batch of our own", so existing
    memories can still fill up ``top_k``;
  * the similarity filter is strict less-than (skip only when ``sim <
    min_sim``; exactly equal to the threshold **counts** as a hit);
  * when one existing memory is hit by several new memories, **keep the hit
    with the highest similarity** (first come first served, replaced only by a
    higher one);
  * ``similarity`` rounds to 3 decimal places, ``content`` is cut to the
    first 200 characters, ``source`` defaults to an empty string;
  * truncate only when ``top_k > 0``; ``top_k <= 0`` returns everything
    (negative = unlimited).
"""
from __future__ import annotations

from typing import Any


def near_duplicates(
    memory_ids: list[str],
    *,
    store: Any,
    vector_of: Any,
    top_k: int,
    min_sim: float,
) -> list[dict[str, Any]]:
    """Find the **existing** memories most similar to "the memories just
    written in this batch".

    Only vector similarity is considered (no contradiction judgment anymore --
    that is consolidate's job), but ``state`` / ``superseded_by`` / ``source``
    are handed over too: already-superseded old versions and synonymous
    phrasings from other files are things the host should see on the spot.
    """
    own = set(memory_ids)
    found: dict[str, dict[str, Any]] = {}
    for mid in memory_ids:
        rec = store.get(mid)
        if rec is None:
            continue
        q = vector_of(rec)
        if q is None:
            continue
        for other, sim in store.search(q, top_k=top_k + len(own) + 2):
            if other.id in own or sim < min_sim:
                continue
            prev = found.get(other.id)
            if prev is None or sim > prev["similarity"]:
                found[other.id] = {
                    "memory_id": other.id,
                    "similarity": round(float(sim), 3),
                    "state": other.state,
                    "superseded_by": other.superseded_by,
                    "source": other.source or "",
                    "content": other.content[:200],
                }
    rows = sorted(found.values(), key=lambda r: -r["similarity"])
    return rows[:top_k] if top_k > 0 else rows
