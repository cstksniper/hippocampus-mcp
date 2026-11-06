"""Write-side supersede detection: a new memory carries an update marker +
is highly similar to an old memory + shares content words -> supersede the
most similar one.

Extracted from ``brain/core.py``'s ``_detect_supersede``.

All three conditions are required: update marker + similarity threshold +
(shared entities OR high content-word overlap), preventing unrelated long
-term facts like "peanut allergy" from being mislabeled as superseded by "now
keeping cats".

Placement rationale: **query composition over stock records + state machine
write-back** -- the same family as ``contradictions.py`` (reusing ``store``'s
``get`` / ``iter_records`` / ``update`` and ``MemoryState``). Constrained by
the dependency matrix, domain cannot touch
``store.memory.MemoryState``, hence it lands on the store side.

Same discipline as above: this module **does not import vector_store /
brain** -- the store instance is injected by the caller (the brain wrapper
passes ``self.manager.store``); the vector query ``vector_of`` and the
criterion ``is_restatement`` are likewise injected **callables** (not call
results):

* ``vector_of`` is only consulted for new memories that "carry an update
  marker"; marker filtering comes first -- never pay the embedding query cost
  for writes without an update marker;
* ``is_restatement`` is not imported in order to preserve the test
  monkeypatch surface on ``brain._is_restatement``, and also so the read side
  (``domain.conflict_annotate``) and the write side share one criterion
  (separately written thresholds would sooner or later contradict each
  other).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..common.constants import UPDATE_MARKERS
from .memory import MemoryState

__all__ = ["detect_supersede"]


def detect_supersede(
    new_memory_ids: list[str],
    ts: float,
    store: Any,
    vector_of: Callable[[Any], Any | None],
    is_restatement: Callable[..., bool],
    update_markers: tuple[str, ...] = UPDATE_MARKERS,
) -> list[dict[str, str]]:
    """Among the active memories stored before ``ts``, find those superseded
    by ``new_memory_ids``.

    Args:
        new_memory_ids: the memory ids newly written this time.
        ts: this write's timestamp; **candidates with ``>= ts`` are always
            skipped** (only older ones are superseded; same-batch writes never
            supersede each other).
        store: the stock record storage (needs ``get`` / ``iter_records`` /
            ``update``).
        vector_of: record -> vector (``None`` means this record has no
            computed vector; skip).
        is_restatement: the same-fact criterion; signature in
            ``domain.policies.is_restatement``.
        update_markers: the update marker lexicon.

    Returns:
        One entry per supersede: ``old_id`` / ``new_id`` / ``sim`` / ``old`` /
        ``new`` (the last two are the first 60 characters of content, for echo
        only).

    Side effect: the superseded record is set to ``state=superseded`` with
    ``superseded_by`` attached -- this is the function's **only** write
    operation; if nothing is found, nothing is written. R2: it also clears
    ``pin``; a superseded card must not keep occupying a pinned slot.

    Preserved behavior (aligned with the original implementation item by
    item; do not "fix it in passing"):

    * only the **most similar** one is superseded (``sim > best_sim`` strict
      inequality => on ties the first encountered wins), to avoid collateral
      damage to unrelated memories;
    * the new memory itself (``rec.id == nid``), non-active records, and those
      with ``timestamp >= ts`` are never candidates;
    * a vectorless new memory is **skipped entirely** (never enters the inner
      loop); a vectorless candidate skips only that candidate.
    """
    out: list[dict[str, str]] = []
    for nid in new_memory_ids:
        new_rec = store.get(nid)
        if new_rec is None:
            continue
        if not any(m in new_rec.content.lower() for m in update_markers):
            continue

        q = vector_of(new_rec)
        if q is None:
            continue
        new_ents = set(new_rec.entity_ids or [])
        best, best_sim = None, 0.0
        for rec in store.iter_records():
            if rec.id == nid or rec.state != MemoryState.ACTIVE.value:
                continue
            if rec.timestamp >= ts:
                continue
            v = vector_of(rec)
            if v is None:
                continue
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1e-12
            sim = float(q @ v / denom)
            # The criterion uniformly goes through is_restatement (the read
            # side's coexistence-conflict annotation uses the same one);
            # writing a separate threshold set on each side would sooner or
            # later contradict itself.
            if sim > best_sim and is_restatement(
                sim, new_rec.content, rec.content, new_ents, rec.entity_ids
            ):
                best, best_sim = rec, sim

        if best is not None:
            store.update(
                best.id,
                state=MemoryState.SUPERSEDED.value,
                superseded_by=nid,
                pin=False,
            )
            out.append({
                "old_id": best.id, "new_id": nid, "sim": round(best_sim, 3),
                "old": best.content[:60], "new": new_rec.content[:60],
            })
    return out
