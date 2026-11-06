# -*- coding: utf-8 -*-
"""The per-sentence write loop of ingest (extracted inline from
``brain.core.ingest``).

**Why it is worth moving**: this is the largest piece of inline orchestration in
the ingest body (about 50 lines) -- it calls ``manager.store_memory`` per
sentence, then reads three **side channels** (``last_store_near_miss`` /
``last_truncation`` / ``last_store_merge_of``) to decide what to echo back, and
finally merges newly attached entity ids back into the memory as needed. It is
tangled with the L1 storage port and brain's entity-linking entry point, and
inside the 6299-line core.py it cannot be unit-tested apart from the whole
brain.

Guarantees that the move is **zero behavior change**:
1. The loop body is copied verbatim, with only ``self.`` replaced by injected
   ports fields;
2. The ``memory_type`` decision (``cognitive_type == EXPERIENCE``) is a loop
   invariant, computed by the wrapper and passed in -- ``CognitiveType`` lives
   in the store package, invisible to domain;
3. The parity probe (``_probe_out/m5_batch19_parity.py``) runs the HEAD old
   ingest and the new ingest against the same seeded library and compares the
   complete response bodies field by field; DRIFT must be 0.

Dependency direction: this module imports only the standard library (the
"domain -> common only" discipline; here not even common is used -- everything
is injected via :class:`SentenceWritePorts`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SentenceWritePorts:
    """All dependencies of the per-sentence write loop (injected by the brain
    side; domain imports no upper layer).

    Attributes:
        manager: the L1 storage port. Uses six surfaces: ``store_memory`` /
            ``last_store_near_miss`` / ``last_truncation`` /
            ``last_store_merge_of`` / ``store.get`` / ``store.update``.
        link_entities: brain's ``_link_entities(names, memory_id, ts, origins)``.
        merge_ids: ``store.memory.merge_id_lists`` (ordered union).
        guess_importance: brain's ``_guess_importance(text)`` (delegates to
            ``domain.policies.guess_importance``).
    """
    manager: Any
    link_entities: Callable[..., list[str]]
    merge_ids: Callable[..., list[str]]
    guess_importance: Callable[[str], float]
    # D2a write-side isolation (default False = status quo): when True, every L1
    # card in this batch carries the quarantine marker and **does not
    # participate in dedup merging** (see manager.store_memory).
    # sensitivity is one label shared by the whole batch (the result of
    # classifying the ingest source text once; re-classifying per sentence is
    # over-engineering -- quarantined cards get manual review anyway, better
    # safe than sorry).
    quarantine: bool = False
    sensitivity: str = ""
    # P0 (Super-agent adaptation):
    #   origin  = the write credential {session/task/surface}, shared by the
    #             whole batch (one tool call is one credential unit;
    #             recomputing per sentence is meaningless);
    #   idem_key= the idempotency key, shared by the whole batch -- the
    #             **first sentence** of the L1 cards split from multiple
    #             sentences takes the key, so a replay is stopped at the
    #             entrance (see brain.ingest's idempotency precheck).
    # With both at their defaults, the record side sees zero change (origin
    # sparse / idem empty string).
    origin: dict[str, str] | None = None
    idem_key: str = ""


@dataclass(frozen=True)
class SentenceWriteResult:
    """The four outputs of one per-sentence write loop (raw material for ingest's echo body)."""
    memory_ids: list[str] = field(default_factory=list)
    merged_into: list[dict[str, Any]] = field(default_factory=list)
    kept_separate: list[dict[str, Any]] = field(default_factory=list)
    truncations: list[dict[str, Any]] = field(default_factory=list)


def write_sentences(
    ports: SentenceWritePorts,
    *,
    sentences: list[str],
    importance: float,
    explicit_importance: bool,
    memory_type: str,
    cognitive_type: str,
    timestamp: float,
    raw_id: str,
    entity_ids: list[str],
    source: str,
    tags: dict[str, Any] | None,
    names: list[str],
    origins: dict[str, str],
) -> SentenceWriteResult:
    """Write into L1 sentence by sentence, collecting echo material from the
    side channels.

    Behavior is verbatim identical to the inline loop in ``brain.core.ingest``
    (including three preserved semantics):

    * importance is judged **per sentence** (when passed explicitly, the whole
      batch shares the block-level value);
    * non-empty ``last_store_merge_of`` = this content was merged into an
      existing memory (not newly created); record it in ``merged_into``;
    * newly attached entities are **unioned** with the memory's existing
      ``entity_ids`` (defect D3: overwriting directly would drop previously
      accumulated entity associations when a dedup merge hits).
    """
    manager = ports.manager
    memory_ids: list[str] = []
    merged_into: list[dict[str, Any]] = []
    # C10 (plan B): those near-duplicates that hit dedup but carry new hard
    # information and were **not merged**
    kept_separate: list[dict[str, Any]] = []
    # C13: the parts of this batch truncated by the embedding model's token cap
    truncations: list[dict[str, Any]] = []
    for sent in sentences:
        # Importance is judged **per sentence**, not shared by the whole block
        # (coverage report, high 5): block-level judgment would let one sentence
        # saying "important!" inside the block lift dozens of sentences into the
        # high band, some of which (>=0.7) would then be auto-pinned, exempted
        # from time decay, and fed into the L3 profile.
        sent_importance = (
            importance if explicit_importance else ports.guess_importance(sent)
        )
        mid = manager.store_memory(
            content=sent,
            importance=sent_importance,
            memory_type=memory_type,
            timestamp=timestamp,
            cognitive_type=cognitive_type,
            quarantine=ports.quarantine,
            sensitivity=ports.sensitivity,
            source_ids=[raw_id],
            entity_ids=list(entity_ids),
            source=source or "",
            tags=tags,
            origin=ports.origin,
            idem_key=ports.idem_key,
        )
        memory_ids.append(mid)
        if manager.last_store_near_miss:
            # C10: this one closely resembles an existing memory, but because it
            # carries numbers/dates that memory has not seen, plan B **did not
            # merge** -- this mid is a genuinely new record.
            kept_separate.append(
                dict(manager.last_store_near_miss, new_memory_id=mid)
            )
        _tcov = manager.last_truncation or {}
        if _tcov.get("truncated"):
            # C13: it was written, but the source text beyond embedded_chars is **unretrievable**.
            truncations.append(dict(_tcov, memory_id=mid))
        if manager.last_store_merge_of:
            # Dedup hit: this content did not create a new entry but merged into
            # that existing memory. In dual-write scenarios this is the easiest
            # trap -- the host believes a new record was written, while the old
            # wording is still in place (content takes the longer, entities take
            # the union).
            rec = manager.store.get(mid)
            merged_into.append({
                "memory_id": mid,
                "content": (rec.content[:200] if rec else ""),
                "source": (rec.source or "") if rec else "",
                "note": "This content was merged into that existing memory (dedup threshold hit; no new record)",
            })
        eids = ports.link_entities(names, mid, timestamp, origins)
        if eids:
            rec = manager.store.get(mid)
            if rec is not None:
                # Union, not overwrite (defect D3): when a dedup merge hits, mid
                # is an **existing** memory; overwriting directly would drop all
                # of its previously accumulated entity associations, and the
                # entity inverted index breaks with it (_recompute_entity_stats
                # trusts the record's entity_ids and would thereupon judge the
                # entities dangling and remove them).
                merged_eids = ports.merge_ids(rec.entity_ids, eids)
                if merged_eids != list(rec.entity_ids or []):
                    manager.store.update(mid, entity_ids=merged_eids)
    return SentenceWriteResult(
        memory_ids=memory_ids,
        merged_into=merged_into,
        kept_separate=kept_separate,
        truncations=truncations,
    )
