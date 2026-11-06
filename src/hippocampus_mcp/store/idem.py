# -*- coding: utf-8 -*-
"""Idempotency index: an in-process inverted map of ``idem_key -> memory_id``.

**Why it exists**: the orchestration layer (Router + multiple Workers) will
retry -- network blips, effect-queue backoff replays, Tickets re-run after
resume. The current write path only dedups by **similarity**
(``dedup_threshold``), which answers "do the contents look alike", not "is
this the same request":

* after a similarity merge the id changes (folded into the old card), so a
  replay cannot get a stable receipt;
* when two sentences score 0.84 < 0.85 similarity, each gets its own card,
  and every replay adds one more entry to the library;
* worse, "the same event" and "two events worded alike" are
  indistinguishable in similarity space.

The real need is an **exact key**: the host says "this is retry N" and the
server returns the first attempt's id.

Design decisions
----------------
1. **The index is a derivative, not a new layer on disk**. ``idem_key``
   persists with the record (the sparse ``MemoryRecord.idem`` field); after
   process start the index is rebuilt by scanning the memory records **the
   first time it is used** -- no new sidecar file (every extra file is one
   more inconsistency surface; see the carnage documented in
   ``common/sidecar_lock``).
2. **Keys normalize whitespace and length, not content**. Treating
   ``"a" -> " a "`` as the same key is intentional on the host's part (an
   accidental extra space in a retry should not count as a new request);
   case is **not** normalized -- Base62 random strings are case-sensitive,
   and wrongly normalizing = two different requests swallowed as one.
3. **A hit means no write**. Write only when lookup misses; the writer is
   responsible for registering the new id (``remember``). **No history**:
   we keep only "which memory this key finally landed on", not the call
   count (that is the ledger's business).
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["MAX_KEY_CHARS", "IdemIndex", "normalize_key"]

#: Idempotency key length cap. An over-long key is almost certainly the host
#: stuffing an entire JSON blob in (misuse); after truncation it still works
#: (the prefix is distinctive enough), but each memory does not carry extra KBs.
MAX_KEY_CHARS = 200


def normalize_key(key: Any) -> str:
    """Normalize an idempotency key: strip both ends + collapse internal whitespace runs to one space + truncate.

    Empty string / None always returns ``""``; the caller treats that as
    "no idempotency" (**never** treat the empty string as a valid key --
    that would make all keyless writes hit each other).
    """
    if key is None:
        return ""
    text = str(key).strip()
    if not text:
        return ""
    # Internal whitespace collapse: an extra newline in a replay should not become a new request
    text = " ".join(text.split())
    return text[:MAX_KEY_CHARS]


class IdemIndex:
    """A thin dict of ``idem_key -> memory_id`` (no I/O, no locks).

    Thread safety is guaranteed by the caller's (``MemoryManager``)
    write-path lock -- this class only does mapping and does not guess a
    concurrency model. Under the daemon, N connections share one manager
    instance, and that lock already serializes writes anyway.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._map)

    def rebuild(self, records: Iterable[Any]) -> int:
        """Rebuild the index from memory records, returning the number registered.

        When the same key appears on multiple records (the merge path can
        leave such history), **take the one with the newest timestamp** --
        it is the closest truth to "where this request finally ended up".
        """
        self._map.clear()
        best: dict[str, float] = {}
        for rec in records:
            key = normalize_key(getattr(rec, "idem", ""))
            if not key:
                continue
            ts = float(getattr(rec, "timestamp", 0.0) or 0.0)
            if key in best and best[key] >= ts:
                continue
            best[key] = ts
            self._map[key] = getattr(rec, "id", "")
        return len(self._map)

    def lookup(self, key: Any) -> str | None:
        """Look up an existing memory id by key (returns ``None`` on miss)."""
        norm = normalize_key(key)
        if not norm:
            return None
        return self._map.get(norm)

    def remember(self, key: Any, memory_id: str) -> bool:
        """Register one mapping. **First come, first served**: an existing key is not overwritten; returns False.

        Why no overwrite: a second write under the same key should not
        happen in the first place (``lookup`` would have blocked it);
        anything reaching here is the concurrent window of "wasn't there at
        lookup time", and the id that landed first is the consistent
        receipt a replay should get.
        """
        norm = normalize_key(key)
        if not norm or not memory_id:
            return False
        if norm in self._map:
            return False
        self._map[norm] = memory_id
        return True

    def forget(self, key: Any) -> bool:
        """Delete one mapping (called after a memory is truly deleted by ``forget()``, preventing dangling receipts)."""
        norm = normalize_key(key)
        if not norm:
            return False
        return self._map.pop(norm, None) is not None

    def forget_memory(self, memory_id: str) -> int:
        """Delete all mappings pointing at one memory, returning how many were removed."""
        if not memory_id:
            return 0
        dead = [k for k, v in self._map.items() if v == memory_id]
        for key in dead:
            del self._map[key]
        return len(dead)
