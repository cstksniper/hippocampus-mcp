# -*- coding: utf-8 -*-
"""Miss-log write flow (extracted from core).

**Why worth moving**: the two phases share one thing — **every deletion must
raise a tombstone** (deletion-ledger discipline) — cap trimming and clearing
are both real deletions; without a tombstone, the stale snapshot still held by
another process gets persisted and merged back ("trimmed then grown back" /
"the cleared-empty was fake"). With this discipline logic separated from entry
construction (brain-state reads), the shell is a single straight line:
"build entry → delegate".

**Dependency direction**: this module has zero imports — the ``miss_log`` list
is passed straight through by the shell (in-place append/del/clear, same-object
semantics as the original); the cap ``max_n`` is fetched fresh by the shell from
its own namespace (``MISS_LOG_MAX``; patching the caller takes effect
immediately); tombstoning is injected via the ``note_deletions`` callback
(brain-side it is ``note_layer_deletion_items``; the deletion-ledger write
surface stays in brain).

**Replicated quirks (changing any one is a behavior change; parity will catch
it)**:

1. Trim criterion ``if max_n and len(...) > max_n`` — ``max_n=0`` = cap off
   (truthiness check, matching the original ``MISS_LOG_MAX`` falsy semantics);
2. Trim amount ``drop_n = len - max_n``; what gets tombstoned is the
   **trimmed head slice** ``miss_log[:drop_n]`` (oldest leaves first); ``del``
   and tombstoning act on the same batch;
3. Tombstone first, delete after — the callback runs before ``del`` (the order
   carries meaning: deleting without the ledger write succeeding is a wasted
   deletion);
4. Clear: an empty list does **not** call ``note_deletions`` (0 tombstones) and
   returns 0; a non-empty list tombstones the whole batch first, then
   ``clear``s, returning the count;
5. Both phases keep the **original return/no-return design**: append returns
   the entry itself (the caller receives the one just written), clear returns
   the count.
"""
from __future__ import annotations

from typing import Any, Callable


def append_miss_entry(
    *,
    miss_log: list[dict[str, Any]],
    entry: dict[str, Any],
    max_n: int,
    note_deletions: Callable[[str, list[dict[str, Any]]], Any],
    ledger_name: str,
) -> dict[str, Any]:
    """Append one miss-log entry + cap trimming (the deletion-must-tombstone discipline).

    Args:
        miss_log: the miss-log list (**mutated in place**, same-object
            semantics as the original).
        entry: the entry built by the shell (ts/truncation/normalization all
            happen in the shell, brain-state reads).
        max_n: the miss-log cap (fetched fresh by the shell from its namespace,
            ``MISS_LOG_MAX``; **0 = cap off**, truthiness check).
        note_deletions: tombstone callback ``(ledger_name, items)`` (deletion
            ledger).
        ledger_name: ledger layer name (``"brain_misslog.json"``).

    Returns:
        The entry just written.
    """
    miss_log.append(entry)
    if max_n and len(miss_log) > max_n:
        # This too is a **real deletion** and must be tombstoned. Without a
        # tombstone, once the other process's stale un-trimmed snapshot gets
        # persisted, these trimmed entries are merged back and the cap becomes
        # decoration ("trimmed then grown back" is the easiest thing to
        # misread as "trimming didn't take effect").
        drop_n = len(miss_log) - max_n
        note_deletions(ledger_name, miss_log[:drop_n])
        del miss_log[:drop_n]
    return entry


def clear_miss_log(
    *,
    miss_log: list[dict[str, Any]],
    note_deletions: Callable[[str, list[dict[str, Any]]], Any],
    ledger_name: str,
) -> int:
    """Clear the miss log (tombstone the whole batch first, then clear).

    Previously a plain ``miss_log.clear()`` only worked when "I am the last
    writer" — once another process's stale snapshot gets persisted, the whole
    batch is merged back, and the "cleared" the user saw was fake. An empty
    list raises no tombstone and clears nothing (0 tombstones, returns 0).

    Returns:
        How many entries were cleared.
    """
    n = len(miss_log)
    if n:
        note_deletions(ledger_name, miss_log)
        miss_log.clear()
    return n
