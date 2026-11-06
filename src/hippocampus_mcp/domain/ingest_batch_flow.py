# -*- coding: utf-8 -*-
"""The three inline logic segments of ``brain.core.ingest_batch`` (extracted
from core).

**Why worth moving**: ``ingest_batch``'s shell had three unrelated
responsibilities tangled together -- per-item argument normalization (~21
lines), per-item result accumulation (~22 lines), response body assembly (~26
lines). They are all pure dict / list operations, yet untestable in core.py
because they read the ``self.items`` loop context. After the move,
``ingest_batch`` keeps only the four things that truly need brain state:
"batch side-effect switches + loop + call :meth:`ingest` + final persist".

**Deliberately left in the shell**: the final persist (``manager.save()`` ->
``_save_l0_only()`` -> ``save_warning`` text -> ``last_save_error`` /
``autosave_failures`` bookkeeping). It depends on five self-facing attributes
at once and carries three side effects; an injection port would be longer than
the original code -- not worth it.

Dependency direction: this module imports only the standard library (the
domain -> common-only discipline: here not even common is used --
``max_seq_length`` / ``l0_pending`` / batch-level defaults are all injected by
the shell).

**Replicated quirks (touching any one is a behavior change; parity will
catch it)**:

1. The capacity check's wording carries the argument values (``{max_items}`` /
   ``{len(items)}``), and both a non-list ``items`` and an over-limit count
   raise ``ValueError`` -- they **do not go into errors**: these two are "the
   caller wrote it wrong", not "one item went bad".
2. When an item lacks the ``tags`` key it falls back to the **batch-level**
   ``default_tags``; even an empty string counts as item-level parsing (an
   empty string makes ``json.loads`` raise, landing in that item's error).
3. The ``entities`` string branch does ``e.strip()``, the list branch does
   ``str(e).strip()`` -- the asymmetry is pre-existing behavior.
4. ``importance`` negativity is checked only for ``int/float`` (``bool`` is an
   int subclass, but ``True < 0`` is false, so behavior is consistent).
5. ``per_item``'s ``source`` and the one fed to ``ingest`` are the results of
   the same expression; this module computes it once and stores it in
   :attr:`ItemPlan.source_str` for reuse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ItemError:
    """An item judged "unwritable" -- recorded into ``errors``, **without
    aborting the batch**.

    Attributes:
        index: the item's index in the input (smoke tests rely on this order).
        error: human-readable reason.
        preview: the item's content preview (``str(it)[:80]`` or ``text[:80]``).
    """
    index: int
    error: str
    preview: str


@dataclass(frozen=True)
class ItemPlan:
    """The normalized result of an item that passed validation.

    Attributes:
        index: same as :attr:`ItemError.index`.
        payload: keyword arguments expanded directly into ``ingest(**payload)``,
            with key order identical word-for-word to the original
            implementation (kwargs ignore order, but parity compares results).
        source_str: the result of ``str(it.get("source") or source or "")``;
            the ``per_item`` summary must reuse the same value.
    """
    index: int
    payload: dict[str, Any]
    source_str: str


@dataclass(frozen=True)
class CollectedRow:
    """The eight outputs of a single ingestion result.

    Corresponds to what the original implementation's eight accumulators --
    ``raw_ids / memory_ids / all_entities / merged_into / kept_separate /
    truncations / superseded_total / per_item`` -- received from **this one
    item**. :func:`assemble_output` stitches them back together horizontally.
    """
    raw_id: str
    memory_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    merged: list[dict[str, Any]] = field(default_factory=list)
    kept: list[dict[str, Any]] = field(default_factory=list)
    truncations: list[dict[str, Any]] = field(default_factory=list)
    superseded_n: int = 0
    item: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- input validation and normalization


def check_capacity(items: Any, *, max_items: int) -> None:
    """The batch entry's capacity gate -- violations raise directly, *not*
    demoted into errors.

    Args:
        items: the list of items to write.
        max_items: per-batch cap (``INGEST_BATCH_MAX`` on the core side, fetched
            at call time and passed in).

    Raises:
        ValueError: ``items`` is not a list; or the count exceeds ``max_items``.
            The wording carries the concrete numbers -- the host can compute
            how many batches it needs just from the message.
    """
    if not isinstance(items, list):
        raise ValueError(f"items must be an array, got {type(items).__name__}")
    if len(items) > max_items:
        raise ValueError(
            f"batch limit is {max_items} items, got {len(items)}; split into multiple calls"
        )


def normalize_item(
    index: int,
    it: Any,
    *,
    default_tags: dict[str, Any] | None,
    default_source: str,
    default_granularity: str,
    default_detect_supersede: bool | None,
    default_client: str | None,
    default_auto_tags: bool,
) -> ItemPlan | ItemError:
    """Normalize one raw input item into ``ingest`` keyword arguments, or judge
    it skippable.

    The three shape corrections that should happen before ingesting all live
    here:

    - ``importance``: negatives are treated as "unspecified" (consistent with
      the MCP side);
    - ``entities``: accepts a list or a comma-separated string, strips each
      entry and drops empties; all other types (including missing) count as
      empty;
    - ``tags``: a missing key falls back to the batch-level default; the string
      form is ``json.loads``-ed in place, and **a parse failure must raise on
      the spot for this item** (collected by the caller's ``except``) -- if the
      host wrote ``tags="{...}"`` but malformed, silently treating it as empty
      would make it believe the tags were attached.

    Args:
        index: item index.
        it: the raw item.
        default_tags / default_source / default_granularity /
            default_detect_supersede / default_client / default_auto_tags:
            batch-level defaults; **item-level fields win**.

    Returns:
        :class:`ItemPlan` means writable; :class:`ItemError` means record into
        ``errors`` and skip (item not a dict / ``text`` missing or blank).

    Raises:
        ValueError: ``tags`` is a string but JSON parsing failed (raises from
            ``json.loads``).
    """
    if not isinstance(it, dict):
        return ItemError(index=index, error="item is not an object (dict)",
                         preview=str(it)[:80])
    text = it.get("text")
    if not isinstance(text, str) or not text.strip():
        return ItemError(index=index, error="text is missing or empty",
                         preview=str(text)[:80])

    imp = it.get("importance")
    if isinstance(imp, (int, float)) and imp < 0:
        imp = None

    ents = it.get("entities")
    if isinstance(ents, str):
        ents = [e.strip() for e in ents.split(",") if e.strip()]
    elif isinstance(ents, list):
        ents = [str(e).strip() for e in ents if str(e).strip()]
    else:
        ents = []

    it_tags = it.get("tags", default_tags)
    if isinstance(it_tags, str):
        # A parse failure must blow up on the spot for this item (the caller's
        # except collects it into errors); it must not silently become empty --
        # the host would believe the tags were attached.
        it_tags = json.loads(it_tags)

    source_str = str(it.get("source") or default_source or "")
    payload: dict[str, Any] = {
        "text": text,
        "speaker": str(it.get("speaker") or "user"),
        "timestamp": it.get("timestamp"),
        "importance": imp,
        "cognitive_type": it.get("cognitive_type"),
        "entities": ents or None,
        "source": source_str,
        "detect_supersede": it.get("detect_supersede", default_detect_supersede),
        "granularity": str(it.get("granularity") or default_granularity),
        "tags": it_tags,
        "auto_tags": bool(it.get("auto_tags", default_auto_tags)),
        "client": it.get("client") or default_client,
    }
    return ItemPlan(index=index, payload=payload, source_str=source_str)


# ---------------------------------------------------------------- result accumulation


def error_preview(it: Any, plan: ItemPlan | None = None) -> str:
    """Compute the preview for an errors entry about "this item blew up", with
    the same value-picking contract as the original implementation.

    Three paths, in decreasing availability:

    1. Normalization already succeeded (``plan`` is :class:`ItemPlan`) -> use
       ``payload["text"]``;
    2. Normalization raised **midway** (typically the item-level ``tags`` JSON
       parse failure, at which point ``plan`` is still None) -> take the
       item's ``text`` directly -- the preview the host sees should still be
       that sentence, not the str of the whole dict;
    3. Theoretically unreachable (non-dict items are already skipped via the
       :class:`ItemError` path) -> fall back to ``str(it)``.

    Path 2 was caught by the parity probe: before extraction, the ``text``
    variable is assigned **before** the try, so the exception handler can
    always reach it; after extraction, ``plan`` only has a value once
    normalize_item returns, so a mid-way raise leaves nothing -- missing this
    detail leads to mistakenly writing ``str(it)``, turning the preview into
    the whole object.

    Args:
        it: the raw item.
        plan: the normalization result (may be None -- meaning normalization
            itself blew up).

    Returns:
        Preview string (at most 80 characters).
    """
    if isinstance(plan, ItemPlan):
        return plan.payload["text"][:80]
    if isinstance(it, dict) and isinstance(it.get("text"), str):
        return it["text"][:80]
    return str(it)[:80]


def collect_item(res: dict[str, Any], *, index: int, source: str) -> CollectedRow:
    """Flatten one ``ingest`` response body into a :class:`CollectedRow`.

    Two details worth calling out:

    - ``merged_into`` / ``kept_separate`` / ``truncation.items`` get
      **stamped with the index** pointing back at the input -- in the batch
      context the host holding an id cannot tell which write produced it.
      ``index`` is added here without mutating the original dict
      (``{**m, "index": index}``).
    - ``truncation`` only collects ``items`` when ``truncated`` is true: when
      false the whole entry is skipped, consistent with the contract "if
      nothing was truncated, do not report".
    """
    merged = [{**m, "index": index} for m in res.get("merged_into", [])]
    kept = [{**k, "index": index} for k in res.get("kept_separate", []) or []]
    _tr = res.get("truncation") or {}
    truncations: list[dict[str, Any]] = []
    if _tr.get("truncated"):
        truncations = [dict(t, index=index) for t in (_tr.get("items") or [])]
    item = {
        "index": index,
        "raw_id": res["raw_id"],
        "memory_ids": res["memory_ids"],
        "n_memories": len(res["memory_ids"]),
        "source": source,
        "superseded": len(res.get("superseded") or []),
        "merged": len(res.get("merged_into") or []),
    }
    return CollectedRow(
        raw_id=res["raw_id"],
        memory_ids=list(res["memory_ids"]),
        entities=list(res["entities"]),
        merged=merged,
        kept=kept,
        truncations=truncations,
        superseded_n=len(res.get("superseded") or []),
        item=item,
    )


# ---------------------------------------------------------------- response body assembly


def assemble_output(
    rows: list[CollectedRow],
    *,
    errors: list[dict[str, Any]],
    total_input: int,
    max_seq_length: int,
    l0_pending: int,
) -> dict[str, Any]:
    """Stitch several :class:`CollectedRow`s horizontally into ``ingest_batch``'s
    echo body.

    The key order is deliberate: **merge warnings come first** --
    ``merged_into`` / ``kept_separate`` / ``truncation`` are placed before
    ``count``, because in the batch-write scenario "how many entries did not
    get stored individually" deserves to be seen before "how many were written
    in total".

    Args:
        rows: successful items' outputs, in input order.
        errors: records of skipped items (``[{"index", "error", "preview"}]``).
        total_input: total number of input items (**including** skipped ones --
            the host reconciles it against count).
        max_seq_length: the embedding model's sequence cap
            (``embed_max_seq_length``; domain cannot reach engine, injected by
            the shell).
        l0_pending: number of L0 entries not yet persisted (also injected;
            always self-reports the outstanding debt).

    Returns:
        A fifteen-key echo body. ``saved`` / ``save_warning`` are in their
        initial form here; the final-persist branch mutates those two keys in
        place -- that part stays in the shell.
    """
    merged_into = [x for r in rows for x in r.merged]
    kept_separate = [x for r in rows for x in r.kept]
    truncations = [x for r in rows for x in r.truncations]
    return {
        # Same as ingest: **merge warnings come first**; in a batch write,
        # "how many entries did not get stored individually" deserves to be
        # seen before "how many were written in total".
        "merged_into": merged_into,
        # Near-duplicates not merged because of new hard info (non-empty ==
        # these were newly created)
        "kept_separate": kept_separate,
        # Truncation self-report, always present
        "truncation": {
            "truncated": bool(truncations),
            "max_seq_length": max_seq_length,
            "items": truncations,
        },
        "count": len(rows),
        "failed": len(errors),
        "total_input": total_input,
        "raw_ids": [r.raw_id for r in rows],
        "memory_ids": [x for r in rows for x in r.memory_ids],
        "entities": list(dict.fromkeys(x for r in rows for x in r.entities)),
        "superseded": sum(r.superseded_n for r in rows),
        "items": [r.item for r in rows],
        "errors": errors,
        "saved": False,
        "save_warning": "",
        # The batch itself does not persist L0; the finalization persists once
        # for all. This always self-reports the outstanding count.
        "l0_pending": l0_pending,
    }
