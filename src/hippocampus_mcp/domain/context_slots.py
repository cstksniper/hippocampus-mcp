# -*- coding: utf-8 -*-
"""5-slot context assembly (modeled on bigeye AttentionFocus) — the orchestration layer.

The product is a text block that can be spliced directly into a system prompt (``text``)
plus structured slots (``blocks``) plus hit details (``hits``):

    block_1 direct hits for the current query  block_2 pending questions/open intents
    block_3 timeline position (recent session summaries)  block_4 related entity state
    block_5 last conclusions / open topics

block_4 has **its own budget** (four injected ``block4_*`` values: at most
``block4_max_entities`` entities, at most ``block4_max_facts_per_entity``
facts each, and the whole block ≤ ``block4_max_chars`` chars), and candidates prioritize
**query-word entities** with the first few hit entities as fallback — **it does not
inflate with ``top_k``** (the main body of zcode D2).
Its budget does not compete with the global ``token_budget``.

``item_max_chars`` = 0 means no truncation (numeric tails of ledger/clause-style long
sentences were often cut off by 200, per user feedback); it only affects the text spliced
into the prompt; the returned ``hits`` are always the full content. When ``filter_tags``
is given, block_4 **skips query-word entity expansion** and only expands within the hit
set (no hits means no entities to expand) — when writing per-chapter, entity state from
other chapters must not leak into the context.

Placement note: the body is pure orchestration (branching + candidate collection + budget
loading); the six brain entry points (recall / pinned_brief / trusted_entity_ids /
extract_entities / entity_facts / render_blocks) are injected as **callables**, and the
four ``BLOCK4_*`` caps are fetched fresh by the caller and passed in (they live in
common/constants; patching the caller's namespace takes effect immediately).
The four data faces (entities / intents / summaries / open_topics) are zero-cost attribute
reads, passed through as-is — in the original implementation they too were read sequentially
within a single call, with no writer during the call
(recall / entity_facts only read them).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

# Empty-slot placeholder rows. Centralized so rendering and confidence
# scoring agree on what is *not* real content.
BLOCK_PLACEHOLDERS = frozenset({
    "(No todos)", "(No session history)",
    "(No related entity state)", "(No open topics)",
})


def memory_context(
    *,
    query: str = "",
    top_k: int = 5,
    token_budget: int = 1200,
    item_max_chars: int = 200,
    filter_tags: dict[str, Any] | None = None,
    recall: Callable[..., list[dict[str, Any]]],
    pinned_brief: Callable[..., list[dict[str, Any]]],
    trusted_entity_ids: Callable[[], set[str]],
    entities: Mapping[str, Any],
    extract_entities: Callable[[str], list[str]],
    entity_facts: Callable[..., list[str]],
    intents: Sequence[Any],
    summaries: Sequence[Any],
    open_topics: Sequence[Any],
    render_blocks: Callable[[dict[str, list[str]], int, int], str],
    block4_max_entities: int,
    block4_max_facts_per_entity: int,
    block4_max_chars: int,
    block4_hit_entity_topn: int,
) -> dict[str, Any]:
    """Assemble the 5-slot context, ready to be spliced into a system prompt.

    Args:
        query: the query text; when empty, block_1 falls back to the pin brief
            (``pinned_brief``).
        top_k: number of recall entries, passed through to recall (same value on the
            pinned_brief path).
        token_budget: token budget for the whole block (coarse-grained: Chinese is roughly
            2 chars/token).
        item_max_chars: per-item truncation length inside the context, default 200;
            **0 means no truncation**. Only affects ``text``; ``hits`` is always complete.
        filter_tags: structured tag filtering, passed through to block_1's hit retrieval;
            when given, block_4 skips query-word entity expansion.
        recall: recall entry point, called as
            ``recall(query, top_k=..., filter_tags=...)``.
        pinned_brief: pin brief entry point, called as ``pinned_brief(top_k)``.
        trusted_entity_ids: the set of trusted entity ids, fetched fresh at call time.
        entities: the entity table ``{lowercase name: entity dict}`` (passed through
            as-is, read-only during the call).
        extract_entities: entity-name extraction, called as ``extract_entities(query)``.
        entity_facts: entity facts, called as
            ``entity_facts(ent["key"], limit=...)``.
        intents / summaries / open_topics: the open intents / session summaries /
            open topics time series (passed through as-is, read-only during the call).
        render_blocks: block rendering + budget loading, called as
            ``render_blocks(blocks, token_budget, item_max_chars)``.
        block4_max_entities / block4_max_facts_per_entity /
            block4_max_chars / block4_hit_entity_topn: block_4's four
            independent budgets (fetched fresh and injected by the caller).
    """
    hits = (
        recall(query, top_k=top_k, filter_tags=filter_tags)
        if query else pinned_brief(top_k)
    )

    # block_4: related entity state. **Already narrowed** —
    # the old implementation started "from the entities of the **hit memories**", so the
    # larger top_k was, the more entities it dragged in, and this block degenerated into
    # a noise wall. The candidate order is now:
    #   ① entities from the query words (most on-topic);
    #   ② entities of the hit memories, taking only the first block4_hit_entity_topn hits
    #      (fallback for when the query mentions no entity).
    # Then constrained by the independent block4_* budgets — no competing with the global
    # token_budget for quota.
    # **Trusted entities only**: this block is spliced straight into the system prompt;
    # short-phrase noise getting in is pure interference.
    # Both paths pass this gate (query-word entities too); otherwise the origin gating
    # is silently left open a crack.
    trusted_ids = trusted_entity_ids()
    by_id = {e["id"]: e for e in entities.values()}
    cand_ids: list[str] = []

    def _push(eid: str) -> None:
        if eid and eid in trusted_ids and eid not in cand_ids:
            cand_ids.append(eid)

    # ① Query-word entities. **Not enabled when narrowing by tags** — in that case the whole
    #    context block should only expand within the hit set (dragging entity state from other
    #    chapters into a "writing per chapter" prompt would break isolation).
    if query and filter_tags is None:
        for name in extract_entities(query):
            ent = entities.get(name.lower())
            if ent:
                _push(ent["id"])
    # ② Entities of the hit memories (taking only the first few hits)
    for h in hits[:block4_hit_entity_topn]:
        for eid in (h.get("entity_ids") or []):
            _push(eid)

    ent_facts: list[str] = []
    seen_facts: set[str] = set()
    used_chars = 0
    for eid in cand_ids[:block4_max_entities]:
        ent = by_id.get(eid)
        if not ent:
            continue
        facts = [f for f in entity_facts(
            ent["key"], limit=block4_max_facts_per_entity) if f not in seen_facts]
        if not facts:
            continue
        line = f"{ent['name']}: " + " / ".join(facts)
        # Independent budget: once exceeded, **drop the whole line of each subsequent entity**
        # (never cut in the middle of a line — facts within the block are already in reverse
        # chronological order, so the first few are the most recent). The first line is kept
        # unconditionally, otherwise the whole block could end up empty.
        if ent_facts and used_chars + len(line) > block4_max_chars:
            break
        seen_facts.update(facts)
        ent_facts.append(line)
        used_chars += len(line)

    # Coexisting-accounts warning: if the hits contain "suspected different accounts of the
    # same fact", call it out at the end of the block.
    # No collapsing, no deleting entries — just let the reader of this block (human or model)
    # know "the next two lines are two versions of the same thing; do not mix them".
    # The old account may be a historical fact being cited, or it may be the correct one.
    # Each hit carries a date prefix (2026-11-05): assembled context is the host's most-used
    # entry point; without time here, the library-side derived `occurred_at` would be given
    # away for nothing — the Agent would still receive a pile of time-less fragments, and
    # questions like "a few weeks ago / last month" would remain unanswerable.
    # Added only when the derived fields exist (entries with unknown time do not get a
    # made-up date).
    # Costs about 17 chars per entry; the block budget is trimmed uniformly by render_blocks,
    # so the cap will not be breached.
    hit_lines = [
        (f"[{h['occurred_at']} {h['weekday']}] {h['content']}"
         if h.get("occurred_at") and h.get("weekday") else h["content"])
        for h in hits
    ]
    conflicted = sorted({
        h["conflict_group"] for h in hits if h.get("conflict_group")
    })
    if conflicted:
        hit_lines.append(
            f"⚠️ Among them, {len(conflicted)} groups look like different accounts of the same fact "
            f"(kept side by side, not collapsed): defer to the authoritative source; within a group, newer=true is the more recent one."
        )

    blocks = {
        "block_1_current_hits": hit_lines,
        "block_2_pending": [
            f"[{i['status']}] {i['text']}" for i in intents if i["status"] == "open"
        ][:5] or ["(No todos)"],
        "block_3_timeline": [s["text"] for s in summaries[-2:]] or ["(No session history)"],
        "block_4_entity_state": ent_facts or ["(No related entity state)"],
        "block_5_open_topics": [t["text"] for t in open_topics[-3:]] or ["(No open topics)"],
    }
    text = render_blocks(blocks, token_budget, item_max_chars)
    # Self-report per-item clipping.  Numeric tails (amounts, counts,
    # thresholds) often live at the end of a line, so a silent 200-char
    # cut destroys the payload.  Callers can pass item_max_chars=0.
    truncated_items: list[dict] = []
    if item_max_chars > 0:
        for slot, values in blocks.items():
            for idx, value in enumerate(values or []):
                if isinstance(value, str) and len(value) > item_max_chars:
                    truncated_items.append({
                        "slot": slot, "index": idx,
                        "chars": len(value),
                        "kept_chars": item_max_chars,
                        "dropped_chars": len(value) - item_max_chars,
                    })
    return {
        "blocks": blocks,
        "text": text,
        "hits": hits,
        "token_estimate": len(text) // 2,
        "truncated_items": truncated_items,
        "truncated_count": len(truncated_items),
    }
