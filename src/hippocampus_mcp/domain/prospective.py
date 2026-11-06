# -*- coding: utf-8 -*-
"""O6: prospective-memory contextual triggering (domain pure functions, M2 discipline).

Biological counterpart: cue-triggered prospective memory — beyond "when the
time comes, do the thing", **the intent surfaces automatically when a related
situation is encountered**.

Current gap: open intents registered via ``remember_intent`` are only brought
back by session_start's ``block_pending_intents`` (one-shot at the opening,
truncated to 5 by list order); mid-session, when ``get_memory_context``
touches a related topic, relevant intents are mixed into the "first 5 open"
and ones ranked later are never recalled.

Mechanism (all pure functions; trigger sets are **derived at registration
time**, assembly only does intersection tests):

* Registration side (``derive_intent_triggers``): extract entity names from the
  intent text (same extractor as retrieval, lexicon hits first) + a content-
  word set — ``tokens_of`` (Chinese bigrams / English words, stopwords removed)
  **plus Chinese trigrams**. Why add trigrams (measured): ``tokens_of``'s
  bigrams are **non-overlapping** segments; a change of wording order kills
  them all — "写第三章初稿" segments as 写第/三章/初稿, "第三章写到哪了"
  segments as 第三/章写/到哪; obviously the same topic yet zero intersection.
  The trigram 「第三章」 appears on both sides, and a three-character string is
  specific enough without introducing noise. Stored in the intent entry's
  ``trigger_entities`` / ``trigger_topic`` — persisted with
  brain_intents.json; the multi-session merge layer preserves extra keys
  as-is.
* Assembly side (``prospective_block2``): the query's entity names/content
  words (including trigrams, same convention as the registration side)
  **intersect** with an active intent's trigger set → that intent is hoisted to
  the front of block_2 and its line gains the "⚡ related to the current
  topic" marker; untriggered intents keep their original order at the back.
  **cap 5 does not grow** (block_2 budget unchanged); trigger details are
  self-reported via the return body's ``prospective_intents`` key (when it
  triggered and on what grounds).

Non-intrusiveness guarantee (the acceptance test's negative control): empty
query, intent without a trigger set, empty intersection — in all three cases
``prospective_block2``'s lines are **byte-for-byte identical** to the original
implementation and produce no ``prospective_intents`` key (zero traces; the
same discipline as U5/D4).

No fight with completion semantics (acceptance item three): this module only
reorders block_2's **display order** and adds a display marker; it never
changes an intent's ``text`` / ``status`` — complete_intent matches on text by
keyword, resolve_intent filters by status; both channels see the original
entries. Triggering produces no automatic completion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "INTENT_BLOCK_CAP", "TRIGGER_SUFFIX",
    "IntentTriggers", "ProspectivePlan",
    "derive_intent_triggers", "prospective_block2", "zh_trigrams",
]

#: block_2 intent count cap — consistent with the existing ``[:5]`` in
#: domain/context_slots.py (copies the status quo, adds no budget; changing the
#: cap requires syncing both sides, with comments cross-referencing).
INTENT_BLOCK_CAP = 5

#: Marker appended to a triggered intent's block_2 line. Inline so the text slot
#: stays directly readable; structured details go through the
#: ``prospective_intents`` key — both from the same source.
TRIGGER_SUFFIX = " (⚡ related to the current topic)"

_ZH_RUN = re.compile(r"[\u4e00-\u9fa5]{2,}")


def zh_trigrams(text: str) -> set[str]:
    """Chinese trigrams (a three-character sliding window over runs of consecutive Han characters; runs shorter than three produce nothing).

    Provided by this module rather than modifying ``tokens_of``: that function
    is the tokenization source of truth for the write/read sides' "same fact"
    criterion (pinned by 44 golden cases); the trigger set has no standing to
    change its convention — only an **increment** (union) is applied, and the
    bigram / English-word conventions are preserved as-is.
    """
    out: set[str] = set()
    for run in _ZH_RUN.findall(text or ""):
        for i in range(len(run) - 2):
            out.add(run[i:i + 3])
    return out


@dataclass(frozen=True)
class IntentTriggers:
    """One intent's trigger conditions (derived once at registration, persisted with the layer)."""
    entities: tuple[str, ...] = ()      # entity names (extractor convention, kept as-is)
    topic: frozenset[str] = field(default_factory=frozenset)  # content words + trigrams


def derive_intent_triggers(
    text: str,
    *,
    extract_entities: Callable[[str], Iterable[str]],
    tokens_of: Callable[[str], set[str]],
) -> IntentTriggers:
    """Derive trigger conditions from an intent text (registration side; pure function — extractor injected).

    Entity names are kept exactly as the extractor produced them (casing /
    normalization is the extractor's job); matching lowercases both sides.
    Content-word set = ``tokens_of`` (the same one retrieval uses) ∪
    ``zh_trigrams`` (the patch for wording-order fragility; see the module
    docstring).
    """
    ents = tuple(dict.fromkeys(
        e for e in (extract_entities(text or "") or []) if e))
    toks = frozenset(tokens_of(text or "")) | frozenset(zh_trigrams(text or ""))
    return IntentTriggers(entities=ents, topic=toks)


@dataclass(frozen=True)
class ProspectivePlan:
    """The reordering result for block_2 (assembly-side product).

    Attributes:
        lines: block_2's line list (format identical to the original
            implementation ``[status] text``, triggered lines carrying
            ``TRIGGER_SUFFIX`` at the end). When nothing triggered, byte-for-byte
            identical to the original implementation.
        triggered: trigger details (``[{id, text, matched_on}]``) — an empty
            list = nothing triggered this time (the caller uses this to decide
            whether to add the ``prospective_intents`` key).
    """
    lines: list[str] = field(default_factory=list)
    triggered: list[dict[str, Any]] = field(default_factory=list)


def _matched_on(
    it: dict[str, Any],
    *,
    query_entities: set[str],
    query_tokens: frozenset[str],
) -> list[str]:
    """One intent's list of hit reasons (entity / content-word channels; for reporting)."""
    reasons: list[str] = []
    hits_e = sorted(set(e.lower() for e in (it.get("trigger_entities") or []))
                    & query_entities)
    reasons.extend("entity:" + e for e in hits_e)
    hits_t = sorted(set(it.get("trigger_topic") or ()) & set(query_tokens))
    reasons.extend("topic:" + t for t in hits_t)
    return reasons


def prospective_block2(
    intents: Sequence[Any],
    *,
    query: str,
    extract_entities: Callable[[str], Iterable[str]],
    tokens_of: Callable[[str], set[str]],
    cap: int = INTENT_BLOCK_CAP,
) -> ProspectivePlan:
    """Assemble block_2's intent lines: triggered ones hoisted to the front, the rest keep their original order (pure function).

    Args:
        intents: brain's intent sequence (read-only as-is; a missing trigger-set
            field = old data, never triggers — same behavior as having no
            trigger set).
        query: the current query; an empty string = original behavior (the
            trigger mechanism only applies with a query context).
        extract_entities / tokens_of: query-side entity extraction and
            tokenization (injected).
        cap: count cap (default 5, same as the original ``[:5]``; sort first,
            cut after — triggered intents are not squeezed out of the cap by
            unrelated ones).

    Returns:
        :class:`ProspectivePlan`. When ``triggered`` is empty, ``lines`` is
        byte-for-byte identical to context_slots' original
        ``[f"[{i['status']}] {i['text']}" ...][:cap]`` (the negative control's
        anchor).
    """
    open_intents = [i for i in intents
                    if isinstance(i, dict) and i.get("status") == "open"]

    def _render(items: list[dict[str, Any]], marked: set[int]) -> list[str]:
        lines = [f"[{i['status']}] {i['text']}"
                 + (TRIGGER_SUFFIX if idx in marked else "")
                 for idx, i in enumerate(items)]
        cut = lines[:max(int(cap), 0)]
        return cut or ["(No todos)"]

    if not (query or "").strip():
        return ProspectivePlan(lines=_render(open_intents, set()), triggered=[])

    q_entities = {e.lower() for e in (extract_entities(query) or []) if e}
    # Query side uses the same convention as the registration side: tokens_of (bigrams/English) ∪ Chinese trigrams.
    q_tokens = frozenset(tokens_of(query or "")) | frozenset(
        zh_trigrams(query or ""))
    if not q_entities and not q_tokens:
        return ProspectivePlan(lines=_render(open_intents, set()), triggered=[])

    triggered_items: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for it in open_intents:
        reasons = _matched_on(
            it, query_entities=q_entities, query_tokens=q_tokens)
        if reasons:
            triggered_items.append(it)
            details.append({
                "id": it.get("id", ""),
                "text": it.get("text", ""),
                "matched_on": reasons,
            })
        else:
            rest.append(it)
    if not triggered_items:
        return ProspectivePlan(lines=_render(open_intents, set()), triggered=[])

    ordered = triggered_items + rest
    marked = set(range(len(triggered_items)))
    return ProspectivePlan(lines=_render(ordered, marked), triggered=details)
