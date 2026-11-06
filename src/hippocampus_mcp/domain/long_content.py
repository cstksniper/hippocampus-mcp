# -*- coding: utf-8 -*-
"""Handling policy for over-long content: **pure decisions and pure
splitting**, no disk writes, no config reads.

## Origin (measured, not conjectured)

Embedding models have a token **hard cap** (BGE / MiniLM = 512). A 3372-char
card measured ``embedded_chars=511`` and ``dropped_ratio=0.8485`` -- **the
text after char 512 contributes zero to the vector, yet the write still
returns success**. With the BM25 literal channel off, tail facts
(/etc/secret/db.key) are **completely unrecallable** by semantic phrasing.

This is "written in but invisible to retrieval" -- the silent failure this
project fears most.

## Three-state policy (default = status quo, zero regression)

* ``warn`` (**default**) -- self-report only; behavior byte-identical to
  before the change.
* ``split`` -- when over the window, split into multiple cards by sentence
  and write each; every card enters the vector index.
* ``reject`` -- refuse the write when over the window; the response body
  explains how to fix it.

**Whitelist parsing**: only an explicit ``split`` / ``reject`` takes effect;
unknown values always fall back to ``warn``. Same reasoning as the
home-library write boundary switch: a default-off switch with blacklist
semantics would be quietly opened by ``''`` / ``'maybe'``; here it is the
reverse -- **the two modes that change write behavior must be named
explicitly by the user**; a typo equals not setting it.

## Why the splitter is injected

``domain`` may depend only on ``common`` (architectural constraint), while
the real sentence splitter (``brain._split_facts``, with quirks like "short
sentences merge into the previous one") lives in brain. Writing **a second
splitter** in this module would instantly create a second source of truth;
therefore: the splitter is injected by the composition root, and when
unavailable we degrade to "hard split at the limit" -- the degradation is
**explicitly visible** (the response body carries a note), never silent.

## Splitting rules

1. Split into sentences with the injected splitter first;
2. When a single sentence still exceeds the limit, break it at the **last
   strong break** (``。！？；`` and newlines) within the ``[1/2, 1]`` region
   of the limit; if none is found, hard-split.
3. The limit is ``max_seq_length * SAFE_RATIO`` (default 0.9): a 512-token
   model must break at 460 chars. This leaves room for ``[CLS]``/``[SEP]``
   and for Chinese's 1 char ≈ 1 token -- **cutting right at 512 would hit
   the boundary exactly once more**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

POLICY_WARN = "warn"
POLICY_SPLIT = "split"
POLICY_REJECT = "reject"

POLICIES: tuple[str, ...] = (POLICY_WARN, POLICY_SPLIT, POLICY_REJECT)

#: Discount from the token cap to the char cap (see module docstring,
#: splitting rule 3).
SAFE_RATIO = 0.9

#: Maximum number of cards one write may be split into. Over that, refuse the
#: write -- better an error than flooding the library with thousands of
#: entries at once (that situation is usually a whole book stuffed in as one
#: memory).
#: 200 rather than smaller: ``ingest`` splitting a long article into hundreds
#: of sentences is normal (measured: a 100-sentence article = 101 pieces);
#: setting 50 would misjudge normal documents as "library flooding".
MAX_PIECES = 200

#: Strong breaks: sentence cuts prefer these (``_hard_split`` searches from
#: the end backwards).
_STRONG_BREAKS = "。！？；!?;\n"
_WEAK_BREAKS = "，、, "


@dataclass(frozen=True)
class LongContentPlan:
    """The handling plan for one over-long write (pure data; the caller
    executes it)."""

    action: str                      # "write" | "split" | "reject"
    policy: str                      # the effective policy (after parsing)
    pieces: tuple[str, ...] = ()     # the pieces when action=="split"
    reason: str = ""                 # why this verdict (echoed to the user, never silent)
    note: str = ""                   # degradation/boundary note
    fields: dict[str, object] = field(default_factory=dict)


def parse_policy(raw: Optional[str]) -> str:
    """Parse a config value into one of the three states. **Whitelist**:
    unknown value → ``warn`` (= status quo)."""
    v = (raw or "").strip().lower()
    if v in POLICIES:
        return v
    return POLICY_WARN


def char_limit(max_seq_length: int, *, safe_ratio: float = SAFE_RATIO) -> int:
    """Token cap → safe character cap.

    Returns 0 for ``max_seq_length <= 0`` (hash backend / unavailable); the
    caller then **skips all intervention** -- with no cap there is no
    over-long window to speak of.
    """
    if not max_seq_length or max_seq_length <= 0:
        return 0
    return max(1, int(max_seq_length * safe_ratio))


def _hard_split(text: str, limit: int) -> list[str]:
    """Cut an over-long text at ``limit``, preferring sentence breaks."""
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        if n - start <= limit:
            out.append(text[start:])
            break
        end = start + limit
        # Find the last break in the second half, avoiding a cut mid-sentence.
        window_lo = start + max(1, limit // 2)
        cut = -1
        for ch in _STRONG_BREAKS:
            cut = max(cut, text.rfind(ch, window_lo, end))
        if cut < window_lo:
            for ch in _WEAK_BREAKS:
                cut = max(cut, text.rfind(ch, window_lo, end))
        idx = (cut + 1) if cut >= window_lo else end
        out.append(text[start:idx])
        start = idx
    return out


def split_pieces(
    text: str,
    limit: int,
    split_facts: Optional[Callable[[str], list[str]]] = None,
) -> tuple[list[str], str]:
    """Returns ``(pieces, degradation note)``.

    When the splitter is available, sentence-split first (preserving brain's
    "short sentences merge into the previous one" quirk); when unavailable,
    hard-split the whole thing and say so in the second return value -- **the
    degradation must be visible**.
    """
    note = ""
    sents: list[str] = []
    if split_facts is not None:
        try:
            sents = [s for s in (split_facts(text) or []) if s and s.strip()]
        except Exception as exc:  # a broken splitter must not take the write down
            note = f"the splitter raised {type(exc).__name__}; degraded to hard length-based splitting"
            sents = []
    if not sents:
        if split_facts is None and not note:
            note = "no splitter injected; hard length-based splitting"
        sents = [text]

    out: list[str] = []
    for s in sents:
        if len(s) <= limit:
            out.append(s)
        else:
            out.extend(_hard_split(s, limit))
    return out, note


def plan_long_content(
    content: str,
    max_seq_length: int,
    policy_raw: Optional[str],
    *,
    truncated: bool,
    split_facts: Optional[Callable[[str], list[str]]] = None,
    safe_ratio: float = SAFE_RATIO,
) -> LongContentPlan:
    """Decide how one write should be handled.

    Args:
        content: The content to write.
        max_seq_length: The embedding model's token hard cap (0 = no cap /
            unavailable).
        policy_raw: The raw config value (unparsed).
        truncated: The precise truncation verdict already computed by the
            engine.
        split_facts: The injected splitter (may be None).
    """
    policy = parse_policy(policy_raw)
    limit = char_limit(max_seq_length, safe_ratio=safe_ratio)

    # No cap information, or genuinely not over the window → write as-is
    # (the policy does not intervene).
    if limit <= 0 or not truncated:
        return LongContentPlan(action="write", policy=policy,
                               pieces=(content,))

    if policy == POLICY_WARN:
        return LongContentPlan(
            action="write", policy=policy, pieces=(content,),
            reason=f"content exceeds the embedding model's token limit {max_seq_length}; "
                   f"text beyond it is not retrievable (policy warn: warn only, no rewrite)")

    if policy == POLICY_REJECT:
        return LongContentPlan(
            action="reject", policy=policy,
            reason=f"content exceeds the embedding model's token limit {max_seq_length}; "
                   f"policy reject: the write was refused. "
                   f"Split it into cards of <{limit} chars, or use ingest (splits by sentence by default)")

    # split
    pieces, note = split_pieces(content, limit, split_facts)
    if len(pieces) > MAX_PIECES:
        return LongContentPlan(
            action="reject", policy=policy,
            reason=f"the content would need {len(pieces)} cards, over the limit {MAX_PIECES} "
                   f"(most likely a whole document was written as one memory). "
                   f"Use ingest first, or split the text yourself",
            note=note)
    return LongContentPlan(
        action="split", policy=policy, pieces=tuple(pieces), note=note,
        reason=f"content exceeds the embedding model's token limit {max_seq_length}; "
               f"split into {len(pieces)} cards written separately")
