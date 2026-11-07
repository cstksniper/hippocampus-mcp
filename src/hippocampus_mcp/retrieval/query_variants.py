# -*- coding: utf-8 -*-
"""Query-variant generation and candidate-list fusion for recall fanout (V2).

Motivation (plan ``docs/design/OpenViking借鉴点执行计划书_20261106.md`` §3):
colloquial whole-sentence queries score poorly — the dev library's miss log
recorded "之前 mem0 对比我们拍板的结论是什么？" at top-similarity 0.268. V2
adds an opt-in fanout: when the query looks like a natural sentence, a
keyword variant is derived and BOTH queries run the existing retrieval; the
two hit lists are merged with Reciprocal Rank Fusion.

Discipline (plan §3.2):

* **Scoring is untouched.** This module never sees a similarity number —
  fusion reorders candidate lists by RRF and keeps each item's own
  ``similarity``/``score`` fields verbatim from its source run. R1 owns the
  scoring formulas; V2 only widens the candidate gate in front of them.
* **Envelope stability.** Fused items are verbatim dicts from one of the two
  runs; no new keys enter the 14-key MCP envelope.
* Pure stdlib. No jieba/segmentation dependency: CJK content is reduced to
  character bigrams with function characters filtered, which is the same
  granularity the in-memory BM25 lane already uses.
"""
from __future__ import annotations

from typing import Any, Iterable

__all__ = ["build_variant", "fuse_variant_items"]

#: Queries shorter than this are treated as keywords already (no fanout).
MIN_QUERY_CHARS = 10

#: CJK function characters — a bigram containing any of these is dropped from
#: the variant. Content-bearing characters (对比/拍板/结论...) are NOT here;
#: the set only covers pronouns, particles, question scaffolding and
#: copulas, so "对比我们拍板的结论" keeps 对比/拍板/结论 and loses 我们/的.
_CJK_STOP_CHARS = frozenset(
    "的了是在有和与就都还呢吗吧啊呀哦么什怎这那个把被从向于或及跟让使要会"
    "很才刚只更太也我你他她它们自己们说问想看看听讲做去吧来过"
)

#: Latin stop words (lowercase) — kept minimal; memory queries are mostly
#: Chinese and the hard relevance floor discards junk candidates anyway.
_STOP_LATIN = frozenset(
    "the a an of to is are was were what how why when and or in on at for "
    "do does did can could will would should i you we they it this that".split()
)


def _cjk_bigrams(text: str) -> list[str]:
    """Character bigrams over CJK runs, function-character pairs dropped.

    A bigram survives only when both characters are outside
    ``_CJK_STOP_CHARS``. Single remaining content characters are not emitted
    (BM25's bigram lane would not match them consistently).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n - 1:
        a, b = text[i], text[i + 1]
        if a in _CJK_STOP_CHARS or b in _CJK_STOP_CHARS:
            i += 1
            continue
        out.append(a + b)
        i += 1
    return out


def build_variant(query: str, entity_names: Iterable[str] = (),
                  min_query_chars: int = MIN_QUERY_CHARS) -> str:
    """Derive a keyword variant of a natural-language query.

    Returns ``""`` whenever a second run is not worth it: the switch is
    off at the call site, the query is short (already keyword-like), the
    variant collapses to the original query, or nothing survives filtering.

    Args:
        query: the raw user query.
        entity_names: known entity names/aliases — a case-insensitive
            substring hit is promoted verbatim into the variant (these are
            the highest-value anchors; ``mem0`` in the motivating miss).
        min_query_chars: minimum raw length for fanout to apply.
    """
    q = (query or "").strip()
    if len(q) < max(int(min_query_chars), 1):
        return ""
    hits = q.lower()
    pieces: list[str] = []
    for name in entity_names:
        name = str(name or "").strip()
        if len(name) >= 2 and name.lower() in hits:
            pieces.append(name)
    keep_latin: list[str] = []
    keep_cjk: list[str] = []
    for token in q.replace("？", " ").replace("?", " ").replace("，", " ") \
            .replace(",", " ").replace("。", " ").replace(".", " ").split():
        if all("\u4e00" <= ch <= "\u9fff" for ch in token):
            keep_cjk.extend(_cjk_bigrams(token))
        else:
            low = token.lower()
            if len(low) >= 2 and low not in _STOP_LATIN and low not in pieces:
                keep_latin.append(token)
    variant_tokens: list[str] = []
    for piece in pieces + keep_latin + keep_cjk:
        if piece not in variant_tokens:
            variant_tokens.append(piece)
    variant = " ".join(variant_tokens).strip()
    if not variant or variant == q:
        return ""
    return variant[:64].strip()


def fuse_variant_items(items_a: list[dict[str, Any]],
                       items_b: list[dict[str, Any]],
                       top_k: int, k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal-Rank-Fusion of two candidate lists (V2 fanout merge).

    RRF score = sum of ``1 / (k + rank)`` over the lists containing the id
    (standard k=60). Items keep their own ``similarity``/``score`` fields
    verbatim from their first occurrence — the fused score only decides the
    ORDER and is never emitted (plan §3.3: 融合分单列，不混入 similarity).
    A memory present in both lists surfaces once, at its fused rank.
    """
    a = list(items_a or [])
    b = list(items_b or [])
    if not b:
        return a[:max(int(top_k), 0)]
    if not a:
        return b[:max(int(top_k), 0)]

    def key(it: dict[str, Any]) -> str:
        return str(it.get("memory_id") or it.get("id") or "")

    fused: dict[str, list] = {}
    for rank, it in enumerate(a):
        entry = fused.setdefault(key(it), [0.0, it])
        entry[0] += 1.0 / (k + rank + 1)
    for rank, it in enumerate(b):
        entry = fused.setdefault(key(it), [0.0, it])
        entry[0] += 1.0 / (k + rank + 1)
    ordered = sorted(fused.values(), key=lambda p: -p[0])
    return [dict(p[1]) for p in ordered[:max(int(top_k), 0)]]
