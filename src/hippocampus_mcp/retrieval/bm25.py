# -*- coding: utf-8 -*-
"""Lightweight BM25 (mixed Chinese/English).

Benchmark conclusion (V9d): vector + BM25 hybrid (RRF fusion) beats pure
vector by MRR +38%. But BM25 previously existed only in the benchmark harness,
not in the production src package -- this module fills that gap.

Chinese has no space tokenization, so bigram splitting is used
("布偶猫" -> "布偶","偶猫"); English splits by word. No external dependencies.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

_K1 = 1.5
_B = 0.75

_EN_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English: English by word, Chinese by bigram
    (single characters are kept too, so short words can still hit)."""
    text = (text or "").lower()
    toks: list[str] = []
    for en in _EN_WORD.findall(text):
        if len(en) > 1:
            toks.append(en)
    zh_runs = re.findall(r"[\u4e00-\u9fa5]+", text)
    for run in zh_runs:
        if len(run) == 1:
            toks.append(run)
            continue
        for i in range(len(run) - 1):
            toks.append(run[i : i + 2])
    return toks


class BM25:
    """In-memory BM25 index.

    Usage:
        idx = BM25()
        idx.index(doc_ids, texts)
        scores = idx.query("查询文本")  # {doc_id: score}
    """

    def __init__(self) -> None:
        self.docs: dict[str, list[str]] = {}
        self.tf: dict[str, dict[str, int]] = {}
        self.df: dict[str, int] = {}
        self.len: dict[str, int] = {}
        self.avg_len = 0.0
        self._dirty = False
        # For incremental maintenance: total token count across documents
        # (numerator of avg_len)
        self._total_len = 0

    def index(self, doc_ids: Iterable[str], texts: Iterable[str]) -> None:
        self.docs.clear()
        self.tf.clear()
        self.df.clear()
        self.len.clear()
        self._total_len = 0
        for did, text in zip(doc_ids, texts):
            toks = tokenize(text)
            self.docs[did] = toks
            self.len[did] = len(toks)
            self._total_len += len(toks)
            tf_map: dict[str, int] = {}
            for t in toks:
                tf_map[t] = tf_map.get(t, 0) + 1
            self.tf[did] = tf_map
            for t in tf_map:
                self.df[t] = self.df.get(t, 0) + 1
        self._refresh_avg_len()
        self._dirty = False

    # ------------------------------------------------------------------
    # Incremental maintenance (coverage report, low 14)
    # ------------------------------------------------------------------
    def _refresh_avg_len(self) -> None:
        n = len(self.docs)
        self.avg_len = (self._total_len / n) if n else 0.0

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def add_doc(self, doc_id: str, text: str) -> None:
        """Add/update the index for a single document (O(length of that
        document), no full index rebuild).

        Previously ``recall`` re-ran ``index()`` over all alive records on every
        call: invisible at 8k entries, but at scale it becomes the dominant
        chunk of query latency. Now maintained incrementally as writes happen.
        """
        if doc_id in self.docs:
            self.remove_doc(doc_id)
        toks = tokenize(text)
        tf_map: dict[str, int] = {}
        for t in toks:
            tf_map[t] = tf_map.get(t, 0) + 1
        self.docs[doc_id] = toks
        self.tf[doc_id] = tf_map
        self.len[doc_id] = len(toks)
        self._total_len += len(toks)
        for t in tf_map:
            self.df[t] = self.df.get(t, 0) + 1
        self._refresh_avg_len()
        self._dirty = False

    def remove_doc(self, doc_id: str) -> bool:
        """Remove a single document's index entry."""
        tf_map = self.tf.pop(doc_id, None)
        if tf_map is None:
            return False
        for t in tf_map:
            left = self.df.get(t, 0) - 1
            if left > 0:
                self.df[t] = left
            else:
                self.df.pop(t, None)
        self._total_len -= self.len.pop(doc_id, 0)
        self.docs.pop(doc_id, None)
        self._refresh_avg_len()
        return True

    def query(self, text: str, top_k: int = 50) -> dict[str, float]:
        """Return {doc_id: bm25_score}, score descending (truncated to top_k)."""
        if not self.docs:
            return {}
        n = len(self.docs)
        q_tokens = tokenize(text)
        scores: dict[str, float] = {}
        for qt in q_tokens:
            df = self.df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for did, tf_map in self.tf.items():
                f = tf_map.get(qt, 0)
                if not f:
                    continue
                dl = self.len[did] or 1
                denom = f + _K1 * (1 - _B + _B * dl / (self.avg_len or 1.0))
                scores[did] = scores.get(did, 0.0) + idf * (f * (_K1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return dict(ranked)


def rrf_fuse(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion: multi-ranked-list fusion (the method adopted by
    the V9d benchmark).

    score(d) = sum_i w_i / (k + rank_i(d))
    """
    weights = weights or [1.0] * len(ranked_lists)
    out: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(lst):
            out[doc_id] = out.get(doc_id, 0.0) + w / (k + rank + 1)
    return out
