# -*- coding: utf-8 -*-
"""轻量 BM25（中英文混排）。

评测结论（V9d）：向量 + BM25 混合（RRF 融合）比纯向量 MRR +38%。
但 BM25 此前只存在于评测 harness，生产包 src 里没有——本模块把它补齐。

中文没有空格分词，采用 bigram 切分（"布偶猫" → "布偶","偶猫"），
英文按词切分。无需外部依赖。
"""

from __future__ import annotations

import math
import re
from typing import Iterable

_K1 = 1.5
_B = 0.75

_EN_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """中英文混排切词：英文按词、中文按 bigram（单字也保留，保证短词可命中）。"""
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
    """内存版 BM25 索引。

    用法:
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
        # 增量维护用：文档总 token 数（avg_len 的分子）
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
    # 增量维护（覆盖报告·低 14）
    # ------------------------------------------------------------------
    def _refresh_avg_len(self) -> None:
        n = len(self.docs)
        self.avg_len = (self._total_len / n) if n else 0.0

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def add_doc(self, doc_id: str, text: str) -> None:
        """新增/更新单篇文档的索引（O(该文档长度)，不重建整个索引）。

        原先 ``recall`` 每次都要对全部存活记录重跑 ``index()``：8k 条无感，
        量级上去就是查询延迟的大头。改为随写入增量维护。
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
        """移除单篇文档的索引。"""
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
        """返回 {doc_id: bm25_score}，按分数降序（截断 top_k）。"""
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
    """Reciprocal Rank Fusion：多路排序融合（V9d 评测采用的方式）。

    score(d) = Σ_i w_i / (k + rank_i(d))
    """
    weights = weights or [1.0] * len(ranked_lists)
    out: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(lst):
            out[doc_id] = out.get(doc_id, 0.0) + w / (k + rank + 1)
    return out
