# -*- coding: utf-8 -*-
"""Standard retrieval and QA evaluation metrics.

Implements the core metrics used by the LoCoMo, LongMemEval and Mem0
evaluation frameworks:
    - Retrieval metrics: MRR, Hit@k, Recall@k (recall_any / recall_all), NDCG@k
    - QA metrics:   F1 (token-level, Porter stemmer), Exact Match

Reference sources:
    - LoCoMo: https://github.com/snap-research/LoCoMo (task_eval/evaluation.py)
    - LongMemEval: https://github.com/xiaowu0162/LongMemEval (src/retrieval/eval_utils.py)
    - Mem0: https://github.com/mem0ai/memory-benchmarks
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Sequence


# ==================================================================
# Retrieval metrics
# ==================================================================

def mrr(rankings: list[list[str]], ground_truths: list[list[str]]) -> float:
    """Mean Reciprocal Rank.

    Args:
        rankings: per-query retrieval result lists (ordered doc id lists).
        ground_truths: per-query lists of correct doc ids.

    Returns:
        MRR score [0, 1].
    """
    if not rankings:
        return 0.0
    total = 0.0
    for ranked, gts in zip(rankings, ground_truths):
        gt_set = set(gts)
        for i, doc_id in enumerate(ranked):
            if doc_id in gt_set:
                total += 1.0 / (i + 1)
                break
    return total / len(rankings)


def hit_at_k(
    rankings: list[list[str]], ground_truths: list[list[str]], k: int = 5
) -> float:
    """Hit@k: whether any correct doc is hit within the top-k.

    Args:
        rankings: retrieval result lists.
        ground_truths: correct doc lists.
        k: truncation position.

    Returns:
        Hit@k score [0, 1].
    """
    if not rankings:
        return 0.0
    hits = 0
    for ranked, gts in zip(rankings, ground_truths):
        gt_set = set(gts)
        top_k = ranked[:k]
        if any(doc_id in gt_set for doc_id in top_k):
            hits += 1
    return hits / len(rankings)


def recall_any_at_k(
    rankings: list[list[str]], ground_truths: list[list[str]], k: int = 5
) -> float:
    """Recall (any) @k: proportion of queries hitting any correct doc within the top-k.

    Equivalent to Hit@k; the naming follows LongMemEval.
    """
    return hit_at_k(rankings, ground_truths, k)


def recall_all_at_k(
    rankings: list[list[str]], ground_truths: list[list[str]], k: int = 5
) -> float:
    """Recall (all) @k: proportion of queries hitting all correct docs within the top-k.

    Follows LongMemEval's recall_all@k definition.
    """
    if not rankings:
        return 0.0
    hits = 0
    for ranked, gts in zip(rankings, ground_truths):
        gt_set = set(gts)
        top_k = set(ranked[:k])
        if gt_set.issubset(top_k):
            hits += 1
    return hits / len(rankings)


def ndcg_at_k(
    rankings: list[list[str]], ground_truths: list[list[str]], k: int = 5
) -> float:
    """NDCG@k: normalized discounted cumulative gain.

    Follows LongMemEval's NDCG implementation:
        relevance = 1 if doc in ground_truth else 0
        DCG = rel[0] + sum(rel[i] / log2(i+1) for i in 1..k)
        NDCG = DCG / IDCG (the DCG of the ideal ranking)

    Args:
        rankings: retrieval result lists.
        ground_truths: correct doc lists.
        k: truncation position.

    Returns:
        NDCG@k score [0, 1].
    """
    if not rankings:
        return 0.0

    def dcg(relevances: list[int]) -> float:
        if not relevances:
            return 0.0
        result = relevances[0]
        for i in range(1, len(relevances)):
            result += relevances[i] / math.log2(i + 1)
        return result

    total = 0.0
    for ranked, gts in zip(rankings, ground_truths):
        gt_set = set(gts)
        top_k = ranked[:k]

        # Actual DCG
        rels = [1 if doc_id in gt_set else 0 for doc_id in top_k]
        actual_dcg = dcg(rels)

        # Ideal DCG
        n_relevant = min(len(gt_set), k)
        ideal_rels = [1] * n_relevant + [0] * (k - n_relevant)
        ideal_dcg = dcg(ideal_rels)

        if ideal_dcg > 0:
            total += actual_dcg / ideal_dcg

    return total / len(rankings)


# ==================================================================
# QA metrics (LoCoMo style)
# ==================================================================

# Simplified Porter Stemmer implementation (avoids the NLTK dependency)
_STEM_CACHE: dict[str, str] = {}


def _simple_stem(word: str) -> str:
    """Simplified stemming (Porter style).

    Handles common English suffixes: -ing, -ed, -ly, -s, -es, -ment, -tion, -ness.
    """
    if word in _STEM_CACHE:
        return _STEM_CACHE[word]

    original = word
    suffixes = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("logi", "log"),
        ("fulli", "ful"),
        ("lessli", "less"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("ly", ""),
        ("ment", ""),
        ("ness", ""),
        ("s", ""),
        ("es", ""),
        ("e", ""),
    ]
    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            word = word[: -len(suffix)] + replacement
            break

    _STEM_CACHE[original] = word
    return word


# Stopwords / non-informative words (following LoCoMo)
_ARTICLES = {"a", "an", "the", "and"}


def _normalize_text(text) -> list[str]:
    """Text preprocessing: strip punctuation, lowercase, remove stopwords, stem.

    Follows the preprocessing pipeline of LoCoMo evaluation.py.
    """
    # Ensure the input is a string (answer may be int/float in LoCoMo data)
    text = str(text) if not isinstance(text, str) else text
    # Remove commas
    text = text.replace(",", " ")
    # Lowercase
    text = text.lower()
    # Strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize
    tokens = text.split()
    # Remove stopwords + stem
    tokens = [_simple_stem(t) for t in tokens if t and t not in _ARTICLES]
    return tokens


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score (LoCoMo style).

    Preprocessing: remove commas, remove articles, strip punctuation,
    lowercase, Porter stemming.

    Args:
        prediction: predicted answer.
        ground_truth: reference answer.

    Returns:
        F1 score [0, 1].
    """
    pred_tokens = _normalize_text(prediction)
    gt_tokens = _normalize_text(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match: token sets match exactly after preprocessing.

    Args:
        prediction: predicted answer.
        ground_truth: reference answer.

    Returns:
        1.0 match, 0.0 no match.
    """
    pred_tokens = set(_normalize_text(prediction))
    gt_tokens = set(_normalize_text(ground_truth))
    return 1.0 if pred_tokens == gt_tokens else 0.0


def multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """F1 for multi-hop questions: split the answer into sub-answers by comma, compute F1 for each, then average.

    Follows LoCoMo's category=1 evaluation.
    """
    gt_str = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    sub_answers = [a.strip() for a in gt_str.split(",") if a.strip()]
    if not sub_answers:
        return f1_score(prediction, ground_truth)

    # Compute F1 for each sub-answer, then average
    scores = []
    for sub_ans in sub_answers:
        scores.append(f1_score(prediction, sub_ans))
    return sum(scores) / len(scores)


def adversarial_score(prediction: str) -> float:
    """Adversarial question scoring (LoCoMo category=5).

    Checks whether the output contains a refusal phrase ("no information
    available" / "not mentioned"). Correct refusal scores 1, a wrong answer
    scores 0.
    """
    pred_lower = str(prediction).lower() if not isinstance(prediction, str) else prediction.lower()
    refusal_phrases = [
        "no information available",
        "not mentioned",
        "not enough information",
        "cannot answer",
        "don't know",
        "do not know",
        "no information",
    ]
    return 1.0 if any(phrase in pred_lower for phrase in refusal_phrases) else 0.0


# ==================================================================
# Aggregate metrics
# ==================================================================

class EvalReport:
    """Evaluation results report."""

    def __init__(self, method_name: str = "") -> None:
        self.method_name = method_name
        self.retrieval_metrics: dict[str, float] = {}
        self.qa_metrics: dict[str, float] = {}
        self.per_category: dict[str, dict[str, float]] = {}

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "retrieval": self.retrieval_metrics,
            "qa": self.qa_metrics,
            "per_category": self.per_category,
        }

    def summary(self) -> str:
        lines = [f"=== {self.method_name} ==="]
        if self.retrieval_metrics:
            lines.append("  Retrieval metrics:")
            for k, v in sorted(self.retrieval_metrics.items()):
                lines.append(f"    {k}: {v:.4f}")
        if self.qa_metrics:
            lines.append("  QA metrics:")
            for k, v in sorted(self.qa_metrics.items()):
                lines.append(f"    {k}: {v:.4f}")
        if self.per_category:
            lines.append("  Per category:")
            for cat, metrics in sorted(self.per_category.items()):
                lines.append(f"    {cat}: {metrics}")
        return "\n".join(lines)
