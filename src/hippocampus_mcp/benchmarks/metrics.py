# -*- coding: utf-8 -*-
"""标准检索与 QA 评估指标.

实现 LoCoMo、LongMemEval 和 Mem0 评估框架中使用的核心指标:
    - 检索指标: MRR, Hit@k, Recall@k (recall_any / recall_all), NDCG@k
    - QA 指标:   F1 (token-level, Porter stemmer), Exact Match

参考来源:
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
# 检索指标
# ==================================================================

def mrr(rankings: list[list[str]], ground_truths: list[list[str]]) -> float:
    """Mean Reciprocal Rank.

    Args:
        rankings: 每个查询的检索结果列表 (doc id 有序列表).
        ground_truths: 每个查询的正确文档 id 列表.

    Returns:
        MRR 分数 [0, 1].
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
    """Hit@k: top-k 中是否命中任意一个正确文档.

    Args:
        rankings: 检索结果列表.
        ground_truths: 正确文档列表.
        k: 截断位置.

    Returns:
        Hit@k 分数 [0, 1].
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
    """Recall (any) @k: top-k 中命中任意一个正确文档的比例.

    与 Hit@k 等价, 保持命名与 LongMemEval 一致.
    """
    return hit_at_k(rankings, ground_truths, k)


def recall_all_at_k(
    rankings: list[list[str]], ground_truths: list[list[str]], k: int = 5
) -> float:
    """Recall (all) @k: top-k 中命中全部正确文档的比例.

    参照 LongMemEval 的 recall_all@k 定义.
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
    """NDCG@k: 归一化折损累积增益.

    参照 LongMemEval 的 NDCG 实现:
        relevance = 1 if doc in ground_truth else 0
        DCG = rel[0] + sum(rel[i] / log2(i+1) for i in 1..k)
        NDCG = DCG / IDCG (理想排序的 DCG)

    Args:
        rankings: 检索结果列表.
        ground_truths: 正确文档列表.
        k: 截断位置.

    Returns:
        NDCG@k 分数 [0, 1].
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

        # 实际 DCG
        rels = [1 if doc_id in gt_set else 0 for doc_id in top_k]
        actual_dcg = dcg(rels)

        # 理想 DCG
        n_relevant = min(len(gt_set), k)
        ideal_rels = [1] * n_relevant + [0] * (k - n_relevant)
        ideal_dcg = dcg(ideal_rels)

        if ideal_dcg > 0:
            total += actual_dcg / ideal_dcg

    return total / len(rankings)


# ==================================================================
# QA 指标 (LoCoMo 风格)
# ==================================================================

# Porter Stemmer 的简化实现 (避免 NLTK 依赖)
_STEM_CACHE: dict[str, str] = {}


def _simple_stem(word: str) -> str:
    """简化的词干提取 (Porter 风格).

    处理常见的英文后缀: -ing, -ed, -ly, -s, -es, -ment, -tion, -ness.
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


# 停用词 / 无信息词 (参照 LoCoMo)
_ARTICLES = {"a", "an", "the", "and"}


def _normalize_text(text) -> list[str]:
    """文本预处理: 去标点、转小写、去停用词、词干化.

    参照 LoCoMo evaluation.py 的预处理流程.
    """
    # 确保输入为字符串 (LoCoMo 数据中 answer 可能是 int/float)
    text = str(text) if not isinstance(text, str) else text
    # 去逗号
    text = text.replace(",", " ")
    # 转小写
    text = text.lower()
    # 去标点
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 分词
    tokens = text.split()
    # 去停用词 + 词干化
    tokens = [_simple_stem(t) for t in tokens if t and t not in _ARTICLES]
    return tokens


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token 级别 F1 分数 (LoCoMo 风格).

    预处理: 去逗号、去冠词、去标点、转小写、Porter 词干化.

    Args:
        prediction: 预测答案.
        ground_truth: 标准答案.

    Returns:
        F1 分数 [0, 1].
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
    """Exact Match: 预处理后 token 集合完全匹配.

    Args:
        prediction: 预测答案.
        ground_truth: 标准答案.

    Returns:
        1.0 匹配, 0.0 不匹配.
    """
    pred_tokens = set(_normalize_text(prediction))
    gt_tokens = set(_normalize_text(ground_truth))
    return 1.0 if pred_tokens == gt_tokens else 0.0


def multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """多跳问题的 F1: 将答案按逗号切分为子答案, 对每个子答案计算 F1 后取均值.

    参照 LoCoMo category=1 的评估方式.
    """
    gt_str = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    sub_answers = [a.strip() for a in gt_str.split(",") if a.strip()]
    if not sub_answers:
        return f1_score(prediction, ground_truth)

    # 对每个子答案计算 F1, 取均值
    scores = []
    for sub_ans in sub_answers:
        scores.append(f1_score(prediction, sub_ans))
    return sum(scores) / len(scores)


def adversarial_score(prediction: str) -> float:
    """对抗问题评分 (LoCoMo category=5).

    检查输出是否包含拒答短语 ("no information available" / "not mentioned").
    正确拒答得 1, 错误回答得 0.
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
# 聚合指标
# ==================================================================

class EvalReport:
    """评估结果报告."""

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
            lines.append("  检索指标:")
            for k, v in sorted(self.retrieval_metrics.items()):
                lines.append(f"    {k}: {v:.4f}")
        if self.qa_metrics:
            lines.append("  QA 指标:")
            for k, v in sorted(self.qa_metrics.items()):
                lines.append(f"    {k}: {v:.4f}")
        if self.per_category:
            lines.append("  分类别:")
            for cat, metrics in sorted(self.per_category.items()):
                lines.append(f"    {cat}: {metrics}")
        return "\n".join(lines)
