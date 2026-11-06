# -*- coding: utf-8 -*-
"""Main evaluation-framework module.

Orchestrates baseline methods and evaluation metrics into a unified
evaluation pipeline; supports several data formats:

1. Built-in synthetic dataset (for quick validation)
2. LoCoMo format (snap-research/LoCoMo)
3. LongMemEval format (xiaowu0162/LongMemEval)

Evaluation pipeline:
    for each method:
        1. reset the method's state
        2. ingest the conversation history item by item (store)
        3. retrieve for each evaluation question (retrieve)
        4. compute retrieval metrics (MRR, Hit@k, Recall@k, NDCG@k)
        5. (optional) compute QA metrics (F1, EM) -- requires an answer
           generation module
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import numpy as np

from .baselines import (
    BaseRetriever,
    FullMethod,
    LNNBaseline,
    MeanEmbedding,
    PureEmbedding,
    RandomBaseline,
    TimeDecayBaseline,
)
from .metrics import (
    EvalReport,
    exact_match,
    f1_score,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_all_at_k,
    recall_any_at_k,
)


# ==================================================================
# Data-format adapters
# ==================================================================

class EvalSample:
    """A single evaluation sample."""

    def __init__(
        self,
        sample_id: str,
        messages: list[dict[str, str]],
        questions: list[dict[str, Any]],
    ) -> None:
        self.sample_id = sample_id
        self.messages = messages  # [{"role": "user"/"assistant", "content": "..."}]
        self.questions = questions  # [{"question": "...", "answer": "...", "category": int, "evidence": [...]}]


def load_locomo(data_path: str) -> list[EvalSample]:
    """Load a LoCoMo-format dataset.

    LoCoMo JSON structure:
        [{
            "sample_id": "...",
            "conversation": {
                "session_1": [{"name": "...", "text": "..."}],
                "session_2": [...],
                ...
            },
            "qa": [{"question": "...", "answer": "...", "category": 1, "evidence": [...]}]
        }]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        conv = item.get("conversation", {})
        messages = []
        for key in sorted(conv.keys()):
            if key.startswith("session_") and not key.endswith(("_date_time", "_observation", "_summary")):
                session = conv[key]
                if isinstance(session, list):
                    for turn in session:
                        role = "user" if turn.get("name") == conv.get("speaker_a") else "assistant"
                        messages.append({
                            "role": role,
                            "content": turn.get("text", ""),
                            "turn_id": turn.get("dia_id", ""),
                        })

        questions = []
        for qa in item.get("qa", []):
            questions.append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "category": qa.get("category", 2),
                "evidence": qa.get("evidence", []),
            })

        samples.append(EvalSample(
            sample_id=item.get("sample_id", ""),
            messages=messages,
            questions=questions,
        ))

    return samples


def load_longmemeval(data_path: str) -> list[EvalSample]:
    """Load a LongMemEval-format dataset.

    LongMemEval JSON structure:
        [{
            "question_id": "...",
            "question_type": "...",
            "question": "...",
            "answer": "...",
            "haystack_sessions": [[{"role": "...", "content": "...", "has_answer": bool}]],
            "answer_session_ids": [...]
        }]

    Each instance is converted into one EvalSample (one question).
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        messages = []
        session_idx = 0
        for session in item.get("haystack_sessions", []):
            for turn in session:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                    "turn_id": f"sess_{session_idx}_{turn.get('role', '')}",
                    "has_answer": turn.get("has_answer", False),
                })
            session_idx += 1

        # LongMemEval has only one question per instance
        samples.append(EvalSample(
            sample_id=item.get("question_id", ""),
            messages=messages,
            questions=[{
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("question_type", ""),
                "evidence": item.get("answer_session_ids", []),
            }],
        ))

    return samples


def generate_synthetic_dataset(
    n_conversations: int = 20,
    n_turns_per_conv: int = 15,
    n_questions_per_conv: int = 5,
    seed: int = 42,
) -> list[EvalSample]:
    """Generate a synthetic conversation dataset (for quickly validating
    the evaluation framework).

    Each conversation revolves around a few topics; questions require
    retrieving answers from the conversation history.
    """
    rng = random.Random(seed)

    topics = [
        ("Python 编程", ["列表推导式", "装饰器", "生成器", "上下文管理器", "类型注解"]),
        ("机器学习", ["梯度下降", "正则化", "交叉验证", "特征工程", "过拟合"]),
        ("数据库设计", ["范式", "索引", "事务", "外键", "查询优化"]),
        ("Web 开发", ["REST API", "JWT 认证", "CORS", "WebSocket", "缓存策略"]),
        ("DevOps", ["Docker", "Kubernetes", "CI/CD", "监控", "日志聚合"]),
        ("算法", ["动态规划", "贪心算法", "图遍历", "排序", "二分查找"]),
        ("操作系统", ["进程调度", "内存管理", "文件系统", "IO 模型", "锁机制"]),
        ("网络安全", ["SQL 注入", "XSS", "CSRF", "HTTPS", "对称加密"]),
    ]

    samples = []
    for conv_idx in range(n_conversations):
        # randomly pick 2-3 topics
        n_topics = rng.randint(2, 3)
        chosen = rng.sample(topics, n_topics)

        messages = []
        facts = {}  # topic_subtopic -> content

        for turn_idx in range(n_turns_per_conv):
            topic_name, subtopics = rng.choice(chosen)
            sub = rng.choice(subtopics)
            role = "user" if turn_idx % 2 == 0 else "assistant"

            if role == "user":
                content = f"请问{topic_name}中的{sub}是什么?"
            else:
                detail = rng.choice([
                    f"{sub}是一种重要的概念, 用于解决实际问题",
                    f"在{topic_name}中, {sub}指的是一种核心机制",
                    f"{sub}的主要作用是提高系统的效率和可靠性",
                    f"使用{sub}时需要注意性能和安全性",
                    f"{sub}是{topic_name}领域的基础知识之一",
                ])
                content = detail
                facts[f"{topic_name}_{sub}"] = detail

            messages.append({
                "role": role,
                "content": content,
                "turn_id": f"turn_{turn_idx}",
            })

        # generate questions
        questions = []
        fact_items = list(facts.items())
        rng.shuffle(fact_items)

        for q_idx in range(min(n_questions_per_conv, len(fact_items))):
            key, answer = fact_items[q_idx]
            topic_name, sub = key.rsplit("_", 1)

            q = rng.choice([
                f"之前提到的{sub}是什么?",
                f"关于{sub}你说了什么?",
                f"{topic_name}中的{sub}有什么作用?",
                f"回忆一下{sub}的内容",
            ])

            questions.append({
                "question": q,
                "answer": answer,
                "category": 2,  # single-hop
                "evidence": [f"turn_{rng.randint(0, n_turns_per_conv - 1)}"],
            })

        samples.append(EvalSample(
            sample_id=f"synth_{conv_idx}",
            messages=messages,
            questions=questions,
        ))

    return samples


# ==================================================================
# Evaluator
# ==================================================================

class Evaluator:
    """Memory-system evaluator.

    Args:
        dim: embedding dimension (only effective for the hash backend).
        top_k_values: list of k values to evaluate.
        embedder_backend: embedding backend, "sentence_transformer"
            (default) or "hash".
    """

    def __init__(
        self,
        dim: int = 384,
        top_k_values: list[int] | None = None,
        embedder_backend: str = "sentence_transformer",
    ) -> None:
        self.dim = dim
        self.embedder_backend = embedder_backend
        self.top_k_values = top_k_values or [1, 3, 5, 10]

    def get_methods(self) -> dict[str, BaseRetriever]:
        """Return all methods under evaluation."""
        backend = self.embedder_backend
        return {
            "Random": RandomBaseline(dim=self.dim, embedder_backend=backend),
            "PureEmbedding": PureEmbedding(dim=self.dim, embedder_backend=backend),
            "TimeDecay": TimeDecayBaseline(dim=self.dim, embedder_backend=backend),
            "MeanEmbedding": MeanEmbedding(dim=self.dim, embedder_backend=backend),
            "LNN-CfC": LNNBaseline(dim=self.dim, embedder_backend=backend),
            "HippocampusMCP": FullMethod(dim=self.dim, embedder_backend=backend),
        }

    def evaluate_sample(
        self,
        method: BaseRetriever,
        sample: EvalSample,
    ) -> dict[str, Any]:
        """Evaluate a single sample.

        Pipeline:
            1. ingest all conversation messages
            2. retrieve for each question
            3. compute metrics

        Returns:
            A dict containing the retrieval results.
        """
        # ingest the conversation history
        method.reset()
        doc_id_to_content = {}
        for msg in sample.messages:
            doc_id = method.store(msg["content"])
            doc_id_to_content[doc_id] = msg["content"]

        # retrieve for each question
        rankings = []
        ground_truths = []
        qa_results = []

        for q in sample.questions:
            # retrieve
            results = method.retrieve(q["question"], top_k=max(self.top_k_values))
            ranked_ids = [doc_id for doc_id, _ in results]
            rankings.append(ranked_ids)

            # correct documents = documents containing the answer keywords
            answer_keywords = q["answer"]
            gt_ids = []
            for doc_id, content in doc_id_to_content.items():
                # simple match: a keyword from the answer appears in the document
                if any(kw in content for kw in answer_keywords.split() if len(kw) > 1):
                    gt_ids.append(doc_id)
            if not gt_ids:
                # degenerate case: pick one at random as ground truth
                gt_ids = list(doc_id_to_content.keys())[:1]
            ground_truths.append(gt_ids)

            # QA metrics (use the top-1 retrieval result as the answer)
            if results:
                top_content = doc_id_to_content.get(results[0][0], "")
                qa_results.append({
                    "f1": f1_score(top_content, q["answer"]),
                    "em": exact_match(top_content, q["answer"]),
                })
            else:
                qa_results.append({"f1": 0.0, "em": 0.0})

        # compute metrics
        metrics: dict[str, float] = {}
        for k in self.top_k_values:
            metrics[f"MRR@{k}"] = mrr(
                [r[:k] for r in rankings], ground_truths
            )
            metrics[f"Hit@{k}"] = hit_at_k(rankings, ground_truths, k)
            metrics[f"Recall_any@{k}"] = recall_any_at_k(rankings, ground_truths, k)
            metrics[f"Recall_all@{k}"] = recall_all_at_k(rankings, ground_truths, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(rankings, ground_truths, k)

        # QA metrics
        if qa_results:
            metrics["F1"] = np.mean([r["f1"] for r in qa_results])
            metrics["EM"] = np.mean([r["em"] for r in qa_results])

        return metrics

    def evaluate(
        self,
        samples: list[EvalSample],
        methods: dict[str, BaseRetriever] | None = None,
    ) -> list[EvalReport]:
        """Evaluate all methods on the given dataset.

        Args:
            samples: list of evaluation samples.
            methods: dict of methods to evaluate. None means use
                get_methods().

        Returns:
            A list of evaluation reports, one per method.
        """
        if methods is None:
            methods = self.get_methods()

        reports = []
        for method_name, method in methods.items():
            print(f"\n  评估方法: {method_name} ...")

            all_metrics: dict[str, list[float]] = {}
            per_category: dict[str, dict[str, list[float]]] = {}

            for sample in samples:
                metrics = self.evaluate_sample(method, sample)

                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)

                # aggregate by question category
                for q in sample.questions:
                    cat = str(q.get("category", "unknown"))
                    per_category.setdefault(cat, {})
                    for k, v in metrics.items():
                        per_category[cat].setdefault(k, []).append(v)

            # aggregate
            report = EvalReport(method_name)
            for k, values in all_metrics.items():
                report.retrieval_metrics[k] = float(np.mean(values))

            for cat, metrics_dict in per_category.items():
                report.per_category[cat] = {
                    k: float(np.mean(v)) for k, v in metrics_dict.items()
                }

            reports.append(report)
            print(f"    MRR@5={report.retrieval_metrics.get('MRR@5', 0):.4f}  "
                  f"Hit@5={report.retrieval_metrics.get('Hit@5', 0):.4f}  "
                  f"NDCG@5={report.retrieval_metrics.get('NDCG@5', 0):.4f}")

        return reports

    @staticmethod
    def generate_comparison_table(reports: list[EvalReport]) -> str:
        """Generate a Markdown comparison table."""
        if not reports:
            return "无评估结果"

        # collect all metric names
        all_metrics = set()
        for r in reports:
            all_metrics.update(r.retrieval_metrics.keys())
        all_metrics = sorted(all_metrics)

        # header
        lines = [
            "| 方法 | " + " | ".join(all_metrics) + " |",
            "|------|" + "|".join(["------"] * len(all_metrics)) + "|",
        ]

        # data rows
        for r in reports:
            row = [r.method_name]
            for m in all_metrics:
                val = r.retrieval_metrics.get(m, 0.0)
                row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
