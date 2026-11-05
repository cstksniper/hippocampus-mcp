# -*- coding: utf-8 -*-
"""评估框架主模块.

将基线方法与评估指标编排为统一的评估流程, 支持多种数据格式:

1. 内置合成数据集 (用于快速验证)
2. LoCoMo 格式 (snap-research/LoCoMo)
3. LongMemEval 格式 (xiaowu0162/LongMemEval)

评估流程:
    对每个方法:
        1. 重置方法状态
        2. 逐条摄入对话历史 (store)
        3. 对每个评估问题执行检索 (retrieve)
        4. 计算检索指标 (MRR, Hit@k, Recall@k, NDCG@k)
        5. (可选) 计算 QA 指标 (F1, EM) — 需要答案生成模块
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
# 数据格式适配器
# ==================================================================

class EvalSample:
    """单个评估样本."""

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
    """加载 LoCoMo 格式数据集.

    LoCoMo JSON 结构:
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
    """加载 LongMemEval 格式数据集.

    LongMemEval JSON 结构:
        [{
            "question_id": "...",
            "question_type": "...",
            "question": "...",
            "answer": "...",
            "haystack_sessions": [[{"role": "...", "content": "...", "has_answer": bool}]],
            "answer_session_ids": [...]
        }]

    每个 instance 转换为一个 EvalSample (一个 question).
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

        # LongMemEval 每个 instance 只有一个 question
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
    """生成合成对话数据集 (用于快速验证评估框架).

    每段对话围绕若干话题展开, 问题需要从对话历史中检索答案.
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
        # 随机选 2-3 个话题
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

        # 生成问题
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
# 评估器
# ==================================================================

class Evaluator:
    """记忆系统评估器.

    Args:
        dim: 嵌入维度 (仅对 hash 后端有效).
        top_k_values: 评估的 k 值列表.
        embedder_backend: 嵌入后端, "sentence_transformer" (默认) 或 "hash".
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
        """返回所有待评估的方法."""
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
        """评估单个样本.

        流程:
            1. 摄入所有对话消息
            2. 对每个问题执行检索
            3. 计算指标

        Returns:
            包含检索结果的字典.
        """
        # 摄入对话历史
        method.reset()
        doc_id_to_content = {}
        for msg in sample.messages:
            doc_id = method.store(msg["content"])
            doc_id_to_content[doc_id] = msg["content"]

        # 对每个问题检索
        rankings = []
        ground_truths = []
        qa_results = []

        for q in sample.questions:
            # 检索
            results = method.retrieve(q["question"], top_k=max(self.top_k_values))
            ranked_ids = [doc_id for doc_id, _ in results]
            rankings.append(ranked_ids)

            # 正确文档 = 包含答案关键词的文档
            answer_keywords = q["answer"]
            gt_ids = []
            for doc_id, content in doc_id_to_content.items():
                # 简单匹配: 答案中的关键词出现在文档中
                if any(kw in content for kw in answer_keywords.split() if len(kw) > 1):
                    gt_ids.append(doc_id)
            if not gt_ids:
                # 退化: 随机选一个作为 ground truth
                gt_ids = list(doc_id_to_content.keys())[:1]
            ground_truths.append(gt_ids)

            # QA 指标 (用 top-1 检索结果作为答案)
            if results:
                top_content = doc_id_to_content.get(results[0][0], "")
                qa_results.append({
                    "f1": f1_score(top_content, q["answer"]),
                    "em": exact_match(top_content, q["answer"]),
                })
            else:
                qa_results.append({"f1": 0.0, "em": 0.0})

        # 计算指标
        metrics: dict[str, float] = {}
        for k in self.top_k_values:
            metrics[f"MRR@{k}"] = mrr(
                [r[:k] for r in rankings], ground_truths
            )
            metrics[f"Hit@{k}"] = hit_at_k(rankings, ground_truths, k)
            metrics[f"Recall_any@{k}"] = recall_any_at_k(rankings, ground_truths, k)
            metrics[f"Recall_all@{k}"] = recall_all_at_k(rankings, ground_truths, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(rankings, ground_truths, k)

        # QA 指标
        if qa_results:
            metrics["F1"] = np.mean([r["f1"] for r in qa_results])
            metrics["EM"] = np.mean([r["em"] for r in qa_results])

        return metrics

    def evaluate(
        self,
        samples: list[EvalSample],
        methods: dict[str, BaseRetriever] | None = None,
    ) -> list[EvalReport]:
        """评估所有方法在给定数据集上的表现.

        Args:
            samples: 评估样本列表.
            methods: 待评估的方法字典. None 则使用 get_methods().

        Returns:
            每个方法的评估报告列表.
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

                # 按问题类别统计
                for q in sample.questions:
                    cat = str(q.get("category", "unknown"))
                    per_category.setdefault(cat, {})
                    for k, v in metrics.items():
                        per_category[cat].setdefault(k, []).append(v)

            # 聚合
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
        """生成 Markdown 格式的对比表."""
        if not reports:
            return "无评估结果"

        # 收集所有指标名
        all_metrics = set()
        for r in reports:
            all_metrics.update(r.retrieval_metrics.keys())
        all_metrics = sorted(all_metrics)

        # 表头
        lines = [
            "| 方法 | " + " | ".join(all_metrics) + " |",
            "|------|" + "|".join(["------"] * len(all_metrics)) + "|",
        ]

        # 数据行
        for r in reports:
            row = [r.method_name]
            for m in all_metrics:
                val = r.retrieval_metrics.get(m, 0.0)
                row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
