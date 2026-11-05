# -*- coding: utf-8 -*-
"""打分引擎.

综合相似度、时间衰减与回忆频次, 对候选记忆进行打分与排序,
模拟海马体在回忆时对记忆的"巩固 / 衰减"机制.

支持两种打分模式:
    - 乘法公式 (legacy): score = sim * time_decay * recall_factor
    - 加法公式 (优化, 默认): score = w_sim*sim + w_time*time_decay + w_recall*recall_boost

加法公式在 LoCoMo 评测中 MRR@5 从 0.042 提升至 0.183 (4.4倍),
关键创新是负 recall 权重作为多样性惩罚.
"""

from __future__ import annotations

import numpy as np


class ScoringEngine:
    """记忆综合打分引擎.

    打分公式 (加法模式, 默认)::

        score = w_sim * sim + w_time * exp(-age/lambda) + w_recall * log1p(recall) * exp(-age/mu)

    - ``sim``: 相似度, 越高代表与当前查询越相关.
    - ``exp(-age/lambda)``: 时间衰减项, age 越大记忆越陈旧.
    - ``log1p(recall) * exp(-age/mu)``: 回忆频次项, 负权重时作为多样性惩罚.

    Args:
        lambda_decay: 时间衰减时间常数 (秒), 默认 7776000.0 (90天).
        mu_importance: 回忆频次衰减时间常数 (秒), 默认 15552000.0 (180天).
        mode: 打分模式, "additive" (默认, 加法公式) 或 "multiplicative" (乘法公式).
        w_sim: 加法模式下相似度权重, 默认 0.95.
        w_time: 加法模式下时间衰减权重, 默认 0.03.
        w_recall: 加法模式下回忆频次权重, 默认 -0.02 (负值=多样性惩罚).
    """

    def __init__(
        self,
        lambda_decay: float = 7776000.0,
        mu_importance: float = 15552000.0,
        mode: str = "additive",
        w_sim: float = 0.95,
        w_time: float = 0.03,
        w_recall: float = -0.02,
        w_strength: float = 0.03,
    ) -> None:
        self.lambda_decay = lambda_decay
        self.mu_importance = mu_importance
        self.mode = mode
        self.w_sim = w_sim
        self.w_time = w_time
        self.w_recall = w_recall
        self.w_strength = w_strength

    def score(
        self,
        similarities: np.ndarray,
        timestamps: np.ndarray,
        recall_counts: np.ndarray,
        current_time: float,
        weight_multipliers: dict[str, float] | None = None,
        strengths: np.ndarray | None = None,
    ) -> np.ndarray:
        """对一批候选记忆计算综合得分.

        Args:
            similarities: 相似度数组 (N,).
            timestamps: 各记忆的时间戳数组 (N,).
            recall_counts: 各记忆的回忆次数数组 (N,).
            current_time: 当前时间戳, 用于计算记忆年龄.
            weight_multipliers: C5 意图路由的**权重乘子**（如
                ``{"w_time": 4.0}``），作用于实例当前权重；None/空表 = 行为
                与原先逐位一致（老调用点零改动）。
            strengths: O1 检索强度数组 (N,)。None = 不加该项（与旧调用
                逐位一致）；提供时按 ``+w_strength·log1p(强度)`` 加权——
                生物侧"成功回忆巩固痕迹"（测试效应）。与 recall_counts
                的分工：recall_count 记**候选集**（负权＝多样性惩罚），
                strength 记**结果集/重放**（正权＝巩固）。

        Returns:
            得分数组 (N,).
        """
        similarities = np.asarray(similarities, dtype=np.float64).reshape(-1)
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        recall_counts = np.asarray(recall_counts, dtype=np.float64).reshape(-1)

        # 记忆年龄 (秒), 不小于 0
        age = np.maximum(current_time - timestamps, 0.0)

        # 时间衰减: 记忆越陈旧, 衰减越大
        time_decay = np.exp(-age / self.lambda_decay)

        # 回忆频次因子: log1p(recall) * 时间衰减
        recall_boost = np.log1p(recall_counts) * np.exp(-age / self.mu_importance)

        m = weight_multipliers or {}
        w_sim = self.w_sim * float(m.get("w_sim", 1.0))
        w_time = self.w_time * float(m.get("w_time", 1.0))
        w_recall = self.w_recall * float(m.get("w_recall", 1.0))
        w_strength = self.w_strength * float(m.get("w_strength", 1.0))

        # O1 检索强度项：None → 零向量（不加权，旧调用点零改动）
        strength_boost = (
            np.log1p(np.maximum(np.asarray(strengths, dtype=np.float64).reshape(-1), 0.0))
            if strengths is not None
            else np.zeros_like(similarities)
        )

        if self.mode == "multiplicative":
            # 乘法公式 (legacy, 已弃用)——不引入强度项（保持历史形态）
            recall_factor = 1.0 + recall_boost
            return similarities * time_decay * recall_factor
        else:
            # 加法公式 (优化, 默认)
            # 负 recall 权重 = 多样性惩罚, 使检索结果更多样化
            # O1: +w_strength·log1p(strength) = 检索即巩固（正权）
            return (
                w_sim * similarities
                + w_time * time_decay
                + w_recall * recall_boost
                + w_strength * strength_boost
            )

    def rank(self, scores: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """按得分降序返回前 top_k 个 (索引, 分数) 对.

        Args:
            scores: 得分数组 (N,).
            top_k: 返回的顶部数量.

        Returns:
            ``(index, score)`` 列表, 按分数从高到低排列.
        """
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = scores.shape[0]
        if n == 0:
            return []

        top_k = min(top_k, n)
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[int(i)])) for i in top_idx]
