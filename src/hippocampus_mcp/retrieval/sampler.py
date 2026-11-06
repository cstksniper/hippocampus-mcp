"""距离加权抽样。

依据相似度进行 softmax 加权抽样：距离越近（相似度越高）被抽中概率越大，
同时通过 ``epsilon`` 为低相似度记忆保留探索能力，避免检索陷入局部最优。
"""

from __future__ import annotations

import numpy as np


class DistanceWeightedSampler:
    """距离加权抽样器。

    Args:
        beta: softmax 的温度系数。``beta`` 越大，分布越倾向于高相似度项；
            ``beta`` 趋近 0 时分布趋近均匀。
        epsilon: 探索项。加到 softmax 概率上后再归一化，保证每个候选
            都有非零的被抽中概率。
    """

    def __init__(self, beta: float = 1.0, epsilon: float = 0.1) -> None:
        self.beta = beta
        self.epsilon = epsilon

    def sample(self, similarities: np.ndarray, k: int = 5) -> list[int]:
        """按 softmax(beta * sim) + epsilon 抽样 k 个索引。

        Args:
            similarities: 候选项的相似度数组（1D）。
            k: 抽样数量。

        Returns:
            被抽中的候选索引列表（无放回抽样，索引唯一）。
        """
        similarities = np.asarray(similarities, dtype=np.float64).reshape(-1)
        n = similarities.shape[0]
        if n == 0:
            return []

        k = min(k, n)

        # softmax(beta * sim)，减去最大值保证数值稳定
        logits = self.beta * similarities
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)

        # 加上 epsilon 探索项后重新归一化
        probs = probs + self.epsilon
        probs = probs / np.sum(probs)

        # 无放回加权抽样
        chosen = np.random.choice(n, size=k, replace=False, p=probs)
        return chosen.tolist()
