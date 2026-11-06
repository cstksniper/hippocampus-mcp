# -*- coding: utf-8 -*-
"""Scoring engine.

Combines similarity, time decay, and recall frequency to score and rank candidate
memories, simulating the hippocampus's "consolidation / decay" of memories during recall.

Two scoring modes are supported:
    - Multiplicative formula (legacy): score = sim * time_decay * recall_factor
    - Additive formula (optimized, default): score = w_sim*sim + w_time*time_decay + w_recall*recall_boost

The additive formula raised MRR@5 on the LoCoMo eval from 0.042 to 0.183 (4.4x);
the key innovation is a negative recall weight acting as a diversity penalty.
"""

from __future__ import annotations

import numpy as np


class ScoringEngine:
    """Composite memory scoring engine.

    Scoring formula (additive mode, default)::

        score = w_sim * sim + w_time * exp(-age/lambda) + w_recall * log1p(recall) * exp(-age/mu)

    - ``sim``: similarity; higher means more relevant to the current query.
    - ``exp(-age/lambda)``: time-decay term; the larger the age, the staler the memory.
    - ``log1p(recall) * exp(-age/mu)``: recall-frequency term; with a negative weight it
      acts as a diversity penalty.

    Args:
        lambda_decay: time-decay time constant (seconds), default 7776000.0 (90 days).
        mu_importance: recall-frequency decay time constant (seconds), default 15552000.0 (180 days).
        mode: scoring mode, "additive" (default, additive formula) or "multiplicative" (multiplicative formula).
        w_sim: similarity weight in additive mode, default 0.95.
        w_time: time-decay weight in additive mode, default 0.03.
        w_recall: recall-frequency weight in additive mode, default -0.02 (negative = diversity penalty).
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
        """Compute composite scores for a batch of candidate memories.

        Args:
            similarities: similarity array (N,).
            timestamps: per-memory timestamp array (N,).
            recall_counts: per-memory recall-count array (N,).
            current_time: current timestamp, used to compute memory age.
            weight_multipliers: **weight multipliers** for C5 intent routing (e.g.
                ``{"w_time": 4.0}``), applied on top of the instance's current weights;
                None/empty = behavior bit-for-bit identical to before (old call sites
                unchanged).
            strengths: O1 recall-strength array (N,). None = the term is not added
                (bit-for-bit identical to old calls); when provided it is weighted as
                ``+w_strength·log1p(strength)`` -- the biological side of "successful
                recall consolidates the trace" (testing effect). Division of labor with
                recall_counts: recall_count tracks the **candidate set** (negative weight
                = diversity penalty), strength tracks the **result set / replay**
                (positive weight = consolidation).

        Returns:
            Score array (N,).
        """
        similarities = np.asarray(similarities, dtype=np.float64).reshape(-1)
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        recall_counts = np.asarray(recall_counts, dtype=np.float64).reshape(-1)

        # Memory age (seconds), clamped to >= 0
        age = np.maximum(current_time - timestamps, 0.0)

        # Time decay: the staler the memory, the larger the decay
        time_decay = np.exp(-age / self.lambda_decay)

        # Recall-frequency factor: log1p(recall) * time decay
        recall_boost = np.log1p(recall_counts) * np.exp(-age / self.mu_importance)

        m = weight_multipliers or {}
        w_sim = self.w_sim * float(m.get("w_sim", 1.0))
        w_time = self.w_time * float(m.get("w_time", 1.0))
        w_recall = self.w_recall * float(m.get("w_recall", 1.0))
        w_strength = self.w_strength * float(m.get("w_strength", 1.0))

        # O1 recall-strength term: None -> zero vector (no weighting, old call sites unchanged)
        strength_boost = (
            np.log1p(np.maximum(np.asarray(strengths, dtype=np.float64).reshape(-1), 0.0))
            if strengths is not None
            else np.zeros_like(similarities)
        )

        if self.mode == "multiplicative":
            # Multiplicative formula (legacy, deprecated) -- no strength term (keeps the historical shape)
            recall_factor = 1.0 + recall_boost
            return similarities * time_decay * recall_factor
        else:
            # Additive formula (optimized, default)
            # Negative recall weight = diversity penalty, diversifying the retrieval results
            # O1: +w_strength·log1p(strength) = retrieval as consolidation (positive weight)
            return (
                w_sim * similarities
                + w_time * time_decay
                + w_recall * recall_boost
                + w_strength * strength_boost
            )

    def rank(self, scores: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Return the top_k (index, score) pairs by descending score.

        Args:
            scores: score array (N,).
            top_k: number of top entries to return.

        Returns:
            A ``(index, score)`` list, ordered from highest to lowest score.
        """
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = scores.shape[0]
        if n == 0:
            return []

        top_k = min(top_k, n)
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[int(i)])) for i in top_idx]
