"""Distance-weighted sampling.

Performs softmax-weighted sampling by similarity: the closer (more similar) a memory is,
the higher its chance of being drawn, while ``epsilon`` preserves exploration ability
for low-similarity memories, keeping retrieval out of local optima.
"""

from __future__ import annotations

import numpy as np


class DistanceWeightedSampler:
    """Distance-weighted sampler.

    Args:
        beta: softmax temperature coefficient. The larger ``beta``, the more the
            distribution favors high-similarity items; as ``beta`` approaches 0 the
            distribution approaches uniform.
        epsilon: exploration term. Added to the softmax probabilities before
            renormalizing, guaranteeing every candidate keeps a nonzero chance of being
            drawn.
    """

    def __init__(self, beta: float = 1.0, epsilon: float = 0.1) -> None:
        self.beta = beta
        self.epsilon = epsilon

    def sample(self, similarities: np.ndarray, k: int = 5) -> list[int]:
        """Sample k indices via softmax(beta * sim) + epsilon.

        Args:
            similarities: similarity array of the candidates (1D).
            k: number of samples to draw.

        Returns:
            List of drawn candidate indices (sampling without replacement; indices unique).
        """
        similarities = np.asarray(similarities, dtype=np.float64).reshape(-1)
        n = similarities.shape[0]
        if n == 0:
            return []

        k = min(k, n)

        # softmax(beta * sim); subtracting the max for numerical stability
        logits = self.beta * similarities
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)

        # Add the epsilon exploration term, then renormalize
        probs = probs + self.epsilon
        probs = probs / np.sum(probs)

        # Weighted sampling without replacement
        chosen = np.random.choice(n, size=k, replace=False, p=probs)
        return chosen.tolist()
