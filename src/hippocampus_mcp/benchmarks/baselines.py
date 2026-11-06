# -*- coding: utf-8 -*-
"""Baseline method implementations.

Provides comparison baselines for evaluating hippocampus_mcp (LNN + scoring):
    - RandomBaseline:       random retrieval
    - PureEmbedding:        pure text-embedding cosine similarity
    - TimeDecayBaseline:    embedding similarity × time decay
    - MeanEmbedding:        mean of the most recent N embeddings as the query
    - LNNBaseline:          CfC LNN query (no scoring)
    - FullMethod:           LNN + scoring + sampling (hippocampus_mcp full method)

All baselines implement the same interface:
    store(content, timestamp) -> None
    retrieve(query, top_k) -> list[(doc_id, score)]
    reset() -> None
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import uuid
from typing import Optional

import numpy as np

# Ensure the project root is in sys.path (for importing the embeddings module)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the main project's embedders
from hippocampus_mcp.engine.embeddings import HashEmbedder as _ProjectHashEmbedder
from hippocampus_mcp.engine.embeddings import SentenceTransformerEmbedder as _STEmbedder

# Lazily import torch and ncps; only the LNN baselines need them
_torch = None
_ncps = None


def _ensure_torch():
    global _torch, _ncps
    if _torch is None:
        import torch
        import torch.nn.functional as F
        _torch = torch
        from ncps.torch import CfC
        from ncps.wirings import AutoNCP
        _ncps = (CfC, AutoNCP)
        return torch, F, CfC, AutoNCP
    return _torch, _torch.nn.functional, _ncps[0], _ncps[1]


def create_eval_embedder(backend: str = "sentence_transformer", dim: int = 384):
    """Create an embedder for evaluation.

    Args:
        backend: "sentence_transformer" or "hash".
        dim: the hash backend's dimension (sentence_transformer is decided by the model).

    Returns:
        An embedder instance.
    """
    if backend == "sentence_transformer":
        return _STEmbedder()
    else:
        return _ProjectHashEmbedder(dim=dim, salt="eval")


class BaseRetriever:
    """Retrieval base class, defining the unified interface."""

    name: str = "base"

    def __init__(
        self,
        dim: int = 384,
        embedder_backend: str = "sentence_transformer",
    ) -> None:
        self.embedder_backend = embedder_backend
        if embedder_backend == "sentence_transformer":
            self.embedder = _STEmbedder()
            self.dim = self.embedder.dim
        else:
            self.embedder = _ProjectHashEmbedder(dim=dim, salt="eval")
            self.dim = dim
        self._docs: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._ids: list[str] = []

    def store(self, content: str, timestamp: float | None = None) -> str:
        emb = self.embedder.embed(content)
        doc_id = str(uuid.uuid4())
        self._docs.append(content)
        self._embeddings.append(emb)
        self._timestamps.append(timestamp if timestamp is not None else time.time())
        self._ids.append(doc_id)
        return doc_id

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        raise NotImplementedError

    def reset(self) -> None:
        self._docs.clear()
        self._embeddings.clear()
        self._timestamps.clear()
        self._ids.clear()

    def _get_reference_time(self) -> float:
        """Get the reference time: use the newest stored timestamp, not the wall clock.

        This ensures time decay is based on the data's internal time span (e.g. sessions
        in LoCoMo weeks apart), not on the huge gap between when the data was generated
        and the actual current time.
        """
        if self._timestamps:
            return max(self._timestamps)
        return time.time()

    def _cosine_sim(self, query_emb: np.ndarray) -> np.ndarray:
        if not self._embeddings:
            return np.array([])
        matrix = np.vstack(self._embeddings)
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(query_emb)
        denom = norms * q_norm
        denom[denom == 0] = 1e-12
        return (matrix @ query_emb) / denom


class RandomBaseline(BaseRetriever):
    """Random-retrieval baseline."""

    name = "Random"

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        n = len(self._ids)
        if n == 0:
            return []
        k = min(top_k, n)
        indices = np.random.choice(n, size=k, replace=False)
        return [(self._ids[i], 1.0 / (j + 1)) for j, i in enumerate(indices)]


class PureEmbedding(BaseRetriever):
    """Pure text-embedding cosine similarity retrieval."""

    name = "PureEmbedding"

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        q_emb = self.embedder.embed(query)
        sims = self._cosine_sim(q_emb)
        k = min(top_k, len(sims))
        top_idx = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in top_idx]


class TimeDecayBaseline(BaseRetriever):
    """Embedding similarity × time decay."""

    name = "TimeDecay"

    def __init__(
        self,
        dim: int = 384,
        lambda_decay: float = 604800.0,
        embedder_backend: str = "sentence_transformer",
    ) -> None:
        super().__init__(dim, embedder_backend)
        self.lambda_decay = lambda_decay

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        q_emb = self.embedder.embed(query)
        sims = self._cosine_sim(q_emb)
        current_time = self._get_reference_time()
        ages = np.array([current_time - t for t in self._timestamps])
        ages = np.maximum(ages, 0.0)
        decay = np.exp(-ages / self.lambda_decay)
        scores = sims * decay
        k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]


class MeanEmbedding(BaseRetriever):
    """Mean of the most recent N embeddings as the query (sliding window)."""

    name = "MeanEmbedding"

    def __init__(
        self,
        dim: int = 384,
        window_size: int = 5,
        embedder_backend: str = "sentence_transformer",
    ) -> None:
        super().__init__(dim, embedder_backend)
        self.window_size = window_size

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        # query vector = fusion of the text embedding and the recent-window mean
        q_text = self.embedder.embed(query)
        recent = self._embeddings[-self.window_size :]
        if recent:
            q_recent = np.mean(recent, axis=0)
            q_vec = 0.5 * q_text + 0.5 * q_recent
            norm = np.linalg.norm(q_vec)
            if norm > 1e-12:
                q_vec = q_vec / norm
        else:
            q_vec = q_text

        sims = self._cosine_sim(q_vec)
        k = min(top_k, len(sims))
        top_idx = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in top_idx]


class LNNBaseline(BaseRetriever):
    """CfC LNN query vector (no scoring engine).

    Uses CfC's hidden state h(t) as the query vector, weighted-fused with the text
    embedding.
    """

    name = "LNN-CfC"

    def __init__(
        self,
        dim: int = 384,
        n_neurons: int = 512,
        window_size: int = 5,
        embedder_backend: str = "sentence_transformer",
    ) -> None:
        super().__init__(dim, embedder_backend)
        torch, F, CfC, AutoNCP = _ensure_torch()
        self._F = F
        # AutoNCP requires output_dim < n_neurons - 2
        actual_n_neurons = max(n_neurons, self.dim + 3)
        wiring = AutoNCP(actual_n_neurons, self.dim)
        self.cfc = CfC(self.dim, wiring, batch_first=True)
        self.hx = None
        self.last_output = torch.zeros(self.dim)
        self.window_size = window_size
        self.window: list = []

    def store(self, content: str, timestamp: float | None = None) -> str:
        doc_id = super().store(content, timestamp)
        # update the LNN state
        emb = self.embedder.embed(content)
        x = self._torch_tensor(emb)
        output, self.hx = self.cfc(x, self.hx)
        self.hx = self.hx.detach()
        self.last_output = output[0, -1, :].detach()
        self.window.append(self.last_output.clone())
        if len(self.window) > self.window_size:
            self.window.pop(0)
        return doc_id

    def _torch_tensor(self, emb: np.ndarray):
        torch = _ensure_torch()[0]
        return torch.from_numpy(emb).float().unsqueeze(0).unsqueeze(0)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        torch = _ensure_torch()[0]
        q_text = self.embedder.embed(query)

        # LNN query vector
        if self.window:
            window_mean = torch.stack(list(self.window)).mean(dim=0)
            lnn_query = 0.5 * self.last_output + 0.5 * window_mean
            lnn_np = self._F.normalize(lnn_query, dim=0).numpy().astype(np.float64)
        else:
            lnn_np = q_text

        # fusion: 0.6 * text + 0.4 * LNN
        q_vec = 0.6 * q_text + 0.4 * lnn_np
        norm = np.linalg.norm(q_vec)
        if norm > 1e-12:
            q_vec = q_vec / norm

        sims = self._cosine_sim(q_vec)
        k = min(top_k, len(sims))
        top_idx = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in top_idx]

    def reset(self) -> None:
        super().reset()
        self.hx = None
        self.last_output = _ensure_torch()[0].zeros(self.dim)
        self.window.clear()

    def load_trained_weights(self, model_path: str) -> None:
        """Load the trained CfC weights.

        Args:
            model_path: path to the .pt file saved by train_cfc.py.
        """
        torch = _ensure_torch()[0]
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        # keys saved by the training script carry the "cfc." prefix; strip it before loading into self.cfc
        cfc_state = {}
        for k, v in checkpoint["model_state"].items():
            if k.startswith("cfc."):
                cfc_state[k[4:]] = v  # drop the "cfc." prefix
        if cfc_state:
            self.cfc.load_state_dict(cfc_state)
            print(f"  [LNNBaseline] training weights loaded: {model_path} (val_loss={checkpoint.get('best_val_loss', '?')})")
        else:
            self.cfc.load_state_dict(checkpoint["model_state"])
            print(f"  [LNNBaseline] training weights loaded (direct mode): {model_path}")


class FullMethod(LNNBaseline):
    """hippocampus_mcp full method: LNN + scoring engine + sampling.

    On top of the LNN query, adds time-decay and recall-frequency scoring.
    Supports conditional time decay: time weights adjusted dynamically by question type.
    """

    name = "HippocampusMCP"

    def __init__(
        self,
        dim: int = 384,
        n_neurons: int = 512,
        window_size: int = 5,
        lambda_decay: float = 604800.0,
        mu_importance: float = 1209600.0,
        embedder_backend: str = "sentence_transformer",
        w_sim: float = 0.95,
        w_time: float = 0.03,
        w_recall: float = -0.02,
        conditional_time: bool = False,
    ) -> None:
        super().__init__(dim, n_neurons, window_size, embedder_backend)
        self.lambda_decay = lambda_decay
        self.mu_importance = mu_importance
        self.w_sim = w_sim
        self.w_time = w_time
        self.w_recall = w_recall
        self.conditional_time = conditional_time
        # conditional time-decay weight map (by LoCoMo category)
        # category=2 (temporal reasoning): high time weight; others: extremely low
        self._cat_time_weights = {
            1: 0.01,   # multi-hop reasoning — time not important
            2: 0.10,   # temporal reasoning — time critical
            3: 0.01,   # open-domain — time not important
            4: 0.01,   # single-hop reasoning — time not important
            5: 0.01,   # adversarial questions — time not important
        }
        self._recall_counts: list[int] = []

    def store(self, content: str, timestamp: float | None = None) -> str:
        doc_id = super().store(content, timestamp)
        self._recall_counts.append(0)
        return doc_id

    def retrieve(
        self, query: str, top_k: int = 5, category: int | None = None
    ) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []

        # reuse the LNN similarity computation
        torch = _ensure_torch()[0]
        q_text = self.embedder.embed(query)

        if self.window:
            window_mean = torch.stack(list(self.window)).mean(dim=0)
            lnn_query = 0.5 * self.last_output + 0.5 * window_mean
            lnn_np = self._F.normalize(lnn_query, dim=0).numpy().astype(np.float64)
        else:
            lnn_np = q_text

        q_vec = 0.6 * q_text + 0.4 * lnn_np
        norm = np.linalg.norm(q_vec)
        if norm > 1e-12:
            q_vec = q_vec / norm

        sims = self._cosine_sim(q_vec)

        # determine the time weight: conditional decay or the fixed weight
        if self.conditional_time and category is not None:
            w_time_eff = self._cat_time_weights.get(category, self.w_time)
        else:
            w_time_eff = self.w_time

        # scoring: weighted additive formula (similarity dominant, time/recall auxiliary)
        current_time = self._get_reference_time()
        ages = np.maximum(
            np.array([current_time - t for t in self._timestamps]), 0.0
        )
        time_decay = np.exp(-ages / self.lambda_decay)
        recall_arr = np.array(self._recall_counts, dtype=np.float64)
        recall_boost = np.log1p(recall_arr) * np.exp(-ages / self.mu_importance)

        # weighting: configurable similarity/time/recall weights
        scores = self.w_sim * sims + w_time_eff * time_decay + self.w_recall * recall_boost

        k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:k]

        # update the recall counts
        for i in top_idx:
            self._recall_counts[i] += 1

        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def reset(self) -> None:
        super().reset()
        self._recall_counts.clear()


class FullMethodRejection(FullMethod):
    """HippocampusMCP + rejection mechanism.

    Adds a confidence threshold on top of FullMethod:
    - when the top-1 retrieval score is below the threshold, return an empty result (reject)
    - mainly for adversarial questions (category=5), to avoid giving wrong answers
    """

    name = "HippocampusMCP-Reject"

    def __init__(
        self,
        *args,
        reject_threshold: float = 0.15,
        reject_category: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reject_threshold = reject_threshold
        self.reject_category = reject_category

    def retrieve(
        self, query: str, top_k: int = 5, category: int | None = None
    ) -> list[tuple[str, float]]:
        results = super().retrieve(query, top_k=top_k, category=category)

        # adversarial questions: reject on low confidence
        if category == self.reject_category and results:
            top_score = results[0][1]
            if top_score < self.reject_threshold:
                return []  # reject

        return results
