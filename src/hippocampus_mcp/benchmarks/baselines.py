# -*- coding: utf-8 -*-
"""基线方法实现.

为评估 hippocampus_mcp (LNN + 打分) 提供对照基线:
    - RandomBaseline:       随机检索
    - PureEmbedding:        纯文本嵌入余弦相似度
    - TimeDecayBaseline:    嵌入相似度 × 时间衰减
    - MeanEmbedding:        最近 N 条嵌入均值作为查询
    - LNNBaseline:          CfC LNN 查询 (不含打分)
    - FullMethod:           LNN + 打分 + 抽样 (hippocampus_mcp 完整方法)

所有基线实现统一接口:
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

# 确保项目根目录在 sys.path 中 (用于导入 embeddings 模块)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 导入主项目的嵌入器
from hippocampus_mcp.engine.embeddings import HashEmbedder as _ProjectHashEmbedder
from hippocampus_mcp.engine.embeddings import SentenceTransformerEmbedder as _STEmbedder

# 延迟导入 torch 和 ncps, 仅 LNN 基线需要
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
    """创建评测用嵌入器.

    Args:
        backend: "sentence_transformer" 或 "hash".
        dim: hash 后端的维度 (sentence_transformer 由模型决定).

    Returns:
        嵌入器实例.
    """
    if backend == "sentence_transformer":
        return _STEmbedder()
    else:
        return _ProjectHashEmbedder(dim=dim, salt="eval")


class BaseRetriever:
    """检索基类, 定义统一接口."""

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
        """获取参考时间: 使用已存储的最新时间戳, 而非 wall-clock.

        这确保时间衰减基于数据内部的时间跨度 (如 LoCoMo 中会话间隔数周),
        而非数据生成时间与当前实际时间的巨大差异.
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
    """随机检索基线."""

    name = "Random"

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        n = len(self._ids)
        if n == 0:
            return []
        k = min(top_k, n)
        indices = np.random.choice(n, size=k, replace=False)
        return [(self._ids[i], 1.0 / (j + 1)) for j, i in enumerate(indices)]


class PureEmbedding(BaseRetriever):
    """纯文本嵌入余弦相似度检索."""

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
    """嵌入相似度 × 时间衰减."""

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
    """最近 N 条嵌入均值作为查询 (滑动窗口)."""

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
        # 查询向量 = 文本嵌入与最近窗口均值的融合
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
    """CfC LNN 查询向量 (不含打分引擎).

    使用 CfC 的隐藏状态 h(t) 作为查询向量, 与文本嵌入加权融合.
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
        # AutoNCP 要求 output_dim < n_neurons - 2
        actual_n_neurons = max(n_neurons, self.dim + 3)
        wiring = AutoNCP(actual_n_neurons, self.dim)
        self.cfc = CfC(self.dim, wiring, batch_first=True)
        self.hx = None
        self.last_output = torch.zeros(self.dim)
        self.window_size = window_size
        self.window: list = []

    def store(self, content: str, timestamp: float | None = None) -> str:
        doc_id = super().store(content, timestamp)
        # 更新 LNN 状态
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

        # LNN 查询向量
        if self.window:
            window_mean = torch.stack(list(self.window)).mean(dim=0)
            lnn_query = 0.5 * self.last_output + 0.5 * window_mean
            lnn_np = self._F.normalize(lnn_query, dim=0).numpy().astype(np.float64)
        else:
            lnn_np = q_text

        # 融合: 0.6 * 文本 + 0.4 * LNN
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
        """加载训练后的 CfC 权重.

        Args:
            model_path: train_cfc.py 保存的 .pt 文件路径.
        """
        torch = _ensure_torch()[0]
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        # 训练脚本保存的键带 "cfc." 前缀, 需剥离后加载到 self.cfc
        cfc_state = {}
        for k, v in checkpoint["model_state"].items():
            if k.startswith("cfc."):
                cfc_state[k[4:]] = v  # 去掉 "cfc." 前缀
        if cfc_state:
            self.cfc.load_state_dict(cfc_state)
            print(f"  [LNNBaseline] 已加载训练权重: {model_path} (val_loss={checkpoint.get('best_val_loss', '?')})")
        else:
            self.cfc.load_state_dict(checkpoint["model_state"])
            print(f"  [LNNBaseline] 已加载训练权重 (直接模式): {model_path}")


class FullMethod(LNNBaseline):
    """hippocampus_mcp 完整方法: LNN + 打分引擎 + 抽样.

    在 LNN 查询的基础上, 叠加时间衰减与回忆频次打分.
    支持条件性时间衰减: 根据问题类型动态调整时间权重.
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
        # 条件性时间衰减权重映射 (按 LoCoMo category)
        # category=2 (时序推理): 高时间权重; 其他: 极低
        self._cat_time_weights = {
            1: 0.01,   # 多跳推理 — 时间不重要
            2: 0.10,   # 时序推理 — 时间关键
            3: 0.01,   # 开放域 — 时间不重要
            4: 0.01,   # 单跳推理 — 时间不重要
            5: 0.01,   # 对抗问题 — 时间不重要
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

        # 复用 LNN 的相似度计算
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

        # 确定时间权重: 条件性衰减或固定权重
        if self.conditional_time and category is not None:
            w_time_eff = self._cat_time_weights.get(category, self.w_time)
        else:
            w_time_eff = self.w_time

        # 打分: 加权加法公式 (相似度为主, 时间/回忆为辅)
        current_time = self._get_reference_time()
        ages = np.maximum(
            np.array([current_time - t for t in self._timestamps]), 0.0
        )
        time_decay = np.exp(-ages / self.lambda_decay)
        recall_arr = np.array(self._recall_counts, dtype=np.float64)
        recall_boost = np.log1p(recall_arr) * np.exp(-ages / self.mu_importance)

        # 加权: 可配置的相似度/时间/回忆权重
        scores = self.w_sim * sims + w_time_eff * time_decay + self.w_recall * recall_boost

        k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:k]

        # 更新回忆计数
        for i in top_idx:
            self._recall_counts[i] += 1

        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def reset(self) -> None:
        super().reset()
        self._recall_counts.clear()


class FullMethodRejection(FullMethod):
    """HippocampusMCP + 拒答机制.

    在 FullMethod 基础上增加置信度阈值:
    - 当 top-1 检索分数低于阈值时, 返回空结果 (拒答)
    - 主要用于对抗类问题 (category=5), 避免给出错误答案
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

        # 对抗问题: 低置信度时拒答
        if category == self.reject_category and results:
            top_score = results[0][1]
            if top_score < self.reject_threshold:
                return []  # 拒答

        return results
