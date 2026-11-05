# -*- coding: utf-8 -*-
"""CfC 记忆核的纯 numpy 实现.

逐算子对照 ncps 1.0.x ``WiredCfCCell(mode="default")`` 的前向：

    对每层 l（AutoNCP 多层，层 0 输入=嵌入向量，其余层输入=上层输出）：
        xl  = cat([inp, h_l])
        ff1 = tanh(xl @ (W_ff1 ⊙ mask_l)ᵀ + b_ff1)      # mask 仅作用于 ff1/ff2
        ff2 = tanh(xl @ (W_ff2 ⊙ mask_l)ᵀ + b_ff2)
        t   = sigmoid((xl @ W_taᵀ + b_ta)·ts + (xl @ W_tbᵀ + b_tb))  # 照抄：time 头不加 mask
        new_h_l = ff1·(1−t) + t·ff2
    输出 = 末层 new_h（motor 层）；hx_new = cat(各层 new_h)

为什么敢做：
    - 生产配置 ``AutoNCP(771, 768)`` build 后 3 层 ``[2, 1, 768]``，末层即 motor；
    - 接线由 ``np.random.default_rng(1111)`` 种子确定性生成，邻接矩阵可整体导出；
    - 生产路径的 CfC **从未被训练**（``HIPPOCAMPUS_CRL_MODEL`` 未配置、src 无
      manual_seed，权重每次重启都在变），等价性验收 therefore 是"同权重对拍"，
      而不是"复现某次训练结果"。

掩码预乘：torch 版每次 forward 都做 ``weight * sparsity_mask``；这里 mask 固定，
加载时预乘一次（``W_eff = W ⊙ mask``），数学等价、省一半乘法。
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

# 权重初始化种子（生产确定性初始化用）。与 AutoNCP 接线的默认 seed(1111) 同源，
# 让"接线确定 + 权重确定"构成一个完全确定的核。
DEFAULT_INIT_SEED = 1111

_MODE = "default"
_TS = 1.0  # timespans=None 时 ncps 的固定时间步

# 4 个线性头在 npz / 状态字典里的名字（与 ncps 模块属性一一对应）
_HEADS = ("ff1", "ff2", "time_a", "time_b")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NumpyCfcLayer:
    """单层 CfCCell 等价物（backbone_layers=0 的形态，即 wired 模式的全部形态）。

    权重布局与 torch ``nn.Linear`` 一致：weight (out, in)、bias (out,)。
    ``w_ff1/w_ff2`` 存的是**已预乘掩码**的有效权重。
    """

    __slots__ = (
        "w_ff1", "b_ff1", "w_ff2", "b_ff2",
        "w_ta", "b_ta", "w_tb", "b_tb",
    )

    def __init__(self, w_ff1, b_ff1, w_ff2, b_ff2, w_ta, b_ta, w_tb, b_tb):
        self.w_ff1 = w_ff1
        self.b_ff1 = b_ff1
        self.w_ff2 = w_ff2
        self.b_ff2 = b_ff2
        self.w_ta = w_ta
        self.b_ta = b_ta
        self.w_tb = w_tb
        self.b_tb = b_tb

    def forward(self, x: np.ndarray, ts: float = _TS) -> np.ndarray:
        """x: (B, in) → (B, H)。"""
        ff1 = np.tanh(x @ self.w_ff1.T + self.b_ff1)
        ff2 = np.tanh(x @ self.w_ff2.T + self.b_ff2)
        t = _sigmoid((x @ self.w_ta.T + self.b_ta) * ts + (x @ self.w_tb.T + self.b_tb))
        return ff1 * (1.0 - t) + t * ff2


class NumpyWiredCfc:
    """多层（wired）CfC 核的 numpy 等价物。

    Args:
        input_dim: 嵌入输入维度（层 0 的外部输入宽度）。
        layer_sizes: 各层神经元数（如 AutoNCP(771,768) → [2, 1, 768]）。
            末层必须是输出层（``output_dim == layer_sizes[-1]``）。
        masks: 每层的掩码列表，形状 (H_l, in_l)，``in_l = 上层宽 + H_l``。
            传 ``None`` 时按"全连接"生成全 1 掩码（等价无稀疏接线，测试用）。
        seed: 确定性初始化种子。
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        output_dim: int,
        masks: Optional[list[np.ndarray]] = None,
        seed: int = DEFAULT_INIT_SEED,
        weights: Optional[dict[str, np.ndarray]] = None,
    ):
        if layer_sizes[-1] != output_dim:
            raise ValueError(
                f"末层必须是输出层：layer_sizes[-1]={layer_sizes[-1]} != "
                f"output_dim={output_dim}（ncps 里 fc=Identity，输出即末层隐状态）"
            )
        self.input_dim = int(input_dim)
        self.layer_sizes = [int(n) for n in layer_sizes]
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        # 掩码留档：load_weights 换权重时要重新预乘 ff1/ff2
        self.masks: list[np.ndarray] = []

        rng = np.random.default_rng(self.seed)
        self.layers: list[NumpyCfcLayer] = []
        prev_out = self.input_dim  # 层 0 的"外部输入"宽度；层 l>0 输入是上层输出
        for l, h in enumerate(self.layer_sizes):
            in_l = prev_out + h  # cat([inp, h_l])
            if masks is not None:
                mask = np.asarray(masks[l], dtype=np.float32)
                if mask.shape != (h, in_l):
                    raise ValueError(
                        f"layer{l} mask 形状 {mask.shape} != 期望 ({h}, {in_l})"
                    )
            else:
                mask = np.ones((h, in_l), dtype=np.float32)
            if weights is None:
                ws = {head: self._xavier_init(rng, h, in_l) for head in _HEADS}
                bs = {head: self._bias_init(rng, h, in_l) for head in _HEADS}
            else:
                ws, bs = {}, {}
                for head in _HEADS:
                    w = np.asarray(weights[f"layer{l}.{head}.weight"], dtype=np.float32)
                    b = np.asarray(weights[f"layer{l}.{head}.bias"], dtype=np.float32)
                    if w.shape != (h, in_l):
                        raise ValueError(
                            f"layer{l}.{head}.weight 形状 {w.shape} "
                            f"!= 期望 ({h}, {in_l})"
                        )
                    ws[head], bs[head] = w, b
            self.layers.append(
                NumpyCfcLayer(
                    (ws["ff1"] * mask).astype(np.float32),
                    bs["ff1"].astype(np.float32),
                    (ws["ff2"] * mask).astype(np.float32),
                    bs["ff2"].astype(np.float32),
                    ws["time_a"].astype(np.float32),
                    bs["time_a"].astype(np.float32),
                    ws["time_b"].astype(np.float32),
                    bs["time_b"].astype(np.float32),
                )
            )
            self.masks.append(mask)
            prev_out = h

    # ------------------------------------------------------------------
    # 初始化（对齐 torch：2 维权重 Xavier uniform，bias 走 Linear 默认）
    # ------------------------------------------------------------------
    @staticmethod
    def _xavier_init(rng: np.random.Generator, out_f: int, in_f: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (in_f + out_f))
        return rng.uniform(-limit, limit, size=(out_f, in_f)).astype(np.float32)

    @staticmethod
    def _bias_init(rng: np.random.Generator, out_f: int, in_f: int) -> np.ndarray:
        limit = 1.0 / np.sqrt(in_f)
        return rng.uniform(-limit, limit, size=(out_f,)).astype(np.float32)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self, x: np.ndarray, hx: Optional[np.ndarray], ts: float = _TS
    ) -> tuple[np.ndarray, np.ndarray]:
        """前向（与 torch ``CfC.forward`` 同契约）。

        Args:
            x: (B, input_dim) 或 (B, S, input_dim)。
            hx: (B, total_units) 或 None（零初始化）。
            ts: 时间步（本服务恒为 1.0）。

        Returns:
            (output, hx_new)：output (B, S, output_dim)（与 torch 的
            ``fc=Identity`` 语义一致，即末层隐状态的完整序列）；hx_new
            (B, total_units)。
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError(f"forward 输入应为 2/3 维，得到 {x.ndim} 维")
        batch, seq = x.shape[0], x.shape[1]
        total = sum(self.layer_sizes)
        if hx is None:
            h = np.zeros((batch, total), dtype=np.float32)
        else:
            h = np.asarray(hx, dtype=np.float32)
            if h.shape != (batch, total):
                raise ValueError(
                    f"hx 形状 {h.shape} != 期望 ({batch}, {total})"
                )
        # 按层切分（与 torch.split(hx, layer_sizes, dim=1) 一致）
        bounds = np.cumsum([0] + self.layer_sizes)
        h_layers = [h[:, bounds[i]:bounds[i + 1]] for i in range(len(self.layer_sizes))]

        outs = np.zeros((batch, seq, self.output_dim), dtype=np.float32)
        for s in range(seq):
            inp = x[:, s, :]
            new_layers = []
            for l, layer in enumerate(self.layers):
                xl = np.concatenate([inp, h_layers[l]], axis=1)
                nh = layer.forward(xl, ts=ts)
                new_layers.append(nh)
                inp = nh
            outs[:, s, :] = inp  # 末层输出（motor）
            h_layers = new_layers
        hx_new = np.concatenate(h_layers, axis=1)
        return outs, hx_new

    def load_weights(self, arrays: dict[str, np.ndarray]) -> None:
        """按 ``layer{i}.{head}.{weight|bias}`` 键原地换权重（ff1/ff2 重乘掩码）。

        与 ncps ``load_state_dict(strict=False)`` 同语义：缺键不炸（保留原值）。
        """
        for l, layer in enumerate(self.layers):
            h = self.layer_sizes[l]
            mask = self.masks[l]
            in_l = mask.shape[1]
            for head in _HEADS:
                w_key = f"layer{l}.{head}.weight"
                b_key = f"layer{l}.{head}.bias"
                if w_key in arrays:
                    w = np.asarray(arrays[w_key], dtype=np.float32)
                    if w.shape != (h, in_l):
                        raise ValueError(
                            f"{w_key} 形状 {w.shape} != 期望 ({h}, {in_l})"
                        )
                    w_eff = (w * mask) if head in ("ff1", "ff2") else w
                    setattr(
                        layer,
                        "w_ff1" if head == "ff1" else
                        "w_ff2" if head == "ff2" else
                        "w_ta" if head == "time_a" else "w_tb",
                        w_eff.astype(np.float32),
                    )
                if b_key in arrays:
                    b = np.asarray(arrays[b_key], dtype=np.float32)
                    if b.shape != (h,):
                        raise ValueError(f"{b_key} 形状 {b.shape} != 期望 ({h},)")
                    setattr(
                        layer,
                        "b_ff1" if head == "ff1" else
                        "b_ff2" if head == "ff2" else
                        "b_ta" if head == "time_a" else "b_tb",
                        b.astype(np.float32),
                    )

    # ------------------------------------------------------------------
    # 权重读写（npz 通道）
    # ------------------------------------------------------------------
    def state_arrays(self) -> dict[str, np.ndarray]:
        """导出为 npz 友好的数组字典（键与 ncps state_dict 对齐后剥前缀）。"""
        out: dict[str, np.ndarray] = {}
        for l, layer in enumerate(self.layers):
            out[f"layer{l}.ff1.weight"] = layer.w_ff1
            out[f"layer{l}.ff1.bias"] = layer.b_ff1
            out[f"layer{l}.ff2.weight"] = layer.w_ff2
            out[f"layer{l}.ff2.bias"] = layer.b_ff2
            out[f"layer{l}.time_a.weight"] = layer.w_ta
            out[f"layer{l}.time_a.bias"] = layer.b_ta
            out[f"layer{l}.time_b.weight"] = layer.w_tb
            out[f"layer{l}.time_b.bias"] = layer.b_tb
        return out

    @classmethod
    def from_npz(cls, path: str) -> "NumpyWiredCfc":
        """从导出的 npz 构造（含掩码与元信息；权重导出脚本产出此格式）。"""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            arrays = {k: np.asarray(z[k]) for k in z.files if k != "meta_json"}
        return cls(
            input_dim=meta["input_dim"],
            layer_sizes=list(meta["layer_sizes"]),
            output_dim=meta["output_dim"],
            masks=[arrays.pop(f"layer{l}.mask") for l in range(len(meta["layer_sizes"]))],
            seed=meta.get("seed", DEFAULT_INIT_SEED),
            weights=arrays,
        )
