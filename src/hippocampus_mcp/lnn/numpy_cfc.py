# -*- coding: utf-8 -*-
"""A pure-numpy implementation of the CfC memory core.

Operator-by-operator comparison with ncps 1.0.x ``WiredCfCCell(mode="default")``
forward:

    For each layer l (AutoNCP multi-layer; layer 0 input = embedding vector,
    other layers' input = previous layer's output):
        xl  = cat([inp, h_l])
        ff1 = tanh(xl @ (W_ff1 ⊙ mask_l)ᵀ + b_ff1)      # mask applies to ff1/ff2 only
        ff2 = tanh(xl @ (W_ff2 ⊙ mask_l)ᵀ + b_ff2)
        t   = sigmoid((xl @ W_taᵀ + b_ta)·ts + (xl @ W_tbᵀ + b_tb))  # copied as-is: the time head gets no mask
        new_h_l = ff1·(1−t) + t·ff2
    output = last layer's new_h (motor layer); hx_new = cat(all layers' new_h)

Why we dare:
    - The production config ``AutoNCP(771, 768)`` builds 3 layers ``[2, 1, 768]``,
      the last of which is the motor;
    - The wiring is generated deterministically from the
      ``np.random.default_rng(1111)`` seed, so the adjacency matrices can be
      exported wholesale;
    - The production-path CfC was **never trained** (``HIPPOCAMPUS_CRL_MODEL``
      unset, no manual_seed in src; weights change on every restart), so the
      equivalence acceptance is therefore a "same-weights head-to-head check",
      not "reproducing some training run".

Mask pre-multiplication: the torch version does ``weight * sparsity_mask`` on
every forward; here the mask is fixed and pre-multiplied once at load time
(``W_eff = W ⊙ mask``) — mathematically equivalent, half the multiplications.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

# Weight initialization seed (for deterministic production initialization).
# Same origin as AutoNCP wiring's default seed(1111), so "deterministic wiring
# + deterministic weights" compose a fully deterministic core.
DEFAULT_INIT_SEED = 1111

_MODE = "default"
_TS = 1.0  # ncps's fixed timestep when timespans=None

# Names of the 4 linear heads in npz / state dicts (1:1 with ncps module attributes)
_HEADS = ("ff1", "ff2", "time_a", "time_b")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NumpyCfcLayer:
    """Single-layer CfCCell equivalent (the backbone_layers=0 form, i.e. the whole of wired mode).

    Weight layout matches torch ``nn.Linear``: weight (out, in), bias (out,).
    ``w_ff1/w_ff2`` store the **already mask-pre-multiplied** effective weights.
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
        """x: (B, in) → (B, H)."""
        ff1 = np.tanh(x @ self.w_ff1.T + self.b_ff1)
        ff2 = np.tanh(x @ self.w_ff2.T + self.b_ff2)
        t = _sigmoid((x @ self.w_ta.T + self.b_ta) * ts + (x @ self.w_tb.T + self.b_tb))
        return ff1 * (1.0 - t) + t * ff2


class NumpyWiredCfc:
    """Numpy equivalent of the multi-layer (wired) CfC core.

    Args:
        input_dim: Embedding input dimension (layer 0's external input width).
        layer_sizes: Neuron count per layer (e.g. AutoNCP(771,768) →
            [2, 1, 768]). The last layer must be the output layer
            (``output_dim == layer_sizes[-1]``).
        masks: Per-layer mask list, shape (H_l, in_l),
            ``in_l = previous layer width + H_l``. Passing ``None`` generates
            all-ones "fully connected" masks (equivalent to no sparse wiring;
            for tests).
        seed: Deterministic initialization seed.
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
                f"the last layer must be the output layer: layer_sizes[-1]={layer_sizes[-1]} != "
                f"output_dim={output_dim}(in ncps, fc=Identity and the output is the last layer's hidden state)"
            )
        self.input_dim = int(input_dim)
        self.layer_sizes = [int(n) for n in layer_sizes]
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        # Masks kept on file: load_weights must re-pre-multiply ff1/ff2 when
        # swapping weights
        self.masks: list[np.ndarray] = []

        rng = np.random.default_rng(self.seed)
        self.layers: list[NumpyCfcLayer] = []
        prev_out = self.input_dim  # layer 0's "external input" width; layer l>0 takes the previous layer's output
        for l, h in enumerate(self.layer_sizes):
            in_l = prev_out + h  # cat([inp, h_l])
            if masks is not None:
                mask = np.asarray(masks[l], dtype=np.float32)
                if mask.shape != (h, in_l):
                    raise ValueError(
                        f"layer{l} mask shape {mask.shape} != expected ({h}, {in_l})"
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
                            f"layer{l}.{head}.weight shape {w.shape} "
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
    # Initialization (aligned with torch: 2D weights Xavier uniform, bias per Linear default)
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
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: np.ndarray, hx: Optional[np.ndarray], ts: float = _TS
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward (same contract as torch ``CfC.forward``).

        Args:
            x: (B, input_dim) or (B, S, input_dim).
            hx: (B, total_units) or None (zero-initialized).
            ts: Timestep (always 1.0 in this service).

        Returns:
            (output, hx_new): output (B, S, output_dim) (consistent with
            torch's ``fc=Identity`` semantics, i.e. the full sequence of the
            last layer's hidden states); hx_new (B, total_units).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError(f"forward input must be 2/3 dimensional, got {x.ndim}D")
        batch, seq = x.shape[0], x.shape[1]
        total = sum(self.layer_sizes)
        if hx is None:
            h = np.zeros((batch, total), dtype=np.float32)
        else:
            h = np.asarray(hx, dtype=np.float32)
            if h.shape != (batch, total):
                raise ValueError(
                    f"hx shape {h.shape} != expected ({batch}, {total})"
                )
        # Split by layer (consistent with torch.split(hx, layer_sizes, dim=1))
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
            outs[:, s, :] = inp  # last layer's output (motor)
            h_layers = new_layers
        hx_new = np.concatenate(h_layers, axis=1)
        return outs, hx_new

    def load_weights(self, arrays: dict[str, np.ndarray]) -> None:
        """Swap weights in place by ``layer{i}.{head}.{weight|bias}`` keys (ff1/ff2 re-multiplied by mask).

        Same semantics as ncps ``load_state_dict(strict=False)``: missing keys
        don't blow up (original values kept).
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
                            f"{w_key} shape {w.shape} != expected ({h}, {in_l})"
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
                        raise ValueError(f"{b_key} shape {b.shape} != expected ({h},)")
                    setattr(
                        layer,
                        "b_ff1" if head == "ff1" else
                        "b_ff2" if head == "ff2" else
                        "b_ta" if head == "time_a" else "b_tb",
                        b.astype(np.float32),
                    )

    # ------------------------------------------------------------------
    # Weight read/write (npz channel)
    # ------------------------------------------------------------------
    def state_arrays(self) -> dict[str, np.ndarray]:
        """Export as an npz-friendly array dict (keys aligned with ncps state_dict, prefix stripped)."""
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
        """Construct from an exported npz (includes masks and metadata; produced by the weight-export script)."""
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
