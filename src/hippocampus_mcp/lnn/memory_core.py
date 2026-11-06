# -*- coding: utf-8 -*-
"""CfC 动态记忆核（默认纯 numpy，可回退 torch/ncps）.

封装 CfC (Closed-form Continuous-time) 网络, 维护跨 step 的隐藏状态以实现
时序记忆. 同时维护一个最近消息的滑动窗口, get_query 返回 CfC 输出与窗口均值
的加权融合, 构成混合架构.

**后端**：``HIPPOCAMPUS_LNN_BACKEND``

* ``numpy``（默认）—— :class:`lnn.numpy_cfc.NumpyWiredCfc` 逐算子复刻 ncps
  ``WiredCfCCell(mode="default")``；生产进程**不再 import torch**（内存 −370MB
  级收益的前提）。接线仍用 ``ncps.wirings.AutoNCP``（纯 numpy、种子确定）。
* ``torch`` —— 原 ncps.torch 实现，惰性导入；等价性由
  ``tests/test_h7_numpy_lnn.py`` 的 A/B 对拍钉住，出问题一键回退。

环境约束:
    - AutoNCP 要求 output_dim < n_neurons - 2
    - CfC(input_dim, wiring) 返回 (output, hx)
        output shape = (batch, time, output_dim)
        hx     shape = (batch, n_neurons)
"""

import os
import sys
from collections import deque
from typing import Any, Optional

import numpy as np

from .numpy_cfc import DEFAULT_INIT_SEED, NumpyWiredCfc

#: 后端开关。numpy = 默认零 torch；torch = ncps 实现（惰性导入）。
ENV_LNN_BACKEND = "HIPPOCAMPUS_LNN_BACKEND"

# 构造注入缝：旋钮登记在 config（lnn_backend），生产路径由 engine
# import 时灌入；解析序（env 先、注入值回退）见 common/config_bridge。
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

# AutoNCP(wiring) 生成只依赖 numpy；ncps.torch 才依赖 torch。
from ncps.wirings import AutoNCP  # noqa: E402  纯 numpy，不引 torch


def _stderr_print(*args, **kwargs) -> None:
    """诊断输出一律走 stderr。

    **MCP stdio 模式下 stdout 是 JSON-RPC 协议通道**：写到 stdout 的内容会被客户端
    当成协议消息解析（实测报 "Invalid JSON"），可能直接导致断连。本模块的
    "忽略不兼容的 LNN 状态" 等提示在换模型后首次启动时**必然触发**，故必须走 stderr。
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _backend_from_env() -> str:
    raw = (_resolve(ENV_LNN_BACKEND, "lnn_backend", "") or "").strip().lower()
    return raw if raw in ("numpy", "torch") else "numpy"


class MemoryCore:
    """CfC 动态记忆核.

    通过 CfC 网络在多次 step 调用间维持隐藏状态 ``hx``, 形成"液体状态机"
    式的时序记忆. 每次输入消息都会更新隐藏状态, 使核能够记住最近的上下文.

    混合架构: 除 CfC 的隐式时序记忆外, 显式维护一个最近输出向量的滑动窗口,
    get_query 返回 CfC 当前输出与窗口均值的加权融合 (默认 0.5 / 0.5),
    兼顾对最新输入的响应性与近期上下文的稳定性.

    Args:
        input_dim: 输入嵌入维度 (默认 32).
        n_neurons: CfC 神经元数量 (默认 64). 需满足 output_dim < n_neurons - 2.
        output_dim: 输出维度 (默认 32).
        window_size: 滑动窗口大小 (默认 5).
        cfc_weight: get_query 中 CfC 输出的权重 (默认 0.5).
        window_weight: get_query 中窗口均值的权重 (默认 0.5).
    """

    def __init__(
        self,
        input_dim: int = 32,
        n_neurons: int = 64,
        output_dim: int = 32,
        window_size: int = 5,
        cfc_weight: float = 0.5,
        window_weight: float = 0.5,
    ):
        # AutoNCP 要求 output_dim < n_neurons - 2
        assert output_dim < n_neurons - 2, (
            f"AutoNCP 要求 output_dim < n_neurons - 2, "
            f"但得到 output_dim={output_dim}, n_neurons={n_neurons} "
            f"(需 n_neurons > {output_dim + 2})"
        )

        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.output_dim = output_dim
        self.window_size = window_size
        self.cfc_weight = float(cfc_weight)
        self.window_weight = float(window_weight)
        self.backend = _backend_from_env()

        # 接线：两后端共用同一份确定性生成（AutoNCP 默认 seed=22222，
        # 与 torch 路径构造的 wiring 逐位一致 ⇒ A/B 对拍两侧掩码天然相同）。
        wiring = AutoNCP(n_neurons, output_dim)
        wiring.build(input_dim)
        self._wiring = wiring
        self._layer_sizes = [
            len(wiring.get_neurons_of_layer(l)) for l in range(wiring.num_layers)
        ]

        if self.backend == "torch":
            import torch  # 惰性：只有显式回退才付这份内存
            from ncps.torch import CfC

            self._torch = torch
            self.cfc = CfC(input_dim, wiring, batch_first=True)
        else:
            # numpy 核：掩码按 WiredCfCCell.__init__ 的同一公式推导
            masks: list[np.ndarray] = []
            prev_neurons: Any = None
            for l in range(wiring.num_layers):
                neurons = wiring.get_neurons_of_layer(l)
                if l == 0:
                    sparsity = wiring.sensory_adjacency_matrix[:, neurons]
                else:
                    sparsity = wiring.adjacency_matrix[prev_neurons][:, neurons]
                sparsity = np.concatenate(
                    [sparsity, np.ones((len(neurons), len(neurons)), dtype=np.float64)],
                    axis=0,
                )
                masks.append(np.abs(sparsity.T).astype(np.float32))
                prev_neurons = neurons
            self.cfc = NumpyWiredCfc(
                input_dim=input_dim,
                layer_sizes=self._layer_sizes,
                output_dim=output_dim,
                masks=masks,
                seed=DEFAULT_INIT_SEED,
            )

        # 跨 step 维持的隐藏状态, shape = (batch, n_neurons)
        # 初始为 None, 首次 step 时由 CfC 自动初始化
        self.hx: Optional[Any] = None

        # 最近一次输出向量 (detached), shape = (output_dim,)
        self.last_output = self._zeros(output_dim)

        # 最近消息输出的滑动窗口
        self.window: deque = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # 后端细节
    # ------------------------------------------------------------------
    @property
    def runtime(self) -> str:
        """自曝实际运行时（范式：报实际值，不报配置值）。"""
        return "torch(ncps)" if self.backend == "torch" else "numpy"

    def _zeros(self, *shape: int) -> Any:
        if self.backend == "torch":
            return self._torch.zeros(*shape)
        return np.zeros(shape, dtype=np.float32)

    def _to_input(self, x: Any) -> Any:
        """把输入统一成 (B, S, input_dim)。收 numpy（生产路径）或 torch 张量。"""
        if self.backend == "torch":
            t = self._torch
            if not t.is_tensor(x):
                x = t.from_numpy(np.asarray(x, dtype=np.float32))
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() != 3:
                raise ValueError(
                    f"step 输入维度应为 2 (1, input_dim) 或 "
                    f"3 (batch, seq, input_dim), 但得到 {x.dim()}D tensor"
                )
            return x.float()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        elif x.ndim != 3:
            raise ValueError(
                f"step 输入维度应为 2 (1, input_dim) 或 "
                f"3 (batch, seq, input_dim), 但得到 {x.ndim}D tensor"
            )
        return x

    def reset_state(self) -> None:
        """清空时序状态（hx / last_output / window）。全局上下文重置（reset_context）走这里。"""
        self.hx = None
        self.last_output = self._zeros(self.output_dim)
        self.window.clear()

    def load_weights_npz(self, path: str) -> bool:
        """从 npz 加载训练权重（CRL 导出物）。两后端都可走这条通道。"""
        if self.backend == "torch":
            import numpy as _np

            with _np.load(path, allow_pickle=False) as z:
                arrays = {k: _np.asarray(z[k]) for k in z.files if k != "meta_json"}
            self.load_weights(arrays)
            return True
        self.cfc = NumpyWiredCfc.from_npz(path)
        return True

    def load_weights(self, arrays: dict[str, "Any"]) -> bool:
        """把训练权重灌进当前核。键名两种风格都收：

        * ``layer{i}.{head}.{weight|bias}``（numpy_cfc / npz 导出风格）
        * ``rnn_cell.layer_{i}.…``（ncps state_dict 原生风格，legacy CRL .pt）

        torch 核转 tensor 走 ``load_state_dict(strict=False)``（与旧行为一致，
        缺键/多键不炸）；numpy 核校验形状后原地换权重（ff1/ff2 重乘掩码）。
        """
        norm = {}
        for k, v in arrays.items():
            nk = k
            if nk.startswith("rnn_cell.layer_"):
                # ncps 原生 → numpy_cfc 风格
                nk = nk[len("rnn_cell."):].replace("layer_", "layer", 1)
            norm[nk] = v
        if self.backend == "torch":
            t = self._torch
            state = {}
            for k, v in norm.items():
                # numpy_cfc 风格 → ncps state_dict 原生键
                nk = k
                if k.startswith("layer") and "." in k:
                    head, rest = k.split(".", 1)
                    nk = f"rnn_cell.{head.replace('layer', 'layer_', 1)}.{rest}"
                state[nk] = v if t.is_tensor(v) else t.from_numpy(np.asarray(v))
            self.cfc.load_state_dict(state, strict=False)
            return True
        self.cfc.load_weights(norm)
        return True

    # ------------------------------------------------------------------
    # 前向推理
    # ------------------------------------------------------------------
    def step(self, x: Any) -> Any:
        """处理输入, 更新隐藏状态, 返回输出向量.

        Args:
            x: 输入张量, 支持两种形式:
                - (1, input_dim): 单个嵌入向量
                - (batch, seq, input_dim): 嵌入序列

        Returns:
            输出向量, shape = (output_dim,), 取最后一个时间步的结果.
            CfC 的隐藏状态 hx 会跨调用维持 (时序记忆的核心).
        """
        x = self._to_input(x)

        if self.backend == "torch":
            t = self._torch
            # CfC 前向: output=(batch, seq, output_dim), hx=(batch, n_neurons)
            # hx 跨 step 维持 —— 这是时序记忆的核心
            output, hx = self.cfc(x, self.hx)
            # 切断跨 step 的梯度图, 防止长序列下计算图无限增长 (适合推理/检索)
            self.hx = hx.detach()
            last = output[:, -1, :]
            out_vec = last.squeeze(0) if last.shape[0] == 1 else last[0]
            out_vec = out_vec.detach()
            self.last_output = out_vec.clone()
            self.window.append(out_vec.clone())
            return out_vec

        output, hx = self.cfc.forward(x, self.hx)
        self.hx = hx
        last = output[:, -1, :]
        out_vec = np.asarray(last[0], dtype=np.float32).copy()
        self.last_output = out_vec.copy()
        self.window.append(out_vec.copy())
        return out_vec

    # ------------------------------------------------------------------
    # 查询向量
    # ------------------------------------------------------------------
    def get_query(self) -> Any:
        """返回当前输出与窗口均值加权融合后的归一化查询向量.

        混合架构: query = cfc_weight * last_output + window_weight * window_mean,
        再做 L2 归一化. 默认权重 0.5 / 0.5.

        Returns:
            归一化查询向量, shape = (output_dim,).
        """
        cur = self.last_output

        if len(self.window) > 0:
            if self.backend == "torch":
                window_mean = self._torch.stack(list(self.window)).mean(dim=0)
            else:
                window_mean = np.stack(list(self.window)).mean(axis=0)
        else:
            window_mean = cur

        fused = self.cfc_weight * cur + self.window_weight * window_mean
        if self.backend == "torch":
            import torch.nn.functional as F

            return F.normalize(fused, dim=0)
        norm = float(np.linalg.norm(fused))
        return fused / max(norm, 1e-12)

    # ------------------------------------------------------------------
    # 状态持久化
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """返回当前隐藏状态用于 checkpoint.

        **统一返回 numpy 数组**（torch 核在出口处转换），checkpoint 层因此零 torch。

        Returns:
            包含 hx / last_output / window 及配置信息的字典.
        """
        return {
            "hx": self._state_array(self.hx),
            "last_output": self._state_array(self.last_output),
            "window": [self._state_array(t) for t in self.window],
            "input_dim": self.input_dim,
            "n_neurons": self.n_neurons,
            "output_dim": self.output_dim,
            "window_size": self.window_size,
        }

    @staticmethod
    def _state_array(t: Any) -> Optional[np.ndarray]:
        """torch 张量 / numpy 数组 → numpy（None 透传）。"""
        if t is None:
            return None
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def load_state(self, state: dict) -> None:
        """从 checkpoint 恢复状态.

        隐藏状态与嵌入维度绑定：若 checkpoint 来自另一个嵌入模型（维度不同），
        直接恢复会让后续 ``get_query()`` 与查询向量维度不匹配。
        此时安全跳过恢复（重置为空状态），由后续 step 重新积累。

        收 **numpy 数组**（checkpoint npz 的原生形态）；兼容旧 `.pt` 路径读出的
        torch 张量（自动转换）。

        Args:
            state: get_state 返回的字典. 允许为空字典 —— ``StateCheckpoint.load()``
                在主文件与备份都读不出来时会返回 ``{}``（例如状态文件被写坏、
                磁盘故障）。此时必须安全退化为初始状态，而不是抛异常：这条
                路径在服务启动时就会走到，崩掉等于整个 MCP 服务起不来。
        """
        if not state:
            self.reset_state()
            return

        saved_dim = state.get("output_dim")
        if saved_dim is not None and int(saved_dim) != int(self.output_dim):
            _stderr_print(
                f"[MemoryCore] ignoring incompatible LNN state: dimension {saved_dim} "
                f"!= 当前 {self.output_dim}(the embedding model changed; state will re-accumulate)"
            )
            self.reset_state()
            return

        hx = state.get("hx")
        if hx is not None and tuple(np.asarray(hx).shape[-1:]) != (self.n_neurons,):
            self.hx = None
        else:
            self.hx = self._restore_array(hx) if hx is not None else None

        last = state.get("last_output")
        if last is None or tuple(np.asarray(last).shape) != (self.output_dim,):
            if last is not None:
                _stderr_print(
                    f"[MemoryCore] ignoring incompatible last_output: "
                    f"{tuple(np.asarray(last).shape)} != ({self.output_dim},)"
                )
            self.last_output = self._zeros(self.output_dim)
        else:
            self.last_output = self._restore_array(last)

        window_arrays = state.get("window", []) or []
        if any(
            tuple(np.asarray(t).shape) != (self.input_dim,) for t in window_arrays
        ):
            self.window.clear()
        else:
            self.window = deque(
                (self._restore_array(t) for t in window_arrays),
                maxlen=self.window_size,
            )

    def _restore_array(self, t: Any) -> Any:
        """恢复一路状态：numpy（或旧 .pt 的 torch 张量）→ 本后端的数组。"""
        arr = np.asarray(t, dtype=np.float32)
        if self.backend == "torch":
            return self._torch.from_numpy(arr)
        return arr
