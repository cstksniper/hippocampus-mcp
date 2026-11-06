# -*- coding: utf-8 -*-
"""CfC dynamic memory core (pure numpy by default, torch/ncps fallback).

Wraps a CfC (Closed-form Continuous-time) network, maintaining the hidden
state across steps to realize temporal memory. It also maintains a sliding
window of recent messages; get_query returns a weighted fusion of the CfC
output and the window mean, forming a hybrid architecture.

**Backend**: ``HIPPOCAMPUS_LNN_BACKEND``

* ``numpy`` (default) -- :class:`lnn.numpy_cfc.NumpyWiredCfc` replicates
  ncps ``WiredCfCCell(mode="default")`` operator by operator; the
  production process **no longer imports torch** (the precondition of the
  ~-370MB memory win). Wiring still uses ``ncps.wirings.AutoNCP`` (pure
  numpy, deterministic seed).
* ``torch`` -- the original ncps.torch implementation, lazily imported;
  equivalence is pinned by the A/B cross-check in
  ``tests/test_h7_numpy_lnn.py``, so problems fall back with one switch.

Environment constraints:
    - AutoNCP requires output_dim < n_neurons - 2
    - CfC(input_dim, wiring) returns (output, hx)
        output shape = (batch, time, output_dim)
        hx     shape = (batch, n_neurons)
"""

import os
import sys
from collections import deque
from typing import Any, Optional

import numpy as np

from .numpy_cfc import DEFAULT_INIT_SEED, NumpyWiredCfc

#: Backend switch. numpy = zero torch by default; torch = ncps implementation (lazy import).
ENV_LNN_BACKEND = "HIPPOCAMPUS_LNN_BACKEND"

# Construction injection seam: the knob is registered in config
# (lnn_backend), and the production path pours it in at engine import; the
# resolution order (env first, injected value as fallback) is in
# common/config_bridge.
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

# AutoNCP(wiring) generation depends only on numpy; only ncps.torch needs torch.
from ncps.wirings import AutoNCP  # noqa: E402  pure numpy, no torch import


def _stderr_print(*args, **kwargs) -> None:
    """Diagnostics always go to stderr.

    **In MCP stdio mode stdout is the JSON-RPC protocol channel**: anything
    written to stdout is parsed by the client as a protocol message (seen in
    testing as "Invalid JSON"), possibly causing a direct disconnect. This
    module's "ignoring incompatible LNN state" notice **necessarily fires**
    on first start after switching models, so it must go to stderr.
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _backend_from_env() -> str:
    raw = (_resolve(ENV_LNN_BACKEND, "lnn_backend", "") or "").strip().lower()
    return raw if raw in ("numpy", "torch") else "numpy"


class MemoryCore:
    """CfC dynamic memory core.

    The CfC network maintains a hidden state ``hx`` across step calls,
    forming "liquid state machine"-style temporal memory. Each input message
    updates the hidden state, letting the core remember recent context.

    Hybrid architecture: besides the CfC's implicit temporal memory, a
    sliding window of recent output vectors is explicitly maintained;
    get_query returns a weighted fusion of the CfC's current output and the
    window mean (default 0.5 / 0.5), balancing responsiveness to the latest
    input with stability of recent context.

    Args:
        input_dim: input embedding dimension (default 32).
        n_neurons: number of CfC neurons (default 64). Must satisfy
            output_dim < n_neurons - 2.
        output_dim: output dimension (default 32).
        window_size: sliding window size (default 5).
        cfc_weight: weight of the CfC output in get_query (default 0.5).
        window_weight: weight of the window mean in get_query (default 0.5).
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
        # AutoNCP requires output_dim < n_neurons - 2
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

        # Wiring: both backends share one deterministic generation (AutoNCP's
        # default seed=22222, bit-identical to the torch path's wiring => the
        # A/B cross-check's masks are naturally identical on both sides).
        wiring = AutoNCP(n_neurons, output_dim)
        wiring.build(input_dim)
        self._wiring = wiring
        self._layer_sizes = [
            len(wiring.get_neurons_of_layer(l)) for l in range(wiring.num_layers)
        ]

        if self.backend == "torch":
            import torch  # lazy: only the explicit fallback pays this memory
            from ncps.torch import CfC

            self._torch = torch
            self.cfc = CfC(input_dim, wiring, batch_first=True)
        else:
            # numpy core: masks derived by the same formula as WiredCfCCell.__init__
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

        # Hidden state maintained across steps, shape = (batch, n_neurons)
        # Starts as None; the CfC initializes it on first step
        self.hx: Optional[Any] = None

        # Most recent output vector (detached), shape = (output_dim,)
        self.last_output = self._zeros(output_dim)

        # Sliding window of recent message outputs
        self.window: deque = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Backend details
    # ------------------------------------------------------------------
    @property
    def runtime(self) -> str:
        """Self-expose the actual runtime (convention: report the actual value, not the configured value)."""
        return "torch(ncps)" if self.backend == "torch" else "numpy"

    def _zeros(self, *shape: int) -> Any:
        if self.backend == "torch":
            return self._torch.zeros(*shape)
        return np.zeros(shape, dtype=np.float32)

    def _to_input(self, x: Any) -> Any:
        """Unify the input into (B, S, input_dim). Accepts numpy (production path) or torch tensors."""
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
        """Clear the temporal state (hx / last_output / window). Global context reset (reset_context) goes through here."""
        self.hx = None
        self.last_output = self._zeros(self.output_dim)
        self.window.clear()

    def load_weights_npz(self, path: str) -> bool:
        """Load trained weights from npz (CRL export). Both backends can use this channel."""
        if self.backend == "torch":
            import numpy as _np

            with _np.load(path, allow_pickle=False) as z:
                arrays = {k: _np.asarray(z[k]) for k in z.files if k != "meta_json"}
            self.load_weights(arrays)
            return True
        self.cfc = NumpyWiredCfc.from_npz(path)
        return True

    def load_weights(self, arrays: dict[str, "Any"]) -> bool:
        """Pour trained weights into the current core. Both key styles are
        accepted:

        * ``layer{i}.{head}.{weight|bias}`` (numpy_cfc / npz export style)
        * ``rnn_cell.layer_{i}.…`` (ncps state_dict native style, legacy CRL .pt)

        The torch core converts to tensors and goes through
        ``load_state_dict(strict=False)`` (same as the old behavior;
        missing/extra keys do not blow up); the numpy core validates shapes
        and swaps weights in place (ff1/ff2 re-multiplied by masks).
        """
        norm = {}
        for k, v in arrays.items():
            nk = k
            if nk.startswith("rnn_cell.layer_"):
                # ncps native -> numpy_cfc style
                nk = nk[len("rnn_cell."):].replace("layer_", "layer", 1)
            norm[nk] = v
        if self.backend == "torch":
            t = self._torch
            state = {}
            for k, v in norm.items():
                # numpy_cfc style -> ncps state_dict native keys
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
    # Forward inference
    # ------------------------------------------------------------------
    def step(self, x: Any) -> Any:
        """Process input, update the hidden state, return the output vector.

        Args:
            x: input tensor, in either form:
                - (1, input_dim): a single embedding vector
                - (batch, seq, input_dim): an embedding sequence

        Returns:
            Output vector, shape = (output_dim,), taking the last time
            step's result. The CfC hidden state hx is maintained across
            calls (the core of temporal memory).
        """
        x = self._to_input(x)

        if self.backend == "torch":
            t = self._torch
            # CfC forward: output=(batch, seq, output_dim), hx=(batch, n_neurons)
            # hx is maintained across steps -- the core of temporal memory
            output, hx = self.cfc(x, self.hx)
            # Cut the cross-step gradient graph so it cannot grow unboundedly
            # on long sequences (suited to inference/retrieval)
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
    # Query vector
    # ------------------------------------------------------------------
    def get_query(self) -> Any:
        """Return the normalized query vector, a weighted fusion of the current
        output and the window mean.

        Hybrid architecture: query = cfc_weight * last_output + window_weight
        * window_mean, then L2 normalized. Default weights 0.5 / 0.5.

        Returns:
            Normalized query vector, shape = (output_dim,).
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
    # State persistence
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Return the current hidden state for checkpointing.

        **Always returns numpy arrays** (the torch core converts at the
        exit), so the checkpoint layer is zero-torch.

        Returns:
            Dict with hx / last_output / window and configuration info.
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
        """torch tensor / numpy array -> numpy (None passes through)."""
        if t is None:
            return None
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def load_state(self, state: dict) -> None:
        """Restore state from a checkpoint.

        The hidden state is bound to the embedding dimension: if the
        checkpoint came from another embedding model (different dimension),
        restoring directly would make later ``get_query()`` dimension-
        mismatch the query vector. In that case safely skip restoration
        (reset to empty state); later steps re-accumulate.

        Takes **numpy arrays** (the checkpoint npz's native form); also
        accepts torch tensors read from the legacy `.pt` path
        (auto-converted).

        Args:
            state: the dict returned by get_state. May be an empty dict --
                ``StateCheckpoint.load()`` returns ``{}`` when neither the
                main file nor the backup can be read (e.g. the state file
                was written broken, disk failure). It must degrade safely to
                the initial state rather than raise: this path runs at
                service startup, and crashing there means the whole MCP
                service cannot start.
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
        """Restore one piece of state: numpy (or a legacy .pt torch tensor) -> this backend's array."""
        arr = np.asarray(t, dtype=np.float32)
        if self.backend == "torch":
            return self._torch.from_numpy(arr)
        return arr
