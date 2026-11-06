"""ONNX Runtime embedder: **does not load torch / sentence-transformers** at runtime.

Why this module exists
----------------------
The production process holds a single steady-state footprint of ~1.25GB, of
which the torch runtime ~160MB + sentence-transformers ~210MB is **pure
inference overhead** (measured with a segmented bill).
This module reproduces the same vectors using ``onnxruntime`` +
``tokenizers`` (Rust, both very light), replacing those two pieces with
ONNX inference; measured saving ~280MB.

Its relationship with SentenceTransformerEmbedder
-------------------------------------------------
**Fully aligned interface** (``embed`` / ``embed_query`` / ``embed_batch`` /
``max_seq_length`` / ``embedding_coverage`` / ``dim`` / ``model_key``),
so callers do not need to change a line. Switching is via the environment
variable ``HIPPOCAMPUS_EMB_BACKEND``:

* ``torch`` -- the old sentence-transformers path (behavior identical to today)
* ``onnx``  -- this module; **exported artifacts are mandatory**, missing ones raise, no silent fallback
* ``auto``  -- onnx when artifacts exist, otherwise torch (default)

Four easy-to-trip pitfalls (all tripped, or nearly)
---------------------------------------------------
1. **pooling must not be guessed**: bge-base-zh-v1.5 in sentence-transformers
   is **CLS pooling** (``1_Pooling/config.json: pooling_mode_cls_token=true``),
   not the common mean. Guessing mean makes the vectors completely mismatch,
   and **without raising any error**.
   This module only trusts the ``pooling`` field written in ``onnx_meta.json``.
2. **No silent fallback**: if a missing ONNX artifact quietly falls back to
   torch, nobody will ever know the memory saving never took effect. So in
   ``onnx`` mode, missing artifacts must raise.
3. **The dimension must not be guessed either** (a defect actually hit):
   ``word_embedding_dimension`` used to only recognize the top-level field,
   falling back to ``or 768`` when unreadable, while the exporter once wrote
   the dimension nested inside ``pooling_config`` => for a non-bge model
   (MiniLM 384) on ONNX, ``dim`` always reported 768, the LNN core was built
   for 768 while vectors actually came out 384, and it only blew up at
   ``matmul`` on the first ingest (x=386 vs w=770) -- the error site was
   miles away from the root cause. Now both nested forms are recognized,
   and if neither exists an explicit error is raised; see
   :func:`_resolve_embedding_dim`.
4. **Concurrency** (🔴 an early comment got this wrong, since corrected):
   The early comment here said "onnxruntime and the Rust tokenizer both lack
   the mutable-borrow problem ST has, therefore **no global encode lock**"
   -- **that was wrong, and the consequence is a process crash**.

   This class has in fact always been using ``enable_truncation`` /
   ``no_truncation`` (at line 185 below and inside ``embedding_coverage``),
   which hit the **same** ``tokenizers`` Rust mutable borrow as the ST side:
   concurrent calls panic. The difference is that the ST side raises a
   Python-level ``RuntimeError: Already borrowed`` (which reaches logs),
   while this path, running in the daemon's worker thread, takes the whole
   **process** down -- no traceback, no faulthandler dump; it presents as
   "the daemon disappearing silently".

   Reproduction scenario: two workspaces reuse the same daemon, two brains
   encode concurrently for the first time => daemon crashes, both host
   sessions hang together (waiting on a connection that is already dead).
   This is exactly the **normal usage** of "two tasks opened at the same
   time", not an edge case.

   The lock only wraps the tokenizer part (``session.run`` remains
   concurrent, no throughput sacrificed). Not using
   ``embeddings._ENCODE_LOCK`` is a dependency-direction issue:
   ``embeddings`` imports this module (for ``backend_kind`` /
   ``OnnxEmbedder``); importing back would create a cycle. The two
   embedders each guard their own tokenizer instance, which is
   semantically correct.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import threading
from typing import Optional

import numpy as np

#: Mutex for the tokenizer's **mutable borrow** (rationale: item 3 at the top
#: of this module). Guards only the tokenizer, not ``session.run``.
_ENCODE_LOCK = threading.RLock()

logger = logging.getLogger(__name__)

#: Environment variable: select the embedding backend (torch / onnx / auto)
ENV_BACKEND = "HIPPOCAMPUS_EMB_BACKEND"

#: Subdirectory name holding the ONNX artifacts under the model directory
ONNX_SUBDIR = "onnx"
ONNX_MODEL_FILE = "model.onnx"
ONNX_META_FILE = "onnx_meta.json"

# BGE zh family official query instruction prefix (kept consistent with
# SentenceTransformerEmbedder)
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# Model-load serial lock (same as the ST version: concurrent import /
# session construction would wait on each other)
_MODEL_LOAD_LOCK = threading.RLock()

# Model lifecycle (lazy load / idle unload). Shares the same state as the
# torch side.
from . import model_lifecycle as _ML  # noqa: E402
from ..config import load_config  # noqa: E402  (env reads folded into config)


def _with_model_use(fn):
    """Wraps methods that touch the session/tokenizer in a lease (rationale same as the ST-version decorator)."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with _ML.model_use():
            return fn(self, *args, **kwargs)

    return wrapper


def backend_kind() -> str:
    """Reads ``HIPPOCAMPUS_EMB_BACKEND``; unset or invalid values are all treated as ``auto``."""
    raw = (load_config().onnx_backend or "").strip().lower()
    return raw if raw in ("torch", "onnx", "auto") else "auto"


def find_onnx_artifacts(model_dir: str) -> tuple[str, dict] | None:
    """Find ONNX artifacts under the model directory. Returns ``(model_path, meta)`` when found, else None."""
    d = os.path.join(model_dir, ONNX_SUBDIR)
    model_path = os.path.join(d, ONNX_MODEL_FILE)
    meta_path = os.path.join(d, ONNX_META_FILE)
    if not (os.path.isfile(model_path) and os.path.isfile(meta_path)):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model_path, meta


def _resolve_embedding_dim(meta: dict, meta_file: str) -> int:
    """Parse the vector dimension from ``onnx_meta.json``; if unparseable,
    **raise explicitly** -- never guess.

    A defect actually hit: this used to read
    ``int(meta.get("word_embedding_dimension", 0)) or 768``, while
    ``scripts/export_onnx.py`` once wrote the dimension nested inside
    ``pooling_config`` => for a non-bge model (MiniLM 384) on ONNX,
    ``dim`` always reported 768 => the LNN core was built for 768 while
    vectors actually came out 384 => it only blew up at ``matmul`` on the
    first ingest (x=386 vs w=770), the error site miles away from the
    root cause.

    Two forms are recognized in order: top-level
    ``word_embedding_dimension`` (current exporter) and
    ``pooling_config.word_embedding_dimension`` (early exporter); if
    neither exists or is not a positive number => raise, pointing out
    the re-export command.
    """
    raw = meta.get("word_embedding_dimension")
    if raw in (None, "", 0):
        nested = meta.get("pooling_config")
        if isinstance(nested, dict):
            raw = nested.get("word_embedding_dimension")
    try:
        dim = int(raw)
    except (TypeError, ValueError):
        dim = 0
    if dim <= 0:
        raise ValueError(
            f"{meta_file} 里没有可用的 word_embedding_dimension"
            f"（顶层与 pooling_config 嵌套两处都没有）。不猜维度——猜错会让"
            f"向量与 torch 版完全对不上且不报错。请重跑 "
            f"python scripts/export_onnx.py --model <model dir>, "
            f"或改回 {ENV_BACKEND}=torch。"
        )
    return dim


class OnnxEmbedder:
    """Run a sentence-transformers-exported model via onnxruntime.

    Args:
        model_dir: **local model directory** (e.g.
            ``.../libs/models/bge-base-zh-v1.5``). This class does not accept
            Hub names -- its design premise is "no network access and no
            torch at runtime".
        model_name: model name reported outward (used as the vector slot key
            ``model_key``); must match the torch version, otherwise the same
            corpus would be treated as vectors of two different models.
    """

    def __init__(
        self,
        model_dir: str,
        model_name: str = "",
        libs_path: Optional[str] = None,
    ) -> None:
        self.model_dir = os.path.abspath(model_dir)
        self.model_name = model_name or os.path.basename(self.model_dir.rstrip("/\\"))
        self.model_key = self.model_name
        self.is_bge = "bge" in self.model_name.lower()
        self.model_source = "onnx"
        self.model_explicit = True
        self.resolved_model_path: str | None = None
        self._libs_path = libs_path
        #: F6: whether this instance is held by the process-level shared pool
        #: (True = another library for the same model is also using this
        #: copy). Stamped at **the moment of entering the pool**; see
        #: ``get_shared_onnx``.
        self.shared_from_pool = False

        found = find_onnx_artifacts(self.model_dir)
        if found is None:
            raise FileNotFoundError(
                f"The ONNX backend was selected explicitly ({ENV_BACKEND}=onnx), but {self.model_dir} has no "
                f"{ONNX_SUBDIR}/{ONNX_MODEL_FILE} + {ONNX_META_FILE}.\n"
                f"Run first: python scripts/export_onnx.py --model {self.model_dir}\n"
                f"or switch back to {ENV_BACKEND}=torch."
            )
        self._onnx_path, self._meta = found

        self._pooling = self._meta.get("pooling") or ""
        if self._pooling not in ("cls", "mean", "max"):
            raise ValueError(
                f"{self._onnx_path}: invalid pooling field in onnx_meta.json at "
                f"{self._pooling!r}（只认 cls/mean/max）。不猜 pooling —— "
                f"猜错会让向量与 torch 版完全对不上且不报错。"
            )
        self._max_seq_length = int(self._meta.get("max_seq_length", 512))
        self._meta_file = os.path.join(os.path.dirname(self._onnx_path), ONNX_META_FILE)
        self.dim = _resolve_embedding_dim(self._meta, self._meta_file)

        self._session = None
        self._tok = None
        self._model_ready = False
        # Register with the process-level lifecycle (weak reference; does not block reclamation)
        _ML.register(self)

    def release_model(self) -> bool:
        """Unload the session + tokenizer, returning ~400MB to the OS.

        Same reasoning as the ST version: **keep this instance** (its
        ``model_dir`` / ``_meta`` / pooling config are still reused), drop
        only the two memory-hogging objects, and set ``_model_ready`` back
        to False -- the next :meth:`_ensure_model` reloads it as before.

        Returns:
            Whether anything was actually unloaded (False if nothing was
            loaded in the first place).
        """
        did = False
        if self._session is not None:
            # ort.InferenceSession has no close(): dropping the reference is
            # enough; its native memory is freed when the refcount hits zero
            # (measured locally: PrivateUsage fell from 537MB to 138MB after
            # unload; see the mem_lifecycle probe).
            self._session = None
            did = True
        if self._tok is not None:
            self._tok = None
            did = True
        if did:
            self._input_names = []
            self._output_name = ""
            self._model_ready = False
        return did

    def __repr__(self) -> str:
        where = self.resolved_model_path or "(not yet loaded)"
        return f"{self.model_name} @ {where} [onnx]"

    # -- Loading ---------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model_ready:
            return
        with _MODEL_LOAD_LOCK:
            if self._model_ready:
                return
            self._load_model()

    def _load_model(self) -> None:
        """Load the onnxruntime session and the Rust tokenizer (**neither needs torch**)."""
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - missing-dependency branch
            raise ImportError(
                "ONNX 后端需要 onnxruntime：`pip install onnxruntime`（"
                "注意：它不依赖 torch，装它不会把 torch 拉回来）。\n"
                f"错误: {e}"
            ) from e
        try:
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ONNX 后端需要 tokenizers（HuggingFace 的 Rust 分词库）："
                "`pip install tokenizers`。\n"
                f"错误: {e}"
            ) from e

        # Flush the loading status to stderr first (stdout is the dedicated
        # JSON-RPC channel): keeps "slow" and "broken" distinguishable --
        # a few seconds of cold start is normal; repeated hangs mean look
        # at the daemon log.
        print(f"[hippocampus] loading embedding model (ONNX): {self._onnx_path}",
              file=sys.stderr, flush=True)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 🔴 Memory/latency trade-off for batch embed (measured evidence):
        # * arena/pattern **on** (default): single-sentence latency ~34ms
        #   (the lifeline of the recall path); but under certain batch
        #   contents the first large batch allocates ~2GB at once and the
        #   arena keeps it resident (no leak, does not grow with repeats;
        #   exact trigger conditions unknown; the production single-sentence
        #   path is unaffected).
        # * **off** (HIPPOCAMPUS_ONNX_NO_ARENA=1): batch memory completely
        #   flat (constant 820MB), but single-sentence latency ~1048ms
        #   (30x) -- for offline batch scripts only (reembed etc.).
        if (load_config().onnx_no_arena or "").strip() not in ("1", "true"):
            so.enable_mem_pattern = True
            so.enable_cpu_mem_arena = True
        else:
            so.enable_mem_pattern = False
            so.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(
            self._onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name

        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"the model directory lacks tokenizer.json: {tok_path}")
        self._tok = Tokenizer.from_file(tok_path)
        # Truncate to the model's hard cap: aligned with ST's max_seq_length
        # behavior (overflow silently dropped). These two calls mutate the
        # tokenizer's **mutable state**, so they also go through the lock.
        with _ENCODE_LOCK:
            self._tok.enable_truncation(max_length=self._max_seq_length)
            # 🔴 Measured: padding must **not** be fixed to max_seq_length.
            # The original `enable_padding(length=512)` made every query run
            # the forward pass at 512 tokens -- single-sentence latency
            # 480ms (batch-longest dynamic padding takes ~30ms, **16x**),
            # and with 36 sentences per batch the attention intermediates
            # (36×12×512×512) bloated the ORT arena by ~2GB resident. Under
            # CLS pooling + attention mask, padding is numerically
            # irrelevant (verified by cos=1.0), so batch-longest is pure win.
            self._tok.enable_padding()

        self.resolved_model_path = self._onnx_path
        # The session output is authoritative for the dim (the meta value is only an estimate)
        shape = self._session.get_outputs()[0].shape
        if len(shape) == 3 and isinstance(shape[2], int):
            self.dim = int(shape[2])
        # Same discipline as the ST version: set ready only after one
        # successful run, so other threads cannot slip in early.
        # 🔴 This must go through ``_forward`` rather than ``_encode``:
        # ``_encode`` calls ``_ensure_model``, but ``_model_ready`` is still
        # False here => infinite recursion (measured: RecursionError, and
        # it also masked the real loading error).
        _ = self._forward(["warmup"])
        print(f"[hippocampus] embedding model ready (ONNX): {self.model_name}"
              f"（{self.dim}D)", file=sys.stderr, flush=True)
        self._model_ready = True

    # -- Encoding --------------------------------------------------------
    def _forward(self, texts: list[str]) -> np.ndarray:
        """Pure forward pass: **no** load check (the caller guarantees the
        model is loaded).

        The only reason this is separate from :meth:`_encode` is the
        recursion red line above -- the warmup at the end of loading can
        only go through here.
        """
        # 🔴 The tokenizer's mutable borrow must be serialized (see item 3
        # at the top of the module): concurrent encode panics on the Rust
        # side, which in the daemon's worker thread presents as **the whole
        # process vanishing silently**. ``session.run`` stays outside the
        # lock -- that is the time-consuming part.
        with _ENCODE_LOCK:
            encodings = self._tok.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Some models have no token_type_ids input (recorded in meta.input_names at export time)
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = token_type_ids

        hidden = self._session.run([self._output_name], feed)[0]  # (B, L, H)

        if self._pooling == "cls":
            pooled = hidden[:, 0, :]
        elif self._pooling == "mean":
            mask = attention_mask[:, :, None].astype(np.float32)
            summed = (hidden * mask).sum(axis=1)
            counts = np.maximum(mask.sum(axis=1), 1e-9)
            pooled = summed / counts
        else:  # max
            mask = attention_mask[:, :, None].astype(np.float32)
            pooled = (hidden * mask + (1 - mask) * (-1e9)).max(axis=1)

        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return (pooled / norm).astype(np.float64)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """tokenize → onnx forward → pooling → L2 normalization (including lazy load)."""
        self._ensure_model()
        return self._forward(texts)

    @_with_model_use
    def embed(self, text: str) -> np.ndarray:
        """Document-side embedding (no query prefix). Returns an L2-normalized vector."""
        if not text:
            self._ensure_model()
            return np.zeros(self.dim, dtype=np.float64)
        return self._encode([text])[0]

    @_with_model_use
    def embed_query(self, text: str) -> np.ndarray:
        """Query-side embedding (BGE family gets the official instruction prefix)."""
        if self.is_bge:
            text = BGE_ZH_QUERY_PREFIX + text
        return self.embed(text)

    @_with_model_use
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Batch embedding. Returns ``(N, dim)``, each row already
        L2-normalized.

        🔴 **Do not skip empty strings**: sentence-transformers'
        ``encode(["", ...])`` still runs a forward pass for an empty string
        ([CLS][SEP]) and yields a **non-zero** vector, whereas a single
        :meth:`embed("")` returns a **zero** vector by existing convention.
        These two inconsistencies are the torch version's existing behavior,
        and this class must **replicate them verbatim** -- an early version
        short-circuited empty strings to the zero vector here, and the batch
        path then scored cos=0.000000 (acceptance failed outright).
        """
        if not texts:
            self._ensure_model()
            return np.zeros((0, self.dim), dtype=np.float64)
        # Chunking: stuffing tens of thousands at once blows up memory, and it aligns with ST's batch_size semantics
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        step = max(1, int(batch_size))
        for start in range(0, len(texts), step):
            chunk = texts[start : start + step]
            out[start : start + len(chunk)] = self._encode(chunk)
        return out

    @property
    def max_seq_length(self) -> int:
        """The model's hard cap on tokens (overflow silently dropped; same semantics as the torch version)."""
        return self._max_seq_length

    @_with_model_use
    def embedding_coverage(self, text: str) -> dict:
        """Truncation self-report -- **the returned keys must match the torch
        version verbatim**.

        The upper layer (``brain`` / ``brain_stats``) reads ``truncated`` /
        ``embedded_chars`` / ``dropped_ratio``. One missing key means
        **silent degradation**: writes still succeed, the data is still in
        the library, only nobody knows that "nothing beyond char N is
        retrievable" -- a red line even more insidious than wrong vectors.

        ``no_truncation()`` is mandatory: this class's tokenizer permanently
        holds ``enable_truncation(512)``; without turning it off, ids length
        is always <= 512 and ``truncated`` stays False -- the self-report
        would be rendered meaningless.
        """
        total = len(text or "")
        out = {
            "truncated": False,
            "total_chars": total,
            "embedded_chars": total,
            "tokens": 0,
            "max_seq_length": int(self._max_seq_length),
            "dropped_ratio": 0.0,
        }
        if total == 0 or out["max_seq_length"] <= 0:
            return out
        try:
            self._ensure_model()
            # Turning off truncation **temporarily** mutates tokenizer state;
            # colliding with another thread that is mid-encode is the same
            # mutable-borrow conflict -- must use the same lock as encode.
            # 🔴 no_truncation() is a plain method (permanently disables,
            # returns None), not a context manager -- `with
            # self._tok.no_truncation():` would raise TypeError right after
            # disabling truncation and get swallowed by the except below,
            # truncation would never come back, and any text over 512 tokens
            # would punch through to an ORT broadcast error. Must disable
            # explicitly and restore explicitly in finally.
            with _ENCODE_LOCK:
                self._tok.no_truncation()
                try:
                    enc = self._tok.encode(text, add_special_tokens=True)
                finally:
                    self._tok.enable_truncation(max_length=self._max_seq_length)
            ids = enc.ids
            n = len(ids)
            out["tokens"] = n
            limit = out["max_seq_length"]
            if n <= limit:
                return out
            out["truncated"] = True
            kept = 0
            offs = enc.offsets
            if offs:
                idx = min(limit, len(offs)) - 1
                end = int(offs[idx][1])
                if 0 < end <= total:
                    kept = end
            if kept <= 0:
                # Same fallback as the torch version: without offsets,
                # estimate proportionally -- better an approximate value
                # than regressing to "not truncated".
                kept = int(total * limit / max(n, 1))
            out["embedded_chars"] = max(0, min(total, kept))
            out["dropped_ratio"] = round(1.0 - out["embedded_chars"] / total, 4)
        except Exception as exc:      # the self-report must never break the write in return
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out


# ==================================================================
# F6 · Process-level shared pool (one resident copy per model)
# ==================================================================
# Why share only "the heavy layer" instead of "the whole embedder instance":
#   ``OnnxEmbedder`` carries only session + Rust tokenizer, no state that
#   varies per library, so it can be shared safely; whereas the upper layer
#   (``SentenceTransformerEmbedder`` / ``MemoryManager``) carries
#   **decision metadata** -- ``model_source`` (whether the model name was
#   given explicitly or fell out of a fallback) decides "when the
#   in-library model conflicts with the current model, error out or follow
#   the library data" (see ``manager._sync_embedder_with_store``). Sharing
#   the whole instance, the second library inherits the first library's
#   source; a library that should "follow the library data" may be judged
#   "explicitly specified" and raise ``EmbedderModelMismatchError`` outright
#   => **the library fails to start**. So only the heavy object is shared,
#   and each library keeps its own thin shell and its own metadata.
#
# Benefit (measured baseline: single-library daemon 543.5MB, two libraries
# 947.8MB, second-library delta ≈404MB = the second copy of weights): after
# sharing, a second library for the same model no longer loads weights.

#: Escape hatch: ``HIPPOCAMPUS_SHARED_EMBEDDER=0`` disables sharing (behavior reverts to pre-F6).
ENV_SHARED = "HIPPOCAMPUS_SHARED_EMBEDDER"

#: Pool cap (a fuse). Normal operation needs only 1 copy; the headroom is
#: for the "libraries with different in-library models" scenario (e.g. one
#: bge library + one MiniLM library => 2 copies, which is right). Beyond
#: the cap, new instances simply stay out of the pool, **no error** -- they
#: still work, just without reuse. This guards against model-swap
#: experiments pinning N copies of weights in the pool forever.
POOL_MAX = 4

_ONNX_POOL: "dict[tuple[str, str], OnnxEmbedder]" = {}
_ONNX_POOL_LOCK = threading.Lock()


def shared_embedder_enabled() -> bool:
    """Master switch for the shared embedder (default on; ``HIPPOCAMPUS_SHARED_EMBEDDER=0`` turns it off)."""
    raw = (load_config().shared_embedder or "").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _pool_key(model_dir: str, model_name: str) -> tuple[str, str]:
    """Pool key = (absolute path, outward-facing model name).

    ``model_name`` is included because the same directory can be referenced
    under different names, and ``model_key`` (the vector slot key) takes the
    name -- different names must occupy two slots; one shared instance must
    not report the same ``model_key``.
    """
    return (os.path.abspath(model_dir), model_name or "")


def get_shared_onnx(
    model_dir: str,
    model_name: str = "",
    libs_path: Optional[str] = None,
) -> "OnnxEmbedder":
    """Fetch an ``OnnxEmbedder``: **the same (dir, name) reuses one instance**.

    On a pool hit, no session is rebuilt and tokenizer.json is not re-read
    -- that is exactly the ~400MB and the 30s load that F6 saves. The
    returned instance may **not yet** have run ``_ensure_model()``
    (construction is light; loading happens on first encode); callers
    simply call ``_ensure_model`` as usual -- it has a double-checked lock
    inside.

    ``shared_from_pool`` gets stamped on the pooled instance: receiving
    True means these weights are process-level shared (the second library
    necessarily receives True).
    """
    key = _pool_key(model_dir, model_name)
    if shared_embedder_enabled():
        with _ONNX_POOL_LOCK:
            hit = _ONNX_POOL.get(key)
        if hit is not None:
            return hit
    emb = OnnxEmbedder(model_dir=key[0], model_name=model_name, libs_path=libs_path)
    if shared_embedder_enabled():
        with _ONNX_POOL_LOCK:
            if len(_ONNX_POOL) < POOL_MAX:
                _ONNX_POOL[key] = emb
                emb.shared_from_pool = True
    return emb


def onnx_pool_stats() -> dict:
    """Self-report the pool's current state (for ``brain_stats`` and tests)."""
    with _ONNX_POOL_LOCK:
        return {
            "enabled": shared_embedder_enabled(),
            "size": len(_ONNX_POOL),
            "max": POOL_MAX,
            "keys": [f"{k[1]}@{k[0]}" for k in _ONNX_POOL],
        }


def reset_onnx_pool() -> int:
    """Clear the ONNX pool (**tests only**); returns the number cleared.

    Tests need to verify within one process that "disabling sharing really
    builds separate instances"; without a reset, instances left in the pool
    by a previous case pollute the result, producing a false green.
    """
    with _ONNX_POOL_LOCK:
        n = len(_ONNX_POOL)
        _ONNX_POOL.clear()
    return n
