# -*- coding: utf-8 -*-
"""Text embedding utilities.

Provides two embedders:
    - SentenceTransformerEmbedder: semantic embedder based on sentence-transformers (recommended).
      When the model name is empty, fall back in order along ``HIPPOCAMPUS_EMB_MODEL`` → ``all-MiniLM-L6-v2``;
      the fallback emits a warning; for Chinese corpora explicitly specify ``BAAI/bge-base-zh-v1.5``.
    - HashEmbedder: deterministic embedder based on hash projection (MVP/fallback).
      No external dependencies, but cannot capture semantic similarity.

Both implement a unified interface:
    embed(text)        -> document-side embedding, used when writing memories
    embed_query(text)  -> query-side embedding, used at recall time (BGE and similar models add an instruction prefix here)
    embed_batch(texts) -> batch embedding
    model_key          -> slot key for the multi-model vector store

This interface is enforced by ``create_embedder()``: a new backend missing any
of these methods errors out at initialization instead of raising AttributeError
at recall time.

Model cache directory resolution (fix for coverage-report blocker 2):
    The local model directory no longer only recognizes ``<project>/libs/models``;
    instead it probes a candidate list, and the ``HIPPOCAMPUS_EMB_CACHE``
    environment variable can explicitly override it. Candidate order:

        1. ``HIPPOCAMPUS_EMB_CACHE`` (if set, only this one is honored)
        2. ``<project>/libs/models``
        3. ``<project parent>/libs/models``   ← where BGE actually lives on this machine
        4. ``<project parent>/libs``
        5. ``~/.cache/huggingface/hub`` (standard HF cache, with cache_folder structure)

    When offline (``HF_HUB_OFFLINE=1``) and no candidate contains the model,
    **raise immediately** with an error listing the attempted paths instead of
    letting SentenceTransformer go online and wait for a timeout.
"""

from __future__ import annotations

import functools
import hashlib
import locale
import logging
import os
import sys
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# H2: embedding backend selection (torch / onnx / auto). Only a function that
# reads env vars is imported here; the real heavy dependencies (onnxruntime /
# tokenizers) are lazily imported inside _attach_onnx — a process without ONNX
# enabled should pay no cost for it.
from .onnx_embedder import (  # noqa: E402
    backend_kind,
    get_shared_onnx,
    onnx_pool_stats,
    shared_embedder_enabled,
)
from ..config import load_config  # noqa: E402  (env reading folded into config)

# Model lifecycle (lazy loading / idle unload). It lives in a **separate module**
# because both the torch side and the onnx side need it; putting it into either
# one would create a circular import.
from . import model_lifecycle as _ML


def _with_model_use(fn):
    """Wrap methods that use the model in a "lease".

    Why the lease exists: the daemon is one process with N threads sharing one
    brain; if unloading fires while a request is mid-flight, that request ends
    up with half a model. Once registered, :func:`_ML.release_models` **refuses
    to run** while the lease count is non-zero and reports the holder, instead
    of leaving the memory mess for someone else to clean up.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with _ML.model_use():
            return fn(self, *args, **kwargs)

    return wrapper


# **Process-level serial lock** for model loading (background in
# SentenceTransformerEmbedder._ensure_model).
# RLock instead of Lock: re-entry from an exception stack must not self-deadlock.
_MODEL_LOAD_LOCK = threading.RLock()

# **Process-level serial lock** for inference (encode).
#
# Why it exists: every call to ``SentenceTransformer.encode`` goes through
# ``tokenization_utils_tokenizers.set_truncation_and_padding`` →
# ``self._tokenizer.enable_truncation(...)``, which is a **mutable borrow** of
# the Rust-side tokenizer. Two threads encoding concurrently hit Rust's RefCell
# double-borrow and raise
#
#     RuntimeError: Already borrowed
#
# In direct stdio mode there is one thread per process, so this never happens;
# after daemonization one process serves N sessions = N threads sharing the
# same embedder, and collisions become the norm. Measured (two connections
# concurrently writing 15 records each): several ``store_memory`` calls raised
# this error at random, the tool returned isError, and records were silently
# lost (only 20~29 of 30 survived, varying with timing).
#
# Cost: concurrent encodes are serialized. The daemon's goal is saving memory,
# not raising throughput; correctness comes first. A single encode takes tens
# of milliseconds, so serialization will not become a bottleneck.
_ENCODE_LOCK = threading.RLock()


# ==================================================================
# F6 · process-level shared pool (torch side: SentenceTransformer model objects)
# ==================================================================
# What gets pooled is the **model object** (the ~400MB blob), not the embedder
# instance — see the F6 section at the end of the ``onnx_embedder`` module for
# why: an embedder instance carries "decision metadata" such as ``model_source``;
# sharing it would let a second library inherit the first library's semantics,
# possibly judging "follow the library's model" as "explicitly specified" and
# erroring out so the library cannot even start.
#
# Value = ``(model, dim)``: dim is stored together because a pool hit must skip
# the test encode (that encode only probes the dimension; rerunning it is slow
# and risks the tokenizer mutable-borrow collision).
_MODEL_POOL: "dict[tuple[str, str], tuple[object, int]]" = {}
_MODEL_POOL_LOCK = threading.Lock()

#: Pool cap (same as the onnx side's POOL_MAX)
POOL_MAX = 4


def reset_model_pool() -> int:
    """Clear the torch-side model pool (**for tests only**); return the number of entries removed."""
    with _MODEL_POOL_LOCK:
        n = len(_MODEL_POOL)
        _MODEL_POOL.clear()
    return n


def embedder_pool_stats() -> dict:
    """Self-report of the process-level shared pool state (``brain_stats`` / tests).

    ``torch_size`` and ``onnx_size`` are reported separately: in a production
    process running only ONNX, ``torch_size`` is always 0; a non-zero value means
    this process is actually still on the torch fallback — exactly the
    self-report channel for "I thought I was saving memory" style silent failures.
    """
    with _MODEL_POOL_LOCK:
        torch_size = len(_MODEL_POOL)
        keys = [f"{k[1]}@{k[0]}" for k in _MODEL_POOL]
    onnx = onnx_pool_stats()
    return {
        "enabled": shared_embedder_enabled(),
        "torch_size": torch_size,
        "torch_keys": keys,
        "onnx_size": onnx["size"],
        "onnx_keys": onnx["keys"],
        "max": POOL_MAX,
    }
# Env var name: this string used to be scattered across server.py / brain.py /
# various scripts, each site writing its own copy with its own default, so "one
# entry point forgot to set it" would silently fall back to MiniLM.
ENV_MODEL = "HIPPOCAMPUS_EMB_MODEL"

# Allows force-switching when "the explicitly specified model mismatches the
# library's vectors" (for offline experiments comparing different models).
# Unset: a mismatch **errors out with clear instructions**; never silently swap
# the model and recompute the whole library.
ENV_ALLOW_MODEL_SWITCH = "HIPPOCAMPUS_ALLOW_MODEL_SWITCH"

# Application-layer default model (used only when nothing is explicit and the
# library has no model to follow either).
# Why MiniLM: it is the common baseline for English semantic retrieval, and both
# the LoCoMo (English) evaluation and the Mem0 comparison evaluation pinned it
# (see scripts/eval_mem0_*.py, bench_bge.py).
# **Chinese corpora must explicitly specify BGE** — this default is not meant
# for Chinese production libraries.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

#: Default model for Chinese corpora (BGE-base-zh-v1.5, 768D).
#: The semantics are **routing**, not replacement: English and other languages
#: still go through :data:`DEFAULT_MODEL_NAME`.
#: Takes effect as the application default only when "the system language is
#: Chinese" AND the user **did not** explicitly configure a model.
ZH_DEFAULT_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

#: Switch to turn off "pick the default model by system language" (on by default).
#: Set ``0/false/no/off`` to disable → always MiniLM.
#: An explicit ``HIPPOCAMPUS_EMB_MODEL`` always takes priority over this switch.
ENV_MODEL_AUTO = "HIPPOCAMPUS_EMB_MODEL_AUTO"


def _default_libs_path() -> str:
    """Infer the project ``libs/`` from this file's location (same rule as the constructor's inference)."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return os.path.join(project_root, "libs")


def _cache_roots(libs_path: str = "", cache_folder: str = "") -> list[str]:
    """Return model cache root candidates by priority (source of truth for
    ``_resolve_cache_candidates``).

    Hoisted to module level so "pick the default model by language" can check
    whether the Chinese model exists locally **without constructing an embedder**
    (constructing one would drag in torch / ONNX, which is too costly).
    """
    explicit = cache_folder or load_config().emb_cache or ""
    if explicit:
        return [os.path.abspath(explicit)]

    project_libs = libs_path or ""
    parent_libs = (
        os.path.join(os.path.dirname(os.path.dirname(project_libs)), "libs")
        if project_libs
        else ""
    )
    candidates = [
        os.path.join(project_libs, "models") if project_libs else "",
        os.path.join(parent_libs, "models") if parent_libs else "",
        parent_libs,
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ]
    out: list[str] = []
    for c in candidates:
        if c and os.path.abspath(c) not in out:
            out.append(os.path.abspath(c))
    return out


def local_model_dir(model_name: str, roots: list[str]) -> str | None:
    """Find the local copy of ``model_name`` under ``roots``; return ``None`` if absent.

    Source of truth for ``SentenceTransformerEmbedder._find_local_model`` (the
    latter now delegates here; behavior is bit-for-bit identical). The three hit
    shapes are documented in that method's docstring.
    """
    short = model_name.split("/")[-1]
    cls = SentenceTransformerEmbedder
    for root in roots:
        cand = os.path.join(root, short)
        if os.path.isdir(cand) and cls._looks_like_model(cand):
            return cand
        if os.path.basename(root) == short and cls._looks_like_model(root):
            return root
        hit = cls._find_hf_snapshot(root, short)
        if hit:
            return hit
    return None


def _win_ui_language_id() -> int | None:
    """On Windows, get the LANGID of the "system UI language"; return ``None`` on non-Windows or on failure.

    🔴 **Do not use ``LANG`` to decide**: Git Bash sets ``LANG`` to
    ``en_US.UTF-8`` while the system is actually Chinese — measured on this very
    machine. Trusting the env would misclassify a Chinese system as English.
    """
    if os.name != "nt":  # pragma: no cover - non-Windows goes through LANG/locale
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
    except Exception:  # noqa: BLE001 - detection failure must not take down the main flow
        return None


def _is_zh_language_id(langid: int) -> bool:
    """Is the LANGID's primary language Chinese (``LANG_CHINESE == 0x04``, covering Simplified and Traditional)?"""
    return (langid & 0x03FF) == 0x04


def _is_zh_locale(tag: str) -> bool:
    """``zh_CN.UTF-8`` / ``Chinese (Simplified)_China`` / ``zh-TW`` all count as Chinese."""
    t = (tag or "").strip().lower()
    return t.startswith("zh") or t.startswith("chinese")


def detect_system_language() -> str:
    """Detect the system language; return ``"zh"`` for Chinese, ``""`` otherwise.

    Probe order (system level first, then process environment, so a shell's env
    cannot mask the system language):

        1. Primary language of the Windows ``GetUserDefaultUILanguage()`` LANGID;
        2. POSIX ``LC_ALL`` / ``LC_MESSAGES`` / ``LANG``;
        3. ``locale.getlocale()[0]`` as a fallback (on Windows this also yields
           ``"Chinese (Simplified)_China"``).

    Any step that fails **continues to the next**; if all fail, treat as
    non-Chinese (fall back to the English baseline — the only safe default
    direction for this function).
    """
    langid = _win_ui_language_id()
    if langid is not None:
        return "zh" if _is_zh_language_id(langid) else ""

    for env_key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        val = os.environ.get(env_key, "").strip()
        if val and val.upper() not in ("C", "POSIX"):
            return "zh" if _is_zh_locale(val) else ""

    try:
        tag = (locale.getlocale() or ("", ""))[0] or ""
    except Exception:  # noqa: BLE001
        tag = ""
    return "zh" if _is_zh_locale(tag) else ""


def auto_model_selection_enabled() -> bool:
    """The "pick the default model by system language" switch (on by default; ``HIPPOCAMPUS_EMB_MODEL_AUTO=0`` disables).

    Same rule as ``onnx_embedder.shared_embedder_enabled``: blacklist semantics —
    only ``0/false/off/no`` disable; unset/empty/odd values all count as enabled.
    """
    raw = (load_config().emb_model_auto or "").strip().lower()
    return raw not in ("0", "false", "off", "no")


def onnx_runtime_available() -> bool:
    """Whether the onnxruntime dependency is importable (cheap spec check —
    no import, no model).

    FZ (2026-11-08 from-zero install test): local ONNX artifacts do NOT imply
    this dependency — a clean install without onnxruntime mounted the ONNX
    backend (artifacts were cached) and then crashed at first embed. The
    mount decision must ask THIS, not artifact presence; ``_attach_onnx``
    degrades to the torch backend (or raises for an explicit onnx request)
    when this returns False.
    """
    import importlib.util

    return importlib.util.find_spec("onnxruntime") is not None


def default_model_for_system() -> tuple[str, str]:
    """Return the "application default model" based on system language, as ``(model_name, reason)``.

    **Only affects the "nothing configured at all" path**: explicit argument >
    ``HIPPOCAMPUS_EMB_MODEL`` > this function. The returned ``source`` is still
    ``"default"`` (not ``"env"``), and this **must not change** —
    ``MemoryManager._sync_embedder_with_store`` uses ``source == "default"`` to
    decide "may follow the library's existing model"; changing it to ``"env"``
    would make every existing Chinese library raise
    :class:`EmbedderModelMismatchError` right after upgrading.

    ``reason`` values:

        * ``auto_off``          — switch off, always the English baseline;
        * ``not_zh``            — system is not Chinese, use the English baseline;
        * ``zh_local``          — Chinese and the Chinese model is present locally (use it directly);
        * ``zh_download``       — Chinese and a first-time download is needed (online; works but slow);
        * ``zh_unavailable``    — Chinese but offline with no local copy → **fall back to the English baseline**
          (slightly worse retrieval beats a service that cannot start).
    """
    if not auto_model_selection_enabled():
        return DEFAULT_MODEL_NAME, "auto_off"
    if detect_system_language() != "zh":
        return DEFAULT_MODEL_NAME, "not_zh"

    roots = _cache_roots(_default_libs_path())
    if local_model_dir(ZH_DEFAULT_MODEL_NAME, roots) is not None:
        return ZH_DEFAULT_MODEL_NAME, "zh_local"

    offline = os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False")
    if offline:
        return DEFAULT_MODEL_NAME, "zh_unavailable"
    return ZH_DEFAULT_MODEL_NAME, "zh_download"


def _emit_warning(msg: str) -> None:
    """Send the warning through both logging and stderr.

    An MCP service speaks the stdio protocol: logs must go to stderr or they get
    mistaken for protocol frames, while script scenarios often configure only
    logging. Emitting on both sides guarantees "this warning is always seen by
    someone".
    """
    logging.warning(msg)
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:  # pragma: no cover - a closed stderr must not take down the main flow
        pass


def _warn_default_model(model_name: str) -> None:
    """When falling back to the application default model, explain "why this model" (prevents "silently using the wrong model" from recurring).

    The reason is fetched fresh via :func:`default_model_for_system` — the warning
    text and the selection decision share the same judgment, avoiding a
    self-contradiction like "the text says English baseline but a Chinese model
    is actually running".
    """
    _, reason = default_model_for_system()
    if reason == "zh_unavailable":
        _emit_warning(
            f"[embeddings] system language is Chinese, but the Chinese model {ZH_DEFAULT_MODEL_NAME} is not present "
            f"locally and offline mode is active; falling back to the English baseline {model_name} (384D) -- "
            f"Chinese semantic retrieval will be noticeably weaker. Fix: put {ZH_DEFAULT_MODEL_NAME} into the model "
            f"cache directory (or set {ENV_CACHE}), or remove HF_HUB_OFFLINE to let it download."
        )
        return
    if reason == "zh_download":
        _emit_warning(
            f"[embeddings] system language is Chinese; using {ZH_DEFAULT_MODEL_NAME} (768D) by default. "
            f"The first run downloads about 400MB and fails in offline environments -- prefetch in advance, "
            f"or set HIPPOCAMPUS_EMB_MODEL={DEFAULT_MODEL_NAME} to use the English baseline explicitly."
        )
        return
    if reason == "zh_local":
        logging.info(
            "[embeddings] system language is Chinese; using %s by default (768D, local copy already present)",
            model_name,
        )
        return
    _emit_warning(
        f"[embeddings] no embedding model specified; falling back to the default {model_name} "
        f"(English baseline, 384D). For Chinese corpora set {ENV_MODEL}=BAAI/bge-base-zh-v1.5, "
        f"or pass model_name explicitly in the constructor."
    )


def resolve_embedder_model(explicit: str | None = None) -> tuple[str, str]:
    """Resolve the effective embedding model name; return ``(model_name, source)``.

    ``source`` values:

        - ``"explicit"`` — passed in explicitly by the caller (argument or ``--model``);
        - ``"env"``      — from the :data:`ENV_MODEL` environment variable;
        - ``"default"``  — neither given, falls to :func:`default_model_for_system`:
          **system language Chinese and the Chinese model available → BGE-zh,
          otherwise → MiniLM**. Whichever is chosen, the source is recorded as
          ``"default"`` (for the "follow the library's model" decision).

    **This is the single source of truth for model resolution.** Previously each
    entry point wrote its own ``os.environ.get("HIPPOCAMPUS_EMB_MODEL", "")``
    with its own default; missing one spot silently ran MiniLM. Centralized here,
    callers only need to log the source, and "which model am I actually using
    right now" never needs guessing again.
    """
    name = (explicit or "").strip()
    if name:
        return name, "explicit"
    name = (load_config().emb_model or "").strip()
    if name:
        return name, "env"
    # Nothing set → pick the application default by **system language**
    # (Chinese → BGE-zh, otherwise → MiniLM).
    # 🔴 source must still be marked "default", not "env":
    # MemoryManager._sync_embedder_with_store uses `source == "default"` to decide
    # "may follow the library's existing model". Marking "env" would make every
    # existing Chinese library (MiniLM 384D) raise EmbedderModelMismatchError right
    # after upgrading — i.e. this auto-selection must stay "transparent" to old
    # libraries and only take effect for **newly created** libraries.
    default_name, _reason = default_model_for_system()
    return default_name, "default"


# ==================================================================
# HashEmbedder — dependency-free fallback embedder
# ==================================================================

class HashEmbedder:
    """Deterministic text embedder based on hash projection.

    No external dependencies; guarantees the same text always produces the same
    embedding. But it cannot capture semantic similarity and is only suitable
    for quick tests.

    Args:
        dim: output embedding dimension, default 32.
        salt: hash salt, used to separate different embedding spaces.
    """

    def __init__(self, dim: int = 32, salt: str = "hippocampus_mcp") -> None:
        self.dim = dim
        self.salt = salt
        # Model identifier: serves as the slot key in the multi-model vector store
        self.model_key = f"hash:{salt}:{dim}"

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a fixed-dimension normalized vector.

        Args:
            text: text to embed.

        Returns:
            L2-normalized numpy vector, shape = (dim,).
        """
        if not text:
            return np.zeros(self.dim, dtype=np.float64)

        vec = np.zeros(self.dim, dtype=np.float64)

        words = text.strip().lower().split()
        if not words:
            words = [text.strip().lower()]

        for n in (1, 2, 3):
            for i in range(max(len(words) - n + 1, 1)):
                if n == 1 and len(words) > 1:
                    gram = words[i]
                elif len(words) >= n:
                    gram = " ".join(words[i : i + n])
                else:
                    gram = " ".join(words)

                h = hashlib.sha256(
                    f"{self.salt}:{gram}".encode("utf-8")
                ).digest()

                for j in range(self.dim):
                    byte_idx = j % len(h)
                    val = (h[byte_idx] / 127.5) - 1.0
                    vec[j] += val

        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec = np.ones(self.dim, dtype=np.float64)
            vec = vec / np.linalg.norm(vec)

        return vec

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query text.

        The hash embedder has no instruction-prefix concept; fully equivalent to
        :meth:`embed`. This method exists to satisfy the embedder protocol —
        ``MemoryManager.recall()`` always goes through ``embed_query``; without
        it the fallback backend would raise ``AttributeError`` at recall time
        (coverage report D5, fixed).

        Args:
            text: query text.

        Returns:
            L2-normalized numpy vector, shape = (dim,).
        """
        return self.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batch.

        Args:
            texts: list of texts to embed.

        Returns:
            (N, dim) numpy matrix.
        """
        return np.vstack([self.embed(t) for t in texts])


    @property
    def max_seq_length(self) -> int:
        """The hash backend hashes character n-grams and never looks at tokens, so there is no truncation cap (0 = no cap)."""
        return 0

    def embedding_coverage(self, text: str) -> dict:
        """The hash backend sees the entire text and never truncates."""
        total = len(text or "")
        return {
            "truncated": False,
            "total_chars": total,
            "embedded_chars": total,
            "tokens": total,
            "max_seq_length": 0,
            "dropped_ratio": 0.0,
        }

# ==================================================================
# SentenceTransformerEmbedder — semantic embedder (recommended)
# ==================================================================

class SentenceTransformerEmbedder:
    """Semantic text embedder based on sentence-transformers.

    Defaults to all-MiniLM-L6-v2 (384 dims, English). For Chinese, the BGE
    family is recommended:
        - BAAI/bge-small-zh-v1.5 (512 dims, 24M, fast on CPU)
        - BAAI/bge-base-zh-v1.5 (768 dims, 102M, runs on CPU, Chinese retrieval ndcg@10 ~70)

    The BGE family follows the official usage: the query side gets the
    instruction prefix, the document side does not.
    embed() = document side (default), embed_query() = query side (prefix added automatically).

    The model is lazily loaded on the first embed() call (avoids startup delay).
    The first load downloads the model weights; afterwards it loads from the
    local cache.

    Args:
        model_name: model name. **Explicitly passing it is recommended**; when
            empty, take the ``HIPPOCAMPUS_EMB_MODEL`` environment variable, then
            fall back to ``all-MiniLM-L6-v2`` (a warning is emitted on fallback,
            see :func:`resolve_embedder_model`).
            For Chinese, "BAAI/bge-base-zh-v1.5" or "BAAI/bge-small-zh-v1.5" is suggested.
        cache_folder: model cache directory. Defaults to the models directory under the project libs/.
        libs_path: project libs/ directory path, added to sys.path to load dependencies.
    """

    # Default model and dimension. Note: these are the **application fallback
    # defaults**, not "recommended values for Chinese production libraries".
    # For Chinese corpora explicitly specify BAAI/bge-*-zh-v1.5 (see
    # resolve_embedder_model).
    DEFAULT_MODEL = DEFAULT_MODEL_NAME
    DEFAULT_DIM = 384

    # Official BGE zh query instruction prefix (query side only)
    BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    # Known model dimensions (to estimate before init; the model's actual output is authoritative)
    KNOWN_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "BAAI/bge-small-zh-v1.5": 512,
        "BAAI/bge-base-zh-v1.5": 768,
        "BAAI/bge-large-zh-v1.5": 1024,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    }

    # Env var: explicitly specify the model cache root directory
    ENV_CACHE = "HIPPOCAMPUS_EMB_CACHE"

    # Required files of a model directory (to judge "is this directory really a model")
    _MODEL_MARKERS = ("config.json", "modules.json")

    def __init__(
        self,
        model_name: str = "",
        cache_folder: Optional[str] = None,
        libs_path: Optional[str] = None,
    ) -> None:
        # Not explicitly specified → check env → default in order, and record the
        # **source**. When `model_source == "default"` a loud warning fires: this
        # path is the historic entry point of "the service silently fell back to
        # MiniLM" and must be made visible.
        resolved, source = resolve_embedder_model(model_name)
        self.model_name = resolved
        self.model_source = source
        self.model_explicit = source != "default"
        if source == "default":
            _warn_default_model(resolved)
        model_name = resolved
        # BGE family takes the full name (BAAI/xxx); short names are compatible
        if model_name in ("bge-base-zh-v1.5", "bge-base-zh"):
            model_name = "BAAI/bge-base-zh-v1.5"
        elif model_name in ("bge-small-zh-v1.5", "bge-small-zh"):
            model_name = "BAAI/bge-small-zh-v1.5"
        elif model_name in ("bge-large-zh-v1.5", "bge-large-zh"):
            model_name = "BAAI/bge-large-zh-v1.5"
        self.model_name = model_name
        self.dim = self.KNOWN_DIMS.get(model_name, self.DEFAULT_DIM)
        self._model = None
        #: True only when the model is **fully ready** (including the
        #: dimension-probing test encode). Letting threads through when ``_model``
        #: is merely non-empty would let them race with an unfinished test encode.
        self._model_ready = False
        self._cache_folder = cache_folder
        self._libs_path = libs_path
        # Model directory actually used for loading (diagnostics; known only
        # after loading, None on failure)
        self.resolved_model_path: str | None = None

        # Model identifier: slot key in the multi-model vector store (switching
        # models switches slots; no cross-overwriting)
        self.model_key = self.model_name

        # The BGE family needs the instruction prefix on the query side
        self.is_bge = model_name.startswith("BAAI/bge")

        # Infer the default libs path (project root/libs)
        if self._libs_path is None:
            # Infer the project root from this file's location
            # embeddings.py sits at src/hippocampus_mcp/embeddings.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # src/hippocampus_mcp -> src -> project_root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self._libs_path = os.path.join(project_root, "libs")

        # Resolve cache candidate directories (defect blocker 2)
        self._cache_candidates = self._resolve_cache_candidates(cache_folder)
        # Back-compat attribute: default to the first candidate (existing ones
        # preferred), for os.makedirs / logging
        self._cache_folder = next(
            (d for d in self._cache_candidates if os.path.isdir(d)),
            self._cache_candidates[0],
        )

        # H2: ONNX backend (composition). If mounted, this instance becomes a thin
        # shell and all encoding is forwarded to _onnx; torch / sentence-transformers
        # are **never imported even once** — that is exactly the ~280MB saving mechanism.
        self._onnx: object | None = None
        #: F6: whether this instance's heavy object came from the process-level
        #: shared pool (other libraries of the same model are using it too)
        self.shared_from_pool = False
        self._attach_onnx()
        # Register with the process-level lifecycle. Unloading needs to know
        # "whom to notify", and a strong reference must not be used here
        # (otherwise the embedder is never collected and unloading is meaningless).
        _ML.register(self)

    def release_model(self) -> bool:
        """Unload the model itself, **keeping this instance and its backend decision**.

        Deliberately **not** setting ``_onnx`` to None: ``_ensure_model``'s fast
        path relies on ``self._onnx is not None`` to take the ONNX branch; nulling
        it would make the next load fall back to the torch path (slow and ~280MB
        heavier), i.e. self-inflicted harm. Let OnnxEmbedder unload its own
        session/tokenizer and set ``_model_ready`` False; a reload naturally
        returns to the original backend.

        Returns:
            Whether the "loaded → unloaded" state reset happened (False if it was never loaded).
        """
        # 🔴 Cannot use the child object's return value to decide whether to reset
        # self: the copy in the shared pool may already have been unloaded by another
        # embedder (in which case it returns False), so this instance has no model
        # left while ``_model_ready`` still sits at True. On the torch path this is
        # a real accident — ``embed()``'s fast path fails
        # ``_model is not None and _model_ready`` and falls through to
        # ``self._model.encode(...)`` while _model is already None.
        was_loaded = bool(
            self._model_ready or self._model is not None
            or self._onnx is not None)
        released_any = False
        if self._onnx is not None:
            try:
                released_any = bool(self._onnx.release_model()) or released_any
            except Exception as exc:  # noqa: BLE001
                logger.warning("释放 ONNX 会话失败: %s: %s",
                               type(exc).__name__, exc)
        if self._model is not None:
            self._model = None
            released_any = True
        if was_loaded:
            self._model_ready = False
            self.shared_from_pool = False
            # ⚠️ Do not clear resolved_model_path: on the ONNX path a reload goes
            # through ``_onnx._ensure_model`` and never re-enters ``_load_model``;
            # clearing it permanently loses the self-report of "which directory the
            # model actually came from".
        return bool(was_loaded or released_any)

    def _pool_key(self, load_path: str) -> tuple[str, str]:
        """Torch-side pool key = (backend kind, absolute path / model name).

        The kind is included because ``auto`` falling back to torch and explicit
        ``torch`` get the same weights and could share; but kind never changes
        within a process, so carrying it only makes the self-report more readable.
        """
        p = os.path.abspath(load_path) if os.path.isdir(load_path) else load_path
        return (backend_kind(), p)

    # ------------------------------------------------------------------
    # Cache directory resolution
    # ------------------------------------------------------------------
    def _resolve_cache_candidates(self, cache_folder: Optional[str]) -> list[str]:
        """Return the cache root directory candidates ordered by priority.

        When the explicit argument or ``HIPPOCAMPUS_EMB_CACHE`` exists, **only it
        is honored** (respect explicit configuration, avoiding the confusion of
        "I pointed at a path and it still looked elsewhere"); otherwise use the
        default candidate chain.

        The implementation moved to module-level :func:`_cache_roots` (so "pick
        the default model by language" can probe local models without
        constructing an embedder); this method remains a thin shell.
        """
        return _cache_roots(self._libs_path or "", cache_folder or "")

    def _find_local_model(self) -> str | None:
        """Find the local model directory among the candidates; return ``None`` if absent.

        Three hit shapes:
            - ``<candidate root>/<short name>`` (e.g. ``libs/models/bge-base-zh-v1.5``) —
              the shape BGE has on this machine;
            - ``<candidate root>`` itself is the model directory (basename == short
              name, and config.json exists);
            - **HF hub layout** ``<candidate root>/models--<org>--<name>/snapshots/<hash>/``
              — what sentence-transformers persists after downloading from
              HuggingFace. Without this case, a hub-layout-only model on this
              machine (e.g. all-MiniLM-L6-v2) would raise FileNotFoundError offline
              even though **the cache is actually complete** — just unrecognized.
        """
        return local_model_dir(self.model_name, self._cache_candidates)

    @classmethod
    def _find_hf_snapshot(cls, root: str, short: str) -> str | None:
        """Find the model under ``root`` following the HF hub layout; return the snapshot directory or ``None``.

        Target shape: ``<root>/models--<org>--<short>/snapshots/<hash>/``.

        Recovering the org from ``model_name`` is unreliable (``"all-MiniLM-L6-v2"``
        carries no org at all, yet the actual directory is named
        ``models--sentence-transformers--all-MiniLM-L6-v2``), so this matches the
        directory-name suffix fuzzily and never constructs an org prefix.

        Directory-name case is insensitive on Windows and sensitive on Linux, so
        both sides compare lowercased. With multiple snapshots, take the largest
        in sorted order (and filter empty shells via ``_looks_like_model``).
        """
        if not os.path.isdir(root):
            return None
        suffix = "--" + short.lower()
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("models--") or not entry.lower().endswith(suffix):
                continue
            snaps = os.path.join(root, entry, "snapshots")
            if not os.path.isdir(snaps):
                continue
            for h in sorted(os.listdir(snaps), reverse=True):
                cand = os.path.join(snaps, h)
                if os.path.isdir(cand) and cls._looks_like_model(cand):
                    return cand
        return None

    @classmethod
    def _looks_like_model(cls, path: str) -> bool:
        """Does the directory contain the model-required files (missing means it is just an empty shell / half-finished download)?"""
        return os.path.isfile(os.path.join(path, "config.json")) or os.path.isfile(
            os.path.join(path, "modules.json")
        )

    def describe_source(self) -> str:
        """One-line description of "where this embedder will finally load from and whether it has loaded"."""
        where = self.resolved_model_path or "(not yet loaded)"
        return f"{self.model_name} @ {where}"

    def _attach_onnx(self, *, from_store: bool = False) -> bool:
        """Try to mount the ONNX backend; return **whether it mounted**.

        🔴 **The call timing is the entire difficulty of this method** (a pit fixed
        once): at construction time ``self.model_name`` may just be a **fallback
        default**, while the actually effective model is only settled in
        ``manager._follow_store_model()`` (after loading the library). So this
        method **must be callable repeatedly**, and when "the model name is not
        settled yet" it **must not raise** — otherwise explicit ``onnx`` would blow
        up during assembly, and ``auto`` would silently fall back to torch
        (presenting as "I thought I was on ONNX, but I was not").

        Three-state semantics:

        * ``torch`` → return False directly (today's behavior, zero change)
        * ``auto``  → use artifacts if present, quietly go torch otherwise
        * ``onnx``  → missing artifacts/dependencies **raise**, with two exceptions
          (neither counts as the user explicitly specifying a model; raising would
          only cause collateral damage):
          - ``model_source == "default"`` (the model name came from a fallback)
          - ``from_store=True`` (the model name came from following the library's data)
        """
        kind = backend_kind()
        if kind == "torch":
            return False
        # Is the model name "explicitly specified by the user": strict raising only when explicit
        strict = kind == "onnx" and not from_store and self.model_source != "default"
        # FZ (2026-11-08 from-zero install test): local ONNX artifacts do NOT
        # imply a usable ONNX backend — a clean install without onnxruntime
        # mounted the backend (artifacts were cached) and then crashed at
        # first embed. Check the dependency, not just the artifacts: ``auto``
        # falls back to torch loudly; explicit ``onnx`` raises HERE with the
        # actionable pip hint instead of crashing at embed time.
        if not onnx_runtime_available():
            if strict:
                raise ImportError(
                    "HIPPOCAMPUS_EMB_BACKEND=onnx but onnxruntime is not "
                    "installed: `pip install onnxruntime` (it does not pull "
                    "torch), or switch HIPPOCAMPUS_EMB_BACKEND to auto/torch."
                )
            logger.warning(
                "embedding backend: torch (ONNX artifacts present but "
                "onnxruntime is not installed — `pip install onnxruntime`, "
                "or the [torch] extra for the sentence-transformers backend)")
            return False
        try:
            from .onnx_embedder import OnnxEmbedder, find_onnx_artifacts
        except ImportError as e:
            if strict:
                raise ImportError(
                    "HIPPOCAMPUS_EMB_BACKEND=onnx 但 onnxruntime/tokenizers 不可用：%s" % e
                ) from e
            logger.info(
                "embedding backend: torch (ONNX dependencies unavailable: %s)", e,
            )
            return False
        # The ONNX path only accepts local directories (it never goes online nor touches torch)
        model_dir = self._find_local_model()
        if not model_dir and os.path.isdir(self.model_name):
            model_dir = self.model_name
        if not model_dir:
            if strict:
                raise FileNotFoundError(
                    "HIPPOCAMPUS_EMB_BACKEND=onnx 但找不到本地模型目录 %r"
                    % self.model_name
                )
            logger.info(
                "embedding backend: torch (%s has no local model directory; retry once the model name is fixed)",
                self.model_name,
            )
            return False
        if find_onnx_artifacts(model_dir) is None:
            if strict:
                raise FileNotFoundError(
                    "HIPPOCAMPUS_EMB_BACKEND=onnx 但 %s 下没有 ONNX 产物。"
                    "先跑：python scripts/export_onnx.py --model %s"
                    % (model_dir, model_dir)
                )
            logger.info(
                "embedding backend: torch (no ONNX artifacts under %s%s)",
                model_dir,
                ", and the model name is not final yet" if not from_store else "",
            )
            return False
        # F6: on a shared-pool hit, do not create a new session / re-read tokenizer.json.
        # ``OnnxEmbedder`` only holds session + tokenizer, no per-library state,
        # so sharing is safe; this instance's own ``model_source`` / ``model_explicit``
        # are unaffected (that is "where the model name came from" and must be
        # computed per library).
        self._onnx = get_shared_onnx(
            model_dir=model_dir, model_name=self.model_name
        )
        self.shared_from_pool = bool(getattr(self._onnx, "shared_from_pool", False))
        self.dim = getattr(self._onnx, "dim", self.dim)
        # The pooled instance may already have been loaded by another library; take
        # the path self-report from the instance itself (None when not loaded,
        # consistent with the fact that "this instance has not encoded yet").
        self.resolved_model_path = getattr(self._onnx, "resolved_model_path", None)
        logger.info("embedding backend: ONNX (%s) %s; torch/ST are not loaded at runtime",
                    model_dir, "(reusing the in-process shared instance)" if self.shared_from_pool else "")
        return True

    def _ensure_model(self) -> None:
        """Lazily load the model. Load on first call, reuse afterwards.

        **Must be serialized across threads** (see ``_MODEL_LOAD_LOCK``): model
        loading triggers ``import sentence_transformers`` → ``scipy.special`` →
        loading the ``_ufuncs.pyd`` C extension. If two threads import the same
        modules simultaneously, CPython's per-module locks make them wait on each
        other (measured: the warmup thread and a tool thread both entered
        ``_ensure_model`` and both stalled on importlib's module locks).
        Warmup has moved to a background thread, so this concurrency is the norm
        rather than an edge case.
        """
        # Fast path: if already loaded, zero lock overhead. **Must check
        # _model_ready**: after the model object is assigned, a test encode still
        # has to run before it is truly usable; checking only _model would let
        # concurrent threads through early (see the comment at the end of _load_model).
        if self._onnx is not None:
            # ONNX path: the model is just session + Rust tokenizer; loading is
            # very fast and lacks ST's "Rust mutable borrow" concurrency problem.
            self._onnx._ensure_model()
            self.dim = self._onnx.dim
            self._model_ready = True
            return
        if self._model is not None and self._model_ready:
            return
        with _MODEL_LOAD_LOCK:
            if self._model is not None and self._model_ready:
                return               # Double check: another thread may have finished loading while we waited for the lock
            # Loading also counts as "in use": otherwise this load, which happens
            # to straddle the idle threshold, would have its session pulled out
            # from under it by the other side's unload.
            _ML.touch()
            self._load_model()

    def _load_model(self) -> None:
        """Actually construct the model (called by ``_ensure_model`` only while holding the lock)."""
        # Use the HuggingFace China mirror to avoid connection timeouts
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # Append libs/ to sys.path as a **fallback** (not overriding
        # system-installed dependencies).
        # 🔴 Measured 2026-11-03: this used to be `sys.path.insert(0, libs)`; a
        # leftover `regex` in libs (cp310's `_regex.cp310-win_amd64.pyd`) failed to
        # load on 3.11, shadowing the good package ahead of site-packages, and the
        # error surfaced as "cannot import name '_regex' … circular import",
        # looking like missing dependencies.
        # Appending to the end is the correct semantics for "in-project portable
        # dependencies": use the system's if present, only then the project's —
        # insert(0) lets one stale wheel wreck the whole embedding chain.
        if self._libs_path and self._libs_path not in sys.path:
            sys.path.append(self._libs_path)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            # Lean-core era (0.2.2): sentence-transformers lives in the
            # [torch] extra, not the core install — the hint must say so.
            raise ImportError(
                "sentence_transformers is not installed in this interpreter. "
                "Install the torch extra: `pip install \"hippocampus-mcp[torch]\"` "
                "(or `pip install sentence-transformers` here).\n"
                f"Error: {e}\n"
                f"libs_path: {self._libs_path}"
            ) from e

        # Local directory first: look for a ready model directory along the candidate chain
        load_path = self._find_local_model()
        offline = os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False")

        if load_path is None:
            if os.path.isdir(self.model_name):
                # model_name itself is a directory path
                load_path = self.model_name
            elif offline:
                # Offline with no local copy → error out explicitly; do not let
                # SentenceTransformer go online and wait for a timeout
                tried = "\n".join(f"    - {c}" for c in self._cache_candidates)
                raise FileNotFoundError(
                    f"离线模式（HF_HUB_OFFLINE=1）下找不到模型 {self.model_name!r} 的本地副本。\n"
                    f"已尝试的缓存根目录：\n{tried}\n"
                    f"  每个根目录下都找过 <根>/"
                    f"{self.model_name.split('/')[-1]}/\n"
                    f"修复方式（任选）：\n"
                    f"  1) 把模型放到上述任一目录下（推荐 <项目上级>/libs/models/…）；\n"
                    f"  2) 设置环境变量 {self.ENV_CACHE}=<模型所在目录>；\n"
                    f"  3) 去掉 HF_HUB_OFFLINE 让它可以联网下载。"
                )
            else:
                # Online: let SentenceTransformer resolve/download the usual way
                load_path = self.model_name

        if os.path.dirname(load_path):
            os.makedirs(os.path.dirname(load_path), exist_ok=True)
        else:
            os.makedirs(self._cache_folder, exist_ok=True)

        # F6: shared-pool hit ⇒ use the model object another library already
        # loaded; no import of sentence_transformers, no weight reading, no test
        # encode. With multiple libraries open at once this saves the second copy
        # of the weights (measured ≈404MB) and a second ~30s load.
        pool_key = self._pool_key(load_path)
        if shared_embedder_enabled():
            with _MODEL_POOL_LOCK:
                hit = _MODEL_POOL.get(pool_key)
            if hit is not None:
                self._model, self.dim = hit
                self.resolved_model_path = load_path
                self._model_ready = True
                self.shared_from_pool = True
                logger.info(
                    "嵌入模型复用进程内共享实例：%s（%dD)，跳过加载",
                    load_path, self.dim,
                )
                return

        # Load the model
        # Loading takes tens of seconds; flush one stderr status line first —
        # making "slow" distinguishable from "broken" (stdout is the JSON-RPC-only
        # channel; status goes to stderr, same as launcher._diag).
        print(f"[hippocampus] loading embedding model: {load_path}"
              f"(tens of seconds of cold start is normal; for repeated hangs check the daemon log)",
              file=sys.stderr, flush=True)
        self._model = SentenceTransformer(
            load_path,
            cache_folder=self._cache_folder,
        )
        self.resolved_model_path = load_path

        # H7: _prepare_torch no longer imports torch unconditionally (the numpy
        # LNN production path has zero torch). This ST fallback path is the other
        # door through which torch actually enters, so the single-thread limit is
        # added here (Windows OpenMP multithreading occasionally access-violates;
        # see server._prepare_torch).
        try:
            import torch as _torch

            _torch.set_num_threads(1)
        except Exception:  # noqa: BLE001 - a failed thread limit must not fail model loading
            pass

        # Update the actual dimension (in case the model differs from expectations).
        # This encode must also go through _ENCODE_LOCK: colliding with any
        # business-side encode would raise "Already borrowed" (the Rust
        # tokenizer's mutable borrow).
        with _ENCODE_LOCK:
            test_vec = self._model.encode("test", normalize_embeddings=True)
        self.dim = len(test_vec)
        print(f"[hippocampus] embedding model ready: {load_path}（{self.dim}D)",
              file=sys.stderr, flush=True)
        # Set the flag **last**: before this, ``self._model`` is already non-empty;
        # without this flag another thread would sail through ``_ensure_model``'s
        # fast path and race with this thread's unfinished test encode — measured
        # as exactly why the daemon's first tool call blew up.
        self._model_ready = True

        # F6: put into the pool for other libraries of the same model in this
        # process to reuse (over the cap, skip pooling without erroring — this
        # copy just will not be reused).
        if shared_embedder_enabled():
            with _MODEL_POOL_LOCK:
                if len(_MODEL_POOL) < POOL_MAX:
                    _MODEL_POOL[pool_key] = (self._model, self.dim)
                    self.shared_from_pool = True

    @_with_model_use
    def embed(self, text: str) -> np.ndarray:
        """Embed text into a fixed-dimension normalized semantic vector (document side, no instruction prefix).

        Args:
            text: text to embed.

        Returns:
            L2-normalized numpy vector, shape = (dim,).
        """
        if self._onnx is not None:
            return self._onnx.embed(text)

        if not text:
            return np.zeros(self.dim, dtype=np.float64)

        self._ensure_model()

        # encode returns a numpy array; normalize_embeddings=True does L2 normalization.
        # Why the lock: see _ENCODE_LOCK — the tokenizer's enable_truncation is a
        # mutable borrow; concurrent calls raise "Already borrowed", and after the
        # tool layer swallows the exception it presents as **silently lost records**.
        with _ENCODE_LOCK:
            vec = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vec, dtype=np.float64)

    @_with_model_use
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query text (query side).

        The official BGE zh requirement: queries get the instruction prefix,
        documents do not; other models behave the same as embed().

        Args:
            text: query text.

        Returns:
            L2-normalized numpy vector, shape = (dim,).
        """
        if self.is_bge:
            text = self.BGE_ZH_QUERY_PREFIX + text
        return self.embed(text)

    @_with_model_use
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batch, exploiting GPU batch acceleration.

        Args:
            texts: list of texts to embed.

        Returns:
            (N, dim) numpy matrix, each row L2-normalized.
        """
        if self._onnx is not None:
            return self._onnx.embed_batch(texts)

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)

        self._ensure_model()

        with _ENCODE_LOCK:
            vecs = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
        return np.asarray(vecs, dtype=np.float64)

    @property
    def max_seq_length(self) -> int:
        """The maximum number of tokens the embedding model accepts (0 = unavailable, treated as "unknown").

        BGE / MiniLM transformer backends are **512**, a **hard model cap**:
        anything beyond it is silently dropped at encode time and contributes
        **zero** to the vector. Measured (a Chinese library's main card at 1374
        tokens): ``cos(embed(first 600 chars), embed(full text)) = 1.000000`` —
        nothing after character 600 is retrievable at all. The write side must use
        it for truncation self-reporting, otherwise "it was written but is
        invisible to retrieval" would never be known to anyone.
        """
        # On the torch path this (indirectly) loads the model, so record a lease
        # here too, or it may collide with idle unloading; the ONNX path only
        # reads metadata, so the wrapper overhead is negligible.
        _ML.touch()
        if self._onnx is not None:
            return self._onnx.max_seq_length
        try:
            self._ensure_model()
            limit = getattr(self._model, "max_seq_length", 0)
            if not limit:
                tok = getattr(self._model, "tokenizer", None)
                limit = getattr(tok, "model_max_length", 0) if tok is not None else 0
        except Exception:
            return 0
        try:
            return int(limit)
        except (TypeError, ValueError):
            return 0

    @_with_model_use
    def embedding_coverage(self, text: str) -> dict:
        """Self-report: the visible fraction of this text that **actually enters the vector** (truncation self-report).

        Content beyond ``max_seq_length`` is silently truncated — the write
        succeeds, read-back finds it in the library, ``brain_stats`` looks normal,
        and **only retrieval quietly gets worse**. This computes "how much was
        actually included" explicitly for the caller.

        Returns:
            ``{"truncated": bool, "total_chars": int, "embedded_chars": int,
             "tokens": int, "max_seq_length": int, "dropped_ratio": float}``.
            When ``truncated`` is True, original text after ``embedded_chars``
            is **unretrievable**.
        """
        if self._onnx is not None:
            return self._onnx.embedding_coverage(text)

        total = len(text or "")
        out = {
            "truncated": False,
            "total_chars": total,
            "embedded_chars": total,
            "tokens": 0,
            "max_seq_length": int(self.max_seq_length),
            "dropped_ratio": 0.0,
        }
        if total == 0 or out["max_seq_length"] <= 0:
            return out
        try:
            self._ensure_model()
            tok = getattr(self._model, "tokenizer", None)
            if tok is None:
                return out
            fast = bool(getattr(tok, "is_fast", False))
            # Same _ENCODE_LOCK: ``tokenizer.__call__`` internally also goes
            # through ``set_truncation_and_padding`` → ``enable_truncation`` (even
            # with truncation=False), and colliding with a concurrent encode means
            # "Already borrowed". Self-reporting must never in turn break the
            # write, nor break someone else's write.
            with _ENCODE_LOCK:
                enc = tok(
                    text,
                    truncation=False,
                    add_special_tokens=True,
                    return_offsets_mapping=fast,
                )
            ids = enc.get("input_ids") or []
            n = len(ids)
            out["tokens"] = n
            limit = out["max_seq_length"]
            if n <= limit:
                return out
            out["truncated"] = True
            kept = 0
            offs = enc.get("offset_mapping") if fast else None
            if offs:
                idx = min(limit, len(offs)) - 1
                end = int(offs[idx][1])
                if 0 < end <= total:
                    kept = end
            if kept <= 0:
                # No offsets available (non-fast tokenizer / special tokens) →
                # estimate proportionally; better an approximation than falling
                # back to "not truncated".
                kept = int(total * limit / max(n, 1))
            out["embedded_chars"] = max(0, min(total, kept))
            out["dropped_ratio"] = round(1.0 - out["embedded_chars"] / total, 4)
        except Exception as exc:  # self-reporting must never in turn break the write
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out



def embedding_coverage(embedder: object, text: str) -> dict:
    """Truncation self-report for any embedder.

    When the backend does not implement ``embedding_coverage`` this does **not
    silently skip**, but returns an "explicitly unknown" dict with
    ``truncated=None`` — per the same criterion as B2-1: silent degradation is
    more dangerous than an explicit error, and the caller must be able to tell
    "confirmed not truncated" from "not checked".
    """
    fn = getattr(embedder, "embedding_coverage", None)
    if callable(fn):
        try:
            out = fn(text)
            if isinstance(out, dict):
                return out
        except Exception as exc:
            return {
                "truncated": None, "total_chars": len(text or ""),
                "embedded_chars": None, "tokens": None,
                "max_seq_length": None, "dropped_ratio": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "truncated": None, "total_chars": len(text or ""),
        "embedded_chars": None, "tokens": None,
        "max_seq_length": None, "dropped_ratio": None,
        "error": "the embedder does not provide embedding_coverage; truncation cannot be determined",
    }

# ==================================================================
# Factory function — pick the embedder from configuration
# ==================================================================

def create_embedder(
    backend: str = "sentence_transformer",
    dim: int = 384,
    **kwargs,
) -> "HashEmbedder | SentenceTransformerEmbedder":
    """Create an embedder instance.

    Args:
        backend: embedding backend type.
            "sentence_transformer" — use SentenceTransformerEmbedder (recommended).
            "hash" — use HashEmbedder (dependency-free, quick tests).
        dim: embedding dimension (only effective for HashEmbedder;
             SentenceTransformerEmbedder's dimension is model-determined).
        **kwargs: extra arguments passed to the embedder constructor.

    Returns:
        An embedder instance implementing the embed/embed_query/embed_batch interface.

    Raises:
        TypeError: the embedder lacks a protocol-required method. Every backend
            must provide ``embed`` / ``embed_query`` / ``embed_batch`` and
            ``model_key`` together: ``recall()`` goes through ``embed_query``,
            and missing it would only blow up **at recall time** (coverage report
            D5), hence the early interception here.
    """
    if backend == "sentence_transformer":
        embedder = SentenceTransformerEmbedder(**kwargs)
    elif backend == "hash":
        embedder = HashEmbedder(dim=dim, **kwargs)
    else:
        raise ValueError(
            f"未知的嵌入后端: {backend}. "
            f"支持: 'sentence_transformer', 'hash'"
        )

    missing = [
        name
        for name in ("embed", "embed_query", "embed_batch", "model_key")
        if not hasattr(embedder, name)
    ]
    if missing:
        raise TypeError(
            f"embedding backend {backend!r} does not satisfy the embedder protocol; missing: {', '.join(missing)}"
        )
    return embedder
