# -*- coding: utf-8 -*-
"""Embedding model lifecycle: lazy load + idle unload.

Why a separate module
---------------------
``embeddings`` **lazily** imports ``onnx_embedder`` (inside
``_attach_onnx``), and the onnx side needs to register into the same
"who is using the model" state. Putting it in either side would degrade
into a circular import; a standalone module is the only clean spot.

Two hard disciplines (both learned the hard way)
------------------------------------------------
1. **Leases**: the daemon is "one process, N threads, one shared brain".
   If a request is in flight during unload, that request gets half a model
   -- so every thread using the model must register a lease, and unload
   must confirm the lease count is 0 first.
2. **Clear the shared pools too**: ``_ONNX_POOL`` / ``_MODEL_POOL`` are the
   holders of the model objects. The details matter; do not copy vague
   rules -- measured:
   * On the ONNX path, **releasing the session alone** returns hundreds of
     MB immediately; what remains in the pool is just an empty shell;
   * but on the **torch path the pool is the model's only holder**; without
     clearing the pool the reclaim rate is really ≈0.
   Clearing both uniformly is the least trouble, and it also prevents
   stale instances from being inherited by later libraries
   (`get_shared_onnx` would hand back the released instance; it can re-run
   `_ensure_model`, but this kind of "reusing an old shell" is the most
   likely to become a real incident in some future change).

Measured baseline of reclaim effectiveness (bge-base-zh / ONNX)
---------------------------------------------------------------
Load +418.9MB → after unload only +19.8MB remains, **reclaim rate 94.3%**;
a second load shows no cumulative leak; vectors after reload have **cosine
similarity 1.000000** with the first. Loading itself takes 2.5–3.7s.

    ⚠️ The numbers above were **measured on the ONNX backend only**. The
    torch backend has its own caching allocator; whether it reclaims the
    same way is unmeasured -- that is the decision point for "should this
    be changed to subprocess isolation".
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import gc
import logging
import os
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Global state lock: leases and _LAST_USE_AT must be read/written consistently
_LOCK = threading.Lock()

#: Threads currently using the model: ``thread_ident -> entry time``.
#: A dict instead of an int counter, so that on failure we can point at
#: **who** failed to let go.
_LEASES: dict[int, float] = {}

#: The last moment anyone used the model (monotonic clock).
_LAST_USE_AT = time.time()

#: All living embedders (weak refs, so we don't pin them and make them impossible to free).
_EMBEDDERS: "weakref.WeakSet[Any]" = weakref.WeakSet()

_UNLOAD_COUNT = 0
_LAST_RELEASE: dict[str, Any] = {}


# ------------------------------------------------------------------
# Registration and leases
# ------------------------------------------------------------------
def register(obj: Any) -> None:
    """Embedders register themselves at construction time, so unload knows whom to notify."""
    try:
        _EMBEDDERS.add(obj)
    except TypeError:          # Not weakref-able (should never happen)
        pass


@contextmanager
def model_use() -> Iterator[None]:
    """Mark "this thread is using the model". Unload refuses to run while leases remain."""
    ident = threading.get_ident()
    with _LOCK:
        _LEASES[ident] = time.time()
    global _LAST_USE_AT
    _LAST_USE_AT = time.time()
    try:
        yield
    finally:
        with _LOCK:
            _LEASES.pop(ident, None)
        _LAST_USE_AT = time.time()


def touch() -> None:
    """Refresh "last used" without holding a lease (for calls without a clear boundary)."""
    global _LAST_USE_AT
    _LAST_USE_AT = time.time()


def active_leases() -> int:
    with _LOCK:
        return len(_LEASES)


def idle_seconds() -> float:
    """Seconds since the model was last used. Returns 0 if never used."""
    return max(0.0, time.time() - _LAST_USE_AT)


def live_embedders() -> int:
    return len(_EMBEDDERS)


# ------------------------------------------------------------------
# Reading our own process stats
# ------------------------------------------------------------------
class _PMC_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def own_private_mb() -> float:
    """Private committed bytes of this process (MB). Returns 0.0 on non-Windows / when unavailable.

    Used instead of RSS: RSS gets trimmed by the system and cannot answer
    "did the memory actually come back".
    """
    if os.name != "nt":
        return 0.0
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        # 🔴 Use a **real handle**, not the pseudo handle from GetCurrentProcess() (-1):
        # on 64-bit it is interpreted as 0xFFFFFFFFFFFFFFFF, and int() raises OverflowError outright.
        k32.OpenProcess.restype = wt.HANDLE
        k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        h = k32.OpenProcess(0x0400 | 0x0010, False, os.getpid())
        if not h:
            return 0.0
        try:
            pmc = _PMC_EX()
            pmc.cb = ctypes.sizeof(pmc)
            if not psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return 0.0
            return pmc.PrivateUsage / 1048576.0
        finally:
            k32.CloseHandle(h)
    except Exception:  # noqa: BLE001 -- a failed self-read must not affect the business path
        return 0.0


def stats() -> dict[str, Any]:
    return {
        "leases": active_leases(),
        "idle_seconds": round(idle_seconds(), 1),
        "live_embedders": live_embedders(),
        "private_mb": round(own_private_mb(), 1),
        "unload_count": _UNLOAD_COUNT,
        "last_release": dict(_LAST_RELEASE),
    }


# ------------------------------------------------------------------
# Unload
# ------------------------------------------------------------------
def release_models(reason: str = "idle") -> dict[str, Any]:
    """Unload every embedder's model body -- the means of giving ~400MB back to the system when idle.

    Returns a dict: ``ok`` (whether an unload really happened) /
    ``reclaimed_mb`` / ``released`` / ``leases`` / ``pools``. ``ok=False``
    always carries ``reason`` -- silently not unloading would make the
    caller believe memory already went down.
    """
    global _UNLOAD_COUNT, _LAST_RELEASE
    with _LOCK:
        leases = dict(_LEASES)
        if leases:
            return {
                "ok": False,
                "reason": "busy",
                "leases": len(leases),
                "held_by": [round(time.time() - t, 1) for t in leases.values()],
            }
        before = own_private_mb()

        released = 0
        for obj in list(_EMBEDDERS):
            rel = getattr(obj, "release_model", None)
            if rel is None:
                continue
            try:
                released += 1 if rel() else 0
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to release the embedder model: %s: %s",
                               type(exc).__name__, exc)

        # 🔴 Just nulling the references inside the embedders does nothing: the pools are the real holders.
        pools: dict[str, Any] = {}
        try:
            from . import embeddings as _E            # Lazy import, avoids a cycle
            pools["torch"] = _E.reset_model_pool()
        except Exception as exc:  # noqa: BLE001
            pools["torch"] = f"n/a({type(exc).__name__})"
        try:
            from . import onnx_embedder as _OE
            pools["onnx"] = _OE.reset_onnx_pool()
        except Exception as exc:  # noqa: BLE001
            pools["onnx"] = f"n/a({type(exc).__name__})"

        gc.collect()
        gc.collect()
        after = own_private_mb()

        _UNLOAD_COUNT += 1
        _LAST_RELEASE = {
            "at": time.time(),
            "reason": reason,
            "released": released,
            "reclaimed_mb": round(before - after, 1),
            "pools": pools,
        }
        res = {
            "ok": True,
            "reason": reason,
            "released": released,
            "pools": pools,
            "before_mb": round(before, 1),
            "after_mb": round(after, 1),
            "reclaimed_mb": round(before - after, 1),
        }
    logger.info("embedding model unloaded: %s", res)
    return res


def set_last_use(ts: float) -> None:
    """**Test-only**: wind "last used" back into the past to simulate idleness."""
    global _LAST_USE_AT
    _LAST_USE_AT = ts
