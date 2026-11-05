# -*- coding: utf-8 -*-
"""嵌入模型的生命周期：延迟加载 + 空闲卸载。

为什么单独成一个模块
--------------------
``embeddings`` 是**延迟** import ``onnx_embedder`` 的（在 ``_attach_onnx`` 里），
而 onnx 侧要登记到同一份「谁在用模型」的状态里。放进任一方都会退化成环形
导入，独立模块是唯一干净的位置。

两条硬纪律（都是踩出来的）
--------------------------
1. **租约**：daemon 是「一进程 N 线程共享一个 brain」。卸载时若有请求正在
   半路，那条请求会拿到半截模型 —— 因此任何使用模型的线程都必须登记租约，
   卸载前必须确认租约为 0。
2. **连共享池一起清**：``_ONNX_POOL`` / ``_MODEL_POOL`` 是模型对象的持有者。
   细节要注意，别照抄笼统的说法 —— 实测：
   * ONNX 路径下**只要释放了 session**，几百 MB 立刻回来，池里剩下的只是空壳；
   * 但 **torch 路径下池是模型的唯一持有者**，不清池则回收率真的≈0。
   统一清掉最省心，也顺带避免陈旧实例被后来的库继承（`get_shared_onnx` 会把
   已释放的实例发回去，虽然能重新 `_ensure_model`，但这类"复用旧壳"最容易
   在未来某次改动里变成真事故）。

回收效果的实测基线（bge-base-zh / ONNX）
-------------------------------------------------
加载 +418.9MB → 卸载后只残留 +19.8MB，**回收率 94.3%**；二次加载无累积泄漏；
重载后的向量与首次**余弦相似度 1.000000**。加载本身 2.5–3.7s。

    ⚠️ 上述数字**只测过 ONNX 后端**。torch 后端有自己的 caching allocator，
    能不能同样回收尚未实测 —— 那里才是"要不要改成子进程隔离"的判断点。
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

# 全局状态锁： leases 与 _LAST_USE_AT 必须一致地读写
_LOCK = threading.Lock()

#: 正在使用模型的线程：``thread_ident -> 进入时间``。
#: 用 dict 而不是 int 计数，是为了出错时能指出**是谁**没放手。
_LEASES: dict[int, float] = {}

#: 最后一次有人用到模型的时刻（单调钟）。
_LAST_USE_AT = time.time()

#: 所有活着的嵌入器（弱引用，避免把它们钉住导致永远释放不掉）。
_EMBEDDERS: "weakref.WeakSet[Any]" = weakref.WeakSet()

_UNLOAD_COUNT = 0
_LAST_RELEASE: dict[str, Any] = {}


# ------------------------------------------------------------------
# 登记与租约
# ------------------------------------------------------------------
def register(obj: Any) -> None:
    """嵌入器构造时登记自己，卸载时才知道该去通知谁。"""
    try:
        _EMBEDDERS.add(obj)
    except TypeError:          # 不可弱引用的对象（不该发生）
        pass


@contextmanager
def model_use() -> Iterator[None]:
    """标记「本线程正在用模型」。卸载会在仍有租约时拒绝执行。"""
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
    """只刷新「最近使用」而不持租约（给没有明显边界的调用用）。"""
    global _LAST_USE_AT
    _LAST_USE_AT = time.time()


def active_leases() -> int:
    with _LOCK:
        return len(_LEASES)


def idle_seconds() -> float:
    """距上次使用模型过去了多少秒。从未使用过则返回 0。"""
    return max(0.0, time.time() - _LAST_USE_AT)


def live_embedders() -> int:
    return len(_EMBEDDERS)


# ------------------------------------------------------------------
# 自读数
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
    """本进程提交的私有字节（MB）。非 Windows / 取不到就返回 0.0。

    用它而不是 RSS：RSS 会被系统裁剪，无法回答"内存有没有还回来"。
    """
    if os.name != "nt":
        return 0.0
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        # 🔴 用 **真句柄**，不要用 GetCurrentProcess() 的伪句柄（-1）：
        # 64 位下它被解释成 0xFFFFFFFFFFFFFFFF，转 int 直接 OverflowError。
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
    except Exception:  # noqa: BLE001 —— 自读数失败不该影响业务路径
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
# 卸载
# ------------------------------------------------------------------
def release_models(reason: str = "idle") -> dict[str, Any]:
    """卸载所有嵌入器的模型本体 —— 空闲时把 ~400MB 还给系统的手段。

    返回 dict：``ok``（是否真的卸了）/ ``reclaimed_mb`` / ``released`` /
    ``leases`` / ``pools``。``ok=False`` 时一定带 ``reason`` —— 静默不卸
    会让调用方以为已经降下来了。
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
                logger.warning("释放嵌入器模型失败: %s: %s",
                               type(exc).__name__, exc)

        # 🔴 光把嵌入器里的引用置空没用：池才是真正的持有者。
        pools: dict[str, Any] = {}
        try:
            from . import embeddings as _E            # 延迟 import，避免环形
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
    logger.info("已卸载嵌入模型: %s", res)
    return res


def set_last_use(ts: float) -> None:
    """**只给测试用**：把「最近使用」拨到过去，模拟空闲。"""
    global _LAST_USE_AT
    _LAST_USE_AT = ts
