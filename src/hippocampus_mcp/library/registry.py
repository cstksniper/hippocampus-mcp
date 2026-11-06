# -*- coding: utf-8 -*-
"""进程内多库注册表（库路径 -> HippoBrain）。

**daemon 与 server 共用同一份**：这份实现最初为「一个 daemon 持多个库」
而写，后来直连 stdio 也要多库，于是下沉成公共模块——
``daemon.BrainRegistry`` 这个名字保持不变（对外导入兼容）。

三件必须做成这样的事：

1. **同一 key 只建一次** —— 这是「模型 N→1」的落点。两个 brain 指向同一份
   ``memories.json`` 是 v0.4 明确禁止的（写入互不可见 + 整体覆写丢数据）。
2. **每库独立锁**（不搞一个全局 ``_INIT_LOCK`` 管所有库）。
   构造一个 brain 要 25–33 秒（加载全库 + 嵌入模型）。若全局加锁，库 A 正在
   构造时库 B 的**任何**调用都得陪着等 —— 多库的价值（互不干扰）就没了。
   折中：分派 dict 用一把短锁，**构造过程**只用该库自己的锁。
3. **换出前先落盘**：LRU 换出绝不静默丢弃未保存的改动（换出失败也要继续，
   不能卡死整个注册表）。

刻意做成 factory 可注入：测试塞假 brain，不必加载真模型。且本模块**纯标准库**
（不 import torch），daemon 与 server 都能安全引用。
"""
from __future__ import annotations

import os
import threading

# 构造注入缝：见 common/config_bridge。
from ..common.config_bridge import apply_config, resolve as _resolve
from typing import Any

#: 未指定库时的 key（与多库寻址会话 key 的 ``__default__`` 同一命名空间）
DEFAULT_KEY = "__default__"


class BrainRegistry:
    """库 id/路径 -> HippoBrain。**同一 key 只建一次**。"""

    def __init__(self, factory: Any = None, max_resident: int | None = None) -> None:
        self._factory = factory
        self._brains: dict[str, Any] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._lock = threading.RLock()      # 只保护 dict 本身，绝不覆盖构造过程
        self._lru: list[str] = []
        # 并发下"白建了一个"的次数。它必须是**可见**的：白建一次 = 白加载一次
        # 模型（~1GB / 30s），静默吞掉就再也查不出为什么切库这么慢。
        self.duplicate_builds = 0
        # 默认驻留 3：跨任务读常态化后，「家库 + 被他库 + 用户库」是常态
        # 组合，2 会把家库换出、下次写家库重付加载（25–33s）；共享嵌入器后
        # 同模型边际 ~13.6MB，代价可承受。
        # env 直读降级为「直连回退」（env 先、注入值回退；显式参数
        # 仍最优先）——生产值由 engine import 时经 config_bridge 灌入。
        self._max_resident = int(
            max_resident if max_resident is not None
            else _resolve("HIPPOCAMPUS_MAX_RESIDENT_LIBS", "max_resident_libs", "3") or 3
        )

    # -- key ----------------------------------------------------------
    def key_of(self, library: str | None) -> str:
        """库标识归一。

        留空 → ``__default__``；否则按**归一后的绝对路径**（实测 Windows 盘符大小写
        不一致，不归一会把同一个库建成两个 brain）。
        """
        lib = (library or "").strip()
        if not lib:
            return DEFAULT_KEY
        try:
            return os.path.normcase(os.path.abspath(lib))
        except Exception:  # noqa: BLE001 —— 非法路径就按原样当 key，不在此处抛
            return lib

    def _key_lock(self, key: str) -> threading.RLock:
        with self._lock:
            lk = self._locks.get(key)
            if lk is None:
                lk = threading.RLock()
                self._locks[key] = lk
            return lk

    # -- 基本操作 ------------------------------------------------------
    def get(self, library: str | None = None) -> Any:
        """取（必要时构造）某库的 brain。构造**只**在该库自己的锁内进行。"""
        key = self.key_of(library)
        with self._key_lock(key):
            with self._lock:
                if key in self._brains:
                    self._touch(key)
                    return self._brains[key]
            if self._factory is None:
                raise RuntimeError("BrainRegistry has no factory configured")
            brain = self._factory(library)
            with self._lock:
                if key in self._brains:
                    # 并发窗口内别人先建好了。必须**丢弃**这个刚建的：
                    # 留着就是两个 brain 指向同一份存储（写入互不可见）。
                    self.duplicate_builds += 1
                    self._touch(key)
                    return self._brains[key]
                self._brains[key] = brain
                self._touch(key)
            self._evict_if_needed()
            return brain

    def _touch(self, key: str) -> None:
        """调用方必须已持 ``self._lock``。"""
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)

    def _evict_if_needed(self) -> None:
        """超出驻留上限就换出**最久未用**的（换出前先落盘，绝不静默丢弃）。"""
        if self._max_resident <= 0:
            return
        with self._lock:
            while len(self._brains) > self._max_resident and self._lru:
                victim = self._lru[0]
                if victim not in self._brains:
                    self._lru.pop(0)
                    continue
                brain = self._brains.pop(victim)
                self._lru.pop(0)
                try:
                    if getattr(brain, "persist_dir", None):
                        brain.save()
                except Exception:  # noqa: BLE001 —— 换出失败也要继续，不能卡死
                    pass

    def drop(self, library: str | None = None) -> bool:
        """主动换出一个库（换出前先落盘）。返回是否真的换出了。"""
        key = self.key_of(library)
        with self._lock:
            brain = self._brains.pop(key, None)
            if key in self._lru:
                self._lru.remove(key)
        if brain is None:
            return False
        try:
            if getattr(brain, "persist_dir", None):
                brain.save()
        except Exception:  # noqa: BLE001
            pass
        return True

    # -- 自曝 ----------------------------------------------------------
    def keys(self) -> list[str]:
        with self._lock:
            return list(self._brains)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "resident": len(self._brains),
                "max_resident": self._max_resident,
                "keys": list(self._brains),
                "lru": list(self._lru),
                "duplicate_builds": self.duplicate_builds,
            }

    def save_all(self) -> dict[str, str]:
        """收尾：全部落盘。返回 {key: error}（空 dict = 全成功）。"""
        errors: dict[str, str] = {}
        with self._lock:
            for key, brain in list(self._brains.items()):
                try:
                    if getattr(brain, "persist_dir", None):
                        brain.save()
                except Exception as exc:  # noqa: BLE001
                    errors[key] = f"{type(exc).__name__}: {exc}"
        return errors
