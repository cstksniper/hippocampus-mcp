# -*- coding: utf-8 -*-
"""In-process multi-library registry (library path -> HippoBrain).

**Shared verbatim by daemon and server**: this implementation was first written
for "one daemon holding multiple libraries", then direct stdio also needed
multi-library, so it sank into a common module — the ``daemon.BrainRegistry``
name is kept unchanged (import compatibility).

Three things that had to be built this way:

1. **Build each key only once** — this is where the "model N→1" lands. Two
   brains pointing at the same ``memories.json`` is explicitly forbidden since
   v0.4 (writes invisible to each other + wholesale overwrite loses data).
2. **A separate lock per library** (no single global ``_INIT_LOCK`` governing
   all libraries). Constructing one brain takes 25–33 seconds (load the whole
   library + the embedding model). Under a global lock, **any** call to library
   B would have to wait while library A is being constructed — the whole value
   of multi-library (non-interference) would be gone. Compromise: the dispatch
   dict uses one short lock; **construction** uses only that library's own lock.
3. **Persist before eviction**: LRU eviction never silently discards unsaved
   changes (eviction failure must not stall the whole registry either).

Deliberately built with an injectable factory: tests can plug in fake brains
without loading real models. And this module is **pure standard library** (no
torch import), so both daemon and server can reference it safely.
"""
from __future__ import annotations

import os
import threading

# Construction injection seam: see common/config_bridge.
from ..common.config_bridge import apply_config, resolve as _resolve
from typing import Any

#: Key used when no library is specified (same namespace as the multi-library addressing session key's ``__default__``)
DEFAULT_KEY = "__default__"


class BrainRegistry:
    """Library id/path -> HippoBrain. **Each key is built only once**."""

    def __init__(self, factory: Any = None, max_resident: int | None = None) -> None:
        self._factory = factory
        self._brains: dict[str, Any] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._lock = threading.RLock()      # only protects the dict itself, never spans a construction
        self._lru: list[str] = []
        # Count of "built one for nothing" under concurrency. It must be
        # **visible**: one wasted build = one wasted model load (~1GB / 30s);
        # swallowing it silently means nobody can ever find out why library
        # switching is so slow.
        self.duplicate_builds = 0
        # Default resident cap of 3: once cross-task reads became routine,
        # "home library + a foreign library + user library" is the common
        # combination; 2 would evict the home library and the next write to it
        # would pay the load again (25–33s); with the shared embedder the
        # marginal cost of the same model is ~13.6MB, an acceptable price.
        # Direct env reads are demoted to "direct-connection fallback" (env
        # first, injected value as fallback; explicit argument still wins) —
        # the production value is funneled in via config_bridge at engine
        # import time.
        self._max_resident = int(
            max_resident if max_resident is not None
            else _resolve("HIPPOCAMPUS_MAX_RESIDENT_LIBS", "max_resident_libs", "3") or 3
        )

    # -- key ----------------------------------------------------------
    def key_of(self, library: str | None) -> str:
        """Normalize a library identifier.

        Empty → ``__default__``; otherwise the **normalized absolute path**
        (measured: Windows drive-letter case is inconsistent; without
        normalization the same library could be built as two brains).
        """
        lib = (library or "").strip()
        if not lib:
            return DEFAULT_KEY
        try:
            return os.path.normcase(os.path.abspath(lib))
        except Exception:  # noqa: BLE001 — an invalid path just becomes the key as-is; do not raise here
            return lib

    def _key_lock(self, key: str) -> threading.RLock:
        with self._lock:
            lk = self._locks.get(key)
            if lk is None:
                lk = threading.RLock()
                self._locks[key] = lk
            return lk

    # -- Basic operations ------------------------------------------------------
    def get(self, library: str | None = None) -> Any:
        """Fetch (constructing if needed) a library's brain. Construction happens **only** inside that library's own lock."""
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
                    # Someone else finished building within the concurrency
                    # window. The freshly built one must be **discarded**:
                    # keeping it means two brains pointing at the same storage
                    # (writes invisible to each other).
                    self.duplicate_builds += 1
                    self._touch(key)
                    return self._brains[key]
                self._brains[key] = brain
                self._touch(key)
            self._evict_if_needed()
            return brain

    def _touch(self, key: str) -> None:
        """The caller must already hold ``self._lock``."""
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)

    def _evict_if_needed(self) -> None:
        """When over the resident cap, evict the **least recently used** (persist before eviction; never silently discard)."""
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
                except Exception:  # noqa: BLE001 — eviction failure must not stall; keep going
                    pass

    def drop(self, library: str | None = None) -> bool:
        """Actively evict one library (persisting before eviction). Returns whether an eviction actually happened."""
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

    # -- Self-report ----------------------------------------------------------
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
        """Wrap-up: persist everything. Returns {key: error} (empty dict = all succeeded)."""
        errors: dict[str, str] = {}
        with self._lock:
            for key, brain in list(self._brains.items()):
                try:
                    if getattr(brain, "persist_dir", None):
                        brain.save()
                except Exception as exc:  # noqa: BLE001
                    errors[key] = f"{type(exc).__name__}: {exc}"
        return errors
