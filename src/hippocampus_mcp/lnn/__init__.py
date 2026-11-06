# -*- coding: utf-8 -*-
"""LNN dynamic memory core (Liquid Neural Network Dynamic Memory Core).

Defaults to a **pure numpy** CfC implementation (see ``numpy_cfc`` /
``memory_core``); production processes no longer import torch; ``HIPPOCAMPUS_LNN_BACKEND=torch``
falls back to the ncps.torch implementation (lazy import).

``Projector`` depends on torch and is therefore lazily exported — a top-level ``from .projector import
Projector`` would drag torch into every process that does ``import hippocampus_mcp.lnn``,
which is exactly the implicit dependency chain this module exists to break.
"""

from .checkpoint import StateCheckpoint
from .memory_core import MemoryCore

__all__ = ["MemoryCore", "Projector", "StateCheckpoint"]


def __getattr__(name: str):
    """PEP 562 lazy export: Projector (and its torch dependency) is loaded only when actually used."""
    if name == "Projector":
        from .projector import Projector

        return Projector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
