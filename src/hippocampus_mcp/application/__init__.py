# -*- coding: utf-8 -*-
"""Application layer: use-case orchestration.

Five use-case services: Write + Bootstrap (14 tools), Recall (11 tools),
Organize (10 tools), Maintain 5 + LibraryAdmin 5 (the original Maintain 11
was split into two classes by **whether bound to a single brain**; see the
``library_admin_service`` module docstring). Three disciplines:

1. **Do not import brain**: ``brain/core.py`` is the big facade class; the
   runtime is constructed and injected by the composition root (server.py's
   ``_write_service`` / ``_bootstrap_service`` factories), and services hold
   it only as ``Any`` -- on the import graph this edge is always zero. When
   brain's method bodies are later lifted into the Services, this package's
   API is unchanged and outsiders notice nothing.
2. **No I/O**: ``open()`` / ``sqlite3`` / ``store._records`` do not appear
   in this package; storage access goes through the injected runtime's
   public API (ports discipline).
3. **Envelopes are self-produced**: the write-family return body's ``library``
   first key (library self-exposure) is assembled by the service; tools only
   ``json.dumps`` -- response assembly belongs to the use-case layer.

Dependency direction: application -> common / domain / store / map / engine
/ retrieval / lnn / library, **never -> brain**.
"""
from __future__ import annotations

from .base import BaseService
from .bootstrap_service import BootstrapService
from .library_admin_service import LibraryAdminService, LibraryPorts
from .maintain_service import MaintainService
from .organize_service import OrganizeService
from .recall_service import RecallService
from .write_service import WriteService

__all__ = ["BaseService", "BootstrapService", "LibraryAdminService",
           "LibraryPorts", "MaintainService", "OrganizeService",
           "RecallService", "WriteService"]
