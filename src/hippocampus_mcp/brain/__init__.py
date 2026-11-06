# -*- coding: utf-8 -*-
"""brain subpackage: the memory orchestration layer.

``from hippocampus_mcp.brain import HippoBrain`` is the established contract for
out-of-package scripts and tests (60+ files use it), so this module must keep
**namespace equivalence with the pre-split layout**:

* the ``import *`` from ``..common.constants`` goes through its own ``__all__``, which includes
  underscore names like ``_now`` / ``_nid`` / ``_env_int`` — ``import *`` by default
  does not export underscore names; only ``__all__`` brings them through, and missing one
  is a NameError that explodes only on certain code paths;
* the ``import *`` from ``.core`` completes ``HippoBrain`` and the remaining public symbols —
  matching the export behavior of the original ``brain.py`` when it defined no ``__all__``.

Both ``import *`` lines carry noqa: this is a deliberate re-export, not a redundant import.

⚠️ **The real tunable constants live in ``brain.core``**: ``import *`` merely copies the names
into the package namespace as a snapshot. Writing ``hippocampus_mcp.brain.UPDATE_TAGS_CONFIRM_ABOVE = 2`` only changes
the package copy; functions in ``core`` read ``core``'s own globals — a module-level global patch
only takes effect in **the module that both defines and uses it**. To tune a parameter, change ``hippocampus_mcp.brain.core.XXX``,
and read from ``.core`` too. The pre-existing patches in two tests have been fixed accordingly
(j1_update_tags / s2_tag_ops_undo); this is the only semantic boundary introduced by the packaging.

Module responsibilities:
* ``core``    — the ``HippoBrain`` main class (the original ``brain.py``, ~6300 lines)
* ``pure``    — the stateless mixin (the original ``brain_pure.py``)
* ``entity``  — the entity lexicon mixin (the original ``brain_entity.py``)
* ``session`` — the session-resume mixin (the original ``brain_session.py``)
"""

from ..common.constants import *  # noqa: F401,F403
from .core import *  # noqa: F401,F403
