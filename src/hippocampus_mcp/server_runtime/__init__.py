# -*- coding: utf-8 -*-
"""server_runtime: the adapter-layer collaborator subpackage for server.py.

This package's modules are the wholesale split of server.py's non-tool blocks
(warmup / multi-library addressing / profile gating / dispatch / entry
sequence). Discipline:

* **State and the monkeypatch surface stay in server.py**; modules in this
  package always fetch-live via a lazy ``_srv()`` import — tests replacing or
  assigning ``server._X`` all keep working as before;
* server.py re-exports them as-is by import name; existing call sites and
  test references change zero lines;
* This package does not take part in the layer-direction static check (same
  treatment as the entry layer: its import edges are not direction-judged and
  create no new violation edges — the import surface is exactly identical to
  the current server.py: application / common / library / config).

Module list (parsing helpers are not in this package — they landed in
``common/params.py``, being low-level pure functions):

* ``warmup`` — warmup block (background preload session / exit sentinel);
* ``routing`` — multi-library addressing (library_use stickiness / home
  library resolution / concurrency protection);
* ``composition`` — composition-root factory (the assembly point for
  application services);
* ``profiles`` — profile gating (common/full tool projection and
  _TOOL_GROUPS);
* ``dispatch`` — tool dispatch (the scheduling shell outside the tool bodies);
* ``stdout_guard`` — stdout guard (_ThreadAwareStdout, keeps MCP frames from
  being polluted);
* ``entry`` — entry sequence (daemon/direct-connect startup orchestration).

Import-surface discipline (a supplement to the three points above):
``dispatch``/``stdout_guard`` use only the standard library; ``profiles``
depends only on config; ``routing``/``warmup`` depend on library+common;
``composition`` depends on application; ``entry`` depends only on this
package — **no new out-of-package dependency edges**: whatever import surface
the moved-in code had, this package keeps.
"""
