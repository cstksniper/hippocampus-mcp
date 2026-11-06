# -*- coding: utf-8 -*-
""" hippocampus_mcp -- the memory substrate for agents (LNN-enhanced).

v0.3: four memory layers (L0 raw / L1 facts / L2 summaries / L3 profile) + entity index layer + usage layer.
"""

# Exemption registry: third-party library thread env vars (must be set
# before numpy import); the list lives in config.py.

import os as _os

# Memory bill:
# scipy-openblas bundled with numpy reserves per-thread buffers by core
# count by default -- measured on a 32-core machine: the import alone
# commits 754.7MB of private virtual memory (untouched pages don't count
# toward RSS, but they eat the commit limit; multi-library residency /
# daemon / tight pagefile scenarios suffer). Capped at 4 threads, commit
# drops to 111.8MB, and small ops (9841×768 gemv) actually get faster than
# all-core (1.74ms vs 1.95ms).
# setdefault: explicitly set env wins; this must take effect before any
# numpy import, hence in the package __init__ (python -m
# hippocampus_mcp.server always passes through here).
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily export ``HippoBrain`` (PEP 562).

    This used to be a top-level ``from .brain import HippoBrain``; the
    consequence was that **any** ``import hippocampus_mcp.xxx`` would also
    load brain → manager → torch (about 5.7s warm on this machine). MCP's
    ``initialize`` sits on this path, and the extra time eats directly into
    the host's tool-registration window, so it was changed to on-demand
    resolution.

    ``from hippocampus_mcp import HippoBrain`` and attribute access both
    still work.
    """
    if name == "HippoBrain":
        from .brain import HippoBrain

        return HippoBrain
    if name == "TagExtractor":
        # The tag extractor is lazily exported the same way: it only depends
        # on the stdlib, but exporting it here gives one unified exit, so
        # hosts/scripts need not remember it lives in the tagging module.
        from .domain.tagging import TagExtractor

        return TagExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
