# -*- coding: utf-8 -*-
"""MCP stdio startup sequence (the implementation body of server.py's
``main()`` startup orchestration; server.py keeps only a thin shell).

Implemented here: all of main()'s startup orchestration -- file log
hookup, stdout guard-rail installation, import-chain preload, diagnostic
watchdog, the three warmup tiers (sync/off/auto), startup-time
self-report, ``server.run()``.

server.py keeps the ``def main(): _entry.run_main()`` shell (cli.py /
launcher.py's ``from .server import main`` needs zero changes).

🔴 Fetch-live discipline: the server instance and all startup
dependencies (``_PERSIST_DIR`` / ``logger`` / ``_preload_import_chain``
/ ``_WARMUP_DIAG`` / ``_WARMUP_MODE`` / state constants /
``start_background_warmup`` / ``_PROCESS_T0`` / ``_HANDSHAKE_WINDOW`` /
``server``) are all fetched live at call time via :func:`_srv` -- test
replacement of ``server.server`` (a fake FastMCP) or patching any state
variable means the startup sequence receives the stub.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from hippocampus_mcp.server_runtime.stdout_guard import _ThreadAwareStdout

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """The server module itself (fetch-live entry; see the module docstring)."""
    from hippocampus_mcp import server as _s

    return _s


def run_main() -> None:
    """Start the MCP stdio server (implementation body of server.py's ``main()`` thin shell)."""
    s = _srv()
    # Debug file logging (in stdio mode stderr can be truncated; a file log reproduces the crash scene in full)
    os.makedirs(s._PERSIST_DIR, exist_ok=True)
    _fh = logging.FileHandler(
        os.path.join(s._PERSIST_DIR, "mcp_server_debug.log"), encoding="utf-8"
    )
    _fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    s.logger.addHandler(_fh)
    for _name in ("mcpserver", "FastMCP", "mcp"):
        logging.getLogger(_name).addHandler(_fh)
        logging.getLogger(_name).setLevel(logging.DEBUG)

    # Note: torch is **no longer** imported here. torch is imported by
    # ``_prepare_torch()`` when first truly needed (see ``get_brain``);
    # otherwise its ~5.7s would sit directly on the MCP handshake path.
    # ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` accordingly sink into that
    # function (they must be set in the environment before
    # ``import torch``).

    # stdout guard rail: third-party output in background threads is
    # rerouted to stderr, avoiding JSON-RPC pollution
    if not isinstance(sys.stdout, _ThreadAwareStdout):
        sys.stdout = _ThreadAwareStdout(sys.stdout)

    # Import-chain preload: must happen before server.run(), otherwise it
    # hangs (see _preload_import_chain)
    s._preload_import_chain()

    # Diagnostic watchdog: independent of the warmup mode -- hangs can
    # happen in the main thread (off mode's lazy initialization) or in
    # the background thread (auto); watching only auto misses half the
    # scenes.
    if s._WARMUP_DIAG:
        s._start_warmup_watchdog()

    if s._WARMUP_MODE == "sync":
        # Old behavior: answer the handshake only after warmup. **Do not
        # give this to MCP hosts** -- host sessions usually finish tool
        # registration within 10-15 seconds, and tools during warmup
        # would be missed entirely.
        try:
            brain = s.get_brain()
            brain._get_nlp()
            brain.manager.embedder.embed("warmup")
            s.logger.info(
                "warmup done: L1=%d entities=%d",
                brain.manager.store.count(), len(brain.entities),
            )
        except Exception as e:  # warmup failure does not block the service (the first call retries)
            s.logger.warning("warmup failed (will retry on first call): %s", e)
    elif s._WARMUP_MODE == "off":
        s.logger.info("HIPPOCAMPUS_WARMUP=off: skipping warmup; the first tool call will initialize")
    else:
        s.start_background_warmup()

    # Startup-time self-report: reports the number only, does not block.
    # Hung and slow look identical on the host side; without this line
    # you can only guess -- guessing wrong costs the whole onboarding
    # discipline another round of spinning its wheels.
    _boot_s = time.time() - s._PROCESS_T0
    if _boot_s > s._HANDSHAKE_WINDOW:
        s.logger.warning(
            "startup took %.1fs, over the host tool-registration window (about %.0fs) -- "
            "this service's tools may have been missed at registration entirely; the durable fix is launcher + resident daemon.",
            _boot_s, s._HANDSHAKE_WINDOW)
    else:
        s.logger.info("startup took %.1fs (within the %.0fs window)",
                      _boot_s, s._HANDSHAKE_WINDOW)

    s.server.run()
