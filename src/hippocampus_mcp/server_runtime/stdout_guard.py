# -*- coding: utf-8 -*-
"""Header infrastructure for server.py (sys.path fallback / logging setup / stdout guard).

Implemented here: the ``sys.path`` fallback (append the libs dir + insert the src dir),
logging setup (stderr stream handler + level), and the stdio-transport stdout guard
class ``_ThreadAwareStdout``.

server.py re-exports ``_ThreadAwareStdout`` under the same import name
(tests/test_warmup.py instantiates it directly); ``logger`` is returned by
:func:`configure_logging`.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any


def ensure_paths(server_file: str) -> None:
    """``sys.path`` fallback (caller passes server.py's ``__file__``).

    The libs dir is fallback-only (appending does not shadow properly installed packages;
    what is inside libs are old packages compiled for py3.10, and putting them first would
    shadow site-packages, causing "cannot import sentence_transformers").
    """
    _LIBS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(server_file)))),
        "libs",
    )
    if _LIBS_DIR not in sys.path:
        sys.path.append(_LIBS_DIR)

    # Make sure the src dir is on the path
    _SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(server_file)))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)


def configure_logging() -> logging.Logger:
    """Logging setup: stderr stream handler + configured level."""
    logger = logging.getLogger("hippocampus_mcp")
    if not logger.handlers:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(_handler)
        from hippocampus_mcp.config import load_config

        _log_level = load_config().log_level.upper()
        logger.setLevel(getattr(logging, _log_level, logging.INFO))
    return logger


class _ThreadAwareStdout:
    """stdout guard for the stdio transport.

    In MCP stdio mode fd 1 may **only** carry JSON-RPC. But third-party libraries
    (transformers / tqdm / spaCy) may print to stdout on first load -- previously warmup
    finished before the handshake, so at worst it was ugly; now warmup runs on a
    background thread, concurrently with the client's tools/list, and once anything slips
    into the protocol stream the host fails to parse (symptom: "connected, but no tools").

    Here writes to stdout from **non-main threads** are rerouted to stderr; the main
    thread (= the legitimate protocol writer) is unaffected. Note that
    ``contextlib.redirect_stdout`` cannot be used: it swaps the global ``sys.stdout``,
    which would drag the main thread's protocol output to stderr as well.
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    @staticmethod
    def _sink(real: Any) -> Any:
        # Main thread -> real stdout (protocol stream); other threads -> stderr.
        # Must not return sys.stdout: once installed it points at this proxy, which would recurse forever.
        return real if threading.current_thread() is threading.main_thread() else sys.stderr

    def write(self, s: str) -> int:
        return self._sink(self._real).write(s)

    def flush(self) -> None:
        self._sink(self._real).flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)
