"""prewarm -- moving the daemon's creator from the "host process tree" to
the "user session level".

## Why this module exists

Measured conclusions:

- the daemon's creator is the **launcher spawned by the host** ⇒ the daemon
  lands in the host's **process tree**
- the host reclaims the session after roughly 5 minutes idle, terminating
  the daemon with it
  -- the log shows **0 exit records and 0 crash traces**, looking like a
  "silent disappearance"
- ``CREATE_BREAKAWAY_FROM_JOB`` cannot cure it: already falsified by a
  behavioral experiment, the daemon was never inside the host's Job
  (``0x2400 = JOB_MEMORY|SILENT_BREAKAWAY_OK``); that was a pseudo
  requirement

**The root fix is changing the creator**. Let the daemon be started by
"login startup / a scheduled task"; its parent is then explorer or the
Schedule service, out of the host's reach when it reclaims the session.

The launcher side needs **zero changes**: it already checks the registry
first and reuses a live daemon when found (path ① of ``ensure_daemon``).
So prewarming is just "filling the registry ahead of time".

## Usage

::

    # Prewarm (idempotent: skips when a usable daemon already exists)
    python -m hippocampus_mcp.prewarm --cwd D:\\work\\proj

    # With explicit library / model (must match the env vars the host gives
    # the launcher)
    python -m hippocampus_mcp.prewarm --cwd D:\\work\\proj \\
        --library D:\\data\\hippo_proj --model BAAI/bge-base-zh-v1.5

    # Query only, no start
    python -m hippocampus_mcp.prewarm --status --cwd D:\\work\\proj

    # Machine-readable (for scheduled tasks / wrapper scripts)
    python -m hippocampus_mcp.prewarm --cwd ... --json

Exit codes: 0 = already there or started fine; 1 = could not start; 2 =
argument error.

## 🔴 Correctness preconditions (getting them wrong does not error; it just
starts a useless one)

1. **The key must match**: ``key = daemon_key(library, model, version)``
   must be the same one the launcher computes from its env vars. If any of
   the three differs ⇒ prewarming starts **another** daemon; the launcher
   in the host session will start its own real one, and the prewarmed one
   becomes an orphan nobody uses.
2. **The cwd must be right**: when ``library`` is empty, the daemon locates
   the library by **its own cwd**. A wrong cwd starts a daemon pointing
   somewhere else -- it starts fine and registers fine, just against the
   wrong library.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .launcher import (
    _registry_usable,
    daemon_key,
    read_registry,
    runtime_dir,
    spawn_daemon,
)

__all__ = ["prewarm", "state", "main"]


def state(key: str) -> dict[str, Any]:
    """Whether a usable daemon currently exists for this key. Read-only,
    starts nothing."""
    info = read_registry(key)
    return {
        "key": key,
        "running": bool(_registry_usable(info)),
        "info": info,
        "registry": os.path.join(runtime_dir(), f"daemon-{key}.json"),
    }


def prewarm(cwd: str = "", library: str = "", model: str = "",
            version: str = "") -> dict[str, Any]:
    """Start the resident daemon (if none usable yet). **Idempotent**.

    Returns a dict with at least ``ok`` / ``action`` / ``key``.
    ``action`` ∈ ``skip`` (already there, reused) | ``spawned`` (started
    this time) | ``failed``.
    """
    key = daemon_key(library, model, version)
    before = read_registry(key)
    if _registry_usable(before):
        return {
            "ok": True,
            "action": "skip",
            "key": key,
            "pid": before.get("pid"),      # type: ignore[union-attr]
            "port": before.get("port"),    # type: ignore[union-attr]
            "detail": "an existing usable daemon was found; reusing it (no new process spawned)",
        }

    info = spawn_daemon(key, library=library, model=model, cwd=cwd)
    if not info:
        return {
            "ok": False,
            "action": "failed",
            "key": key,
            "cwd": cwd or os.getcwd(),
            "detail": "spawn_daemon returned None; see "
                      f"{os.path.join(runtime_dir(), f'daemon-{key}.log')}",
        }
    return {
        "ok": True,
        "action": "spawned",
        "key": key,
        "pid": info.get("pid"),
        "port": info.get("port"),
        "cwd": cwd or os.getcwd(),
        "detail": "spawned this time (creator = this process; it should have left the host process tree)",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hippocampus_mcp.prewarm",
        description="Prewarm the resident daemon so its creator is not inside the host process tree.")
    p.add_argument("--cwd", default="",
                   help="the daemon's working directory. When library is empty it decides library location; "
                        "give it explicitly for prewarm; defaults to the current directory")
    p.add_argument("--library", default="",
                   help="library path (maps to HIPPOCAMPUS_DAEMON_LIBRARY)")
    p.add_argument("--model", default="",
                   help="embedding model (maps to HIPPOCAMPUS_EMB_MODEL)")
    p.add_argument("--version", default="",
                   help="code version (maps to HIPPOCAMPUS_DAEMON_VERSION)")
    p.add_argument("--status", action="store_true",
                   help="query only, do not start")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="output JSON (for scheduled tasks / wrapper scripts to parse)")
    args = p.parse_args(argv)

    key = daemon_key(args.library, args.model, args.version)
    run_cwd = args.cwd or os.getcwd()

    if not os.path.isdir(run_cwd):
        print(f"[prewarm] x --cwd is not a directory: {run_cwd}", file=sys.stderr)
        return 2

    if args.status:
        res: dict[str, Any] = state(key)
        if args.as_json:
            print(json.dumps(res, ensure_ascii=False, indent=1))
        else:
            print(f"key={key}  running={res['running']}")
            print(f"registry={res['registry']}")
            if res["info"]:
                print(f"  pid={res['info'].get('pid')} "
                      f"port={res['info'].get('port')}")
        return 0

    res = prewarm(cwd=args.cwd, library=args.library,
                  model=args.model, version=args.version)
    if args.as_json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        mark = "✓" if res["ok"] else "✗"
        print(f"[prewarm] {mark} action={res['action']} key={key}")
        print(f"          {res['detail']}")
        if res["ok"]:
            print(f"          pid={res.get('pid')} port={res.get('port')}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
