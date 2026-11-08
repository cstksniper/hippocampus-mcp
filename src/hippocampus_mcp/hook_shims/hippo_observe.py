#!/usr/bin/env python3
"""Stop hook (capture lane): L-raw observation slices, OFF by default.

Second Stop entry WITHOUT a matcher — fires on every response end. When
capture is enabled (installer constant ``CAPTURE_ENABLED``), it takes the
response text straight from stdin (verified 2026-11-08: ``responseText`` /
``sessionId`` / ``turnId`` ride stdin; no transcript parsing), applies the
blacklist, and appends an idempotent L-raw record under the bound library's
``observations/`` directory.

Never prints anything (zero tokens, zero noise) and NEVER writes to formal
memory — candidates become memories only via explicit ``ingest``.

Deployed by ``python -m hippocampus_mcp.cli install``; reinstall = repair.
"""

import json
import os
import sys

_SRC = r"@@SRC@@"
if _SRC:
    sys.path.insert(0, _SRC)

# Mirror constants — MUST match mcp.servers.hippocampus env (doctor checks).
HELLO_LIBRARY = r"@@HELLO_LIBRARY@@"

# Opt-in switch, substituted by the installer (``--stop-capture``).
# Changing it requires re-running install; default OFF by design.
CAPTURE_ENABLED = "@@STOP_CAPTURE@@" == "1"


def _resolve_cwd(data: dict) -> str:
    return (os.environ.get("ZCODE_PROJECT_DIR")
            or os.environ.get("CLAUDE_PROJECT_DIR")
            or str(data.get("cwd") or "")
            or os.getcwd())


def main() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if not isinstance(data, dict) or not CAPTURE_ENABLED:
        return
    text = str(data.get("responseText")
               or data.get("last_assistant_message") or "")
    if not text.strip():
        return

    from hippocampus_mcp.observe_capture import (append_raw, build_record,
                                                 reverse_lookup_library)
    lib_dir = reverse_lookup_library(_resolve_cwd(data), HELLO_LIBRARY)
    if not lib_dir:
        return  # unbound workspace: capture is a gift, never an error
    record = build_record(
        session_id=str(data.get("session_id") or data.get("sessionId") or ""),
        turn_id=str(data.get("turnId") or ""),
        text=text,
        cwd=_resolve_cwd(data),
        now=__import__("time").time())
    append_raw(lib_dir, record)
    # No output by design: this lane must stay invisible to the model.


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # fail-open: capture must never stall the session
    sys.exit(0)
