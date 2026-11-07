#!/usr/bin/env python3
"""UserPromptSubmit hook: per-turn memory context injection (mechanism, not
discipline).

Deployed by ``python -m hippocampus_mcp.cli install`` from
``src/hippocampus_mcp/hook_shims/`` — the @@TOKEN@@ placeholders below are
substituted at deploy time. Reinstall after a host upgrade REPAIRS the
deployment; do not hand-edit the deployed copy.

Discipline identical to hippo_stop.py: any exception exits 0 with no
output — a broken hook must never stall the session. Empty result prints
nothing (zero tokens, zero noise).
"""

import json
import os
import sys

# -I isolation ignores PYTHONPATH; add the package source manually.
# @@SRC@@ is substituted by the installer (empty string when the package is
# already importable, e.g. pip-installed).
_SRC = r"@@SRC@@"
if _SRC:
    sys.path.insert(0, _SRC)

# Mirror constants: these MUST match the mcp.servers.hippocampus env in
# ~/.zcode/cli/config.json (installer keeps them in sync; doctor's drift
# check compares them). Changing the config by hand requires changing the
# deployed shim too — or just re-run the installer.
KEY_LIBRARY = r"@@KEY_LIBRARY@@"
EMB_MODEL = "@@EMB_MODEL@@"
HELLO_LIBRARY = r"@@HELLO_LIBRARY@@"

# Injection shape knobs (hook-side budget; the server assembles the blocks).
# The installer can raise these when the adaptive-recall knob is on.
HOOK_TOP_K = 3
HOOK_BUDGET = 600
HOOK_ITEM_CHARS = 150


def _resolve_cwd(data: dict) -> str:
    """Workspace root: the hook env var is most trustworthy; fall back to
    stdin cwd, then the process cwd."""
    return (os.environ.get("ZCODE_PROJECT_DIR")
            or os.environ.get("CLAUDE_PROJECT_DIR")
            or str(data.get("cwd") or "")
            or os.getcwd())


def main() -> None:
    # -I ignores every PYTHON* env (including global PYTHONUTF8); child
    # processes fall back to GBK and UTF-8 Chinese stdin decoding fails.
    # Pin UTF-8 BEFORE reading stdin (runtime API, unaffected by -I).
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    prompt = str(data.get("prompt") or "")
    if not prompt.strip():
        return

    from hippocampus_mcp.hook_client import fetch_context
    text = fetch_context(
        prompt, _resolve_cwd(data),
        key_library=KEY_LIBRARY, emb_model=EMB_MODEL,
        hello_library=HELLO_LIBRARY,
        top_k=HOOK_TOP_K, token_budget=HOOK_BUDGET,
        item_max_chars=HOOK_ITEM_CHARS,
        session_id=str(data.get("session_id") or ""),
        dedup=True)
    if not text:
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        },
        "suppressOutput": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # fail-open: any exception silently lets the session proceed
    sys.exit(0)
