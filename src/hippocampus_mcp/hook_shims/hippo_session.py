#!/usr/bin/env python3
"""SessionStart hook: read-only opening anchor from the memory library.

Injects pins / open todos / recent session summaries / unfinished topics at
session start, resume, clear and AFTER COMPACTION — the moments where
discipline is weakest and the need is strongest. Deliberately does NOT call
``session_start`` (that would create a session and supersede the previous
one); the injection is a read-only anchor, the full package stays one
explicit ``session_start`` away.

Deployed by ``python -m hippocampus_mcp.cli install`` from
``src/hippocampus_mcp/hook_shims/`` — @@TOKEN@@ placeholders substituted at
deploy time. Reinstall = repair.

Discipline identical to the other shims: any exception exits 0 with no
output; empty result prints nothing.
"""

import json
import os
import sys

# -I isolation ignores PYTHONPATH; add the package source manually
# (substituted by the installer; empty when already importable).
_SRC = r"@@SRC@@"
if _SRC:
    sys.path.insert(0, _SRC)

# Mirror constants — MUST match mcp.servers.hippocampus env in
# ~/.zcode/cli/config.json (installer keeps them in sync; doctor checks).
KEY_LIBRARY = r"@@KEY_LIBRARY@@"
EMB_MODEL = "@@EMB_MODEL@@"
HELLO_LIBRARY = r"@@HELLO_LIBRARY@@"

BOOTSTRAP_TOP_K = 4
BOOTSTRAP_BUDGET = 1600
BOOTSTRAP_ITEM_CHARS = 200

_COMPACT_REMINDER = ("(Context was compacted — to rebuild the full opening "
                     "package (profile / task recall / maps), call "
                     "mcp__hippocampus__session_start.)")


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
    if not isinstance(data, dict):
        return

    from hippocampus_mcp.hook_client import fetch_bootstrap
    text = fetch_bootstrap(
        _resolve_cwd(data),
        key_library=KEY_LIBRARY, emb_model=EMB_MODEL,
        hello_library=HELLO_LIBRARY,
        top_k=BOOTSTRAP_TOP_K, token_budget=BOOTSTRAP_BUDGET,
        item_max_chars=BOOTSTRAP_ITEM_CHARS)
    if not text:
        return
    # Compaction is the moment the receipt protocol matters most; the
    # static reminder line costs one line and points at the escape hatch.
    if str(data.get("source") or "") == "compact":
        text += "\n" + _COMPACT_REMINDER
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": text,
        },
        "suppressOutput": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # fail-open
    sys.exit(0)
