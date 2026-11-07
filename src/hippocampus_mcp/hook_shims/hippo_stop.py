#!/usr/bin/env python3
"""Stop hook: remind the agent to close the hippocampus session.

ZCode has no SessionEnd event; the Stop event plus a closing-phrase matcher
stands in: when a reply looks like a wrap-up, block the stop ONCE and ask
the agent to call ``session_end`` first.
- Once per session (state file in the temp dir, 12h TTL).
- Any exception releases the stop (exit 0, no output) — a broken hook must
  never stall the session.

Deployed by ``python -m hippocampus_mcp.cli install``; reinstall = repair.
"""

import json
import os
import sys
import time
import tempfile

STATE_TTL = 12 * 3600  # after expiry the same session may be reminded again

REASON = (
    "[HippoMCP wrap-up check] This reply looks like a session wrap-up. "
    "Close the memory session before stopping:\n"
    "1. Call mcp__hippocampus__session_end with summary (one-line session "
    "summary) and open_topics (empty when none).\n"
    "2. Ingest any key conclusions/decisions not yet persisted first.\n"
    "3. If this session never called ANY hippocampus tool (not even "
    "session_start), ignore this reminder and stop.\n"
    "Afterwards stop normally; no long explanation to the user is needed."
)


def state_path(sid: str) -> str:
    safe = "".join(c for c in sid if c.isalnum() or c in "-_")[:64] or "unknown"
    return os.path.join(tempfile.gettempdir(),
                        f"hippo_stop_reminded_{safe}.json")


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    # ZCode already continued one round (stop_hook_active) — do not block
    # again (anti-loop).
    if data.get("stop_hook_active"):
        return

    sid = str(data.get("session_id") or "unknown")
    sp = state_path(sid)

    try:
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                marked = json.load(f).get("ts", 0)
            if time.time() - marked < STATE_TTL:
                return  # already reminded this session; release
    except Exception:
        pass

    try:
        with open(sp, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "sid": sid}, f)
    except Exception:
        pass  # unwritable state still blocks once; ZCode's 3-continuation
        # cap bounds the loop

    print(json.dumps({"decision": "block", "reason": REASON},
                     ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # any exception: release silently
    sys.exit(0)
