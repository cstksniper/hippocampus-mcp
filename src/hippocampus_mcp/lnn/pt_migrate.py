# -*- coding: utf-8 -*-
"""One-shot migration of a legacy ``lnn_state.pt`` → npz (run in a **separate subprocess**).

Why a subprocess: reading ``.pt`` requires torch; the main process (the MCP
server) aims for **zero resident torch**. This module is spawned only by
``StateCheckpoint._load_legacy_pt`` as
``python -m hippocampus_mcp.lnn.pt_migrate <src.pt> <dst.npz>``; it exits
right after reading, and torch never enters the main process.

The migration semantics are independent of the in-memory format: ``hx`` drifts
with the weights (historical libraries' weights are randomized on every
restart), so this migration merely **brings the state over as a safety net**;
if it cannot be read, reset safely — no memory data is affected.
"""

from __future__ import annotations

import faulthandler
import sys

# If the subprocess gets unexpectedly stuck in some parent-process (server)
# context, dump its own thread stacks to stderr after 30s (the parent's
# capture_output will receive them) — the scene self-exposes, no more
# black-box guessing.
faulthandler.dump_traceback_later(30, exit=False)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m hippocampus_mcp.lnn.pt_migrate <src.pt> <dst.npz>",
              file=sys.stderr)
        return 2
    src, dst = argv
    try:
        import torch

        data = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"pt_migrate: read failed ({src}): {exc}", file=sys.stderr)
        return 1

    from .checkpoint import payload_from_state

    state = dict(data or {})
    # Keys outside get_state (step_count / reference_time, etc.) are collected
    # into scalars_json by payload_from_state; hx/last_output/window become
    # arrays.
    try:
        payload = payload_from_state(state)
        import numpy as np

        np.savez(dst, **payload)
    except Exception as exc:
        print(f"pt_migrate: write failed ({dst}): {exc}", file=sys.stderr)
        return 1
    print(f"pt_migrate: {src} → {dst} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
