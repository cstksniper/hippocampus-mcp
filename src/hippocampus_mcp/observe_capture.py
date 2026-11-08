# -*- coding: utf-8 -*-
"""Observation-layer capture: L-raw slices + candidate extraction (WS-1).

Implements the capture half of ``docs/design/Stop观察分层_设计书_20261107.md``
as amended by the 2026-11-08 harness finding: ZCode's Stop hook stdin carries
``responseText`` / ``last_assistant_message`` directly (plus ``sessionId`` /
``turnId`` / ``transcript_path``), so capture reads STDIN — no transcript
file parsing.

Three hard rules from the design (each enforced here or by construction):

1. **Capture is OFF unless explicitly enabled** — the Stop shim calls into
   this module only behind an installer-substituted constant; nothing fires
   by default.
2. **Blacklist before storage** — injected memory blocks, host system
   reminders, tool-JSON shapes and injection-rendering artifacts are dropped
   line-by-line BEFORE anything is written (§2 of the design: 宁可少记不错记).
   Sensitive-word filtering is enforced at PROMOTION time by the existing
   ingest pipeline (quarantine/sensitive tags) — this module deliberately
   does not duplicate it.
3. **Candidates never touch formal memory** — this module only appends
   JSONL under ``<library>/observations/``; it has no imports from the write
   path and cannot supersede anything.

Idempotency: each record carries ``key = sha1(session|turn|text)``; appends
skip when the key already exists in the target day file (Stop can legally
refire up to 3 times per turn).

Stdlib only — the Stop shim imports this under ``-I``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

__all__ = ["extract_candidates", "build_record", "append_raw",
           "reverse_lookup_library", "observations_dir"]

# --------------------------------------------------------------------------
# Blacklist (line-level, structural). Order matters only for readability.
_DENY_PATTERNS = tuple(re.compile(p) for p in (
    # memory-injection rendering (both hook headers + section glyphs)
    r"HippoMCP (auto-injected|session bootstrap)",
    r"^\s*◆",
    r"^\s*【[^】]*】\s*$",
    # host/system noise
    r"<system-reminder>",
    r"\[SYSTEM NOTIFICATION",
    r"^\s*\{?\s*\"(memory_id|hookSpecificOutput|tool_use|tool_result)\"",
    # injection-rendering footers/artifacts
    r"Slot budget:",
    r"Truncated to budget\)",
    r"suppressed: already injected",
    r"retrieval misses in the last 24h",
    r"item\(s\) clipped to",
    # empty-slot placeholders (EN + stale-daemon CN variants)
    r"^\((No todos|No session history|No related entity state|No open topics)\)",
    r"^（无待办|无会话历史|无相关实体状态|无未完结话题）",
))

_MIN_LINE_CHARS = 8  # drop greetings/fragments; conclusions survive


def extract_candidates(text: str) -> list[str]:
    """Response text -> surviving candidate lines (blacklist applied).

    Line-level filtering keeps provenance simple (each candidate is one
    original line) and is intentionally conservative: anything that LOOKS
    like injection/tool/system noise is dropped even if that occasionally
    loses a real sentence — the design's error budget favors silence.
    """
    out: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if len(line) < _MIN_LINE_CHARS:
            continue
        if any(p.search(line) for p in _DENY_PATTERNS):
            continue
        out.append(line)
    return out


def build_record(session_id: str, turn_id: str, text: str, cwd: str,
                 now: float) -> dict[str, Any]:
    """One L-raw record (idempotent by construction)."""
    lines = extract_candidates(text)
    digest = hashlib.sha1(
        f"{session_id}|{turn_id}|{text}".encode("utf-8", "replace")
    ).hexdigest()[:16]
    return {
        "ts": float(now),
        "key": digest,
        "session": str(session_id or ""),
        "turn": str(turn_id or ""),
        "cwd": str(cwd or ""),
        "lines": lines,
    }


def observations_dir(lib_dir: str) -> str:
    return os.path.join(lib_dir, "observations")


def _day_file(lib_dir: str, now: float) -> str:
    import time as _time
    day = _time.strftime("%Y%m%d", _time.localtime(now))
    return os.path.join(observations_dir(lib_dir), f"raw-{day}.jsonl")


def _existing_keys(path: str) -> set[str]:
    keys: set[str] = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    k = json.loads(line).get("key")
                except json.JSONDecodeError:
                    continue
                if k:
                    keys.add(str(k))
    except OSError:
        pass
    return keys


def append_raw(lib_dir: str, record: dict[str, Any]) -> tuple[str, bool]:
    """Append one record to the day file. Returns ``(path, appended)``;
    ``appended=False`` when the idempotency key already exists."""
    path = _day_file(lib_dir, float(record.get("ts") or 0.0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if str(record.get("key")) in _existing_keys(path):
        return path, False
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path, True


# --------------------------------------------------------------------------
# Workspace -> library reverse lookup (stdlib twin of the daemon's C1 rule:
# deepest workspace_root prefix wins). Kept independent from cli.py because
# the Stop shim imports this module under -I; parity is pinned by tests.
# --------------------------------------------------------------------------

def reverse_lookup_library(cwd: str, user_lib: str) -> str:
    """Deepest ``workspace_root`` containing ``cwd`` -> library path; '' when
    unbound/sidecar missing (the caller then stays silent — capture is a
    gift, never an error)."""
    from .library import normpath
    from .library.task_directory import load_directory  # stdlib-only module

    try:
        data, issue = load_directory(user_lib)
    except Exception:
        return ""
    if issue:
        return ""
    want = os.path.normcase(os.path.abspath(os.path.expanduser(cwd or "")))
    best = ""
    best_len = -1
    for e in data.get("entries") or []:
        wr = str(e.get("workspace_root") or "").strip()
        lib = str(e.get("path") or "").strip()
        if not wr or not lib:
            continue
        wrn = os.path.normcase(os.path.abspath(os.path.expanduser(wr)))
        try:
            if want != wrn and not want.startswith(wrn + os.sep):
                continue
        except (ValueError, OSError):
            continue
        if len(wrn) > best_len:
            best, best_len = lib, len(wrn)
    return normpath(best) if best and os.path.isdir(best) else ""
