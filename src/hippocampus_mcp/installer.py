# -*- coding: utf-8 -*-
"""Idempotent ZCode deployment installer + hook-chain doctor (P0, 2026-11-07).

Design doc: ``docs/design/Agent使用自然化_设计提案_20261107.md`` §四 P0 (as
revised by the 2026-11-07 review: "repairable install + observable health",
NOT magical self-healing). Core promise: **reinstall = repair** — the
deployed shims/config entry are products of this module, never hand-placed
one-offs, and re-running it fixes drift.

What it manages (and nothing else):

1. ``~/.zcode/cli/config.json`` — the ``mcp.servers.hippocampus`` entry
   (created when missing; an EXISTING entry is left untouched unless
   ``--force``, so user tuning is never silently overwritten) and the
   ``hooks`` block (merged event-by-event; other events/servers preserved).
   A timestamped ``.bak-before-install-<ts>`` is written before any change.
2. ``~/.zcode/hooks/{hippo_context,hippo_stop}.py`` — copied from
   ``hook_shims/`` with @@TOKEN@@ substitution; existing files are backed
   up as ``.bak-<ts>`` when their content differs (overwrite = repair).
3. Optionally the ``hippo-memory`` user skill (copy directory).

Dry-run: :func:`plan_install` returns the action list; ``cli install
--dry-run`` prints it without touching anything.

Doctor support: :func:`hook_chain_status` returns the L1/L2 layer states
plus the shim-mirror-constant drift check (the 2026-11-05 lesson: shim
constants silently drifting from the config env).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .launcher import daemon_key  # key parity with the runtime (single truth)

__all__ = ["plan_install", "apply_install", "hook_chain_status",
           "HOOK_SHIM_NAMES", "HOOKS_EVENTS"]

HOOK_SHIM_NAMES = ("hippo_context.py", "hippo_stop.py", "hippo_session.py")
HOOKS_EVENTS = ("UserPromptSubmit", "Stop", "SessionStart")
_SKILL_NAME = "hippo-memory"

_ZCODE_DIR = ".zcode"
_USER_SKILL_DIR = ".agents/skills"


def _config_path(home: Path) -> Path:
    return home / _ZCODE_DIR / "cli" / "config.json"


def _hooks_dir(home: Path) -> Path:
    return home / _ZCODE_DIR / "hooks"


def _shims_source_dir() -> Path:
    return Path(__file__).resolve().parent / "hook_shims"


def _package_src_dir() -> str:
    """Repo src dir when the package is NOT importable standalone (the
    deployed shim needs a PYTHONPATH anchor under ``-I``); empty string
    when the package lives in site-packages (nothing to anchor)."""
    pkg_root = Path(__file__).resolve().parent.parent  # .../src/hippocampus_mcp
    if pkg_root.parent.name == "src":
        return str(pkg_root.parent)
    return ""


def _mcp_entry(home: Path) -> dict[str, Any]:
    """The user-level entry template (mirrors the 2026-10-30 migration
    shape: launcher + common profile + user library binding)."""
    user_lib = str(Path(home) / ".hippocampus_mcp" / "user")
    env: dict[str, str] = {
        "HIPPOCAMPUS_PERSIST_DIR": user_lib,
        "HIPPOCAMPUS_EMB_MODEL": "BAAI/bge-base-zh-v1.5",
        "HIPPOCAMPUS_TOOL_PROFILE": "common",
        "HIPPOCAMPUS_LAUNCHER": "1",
        "HF_HUB_OFFLINE": "1",
    }
    src = _package_src_dir()
    if src:
        env["PYTHONPATH"] = src
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "hippocampus_mcp.launcher"],
        "timeoutMs": 120000,
        "env": env,
    }


def _substitute(template: str, *, entry: dict[str, Any]) -> str:
    env = (entry.get("env") or {})
    return (template
            .replace("@@SRC@@", _package_src_dir())
            .replace("@@KEY_LIBRARY@@", env.get("HIPPOCAMPUS_DAEMON_LIBRARY", ""))
            .replace("@@EMB_MODEL@@", env.get("HIPPOCAMPUS_EMB_MODEL", ""))
            .replace("@@HELLO_LIBRARY@@", env.get("HIPPOCAMPUS_PERSIST_DIR", "")))


def _hooks_block(home: Path, entry: dict[str, Any]) -> dict[str, Any]:
    hooks_dir = _hooks_dir(home).as_posix()
    py = entry.get("command") or "python"
    return {"enabled": True, "events": {
        "UserPromptSubmit": [{"hooks": [{
            "type": "process", "command": py,
            "args": ["-I", f"{hooks_dir}/hippo_context.py"],
            "timeoutMs": 8000,
            "statusMessage": "hippocampus context injection"}]}],
        "Stop": [{"matcher": ("先这样|今天到这|就到这|到这里|到这儿|收工|到此为止"
                              "|下次继续|晚安|回头见"),
                  "hooks": [{
                      "type": "process", "command": py,
                      "args": ["-I", f"{hooks_dir}/hippo_stop.py"],
                      "timeoutMs": 10000,
                      "statusMessage": "hippocampus wrap-up check"}]}],
        # No matcher: fires for every source (startup/resume/clear/compact);
        # the shim branches on stdin `source` for the compact reminder.
        "SessionStart": [{"hooks": [{
            "type": "process", "command": py,
            "args": ["-I", f"{hooks_dir}/hippo_session.py"],
            "timeoutMs": 8000,
            "statusMessage": "hippocampus session bootstrap"}]}],
    }}


def plan_install(home: Path, *, skill: bool = True,
                 force_entry: bool = False) -> tuple[list[str], list[dict[str, Any]]]:
    """Compute the install action list WITHOUT touching anything.

    Returns ``(descriptions, actions)``; ``actions`` entries are
    ``{"kind": ..., ...}`` understood by :func:`apply_install`. Idempotent:
    already-correct pieces produce no action (the report says "ok").
    """
    home = Path(home).expanduser()
    descs: list[str] = []
    actions: list[dict[str, Any]] = []
    cfg_path = _config_path(home)
    entry = _mcp_entry(home)

    descs.append(f"config: {cfg_path}"
                 f" (backup suffix .bak-before-install-<ts>)")
    actions.append({"kind": "config", "path": str(cfg_path), "entry": entry,
                    "force": force_entry})

    src_dir = _shims_source_dir()
    for name in HOOK_SHIM_NAMES:
        template = (src_dir / name).read_text(encoding="utf-8")
        rendered = _substitute(template, entry=entry)
        target = _hooks_dir(home) / name
        descs.append(f"shim: {target}"
                     f" (backup .bak-<ts> when content differs)")
        actions.append({"kind": "shim", "path": str(target),
                        "content": rendered})

    descs.append("hooks block: merge events "
                 + ", ".join(HOOKS_EVENTS) + " (enabled=true)")
    actions.append({"kind": "hooks", "path": str(cfg_path),
                    "block": _hooks_block(home, entry)})

    if skill:
        skill_src = Path(__file__).resolve().parent.parent.parent / "skills" / _SKILL_NAME
        descs.append(f"skill: {skill_src} -> "
                     f"{home / _USER_SKILL_DIR / _SKILL_NAME} (optional)")
        actions.append({"kind": "skill", "src": str(skill_src),
                        "dst": str(home / _USER_SKILL_DIR / _SKILL_NAME)})
    return descs, actions


def _backup(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_name(path.name + suffix)
    shutil.copy2(path, bak)
    return bak


def apply_install(home: Path, actions: list[dict[str, Any]]) -> list[str]:
    """Execute the planned actions (backup-first, idempotent).

    Config write is a whole-file JSON merge that PRESERVES every other
    server/hook key. Returns human-readable result lines.
    """
    home = Path(home).expanduser()
    ts = time.strftime("%Y%m%d-%H%M%S")
    results: list[str] = []
    cfg_path: Path | None = None
    for act in actions:
        kind = act["kind"]
        if kind == "config":
            cfg_path = Path(act["path"])
        if kind == "hooks":
            cfg_path = Path(act["path"])

    for act in actions:
        kind = act["kind"]
        if kind == "shim":
            target = Path(act["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_text(encoding="utf-8") == act["content"]:
                results.append(f"shim ok (unchanged): {target}")
                continue
            bak = _backup(target, f".bak-{ts}")
            target.write_text(act["content"], encoding="utf-8", newline="\n")
            results.append(f"shim written: {target}"
                           + (f" (backup {bak.name})" if bak else ""))
        elif kind in ("config", "hooks"):
            path = Path(act["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            bak = _backup(path, f".bak-before-install-{ts}")
            changed = False
            if kind == "config":
                servers = data.setdefault("mcp", {}).setdefault("servers", {})
                if "hippocampus" not in servers or act.get("force"):
                    servers["hippocampus"] = act["entry"]
                    changed = True
            else:
                hooks = data.setdefault("hooks", {})
                if not hooks.get("enabled"):
                    hooks["enabled"] = True
                    changed = True
                events = hooks.setdefault("events", {})
                for ev_name, ev_block in act["block"]["events"].items():
                    if events.get(ev_name) != ev_block:
                        events[ev_name] = ev_block
                        changed = True
            if changed or bak is None:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8", newline="\n")
                results.append(f"{kind} written: {path}"
                               + (f" (backup {bak.name})" if bak else ""))
            else:
                results.append(f"{kind} ok (unchanged): {path}")
        elif kind == "skill":
            src, dst = Path(act["src"]), Path(act["dst"])
            if not src.exists():
                results.append(f"skill skipped (source missing): {src}")
                continue
            changed = not dst.exists() or any(
                (src / f).read_bytes() != (dst / f).read_bytes()
                for f in os.listdir(src))
            if changed:
                if dst.exists():
                    _backup(dst, f".bak-{ts}")
                shutil.copytree(src, dst, dirs_exist_ok=True)
                results.append(f"skill written: {dst}")
            else:
                results.append(f"skill ok (unchanged): {dst}")
    return results


# --------------------------------------------------------------------------
# Doctor: hook-chain layers (L1 config entry, L2 shims + hooks.enabled,
# plus the shim mirror-constant drift check).
_SHIM_CONST_RE = {
    "KEY_LIBRARY": re.compile(r"^KEY_LIBRARY = r?\"(.*)\"", re.M),
    "EMB_MODEL": re.compile(r"^EMB_MODEL = \"(.*)\"", re.M),
    "HELLO_LIBRARY": re.compile(r"^HELLO_LIBRARY = r?\"(.*)\"", re.M),
}


def hook_chain_status(home: Path) -> dict[str, Any]:
    """Read-only hook-chain health for ``cli doctor`` (never throws).

    Returns rows: ``config_entry`` (L1), ``shims`` (L2 file existence),
    ``hooks_enabled`` (L2b), ``events`` (L2c per-event registered),
    ``drift`` (shim mirror constants vs the config entry env; empty = OK).
    """
    home = Path(home).expanduser()
    out: dict[str, Any] = {"config_entry": None, "shims": {},
                           "hooks_enabled": None, "events": {},
                           "drift": []}
    cfg_path = _config_path(home)
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    entry = ((data.get("mcp") or {}).get("servers") or {}).get("hippocampus")
    out["config_entry"] = bool(entry)
    if entry:
        is_launcher = entry.get("args") == ["-m", "hippocampus_mcp.launcher"]
        out["config_entry"] = {"present": True, "launcher": is_launcher,
                               "command": entry.get("command", "")}
    env = (entry or {}).get("env") or {}
    hooks_dir = _hooks_dir(home)
    for name in HOOK_SHIM_NAMES:
        p = hooks_dir / name
        out["shims"][name] = p.exists()
        if name == "hippo_context.py" and p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            for key, rx in _SHIM_CONST_RE.items():
                m = rx.search(text)
                shim_val = m.group(1) if m else None
                env_key = {"KEY_LIBRARY": "HIPPOCAMPUS_DAEMON_LIBRARY",
                           "EMB_MODEL": "HIPPOCAMPUS_EMB_MODEL",
                           "HELLO_LIBRARY": "HIPPOCAMPUS_PERSIST_DIR"}[key]
                if shim_val is not None and shim_val != env.get(env_key, ""):
                    out["drift"].append(
                        f"{name}:{key}={shim_val!r} != config env "
                        f"{env_key}={env.get(env_key, '')!r}")
    hooks = data.get("hooks") or {}
    out["hooks_enabled"] = bool(hooks.get("enabled"))
    events = hooks.get("events") or {}
    for ev in HOOKS_EVENTS:
        block = events.get(ev) or []
        registered = any(
            any(name in str(h.get("args", ""))
                for h in item.get("hooks", [])
                for name in HOOK_SHIM_NAMES)
            for item in block)
        out["events"][ev] = registered
    return out


def daemon_key_hint(entry: dict[str, Any] | None) -> str:
    """Best-effort daemon registry key hint (doctor L3 uses it via the
    hook_client registry read; kept here for a single key algorithm)."""
    env = (entry or {}).get("env") or {}
    return daemon_key(env.get("HIPPOCAMPUS_DAEMON_LIBRARY", ""),
                      env.get("HIPPOCAMPUS_EMB_MODEL", ""), "")
