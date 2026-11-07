"""``hippocampus-mcp`` command line entry point (setup / sync / doctor /
models / serve).

Five subcommands::

    hippocampus-mcp serve     start the MCP service (via the launcher by
                              default, shared daemon)
    hippocampus-mcp setup     generate / merge MCP configuration for one host
    hippocampus-mcp sync      discover all host configs and sync the same
                              entry (preview by default)
    hippocampus-mcp doctor    health check: paths / library / model /
                              dependencies ready
    hippocampus-mcp models    local model cache: usage detail / who uses it /
                              reclaim redundancy

**Why setup exists**: the MCP protocol only specifies transport
(stdio / SSE) and JSON-RPC; **it does not specify where config files live**.
Each host implements its own, differing even in the JSON top-level key name
(WorkBuddy and TRAE use ``mcpServers``, ZCode uses ``mcp.servers``); this
project cannot make hosts change those differences.

But the differences can be **absorbed into this command**: the user runs one
command and maintains one intent (which library, which profile, which
model), and this command translates it into each host's own format. What
the user experiences is uniform -- that is the entire reason setup exists.

**Safety discipline** (the four easiest ways writing host configs goes
wrong):

1. **Back up before writing**: copy the target file to
   ``<name>.bak-<timestamp>`` first, then write.
2. **Stop on parse failure**: when an existing config's JSON is corrupt,
   **do not overwrite**; raise and let a human fix it manually -- it may
   contain other MCP servers, and overwriting them is collateral damage.
3. **Touch only our own entry**: merge rather than replace the whole file;
   other server entries are preserved as-is.
4. **Post-write verification**: after writing, automatically perform one
   **real handshake** (initialize + tools/list) with the written config;
   if it cannot start, give the reason on the spot -- do not leave
   discovery to the moment the user restarts the host and hits "0 tools".
   ``--no-verify`` skips it.
5. **sync previews by default**: it targets hosts outside ``_HOSTS``
   (filename conventions + shape sniffing, see ``host_configs``), where
   writing to the wrong place costs more than setup, so it follows the same
   discipline as ``models``' disk deletion -- without ``--yes`` not one
   byte is touched, and only existing files are modified, never created.
"""

# Exemption registry: this file's direct env reads are display pass-throughs
# of introspection commands (read for humans; values match config); the
# closed list is in config.py's "root-layer exemption list".

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import host_configs as hc

SERVER_NAME_DEFAULT = "hippocampus"
LIBRARY_DEFAULT = "~/.hippocampus/brain"
MODEL_DEFAULT = "BAAI/bge-base-zh-v1.5"
PROFILE_DEFAULT = "common"

# Host -> JSON key path of the config file (**the difference lives right
# here**; nowhere else). key_path is a nested-key tuple: both ("mcpServers",)
# and ("mcp", "servers") must be readable and writable.
_HOSTS: dict[str, dict] = {
    "workbuddy": {
        "label": "WorkBuddy",
        "key_path": ("mcpServers",),
        "default_scope": "user",
    },
    "zcode": {
        "label": "ZCode",
        "key_path": ("mcp", "servers"),
        "default_scope": "project",
    },
    "trae": {
        "label": "TRAE SOLO CN",
        "key_path": ("mcpServers",),
        "default_scope": "user",
    },
    "claude": {
        "label": "Claude Desktop / Code",
        "key_path": ("mcpServers",),
        "default_scope": "user",
    },
}


# --------------------------------------------------------------------------
# Path derivation
# --------------------------------------------------------------------------
def user_config_path(host: str) -> Path | None:
    """The host's user-level config path; None when the host has no user-level config."""
    home = Path.home()
    appdata = os.environ.get("APPDATA", "")
    if host == "workbuddy":
        return home / ".workbuddy" / "mcp.json"
    if host == "trae":
        if appdata:
            return Path(appdata) / "TRAE SOLO CN" / "User" / "mcp.json"
        return (home / "Library" / "Application Support" / "TRAE SOLO CN"
                / "User" / "mcp.json")
    if host == "claude":
        if sys.platform == "darwin":
            return (home / "Library" / "Application Support" / "Claude"
                    / "claude_desktop_config.json")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        return home / ".config" / "Claude" / "claude_desktop_config.json"
    if host == "zcode":
        # After the 2026-10 upgrade the user-level config moved to the cli/
        # subdirectory (the old ~/.zcode/config.json is deprecated).
        # Previously this function returned None for zcode -- doctor/sync
        # could not see the user-level config, and when the 09-28 entry added
        # at the old location was lost in the upgrade, no health check could
        # catch it (blind-spot incident, see
        # docs/design/ZCode用户级接入迁移_20261030.md section 1).
        return home / ".zcode" / "cli" / "config.json"
    return None


def project_config_path(host: str, workspace: Path) -> Path | None:
    """The host's workspace-level config path."""
    if host == "workbuddy":
        return workspace / ".mcp.json"
    if host == "zcode":
        return workspace / ".zcode" / "config.json"
    return None


def package_src_dir() -> tuple[Path | None, Path]:
    """Return (src dir | None, package dir).

    When running from ``pip install``'s site-packages, PYTHONPATH is **not
    needed** (the package is already on sys.path) and None is returned; it
    is only needed when running from the repository's ``src/`` layout. This
    distinction matters -- blindly writing PYTHONPATH would point at a
    nonexistent directory on other people's machines.
    """
    pkg_dir = Path(__file__).resolve().parent
    parent = pkg_dir.parent
    if parent.name == "src":
        return parent, pkg_dir
    return None, pkg_dir


# --------------------------------------------------------------------------
# Config read/write (touch only our own entry)
# --------------------------------------------------------------------------
def _dig(cfg: dict, key_path: tuple[str, ...], create: bool = False) -> dict | None:
    """Fetch the container at the key path; with ``create=True`` fill in empty dicts level by level."""
    node: dict = cfg
    for key in key_path:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            if not create:
                return None
            nxt = {}
            node[key] = nxt
        node = nxt
    return node


def build_entry(*, python: str, src_dir: Path | None, library: str,
                model: str, profile: str, offline: bool) -> dict:
    """Build a single MCP server entry (all hosts share the same one; the difference is only the shell)."""
    env: dict[str, str] = {}
    if src_dir is not None:
        env["PYTHONPATH"] = str(src_dir)
    env["HIPPOCAMPUS_PERSIST_DIR"] = library
    env["HIPPOCAMPUS_EMB_MODEL"] = model
    env["HIPPOCAMPUS_STORE_BACKEND"] = "sqlite"
    env["HIPPOCAMPUS_FTS5"] = "1"
    env["HIPPOCAMPUS_TOOL_PROFILE"] = profile
    env["HIPPOCAMPUS_LAUNCHER"] = "1"
    if offline:
        # Only set when the user explicitly asks for offline: open-source
        # users need to download the model on first run; enabling offline by
        # default would leave them stuck at "model fails to load" with no hint.
        env["HF_HUB_OFFLINE"] = "1"
    return {
        "type": "stdio",
        "command": python,
        "args": ["-m", "hippocampus_mcp.launcher"],
        "env": env,
    }


def _read_config(path: Path) -> dict:
    cfg, err = hc.load_json_config(path)
    if cfg is None:
        # A bad file escalates to a full stop: setup/sync's write path must
        # die here rather than take other servers' entries down with it
        # (discovery-style scanning can continue wounded; writing must not --
        # the semantic difference lives in host_configs.load_json_config).
        raise SystemExit(
            f"Configuration file {err}; stopped without writing: {path}\n"
            f"  It may contain other MCP servers; overwriting it would lose them. "
            f"Fix the file manually and rerun.")
    return cfg


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, dst)
    return dst


def _detect_host(workspace: Path) -> str:
    """Guess the host from workspace traits when ``--host`` is omitted.

    The criteria are deliberately conservative: only recognize "host traces
    already present in the directory" (a ``.zcode/`` directory, a
    ``.mcp.json`` file); when guessing fails, raise and list the choices --
    never silently pick one. setup writes the host's real configuration;
    writing to the wrong host is worse than not writing.
    """
    if (workspace / ".zcode").is_dir():
        return "zcode"
    if (workspace / ".mcp.json").is_file():
        return "workbuddy"
    raise SystemExit(
        "Cannot determine target host: workspace has neither .zcode/ nor .mcp.json.\n"
        f"  Specify --host explicitly (choices: {', '.join(sorted(_HOSTS))}).")


# --------------------------------------------------------------------------
# Post-write verification (real handshake)
# --------------------------------------------------------------------------
VERIFY_TIMEOUT = 120.0


def _stderr_tail(fh) -> str:
    """Read the tail of the child process's stderr (the first-hand clue given to the user when probing fails)."""
    try:
        fh.seek(0)
        txt = (fh.read() or "").strip()
    except Exception:  # noqa: BLE001 -- if the tail cannot be read, so be it; do not mask the primary cause
        return ""
    if not txt:
        return ""
    if len(txt) > 600:
        txt = "…" + txt[-600:]
    return f"stderr tail: {txt}"


def _verify_entry(entry: dict, cwd: str = "",
                  timeout: float = VERIFY_TIMEOUT) -> tuple[bool, str]:
    """Actually start one handshake using the command/args/env written into
    the config.

    setup previously only guaranteed "the config got written", not "the host
    can actually start from it" -- a wrong interpreter path, PYTHONPATH not
    taking effect, or the package not installed would only surface when the
    user restarted the host and hit "0 tools". Here discovery is moved up to
    setup itself: spawn a child process, run through initialize +
    tools/list, and usable/unusable is settled within seconds (the handshake
    does not trigger model loading, so it will not hang on minute-scale
    downloads).

    Returns ``(ok, detail)``. The failure detail must **carry the reason**
    (timeout / early exit / error response / stderr tail), otherwise the
    user still has to dig through logs.

    When going through the launcher path (``HIPPOCAMPUS_LAUNCHER=1`` in the
    config), the probe also brings the daemon up as a side effect; after the
    probe process exits the daemon stays warm and is reclaimed when idle, so
    the host's next start is actually hot -- this is a side effect, not a
    defect.
    """
    import queue
    import subprocess
    import tempfile
    import threading

    cmd = [entry.get("command") or sys.executable] + list(entry.get("args") or [])
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (entry.get("env") or {}).items()})
    try:
        err_fh = tempfile.TemporaryFile(mode="w+", encoding="utf-8",
                                        errors="replace")
        proc = subprocess.Popen(cmd, cwd=cwd or None, env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=err_fh)
    except OSError as exc:
        return False, f"Cannot start process {cmd[0]}: {type(exc).__name__}: {exc}"

    lines_q: queue.Queue = queue.Queue()

    def _reader() -> None:
        try:
            for raw in proc.stdout:          # blocking line reads; loop ends naturally when the process exits
                lines_q.put(raw)
        finally:
            lines_q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    def _readline(deadline: float) -> bytes:
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise TimeoutError("timeout")
        try:
            item = lines_q.get(timeout=remain)
        except queue.Empty:
            raise TimeoutError("timeout") from None
        if item is None:
            raise EOFError("child process exited before a response arrived")
        return item

    def _send(obj: dict) -> None:
        proc.stdin.write(
            (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        proc.stdin.flush()

    deadline = time.monotonic() + timeout
    try:
        _send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                          "clientInfo": {"name": "hippocampus-setup-verify",
                                         "version": "0"}}})
        # In theory the first response is initialize's; matching by id
        # withstands out-of-band lines.
        while True:
            obj = json.loads(_readline(deadline).decode("utf-8", "replace"))
            if obj.get("id") == 1:
                break
        if "result" not in obj:
            return False, f"initialize returned error: {obj.get('error')}"
        _send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        while True:
            obj = json.loads(_readline(deadline).decode("utf-8", "replace"))
            if obj.get("id") == 2:
                break
        if "result" not in obj:
            return False, f"tools/list returned error: {obj.get('error')}"
        tools = obj["result"].get("tools") or []
        if not tools:
            return True, "initialize + tools/list succeeded (but 0 tools? check profile)"
        return True, f"initialize + tools/list succeeded; {len(tools)} tools visible"
    except TimeoutError:
        return False, (f"No handshake response within {timeout:.0f}s (possibly model loading or network). "
                       "Slow is not necessarily broken, but repeated failures need daemon logs.")
    except EOFError:
        return False, (f"Child process exited before completing handshake (exit={proc.poll()}). "
                       f"{_stderr_tail(err_fh)}")
    except Exception as exc:  # noqa: BLE001 -- the prober itself must not take setup down
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        for closer in (proc.stdin, proc.stdout):
            try:
                closer.close()
            except Exception:  # noqa: BLE001
                pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
        err_fh.close()


# --------------------------------------------------------------------------
# Subcommand: setup
# --------------------------------------------------------------------------
def _pick_value(value: str | None, old_value: str | None, default: str
                ) -> tuple[str, str]:
    """Value priority: command line > existing config > default.

    The middle tier **cannot be skipped**: skipping it means one run of
    setup / sync on this machine would swap the existing library for the
    new-user default -- the user thinks they are just "filling in a config"
    while the library is silently moved away. This is the most damaging
    mistake writing host configs can cause, hence spelled out explicitly.
    setup and sync share this rule (sync additionally requires **per-file**
    inheritance: different hosts pointing at different libraries is a legal
    state, not a difference to be flattened).
    """
    if value:
        return value, "CLI argument"
    if old_value:
        return old_value, "existing config"
    return default, "fresh-install default"


def cmd_setup(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    if args.host:
        host = args.host
    else:
        # S4: without --host, infer from workspace markers (has .zcode/ ->
        # zcode; has .mcp.json -> workbuddy). Guessing fails loudly; never
        # pick blindly.
        host = _detect_host(workspace)
        print(f"--host omitted; inferred from workspace markers: {host}")
    spec = _HOSTS[host]

    if args.file:
        target = Path(args.file).expanduser().resolve()
    else:
        scope = args.scope or spec["default_scope"]
        guess = (user_config_path(host) if scope == "user"
                 else project_config_path(host, workspace))
        if guess is None:
            raise SystemExit(
                f"{spec['label']} has no {scope}-level configuration path; "
                f"use --file to specify the config file explicitly."
            )
        target = guess

    cfg = _read_config(target)
    container = _dig(cfg, spec["key_path"], create=True)
    old = container.get(args.name)
    old = old if isinstance(old, dict) else {}
    old_env = old.get("env") if isinstance(old.get("env"), dict) else {}
    existed = bool(old)

    src_dir, _pkg = package_src_dir()

    library, lib_src = _pick_value(args.library,
                                   old_env.get("HIPPOCAMPUS_PERSIST_DIR"),
                                   str(Path(LIBRARY_DEFAULT).expanduser()))
    model, model_src = _pick_value(args.model,
                                   old_env.get("HIPPOCAMPUS_EMB_MODEL"),
                                   MODEL_DEFAULT)
    profile, prof_src = _pick_value(args.profile,
                                    old_env.get("HIPPOCAMPUS_TOOL_PROFILE"),
                                    PROFILE_DEFAULT)
    python = args.python or old.get("command") or sys.executable
    offline = args.offline or old_env.get("HF_HUB_OFFLINE") == "1"

    entry = build_entry(python=python, src_dir=src_dir, library=library,
                        model=model, profile=profile, offline=offline)
    container[args.name] = entry

    payload = json.dumps(cfg, ensure_ascii=False, indent=2)

    print(f"Host         : {spec['label']} ({host})")
    print(f"Config file  : {target}")
    print(f"Entry        : {args.name} ({'overwrite existing' if existed else 'new'}; "
          f"other server entries are preserved)")
    print(f"Interpreter  : {python}")
    print(f"Library dir  : {library}  [{lib_src}]")
    print(f"Model        : {model}  [{model_src}]")
    print(f"Tool profile : {profile}  [{prof_src}]")
    if src_dir is None:
        print("PYTHONPATH: not set (package is already importable)")
    else:
        print(f"PYTHONPATH: {src_dir}")

    if args.dry_run:
        print("\n[dry-run] The following was NOT written; preview only:\n")
        print(payload)
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup(target)
    if bak:
        print(f"\nBackup       : {bak}")
    target.write_text(payload + "\n", encoding="utf-8")
    print(f"\nWritten      : {target}")

    if args.no_verify:
        print("[Post-write verification skipped] (--no-verify): availability is unverified; "
              "after restarting the host, call library_whoami to confirm")
    else:
        # S2 post-write verification: perform one real handshake with the
        # **just-written** entry. Failures come with reasons, so "config is
        # broken" shows up right here in setup instead of silently leaving
        # the user with 0 tools after restarting the host.
        print("\nPost-write verification: performing a real handshake with the written "
              "config (initialize + tools/list)...")
        ok, detail = _verify_entry(entry, cwd=str(workspace))
        print(f"  {'✓ config verified' if ok else '✗ handshake verification failed'}: {detail}")
        if not ok:
            print("  Config was written, but the host may fail to start. Diagnose:")
            print(f"    python -m hippocampus_mcp.cli doctor --host {host} "
                  f"--project {workspace}")
            return 1

    print("\nNext steps:")
    # When the model is not local, say it up front: the model is downloaded
    # on first start; a failed download leaves the host with "not a single
    # tool", and by then the user is already inside a session -- the hardest
    # thing to diagnose.
    missing_model = False
    try:
        from .engine.embeddings import _cache_roots, _default_libs_path, local_model_dir
        missing_model = local_model_dir(model, _cache_roots(_default_libs_path())) is None
    except Exception:                                        # noqa: BLE001
        missing_model = False
    step = 0
    if missing_model:
        step += 1
        print(f"  {step}) Prefetch the model first ({model} is not local; a failed "
              f"first-start download can leave the host with no tools):")
        print(f"     hippocampus-mcp models --fetch {model}")
    step += 1
    print(f"  {step}) Restart that host (MCP config is read at startup)")
    step += 1
    print(f"  {step}) In a new session, call library_whoami first to confirm the active library")
    step += 1
    print(f"  {step}) If in doubt, run hippocampus-mcp doctor")
    return 0


# --------------------------------------------------------------------------
# Subcommand: sync
# --------------------------------------------------------------------------
def cmd_sync(args: argparse.Namespace) -> int:
    """Discover all host configs and sync the same entry into them ("configure
    once, sync everywhere").

    Division of labor vs setup: the 4 hosts ``setup --host`` knows are
    written directly without scanning; sync targets **all other** hosts --
    no host list is maintained; configs are found via host_configs'
    "filename conventions + shape sniffing". Preview only by default
    (candidates + verdicts listed); --yes to write.

    Write discipline: each file inherits values independently (different
    hosts pointing at different libraries is a legal state, not flattened);
    touch only our own entry; re-parse before writing and stop-and-skip on
    bad files; skip when the entry is already identical (idempotent, no
    redundant backups); after writing, handshake **each deduplicated entry**
    once (per-file inheritance can produce different entries; verifying only
    one would miss the others).
    """
    workspace = Path(args.workspace).expanduser().resolve()
    candidates = hc.discover_configs(workspace, entry_name=args.name)
    action = {"mine": "update", "new": "add", "suspect": "add (low confidence)",
              "broken": "stop", "skip": "skip"}
    targets = [c for c in candidates if c.kind in ("mine", "new", "suspect")]

    print(f"sync preview (workspace {workspace}, "
          f"{'--yes will write' if args.yes else 'not written'})")
    n_skipped = 0
    for c in candidates:
        if c.kind == "skip":            # shape mismatches are only counted, not printed
            n_skipped += 1
            continue
        note = c.reason or {"mine": "our entry already exists",
                            "new": "other server entries exist"}.get(c.kind, "")
        print(f"  [{action[c.kind]}] {c.origin:<7} {c.path}"
              f"{'  ' + note if note else ''}")
    if n_skipped:
        print(f"  ({n_skipped} files omitted because their shape did not match)")
    if not targets:
        print("  No syncable host configs found -- use setup --host for installed hosts,")
        print("  or place a config at a conventional path (e.g. *mcp*.json) and rerun.")
        return 0
    if not args.yes:
        print("\nPreview mode; nothing was written. Confirm the list, then add --yes.")
        return 0

    written: list[tuple[Path, Path | None]] = []
    distinct: dict[str, dict] = {}
    print()
    for c in targets:
        cfg, err = hc.load_json_config(c.path)
        if cfg is None:
            # The file may have gone bad between preview and write: a failed
            # re-parse = stop and skip, never overwrite.
            print(f"  ⚠ Stopped at {c.path}: {err} (bad file not written; fix it manually)")
            continue
        key_path = c.key_path or ("mcpServers",)
        container = _dig(cfg, key_path, create=True)
        old = container.get(args.name)
        old = old if isinstance(old, dict) else {}
        old_env = old.get("env") if isinstance(old.get("env"), dict) else {}
        library, _ = _pick_value(args.library,
                                 old_env.get("HIPPOCAMPUS_PERSIST_DIR"),
                                 str(Path(LIBRARY_DEFAULT).expanduser()))
        model, _ = _pick_value(args.model,
                               old_env.get("HIPPOCAMPUS_EMB_MODEL"),
                               MODEL_DEFAULT)
        profile, _ = _pick_value(args.profile,
                                 old_env.get("HIPPOCAMPUS_TOOL_PROFILE"),
                                 PROFILE_DEFAULT)
        python = args.python or old.get("command") or sys.executable
        offline = args.offline or old_env.get("HF_HUB_OFFLINE") == "1"
        src_dir, _pkg = package_src_dir()
        entry = build_entry(python=python, src_dir=src_dir, library=library,
                            model=model, profile=profile, offline=offline)
        if container.get(args.name) == entry:
            print(f"  = Already up to date {c.path}")
            continue
        container[args.name] = entry
        c.path.parent.mkdir(parents=True, exist_ok=True)
        bak = _backup(c.path)
        c.path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"  ✓ Written {c.path}" + (f" (backup {bak})" if bak else ""))
        written.append((c.path, bak))
        distinct[json.dumps(entry, ensure_ascii=False, sort_keys=True)] = entry

    if not written:
        print("\nNo changes to write.")
        return 0

    if args.no_verify:
        print("\n[Post-write verification skipped] (--no-verify): availability is unverified; "
              "after restarting the host, call library_whoami to confirm")
        return 0

    print("\nPost-write verification: performing a real handshake for each unique entry...")
    failed = False
    for entry in distinct.values():
        ok, detail = _verify_entry(entry, cwd=str(workspace))
        print(f"  {'✓' if ok else '✗'} {detail}")
        failed = failed or not ok
    if failed:
        print("\nHandshake verification failed -- written files and backups (use backups to roll back):")
        for path, bak in written:
            print(f"  {path}" + (f" (backup {bak})" if bak else ""))
        return 1

    print("\nNext: restart each host (MCP config is read at startup); in a new session, "
          "call library_whoami to confirm libraries were not crossed.")
    return 0


# --------------------------------------------------------------------------
# Subcommand: doctor
# --------------------------------------------------------------------------
def _mark(ok: bool, warn: bool = False) -> str:
    return "[FAIL]" if not ok else ("[WARN]" if warn else "[ OK ]")


def _entry_verdict(cpath: Path, spec: dict) -> str:
    """Verdict line for one config file: missing / exists without our entry / exists and where its library points."""
    if not cpath.exists():
        return "Missing (not wired up here)"
    try:
        cfg = _read_config(cpath)
    except SystemExit:
        return f"Exists but is not valid JSON (host cannot read any server until fixed) -- {cpath}"
    ent = (_dig(cfg, spec["key_path"]) or {}).get(SERVER_NAME_DEFAULT)
    if not isinstance(ent, dict):
        return f"Exists but has no {SERVER_NAME_DEFAULT} entry (not wired up)"
    env = ent.get("env") or {}
    lib = env.get("HIPPOCAMPUS_PERSIST_DIR", "")
    prof = env.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common (default)"
    return f"Exists; library={lib or '(unset; falls back to default)'}  profile={prof}"


def _doctor_host_project(host: str, project: Path) -> None:
    """**Verdict-style** check of host x directory (``doctor --host X --project
    <dir>``).

    The old matrix only "listed what exists", missing "what is missing" and
    "which file should be read" -- the typical wiring incident (ZCode's
    config mistakenly written as ``.mcp.json``) looked all green in the old
    matrix. Here a verdict is given along "what the host reads -> does it
    exist -> where does the library point", and config files in the
    directory that **exist but this host does not read** are named
    explicitly.
    """
    spec = _HOSTS[host]
    print(f"\nHost/workspace verdict: {spec['label']} @ {project}")
    for scope, cpath in (("user", user_config_path(host)),
                         ("project", project_config_path(host, project))):
        if cpath is None:
            continue
        print(f"  Reads {scope} level : {cpath}")
        print(f"                → {_entry_verdict(cpath, spec)}")
    for other, ospec in _HOSTS.items():
        if other == host:
            continue
        opath = project_config_path(other, project)
        if opath is not None and opath.exists():
            print(f"  ⚠ Exists but {spec['label']} does not read it: {opath}"
                  f" ({ospec['label']} project-level config)")


def _doctor_workspace_binding(project: Path) -> None:
    """D3: workspace binding verdict (conclusion-style output of ``doctor
    --project``).

    Is this project directory registered in the user-level task directory?
    Hit = the first session's HELLO binds straight to that library (with its
    read-side access level); unregistered = first use self-reports
    "unbound", with a one-command onboarding form. library/task_directory
    are lightweight imports (no torch); any failure degrades to one line of
    "cannot check" -- the health check itself must never blow up.
    """
    print("\nWorkspace binding verdict (C1/D3):")
    try:
        from .domain.task_directory import (normalize_entries,
                                            resolve_user_access,
                                            resolve_workspace)
        from .library import resolve_library, scan_libraries
        from .library.task_directory import load_directory
        try:
            ulib = resolve_library("user", libs=scan_libraries())
        except Exception:
            ulib = None
        if ulib is None:
            print("  User-level library not found; binding/task-directory features are not enabled "
                  "(single-library use is unaffected).")
            return
        data, issue = load_directory(ulib.path)
        if issue:
            print(f"  ⚠ Task directory is unreadable ({issue}); binding status cannot be checked, "
                  "and HELLO reverse lookup will fall back to the declared library.")
            return
        entry, status = resolve_workspace(
            normalize_entries((data or {}).get("entries")), str(project))
        if status == "bound":
            print(f"  Bound: this project -> {entry['path']} (task={entry['task']}, "
                  f"read access={entry.get('access') or 'masked'})")
        elif status == "ambiguous":
            print("  ⚠ Ambiguous: multiple entries declare the same workspace_root; HELLO will "
                  "fall back to the declared library and self-report. Fix directory.json "
                  "(one workspace should point to one library).")
        elif status == "unbound":
            print("  Not registered: first use in this project will self-report as unbound. Onboarding:")
            print("    library_create(name=..., workspace_root=r'%s')"
                  % project)
        else:
            print("  No workspace path to evaluate (--project was omitted or empty).")
    except Exception as exc:  # noqa: BLE001 -- the health-check freebie must not crash the main flow
        print(f"  (Binding status unavailable: {type(exc).__name__}: {exc})")


def _doctor_off_registry(project: Path) -> None:
    """Off-registry host configs (same scan source as sync, read-only) --
    doctor must not only know _HOSTS.

    Crossed-library incidents do not only happen with the registered hosts:
    if the user casually wires hippocampus into some unknown host with the
    library pointing wrong, it silently writes wrong all the same. This
    names the config files discovered by scanning that _HOSTS does not
    register (including broken files and low-confidence inferences; skipped
    pure noise is not counted).
    """
    print("\nOff-registry host configs discovered (MCP configs outside _HOSTS):")
    known: set[str] = set()
    for host in _HOSTS:
        for cpath in (user_config_path(host),
                      project_config_path(host, project)):
            if cpath is not None:
                known.add(os.path.normcase(str(cpath.resolve())))
    try:
        candidates = hc.discover_configs(project)
    except Exception as exc:  # noqa: BLE001 -- the health-check freebie must not crash the main flow
        print(f"  (Scan failed: {type(exc).__name__}: {exc})")
        return
    found = False
    for c in candidates:
        if c.kind == "skip" or (os.path.normcase(str(c.path.resolve()))
                                in known):
            continue
        found = True
        if c.kind == "mine":
            env = (c.entry or {}).get("env") or {}
            lib = env.get("HIPPOCAMPUS_PERSIST_DIR", "")
            prof = env.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common (default)"
            print(f"  {c.origin} <{c.path}> : our entry  library={lib or '(unset; falls back to default)'}"
                  f"  profile={prof}")
        elif c.kind == "broken":
            print(f"  {c.origin} <{c.path}> : {c.reason}"
                  f" -- host cannot read it and it will not be written automatically until fixed")
        elif c.kind == "new":
            print(f"  {c.origin} <{c.path}> : other server entries exist"
                  f" (not wired to this service; run sync --yes to add it)")
        else:
            print(f"  {c.origin} <{c.path}> : {c.reason}"
                  f" (not wired to this service; run sync --yes to add it)")
    if not found:
        print("  (No off-registry configs found)")


# Field feedback 2026-10-31: a library (especially the user library, with
# embeddings, only ever growing) landing on the system drive keeps squeezing
# the C drive. Here we only health-check + give a three-step manual
# migration -- deliberately **no relocate command**: one-shot operations do
# not get resident commands (moving GB-scale vectors also means dealing with
# daemon file locks; the cost far exceeds the benefit). The migration story
# is spelled out in the health-check output below.
LIB_ON_SYSTEM_DISK_WARN_BYTES = 1024 ** 3  # 1 GB


def _dir_size_bytes(path: Path) -> int:
    """Directory size (sum of st_size). walk / stat failures count as 0 -- the health check must not blow up."""
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _system_disk_migration_hint(lib_path: Path, size_bytes: int,
                                system_drive: str) -> str | None:
    """Health-check verdict when the library sits on the system drive (pure
    function, unit-testable).

    Non-system drive -> None (no nagging); system drive + over 1 GB -> the
    three migration steps; system drive but under -> one line of "no move
    needed yet". Library identity is determined by manifest name, not path,
    so moving drives requires zero code change (edit env + move the
    directory + restart).
    """
    try:
        lib_drive = os.path.splitdrive(str(lib_path))[0].upper() + "\\"
    except Exception:  # noqa: BLE001 -- if the drive letter cannot be determined, pretend it is not an issue
        return None
    if not system_drive or lib_drive.upper() != system_drive.upper():
        return None
    gb = size_bytes / 1024 ** 3
    if size_bytes >= LIB_ON_SYSTEM_DISK_WARN_BYTES:
        return (f"⚠ Library is on the system drive {lib_drive}; current size {gb:.1f} GB"
                f" (embeddings only grow and will keep consuming the system drive).\n"
                f"  Migration steps (there is no relocate command; move manually; library identity "
                f"is determined by manifest name, not path):\n"
                f"  1) Stop all host sessions (the daemon must not hold library files while moving);\n"
                f"  2) Move the library directory to the new drive, e.g. D:\\hippocampus\\user;\n"
                f"  3) Update HIPPOCAMPUS_PERSIST_DIR in the host config, "
                f"then verify with library_whoami in a new session.")
    return (f"Library is on the system drive {lib_drive}; current size {gb:.2f} GB "
            f"(<1 GB; no move needed yet; use migration steps above 1 GB)")


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util

    print("hippocampus-mcp doctor")
    print("-" * 60)

    if getattr(args, "host", None):
        _doctor_host_project(args.host,
                             Path(args.project).expanduser().resolve())
        _doctor_workspace_binding(Path(args.project).expanduser().resolve())
        print("-" * 60)

    py_ok = sys.version_info >= (3, 10)
    print(f"{_mark(py_ok)} Python {sys.version.split()[0]} "
          f" (requires >=3.10)")

    src_dir, pkg_dir = package_src_dir()
    print(f"{_mark(True)} Package location   {pkg_dir}")
    print(f"{_mark(True)} PYTHONPATH requirement "
          f"{'required: ' + str(src_dir) if src_dir else 'not required (already importable)'}")

    for mod, required in (("mcp", True), ("torch", True),
                          ("numpy", True), ("sentence_transformers", False)):
        found = importlib.util.find_spec(mod) is not None
        print(f"{_mark(found, warn=not found and not required)} Dependency {mod}"
              f"{'' if found else (' (missing; core functionality unavailable)' if required else ' (missing; degraded)')}")

    # S7 self-disclosure: running doctor bare in a shell has no host env, so
    # what is checked is the default library -- without saying so it would be
    # misread as "the library in the host config is broken". Same for the
    # model.
    persist_env = os.environ.get("HIPPOCAMPUS_PERSIST_DIR", "").strip()
    library = Path(persist_env or LIBRARY_DEFAULT).expanduser()
    # A library / model "not there yet" is not a fault: first start creates /
    # auto-downloads it. Reporting FAIL would make people think the install
    # is broken, so these two items only get WARN.
    lib_exists = library.exists()
    print(f"{_mark(True, warn=not lib_exists)} Library dir   {library}"
          f"{'' if lib_exists else ' (not found yet; first start will create it)'}")
    if not persist_env:
        print("       ⚠ HIPPOCAMPUS_PERSIST_DIR is not set; this checks the default library, "
              "not a library from any host config")
    if lib_exists:
        backend = ("sqlite" if (library / "hippocampus.db").exists()
                   else ("json" if (library / "memories.json").exists() else "empty"))
        print(f"{_mark(True)} Library backend   {backend}")
        if backend != "empty":
            # V1-c: L0 citation health (dangling/orphan) — read-only probe via
            # tree_view's snapshot loader (same numbers audit_l0_dangling prints).
            try:
                from .tree_view import l0_health, snapshot_library
                h = l0_health(snapshot_library(str(library)))
                ok = h["dangling_ids"] == 0
                print(f"{_mark(ok, warn=not ok)} L0 health   raw {h['l0_raw_total']}, "
                      f"refs {h['refs_total']}, dangling {h['dangling_ids']} "
                      f"(affecting {h['affected_memories']} memories), "
                      f"orphan raw {h['orphan_raw']}")
                if not ok:
                    print("       details: python scripts/audit_l0_dangling.py "
                          f"--persist-dir {library}")
            except Exception as exc:  # noqa: BLE001 — probe must not fail doctor
                print(f"{_mark(False, warn=True)} L0 health   probe failed: {exc}")
        # Field feedback 2026-10-31: drive letter / size check (the relocate
        # story is spelled out here).
        size_b = _dir_size_bytes(library)
        home_drive = (os.path.splitdrive(os.path.expanduser("~"))[0]
                      + "\\")
        hint = _system_disk_migration_hint(library, size_b, home_drive)
        if hint:
            print("       " + hint.replace("\n", "\n       "))

    model_env = os.environ.get("HIPPOCAMPUS_EMB_MODEL", "").strip()
    model = model_env or MODEL_DEFAULT
    cache = os.environ.get("HIPPOCAMPUS_EMB_CACHE", "") or str(
        Path.home() / ".cache" / "huggingface" / "hub")
    slug = "models--" + model.replace("/", "--")
    have_model = (Path(cache) / slug).exists()
    print(f"{_mark(True, warn=not have_model)} Model     {model}"
          f"{'' if have_model else f' (not found under {cache}; normal for a custom model directory, and first start may download it)'}")
    if not model_env:
        print("       ⚠ HIPPOCAMPUS_EMB_MODEL is not set; this checks the default model "
              "(actual model is determined by host config env)")

    profile = os.environ.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common (default)"
    print(f"{_mark(True)} Tool profile {profile}")

    # Naturalization P0-3: layered hook-chain health (L1/L2 + drift).
    try:
        _doctor_hook_chain()
    except Exception as exc:  # noqa: BLE001 — doctor must never crash here
        print(f"{_mark(False, warn=True)} L1/L2 hook-chain check failed: {exc}")

    # The most valuable section: **which library each host actually points
    # at**. Crossed libraries are this project's costliest pit (writing into
    # the wrong library raises no error); listing host by host is the only
    # way to see it at a glance.
    print("\nLibraries configured by host (check for crossed libraries):")
    found_any = False
    for host, spec in _HOSTS.items():
        candidates = (("user", user_config_path(host)),
                      ("project", project_config_path(host, Path.cwd())))
        for scope, cpath in candidates:
            if cpath is None or not cpath.exists():
                continue
            found_any = True
            try:
                cfg = _read_config(cpath)
            except SystemExit:
                print(f"  {spec['label']}({scope}) <{cpath}> : "
                      f"config is not valid JSON; skipped")
                continue
            cont = _dig(cfg, spec["key_path"])
            ent = (cont or {}).get(SERVER_NAME_DEFAULT)
            if not isinstance(ent, dict):
                print(f"  {spec['label']}({scope}) <{cpath}> : "
                      f"no {SERVER_NAME_DEFAULT} entry (not wired up)")
                continue
            env = ent.get("env") or {}
            lib = env.get("HIPPOCAMPUS_PERSIST_DIR", "")
            prof = env.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common (default)"
            # Each line carries the config file path: once "crossed library /
            # not wired" is spotted, the next step is always "which file to
            # edit" -- the path is right there, no guessing.
            print(f"  {spec['label']}({scope}) <{cpath}> : "
                  f"library={lib or '(unset; falls back to default)'}  profile={prof}")
    if not found_any:
        print("  (No host config files found -- run setup --host <host> first)")

    # The off-registry scan complements the matrix above: the matrix answers
    # "what do the known hosts read"; this section answers "is a config lying
    # around inside an unknown host" -- crossed libraries do not care whether
    # the host is known.
    _doctor_off_registry(Path.cwd())

    print("-" * 60)
    print("Note: this command performs static checks only and does not load models (so it is fast).")
    print("      Actual availability is what library_whoami reports in the host session.")
    print("      workspace=undeclared means a direct connection without a HELLO "
          "declaration; it is normal, not a fault.")
    return 0


# --------------------------------------------------------------------------
# Subcommand: models
# --------------------------------------------------------------------------
def cmd_models(args: argparse.Namespace) -> int:
    """List the local embedding model cache and reclaim redundancy on demand.

    **Nothing is deleted by default**: without ``--prune`` only the inventory
    is shown; with ``--prune`` only a plan is produced; actually deleting
    from disk additionally requires ``--yes``. The three gates are
    intentional -- the only thing this command can do is delete files, and
    the cost of deleting wrongly is "some library fails to load its model on
    next start", unfixable in an offline environment.
    """
    from .config import load_config
    from .engine import model_cache as MC
    from .engine.embeddings import (
        _cache_roots,
        _default_libs_path,
        default_model_for_system,
        local_model_dir,
    )
    from .engine.onnx_embedder import backend_kind
    from .library import scan_libraries

    roots = ([os.path.abspath(os.path.expanduser(args.cache))]
             if args.cache else _cache_roots(_default_libs_path()))
    models = MC.scan_models(roots)

    # Reference relations: the library manifest's embedder_model is the **only**
    # trusted source (empty string = not pinned)
    refs: dict[str, list[str]] = {}
    for info in scan_libraries():
        mid = (info.embedder_model or "").strip()
        if mid:
            refs.setdefault(mid, []).append(info.name or os.path.basename(info.path))

    default_id, default_why = default_model_for_system()
    protected = {default_id}
    cfg = load_config()
    if cfg.emb_model.strip():
        protected.add(cfg.emb_model.strip())

    if args.fetch:
        # Download the model ahead of time, at install time.
        # Why this step must exist separately: the model is **downloaded on
        # first start**; once offline / blocked, the MCP service simply fails
        # to start, which inside the host looks like "not a single tool" --
        # neither user nor agent can tell the root cause is a missing model.
        # Move this to install time, and on failure lay out the three ways
        # out clearly.
        want = args.fetch if isinstance(args.fetch, str) else default_id
        ready = None
        try:
            ready = local_model_dir(want, roots)
        except Exception:                                    # noqa: BLE001
            ready = None
        if ready:
            print(f"[Ready] {want} already local: {ready}")
            return 0

        # setdefault's return value is the effective value: no need to read
        # the env var again (the root-layer exemption list pins cli.py's env
        # read count; see the table in config.py).
        hf_endpoint = os.environ.setdefault(
            "HF_ENDPOINT", "https://hf-mirror.com")
        cache = os.path.abspath(os.path.expanduser(
            args.cache or (roots[0] if roots else
                           os.path.join(_default_libs_path(), "models"))))
        print(f"Prefetch model   : {want}")
        print(f"Cache root       : {cache} (using HF mirror {hf_endpoint})")
        print("First download is about 90MB (English baseline) to 400MB (Chinese engine); please wait...")
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(want, cache_folder=cache)
            # Actually run one encode: checking "files are in place" is not
            # enough -- broken weights/tokenizers only explode on first
            # retrieval, by which time the user is already inside a session.
            vec = st.encode(["warmup"], show_progress_bar=False)
            dim = int(len(vec[0]))
        except Exception as e:                               # noqa: BLE001
            print(f"\n[Failed] {want} download or load failed: {type(e).__name__}: {e}")
            print("Three options (lowest cost first):")
            print(f"  1) Retry with a mirror: set HF_ENDPOINT=https://hf-mirror.com and rerun"
                  f" `hippocampus-mcp models --fetch {want}`")
            print(f"  2) Manual placement: put the model directory under {cache}{os.sep}<model name>/ and rerun this command")
            print("  3) Switch to the English baseline (88MB; Chinese retrieval will degrade significantly): "
                  "`hippocampus-mcp models --fetch all-MiniLM-L6-v2`，"
                  "and add --model all-MiniLM-L6-v2 to setup")
            return 1
        print(f"[Done] {want} is ready ({dim} dimensions, cached at {cache})")
        return 0

    if args.json:
        print(json.dumps({
            "ok": True,
            "cache_roots": roots,
            "default_model": default_id,
            "default_reason": default_why,
            "backend": backend_kind(),
            "models": [
                {
                    "model_id": m.model_id,
                    "path": m.path,
                    "total_bytes": m.total_bytes,
                    "weight_bytes": m.weight_bytes,
                    "onnx_bytes": m.onnx_bytes,
                    "int8_bytes": m.int8_bytes,
                    "other_bytes": m.other_bytes,
                    "libraries": sorted(refs.get(m.model_id) or []),
                    "is_default": m.model_id in protected,
                }
                for m in models
            ],
        }, ensure_ascii=False, indent=2))
        return 0

    print(MC.format_report(models, in_use=refs, protected=protected,
                           cache_roots=roots))
    print(f"Current default model : {default_id} ({default_why})")
    print(f"Inference backend     : {backend_kind()}")

    if not (args.prune or args.drop):
        print("\nTip: add --prune to see reclaimable space (still no deletion); add --yes to delete.")
        return 0

    keep_onnx = backend_kind() != "torch"
    plan = MC.plan_prune(models, in_use=refs, protected=protected,
                         keep_onnx=keep_onnx)

    if args.drop:
        by_id = {m.model_id: m for m in models}
        picked: list[MC.PruneAction] = []
        for mid in [d.strip() for d in args.drop if d.strip()]:
            m = by_id.get(mid)
            if m is None:
                print(f"[Skipped] {mid}: model is not in the cache")
                continue
            if refs.get(mid):
                print(f"[Refused] {mid}: referenced by libraries {', '.join(sorted(refs[mid]))}; "
                      f"that library could not load the model after deletion")
                continue
            if mid in protected:
                print(f"[Refused] {mid}: this is the current default model; deleting it requires re-downloading on next start")
                continue
            picked.append(MC.PruneAction(
                kind=MC.KIND_MODEL, model_id=mid, path=m.path,
                bytes_=m.total_bytes, reason="explicitly selected by --drop"))
        # Keep the int8-type actions that "are reclaimable whether named or not"
        plan.actions = picked + [a for a in plan.actions
                                 if a.kind != MC.KIND_MODEL]

    print()
    print(MC.format_plan(plan))
    if not plan.actions:
        return 0

    if not args.yes:
        print(f"\n[dry-run] The above {MC.human_size(plan.total_bytes)} was not deleted. "
              f"Confirm, then add --yes to execute.")
        return 0

    freed, errors = MC.apply_prune(plan.actions, roots)
    print(f"\nFreed {MC.human_size(freed)}")
    for e in errors:
        print(f"  [FAIL] {e}")
    if errors:
        return 1
    print("Note: the next start will rescan the cache; if you want the ONNX fast path back, "
          "跑 scripts/export_onnx.py 重生成 onnx/。")
    return 0


# --------------------------------------------------------------------------
# Subcommand: serve
# --------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> int:
    """Start the service.

    Goes through the **launcher** by default: multiple hosts / sessions
    share one daemon, and the embedding model is installed only once.
    ``--direct`` falls back to the direct server (equivalent to the old
    behavior; one model copy per session).
    """
    if args.direct:
        from .server import main as _server_main
        return _server_main() or 0
    from .launcher import main as _launcher_main
    return _launcher_main([])


# --------------------------------------------------------------------------
# Subcommand: context
# --------------------------------------------------------------------------
def cmd_install(args: argparse.Namespace) -> int:
    """Idempotent ZCode deployment: config entry + shims + hooks + skill.

    Reinstall = repair (naturalization P0-2): shims are copied from the
    versioned ``hook_shims/`` templates with token substitution; a config
    entry that already exists is left untouched unless --force, so user
    tuning is never silently overwritten.
    """
    from .installer import plan_install, apply_install
    home = Path.home()
    descs, actions = plan_install(home, skill=not args.no_skill,
                                  force_entry=args.force)
    print("hippocampus install (host=zcode)"
          + ("  [DRY-RUN]" if args.dry_run else ""))
    print("-" * 60)
    for d in descs:
        print(" plan:", d)
    if args.dry_run:
        print("-" * 60)
        print("dry-run: nothing written. Re-run without --dry-run to apply.")
        return 0
    for line in apply_install(home, actions):
        print(" done:", line)
    print("-" * 60)
    print("Next: restart the host (hook/config changes need a NEW session), "
          "then run `hippocampus doctor` to verify the chain.")
    return 0


def _doctor_hook_chain() -> None:
    """L1/L2 hook-chain health + shim mirror-constant drift (P0-3)."""
    from .installer import hook_chain_status
    st = hook_chain_status(Path.home())
    entry = st["config_entry"]
    if entry is True or (isinstance(entry, dict) and entry.get("present")):
        launcher = entry.get("launcher") if isinstance(entry, dict) else True
        print(f"{_mark(bool(launcher))} L1 user-level MCP entry "
              + ("" if launcher else "(present but NOT the launcher entry)"))
    else:
        print(f"{_mark(False)} L1 user-level MCP entry missing "
              f"({_config_hint()})")
    for name in ("hippo_context.py", "hippo_stop.py", "hippo_session.py"):
        ok = st["shims"].get(name)
        print(f"{_mark(bool(ok))} L2 shim {name} "
              + ("" if ok else "(missing; run `hippocampus install`)"))
    print(f"{_mark(bool(st['hooks_enabled']))} L2 hooks.enabled "
          + ("" if st["hooks_enabled"] else "(false/absent)"))
    for ev, ok in st["events"].items():
        print(f"{_mark(bool(ok))} L2 hook event {ev} registered")
    for d in st["drift"]:
        print(f"{_mark(False, warn=True)} L2 shim drift: {d}")
    print("       (L3 daemon reachability: "
          "`hippocampus context \"ping\"` answers within 6s when alive; "
          "L4 binding: `hippocampus doctor --host zcode --project .`)")


def _config_hint() -> str:
    return str(Path.home() / ".zcode" / "cli" / "config.json")


def cmd_context(args: argparse.Namespace) -> int:
    """Hook injection preview: run one query through the full hook_client
    chain.

    Shares :func:`hook_client.fetch_context` with the UserPromptSubmit hook,
    for manual verification and troubleshooting (works only while the daemon
    is alive; this command does **not** start the daemon). Env defaults match
    the launcher: the key's three components read HIPPOCAMPUS_DAEMON_LIBRARY
    / HIPPOCAMPUS_EMB_MODEL / HIPPOCAMPUS_DAEMON_VERSION; the HELLO-declared
    library reads HIPPOCAMPUS_PERSIST_DIR.
    """
    from .hook_client import fetch_context
    trace: list[str] = []
    text = fetch_context(
        args.query or "", str(Path(args.cwd).resolve()),
        top_k=args.top_k, token_budget=args.budget,
        item_max_chars=args.item_chars,
        key_library=os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", ""),
        emb_model=args.model or os.environ.get("HIPPOCAMPUS_EMB_MODEL",
                                               MODEL_DEFAULT),
        daemon_version=os.environ.get("HIPPOCAMPUS_DAEMON_VERSION", ""),
        hello_library=os.environ.get("HIPPOCAMPUS_PERSIST_DIR", ""),
        trace=trace)
    if not text:
        print("(No injection content)", file=sys.stderr)
        for line in trace:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(text)
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """SessionStart hook preview: run the read-only opening anchor through
    fetch_bootstrap (pins / open todos / recent sessions / unfinished
    topics; does NOT create a session and does not start the daemon)."""
    from .hook_client import fetch_bootstrap
    trace: list[str] = []
    text = fetch_bootstrap(
        str(Path(args.cwd).resolve()),
        top_k=args.top_k, token_budget=args.budget,
        item_max_chars=args.item_chars,
        key_library=os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", ""),
        emb_model=os.environ.get("HIPPOCAMPUS_EMB_MODEL", MODEL_DEFAULT),
        daemon_version=os.environ.get("HIPPOCAMPUS_DAEMON_VERSION", ""),
        hello_library=os.environ.get("HIPPOCAMPUS_PERSIST_DIR", ""),
        trace=trace)
    if not text:
        print("(No bootstrap content)", file=sys.stderr)
        for line in trace:
            print(f"  - {line}", file=sys.stderr)
        return 1
    if args.compact:
        text += "\n(Context was compacted -- full-package reminder line "                 "appended by the shim)"
    print(text)
    return 0


def _cli_daemon_key() -> str:
    """Recompute the key exactly as launcher.main() does."""
    from .launcher import daemon_key
    library = os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", "")
    model = os.environ.get("HIPPOCAMPUS_EMB_MODEL", "")
    version = os.environ.get("HIPPOCAMPUS_DAEMON_VERSION", "")
    return daemon_key(library, model, version)


def _clear_stop_marker(key: str) -> None:
    from .launcher import stop_request_path
    try:
        os.remove(stop_request_path(key))
    except OSError:
        pass


def _library_for_workspace(cwd: str, user_lib: str | None = None) -> str | None:
    """Reverse-lookup the library bound to a workspace (read-only, daemon-free).

    Reads the same mapping the host session uses — the user task directory
    (``<user library>/directory.json``) maps ``workspace_root`` to a library
    ``path`` — and returns the library directory of the **deepest** bound
    workspace root that contains *cwd*. Returns ``None`` when nothing matches
    or the sidecar is unreadable; the caller decides on the fallback.
    """
    from .library.task_directory import load_directory

    lib = user_lib or os.path.join(os.path.expanduser("~"), ".hippocampus_mcp",
                                   "user")
    try:
        data, _issue = load_directory(lib)
    except Exception:  # noqa: BLE001 — a lookup failure must not break the command
        return None
    key = os.path.normcase(os.path.abspath(cwd))
    best: tuple[str, str] | None = None
    for entry in data.get("entries") or []:
        wr = str(entry.get("workspace_root") or "").strip()
        path = str(entry.get("path") or "").strip()
        if not wr or not path:
            continue
        wrn = os.path.normcase(os.path.abspath(os.path.expanduser(wr)))
        if key == wrn or key.startswith(wrn + os.sep):
            if best is None or len(wrn) > len(best[0]):
                best = (wrn, path)
    return best[1] if best else None


# --------------------------------------------------------------------------
# Subcommand: tree (V3, read-only library tree + static HTML viewer)
# --------------------------------------------------------------------------
def cmd_tree(args: argparse.Namespace) -> int:
    """Render a read-only tree of a library (markdown, or single-file HTML).

    Read-only guarantees and privacy red lines live in ``tree_view``:
    the library is opened ``mode=ro`` (sqlite) / plain-read (json); the user
    library is rendered with zero content and ``--full`` is refused there;
    L0 raw text never enters the output.
    """
    import os

    from .tree_view import (
        LibraryNotFoundError,
        TreePrivacyError,
        aggregate,
        is_user_library,
        library_kind,
        render_html,
        render_markdown,
        snapshot_library,
    )

    if args.persist_dir:
        persist = str(Path(args.persist_dir).expanduser())
    else:
        workspace = os.path.abspath(args.project or os.getcwd())
        bound = _library_for_workspace(workspace)
        if bound:
            persist = bound
            print(f"note: library resolved from workspace binding: {persist}",
                  file=sys.stderr)
        else:
            persist = str(Path(LIBRARY_DEFAULT).expanduser())
            print(f"note: no workspace binding for {workspace}; using default "
                  f"library {persist}", file=sys.stderr)
    if library_kind(persist) is None:
        print(f"error: not a hippocampus library (no hippocampus.db / "
              f"memories.json): {persist}")
        return 2
    if args.full and is_user_library(persist):
        print("error: --full is refused on the user library "
              "(privacy: zero content in every channel)")
        return 2
    if args.html and not args.out:
        print("error: --html writes a file containing library content, "
              "so an explicit --out path is required")
        return 2

    try:
        snapshot = snapshot_library(persist)
        states = ("active",) if args.state == "active" else (
            "active", "superseded", "archived")
        agg = aggregate(
            snapshot,
            include_states=states,
            full=bool(args.full),
            limit=args.limit,
        )
        if args.html:
            html = render_html(agg, title=f"Library tree — {os.path.basename(persist)}")
            out = str(Path(args.out).expanduser())
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(html)
            size_kb = os.path.getsize(out) / 1024.0
            print(f"written {out} ({size_kb:.1f} KB) — this file contains "
                  f"library content, store it accordingly")
            print(f"entries included: {agg['health']['included_total']} "
                  f"(states: {', '.join(agg['health']['include_states'])})")
            return 0
        markdown = render_markdown(agg)
        if args.out:
            out = str(Path(args.out).expanduser())
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(markdown + "\n")
            print(f"written {out} (contains library content)")
        else:
            print(markdown)
        return 0
    except (LibraryNotFoundError, TreePrivacyError) as exc:
        print(f"error: {exc}")
        return 2


def cmd_stop(args) -> int:
    """Cooperative stop: write a marker and wait for a graceful, saved exit.

    Killing the process directly would discard the in-memory tail (see the
    step-based autosave window).  The daemon picks the marker up in its
    maintenance thread and runs its normal ``stop()`` path instead.
    """
    from . import launcher
    key = _cli_daemon_key()
    info = launcher.read_registry(key)
    if not info:
        print(f"No running daemon for key={key}.")
        _clear_stop_marker(key)
        return 0
    pid = int(info["pid"])
    marker = launcher.stop_request_path(key)
    try:
        os.makedirs(launcher.runtime_dir(), exist_ok=True)
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"pid": pid, "ts": time.time(), "reason": "cli stop"}, f)
    except OSError as exc:
        print(f"Failed to write stop marker: {exc}", file=sys.stderr)
        return 1
    timeout = float(getattr(args, "timeout", 30) or 30)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not launcher._pid_alive(pid) or launcher.read_registry(key) is None:
            print(f"Daemon pid={pid} stopped gracefully (state saved).")
            _clear_stop_marker(key)
            return 0
        time.sleep(0.5)
    print(f"Daemon pid={pid} did not exit within {timeout:.0f}s; marker left at "
          f"{marker}. Check the daemon log before retrying.", file=sys.stderr)
    return 1


def cmd_restart(args) -> int:
    """Stop the old daemon, then start a fresh one with current code/config."""
    from . import launcher
    rc = cmd_stop(args)
    if rc:
        return rc
    key = _cli_daemon_key()
    # A daemon killed before it could clean up leaves the marker behind; a new
    # daemon would read it and exit immediately.
    _clear_stop_marker(key)
    library = os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", "")
    model = os.environ.get("HIPPOCAMPUS_EMB_MODEL", "")
    info = launcher.ensure_daemon(key, library=library, model=model)
    if not info:
        print("Failed to start a new daemon; the next host session will spawn one.",
              file=sys.stderr)
        return 1
    print(f"New daemon pid={info.get('pid')} port={info.get('port')} key={key}.")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hippocampus-mcp",
        description="HippoMCP memory service: serve, install into hosts, and diagnose.")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("serve", help="Run the MCP service (shared daemon via launcher by default)")
    sp.add_argument("--direct", action="store_true",
                    help="Fall back to the direct server (one model per session; not recommended)")
    sp.set_defaults(func=cmd_serve)

    su = sub.add_parser("setup",
                        help="Create or merge MCP configuration for one host "
                             "(backs up before writing, verifies afterward)")
    su.add_argument("--host", default=None, choices=sorted(_HOSTS),
                    help="Target host; when omitted, infer from workspace markers "
                         "(.zcode/ -> zcode, .mcp.json -> workbuddy)")
    su.add_argument("--scope", choices=("user", "project"), default=None,
                    help="User-level or workspace-level scope; defaults follow host convention")
    su.add_argument("--workspace", default=".",
                    help="Workspace path for project-level configuration; defaults to current directory")
    su.add_argument("--file", default="",
                    help="Explicit configuration file path (overrides --scope inference)")
    su.add_argument("--library", default=None,
                    help="Memory library directory; reuse existing config when present, "
                         f"otherwise use {LIBRARY_DEFAULT}")
    su.add_argument("--model", default=None,
                    help="Embedding model; reuse existing config when present, "
                         f"otherwise use {MODEL_DEFAULT}")
    su.add_argument("--profile", default=None,
                    help="Tool profile; reuse existing config when present, "
                         f"otherwise use {PROFILE_DEFAULT}")
    su.add_argument("--python", default="",
                    help="Absolute interpreter path; defaults to current sys.executable")
    su.add_argument("--name", default=SERVER_NAME_DEFAULT,
                    help=f"Server entry name in the host config; defaults to {SERVER_NAME_DEFAULT}")
    su.add_argument("--offline", action="store_true",
                    help="Set HF_HUB_OFFLINE=1 (use only when the model is already downloaded; "
                         "do not use on first install)")
    su.add_argument("--dry-run", action="store_true",
                    help="Preview the config that would be written; do not write it")
    su.add_argument("--no-verify", action="store_true",
                    help="Skip post-write verification (by default performs a real handshake)")
    su.set_defaults(func=cmd_setup)

    sy = sub.add_parser(
        "sync",
        help="Discover host configs and sync the same entry (preview by default, write with --yes)")
    sy.add_argument("--workspace", default=".",
                    help="Workspace path to scan for project-level configs; defaults to current directory")
    sy.add_argument("--library", default=None,
                    help="Memory library directory; inherit per file when present, "
                         f"otherwise use {LIBRARY_DEFAULT}")
    sy.add_argument("--model", default=None,
                    help="Embedding model; inherit per file when present, "
                         f"otherwise use {MODEL_DEFAULT}")
    sy.add_argument("--profile", default=None,
                    help="Tool profile; inherit per file when present, "
                         f"otherwise use {PROFILE_DEFAULT}")
    sy.add_argument("--python", default="",
                    help="Absolute interpreter path; defaults to current sys.executable")
    sy.add_argument("--name", default=SERVER_NAME_DEFAULT,
                    help=f"Server entry name in the host config; defaults to {SERVER_NAME_DEFAULT}")
    sy.add_argument("--offline", action="store_true",
                    help="Set HF_HUB_OFFLINE=1 (use only when the model is already downloaded; "
                         "do not use on first install)")
    sy.add_argument("--yes", action="store_true",
                    help="Actually write (default is preview only)")
    sy.add_argument("--no-verify", action="store_true",
                    help="Skip post-write verification (by default performs a real handshake)")
    sy.set_defaults(func=cmd_sync)

    sd = sub.add_parser(
        "doctor",
        help="Check paths, libraries, models, dependencies, and host/workspace binding")
    sd.add_argument("--host", default=None, choices=sorted(_HOSTS),
                    help="Add a host/workspace verdict: which config this host reads, "
                         "whether it exists, and what is missing")
    sd.add_argument("--project", default=".",
                    help="Workspace used for host/workspace verdicts; defaults to current directory")
    sd.set_defaults(func=cmd_doctor)

    si = sub.add_parser(
        "install",
        help="Idempotent ZCode deployment: config entry + versioned hook "
             "shims + hooks block (+ optional user skill). Reinstall = repair.")
    si.add_argument("--dry-run", action="store_true",
                    help="Print the action plan without writing anything")
    si.add_argument("--force", action="store_true",
                    help="Overwrite an existing mcp.servers.hippocampus entry "
                         "(default: leave an existing entry untouched)")
    si.add_argument("--no-skill", action="store_true",
                    help="Skip copying the hippo-memory user skill")
    si.set_defaults(func=cmd_install)

    sm = sub.add_parser(
        "models", help="Inspect local model cache and reclaim unused models (no deletion by default)")
    sm.add_argument("--prune", action="store_true",
                    help="List reclaimable models (still does not delete; add --yes to execute)")
    sm.add_argument("--yes", action="store_true",
                    help="Actually delete (requires --prune or --drop)")
    sm.add_argument("--drop", action="append", default=[], metavar="MODEL",
                    help="Explicit model id to delete, such as all-MiniLM-L6-v2 "
                         "(models referenced by a library are refused)")
    sm.add_argument("--cache", default="",
                    help="Explicit model cache root directory (auto-detected by default)")
    sm.add_argument("--fetch", nargs="?", const=True, default=None, metavar="MODEL",
                    help="Prefetch an embedding model into the local cache (default model when omitted). "
                         "Run once during installation; a failed first-start download can leave the host "
                         "with no visible MCP tools")
    sm.add_argument("--json", action="store_true", help="Output JSON for scripting")
    sm.set_defaults(func=cmd_models)

    cx = sub.add_parser(
        "context",
        help="Preview hook injection by running one query through hook_client (does not start daemon)")
    cx.add_argument("query", nargs="?", default="",
                    help="Simulated user prompt (used as the retrieval query)")
    cx.add_argument("--cwd", default=".",
                    help="Workspace root used for library reverse lookup; defaults to current directory")
    cx.add_argument("--top-k", type=int, default=3, help="Number of hits (default 3)")
    cx.add_argument("--budget", type=int, default=600,
                    help="Injection character budget (default 600)")
    cx.add_argument("--item-chars", type=int, default=150,
                    help="Per-item truncation length (default 150; 0 disables truncation)")
    cx.add_argument("--model", default="",
                    help="Embedding model component of the daemon key (defaults to env "
                         "HIPPOCAMPUS_EMB_MODEL or the built-in default)")
    cx.set_defaults(func=cmd_context)

    bs = sub.add_parser(
        "bootstrap",
        help="SessionStart hook preview: read-only opening anchor (pins / "
             "todos / recent sessions / topics; no session created)")
    bs.add_argument("--cwd", default=".",
                    help="Workspace root used for library reverse lookup")
    bs.add_argument("--top-k", type=int, default=4)
    bs.add_argument("--budget", type=int, default=1600)
    bs.add_argument("--item-chars", type=int, default=200)
    bs.add_argument("--compact", action="store_true",
                    help="Also show the compaction reminder line the shim "
                         "appends on source=compact")
    bs.set_defaults(func=cmd_bootstrap)

    tr = sub.add_parser(
        "tree",
        help="Read-only library tree (markdown, or single-file HTML viewer "
             "for the browser)")
    tr.add_argument("--persist-dir", default=None,
                    help="Library directory. Omit to resolve the bound library "
                         "from the workspace (see --project); falls back to "
                         + LIBRARY_DEFAULT)
    tr.add_argument("--project", default=None,
                    help="Workspace root for library reverse lookup via the "
                         "user task directory (default: current directory)")
    tr.add_argument("--state", choices=("active", "all"), default="active",
                    help="Include only active entries, or all states "
                         "(default: active)")
    tr.add_argument("--full", action="store_true",
                    help="Include whole content instead of the 60-char "
                         "preview (refused on the user library)")
    tr.add_argument("--html", action="store_true",
                    help="Emit a self-contained single-file HTML viewer "
                         "instead of markdown (requires --out)")
    tr.add_argument("--out", default=None,
                    help="Write the output to this file instead of stdout "
                         "(required for --html)")
    tr.add_argument("--limit", type=int, default=200,
                    help="Max entries rendered in the markdown tree "
                         "(default 200; counts stay complete)")
    tr.set_defaults(func=cmd_tree)

    sp_stop = sub.add_parser(
        "stop", help="Gracefully stop the shared daemon (saves state before exit)")
    sp_stop.add_argument("--timeout", type=float, default=30.0,
                         help="Seconds to wait for a graceful exit (default 30)")
    sp_stop.set_defaults(func=cmd_stop)

    sp_restart = sub.add_parser(
        "restart", help="Stop the stale daemon and start one with current code/config")
    sp_restart.add_argument("--timeout", type=float, default=30.0,
                            help="Seconds to wait for the old daemon to exit (default 30)")
    sp_restart.set_defaults(func=cmd_restart)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # No subcommand = start the service (same as the old entry point's
        # behavior, but now via the launcher)
        return cmd_serve(argparse.Namespace(direct=False))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
