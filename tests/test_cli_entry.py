# -*- coding: utf-8 -*-
"""CLI entry and wiring self-check: console-script binding / subcommand dispatch / doctor verdicts.

The background is a real onboarding incident: the old installed ``hippocampus-mcp.exe``
entry point **silently swallowed** subcommands (``doctor`` / ``setup`` became "start the
server and exit 0"), so nobody ever got an error about "the config was actually wrong" —
the user only saw ``mcp__hippocampus__*`` tools missing in the host, and an unloaded MCP
server is physically unable to speak for itself. This suite guards that chain's **discoverability**.

Coverage:
    A console-script entry binds cli:main (source check + mutation negative control)
    B subcommand dispatch: doctor reports / unknown subcommand non-zero / not swallowed into startup
    C workspace-marker detection when setup runs without --host (incl. "no guessing")
    D verdict-style doctor --host × --project (what the host reads, what is missing)
    E setup verify-after-write: real subprocess handshake, failure must carry a reason and exit non-zero
    F launcher preflight (library-dir writability fail-fast)
    G doctor self-report: which library this run checks / which file each config line comes from

Discipline:
  · Falsifiable assertions — stub validity is pinned by negative controls
    (B6 / A4 / D6), else "not called" may just be a dead stub;
  · Behavior-level first: group E spawns a real subprocess, no mocked handshake;
  · ``preflight_persist_dir``'s return value is judged with ``is None``, not
    falsiness — ``0`` is falsy too and 0 means "started successfully", the opposite.

Usage:
    python tests/test_cli_entry.py
"""
from __future__ import annotations

import io
import json
import ntpath
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import tomllib  # requires-python >= 3.11, guaranteed in the stdlib

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hippocampus_mcp import cli  # noqa: E402
from hippocampus_mcp import launcher as L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT = os.path.join(ROOT, "pyproject.toml")

_PASS = 0
_FAIL = 0
_FAILED: list[tuple[str, str]] = []


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILED.append((name, detail))


def run_cli(argv, env=None):
    """Run ``cli.main(argv)``; return (return code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    old = dict(os.environ)
    try:
        if env is not None:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(argv)
    except SystemExit as exc:          # argparse's error() takes this path
        rc = exc.code
    finally:
        os.environ.clear()
        os.environ.update(old)
    return rc, out.getvalue(), err.getvalue()


def tmpdir():
    return tempfile.mkdtemp(prefix="hippo-cli-test-")


# ======================================================================
# A group: console-script entry binding (guards against reverting to server:main)
# ======================================================================
def _scripts_of(text: str) -> dict:
    return tomllib.loads(text).get("project", {}).get("scripts", {})


def _binds_cli(text: str) -> bool:
    """Whether the entry binds the CLI dispatcher (asserting the bound value itself is the caller's job)."""
    return _scripts_of(text).get("hippocampus-mcp") == "hippocampus_mcp.cli:main"


def test_a_entry_binding():
    with open(PYPROJECT, "r", encoding="utf-8") as f:
        text = f.read()
    scripts = _scripts_of(text)
    check("A1 console-script binds cli:main",
          scripts.get("hippocampus-mcp") == "hippocampus_mcp.cli:main",
          str(scripts))
    check("A2 no entry points at the server module",
          not any("hippocampus_mcp.server" in str(v) for v in scripts.values()),
          str(scripts))
    check("A3 binding target resolvable and callable",
          callable(getattr(cli, "main", None)), str(getattr(cli, "main", None)))
    # Negative control: revert the binding to server:main; A1's check must fail — otherwise A1 is vacuously true.
    mutated = text.replace("hippocampus_mcp.cli:main",
                           "hippocampus_mcp.server:main")
    check("A4 negative control: reverted server:main must fail",
          (not _binds_cli(mutated)) and mutated != text,
          "变异没生效或判据恒真")


# ======================================================================
# B group: subcommand dispatch (core regression: subcommands silently swallowed)
# ======================================================================
def test_b_dispatch():
    # doctor: must print its report and must **not** start the server
    real_serve = cli.cmd_serve
    calls: list[object] = []
    cli.cmd_serve = lambda args: (calls.append(args), 0)[1]
    try:
        rc, out, _err = run_cli(["doctor"])
        check("B1 doctor exit code 0", rc == 0, str(rc))
        check("B2 doctor prints its report",
              "hippocampus-mcp doctor" in out and "[ OK ]" in out
              and "Libraries configured by host" in out, out[:200])
        check("B3 doctor does not start the server (subcommand not swallowed)", not calls, str(calls))

        ws = tmpdir()
        rc2, out2, _err2 = run_cli(
            ["setup", "--host", "zcode", "--dry-run", "--workspace", ws])
        check("B4 setup --dry-run exit code 0", rc2 == 0, str(rc2))
        check("B5 setup does not start the server", not calls, str(calls))
        check("B6 setup preview content lands", "dry-run" in out2 and "zcode" in out2,
              out2[:200])
    finally:
        cli.cmd_serve = real_serve

    # The verdict must also be visible after dispatch through ``main()`` — testing only
    # _doctor_host_project would miss "dispatch not wired" (the old entry broke at that layer).
    ws2 = tmpdir()
    os.makedirs(os.path.join(ws2, ".zcode"))
    rc_h, out_h, _err_h = run_cli(["doctor", "--host", "zcode",
                                   "--project", ws2])
    check("B12 doctor --host prints the verdict via main too",
          rc_h == 0 and "Host/workspace verdict" in out_h, out_h[:200])
    # Compare directory names only: resolve() may rewrite drive case/separators; full-string compare would false-fail
    check("B13 verdict names the target directory", os.path.basename(ws2) in out_h, out_h[:300])

    # setup without --host: end-to-end auto-detection
    rc_a, out_a, _err_a = run_cli(["setup", "--dry-run", "--workspace", ws2])
    check("B14 without --host auto-detects and says so",
          rc_a == 0 and "inferred from workspace markers" in out_a and "zcode" in out_a, out_a[:300])

    # Unknown subcommand: must error out with a **non-zero** exit (the old entry's disease was a silent exit 0)
    rc3, _out3, err3 = run_cli(["definitely-not-a-subcommand"])
    check("B7 unknown subcommand exits non-zero", isinstance(rc3, int) and rc3 != 0, str(rc3))
    check("B8 unknown subcommand prints an error", "invalid choice" in err3, err3[:200])

    # No subcommand = start the server (old entry point behavior must be kept)
    calls2: list[object] = []
    real_serve2 = cli.cmd_serve
    cli.cmd_serve = lambda args: (calls2.append(args), 0)[1]
    try:
        rc4, _o4, _e4 = run_cli([])
        check("B9 no subcommand still starts the server", len(calls2) == 1 and rc4 == 0,
              f"calls={calls2} rc={rc4}")
        check("B10 default goes through the launcher (direct=False)",
              len(calls2) == 1 and getattr(calls2[0], "direct", None) is False,
              str(calls2))
        # Negative control: the same stub does record calls in a scenario where it
        # SHOULD be called — otherwise B3/B5's "not called" is just a dead stub's illusion.
        check("B11 negative control: stub itself works (B9 scenario recorded the call)", len(calls2) == 1,
              str(calls2))
    finally:
        cli.cmd_serve = real_serve2


# ======================================================================
# C group: workspace-marker detection when setup runs without --host
# ======================================================================
def test_c_detect_host():
    ws = tmpdir()
    try:
        cli._detect_host(Path(ws))
        check("C1 must error when undecidable (no guessing)", False, "没有抛 SystemExit")
    except SystemExit:
        check("C1 must error when undecidable (no guessing)", True)
    except Exception as exc:  # noqa: BLE001
        check("C1 must error when undecidable (no guessing)", False,
              f"{type(exc).__name__}: {exc}")

    ws2 = tmpdir()
    os.makedirs(os.path.join(ws2, ".zcode"))
    check("C2 .zcode/ present -> zcode", cli._detect_host(Path(ws2)) == "zcode",
          cli._detect_host(Path(ws2)))

    ws3 = tmpdir()
    Path(ws3, ".mcp.json").write_text("{}", encoding="utf-8")
    check("C3 .mcp.json present -> workbuddy",
          cli._detect_host(Path(ws3)) == "workbuddy",
          cli._detect_host(Path(ws3)))

    # Both present: the precedence order must be pinned, otherwise results flip across machines
    ws4 = tmpdir()
    os.makedirs(os.path.join(ws4, ".zcode"))
    Path(ws4, ".mcp.json").write_text("{}", encoding="utf-8")
    check("C4 both present -> zcode wins",
          cli._detect_host(Path(ws4)) == "zcode", cli._detect_host(Path(ws4)))

    # Negative control: a same-named **file** does not count as a host marker (is_dir semantics)
    ws5 = tmpdir()
    Path(ws5, ".zcode").write_text("{}", encoding="utf-8")
    try:
        cli._detect_host(Path(ws5))
        check("C5 negative control: .zcode as a file does not count", False, "被当成目录了")
    except SystemExit:
        check("C5 negative control: .zcode as a file does not count", True)
    except Exception as exc:  # noqa: BLE001
        check("C5 negative control: .zcode as a file does not count", False,
              f"{type(exc).__name__}: {exc}")


# ======================================================================
# D group: verdict-style doctor --host × --project
# ======================================================================
def test_d_host_project():
    ws = Path(tmpdir())
    real_user_path = cli.user_config_path
    # Isolate the host user-level path into tmp: on dev machines ~/.workbuddy/mcp.json is
    # real; unisolated, the verdict mixes in local state and the suite fails elsewhere.
    cli.user_config_path = lambda host: Path(tmpdir()) / f"{host}-user.json"
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            cli._doctor_host_project("zcode", ws)
        text = out.getvalue()
        check("D1 names which config this host reads",
              ".zcode" in text and "config.json" in text, text[:300])
        check("D2 missing state is explicit", "Missing (not wired up here)" in text, text[:300])

        # Write a correct ZCode config
        cfgdir = ws / ".zcode"
        cfgdir.mkdir(parents=True, exist_ok=True)
        (cfgdir / "config.json").write_text(json.dumps({
            "mcp": {"servers": {"hippocampus": {
                "command": "python",
                "env": {"HIPPOCAMPUS_PERSIST_DIR": "/lib/expected"}}}}
        }), encoding="utf-8")
        out2 = io.StringIO()
        with redirect_stdout(out2):
            cli._doctor_host_project("zcode", ws)
        t2 = out2.getvalue()
        check("D3 existing entry reports library", "library=/lib/expected" in t2, t2[:400])

        # The incident's **direct detection point**: a config in the same directory that the host does not read
        (ws / ".mcp.json").write_text(json.dumps(
            {"mcpServers": {"hippocampus": {"command": "python"}}}),
            encoding="utf-8")
        out3 = io.StringIO()
        with redirect_stdout(out3):
            cli._doctor_host_project("zcode", ws)
        t3 = out3.getvalue()
        check("D4 names the 'exists but ZCode does not read it' state",
              "does not read it" in t3 and ".mcp.json" in t3, t3[:400])

        # Broken config: must not crash, must still give a verdict
        (cfgdir / "config.json").write_text("{ 这不是 JSON", encoding="utf-8")
        out4 = io.StringIO()
        try:
            with redirect_stdout(out4):
                cli._doctor_host_project("zcode", ws)
            check("D5 bad config gets verdict without crashing", "is not valid JSON" in out4.getvalue(),
                  out4.getvalue()[:400])
        except Exception as exc:  # noqa: BLE001
            check("D5 broken config gives a verdict without crashing", False, f"{type(exc).__name__}: {exc}")

        # An entry exists but it is not ours
        (cfgdir / "config.json").write_text(json.dumps(
            {"mcp": {"servers": {"other": {}}}}), encoding="utf-8")
        out5 = io.StringIO()
        with redirect_stdout(out5):
            cli._doctor_host_project("zcode", ws)
        check("D6 no our-entry is not wired up", "not wired up" in out5.getvalue(),
              out5.getvalue()[:400])

        # Negative control: the states must produce **distinct** verdicts
        v_missing = cli._entry_verdict(ws / "nope.json", cli._HOSTS["zcode"])
        v_broken = cli._entry_verdict(cfgdir / "config.json",
                                      cli._HOSTS["zcode"])
        check("D7 negative control: missing/no-entry/broken verdicts all differ",
              len({v_missing, v_broken}) == 2, f"{v_missing} | {v_broken}")
    finally:
        cli.user_config_path = real_user_path


# ======================================================================
# E group: setup verify-after-write (real subprocess handshake)
# ======================================================================
#: Fake MCP server: read one line -> reply one line (verifies only the handshake, loads no models)
_FAKE_SRV = r"""
import json, sys
sys.stdin.readline()
print(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}), flush=True)
sys.stdin.readline()
sys.stdin.readline()
tools = [{"name": "a"}, {"name": "b"}] if __import__("os").environ.get("FAKE_TOOLS", "1") == "1" else []
print(json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"tools": tools}}), flush=True)
"""
_FAKE_ERR = r"""
import json, sys
sys.stdin.readline()
print(json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "boom"}}), flush=True)
"""
_FAKE_SILENT = "import sys\nsys.stdin.readline()\n"


def _entry(command, args, env=None):
    return {"command": command, "args": args, "env": env or {}}


def test_e_verify():
    py = sys.executable
    ok, detail = cli._verify_entry(
        _entry(py, ["-c", _FAKE_SRV]), timeout=20.0)
    check("E1 handshake succeeds", ok is True, detail)
    check("E2 success detail includes tool count", "2 tools visible" in detail, detail)

    ok2, d2 = cli._verify_entry(
        _entry(os.path.join(tmpdir(), "no-such-python.exe"), ["-m", "x"]),
        timeout=20.0)
    check("E3 process cannot start includes reason", ok2 is False and "Cannot start process" in d2, d2)

    ok3, d3 = cli._verify_entry(_entry(py, ["-c", _FAKE_SILENT]), timeout=20.0)
    check("E4 subprocess exits early -> failure carries the reason",
          ok3 is False and "before completing handshake" in d3, d3)

    ok4, d4 = cli._verify_entry(_entry(py, ["-c", _FAKE_ERR]), timeout=20.0)
    check("E5 error response includes reason", ok4 is False and "returned error" in d4, d4)

    # Negative control: handshake OK but 0 tools — must not silently pass as "verified"
    ok5, d5 = cli._verify_entry(
        _entry(py, ["-c", _FAKE_SRV], env={"FAKE_TOOLS": "0"}), timeout=20.0)
    check("E6 negative control names zero tools", ok5 is True and "0 tools" in d5, d5)

    # setup exit: verification failure must exit non-zero (no "written so it worked")
    ws = tmpdir()
    cfg = os.path.join(ws, "cfg.json")
    real_verify, calls = cli._verify_entry, []
    cli._verify_entry = lambda *a, **k: (calls.append(a), (False, "假失败"))[1]
    try:
        rc, out, _err = run_cli(["setup", "--host", "zcode", "--workspace", ws,
                                 "--file", cfg])
        check("E7 verify-after-write fails -> setup exits non-zero", rc == 1, str(rc))
        check("E8 failure is reported on stdout", "handshake verification failed" in out, out[-300:])
        check("E9 verification was actually called once", len(calls) == 1, str(calls))
    finally:
        cli._verify_entry = real_verify

    # --no-verify must really skip (being switchable is the discipline)
    calls2: list[object] = []
    cli._verify_entry = lambda *a, **k: (calls2.append(a), (True, "ok"))[1]
    try:
        rc2, _o2, _e2 = run_cli(["setup", "--host", "zcode", "--workspace", ws,
                                 "--file", cfg, "--no-verify"])
        check("E10 --no-verify skips verification and still succeeds", rc2 == 0 and not calls2,
              f"rc={rc2} calls={calls2}")
    finally:
        cli._verify_entry = real_verify

    # Negative control: run it for real (E1's fake script is a genuine subprocess, not a mock fallback)
    p = subprocess.run([py, "-c", _FAKE_SRV], input='{"id":1}\n{"x"}\n{"id":2}\n',
                       capture_output=True, text=True, timeout=30)
    check("E11 negative control: fake server works on its own (real process)",
          p.returncode == 0 and '"id": 2' in (p.stdout or ""),
          f"rc={p.returncode} out={p.stdout[:120]}")


# ======================================================================
# F group: launcher preflight (library-dir writability fail-fast)
# ======================================================================
def test_f_preflight():
    old = os.environ.get(L.ENV_PERSIST)
    try:
        os.environ.pop(L.ENV_PERSIST, None)
        err = io.StringIO()
        with redirect_stderr(err):
            rc = L.preflight_persist_dir()
        check("F1 library dir unset -> pass through", rc is None, str(rc))
        check("F2 self-reports when unset (a cross-library leak never errors)",
              "not set" in err.getvalue() and "doctor" in err.getvalue(),
              err.getvalue()[:200])

        ws = tmpdir()
        os.environ[L.ENV_PERSIST] = ws
        err2 = io.StringIO()
        with redirect_stderr(err2):
            rc2 = L.preflight_persist_dir()
        check("F3 writable dir -> pass through", rc2 is None, str(rc2))
        check("F4 probe file leaves no residue",
              not os.path.exists(os.path.join(ws, ".hippo-write-probe")),
              str(os.listdir(ws)))

        # Unwritable: use an existing **file** as the directory name (more reliable than
        # permission tricks — under admin, "read-only directories" may still be writable)
        blocker = os.path.join(tmpdir(), "i-am-a-file")
        Path(blocker).write_text("x", encoding="utf-8")
        os.environ[L.ENV_PERSIST] = blocker
        err3 = io.StringIO()
        with redirect_stderr(err3):
            rc3 = L.preflight_persist_dir()
        e3 = err3.getvalue()
        check("F5 unwritable -> refuse to start", rc3 == 2, str(rc3))
        check("F6 refusal carries a reason and fix guidance",
              "refusing to start" in e3 and L.ENV_PERSIST in e3, e3[:300])
        # Negative control: the return value must not be judged by falsiness — 0 means "started successfully", the opposite
        check("F7 negative control: pass-through is None, not 0",
              rc2 is None and rc3 != 0 and bool(rc3) is True,
              f"ok={rc2!r} fail={rc3!r}")
    finally:
        if old is None:
            os.environ.pop(L.ENV_PERSIST, None)
        else:
            os.environ[L.ENV_PERSIST] = old


# ======================================================================
# G group: doctor self-report (which library this run checks / which file each config comes from)
# ======================================================================
def test_g_doctor_selfreport():
    real_user_path = cli.user_config_path
    udir = Path(tmpdir())
    upath = udir / "user-mcp.json"
    upath.write_text(json.dumps({"mcpServers": {"hippocampus": {
        "command": "python", "env": {"HIPPOCAMPUS_PERSIST_DIR": "/lib/from-user"}}}}),
        encoding="utf-8")
    cli.user_config_path = lambda host: upath if host == "workbuddy" else None
    try:
        rc, out, _err = run_cli(["doctor"], env={"HIPPOCAMPUS_PERSIST_DIR": None})
        check("G1 doctor exit code 0", rc == 0, str(rc))
        check("G2 bare run self-reports 'this checks the default library'",
              "default library" in out and "not a library from any host config" in out, out[:600])
        check("G3 default model is self-reported",
              "this checks the default model" in out, out[:600])

        rc2, out2, _e2 = run_cli(
            ["doctor"], env={"HIPPOCAMPUS_PERSIST_DIR": "/lib/explicit"})
        check("G4 with a library set, no default-library self-report",
              "不是任何宿主配置里的库" not in out2, out2[:600])
        # Negative control: the self-report follows env only, not config content
        check("G5 negative control: self-report driven by env (G2/G4 outcomes opposite)",
              ("not a library from any host config" in out)
              != ("not a library from any host config" in out2), "self-report did not follow env")

        # Every library-matrix row must carry its config file path — after spotting a problem the next step is always "which file to fix"
        check("G6 library-matrix rows carry the config source path",
              f"<{upath}>" in out2 and "library=/lib/from-user" in out2, out2[-600:])
    finally:
        cli.user_config_path = real_user_path


# ======================================================================
# I group: doctor drive-letter / size check (field feedback 2026-10-31; the
# relocate scenario is spelled out in the check output — deliberately no relocate command)
# ======================================================================
_GB = 1024 ** 3


def test_i_disk_hint():
    # This group tests Windows drive-letter semantics (C:\ vs c:). Linux's
    # os.path.splitdrive always returns an empty drive, so the pure function returns
    # None without a drive — not a code regression, the platform lacks the concept
    # (the source of CI's false failures on I2/I3/I4). Non-Windows: temporarily swap
    # splitdrive to ntpath semantics so the 5 assertions stay sharp on Linux; production code untouched.
    _drive_ctx = (mock.patch.object(os.path, "splitdrive", ntpath.splitdrive)
                  if os.name != "nt" else nullcontext())
    with _drive_ctx:
        check("I1 non-system drive -> no nagging",
              cli._system_disk_migration_hint(Path("D:/libs/user"), 5 * _GB,
                                              "C:\\") is None)
        check("I2 system drive + small size -> one-line 'no move needed yet'",
              (h := cli._system_disk_migration_hint(Path("C:/libs/user"),
                                                    0.3 * _GB, "C:\\")) is not None
              and "no move needed yet" in h and "Migration steps" not in h, str(h)[:200])
        big = cli._system_disk_migration_hint(Path("C:/libs/user"), 2.4 * _GB,
                                              "C:\\")
        check("I3 system drive + large size -> three migration steps + env and verify command pointers",
              big is not None and "⚠" in big and "Migration steps" in big
              and "HIPPOCAMPUS_PERSIST_DIR" in big
              and "library_whoami" in big, str(big)[:200])
        check("I4 drive-letter case-insensitive",
              cli._system_disk_migration_hint(Path("c:/libs/user"), 2 * _GB,
                                              "C:\\") is not None)
        check("I5 empty system_drive -> undecidable, no warning",
              cli._system_disk_migration_hint(Path("C:/libs/user"), 2 * _GB,
                                              "") is None)


# ======================================================================
# H group: ZCode user-level config path (regression pin for the 2026-10-30 blind-spot incident)
# ======================================================================
def test_h_zcode_user_config_path():
    """user_config_path("zcode") must point at ~/.zcode/cli/config.json.

    Background: a real incident where a ZCode upgrade moved the user-level config from
    ~/.zcode/config.json into cli/ while this function returned None for zcode — doctor/sync
    went blind to it and 9-28 entries re-added at the old spot were lost undetected (migration
    design doc §1). Pinned: the mapping exists and doctor's host-library reconciliation reads it.
    """
    got = cli.user_config_path("zcode")
    check("H1 zcode user-level path = ~/.zcode/cli/config.json",
          got is not None
          and got.parts[-3:] == (".zcode", "cli", "config.json"),
          str(got))
    # Negative control: unknown host still returns None (no inventing paths that do not exist)
    check("H2 unknown host returns None", cli.user_config_path("nonexistent") is None,
          str(cli.user_config_path("nonexistent")))
    # End-to-end: put a user-level config in a temp home; doctor's host-library reconciliation must see it
    real_home = Path.home
    fake_home = Path(tmpdir())
    (fake_home / ".zcode" / "cli").mkdir(parents=True)
    (fake_home / ".zcode" / "cli" / "config.json").write_text(json.dumps({
        "mcp": {"servers": {"hippocampus": {
            "command": "python",
            "env": {"HIPPOCAMPUS_PERSIST_DIR": "/lib/user-level"}}}}}),
        encoding="utf-8")
    Path.home = lambda: fake_home
    try:
        # Key-shape consistency: _HOSTS["zcode"]'s key_path must really dig our entry
        # out of the user-level file (right path + right key shape = fully discoverable)
        upath = cli.user_config_path("zcode")
        check("H3 user-level file exists", upath.is_file(), str(upath))
        cfg, err = cli.hc.load_json_config(upath)
        container = cli._dig(cfg or {}, cli._HOSTS["zcode"]["key_path"])
        entry = (container or {}).get("hippocampus")
        check("H4 key shape (mcp,servers) resolves our entry",
              not err and isinstance(entry, dict)
              and entry.get("env", {}).get("HIPPOCAMPUS_PERSIST_DIR") == "/lib/user-level",
              f"err={err} entry={str(entry)[:200]}")
    finally:
        Path.home = real_home


def main() -> int:
    for fn in (test_a_entry_binding, test_b_dispatch, test_c_detect_host,
               test_d_host_project, test_e_verify, test_f_preflight,
               test_g_doctor_selfreport, test_h_zcode_user_config_path,
               test_i_disk_hint):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} 抛异常", False, f"{type(exc).__name__}: {exc}")
    print("=" * 70)
    print(f"结果：{_PASS}/{_PASS + _FAIL} 通过")
    if _FAILED:
        print("\n失败明细：")
        for n, d in _FAILED:
            print(f"  - {n}  {d}")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
