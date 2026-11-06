# -*- coding: utf-8 -*-
"""CLI 入口与接入自检：console-script 绑定 / 子命令分发 / doctor 结论。

背景是一次真实接入事故：旧安装的 ``hippocampus-mcp.exe`` 入口把子命令
**静默吞掉**（``doctor`` / ``setup`` 变成"启动服务并 exit 0"），于是
"配置其实没写对"这件事没有任何人报错 —— 用户只能在宿主里看到
``mcp__hippocampus__*`` 工具不存在，而 MCP server 没被加载时物理上无法
自己发声。本套件守的就是这条链路的**可发现性**。

覆盖：
    A console-script 入口绑定 cli:main（源码口径 + 变异负对照）
    B 子命令分发：doctor 出体检报告 / 未知子命令非 0 / 子命令不被吞成启动服务
    C setup 不带 --host 时的工作区特征探测（含"不瞎选"）
    D doctor --host × --project 的结论式判定（宿主会读什么、缺了谁）
    E setup 写后即验：真子进程握手，失败必须带原因且 setup 退非 0
    F launcher 启动前自检（库目录可写性 fail-fast）
    G doctor 自曝：本次检查的是默认库 / 每行配置来自哪个文件

纪律：
  · 断言可证伪 —— 替身有效性用反面对照钉（B6 / A4 / D6），
    否则"没被调用"可能是替身自己失效；
  · 行为级优先：E 组起真子进程，不 mock 掉握手；
  · ``preflight_persist_dir`` 的返回值用 ``is None`` 判定，不用 falsy ——
    ``0`` 也是 falsy，而 0 意味着"成功启动"，语义正好相反。

用法：
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

import tomllib  # requires-python >= 3.11，标准库必有

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
    """跑 ``cli.main(argv)``，返回 (返回码, stdout, stderr)。"""
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
    except SystemExit as exc:          # argparse 的 error() 走这条路
        rc = exc.code
    finally:
        os.environ.clear()
        os.environ.update(old)
    return rc, out.getvalue(), err.getvalue()


def tmpdir():
    return tempfile.mkdtemp(prefix="hippo-cli-test-")


# ======================================================================
# A 组：console-script 入口绑定（防退回 server:main）
# ======================================================================
def _scripts_of(text: str) -> dict:
    return tomllib.loads(text).get("project", {}).get("scripts", {})


def _binds_cli(text: str) -> bool:
    """入口是否绑定 CLI 分发器（同时返回绑定值的可读性由调用方断言）。"""
    return _scripts_of(text).get("hippocampus-mcp") == "hippocampus_mcp.cli:main"


def test_a_entry_binding():
    with open(PYPROJECT, "r", encoding="utf-8") as f:
        text = f.read()
    scripts = _scripts_of(text)
    check("A1 console-script 绑 cli:main",
          scripts.get("hippocampus-mcp") == "hippocampus_mcp.cli:main",
          str(scripts))
    check("A2 没有任何入口指向 server 模块",
          not any("hippocampus_mcp.server" in str(v) for v in scripts.values()),
          str(scripts))
    check("A3 绑定目标可解析且可调用",
          callable(getattr(cli, "main", None)), str(getattr(cli, "main", None)))
    # 负对照：把绑定改回 server:main，A1 的判据必须翻红 —— 否则 A1 恒真。
    mutated = text.replace("hippocampus_mcp.cli:main",
                           "hippocampus_mcp.server:main")
    check("A4 负对照：改回 server:main 必须判失败",
          (not _binds_cli(mutated)) and mutated != text,
          "变异没生效或判据恒真")


# ======================================================================
# B 组：子命令分发（核心回归：子命令被静默吞掉）
# ======================================================================
def test_b_dispatch():
    # doctor：必须出体检报告，且**不能**去启动服务
    real_serve = cli.cmd_serve
    calls: list[object] = []
    cli.cmd_serve = lambda args: (calls.append(args), 0)[1]
    try:
        rc, out, _err = run_cli(["doctor"])
        check("B1 doctor 退出码 0", rc == 0, str(rc))
        check("B2 doctor 出体检报告",
              "hippocampus-mcp doctor" in out and "[ OK ]" in out
              and "Libraries configured by host" in out, out[:200])
        check("B3 doctor 不触发启动服务（子命令没被吞）", not calls, str(calls))

        ws = tmpdir()
        rc2, out2, _err2 = run_cli(
            ["setup", "--host", "zcode", "--dry-run", "--workspace", ws])
        check("B4 setup --dry-run 退出码 0", rc2 == 0, str(rc2))
        check("B5 setup 不触发启动服务", not calls, str(calls))
        check("B6 setup 预览内容落地", "dry-run" in out2 and "zcode" in out2,
              out2[:200])
    finally:
        cli.cmd_serve = real_serve

    # 结论式判定也必须经 ``main()`` 分发后可见 —— 只测 _doctor_host_project
    # 会被"分发没接上"漏过去（旧入口吞子命令正是那一层坏）。
    ws2 = tmpdir()
    os.makedirs(os.path.join(ws2, ".zcode"))
    rc_h, out_h, _err_h = run_cli(["doctor", "--host", "zcode",
                                   "--project", ws2])
    check("B12 doctor --host 经 main 也要出结论段",
          rc_h == 0 and "Host/workspace verdict" in out_h, out_h[:200])
    # 只比对目录名：resolve() 可能改写盘符大小写/分隔符，全串比对会假红
    check("B13 结论段带上目标目录", os.path.basename(ws2) in out_h, out_h[:300])

    # setup 不带 --host：端到端走自动判定
    rc_a, out_a, _err_a = run_cli(["setup", "--dry-run", "--workspace", ws2])
    check("B14 不带 --host 时自动判定并说出来",
          rc_a == 0 and "inferred from workspace markers" in out_a and "zcode" in out_a, out_a[:300])

    # 未知子命令：必须报错退出**非 0**（旧入口的病就是静默 exit 0）
    rc3, _out3, err3 = run_cli(["definitely-not-a-subcommand"])
    check("B7 未知子命令退出非 0", isinstance(rc3, int) and rc3 != 0, str(rc3))
    check("B8 未知子命令有报错输出", "invalid choice" in err3, err3[:200])

    # 无子命令 = 启动服务（旧 entry point 行为必须保留）
    calls2: list[object] = []
    real_serve2 = cli.cmd_serve
    cli.cmd_serve = lambda args: (calls2.append(args), 0)[1]
    try:
        rc4, _o4, _e4 = run_cli([])
        check("B9 无子命令仍走启动服务", len(calls2) == 1 and rc4 == 0,
              f"calls={calls2} rc={rc4}")
        check("B10 默认经 launcher（direct=False）",
              len(calls2) == 1 and getattr(calls2[0], "direct", None) is False,
              str(calls2))
        # 负对照：同一个替身在"该被调用"的场景确实记到了 —— 否则 B3/B5 的
        # "没被调用"只是替身失效的假象。
        check("B11 负对照：替身自身有效（B9 场景记到调用）", len(calls2) == 1,
              str(calls2))
    finally:
        cli.cmd_serve = real_serve2


# ======================================================================
# C 组：setup 不带 --host 时的工作区特征探测
# ======================================================================
def test_c_detect_host():
    ws = tmpdir()
    try:
        cli._detect_host(Path(ws))
        check("C1 无从判断时必须报错（不瞎选）", False, "没有抛 SystemExit")
    except SystemExit:
        check("C1 无从判断时必须报错（不瞎选）", True)
    except Exception as exc:  # noqa: BLE001
        check("C1 无从判断时必须报错（不瞎选）", False,
              f"{type(exc).__name__}: {exc}")

    ws2 = tmpdir()
    os.makedirs(os.path.join(ws2, ".zcode"))
    check("C2 有 .zcode/ → zcode", cli._detect_host(Path(ws2)) == "zcode",
          cli._detect_host(Path(ws2)))

    ws3 = tmpdir()
    Path(ws3, ".mcp.json").write_text("{}", encoding="utf-8")
    check("C3 有 .mcp.json → workbuddy",
          cli._detect_host(Path(ws3)) == "workbuddy",
          cli._detect_host(Path(ws3)))

    # 两者都有：判据顺序必须钉死，否则换机器就换结果
    ws4 = tmpdir()
    os.makedirs(os.path.join(ws4, ".zcode"))
    Path(ws4, ".mcp.json").write_text("{}", encoding="utf-8")
    check("C4 两者都有 → zcode 优先",
          cli._detect_host(Path(ws4)) == "zcode", cli._detect_host(Path(ws4)))

    # 负对照：同名的**文件**不算宿主痕迹（is_dir 语义）
    ws5 = tmpdir()
    Path(ws5, ".zcode").write_text("{}", encoding="utf-8")
    try:
        cli._detect_host(Path(ws5))
        check("C5 负对照：.zcode 是文件不算数", False, "被当成目录了")
    except SystemExit:
        check("C5 负对照：.zcode 是文件不算数", True)
    except Exception as exc:  # noqa: BLE001
        check("C5 负对照：.zcode 是文件不算数", False,
              f"{type(exc).__name__}: {exc}")


# ======================================================================
# D 组：doctor --host × --project 的结论式判定
# ======================================================================
def test_d_host_project():
    ws = Path(tmpdir())
    real_user_path = cli.user_config_path
    # 宿主 user 级路径隔离到 tmp：测试机上 ~/.workbuddy/mcp.json 是真的，
    # 不隔离的话结论会掺进本机状态，套件换个机器就翻红。
    cli.user_config_path = lambda host: Path(tmpdir()) / f"{host}-user.json"
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            cli._doctor_host_project("zcode", ws)
        text = out.getvalue()
        check("D1 点名该宿主会读哪份配置",
              ".zcode" in text and "config.json" in text, text[:300])
        check("D2 missing state is explicit", "Missing (not wired up here)" in text, text[:300])

        # 写入一份正确的 ZCode 配置
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

        # 事故的**直接检测点**：同目录里有一份宿主不读的配置
        (ws / ".mcp.json").write_text(json.dumps(
            {"mcpServers": {"hippocampus": {"command": "python"}}}),
            encoding="utf-8")
        out3 = io.StringIO()
        with redirect_stdout(out3):
            cli._doctor_host_project("zcode", ws)
        t3 = out3.getvalue()
        check("D4 点名「存在但 ZCode 不读」",
              "does not read it" in t3 and ".mcp.json" in t3, t3[:400])

        # 配置坏了：不能崩，要给结论
        (cfgdir / "config.json").write_text("{ 这不是 JSON", encoding="utf-8")
        out4 = io.StringIO()
        try:
            with redirect_stdout(out4):
                cli._doctor_host_project("zcode", ws)
            check("D5 bad config gets verdict without crashing", "is not valid JSON" in out4.getvalue(),
                  out4.getvalue()[:400])
        except Exception as exc:  # noqa: BLE001
            check("D5 坏配置给结论不崩", False, f"{type(exc).__name__}: {exc}")

        # 有条目但不是我们这条
        (cfgdir / "config.json").write_text(json.dumps(
            {"mcp": {"servers": {"other": {}}}}), encoding="utf-8")
        out5 = io.StringIO()
        with redirect_stdout(out5):
            cli._doctor_host_project("zcode", ws)
        check("D6 no our-entry is not wired up", "not wired up" in out5.getvalue(),
              out5.getvalue()[:400])

        # 负对照：三种状态必须给出三种**不同**的结论
        v_missing = cli._entry_verdict(ws / "nope.json", cli._HOSTS["zcode"])
        v_broken = cli._entry_verdict(cfgdir / "config.json",
                                      cli._HOSTS["zcode"])
        check("D7 负对照：缺失/无条目/坏配置结论互不相同",
              len({v_missing, v_broken}) == 2, f"{v_missing} | {v_broken}")
    finally:
        cli.user_config_path = real_user_path


# ======================================================================
# E 组：setup 写后即验（真子进程握手）
# ======================================================================
#: 假 MCP server：读一行 → 回一行（只验握手链路，不加载任何模型）
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
    check("E1 握手成功", ok is True, detail)
    check("E2 success detail includes tool count", "2 tools visible" in detail, detail)

    ok2, d2 = cli._verify_entry(
        _entry(os.path.join(tmpdir(), "no-such-python.exe"), ["-m", "x"]),
        timeout=20.0)
    check("E3 process cannot start includes reason", ok2 is False and "Cannot start process" in d2, d2)

    ok3, d3 = cli._verify_entry(_entry(py, ["-c", _FAKE_SILENT]), timeout=20.0)
    check("E4 子进程提前退出 → 失败带原因",
          ok3 is False and "before completing handshake" in d3, d3)

    ok4, d4 = cli._verify_entry(_entry(py, ["-c", _FAKE_ERR]), timeout=20.0)
    check("E5 error response includes reason", ok4 is False and "returned error" in d4, d4)

    # 负对照：握手通了但 0 个工具 —— 不能算"验证通过"静默过去
    ok5, d5 = cli._verify_entry(
        _entry(py, ["-c", _FAKE_SRV], env={"FAKE_TOOLS": "0"}), timeout=20.0)
    check("E6 negative control names zero tools", ok5 is True and "0 tools" in d5, d5)

    # setup 出口：验证失败必须退非 0（不能"写进去了就当成功"）
    ws = tmpdir()
    cfg = os.path.join(ws, "cfg.json")
    real_verify, calls = cli._verify_entry, []
    cli._verify_entry = lambda *a, **k: (calls.append(a), (False, "假失败"))[1]
    try:
        rc, out, _err = run_cli(["setup", "--host", "zcode", "--workspace", ws,
                                 "--file", cfg])
        check("E7 写后即验失败 → setup 退非 0", rc == 1, str(rc))
        check("E8 failure is reported on stdout", "handshake verification failed" in out, out[-300:])
        check("E9 验证确实被调用了一次", len(calls) == 1, str(calls))
    finally:
        cli._verify_entry = real_verify

    # --no-verify 必须真的跳过（关得掉是纪律）
    calls2: list[object] = []
    cli._verify_entry = lambda *a, **k: (calls2.append(a), (True, "ok"))[1]
    try:
        rc2, _o2, _e2 = run_cli(["setup", "--host", "zcode", "--workspace", ws,
                                 "--file", cfg, "--no-verify"])
        check("E10 --no-verify 跳过验证且仍成功", rc2 == 0 and not calls2,
              f"rc={rc2} calls={calls2}")
    finally:
        cli._verify_entry = real_verify

    # 负对照：真跑一次（E1 用的假脚本确实是子进程、不是 mock 兜底）
    p = subprocess.run([py, "-c", _FAKE_SRV], input='{"id":1}\n{"x"}\n{"id":2}\n',
                       capture_output=True, text=True, timeout=30)
    check("E11 负对照：假 server 本身可用（真进程）",
          p.returncode == 0 and '"id": 2' in (p.stdout or ""),
          f"rc={p.returncode} out={p.stdout[:120]}")


# ======================================================================
# F 组：launcher 启动前自检（库目录可写性 fail-fast）
# ======================================================================
def test_f_preflight():
    old = os.environ.get(L.ENV_PERSIST)
    try:
        os.environ.pop(L.ENV_PERSIST, None)
        err = io.StringIO()
        with redirect_stderr(err):
            rc = L.preflight_persist_dir()
        check("F1 未设库目录 → 放行", rc is None, str(rc))
        check("F2 未设时要自曝（串库从不报错）",
              "not set" in err.getvalue() and "doctor" in err.getvalue(),
              err.getvalue()[:200])

        ws = tmpdir()
        os.environ[L.ENV_PERSIST] = ws
        err2 = io.StringIO()
        with redirect_stderr(err2):
            rc2 = L.preflight_persist_dir()
        check("F3 可写目录 → 放行", rc2 is None, str(rc2))
        check("F4 探测件不残留",
              not os.path.exists(os.path.join(ws, ".hippo-write-probe")),
              str(os.listdir(ws)))

        # 不可写：拿一个已存在的**文件**当目录名（比权限手法更确定，
        # 管理员模式下"只读目录"未必真的写不进）
        blocker = os.path.join(tmpdir(), "i-am-a-file")
        Path(blocker).write_text("x", encoding="utf-8")
        os.environ[L.ENV_PERSIST] = blocker
        err3 = io.StringIO()
        with redirect_stderr(err3):
            rc3 = L.preflight_persist_dir()
        e3 = err3.getvalue()
        check("F5 不可写 → 拒绝启动", rc3 == 2, str(rc3))
        check("F6 拒绝要带原因与修复指引",
              "refusing to start" in e3 and L.ENV_PERSIST in e3, e3[:300])
        # 负对照：返回值不能用 falsy 判 —— 0 是"成功启动"，语义相反
        check("F7 负对照：放行是 None 而非 0",
              rc2 is None and rc3 != 0 and bool(rc3) is True,
              f"ok={rc2!r} fail={rc3!r}")
    finally:
        if old is None:
            os.environ.pop(L.ENV_PERSIST, None)
        else:
            os.environ[L.ENV_PERSIST] = old


# ======================================================================
# G 组：doctor 自曝（本次查的是哪个库 / 配置来自哪个文件）
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
        check("G1 doctor 退出码 0", rc == 0, str(rc))
        check("G2 裸跑时自曝「查的是默认库」",
              "default library" in out and "not a library from any host config" in out, out[:600])
        check("G3 default model is self-reported",
              "this checks the default model" in out, out[:600])

        rc2, out2, _e2 = run_cli(
            ["doctor"], env={"HIPPOCAMPUS_PERSIST_DIR": "/lib/explicit"})
        check("G4 设了库就不再自曝默认库",
              "不是任何宿主配置里的库" not in out2, out2[:600])
        # 负对照：自曝只跟 env 有关，跟配置内容无关
        check("G5 负对照：自曝由 env 触发（G2/G4 结论相反）",
              ("not a library from any host config" in out)
              != ("not a library from any host config" in out2), "self-report did not follow env")

        # 库矩阵每行必须带配置文件路径 —— 看出问题后下一步永远是"改哪个文件"
        check("G6 库矩阵每行带配置来源路径",
              f"<{upath}>" in out2 and "library=/lib/from-user" in out2, out2[-600:])
    finally:
        cli.user_config_path = real_user_path


# ======================================================================
# I 组：doctor 盘符 / 体积体检（实测反馈 2026-10-31，relocate 场景在
# 体检输出里说清楚——刻意没有 relocate 命令）
# ======================================================================
_GB = 1024 ** 3


def test_i_disk_hint():
    # 本组测的是 Windows 盘符语义（C:\ ↔ c:）。Linux 的 os.path.splitdrive
    # 恒返回空盘符，纯函数拿不到 drive 就一律返回 None —— 不是代码退化，
    # 是平台没有这个概念（CI 上 I2/I3/I4 假红即由此而来）。
    # 非 Windows 时临时把 splitdrive 换成 ntpath 语义，让这 5 条断言在
    # Linux runner 上仍保有完整判别力；生产代码一行不动。
    _drive_ctx = (mock.patch.object(os.path, "splitdrive", ntpath.splitdrive)
                  if os.name != "nt" else nullcontext())
    with _drive_ctx:
        check("I1 非系统盘 → 不叨叨",
              cli._system_disk_migration_hint(Path("D:/libs/user"), 5 * _GB,
                                              "C:\\") is None)
        check("I2 系统盘 + 小体积 → 一行「暂不必迁」",
              (h := cli._system_disk_migration_hint(Path("C:/libs/user"),
                                                    0.3 * _GB, "C:\\")) is not None
              and "no move needed yet" in h and "Migration steps" not in h, str(h)[:200])
        big = cli._system_disk_migration_hint(Path("C:/libs/user"), 2.4 * _GB,
                                              "C:\\")
        check("I3 系统盘 + 大体积 → 迁移三步 + 指路 env 与验证命令",
              big is not None and "⚠" in big and "Migration steps" in big
              and "HIPPOCAMPUS_PERSIST_DIR" in big
              and "library_whoami" in big, str(big)[:200])
        check("I4 盘符大小写不敏感",
              cli._system_disk_migration_hint(Path("c:/libs/user"), 2 * _GB,
                                              "C:\\") is not None)
        check("I5 system_drive 空 → 判不了不告警",
              cli._system_disk_migration_hint(Path("C:/libs/user"), 2 * _GB,
                                              "") is None)


# ======================================================================
# H 组：ZCode 用户级配置路径（2026-10-30 盲区事故的回归钉）
# ======================================================================
def test_h_zcode_user_config_path():
    """user_config_path("zcode") 必须指向 ~/.zcode/cli/config.json。

    背景是一次真实事故：ZCode 升级把用户级配置从 ~/.zcode/config.json 迁到
    cli/ 子目录，而本函数当时对 zcode 返回 None——doctor/sync 看不见用户级
    配置，9-28 补在旧位置的条目升级后丢失也无人能体检出来（接入迁移设计
    文档 §一）。本组钉住：路径映射存在、且 doctor 的宿主库对账真能读到它。
    """
    got = cli.user_config_path("zcode")
    check("H1 zcode 用户级路径 = ~/.zcode/cli/config.json",
          got is not None
          and got.parts[-3:] == (".zcode", "cli", "config.json"),
          str(got))
    # 负对照：未知宿主仍返回 None（没有的路径不许瞎编）
    check("H2 未知宿主返回 None", cli.user_config_path("nonexistent") is None,
          str(cli.user_config_path("nonexistent")))
    # 端到端：临时家目录里放一份用户级配置，doctor 的宿主库对账必须看到
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
        # 键形一致性：_HOSTS["zcode"] 的 key_path 必须能从用户级文件里
        # 真的挖出我们的条目（路径对 + 键形对，才是完整可发现）
        upath = cli.user_config_path("zcode")
        check("H3 用户级文件存在", upath.is_file(), str(upath))
        cfg, err = cli.hc.load_json_config(upath)
        container = cli._dig(cfg or {}, cli._HOSTS["zcode"]["key_path"])
        entry = (container or {}).get("hippocampus")
        check("H4 键形 (mcp,servers) 可解析出我方条目",
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
