# -*- coding: utf-8 -*-
"""模拟新用户：从 wheel 装完后，验证「能导入 / 能跑通 / 能接入宿主」。

CI 与本机共用同一份验收口径。**必须在已安装 wheel 的解释器里执行**::

    python scripts/verify_install_e2e.py --repo <仓库根>

刻意不加载嵌入模型：quickstart 走 hash 后端，setup 的写后握手按设计不触发
模型加载，因此这个 job 不需要预下载模型、也不会卡在分钟级的 HF 请求上。
所有宿主配置都写进临时 HOME，绝不碰运行者真实的 MCP 配置。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"[PASS] {name}")
    else:
        _FAIL += 1
        _FAILURES.append(f"{name}: {detail}")
        print(f"[FAIL] {name}" + (f"  <- {detail}" if detail else ""))


def run(cmd: list[str], *, env=None, cwd=None, timeout: int = 300):
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=cwd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        return -1, f"TIMEOUT after {timeout}s\n{exc.stdout or ''}"
    except OSError as exc:
        return -1, f"SPAWN ERROR: {exc}"


def isolated_env(root: Path) -> dict:
    """把宿主配置与模型缓存都圈在临时目录里，跨平台生效。"""
    env = dict(os.environ)
    for key in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        env[key] = str(root)
    env["APPDATA"] = str(root / "AppData" / "Roaming")
    (root / "AppData" / "Roaming").mkdir(parents=True, exist_ok=True)
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    repo = Path(ap.parse_args().repo).resolve()
    py = sys.executable

    print("== 0. Interpreter / repo ==")
    print("python :", py)
    print("repo   :", repo)

    import hippocampus_mcp
    pkg = Path(hippocampus_mcp.__file__).resolve()

    print("\n== 1. Package origin ==")
    print("package:", pkg)
    check("wheel install layout (package not inside the repo src/ layout)",
          pkg.parent.name != "src" and repo not in pkg.parents, str(pkg))

    print("\n== 2. Top-level imports ==")
    for mod in ("hippocampus_mcp.server", "hippocampus_mcp.cli",
                "hippocampus_mcp.launcher"):
        rc, out = run([py, "-c", f"import {mod}"], timeout=180)
        check(f"import {mod}", rc == 0, out[-600:])

    print("\n== 3. quickstart full pipeline ==")
    quickstart = repo / "examples" / "quickstart.py"
    rc, out = run([py, str(quickstart)], cwd=str(repo), timeout=300)
    check("quickstart exit code 0", rc == 0, out[-800:])
    check("quickstart full pipeline", "All steps completed" in out, out[-400:])

    cli = [py, "-m", "hippocampus_mcp.cli"]
    # setup 的写后握手会顺手拉起 daemon 并留存保温；Windows 上它仍握着
    # hippocampus.db 句柄，退出时删临时目录会撞 WinError 32。这是设计内的
    # 副作用（宿主下次启动反而是热的），不是验收失败 ⇒ 清理失败一律容忍。
    with tempfile.TemporaryDirectory(prefix="hippo_install_e2e_",
                                     ignore_cleanup_errors=True) as td:
        home = Path(td)
        env = isolated_env(home)

        print("\n== 4. setup --dry-run ==")
        rc, out = run(cli + ["setup", "--host", "workbuddy", "--dry-run"],
                      env=env, cwd=str(repo), timeout=180)
        print(out[-900:])
        check("dry-run exit code 0", rc == 0, out[-800:])
        check("wheel layout writes no PYTHONPATH", "PYTHONPATH: not set" in out,
              out[-500:])

        print("\n== 5. real setup + post-write handshake ==")
        rc, out = run(cli + ["setup", "--host", "workbuddy"],
                      env=env, cwd=str(repo), timeout=300)
        print(out[-1400:])
        check("setup exit code 0", rc == 0, out[-1000:])
        check("initialize + tools/list handshake OK", "✓ config verified" in out,
              out[-800:])

        cfg = home / ".workbuddy" / "mcp.json"
        check("config file generated", cfg.is_file(), str(cfg))
        if cfg.is_file():
            try:
                entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["hippocampus"]
                check("entry uses the current interpreter", entry.get("command") == py,
                      repr(entry.get("command")))
                check("entry has no PYTHONPATH", "PYTHONPATH" not in entry.get("env", {}),
                      repr(entry.get("env")))
                lib = entry.get("env", {}).get("HIPPOCAMPUS_PERSIST_DIR", "")
                check("library dir inside the temp HOME", str(home) in lib, repr(lib))
            except Exception as exc:  # noqa: BLE001
                check("config JSON parseable", False, repr(exc))

        print("\n== 6. doctor read-only check ==")
        rc, out = run(cli + ["doctor", "--host", "workbuddy", "--project", "."],
                      env=env, cwd=str(repo), timeout=180)
        print(out[-1200:])
        check("doctor exit code 0", rc == 0, f"rc={rc}\n{out[-800:]}")

    print("\n" + "=" * 64)
    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for item in _FAILURES:
        print("  -", item)
    print("=" * 64)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
