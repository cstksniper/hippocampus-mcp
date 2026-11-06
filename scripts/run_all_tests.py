#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零依赖回归 runner —— 一条命令跑完 tests/ 下的全部套件。

## 为什么不用 pytest

项目的用例同时支持两种跑法：

    python tests/test_forget.py          # 单跑，走各文件自带的 main()
    pytest tests/                        # 装了 pytest 的环境（见 pytest.ini）

但**不装 pytest 也必须能跑全套** —— 这个项目的重活是 numpy/torch，
不该为了跑测试再引入一个测试框架依赖。所以这里提供一个只用标准库的 runner。

## 用法

    python scripts/run_all_tests.py                # 跑 tests/ 全部
    python scripts/run_all_tests.py -k forget      # 只跑文件名含 forget 的
    python scripts/run_all_tests.py --list         # 只列出会跑哪些
    python scripts/run_all_tests.py --timeout 600  # 单套件超时（秒）
    python scripts/run_all_tests.py --min-free-gb 0 # 跳过磁盘空间检查
    python scripts/run_all_tests.py --include-scripts
                                                   # 连同 scripts/ 下未公开的
                                                   # 私有套件一起跑
    python scripts/run_all_tests.py --log-dir .ci-logs
                                                   # 失败套件落**完整**输出
                                                   # （CI 上传 artifact 用）

## 退出码

    0 = 全部通过    1 = 有套件失败    2 = 发现阶段出错
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT, "tests")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

# 磁盘空间守卫
#   测试套件大量写临时文件（mkdtemp 库 / 模型缓存），满盘的症状是
#   「大片退出码 1」而不是一条清晰的错误——单跑同一套件却立刻全绿，
#   极易误判成套件本身退化，排查成本很高。所以要在开跑前拦一道。
#   阈值取 1GB 而不是更保守的 5GB：宿主盘常年在低位运行，
#   拦的是「再写就爆」，不是「紧张」。
MIN_FREE_GB_DEFAULT = 1.0

# 各套件末尾会打印「结果：N/M 通过」；用它做汇总
RESULT_RE = re.compile(r"结果[:：]\s*(\d+)\s*/\s*(\d+)\s*通过")
# 兜底：早期套件写的是「断言 N/M」。只认这两个口径，不匹配就回落退出码判定，
# 免得把「没匹配上」误当成「1/1 全过」——那会让回归统计悄悄失真。
ASSERT_RE = re.compile(r"断言\s*(\d+)\s*/\s*(\d+)")
FAIL_HINT_RE = re.compile(r"^\s*-\s+(.+?)\s*→\s*(.*)$", re.M)


def check_free_space(min_gb: float) -> list[tuple[str, float]]:
    """返回 [(盘符/路径, 剩余GB)] 中低于 ``min_gb`` 的项；正常返回空表。

    看两个盘：当前盘（python 自身与 %TEMP% 的默认落点）与 TMPDIR 所在盘
    （临时目录可能被挪到另一块盘）。
    """
    low: list[tuple[str, float]] = []
    seen: set[str] = set()
    for path in (os.getcwd(), os.environ.get("TMPDIR") or os.getcwd()):
        drive = os.path.splitdrive(os.path.abspath(path))[0] or path
        if drive in seen:
            continue
        seen.add(drive)
        try:
            free_gb = shutil.disk_usage(drive + os.sep if os.name == "nt"
                                        else drive).free / (1024 ** 3)
        except OSError:
            continue                    # 查不到就不拦（别把工具本身变成故障源）
        if free_gb < min_gb:
            low.append((drive, free_gb))
    return low


def discover(include_scripts: bool, keyword: str = ""):
    """返回 [(显示名, 绝对路径)]，按名称排序。"""
    found = []
    for base, label in ((TESTS_DIR, "tests"), (SCRIPTS_DIR, "scripts")):
        if base == SCRIPTS_DIR and not include_scripts:
            continue
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not (name.startswith("test_") and name.endswith(".py")):
                continue
            if keyword and keyword not in name:
                continue
            found.append((f"{label}/{name}", os.path.join(base, name)))
    return found


def save_full_log(log_dir: str, name: str, passed: int, total: int,
                  elapsed: float, fails, out: str) -> str:
    """把失败套件的**完整**原始输出落盘，返回文件路径。

    ``-v`` 只打印尾部（8K 字符 / 40 行，避免 153 个套件刷屏），但长输出
    里的 traceback 与子进程 stderr 仍可能在更早的位置被截掉 —— 汇总里就
    只剩一条无信息量的 ``<退出码> → 返回 1``，无从定性 daemon 类失败。
    CI 用 artifact 把这份完整日志带回来；只对非 PASS 套件落盘，不制造
    上百个空文件。
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, os.path.basename(name).replace(".py", "") + ".log")
    head = (f"# suite: {name}\n"
            f"# assertions: {passed}/{total}  elapsed: {elapsed:.1f}s\n"
            f"# failures: {fails}\n" + "=" * 62 + "\n")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(head + out)
    return path


def prepare_log_dir(log_dir: str) -> None:
    """开跑前清掉本目录里旧的 *.log，避免本机复跑时看到过期诊断。"""
    os.makedirs(log_dir, exist_ok=True)
    for name in os.listdir(log_dir):
        if name.endswith(".log"):
            try:
                os.remove(os.path.join(log_dir, name))
            except OSError:
                pass


def run_one(path: str, timeout: float):
    """跑一个套件，返回 (通过数, 总数, 耗时, 失败明细, 完整原始输出)。"""
    env = dict(os.environ)
    # 与项目约定一致：模型走镜像、禁用 Xet 通道
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("PYTHONPATH", os.path.join(ROOT, "src"))

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=ROOT,
            env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return 0, 0, time.time() - t0, [("<timeout>", f"exceeded {timeout:.0f}s")], ""
    elapsed = time.time() - t0

    # 优先用套件自报的「结果：N/M 通过」
    m = None
    for m in RESULT_RE.finditer(out):
        pass
    if not m:
        for m in ASSERT_RE.finditer(out):
            pass
    if m:
        passed, total = int(m.group(1)), int(m.group(2))
    else:
        # 没自报结果的（例如 pytest 风格文件单跑无输出），用退出码判定
        passed, total = (1, 1) if rc == 0 else (0, 1)

    fails = [tuple(g) for g in FAIL_HINT_RE.findall(out)]
    if rc != 0 and not fails:
        fails = [("<exit-code>", f"returned {rc}")]
    return passed, total, elapsed, fails, out


def main() -> int:
    ap = argparse.ArgumentParser(description="Zero-dependency regression runner")
    ap.add_argument("-k", "--keyword", default="",
                    help="only run suites whose filename contains this keyword")
    ap.add_argument("--list", action="store_true",
                    help="only list which suites would run")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-suite timeout in seconds (default 600)")
    ap.add_argument("--include-scripts", action="store_true",
                    help="also run the private test_* suites under scripts/ "
                         "that are not part of the public layer")
    # perf_write 这类**耗时阈值**用例在负载高的机器上会偶发红
    # （连跑 12/13、单跑 13/13），红的是机器忙、不是代码退化。
    # 默认仍按失败处理（否则永远发现不了真退化）；CI 里显式声明哪些套件
    # 允许红，红也要打印出来，只是不影响退出码。
    ap.add_argument("--allow-fail", default="",
                    help="comma-separated suite-name fragments; failures in these "
                         "suites only warn and do not affect the exit code")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="on failure, print the tail of raw output (8K chars / 40 lines)")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB_DEFAULT,
                    help="required free disk space (GB) before starting; abort if "
                         "lower; 0 = skip the check")
    ap.add_argument("--log-dir", default="",
                    help="write full stdout+stderr of non-PASS suites to this "
                         "directory (CI artifact)")
    args = ap.parse_args()

    if args.log_dir:
        prepare_log_dir(args.log_dir)

    suites = discover(args.include_scripts, args.keyword)
    if not suites:
        print("no test_*.py suites found", file=sys.stderr)
        return 2

    if args.min_free_gb > 0:
        low = check_free_space(args.min_free_gb)
        if low:
            print("not enough free disk space, refusing to start (low space makes "
                  "suites fail in batches; this wasted two real debugging rounds):",
                  file=sys.stderr)
            for drive, free_gb in low:
                print(f"  {drive} has {free_gb:.1f} GB free < "
                      f"{args.min_free_gb:.1f} GB", file=sys.stderr)
            print("Free up space and rerun, or pass --min-free-gb 0 to skip this "
                  "check.", file=sys.stderr)
            return 2

    # ------------------------------------------------------------------
    # 引导文件陷阱
    #   仓库根的 `.hippocampus.json`（{"library": "default"}）优先级高于
    #   HIPPOCAMPUS_PERSIST_DIR ⇒ 从仓库根跑测试时，**所有按连接/按目录解析
    #   库的用例都会被静默重定向到默认库**。
    #   症状是多库类套件成片假失败，而它与被测代码无关——移开该文件即全部
    #   转绿。不早早说一句，很容易被误读成"谁改坏了 connect-level 库解析"。
    # ------------------------------------------------------------------
    guide = os.path.join(os.getcwd(), ".hippocampus.json")
    if os.path.exists(guide):
        try:
            shown = open(guide, encoding="utf-8").read().strip().replace("\n", " ")[:80]
        except Exception:  # noqa: BLE001
            shown = "<unreadable>"
        print("!" * 62)
        print("⚠ A guide file .hippocampus.json exists in the current directory: "
              f"{shown}\n"
              "  It **overrides** HIPPOCAMPUS_PERSIST_DIR ⇒ cases that rely on\n"
              "  \"specifying the library directory themselves\" get silently\n"
              "  redirected. **If connect-level / multi-library failures appear,\n"
              "  suspect this file first**:\n"
              "    mv .hippocampus.json /tmp/ && python scripts/run_all_tests.py\n"
              "  (move it back when the run finishes)")
        print("!" * 62 + "\n")

    print(f"discovered {len(suites)} suites, starting\n" + "=" * 62)
    if args.list:
        for name, _ in suites:
            print(" ", name)
        print(f"{len(suites)} suites total")
        return 0

    allow = [s.strip() for s in args.allow_fail.split(",") if s.strip()]
    if allow:
        print(f"(non-blocking list: {', '.join(allow)} — failures in these suites "
              f"only warn)")
    total_pass = total_all = 0
    failed_suites = []
    allowed_failures = []
    t_start = time.time()

    for name, path in suites:
        passed, total, elapsed, fails, out = run_one(path, args.timeout)
        # 8K / 40 行：CI 日志里也留一份够用的尾部，artifact 万一没传成功
        # 不至于整轮白跑（1bf3181 就吃过只留 1200 字符 / 12 行的亏）。
        tail = out[-8000:]
        total_pass += passed
        total_all += total
        mark = "PASS" if (total and passed == total) else "FAIL"
        tolerated = mark == "FAIL" and any(a in name for a in allow)
        if tolerated:
            mark = "WARN"
        print(f"[{mark}] {name:44s} {passed:>4}/{total:<4}  {elapsed:6.1f}s")
        if mark == "FAIL":
            failed_suites.append((name, fails))
        elif tolerated:
            allowed_failures.append((name, fails))
        if mark != "PASS" and args.log_dir:
            log_path = save_full_log(args.log_dir, name, passed, total,
                                     elapsed, fails, out)
            print(f"      full log → {log_path}")
        if mark != "PASS" and args.verbose and tail:
            print("      ── output tail ──")
            for line in tail.strip().splitlines()[-40:]:
                print("      " + line)

    print("=" * 62)
    print(f"suites {len(suites) - len(failed_suites) - len(allowed_failures)}"
          f"/{len(suites)} all passed | "
          f"assertions {total_pass}/{total_all} | total {time.time() - t_start:.1f}s")
    if allowed_failures:
        print(f"\nnon-blocking failures: {len(allowed_failures)} "
              "(warned only, exit code unaffected):")
        for name, fails in allowed_failures:
            print(f"  {name}")
            for item, detail in fails[:6]:
                print(f"    - {item} → {detail}"[:150])

    if failed_suites:
        print("\nfailure details:")
        for name, fails in failed_suites:
            print(f"  {name}")
            for item, detail in fails[:6]:
                print(f"    - {item} → {detail}"[:150])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
