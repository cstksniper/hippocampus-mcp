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
    head = (f"# 套件: {name}\n"
            f"# 断言: {passed}/{total}  耗时: {elapsed:.1f}s\n"
            f"# 失败项: {fails}\n" + "=" * 62 + "\n")
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
        return 0, 0, time.time() - t0, [("<超时>", f"超过 {timeout:.0f}s")], ""
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
        fails = [("<退出码>", f"返回 {rc}")]
    return passed, total, elapsed, fails, out


def main() -> int:
    ap = argparse.ArgumentParser(description="零依赖回归 runner")
    ap.add_argument("-k", "--keyword", default="", help="只跑文件名含该关键词的套件")
    ap.add_argument("--list", action="store_true", help="只列出会跑哪些套件")
    ap.add_argument("--timeout", type=float, default=600.0, help="单套件超时秒数（默认 600）")
    ap.add_argument("--include-scripts", action="store_true",
                    help="连同 scripts/ 下未进公开层的私有套件一起跑")
    # perf_write 这类**耗时阈值**用例在负载高的机器上会偶发红
    # （连跑 12/13、单跑 13/13），红的是机器忙、不是代码退化。
    # 默认仍按失败处理（否则永远发现不了真退化）；CI 里显式声明哪些套件
    # 允许红，红也要打印出来，只是不影响退出码。
    ap.add_argument("--allow-fail", default="",
                    help="逗号分隔的套件名（可只写片段）；这些套件失败只警告、"
                         "不影响退出码")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="失败时打印原始输出尾部（8K 字符 / 40 行）")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB_DEFAULT,
                    help="开跑前要求各盘剩余空间（GB），不足即中止；0 = 不检查")
    ap.add_argument("--log-dir", default="",
                    help="非 PASS 套件的完整 stdout+stderr 落到此目录（CI artifact）")
    args = ap.parse_args()

    if args.log_dir:
        prepare_log_dir(args.log_dir)

    suites = discover(args.include_scripts, args.keyword)
    if not suites:
        print("没有发现任何 test_*.py 套件", file=sys.stderr)
        return 2

    if args.min_free_gb > 0:
        low = check_free_space(args.min_free_gb)
        if low:
            print("磁盘空间不足，拒绝开跑（低剩余会让套件成片假红，"
                  "实测曾白耗两轮排查）：", file=sys.stderr)
            for drive, free_gb in low:
                print(f"  {drive} 剩 {free_gb:.1f} GB < "
                      f"{args.min_free_gb:.1f} GB", file=sys.stderr)
            print("清理后重跑，或显式 --min-free-gb 0 跳过本检查。",
                  file=sys.stderr)
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
            shown = "<读不出来>"
        print("!" * 62)
        print("⚠ 当前目录存在引导文件 .hippocampus.json："
              f"{shown}\n"
              "  它会**盖过** HIPPOCAMPUS_PERSIST_DIR ⇒ 依赖「自行指定库目录」的用例\n"
              "  会被静默重定向。**若出现 connect-level / 多库类失败，先怀疑它**：\n"
              "    mv .hippocampus.json /tmp/ && python scripts/run_all_tests.py\n"
              "  （跑完记得 mv 回来）")
        print("!" * 62 + "\n")

    print(f"发现 {len(suites)} 个套件，开始运行\n" + "=" * 62)
    if args.list:
        for name, _ in suites:
            print(" ", name)
        print(f"共 {len(suites)} 个套件")
        return 0

    allow = [s.strip() for s in args.allow_fail.split(",") if s.strip()]
    if allow:
        print(f"（非阻断名单：{', '.join(allow)} —— 这些套件失败只警告）")
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
            print(f"      完整输出 → {log_path}")
        if mark != "PASS" and args.verbose and tail:
            print("      ── 输出尾部 ──")
            for line in tail.strip().splitlines()[-40:]:
                print("      " + line)

    print("=" * 62)
    print(f"套件 {len(suites) - len(failed_suites) - len(allowed_failures)}"
          f"/{len(suites)} 全通过 ｜ "
          f"断言 {total_pass}/{total_all} ｜ 总耗时 {time.time() - t_start:.1f}s")
    if allowed_failures:
        print(f"\n非阻断失败 {len(allowed_failures)} 个（已计入警告，不影响退出码）：")
        for name, fails in allowed_failures:
            print(f"  {name}")
            for item, detail in fails[:6]:
                print(f"    - {item} → {detail}"[:150])

    if failed_suites:
        print("\n失败明细：")
        for name, fails in failed_suites:
            print(f"  {name}")
            for item, detail in fails[:6]:
                print(f"    - {item} → {detail}"[:150])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
