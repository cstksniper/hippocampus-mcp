# -*- coding: utf-8 -*-
"""F2'·② 工作区引导文件 ``.hippocampus.json`` 的回归测试。

被钉住的是**「零配置便利项」的四条边界**——它的价值全在这些边界上：

  * 文件不存在 ⇒ **完全不生效**，行为与多库之前逐位一致；
  * 解析不明 / 文件坏了 ⇒ **出声**回退 env 默认（串库比回退危险得多）；
  * 向上找**不超过 5 层**（防止误吃到无关上级目录的配置）；
  * 只认**精确文件名**（``hippocampus.json`` 无点前缀的不算）。

为什么值得单开一个套件：这段逻辑**此前没有任何测试覆盖**，而它是
「便利项」——一旦它悄悄生效却指错库，症状是**静默串库**（写到别人的库里），
比报错难查得多。

用法：
    python tests/test_workspace_guide.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hippocampus_mcp.library as L  # noqa: E402
import hippocampus_mcp.server as S  # noqa: E402

_results: list[tuple[str, bool, str]] = []
_TMPDIRS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    line = f"  [{tag}] {name}"
    if detail and not cond:
        line += f"\n         → {detail}"
    print(line, flush=True)


def _tmpdir() -> str:
    d = tempfile.mkdtemp(prefix="hippo_guide_")
    _TMPDIRS.append(d)
    return d


def _np(p: str) -> str:
    """与被测代码同一把归一化尺子（Windows 下会得到 8.3 短名，必须两边一致）。"""
    return L.normpath(os.path.expanduser(p))


class _Info:
    """够用的 ``LibraryInfo`` 替身：被测代码只用到 ``name`` / ``path``。"""

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path


def _write_guide(d: str, obj: object = None, raw: str | None = None) -> str:
    p = os.path.join(d, S.GUIDE_FILENAME)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(raw if raw is not None else json.dumps(obj))
    return p


def _patch_resolve(known: dict[str, tuple[str, str]]):
    """把库解析换成受控实现：只有 known 里的名字能解析，其余抛解析不明。"""
    orig = L.resolve_library

    def fake(query, roots=None, libs=None, default_path=None):
        if query in known:
            n, p = known[query]
            return _Info(n, p)
        raise L.LibraryResolutionError(f"找不到记忆库：{query}")

    L.resolve_library = fake
    return orig


def _patch_warning():
    """捕获 logger.warning 的调用（比挂 Handler 稳，不受 level/propagate 影响）。"""
    calls: list[str] = []
    orig = S.logger.warning

    def fake(msg, *a, **k):
        try:
            calls.append(str(msg) % a if a else str(msg))
        except Exception:
            calls.append(str(msg))
    S.logger.warning = fake
    return orig, calls


# ----------------------------------------------------------------------
# A · _find_workspace_guide：向上找，最多 5 层，只认精确文件名
# ----------------------------------------------------------------------
def part_a() -> None:
    print("\n【A】引导文件查找（向上 ≤5 层 / 精确文件名）")

    check("A1 无引导文件 ⇒ None", S._find_workspace_guide(_tmpdir()) is None)

    direct = _tmpdir()
    g = _write_guide(direct, {"library": "hippo_probe"})
    got = S._find_workspace_guide(direct)
    check("A2 同目录有 ⇒ 返回该文件", got is not None and _np(got) == _np(g),
          f"got={got}")

    root = _tmpdir()
    deep = os.path.join(root, "a", "b", "c")
    os.makedirs(deep, exist_ok=True)
    g3 = _write_guide(root, {"library": "hippo_probe"})
    got3 = S._find_workspace_guide(deep)
    check("A3 向上 3 层能找到", got3 is not None and _np(got3) == _np(g3),
          f"got={got3}")

    root2 = _tmpdir()
    d6 = os.path.join(root2, "l1", "l2", "l3", "l4", "l5", "l6")
    os.makedirs(d6, exist_ok=True)
    _write_guide(root2, {"library": "hippo_probe"})
    check("A4 超过 5 层 ⇒ None（不误吃上级配置）",
          S._find_workspace_guide(d6) is None,
          f"got={S._find_workspace_guide(d6)}")

    near = _tmpdir()
    with open(os.path.join(near, "hippocampus.json"), "w",
              encoding="utf-8") as fh:
        fh.write(json.dumps({"library": "hippo_probe"}))
    check("A5 缺前导点的近似名不认", S._find_workspace_guide(near) is None)


# ----------------------------------------------------------------------
# B · _resolve_default_lib_path：引导文件 > env，且坏情况必须出声
# ----------------------------------------------------------------------
def part_b() -> None:
    print("\n【B】默认库解析：引导文件 > env，坏情况出声")

    root = _tmpdir()
    base = _np(os.path.join(root, "env_default_lib"))
    os.makedirs(base, exist_ok=True)

    orig_r = _patch_resolve({"hippo_probe": ("hippo_probe", "/tmp/target_lib")})
    orig_w, warns = _patch_warning()
    try:
        got = S._resolve_default_lib_path(base, root)
        check("B1 无引导文件 ⇒ 回退 env 默认", _np(got) == base, f"got={got}")

        _write_guide(root, {"library": "hippo_probe"})
        got2 = S._resolve_default_lib_path(base, root)
        check("B2 引导文件优先于 env",
              _np(got2) == _np("/tmp/target_lib"), f"got={got2}")

        _write_guide(root, {"library": "   "})
        got3 = S._resolve_default_lib_path(base, root)
        check("B3 library 空白 ⇒ 回退", _np(got3) == base, f"got={got3}")

        _write_guide(root, {"other": 1})
        got4 = S._resolve_default_lib_path(base, root)
        check("B4 缺 library 键 ⇒ 回退", _np(got4) == base, f"got={got4}")

        warns.clear()
        _write_guide(root, raw="{not json")
        with_ = S._resolve_default_lib_path(base, root)
        check("B5 坏 JSON ⇒ 回退", _np(with_) == base, f"got={with_}")
        check("B5 坏 JSON 必须出声（不静默回退）",
              any("读取失败" in m for m in warns), f"warns={warns[:2]}")

        warns.clear()
        _write_guide(root, {"library": "不存在的库"})
        got6 = S._resolve_default_lib_path(base, root)
        check("B6 解析不明 ⇒ 回退", _np(got6) == base, f"got={got6}")
        check("B6 解析不明必须出声（串库比回退危险）",
              any("解析不明" in m for m in warns), f"warns={warns[:2]}")

        warns.clear()
        _write_guide(root, raw="[1,2,3]")
        got7 = S._resolve_default_lib_path(base, root)
        check("B7 顶层非 dict ⇒ 回退不崩", _np(got7) == base, f"got={got7}")
    finally:
        L.resolve_library = orig_r
        S.logger.warning = orig_w


def main() -> int:
    part_a()
    part_b()
    ok = sum(1 for _, p, _ in _results if p)
    print(f"\n结果：{ok} 通过 / {len(_results) - ok} 失败")
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)
    return 1 if ok != len(_results) else 0


if __name__ == "__main__":
    sys.exit(main())
