# -*- coding: utf-8 -*-
"""Regression tests for the F2'·② workspace guide file ``.hippocampus.json``.

What gets pinned are the **four boundaries of the zero-config convenience** — its whole value lives on them:

  * file absent ⇒ **fully inert**, behavior bit-identical to pre-multi-library;
  * unresolvable / broken file ⇒ **loud** fallback to env default (a cross-library leak is far more dangerous);
  * upward search **no more than 5 levels** (no swallowing unrelated upper-directory configs);
  * only the **exact filename** counts (``hippocampus.json`` without the dot prefix does not).

Why a separate suite: this logic previously had **no test coverage at all**, and it is a
"convenience" — if it quietly activates yet points at the wrong library, the symptom is a
**silent cross-library leak** (writing into someone else's library), far harder to debug than an error.

Usage:
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
    """Same normalization ruler as the code under test (Windows may yield 8.3 short names; both sides must agree)."""
    return L.normpath(os.path.expanduser(p))


class _Info:
    """A good-enough ``LibraryInfo`` stub: the code under test only uses ``name`` / ``path``."""

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path


def _write_guide(d: str, obj: object = None, raw: str | None = None) -> str:
    p = os.path.join(d, S.GUIDE_FILENAME)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(raw if raw is not None else json.dumps(obj))
    return p


def _patch_resolve(known: dict[str, tuple[str, str]]):
    """Swap library resolution for a controlled fake: only names in known resolve, everything else raises unresolvable."""
    orig = L.resolve_library

    def fake(query, roots=None, libs=None, default_path=None):
        if query in known:
            n, p = known[query]
            return _Info(n, p)
        raise L.LibraryResolutionError(f"找不到记忆库：{query}")

    L.resolve_library = fake
    return orig


def _patch_warning():
    """Capture logger.warning calls (sturdier than attaching a Handler; immune to level/propagate)."""
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
# A · _find_workspace_guide: search upward, max 5 levels, exact filename only
# ----------------------------------------------------------------------
def part_a() -> None:
    print("\n【A】引导文件查找（向上 ≤5 层 / 精确文件名）")

    check("A1 no guide file => None", S._find_workspace_guide(_tmpdir()) is None)

    direct = _tmpdir()
    g = _write_guide(direct, {"library": "hippo_probe"})
    got = S._find_workspace_guide(direct)
    check("A2 same-dir file => returns it", got is not None and _np(got) == _np(g),
          f"got={got}")

    root = _tmpdir()
    deep = os.path.join(root, "a", "b", "c")
    os.makedirs(deep, exist_ok=True)
    g3 = _write_guide(root, {"library": "hippo_probe"})
    got3 = S._find_workspace_guide(deep)
    check("A3 found 3 levels up", got3 is not None and _np(got3) == _np(g3),
          f"got={got3}")

    root2 = _tmpdir()
    d6 = os.path.join(root2, "l1", "l2", "l3", "l4", "l5", "l6")
    os.makedirs(d6, exist_ok=True)
    _write_guide(root2, {"library": "hippo_probe"})
    check("A4 beyond 5 levels => None (no swallowing upper configs)",
          S._find_workspace_guide(d6) is None,
          f"got={S._find_workspace_guide(d6)}")

    near = _tmpdir()
    with open(os.path.join(near, "hippocampus.json"), "w",
              encoding="utf-8") as fh:
        fh.write(json.dumps({"library": "hippo_probe"}))
    check("A5 near-name without the leading dot is rejected", S._find_workspace_guide(near) is None)


# ----------------------------------------------------------------------
# B · _resolve_default_lib_path: guide file > env, and broken cases must be loud
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
        check("B1 no guide file => fall back to env default", _np(got) == base, f"got={got}")

        _write_guide(root, {"library": "hippo_probe"})
        got2 = S._resolve_default_lib_path(base, root)
        check("B2 guide file takes precedence over env",
              _np(got2) == _np("/tmp/target_lib"), f"got={got2}")

        _write_guide(root, {"library": "   "})
        got3 = S._resolve_default_lib_path(base, root)
        check("B3 blank library => fallback", _np(got3) == base, f"got={got3}")

        _write_guide(root, {"other": 1})
        got4 = S._resolve_default_lib_path(base, root)
        check("B4 missing library key => fallback", _np(got4) == base, f"got={got4}")

        warns.clear()
        _write_guide(root, raw="{not json")
        with_ = S._resolve_default_lib_path(base, root)
        check("B5 broken JSON => fallback", _np(with_) == base, f"got={with_}")
        check("B5 broken JSON must be loud (no silent fallback)",
              any("read failed" in m for m in warns), f"warns={warns[:2]}")

        warns.clear()
        _write_guide(root, {"library": "不存在的库"})
        got6 = S._resolve_default_lib_path(base, root)
        check("B6 unresolvable => fallback", _np(got6) == base, f"got={got6}")
        check("B6 unresolvable must be loud (a leak is riskier than a fallback)",
              any("unresolvable" in m for m in warns), f"warns={warns[:2]}")

        warns.clear()
        _write_guide(root, raw="[1,2,3]")
        got7 = S._resolve_default_lib_path(base, root)
        check("B7 non-dict top level => fallback without crashing", _np(got7) == base, f"got={got7}")
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
