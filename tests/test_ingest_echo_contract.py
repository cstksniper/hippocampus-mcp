# -*- coding: utf-8 -*-
"""写入侧回显契约：合并警示必须在返回体首位。

背景：命中 0.85 去重阈值时，本次写入**不会新建条目**，内容被并入既有记忆。
这是宿主最该先看到的事——实测后果是「燕行日程·约二千余里」这类带新数字的卡片
被整条吞掉，宿主却毫无察觉（只能靠改写措辞降相似度来救）。

此前 `merged_into` 排在返回体倒数第二，被 raw_id / memory_ids 挤到末尾；
宿主只看开头几个字段时等于没看见。**JSON 保序，所以字段顺序就是优先级。**

覆盖：
    A1 ingest 返回体首键 = merged_into
    A2 ingest_batch 返回体首键 = merged_into
    A3 命中去重时 merged_into 非空，且本次确实没新建（memory_ids 并入既有 id）
    A4 未命中去重时 merged_into 为空、新建了条目

用法：
    <py311> tests/test_ingest_echo_contract.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hippocampus_mcp.brain import HippoBrain  # noqa: E402

_PASS = 0
_FAIL = 0
_FAILED: list[tuple[str, str]] = []
_TMPDIRS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        _FAILED.append((name, detail))
        print(f"  [FAIL] {name}" + (f"  <- {detail}" if detail else ""))


def _new_brain() -> HippoBrain:
    d = tempfile.mkdtemp(prefix="echo16_")
    _TMPDIRS.append(d)
    return HippoBrain(persist_dir=d, embedder_backend="hash", auto_save=False)


def main() -> int:
    print("=" * 70)
    print("追加 16②：写入侧合并警示前置")
    print("=" * 70)

    text = "燕行日程往返约二千余里"

    # A1 / A3：重复写入触发去重
    b = _new_brain()
    r1 = b.ingest(text)
    check("A3 首次写入新建了条目", len(r1["memory_ids"]) == 1, str(r1["memory_ids"]))
    first_id = r1["memory_ids"][0]

    r2 = b.ingest(text)
    keys = list(r2.keys())
    check("A1 ingest 返回体首键 = merged_into", keys[0] == "merged_into", str(keys[:3]))
    check("A3 命中去重时 merged_into 非空", bool(r2["merged_into"]), str(r2["merged_into"])[:120])
    check("A3 本次并入既有 id（未新建）",
          r2["memory_ids"] == [first_id], f"{r2['memory_ids']} vs {[first_id]}")

    # A2：批写
    b2 = _new_brain()
    b2.ingest(text)
    out = b2.ingest_batch([text, text])
    keys = list(out.keys())
    check("A2 ingest_batch 返回体首键 = merged_into", keys[0] == "merged_into", str(keys[:3]))

    # A4：未命中去重
    b3 = _new_brain()
    r = b3.ingest("一条完全独立的新内容，与其他条目都不相似")
    check("A4 未命中时 merged_into 为空", r["merged_into"] == [], str(r["merged_into"])[:120])
    check("A4 未命中时新建了条目", len(r["memory_ids"]) >= 1, str(r["memory_ids"]))

    print("=" * 70)
    print(f"结果：{_PASS}/{_PASS + _FAIL} 通过")
    if _FAIL:
        print("失败项：")
        for name, detail in _FAILED:
            print(f"  - {name}  → {detail}")
    return 0 if not _FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
