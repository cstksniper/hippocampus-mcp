# -*- coding: utf-8 -*-
"""Write-side echo contract: the merge warning must lead the payload.

Background: when the 0.85 dedup threshold is hit, the write **creates no new
record**; the content is folded into the existing memory. This is what the
host most needs to see first -- in practice, cards carrying new numbers like
「燕行日程·约二千余里」 got swallowed whole while the host never noticed
(the only rescue was rewording to lower similarity).

Previously `merged_into` sat second-to-last in the payload, squeezed to the
end behind raw_id / memory_ids; a host reading only the first few fields saw
nothing. **JSON preserves order, so field order is priority.**

Coverage:
    A1 first key of the ingest payload = merged_into
    A2 first key of the ingest_batch payload = merged_into
    A3 on a dedup hit, merged_into is non-empty and nothing new was created (memory_ids fold into the existing id)
    A4 on no dedup hit, merged_into is empty and a new record is created

Usage:
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

    # A1 / A3: duplicate write triggers dedup
    b = _new_brain()
    r1 = b.ingest(text)
    check("A3 first write creates a record", len(r1["memory_ids"]) == 1, str(r1["memory_ids"]))
    first_id = r1["memory_ids"][0]

    r2 = b.ingest(text)
    keys = list(r2.keys())
    check("A1 ingest payload first key = merged_into", keys[0] == "merged_into", str(keys[:3]))
    check("A3 merged_into non-empty on dedup hit", bool(r2["merged_into"]), str(r2["merged_into"])[:120])
    check("A3 folded into the existing id (nothing new created)",
          r2["memory_ids"] == [first_id], f"{r2['memory_ids']} vs {[first_id]}")

    # A2: batch write
    b2 = _new_brain()
    b2.ingest(text)
    out = b2.ingest_batch([text, text])
    keys = list(out.keys())
    check("A2 ingest_batch payload first key = merged_into", keys[0] == "merged_into", str(keys[:3]))

    # A4: no dedup hit
    b3 = _new_brain()
    r = b3.ingest("一条完全独立的新内容，与其他条目都不相似")
    check("A4 merged_into empty on no dedup hit", r["merged_into"] == [], str(r["merged_into"])[:120])
    check("A4 new record created on no dedup hit", len(r["memory_ids"]) >= 1, str(r["memory_ids"]))

    print("=" * 70)
    print(f"结果：{_PASS}/{_PASS + _FAIL} 通过")
    if _FAIL:
        print("失败项：")
        for name, detail in _FAILED:
            print(f"  - {name}  → {detail}")
    return 0 if not _FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
