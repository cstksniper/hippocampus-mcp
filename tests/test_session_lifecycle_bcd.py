# -*- coding: utf-8 -*-
"""Regression suite for the B/C/D proposals (verifier feedback 2026-11-01).

Coverage (design: docs/design/验证反馈B_C_D设计提案_20261101.md):
  B session_end returns saved_layers layer detail (written/skipped/failed); a touched
    layer (brain_topics) must be written after close — the mirror that exposes A-type silence.
  C-1 a second session_start by the same client auto-ends the old session (superseded_session self-report).
  C-2 session_end closes precisely by session_id; cross-client proxy close is refused with zero side effects.
  C-3 idle-timeout recycling (ended + recycled set in the active_sessions display); env=0 disables it.
  D  similarity dedup defaults on (0.85), hits echo dedup_hits; env=0 leaves only exact dedup.

Usage:
    <py311> tests/test_session_lifecycle_bcd.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from hippocampus_mcp.brain import HippoBrain

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    line = f"  [{tag}] {name}"
    if detail and not cond:
        line += f"\n         → {detail}"
    print(line)


def test_b_saved_layers(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    b.session_start(client="a")
    out = b.session_end(summary="", open_topics=["话题甲"], client="a")
    sl = out.get("saved_layers") or {}
    check("B1 payload carries saved_layers covering the topics layer",
          isinstance(sl, dict) and "brain_topics.json" in sl, str(sl)[:200])
    check("B2 values limited to written/skipped",
          sl and all(v in ("written", "skipped") for v in sl.values()),
          str(set(sl.values())))
    # close touched the topics layer => this run's layer must be written (the mirror that exposes A-type silence).
    b.session_start(client="a")
    out2 = b.session_end(summary="", close_topics=["话题甲"], client="a")
    check("B3 topics layer touched by close honestly marked written",
          (out2.get("saved_layers") or {}).get("brain_topics.json")
          == "written",
          str((out2.get("saved_layers") or {}).get("brain_topics.json")))


def test_c1_supersede(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    s1 = b.session_start(client="w", recent_limit=0)["session_id"]
    out2 = b.session_start(client="w", recent_limit=0)
    s2 = out2["session_id"]
    check("C1a second start self-reports superseded_session",
          out2.get("superseded_session") == s1, f"{s1} → {s2}")
    check("C1b only one active session left",
          len(b.active_sessions(client="w")) == 1)
    old = b.sessions[s1]
    check("C1c old session ended and attributable",
          bool(old.get("ended_at"))
          and old.get("superseded_by") == s2)
    check("C1d first start has no self-report key (zero traces)",
          "superseded_session" not in b.session_start(
              client="fresh", recent_limit=0))


def test_c2_end_by_id(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    s_a = b.session_start(client="a", recent_limit=0)["session_id"]
    # Cross-client proxy close: refused + zero side effects (session still active, topics untouched).
    b.session_end(summary="", open_topics=["不该出现的话题"],
                  session_id=s_a, client="b")
    check("C2a proxy close refused", "error" in b.session_end(
        summary="", open_topics=["不该出现的话题"],
        session_id=s_a, client="b"))
    check("C2b refusal has zero side effects (session still active)",
          not b.sessions[s_a].get("ended_at")
          and b.current_session_of("a") == s_a)
    check("C2c refusal has zero side effects (topics untouched)",
          not any(t["text"] == "不该出现的话题" for t in b.open_topics))
    # Closing one's own current by id: equivalent to a regular close.
    out = b.session_end(summary="收", open_topics=["正常话题"],
                        session_id=s_a, client="a")
    check("C2d closing current by id succeeds", out.get("session_id") == s_a
          and b.sessions[s_a].get("ended_at")
          and any(t["text"] == "正常话题" for t in b.open_topics))
    # Closing an already-closed session: reports an error without crashing.
    check("C2e duplicate close reports an error", "error" in b.session_end(
        session_id=s_a, client="a"))


def test_c3_idle_recycle(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    sid = b.session_start(client="z", recent_limit=0)["session_id"]
    b.sessions[sid]["last_seen_at"] = time.time() - 25 * 3600
    check("C3a idle timeout auto-recycles (recycled marker)",
          b.active_sessions() == [] and b.sessions[sid].get("recycled")
          and bool(b.sessions[sid].get("ended_at")))
    check("C3b current pointer removed along with it", b.current_session_of("z") is None)
    os.environ["HIPPOCAMPUS_SESSION_IDLE_RECYCLE"] = "0"
    try:
        sid2 = b.session_start(client="z2", recent_limit=0)["session_id"]
        b.sessions[sid2]["last_seen_at"] = time.time() - 25 * 3600
        check("C3c env=0 disables recycling (not marked recycled; visible only with a wider display TTL)",
              any(x["session_id"] == sid2
                  for x in b.active_sessions(ttl=30 * 3600))
              and not b.sessions[sid2].get("recycled"))
    finally:
        os.environ.pop("HIPPOCAMPUS_SESSION_IDLE_RECYCLE", None)


def test_d_sim_dedup(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    b.session_start(client="d", recent_limit=0)
    b.session_end(summary="", open_topics=["tick 续传待办"], client="d")
    os.environ["HIPPOCAMPUS_TOPIC_SIM_DEDUP"] = "0.85"
    try:
        out = b.session_end(summary="",
                            open_topics=["tick 续传待办事项"], client="d")
        check("D1 similar topic blocked",
              out.get("topics_deduped") == 1
              and len(b.open_topics) == 1)
        hits = out.get("dedup_hits") or []
        check("D2 hit echoed (matched old entry visible verbatim)",
              hits and hits[0].get("matched") == "tick 续传待办"
              and hits[0].get("input") == "tick 续传待办事项"
              and hits[0].get("sim", 0) >= 0.85, str(hits))
        os.environ["HIPPOCAMPUS_TOPIC_SIM_DEDUP"] = "0"
        out2 = b.session_end(summary="",
                             open_topics=["tick 续传待办事项"], client="d")
        check("D3 env=0 disables it (exact dedup only)",
              out2.get("topics_deduped") == 0
              and len(b.open_topics) == 2
              and "dedup_hits" not in out2)
        # Exact duplicates are blocked at any threshold.
        out3 = b.session_end(summary="",
                             open_topics=["tick 续传待办事项 "], client="d")
        check("D4 exact dedup unaffected by the threshold",
              out3.get("topics_deduped") == 1)
    finally:
        os.environ.pop("HIPPOCAMPUS_TOPIC_SIM_DEDUP", None)
    # D5 (real phrase pair the verifier's 2026-11-01 run missed; regression pin at the
    # default threshold): rewording + separator change (/ vs 、) + added words — pure
    # difflib 0.85 missed it (raw ratio 0.469); normalization + bigram gives 0.63 >= 0.60.
    b2 = HippoBrain(persist_dir=tmp + "_d5", embedder_backend="hash")
    b2.session_start(client="v", recent_limit=0)
    b2.session_end(
        summary="",
        open_topics=["TA/I/J/JM/FG/SA/UR tick 下载待用户另立项按月续传"],
        client="v")
    out5 = b2.session_end(
        summary="",
        open_topics=["其余品种（TA、I、J、JM、FG、SA、UR）tick 历史行情"
                     "需要另立专项分期补齐下载"],
        client="v")
    check("D5 verifier's real phrase pair blocked (default threshold)",
          out5.get("topics_deduped") == 1
          and (out5.get("dedup_hits") or [{}])[0].get("matched")
          == "TA/I/J/JM/FG/SA/UR tick 下载待用户另立项按月续传",
          str(out5.get("dedup_hits")))
    # D6 hard-identifier penalty: chapter-number/volume-number/name-list-char differences must pass (reverse pin for the live T2/T3 offenders).
    b3 = HippoBrain(persist_dir=tmp + "_d6", embedder_backend="hash")
    b3.session_start(client="n", recent_limit=0)
    b3.session_end(summary="", open_topics=["示例小说第43章情节校对"],
                   client="n")
    out6 = b3.session_end(summary="", open_topics=["示例小说第44章情节校对"],
                          client="n")
    check("D6a different chapter number is not falsely blocked",
          out6.get("topics_deduped") == 0 and len(b3.open_topics) == 2)
    out7 = b3.session_end(summary="", open_topics=["示例古籍卷五校对"],
                          client="n")
    out8 = b3.session_end(summary="", open_topics=["示例古籍卷六校对"],
                          client="n")
    check("D6b different Chinese-numeral volume number is not falsely blocked",
          out8.get("topics_deduped") == 0 and len(b3.open_topics) == 4,
          str(out8.get("dedup_hits")))


def main() -> int:
    for tag, fn in (("[B] saved_layers", test_b_saved_layers),
                    ("[C1] 自动终结", test_c1_supersede),
                    ("[C2] 按 id 收尾/代收拒绝", test_c2_end_by_id),
                    ("[C3] idle 回收", test_c3_idle_recycle),
                    ("[D] 相似度去重", test_d_sim_dedup)):
        with tempfile.TemporaryDirectory() as tmp:
            print(f"\n{tag}")
            fn(tmp)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n结果：{len(_results) - len(failed)}/{len(_results)} 通过")
    for n in failed:
        print("  ❌ " + n)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
