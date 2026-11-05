# -*- coding: utf-8 -*-
"""B/C/D 提案落地回归（验证反馈 2026-11-01）。

覆盖（设计：docs/design/验证反馈B_C_D设计提案_20261101.md）：
  B session_end 返回 saved_layers 层级明细（written/skipped/failed）；
    触及层（brain_topics）close 后必须 written——A 型静默的照妖镜。
  C-1 同 client 二次 session_start 自动终结旧会话（superseded_session 自曝）。
  C-2 session_end 按 session_id 精确收尾；跨 client 代收被拒且零副作用。
  C-3 idle 超时回收（active_sessions 展示时置 ended + recycled）；env=0 关闭。
  D  相似度去重默认开（0.85），命中回显 dedup_hits；env=0 关闭只剩精确去重。

用法：
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
    check("B1 返回体带 saved_layers 且覆盖 topics 层",
          isinstance(sl, dict) and "brain_topics.json" in sl, str(sl)[:200])
    check("B2 值域只含 written/skipped",
          sl and all(v in ("written", "skipped") for v in sl.values()),
          str(set(sl.values())))
    # close 改动了 topics 层 ⇒ 本次该层必须 written（A 型静默的照妖镜）。
    b.session_start(client="a")
    out2 = b.session_end(summary="", close_topics=["话题甲"], client="a")
    check("B3 close 触及的 topics 层如实标 written",
          (out2.get("saved_layers") or {}).get("brain_topics.json")
          == "written",
          str((out2.get("saved_layers") or {}).get("brain_topics.json")))


def test_c1_supersede(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    s1 = b.session_start(client="w", recent_limit=0)["session_id"]
    out2 = b.session_start(client="w", recent_limit=0)
    s2 = out2["session_id"]
    check("C1a 二次开场自曝 superseded_session",
          out2.get("superseded_session") == s1, f"{s1} → {s2}")
    check("C1b active 只剩一个",
          len(b.active_sessions(client="w")) == 1)
    old = b.sessions[s1]
    check("C1c 旧会话已终结且可归因",
          bool(old.get("ended_at"))
          and old.get("superseded_by") == s2)
    check("C1d 首次开场无自曝键（零痕迹）",
          "superseded_session" not in b.session_start(
              client="fresh", recent_limit=0))


def test_c2_end_by_id(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    s_a = b.session_start(client="a", recent_limit=0)["session_id"]
    # 跨 client 代收：拒绝 + 零副作用（会话仍活跃、topics 未被动）。
    b.session_end(summary="", open_topics=["不该出现的话题"],
                  session_id=s_a, client="b")
    check("C2a 代收被拒", "error" in b.session_end(
        summary="", open_topics=["不该出现的话题"],
        session_id=s_a, client="b"))
    check("C2b 拒绝零副作用（会话仍活跃）",
          not b.sessions[s_a].get("ended_at")
          and b.current_session_of("a") == s_a)
    check("C2c 拒绝零副作用（topics 未被塞入）",
          not any(t["text"] == "不该出现的话题" for t in b.open_topics))
    # 按 id 收自己的 current：等价常规收尾。
    out = b.session_end(summary="收", open_topics=["正常话题"],
                        session_id=s_a, client="a")
    check("C2d 按 id 收 current 成功", out.get("session_id") == s_a
          and b.sessions[s_a].get("ended_at")
          and any(t["text"] == "正常话题" for t in b.open_topics))
    # 已收尾的再收：报错不炸。
    check("C2e 重复收尾报错", "error" in b.session_end(
        session_id=s_a, client="a"))


def test_c3_idle_recycle(tmp: str) -> None:
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    sid = b.session_start(client="z", recent_limit=0)["session_id"]
    b.sessions[sid]["last_seen_at"] = time.time() - 25 * 3600
    check("C3a idle 超时自动回收（recycled 标记）",
          b.active_sessions() == [] and b.sessions[sid].get("recycled")
          and bool(b.sessions[sid].get("ended_at")))
    check("C3b current 指针一并摘除", b.current_session_of("z") is None)
    os.environ["HIPPOCAMPUS_SESSION_IDLE_RECYCLE"] = "0"
    try:
        sid2 = b.session_start(client="z2", recent_limit=0)["session_id"]
        b.sessions[sid2]["last_seen_at"] = time.time() - 25 * 3600
        check("C3c env=0 关闭回收（未标 recycled；放宽展示 TTL 才可见）",
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
        check("D1 相似话题被拦截",
              out.get("topics_deduped") == 1
              and len(b.open_topics) == 1)
        hits = out.get("dedup_hits") or []
        check("D2 命中回显（撞的旧条原文可见）",
              hits and hits[0].get("matched") == "tick 续传待办"
              and hits[0].get("input") == "tick 续传待办事项"
              and hits[0].get("sim", 0) >= 0.85, str(hits))
        os.environ["HIPPOCAMPUS_TOPIC_SIM_DEDUP"] = "0"
        out2 = b.session_end(summary="",
                             open_topics=["tick 续传待办事项"], client="d")
        check("D3 env=0 关闭（只剩精确去重）",
              out2.get("topics_deduped") == 0
              and len(b.open_topics) == 2
              and "dedup_hits" not in out2)
        # 精确重复在任何阈值下都拦。
        out3 = b.session_end(summary="",
                             open_topics=["tick 续传待办事项 "], client="d")
        check("D4 精确去重不受阈值影响",
              out3.get("topics_deduped") == 1)
    finally:
        os.environ.pop("HIPPOCAMPUS_TOPIC_SIM_DEDUP", None)
    # D5（验证方 2026-11-01 实测未过的真实原话对，默认阈值回归钉）：
    # 换措辞 + 换分隔符（/ vs 、）+ 添词——首版 0.85 纯 difflib 漏拦
    # （原样 ratio 0.469），归一 + bigram 双指标后 0.63 ≥ 0.60。
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
    check("D5 验证方真实原话对被拦（默认阈值）",
          out5.get("topics_deduped") == 1
          and (out5.get("dedup_hits") or [{}])[0].get("matched")
          == "TA/I/J/JM/FG/SA/UR tick 下载待用户另立项按月续传",
          str(out5.get("dedup_hits")))
    # D6 硬标识罚项：章号/卷号/占名单字之差必须放行（T2/T3 现行犯的反向钉）。
    b3 = HippoBrain(persist_dir=tmp + "_d6", embedder_backend="hash")
    b3.session_start(client="n", recent_limit=0)
    b3.session_end(summary="", open_topics=["示例小说第43章情节校对"],
                   client="n")
    out6 = b3.session_end(summary="", open_topics=["示例小说第44章情节校对"],
                          client="n")
    check("D6a 章号不同不误拦",
          out6.get("topics_deduped") == 0 and len(b3.open_topics) == 2)
    out7 = b3.session_end(summary="", open_topics=["示例古籍卷五校对"],
                          client="n")
    out8 = b3.session_end(summary="", open_topics=["示例古籍卷六校对"],
                          client="n")
    check("D6b 中文数字卷号不同不误拦",
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
