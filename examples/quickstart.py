# -*- coding: utf-8 -*-
"""HippoMCP 最小可运行示例 —— 不需要下载模型，不需要任何 API Key。

演示记忆底座的四个核心动作：

    1. ingest()           写入：原始文本进 L0，萃取出的句子进 L1，并挂上实体
    2. recall()           召回：融合检索（向量 + BM25 + 时序）并标注口径冲突
    3. get_memory_context()  用层：按 5 个固定槽位装配，可直接拼进 system prompt
    4. forget()           删除：级联清掉 L2/L3/索引层，不留孤儿

运行::

    PYTHONPATH=src python examples/quickstart.py

说明：本示例刻意使用 ``embedder_backend="hash"``，好处是**不下载模型、秒级启动**，
便于快速确认整套流程能跑通。哈希嵌入不具备语义泛化能力，检索只按词面相似——
要看真实效果请换成句向量模型::

    PYTHONPATH=src python examples/quickstart.py --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile

from hippocampus_mcp.brain import HippoBrain


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(description="HippoMCP 最小可运行示例")
    ap.add_argument(
        "--model", default="",
        help="句向量模型名；留空则用 hash 后端（零下载）",
    )
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="hippocampus_quickstart_")
    backend = "sentence_transformer" if args.model else "hash"
    print(f"临时记忆库目录: {workdir}")
    print(f"嵌入后端      : {backend}" + (f" ({args.model})" if args.model else ""))

    # 示例语料：一个虚构项目组的三次对话。全部为通用内容，不含任何私人数据。
    corpus = [
        ("设计评审定下来用 PostgreSQL，不用 MongoDB，理由是事务和报表。", "会议纪要.md"),
        ("张工的邮箱是 zhang@example.com，负责后端。", "团队花名册.md"),
        ("部署窗口改到每周三凌晨，运维说周二有例行维护。", "会议纪要.md"),
        ("之前定的部署窗口是周一，后来改成周三了。", "会议纪要.md"),
    ]

    brain = HippoBrain(
        persist_dir=workdir,
        embedder_backend=backend,
        embedder_model=args.model,
        auto_save=False,
    )

    try:
        # ---------------- 1. 写入 ----------------
        banner("1. ingest —— 写入并萃取")
        for text, source in corpus:
            r = brain.ingest(text, source=source)
            n = len(r.get("memory_ids") or r.get("memories") or [])
            print(f"  + [{source}] {text[:34]}…  → 新增 {n} 条 L1")

        st = brain.stats()
        print(f"\n  L0 原文: {st.get('L0_raw')} 条   L1 记忆: {st.get('L1_memories')} 条   "
              f"实体: {st.get('entities')}   嵌入模型: {st.get('embed_model')}")

        # ---------------- 2. 召回 ----------------
        banner("2. recall —— 融合检索")
        for q in ("部署窗口是什么时候", "谁负责后端"):
            hits = brain.recall(q, top_k=3)
            print(f"\n  查询: {q}")
            if not hits:
                print("    （无命中）")
            for h in hits:
                score = h.get("score", h.get("similarity", 0.0))
                flags = " ".join(
                    k for k in ("conflict_group", "newer", "superseded") if k in h
                )
                print(f"    {score:.3f}  {str(h.get('content', ''))[:44]}  {flags}")

        # ---------------- 3. 用层装配 ----------------
        banner("3. get_memory_context —— 5 槽位装配")
        ctx = brain.get_memory_context("准备写部署文档", top_k=5)
        blocks = ctx.get("blocks", {})
        print("  槽位装载情况:")
        for key, val in blocks.items():
            print(f"    {key:24s} {len(val)} 项")
            for item in list(val)[:2]:
                print(f"        - {str(item)[:58]}")
        print(f"\n  估算 token: {ctx.get('token_estimate')}")
        print("  ---- 可直接拼进 system prompt 的文本 ----")
        for line in (ctx.get("text") or "（空）").splitlines()[:12]:
            print(f"    {line}")

        # ---------------- 4. 级联删除 ----------------
        banner("4. forget —— 级联删除")
        victim = brain.recall("张工的邮箱", top_k=1)
        if victim:
            mid = victim[0].get("id") or victim[0].get("memory_id")
            rep = brain.forget(mid, reason="quickstart 演示：级联删除")
            print(f"  删除 {mid}")
            print(f"  级联报告: {rep}")
            left = brain.recall("张工的邮箱", top_k=1)
            print(f"  删除后同查询命中: {len(left)} 条")

        banner("完成")
        print("  全部流程跑通。真实使用时把 persist_dir 指向你的记忆库目录即可。")
        print("  详见 docs/ARCHITECTURE.md")
        return 0

    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
