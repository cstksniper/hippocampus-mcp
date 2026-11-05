# -*- coding: utf-8 -*-
"""HippoBrain v0.3 冒烟 demo —— 不需要 MCP 宿主，直接跑。

演示一个智能体记忆底座该做的五件事：
    1. 存：原始对话写入 L0 + 萃取进 L1 + 实体入索引
    2. 想起：跨会话召回（换一天再问，还能想起来）
    3. 知识更新：事实被取代时标 superseded，旧记忆保留
    4. 用：5 槽位上下文（可直接拼进 system prompt）
    5. 整理：巩固任务（去重/取代/沉寂/矛盾）+ 画像沉淀

运行：
    python scripts/demo_brain.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# 从脚本自身位置推导项目根（scripts/ 的上一级），不写死绝对路径——
# 否则 clone 到别的机器上第一件事就是改路径。
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from hippocampus_mcp.brain import HippoBrain  # noqa: E402

# 演示库目录。`data/` 已在 .gitignore 里，所以跑 demo 不会污染仓库。
# 想放到别处：设 HIPPOCAMPUS_PERSIST_DIR 环境变量即可。
# ⚠️ 这里是**每次新建的临时目录**，且脚本全程不做任何删除操作。
# 旧版本会 rmtree 一个固定目录——一旦用户把 HIPPOCAMPUS_PERSIST_DIR
# 指向自己的真实记忆库，跑一次 demo 就把库清空了。demo 无权删除用户数据，
# 所以改成"只新建、不删除"：想看持久化就把下面的路径打印出来自己去看。
DEMO_DIR = os.environ.get("HIPPOCAMPUS_PERSIST_DIR") or tempfile.mkdtemp(
    prefix="hippocampus_demo_"
)


def line(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> None:
    # 不做任何删除：DEMO_DIR 每次都是新建的临时目录（见文件头说明）。
    print(f"演示库目录：{DEMO_DIR}")
    print("初始化 HippoBrain（首次会加载嵌入模型，稍等）...")
    brain = HippoBrain(persist_dir=DEMO_DIR)
    day = 86400.0
    t0 = time.time() - 30 * day  # 30 天前开始

    # ── 会话 1：认识用户 ────────────────────────────────
    line("会话 1（30 天前）：认识用户")
    s1 = brain.session_start(task="了解用户")
    print("预激活包（首次为空/极少）:")
    for k, v in s1.items():
        print(f"  {k}: {v}")

    brain.ingest("我叫陈默，在深圳做跨境电商，主要做家居品类。",
                 speaker="user", timestamp=t0, entities=["陈默", "深圳", "家居品类"])
    brain.ingest("我养了一只叫豆豆的柯基，每天早晚各遛一次。",
                 speaker="user", timestamp=t0 + 60, entities=["豆豆"])
    brain.ingest("记住：我对花生过敏，任何方案里都不要出现花生相关的产品。", speaker="user", timestamp=t0 + 120)
    brain.ingest("帮我记住下个月要给供应商付尾款。", speaker="user", timestamp=t0 + 180)
    brain.track_intent("给供应商付尾款", due="下个月")
    print(f"\n写入后: {brain.stats()}")

    brain.session_end(
        summary="认识用户：陈默，深圳跨境电商（家居），养柯基豆豆，花生过敏，有供应商尾款待办",
        open_topics=["家居品类的选品方向还没聊"],
    )

    # ── 会话 2：跨会话想起 ──────────────────────────────
    line("会话 2（15 天后）：跨会话还能想起来吗？")
    t2 = t0 + 15 * day
    s2 = brain.session_start(task="继续讨论家居选品")
    print("预激活包:")
    print(f"  画像: {s2['block_profile']}")
    print(f"  最近摘要: {s2['block_recent_summaries']}")
    print(f"  未完结: {s2['block_open_topics']}")
    print(f"  待办: {s2['block_pending_intents']}")

    print("\n问：用户是做什么的？有什么禁忌？")
    ctx = brain.get_memory_context("用户做什么工作，有什么过敏或禁忌", top_k=3)
    print(ctx["text"])

    # ── 会话 3：知识更新（事实被取代）────────────────────
    line("会话 3（今天）：知识更新 — 事实被取代时怎么处理")
    t3 = time.time()
    brain.ingest("我以前养的是柯基豆豆，现在养的是一只叫年糕的布偶猫。",
                 speaker="user", timestamp=t3, entities=["豆豆", "年糕"])
    brain.ingest("我们不再做家居品类了，现在主要做宠物用品。",
                 speaker="user", timestamp=t3 + 60, entities=["家居品类", "宠物用品"])

    print("\n问：用户现在养什么宠物？")
    ctx3 = brain.get_memory_context("用户现在养什么宠物", top_k=4)
    print(ctx3["text"])
    print("\n命中记忆的状态（注意 superseded 的旧事实仍在库里但被标注）:")
    for h in ctx3["hits"]:
        print(f"  [{h['state']:<10s}] sim={h['similarity']:.3f} imp={h['importance']:.2f} "
              f"pin={h['pin']} | {h['content'][:50]}")

    brain.session_end(summary="用户转做宠物用品，宠物换成布偶猫年糕")

    # ── 巩固 ─────────────────────────────────────────
    line("巩固任务（睡眠修剪）")
    print(brain.consolidate())

    # ── 最终状态 ──────────────────────────────────────
    line("最终状态")
    st = brain.stats()
    for k, v in st.items():
        print(f"  {k}: {v}")

    # ── 持久化 & 重载验证 ──────────────────────────────
    line("持久化 / 重载验证")
    brain.save()
    brain2 = HippoBrain(persist_dir=DEMO_DIR)
    print(f"重载后: L0={len(brain2.raw_log)} 条, L1={brain2.manager.store.count()} 条, "
          f"L2={len(brain2.summaries)} 条, 实体={len(brain2.entities)} 个")
    r = brain2.recall("用户现在做什么品类", top_k=2)
    print("重载后召回测试 →", [x["content"][:40] for x in r])

    print("\n✅ demo 完成。数据目录:", DEMO_DIR)


if __name__ == "__main__":
    main()
