# -*- coding: utf-8 -*-
"""意图感知查询路由：查询意图的零依赖词面分类 + 权重路由表.

动机：不同意图的查询对「新鲜度 vs 相似度 vs 多样性」的需求
不同——事实 Lookup 要相似度主导、多样性惩罚更狠；时间型问题要新鲜度；
探索型要放松多样性惩罚。现行单一权重组合对所有意图一视同仁。

诚实边界：词面规则是**便宜的第一版**，中文写作查询里准确率有限；它的价值
是把「意图 → 权重」这条路由先立起来（骨架 + 开关 + 自曝），将来换分类器
只动 :func:`classify_intent` 一处。

开关：``HIPPOCAMPUS_INTENT_ROUTING``（默认 0 = 关）。关闭时 manager.recall
连分类都不调，行为逐位不变——这是评测基线不受影响的硬保证。
"""

from __future__ import annotations

import os

# 构造注入缝：旋钮登记在 config（intent_routing），生产路径由 engine
# import 时灌入；解析序（env 先、注入值回退）见 common/config_bridge。
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

# 意图常量（general = 兜底，路由不动权重，与关闭时等价）
INTENT_TEMPORAL = "temporal"
INTENT_LOOKUP = "lookup"
INTENT_EXPLORE = "explore"
INTENT_GENERAL = "general"

# 词面触发词（中文写作场景为主；命中计数最多者胜，并列/全零 → general）
_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    INTENT_TEMPORAL: (
        "什么时候", "何时", "哪年", "哪月", "哪天", "哪一年", "多久",
        "最近", "之前", "后来", "先后", "顺序", "时间线", "历程",
    ),
    INTENT_LOOKUP: (
        "是谁", "是什么", "名字", "叫什么", "在哪里", "在哪儿", "哪个",
        "多少", "几个", "数量",
    ),
    INTENT_EXPLORE: (
        "为什么", "为何", "怎么", "如何", "原因", "解释", "分析", "影响",
    ),
}


def intent_routing_enabled() -> bool:
    """意图路由是否开启（每次读 env，便于测试/运行时切换）。默认关。"""
    return _resolve(
        "HIPPOCAMPUS_INTENT_ROUTING", "intent_routing", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def classify_intent(query: str) -> str:
    """按词面规则把查询分到四类意图之一。

    计分 = 各意图触发词的命中次数（同一词多次命中只计一次）；
    最高者胜，全零或并列 → :data:`INTENT_GENERAL`（兜底不动权重）。
    """
    if not query:
        return INTENT_GENERAL
    scores: dict[str, int] = {}
    for intent, words in _INTENT_PATTERNS.items():
        scores[intent] = sum(1 for w in words if w in query)
    best = max(scores.values())
    if best <= 0:
        return INTENT_GENERAL
    winners = [k for k, v in scores.items() if v == best]
    return winners[0] if len(winners) == 1 else INTENT_GENERAL


# 路由表：对 ScoringEngine **当前权重**的乘子（不是绝对值——库/配置可能
# 自定义了权重，乘子语义保证路由在任意基线上都成立）。
# general = 空表 = 与关闭时同权重。
INTENT_WEIGHT_MULTIPLIERS: dict[str, dict[str, float]] = {
    INTENT_TEMPORAL: {"w_time": 4.0, "w_recall": 0.5},
    INTENT_LOOKUP: {"w_sim": 1.05, "w_recall": 1.5},
    INTENT_EXPLORE: {"w_recall": 0.5},
    INTENT_GENERAL: {},
}
