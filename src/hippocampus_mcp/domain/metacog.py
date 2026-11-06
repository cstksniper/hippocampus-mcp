# -*- coding: utf-8 -*-
"""O7：元记忆置信度外显（domain 纯函数，M2 纪律）。

生物对应：元认知监测——海马体不只给出「想起什么」，还给出「这个想起
有多大把握」（知道自己"记不清"）。

现状缺口：检索结果不带逐块置信度，宿主无法区分「确证」与「似是而非」
——与反幻觉目标（L0 回指可核）不配套的最后一公里。

置信度公式（**校准锚点全部用仓内既有常量**，不另立标尺）：

    conf = W_SIM   · min(sim / SUPERSEDE_SIM, 1)     # 0.5 —— 相似度：
                                                          达到"同一事实"
                                                          档（0.72）即满分
         + W_STATE · state_score                      # 0.2 —— 生命周期：
                                                          active 1.0 /
                                                          superseded 0.2 /
                                                          archived 0.0
         + W_SRC   · source_resolved_ratio            # 0.2 —— L0 可解性：
                                                          source_ids 里能在
                                                          raw_log 对上的
                                                          比例；**无溯源 = 0**
         + W_PIN   · (1.0 if pin else 0.0)            # 0.1 —— 钉住 =
                                                          被确认过

校准合理性（验收抽查的两条，公式保证恒成立）：

* pinned 高于普通：同 sim / state / 溯源下 pin 项 +0.1；
* 无溯源低于有溯源：``source_ids`` 为空时 W_SRC 项记 0，有且可解时记满。

BM25 单路命中（``similarity`` 恒 0.0）自然落在低置信区——字面撞上但
向量无法确证，正是"似是而非"该有的分数。

自曝承收据纪律：返回体带 ``confidence`` 收据
块（各块 value / basis / 全体 inputs / weights / computed_at），回答
"这些把握是怎么算出来的"；逐条命中在 ``hits[i]["confidence"]`` 上。
**不加新工具、不改既有键**——只添新键，向后兼容。

依赖方向：只 import common 常量（L0）。
"""
from __future__ import annotations

from typing import Any, Iterable

from ..common.constants import SUPERSEDE_SIM

__all__ = [
    "W_SIM", "W_STATE", "W_SRC", "W_PIN", "STATE_SCORE", "FIRST_PARTY_SCORE",
    "hit_confidence", "mean_confidence", "confidence_receipt",
]

W_SIM: float = 0.5
W_STATE: float = 0.2
W_SRC: float = 0.2
W_PIN: float = 0.1

#: 生命周期状态分。superseded 不是 0：被取代的旧口径仍是"真实说过的话"
#: （历史事实引用价值），只是不能当当前事实用。
STATE_SCORE: dict[str, float] = {
    "active": 1.0,
    "superseded": 0.2,
    "archived": 0.0,
}

#: 每个非空块的基础分：block_2/3/5 出自 first-party 层（intents /
#: summaries / open_topics 是本系统自己记的账，不是检索来的），给满 1.0
#: 并在 basis 里写明——"确凿"与"待核"的区分才有意义。
FIRST_PARTY_SCORE: float = 1.0


def hit_confidence(
    *,
    similarity: float,
    state: str,
    source_ids: Iterable[str],
    source_resolved: int,
    pin: bool,
) -> float:
    """单条命中的置信度（0~1，四位小数）。入参口径见模块 docstring。

    Args:
        similarity: 向量相似度（BM25 单路命中恒 0.0——按"无法确证"计）。
        state: 生命周期状态（未知状态按 0.5 计，不武断加分或清零）。
        source_ids: L0 回指 id 列表（空 = 无溯源，该项 0 分）。
        source_resolved: 其中能在 L0 raw_log 对上的个数。
        pin: 是否钉住。
    """
    sim_n = max(0.0, min(float(similarity) / SUPERSEDE_SIM, 1.0))
    state_score = STATE_SCORE.get(str(state or ""), 0.5)
    ids = [s for s in (source_ids or []) if s]
    if not ids:
        src_score = 0.0
    else:
        src_score = max(0.0, min(float(source_resolved) / len(ids), 1.0))
    v = (W_SIM * sim_n + W_STATE * state_score
         + W_SRC * src_score + W_PIN * (1.0 if pin else 0.0))
    return round(min(max(v, 0.0), 1.0), 4)


def mean_confidence(values: Iterable[float]) -> float | None:
    """逐条置信度的块级汇总（算术平均，四位小数；空集 = None）。

    平均而非取 max：块级"把握"要为**整块**背书，一条似是而非的命中混进
    确证命中里，块级就该掉分——宿主据底下沉到逐条 ``confidence`` 查是谁。
    """
    vals = [float(v) for v in values]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def confidence_receipt(
    block_values: dict[str, dict[str, Any]],
    *,
    computed_at: float,
) -> dict[str, Any]:
    """K4 收据式的置信度自曝块（get_memory_context 返回体的新键）。

    Args:
        block_values: ``{块名: {"value": float|None, "basis": str}}``。
        computed_at: 计算时刻（Unix 秒）。

    Returns:
        ``{"blocks": ..., "inputs": [...], "weights": ..., "per_hit_key":
        "confidence", "computed_at": ..., "note": ...}``。
    """
    return {
        "blocks": dict(block_values),
        "inputs": ["similarity", "state", "source_ids_resolvable", "pin"],
        "weights": {"similarity": W_SIM, "state": W_STATE,
                    "source_ids_resolvable": W_SRC, "pin": W_PIN},
        "per_hit_key": "confidence",
        "computed_at": computed_at,
        "note": ("value=None = 该块不可评估（空块 / 派生层无法逐条溯源），"
                 "不是 0 分。block_1 为逐条命中的平均；校准锚点：相似度以"
                 "SUPERSEDE_SIM（同一事实档）为满分线，pinned 高于普通、"
                 "无溯源低于有溯源。"),
    }
