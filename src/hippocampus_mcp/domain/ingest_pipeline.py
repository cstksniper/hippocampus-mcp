# -*- coding: utf-8 -*-
"""``brain.core.ingest`` 的五段内联逻辑（从 core 抽取）。

**为什么值得搬**：``ingest`` 是宿主最常用的写入口，壳里除了「写 L0 / 链接
实体 / 调 manager / 记账」这些真正需要 brain 状态的事，还缠着五块纯逻辑——
auto_tags 合并（6 行）、认知类型与重要性判定（7 行）、分句与 memory_type
（11 行）、kept_separate 邻居标注（10 行）、回显体组装（33 行）。
它们全是纯 dict/list 运算，却因为散落在 self 面的上下文里而无法单测。
搬出来后，``ingest`` 只剩一条直线：记账 → 写 L0 → 判定 → 分句 → 写循环 →
检测 → 邻居 → 回显。

**依赖方向**：本模块只 import 标准库（domain → 仅 common 的纪律这里连
common 都没用上——所有回调与常量均由壳在调用时现取后注入）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``resolve_tags`` 的短路次序：``auto_tags`` 为假、抽取结果为空、抽取结果
   非空三种形态都走**同一个表达式**（``merge_tag_dicts(tags, auto)``）——
   包括 ``tags=None`` 且 ``auto=None`` 时返回 None（不是 ``{}``）。
2. ``guess_basics`` 的 ``explicit_importance`` 只看入参是不是 None，
   与最终值无关；``importance`` 传 0 也算法定显式（0 is not None）。
3. ``split_sentences`` 的 chunk 分支对全空白文本产出**空列表**（不写 L1），
   sentence 分支则把空串交给 ``_split_facts``（它的既有行为）。
4. ``annotate_neighbors`` 只补三个键、不动既有键；命中数即返回值。
   没有 kept_separate 时整段跳过（连 neighbors 都不遍历）。
5. ``assemble_ingest_result`` 的键序是有意的（zcode 追加 16）：
   ``merged_into`` 必须第一个——JSON 保序，顺序传达优先级。
6. ``truncation`` 子体的 ``truncated`` 恒存在（不只在截断时才给），
   ``max_seq_length`` 由壳现取后传入（0 = 未知）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ======================================================================
# 段 1：auto_tags 合并（旧壳 874-878）
# ======================================================================
def resolve_tags(
    tags: dict[str, Any] | None,
    *,
    auto_tags: bool,
    text: str,
    extract: Callable[[str], dict[str, Any] | None],
    merge: Callable[[dict[str, Any] | None, dict[str, Any] | None],
                    dict[str, Any] | None],
) -> dict[str, Any] | None:
    """把自动抽取的标签并进入显式 tags（显式优先，只补空缺键）。

    Args:
        tags: 调用方显式传入的标签（可为 None）。
        auto_tags: 开关。为假时**连抽取都不做**（省一次正则扫描）。
        text: 正文（抽标签的语料）。
        extract: ``brain.tag_extractor.extract`` 本体。
        merge: ``store.memory.merge_tag_dicts`` 本体（保 left，
            即显式传入的值不被自动结果覆盖）。

    Returns:
        合并后的标签；``auto_tags`` 关、抽取为空、或两侧皆空时返回入参
        ``tags`` 原值（None 原样返回，见模块 docstring 怪癖 1）。
    """
    if auto_tags:
        # 显式传入的优先：调用方知道的东西比正则准（merge 保 left）
        auto = extract(text or "")
        if auto:
            tags = merge(tags, auto)
    return tags


# ======================================================================
# 段 2：认知类型与重要性判定（旧壳 902-908）
# ======================================================================
@dataclass(frozen=True)
class IngestBasics:
    """认知类型 / 重要性的归一结果。

    Attributes:
        cognitive_type: 最终采用的值（调用方显式传入的优先）。
        importance: 最终采用的值（None 时走自动判定）。
        explicit_importance: 调用方是否显式传了 importance——
            写循环据此决定要不要再跑一次句级判定（块级只用于摘要）。
    """
    cognitive_type: str
    importance: float
    explicit_importance: bool


def guess_basics(
    text: str,
    speaker: str,
    cognitive_type: str | None,
    importance: float | None,
    *,
    guess_cognitive: Callable[[str, str], str],
    guess_importance: Callable[[str], float],
) -> IngestBasics:
    """认知类型自动判定 + 重要性自动判定（块级，仅用于返回摘要值）。

    复刻怪癖：``explicit_importance`` 只看入参是不是 None——传 0 也算法定
    显式（见模块 docstring 怪癖 2）。
    """
    if cognitive_type is None:
        cognitive_type = guess_cognitive(text, speaker)
    explicit_importance = importance is not None
    if importance is None:
        importance = guess_importance(text)
    return IngestBasics(
        cognitive_type=cognitive_type,
        importance=importance,
        explicit_importance=explicit_importance,
    )


# ======================================================================
# 段 3：分句与 memory_type（旧壳 919-931）
# ======================================================================
def split_sentences(
    text: str,
    granularity: str,
    *,
    split_facts: Callable[[str], list[str]],
) -> list[str]:
    """按粒度把原文切成 L1 候选句。

    Args:
        text: 原始正文。
        granularity: ``"chunk"`` = 整段一条；其余（含 ``"sentence"``）
            走分句器。
        split_facts: ``brain._split_facts`` 本体（短句并入上一条等怪癖
            都在它里面，本函数不重复实现）。

    Returns:
        候选句列表。chunk 粒度下全空白文本返回**空列表**（不写 L1）；
        sentence 粒度下空串原样交给 ``split_facts``（见模块 docstring 怪癖 3）。
    """
    if granularity == "chunk":
        stripped = text.strip()
        return [stripped] if stripped else []
    return split_facts(text)


def memory_type_for(cognitive_type: str, experience_value: str) -> str:
    """认知类型 → memory_type 映射（CognitiveType 住在 store 包，domain
    看不见，按值注入——循环不变量，只算一次）。

    Returns:
        ``"episodic"``（经验类）或 ``"semantic"``（其余一切）。
    """
    return "episodic" if cognitive_type == experience_value else "semantic"


# ======================================================================
# 段 4：kept_separate 邻居标注（旧壳 970-979）
# ======================================================================
def annotate_neighbors(
    neighbors: list[dict[str, Any]],
    kept_separate: list[dict[str, Any]],
) -> int:
    """C10：把「因含新硬信息而故意没合并」的标记补到 neighbors 上。

    宿主就算只看 neighbors 也能当场发现：这两条很像，但不是重复写入。

    Args:
        neighbors: 写入侧回显的 top3 近邻（**就地修改**）。
        kept_separate: 写循环产出的「未合并」清单。

    Returns:
        实际标注了几条（便于测试与自曝）。

    复刻怪癖：只补三个键（``hard_info_kept_separate`` /
    ``missing_hard_info`` / ``new_memory_id``），不动既有键；
    ``kept_separate`` 为空时整段跳过——连 neighbors 都不遍历
    （见模块 docstring 怪癖 4）。
    """
    if not kept_separate:
        return 0
    ks = {k["memory_id"]: k for k in kept_separate}
    n = 0
    for nb in neighbors:
        hit = ks.get(nb.get("memory_id"))
        if hit is not None:
            nb["hard_info_kept_separate"] = True
            nb["missing_hard_info"] = hit["missing_hard_info"]
            nb["new_memory_id"] = hit["new_memory_id"]
            n += 1
    return n


# ======================================================================
# 段 5：回显体组装（旧壳 992-1024）
# ======================================================================
def assemble_ingest_result(
    *,
    merged_into: list[dict[str, Any]],
    raw_id: str,
    memory_ids: list[str],
    entities: list[str],
    cognitive_type: str,
    importance: float,
    superseded: list[Any],
    tags: dict[str, Any] | None,
    neighbors: list[dict[str, Any]],
    kept_separate: list[dict[str, Any]],
    l0_pending: int,
    truncations: list[dict[str, Any]],
    max_seq_length: int,
) -> dict[str, Any]:
    """组装 ingest 的返回体（键序即优先级，见模块 docstring 怪癖 5）。

    Args:
        merged_into: 命中去重被并入的既有记忆。**非空 = 本次未新建**，
            排第一个是 zcode 追加 16 的决定。
        raw_id: 本次 L0 记录 id。
        memory_ids: 本次实际写出的 L1 id 列表。
        entities: 归一后的实体名列表（别名已归并）。
        cognitive_type / importance: 最终采用的值。
        superseded: 被本批取代的旧记忆（可为空列表）。
        tags: 实际挂上的标签（含 auto_tags 抽出来的）；回显是拷贝。
        neighbors: 近邻回显（段 4 可能已就地标注）。
        kept_separate: 未合并的近重复（与 merged_into 互斥）。
        l0_pending: 当前 L0 欠账数（恒存在）。
        truncations: 截断明细（空列表 = 没截断）。
        max_seq_length: 嵌入模型 token 上限（0 = 未知）。

    Returns:
        回显体。``tags`` 是 ``dict(tags or {})`` 的拷贝；
        ``truncation.truncated`` 恒存在（见模块 docstring 怪癖 6）。
    """
    return {
        # 写入侧回显：命中去重被并入的既有记忆。**非空 = 本次未新建**。
        "merged_into": merged_into,
        "raw_id": raw_id,
        "memory_ids": memory_ids,
        "entities": entities,
        "cognitive_type": cognitive_type,
        "importance": importance,
        "superseded": superseded,
        # 本批实际挂上的标签（含 auto_tags 抽出来的）——回显出来，
        # 否则"我开了 auto_tags 到底抽到什么"只能靠再查一次库
        "tags": dict(tags or {}),
        "neighbors": neighbors,
        # 去重守卫（方案 B）：命中去重但因含新数字/日期而
        # **未合并**的近重复。非空 == 这些记忆是新建的（与 merged_into 互斥）。
        "kept_separate": kept_separate,
        # 还有几条新 raw 没落到 brain_raw.json。恒存在，
        # 否则调用方分不清「确定没欠账」与「压根没查」。
        "l0_pending": l0_pending,
        # C13：截断自曝。**恒存在**（不只在截断时才给），否则调用方分不清
        # 「确定没截断」与「压根没查」。truncated=True 时 items 里
        # embedded_chars 之后的原文对向量检索不可见 —— 必须拆卡。
        "truncation": {
            "truncated": bool(truncations),
            "max_seq_length": max_seq_length,
            "items": truncations,
        },
    }
