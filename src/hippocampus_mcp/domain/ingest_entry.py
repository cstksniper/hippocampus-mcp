# -*- coding: utf-8 -*-
"""M5 SessionC 批C1：``brain.core.ingest`` 的剩余内联纯段（从 core 抽取）。

**背景**：``ingest`` 的五块大纯逻辑已搬进 ingest_flow /
ingest_pipeline，壳是一条「记账 → 写 L0 → 判定 → 分句 → 写循环 → 检测 →
邻居 → 回显」的直线。本批把直线上**还剩的三小段纯计算**也搬出来——
它们各自踩着一个此前没有独立单测钉住的语义怪癖（时钟 None 语义 /
session_id 短路 / supersede 三态），此前回归只能靠整条 ingest 的端到端
parity 兜底；搬出来后怪癖有了自己的断言面。

**为什么壳不再继续瘦**：``ingest`` 被几乎所有写入侧套件直调（侦察：
31 个测试文件 111 处 ``.ingest(``），剩余行几乎全是「状态写副作用 +
组合点」——``raw_log.append`` / ``record_source_sig`` / ``_guard_map_entry_dup``
/ ``bump_daily`` 是状态写，五个 domain 模块的调用组装因「domain 同层
禁互引」必须留在壳里当组合点。

**依赖方向**：本模块只 import 标准库（domain → 仅 common 的纪律这里连
common 都没用上——时钟与会话表两个依赖由壳在调用时现取后注入）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``resolve_timestamp`` 的 None 语义：**只有 ``timestamp is None`` 才取
   时钟**——传 0.0 也是显式值（历史回放的纪元时间戳真实存在，不能被
   "补当前时间"吞掉）；负数同理。
2. ``build_raw_entry`` 的 session_id 短路：显式传入的 session_id 为真值时
   **不查**会话表（``resolve_session`` 不被调）；为假值（None / 空串）才
   回落 ``current_session_of(client)``——回落结果为 None 时**不再兜底**，
   session_id 就是 None。
3. ``build_raw_entry`` 的 source 兜底：``source or ""``——None 与空串
   同形（L0 条目里 source 恒为字符串）。
4. ``resolve_supersede_switch`` 的三态：``None`` = 跟随实例开关
   ``supersede_enabled``；True/False 原样采用——批量导入逐条传 False
   时，不得被实例开关翻上去。
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# 段 1：入站时间戳的 None 语义（旧壳 876 行）
# ======================================================================
def resolve_timestamp(
    timestamp: float | None,
    *,
    now: Callable[[], float],
) -> float:
    """入站时间戳归一：``None`` 才取时钟，其余（含 0.0）显式采用。

    Args:
        timestamp: 调用方显式传入的时间戳（可为 None）。
        now: ``brain_core._now`` 本体（壳调用时现取——monkeypatch 与
            parity 冻结都打在它上，注入保持这层可换）。

    Returns:
        显式时间戳或时钟当前值。
    """
    return timestamp if timestamp is not None else now()


# ======================================================================
# 段 2：L0 raw 记录组装（旧壳 894-904 行；append 状态写留壳）
# ======================================================================
def build_raw_entry(
    *,
    raw_id: str,
    text: str,
    speaker: str,
    timestamp: float,
    session_id: str | None,
    source: str | None,
    resolve_session: Callable[[str | None], str | None],
    client: str | None,
) -> dict[str, Any]:
    """组装 L0 原始记录 dict（六键，键序即 ``raw_log`` 条目形状）。

    Args:
        raw_id: ``_nid("raw")`` 产物（uuid 有副作用，由壳生成后传入）。
        text / speaker / timestamp: 原样透传，不拷贝不变换。
        session_id: 显式会话 id；为真值时**不查**会话表（怪癖 2）。
        source: 来源标签；``or ""`` 兜底（怪癖 3）。
        resolve_session: ``brain.current_session_of`` 本体。
        client: 会话归属的客户端标识（原样转给 resolve_session）。

    Returns:
        六键 dict。append 由壳做——本函数只组装，不碰状态。
    """
    return {
        "id": raw_id,
        "text": text,
        "speaker": speaker,
        "timestamp": timestamp,
        "session_id": session_id or resolve_session(client),
        "source": source or "",
    }


# ======================================================================
# 段 3：supersede 开关三态归一（旧壳 966-969 行；检测调用留壳）
# ======================================================================
def resolve_supersede_switch(
    detect_supersede: bool | None,
    *,
    instance_enabled: bool,
) -> bool:
    """事实更新检测的三态开关归一（怪癖 4）。

    Args:
        detect_supersede: 本次调用的三态开关（None = 跟随实例）。
        instance_enabled: ``brain.supersede_enabled`` 实例开关。

    Returns:
        是否对本次写入跑取代检测。
    """
    return instance_enabled if detect_supersede is None else detect_supersede


# ======================================================================
# 段 4：幂等重放返回体（旧壳 925-934 行；查 idem 的状态读留壳）
# ======================================================================
def duplicate_body(memory_id: str) -> dict[str, Any]:
    """同一 ``idem_key`` 重放时的返回体。

    **为什么单独抽出来**：``brain.core.ingest`` 有 S1 结构守卫
    （``tests/test_m5_batch19_ingest_flow.py``）钉着壳 ≤170 行（去
    docstring 计体）。这 10 行是纯字典构造、不含任何状态写，是壳里
    唯一还剩的整块纯计算——搬走后壳只留「查 → 回」两行。

    键名、顺序与措辞逐字保持（``tests/test_p0_origin_idem.py`` 与
    parity 套件按原样断言，改一个字都算行为变化）。

    Args:
        memory_id: 命中的既有记忆 id（``manager.idem_lookup`` 的返回值）。

    Returns:
        幂等重放的标准返回体。
    """
    return {
        "status": "duplicate",
        "idempotent": True,
        "memory_id": memory_id,
        "memory_ids": [memory_id],
        "raw_id": None,
        "note": ("同一 idem_key 的重放：本次未写 L0 / L1，直接回既有"
                 "记忆 id。若这次其实是**新内容**，说明宿主复用了键"
                 "——请换键重发（幂等键应按请求唯一，不按内容）。"),
    }
