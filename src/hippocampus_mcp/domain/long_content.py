# -*- coding: utf-8 -*-
"""超长内容的处置策略：**纯判定与纯切分**，不落盘、不读配置。

## 起因（实测，不是推测）

嵌入模型有 token **硬上限**（BGE / MiniLM = 512）。3372 字的卡实测
``embedded_chars=511``、``dropped_ratio=0.8485``——**第 512 字之后的原文
对向量零贡献，但写入照样返回成功**。关掉 BM25 字面通道后，尾部事实
（/etc/secret/db.key）用语义问法**完全召回不到**。

这是「写进去了但检索看不见」，属于本项目最忌的静默失败。

## 三态策略（默认 = 现状，零回归）

* ``warn``（**默认**）—— 只自曝，行为与改动前逐字节一致。
* ``split`` —— 超窗时按句拆成多张卡写入，每张都进向量。
* ``reject`` —— 超窗时拒写，返回体说明怎么改。

**白名单解析**：只有明确写 ``split`` / ``reject`` 才生效，未知值一律回落
``warn``。理由同家库写边界开关：默认关闭的开关若用黑名单语义，
``''`` / ``'maybe'`` 会把它悄悄打开；这里反过来——**改变写入行为的两个档
必须由用户显式点名**，写错等同于没写。

## 为什么切分器要注入

``domain`` 只准依赖 ``common``（架构约束），而真正的分句器
（``brain._split_facts``，含「短句并入上一条」等怪癖）住在 brain。
在本模块**另写一份分句**会立刻产生第二份真相，因此：分句器由组合根注入，
取不到就退化为「按上限硬切」——降级**显式可见**（返回体带 note），不静默。

## 切分口径

1. 先按注入的分句器切句；
2. 单句仍超过上限时，在上限的 ``[1/2, 1]`` 区间内找**最后一个强断点**
   （``。！？；`` 及换行）断开；找不到就硬切。
3. 上限取 ``max_seq_length * SAFE_RATIO``（默认 0.9）：512 token 的模型
   要到 460 字就断。留出 ``[CLS]``/``[SEP]`` 与中文 1 字≈1 token 的余量
   ——**贴着 512 切会刚好再撞一次边界**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

POLICY_WARN = "warn"
POLICY_SPLIT = "split"
POLICY_REJECT = "reject"

POLICIES: tuple[str, ...] = (POLICY_WARN, POLICY_SPLIT, POLICY_REJECT)

#: 字符上限相对 token 上限的折扣（见模块 docstring 切分口径 3）。
SAFE_RATIO = 0.9

#: 单条写入最多拆出多少张卡。超了就拒写——宁可报错，也不要一次灌进去
#: 上千条把库冲垮（那种情况多半是整本书被当成一条记忆塞进来了）。
#: 200 而不是更小：``ingest`` 把一篇长文拆成上百句本来就是常态（实测
#: 100 句的长文 = 101 片），定 50 会把正常文档误判成"灌库"。
MAX_PIECES = 200

#: 强断点：断句优先落在这里（``_hard_split`` 从后往前找）。
_STRONG_BREAKS = "。！？；!?;\n"
_WEAK_BREAKS = "，、, "


@dataclass(frozen=True)
class LongContentPlan:
    """一次超长写入的处置计划（纯数据，调用方据此执行）。"""

    action: str                      # "write" | "split" | "reject"
    policy: str                      # 生效的策略（解析后）
    pieces: tuple[str, ...] = ()     # action=="split" 时的分片
    reason: str = ""                 # 为什么这么判（回显给用户，不静默）
    note: str = ""                   # 降级/边界说明
    fields: dict[str, object] = field(default_factory=dict)


def parse_policy(raw: Optional[str]) -> str:
    """把配置值解析成三态之一。**白名单**：未知值 → ``warn``（= 现状）。"""
    v = (raw or "").strip().lower()
    if v in POLICIES:
        return v
    return POLICY_WARN


def char_limit(max_seq_length: int, *, safe_ratio: float = SAFE_RATIO) -> int:
    """token 上限 → 安全的字符上限。

    ``max_seq_length <= 0``（hash 后端 / 取不到）返回 0，调用方据此**跳过
    一切干预**——没有上限就谈不上超窗。
    """
    if not max_seq_length or max_seq_length <= 0:
        return 0
    return max(1, int(max_seq_length * safe_ratio))


def _hard_split(text: str, limit: int) -> list[str]:
    """把一段超长文本按 ``limit`` 切开，尽量落在断句处。"""
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        if n - start <= limit:
            out.append(text[start:])
            break
        end = start + limit
        # 在后半段里找最后一个断点，避免切出「半句话」。
        window_lo = start + max(1, limit // 2)
        cut = -1
        for ch in _STRONG_BREAKS:
            cut = max(cut, text.rfind(ch, window_lo, end))
        if cut < window_lo:
            for ch in _WEAK_BREAKS:
                cut = max(cut, text.rfind(ch, window_lo, end))
        idx = (cut + 1) if cut >= window_lo else end
        out.append(text[start:idx])
        start = idx
    return out


def split_pieces(
    text: str,
    limit: int,
    split_facts: Optional[Callable[[str], list[str]]] = None,
) -> tuple[list[str], str]:
    """返回 ``(分片, 降级说明)``。

    分句器可用就先分句（保留 brain 侧「短句并入上一条」的怪癖）；不可用
    则整体走硬切，并在第二个返回值里说明——**降级要看得见**。
    """
    note = ""
    sents: list[str] = []
    if split_facts is not None:
        try:
            sents = [s for s in (split_facts(text) or []) if s and s.strip()]
        except Exception as exc:  # 分句器坏了不能让写入挂掉
            note = f"分句器异常（{type(exc).__name__}），已退化为按长度硬切"
            sents = []
    if not sents:
        if split_facts is None and not note:
            note = "未注入分句器，按长度硬切"
        sents = [text]

    out: list[str] = []
    for s in sents:
        if len(s) <= limit:
            out.append(s)
        else:
            out.extend(_hard_split(s, limit))
    return out, note


def plan_long_content(
    content: str,
    max_seq_length: int,
    policy_raw: Optional[str],
    *,
    truncated: bool,
    split_facts: Optional[Callable[[str], list[str]]] = None,
    safe_ratio: float = SAFE_RATIO,
) -> LongContentPlan:
    """判定一次写入该怎么写。

    Args:
        content: 待写正文。
        max_seq_length: 嵌入模型的 token 硬上限（0 = 无上限 / 取不到）。
        policy_raw: 配置原值（未解析）。
        truncated: 引擎侧已算出的精确截断结论。
        split_facts: 注入的分句器（可空）。
    """
    policy = parse_policy(policy_raw)
    limit = char_limit(max_seq_length, safe_ratio=safe_ratio)

    # 没有上限信息，或确实没超窗 → 原样写（策略不介入）。
    if limit <= 0 or not truncated:
        return LongContentPlan(action="write", policy=policy,
                               pieces=(content,))

    if policy == POLICY_WARN:
        return LongContentPlan(
            action="write", policy=policy, pieces=(content,),
            reason=f"内容超过嵌入模型 token 上限 {max_seq_length}，"
                   f"其后的原文检索不到（当前策略 warn：只告警不改写）")

    if policy == POLICY_REJECT:
        return LongContentPlan(
            action="reject", policy=policy,
            reason=f"内容超过嵌入模型 token 上限 {max_seq_length}，"
                   f"当前策略 reject：已拒写。"
                   f"请拆成多张 <{limit} 字的卡，或改走 ingest（默认按句拆）")

    # split
    pieces, note = split_pieces(content, limit, split_facts)
    if len(pieces) > MAX_PIECES:
        return LongContentPlan(
            action="reject", policy=policy,
            reason=f"内容需拆成 {len(pieces)} 张卡，超过上限 {MAX_PIECES}"
                   f"（多半是整篇文档被当成一条记忆写入）。"
                   f"请先用 ingest 或自行分段",
            note=note)
    return LongContentPlan(
        action="split", policy=policy, pieces=tuple(pieces), note=note,
        reason=f"内容超过嵌入模型 token 上限 {max_seq_length}，"
               f"已拆成 {len(pieces)} 张卡分别写入")
