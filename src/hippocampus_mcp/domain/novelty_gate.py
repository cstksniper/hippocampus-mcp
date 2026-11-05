# -*- coding: utf-8 -*-
"""O5：新颖度/预测误差门控编码（domain 纯函数，M2 纪律）。

生物对应：海马体对「与既有图式一致的输入」低耗编码（印证 → 不加强，
重复归 dedup 管），对「违反预期的输入」显著强化（矛盾 → importance
提升 + 复用既有冲突标注词汇回显），对「全新刺激」小幅加权（全新实体
→ 小幅提升）。

**判据是有意的保守启发式**（生产路径不内嵌 LLM，本系统的一贯约束）：

* 矛盾判定 = 新句含**否定/更正标记词** 且 与某条高置信图式目标实词重叠
  达到 ``SUPERSEDE_OVERLAP``（0.6，与写侧 supersede / 读侧冲突标注共用的
  "同一事实"重叠档对齐——**阈值只此一处**，不另立一套）。它抓的是
  「张岱死期 1689」vs「张岱其实死于 1690」这类显式更正；隐式矛盾
  （数值不同而无更正词）交给读侧 contradictions 工具与 dedup 回显。
* 印证判定 = 重叠达标但**无**标记词 → 高耗编码没必要，importance 保持
  关键词启发式原值。
* 全新实体判定 = 分句抽出的实体名（小写）有库里从没见过的 → +小幅。
  三种结论按 矛盾 > 印证 > 全新实体 的优先级取一。

**两条边界（与既有机制的零回归保证）**：

1. 门控只调整「编码强度」——写库时的 importance——**不制造 pin**：
   base < ``PIN_IMPORTANCE`` 时提升后被钳在 ``PIN_IMPORTANCE - 0.01``，
   绝不新跨越自动钉住门槛（pin 单调堆积是 ``domain/policies.py`` 记录
   在案的教训：实测某库 78% 自动 pin）。0.7 档的画像候选 / 提升候选 /
   用户级继承槽同理一个都不会被门控新送进来。
2. dedup / supersede 判据**零接触**：合并阈值、取代检测的触发条件
   （UPDATE_MARKERS + 相似度 + is_restatement）一条都不在本模块里；
   本模块只在写循环前给出 importance 与冲突对象回显，不改变任何
   「谁取代谁 / 谁并入谁」的判定。

依赖方向：只 import common 常量（L0），符合目标架构 §3 分层。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..common.constants import (
    _GUESS_IMPORTANCE_CAP,
    SUPERSEDE_OVERLAP,
)

__all__ = [
    "CONTRADICTION_MARKERS", "HIGH_CONF_IMPORTANCE", "CONTRADICT_BOOST",
    "NOVEL_BOOST", "SCHEMA_L1_CAP", "NoveltyOutcome", "SchemaTarget",
    "build_schema_targets", "gate_sentence", "summarize_events",
]

# 否定 / 更正标记词。与 store.supersede 的 UPDATE_MARKERS 刻意**不同表**：
# 那张表混着时间迁移词（现在 / 以前 / 目前 / recently——"我现在住北京"
# 是正常陈述不是更正），本表只收「否定此前说法」语义的词，压误报。
CONTRADICTION_MARKERS: tuple[str, ...] = (
    "不再", "不是", "其实", "错了", "搞错", "不对", "弄错", "记错",
    "取消", "撤销", "作废", "收回", "纠正", "更正", "改成", "改为",
    "换成", "改成是", "no longer", "not anymore", "actually", "wrong",
    "mistake", "correction", "instead", "changed to", "replaced by",
    " rather than", "my bad",
)

#: 高置信 L1 门槛：与自动钉住门槛（``store.memory.PIN_IMPORTANCE``）、
#: 画像候选、提升候选、用户级继承槽同一档（0.7）——「图式」由被反复确认
#: 过的记忆构成，0.5 基线分的一般陈述不构成"预期"，违背它谈不上预测误差。
#: （domain 不 import store——``consolidate_flow`` / ``stats_views`` 同款
#: 纪律：门槛值由调用方注入，本常量只是无注入时的同值兜底，测试钉对齐。）
HIGH_CONF_IMPORTANCE: float = 0.7

#: 钉住门槛的同值兜底（真值以 core 侧注入的 ``store.memory.PIN_IMPORTANCE``
#: 为准；两者脱钩 = 测试抓红）。
PIN_IMPORTANCE_DEFAULT: float = 0.7

#: 矛盾判定的实词重叠档：与 ``is_restatement`` 的无实体重叠档共用
#: ``SUPERSEDE_OVERLAP``（0.6）——"同一事实"的口径只此一处。
CONTRADICT_OVERLAP: float = SUPERSEDE_OVERLAP

#: 矛盾（预测误差）提升幅度；全新实体小幅加权幅度。
CONTRADICT_BOOST: float = 0.1
NOVEL_BOOST: float = 0.05

#: 高置信 L1 比对集上限（按时间取最新的 N 条）。库大了全量比对白烧 CPU，
#: 近期高置信记忆才是"当前图式"；画像与知识点摘要全量保留（本来就小）。
SCHEMA_L1_CAP: int = 200


@dataclass(frozen=True)
class SchemaTarget:
    """一个图式比对目标（L3 画像 / L2T 知识点摘要 / 高置信 L1 各一条）。"""
    ref: str          # 回指：memory_id / "profile:<slot>" / "topic:<id>"
    kind: str         # "l1" | "l3" | "l2t"
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class NoveltyOutcome:
    """单句门控结论（``gate_sentence`` 的产物）。"""
    verdict: str                    # contradict | corroborate | novel_entity | none
    importance: float               # 调整后（未调整 = 原值）
    base_importance: float
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    novel_entities: list[str] = field(default_factory=list)


def build_schema_targets(
    *,
    profile_texts: Iterable[str],
    topic_texts: Iterable[str],
    high_conf_records: Iterable[tuple[str, str, float]],
    tokens_of: Callable[[str], set[str]],
    cap: int = SCHEMA_L1_CAP,
) -> list[SchemaTarget]:
    """装配图式比对目标集（写循环前一次性，core.ingest 注入分词器）。

    Args:
        profile_texts: L3 画像条目文本（core 侧取 ``profile[slot][i]["text"]``）。
        topic_texts: L2T 知识点摘要文本（core 侧取非空 ``summary``）。
        high_conf_records: ``(memory_id, content, importance, timestamp)``
            四元组，**core 侧已过滤** importance >= HIGH_CONF_IMPORTANCE 且
            active、按时间取最新（nlargest）——本函数只裁 cap（domain 不碰
            store 对象）。
        tokens_of: 文本 → 实词集合（brain 的 ``_tokens_of``）。
        cap: 高置信 L1 参与比对的条数上限（画像 / 摘要不占这个名额）。
    """
    targets: list[SchemaTarget] = []
    for t in profile_texts:
        t = (t or "").strip()
        if t:
            targets.append(SchemaTarget(
                ref="", kind="l3", text=t, tokens=frozenset(tokens_of(t))))
    for t in topic_texts:
        t = (t or "").strip()
        if t:
            targets.append(SchemaTarget(
                ref="", kind="l2t", text=t, tokens=frozenset(tokens_of(t))))
    n = 0
    for mid, content, _imp, _ts in high_conf_records:
        if n >= max(int(cap), 0):
            break
        content = (content or "").strip()
        if not content:
            continue
        targets.append(SchemaTarget(
            ref=mid, kind="l1", text=content, tokens=frozenset(tokens_of(content))))
        n += 1
    return targets


def _clamp_boosted(
    base: float, boosted: float, pin_importance: float
) -> float:
    """提升后的 importance：**不制造 pin**，不越过启发式上限。

    base 已 >= 钉住门槛的句子本来就处于钉住区间（0.7/0.75 档），继续抬升
    不改变 pin 状态；base < 门槛的句子被钳在门槛下方——门控的职责边界是
    「编码强度」，不是「留存策略」。
    """
    v = min(float(boosted), float(_GUESS_IMPORTANCE_CAP))
    if float(base) < float(pin_importance):
        v = min(v, float(pin_importance) - 0.01)
    return round(v, 4)


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def gate_sentence(
    text: str,
    *,
    base_importance: float,
    sentence_entities: Iterable[str],
    known_entity_keys: Iterable[str],
    targets: list[SchemaTarget],
    tokens_of: Callable[[str], set[str]],
    markers: tuple[str, ...] = CONTRADICTION_MARKERS,
    overlap_threshold: float = CONTRADICT_OVERLAP,
    contradict_boost: float = CONTRADICT_BOOST,
    novel_boost: float = NOVEL_BOOST,
    pin_importance: float = PIN_IMPORTANCE_DEFAULT,
) -> NoveltyOutcome:
    """对一条分句给出新颖度结论与调整后的 importance（纯函数）。

    判定顺序（互斥取一）：矛盾 > 印证 > 全新实体 > 不动。

    Args:
        text: 分句原文。
        base_importance: 关键词启发式给出的基准分（``guess_importance(sent)``）。
        sentence_entities: 本句抽出的实体名（小写；core 侧负责归一）。
        known_entity_keys: 写入前实体表的键集合（小写）——"全新"以**写入前**
            的库为准，本句自己的实体不算已知。
        targets: :func:`build_schema_targets` 的产物。
        tokens_of: 分词器（注入）。
        markers / overlap_threshold / 两个 boost: 缺省即模块常量，参数化
            仅为测试探边界。
        pin_importance: 钉住门槛（core 侧注入 ``store.memory.PIN_IMPORTANCE``
            ——"不制造 pin"的钳制基准；domain 不 import store）。
    """
    base = float(base_importance)
    t = (text or "").lower()
    toks = frozenset(tokens_of(text or ""))
    ents = [e.lower() for e in (sentence_entities or []) if e]
    known = {k.lower() for k in (known_entity_keys or [])}

    # ① 矛盾（预测误差）：更正标记 + 与某图式目标"同一事实"级重叠。
    #    取重叠最大的目标为冲突对象（并列取先出现的——targets 顺序即
    #    l3/l2t 在前、l1 按时间倒序，先出现者更"近期/高层"）。
    if toks and any(m in t for m in markers):
        best: tuple[float, SchemaTarget] | None = None
        for tg in targets:
            ov = _overlap(toks, tg.tokens)
            if ov >= overlap_threshold and (best is None or ov > best[0]):
                best = (ov, tg)
        if best is not None:
            ov, tg = best
            return NoveltyOutcome(
                verdict="contradict",
                importance=_clamp_boosted(
                    base, base + contradict_boost, pin_importance),
                base_importance=base,
                conflicts=[{
                    "ref": tg.ref or f"{tg.kind}:（无回指）",
                    "kind": tg.kind,
                    "text": tg.text[:200],
                    "overlap": round(ov, 4),
                    "boosted_from": base,
                    "boosted_to": _clamp_boosted(
                        base, base + contradict_boost, pin_importance),
                }],
            )

    # ② 印证（图式一致）：重叠达标、无更正标记 → 低耗编码，一分不加。
    #    （真重复走命中去重合并，与门控无关。）
    if toks:
        for tg in targets:
            if _overlap(toks, tg.tokens) >= overlap_threshold:
                return NoveltyOutcome(
                    verdict="corroborate", importance=base,
                    base_importance=base)

    # ③ 全新实体：出现库里没见过的实体名 → 新刺激小幅加权。
    fresh = [e for e in ents if e and e not in known]
    if fresh:
        return NoveltyOutcome(
            verdict="novel_entity",
            importance=_clamp_boosted(base, base + novel_boost, pin_importance),
            base_importance=base,
            novel_entities=fresh,
        )

    return NoveltyOutcome(verdict="none", importance=base, base_importance=base)


#: 单次 ingest 回显的冲突对象上限（批量导入可能命中大量目标，回显体
#: 不是审计日志——读侧 contradictions 才是全景入口）。
CONFLICTS_ECHO_CAP: int = 20


def summarize_events(
    events: list[tuple[str, NoveltyOutcome]],
) -> dict[str, Any] | None:
    """把分句事件汇总成 ingest 返回体的 ``novelty`` 键（无门控产出 = None）。

    ``None`` = 本次写入没有任何新颖度信号（全部 ``none``）或门控未跑——
    返回体零痕迹（U5/D4 同款纪律），宿主判真值即可。
    """
    if not events:
        return None
    counts = {"contradict": 0, "corroborate": 0, "novel_entity": 0, "none": 0}
    conflicts: list[dict[str, Any]] = []
    fresh_all: list[str] = []
    for _sent, out in events:
        counts[out.verdict] = counts.get(out.verdict, 0) + 1
        if out.conflicts and len(conflicts) < CONFLICTS_ECHO_CAP:
            c = dict(out.conflicts[0])
            c["sentence"] = " ".join(_sent.split())[:200]
            conflicts.append(c)
        for e in out.novel_entities:
            if e not in fresh_all:
                fresh_all.append(e)
    if counts["contradict"] == counts["corroborate"] == counts["novel_entity"] == 0:
        return None
    return {
        "verdicts": counts,
        "conflicts": conflicts,
        "novel_entities": fresh_all[:20],
        "note": ("矛盾→importance 提升（不制造 pin）；印证→不加强"
                 "（重复归 dedup 管）；全新实体→小幅加权。冲突对象回显只"
                 "是提示，裁决走 update_memory / 读侧 contradictions。"),
    }
