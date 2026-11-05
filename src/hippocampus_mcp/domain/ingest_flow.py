# -*- coding: utf-8 -*-
"""ingest 的分句写循环（从 ``brain.core.ingest`` 内联抽取）。

**为什么值得搬**：这是 ingest 主体里最大的一块内联编排（约 50 行）——
逐句调 ``manager.store_memory``，再读三个**侧信道**（``last_store_near_miss``
/ ``last_truncation`` / ``last_store_merge_of``）决定回显什么，最后按需把
新挂的实体 id 并回记忆。它缠着 L1 存储端口与 brain 的实体链接入口，
在 6299 行的 core.py 里无法脱离整个 brain 单测。

搬动**零行为变化**的保障：
1. 循环体逐字复制，只把 ``self.`` 换成注入的 ports 字段；
2. ``memory_type`` 的判定（``cognitive_type == EXPERIENCE``）是循环不变量，
   由包装层算好传入——``CognitiveType`` 住在 store 包，domain 看不着；
3. parity 探针（``_probe_out/m5_batch19_parity.py``）用 HEAD 的旧 ingest
   与新 ingest 跑同种子库，完整返回体逐字段比对，DRIFT 必须为 0。

依赖方向：本模块只 import 标准库（domain → 仅 common 的纪律这里连 common
都没用上——全部经 :class:`SentenceWritePorts` 注入）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SentenceWritePorts:
    """分句写循环的全部依赖（brain 侧注入；domain 不 import 任何上层）。

    Attributes:
        manager: L1 存储端口。用到 ``store_memory`` / ``last_store_near_miss``
            / ``last_truncation`` / ``last_store_merge_of`` / ``store.get`` /
            ``store.update`` 六个面。
        link_entities: brain 的 ``_link_entities(names, memory_id, ts, origins)``。
        merge_ids: ``store.memory.merge_id_lists``（有序并集）。
        guess_importance: brain 的 ``_guess_importance(text)``（委托到
            ``domain.policies.guess_importance``）。
    """
    manager: Any
    link_entities: Callable[..., list[str]]
    merge_ids: Callable[..., list[str]]
    guess_importance: Callable[[str], float]
    # D2a 写侧隔离（默认 False = 现状）：True 时本批每张 L1 卡带
    # quarantine 标记，且**不参与去重合并**（见 manager.store_memory）。
    # sensitivity 是整批共用一个标签（对 ingest 原文分类一次的结果，
    # 逐句重打是过度工程——待审卡本来就要人工复审，宁枉勿纵）。
    quarantine: bool = False
    sensitivity: str = ""
    # P0（Super-agent 适配）：
    #   origin  = 写入凭据 {session/task/surface}，整批共用（一次工具调用
    #             就是一个凭据单元，逐句重算没有意义）；
    #   idem_key= 幂等键，整批共用——多句拆出来的 L1 里**第一句**落键，
    #             重放时在入口就被拦住（见 brain.ingest 的幂等前置）。
    # 两者都为默认值时记录侧零变化（origin 稀疏 / idem 空串）。
    origin: dict[str, str] | None = None
    idem_key: str = ""


@dataclass(frozen=True)
class SentenceWriteResult:
    """一次分句写循环的四路产出（ingest 回显体的原料）。"""
    memory_ids: list[str] = field(default_factory=list)
    merged_into: list[dict[str, Any]] = field(default_factory=list)
    kept_separate: list[dict[str, Any]] = field(default_factory=list)
    truncations: list[dict[str, Any]] = field(default_factory=list)


def write_sentences(
    ports: SentenceWritePorts,
    *,
    sentences: list[str],
    importance: float,
    explicit_importance: bool,
    memory_type: str,
    cognitive_type: str,
    timestamp: float,
    raw_id: str,
    entity_ids: list[str],
    source: str,
    tags: dict[str, Any] | None,
    names: list[str],
    origins: dict[str, str],
) -> SentenceWriteResult:
    """逐句写入 L1，并按侧信道收集回显原料。

    行为与 ``brain.core.ingest`` 内联循环逐字一致（含三条保留语义）：

    * 重要性**按分句**判定（显式传入时整批共用块级值）；
    * ``last_store_merge_of`` 非空 = 本次内容并进了既有记忆（未新建），
      记 ``merged_into``；
    * 新挂实体与记忆既有 ``entity_ids`` 取**并集**（缺陷 D3：直接覆盖会
      让命中去重合并时丢掉此前积累的实体关联）。
    """
    manager = ports.manager
    memory_ids: list[str] = []
    merged_into: list[dict[str, Any]] = []
    # C10（方案 B）：命中去重但含新硬信息、**未合并**的那些近重复
    kept_separate: list[dict[str, Any]] = []
    # C13：本批里被嵌入模型 token 上限截断的部分
    truncations: list[dict[str, Any]] = []
    for sent in sentences:
        # 重要性**按分句**判定，而不是整块共用（覆盖报告·高 5）：
        # 块级判定会让块内一句"重要！"把几十条句子全部抬到高分区，
        # 其中一部分（>=0.7）还会被自动钉住、豁免时间衰减、灌进 L3 画像。
        sent_importance = (
            importance if explicit_importance else ports.guess_importance(sent)
        )
        mid = manager.store_memory(
            content=sent,
            importance=sent_importance,
            memory_type=memory_type,
            timestamp=timestamp,
            cognitive_type=cognitive_type,
            quarantine=ports.quarantine,
            sensitivity=ports.sensitivity,
            source_ids=[raw_id],
            entity_ids=list(entity_ids),
            source=source or "",
            tags=tags,
            origin=ports.origin,
            idem_key=ports.idem_key,
        )
        memory_ids.append(mid)
        if manager.last_store_near_miss:
            # C10：这条很像某条既有记忆，但因为带它没见过的数字/日期，
            # 按方案 B **没有合并**——本次 mid 是真正新建的。
            kept_separate.append(
                dict(manager.last_store_near_miss, new_memory_id=mid)
            )
        _tcov = manager.last_truncation or {}
        if _tcov.get("truncated"):
            # C13：写进去了，但 embedded_chars 之后的原文**检索不到**。
            truncations.append(dict(_tcov, memory_id=mid))
        if manager.last_store_merge_of:
            # 命中去重：本次内容没有新建条目，而是并进了这条既有记忆。
            # 双写场景下这是最容易踩的坑——宿主以为写进了新的一条，
            # 实际旧口径还在原地（内容取更长者、实体取并集）。
            rec = manager.store.get(mid)
            merged_into.append({
                "memory_id": mid,
                "content": (rec.content[:200] if rec else ""),
                "source": (rec.source or "") if rec else "",
                "note": "本次内容并入这条既有记忆（命中去重阈值，未新建）",
            })
        eids = ports.link_entities(names, mid, timestamp, origins)
        if eids:
            rec = manager.store.get(mid)
            if rec is not None:
                # 并集而非覆盖（缺陷 D3）：命中去重合并时 mid 是**既存**记忆，
                # 直接覆盖会把它此前积累的实体关联全部丢掉，实体倒排索引随之
                # 断裂（_recompute_entity_stats 以记录上的 entity_ids 为准，
                # 会据此把实体判成悬挂而摘除）。
                merged_eids = ports.merge_ids(rec.entity_ids, eids)
                if merged_eids != list(rec.entity_ids or []):
                    manager.store.update(mid, entity_ids=merged_eids)
    return SentenceWriteResult(
        memory_ids=memory_ids,
        merged_into=merged_into,
        kept_separate=kept_separate,
        truncations=truncations,
    )
