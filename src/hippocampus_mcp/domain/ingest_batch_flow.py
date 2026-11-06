# -*- coding: utf-8 -*-
"""``brain.core.ingest_batch`` 的三段内联逻辑（从 core 抽取）。

**为什么值得搬**：``ingest_batch`` 的壳里有三块互不相干的职责缠在一起——
逐项入参归一化（约 21 行）、逐项结果累加（约 22 行）、回显体组装（约 26 行）。
它们全是纯 dict / list 操作，却在 core.py 里因为要读 ``self.items`` 的循环
上下文而无法单测。搬出来后，``ingest_batch`` 只剩「批次副作用开关 + 循环 +
调用 :meth:`ingest` + 收尾落盘」这四件真正需要 brain 状态的事。

**刻意留在壳里的一段**：收尾落盘（``manager.save()`` → ``_save_l0_only()``
→ ``save_warning`` 文案 → ``last_save_error`` / ``autosave_failures`` 记账）。
它同时依赖五个 self 面并带三处副作用，注入端口会比原代码还长，得不偿失。

依赖方向：本模块只 import 标准库（domain → 仅 common 的纪律这里连 common
都没用上——``max_seq_length`` / ``l0_pending`` / 批次级默认值全部由壳注入）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. 容量检查的措辞带参数值（``{max_items}`` / ``{len(items)}``），且
   ``items`` 非 list 与超上限都抛 ``ValueError``——**不入 errors**，这两类
   是「调用方写错了」而不是「某项坏了」。
2. 项缺 ``tags`` 键时回落到**批次级** ``default_tags``；写了哪怕是空串，
   也走项级解析（空串会引起 ``json.loads`` 抛错，落到该项的 error）。
3. ``entities`` 的字符串分支做 ``e.strip()``，列表分支做
   ``str(e).strip()``——不对称是既有行为。
4. ``importance`` 只对 ``int/float`` 判负（``bool`` 是 int 子类，但
   ``True < 0`` 为假，行为一致）。
5. ``per_item`` 里的 ``source`` 与喂给 ``ingest`` 的是同一个表达式的结果，
   本模块只算一次存进 :attr:`ItemPlan.source_str` 复用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ItemError:
    """一项被判定「没法写」——记进 ``errors``，**不中断整批**。

    Attributes:
        index: 该项在入参里的下标（冒烟的顺序性靠它）。
        error: 人话原因。
        preview: 该项的内容预览（``str(it)[:80]`` 或 ``text[:80]``）。
    """
    index: int
    error: str
    preview: str


@dataclass(frozen=True)
class ItemPlan:
    """一项通过校验后的归一化结果。

    Attributes:
        index: 同 :attr:`ItemError.index`。
        payload: 直接展开给 ``ingest(**payload)`` 的关键字参数，键序与
            原实现逐字一致（虽然 kwargs 不看顺序，parity 比的是结果）。
        source_str: ``str(it.get("source") or source or "")`` 的结果，
            ``per_item`` 摘要要复用同一个值。
    """
    index: int
    payload: dict[str, Any]
    source_str: str


@dataclass(frozen=True)
class CollectedRow:
    """单项 ingestion 结果的八路产出。

    对应原实现里 ``raw_ids / memory_ids / all_entities / merged_into /
    kept_separate / truncations / superseded_total / per_item`` 八个累加器
    在**这一项上**的贡献。:func:`assemble_output` 负责把它们横着拼回去。
    """
    raw_id: str
    memory_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    merged: list[dict[str, Any]] = field(default_factory=list)
    kept: list[dict[str, Any]] = field(default_factory=list)
    truncations: list[dict[str, Any]] = field(default_factory=list)
    superseded_n: int = 0
    item: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 入参校验与归一化


def check_capacity(items: Any, *, max_items: int) -> None:
    """批量入口的容量闸门——不合规直接抛，*不*降级成 errors。

    Args:
        items: 待写入的项列表。
        max_items: 单批上限（core 侧是 ``INGEST_BATCH_MAX``，调用时现取传入）。

    Raises:
        ValueError: ``items`` 不是 list；或条数超过 ``max_items``。措辞带
            具体数值——宿主看到就能自己算要分几批。
    """
    if not isinstance(items, list):
        raise ValueError(f"items must be an array, got {type(items).__name__}")
    if len(items) > max_items:
        raise ValueError(
            f"batch limit is {max_items} items, got {len(items)}; split into multiple calls"
        )


def normalize_item(
    index: int,
    it: Any,
    *,
    default_tags: dict[str, Any] | None,
    default_source: str,
    default_granularity: str,
    default_detect_supersede: bool | None,
    default_client: str | None,
    default_auto_tags: bool,
) -> ItemPlan | ItemError:
    """把一项原始入参归一化成 ``ingest`` 的关键字参数，或判定跳过。

    三项应当在 ingested 之前完成的形状归正都在这里：

    - ``importance``：负数视为「未指定」（与 MCP 侧一致）；
    - ``entities``：接受列表或逗号分隔字符串，逐项 strip 并丢空；其余类型
      （含缺失）一律当空；
    - ``tags``：缺键回落到批次级默认值；字符串形态就地 ``json.loads``，
      **解析失败要当场抛给这一项**（由调用方的 ``except`` 收编）——宿主
      写了 ``tags="{...}"`` 却写坏了，静默当空会让它以为标签挂上了。

    Args:
        index: 项下标。
        it: 原始项。
        default_tags / default_source / default_granularity /
            default_detect_supersede / default_client / default_auto_tags:
            批次级默认值，**项级字段优先**。

    Returns:
        :class:`ItemPlan` 表示可以写；:class:`ItemError` 表示应记进 ``errors``
        并跳过（项不是 dict / ``text`` 缺失或空白）。

    Raises:
        ValueError: ``tags`` 是字符串但 JSON 解析失败（``json.loads`` 抛出）。
    """
    if not isinstance(it, dict):
        return ItemError(index=index, error="item is not an object (dict)",
                         preview=str(it)[:80])
    text = it.get("text")
    if not isinstance(text, str) or not text.strip():
        return ItemError(index=index, error="text is missing or empty",
                         preview=str(text)[:80])

    imp = it.get("importance")
    if isinstance(imp, (int, float)) and imp < 0:
        imp = None

    ents = it.get("entities")
    if isinstance(ents, str):
        ents = [e.strip() for e in ents.split(",") if e.strip()]
    elif isinstance(ents, list):
        ents = [str(e).strip() for e in ents if str(e).strip()]
    else:
        ents = []

    it_tags = it.get("tags", default_tags)
    if isinstance(it_tags, str):
        # 解析失败必须当场炸给这一项（调用方的 except 会收进 errors），
        # 不能静默当空——宿主会以为标签挂上了。
        it_tags = json.loads(it_tags)

    source_str = str(it.get("source") or default_source or "")
    payload: dict[str, Any] = {
        "text": text,
        "speaker": str(it.get("speaker") or "user"),
        "timestamp": it.get("timestamp"),
        "importance": imp,
        "cognitive_type": it.get("cognitive_type"),
        "entities": ents or None,
        "source": source_str,
        "detect_supersede": it.get("detect_supersede", default_detect_supersede),
        "granularity": str(it.get("granularity") or default_granularity),
        "tags": it_tags,
        "auto_tags": bool(it.get("auto_tags", default_auto_tags)),
        "client": it.get("client") or default_client,
    }
    return ItemPlan(index=index, payload=payload, source_str=source_str)


# ---------------------------------------------------------------- 结果累加


def error_preview(it: Any, plan: ItemPlan | None = None) -> str:
    """给「这一项炸了」的 errors 条目算 preview，取值口径与原实现一致。

    三路取值，按可用性递减：

    1. 归一化已经成功（``plan`` 是 :class:`ItemPlan`）→ 用 ``payload["text"]``；
    2. 归一化**中途**抛了（典型是项级 ``tags`` 的 JSON 解析失败，此时
       ``plan`` 还是 None）→ 直接取项里的 ``text``——宿主看到的预览应该
       仍是那句话，而不是整个 dict 的 str；
    3. 理论上到不了（非 dict 的项已在 :class:`ItemError` 路径跳过）→
       退回 ``str(it)`` 兜底。

    第 2 路是 parity 探针抓出来的：抽取前 ``text`` 变量在 try
    **之前**赋值，异常处理器里必然拿得到它；抽取后 ``plan`` 要等
    normalize_item 返回才有值，中途抛了就没有——漏看这条会误写成
    ``str(it)``，把预览变成整个对象。

    Args:
        it: 原始项。
        plan: 归一化结果（可能为 None——表示归一化自己炸了）。

    Returns:
        预览字符串（最多 80 字符）。
    """
    if isinstance(plan, ItemPlan):
        return plan.payload["text"][:80]
    if isinstance(it, dict) and isinstance(it.get("text"), str):
        return it["text"][:80]
    return str(it)[:80]


def collect_item(res: dict[str, Any], *, index: int, source: str) -> CollectedRow:
    """把一次 ``ingest`` 的返回体摊平成 :class:`CollectedRow`。

    两个细节值得点名：

    - ``merged_into`` / ``kept_separate`` / ``truncation.items`` 三项会被
      **打上下标**回指入参——批量的语境里宿主拿着 id 找不到是第几条写的。
      这里顺手补 ``index``，不改原 dict（``{**m, "index": index}``）。
    - ``truncation`` 只在 ``truncated`` 为真时收 ``items``：假则整项跳过，
      与「没截断就不该报 chast⟩」的口径一致。
    """
    merged = [{**m, "index": index} for m in res.get("merged_into", [])]
    kept = [{**k, "index": index} for k in res.get("kept_separate", []) or []]
    _tr = res.get("truncation") or {}
    truncations: list[dict[str, Any]] = []
    if _tr.get("truncated"):
        truncations = [dict(t, index=index) for t in (_tr.get("items") or [])]
    item = {
        "index": index,
        "raw_id": res["raw_id"],
        "memory_ids": res["memory_ids"],
        "n_memories": len(res["memory_ids"]),
        "source": source,
        "superseded": len(res.get("superseded") or []),
        "merged": len(res.get("merged_into") or []),
    }
    return CollectedRow(
        raw_id=res["raw_id"],
        memory_ids=list(res["memory_ids"]),
        entities=list(res["entities"]),
        merged=merged,
        kept=kept,
        truncations=truncations,
        superseded_n=len(res.get("superseded") or []),
        item=item,
    )


# ---------------------------------------------------------------- 回显体组装


def assemble_output(
    rows: list[CollectedRow],
    *,
    errors: list[dict[str, Any]],
    total_input: int,
    max_seq_length: int,
    l0_pending: int,
) -> dict[str, Any]:
    """把若干 :class:`CollectedRow` 横着拼成 ``ingest_batch`` 的回显体。

    键序是有讲究的：**合并警示前置**——``merged_into`` / ``kept_separate`` /
    ``truncation`` 排在 ``count`` 之前，因为批写场景里「有几条没单独存下来」
    比「总共写了几条」更该先看到。

    Args:
        rows: 成功项的产出，按入参顺序。
        errors: 跳过项的记录（``[{"index", "error", "preview"}]``）。
        total_input: 入参总条数（**含**被跳过的——宿主拿它跟 count 对账）。
        max_seq_length: 嵌入模型的序列上限（``embed_max_seq_length``，
            -domain 够不着 engine，由壳注入）。
        l0_pending: 未落盘的 L0 条数（同样注入；恒自曝欠账）。

    Returns:
        十五键回显体。``saved`` / ``save_warning`` 此时是初始形态，收尾落盘
        的分支会就地改这两键——那部分留在壳里。
    """
    merged_into = [x for r in rows for x in r.merged]
    kept_separate = [x for r in rows for x in r.kept]
    truncations = [x for r in rows for x in r.truncations]
    return {
        # 同 ingest：**合并警示前置**，批写时"有几条没单独存下来"
        # 比"总共写了几条"更该先看到。
        "merged_into": merged_into,
        # 因含新硬信息而未合并的近重复（非空 == 这些是新建的）
        "kept_separate": kept_separate,
        # 截断自曝，恒存在
        "truncation": {
            "truncated": bool(truncations),
            "max_seq_length": max_seq_length,
            "items": truncations,
        },
        "count": len(rows),
        "failed": len(errors),
        "total_input": total_input,
        "raw_ids": [r.raw_id for r in rows],
        "memory_ids": [x for r in rows for x in r.memory_ids],
        "entities": list(dict.fromkeys(x for r in rows for x in r.entities)),
        "superseded": sum(r.superseded_n for r in rows),
        "items": [r.item for r in rows],
        "errors": errors,
        "saved": False,
        "save_warning": "",
        # 批内不落 L0，收尾统一落一次；这里恒自曝欠账条数
        "l0_pending": l0_pending,
    }
