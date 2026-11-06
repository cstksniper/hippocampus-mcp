# -*- coding: utf-8 -*-
"""``brain.core.recall`` 的三段内联逻辑（从 core 抽取）。

**为什么值得搬**：``recall`` 的壳里混着三块互不相干的职责——查询实体收集
（约 6 行）、向量/字面两路融合（约 15 行）、候选过滤与打分构建（约 68 行）。
第三块是主体：硬门槛过滤 + 命中归因 + 四路加分 + 导航降权 + 回显体组装，
全是纯 dict/list 运算，却因为要读 ``self.manager`` 的十几个面而无法单测。
搬出来后，``recall`` 只剩「空库短路 / 标签白名单 / 取两路候选 / 调 manager /
MMR 与标注编排 / 三处副作用记账」这些真正需要 brain 状态的事。

**依赖方向**：本模块只 import 标准库（domain → 仅 common 的纪律这里连
common 都没用上——所有常量、回调、``np.exp`` 均由壳在调用时现取后注入）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. ``literal_ids`` 的取法与旧实现**同源**：旧壳先算
   ``bm25_ranked = list(bm25_scores.keys())``，再取前 ``MMR_LITERAL_EXEMPT_TOPN``
   名。本模块直接用 ``bm25_scores`` 的键序算——``use_bm25=False`` 时两侧同为
   空，``fts5`` 分支也同为 ``dict(hits)`` 的键序，逐位等价。
2. 回显体里 ``literal`` / ``_literal`` / ``_emb`` **三键并存**（``_literal``
   与 ``literal`` 同值，是历史冗余）。``_emb`` 供 :meth:`_mmr_select` 算多样性
   惩罚、``_annotate_conflicts`` 算近邻，删任一个都会改下游行为。
3. ``_emb`` 只在候选**通过全部过滤**后才取（旧代码在 append 里调
   ``manager.vector_of``），抽到 :meth:`RecallCandidateContext.embed` 后保持
   同一惰性——被淘汰的候选不会碰嵌入模型。
4. ``hit_entity`` 是 ``bool(query_ent_ids) and bool(交集)`` 的短路形态：
   ``query_ent_ids`` 为空时**不碰** ``rec.entity_ids``。
5. ``matched_by`` 按固定顺序收集（vector → bm25 → entity → pin）后
   ``sorted()`` 输出；顺序本身不影响结果，但漏一路会漏归因。
6. 钉住豁免的指数衰减**外层套 ``float()``**（``np.exp`` 返回 ``np.float64``，
   不转会让回显体里出现 numpy 标量）。
7. 导航降权是**乘在最终分上**且带双条件：``nav_penalty < 1.0`` 才生效、
   且标签键有真值；默认 ``nav_penalty == 1.0`` 时整段跳过（不碰分）。
8. 硬门槛豁免只认字面路：``sim < hard_floor`` 且 ``did not in bm25_scores``
   才淘汰——BM25 单路命中的条目 ``similarity`` 是 0.0 也不该被杀。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ======================================================================
# 块 A：查询实体（旧壳 1451-1456）
# ======================================================================
def query_entity_ids(
    names: list[str],
    entities: dict[str, Any],
    trusted_ids: set[str],
) -> set[str]:
    """把查询里的实体名解析成**受信任**的实体 id 集合。

    Args:
        names: :meth:`HippoBrain.extract_entities` 的产出（四路次序的实体名）。
        entities: 实体表（``brain.entities``，键为小写实体名）。
        trusted_ids: 受信任来源的实体 id 白名单（启发式产出不参与加成）。

    Returns:
        去重后的实体 id 集合。查询里没有任何受信任实体时为空集——
        调用方据此短路 :meth:`collect_recall_items` 的实体加成。

    复刻怪癖：查表键是 ``name.lower()`` 而**不是** name 本身；实体表里
    ``id`` 缺失的条目（``ent["id"]``）会 KeyError——既有行为，不兜底。
    """
    out: set[str] = set()
    for name in names:
        ent = entities.get(name.lower())
        if ent and ent["id"] in trusted_ids:
            out.add(ent["id"])
    return out


# ======================================================================
# 块 B：两路融合（旧壳 1484-1500）
# ======================================================================
@dataclass(frozen=True)
class RecallFusion:
    """向量路与字面路的融合中间态。

    Attributes:
        vec_ranked: 向量路排序结果（原始顺序，RRF 输入）。
        vec_set: 同上的集合形态，只用于命中归因。
        fused: RRF 融合分（含 BM25 权重），**只含参与融合的 id**。
        candidate_ids: 向量路 ∪ 字面路的并集（向量路在前，字面路补尾）。
        sim_by_id: 向量相似度（字面路单捞的条目取 0.0）。
        vec_score_by_id: 向量路原始分（次级排序信号，权重 0.1）。
    """
    vec_ranked: list[str]
    vec_set: set[str]
    fused: dict[str, float]
    candidate_ids: list[str]
    sim_by_id: dict[str, float]
    vec_score_by_id: dict[str, float]


def build_fusion(
    raw: list[Any],
    bm25_ranked: list[str],
    *,
    bm25_weight: float,
    fuse: Callable[[list[list[str]], Any], dict[str, float]],
) -> RecallFusion:
    """把两路候选融合成统一的打分/并集视图（旧壳 1484-1500 整块）。

    Args:
        raw: manager.recall 的返回（鸭子类型：``memory_id`` / ``similarity``
            / ``score`` 三个属性）。
        bm25_ranked: 字面路排序结果；**空列表表示字面通道关闭**，此时
            ``fuse`` 只收到单路（权重表退化为 ``[1.0]``）。
        bm25_weight: 字面路在 RRF 里的权重（``brain.bm25_weight``）。
        fuse: :func:`~hippocampus_mcp.retrieval.bm25.rrf_fuse` 本体
            （retrieval 不在 domain 的可依赖范围内，按第 16 条铁律注入）。

    Returns:
        :class:`RecallFusion`。空 ``raw`` 且空 ``bm25_ranked`` 时全空，
        调用方循环自然不产出。

    复刻怪癖：``fuse`` 的调用形态随 ``bm25_ranked`` **真假**分岔——空列表
    时权重表只有 ``[1.0]`` 一项，与旧壳逐字一致（不是传 ``[1.0, w]`` 的空路）。
    """
    vec_ranked = [r.memory_id for r in raw]
    vec_set = set(vec_ranked)
    if bm25_ranked:
        fused = fuse([vec_ranked, bm25_ranked], weights=[1.0, bm25_weight])
    else:
        fused = fuse([vec_ranked], weights=[1.0])

    candidate_ids = list(vec_ranked)
    for did in bm25_ranked:
        if did not in candidate_ids:
            candidate_ids.append(did)

    sim_by_id = {r.memory_id: r.similarity for r in raw}
    vec_score_by_id = {r.memory_id: r.score for r in raw}
    return RecallFusion(
        vec_ranked=vec_ranked,
        vec_set=vec_set,
        fused=fused,
        candidate_ids=candidate_ids,
        sim_by_id=sim_by_id,
        vec_score_by_id=vec_score_by_id,
    )


# ======================================================================
# 块 C：候选过滤与打分构建（旧壳 1502-1570）
# ======================================================================
@dataclass(frozen=True)
class RecallCandidateContext:
    """候选遍历所需的全部旋钮与回调（壳在调用时现取后一次性注入）。

    Attributes:
        allow_ids: 标签白名单；``None`` 表示不过滤（**不是空集**）。
        include_superseded: 是否放行 superseded 记录。
        archived_state: ``MemoryState.ARCHIVED.value``——状态是字符串，
            按值注入（store.memory 不在 domain 依赖范围内）。
        superseded_state: 同上，``MemoryState.SUPERSEDED.value``。
        hard_floor: 相关性硬门槛（``RELEVANCE_HARD_FLOOR``）。
        entity_bonus: 实体命中加成（``ENTITY_BONUS``）。
        literal_topn: 字面强命中豁免名次（``MMR_LITERAL_EXEMPT_TOPN``）；
            ``<= 0`` 时豁免集为空。
        w_time: 时间衰减权重（``manager.scorer.w_time``，补钉住豁免用）。
        lambda_decay: 衰减时间常数（``brain.lambda_decay``）。
        nav_penalty: 导航类内容折扣（``< 1.0`` 才生效）。
        nav_tag_key: 导航标签键（``brain.nav_tag_key``）。
        bm25_scores: 字面路分数字典（**键序即名次**）。
        query_ent_ids: 查询实体 id 集合（块 A 的产出）。
        vec_set: 向量路 id 集合（归因用）。
        fused: RRF 融合分（块 B 的产出）。
        vec_score_by_id: 向量路原始分（次级信号）。
        sim_by_id: 向量相似度（缺失取 0.0）。
        get_record: ``manager.store.get`` 本体（鸭子类型返回值，可能 None）。
        embed: ``manager.vector_of`` 本体（**惰性**：只对通过过滤的候选调）。
        exp: ``np.exp`` 本体（numpy 不在 domain 依赖范围内）。
        now: 时间基准（``manager._reference_time or _now()``，调用时算好）。
    """
    allow_ids: set[str] | None
    include_superseded: bool
    archived_state: str
    superseded_state: str
    hard_floor: float
    entity_bonus: float
    literal_topn: int
    w_time: float
    lambda_decay: float
    nav_penalty: float
    nav_tag_key: str
    bm25_scores: dict[str, float]
    query_ent_ids: set[str]
    vec_set: set[str]
    fused: dict[str, float]
    vec_score_by_id: dict[str, float]
    sim_by_id: dict[str, float]
    get_record: Callable[[str], Any]
    embed: Callable[[Any], Any]
    exp: Callable[[float], float]
    now: float


def collect_recall_items(
    candidate_ids: list[str],
    ctx: RecallCandidateContext,
) -> list[dict[str, Any]]:
    """按旧壳 1502-1570 的次序过滤候选、打分、组装回显体。

    Returns:
        通过全部过滤的条目（**顺序即 candidate_ids 的遍历序**；MMR 与
        最终截断在壳里做，这里不裁 ``top_k``）。每条都带 ``_emb`` /
        ``_literal`` 两个内部字段，供 :meth:`_mmr_select` 与
        :meth:`_annotate_conflicts` 消费，最后由 ``strip_internal_fields``
        统一清理——本函数**不负责清理**。

    复刻怪癖见模块 docstring 第 2-8 条；其中：

    - 硬门槛豁免只认字面路（``did not in ctx.bm25_scores``）；
    - 钉住豁免与实体加成**按此顺序**叠加在融合分之后、导航降权之前；
    - 导航降权是乘法，且 ``nav_penalty`` 为 1.0（默认）时整段跳过。
    """
    # 字面强命中豁免集：键序即名次（见模块 docstring 第 1 条）
    if ctx.literal_topn > 0:
        literal_ids = set(list(ctx.bm25_scores.keys())[:ctx.literal_topn])
    else:
        literal_ids = set()

    items: list[dict[str, Any]] = []
    for did in candidate_ids:
        rec = ctx.get_record(did)
        if rec is None:
            continue
        if ctx.allow_ids is not None and did not in ctx.allow_ids:
            continue
        if rec.state == ctx.archived_state:
            continue
        if rec.state == ctx.superseded_state and not ctx.include_superseded:
            continue
        # D2a 写侧隔离：待审卡不进召回池——这是"不进召回池"承诺的**唯一
        # 执行点**（brain.recall 单一调用路径，scope_recall 的用户库流同穿
        # 此处）。getattr 兜底：旧记录对象没有该字段时按未隔离处理。
        if getattr(rec, "quarantine", False):
            continue
        sim = ctx.sim_by_id.get(did, 0.0)
        # 相关性硬门槛：向量和 BM25 都没命中 → 淘汰（防摘要层霸榜）
        if sim < ctx.hard_floor and did not in ctx.bm25_scores:
            continue

        # 命中归因：三路信号各记一笔，供返回体的 matched_by 用
        hit_entity = bool(ctx.query_ent_ids) and bool(
            ctx.query_ent_ids & set(rec.entity_ids or [])
        )
        srcs = []
        if did in ctx.vec_set:
            srcs.append("vector")
        if did in ctx.bm25_scores:
            srcs.append("bm25")
        if hit_entity:
            srcs.append("entity")
        if rec.pin:
            srcs.append("pin")

        # 主排序信号 = RRF 融合分；向量分作为次级信号
        score = ctx.fused.get(did, 0.0) + 0.1 * ctx.vec_score_by_id.get(did, 0.0)
        # 钉住记忆豁免时间衰减（补足 w_time 的最大可能贡献）
        if rec.pin:
            age = max(ctx.now - rec.timestamp, 0.0)
            score += ctx.w_time * (
                1.0 - float(ctx.exp(-age / ctx.lambda_decay))
            )
        # 实体命中加成（跨越语义鸿沟：向量相似度低但实体对上）
        if hit_entity:
            score += ctx.entity_bonus
        # 导航类内容降权：默认 penalty=1.0 不生效，只有导入端显式标了
        # NAV_TAG_KEY 且配了折扣才动，避免默认行为变化。
        if ctx.nav_penalty < 1.0 and (rec.tags or {}).get(ctx.nav_tag_key):
            score *= ctx.nav_penalty
        items.append({
            "memory_id": rec.id,
            "content": rec.content,
            "score": score,
            "similarity": sim,
            "timestamp": rec.timestamp,
            "importance": rec.importance,
            "state": rec.state,
            "cognitive_type": rec.cognitive_type,
            "source_ids": rec.source_ids or [],
            "entity_ids": rec.entity_ids or [],
            "pin": rec.pin,
            "source": rec.source or "",
            "tags": dict(rec.tags or {}),
            # D7（ZCode 追加三）：这条**凭什么**被召回。多路混合检索里
            # "分数高"不等于"检索器认为它相关"——可能只是 BM25 字面撞上了
            # 一个虚词，也可能是实体/钉住加成顶上去的。给出归因才能解释
            # 误召回，也是调权重的依据。
            "matched_by": sorted(srcs),
            "literal": did in literal_ids,
            "_emb": ctx.embed(rec),
            "_literal": did in literal_ids,
        })
    return items
