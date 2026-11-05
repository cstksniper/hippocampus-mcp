# -*- coding: utf-8 -*-
"""``brain.core.stats`` 的回显体组装（从 core 抽取）。

**为什么值得搬**：``stats`` 是宿主必调的自查入口，除两个副作用
调用（词库热感知 / 写者心跳）与已抽出的六个投影外，剩余的 return
大字典约 120 行、45 键、每键带设计 rationale 注释——纯「键序敏感回显体
组装」，与 knowledge_map_view 的 assemble_report 完全同型。

**依赖方向**：本模块零 import——所有值由壳算好传入：``tag_coverage`` /
``_recall_feedback_stats`` / ``_pin_stats`` / ``knowledge_stats`` /
``lexical_self_report`` / ``save_ledger_view`` / ``entity_origin_stats`` /
``tag_extractor.describe`` 等 brain 侧能力按值注入；``load_config`` 的两个
开关（layer_del / source_sig）与六个投影（state_counts / dangling_report
/ daily_activity / source_sig_block / writers_block / forgotten_count）也由
壳现取（不变式：六调用在壳内各恰一次）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. 45 键固定序（键序即信号——JSON 回显保序，宿主与测试都按序取用）；
2. ``layer_del`` 是嵌套子字典：四个标量在组装位现拼，键序 enabled /
   tombstones_n / blocked_last / gc_last；
3. ``source_sig`` 是 enabled 在前 + ``**block``（键序 = enabled /
   recorded / referenced，顺序即信号）；
4. ``vectors_by_model`` 从注入的 model_stats 投影 ``v["vectors"]``；
5. ``embedder_model_source`` 的「n/a（该后端不经模型名解析）」兜底文案是
   契约值（计算留在壳：test_runtime_selfreport C1 按**行为**钉「不谎报
   unknown」；test_f6 F3 按源码含 ``"embedder_shared"`` 字面量钉
   共享嵌入器键——该键的计算同样留在壳里，注释锚点随行）。
"""
from __future__ import annotations

from typing import Any


def assemble_report(
    *, l0_raw: int, l1_memories: int, tags: dict[str, Any],
    l0_dangling_source_ids: int, l0_dangling_memories: int,
    daily_activity: dict[str, Any], l1_by_state: dict[str, Any],
    recall_feedback: dict[str, Any], pin_stats: dict[str, Any],
    l2_summaries: int, topic_layer: dict[str, Any], l3_profile_slots: int,
    entities: int, sessions: int, open_intents: int,
    layer_merge_last: dict[str, Any], layer_del_enabled: bool,
    layer_del_tombstones_n: int, layer_del_blocked_last: int,
    layer_del_gc_last: Any, source_sig_enabled: bool,
    source_sig_block: dict[str, Any], writers: dict[str, Any],
    open_topics: int, pinned: int, tombstones: int, raw_forgotten: int,
    embed_model: str, embedder_backend: str, embedder_runtime: str,
    onnx_attached: bool, lnn_runtime: Any, pending_consolidate: bool,
    embedding_dim: Any, embedder_model_source: str, embedder_shared: bool,
    embedder_pool: dict[str, Any], ner_backend: str,
    lexical: dict[str, Any], model_stats: dict[str, Any],
    vectors_missing: int, supersede_enabled: bool, entity_lexicon_size: int,
    entity_aliases: int, tag_schema: dict[str, Any],
    entity_origins: dict[str, Any], sources: int,
    save_ledger: dict[str, Any], l0_pending: int,
) -> dict[str, Any]:
    """组装 stats 回显体（45 键固定序，键序即信号）。

    壳按旧 dict 字面量的求值序传参（kwargs 自左向右求值），本函数只做
    组装：四个标量现拼 ``layer_del`` 嵌套字典、enabled + ``**block`` 拼
    ``source_sig``、从 model_stats 投影 ``vectors_by_model``，其余逐键
    原样落位——每键上方的 rationale 注释随键转移，一字未删。
    """
    return {
        "L0_raw": l0_raw,
        "L1_memories": l1_memories,
        # 标签覆盖率 / 键分布：B1「按域查得到东西」的前提自曝
        "tags": tags,
        # 被 L1 引用却不在 L0 里的 id 数 / 受影响的 L1 条数。正常应为 0。
        "l0_dangling_source_ids": l0_dangling_source_ids,
        "l0_dangling_memories": l0_dangling_memories,
        "daily_activity": daily_activity,
        "L1_by_state": l1_by_state,
        # 召回反馈覆盖率。**自曝而非回填** —— 覆盖率低时
        # 「多久没被用过」这类归档结论会失真，这里让宿主看得见。
        "recall_feedback": recall_feedback,
        # pin 只进不退，堆积到一定比例会挤掉用层动态位
        "pin_stats": pin_stats,
        "L2_summaries": l2_summaries,
        "topic_layer": topic_layer,
        "L3_profile_slots": l3_profile_slots,
        "entities": entities,
        "sessions": sessions,
        "open_intents": open_intents,
        # 最近一次保存时从磁盘并回来的条数（按层）。
        # 非 0 说明确实有别的写者在动这个库。
        "layer_merge_last": layer_merge_last,
        # 删除账本自曝。
        # tombstones_n = 这张账本上还立着多少碑；blocked_last = 上次保存
        # 拦下了几条迟到的旧条目（>0 = 真的有人在拿旧快照复活东西）。
        # "拦截生效没有"必须有数字，不能靠"我觉得应该生效了"。
        "layer_del": {
            "enabled": layer_del_enabled,
            "tombstones_n": layer_del_tombstones_n,
            "blocked_last": layer_del_blocked_last,
            "gc_last": layer_del_gc_last,
        },
        # 源签名层自曝。recorded=采集过签名的 source 数；
        # referenced=库里有记忆带 source 标签的去重数。两者之差就是
        # drift_audit 里 unbaselined/orphan 的池子——"采集覆盖率"
        #必须看得见，否则"没漂移"和"没采集"分不开。
        "source_sig": {
            # 键序 = 旧实现的 enabled/recorded/referenced（顺序即信号）
            "enabled": source_sig_enabled,
            **source_sig_block,
        },
        # 共库写者可见化。active>1 说明真有人在跟你抢这个库。
        "writers": writers,
        "open_topics": open_topics,
        "pinned": pinned,
        "tombstones": tombstones,
        "raw_forgotten": raw_forgotten,
        "embed_model": embed_model,
        # ---- 运行期自述：让用户能核对"实际生效的是什么"，而不是靠猜 ----
        # 嵌入器后端（sentence_transformer / hash / …）。hash 后端是零依赖
        # 快速验证路径，语义能力弱——跑评测时不该悄悄落在它上面。
        "embedder_backend": embedder_backend,
        # **实际**推理后端（onnx / torch / hash）。与上一行的区别：
        # 上一行是"配了什么"，这一行是"用什么在跑"。产物缺失时 ONNX 会
        # 静默回退 torch，只看配置永远发现不了（真库实测）。
        "embedder_runtime": embedder_runtime,
        # onnx 有两种形态（ST 包装器挂 _onnx / embedder 本身是
        # OnnxEmbedder 直构），统一以 manager 的 runtime 判定为准；
        # lnn_runtime 自曝 CfC 实际运行时（numpy=零 torch）。
        "onnx_attached": onnx_attached,
        "lnn_runtime": lnn_runtime,
        # 会话分路自曝：还有会话未收尾时为 True（全局巩固推迟中）
        "pending_consolidate": pending_consolidate,
        # 实际向量维度（由模型决定，不是构造参数）
        "embedding_dim": embedding_dim,
        # 模型名的**来源**：参数 / env / 默认。这是"静默回落 MiniLM"那类
        # 事故的唯一探测器——来源是 default 且用户以为配了中文模型时，
        # 看这里一眼就能发现。
        # 取不到时说明该后端不经模型名解析（如 hash），此时"来源"没有意义，
        # 明确写出来，别让 unknown 被误读成"解析失败了"。
        "embedder_model_source": embedder_model_source,
        # 嵌入器共享自曝：本库的嵌入器是否复用了进程内共享的那份重对象。
        # 双库同模型时第二库应为 True；为 False 说明它自己又装了一份权重。
        "embedder_shared": embedder_shared,
        # 进程级共享池现状：onnx_size / torch_size 各几份。双库同模型时
        # 这里应当是 1；看到 2 就说明第二库又装了一份权重。
        "embedder_pool": embedder_pool,
        # 英文实体抽取后端。spaCy 装不上时这里会带上失败原因，
        # 不会只显示一句 heuristic-only 让人以为是设计如此。
        # 用**无副作用**版——启动日志走 stats()，若这里用 ner_backend
        # 会把 spaCy→thinc→torch（+160MB 常驻）在启动时强行拉进进程。
        "ner_backend": ner_backend,
        # ---- 字面通道自曝（字面通道优化的验收口）----
        # 「sqlite+FTS5 ⇒ 不建内存 BM25 ⇒ 省 34.5MB」这一优化必须有生产侧
        # 观测入口，否则「到底省没省」只能靠猜。这一节把后端、FTS5 可用性、
        # 内存索引条数、上次走的路一次给全。
        "lexical": lexical,
        "vectors_by_model": {
            k: v["vectors"] for k, v in model_stats.items()
        },
        "vectors_missing": vectors_missing,
        # 覆盖率自检硬性口径：必须为 0，
        # 否则说明库内向量与当前模型不符，检索会失真。
        "supersede_enabled": supersede_enabled,
        "entity_lexicon_size": entity_lexicon_size,
        # 别名条目数：为 0 时"同一实体多个称法"不会归并（读取侧合并
        # 与查询侧归一都依赖它）。宿主核对"我配的别名到底生效没有"看这里。
        "entity_aliases": entity_aliases,
        # 生效的标签抽取规则自述：键名是**调用方契约**，暴露出来才能
        # 双端核对（键名不一致 = 过滤静默失效，且不报错）。
        "tag_schema": tag_schema,
        # 实体来源分布：受信任(manual/lexicon/spacy) 才参与画像与检索加成，
        # heuristic 是候选（中文文档下属噪声主要来源，约 84%）。见 ENTITY_ORIGIN_*。
        "entity_origins": entity_origins,
        "sources": sources,
        # 保存账本投影 —— 把「上次保存是几点、多久没落盘」变成一行。
        "save_ledger": save_ledger,
        # 欠账的 raw 条数。看不见就等于不存在——这个坑拖了两个月，
        # 就是因为「L1 在、L0 不在」这件事无处可见
        "l0_pending": l0_pending,
    }
