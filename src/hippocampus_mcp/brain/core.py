# -*- coding: utf-8 -*-
"""HippoBrain — 智能体记忆底座的编排层 (v0.3).

把 MemoryManager（向量 + LNN + 打分）包装成完整的"记忆大脑"：

    L0 原始层  raw_log    原始对话流水，append-only，L1 的 source_ids 回指它
    L1 事实层  memories   萃取事实 + 状态机 (active/superseded/archived) + 认知类型
    L2 摘要层  summaries  会话摘要 / 主题聚合
    L3 画像层  profile    长期稳定事实与偏好（低置信不入，冲突只提示不覆盖）
    索引层     entities   实体表 + 实体↔记忆倒排（P0 实体图前身）
    用层       session_*  5 槽位上下文装配（借鉴 bigeye AttentionFocus）

设计要点来自研究：
    - 事实被取代 → 标 superseded，不删除（bigeye relation_store）
    - importance >= 0.7 → 钉住，豁免时间衰减（bigeye: >=7 pinned）
    - 相似度硬门槛 0.30 → 防止摘要层"和什么都相似"而霸榜
    - MMR 去重 (λ=0.7) → 抑制召回噪
"""

from __future__ import annotations

import atexit as _atexit
import heapq  # noqa: F401  （新颖度门控：高置信 L1 比对集取前 cap，O(N) 不全排序）
# json 虽已不再被本模块直调（唯一调用点已搬进
# domain.ingest_batch_flow），但**必须留在这里**：
# tests/test_l0_autosave.py monkeypatch ``brain.json.dump`` 注入 OSError
# 模拟磁盘满——``B.json.dump`` 的 B 就是本模块。删掉这个 import，那条
# 「落盘失败不能中断」的守卫测试会失去注入点。
import json  # noqa: F401
import logging
import os
import re
import sys
import threading
import time
import zlib

import numpy as np

try:  # 只拿主机名，失败就留空——不该因为拿不到主机名就起不来
    import socket as _socket
except ImportError:  # pragma: no cover
    _socket = None  # type: ignore[assignment]
from typing import Any

import numpy as np

from ..config import load_config
from ..common.recency import (
    in_time_window,
    iso_date,
    out_of_window,
    weekday_abbr,
)
from ..domain import policies as _policies
from ..domain import novelty_gate as _novelty_gate
from ..domain import prospective as _prospective
from ..domain import forget_cleanup as _forget_cleanup
from ..domain import ingest_flow as _ingest_flow
from ..domain import knowledge_batch as _knowledge_batch
from ..domain import knowledge_map_view as _knowledge_map_view
from ..domain import topic_material as _topic_material
from ..domain import tag_update as _tag_update
from ..domain import consolidate_flow as _consolidate_flow
from ..domain import ingest_batch_flow as _batch_flow
from ..domain import ingest_pipeline as _ingest_pipeline
from ..domain import recall_flow as _recall_flow
from ..domain import refresh_knowledge_flow as _refresh_flow
from ..domain import stats_projection as _stats_projection
from ..domain import stats_report as _stats_report
from ..domain import miss_log_flow as _miss_log_flow
from ..domain import file_registry_flow as _fr_flow
from ..domain import summarize_batch_flow as _summarize_batch_flow
from ..domain import pure_checks as _pure_checks
from ..domain import tag_select as _tag_select
from ..domain import drift_ops as _drift_ops
from ..domain import entity_link as _entity_link
from ..domain import supersede_views as _supersede_views
from ..domain import map_ops as _map_ops
from ..domain import map_drift as _map_drift
from ..domain import knowledge_ops as _knowledge_ops
from ..domain import profile_ops as _profile_ops
from ..store import layer_gc as _layer_gc
from ..domain import ingest_entry as _ingest_entry
from ..domain import forget_entry as _forget_entry

from ..domain import aliases as alias_mod
from ..engine.embeddings import ENV_MODEL, embedder_pool_stats
from ..engine.manager import MemoryManager
from ..retrieval.bm25 import BM25, rrf_fuse
from ..store.memory import (
    PIN_IMPORTANCE,  # noqa: F401  （唯一阈值定义处见 store/memory.py）
    CognitiveType,
    MemoryState,
    merge_id_lists,
    merge_tag_dicts,
)
from ..store.vector_store import multi_writer_enabled
from ..store.backend import get_backend
# 层保存的两段编排（账本 begin/end/fail 三段 + 派生层
# 「读-合并-写」）落 store 侧；core 只留两个薄壳当适配层——brain 记账
# 状态（指纹戳/序号/l0 欠账）的写回也在壳里，store 返回即写盘成功。
from ..store.layer_save import LayerSavePorts, LayerSaveResult, LayerLedgerPorts
from ..store.layer_save import run_layer_save, save_layers_core
# 标签变更账本的 I/O：JSONL ring 的读/扫/追加/裁剪是纯持久
# 化设施，归 store 侧；下面只留三个薄包装当适配层——ring 旋钮在调用时从
# core 命名空间现取，保住 ``core.TAG_OPS_RING`` 被测试 monkeypatch 的面。
from ..store.tag_ops import append_ops, read_ops, scan_ops
# 保存账本的 I/O：同型理由——账本行数/末条的状态
# 语义留在包装层，ring 旋钮（LEDGER_RING / LEDGER_TRIM_AT）在调用时从
# core 命名空间现取，保住 test_save_ledger 对 ``core.LEDGER_RING`` 的
# monkeypatch 面（旋钮快照进 store 模块会让 L5 的环大小断言静默失效）。
from ..store.ledger import (
    append as ledger_append,
    ledger_path,
    read_tail,
    reconcile_on_boot,
    trim as ledger_trim,
)
# 矛盾检测：库存记录上的查询组合。store 实例与
# 可信实体 id 由包装层在调用时注入（self.manager.store /
# self._trusted_entity_ids()）——本模块不 import vector_store 也不碰 brain，
# 同 tag_ops / ledger 纪律。
from ..store.supersede import detect_supersede as _detect_supersede_fn
from ..store.contradictions import (
    contradiction_buckets, find_contradictions, ids_matching_tags, pool_size,
)
# 源文件签名与漂移审计：库存记录上的查询组合 +
# 文件 I/O。store 实例（iter_records）、source_sigs 层、sha1 缓存由包装层
# 注入；source_root / sig_enabled 两个 Config 旋钮在调用时从 core 命名空间
# 现取（store 不许 import config，同 tag_ops / ledger 纪律）。
from ..store.source_sig import (
    drift_audit, file_sig, record_source_sig as _record_source_sig,
    resolve_source_path,
)
# 标签变更撤销：undo_tag_ops 方法体。账本读取/追加、摘要、
# 覆盖率、落盘由包装层注入；两个闸门常量（UPDATE_TAGS_SAMPLE_MAX /
# UPDATE_TAGS_CONFIRM_ABOVE）在调用时从 core 命名空间现取——测试
# monkeypatch core.UPDATE_TAGS_CONFIRM_ABOVE，快照进 store 模块会让闸门
# 断言静默失效（同 tag_ops / ledger / source_sig 纪律）。
from ..store.tag_ops_undo import undo_tag_ops as _undo_tag_ops
# 层指纹投影：只读投影，不写任何东西。``_LAYER_ATTRS``
# 与指纹函数、磁盘戳比较由包装层绑好注入；``layer_fp`` 开关在调用时从 core
# 命名空间现取（store 不许 import config，同 tag_ops / ledger / source_sig 纪律）。
from ..store.layer_view import (
    classify_layer_quadrants,
    compute_all_layer_fps,
    ledger_view,
    pending_layer_names,
)
from ..domain.tagging import TagExtractor, load_extractor, merge_tag_values
from ..map import file_registry as fr_mod
from ..map import store as map_mod
from ..map import auto as map_auto_mod
from ..map import mention as map_mention_mod
from ..common.sidecar_lock import SidecarBusy, SidecarUnreadable
from ..domain import topic as topic_mod
# 任务概况投影：纯派生，不检索不引模型。
# 主题层开关由包装层从 ``_topic_defaults()`` 现取（纪律：做成 import-time
# 快照会让 HIPPOCAMPUS_TOPIC_LAYER=0 静默失效）；五档上限从 core 命名空间
# 现取（arch 冻结常量名在 core）；参考时钟走 manager._reference_time。
from ..domain.task_profile import Limits as _TaskProfileLimits
from ..domain.task_profile import task_profile as _task_profile
# 库级全貌卡 / 状态卡投影：纯派生、不检索。派生层与参考时钟
# 由包装层绑好；两个 Config 开关（主题层 / 状态卡矛盾）与五个上限调用时现取；
# 五个**惰性回调**（tag_coverage / knowledge_stats / contradictions_count /
# active_sessions / file_registry_summary）传的是可调用对象——开关关掉的那些
# 一次都不会被调用（万条库 0.34s 量级的调用，白烧就是白烧）。
from ..domain.conflict_annotate import annotate_conflicts as _annotate_conflicts_fn
from ..domain.project_view import project_overview as _project_overview
from ..domain.project_view import project_status as _project_status
# 开场包两个槽：最近记忆是库存记录查询组合（state 过滤 +
# 排序 + 投影，与 contradictions 同族）⇒ store；知识地图槽是派生投影 ⇒
# domain，但主题口径（ordered / 归档状态 / summary_state）走注入不 import
# （架构约束：domain → 仅 common），store.count() 同样注入以保住惰性。
from ..store.recent_memories import recent_memories as _recent_memories
from ..domain.knowledge_brief import knowledge_brief as _knowledge_brief
from ..domain.map_view import DetailLimits as _MapDetailLimits
from ..domain.map_view import IndexLimits as _MapIndexLimits
from ..domain.map_view import map_detail as _map_detail_fn
from ..domain.map_view import map_index as _map_index_fn
# 统计自曝投影四件：pin / 召回反馈 / 字面通道自曝 + 钉住条目
# 简报，全部只读。``MemoryState.ACTIVE.value`` / ``PIN_IMPORTANCE`` 在 store 包
# ⇒ 调用时现取注入（架构约束：domain → 仅 common）；``_pinned_brief`` 的相关度
# 走回调（``_pin_task_relevance`` 还要读 ``_task_entity_ids`` ⇒ 留在 brain）。
from ..domain.stats_views import lexical_self_report as _lexical_self_report_fn
from ..domain.stats_views import pin_stats as _pin_stats_fn
from ..domain.stats_views import pinned_brief as _pinned_brief_fn
from ..domain.stats_views import recall_feedback_stats as _recall_feedback_stats_fn
# 按自然语言删除的编排层：本体是分流 + 删除循环，
# recall / forget 两个 brain 入口以可调用对象注入；默认门槛
# FORGET_MIN_SIMILARITY 调用时现取（patch 本命名空间即生效）。
from ..domain.forget_flow import forget_by_query as _forget_by_query_fn
# 5 槽位上下文装配：本体是分流 + 候选收集 + 预算装填，
# recall / _pinned_brief / _trusted_entity_ids / extract_entities /
# entity_facts / _render_blocks 六个入口以可调用对象注入，entities /
# intents / summaries / open_topics 零成本属性读原样传值；BLOCK4_*
# 四个上限调用时从本命名空间现取（common/constants 的值，patch 即生效）。
from ..domain.context_slots import memory_context as _memory_context_fn
# 库展示名（提取为自由函数）：要读 library.json（I/O），归 library/
# 包——common 是最底层不许依赖 library。brain→library 合法。
# 三个时间序列统计（recent_ingest / spark_series / recent_miss_count）随
# 全貌卡/状态卡搬进 domain/project_view.py 后**在 core 已无调用点**
# ⇒ 死导入删掉；它们仍在 common/stats.py 里供 domain 与渲染器使用。
from ..library import display_name

logger = logging.getLogger(__name__)


from .pure import _PureMixin
from .entity import _EntityLexiconMixin
from .session import _SessionMixin, _topic_defaults
from ..common.constants import *  # noqa: F401,F403  常量表见 common/constants
# 地图层常量（见 map/constants.py）
from ..map.constants import (
    MAP_DETAIL_MAX_MAPS,
    MAP_DETAIL_MAX_ROWS,
    MAP_ENTRY_KEY,
    MAP_INDEX_MAX,
    MAP_NAME_KEY,
    MAP_STALE_DAYS,
    MAP_STATUS_KEY,
    MAP_WRITE_WAIT_S,
)
# 画像毕业槽常量住 domain/constants（共享底座不是
# 倾销场）；**显式**导入而非靠 common 的 import *，以保住 brain 包再导出
# 名基线与本模块的 monkeypatch 面（测试 patch core 命名空间）。
from ..domain.constants import (
    PROFILE_SLOT_RECURRING,
    PROFILE_STRENGTH_DAYS,
    PROFILE_STRENGTH_MAX,
    PROFILE_STRENGTH_MIN,
)

# M1b：ring 三件改由 Config 供值（仍是 import 时快照，语义未变）；topic 默认
# 值包相反，原实现是 call-time 读 env，现取装配见 session._topic_defaults()。
_cfg_import = load_config()
TAG_OPS_RING, TAG_OPS_TRIM_AT = _cfg_import.tag_ops_ring, _cfg_import.tag_ops_ring * 2
LEDGER_RING, LEDGER_TRIM_AT = _cfg_import.save_ledger_ring, _cfg_import.save_ledger_ring * 2
L0_AUTOSAVE_EVERY = _cfg_import.l0_autosave_every


def _seed_library_map_config() -> None:
    """library/map/domain(tagging) 旋钮的生产注入点。

    与 ``engine.manager._seed_store_config`` 同款（那边管 store/lnn/retrieval；
    engine 按依赖纪律不许 import library，故这三家由本模块——每条生产路径必经
    的 brain——import 时灌入）。消费点的 env 直读降级为「直连回退」，测试
    直连不经 brain 的面零改动；解析序与零漂移论证见 common/config_bridge。
    """
    from ..domain import tagging as _tagging_mod
    from ..library import apply_config as _lib_apply_config
    from ..library import registry as _lib_registry_mod
    from ..map import file_registry as _fr_mod

    _lib_registry_mod.apply_config(max_resident_libs=_cfg_import.max_resident_libs)
    _lib_apply_config(library_roots=_cfg_import.library_roots)
    _fr_mod.apply_config(
        file_registry_ignore=_cfg_import.file_registry_ignore,
        file_registry_max_files=_cfg_import.file_registry_max_files,
        file_registry_budget_s=_cfg_import.file_registry_budget_s,
    )
    _tagging_mod.apply_config(tag_schema=_cfg_import.tag_schema)


_seed_library_map_config()


# 访问驱动毕业的四个入参。抽成模块级函数**只为**让
# ``HippoBrain._update_profile`` 的壳守住行数预算；开关与
# 阈值仍是**调用时现取**（monkeypatch 本模块命名空间即可生效）。关闸返回
# 空 dict——domain 侧默认 ``recurring_slot=""`` 就是「不收毕业条目」。
def _profile_promotion_kwargs() -> dict[str, Any]:
    if not load_config().profile_promotion:
        return {}
    return {
        "recurring_slot": PROFILE_SLOT_RECURRING,
        "strength_min": PROFILE_STRENGTH_MIN,
        "strength_days": PROFILE_STRENGTH_DAYS,
        "strength_max": PROFILE_STRENGTH_MAX,
    }


class HippoBrain(_SessionMixin, _EntityLexiconMixin, _PureMixin):
    """智能体记忆底座：L0-L3 + 索引层 + 用层 + 巩固任务。

    Args:
        persist_dir: 持久化目录。None 则纯内存（重启即丢）。
        embedder_backend: 嵌入后端，默认 sentence_transformer。
        auto_consolidate_every: 每 ingest 多少条自动跑一次巩固，0 关闭。
        auto_save: 每 10 次写入自动落盘（透传给 MemoryManager）。
        mu_importance: 重要性衰减常数（秒），透传给打分引擎。
        w_sim / w_time / w_recall: 加法评分权重，透传给打分引擎。
        supersede_enabled: 是否启用"同一事实被更新"检测。**批量导入文档时
            应关闭**（md 已是当前真相，沿革叙述会被大面积误判）。
        entity_lexicon_path: 项目专名词库文件路径（一行一个专名）。
            缺省用 ``HIPPOCAMPUS_ENTITY_LEXICON``，再缺省用
            ``<persist_dir>/entity_lexicon.txt``（放在记忆库目录里即自动生效）。
            **行内可写别名**：``别名 => 主名``（``->`` / ``→`` 等价），
            抽取与查询两侧都会归一到主名（见 :meth:`canonical_entity`）。
        tag_schema: 结构化标签的抽取配置（JSON 文件路径）。缺省用
            ``HIPPOCAMPUS_TAG_SCHEMA``，再缺省用
            ``<persist_dir>/tag_schema.json``（放在记忆库目录里即自动生效）。
            配置可改名键、新增结构后缀、加自定义正则——见
            ``hippocampus_mcp.domain.tagging``。不配则用内置通用规则。
        auto_backfill: 加载时是否自动补算当前模型缺失的向量。大库补算是
            分钟级且**同步阻塞**在启动路径上，会拖垮 MCP 客户端；此时置 False，
            先离线跑 ``scripts/reembed.py``。

    说明（v0.4）：本类持有的 ``manager`` 是进程内**唯一**的记忆管理器。
    服务端两个入口（旧工具 / v0.3 用层工具）都必须复用它，不能各建一个——
    两条路径各建一个实例会指向同一个 ``memories.json``，
    造成写入互不可见 + save 时整体覆写（实测数据丢失）。
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        embedder_backend: str = "sentence_transformer",
        auto_consolidate_every: int = 0,
        lambda_decay: float = 7776000.0,
        use_bm25: bool = True,
        bm25_weight: float = 1.0,
        embedder_model: str = "",
        auto_save: bool = False,
        mu_importance: float = 15552000.0,
        w_sim: float = 0.95,
        w_time: float = 0.03,
        w_recall: float = -0.02,
        supersede_enabled: bool = True,
        entity_lexicon_path: str | None = None,
        tag_schema: str | None = None,
        auto_backfill: bool = True,
        nav_penalty: float = 1.0,
        nav_tag_key: str = NAV_TAG_KEY,
        recency_window_days: int = 0,
        w_strength: float = 0.03,
        mem_history_ring: int = 200,
    ) -> None:
        self.persist_dir = persist_dir
        # 落盘后端（json 多文件 / sqlite 单库）。只替换读写层，内存里的
        # 层对象与业务逻辑不动。见 store/backend.py 的模块说明。
        self._backend = get_backend(persist_dir) if persist_dir else None
        # 导航类内容（INDEX / 目录 / 表头）的召回折扣（追加八）。1.0 = 不降权，
        # 默认保持不动——它是**显式配置**才生效的机制，不该悄悄改变既有排序。
        self.nav_penalty = float(nav_penalty)
        self.nav_tag_key = str(nav_tag_key or NAV_TAG_KEY)
        # 检索端过时窗口（天）：age > 窗口且未 pin
        # → 不进结果集。**零状态变更**——不删、不归档、不改 state，只在
        # 检索端挡输出。向量路在 MemoryManager（覆盖 MCP recall 工具的
        # 直连路径），这里只管下传旋钮 + 融合路兜底。
        # **默认 0 = 关**：库里的老记忆必须能被相似查询查到——只给排名靠前
        # 的几条，而不是因为没有"近期"就直接遗忘。需要到期窗口就显式配。
        self.recency_window_days = max(int(recency_window_days), 0)
        self._recency_window_seconds = self.recency_window_days * 86400.0
        # 检索强度权重：+w_strength·log1p(强度)——成功召回
        # 巩固痕迹（测试效应）。0 = 关（与旧打分逐位一致）。透传 manager。
        self.w_strength = w_strength
        # 改写历史账本 ring（透传 manager；账本 I/O 在 store/mem_history，
        # 挂钩在 manager.update_memory / store_memory 合并分支，brain 无感知）。
        self.mem_history_ring = mem_history_ring
        self.auto_consolidate_every = auto_consolidate_every
        self.lambda_decay = lambda_decay
        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self._bm25 = BM25()
        # BM25 增量索引：id -> 内容指纹（crc32），用于识别新增/改动/删除
        self._bm25_indexed: dict[str, int] = {}
        # 「上次检索的字面通道走的是哪条路」——H4 的核心收益是**不再建内存
        # BM25 索引**（省 34.5MB），但按下它的那个开关（后端 / FTS5 可用性）在
        # 生产侧一直没有观测口，值守清单第 4 步的验收无从下手。
        self._last_lexical_path: str = "none"
        # 事实更新检测开关。批量导入文档时应关闭：md 文件本身已经是去重后的
        # 当前真相，而 _detect_supersede 是为"对话里的事实更新"设计的，
        # 对沿革叙述（"X 改成 Y"本就是这类文档的常态）会大面积误杀
        # （覆盖报告·高 4）。
        self.supersede_enabled = supersede_enabled
        # 自动补算缺失向量的开关。启动路径上补算大库会**同步阻塞**，
        # 导致 MCP 客户端超时；大库请先离线 reembed 再启动。
        self.auto_backfill = auto_backfill

        # 嵌入模型解析（覆盖报告·阻断 1）：
        # 未显式指定时读 HIPPOCAMPUS_EMB_MODEL（常量见 embeddings.ENV_MODEL，
        # 不要再往这里抄字面量）。此前只有 server.py 读这个 env，
        # import_md.py 与文档里的验证片段都直接 default 到 all-MiniLM-L6-v2，
        # 于是"导进库的全是 MiniLM 向量、服务用 BGE 查"——维度不符，检索必失。
        # 即便如此，这里也不是唯一防线：env 也没给时嵌入器会回落默认值，
        # 而 manager.load() 会把"回落值 vs 库内实际模型"对齐（见
        # MemoryManager._sync_embedder_with_store）——所以"忘配 env"不再致命。
        if not embedder_model:
            embedder_model = load_config().emb_model
        # 专名词库（覆盖报告·高 6）：显式参数 > env > <记忆库>/entity_lexicon.txt
        self._explicit_lexicon_path = entity_lexicon_path
        self.entity_lexicon_path = self._resolve_lexicon_path(
            entity_lexicon_path, persist_dir
        )
        self.entity_lexicon, self.entity_lexicon_aliases = self._read_lexicon(
            self.entity_lexicon_path
        )
        # 词库文件 mtime 快照（热感知用，见 :meth:`_reload_lexicon_if_changed`）
        self._lexicon_mtime = self._lexicon_mtime_of(self.entity_lexicon_path)
        # 别名 → 规范名的反向索引缓存（惰性建；词库 reload 时置 None 失效）
        self._alias_reverse: dict[str, list[str]] | None = None
        # 结构化标签抽取器（通用化）：可配置，不写死任何一种语料的结构词。
        self._explicit_tag_schema = tag_schema
        self.tag_extractor = load_extractor(tag_schema, persist_dir)

        self.manager = MemoryManager(
            persist_dir=persist_dir,
            embedder_backend=embedder_backend,
            embedder_model=embedder_model,
            lambda_decay=lambda_decay,
            mu_importance=mu_importance,
            w_sim=w_sim,
            w_time=w_time,
            w_recall=w_recall,
            auto_save=auto_save,
            recency_window_days=recency_window_days,
            w_strength=w_strength,
            mem_history_ring=mem_history_ring,
        )

        # L0 原始层
        self.raw_log: list[dict[str, Any]] = []
        # L2 摘要层
        self.summaries: list[dict[str, Any]] = []
        # 主题层（L2T）：L1 的派生层——把平铺事实收敛成「知识点」的骨架。
        # 算法与硬约束见 src/hippocampus_mcp/topic.py 的模块说明（为什么不
        # 进检索路径、为什么是派生层、为什么只在 consolidate / 显式调用时建）。
        self.knowledge: dict[str, dict[str, Any]] = {}
        # 主题层自曝：上次刷新时点 / 耗时 / 摘除条数（「多久没梳理了」要看得见）
        self._knowledge_last_at = None
        self._knowledge_last_took = None
        self._knowledge_last_dropped = 0
        # L3 画像层
        self.profile: dict[str, list[dict[str, Any]]] = {}
        # 最近一次画像刷新**摘掉了什么**（追加 19②）。画像此前只增不摘，
        # "画像少了两条"是宿主该看见的事，所以落进返回体自曝。
        self.last_profile_pruned: list[dict[str, str]] = []
        # 索引层：实体表 name -> entity dict
        self.entities: dict[str, dict[str, Any]] = {}
        # 会话状态 / 未完结话题 / 意图队列
        self.sessions: dict[str, dict[str, Any]] = {}
        self.intents: list[dict[str, Any]] = []
        self.open_topics: list[dict[str, Any]] = []
        # 会话身份按 client 分路。未署名调用落在 __default__ 路，
        # 行为与改造前逐位一致（见 current_session_id property）。
        self._current_sessions: dict[str, str] = {}
        # C-1 自曝用：本次 session_start 自动终结了哪个旧会话（无则 None）。
        self._superseded_session: str | None = None
        # B（验证反馈 2026-11-01）：最近一次 save 的层级明细
        # （written/skipped/failed），session_end 返回体回显。
        self.last_save_layers: dict[str, str] = {}
        # 还有别的 client 在会话中时，session_end 只做本路记账 +
        # 落盘，**全局收尾（画像提炼 + 巩固）推迟到最后退出者**。否则 daemon
        # 下 A 的 session_end 会扫全库/动派生层，干扰 B 正在进行中的会话。
        self._pending_consolidate = False
        # 睡眠批状态。水位线**仅内存**——daemon 重启后
        # 首轮按重放窗口重算；要不要落盘等 O1 强度变量接线时随其存储字段
        # 一起定（per-record 戳比全局水位更抗中断，届时一并做）。
        # _sleep_lock 是重入守卫：睡眠批与在线写分属不同线程，同库并发
        # 跑两个睡眠批没有任何好处。
        self._last_sleep_at: float = 0.0
        self._sleep_lock = threading.Lock()
        # 各派生层最近一次「从磁盘并回来几条」，stats() 自曝
        self.last_layer_merge: dict[str, int] = {}
        # 写者可见化。_save_seq 是落盘次数（心跳的一部分）。
        self._created_at = time.time()
        self._save_seq = 0
        # F14：账本末条与行数。行数只用于 ring buffer 的触发时机，不追求精确——
        # 多进程各记各的，偏大只是早裁一点，不影响正确性。
        self._ledger_last: dict[str, Any] | None = None
        self._ledger_lines = 0
        # F14-5：账本尾部若干行（fail 回看同 seq 的 begin 用）+ 启动对账结果。
        self._ledger_tail: list[dict[str, Any]] = []
        self._boot_reconcile: dict[str, Any] | None = None
        # F14-5：begin/end 行的层指纹暂存（begin 在写盘前算，end 在 core 里补齐）。
        self._begin_layer_fps: dict[str, str] = {}
        self._end_layer_fps: dict[str, str] = {}
        self.writers_active: list[dict] = []
        self.writers_stale_cleaned = 0
        # 本进程上次落盘时各层文件的戳（mtime_ns + size）。
        # 戳没变 = 没别人动过 = 不必读盘合并。
        self._layer_stamp: dict[str, tuple[int, int] | None] = {}
        # F14-1 / G0：本进程**上次落盘时**各层的内容指纹（层版本号）。
        # 「内存指纹 != 已写指纹」＝该层有改动还没落盘 —— 这正是切片 1 答不了
        # 的「哪层有改动却没写下去」，也是 G0 派生层脏集的判据（只写脏层）。
        self._layer_written_fp: dict[str, str] = {}
        # 上次 save 因"内容没变"而跳过的层（脏集效果的自曝）
        self._layers_skipped: list[str] = []
        # 墓碑：记录"删过什么、为什么删"，物理删除后仍可追溯（审计用）
        self.tombstones: list[dict[str, Any]] = []
        # 派生层删除账本。结构 {f"{层文件名}::{键}": 删除时刻}。
        # 内存里删掉一条并不等于"它不会回来" —— 别的进程手里还是旧快照，
        # 它一落盘就把这条并回来。只有记下"这个键在什么时候被删了"，
        # 合并时才能把迟到的旧版本挡住。
        self._layer_del: dict[str, float] = {}
        # 自曝：本次合并因为这张账本拦下了几条（0 = 没拦，也可能是没得拦）
        self._layer_del_blocked = 0
        # 自曝：上次 save 时账本瘦身丢掉几条（>0 意味着有墓碑到期了）
        self._layer_del_gc_last = 0
        # C1a：源文件签名层 {source 标签: {path, mtime_ns, size, sha1, sig_at,
        # first_seen}}。drift_audit 的事实源；内容事实没变就不动它（防写放大）。
        self.source_sigs: dict[str, dict[str, Any]] = {}
        # (路径, mtime_ns, size) → sha1 的内存缓存（批量同源导入只算一次）。
        # 纯性能缓存：进程重启丢得起，层文件才是事实源。
        self._sig_cache: dict[tuple[str, int, int], str] = {}
        # S2：update_tags 变更账本的内存态（行数 + 末条 seq）。磁盘才是事实源
        # （tag_ops.jsonl），这里只为了分配 seq 与裁剪判断。
        self._tag_ops_lines = 0
        self._tag_ops_last_seq = 0
        # 未命中留痕：查询召回不到时自动记账，喂归因分析。
        # 判定用"最高相似度是否达到硬门槛"，而不是"结果是否为空"——
        # BM25 单路救回来的条目 similarity 恰好是 0.0，若只看空结果会漏记。
        self.miss_log: list[dict[str, Any]] = []
        self.miss_log_threshold: float = MISS_LOG_THRESHOLD
        self.miss_log_enabled: bool = True
        # 按日活动计数（追加五）：``{"YYYY-MM-DD": {"recall": n, "ingest": n}}``。
        # 平行期评测要的是"该想起几次 vs 实际想起几次"，此前只能考古
        # ``recall_count`` 的总和，看不出哪天用过、用得是否均衡。
        self.daily_counts: dict[str, dict[str, int]] = {}

        self._ingest_since_consolidate = 0
        # 批量写入深度（>0 表示正在 ingest_batch 内）：批量时**不逐条触发**
        # 自动巩固，收尾统一跑一次——否则 500 条的一批会把 consolidate 触发几十次。
        self._batch_depth = 0
        # F11：已写入但还没落到 brain_raw.json 的 raw 条数。
        # 恒自曝（0 也要报）——「确定没欠账」和「压根没查」必须分得开。
        self._l0_pending = 0
        self._l0_autosave_every = L0_AUTOSAVE_EVERY
        self._nlp = None
        self._nlp_tried = False
        # spaCy 不可用时的原因。留空串表示"可用"或"还没试过"。
        # 为什么要记下来而不是静默置 None：装不上 spaCy 会让英文实体抽取
        # 退回启发式，召回结果悄悄变差而用户毫不知情——这正是本项目反复踩的
        # "静默用错默认值"那一类坑（对照：embedder 静默回落 MiniLM）。
        self._nlp_error = ""

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.load()
            # F14：读账本末条 —— 进程换了一茬，但账还在（这正是账本的意义）。
            self._ledger_last, self._ledger_lines, self._ledger_tail = (
                self._ledger_read_tail())
            # F14-5：启动对账——上次保存没走完 / 磁盘与账本不符，当场报出来，
            # 不再等 l0_dangling 事后反推。
            self._reconcile_ledger_on_boot()
            # S2：读标签变更账本（行数 + 末条 seq）——进程换了一茬，
            # undo 还要能找到上一个进程记的 op。
            self._tag_ops_lines, self._tag_ops_last_seq = (
                self._tag_ops_scan())
            # **启动即留名**。只在落盘时留名是不够的——两个进程
            # 同时连一个库、都还没写过东西时，writers/ 是空的，
            # 「有几个人在动这个库」的答案会是 0，而真实答案是 2。
            self._touch_writer()
            try:
                _atexit.register(self._unregister_writer)
            except Exception:  # noqa: BLE001 —— 注销失败不影响主流程
                pass






    def reload_tag_schema(self, schema: str | None = None) -> dict[str, Any]:
        """重新加载标签抽取配置（配置改了不用重启服务）。返回生效规则自述。"""
        if schema:
            self._explicit_tag_schema = schema
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        self.tag_extractor = load_extractor(self._explicit_tag_schema, persist_dir)
        return self.tag_extractor.describe()

    # 词库别名行的分隔符。**方向是"别名 → 规范名"**，与调用方的直觉一致
    # （读作"子安 指的是 顾言"）。
    _ALIAS_SEP = re.compile(r"\s*(?:=>|->|→|\u21d2)\s*")





    # ==================================================================
    # 实体抽取（专名词库 → 启发式；spaCy 仅英文）
    # ==================================================================
    #: 哨兵期把 torch 重新放行的开关（默认封堵；评测脚本需要真 torch 时设 1）
    NER_ALLOW_TORCH_ENV = "HIPPOCAMPUS_NER_ALLOW_TORCH"

    @staticmethod
    def _block_torch_reflow() -> bool:
        """spaCy 加载期间临时封堵 torch 回流。

        ``thinc/compat.py`` 顶部是 ``try: import torch … except ImportError``——
        只要 torch 装在环境里，首次英文 NER 就会把它（~160MB 常驻）拉进进程且
        不释放（实测确认，常驻不回落）。

        做法是 ``import spacy`` 前往 ``sys.modules`` 放 **None 哨兵**：
        ``import torch`` 会抛 ImportError，正好落在 thinc 的兜底分支
        （``has_torch=False``），en_core_web_sm 是 thinc numpy ops 管线，
        不需要 torch。返回 True 表示装了哨兵，**调用方必须负责摘除**。
        torch 已在场（评测脚本先 import torch）或
        ``HIPPOCAMPUS_NER_ALLOW_TORCH=1`` 时不装。
        """
        if "torch" in sys.modules:
            return False
        if load_config().ner_allow_torch:
            return False
        sys.modules["torch"] = None
        return True

    @staticmethod
    def _unblock_torch_reflow(installed: bool) -> None:
        """摘除哨兵。哨兵还在（没人偷换）才删；此后真实 torch 导入不受影响。"""
        if installed and sys.modules.get("torch") is None:
            del sys.modules["torch"]

    def _get_nlp(self):
        if self._nlp_tried:
            return self._nlp
        self._nlp_tried = True
        _blocked = self._block_torch_reflow()
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load("en_core_web_sm")
            self._nlp_error = ""
        except Exception as ex:  # noqa: BLE001
            self._nlp = None
            # 失败原因**必须留痕**：下面 ner_backend / brain_stats 会把它报出去。
            self._nlp_error = f"{type(ex).__name__}: {ex}"[:200]
        finally:
            self._unblock_torch_reflow(_blocked)
        return self._nlp

    @property
    def ner_backend(self) -> str:
        """英文实体抽取实际使用的后端。

        spaCy 装不上（缺包 / 缺 ``en_core_web_sm`` 模型）时退回纯启发式，
        **召回质量会变但不会报错**。这个属性让"到底用的哪个后端"变成可核对的
        事实，而不是要靠猜。中文路径本来就跳过 spaCy，不受此项影响。

        ⚠️ **访问有副作用**：会真正加载 spaCy（~92MB 常驻）。其 thinc.compat
        连带 import torch 的回流已被 :meth:`_block_torch_reflow` 哨兵拦下
        ），加载后进程仍零 torch。只要自曝、不要触发加载的
        场景请用 :attr:`ner_backend_lazy`。
        """
        if self._get_nlp() is not None:
            return "spacy"
        if self._nlp_error:
            return f"heuristic-only（spaCy 不可用：{self._nlp_error}）"
        return "heuristic-only"

    @property
    def ner_backend_lazy(self) -> str:
        """``ner_backend`` 的**无副作用**版。

        🔴 为什么要它：``_build_brain`` 的启动日志走 ``stats()`` → ``ner_backend``
        会把 spaCy（→ thinc → **torch**，实测 +160MB 级常驻）在服务启动那一刻
        强行拉进进程——自曝不该有加载副作用。已试过/已加载时如实报告；还没试
        过就明说"未加载"，不撒谎也不付成本。
        """
        if self._nlp is not None:
            return "spacy"
        if self._nlp_tried:
            if self._nlp_error:
                return f"heuristic-only（spaCy 不可用：{self._nlp_error}）"
            return "heuristic-only"
        return "lazy（未加载——首次英文实体抽取时按需加载，失败退 heuristic-only）"



    def extract_entities(self, text: str) -> list[str]:
        """抽取实体名（返回原始大小写，调用方自行 lower）。

        三条来源，按可靠性排序：
            1. **项目专名词库**（lexicon）：一行一个专名，命中即链——
               文档场景下比任何启发式都准（覆盖报告·高 6 的推荐方案）；
            2. 启发式规则（中文句式 + 《》书名号 + 引号 + 英文大写词）；
            3. spaCy（**仅英文文本**启用；中文文本跳过，en_core_web_sm 抓不出中文）。

        需要知道每个实体来自哪条路径时用 :meth:`extract_entities_with_origin`
        ——origin 决定它能不能参与检索加成/画像/矛盾检测。
        """
        return list(self.extract_entities_with_origin(text))

    def extract_entities_with_origin(self, text: str) -> dict[str, str]:
        """同 :meth:`extract_entities`，但返回 ``{实体名: origin}``。

        实现在 :func:`hippocampus_mcp.domain.entity_link.extract_entities_with_origin`
        本方法只做绑定：预处理/CJK 比/词库/NER/
        标题与启发式抽取/清洗/别名归一全部**调用时现取**并注入——
        四路次序、去重合并与可信度语义见新模块 docstring。
        """
        return _entity_link.extract_entities_with_origin(
            text,
            preprocess=self._preprocess_for_entities,
            cjk_ratio=self._cjk_ratio,
            cjk_skip_ratio=_CJK_SKIP_SPACY_RATIO,
            lexicon=self.entity_lexicon,
            get_nlp=self._get_nlp,
            title_entities=self._title_entities,
            heuristic_entities=self._heuristic_entities,
            clean_name=self._clean_entity_name,
            canonicalize_map=self._canonicalize_map,
            origin_rank=_ORIGIN_RANK,
            origin_lexicon=ENTITY_ORIGIN_LEXICON,
            origin_spacy=ENTITY_ORIGIN_SPACY,
            origin_title=ENTITY_ORIGIN_TITLE,
            origin_heuristic=ENTITY_ORIGIN_HEURISTIC,
        )


    _NOISE_CHARS = ("的", "是", "在", "了", "和", "与", "我", "你", "他", "她", "它",
                    "主要", "做了", "我们", "这个", "那个", "一只", "一个", "一条",
                    "叫", "有", "也", "就", "都", "还", "不", "要", "给", "对")

    # 纯 ASCII 且带数字的短串（H05 / B03 / ch11 / p12）几乎总是文档里的编号，
    # 不是实体——覆盖报告·高 6 点名的垃圾实体。
    _CODE_LIKE = re.compile(r"^(?=.*\d)[A-Za-z0-9_\-]{2,8}$")


    # 对话特征模式：中文里"我叫X/X公司/养了一只X"这类句式本身就框出了专名，
    # 无论文档还是对话场景都是高置信信号，中文下照常启用。
    _STATEMENT_PATTERNS = (
        r"我叫([\u4e00-\u9fa5A-Za-z0-9]{1,8})",
        r"我是([\u4e00-\u9fa5]{2,4})(?:，|,|。)",
        r"叫([\u4e00-\u9fa5]{1,6})的",
        r"养(?:了|的|着)?(?:一只|一条|一个)?([\u4e00-\u9fa5]{2,4})",
        r"([\u4e00-\u9fa5A-Za-z]{2,8})(?:品类|公司|品牌|项目|平台|团队)",
        r"在([\u4e00-\u9fa5]{2,4})(?:做|工作|生活|开|读)",
    )

    # `《》` 是中文唯一明确的"作品名"标记，任何场景都保留，且单列 title 受信任档。
    _TITLE_RE = re.compile(r"《([\u4e00-\u9fa5A-Za-z0-9 ]{2,12})》")

    # 引号模式：**仅英文文本启用**。中文里引号表示强调/引用/台词，不表示专名——
    # 实测中文小说文档库 832 个中文候选里 67%（560 个）由 ASCII 双引号产生，
    # 抓到的是「这笔账没法记」这类短语；真专名由词库与《》负责，不靠引号。
    _QUOTE_PATTERNS = (
        r"「([\u4e00-\u9fa5A-Za-z0-9 ]{2,12})」",
        r'"([\u4e00-\u9fa5A-Za-z0-9 ]{2,12})"',
    )



    def _link_entities(
        self,
        names: list[str],
        memory_id: str,
        ts: float,
        origins: dict[str, str] | None = None,
    ) -> list[str]:
        """把实体写入实体表并维护倒排索引，返回实体 id 列表。

        实现在 :func:`hippocampus_mcp.domain.entity_link.link_entities`
        本方法只做绑定：实体表本体（就地修改，
        容器不搬）、id 生成器与 origin 常量**调用时现取**并注入；
        去重/升级/派生量语义见新模块 docstring。
        """
        return _entity_link.link_entities(
            names, memory_id, ts, origins,
            entities=self.entities, nid=_nid,
            origin_rank=_ORIGIN_RANK,
            origin_heuristic=ENTITY_ORIGIN_HEURISTIC,
        )












    # ==================================================================
    # 源文件签名与漂移审计（借鉴前作的条目绑源 + verify 四分类）
    # ==================================================================
    def _resolve_source_path(self, source: str) -> str | None:
        """把 source 标签解析成磁盘上的绝对路径；解不出返回 None。

        解析次序：绝对路径 → 记忆库目录 → ``HIPPOCAMPUS_SOURCE_ROOT`` →
        进程 CWD。**解不出是常态而不是错误**：source 也承载会话 id、
        自由文本来源（"群聊" / "口头"），它们本来就没有对应文件——
        对它们记一份假签名只会把 unbaselined 池搅浑。
        """
        return resolve_source_path(source, self.persist_dir,
                                   load_config().source_root)

    def _file_sig(self, path: str) -> dict[str, Any] | None:
        """算一个文件的 {mtime_ns, size, sha1}；带 (路径, mtime, size) 缓存。

        缓存键含 mtime/size：文件被改（mtime 变）自然重算，没改则同一
        进程内反复 ingest 同一文件只算一次哈希。OSError（文件正在被删 /
        无权限）返回 None——调用方按"解不出"处理，不炸写入路径。
        """
        return file_sig(path, self._sig_cache)

    def record_source_sig(self, source: str) -> dict[str, Any] | None:
        """写入侧：给一个 source 标签记源文件签名（ingest 调用）。

        **内容事实没变就不动层**：签名条目含 sig_at，若每次 ingest 都重盖，
        层指纹每次都变 ⇒ 每次 save 都写盘（批量导入 500 条 = 500 次写盘）。
        没变时连 dict 都不碰，层保持干净。

        Returns:
            当前生效的签名条目（新记或既有），解不出路径 / 开关关着返回 None。
        """
        return _record_source_sig(
            source, self.source_sigs, self.persist_dir,
            load_config().source_root, self._sig_cache,
            sig_enabled=load_config().source_sig)

    def source_drift(self, source: str = "", limit: int = 50,
                     include_ok: bool = False) -> dict[str, Any]:
        """C1b：源文件漂移审计——"文件改了但记忆没更新"的服务端答案。

        借鉴前作的 verify 四分类，按记忆库的实际形态落地为：

        ============== ======================================================
        分类           含义
        ============== ======================================================
        ``stale``      记过签名、文件还在，但 mtime/size/sha1 有任一不同
                       ⇒ 该源下的记忆可能已过期（文件是较新真相）
        ``missing``    记过签名、文件不见了 ⇒ 记忆还在、源没了（imported_
                       but_deleted）；回查原话的链路已断
        ``unbaselined``能解析到文件、但**从没记过签名**（本功能上线前导入的
                       存量、或 ``HIPPOCAMPUS_SOURCE_SIG=0`` 期间写的）⇒
                       不是"没漂移"，是"无法判断"——单列一类，不混进 ok
        ``orphan``     记过签名、但库里没有任何记忆引用这个 source ⇒ 签名
                       条目本身过期了（记忆被 forget 干净 / 换标签重导）
        ``unresolvable``source 不是文件（会话 id / 自由文本）⇒  informational，
                       不参与漂移判定
        ``ok``         记过签名、文件还在、内容事实一致
        ============== ======================================================

        每条带**分母**（各类计数 + 总源数 + 引用记忆数）：``count:0`` 有两种
        真相（真没漂移 vs 压根没采集），不报分母调用方一律读成前者——
        与 contradictions 的 ``candidate_pool`` 同一条纪律。

        **不做**的事：``never_ingested``（某目录下的文件从没导入过）需要知道
        "监视目录"是什么，那是项目侧脚本的输入（``export_memories(source=)``
        已能算），本工具不假设监视目录。

        Args:
            source: 只审计这一个 source（空 = 全库）。
            limit: 每类最多回多少行（0 = 只给计数）。
            include_ok: 是否把 ``ok`` 的源也列出来（默认只列有问题的）。
        """
        # 记录流 / 签名层 / sha1 缓存由本层注入；两旋钮调用时现取。
        return drift_audit(
            self.manager.store.iter_records(), self.source_sigs,
            source=source, limit=limit, include_ok=include_ok,
            persist_dir=self.persist_dir, source_root=load_config().source_root,
            sig_enabled=load_config().source_sig, sig_cache=self._sig_cache)


    # ==================================================================
    # 写侧：L0 原始 + L1 萃取
    # ==================================================================
    def ingest(
        self,
        text: str,
        speaker: str = "user",
        timestamp: float | None = None,
        importance: float | None = None,
        cognitive_type: str | None = None,
        entities: list[str] | None = None,
        session_id: str | None = None,
        source: str = "",
        detect_supersede: bool | None = None,
        granularity: str = "sentence",
        tags: dict[str, Any] | None = None,
        auto_tags: bool = False,
        client: str | None = None,
        quarantine: bool = False,
        sensitivity: str = "",
        origin: dict[str, str] | None = None,
        idem_key: str = "",
    ) -> dict[str, Any]:
        """写入一条原始消息并萃取为 L1 记忆。

        这是宿主 agent 最常用的写入口：原始文本进 L0（一字不改），
        抽取出的句子/事实进 L1（带 source_ids 回指 L0）。

        Args:
            source: 来源标签（文件名 / 相对路径 / 会话标识）。会写进 L0 与
                L1，recall 命中结果里一并返回，让"这条记忆出自哪个文档"
                可回查（覆盖报告·高 3）。
            detect_supersede: 本次是否做事实更新检测。None = 跟随实例开关
                ``self.supersede_enabled``。批量导入逐条传 False。
            granularity: ``"sentence"``（默认，按句拆成多条 L1）或
                ``"chunk"``（整段作为一条 L1）。文档导入时 chunk 粒度可显著
                减少条数与后续噪声（覆盖报告·中 7 的可配置项）。
            tags: 结构化标签，如 ``{"章": 43, "年": "己丑"}``。写进本批
                每条 L1，供 recall/get_memory_context 按条件过滤——强约束写作
                项目里"按章/按年查"的高频场景从此是查而非猜。命中去重合并时
                取并集（见 :func:`merge_tag_dicts`）。
                行内可写别名（词库 ``别名 => 主名``），实体两侧都会归一。
            auto_tags: 由服务端从**正文**自动抽标签（默认关）。
                **这是写入侧唯一能让标签覆盖率动起来的开关**：靠事后回填只能
                从已有正文里捞（中文小说库 9359 条实测上限 17.1%），而写入时
                抽取同样受"正文里得有标记"这个物理边界限制——所以它并不能
                凭空提高覆盖率，但能让**新写入**的条目从一开始就带上标签，
                且宿主不必自己实现一遍抽取。显式传入的 ``tags`` 优先，自动
                结果只补空缺键（:func:`merge_tag_dicts`）。

        Returns:
            除原有字段外，还带两项**写入侧回显**：
                - ``merged_into``: 命中去重、本次内容并进去的既有记忆（未新建）；
                - ``neighbors``: 与本条最像的 top3 既有记忆，带 state /
                  superseded_by / source，用于当场发现重复口径与已作废的旧版本。
        """
        # ts None 语义（None 才取时钟，0.0 也显式）
        # → ingest_entry.resolve_timestamp（SessionC 批C1）。
        ts = _ingest_entry.resolve_timestamp(timestamp, now=_now)
        # P0-2 幂等前置（必须在 raw_log.append **之前**）：同一 idem_key 重放
        # 时直接回既有 id，不写 L0 也不写 L1。放到后面会留下一份一字不差的
        # 孤儿原文——"没新建记忆"却多了审计记录，是最难排查的那种不一致。
        if idem_key:
            _hit = self.manager.idem_lookup(idem_key)
            if _hit:
                # 返回体本体在 domain/ingest_entry.py::duplicate_body
                # （S1 结构守卫钉着壳 ≤170 行；这里只留「查 → 回」两行）。
                return _ingest_entry.duplicate_body(_hit)
        # C1a：写入侧记源文件签名（drift_audit 的事实源从这里开始攒）。
        # 放在最前面：哪怕后面分句/萃取抛异常，"这个源被读过"也已成立。
        # 解不出路径的 source（会话 id 等）不记——见 record_source_sig。
        self.record_source_sig(source)
        # auto_tags 合并（本体在 domain/ingest_pipeline.py）：
        # extract 与 merge 两个依赖调用时现取后注入。
        tags = _ingest_pipeline.resolve_tags(
            tags, auto_tags=auto_tags, text=text,
            extract=self.tag_extractor.extract,
            merge=merge_tag_dicts)
        # 闸一·legacy 侧守卫：带「地图」+「条目」
        # 保留标签的写入 = 声明一个地图条目。同键已有**活**条目时拒绝——
        # 「旧状态与新状态竞争」正是设计评审否掉「只当记忆管」的失败
        # 形态，ingest 不能成为绕过 map_set / update_tags 的后门
        # （legacy 全靠 agent 自觉，这是把自觉变成机制的那一步）。
        # 归档过的同名条目不挡：那是死条目，重建是正当动作。
        self._guard_map_entry_dup(tags)
        raw_id = _nid("raw")
        # L0 组装 → ingest_entry.build_raw_entry（批C1；append 留壳，
        # session_id 短路 / source 兜底在本体复刻）。
        self.raw_log.append(_ingest_entry.build_raw_entry(
            raw_id=raw_id, text=text, speaker=speaker, timestamp=ts,
            session_id=session_id, source=source,
            resolve_session=self.current_session_of, client=client))
        # F11：L0 是唯一没有自动落盘的层（L1 有 manager.auto_save）。
        # 这里先记账，落盘时机交给 _maybe_autosave_l0：批内不落，
        # 由 ingest_batch 收尾统一落一次，避免 500 条一批重写 500 次。
        self._l0_pending += 1

        # 认知类型自动判定 + 重要性自动判定：
        # 块级 importance 仅用于返回给调用方的摘要值，两个判定器注入。
        _basics = _ingest_pipeline.guess_basics(
            text, speaker, cognitive_type, importance,
            guess_cognitive=self._guess_cognitive_type,
            guess_importance=self._guess_importance)
        cognitive_type = _basics.cognitive_type
        importance = _basics.importance
        explicit_importance = _basics.explicit_importance

        # 实体来源分级（origin）：宿主显式传入的最可信（manual），
        # 抽取得到的按各自路径标注。origins 以小写名为键，与 _link_entities 对齐。
        # 合并本体在 domain/policies.py（含别名归一与 rank 升级）。
        names, origins = _policies.merge_entity_origins(
            entities, self.extract_entities_with_origin(text),
            canonical=self.canonical_entity,
        )
        entity_ids = self._link_entities(names, "", ts, origins)

        # 新颖度/预测误差门控。分句判定前与既有图式
        # （L3 画像 / L2T 知识点摘要 / 高置信 L1）比对：印证→不加强
        # （重复归 dedup 管）；矛盾→importance 提升 + 冲突对象回显；
        # 全新实体→小幅加权。构造本体在 :meth:`_novelty_gate_guess`
        # （保持本壳薄——batch19 S1 结构守卫）；显式 importance 整体旁路；
        # dedup/supersede 判据零接触；纯判定在 domain/novelty_gate.py。
        novelty_events: list[Any] = []
        gate_guess = self._novelty_gate_guess(novelty_events, explicit_importance)

        # 分句 → 每条作为一条 L1 记忆（太短的句子并入上一条）；
        # chunk 粒度则整段一条，不拆。
        # 分句本体在 domain/ingest_pipeline.py::split_sentences。
        sentences = _ingest_pipeline.split_sentences(
            text, granularity, split_facts=self._split_facts)
        # memory_type 是循环不变量：CognitiveType 住在 store 包，domain 看不见，
        # 由本层算好一次传进写循环。
        memory_type = _ingest_pipeline.memory_type_for(
            cognitive_type, CognitiveType.EXPERIENCE.value)
        # 分句写循环本体在 domain/ingest_flow.py：
        # L1 端口 + 实体链接 + 并集合并 + 重要性判定四个依赖注入。
        _sw = _ingest_flow.write_sentences(
            _ingest_flow.SentenceWritePorts(
                manager=self.manager,
                link_entities=self._link_entities,
                merge_ids=merge_id_lists,
                guess_importance=gate_guess,
                # D2a：非家工作区经 user 豁免通道的 ingest，整批 L1 卡
                # 带隔离标记（判据在组合根工厂算好传入，brain 不判定）。
                quarantine=quarantine,
                sensitivity=sensitivity,
                origin=origin,
                idem_key=idem_key,
            ),
            sentences=sentences,
            importance=importance,
            explicit_importance=explicit_importance,
            memory_type=memory_type,
            cognitive_type=cognitive_type,
            timestamp=ts,
            raw_id=raw_id,
            entity_ids=entity_ids,
            source=source or "",
            tags=tags,
            names=names,
            origins=origins,
        )
        memory_ids = _sw.memory_ids
        merged_into = _sw.merged_into
        kept_separate = _sw.kept_separate
        truncations = _sw.truncations

        # 事实更新检测：新记忆是否取代了旧的同一事实。三态归一
        #（None 跟随实例开关）→ ingest_entry.resolve_supersede_switch（批C1）。
        do_detect = _ingest_entry.resolve_supersede_switch(
            detect_supersede, instance_enabled=self.supersede_enabled)
        superseded = self._detect_supersede(memory_ids, ts) if do_detect else []

        # 写入侧回显"与本条最像的既有记忆"（zcode 需求 A3）：
        # 双写落盘时当场就能看到旧口径，立刻决定要不要显式作废，
        # 而不是等矛盾积累到 consolidate 才发现。
        neighbors = self._near_duplicates(memory_ids, top_k=3)

        # C10：把「因为含新硬信息而故意没合并」的老邻居标出来。
        # 宿主就算只看 neighbors 也能发现：这两条很像，但不是重复写入。
        # 邻居标注本体在 domain/ingest_pipeline.py::annotate_neighbors。
        _ingest_pipeline.annotate_neighbors(neighbors, kept_separate)

        # 按日写入计数（追加五）：记**条数**而不是"调用次数"——ingest_batch
        # 一次可以落几十条，记次数会低估批量导入的量级。
        self.bump_daily("ingest", max(len(memory_ids), 1))
        # 把本次写入的记忆 id 累积到当前会话，session_end 挂到 L2 摘要。
        self._note_session_memory(client, memory_ids)

        self._ingest_since_consolidate += 1
        if self._batch_depth == 0:
            self._maybe_auto_consolidate()
            self._maybe_autosave_l0()

        # 字段顺序是有意的：**merged_into 摆在第一个**（用户反馈后调整）。
        # 命中去重时"本次内容没被单独存下来"是宿主最该先看到的事——
        # 此前它排在倒数第二，被 raw_id / memory_ids 挤到末尾，
        # 宿主只看开头几个字段时等于没看见（实测：新数字被 0.85 阈值吞掉
        # 却毫无察觉，只能靠改写措辞降相似度来救）。JSON 保序，故这里用顺序传达优先级。
        # 结果装配本体在 domain/ingest_pipeline.py::assemble_ingest_result。
        result = _ingest_pipeline.assemble_ingest_result(
            merged_into=merged_into,
            raw_id=raw_id,
            memory_ids=memory_ids,
            entities=names,
            cognitive_type=cognitive_type,
            importance=importance,
            superseded=superseded,
            tags=tags,
            neighbors=neighbors,
            kept_separate=kept_separate,
            l0_pending=self._l0_pending,
            truncations=truncations,
            max_seq_length=self._embed_max_seq_length(),
        )
        # O5 回显（零痕迹纪律）：门控没跑或没有任何新颖度信号时不加键。
        # 键落尾——不扰动既有键序（键序即优先级，见 ingest docstring）。
        _novelty = _novelty_gate.summarize_events(novelty_events)
        if _novelty is not None:
            result["novelty"] = _novelty
        return result

    def _novelty_gate_guess(self, events: list[Any],
                            explicit_importance: bool) -> Callable[[str], float]:
        """O5：构造门控版分句判定器（判定核在 ``domain/novelty_gate.py``）。

        从既有图式（L3 画像文本 / L2T 知识点摘要 / 高置信 L1 取最新
        ``SCHEMA_L1_CAP`` 条）构目标集；图式为空（新库）、门关
        （``HIPPOCAMPUS_NOVELTY_GATE``，A 型默认开）或宿主显式传了
        importance（宿主说了算）时原样返回关键词判定器——**零成本旁路**。
        nlargest 而非全排序：ingest_batch 逐条走这里，排序 O(N log N)/条，
        取前 cap 是 O(N)，大库批量导入不为此多付一个数量级。

        Args:
            events: 出参——每次分句判定把 ``(句子, NoveltyOutcome)`` 追加进
                此列表（ingest 收尾汇成 ``novelty`` 回显）。
            explicit_importance: 宿主是否显式传了 importance（真 = 旁路）。
        """
        base_guess = self._guess_importance
        if explicit_importance or not load_config().novelty_gate:
            return base_guess
        targets = _novelty_gate.build_schema_targets(
            profile_texts=[
                it.get("text", "")
                for _items in self.profile.values() for it in _items],
            topic_texts=[
                t.get("summary", "") for t in self.knowledge.values()],
            high_conf_records=heapq.nlargest(
                _novelty_gate.SCHEMA_L1_CAP,
                ((r.id, r.content, r.importance, r.timestamp)
                 for r in self.manager.store.iter_records()
                 if r.state == MemoryState.ACTIVE.value
                 and r.importance >= _novelty_gate.HIGH_CONF_IMPORTANCE),
                key=lambda t: (t[3], t[0])),
            tokens_of=self._tokens_of,
        )
        if not targets:
            return base_guess
        known = frozenset(self.entities.keys())

        def _gated_guess(sent: str) -> float:
            out = _novelty_gate.gate_sentence(
                sent, base_importance=base_guess(sent),
                sentence_entities=self.extract_entities(sent),
                known_entity_keys=known, targets=targets,
                tokens_of=self._tokens_of,
                pin_importance=PIN_IMPORTANCE)
            events.append((sent, out))
            return out.importance

        return _gated_guess

    def _embed_max_seq_length(self) -> int:
        """嵌入模型的 token 硬上限（自曝用）。取不到返回 0 = 未知。

        BGE / MiniLM 一系是 512。超过它的文本在 encode 时被静默丢弃，
        对向量零贡献——写入成功、回读在库、stats 正常，只有检索悄悄差一截。
        """
        try:
            emb = getattr(self.manager, "embedder", None)
            if emb is None:
                return 0
            limit = getattr(emb, "max_seq_length", 0)
            return int(limit) if limit else 0
        except Exception:  # 自曝不得反过来把写入搞挂
            return 0

    def _maybe_auto_consolidate(self) -> bool:
        """达到阈值就跑一次巩固（``auto_consolidate_every`` 为 0 表示关闭）。

        抽出来是为了让 :meth:`ingest_batch` 能在**收尾统一**判定一次，
        而不是在批内被触发几十次。
        """
        if (self.auto_consolidate_every > 0
                and self._ingest_since_consolidate >= self.auto_consolidate_every):
            self.consolidate()
            self._ingest_since_consolidate = 0
            return True
        return False

    def _maybe_autosave_l0(self) -> bool:
        """F11：累积的新 raw 到阈值就**只落 L0** 一次。返回是否落了。

        为什么单独给 L0 做一条自动落盘：L1 有 ``manager.auto_save``（每 10 步），
        L0（``brain_raw.json``）却只由显式 ``save()`` / ``save_state`` / ``session_end``
        落盘 ⇒ **宿主回收进程**即得「L1 在、L0 不在」，L1 的 ``source_ids``
        指向不存在的 raw（实打实发生过的事故）。这不是优化，是补一个不对称。
        """
        every = self._l0_autosave_every
        if every < 0:
            return False          # 显式关闭：只自曝欠账，不自动写盘
        if self._batch_depth > 0:
            return False          # 批内不落：交给 ingest_batch 收尾统一落一次
        if every > 0 and self._l0_pending < every:
            return False
        return self._save_l0_only()

    def _save_l0_only(self) -> bool:
        """只落 ``brain_raw.json``（L0）。成功返回 True，失败返回 False 并自曝。

        实现在 :func:`hippocampus_mcp.store.layer_gc.save_l0_only`
        本方法只做绑定 + 副作用记账：落盘核心在
        store（异常向上抛），``last_save_error`` 记账 / logger 自曝 /
        ``_l0_pending`` 清零留在壳。
        """
        if not self.persist_dir:
            return False
        try:
            self.raw_log = _layer_gc.save_l0_only(
                persist_dir=self.persist_dir, raw_log=self.raw_log,
                multi_writer_fn=multi_writer_enabled,
                read_raw_fn=self._read_raw_log_from_disk,
                merge_raw_fn=self._merge_raw_log,
                backend=self._backend, get_backend_fn=get_backend,
                layer_stamp=self._layer_stamp)
            self._l0_pending = 0
            return True
        except Exception as e:  # noqa: BLE001 —— 见 docstring：不抛，但自曝
            self.manager.last_save_error = (
                f"L0 自动落盘失败：{type(e).__name__}: {e}")
            logger.warning("[l0_autosave] %s", self.manager.last_save_error)
            return False

    def ingest_batch(
        self,
        items: list[dict[str, Any]],
        source: str = "",
        tags: dict[str, Any] | None = None,
        granularity: str = "sentence",
        detect_supersede: bool | None = None,
        save: bool = False,
        auto_tags: bool = False,
        client: str | None = None,
        quarantine: bool = False,
        sensitivity: str = "",
        origin: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """批量写入多条消息（zcode D5）

        逐条语义与 :meth:`ingest` 完全一致，差别只在"批量"这三件事上：

        - **逐项错误隔离**：某一项抛异常只记进 ``errors``，不中断整批。
          批量场景里"第 37 条格式不对"不该让前 36 条白写。
        - **不逐条触发自动巩固**：批内计数被抑制，收尾用
          :meth:`_maybe_auto_consolidate` 统一判定一次。
        - **不逐条落盘**：批内关闭 ``manager.auto_save``（落盘是**整体覆写**，
          逐条 save 的代价是 O(N × 库体积)），收尾按需保存一次。

        Args:
            items: 每项一个 dict，字段是 :meth:`ingest` 的同名子集——
                ``text``（**必填**，非空字符串）、``speaker`` / ``timestamp`` /
                ``importance`` / ``cognitive_type`` / ``entities`` / ``source`` /
                ``tags`` / ``granularity`` / ``detect_supersede``。
                **项级字段覆盖批次级默认**（整批同 source 时不必逐项写）。
                ``importance`` 传负数视为 None（自动判定，与 MCP 侧一致）；
                ``entities`` 接受列表或逗号分隔字符串；``tags`` 接受 dict 或
                JSON 对象字符串。
            source / tags / granularity / detect_supersede: 批次级默认值。
            auto_tags: 批次是否逐项从正文抽标签（见 :meth:`ingest`）；
                项里写 ``auto_tags`` 以项为准。
            save: 收尾是否显式落盘一次。若 ``manager.auto_save`` 原本为真，
                即使 ``save=False`` 也会在收尾保存一次（保住"开了自动保存就
                会落盘"的契约，只是从 N 次变成 1 次）。

        Returns:
            ``count`` 成功条数 · ``failed`` 失败条数 · ``total_input`` 输入条数 ·
            ``raw_ids`` / ``memory_ids``（全批展开）· ``entities``（去重并集）·
            ``merged_into``（每项附 ``index`` 指回入参）· ``superseded``（数量）·
            ``items``（逐项摘要）· ``errors``（``[{index, error, preview}]``）·
            ``saved`` / ``save_warning``。
        """
        _batch_flow.check_capacity(items, max_items=INGEST_BATCH_MAX)

        rows: list[_batch_flow.CollectedRow] = []
        errors: list[dict[str, Any]] = []

        prev_auto_save = self.manager.auto_save
        self.manager.auto_save = False
        self._batch_depth += 1
        try:
            for idx, it in enumerate(items):
                # normalize_item 必须在 try 内：项级 tags 写坏时 json.loads
                # 抛 ValueError，那属于「这一项坏了」，不是「整批坏了」。
                plan: Any = None
                try:
                    plan = _batch_flow.normalize_item(
                        idx, it,
                        default_tags=tags, default_source=source,
                        default_granularity=granularity,
                        default_detect_supersede=detect_supersede,
                        default_client=client, default_auto_tags=auto_tags)
                    if isinstance(plan, _batch_flow.ItemError):
                        errors.append({"index": plan.index, "error": plan.error,
                                       "preview": plan.preview})
                        continue
                    # D2a：隔离是批次级**策略**（服务层判定的结果），不开放
                    # 项级覆盖——注入在 normalize 之后，项里即便塞了同名键
                    # 也会被盖掉。
                    payload = dict(plan.payload)
                    payload["quarantine"] = quarantine
                    payload["sensitivity"] = sensitivity
                    # P0-1：写入凭据整批共用（批次级策略，同隔离标记；
                    # 项里即便塞了同名键也被盖掉）。批量不接幂等键——
                    # 文档导入的重放判据是源文件签名（source_drift），
                    # 不是请求键。
                    payload["origin"] = origin
                    res = self.ingest(**payload)
                except Exception as e:  # noqa: BLE001 —— 逐项隔离，不让一项毁掉整批
                    # plan 可能是 None（normalize_item 自己炸了，典型是项级
                    # tags 的 JSON 解析失败）——此时 preview 仍要是那句话，
                    # 不能退化成整个 dict 的 str（parity 抓出来的口径）。
                    errors.append({
                        "index": idx, "error": f"{type(e).__name__}: {e}",
                        "preview": _batch_flow.error_preview(it, plan)})
                    continue
                rows.append(_batch_flow.collect_item(
                    res, index=idx, source=plan.source_str))
        finally:
            self._batch_depth -= 1
            self.manager.auto_save = prev_auto_save

        # 收尾统一巩固一次（批内被抑制）
        if rows:
            self._maybe_auto_consolidate()

        out = _batch_flow.assemble_output(
            rows, errors=errors, total_input=len(items),
            max_seq_length=self._embed_max_seq_length(),
            l0_pending=self._l0_pending)

        if rows and (save or prev_auto_save):
            try:
                self.manager.save()
                # F11：L1 落了而 L0 不落，等于 L1 的 source_ids 指向不存在的
                # raw。这里补上；失败不中断（与上面同一套约定），写进 warning。
                l0_ok = self._save_l0_only()
                if l0_ok:
                    self.manager.last_save_error = None
                else:
                    out["save_warning"] = (
                        f"L0 未落盘，{self._l0_pending} 条 raw 仅在内存中："
                        f"{self.manager.last_save_error or '未知原因'}。"
                        f"请调用 save_state。"
                    )
                out["saved"] = True
            except Exception as e:  # noqa: BLE001
                # 与 store_memory 的自动保存同一套约定：不中断，但绝不静默。
                out["save_warning"] = (
                    f"落盘失败，本批记忆仅在内存中：{type(e).__name__}: {e}。"
                    f"请调用 save_state；若报 conflict 则先 load_state 或 retry_save。"
                )
                self.manager.last_save_error = f"{type(e).__name__}: {e}"
                self.manager.autosave_failures += 1
                logger.warning("[ingest_batch] %s", out["save_warning"])
        return out

    def _near_duplicates(
        self,
        memory_ids: list[str],
        top_k: int = 3,
        min_sim: float = INGEST_NEIGHBOR_MIN_SIM,
    ) -> list[dict[str, Any]]:
        """找出与"刚写入的这批记忆"最像的**既有**记忆。

        只看向量相似度（不再做矛盾判定——那是 consolidate 的活），
        但会把 ``state`` / ``superseded_by`` / ``source`` 一并给出：
        已作废的旧版本、来自别的文件的同义表述，都是宿主当场该看到的。

        实现在 :func:`hippocampus_mcp.domain.pure_checks.near_duplicates`
        本方法只做绑定：store / 向量查询**调用时现取**
        并注入；``min_sim`` 的默认参数保持 def 时绑定的既有形态。
        """
        return _pure_checks.near_duplicates(
            memory_ids,
            store=self.manager.store,
            vector_of=self.manager.vector_of,
            top_k=top_k,
            min_sim=min_sim,
        )




    def _detect_supersede(self, new_memory_ids: list[str], ts: float) -> list[dict[str, str]]:
        """检测"同一事实被更新"：新记忆含更新标记 + 与旧记忆高度相似 + 共享实词
        → 只取代最相似的那一条（避免误伤无关记忆）。

        三重条件缺一不可：更新标记 + 相似度阈值 + （共同实体 或 高实词重叠），
        防止"花生过敏"这类无关长期事实被"现在养猫"误标为已取代。

        实词重叠门槛由 0.3 提高到 ``SUPERSEDE_OVERLAP``（0.6）：中文 bigram
        口径下，同一份文档里的两段话轻易就能超过 0.3，于是文档沿革叙述
        （"X 降位"/"改成 Y"本就是这类文档的常态）被大面积误判为事实更新
        ——覆盖报告·高 4 实测 28 对误杀，含"张岱死期 1689"这种真事实。

        实现在 :func:`hippocampus_mcp.store.supersede.detect_supersede`本方法只做绑定：store / 向量查询 / 更新标记词表 /
        判据 ``_is_restatement`` 全部**调用时现取**并注入——判据不 import
        是为了保住 monkeypatch 面，也让读侧标注能用同一份判据。
        """
        return _detect_supersede_fn(
            new_memory_ids, ts,
            store=self.manager.store,
            vector_of=self.manager.vector_of,
            is_restatement=self._is_restatement,
            update_markers=UPDATE_MARKERS,
        )




    @classmethod
    def _is_restatement(
        cls,
        sim: float,
        text_a: str,
        text_b: str,
        ents_a: Any = None,
        ents_b: Any = None,
        toks_a: set[str] | None = None,
        toks_b: set[str] | None = None,
    ) -> bool:
        """两条记忆是否"在讲同一件事"（疑似同一事实的不同口径）。

        **判据只此一处**：写入侧（:meth:`_detect_supersede` 判"谁取代谁"）与
        读取侧（:meth:`_annotate_conflicts` 判"谁与谁并存冲突"）共用它。

        为什么要共用一个判据：写侧受限于时机——``supersede`` 关着（文档导入
        的默认）、老数据早于该机制、或当时没配对上的，都会留下**并存的新旧
        口径**；读侧拿到同一判据就能把这些补标出来。若两边各写一套阈值，
        迟早出现"写时判为同一事实、读时判为不同事实"的自相矛盾。

        条件：相似度达标（``sim >= SUPERSEDE_SIM``）**且**（共享实体 **或**
        实词重叠达标（``>= SUPERSEDE_OVERLAP``）。单一条件不够——中文
        bigram 口径下同文档两段话很容易越过重叠门槛，只看重叠会把沿革
        叙述大面积误判（覆盖报告·高 4 实测 28 对误杀）。

        M2：判定核已迁 ``domain.policies.is_restatement``（阈值常量
        ``SUPERSEDE_SIM`` / ``SUPERSEDE_OVERLAP`` 随迁），此处为同名
        委托——写入/读取两侧的调用点一行不动，行为由
        ``tests/test_policies_golden.py``（44 例）钉住。
        """
        return _policies.is_restatement(
            sim, text_a, text_b, ents_a, ents_b, toks_a, toks_b)

    # ==================================================================
    # 读侧：召回 + 5 槽位上下文
    # ==================================================================
    def _sync_bm25(self) -> None:
        """把 BM25 索引同步到当前存活记录（增量，覆盖报告·低 14）。

        维护一张 ``id -> 内容指纹`` 表逐条比对：只在**确实新增/改动/删除**时
        才动索引，未变的文档不重新分词。原先每次 recall 都对全部存活记录
        重跑 ``index()``——8k 条无感，量级上去就是查询延迟的大头。
        """
        current: dict[str, tuple[int, str]] = {}
        for r in self.manager.store.iter_records():
            if r.state == MemoryState.ARCHIVED.value:
                continue
            current[r.id] = (zlib.crc32(r.content.encode("utf-8")), r.content)

        for did in list(self._bm25_indexed):
            if did not in current:
                self._bm25.remove_doc(did)
                del self._bm25_indexed[did]
        for did, (fp, content) in current.items():
            if self._bm25_indexed.get(did) == fp:
                continue
            self._bm25.add_doc(did, content)
            self._bm25_indexed[did] = fp

    def _fts_hits(self, query: str, top_n: int) -> dict[str, float] | None:
        """走 SQLite FTS5 做字面检索。

        Returns:
            ``None`` = 这条路用不了（json 后端 / 未开 FTS5 / 词太短），
            调用方必须回退内存 BM25；``dict`` = 索引结果，**哪怕为空**也是
            权威结论，不要再回退（回退＝白建一次内存索引）。

        失败一律吞掉并回退：全文索引是**加速项**，它挂了不该让 recall 挂。
        """
        backend = getattr(self.manager.store, "_backend", None)
        if backend is None or getattr(backend, "name", "") != "sqlite":
            return None
        try:
            hits = backend.fts5_search(query, top_n)
        except Exception as e:      # noqa: BLE001
            logger.debug("FTS5 检索不可用，回退内存 BM25: %s", e)
            return None
        if hits is None:
            return None
        # 同一 id 理论上只出现一次，dict 化顺便去重
        return dict(hits)

    def lexical_self_report(self) -> dict[str, Any]:
        """字面通道（BM25 / FTS5）的运行期自曝。

        为什么要这一节：sqlite + FTS5 最大的卖点是「**不用再建内存 BM25 索引**，
        省 34.5MB」，而这件事在生产侧此前**一个观测手段都没有** —— 值守清单第 4 步
        的验收条件写着 `_bm25_indexed=0`，但 `get_stats` 里根本没有这个字段，
        于是「到底省没省」只能靠猜。这里按 B2-4 的规矩全部自曝**实测值**：

        - ``store_backend``：实际存储后端（json / sqlite）
        - ``fts5_ready``：FTS5 虚表在本连接里是否真的可用
        - ``bm25_indexed``：内存 BM25 现有几条 —— **H4 真正生效时这里必须是 0**
        - ``last_lexical_path``：上次检索的字面通道实际走的路（fts5 / memory / off）
        """
        # 本体在 domain/stats_views.py：装配层只负责取
        # store._backend（双重 getattr 防御）与两个 brain 实例属性。
        backend = getattr(getattr(self.manager, "store", None), "_backend", None)
        return _lexical_self_report_fn(
            backend,
            len(self._bm25_indexed),
            self._last_lexical_path,
        )

    def _ids_matching_tags(self, filter_tags: dict[str, Any] | None) -> set[str] | None:
        """按结构化标签筛出记忆 id 白名单。

        语义：**所有**键都必须匹配（AND）；单个键的值可以是标量或列表
        （列表内取 OR）。比较前统一 ``str()`` 归一——标签经 JSON 往返后
        可能是 ``43`` 也可能是 ``"43"``，不做归一会让过滤静默失效。

        Returns:
            None 表示"无过滤"（调用方不要把它当成空集）。
        """
        return ids_matching_tags(self.manager.store.iter_records(), filter_tags)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        include_superseded: bool = False,
        use_mmr: bool = True,
        log_miss: bool = True,
        filter_tags: dict[str, Any] | None = None,
        annotate_conflicts: bool = True,
        client: str | None = None,
        occurred_after: float = 0.0,
        occurred_before: float = 0.0,
    ) -> list[dict[str, Any]]:
        """召回记忆：硬门槛过滤 + MMR 去重 + 钉住豁免衰减 + 实体加成。

        Args:
            log_miss: 召回不到时是否自动记入 :attr:`miss_log`。
                默认 True。内部调用（如装配上下文时的间接召回）可传 False 避免噪声。
            annotate_conflicts: 是否对结果集内的**疑似同一事实的不同口径**打标
                （``conflict_group`` / ``conflict_with`` / ``newer``）。
                只增注释，不折叠、不改排序、不影响分数。top_k 很大或高频调用
                时可关掉省开销。
            filter_tags: 结构化标签过滤，如 ``{"章": 43}`` 或
                ``{"年": ["己丑", "庚寅"]}``。**在检索前收窄候选集**，
                而不是检索后再筛——后者会在"匹配项都排在 top_k 之外"时漏召回。
                指定后 BM25 路同样受白名单约束，否则字面命中会绕过过滤。
        """
        if self.manager.store.count() == 0:
            return []

        allow_ids = self._ids_matching_tags(filter_tags)
        if allow_ids is not None and not allow_ids:
            # 过滤条件没有任何命中：直接返回，且按"未命中"留痕（这是真信号）
            if log_miss and query and self.miss_log_enabled:
                self.log_miss(query, 0.0, top_k=top_k,
                              path="brain.recall:tag_empty", client=client)
            return []

        fetch_k = max(top_k * 4, 12)
        raw = self.manager.recall(
            query=query, top_k=fetch_k, use_lnn=True, allow_ids=allow_ids,
            client=client,
            # 时间窗下沉到 manager：候选池在那里取，若不透传，池子先按
            # 无窗口裁到 fetch_k，窗口内的条目可能压根没进候选。
            occurred_after=occurred_after, occurred_before=occurred_before,
        )
        now = self.manager._reference_time or _now()

        # 查询实体 → 命中记忆加实体加成（语义鸿沟时靠实体桥接，T1 结论的最小实现）
        # 只认受信任来源：启发式产出的"这笔账没法记"这类短语若参与加成，
        # 会把纯属巧合的命中抬到相关记忆之上（见 ENTITY_ORIGIN_* 说明）。
        # 实体 id 查询本体在 domain/recall_flow.py::query_entity_ids。
        query_ent_ids = _recall_flow.query_entity_ids(
            self.extract_entities(query), self.entities,
            self._trusted_entity_ids())

        # BM25 混合检索（V9d 评测关键组件，此前只存在于评测 harness）
        bm25_scores: dict[str, float] = {}
        if self.use_bm25:
            # 能用 FTS5 就用——**关键是别再调 _sync_bm25()**，否则内存
            # BM25 索引照样被建起来，省 34.5MB 这件事当场作废。
            fts_hits = self._fts_hits(query, fetch_k * 2)
            if fts_hits is not None:
                bm25_scores = fts_hits
                self._last_lexical_path = "fts5"
            else:
                self._sync_bm25()
                if self._bm25.docs:
                    bm25_scores = self._bm25.query(query, top_k=fetch_k * 2)
                self._last_lexical_path = "memory"
            bm25_ranked = list(bm25_scores.keys())
        else:
            bm25_ranked = []
            self._last_lexical_path = "off"

        # 字面强命中集合：BM25 单路前 N 名（dict 已按分数降序）。
        # 这些候选在 MMR 阶段豁免多样性惩罚，避免"带章号/数额的那条"被近义项折叠。
        # 本集合随打分块（块 C）搬进 domain/recall_flow.py，
        # 由 bm25_scores 的键序同源派生（见模块 docstring 第 1 条）。

        # RRF 融合：向量路 + BM25 路，兼候选并集。
        # rrf_fuse 按第 16 条铁律注入（retrieval 不在 domain 依赖范围内）。
        fusion = _recall_flow.build_fusion(
            raw, bm25_ranked,
            bm25_weight=self.bm25_weight, fuse=rrf_fuse)

        # O8 过时窗口（融合路兜底）：向量路的过时候选已在 manager.recall
        # 被挡掉，这里拦 BM25/FTS 字面路捞回的。rec 为 None 的候选
        # collect_recall_items 本来就会丢，提前丢等价。
        candidate_ids = fusion.candidate_ids
        window_active = occurred_after > 0 or occurred_before > 0
        if self._recency_window_seconds > 0 or window_active:
            kept: list[str] = []
            for cid in candidate_ids:
                rec = self.manager.store.get(cid)
                if rec is None:
                    continue
                if (self._recency_window_seconds > 0 and not window_active
                        and out_of_window(rec.timestamp, rec.pin, now,
                                          self._recency_window_seconds)):
                    continue
                # 用户显式时间窗：BM25/字面路会带回区间外的条目（它只按
                # 文本匹配），这里必须再裁一次，否则时间窗在字面路上失效。
                if window_active and not in_time_window(
                        rec.timestamp, occurred_after, occurred_before):
                    continue
                kept.append(cid)
            candidate_ids = kept

        # 候选过滤与打分构建：全部旋钮调用时现取后注入。
        items = _recall_flow.collect_recall_items(
            candidate_ids,
            _recall_flow.RecallCandidateContext(
                allow_ids=allow_ids,
                include_superseded=include_superseded,
                archived_state=MemoryState.ARCHIVED.value,
                superseded_state=MemoryState.SUPERSEDED.value,
                hard_floor=RELEVANCE_HARD_FLOOR,
                entity_bonus=ENTITY_BONUS,
                literal_topn=MMR_LITERAL_EXEMPT_TOPN,
                w_time=self.manager.scorer.w_time,
                lambda_decay=self.lambda_decay,
                nav_penalty=self.nav_penalty,
                nav_tag_key=self.nav_tag_key,
                bm25_scores=bm25_scores,
                query_ent_ids=query_ent_ids,
                vec_set=fusion.vec_set,
                fused=fusion.fused,
                vec_score_by_id=fusion.vec_score_by_id,
                sim_by_id=fusion.sim_by_id,
                get_record=self.manager.store.get,
                embed=self.manager.vector_of,
                exp=np.exp,
                now=now,
            ))

        if use_mmr and len(items) > top_k:
            items = self._mmr_select(items, top_k)
        else:
            items = sorted(items, key=lambda x: -x["score"])[:top_k]

        # 并存口径标注：必须在**清理内部字段之前**——它要用 _emb 算近邻。
        # 放在排序之后：只为用户真正会看到的那几条算，不为被淘汰的候选白算。
        if annotate_conflicts:
            self._annotate_conflicts(items)

        for it in items:
            # 统一清理内部字段（不再手写单个 pop——`_sel_sim` 就是这样漏出去的）
            strip_internal_fields(it)
            # 时间派生字段（2026-11-05）：宿主拿 `occurred_at` 就能拼出
            # "[2023-01-08 Fri] ..."，不必自己算 epoch→日期、也不必算星期
            # （LLM 算星期是弱项，而「上周五」这类相对定位正卡在这一步）。
            #
            # 为什么派生而不改 `timestamp`：老调用方仍按 float 用（排序、
            # 区间比较、衰减），改类型会全数报错。
            # 为什么放在这里而不是 `recall_flow`：那个模块有 AST 断言守着
            # **只许 import 标准库**（见 test_m5 的 S4），时间格式化属于
            # 呈现、归调用方；manager 那条路径在 `RecallResult.to_dict` 里
            # 同口径补齐。
            it["occurred_at"] = iso_date(it["timestamp"])
            it["weekday"] = weekday_abbr(it["timestamp"])

        # **"最后一次被召回"只记在真正返回给宿主的条目上**。
        # 计数（recall_count）记在候选集上（manager.search），时间戳记在结果集
        # 上：候选会被硬门槛 / MMR / 标签过滤掉一大半，若在候选阶段打戳，
        # 沉寂归档就会被稀释成"每次查询给十来条记录续命"。
        #
        # 时间基准必须与 consolidate 的 now 一致（`_reference_time or wall-clock`）：
        # LoCoMo 评测的数据时间戳可能是 2023 年，若写裸 time.time()，老记忆会被
        # 判成"刚刚被召回"→ **永久豁免沉寂归档**，机制反向失效。
        #
        # 此前 last_recalled_at **全仓零写入**（追加五实测：224 条被召回过的
        # 记忆该字段全为 0），沉寂归档退化成"按写入时间算"，DORMANT_DAYS=180
        # 下归档 0 条等于机制闲置。
        self.manager.stamp_recalled([it["memory_id"] for it in items])

        # 按日召回计数（追加五）：平行期"该想起率"的分子。检索入口不止这一处
        # （MCP 的 recall 工具直连 manager.recall），那边也要记，见 server.py。
        self.bump_daily("recall")

        # 未命中留痕：拿"最高相似度"判定，而不是"结果是否为空"——
        # BM25 单路捞回来的条目 similarity 是 0.0，只看空结果会把这类
        # "其实没找到"记成命中。
        if log_miss and query and self.miss_log_enabled:
            top_sim = max((it["similarity"] for it in items), default=0.0)
            if top_sim < self.miss_log_threshold:
                self.log_miss(query, top_sim, top_k=top_k, path="brain.recall",
                              client=client)
        return items

    def log_miss(
        self,
        query: str,
        top_similarity: float,
        top_k: int = 0,
        path: str = "",
        client: str | None = None,
    ) -> dict[str, Any]:
        """记一条未命中（供归因分析用）。

        平行期三列记录里"没命中的 query"不用再手记：留痕里带查询、当时的
        最高相似度、入口路径与会话 id，可直接喂 80% 门槛那类评测。
        """
        entry = {
            "ts": _now(),
            "query": (query or "")[:300],
            "top_similarity": round(float(top_similarity), 4),
            "threshold": self.miss_log_threshold,
            "top_k": top_k,
            "path": path,
            "session_id": self.current_session_of(client),
        }
        # 追加 + 上限裁剪（删除必立碑纪律）在
        # domain/miss_log_flow.py：
        # max_n 从本命名空间现取，立碑回调裸引用注入。
        return _miss_log_flow.append_miss_entry(
            miss_log=self.miss_log, entry=entry, max_n=MISS_LOG_MAX,
            note_deletions=self.note_layer_deletion_items,
            ledger_name="brain_misslog.json")

    def clear_miss_log(self) -> int:
        """清空未命中留痕（MCP 工具 ``miss_log(clear=True)`` 的落点）。

        清空是**真删除**，先给每一条立碑再清。此前直接
        ``miss_log.clear()`` 只在"我是最后一个写的进程"时有效 —— 别的进程
        手里那份旧快照一落盘，整批就被并回来，用户看到的"已清空"是假的。

        Returns:
            清掉几条。
        """
        # 清空前整批立碑在 domain/miss_log_flow.py
        return _miss_log_flow.clear_miss_log(
            miss_log=self.miss_log,
            note_deletions=self.note_layer_deletion_items,
            ledger_name="brain_misslog.json")


    def bump_daily(self, kind: str, n: int = 1) -> None:
        """按日累计一次活动（追加五）：``kind`` 取 ``"recall"`` / ``"ingest"``。

        实现在 :func:`hippocampus_mcp.domain.profile_ops.bump_daily`
        本方法只做绑定：``daily_counts`` 容器注入
        本体（就地修改）；``base_fn`` 惰性——kind/n 门禁在时间基准计算
        之前，开关关掉时一次都不调。
        """
        _profile_ops.bump_daily(
            kind, n, daily_counts=self.daily_counts,
            base_fn=lambda: self.manager._reference_time or _now(),
            keep_days=DAILY_COUNT_KEEP_DAYS)


    def _annotate_conflicts(self, items: list[dict[str, Any]]) -> int:
        """标注结果集内**疑似同一事实的不同口径**，返回配对数。

        **只做标注**：给配对的条目加 ``conflict_group``（同组同号）/
        ``conflict_with``（对方 id 列表）/ ``newer``（组内时间最新的那条）。
        分数、顺序、条数一律不动——旧口径可能是被引用的历史事实，也可能才是
        对的那条；哪条为准由权威来源（docs）决定，检索器不替用户裁定。

        触发条件是"同一套语义"：判据来自 :meth:`_is_restatement`，与写侧的
        ``supersede`` 检测完全一致。写侧管不到的（机制关闭 / 历史数据 / 当时
        没配上对）在这里补标——这正是"并存新旧口径"这个普遍问题的通用兜底。

        代价：O(n²) 配对 × O(1) 集合交（实词集合预先算好，不重复分词）。
        n 受 ``CONFLICT_ANNOTATE_MAX`` 约束，超出直接跳过（**只影响标注，
        不影响检索结果本身**）。

        实现在 :func:`hippocampus_mcp.domain.conflict_annotate.annotate_conflicts`
        本方法只做绑定：分词 ``_tokens_of``、判据
        ``_is_restatement``、上限 ``CONFLICT_ANNOTATE_MAX`` 全部**调用时现取**
        并注入——判据不 import 是为了保住 monkeypatch 面，也让写侧取代检测
        能用同一份判据。
        """
        return _annotate_conflicts_fn(
            items,
            tokens_of=self._tokens_of,
            is_restatement=self._is_restatement,
            max_items=CONFLICT_ANNOTATE_MAX,
        )

    # ==================================================================
    # 用层：会话生命周期 + 5 槽位上下文
    # ==================================================================
    def recent_memories(
        self, limit: int = 10, source: str | None = None
    ) -> list[dict[str, Any]]:
        """最近写入的记忆（时间倒序），"上次干到哪"的最便宜实现。

        只取 active，内容截断到 200 字——预激活包是要拼进 prompt 的。
        """
        return _recent_memories(
            self.manager.store.iter_records(), limit, source,
            active=MemoryState.ACTIVE.value,
        )

    def export_memories(
        self,
        since: float = 0.0,
        source: str | None = None,
        state: str | None = None,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        """按时间戳 / 来源 / 状态增量导出记忆（不含向量）。

        用于把记忆库里的某一段（比如"上次导出之后新增的"）搬到别处，
        或做按源的审计。``since`` 是 Unix 秒级时间戳，边界语义是 **>=**。

        为什么不做成严格大于：同一时间戳可能有多条记忆（显式指定 timestamp
        的批量写入）。严格大于会把"上批最大 ts 之外的其它同 ts 条目"整批跳过，
        那是静默丢数据；含等于只多回一条，调用方按 id 去重即可。
        """
        rows = []
        for r in self.manager.store.iter_records():
            if r.timestamp < since:
                continue
            if source is not None and (r.source or "") != source:
                continue
            if state is not None and r.state != state:
                continue
            rows.append(r.to_dict())
        rows.sort(key=lambda r: r["timestamp"])
        return rows[:limit] if limit > 0 else rows





    def get_memory_context(self, query: str = "", top_k: int = 5,
                           token_budget: int = 1200,
                           item_max_chars: int = 200,
                           filter_tags: dict[str, Any] | None = None) -> dict[str, Any]:
        """装配 5 槽位上下文（借鉴 bigeye AttentionFocus），可直接拼进 system prompt。

        block_1 当前查询直接命中  block_2 未决问题/待办
        block_3 时间线位置（最近会话摘要）  block_4 相关实体状态
        block_5 上次结论 / 未完结话题

        block_4 有**自己的预算**（``BLOCK4_*``：最多 ``BLOCK4_MAX_ENTITIES`` 个实体、
        每个至多 ``BLOCK4_MAX_FACTS_PER_ENTITY`` 条事实、全块 ≤ ``BLOCK4_MAX_CHARS``
        字符），且候选以**查询词实体**优先、命中实体的前若干条作兜底——**不随
        ``top_k`` 膨胀**（zcode D2 主体）。

        Args:
            item_max_chars: 单条文本在上下文里的截断长度，默认 200。
                **传 0 表示不截断**——账目/条文类长句的数字尾巴常被 200 切掉，
                这类场景应放开（用户反馈）。
                注意它只影响拼进 prompt 的 text，返回的 ``hits`` 始终是完整内容。
            filter_tags: 结构化标签过滤，透传给 block_1 的命中检索。
                给定时 block_4 **不做查询词实体扩展**，只在命中集内发生
                （命中为空则无实体可扩），这是期望行为：按章写作时不希望
                别章的实体状态混进上下文。
        """
        # 本体在 domain/context_slots.py：六个 brain 入口以
        # 可调用对象注入（recall / _pinned_brief / _trusted_entity_ids /
        # extract_entities / entity_facts / _render_blocks），BLOCK4_*
        # 四个上限调用时从本命名空间现取。
        return _memory_context_fn(
            query=query, top_k=top_k, token_budget=token_budget,
            item_max_chars=item_max_chars, filter_tags=filter_tags,
            recall=self.recall,
            pinned_brief=self._pinned_brief,
            trusted_entity_ids=self._trusted_entity_ids,
            entities=self.entities,
            extract_entities=self.extract_entities,
            entity_facts=self.entity_facts,
            intents=self.intents,
            summaries=self.summaries,
            open_topics=self.open_topics,
            render_blocks=self._render_blocks,
            block4_max_entities=BLOCK4_MAX_ENTITIES,
            block4_max_facts_per_entity=BLOCK4_MAX_FACTS_PER_ENTITY,
            block4_max_chars=BLOCK4_MAX_CHARS,
            block4_hit_entity_topn=BLOCK4_HIT_ENTITY_TOPN,
        )



    # ==================================================================
    # L3 画像（保守：只收反复出现/高重要性，冲突只提示不覆盖）
    # ==================================================================
    def _update_profile(self) -> list[dict[str, Any]]:
        """提炼 L3 画像：只收高置信条目，冲突只追加不覆盖。

        实现在 :func:`hippocampus_mcp.domain.profile_ops.update_profile`
        本方法只做绑定：``profile`` 容器注入本体
        （不搬）；prune 前置的结果落 ``last_profile_pruned``；
        ``records_fn`` 惰性（钉住扫描在实体扫描之后才物化）。

        O4：第三来源「访问驱动毕业」由 :func:`_profile_promotion_kwargs`
        决定（开关与阈值调用时现取；关闸只停新增，失效摘除仍无条件）。
        """
        updates, dropped = _profile_ops.update_profile(
            self.profile, prune_fn=self.prune_stale_profile,
            entities=self.entities, entity_trusted=self._entity_trusted,
            store_get=self.manager.store.get,
            records_fn=lambda: self.manager.store.iter_records(),
            now_fn=_now, active_state=MemoryState.ACTIVE.value,
            slot_max=PROFILE_SLOT_MAX, entity_max=PROFILE_ENTITY_MAX,
            stable_slot=PROFILE_SLOT_STABLE, **_profile_promotion_kwargs())
        self.last_profile_pruned = dropped
        return updates

    def prune_stale_profile(self) -> list[dict[str, str]]:
        """摘除**已失效**的 L3 画像条目。

        实现在 :func:`hippocampus_mcp.domain.profile_ops.prune_stale_profile`
        本方法只做绑定：``profile`` 容器注入本体；
        「宁可漏摘不可误摘」的判据与 reason 三态见新模块 docstring。
        O4：毕业槽**无条件**纳入管理（与 ``profile_promotion`` 闸门无关
        ——开关只停新增，不挡失效摘除）。
        """
        return _profile_ops.prune_stale_profile(
            self.profile, store_get=self.manager.store.get,
            stable_slot=PROFILE_SLOT_STABLE,
            active_state=MemoryState.ACTIVE.value,
            recurring_slot=PROFILE_SLOT_RECURRING)

    def pinned_total(self) -> int:
        """当前钉住且活跃的条目总数——与钉住槽实际带回来的条数对照，
        差值就是"被 pin_limit 挡在外面"的数量（追加 19① 的自曝口径）。"""
        return sum(
            1 for r in self.manager.store.iter_records()
            if r.pin and r.state == MemoryState.ACTIVE.value
        )


    def _knowledge_brief(self, limit: int = 8) -> list[str]:
        """开场包的「知识地图」槽：几条一行的知识点（标题 + 规模 + 一句话）。

        🔴 **刻意不在这里自动梳理**。开场包在热路径上（每会话一次），而首次
        梳理是全库聚类——真实大库实测也就 2-3 秒。把这几秒塞进
        每次开场，换来的只是「万一宿主想看地图」。所以这里只读盘上已有的层，
        空就明说「调 knowledge_map 会梳理一次」，把一次性成本留给用户显式调用。

        一句话优先用**宿主归纳**（那才是上层知识），没有才退回抽取式代表句。
        口径与"绝不在这里自动梳理"的理由见 :mod:`..domain.knowledge_brief`。
        """
        return _knowledge_brief(
            self.knowledge, limit,
            topic_layer_enabled=_topic_defaults().layer_enabled,
            # 主题口径注入（架构约束：domain 只准依赖 common）
            ordered_topics=topic_mod.ordered,
            archived_status=topic_mod.STATUS_ARCHIVED,
            summary_state=topic_mod.summary_state,
            # 只在"有料但没梳理过"这条分支上才问库里有没有东西
            store_count=self.manager.store.count,
            clip=KNOWLEDGE_BLOCK_CLIP,
        )

    def _profile_brief(self, limit: int = 5) -> list[str]:
        out = []
        for slot, items in self.profile.items():
            for it in items[-1:]:
                out.append(f"{slot.replace('entity:', '')}: {it['text'][:80]}")
            if len(out) >= limit:
                break
        return out or ["（画像尚在积累）"]

    def _pinned_brief(self, limit: int = 3, task: str = "",
                      task_hit_ids: set[str] | None = None) -> list[dict[str, Any]]:
        """钉住记忆槽。

        排序口径：**任务相关优先**。

        钉住条目的 importance 普遍偏高且彼此接近（实测：书级铁律 0.9 vs
        章节级卡 0.8），纯 importance 排序排出来的其实是"谁先被提权"——
        于是本次会话用得上的铁律被无关的会话卡挤掉。真正有区分度的是
        "跟这次 task 对不对得上"，故分三档：

            ``task_relevance`` = 2  task 语义召回命中了这条
                               = 1  与 task 里的**受信任实体**相交
                               = 0  无关

        同档内再按 importance / timestamp 降序；``task`` 为空时退回纯 importance
        （与旧行为一致）。实体只认受信任来源：启发式实体约 84% 是短语噪声，
        让它们参与"相关性"判定等于随机重排。

        Args:
            limit: 返回条数。**负数 = 关闭该槽**（返回空列表）。
            task: 本次会话任务描述，用于任务相关优先。
            task_hit_ids: task 语义召回命中的 memory_id（``session_start`` 里
                已经算过一次，传进来复用，不重复检索）。
        """
        # 本体在 domain/stats_views.py：相关度走回调
        # （_pin_task_relevance 还要读 _task_entity_ids ⇒ 留在 brain）。
        return _pinned_brief_fn(
            self.manager.store.iter_records(),
            relevance=self._pin_task_relevance,
            limit=limit, task=task, task_hit_ids=task_hit_ids,
            active_state=MemoryState.ACTIVE.value,
        )

    def _pin_task_relevance(self, task: str,
                            task_hit_ids: set[str] | None = None) -> dict[str, int]:
        """钉住条目 vs task 的相关度（0/1/2），判据见 :meth:`_pinned_brief`。"""
        rel: dict[str, int] = {}
        if not (task or "").strip():
            return rel
        for mid in (task_hit_ids or ()):
            rel[mid] = 2
        ent_ids = self._task_entity_ids(task)
        if not ent_ids:
            return rel
        for r in self.manager.store.iter_records():
            if r.id in rel:
                continue
            if ent_ids.intersection(r.entity_ids or ()):
                rel[r.id] = 1
        return rel











    def relink_entities(self, only_ids: list[str] | None = None) -> dict[str, int]:
        """对已有记忆重跑实体抽取并**补挂**（词库扩充的必要配套）。

        实现在 :func:`hippocampus_mcp.domain.drift_ops.relink_entities`
        本方法只做绑定：抽取/挂链/并集/写回全部
        **调用时现取**并注入（``merge_id_lists`` 在 store 层，domain
        不 import，走回调）；``archived_state`` 按值传（未 patch 常量）。
        """
        return _drift_ops.relink_entities(
            self.manager.store.iter_records(), only_ids=only_ids,
            extract_fn=self.extract_entities_with_origin,
            link_fn=self._link_entities,
            merge_fn=merge_id_lists,
            update_fn=self.manager.store.update,
            archived_state=MemoryState.ARCHIVED.value,
        )


    # ==================================================================
    # 别名发现（设计取舍：别名"发现"仍全靠人）
    # ==================================================================
    def _stop_grams(
        self, exclude: list[str] | None = None
    ) -> tuple[set[str], set[str], dict[str, int]]:
        """停用词三层 → ``(原词集合, 展开后的子串集合, 各层条数)``。

        实现在 :func:`hippocampus_mcp.domain.drift_ops.stop_grams`
        本方法只做绑定：词库路径与读取回调、内置
        停用词值**调用时现取**并注入；``lexicon_suggest`` 与
        :meth:`discover_aliases` 共用同一套噪声定义的判据只此一处。
        """
        return _drift_ops.stop_grams(
            self.entity_lexicon_path, self._load_lexicon,
            builtin_stop=_DEFAULT_LEXICON_STOP, exclude=exclude)




    # ==================================================================
    # 意图 / 待办（WorkMemory 简化版）
    # ==================================================================
    def track_intent(self, text: str, due: str = "") -> str:
        entry: dict[str, Any] = {
            "id": _nid("it"), "text": text, "due": due,
            "status": "open", "created_at": _now(),
        }
        # 前瞻记忆情境触发。登记时从待办文本**派生**
        # 触发条件（实体名 + 实词集），随 brain_intents.json 落盘——
        # 之后 get_memory_context 的查询与触发集相交时该待办在 block_2
        # 置前回带（domain/prospective.py）。开关关 = 不派生（老数据
        # 缺字段同样永不触发，与无触发集一致）。
        if load_config().prospective_trigger:
            tri = _prospective.derive_intent_triggers(
                text, extract_entities=self.extract_entities,
                tokens_of=self._tokens_of)
            entry["trigger_entities"] = list(tri.entities)
            entry["trigger_topic"] = sorted(tri.topic)
        self.intents.append(entry)
        return f"已登记待办：{text}"

    def resolve_intent(self, keyword: str) -> str:
        for it in self.intents:
            if it["status"] == "open" and keyword.lower() in it["text"].lower():
                it["status"] = "done"
                return f"已完成待办：{it['text']}"
        return f"未找到匹配的待办：{keyword}"

    # ==================================================================
    # 反馈回流
    # ==================================================================
    def report_feedback(self, memory_id: str, useful: bool = True) -> bool:
        """记忆被实际采用 → 提升重要性；无用 → 标注不降权（失败经验同样有价值）。"""
        rec = self.manager.store.get(memory_id)
        if rec is None:
            return False
        if useful:
            new_imp = min(rec.importance + 0.05, 1.0)
            self.manager.store.update(memory_id, importance=new_imp,
                                      pin=new_imp >= PIN_IMPORTANCE)
        else:
            # 不降权，只降"近期优先度"：清掉 pin 让它自然衰减
            self.manager.store.update(memory_id, pin=False)
        return True

    # ==================================================================
    # 记忆删除（级联清理）与纠错
    # ==================================================================
    def forget(self, memory_id: str, reason: str = "") -> dict[str, Any]:
        """删除一条记忆，并级联清理它在所有派生结构里的痕迹。

        与 ``manager.delete_memory`` 的关键区别：后者只从 L1 物理移除，
        派生层（实体倒排 / 画像 / 被取代者的回指 / L0 来源）会留下残留副本，
        导致"已经删掉的错误信息"仍然通过画像或实体状态进入上下文。
        forget 把这条链路一并清干净，并留下墓碑以便追溯。

        级联范围：
          1. 实体倒排：摘除引用并重算统计，清空则移除整个实体；同时把
             消失的实体从其他记忆的 entity_ids 里剔除（反向悬挂）。
          2. 画像层：移除 evidence 指向该记忆、或文本副本等于该内容的条目。
          3. 被取代者：若某记忆是被这条取代的，恢复为 active（取代理由不成立）。
          4. L0 来源：若没有其他存活记忆引用同一原始消息，标记 forgotten。

        Args:
            memory_id: 要删除的记忆 id。
            reason: 删除原因，写入墓碑供审计。

        Returns:
            各级联层的清理报告。
        """
        rec = self.manager.store.get(memory_id)
        if rec is None:
            return {"ok": False, "error": f"记忆不存在: {memory_id}"}

        # 墓碑先记（保留原文，保证删除后仍可追溯"删了什么、为什么"）。
        # 构建本体 → forget_entry.build_tombstone（批C2；append 留壳，
        # 时钟注入、快照拷贝在本体复刻）。
        self.tombstones.append(_forget_entry.build_tombstone(
            memory_id=memory_id, content=rec.content, reason=reason,
            timestamp=rec.timestamp, state=rec.state,
            entity_ids=rec.entity_ids, source_ids=rec.source_ids, now=_now))
        # 受影响实体键（扫 entities 倒排）→ forget_entry.compute_affected_keys。
        affected_keys = _forget_entry.compute_affected_keys(
            self.entities, memory_id)

        # 1) 物理删除（后续重算以"存活记忆"为准，必须在其之后）
        self.manager.delete_memory(memory_id)
        # 1b) 主题层：摘除成员引用。它是 L1 的派生副本，**只删 L1 等于假
        #     删除**——「知识点」会继续引用一条已经不存在的记忆。
        knowledge_dropped = self._forget_from_knowledge(memory_id)

        # 2) 实体倒排：重算 + 移除空实体 + 反向清理其他记忆的 entity_ids
        self._recompute_entity_stats(affected_keys)
        empty_keys = [
            k for k in affected_keys
            if k in self.entities and not self.entities[k]["memory_ids"]
        ]
        dead_entity_ids = {self.entities[k]["id"] for k in empty_keys}
        for k in empty_keys:
            del self.entities[k]
        relinked = 0
        if dead_entity_ids:
            for r in self.manager.store.iter_records():
                if not r.entity_ids:
                    continue
                kept = [e for e in r.entity_ids if e not in dead_entity_ids]
                if len(kept) != len(r.entity_ids):
                    self.manager.store.update(r.id, entity_ids=kept)
                    relinked += 1

        # 3) 画像层：摘掉指向该记忆的派生条目
        self.profile, dropped_profile = _forget_cleanup.clean_profile(
            self.profile, memory_id=memory_id, content=rec.content,
        )

        # 4) 被这条取代的记忆：恢复 active 并解除回指（取代理由不再成立）
        restored = []
        for r in self.manager.store.iter_records():
            if r.superseded_by == memory_id:
                self.manager.store.update(
                    r.id, superseded_by=None, state=MemoryState.ACTIVE.value
                )
                restored.append(r.id)

        # 5) L0 来源：不再被任何存活记忆引用时标记（原文保留，审计仍可查）
        alive_refs = {
            sid for r in self.manager.store.iter_records()
            for sid in (r.source_ids or [])
        }
        orphan_raws = _forget_cleanup.mark_orphan_raws(
            self.raw_log, source_ids=rec.source_ids,
            alive_refs=alive_refs, reason=reason,
        )

        self.save()
        return {
            "ok": True,
            "memory_id": memory_id,
            "content": rec.content,
            "reason": reason or "",
            "entities_touched": affected_keys,
            "entities_removed": empty_keys,
            "memories_relinked": relinked,
            "profile_dropped": dropped_profile,
            "restored_superseded": restored,
            "knowledge_members_dropped": knowledge_dropped,
            "raw_marked": orphan_raws,
            "remaining_l1": self.manager.store.count(),
        }

    def forget_by_query(self, query: str, top_k: int = 3,
                        dry_run: bool = True,
                        min_similarity: float | None = None,
                        allow_multiple: bool = False) -> dict[str, Any]:
        """按自然语言定位候选记忆并删除（或仅预览）。

        宿主 agent 拿到的通常是"那条关于三月上线的记错了"这类自然语言，
        而不是 memory_id。此方法先召回再删，**默认只预览**（dry_run=True），
        确认候选无误后再以 dry_run=False 执行，避免误删。

        **执行删除（dry_run=False）有两道护栏**：

        1. **必须有向量背书**：只考虑 ``similarity >= min_similarity`` 的候选。
           recall 会放行 BM25 词面命中但语义无关的条目，它们的 similarity 是
           **0.0**，删它们等于"按关键词删库"。
        2. **默认只删最匹配的一条**：``allow_multiple=False`` 时，即便有若干条
           过了门槛，也只删排序第一的那条。"删掉那条记错的"在语义上几乎总是单数，
           多出来的槽位正是误删的来源。确实要批量删请显式传 ``allow_multiple=True``。

        两道都不是靠"调高相似度门槛"实现的——相关性随查询长度整体漂移（见
        ``FORGET_MIN_SIMILARITY`` 注释里的实测分布），定值门槛救不了这两个方向
        中的任意一个；真正稳定的是"靶子恒为 #1"这个相对优势。

        被跳过的候选一律原样列进 ``skipped`` 并带 ``skip_reason``，宿主可据此
        改用 memory_id 精确删除。

        Args:
            min_similarity: 门槛；None 取 FORGET_MIN_SIMILARITY。
            allow_multiple: 允许多条一起删，默认 False。
        """
        # 本体在 domain/forget_flow.py：recall / forget 以
        # 可调用对象注入（不预调用，dry_run 时一次都不删）；默认门槛
        # FORGET_MIN_SIMILARITY 调用时现取——patch 本命名空间即生效。
        return _forget_by_query_fn(
            query=query,
            top_k=top_k,
            dry_run=dry_run,
            min_similarity=min_similarity,
            min_similarity_default=FORGET_MIN_SIMILARITY,
            allow_multiple=allow_multiple,
            recall=self.recall,
            forget=self.forget,
        )

    def list_superseded(
        self, limit: int = 200, source: str | None = None
    ) -> list[dict[str, Any]]:
        """列出被标记为 superseded（已被取代）的记忆，含"被谁取代"。

        实现在 :func:`hippocampus_mcp.domain.supersede_views.list_superseded`
        本方法只做绑定：store 读与取代态常量**调用时
        现取**注入；``limit<=0`` 的切片怪癖原样保留（见新模块 docstring）。
        """
        return _supersede_views.list_superseded(
            self.manager.store.iter_records(),
            get=self.manager.store.get,
            superseded_state=MemoryState.SUPERSEDED.value,
            limit=limit, source=source)

    def un_supersede(
        self,
        memory_ids: list[str] | None = None,
        all: bool = False,  # noqa: A002 - 与 CLI 语义对齐，保留此名
        source: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """把误判为"已被取代"的记忆恢复为 active（解除 superseded_by）。

        实现在 :func:`hippocampus_mcp.domain.supersede_views.un_supersede`
        本方法只做绑定：store 写回与状态常量**调用时
        现取**注入，``save`` 惰性回调（有目标才存，预览零写零存）。
        """
        return _supersede_views.un_supersede(
            self.manager.store.iter_records(),
            update=self.manager.store.update, save=self.save,
            superseded_state=MemoryState.SUPERSEDED.value,
            active_state=MemoryState.ACTIVE.value,
            memory_ids=memory_ids, all=all, source=source, dry_run=dry_run)

    # ==================================================================
    # 巩固任务（睡眠修剪）：沉寂归档 / 刷新画像 / 矛盾清单
    # ==================================================================
    def _contradiction_buckets(self, records) -> dict[str, list[str]]:
        """按受信任实体把活跃记忆分桶（矛盾检测的输入）。

        只认受信任来源的实体：不可信实体（尤其中文高频短语）会把大量互不相关
        的记忆塞进同一个桶，既产生成片的假"矛盾"，又因桶内两两比对是 O(k²)
        而拖慢巩固。抽成独立方法是为了让这条过滤可被单测直接验证。
        """
        return contradiction_buckets(records, self._trusted_entity_ids())

    # ==================================================================
    # 标签批量维护：存量回填 / 批量改键 / 批量删键
    # ==================================================================
    def _select_for_tag_update(
        self,
        source: str = "",
        source_contains: str = "",
        contains: str = "",
        state: str = "",
        untagged_only: bool = False,
        missing_keys: list[str] | None = None,
        memory_ids: list[str] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """按过滤条件挑出待改标签的记录。

        实现在 :func:`hippocampus_mcp.domain.tag_select.select_for_tag_update`
        本方法只做绑定：``store.iter_records()`` 调用时
        现取后传入——筛选语义与过滤条件自曝见新模块 docstring。
        """
        return _tag_select.select_for_tag_update(
            self.manager.store.iter_records(),
            source=source, source_contains=source_contains,
            contains=contains, state=state, untagged_only=untagged_only,
            missing_keys=missing_keys, memory_ids=memory_ids,
        )

    def tag_coverage(self) -> dict[str, Any]:
        """库级标签覆盖率与键分布（验收判据）。

        实现在 :func:`hippocampus_mcp.domain.supersede_views.tag_coverage`
        本方法只做绑定：``store.iter_records()`` 现取
        传入；覆盖率口径与 top 键截断见新模块 docstring。
        """
        return _supersede_views.tag_coverage(
            self.manager.store.iter_records())

    # ------------------------------------------------------------------
    # S2：update_tags 变更账本（可证明地撤销）
    # ------------------------------------------------------------------
    def _tag_ops_read(self) -> list[dict[str, Any]]:
        """读全部账本条目（坏行跳过——账本自己不该让 undo 用不了）。"""
        return read_ops(self.persist_dir)

    def _tag_ops_scan(self) -> tuple[int, int]:
        """启动时扫一遍账本：返回 (行数, 末条 seq)。"""
        return scan_ops(self.persist_dir)

    def _tag_ops_append(self, entry: dict[str, Any]) -> str | None:
        """追加一条账本（ring buffer 裁剪）。返回 op id；没落盘能力返回 None。

        状态三元组（last_seq / lines）落 brain 实例属性；ring 旋钮在**调用
        时**从 core 命名空间现取——``core.TAG_OPS_RING`` 被测试 monkeypatch，
        快照进 store 模块会让 S2 的环大小断言静默失效。
        """
        op_id, seq, n = append_ops(
            self.persist_dir, entry,
            self._tag_ops_last_seq, self._tag_ops_lines,
            TAG_OPS_RING, TAG_OPS_TRIM_AT,
        )
        self._tag_ops_last_seq = seq
        self._tag_ops_lines = n
        return op_id

    def undo_tag_ops(self, target: str = "last", dry_run: bool = True,
                     confirm: bool = False) -> dict[str, Any]:
        """撤销一次 update_tags 执行。

        **fail-closed 回滚**：只把"当前值仍等于记录的 after"的条目改回
        before。期间被第三方（另一次 update_tags / update_memory）改过的
        条目一律跳过并进 ``conflicts`` 自曝——盖掉后来的修改比不回滚更糟。

        撤销本身也记帐（op="undo"），所以撤错了可以再撤一次（对称恢复）。

        Args:
            target: op id（``tagops_3``）或 ``"last"``（最近一次）。
            dry_run: 默认 True —— 只出回滚计划。
            confirm: 回滚条数超过 :data:`UPDATE_TAGS_CONFIRM_ABOVE` 时须为真
                （回滚和写入一样能造成大面积破坏，闸门同级）。
        """
        return _undo_tag_ops(
            self._tag_ops_read(), target, dry_run, confirm,
            store=self.manager.store,
            append_op=self._tag_ops_append,
            brief=self._tag_ops_brief,
            coverage=self.tag_coverage,
            save=self.save,
            sample_max=UPDATE_TAGS_SAMPLE_MAX,
            confirm_above=UPDATE_TAGS_CONFIRM_ABOVE,
        )


    def update_tags(
        self,
        tags: dict[str, Any] | None = None,
        clear_keys: list[str] | None = None,
        source: str = "",
        source_contains: str = "",
        contains: str = "",
        state: str = "",
        untagged_only: bool = False,
        missing_keys: list[str] | None = None,
        memory_ids: list[str] | None = None,
        mode: str = "merge",
        dry_run: bool = True,
        limit: int = 0,
        confirm: bool = False,
        undo: str = "",
    ) -> dict[str, Any]:
        """批量维护已有记忆的标签（带撤销）。

        入库时没打标签 / 标签键后来改了，都需要回填。**这个工具是那唯一的通路**
        —— ``update_memory`` 不接受 ``tags``，写入后标签就改不动了。

        三道闸门（都是踩过同类坑才加的）：

        * **默认 ``dry_run=True``**：先出计划。批量改标签没有事务，跑错了不会
          报错，只会在过滤时表现为「查不到东西」；
        * **超过 :data:`UPDATE_TAGS_CONFIRM_ABOVE` 条必须 ``confirm=True``**：
          真库一次动 2,010 条，过滤条件一个字符写错就是 2,010 条垃圾标签；
        * **值未变不计入 ``to_change``**：这样"重跑一遍"能得到 ``to_change=0``，
          幂等性是批量工具能不能重跑的前提（原需求写的幂等靠 merge 语义本身
          是达不到的——merge 下第二次跑每条的键都已存在，但**值可能不同**）。

        执行记**全量前后影像账本**（``tag_ops.jsonl``），
        ``undo=<op_id|"last">`` 可 fail-closed 回滚。账本是 ring buffer
        （默认留最近 50 次），且撤销本身也记账——撤错了能再撤一次。

        Args:
            tags: 要写入的标签（dict）。``mode="merge"`` 时逐键覆盖，
                ``mode="replace"`` 时整体替换。
            clear_keys: 要删掉的键（在写入之后执行）。
            source / source_contains / contains / state / untagged_only /
                missing_keys / memory_ids: 过滤条件，全部可省略（省略 = 全库）。
            mode: ``merge``（默认）／``replace``。
            dry_run: 默认 True —— 只出计划，不写。
            limit: 最多处理多少条（0 = 不限）。用于先小批试。
            confirm: 影响条数超过阈值时的显式确认。
            undo: 撤销模式。传 op id（``tagops_3``）或 ``"last"`` 时，
                本方法转为 :meth:`undo_tag_ops`，忽略 tags / clear_keys。

        Returns:
            计划或执行结果。``sample`` 只给前若干条的**改前/改后**。
            S2 起服务端记**全量前后影像账本**（``tag_ops.jsonl``）：执行后
            返回体带 ``op_id``，``undo=tagops_N``（或 ``"last"``）即可
            fail-closed 回滚——当前值仍等于记录的 ``after`` 才改回 ``before``，
            期间被改过的条目跳过并自曝，绝不盖掉后来的修改。
        """
        # S2：撤销优先于一切校验——它不依赖 tags/clear_keys。
        if str(undo or "").strip():
            return self.undo_tag_ops(target=str(undo).strip(),
                                     dry_run=dry_run, confirm=confirm)
        mode, new_tags, ckeys, err = _tag_update.normalize_request(
            mode, tags, clear_keys)
        if err is not None:
            return err

        recs, filt = self._select_for_tag_update(
            source=source, source_contains=source_contains,
            contains=contains, state=state, untagged_only=untagged_only,
            missing_keys=missing_keys, memory_ids=memory_ids,
        )
        matched = len(recs)
        capped = False
        if limit and limit > 0 and len(recs) > limit:
            recs = recs[:limit]
            capped = True

        plan = _tag_update.build_plan(
            recs, mode=mode, new_tags=new_tags, ckeys=ckeys)
        to_change = len(plan)
        payload, proceed = _tag_update.build_payload(
            dry_run=dry_run, confirm=confirm, mode=mode,
            new_tags=new_tags, ckeys=ckeys, filt=filt,
            matched=matched, considered=len(recs), capped=capped,
            to_change=to_change, plan=plan,
            sample_max=UPDATE_TAGS_SAMPLE_MAX,
            confirm_above=UPDATE_TAGS_CONFIRM_ABOVE,
        )
        if not proceed:
            return payload

        result = _tag_update.apply_plan(
            plan,
            _tag_update.ApplyPorts(
                store=self.manager.store,
                map_name_key=MAP_NAME_KEY,
                now=_now,
            ),
        )
        payload["applied"] = True
        payload["changed"] = result.changed
        if result.failed:
            payload["failed"] = result.failed[:UPDATE_TAGS_SAMPLE_MAX]
            payload["failed_n"] = len(result.failed)

        # S2：记全量前后影像账本。**先记账后落盘**——内存里的标签已经改了，
        # "改之前是什么"必须留在盘上，否则 save 失败那一刻起撤销依据就没了
        # （与 save_ledger 的 begin 先于写同一序）。只记**真改成功**的条目。
        payload["op_id"] = self._tag_ops_append({
            "op": "update_tags",
            "ts": _now(),
            "filter": filt,
            "mode": mode,
            "tags": new_tags,
            "clear_keys": ckeys,
            "to_change": to_change,
            "applied": result.changed,
            "changes": result.applied_changes,
        })

        # 落盘：记忆写入侧只在 store_memory 每 10 步自动存一次，批量改标签
        # 不走那条路。不显式存的话 daemon 空闲退出就全丢了，且**无任何报错**
        # （内存里一直看得到）——主题层写回归纳踩过同一个坑。
        try:
            self.save()
            payload["saved"] = True
        except Exception as exc:  # noqa: BLE001
            payload["saved"] = False
            payload["save_error"] = f"{type(exc).__name__}: {exc}"
            payload["hint"] = "标签已在内存生效但未落盘，请调 save_state 重试"
        payload["coverage_after"] = self.tag_coverage()
        return payload

    def consolidate(
        self,
        filter_tags: dict[str, Any] | None = None,
        max_items: int = 0,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """周期巩固：归档沉寂记忆、刷新 L3 画像、列出矛盾候选。

        两点说明：
          - **冗余合并**由写入侧负责（``store_memory`` 的 cosine 去重），
            此处不再重复合并；旧版的 ``merged`` 字段是未实现的死值，已移除。
          - **画像刷新**此前只在 ``session_end`` 触发，导致"删掉记忆后
            重跑巩固不能自愈"。现改为每次巩固都刷新（``_update_profile``
            本身幂等，重复调用不会产生重复条目）。

        三个可选参数，**缺省行为与改造前逐位一致**
        （``session_end`` / ``_maybe_auto_consolidate`` 内部都是无参调用）：

        Args:
            filter_tags: 把**沉寂归档与矛盾列举**的候选收窄到某个标签域，
                与 ``recall`` / ``contradictions`` 同一套语义（复用
                :meth:`_ids_matching_tags`，全键 AND、单键多值 OR、str 归一）。
                缺省 = 全库，返回体以 ``scope: null`` 自曝"这次没设域"。
                ⚠️ **不做成必传**：本方法被 ``session_end`` 内部自动调用，
                必参会让你所有现有调用方报错；也不做"缺省报警告"——那会让
                每次会话收尾都刷一条无意义告警。
            max_items: 本次沉寂归档最多处理多少条（0 = 不限）。大库分批次
                归档用，避免一次巩固改动过多样本。
            dry_run: 只干跑——列出**将会**归档的候选与矛盾候选，不做任何
                状态变更（不归档、不刷新画像 / 主题层、不落盘）。
        """
        now = self.manager._reference_time or _now()
        # 标签域白名单。None = 无过滤（**不要**当成空集——那是"查不到"）
        allow = self._ids_matching_tags(filter_tags)

        records = list(self.manager.store.iter_records())
        # 沉寂归档：长期未被调用且低重要性（本体在
        # domain/consolidate_flow.py；两常量 + 两状态字面量 + 归档回调
        # 调用时现取后注入，archive 仅在实归档时调用）。
        dormant, dormant_candidates = _consolidate_flow.archive_dormant(
            records,
            now=now,
            dormant_days=DORMANT_DAYS,
            pin_importance=PIN_IMPORTANCE,
            active_state=MemoryState.ACTIVE.value,
            archived_state=MemoryState.ARCHIVED.value,
            allow=allow,
            dry_run=dry_run,
            max_items=max_items,
            archive=lambda rid, st: self.manager.store.update(rid, state=st),
        )

        # 矛盾候选：同实体、高相似但内容不同且都活跃（提示宿主裁决，不自动改）。
        # 实现抽到 :meth:`_find_contradictions`，与按需工具 :meth:`contradictions`
        # 共用同一段——两套实现迟早会算出不一样的答案。
        #
        # **性能（实测确认，本段是整个 session_end 超时的唯一根因）**：
        # 旧实现是桶内**逐对 Python 循环**，每对做 2 次 list→numpy 转换
        # （manager.vector_of 返回 Python list）+ 2 次 norm + 1 次点积。
        # 中文小说库 293 个桶 / 总对数 2,815,359 → **206.11s**，宿主 120s 超时
        # 必然踩中（实测单次 294.9s），且重试同样超时（无缓存、无短路）。
        # 现改为 store.pairwise_similarities（分块矩阵乘）→ **0.34s，610×**，
        # 数值与逐对版完全一致（抽样 16 对，差 <1e-9）。
        #
        # 另：旧实现把**所有**命中对（实测 12,812 条）都建成 dict 追加进 list，
        # 最后只取 [:5] —— 白建 12,807 个 dict。现在按桶取 top-N 再全局取 top-5，
        # 且补上了 docstring 一直承诺、代码里却**没有实现**的「内容不同」判断。
        #
        # filter_tags 透传，把矛盾列举也收窄到同一标签域。
        # 矛盾投影本体在 domain/consolidate_flow.py::project_contradictions。
        contradictions = _consolidate_flow.project_contradictions(
            self._find_contradictions(CONTRADICTION_REPORT_MAX,
                                      filter_tags=filter_tags)
        )

        # 状态计数本体在 domain/consolidate_flow.py::count_state。
        # 🔴 必须**重新迭代 store**（不是上面的 records 快照）——沉寂归档
        # 刚把若干条改成 archived，原实现的这两个计数反映的是归档**后**的
        # 状态；传快照会让 active_memories 多数几条（行为 drift）。
        superseded_total = _consolidate_flow.count_state(
            self.manager.store.iter_records(), MemoryState.SUPERSEDED.value)
        report = _consolidate_flow.assemble_report(
            filter_tags=filter_tags,
            max_items=max_items,
            dry_run=dry_run,
            superseded_total=superseded_total,
            dormant=dormant,
            contradictions=contradictions,
            active_memories=_consolidate_flow.count_state(
                self.manager.store.iter_records(),
                MemoryState.ACTIVE.value),
            dormant_candidates=dormant_candidates if dry_run else None,
        )
        if dry_run:
            # 干跑到此为止：只列候选，不碰画像 / 主题层 / 记忆状态
            return report
        # 画像刷新（幂等）：让删改后的状态能通过重跑巩固自愈
        profile_updates = self._update_profile()
        # 主题层：只在巩固时刷新（**写路径上不跑**——perf_write 基线 6.7s
        # 是硬约束，而「每写一条就重排主题」的收益是零：地图是给人看的，
        # 不需要实时）。
        knowledge = self.refresh_knowledge()
        report["profile_updates"] = len(profile_updates)
        # 追加 19②：本轮巩固摘掉了几条失效画像（取消钉住 / 归档 / 已删）
        report["profile_pruned"] = len(self.last_profile_pruned)
        report["knowledge"] = knowledge
        return report

    # ------------------------------------------------------------------
    # daemon 空闲「睡眠窗口」——离线重放式巩固
    # ------------------------------------------------------------------
    def sleep_batch(
        self,
        *,
        cancel: "threading.Event | None" = None,
        budget_s: float = 120.0,
    ) -> dict[str, Any]:
        """O2 睡眠批：daemon 空闲期跑的离线巩固（SWR 重放式，Buzsáki 2015）。

        与 :meth:`consolidate` 的分工：consolidate 是「周期维护」（归档沉寂、
        刷新派生层、列矛盾）；睡眠批是「离线学习」——先重放最近会话的
        L1 簇做共激活强化（生物对应：海马体的 sharp-wave ripple 重放，
        是系统巩固的引擎），再复用 consolidate 完成巩固三件套。

        四条硬要求（TODO O2 的验收面）与各自的落法：

        - **幂等**：重放候选选择是纯函数（``domain.consolidate_flow
          .select_replay_candidates``，零状态变更）；巩固三件套本身幂等
          （见 :meth:`consolidate` docstring）。水位线 ``_last_sleep_at``
          只在**干净完成**时推进——中断后下一轮会重选同一窗口，重放侧
          （接线后）必须容忍候选被重复选中，这条写在接缝方法的契约里。
        - **可中断**：``cancel`` 置位后，在每个**单元边界**（每个簇之间、
          每个阶段之间）检查并尽快退出。consolidate 是单单元：一旦进入
          就跑完（内部各步都有限流，秒级～分钟级）；调用前先查 cancel。
          返回体以 ``interrupted`` 自曝。
        - **不阻塞在线写**：全程不持跨单元锁（重入守卫的 Lock 只在
          ``acquire(blocking=False)`` 一瞬持有）；不主动 save——落盘交给
          daemon 的 ``stop()→save_all()`` 与在线写路径的既有 save 时机，
          睡眠批去抢保存文件锁正是「与在线写打架」的反面教材。
        - **不用大模型**：重放用存储向量都不用——实体共现是纯数据；
          巩固三件套也全是存储内计算。睡眠发生在模型已卸载之后也能跑
          （也正因此，睡眠批绝不 touch 模型生命周期时钟，不推迟空闲卸载）。
          需要 LLM 的部分（如簇标题归纳）留给面向宿主的既有巩固流。

        Args:
            cancel: 中断信号（daemon 在新连接到来时置位）。None = 不可中断
                （脚本直接调用）。
            budget_s: 时间预算兜底（秒），防大库上一轮阶段耗时失控。0/负值
                = 不限。超预算按中断处理（``interrupted="budget"``）。

        Returns:
            报告体：``replay_clusters``（重放候选簇数）/ ``replay_records``
            （参与记录数）/ ``strengthened``（本批实际强化条数，由
            :meth:`_sleep_replay_cluster` 经 ``manager.reinforce`` 累加）/
            ``consolidated``
            （巩固三件套是否跑了）/ ``dormant_archived`` 等透传 consolidate
            报告 / ``interrupted``（False | "cancel" | "budget"）/
            ``covered_pending``（是否覆盖了 session_end 欠下的待巩固）。
        """
        # 重入守卫：非阻塞抢锁，抢不到说明另一线程正在睡（理论上是 daemon
        # 单飞守卫的兜底，脚本双跑时也用得上）。
        if not self._sleep_lock.acquire(blocking=False):
            return {"skipped": "already_running", "interrupted": False}
        try:
            return self._sleep_batch_locked(cancel=cancel, budget_s=budget_s)
        finally:
            self._sleep_lock.release()

    def _sleep_batch_locked(
        self,
        *,
        cancel: "threading.Event | None",
        budget_s: float,
    ) -> dict[str, Any]:
        now = self.manager._reference_time or _now()
        deadline = (time.monotonic() + budget_s) if budget_s and budget_s > 0 else None

        def _interrupted() -> str:
            if cancel is not None and cancel.is_set():
                return "cancel"
            if deadline is not None and time.monotonic() >= deadline:
                return "budget"
            return ""

        report: dict[str, Any] = {
            "replay_clusters": 0, "replay_records": 0, "strengthened": 0,
            "consolidated": False, "interrupted": False,
        }
        # ---- 阶段 R：重放候选选择 + 共激活强化 ----------------------
        if _interrupted():
            report["interrupted"] = _interrupted()
            return report
        cfg = load_config()
        since = self._last_sleep_at or (now - cfg.sleep_replay_days * 86400.0)
        clusters = _consolidate_flow.select_replay_candidates(
            self.manager.store.iter_records(),
            now=now,
            since=since,
            active_state=MemoryState.ACTIVE.value,
            max_items=cfg.sleep_max_replay,
            min_cluster=cfg.sleep_min_cluster,
        )
        report["replay_clusters"] = len(clusters)
        report["replay_records"] = sum(len(c["ids"]) for c in clusters)
        for cluster in clusters:
            flag = _interrupted()
            if flag:
                # 中断在簇边界：已强化的簇保持（幂等语义见接缝方法），
                # 未处理的留给下一轮；水位线不动 ⇒ 下一轮重选同一窗口。
                report["interrupted"] = flag
                return report
            report["strengthened"] += self._sleep_replay_cluster(cluster)

        # ---- 阶段 C：巩固三件套（沉寂归档 + 画像 + 主题层） --------------
        flag = _interrupted()
        if flag:
            report["interrupted"] = flag
            return report
        report["consolidated"] = True
        con = self.consolidate()
        had_pending = self._pending_consolidate
        self._pending_consolidate = False
        report["covered_pending"] = bool(had_pending)
        report["dormant_archived"] = con.get("dormant_archived", 0)
        report["profile_updates"] = con.get("profile_updates", 0)
        report["profile_pruned"] = con.get("profile_pruned", 0)

        # ---- 收尾：只有干净完成才推进水位线 ------------------------------
        report["interrupted"] = _interrupted() or False
        if not report["interrupted"]:
            self._last_sleep_at = now
        return report

    def _sleep_replay_cluster(self, cluster: dict[str, Any]) -> int:
        """O1 接缝（已接线）：对一个重放簇做共激活强化，返回实际强化条数。

        实现＝``manager.reinforce(cluster["ids"], amount=1.0)``——重放是
        "进结果集"的**离线形态**：只累加检索强度，不碰
        ``recall_count``（候选集语义，约束 4/5 沉寂归档依赖它）与
        ``last_recalled_at``（结果集时间戳，重放不是真实召回，动了会污染
        沉寂归档的时间基准）。

        幂等两条契约（写在睡眠批 docstring 的幂等条款里）：
        1. 必须容忍同一窗口被重复选中（中断后水位线不推进 ⇒ 下一轮重选，
           强度再次 +1 是符合生物语义的——反复共激活就是反复巩固）；
        2. 见上：只动强度变量。
        """
        ids = [mid for mid in (cluster.get("ids") or []) if mid]
        if not ids:
            return 0
        return self.manager.reinforce(ids, amount=1.0)

    # ------------------------------------------------------------------
    # 矛盾候选（consolidate 与按需工具共用）
    # ------------------------------------------------------------------
    def _find_contradictions(
        self,
        limit: int = CONTRADICTION_REPORT_MAX,
        source: str = "",
        records: list[Any] | None = None,
        filter_tags: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """筛出"同实体 + 高相似 + 内容不同"的活跃记忆对。

        **只读**：不修改任何记录状态。它只是"提示宿主这里可能有两套口径"，
        裁决权在宿主（只增不删的库里，新旧两版本来就该并存，见 D8）。

        Args:
            limit: 最多返回多少对。
            source: 只查某个来源（文件名 / 会话标识）下的记忆。
            records: 候选集，默认全库存活记录。
            filter_tags: 结构化标签过滤，与 ``recall`` 的
                ``filter_tags`` **同一套语义**（全键 AND、单键多值 OR、
                ``str()`` 归一），复用 :meth:`_ids_matching_tags`。
                作用＝把"整库口径矛盾"收窄成"**某个域内的**口径矛盾"——
                六代口径混在一个库里时，全局 top-N 会被无关域占满。

        Returns:
            按相似度降序的候选对，每对带 ``a_id``/``b_id``（可直接交给
            :meth:`update_memory` 作废）与 ``newer``（谁的时间戳更新）。
        """
        return find_contradictions(
            self.manager.store, limit, source=source, records=records,
            filter_tags=filter_tags, trusted_ids=self._trusted_entity_ids(),
        )

    def _contradiction_pool_size(
        self, source: str = "", filter_tags: dict[str, Any] | None = None,
    ) -> int:
        """矛盾候选的**分母**：过滤后还剩多少条进得了比对。

        存在的理由：``count: 0`` 有两种截然不同的真相——
          ① 候选池非空、但筛不出矛盾对（＝域内口径一致，好消息）
          ② **候选池本身就是空的**（＝过滤条件没命中，和 B2 同型的静默失败）
        不把分母报出来，调用方一律读成 ①，于是"域内没问题"这个结论是假的。
        """
        return pool_size(self.manager.store, source=source, filter_tags=filter_tags)

    def contradictions(
        self, limit: int = CONTRADICTION_REPORT_MAX, source: str = "",
        filter_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """按需列出库内疑似"同一事实两种口径"的候选。

        与 :meth:`ingest` 的写入侧回显（``neighbors``）互补：那个是"**刚写
        的这条**跟谁像"，这个是"**整库**现在有哪些并存的口径"。只读，不改状态。

        Args:
            limit: 最多返回多少对（默认 :data:`CONTRADICTION_REPORT_MAX`）。
            source: 只查某个来源（文件名 / 会话标识）。留空 = 全库。
            filter_tags: 按结构化标签把候选收窄到某个域。留空 = 全库。

        Returns:
            ``candidates`` / ``count`` / ``sim_range`` / ``note``。
            每对带 id 与时间戳，可直接用 :meth:`update_memory` 把旧的标
            ``superseded``——**不是**删掉，删了就没有"曾经改过口径"的痕迹了。
        """
        cands = self._find_contradictions(limit=limit, source=source,
                                          filter_tags=filter_tags)
        return {
            "candidates": cands,
            "count": len(cands),
            "limit": limit,
            "source": source,
            # J3：把"这次到底收窄了没有"自曝出来。标签无命中时白名单是空集，
            # 结果自然是 0 对——不报这个数，调用方会以为"域内没有矛盾"，
            # 而真相是"过滤条件没命中任何条目"（与 B2 同型的静默失败）。
            "filter_tags": dict(filter_tags or {}),
            "candidate_pool": self._contradiction_pool_size(
                source=source, filter_tags=filter_tags),
            "sim_range": [SUPERSEDE_SIM, CONTRADICTION_SIM_CEIL],
            "note": (
                "只标注不裁决：同一事实的新旧口径都要留在库里，由宿主决定取哪条。"
                "确认旧的那条作废时，用 update_memory(id, state='superseded', "
                "superseded_by=<新id>)，不要删。"
            ),
        }

    # ==================================================================
    # 统计 / 持久化
    # ==================================================================
    def _recall_feedback_stats(self) -> dict[str, Any]:
        """C12 自曝：召回反馈（``last_recalled_at``）实际覆盖了多少条。

        ``recall_count`` 记的是**候选集**（谁进过检索池），
        ``last_recalled_at`` 记的是**结果集**（谁真的返回给了调用方）。
        两者覆盖率的差 = "检索到了但没被采用"的量。

        为什么要自曝：``last_recalled_at`` 正是 consolidate / 衰减的**时间基准**，
        而历史数据里它曾长期零写入（实测覆盖率约 9.5%）—— 覆盖率低时，
        "这条多久没被用过"这类归档结论会失真，且**不会报错**。

        口径：**不回填**（回填等于编造召回历史），
        改为每次统计自曝，把"可信度打几折"的判断交回给宿主。
        """
        # 本体在 domain/stats_views.py：纯只读投影。
        return _recall_feedback_stats_fn(
            self.manager.store.iter_records(),
            active_state=MemoryState.ACTIVE.value,
        )

    def _pin_stats(self) -> dict[str, Any]:
        """I4 自曝：钉住（pin）到底钉了多少、其中多少是**词面规则自动钉**的。

        为什么要自曝：``PIN_IMPORTANCE = 0.7``，而 :meth:`_guess_importance`
        命中「现在 / 不再 / changed / ！」即给 0.7 ⇒ **自动 pin**；而 pin
        在系统内**只进不退**（唯一"退出"是调用方显式动作）⇒ 长库里 pin 会
        单调堆积（实测某库 78%），挤掉用层的动态位，且**全程无报错**。

        "疑似自动"的判据：``pin=True`` 且 importance 落在 ``_guess_importance``
        的值域内（``[PIN_IMPORTANCE, 0.75]``）。这是**疑似**不是确证 ——
        调用方显式传 0.7 也会落进来，所以字段名带 ``suspected``。
        """
        # 本体在 domain/stats_views.py。PIN_IMPORTANCE 调用时
        # 现取（原实现的函数内 try import 同语义：导入失败兜底 0.7）；
        # 取记录本身失败也要自曝 unknown，不让其它统计项挂掉。
        try:
            from ..store.memory import PIN_IMPORTANCE
        except Exception:      # pragma: no cover —— 导入失败不该让 stats 挂掉
            PIN_IMPORTANCE = 0.7
        try:
            records = self.manager.store.iter_records()
        except Exception as e:  # noqa: BLE001
            return _pin_stats_fn([], active_state=MemoryState.ACTIVE.value,
                                 error=type(e).__name__)
        return _pin_stats_fn(
            records,
            active_state=MemoryState.ACTIVE.value,
            pin_threshold=PIN_IMPORTANCE,
        )

    # ==================================================================
    # 主题层（L2T）：L1 的派生层——「知识点」的骨架
    #
    # 算法、阈值与硬约束全在 src/hippocampus_mcp/topic.py 的模块说明里。
    # 这里只放 brain 侧的编排（取数 / 触发时机 / 自曝），纯计算不落在这里。
    # ==================================================================

    def _knowledge_items(self, require_vec: bool = True) -> list[dict[str, Any]]:
        """把 active 的 L1 记录打包成 topic 模块的输入。

        实现在 :func:`hippocampus_mcp.domain.knowledge_ops.knowledge_items`
        本方法只做绑定：实体表/信任名单/向量查询
        **调用时现取**并注入；require_vec 的语义见新模块 docstring。
        """
        return _knowledge_ops.knowledge_items(
            self.manager.store.iter_records(), entities=self.entities,
            trusted_ids=self._trusted_entity_ids(),
            vector_of=self.manager.vector_of,
            active_state=MemoryState.ACTIVE.value, require_vec=require_vec)

    def _reclip_knowledge(self) -> int:
        """按当前存活的 active 记录重裁主题成员（派生层纪律）。"""
        if not self.knowledge:
            return 0
        alive = [r.id for r in self.manager.store.iter_records()
                 if r.state == MemoryState.ACTIVE.value]
        return topic_mod.reclip(self.knowledge, alive)

    def _forget_from_knowledge(self, memory_id: str) -> int:
        """forget 级联：把这条记忆从所有主题的成员里摘掉。

        成员被摘空的主题标 ``archived`` 而不是直接删——它是**真删除**，
        走删除必立碑口径（并集只会做加法，删了会被别人的旧快照并回来）。
        """
        if not self.knowledge:
            return 0
        dropped = 0
        for t in self.knowledge.values():
            before = list(t.get("member_ids") or [])
            after = [m for m in before if m != memory_id]
            if len(after) != len(before):
                dropped += len(before) - len(after)
                t["member_ids"] = after
                t["size"] = len(after)
                if not after:
                    t["status"] = topic_mod.STATUS_ARCHIVED
        return dropped

    def refresh_knowledge(self, force: bool = False) -> dict[str, Any]:
        """重建 / 增量刷新主题层。**只在显式调用与 consolidate 时跑**。

        三步（顺序有语义）：先按存活记录重裁 → 再增量归入未分配的 → 最后
        重算派生字段。反过来会先把已删的成员算进质心里。

        Args:
            force: **重画簇边界**（换参数、或旧结构已经长歪时用）。
                与增量的区别只有两点：整层世代 +1（合并时整代覆盖，
                所以旧簇真的会被清掉），以及旧宿主归纳按成员重叠迁移
                （见 :func:`topic.migrate_host_summaries`）。
        """
        _td = _topic_defaults()
        if not _td.layer_enabled:
            return {"enabled": False,
                    "note": "主题层已关闭（HIPPOCAMPUS_TOPIC_LAYER=0）"}
        t0 = time.time()
        items = self._knowledge_items()
        # 存活全集（含暂时没算向量的）：判「失效」只认 active，不认有没有
        # 向量（见 _knowledge_items 的 docstring）。标题/实体/标签也需要这些
        # 记录的正文，所以一并传给 refresh —— 否则主题会显示"没有成员"。
        info = {str(it["id"]): it for it in self._knowledge_items(require_vec=False)}
        self.knowledge, out, took, dropped = _refresh_flow.collect(
            items, info, self.knowledge, force=force,
            defaults=_td, max_topics=_topic_defaults().max_topics,
            now=self.manager._reference_time or _now(),
            t0=t0, clock=time.time,
            layer_epoch=topic_mod.layer_epoch, reclip=topic_mod.reclip,
            build=topic_mod.build, refresh=topic_mod.refresh,
            migrate_host_summaries=topic_mod.migrate_host_summaries,
            coverage_of=topic_mod.coverage_of, by_status=topic_mod.by_status,
            summary_kind_host=topic_mod.SUMMARY_KIND_HOST)
        self._knowledge_last_at = _now()
        self._knowledge_last_took = took
        self._knowledge_last_dropped = dropped
        return out

    def knowledge_stats(self) -> dict[str, Any]:
        """brain_stats 自曝：主题层现在什么状态（含覆盖率与上次刷新时点）。

        实现在 :func:`hippocampus_mcp.domain.knowledge_ops.knowledge_stats`
        本方法只做绑定：topic 层能力经 :class:`TopicPort`
        注入（同层不互引），刷新时点三态调用时现取。
        """
        return _knowledge_ops.knowledge_stats(
            _knowledge_ops.TopicPort(
                defaults_fn=_topic_defaults, ordered=topic_mod.ordered,
                pending_summary_ids=topic_mod.pending_summary_ids,
                representative_ids=topic_mod.representative_ids,
                summary_state=topic_mod.summary_state,
                apply_host_summary=topic_mod.apply_host_summary,
                coverage_of=topic_mod.coverage_of,
                by_status=topic_mod.by_status,
                host_summary_stats=topic_mod.host_summary_stats,
                evidence_clip=topic_mod.TOPIC_EVIDENCE_CLIP),
            knowledge=self.knowledge,
            records=self.manager.store.iter_records(),
            active_state=MemoryState.ACTIVE.value,
            last_at=self._knowledge_last_at,
            last_took=self._knowledge_last_took,
            last_dropped=self._knowledge_last_dropped)

    def knowledge_map(self, limit: int = 20,
                      include_archived: bool = False) -> dict[str, Any]:
        """库的知识结构：主题列表 + 每个主题的一句话 + 规模 + 未决项。

        **只读**。它回答的是「这个库有哪些知识点」——此前提不出来的问题：
        L1 是平铺的、L2 只有会话级摘要且写完即沉睡、L3 是热点条目不是知识。
        """
        _td = _topic_defaults()
        if not _td.layer_enabled:
            return {"enabled": False, "topics": [],
                    "note": "主题层已关闭（HIPPOCAMPUS_TOPIC_LAYER=0）"}
        # 库里从没梳理过 → 首次调用自动建一次。刻意**不**每次都建：
        # 地图是给人看的，不需要实时；之后只在 consolidate 时增量刷新。
        auto_built = False
        if (not self.knowledge and self.manager.store.count()):
            self.refresh_knowledge()
            auto_built = True
        # 三块本体在 domain/knowledge_map_view.py；topic 包五符号
        # 调用时现取后注入（domain 同层禁互引）。stats / open_* 按值现取，
        # playbook 惰性回调（只在 pending 非空时调，次数与原实现对齐）。
        return _knowledge_map_view.collect(
            self.knowledge,
            include_archived=include_archived,
            limit=limit,
            defaults=_td,
            ordered=topic_mod.ordered,
            status_archived=topic_mod.STATUS_ARCHIVED,
            summary_kind_host=topic_mod.SUMMARY_KIND_HOST,
            signature_of=topic_mod.signature_of,
            pending_summary_ids=topic_mod.pending_summary_ids,
            stats=self.knowledge_stats(),
            open_topics=[t.get("text") for t in self.open_topics[-5:]],
            open_intents=[i.get("text") for i in self.intents
                          if i.get("status") == "open"][:5],
            playbook=lambda n, m: self._knowledge_playbook(n, m),
            auto_built=auto_built,
        )


    def knowledge_topic(self, topic_id: str = "", action: str = "brief",
                        summary: str = "",
                        evidence: list[str] | None = None,
                        limit: int = 30,
                        topics: list[str] | None = None,
                        items: list[dict[str, Any]] | None = None,
                        per_topic: int = 5,
                        batch: int = 10,
                        fras: dict[str, Any] | None = None) -> dict[str, Any]:
        """主题层读写通道：出素材（brief）/ 写回归纳（summarize）/ 撤销（revert）。

        这是「上层知识」真正的入口。初版只给得出代表句——离质心最近的几条
        原文**摘录**，它回答「这个主题大概在说什么」；用户要的「归纳」是把
        N 条事实写成一句结论，那一步需要智能，而生产路径不内嵌 LLM 是本
        项目的边界（TODO J 段 D1）。所以这里是**写回通道**：本方法出素材 →
        宿主归纳 → 本方法落库。

        落库后 ``summary_kind="host"``，此后的纪律是：
          * refresh（巩固）**只填抽取式**，绝不覆盖 host 归纳；
          * 层合并（多进程并集）**host 优先**，不以时间戳论新旧；
          * 成员集合变了只标 ``stale``，不改写内容——本模块判断不了内容对不对，
            只负责把「归纳所依据的成员已经变了」这个事实摆出来。

        S3：``summarize`` 可附带轻量 **FRAS 四字段**（``fras``）：
        ``f`` 范畴（直接当标题，治"多簇同名"）／``r`` 关联实体与主题／
        ``a`` 可引用的结论（只传 fras 时可代替 summary）／``s`` 不可推断的
        约束（准入：推不出来 + 防错；没有也要显式写"无"）。四键必须齐全，
        结构不过整条拒绝。自由文本写回仍然支持（不传 fras 即可）。

        Args:
            topic_id: 主题 id（来自 ``knowledge_map``）。
            action: ``brief`` 出素材 / ``summarize`` 写回 / ``revert`` 撤销 /
                ``brief_batch`` / ``summarize_batch`` 批量 /
                ``rebuild`` 重画簇边界（世代 +1）/ ``refresh`` 增量刷新。
            summary: 归纳文本（summarize 时必填，≤2000 字；传了 fras 可省）。
            evidence: 出处（L1 记忆 id，**必须是本主题成员**），≤20 条。
            limit: 素材条数（1–200，默认 30）。
            fras: 可选四字段结构，见上。``summarize_batch`` 的每条
                item 里也可各带一个 ``fras`` 键。
        """
        if not _topic_defaults().layer_enabled:
            return {"ok": False, "enabled": False,
                    "error": "主题层已关闭（HIPPOCAMPUS_TOPIC_LAYER=0）"}
        act = str(action or "brief").strip().lower()
        if act not in ("brief", "summarize", "revert",
                       "brief_batch", "summarize_batch",
                       "rebuild", "refresh"):
            return {"ok": False, "enabled": True,
                    "error": ("action 只支持 brief / summarize / revert / "
                              "brief_batch / summarize_batch / rebuild / "
                              "refresh，收到 " + str(action))}
        auto = False
        if not self.knowledge and self.manager.store.count():
            self.refresh_knowledge()
            auto = True
        # 批量先走：它们没有"单个 topic_id"这个概念，硬塞进单条路径会
        # 变成「没有这个主题」的假错误。
        if act == "brief_batch":
            return self._knowledge_brief_batch(
                topics, batch=batch, per_topic=per_topic, auto_built=auto)
        if act == "summarize_batch":
            return self._knowledge_summarize_batch(items)
        # 「重画」也是层级操作，同样没有单个 topic_id。
        # rebuild 与 refresh 的唯一区别是 force：force=True 会把整层世代 +1，
        # 层合并时整代覆盖 ⇒ 旧簇**真的**被清掉（键并集做不到这一点，
        # 见 _merge_topic_layer 的注释）。换参数、或旧结构长歪了才用 rebuild。
        if act in ("rebuild", "refresh"):
            out = self.refresh_knowledge(force=(act == "rebuild"))
            if not out.get("enabled"):
                return {"ok": False, "enabled": False,
                        "action": act, "error": out.get("note")}
            self._persist_knowledge()
            out.update({"ok": True, "action": act, "saved": True,
                        "note": ("已重画簇边界并落盘；旧宿主归纳按成员重叠迁移"
                                 "（标 ⚠️ 待复核），迁移不了的在 lost_host 里"
                                 "自曝——不会静默消失"
                                 if act == "rebuild" else
                                 "已增量刷新并落盘（簇边界不变，不动宿主归纳）")})
            return out

        tid = str(topic_id or "").strip()
        t = self.knowledge.get(tid)
        if t is None:
            return {"ok": False, "enabled": True, "action": act,
                    "error": "没有这个主题：" + (str(topic_id) or "(空)"),
                    "hint": "先调 knowledge_map 列一下 id",
                    "sample_ids": [x.get("id") for x in
                                   topic_mod.ordered(self.knowledge)[:5]],
                    "total": len(self.knowledge)}
        # 存活成员＝active 全集（含暂时没向量的）；素材只用有向量的那些
        # （排序要算余弦）。两者分开：混用会让"成员失效"的
        # 判定跟着向量补齐一起抖。
        by_id = {str(it["id"]): it for it in self._knowledge_items()}
        info = {str(it["id"]): it for it in self._knowledge_items(require_vec=False)}
        alive = [m for m in (t.get("member_ids") or []) if m in info]
        no_vec = sum(1 for m in alive if m not in by_id)

        if act == "summarize":
            ok, err = topic_mod.apply_host_summary(
                t, summary, evidence, alive,
                now=self.manager._reference_time or _now(), fras=fras)
            if not ok:
                return {"ok": False, "enabled": True, "action": act,
                        "topic_id": tid, "error": err}
            self._persist_knowledge()
            return {"ok": True, "enabled": True, "action": act,
                    "topic_id": tid, "written": True, "saved": True,
                    "summary": topic_mod.summary_state(t, alive),
                    "note": ("已写回并落盘。host 归纳不会被 refresh / 层合并"
                             "覆盖；要改就再写一次，要退回抽取式就 action=revert")}

        if act == "revert":
            ok, err = topic_mod.revert_host_summary(t)
            if not ok:
                return {"ok": False, "enabled": True, "action": act,
                        "topic_id": tid, "error": err}
            self._persist_knowledge()
            return {"ok": True, "enabled": True, "action": act,
                    "topic_id": tid, "reverted": True, "saved": True,
                    "summary": topic_mod.summary_state(t, alive),
                    "note": "已撤销并落盘；下次 refresh 会填回抽取式代表句"}

        # ---- brief：出素材 ----
        n = _topic_material.clamp_limit(limit)
        picked = topic_mod.representative_ids(t, by_id, n)
        members = _topic_material.build_members(
            by_id, picked, clip=topic_mod.TOPIC_EVIDENCE_CLIP)
        hint = _topic_material.brief_hint(
            has_alive=bool(alive), has_vec=bool(by_id),
            has_members=bool(members))
        return _topic_material.build_brief_payload(
            tid=tid, topic=t, alive_n=len(alive), members=members,
            # no_vector：存活但没算向量的成员——它们同样属于这个主题
            #（能当 evidence），只是排不进素材顺序。不报出来宿主会以为
            # 这个主题就这么点内容。
            no_vec=no_vec, auto=auto, hint=hint,
            summary=topic_mod.summary_state(t, alive))

    # ------------------------------------------------------------------
    # 批量：让「整理一个库」从 400 次往返降到 ~40 次
    # ------------------------------------------------------------------
    def _persist_knowledge(self) -> None:
        """把主题层写盘（带 F14 账本）。

        为什么必须显式做：``refresh_knowledge`` 和写回都只改内存，落盘要自己
        调。不写这一步的话，宿主辛苦归纳的东西 daemon 一退就全没了——而且
        **没有任何报错**（内存里一直看得到，直到重启才发现）。
        """
        try:
            self._save_layers()
        except Exception as exc:  # noqa: BLE001 —— 落盘失败不能吞掉归纳本身
            logger.warning("[knowledge] 主题层落盘失败：%s", exc)

    def _knowledge_brief_batch(self, topic_ids: list[str] | None,
                               batch: int = 10, per_topic: int = 5,
                               auto_built: bool = False) -> dict[str, Any]:
        """一次取 N 个主题的归纳素材（``action="brief_batch"``）。

        ``topic_ids`` 为空 ⇒ **自动挑最该归纳的**（与 ``knowledge_map`` 的
        ``pending`` 同一口径：未归档 / 成员 ≥3 / 还没有归纳或已 stale）。
        这样宿主不用自己算该做哪些，也不会把已经写好的重做一遍。
        """
        # 四段本体在 domain/knowledge_batch.py；topic 包四回调一
        # 常量 + defaults 调用时现取后注入（domain 同层禁互引）。by_id /
        # info 两次 _knowledge_items 的调用顺序与原实现一致（先有向量、
        # 后存活全集）。注意本方法**不检查 layer_enabled**——门禁在
        # knowledge_topic 开头，行为保持原样。
        return _knowledge_batch.collect(
            topic_ids,
            batch=batch,
            per_topic=per_topic,
            auto_built=auto_built,
            knowledge=self.knowledge,
            by_id={str(it["id"]): it for it in self._knowledge_items()},
            info={str(it["id"]): it
                  for it in self._knowledge_items(require_vec=False)},
            defaults=_topic_defaults(),
            ordered=topic_mod.ordered,
            pending_summary_ids=topic_mod.pending_summary_ids,
            representative_ids=topic_mod.representative_ids,
            summary_state=topic_mod.summary_state,
            clip=topic_mod.TOPIC_EVIDENCE_CLIP,
        )

    def _knowledge_summarize_batch(
            self, items: list[dict[str, Any]] | None) -> dict[str, Any]:
        """一次写回 N 条归纳（``action="summarize_batch"``）。

        **逐条独立**：某条校验不过只退那一条，不影响同批其他条——否则一次
        手抖写错一个 id，整批几十条归纳全丢，宿主还得重新归纳一遍。
        """
        # 五段本体在 domain/summarize_batch_flow.py；topic 两回调
        # （apply_host_summary / pending_summary_ids）+ defaults 现取后注入
        # （domain 同层禁互引）。调用次数语义全保留：存活全集逐条现取
        # （每个通过主题查找的条目各一次 require_vec=False）、pending
        # 恰好两次（pending_count 与 note 各一次）、persist 仅 written
        # 非空。
        return _summarize_batch_flow.collect(
            items,
            knowledge=self.knowledge,
            items_of=self._knowledge_items,
            apply_summary=topic_mod.apply_host_summary,
            now=self.manager._reference_time or _now(),
            defaults=_topic_defaults(),
            pending_summary_ids=topic_mod.pending_summary_ids,
            persist=self._persist_knowledge,
        )

    def stats(self, daily_days: int = 0) -> dict[str, Any]:
        # 词库热感知：宿主补建 entity_lexicon.txt 后不必重启 daemon
        # （详见 :meth:`_reload_lexicon_if_changed`）。放在 stats() 里是因为它
        # 是宿主必调的自查入口，且**不新增 MCP 工具**（工具面每会话随
        # tools/list 注入宿主上下文，加一个工具就是长期 token 成本；改动工具面
        # 要同步本仓多处硬编码的工具数，全 grep `== 42` 那一类断言）。
        self._reload_lexicon_if_changed()
        # 状态分布 + L0↔L1 悬挂引用 + 按日活动窗口（本体在
        # domain/stats_projection.py；records 鸭子类型直传）。
        states = _stats_projection.state_counts(
            self.manager.store.iter_records())
        model_stats = self.manager.store.model_stats()
        missing = len(self.manager.store.missing_ids())
        # L0↔L1 引用完整性**自曝**（zcode 追加五）：L1 的 source_ids 指不到
        # L0 原文，此前只能靠考古文件才发现。这里每次统计都算一遍（O(n)），
        # 让漂移当场可见——宿主看到非 0 就知道该查保存链了。
        raw_ids = {r.get("id") for r in self.raw_log}
        dangling, dangling_recs = _stats_projection.dangling_report(
            self.manager.store.iter_records(), raw_ids)
        # 标签覆盖率自曝（验收判据）：「多少条带了域标签」此前只能靠
        # 导出整库自己数。没有这个数字，宿主只会凭感觉说"好像大部分都有"——
        # 真库实测是 41.8%，一半以上完全没有标签。
        tag_cov = self.tag_coverage()
        # 按日活动（追加五）：平行期"该想起率"的分子/分母此前只能考古
        # recall_count 总和，看不出"哪天用过、用得是否均衡"。
        n_days = self._resolve_daily_days(daily_days)
        daily_block = _stats_projection.daily_activity(
            self.daily_counts, n_days)
        # writers 必须是**此刻**的真相，不是上次落盘时的快照。
        # 这个数字用来判断「现在有几个人在动这个库」——答成过期值
        # 比不答更糟：用户会据此以为自己是独占写者。
        # _touch_writer() 顺带刷新自己的心跳：来 stats 这件事本身就
        # 证明本进程还活着。
        try:
            self._touch_writer()
        except Exception:  # noqa: BLE001 —— 自曝失败绝不能连累 stats
            pass
        # 回显体组装（45 键固定序 + 每键 rationale 注释）在
        # domain/stats_report.py：壳只算值，
        # kwargs 求值序与旧 dict 字面量逐键对齐；六投影留在壳内
        # 各恰一次。test_f6 F3 以源码含 "embedder_shared"
        # 字面量为锚，该键的计算留在壳里。
        return _stats_report.assemble_report(
            l0_raw=len(self.raw_log),
            l1_memories=self.manager.store.count(),
            tags=tag_cov,
            l0_dangling_source_ids=len(dangling),
            l0_dangling_memories=dangling_recs,
            daily_activity=daily_block,
            l1_by_state=states,
            recall_feedback=self._recall_feedback_stats(),
            pin_stats=self._pin_stats(),
            l2_summaries=len(self.summaries),
            topic_layer=self.knowledge_stats(),
            l3_profile_slots=len(self.profile),
            entities=len(self.entities),
            sessions=len(self.sessions),
            open_intents=sum(1 for i in self.intents if i["status"] == "open"),
            layer_merge_last=dict(self.last_layer_merge),
            layer_del_enabled=load_config().layer_del,
            layer_del_tombstones_n=len(self._layer_del),
            layer_del_blocked_last=self._layer_del_blocked,
            layer_del_gc_last=self._layer_del_gc_last,
            source_sig_enabled=load_config().source_sig,
            source_sig_block=_stats_projection.source_sig_block(
                self.source_sigs, self.manager.store.iter_records()),
            writers=_stats_projection.writers_block(
                self.writers_active, self.writers_stale_cleaned,
                os.getpid()),
            open_topics=len(self.open_topics),
            pinned=sum(1 for r in self.manager.store.iter_records() if r.pin),
            tombstones=len(self.tombstones),
            raw_forgotten=_stats_projection.forgotten_count(self.raw_log),
            embed_model=self.manager.model_key,
            embedder_backend=self.manager.embedder_backend,
            embedder_runtime=self.manager.embedder_runtime_backend(),
            onnx_attached=(
                self.manager.embedder_runtime_backend() == "onnx"),
            lnn_runtime=getattr(self.manager.memory_core, "runtime", None),
            pending_consolidate=self._pending_consolidate,
            embedding_dim=getattr(self.manager, "embedding_dim", None),
            embedder_model_source=(
                getattr(self.manager.embedder, "model_source", None)
                or "n/a（该后端不经模型名解析）"),
            # F6 自曝（键名锚点，test_f6 F3 查源码含 "embedder_shared"）：
            # 本库的嵌入器是否复用了进程内共享的那份重对象。
            embedder_shared=bool(
                getattr(self.manager.embedder, "shared_from_pool", False)),
            embedder_pool=embedder_pool_stats(),
            ner_backend=self.ner_backend_lazy,
            lexical=self.lexical_self_report(),
            model_stats=model_stats,
            vectors_missing=missing,
            supersede_enabled=self.supersede_enabled,
            entity_lexicon_size=len(self.entity_lexicon),
            entity_aliases=len(self.entity_lexicon_aliases),
            tag_schema=self.tag_extractor.describe(),
            entity_origins=self.entity_origin_stats(),
            sources=len([s for s in self.manager.sources() if s]),
            save_ledger=self.save_ledger_view(),
            l0_pending=self._l0_pending,
        )

    # ------------------------------------------------------------------
    # 项目概况与状态卡
    # 立场：session_start 预激活包的两个新槽——agent 每次开工时
    # 「这个库整体是什么、现在什么状况」的紧凑投影。纯派生、不检索、
    # 不聚类（热路径纪律：首次知识梳理 2-3 秒绝不进每次开场）。
    # 信号只放「推不出但错了会出事」的。
    # ------------------------------------------------------------------

    def contradictions_count(self, limit: int = STATUS_CONTRADICTION_CAP) -> dict[str, Any]:
        """状态卡用的矛盾计数（带预算上限；Stale 同型信号）。

        全库对扫在万条库上约 0.34s（vector_store.pairwise_similarities 实测），
        热路径（每次开场）不付全量代价，故设上限：达到 limit 即
        ``capped=True``，卡片报「≥N」而不是假精确。``limit<=0`` = 关闭。

        Returns:
            ``{count, capped, limit}``。count 是截断后的实际对数；
            capped=True 时真实值 ≥ limit。
        """
        if limit <= 0:
            return {"count": 0, "capped": False, "limit": 0}
        pairs = self._find_contradictions(limit=limit)
        return {
            "count": len(pairs),
            "capped": len(pairs) >= limit,
            "limit": limit,
        }

    def project_overview(self) -> dict[str, Any]:
        """库级全貌投影（设计文档 §3.1）。纯派生、不检索、O(n) 单遍。

        五样定位信息：身份+规模 / 时间跨度 / 结构画像 / 热点实体。
        全部来自既有派生层（tag_coverage / knowledge_stats / entities），
        唯一新派生量是记录时间戳的 min/max。口径见 :mod:`..domain.project_view`。
        """
        return _project_overview(
            records=list(self.manager.store.iter_records()),
            entities=self.entities,
            knowledge=self.knowledge,
            daily_counts=self.daily_counts,
            ref_time=self.manager._reference_time,
            # display_name 住 library/（arch G1：domain 只准依赖 common）
            name=display_name(self.persist_dir),
            topic_layer_enabled=_topic_defaults().layer_enabled,
            # 惰性回调：开关关掉时一次都不调
            tag_coverage=self.tag_coverage,
            knowledge_stats=self.knowledge_stats,
            recent_days=OVERVIEW_RECENT_DAYS,
            top_entities=OVERVIEW_TOP_ENTITIES,
            top_tag_keys=OVERVIEW_TOP_TAG_KEYS,
        )

    def project_status(self) -> dict[str, Any]:
        """当前状态投影（设计文档 §3.2）。纯派生、不检索。

        四组信号：未决（开放话题/待办/巩固）/ 健康（L0 悬空/矛盾/miss 缺口）/
        协作（活跃会话）/ 活动（sparkline 序列）。矛盾计数带预算上限，
        可用 env 关闭（万条库上约 0.34s，见 contradictions_count）。
        """
        return _project_status(
            records=list(self.manager.store.iter_records()),
            raw_log=self.raw_log,
            open_topics=self.open_topics,
            intents=self.intents,
            miss_log=self.miss_log,
            daily_counts=self.daily_counts,
            ref_time=self.manager._reference_time,
            pending_consolidate=self._pending_consolidate,
            contradictions_enabled=load_config().status_contradictions,
            # 惰性回调：开关关掉时一次都不调
            contradictions_count=self.contradictions_count,
            active_sessions=self.active_sessions,
            file_registry_summary=self.file_registry_summary,
            contradiction_cap=STATUS_CONTRADICTION_CAP,
            miss_days=STATUS_MISS_DAYS,
            spark_days=STATUS_SPARKLINE_DAYS,
        )

    # ==================================================================
    # 地图索引行 + 文件建档
    # maps.json 侧车图合并进索引 + L1 相关图全文
    # ==================================================================

    # ── 闸一·legacy 侧守卫─────────────────────────────────────────

    def find_map_entry_record(self, name: str, key: str):
        """按（地图, 条目）保留标签找**活**条目记忆；没有返回 None。

        归档（consolidate 沉寂清理）过的不算——死条目不挡重建。
        O(册内条数)，只在带地图标签的写入路径上调用（见 _guard_map_entry_dup）。
        """
        name, key = str(name or "").strip(), str(key or "").strip()
        if not name or not key:
            return None
        for rec in self.manager.store.iter_records():
            if rec.state == MemoryState.ARCHIVED.value:
                continue
            tags = rec.tags or {}
            if (str(tags.get(MAP_NAME_KEY) or "").strip() == name
                    and str(tags.get(MAP_ENTRY_KEY) or "").strip() == key):
                return rec
        return None

    def _guard_map_entry_dup(self, tags: dict[str, Any] | None) -> None:
        """ingest 侧同键查重（闸一）。用法错误抛 ValueError——ingest_batch
        会把它隔离进 errors，不中断整批（与 map/store 的护栏同哲学）。
        """
        if not tags:
            return
        name = str(tags.get(MAP_NAME_KEY) or "").strip()
        key = str(tags.get(MAP_ENTRY_KEY) or "").strip()
        if not name or not key:
            return
        dup = self.find_map_entry_record(name, key)
        if dup is None:
            return
        cur = str((dup.tags or {}).get(MAP_STATUS_KEY) or "（无状态）")
        raise ValueError(
            "地图条目「%s / %s」已存在（记忆 %s，当前状态：%s）——地图是"
            "「现在是什么」的按键状态，同键只能有一条。改状态请走 "
            "update_tags（legacy）或 map_set（registry 侧车，推荐），"
            "不要用 ingest 写第二条" % (name, key, dup.id[:8], cur))

    def map_index(self) -> dict[str, Any]:
        """地图索引（L0 递送）：库里有哪几地图、各多少条、多久没更新。

        两个来源合并：
        - **legacy**：带保留标签键的记忆（``{"地图": "合约", "条目":
          "CN2601", "状态": "验证集"}``），一键一条；
        - **registry**：``maps.json`` 侧车（map_set 写入的正式地图层）。

        同名图相加（entries 数相加、取最新、来源标 ``both``）——老数据不丢，
        新数据走新路。只报名+条数+新鲜度+漂移数+冲突数，不带正文（
        零正文是「必要信息」与「不能太长」的平衡点）；正文走 L1 命中进包 /
        L2 ``map_get`` 精确查询。纯 O(n) 读既有层，不检索不聚类。

        设计评审收紧：
        - **归档条目不进索引**（consolidate 会把沉寂的地图条目记忆归档，
          继续计条数会让索引行虚高，且归档条目 recall 不可达）；
        - **双源同键冲突检测**：同一个（图, 条目键）在 legacy 记忆与
          registry 侧车里状态不一致 ⇒ ``conflicts`` 计数 + 索引行「冲突N」。
          相加只解决了计数合并，没解决「两条状态竞争」——那正是设计 §2
          的核否命题，修复前它可以无声无息地活在 both 图里。
        """
        return _map_index_fn(
            self.manager.store.iter_records(),
            map_mod.load(self.persist_dir),
            list_maps=map_mod.list_maps,
            now=_now,
            archived_state=MemoryState.ARCHIVED.value,
            limits=_MapIndexLimits(
                max_maps=MAP_INDEX_MAX, stale_days=MAP_STALE_DAYS,
                name_key=MAP_NAME_KEY, entry_key=MAP_ENTRY_KEY,
                status_key=MAP_STATUS_KEY,
            ),
        )

    # ==================================================================
    # 地图层（maps.json 侧车 + L1 相关图全文）
    # 设计文档 §5-L1 / §6：任务 token/锚点实体 ∩ 图 desc/条目名 ⇒ 该图
    # 整图进开场包（每图 cap 20 行 + 「等 N 条」自曝）；不命中就不递
    # ——反例约束由匹配自然保证，不需要人工配置「谁看哪张图」。
    # ==================================================================
    def map_detail(self, task: str,
                   tp: dict[str, Any] | None) -> dict[str, Any]:
        """任务相关地图的全文投影（L1 递送）。纯读侧车，不检索不引模型。

        token = 任务正文 + tp 里的锚点实体名（与二期 task_profile 同一
        个 :func:`task_tokens`；**不重新检索**）。命中判据（§6）：
        某张图的 desc 含任一 token，或其任一条目键含任一 token。
        """
        return _map_detail_fn(
            task, tp,
            map_mod.load(self.persist_dir),
            list_maps=map_mod.list_maps,
            task_tokens_fn=task_tokens,
            limits=_MapDetailLimits(
                max_maps=MAP_DETAIL_MAX_MAPS, max_rows=MAP_DETAIL_MAX_ROWS,
            ),
        )

    # ── 地图工具后端（server.py 的 map_* 四工具是薄包装，同 drift_audit 先例）──

    def map_source_root(self) -> str | None:
        """truth_source 相对路径的解析根：文件建档的项目根（opt-in）。

        没启用建档返回 None——resolve_source 回落到进程 cwd。
        """
        reg = fr_mod.load(self.persist_dir)
        return (reg or {}).get("root") or None

    def map_set(self, name: str, key: str, status: str = "",
                fields: dict | None = None, desc: str = "",
                truth_source: str = "", note: str = "",
                updated_by: str = "") -> dict[str, Any]:
        """闸一写入口：按键 upsert（同键再写整体替换，旧状态不竞争）。

        实现在 :func:`hippocampus_mcp.domain.map_ops.map_set`。本方法只做绑定：map 包能力经构造注入（``store``/``domain``
        都不许 import ``map``），busy/unreadable 异常类按值传。
        """
        return _map_ops.map_set(
            name, key, status=status, fields=fields, desc=desc,
            truth_source=truth_source, note=note, updated_by=updated_by,
            persist_dir=self.persist_dir,
            set_entry=map_mod.set_entry, update_fn=map_mod.update,
            root_fn=self.map_source_root, busy_exc=map_mod.SidecarBusy,
            unreadable_exc=SidecarUnreadable, wait=MAP_WRITE_WAIT_S)

    def map_get(self, name: str, key: str) -> dict[str, Any]:
        """L2 精确查询：单条目。**未命中显式报**（不许静默空）。"""
        data = map_mod.load(self.persist_dir)
        if data is None:
            return {"found": False, "reason": "no_maps", "map": name,
                    "key": key,
                    "hint": "本地图不存在（未启用地图层）——先 map_set 建图"}
        ent = map_mod.get_entry(data, name, key)
        if ent is not None:
            return {"found": True, "map": name, "key": key, "entry": ent}
        m = (data.get("maps") or {}).get(str(name))
        if m is None:
            return {"found": False, "reason": "no_map", "map": name,
                    "key": key,
                    "hint": "没有这张图；现有图：%s"
                            % ("、".join(map_mod.map_names(data)) or "（空）")}
        return {"found": False, "reason": "no_key", "map": name, "key": key,
                "hint": "该图有 %d 个条目；要全列表走 map_list(name=...)"
                        % len(m.get("entries") or {})}

    def map_list(self, name: str = "") -> dict[str, Any]:
        """列地图（name 空）或列一张图的条目（带 cap + 自曝）。

        实现在 :func:`hippocampus_mcp.domain.map_ops.map_list`。本方法只做绑定：``load`` 调用时现取后把 data 传入；条目
        行截字面量 100 与图数上限分开的怪癖见新模块 docstring。
        """
        return _map_ops.map_list(
            map_mod.load(self.persist_dir), name,
            list_maps_fn=map_mod.list_maps, map_names_fn=map_mod.map_names,
            max_maps=MAP_INDEX_MAX)

    def map_verify(self, name: str = "") -> dict[str, Any]:
        """闸二：重读 truth_source 比对签名（哈希仲裁，touch 不误报）。

        实现在 :func:`hippocampus_mcp.domain.map_ops.map_verify`。本方法只做绑定：map 包能力经构造注入；锁内写 drift、
        busy/unreadable 回显形状与 map_set 刻意不同（见新模块 docstring）。
        """
        return _map_ops.map_verify(
            name, persist_dir=self.persist_dir,
            verify_fn=map_mod.verify, update_fn=map_mod.update,
            list_maps_fn=map_mod.list_maps,
            root_fn=self.map_source_root, busy_exc=map_mod.SidecarBusy,
            unreadable_exc=SidecarUnreadable, wait=MAP_WRITE_WAIT_S)

    def map_auto_verify(self, force: bool = False) -> dict[str, Any]:
        """按信号自动 verify——值级校验 / 目录对账。

        实现在 :func:`hippocampus_mcp.domain.map_drift.auto_verify`
        本方法只做绑定：map 包能力经构造注入；
        config 三旋钮调用时现取按值传；间隔门控/预算/best-effort 落盘
        的运行时纪律见新模块 docstring。
        """
        return _map_drift.auto_verify(
            force, persist_dir=self.persist_dir, now_fn=_now,
            enabled=load_config().map_auto_verify,
            interval_h=load_config().map_auto_interval_h,
            budget_s=load_config().map_auto_budget_s,
            registry_load=fr_mod.load, auto_run=map_auto_mod.run,
            auto_needs_registry=map_auto_mod.needs_registry,
            update_fn=map_mod.update, busy_exc=SidecarBusy)

    def map_mention_drift(self, force: bool = False) -> dict[str, Any]:
        """闸三：更新后被提到「另一个状态」⇒ 条可疑过期。

        实现在 :func:`hippocampus_mcp.domain.map_drift.mention_drift`
        本方法只做绑定：map 包能力经构造注入；
        记录快照走 ``records_fn`` 惰性回调（过门控才物化）；判据三重与
        弱信号不覆盖强事实的纪律见新模块 docstring。
        """
        return _map_drift.mention_drift(
            force, persist_dir=self.persist_dir, now_fn=_now,
            enabled=load_config().map_mention,
            interval_h=load_config().map_mention_interval_h,
            budget_s=load_config().map_mention_budget_s,
            max_records=load_config().map_mention_max_records,
            needs_run_fn=map_mention_mod.needs_run,
            run_fn=map_mention_mod.run,
            records_fn=lambda: list(self.manager.store.iter_records()),
            update_fn=map_mod.update, busy_exc=SidecarBusy)

    def file_registry_summary(self) -> dict[str, Any] | None:
        """文件建档摘要。**没启用（无侧车）返回 None**——opt-in。

        启用且距上次扫描超过间隔（默认 6h，env 可调）时带预算重扫一次：
        session_start 是每次开场的热路径，全量 stat 也不能无上限——
        超了标 scan_complete=false 自曝「扫描不完整」。扫描失败不炸
        （返回上一次的摘要，last_checked 不前进——下次开场会再试）。

        设计评审收紧：扫完与**源签名层对账**
        （:meth:`_fr_reconcile_source_sigs`）——重新 ingest 过的文件，
        ``source_sigs`` 里躺着当前内容版本的签名，此时 Hippo 已反映磁盘
        内容，建档不该再报 stale。否则同一个文件会出现两个矛盾答案：
        drift_audit（记过新签名 ⇒ 自愈）vs file_registry（没手工 sync
        ⇒ 标红），而手工 sync 这一步最容易被忘。
        """
        if load_config().file_registry_off:
            return None
        now = _now()
        interval_h = load_config().file_registry_interval_h
        limits = fr_mod.scan_limits()
        # 编排在 domain/file_registry_flow.py：
        # fr_mod 四入口调用时现取注入（patch core.fr_mod 即生效），
        # 对账回调裸引用（读 self.source_sigs，brain 态）。
        return _fr_flow.summary_flow(
            persist_dir=self.persist_dir,
            now=now, interval_h=interval_h, limits=limits,
            load=fr_mod.load, scan=fr_mod.scan, update=fr_mod.update,
            summarize=fr_mod.summarize,
            reconcile=self._fr_reconcile_source_sigs,
        )

    def _fr_reconcile_source_sigs(self, reg: dict[str, Any]) -> int:
        """建档扫描 × 源签名层对账：hippo 吃过当前内容版本 ⇒ 清 stale。

        判据是 (mtime_ns, size) 双等——与建档层同 threat model（墙钟比对
        的对抗性场景本来就不在防御范围内，见 file_registry docstring 纪律 2）。
        返回清掉的条数（0 = 没得可清，常态）。
        """
        root = reg.get("root") or ""
        if not root or not self.source_sigs:
            return 0
        # 索引构造是 fr 模块的纯函数（CLI 校验 sync 也用它）——两边同口径，
        # 免得一处改了归一化/取最新，另一处慢慢漂移成两套事实。
        sig_by_path = fr_mod.source_sig_index(self.source_sigs)
        # stale 清账循环在 domain/file_registry_flow.py（D-3）：
        # 路径归一化经回调注入（os 不进 domain），last_synced 逐条现取。
        return _fr_flow.reconcile_stale_entries(
            reg=reg, sig_by_path=sig_by_path,
            path_of=lambda rel: os.path.normcase(
                os.path.join(root, str(rel))),
            now=_now,
        )

    # ==================================================================
    # 任务级「问题概况」
    # ==================================================================
    def task_profile(self, task: str,
                     task_hits: list[dict[str, Any]]) -> dict[str, Any]:
        """任务触及的结构投影（设计文档 §6）。纯派生、不检索、不引模型。

        锚点是 session_start **已经算好的** ``task_hits``（预召回跑过一次，
        这里绝不重新检索——热路径纪律与全貌卡相同）。四块口径与空任务时
        「块空 ⟺ 结构化空」的双轨纪律见 :mod:`..domain.task_profile`。
        """
        return _task_profile(
            task, task_hits,
            knowledge=self.knowledge,
            entities=self.entities,
            intents=self.intents,
            miss_log=self.miss_log,
            ref_time=self.manager._reference_time,
            # 开关与五档上限一律调用时现取（包内常量表冻结在 core）
            topic_layer_enabled=_topic_defaults().layer_enabled,
            limits=_TaskProfileLimits(
                topic_max=TASK_TOPIC_MAX, entity_max=TASK_ENTITY_MAX,
                intent_max=TASK_INTENT_MAX, miss_max=TASK_MISS_MAX,
                miss_days=TASK_MISS_DAYS,
            ),
            # 主题口径注入（arch G1：domain 只准依赖 common，同层也不行）
            ordered_topics=topic_mod.ordered,
            archived_status=topic_mod.STATUS_ARCHIVED,
            now=_now,
        )

    def _read_raw_log_from_disk(self) -> list[dict]:
        """读磁盘上的 L0 流水（用于写前并集）。读不到就当空——不阻塞保存。"""
        if not self.persist_dir:
            return []
        # G1：走后端 —— sqlite 下 L0 在 layers 表里，没有单独文件可读，
        # 直接 open 会静默返回空，多写合并时把别人的 raw 写没（同款事故）。
        data = self._read_layer_from_disk("brain_raw.json")
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # 派生层「读-合并-写」
    # ------------------------------------------------------------------

    # 层文件 -> 内存属性
    _LAYER_ATTRS: dict[str, str] = {
        "brain_raw.json": "raw_log",
        "brain_entities.json": "entities",
        "brain_summaries.json": "summaries",
        "brain_profile.json": "profile",
        "brain_sessions.json": "sessions",
        "brain_intents.json": "intents",
        "brain_topics.json": "open_topics",
        "brain_tombstones.json": "tombstones",
        "brain_misslog.json": "miss_log",
        "brain_counters.json": "daily_counts",
        "brain_knowledge.json": "knowledge",
        # 删除账本。它**不是**第 11 份业务数据，
        # 是"哪些键被删过"的墓碑集，合并时被别的层读来拦复活。
        LAYER_DEL_FILE: "_layer_del",
        # 源文件签名层（drift_audit 的事实源）。
        SOURCE_SIG_FILE: "source_sigs",
    }

    # 层文件 -> 合并口径
    _LAYER_MERGE_RULES: dict[str, str] = {
        "brain_raw.json": "raw",
        "brain_summaries.json": "list",
        # 待办不能走通用 list：那会让已完成的状态被旧快照撤销，见
        # :meth:`_merge_intent_layer`。
        "brain_intents.json": "intent",
        # 待办话题也不能走通用 list：纯追加无上限会无声膨胀（同 miss_log
        # 的理由），走 list_cap 裁最旧，上限 OPEN_TOPICS_MAX。
        "brain_topics.json": "list_cap",
        "brain_tombstones.json": "list",
        "brain_misslog.json": "list_cap",
        "brain_entities.json": "entity",
        "brain_profile.json": "dict",
        "brain_sessions.json": "session",
        "brain_counters.json": "counter",
        "brain_knowledge.json": "topic",
        # 删除账本自己走「同键取 max」：两侧都是删除，取更晚的那次才不会
        # 把对方的删除记弱（dict 规则的"本地优先"在这里是错的）。
        LAYER_DEL_FILE: "del_ledger",
        # C1a：签名层同键取 sig_at 较新者——两个进程都 ingest 同一个文件时，
        # 后校验的那份才是磁盘现状；sha1 相同则事实无差，取谁都一样。
        SOURCE_SIG_FILE: "sourcesig",
    }

    # 待办状态的**前进次序**：只能往右走，不许退回来。
    # 多进程共库时若仍按「谁最后落盘谁说了算」，一个进程刚标记完成，
    # 另一个只持有旧快照的进程一次保存就能把它撤销 —— 且两边都没有报错。
    _INTENT_STATUS_RANK: dict[str, int] = {"open": 0, "done": 1}








    def _read_layer_from_disk(self, name: str):
        """读磁盘上一层（读不到 / 损坏 -> None，按"磁盘无此层"处理）。

        G1：读盘走 :attr:`_backend`（json 文件 / sqlite 的 layers 表）。
        """
        if not self.persist_dir:
            return None
        backend = self._backend or get_backend(self.persist_dir)
        return backend.read_layer(name)


    def note_layer_deletion(self, layer: str, keys) -> int:
        """记下"这些键在 **layer** 层被删了"，供后续合并拦截迟到的旧版本。

        Returns:
            新记下几条（0 = 账本被关掉 / 没给有效键）。
        """
        if not load_config().layer_del or not keys:
            return 0
        now = _now()
        n = 0
        for k in keys:
            if not k:
                continue
            self._layer_del[self._del_ledger_key(layer, str(k))] = now
            n += 1
        return n

    def note_layer_deletion_items(self, layer: str, items) -> int:
        """list 层专用：按 :meth:`_layer_key` 给一批被删条目立碑。"""
        if not load_config().layer_del:
            return 0
        keys = [self._layer_key(it) for it in (items or []) if isinstance(it, dict)]
        return self.note_layer_deletion(layer, keys)



    def _apply_layer_deletions(self, name: str, merged):
        """合并之后：把账本里已标记删除的键挡掉。

        实现在 :func:`hippocampus_mcp.store.layer_gc.apply_layer_deletions`
        本方法只做绑定 + 计数回写（``_layer_del_blocked``
        由壳累加）；放行条件与两种形态的拦截见新模块 docstring。
        """
        merged, blocked = _layer_gc.apply_layer_deletions(
            name, merged, enabled=load_config().layer_del,
            layer_del=self._layer_del, layer_key_fn=self._layer_key,
            entry_ts_fn=self._entry_ts)
        self._layer_del_blocked += blocked
        return merged

    def _gc_layer_del(self) -> int:
        """账本瘦身：丢超龄的；仍超量就丢最旧的。返回丢掉几条。

        实现在 :func:`hippocampus_mcp.store.layer_gc.gc_layer_del`
        本方法只做绑定 + 回写（模块返回新账本，
        壳重绑 ``self._layer_del``——与原实现的重绑语义一致）。
        """
        new_ledger, dropped = _layer_gc.gc_layer_del(
            self._layer_del, now_fn=_now, ttl=LAYER_DEL_TTL,
            max_keys=LAYER_DEL_MAX_KEYS)
        self._layer_del = new_ledger
        return dropped

    def _merge_layer(self, name: str, disk, mine):
        """按层口径合并（磁盘打底 + 本地补充），再按删除账本拦截。

        实现在 :func:`hippocampus_mcp.store.layer_gc.merge_layer`，本方法只做
        绑定：规则表与十路合并回调调用时现取注入（helper 在 brain/pure.py）。
        """
        return _layer_gc.merge_layer(
            name, disk, mine, rules=self._LAYER_MERGE_RULES,
            fns={
                "del_ledger": self._merge_del_ledger_layer,
                "raw": self._merge_raw_log,
                "list": self._merge_list_layer,
                "list_cap": self._merge_list_layer,
                "entity": self._merge_entity_layer,
                "session": self._merge_session_layer,
                "intent": self._merge_intent_layer,
                "counter": self._merge_counter_layer,
                "topic": self._merge_topic_layer,
                "sourcesig": self._merge_sourcesig_layer,
                "dict": self._merge_dict_layer,
            },
            apply_del_fn=self._apply_layer_deletions,
            miss_log_max=MISS_LOG_MAX,
            # topics 的 per-层 cap（patch core 侧 OPEN_TOPICS_MAX 即生效）。
            list_caps={"brain_topics.json": OPEN_TOPICS_MAX})



    def _disk_only_count(self, name: str, disk, mine) -> int:
        """磁盘里有、本进程内存里没有的条数（= 这次合并救回来几条）。"""
        if isinstance(disk, dict) and isinstance(mine, dict):
            return len(set(disk) - set(mine))
        if isinstance(disk, list) and isinstance(mine, list):
            mk = {self._layer_key(x) for x in mine if isinstance(x, dict)}
            dk = {self._layer_key(x) for x in disk if isinstance(x, dict)}
            return len(dk - mk)
        return 0


    def _layer_changed_by_other(self, name: str, path: str) -> bool:
        """这个层文件自本进程上次落盘后，是否被**别人**动过。

        没被动过就不必读盘合并 —— 内存这份就是最新的。单库单写者是常态，
        不加这道判断时 perf 用例（N=1000）从 30s 涨到 192s：每次 save 都
        把 10 个层文件读一遍再并一遍，纯浪费。

        首次落盘（没戳）一律返回 True：这个库可能早就在给别的进程服务。
        """
        if name not in self._layer_stamp:
            return True
        # G1：sqlite 没有层文件，戳由后端自己给（updated_at + 长度）
        return self._layer_stamp.get(name) != (
            self._backend.layer_stamp(name) if self._backend
            else self._stamp_of(path)
        )

    def _reclip_derived_layers(self) -> None:
        """并集之后，按**当前真相**重新裁剪派生层。

        实现在 :func:`hippocampus_mcp.store.layer_gc.reclip_derived_layers`
        本方法只做绑定：三容器注入本体（实体摘空壳
        就地 pop）+ 三个 brain 能力回调调用时现取；"宁可漏裁不可误裁"
        的 count 门见新模块 docstring。
        """
        _layer_gc.reclip_derived_layers(
            self.entities, self.profile, self.knowledge,
            count_fn=self.manager.store.count,
            recompute_entity_stats_fn=self._recompute_entity_stats,
            prune_fn=self.prune_stale_profile,
            reclip_knowledge_fn=self._reclip_knowledge)



    # ------------------------------------------------------------------
    # F14 保存账本：一次保存 = begin + end 两行；异常路径记 fail
    # ------------------------------------------------------------------
    def _layer_count(self, attr: str) -> int:
        """该层当前的条目数（账本 begin/end 的条数字段）。"""
        try:
            return len(getattr(self, attr))
        except TypeError:
            return -1

    def _ledger_path(self) -> str | None:
        """账本文件路径；无 persist_dir（纯内存 brain）返回 None。"""
        return ledger_path(self.persist_dir)

    def _ledger_read_tail(self) -> tuple[dict[str, Any] | None, int, list]:
        """读账本最后一条 + 总行数 + 尾部若干行（启动对账用）。

        坏行逐行跳过、坏文件按惯例隔离为 .corrupt-<ts>，
        **绝不阻塞启动**（设计 §8 风险 9：账本缺失/损坏报 unknown 并继续）。
        """
        return read_tail(self.persist_dir)

    def _ledger_append(self, entry: dict[str, Any]) -> None:
        """追加一行。**追加式是硬要求**：全量重写会让它自己变成新的丢数据点。

        落盘成功才更新末条与行数（写失败时「改之前是什么」不能指向一条
        没写成功的记录）；行数达到 trim_at 的裁剪在 store 侧完成，
        ring / trim_at 在**调用时**从 core 命名空间现取——
        ``core.LEDGER_RING`` 被测试 monkeypatch，快照会让 L5 失效。
        """
        ok, n = ledger_append(
            self.persist_dir, entry,
            self._ledger_lines, LEDGER_TRIM_AT, LEDGER_RING,
        )
        if ok:
            self._ledger_last = entry
            self._ledger_lines = n

    def _ledger_trim(self) -> None:
        """ring buffer：只保留最近 LEDGER_RING 条，保证账本体积有界。

        store 侧返回 None（无 persist_dir / 读失败 / 写失败）时行数
        保持原值——账本行数与磁盘实况允许短暂不一致。
        """
        n = ledger_trim(self.persist_dir, LEDGER_RING)
        if n is not None:
            self._ledger_lines = n

    def save_ledger_view(self) -> dict[str, Any]:
        """F14：账本投影（``brain_stats.save_ledger``）。

        范式照抄 ``recall_feedback``：数字 + level + 人话 note。
        空库 / 从未落盘报 ``null`` + ``unknown``，**绝不报 0** ——
        0 会被读成「刚保存过」，那是比 unknown 更糟的谎话。
        """
        # 参数按原实现的求值顺序给：quadrant → l1_dirty → pending。
        # 两次 ``time.monotonic()``（at_mono 兜底 / 算间隔）经 ``now`` 回调，
        # 与原实现逐次对齐（注入固定钟后 interval_s 可复现）。
        return ledger_view(
            self._ledger_last, self._ledger_lines, self._boot_reconcile,
            self.classify_layers(),
            bool(
                getattr(getattr(self.manager, "store", None), "_dirty_ids", None)
                or getattr(getattr(self.manager, "store", None), "_deleted_ids", None)
            ),
            self._pending_layer_names(),
            now=time.monotonic,
            stamp_of=lambda ts: time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(ts)),
        )

    def _save_layers(self) -> None:
        """F14：给本次保存建账，真正的写盘交给 :meth:`_save_layers_core`。

        三段各有理由：``begin`` 必须在写盘**之前**（崩溃时至少留下「想写什么」），
        ``end`` 记实际结果，``fail`` 记失败原因 —— 不记就等于「曾经想写但没写成」
        这件事从未发生过（设计 §8 风险 8）。编排本体在
        :mod:`hippocampus_mcp.store.layer_save`。
        """
        try:
            run_layer_save(LayerLedgerPorts(
                persist_dir=self.persist_dir,
                save_seq=self._save_seq,
                layer_attrs=self._LAYER_ATTRS,
                now_wall=lambda: self.manager._reference_time or time.time(),
                now_mono=time.monotonic,
                get_pid=os.getpid,
                compute_all_fps=self._compute_all_layer_fps,
                layer_count=self._layer_count,
                ledger_append=self._ledger_append,
                set_begin_fps=lambda fps: setattr(self, "_begin_layer_fps", fps),
                core_fn=self._save_layers_core,
            ))
        except BaseException:
            # B（验证反馈 2026-11-01）：失败侧明细——写盘抛异常时全层标
            # failed（部分写入不可知，如实从严），session_end 的
            # saved=false + save_error 与之配对，上层不再只看聚合布尔。
            self.last_save_layers = {n: "failed" for n in self._LAYER_ATTRS}
            raise


    def _pending_layer_names(self) -> list[str]:
        """F14 切片 2/3：此刻**内存里有改动、但还没写下去**的层名。

        四象限的"内存侧"那一维就来自这里：拿当前内存指纹与上次落盘指纹比。
        没有落盘记录时（新库）返回空 —— 不是"没有改动"，是"无从判断"，
        这一点由 :meth:`save_ledger_view` 的 ``pending_note`` 说清楚。
        """
        return pending_layer_names(
            self._LAYER_ATTRS, self._layer_written_fp,
            lambda attr: self._layer_fingerprint(getattr(self, attr)),
        )

    def _compute_all_layer_fps(self) -> dict[str, str]:
        """全部层的内容指纹（账本 begin/end 行的 ``layer_fps`` 字段）。

        指纹开关关闭时返回空 dict —— 账本行里就不带 ``layer_fps``，
        启动对账据此诚实降级（不猜）。
        """
        return compute_all_layer_fps(
            self._LAYER_ATTRS,
            lambda attr: self._layer_fingerprint(getattr(self, attr)),
            # 旋钮调用时现取：monkeypatch ``core.load_config`` 仍生效
            load_config().layer_fp,
        )

    def classify_layers(self) -> dict[str, str]:
        """F14-3 四象限判定（口径 2）：逐层给出当前归属。

        内存侧（**主判据**）＝当前内存指纹 vs 本进程上次落盘指纹；
        磁盘侧（**仅辅证**）＝层文件戳还在不在（mtime 永不单独定罪——
        touch 不改内容时指纹不变，仍算 clean，只是记一笔 external_write）。
        本进程从没写过该层 → ``unknown``（是"还没建"，不是"改了没写"）。
        """
        return classify_layer_quadrants(
            self._LAYER_ATTRS, self._layer_written_fp, self._layer_stamp,
            self.persist_dir,
            lambda attr: self._layer_fingerprint(getattr(self, attr)),
            self._layer_changed_by_other,
        )

    def _reconcile_ledger_on_boot(self) -> None:
        """启动对账：报「哪层有改动却没写下去」（lost_since_last_save）。

        实现在 :func:`hippocampus_mcp.store.layer_gc.reconcile_ledger_on_boot`
        （纯解析委托 store.ledger.reconcile_on_boot）。
        三种可判情形：末条 ``end`` 带指纹 → ``disk_mismatch``；末条
        ``begin`` → 磁盘 ≠ begin 指纹的层报 lost；末条 ``fail`` → 回看
        尾部找 begin 按上一条判。**已知局限**：进程带着改动被杀且从未
        发起 save（无 begin）⇒ 账本无从知晓。双钟纪律：不新增时间字段。
        本方法只做绑定 + 日志自曝。
        """
        self._boot_reconcile = None
        last = self._ledger_last
        if not isinstance(last, dict):
            return
        attrs = self._LAYER_ATTRS
        rec = _layer_gc.reconcile_ledger_on_boot(
            last, self._ledger_tail, list(attrs),
            lambda name: self._layer_fingerprint(getattr(self, attrs[name])),
        )
        self._boot_reconcile = rec
        lost = rec.get("lost_since_last_save")
        mismatch = rec.get("disk_mismatch")
        if lost:
            logger.warning(
                "启动对账：上次保存(seq=%s, phase=%s)没走完，%d 个层声明要写但磁盘"
                "内容不符（lost_since_last_save）：%s",
                last.get("seq"), last.get("phase"), len(lost), rec["note"],
            )
        elif mismatch:
            logger.warning(
                "启动对账：%d 个层的磁盘内容与账本末条(end seq=%s)不符：%s",
                len(mismatch), last.get("seq"), "、".join(mismatch),
            )

    def _save_layers_core(self) -> LayerSaveResult:
        """只写 L0 / L2 / L3 等**本文件负责的层**（不碰 manager 的 L1 + 向量）。

        **多进程下所有派生层都必须与 L1 同语义**：L1（``memories.json``）走
        「冲突即合并」（加法），而本文件负责的层此前是**整份覆写** ——
        两个进程共用一个库时，谁最后落盘谁的版本就是全库版本，对方写的实体
        挂载 / 会话登记 / 墓碑会被静默写没（实测事故）。无异常、无日志。

        所以这里做**读-合并-写**：落盘前先读磁盘版本，按层口径取并集再写。
        单进程（``multi_writer_enabled()`` 为假）行为与改动前完全一致。

        合并进来的条数记在 :attr:`last_layer_merge`，``stats()`` 里自曝 ——
        "别人的东西被我救回来几条"必须是看得见的数字，不能靠考古。
        编排本体在 :mod:`hippocampus_mcp.store.layer_save`；
        本壳只组装注入面，并把 store 段的产出写回 brain 记账状态 ——
        写盘成功才更新指纹戳/序号（失败即抛，到不了写回）。
        """
        result = save_layers_core(LayerSavePorts(
            persist_dir=self.persist_dir,
            layer_attrs=self._LAYER_ATTRS,
            get_layer=lambda attr: getattr(self, attr),
            set_layer=lambda attr, value: setattr(self, attr, value),
            layer_fingerprint=self._layer_fingerprint,
            layer_changed_by_other=self._layer_changed_by_other,
            read_layer_from_disk=self._read_layer_from_disk,
            read_raw_log_from_disk=self._read_raw_log_from_disk,
            merge_raw_log=self._merge_raw_log,
            merge_del_ledger_layer=self._merge_del_ledger_layer,
            merge_layer=self._merge_layer,
            disk_only_count=self._disk_only_count,
            reclip_derived_layers=self._reclip_derived_layers,
            gc_layer_del=self._gc_layer_del,
            reset_del_blocked=lambda: setattr(self, "_layer_del_blocked", 0),
            backend=self._backend,
            begin_layer_fps=self._begin_layer_fps,
            written_fp=self._layer_written_fp,
            layer_fp_on=lambda: load_config().layer_fp,
        ))
        self.last_save_layers = self._save_layer_status(
            result.persisted, result.layers_skipped)
        if not result.persisted:
            return result
        self.last_layer_merge = result.last_layer_merge
        self._layers_skipped = result.layers_skipped
        self._end_layer_fps = result.end_layer_fps
        self._layer_written_fp.update(result.written_fp)
        self._layer_stamp.update(result.layer_stamp)
        if result.del_gc_last is not None:
            self._layer_del_gc_last = result.del_gc_last
        # F11：全量落盘后 L0 不再欠账
        self._l0_pending = 0
        # 落盘即留名（崩溃退出也有据可查，靠 TTL 兜底清理）
        self._save_seq += 1
        self._touch_writer()
        return result

    def _save_layer_status(self, persisted: bool,
                           skipped: list[str]) -> dict[str, str]:
        """B（验证反馈 2026-11-01）：save 的层级明细（written/skipped）。

        persisted=False = 没有写盘动作发生，如实全标 skipped（不是
        failed——没报错，只是无事可写）。failed 侧在 ``_save_layers`` 的
        异常钩子（全层标 failed，与 saved=false + save_error 配对）。
        saved:true 曾把"内存算了没写盘"盖成静默成功（A 的 bug 因此被
        掩盖），这份明细让每层的真实处置可见，session_end 返回体回显。
        """
        if not persisted:
            return {n: "skipped" for n in self._LAYER_ATTRS}
        dropped = set(skipped or [])
        return {n: ("skipped" if n in dropped else "written")
                for n in self._LAYER_ATTRS}

    def save(self) -> None:
        """持久化：向量库 + LNN（manager）+ 其余层（本文件）。

        G3：sqlite 后端下这两步包在**同一个事务**里 —— L1 与 10 个派生层
        同生共死，不会留下"记忆写了、实体层没写"的夹层状态（JSON 方案在
        文件系统层补不了这个原语，10 个层文件只能逐个 ``os.replace``）。
        json 后端的事务是空实现，行为与改动前完全一致。
        """
        backend = self._backend
        if backend is None:
            self.manager.save()
            self._save_layers()
            return
        with backend.transaction():
            self.manager.save()
            self._save_layers()

    def retry_save(self) -> dict[str, Any]:
        """并发冲突后重试保存：**L1 与 L0 一起补上**。

        为什么不直接用 ``manager.retry_save()``：它只保存 **L1**（memories.json
        + 向量 + LNN），结尾的 ``manager.save()`` 根本不写 ``brain_raw.json``。
        于是"本进程刚 ingest 的一批"会出现「L1 进磁盘了、L0 还在内存里」，进程
        一退原文就丢，L1 的 ``source_ids`` 悬空——追加五「L0 簿记漂移」的第一
        个洞。这里补上 L0 的落盘（``_save_layers`` 内部会与磁盘取并集，不会
        盖掉别的进程写进来的 raw）。

        Returns:
            ``manager.retry_save()`` 的原始统计（added / dropped_local_edits /
            disk_only），另加 ``raw_log_written`` 说明 L0 是否真的落盘了。
        """
        result = self.manager.retry_save()
        result["raw_log_written"] = bool(self.persist_dir)
        self._save_layers()
        return result

    def load(self, auto_backfill: bool | None = None) -> None:
        if not self.persist_dir:
            return
        self.manager.load(
            auto_backfill=self.auto_backfill if auto_backfill is None else auto_backfill
        )
        # 存储损坏不静默：坏文件已隔离，这里把原因吼出来（缺陷 N1）
        issue = self.manager.store.last_load_issue
        if issue:
            logger.error(
                "记忆存储加载异常（内存状态未清空，坏文件已隔离）: %s", issue
            )
        # 🔴 原本这里手写了第二份「层文件 -> 属性」映射，与类属性
        # _LAYER_ATTRS **内容重复**。加删除账本层时踩到：
        # 只在 _LAYER_ATTRS 加了新层，这份没加 ⇒ 新层**从不被载入**，
        # 内存恒为空，多进程拦截整体失效，且唯一症状是启动对账一句
        # "brain_layerdel.json 磁盘内容与账本不符"（极易被当成无关的脏账）。
        # 教训：同一份对应关系只要有第二处手写副本，加层时必漏一处，
        # 而且漏的那处不报错。这里改为直接复用 _LAYER_ATTRS。
        mapping = self._LAYER_ATTRS
        backend = self._backend or get_backend(self.persist_dir)
        for name, attr in mapping.items():
            if not backend.has_layer(name):
                continue
            disk_data = backend.read_layer(name)
            if disk_data is None:
                continue
            # L0 **不能**用磁盘直接替换内存：本进程可能还有没落盘的 raw
            # （ingest 默认不落盘），一替换就丢——这正是追加五「L0 簿记漂移」
            # 的第二个洞（`load_state` 是指南推荐的冲突自救动作，调一次就丢一批）。
            # 多进程模式下按 id 并集；单进程保持原语义（直接替换）。
            if attr == "raw_log" and multi_writer_enabled():
                setattr(self, attr,
                        self._merge_raw_log(disk_data, self.raw_log))
            else:
                setattr(self, attr, disk_data)

        # 实体统计自愈：mention_count 旧实现是累加值（虚高），
        # memory_ids 也可能含已删记忆的悬挂引用 —— 以当前存活记忆重算一次。
        dropped = self._recompute_entity_stats()
        if dropped:
            logger.info("实体索引自愈：摘除 %d 条悬挂引用", dropped)
