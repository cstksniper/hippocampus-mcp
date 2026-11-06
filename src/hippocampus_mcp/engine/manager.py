# -*- coding: utf-8 -*-
"""记忆管理器 (MemoryManager).

将 LNN 动态记忆核、向量存储、打分引擎与抽样器编排为统一的记忆服务.
对外提供 store / recall / get / update / delete / save / load / reset 等操作,
是 MCP 工具层与底层组件之间的桥梁.

架构对应关系（海马体隐喻）:
    - MemoryCore  → CA3: 时序整合, 生成查询向量 h(t)
    - VectorStore → CA1/皮层: 长期记忆存储
    - ScoringEngine → 海马体回忆机制: 相似度 × 时间衰减 × 重要性
    - Sampler     → 探索性回忆: 距离加权抽样避免局部最优
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

from ..config import LNN_CTX_MAX_DEFAULT, load_config  # noqa: F401（LNN_CTX_MAX_DEFAULT 再导出）
from ..common.recency import (
    in_time_window,
    iso_date,
    out_of_window,
    weekday_abbr,
)
from ..store import mem_history as _mem_history
from ..store.idem import IdemIndex, normalize_key


def _stderr_print(*args, **kwargs) -> None:
    """诊断输出一律走 stderr。

    **MCP stdio 模式下 stdout 是 JSON-RPC 协议通道**：写到 stdout 的内容会被客户端
    当成协议消息解析，报 "Invalid JSON" 并可能直接断连。本模块的告警（补算向量、
    加载训练权重失败、元数据回退等）都发生在服务启动路径上，故必须走 stderr。
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)

logger = logging.getLogger(__name__)

from .embeddings import (
    ENV_ALLOW_MODEL_SWITCH,
    backend_kind,
    create_embedder,
    embedder_pool_stats,
    embedding_coverage,
)
from ..lnn import MemoryCore, StateCheckpoint
from ..retrieval import DistanceWeightedSampler, ScoringEngine
# C5 意图路由（默认关）与 C6 检索时衰减（默认关）——见 retrieval/intent.py
from ..retrieval.intent import (
    INTENT_WEIGHT_MULTIPLIERS,
    classify_intent,
    intent_routing_enabled,
)


def _recall_decay_enabled() -> bool:
    """C6 检索时记忆衰减是否开启（默认关；每次读 env 便于测试切换）。

    M1：改走 config 单一读取点（白名单语义，见 config._BOOL_SEMANTICS B）。
    """
    return load_config().recall_decay


def _recall_decay_tau() -> float:
    """C6 衰减时间常数 tau（秒）。默认 90 天，与 lambda_decay 同基准。"""
    tau = load_config().recall_decay_tau
    return tau if tau > 0 else 7776000.0
from ..store import (
    CognitiveType,
    MemoryRecord,
    MemoryState,
    MemoryType,
    VectorStore,
    merge_id_lists,
    merge_tag_dicts,
)


def _seed_store_config() -> None:
    """store/lnn/retrieval 旋钮的生产注入点。

    G1 依赖矩阵只许 engine/brain import config；store/lnn/retrieval 的
    16 个已收编旋钮由**唯一生产构造方**（本模块）在 import 时把
    ``load_config()`` 的字符串运输值灌进各自注入缝（``apply_config``）。
    消费点的 env 直读降级为「直连回退」（env 先、注入值回退）——测试
    直连 store/retrieval 不经本模块的面零改动；解析（float/白名单）
    仍留消费点，零漂移。详见 common/config_bridge 模块 docstring。
    """
    from ..lnn import memory_core as _lnn_mod
    from ..retrieval import intent as _intent_mod
    from ..store import backend as _backend_mod
    from ..store import vector_store as _vs_mod

    c = load_config()
    _vs_mod.apply_config(
        snapshot_min_hours=c.snapshot_min_hours,
        multi_writer=c.multi_writer,
        save_lock_timeout=c.save_lock_timeout,
        matrix_dtype=c.matrix_dtype,
        vectors_mmap=c.vectors_mmap,
        allow_stale_save=c.allow_stale_save,
    )
    _backend_mod.apply_config(store_backend=c.store_backend, fts5=c.fts5)
    _lnn_mod.apply_config(lnn_backend=c.lnn_backend)
    _intent_mod.apply_config(intent_routing=c.intent_routing)


_seed_store_config()


class EmbedderModelMismatchError(RuntimeError):
    """显式配置的嵌入模型与库内已有向量不一致，且未允许切换。

    这是"**服务静默用错模型**"这类事故的硬拦截：与其用错模型
    把整库重算一遍（几千条要几十秒到几分钟），或者更糟——悄悄用另一个模型的
    向量去检索（命中率归零却看不出来），不如明确失败并写清处置办法。

    "库里的模型"由 :meth:`VectorStore.preferred_model_key` 判定：
    优先落盘的 ``model_key``，其次覆盖率最高的非 ``__legacy__`` 槽位。

    确实要在同一份数据上比对不同模型的检索效果时，设
    ``HIPPOCAMPUS_ALLOW_MODEL_SWITCH=1`` 显式放行（等于回到旧的"补算"行为）。
    """


class RecallResult:
    """单条回忆结果的封装."""

    def __init__(
        self,
        memory_id: str,
        content: str,
        score: float,
        similarity: float,
        timestamp: float,
        importance: float,
        recall_count: int,
        memory_type: str,
    ) -> None:
        self.memory_id = memory_id
        self.content = content
        self.score = score
        self.similarity = similarity
        self.timestamp = timestamp
        self.importance = importance
        self.recall_count = recall_count
        self.memory_type = memory_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": round(self.score, 4),
            "similarity": round(self.similarity, 4),
            "timestamp": self.timestamp,
            # 时间派生字段（2026-11-05）：与 brain.recall 的返回体同口径。
            # 生产路径（MCP 工具）走的是 manager.recall，不经 recall_flow，
            # 若不在这里补，宿主拿到的结果就**没有**日期——两条路径的
            # 字段面会分叉，那正是"评测好看、真机没有"的典型成因。
            "occurred_at": iso_date(self.timestamp),
            "weekday": weekday_abbr(self.timestamp),
            "importance": self.importance,
            "recall_count": self.recall_count,
            "memory_type": self.memory_type,
        }


# ==================================================================
# 去重合并前先看"会不会丢硬信息"
# ==================================================================

# 阿拉伯数字（含小数、千分位）。不收百分号——"0.1—0.4%"里 % 是单位不是信息。
_RE_ARABIC = re.compile(r"\d+(?:[.,]\d+)*")
# 中文数字串：**只收长度 >= 2 的连续序列**。单字（"一""二"）在中文里绝大多数是
# 助词/量词（"一样""再一次"），收进来会让"几乎每条都带硬信息"、
# 去重形同虚设——那等于变相关掉去重，是本方案最大的误用风险。
_RE_CN_NUM = re.compile(r"[零〇一二三四五六七八九十百千万亿两]{2,}")
# 日期：2026-09-21 / 2026年9月21日 / 9/21 / 八月中 / 己丑年
_RE_DATE = re.compile(
    r"\d{2,4}\s*[-/年.]\s*\d{1,2}(?:\s*[-/月.]\s*\d{1,4}\s*日?)?"
    r"|[〇零一二三四五六七八九十]{1,3}\s*月(?:\s*[〇零一二三四五六七八九十]{1,3}\s*日)?"
    r"|[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]\s*年?"
)


def extract_hard_info(text: str) -> set[str]:
    """抽出文本里的**可判定硬信息**：数字与日期。

    为什么只取这两类：合并时 ``content`` 取**更长者**，较短的一方被整段丢弃，
    而丢得最痛的恰恰是"旧文本里没有的那个数"——实测案例是
    「燕行日程·约二千余里」被 0.85 阈值吞掉后彻底消失。

    这两类是**可判定的**：有没有新数字，机器能算准；
    而"语义上是不是一条新事实"机器判不准，那属于 supersede / neighbors 的活。
    宁可漏判（少建几条重复）也不能让它静默丢数——丢数是不可恢复的。
    """
    if not text:
        return set()
    out: set[str] = set()
    for m in _RE_ARABIC.finditer(text):
        out.add(m.group(0))
    for m in _RE_CN_NUM.finditer(text):
        out.add(m.group(0))
    for m in _RE_DATE.finditer(text):
        out.add(m.group(0).strip())
    return out


# ==================================================================
# LNN 时序隐状态分路
# ==================================================================

# 没有 client 信息时的兜底路。所有未署名调用共享它 —— 与改造前等价。
DEFAULT_LNN_KEY = "__default__"

# 最多同时保留几路上下文（LRU）。每份约 131 KB（hx + window），8 份 ≈ 1 MB，
# 代价可忽略；超了淘汰**最久未用**的那路，绝不动正在用的。
# M1：常量上移 config.py（L0 底座），此处保留再导出防包外引用断裂。


def lnn_ctx_max_from_env() -> int:
    """读 LNN 分路上限（走 config 单一读取点；>0 判定保持现状）。"""
    v = load_config().lnn_ctx_max
    return v if v > 0 else LNN_CTX_MAX_DEFAULT


class LnnContext:
    """一路 LNN 时序上下文（一位 client 一份）。

    打包了四类**进程级运行时状态**：CfC 隐状态 ``hx``、上一步输出
    ``last_output``、滑窗 ``window``、步数 ``step_count``，外加时间基准
    ``reference_time``。这些是共享连接下会串味的全部东西 —— 记忆 / 实体 /
    画像 / 摘要 / 待办 / 钉住槽本就该跨会话共用，不在其中。
    """

    __slots__ = ("hx", "last_output", "window", "step_count",
                 "reference_time", "window_maxlen")

    def __init__(self, window_maxlen: int | None = None) -> None:
        self.hx = None
        self.last_output = None
        self.window_maxlen = window_maxlen
        self.window = deque(maxlen=window_maxlen)
        self.step_count = 0
        self.reference_time: float | None = None

    def snapshot(self) -> dict:
        """给自曝用的轻量快照（不拷张量，只报形态）。"""
        return {
            "step_count": self.step_count,
            "window_size": len(self.window),
            "has_state": self.hx is not None,
            "reference_time": self.reference_time,
        }

class MemoryManager:
    """记忆管理器: 编排 LNN + 向量库 + 打分 + 抽样.

    Args:
        embedding_dim: 嵌入维度, 默认 384（**仅对 hash 后端生效**；
            走 sentence_transformer 时维度由模型决定，会被实际模型覆盖）。
            使用 hash 后端时需手动指定为 32。
        n_neurons: CfC 神经元数, 默认 512 (需 > output_dim + 2).
        window_size: LNN 滑动窗口, 默认 5.
        persist_dir: 持久化目录路径. 为 None 则不持久化.
        lambda_decay: 时间衰减常数 (秒). 默认 7776000 (90天, LoCoMo调优).
        mu_importance: 重要性衰减常数 (秒). 默认 15552000 (180天, LoCoMo调优).
        embedder_backend: 嵌入后端, "sentence_transformer" (默认) 或 "hash".
        w_sim: 加法公式相似度权重, 默认 0.95.
        w_time: 加法公式时间衰减权重, 默认 0.03.
        w_recall: 加法公式回忆频次权重, 默认 -0.02 (负值=多样性惩罚).
        auto_save: 是否在每次 store 后自动持久化, 默认 False.
        backend: 存储后端. 目前只支持 "numpy"（VectorStore）。
            传其他值会**显式报错**而不是悄悄换一条会丢数据的路径。
            （曾有一份基于 ChromaDB 的实验实现，实测有 6 处会静默损坏数据
            的语义差异，已从代码库移除；原因保留在下方错误信息里，
            供将来重新实现后端时对照。）
        dedup_threshold: 去重合并阈值, 默认 0.85. 置 1.0 关闭去重。
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        n_neurons: int = 512,
        window_size: int = 5,
        persist_dir: str | None = None,
        lambda_decay: float = 7776000.0,
        mu_importance: float = 15552000.0,
        embedder_backend: str = "sentence_transformer",
        embedder_model: str = "",
        w_sim: float = 0.95,
        w_time: float = 0.03,
        w_recall: float = -0.02,
        auto_save: bool = False,
        backend: str = "numpy",
        dedup_threshold: float = 0.85,
        recency_window_days: int = 0,
        w_strength: float = 0.03,
        mem_history_ring: int = 200,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.embedder_backend = embedder_backend
        self.auto_save = auto_save
        self.backend = backend
        self.dedup_threshold = dedup_threshold
        # O8 检索端过时窗口（天）：age > 窗口且未 pin → 不进候选。**默认 0 = 关**
        # ——归档的老记忆也必须能被相似查询查到；输出长度靠 top_k 截断，
        # 不靠时间判死。**零状态变更**——不删、不归档、不改 state，只挡输出。MCP recall 工具经 RecallService 直连本方法（不经
        # brain.recall），窗口必须在这一层生效才全覆盖。
        self._recency_window_seconds = max(int(recency_window_days), 0) * 86400.0
        # 改写历史账本（mem_history.jsonl，sidecar）：ring 构造注入
        # （纪律：store 侧函数只认显式参数）；行数/seq 惰性扫描——
        # 首条写入时读一次盘，纯内存 brain（persist_dir=None）整体旁路。
        self.mem_history_ring = max(int(mem_history_ring), 0)
        self._mem_history_lines: int | None = None
        self._mem_history_last_seq: int = 0

        # 嵌入器: 默认使用 sentence-transformers 语义嵌入
        # 统一走 create_embedder(), 它会校验嵌入器协议（embed/embed_query/
        # embed_batch/model_key）——缺任一方法在初始化就报错，而不是等召回时
        # 才抛 AttributeError（覆盖报告 D5）。
        #
        # 模型名的解析不再在这里分支：把（可能为空的）名字原样交给嵌入器，
        # 由 resolve_embedder_model() 单点决定"参数 > env > 默认值"，并把
        # **来源**记在 embedder.model_source 上。"某个入口忘了设 env 就静默
        # 跑成 MiniLM"正是这么来的，集中解析后不再有第二个地方会写错。
        if embedder_backend == "sentence_transformer":
            self.embedder = create_embedder(
                "sentence_transformer", model_name=embedder_model or ""
            )
            # 实际维度由模型决定, 同步更新
            self.embedding_dim = self.embedder.dim
        else:
            self.embedder = create_embedder(embedder_backend, dim=embedding_dim)

        # 构造参数留档：库内模型与本进程不一致且需"跟随数据"时，要按新维度
        # 重建 LNN 核（见 _follow_store_model）。
        self._ctor_n_neurons = n_neurons
        self._ctor_window_size = window_size
        # 最近一次"跟随数据换模型"的报告（None = 没发生），供上层日志/测试断言
        self.model_follow: dict[str, Any] | None = None

        # LNN 动态记忆核
        # AutoNCP 要求 output_dim < n_neurons - 2
        actual_n_neurons = max(n_neurons, self.embedding_dim + 3)
        self.memory_core = MemoryCore(
            input_dim=self.embedding_dim,
            n_neurons=actual_n_neurons,
            output_dim=self.embedding_dim,
            window_size=window_size,
        )

        # 向量存储：当前唯一受支持的后端是 numpy（VectorStore）。
        # model_key 标识当前活跃嵌入模型: 换模型 = 换向量槽位, 无需重新导入文档
        self._model_key = self.embedder.model_key
        if backend != "numpy":
            raise ValueError(
                f"Storage backend {backend!r} is not wired up. The only supported backend is 'numpy'.\n"
                f"An experimental ChromaDB-based implementation (backend='chroma') once existed; testing showed the following defects, "
                f"so wiring it back in would silently corrupt data -- the implementation was removed and this refusal kept:\n"
                f"  1. after switching embedding models, ChromaDB vectors from the old model get tagged with the current model name; "
                f"missing_ids() returns empty -> never backfilled -> retrieval uses wrong vectors (silent pollution);\n"
                f"  2. save/load round-trips lose source_ids / entity_ids -> the L0 audit chain and entity index break;\n"
                f"  3. update() does not sync pin state (related fixes do not cover that path);\n"
                f"  4. no last_load_issue / corruption quarantine (related fixes do not cover that path);\n"
                f"  5. 构造即自动加载，与其他后端\"显式 load()\"的时序不一致。\n"
                f"Once all five are fixed with dual-backend equivalence regression, this restriction can be lifted."
            )
        store_path = None
        if persist_dir:
            import os
            os.makedirs(persist_dir, exist_ok=True)
            store_path = os.path.join(persist_dir, "memories.json")
        self.store = VectorStore(
            dim=self.embedding_dim,
            persist_path=store_path,
            model_key=self._model_key,
        )

        # 打分引擎: 加法公式 (LoCoMo 评测最优配置)
        self.scorer = ScoringEngine(
            lambda_decay=lambda_decay,
            mu_importance=mu_importance,
            mode="additive",
            w_sim=w_sim,
            w_time=w_time,
            w_recall=w_recall,
            w_strength=w_strength,
        )
        self.sampler = DistanceWeightedSampler(beta=1.0, epsilon=0.1)

        # 持久化
        self.persist_dir = persist_dir
        self._checkpoint: StateCheckpoint | None = None
        if persist_dir:
            import os
            # H7：主格式 npz（零 torch 读写）；旧库的 lnn_state.pt 由
            # StateCheckpoint 兼容读（惰性 torch），无需迁移即可启动。
            ckpt_path = os.path.join(persist_dir, "lnn_state.npz")
            self._checkpoint = StateCheckpoint(ckpt_path)

        # 当前会话已处理的步数（进程级累计，供 auto_save 与 checkpoint 用；
        # 分路之后各路还有自己的 step_count，见 LnnContext）
        self._step_count: int = 0

        # 「L1 落盘前先刷 L0」钩子（R2 / L0 不变式，语义见 :meth:`save`）。
        # 为什么挂在 manager 而不让 engine 认识 L0：auto_save / save_state /
        # retry_save / session_end 四个入口最后都汇到 save()，钩子挂这里
        # 一处覆盖全部路径——按路径逐个补，等于同一个 bug 补四次。
        self._pre_persist: Callable[[], None] | None = None

        # LNN 隐状态分路容器。未署名调用全部落在 DEFAULT_LNN_KEY，
        # 因此**不接线时行为与改造前逐位一致**。
        self._lnn_contexts: dict[str, LnnContext] = {}
        self._lnn_ctx_lru: list[str] = []
        self._lnn_ctx_max = lnn_ctx_max_from_env()

        # 参考时间: 使用已存储消息的最新时间戳, 而非 wall-clock
        # (LoCoMo 评测中发现 time.time() 与数据时间戳差距巨大导致时间衰减失效)
        self._reference_time: float | None = None

        # 自动保存的健康度（zcode 反馈：保存被并发拒绝时宿主毫不知情，
        # 以为 ingest 已落盘，重启后才发现丢了一批）。
        # 见 store_memory() 里的自动保存分支：失败不阻断写入，但必须留痕。
        self.last_save_error: str | None = None
        # C5 自曝：最近一次 recall 的意图判定（路由关闭时恒 None）
        self.last_recall_intent: str | None = None
        self.autosave_failures: int = 0
        # 最近一次 store_memory 是否命中去重（并入了哪条既有记忆），见 store_memory
        self.last_store_merge_of: str | None = None
        # C10：命中去重但**因含新硬信息而未合并**时，那条近重复的既有记忆。
        # 与 last_store_merge_of 互斥：本字段非空 == 本次确实新建了。
        self.last_store_near_miss: dict[str, Any] | None = None
        # C13：本次写入内容的截断自曝（嵌入模型 token 硬上限）。
        # truncated 为 True 时，embedded_chars 之后的原文对向量检索不可见。
        self.last_truncation: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # LNN 上下文路由
    # ------------------------------------------------------------------
    def lnn_key(self, client: str | None) -> str:
        """解析本次调用走哪一路。三级：显式 client > 默认路。

        第二级（``ctx.session``：daemon 下每条 TCP 连接一个 ServerSession）
        由 server 层接线时解析后作为 ``client`` 传进来，manager 只认
        显式值 —— 协议层拿不到会话，不该在这里猜。
        """
        c = (client or "").strip()
        return c or DEFAULT_LNN_KEY

    def _lnn_ctx(self, client: str | None) -> LnnContext:
        """取（或建）一路上下文，并把它挪到 LRU 末尾（最近使用）。"""
        key = self.lnn_key(client)
        ctx = self._lnn_contexts.get(key)
        if ctx is None:
            ctx = LnnContext(
                getattr(getattr(self.memory_core, "window", None), "maxlen", None)
            )
            self._lnn_contexts[key] = ctx
        if key in self._lnn_ctx_lru:
            self._lnn_ctx_lru.remove(key)
        self._lnn_ctx_lru.append(key)
        # LRU 淘汰：只淘汰**最久未用**且当前没在路上的
        while len(self._lnn_contexts) > self._lnn_ctx_max and len(self._lnn_ctx_lru) > 1:
            victim = self._lnn_ctx_lru.pop(0)
            self._lnn_contexts.pop(victim, None)
        return ctx

    @contextmanager
    def _lnn_route(self, client: str | None):
        """把某一路的隐状态装进 memory_core，退出时原样取回。

        ``MemoryCore`` 是一个**单例**（权重只有一份，本来就该共享），
        只有隐状态是需要分路的运行时数据 —— 所以这里做的是"换状态"，
        不是"换模型"。用上下文管理器保证异常路径也能把状态还回去。
        """
        ctx = self._lnn_ctx(client)
        core = self.memory_core
        saved = (core.hx, core.last_output, core.window, self._reference_time)
        core.hx = ctx.hx
        if ctx.last_output is not None:
            core.last_output = ctx.last_output
        core.window = ctx.window
        self._reference_time = ctx.reference_time
        try:
            yield ctx
        finally:
            ctx.hx = core.hx
            ctx.last_output = core.last_output
            ctx.window = core.window
            ctx.reference_time = self._reference_time
            core.hx, core.last_output, core.window, self._reference_time = saved
            self._touch_global_reference_time()

    def _touch_global_reference_time(self) -> None:
        """把进程级 ``_reference_time`` 对齐为各路的最大值。

        consolidate / save / load 读的是进程级那一份；分路后它仍应是"见过的最新
        时间戳"，否则老记忆会永久豁免归档（历史故障，见架构文档 §八 4）。
        """
        vals = [c.reference_time for c in self._lnn_contexts.values()
                if c.reference_time is not None]
        if vals:
            mx = max(vals)
            if self._reference_time is None or mx > self._reference_time:
                self._reference_time = mx

    def lnn_context_report(self) -> dict:
        """自曝：当前有几路上下文、各自走到第几步。"""
        return {
            "count": len(self._lnn_contexts),
            "max": self._lnn_ctx_max,
            "keys": {k: self._lnn_contexts[k].snapshot()
                     for k in self._lnn_ctx_lru},
        }

    # ---------------------------------------------------------------- 幂等索引
    def _idem_index(self) -> IdemIndex:
        """取幂等倒排（首次用到时从内存记录重建，**不另开盘**）。

        索引是 ``MemoryRecord.idem`` 的派生物：进程重启后第一次写带键的
        请求会扫一遍记录。万条量级一次线性扫描 ~ms 级，且只在真用了幂等
        键的库上发生——不用这个特性时**零成本、零内存**。
        """
        idx = getattr(self, "_idem", None)
        if idx is None:
            idx = IdemIndex()
            # M3 端口纪律：走 iter_records，不摸 _records
            idx.rebuild(self.store.iter_records())
            self._idem = idx
        return idx

    def _remember_idem(self, idem_key: str, memory_id: str) -> None:
        """写入成功后登记幂等映射（空键 / 空 id 一律不动）。"""
        if idem_key and memory_id:
            self._idem_index().remember(idem_key, memory_id)

    def idem_lookup(self, key: str) -> str | None:
        """幂等键查询：返回既存记忆 id，未命中 / 悬空一律 ``None``。

        供 ``brain.ingest`` / ``WriteService`` 在**写之前**判定（写前判定
        才能连 L0 都不留）。悬空回执（记录已被 ``forget()`` 真删除）在这里
        自动摘除——返回一个库里不存在的 id 比重复写入更坏：宿主会以为
        记忆还在。
        """
        if not key:
            return None
        idx = self._idem_index()
        hit = idx.lookup(key)
        if not hit:
            return None
        if self.store.get(hit) is not None:
            return hit
        idx.forget(key)
        return None

    def store_memory(
        self,
        content: str,
        importance: float = 0.5,
        memory_type: str = "general",
        timestamp: float | None = None,
        cognitive_type: str = "world",
        source_ids: list[str] | None = None,
        entity_ids: list[str] | None = None,
        state: str = "active",
        source: str = "",
        tags: dict[str, Any] | None = None,
        client: str | None = None,
        quarantine: bool = False,
        sensitivity: str = "",
        origin: dict[str, str] | None = None,
        idem_key: str = "",
    ) -> str:
        """存储一条新记忆.

        流程: 文本 → 嵌入 → LNN step (更新时序状态) → 写入向量库.

        Args:
            content: 记忆文本内容.
            importance: 重要程度 [0, 1], 默认 0.5.
            memory_type: 记忆类型 (episodic/semantic/procedural/general).
            timestamp: 自定义时间戳; None 则使用当前时间.
            cognitive_type: 认知类型 (experience/world/opinion), v0.3 新增.
            source_ids: L0 原始层回指 id 列表, v0.3 新增.
            entity_ids: 关联实体 id 列表, v0.3 新增.
            state: 生命周期状态 (active/superseded/archived), v0.3 新增.
            source: 来源标签（文件名/相对路径），透传到 recall 命中结果，
                让"这条记忆出自哪个文档"可回查（覆盖报告·高 3）。
            tags: 结构化标签，如 ``{"章": 43, "年": "己丑"}``，供
                recall/get_memory_context 按条件过滤。命中去重合并时**取并集**
                （与 source_ids/entity_ids 同一原则）：并进来的内容可能带不同
                章的标签，丢掉就再也按章查不到这条混合记录了。
            quarantine: D2a 写侧隔离标记（默认 False = 现状）。True = 待审：
                不进召回池、不进 user_scope 槽，drift_audit 过目后经受管
                通道转正。判据单一真相在 domain/write_quarantine.should_quarantine，
                本层只存标记不做判定。
            sensitivity: D1 隐私打标（默认空串 = 未打）。"personal" /
                "sensitive:类别"，仅随卡存储与回显，本层不消费。

        Returns:
            记忆的唯一 ID。**注意**：若命中去重（与某条 active 记忆相似度
            >= ``dedup_threshold``），本次不会新建，而是合并进那条记忆并返回
            它的 id——此时 ``source_ids`` / ``entity_ids`` 取并集，
            ``content`` 取更长者、``timestamp`` 取最新、``importance`` 取最大，
            类型字段仅当原值为默认值时才被新值替换。
        """
        if timestamp is None:
            timestamp = time.time()

        # 更新参考时间 (用于 recall 时的打分)
        # 按本路记录；进程级全局仍取各路最大值（_touch_global_reference_time），
        # 所以 consolidate / save 的口径与改造前一致。
        _ctx = self._lnn_ctx(client)
        if _ctx.reference_time is None or timestamp > _ctx.reference_time:
            _ctx.reference_time = timestamp
            self._touch_global_reference_time()

        # 本次调用是否命中去重、并入了哪条既有记忆。
        # 返回值保持"只有 id"的旧签名不变，宿主（ingest）需要区分"新建 / 并入"
        # 时读这个属性——双写场景下"我以为写了新条，其实并进了旧条"必须能看到。
        self.last_store_merge_of: str | None = None
        self.last_store_near_miss = None

        # P0-2 幂等：宿主声明的「同一次请求」重放 ⇒ 不嵌入、不写、不新增。
        # 查表放在嵌入**之前**是刻意的：重放本该省掉的正是最贵的那一步。
        self.last_idem_hit: str | None = None
        if idem_key:
            _hit = self.idem_lookup(idem_key)
            if _hit:
                self.last_idem_hit = _hit
                return _hit

        # 生成嵌入
        emb = self.embedder.embed(content)

        # C13：截断自曝。嵌入模型有 token 硬上限（BGE / MiniLM = 512），
        # 超限部分被静默丢弃、对向量零贡献：写入成功、回读在库、stats 正常，
        # **只有检索悄悄差一截**。这里把「实际纳入嵌入多少字」显式算出来，
        # 由 ingest / store_memory 回显给调用方。
        self.last_truncation = embedding_coverage(self.embedder, content)

        # Phase 5 步骤5: 记忆去重 — 相似度 > 阈值时合并而非新建
        if self.dedup_threshold < 1.0 and self.store.count() > 0:
            candidates = self.store.search(emb, top_k=1)
            if candidates:
                existing, sim = candidates[0]
                # 只合并到**仍有效**的记忆。已作废 / 已归档的记录语义上不是
                # "同一条现行事实"，把新写入吞进去会让它一进来就隐身；
                # 那种情况应当新建，再交给上层的事实更新检测去判定取代关系。
                # ---- 合并丢信息守卫----------------------------------------
                # 合并只保留「更长者」——较短的新文本被**整段丢弃**。若新文本
                # 带来旧文本没有的数字/日期，这样的丢弃是不可恢复的信息损失，
                # 因此不合并、独立新建，并把近重复提示留在 last_store_near_miss。
                _is_dup = (
                    sim >= self.dedup_threshold
                    and existing.state == MemoryState.ACTIVE.value
                    # D2a：待审写入**绝不合并**——合并会把未经审查的内容
                    # 并进一张可见卡（quarantine 标记随合并丢弃），等于
                    # 绕过待审。隔离卡一律独立新建，由 drift_audit 复审。
                    and not quarantine
                )
                _missing_hard: set[str] = (
                    extract_hard_info(content) - extract_hard_info(existing.content)
                ) if _is_dup else set()
                if _is_dup and _missing_hard:
                    self.last_store_near_miss = {
                        "memory_id": existing.id,
                        "similarity": round(float(sim), 4),
                        "missing_hard_info": sorted(_missing_hard)[:12],
                        "note": (
                            "the new text contains numbers/dates absent from the old text -> not merged per C10; "
                            "created as a separate record (merging would lose that information)"
                        ),
                        "content": existing.content[:200],
                    }
                if _is_dup and not _missing_hard:
                    merged_content = (
                        content if len(content) > len(existing.content)
                        else existing.content
                    )
                    # 被吞文本必须在 store.update **之前**取——update
                    # 原地改 existing，之后读 existing.content 拿到的已是
                    # 幸存文本，"谁被吞了"就再也算不出来了。
                    discarded_content = (
                        existing.content
                        if merged_content == content else content
                    )
                    merged_importance = max(importance, existing.importance)
                    if merged_content != existing.content:
                        merged_emb = self.embedder.embed(merged_content)
                    else:
                        # 内容维持**旧文本** → 必须沿用旧文本的向量。
                        # 此前这里赋的是 emb（本次新文本的向量），于是
                        # 「内容没变、向量却换了」：记录与它的向量脱钩，
                        # 写入成功、回读在库、stats 正常，只有检索悄悄变差。
                        # 实测后果：两张内容不同的卡落盘向量逐元素相同
                        # （cos=1.000000），其中一条与自身文本只有 0.8908。
                        merged_emb = self.vector_of(existing)
                        if merged_emb is None:
                            # 旧记录在当前模型槽位下没有向量（不该发生）；
                            # 兜底沿用本次的，总好过把记录写成无向量。
                            merged_emb = emb
                    self.store.update(
                        existing.id,
                        content=merged_content,
                        embedding=merged_emb,
                        timestamp=max(timestamp, existing.timestamp),
                        importance=merged_importance,
                        # 缺陷 D3：下面四项此前**被整体丢弃**。后果是
                        #   ① 本次的 L0 原文失去 source_ids 回指，成为永久孤儿
                        #      （"一字不改可回指核对"的审计链静默断裂，且不会被
                        #       forget 的级联清理覆盖到）
                        #   ② 本次提到的实体与这条记忆失去关联，实体倒排索引断裂
                        # 因此合并必须把来源与索引一并带过去（有序并集，非覆盖）。
                        source_ids=merge_id_lists(existing.source_ids, source_ids),
                        entity_ids=merge_id_lists(existing.entity_ids, entity_ids),
                        # 类型取"更具体"的一方：原值仍是默认值时才采纳新值，
                        # 避免后续一次粗分类把已判定的具体类型冲掉。
                        memory_type=(
                            existing.memory_type
                            if existing.memory_type != MemoryType.GENERAL.value
                            else memory_type
                        ),
                        cognitive_type=(
                            existing.cognitive_type
                            if existing.cognitive_type != CognitiveType.WORLD.value
                            else cognitive_type
                        ),
                        # 来源标签同样不能吞：合并进来的内容可能来自另一个文件，
                        # 丢掉 source 就再也查不出这条记忆是哪些文件的混合体。
                        # 单值字段取"先有的那个，空则用新的"。
                        source=existing.source or source,
                        # 标签同样取并集：并进来的内容可能带别的章/年标签，
                        # 覆盖会让这条混合记录在"按章过滤"下彻底查不到。
                        tags=merge_tag_dicts(existing.tags, tags),
                        # O1：合并保留双方检索强度的 max——新轨迹恒为 0，
                        # 等价于保留旧痕迹的巩固成果，合并不丢强度。
                        recall_strength=max(
                            float(existing.recall_strength or 0.0), 0.0),
                    )
                    # 合并留前值：被吞一方的文本入账（两态都记——
                    # "新长旧短"吞旧文本、"旧长新短"吞新文本；两文本完全
                    # 相同不记，没有任何东西丢失）。after 取幸存文本。
                    if discarded_content != merged_content:
                        self._mem_history_append({
                            "kind": "merge",
                            "memory_id": existing.id,
                            "before": {"content": discarded_content},
                            "after": {"content": merged_content},
                        })
                    self.last_store_merge_of = existing.id
                    # 幂等键照样登记：合并路径也算「这次请求有了回执」，
                    # 否则重放会再走一遍全量匹配（结果相同但白付一次嵌入）。
                    self._remember_idem(idem_key, existing.id)
                    return existing.id

        # LNN step: 更新隐藏状态 (时序记忆)
        # 推进的是**本路**的隐状态。共享连接下若不分路，A 的写入会
        # 直接改写 B 的检索起点 —— 无异常、无日志，只掉时序分。
        # H7：MemoryCore 收 numpy（后端内部负责 dtype/形状），进程零 torch。
        emb_input = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        with self._lnn_route(client) as _rctx:
            self.memory_core.step(emb_input)
            _rctx.step_count += 1
        self._step_count += 1

        # 写入向量库（向量挂在当前模型槽位下）
        record = MemoryRecord(
            id="",
            content=content,
            timestamp=timestamp,
            embeddings={self._model_key: emb.tolist()},
            importance=importance,
            recall_count=0,
            memory_type=memory_type,
            cognitive_type=cognitive_type,
            source_ids=list(source_ids) if source_ids else [],
            entity_ids=list(entity_ids) if entity_ids else [],
            state=state,
            source=source or "",
            tags=dict(tags or {}),
            quarantine=bool(quarantine),
            sensitivity=str(sensitivity or ""),
            origin=dict(origin or {}),
            idem=normalize_key(idem_key),
        )
        memory_id = self.store.add(record)
        self._remember_idem(idem_key, memory_id)

        # 自动持久化 (可选)
        if self.auto_save and self._step_count % 10 == 0:
            try:
                self.save()
                self.last_save_error = None
            except Exception as e:
                # 不阻断主流程——但**绝不能静默**：
                # 并发写保护拒绝保存时，这批记忆只在内存里。宿主看到 ingest 成功
                # 就以为落盘了，进程一重启全丢，且没有任何痕迹可查（zcode 反馈）。
                # 这里记下来，由 get_stats / save_state 显式暴露出去。
                self.last_save_error = f"{type(e).__name__}: {e}"
                self.autosave_failures += 1
                logger.warning(
                    "auto-save failed (the batch of memories is still in memory; call save_state or "
                    "retry_save explicitly to retry): %s", self.last_save_error,
                )

        return memory_id

    def recall(
        self,
        query: str | None = None,
        top_k: int = 5,
        use_lnn: bool = True,
        sample: bool = False,
        allow_ids: set[str] | None = None,
        client: str | None = None,
        occurred_after: float = 0.0,
        occurred_before: float = 0.0,
    ) -> list[RecallResult]:
        """回忆与查询相关的记忆.

        流程:
            1. 生成查询向量 (使用 LNN 的 h(t) 或纯文本嵌入).
            2. 向量库检索 top_k * 3 候选.
            3. 打分引擎综合排序.
            4. 可选: 距离加权抽样引入探索性.

        Args:
            query: 查询文本. 为 None 时使用 LNN 当前状态作为隐式查询.
            top_k: 返回的记忆数量.
            use_lnn: 是否使用 LNN 混合查询向量 (False 则纯文本嵌入).
            sample: 是否使用抽样 (True) 或贪心排序 (False).
            allow_ids: 只在这个 id 白名单里检索（标签过滤在检索**前**收窄
                候选集）。None = 不限。空集合 = 直接返回空。
        """
        if self.store.count() == 0:
            return []
        if allow_ids is not None and not allow_ids:
            return []

        # 本次检索读的是**本路**的时序状态（与写入侧同一 key）
        window_active = occurred_after > 0 or occurred_before > 0
        with self._lnn_route(client) as _rctx:
            # 生成查询向量
            if query is not None:
                # 查询侧: BGE 等模型需要指令前缀 (embed_query 对非 BGE 模型等价于 embed)
                query_emb = self.embedder.embed_query(query)
                if use_lnn:
                    # 混合: LNN 查询向量与文本嵌入的加权融合
                    lnn_query = self.memory_core.get_query()
                    lnn_np = np.asarray(lnn_query, dtype=np.float64)
                    # 加权: 0.6 * 文本嵌入 + 0.4 * LNN 状态
                    query_vec = 0.6 * query_emb + 0.4 * lnn_np
                    norm = np.linalg.norm(query_vec)
                    if norm > 1e-12:
                        query_vec = query_vec / norm
                else:
                    query_vec = query_emb
            else:
                # 无显式查询, 使用 LNN 当前状态
                lnn_query = self.memory_core.get_query()
                query_vec = np.asarray(lnn_query, dtype=np.float64)

            # 检索候选 (多取一些用于打分)
            # 标签过滤时上限改用白名单规模：白名单可能只有几十条，
            # 沿用全库条数只会白算一堆被屏蔽的行。
            pool_size = self.store.count() if allow_ids is None else len(allow_ids)
            # 时间窗生效时把候选池放大 10 倍：过滤发生在检索**之后**，
            # 若池子仍是 top_k*3，窗口内可用的条目可能凑不满 top_k ——
            # 那是"过滤太狠"，不是"库里没有"。先多捞再按时间裁。
            candidate_k = min(max(top_k * (10 if window_active else 3), top_k),
                              pool_size)
            candidates = self.store.search(query_vec, top_k=candidate_k, allow_ids=allow_ids)

            if not candidates:
                return []

            # 准备打分输入
            # 使用参考时间 (已存储消息的最新时间戳), 而非 wall-clock
            # (LoCoMo 评测关键修复: time.time() 与数据时间戳差距巨大导致时间衰减失效)
            current_time = self._reference_time if self._reference_time else time.time()

        # O8 检索端过时窗口：age > 窗口且未 pin → 不进候选。放打分前——被挡
        # 条目不计分、不进 rank、不加 recall_count（对被挡条目零状态写入）。
        if self._recency_window_seconds > 0 or window_active:
            candidates = [
                (r, s) for r, s in candidates
                # O8：系统自动挡（对 timestamp<=0 保守保留，见 recency 注释）。
                # **用户显式给了时间窗时让位**：O8 是"默认策略"，时间窗是
                # "显式意图"，两者冲突时显式优先——否则「查三年前那次事故」
                # 会被 180 天窗口先挡掉，时间窗形同虚设。
                if (self._recency_window_seconds <= 0 or window_active
                    or not out_of_window(r.timestamp, r.pin, current_time,
                                         self._recency_window_seconds))
                # 用户显式时间窗（对 timestamp<=0 排除，理由见 in_time_window）
                and (not window_active
                     or in_time_window(r.timestamp, occurred_after,
                                       occurred_before))
            ]
            if not candidates:
                return []
        similarities = np.array([sim for _, sim in candidates])
        timestamps = np.array([r.timestamp for r, _ in candidates])
        recall_counts = np.array([r.recall_count for r, _ in candidates])
        # O1 检索强度（结果集语义的累积量，见 reinforce/stamp_recalled）
        strengths = np.array([r.recall_strength for r, _ in candidates])

        # 综合打分
        # C5（默认关，HIPPOCAMPUS_INTENT_ROUTING=1 开）：按查询意图对权重乘子
        # 路由；关闭时连分类都不调，行为逐位不变。
        intent = None
        multipliers = None
        if intent_routing_enabled() and query:
            intent = classify_intent(query)
            multipliers = INTENT_WEIGHT_MULTIPLIERS.get(intent) or None
        scores = self.scorer.score(
            similarities, timestamps, recall_counts, current_time,
            weight_multipliers=multipliers,
            strengths=strengths,
        )
        # C6（默认关，HIPPOCAMPUS_RECALL_DECAY=1 开）：检索时记忆衰减——
        # 对综合分做乘性 recency 重加权 exp(-age/tau)。加法公式里的时间项
        # 权重太小（~0.03），改权重是噪声（C2 扫描已证）；乘性重加权能把
        # 「最近发生的」整体从排序深处抬上来。关闭时不进乘法，逐位不变。
        if _recall_decay_enabled():
            age = np.maximum(current_time - timestamps, 0.0)
            scores = scores * np.exp(-age / _recall_decay_tau())
        self.last_recall_intent = intent

        if sample:
            # 距离加权抽样
            chosen_idx = self.sampler.sample(similarities, k=top_k)
            ranked = [(i, scores[i]) for i in chosen_idx]
            ranked.sort(key=lambda x: -x[1])
        else:
            # 贪心排序
            ranked = self.scorer.rank(scores, top_k=top_k)

        # 构建结果
        results: list[RecallResult] = []
        for idx, score in ranked:
            record, sim = candidates[idx]
            results.append(
                RecallResult(
                    memory_id=record.id,
                    content=record.content,
                    score=score,
                    similarity=sim,
                    timestamp=record.timestamp,
                    importance=record.importance,
                    recall_count=record.recall_count,
                    memory_type=record.memory_type,
                )
            )

            # 增加回忆计数。
            #
            # 注意**这里不写 last_recalled_at**：本方法返回的是**候选集**
            # （brain.recall 以 fetch_k = max(top_k*4, 12) 取候选，之后还要过
            # 硬门槛 / MMR / 标签过滤），在这里打时间戳会把"只是被取出来看过
            # 一眼、最终没返回给宿主"的记录也算成"被召回过"——沉寂归档
            # （consolidate 以 last_recalled_at 为时间基准）因此被稀释：
            # 每次查询都会给十来条记录续命。真正的"被召回"时间戳由
            # **brain.recall 在最终结果上写**（见该方法末尾）。
            self.store.update(record.id, recall_count=record.recall_count + 1)

        return results

    def stamp_recalled(self, memory_ids) -> int:
        """给"真正返回给宿主的条目"盖上"最后一次被召回"的时间戳。

        为什么单独开一个方法而不是写在 :meth:`search` 里：``search`` 返回的是
        **候选集**（上层还会过硬门槛 / MMR / 标签过滤），在候选阶段打戳等于
        "每次查询给十来条根本没露面的记忆续命"，沉寂归档（``consolidate`` 以
        ``last_recalled_at`` 为时间基准）会被稀释。

        时间基准与 ``consolidate`` 的 ``now`` 必须一致（``_reference_time or
        wall-clock``）：LoCoMo 评测的数据时间戳可能是 2023 年，若写裸
        ``time.time()``，老记忆会被判成"刚刚被召回"→ **永久豁免沉寂归档**。

        Returns:
            实际打戳成功的条数。
        """
        stamp = self._reference_time if self._reference_time else time.time()
        n = 0
        for mid in memory_ids:
            if not mid:
                continue
            rec = self.store.get(mid)
            if rec is None:
                continue
            # O1：召回即巩固——真实召回（进结果集）在盖时间戳的同时给
            # 检索强度 +1。与 recall_count（候选集、多样性惩罚）分工见
            # MemoryRecord.recall_strength 注释。
            new_strength = max(float(rec.recall_strength or 0.0), 0.0) + 1.0
            if self.store.update(mid, last_recalled_at=stamp,
                                 recall_strength=new_strength):
                n += 1
        return n

    def reinforce(self, memory_ids, amount: float = 1.0) -> int:
        """只累加检索强度，不碰 last_recalled_at（睡眠重放的接缝）。

        分工纪律（约束 4/5 的延伸）：``stamp_recalled`` ＝ 真实召回（进
        结果集）——盖时间戳 + 强度 +1；本方法 ＝ **非召回的强化**（睡眠批的
        睡眠重放共激活）——只加强度。重放不是"被宿主想起过"，若盖了
        last_recalled_at，沉寂归档就会被重放稀释成永生。
        """
        n = 0
        for mid in memory_ids:
            if not mid:
                continue
            rec = self.store.get(mid)
            if rec is None:
                continue
            new_strength = max(float(rec.recall_strength or 0.0), 0.0) + float(amount)
            if self.store.update(mid, recall_strength=new_strength):
                n += 1
        return n

    # ------------------------------------------------------------------
    # 改写历史账本（mem_history.jsonl）
    # ------------------------------------------------------------------
    def _mem_history_append(self, entry: dict[str, Any]) -> None:
        """追加一条改写历史（纯旁路——任何失败都不阻断主写路径）。

        行数/seq 惰性初始化（首条写入时扫一次盘）；纯内存 brain 直接放弃。
        写失败由 store.mem_history 落 warning（"改之前是什么从此无处可查"
        必须让运维看见），本方法不重复记、不抛出。
        """
        if not self.persist_dir:
            return
        if self._mem_history_lines is None:
            self._mem_history_lines, self._mem_history_last_seq = (
                _mem_history.scan_history(self.persist_dir))
        _op_id, self._mem_history_last_seq, self._mem_history_lines = (
            _mem_history.append_history(
                self.persist_dir, entry,
                self._mem_history_last_seq, self._mem_history_lines,
                ring=self.mem_history_ring, trim_at=self.mem_history_ring))

    def memory_history(self, memory_id: str | None = None,
                       limit: int = 20) -> list[dict[str, Any]]:
        """读改写历史（最新在前）。``memory_id`` 过滤单条；纯内存
        brain 返回空。账本是审计面：forget 级联**不**清理它（墓碑拦的
        是复活不是历史），已删记录的历史条目自然 ring 老化。"""
        entries = _mem_history.read_history(self.persist_dir)
        if memory_id:
            entries = [e for e in entries if e.get("memory_id") == memory_id]
        entries.reverse()
        return entries[:max(int(limit), 0)]

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取单条记忆."""
        record = self.store.get(memory_id)
        if record is None:
            return None
        return record.to_dict()

    def update_memory(
        self, memory_id: str, **fields: Any
    ) -> bool:
        """更新记忆字段 (content / importance / memory_type 等).

        如果更新了 content, 会重新生成嵌入并更新 LNN 状态.

        content/importance 变更先把旧值记进
        ``mem_history.jsonl``（sidecar 账本，纯旁路——写失败只 warning
        不阻断主路径）。state **单独**变更不记：supersede/归档路径的旧
        记录本身保留（状态机已承担留痕），记了反是噪声；state 随
        content/importance 一起变时旧值一并入账。embedding 不记（可由
        content 重算）。
        """
        record = self.store.get(memory_id)
        if record is None:
            return False

        # 改写留前值：before＝三字段全量快照（审计价值——改写时的
        # 状态也要可考），after＝被改字段的新值。见方法 docstring 的
        # 记/不记边界。
        changed = [k for k in ("content", "importance") if k in fields]
        if changed:
            self._mem_history_append({
                "kind": "update",
                "memory_id": memory_id,
                "before": {k: getattr(record, k)
                           for k in ("content", "importance", "state")},
                "after": {k: fields[k] for k in changed},
            })

        # 如果更新了 content, 重新嵌入
        if "content" in fields:
            new_content = fields["content"]
            new_emb = self.embedder.embed(new_content)
            fields["embedding"] = new_emb.tolist()

        return self.store.update(memory_id, **fields)

    def delete_memory(self, memory_id: str) -> bool:
        """删除指定记忆（同时摘掉指向它的幂等回执，防悬空 id）。"""
        ok = self.store.delete(memory_id)
        # 索引还没建就不用摘：重建时这条记录本就不在了（零成本旁路）
        if ok and getattr(self, "_idem", None) is not None:
            self._idem.forget_memory(memory_id)
        return ok

    def list_memories(
        self,
        limit: int = 100,
        source: str | None = None,
        state: str | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """列出记忆 (不含嵌入向量).

        Args:
            limit: 最多返回条数.
            source: 只看来自该来源的条目（精确匹配；覆盖报告·高 3 要的
                "列某文件入库了哪些条"）。传 ``""`` 可筛"没有来源"的旧记录。
            state: 只看指定状态（active/superseded/archived）。
            offset: 跳过前若干条（配合 limit 分页审计大库）。
        """
        results = []
        for record in self.store.iter_records():
            if source is not None and (record.source or "") != source:
                continue
            if state is not None and record.state != state:
                continue
            if offset > 0:
                offset -= 1
                continue
            if len(results) >= limit:
                break
            results.append(record.to_dict())
        return results

    def sources(self) -> dict[str, int]:
        """统计各来源的条数：``{source: 条数}``。

        导入脚本用它做"按源文件跳过已导入"的幂等判断（覆盖报告·中 9）。
        没有来源的旧记录归到空字符串键下。
        """
        out: dict[str, int] = {}
        for record in self.store.iter_records():
            key = record.source or ""
            out[key] = out.get(key, 0) + 1
        return out

    def reset_context(self, client: str | None = None,
                      clear_all: bool = False) -> list[str]:
        """重置 LNN 隐藏状态 (开始新的对话上下文).

        默认只清**这一路**。共享进程里"清掉所有人"正是最典型的
        串味事故——A 开新会话，B 的时序上下文跟着归零。要全清必须显式
        ``clear_all=True``（并会在返回值里列出被清的路）。

        Returns:
            被清掉的 key 列表。
        """
        if clear_all:
            keys = list(self._lnn_contexts)
            for c in self._lnn_contexts.values():
                c.hx = None
                c.last_output = None
                c.window.clear()
                c.step_count = 0
                c.reference_time = None
            self._lnn_contexts.clear()
            self._lnn_ctx_lru.clear()
            self.memory_core.reset_state()
            self._step_count = 0
            self._reference_time = None
            return keys

        key = self.lnn_key(client)
        ctx = self._lnn_contexts.pop(key, None)
        if key in self._lnn_ctx_lru:
            self._lnn_ctx_lru.remove(key)
        # 若清的正是当前挂在 core 上的那一路，core 也要归零，
        # 否则下一个读它的人会读到残留。
        self.memory_core.reset_state()
        self._step_count = 0
        self._reference_time = None
        return [key] if ctx is not None else []

    # ------------------------------------------------------------------
    # 多模型向量管理（换模型不重导入）
    # ------------------------------------------------------------------
    @property
    def model_key(self) -> str:
        """当前活跃嵌入模型标识."""
        return self._model_key

    def vector_of(self, record: MemoryRecord) -> np.ndarray | None:
        """取记录在当前活跃模型下的向量；未算过返回 None（需补算）."""
        vec = record.vector_for(self._model_key)
        if vec is None:
            return None
        return np.asarray(vec, dtype=np.float64)

    def ensure_vectors(
        self, batch_size: int = 32, verbose: bool = True
    ) -> dict[str, int]:
        """为当前模型补算缺失向量（不重新导入文档、不调 LLM）.

        换嵌入模型后首次加载会走这条路：文本与结构原样保留，
        只用新模型重算向量。万条规模亦仅需分钟级。

        Returns:
            {"missing": 缺失条数, "filled": 实际补算条数}
        """
        missing = self.store.missing_ids(self._model_key)
        if not missing:
            return {"missing": 0, "filled": 0}

        if verbose and len(missing) >= 50:
            # 大库补算是分钟级操作，而且它在 load() 路径上是**同步阻塞**的：
            # MCP 服务启动预热时会卡在这，客户端可能直接超时断连。
            # 所以先把话说清楚，让人知道"不是在挂，是在算"。
            _stderr_print(
                f"[MemoryManager] model {self._model_key} is missing {len(missing)} vectors; "
                f"backfilling now (synchronous and blocking -- do not interrupt; for large libraries run "
                f"scripts/reembed.py offline first).",
                flush=True,
            )

        filled = 0
        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            texts = []
            ids = []
            for mid in chunk:
                rec = self.store.get(mid)
                if rec is None or not rec.content:
                    continue
                ids.append(mid)
                texts.append(rec.content)
            if not texts:
                continue
            vecs = self.embedder.embed_batch(texts)
            for mid, vec in zip(ids, vecs):
                self.store.set_vector(mid, self._model_key, vec)
                filled += 1
            if verbose:
                _stderr_print(f"  backfilled vectors {filled}/{len(missing)}", flush=True)

        if filled:
            # 只落盘向量库；不动 LNN 检查点（避免在 load 过程中覆盖待恢复的状态）
            self.store.save()
        return {"missing": len(missing), "filled": filled}

    def switch_model(self, model_key: str, dim: int) -> dict[str, int]:
        """切换活跃嵌入模型（供离线实验对比不同模型的检索效果）.

        Args:
            model_key: 目标模型标识.
            dim: 目标模型维度.

        Returns:
            ensure_vectors 的结果（缺失向量自动补算）。
        """
        self.store.set_model(model_key, dim)
        self._model_key = model_key
        self.embedding_dim = dim
        return self.ensure_vectors()

    def get_stats(self) -> dict[str, Any]:
        """返回当前系统状态统计."""
        return {
            "total_memories": self.store.count(),
            "lnn_steps": self._step_count,
            "embedding_dim": self.embedding_dim,
            "embed_model": self._model_key,
            "vectors": self.store.model_stats(),
            "lnn_has_state": self.memory_core.hx is not None,
            "window_size": len(self.memory_core.window),
            # 自曝：有几路上下文、各走到第几步。共享连接下这个数字
            # 应当 > 1；恒为 1 说明调用方根本没传 client（接线未完成）。
            "lnn_contexts": self.lnn_context_report(),
            # H7 自曝：CfC 实际运行时（numpy=零 torch；torch(ncps)=回退路径）。
            # 报**实际值**而非配置值，与 embedder_runtime 同一纪律。
            "lnn_runtime": self.memory_core.runtime,
            "lnn_backend": self.memory_core.backend,
            "scoring_mode": self.scorer.mode,
            "w_sim": self.scorer.w_sim,
            "w_time": self.scorer.w_time,
            "w_recall": self.scorer.w_recall,
            "lambda_decay_days": round(self.scorer.lambda_decay / 86400, 1),
            "mu_importance_days": round(self.scorer.mu_importance / 86400, 1),
            # 自动保存健康度：非 None 说明有整批**只在内存里**，重启即丢
            "last_save_error": self.last_save_error,
            "autosave_failures": self.autosave_failures,
            # B2-4 自曝：实际推理后端。**必须自曝实际值而非配置值**——
            # 配了 onnx 但产物缺失时会自动回退 torch，只看配置永远不知道。
            "embedder_runtime": self.embedder_runtime_backend(),
            "embedder_model_source": getattr(
                self.embedder, "model_source", None
            ),
            "onnx_attached": self.embedder_runtime_backend() == "onnx",
            # F6 自曝：本库的嵌入器是不是复用了进程内共享的那份重对象
            # （False 而池里有货 = 共享没生效，第二库仍在白装一份权重）。
            "embedder_shared": bool(
                getattr(self.embedder, "shared_from_pool", False)
            ),
            "embedder_pool": embedder_pool_stats(),
            # C5/C6 自曝（B2-4 范式：开关状态要能从 stats 里看见）
            "intent_routing_enabled": intent_routing_enabled(),
            "recall_decay_enabled": _recall_decay_enabled(),
            "recall_decay_tau_days": round(_recall_decay_tau() / 86400, 1),
            "last_recall_intent": self.last_recall_intent,
        }

    def embedder_runtime_backend(self) -> str:
        """返回**实际**推理后端名：``onnx`` / ``torch`` / ``hash`` / …

        🔴 为什么不能只报配置值：``HIPPOCAMPUS_EMB_BACKEND=onnx`` 在产物缺失时
        会静默回退 torch（或装配阶段失败被吞），"我以为在用 ONNX 其实没有"
        正是真库上发现过的接线缺陷。这里报的是**挂载结果**。
        有两种 onnx 形态：ST 包装器挂了 ``_onnx``（ST 侧路径），
        或 embedder 本身就是 ``OnnxEmbedder``（follow 直构路径）——都算 onnx。
        """
        e = self.embedder
        if getattr(e, "_onnx", None) is not None or type(e).__name__ == "OnnxEmbedder":
            return "onnx"
        name = type(e).__name__
        if name == "SentenceTransformerEmbedder":
            return "torch"
        return (self.embedder_backend or name).lower()

    def load_trained_weights(self, model_path: str) -> bool:
        """加载训练后的 CfC 权重 (CRL 检索导向训练).

        H7：支持两种来源——``.npz``（``scripts/export_lnn_weights.py`` 的导出物，
        numpy/torch 核通吃，**不需要 torch**）与 legacy ``.pt``（惰性 torch 兜底，
        保持老评测链可用）。推荐先离线跑一次导出，运行期就走零 torch 通道。

        Args:
            model_path: .npz 或 .pt 文件路径.

        Returns:
            加载成功返回 True.
        """
        try:
            if model_path.endswith(".npz"):
                with np.load(model_path, allow_pickle=False) as z:
                    arrays = {
                        k: np.asarray(z[k]) for k in z.files if k != "meta_json"
                    }
                # npz 里可能带 mask 键；load_weights 只认 weight/bias，多余键忽略
                if arrays:
                    self.memory_core.load_weights(arrays)
                    return True
                return False
            # legacy .pt：惰性 torch（仅显式传入 .pt 的场景才付出这份依赖）
            import torch

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            # CRL 训练脚本保存的键带 "cfc." 前缀, 需剥离后加载
            cfc_state = {}
            for k, v in checkpoint.get("model_state", checkpoint).items():
                if k.startswith("cfc."):
                    cfc_state[k[4:]] = v  # 去掉 "cfc." 前缀
                else:
                    cfc_state[k] = v
            if cfc_state:
                self.memory_core.load_weights(cfc_state)
                return True
            return False
        except Exception as e:
            _stderr_print(f"[MemoryManager] failed to load training weights: {e}")
            return False

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    @property
    def meta_path(self) -> str | None:
        """标量元数据侧车文件路径（与 checkpoint 同目录）。"""
        if self._checkpoint is None:
            return None
        return self._checkpoint.path + ".meta.json"

    def save(self) -> None:
        """持久化所有状态: 向量库 + LNN 隐藏状态 + 标量元数据.

        checkpoint 走**带备份**写入（``save_with_backup``）：此前用的是不备份的
        ``save_atomic``，导致 ``StateCheckpoint.load`` 里那段"回退 .bak"分支
        永远没有备份可用——标称的"原子写 + 备份"只兑现了原子写（缺陷 N2）。

        🔴 已知坑（已修）：core 是**单例**，``_lnn_route`` 退出时
        会把进入前快照还原到 core 上 ⇒ 此处直接 ``get_state()`` 存的是**过期**
        状态（新进程里 hx 恒为空，老 torch 路径同样中招——权重本就每次重启
        随机，才没人察觉）。实时状态活在默认路的 ctx 里，从那里取；其余具名
        路由不落盘（历史边界不变：checkpoint 只承载默认会话的时序状态）。
        H7 权重固定化之后 hx 的持久化首次有了真实意义。

        🔴 前置钩子（``_pre_persist``）：在动 L1 **之前**先把待落 L0 刷下去，
        保证「盘上任何一条 L1 的 source_ids，都能在盘上 L0 找到」。顺序是
        有意的：先 L0 后 L1，被杀最多留下「有 raw 无人引用」（无害）；
        先 L1 后 L0 则留下「引用指向不存在的 raw」（不可恢复——实测某库
        近一成条目中招，修复上线后仍在新增）。
        """
        # 契约边界：**钩子抛异常 = 前置失败，但绝不阻断本次落盘**。
        # 阻断等于"连 L1 也不存了"，比"L1 在、L0 缺"更糟；错误留到结尾
        # 重新写回 last_save_error —— 否则末尾那句 `= None` 会把 L0 的失败
        # 静默吞掉，而"静默失败"正是这次事故拖到被外部项目发现的原因。
        pre_error: str | None = None
        hook = self._pre_persist
        if hook is not None:
            try:
                hook()
            except Exception as e:  # noqa: BLE001 —— 见上方契约
                pre_error = f"L0 前置落盘失败：{type(e).__name__}: {e}"
                logger.warning("[save] %s", pre_error)
        self.store.save()
        if self._checkpoint is not None:
            state = self.memory_core.get_state()
            dctx = self._lnn_contexts.get(DEFAULT_LNN_KEY)
            if dctx is not None:
                if dctx.hx is not None:
                    state["hx"] = MemoryCore._state_array(dctx.hx)
                if dctx.last_output is not None:
                    state["last_output"] = MemoryCore._state_array(dctx.last_output)
                state["window"] = [
                    MemoryCore._state_array(t) for t in dctx.window
                ]
            state["step_count"] = self._step_count
            state["reference_time"] = self._reference_time
            self._checkpoint.save_with_backup(state)
            self._save_meta()
        # 落盘成功 → 清掉此前的失败痕迹（保留累计计数，供审计）；
        # 但**前置 L0 失败不清**：那是一笔还没还上的账，得留着让人看见。
        self.last_save_error = pre_error

    def set_pre_persist_hook(self, hook: Callable[[], None] | None) -> None:
        """注册「落盘前先刷 L0」钩子（调用点与语义见 :meth:`save`）。

        为什么用钩子而不是让 manager 直接持有 brain：engine 层不认识 L0
        （它只认识 store / checkpoint）。钩子把「层的先后顺序」这条**业务
        不变式**留在 brain，engine 只负责在正确的时机调用。
        """
        self._pre_persist = hook

    def retry_save(self) -> dict[str, Any]:
        """并发冲突后重试保存：**先取磁盘最新版本，再只补磁盘上没有的增量**。

        为什么不能简单地"再 save() 一次"：``disk_stamp`` 已经说明磁盘被别的
        进程改过，此时直接用内存快照覆写 = 拿陈旧版本盖掉别人的新数据。
        ``retry_save`` 的做法是：

            1. 快照本进程内存里的全部记录（此时它们还没落盘）；
            2. 重新 ``load()`` 磁盘最新版本（以磁盘为准）；
            3. 把"磁盘上没有的 id"补回去 —— 这些才是本进程的真实增量；
            4. 再 ``save()``（此时 stamp 已刷新，是安全写入）。

        语义边界（不自动 merge，只做加法）：
            - 磁盘缺失的 id → 补上（增量，安全）；
            - 双方都有但内容不同 → **保留磁盘版本**，本地改动记入
              ``dropped_local_edits`` 供人工判断（不做字段级合并，避免
              半自动合并出谁也说不清的结果）；
            - 本进程删过、磁盘上还在的记录 → 会"复活"，记入
              ``disk_only``（另一进程新增的 id 也落在这里，二者本进程无法区分）。

        Returns:
            统计字典，含 added / dropped_local_edits / disk_only 等。

        Raises:
            ConcurrentWriteError: 重试期间又被写（说明冲突没停，应当先停掉
                其它写入进程，而不是继续重试）。
        """
        before = {rec.id: rec for rec in self.store.iter_records()}

        # 2) 以磁盘为准重新载入（不补算向量：补算是重活，交给调用方显式触发）
        self.load(auto_backfill=False)
        disk = {rec.id: rec for rec in self.store.iter_records()}

        # 3) 只补磁盘上没有的增量
        added: list[str] = []
        for mid, rec in before.items():
            if mid not in disk:
                self.store.add(rec)
                added.append(mid)

        # 双方都有但内容不同 → 保留磁盘版本，登记本地被丢弃的改动
        dropped: list[str] = []
        for mid, rec in before.items():
            other = disk.get(mid)
            if other is None:
                continue
            if rec.to_dict(with_embeddings=False) != other.to_dict(with_embeddings=False):
                dropped.append(mid)

        disk_only = [mid for mid in disk if mid not in before]

        # 4) stamp 已刷新，这次是安全写入
        self.save()
        return {
            "added": added,
            "added_count": len(added),
            "dropped_local_edits": dropped,
            "dropped_local_edits_count": len(dropped),
            "disk_only": disk_only,
            "disk_only_count": len(disk_only),
            "total": self.store.count(),
        }

    def _save_meta(self) -> None:
        """把标量元数据（``reference_time`` / ``step_count``）单独落 JSON。

        ``.pt`` 里装的是张量，任何损坏都让整份状态不可读；而 ``reference_time``
        是打分用的关键标量——丢了会让 ``recall()`` 退回 ``time.time()``，
        时间衰减**静默失真**（LoCoMo 评测踩过的同一个坑）。单独存一份纯文本
        JSON，即使 ``.pt`` 完全丢失也能把它捞回来。
        """
        path = self.meta_path
        if path is None:
            return
        payload = {
            "reference_time": self._reference_time,
            "step_count": self._step_count,
        }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as e:
            _stderr_print(f"[MemoryManager] metadata save failed: {e}")

    def _load_meta(self) -> dict[str, Any]:
        """读取侧车元数据 JSON；不存在或损坏时返回空字典。"""
        path = self.meta_path
        if path is None or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            _stderr_print(f"[MemoryManager] metadata read failed (ignored): {e}")
            return {}

    def _follow_store_model(self, model_key: str) -> None:
        """把嵌入器换成库内实际使用的模型（"模型跟随数据"的执行部分）。"""
        old_key, old_dim = self._model_key, self.embedding_dim
        backend = self.embedder_backend
        if backend == "sentence_transformer":
            probe = create_embedder(
                "sentence_transformer", model_name=model_key
            )
            # H7：能走 ONNX 就**直构 OnnxEmbedder**——不经 ST 包装器，embedder
            # 本身就是零 torch 实现。probe 只借用它的本地模型目录解析（纯路径
            # 逻辑，构造不 import torch/ST，惰性加载在首次 encode 才发生）。
            direct = None
            if backend_kind() != "torch":
                try:
                    from .onnx_embedder import find_onnx_artifacts, get_shared_onnx

                    model_dir = probe._find_local_model()
                    if model_dir and find_onnx_artifacts(model_dir) is not None:
                        # F6：走共享池 —— 同模型的另一个库已经在跑的话，
                        # 这里拿到的是同一份 session/tokenizer（省 ~400MB
                        # 权重与一次加载），而不是再装一份。
                        direct = get_shared_onnx(
                            model_dir=model_dir, model_name=model_key
                        )
                except Exception as e:  # noqa: BLE001 - 直构失败回落包装路径
                    _stderr_print(f"[MemoryManager] direct ONNX construction failed; falling back to the ST wrapper: {e}")
            if direct is not None:
                self.embedder = direct
            else:
                # H2：模型名**到这一刻才最终确定**——构造时可能只是回落的默认值
                # （如 DEFAULT_MODEL_NAME=all-MiniLM-L6-v2），而库内实际是另一个
                # 模型。ONNX 装配必须在这里再跑一次，否则默认路径下 ONNX 永不生效
                # 且不报错（"我以为在省内存其实没省"）。
                self.embedder = probe
                attach = getattr(self.embedder, "_attach_onnx", None)
                if callable(attach):
                    try:
                        attach(from_store=True)
                    except Exception as e:  # noqa: BLE001 - 装配失败不许让库起不来
                        _stderr_print(
                            f"[MemoryManager] ONNX assembly failed; keeping the torch backend: {e}"
                        )
        else:
            self.embedder = create_embedder(backend, dim=old_dim)
        self.embedding_dim = self.embedder.dim
        self._model_key = self.embedder.model_key
        # LNN 核的输入/输出维度跟着嵌入维度走（AutoNCP 要求 output_dim < n-2）
        self.memory_core = MemoryCore(
            input_dim=self.embedding_dim,
            n_neurons=max(self._ctor_n_neurons, self.embedding_dim + 3),
            output_dim=self.embedding_dim,
            window_size=self._ctor_window_size,
        )
        self.store.set_model(self._model_key, self.embedding_dim)
        self.model_follow = {
            "from": old_key,
            "to": self._model_key,
            "from_dim": old_dim,
            "to_dim": self.embedding_dim,
        }
        _stderr_print(
            f"[MemoryManager] embedding model follows the library's data: {old_key} -> {self._model_key} "
            f"(vectors for this model already exist in the library; no recomputation needed)."
        )

    def _sync_embedder_with_store(self) -> None:
        """加载后校验"生效模型"与"库内向量"是否一致（防静默用错模型）。

        三种情形：

        * **一致**（当前模型在库里有覆盖，或库是空的）→ 什么也不做，正常路径；
        * **不一致，且模型是显式指定的** → 抛 :class:`EmbedderModelMismatchError`。
          配置与数据打架时不要猜；确实要比对模型就显式设
          ``HIPPOCAMPUS_ALLOW_MODEL_SWITCH=1``；
        * **不一致，但模型是回落出来的默认值** → **跟随库内数据**换过去
          （这才是"服务没配 env 就静默跑成 MiniLM"的根治：默认值不再能
          悄悄覆盖一个有明确模型的库）。
        """
        self.model_follow = None
        if self.embedder_backend != "sentence_transformer":
            return
        preferred = self.store.preferred_model_key()
        if not preferred or preferred == self._model_key:
            return
        # 库内模型有覆盖、且不是当前模型 → 冲突
        source = getattr(self.embedder, "model_source", "explicit")
        allowed = load_config().allow_model_switch
        if source == "default" and not allowed:
            self._follow_store_model(preferred)
            return
        if allowed:
            _stderr_print(
                f"[MemoryManager] {ENV_ALLOW_MODEL_SWITCH} enabled: the library was built with "
                f"{preferred}; running with {self._model_key} this time and backfilling missing vectors."
            )
            return
        raise EmbedderModelMismatchError(
            f"embedding model mismatch with the library's data: the effective model is {self._model_key!r} "
            f"(source: {source}), but the vectors in {self.store.persist_path!r} belong to "
            f"{preferred!r}.\n"
            f"Using the wrong model silently destroys retrieval (or triggers a full-library recompute), so loading is refused. Remedies:\n"
            f"  1) make them consistent (recommended): set HIPPOCAMPUS_EMB_MODEL={preferred};\n"
            f"  2) to compare other models on this data deliberately: set "
            f"{ENV_ALLOW_MODEL_SWITCH}=1 (all vectors will be backfilled for the new model)."
        )

    def load(self, auto_backfill: bool = True) -> None:
        """从磁盘恢复所有状态.

        Args:
            auto_backfill: 加载后自动为当前嵌入模型补算缺失向量。
                旧库（换模型前建的、或早期单向量格式）无需重新导入，
                首次加载会自动重新嵌入；不需要时置 False 手动控制。
        """
        self.store.load()
        # 加载后**先对齐模型，再碰任何向量**：库里写着用哪个模型在跑，
        # 就按哪个模型来（或明确报错），绝不拿当前模型硬套（见方法文档）。
        self._sync_embedder_with_store()
        if self._checkpoint is not None and (
            # 主文件缺失但 .bak 尚在（上次保存在替换间隙崩溃）也算有状态可恢复
            self._checkpoint.exists()
            or os.path.isfile(self._checkpoint.bak_path)
        ):
            state = self._checkpoint.load()
            self.memory_core.load_state(state)
            self._step_count = state.get("step_count", 0)
            self._reference_time = state.get("reference_time")

        # .pt 不可用（损坏/丢失）时用侧车 JSON 兜底恢复标量
        meta = self._load_meta()
        if self._reference_time is None and meta.get("reference_time") is not None:
            self._reference_time = meta["reference_time"]
            _stderr_print("[MemoryManager] reference_time restored from metadata JSON (.pt unavailable)")
        if not self._step_count and meta.get("step_count"):
            self._step_count = int(meta["step_count"])

        if auto_backfill:
            stats = self.ensure_vectors()
            if stats["filled"]:
                _stderr_print(
                    f"[MemoryManager] backfilled {self._model_key} vectors for model "
                    f"{stats['filled']}"
                )
