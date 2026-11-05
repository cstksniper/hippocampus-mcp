# -*- coding: utf-8 -*-
"""配置单一来源 —— 全库唯一的 env 读取点，产出不可变 Config。

## 为什么要有这个文件

设计动机：env 读取曾散落在 **23 个文件 / 101 处**：server.py
模块级 18 处 import-time 快照、manager/core/pure 各 4 处函数内读取、
domain/topic.py 4 处、constants 里还有 18 处 ``_env_*`` 模块级求值。
后果是「改一个旋钮要 grep 半个仓库」「测试 patch 了一个副本另一个没变」。

本文件把它们收成一份**显式字段清单**：

* :class:`Config` —— frozen dataclass，每个字段 = 一个 env 旋钮，
  默认值与迁移前逐一比对（``tests/test_config_parity.py`` 钉住）；
* :func:`load_config` —— **唯一**的 ``os.environ`` 读取点。刻意**不缓存**：
  测试与评测脚本靠「改 env → 再调一次」观察行为变化（如
  ``test_c1_source_drift.py`` D6 设 ``HIPPOCAMPUS_SOURCE_SIG=0``），
  缓存会把这种切换变成静默失效。

## 边界（防范围蔓延）

* **只收 server/manager/core/pure 的散点**（32 处）。``common/constants.py``
  的 ``_env_*`` 模块级求值与 ``domain/topic.py`` 的参数化散点由第二批
  收编（见下文对应分组）。config 依赖 common/constants 的默认值表
  （最底层底座），不反向依赖任何上层。
* **不动 101 个 env 的名字与语义**：Config 是读取点的搬迁，不是旋钮的
  重新设计。用户环境变量一个都不用改。

依赖方向：``config`` → ``common.constants``（取默认值常量），无其他依赖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

#: LNN 分路上限默认值（config 是最底层，不能反向依赖 engine；
#: manager 从这里 import）
LNN_CTX_MAX_DEFAULT = 8

# ---------------------------------------------------------------------------
# 类型化 env 读取。**四种 bool 语义并存是现状**（如实复现存量散点），
# 逐一复现、不统一——统一就是行为漂移：
#   A 黑名单默认 true ：「false/0/no」关，其余含未设都开
#       （server.py _AUTO_SAVE/_AUTO_BACKFILL/_SUPERSEDE）
#   B 白名单          ：只有「1/true/yes/on」开，其余含未设都关
#       （manager _recall_decay_enabled、core NER_ALLOW_TORCH_ENV）
#   C 黑名单默认 false：「false/0/no」关，其余含未设……未设关、其他值开
#       （server.py _WARMUP_DIAG，get 默认 "0"）
#   D 非空非 0        ：空串/未设/"0" 关，其余（含 "false"！）开
#       （manager ENV_ALLOW_MODEL_SWITCH）
# ---------------------------------------------------------------------------


def _flag_deny_true(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("false", "0", "no")


def _flag_allow_only(name: str, default: bool = False,
                     with_on: bool = True) -> bool:
    """白名单开关。``with_on=False`` 复现 core.py NER 的窄名单
    （("1","true","yes") 无 "on"——与 manager recall_decay 的宽名单差一个词，
    M1 实测抓到的第二处 parity 裂缝，逐字段复现不统一）。"""
    words = ("1", "true", "yes", "on") if with_on else ("1", "true", "yes")
    return os.environ.get(name, "").strip().lower() in words


def _flag_deny_false(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "0")
    return raw.strip().lower() not in ("false", "0", "no")


def _flag_nonempty(name: str, default: bool = False) -> bool:
    return (os.environ.get(name) or "").strip() not in ("", "0")


def _flag_tristate(name: str, default: bool) -> bool:
    """E 型：两份名单都认，**其余值回落到 default**（不像 A 型那样判开）。

    这条很重要： ``HIPPOCAMPUS_MAP_MENTION=maybe`` 在迁移前是什么结果，
    迁移后还得是什么——脏值的行为也是行为的一部分。
    """
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


def _positive_float(name: str, default: float) -> float:
    """取值必须是正数，否则回落默认（原 ``constants.writer_ttl`` 的 ``v > 0`` 门）。

    为什么保留这道门、而不是把它当成普通的「配错兜底」：TTL 这类旋钮，
    ``0`` 的语义是「立刻过期、谁也看不见」，配出来只会制造一批幽灵写者，
    比不生效更糟——现状选了挡掉，收编照搬。
    """
    v = _float(name, default)
    return v if v > 0 else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """全库配置快照（不可变）。字段 = env 旋钮，默认值 = 迁移前实测值。

    每次 :func:`load_config` 重读 env——这是特性不是遗漏：测试与评测
    依赖「改 env 立刻生效」的语义（见模块 docstring）。
    """

    # --- 启动与库位置（server.py 模块级） ---
    log_level: str = "INFO"
    persist_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".hippocampus_mcp"))

    # --- 检索调参（server.py 模块级，LoCoMo 调优结果） ---
    lambda_decay: float = 7776000.0        # 90 天
    mu_importance: float = 15552000.0      # 180 天
    w_sim: float = 0.95
    w_time: float = 0.03
    w_recall: float = -0.02
    # 检索强度权重：+w·log1p(recall_strength)——成功召回
    # 巩固痕迹（测试效应）。0 = 关（与旧打分逐位一致）。
    w_strength: float = 0.03
    # 改写历史账本 ring：mem_history.jsonl 保留最新 N 条
    # （update/merge 的 before 流水，sidecar 审计面；形态镜像 tag_ops）。
    # 0 的切片语义见 store/mem_history 模块 docstring（[-0:] 全量怪癖，
    # 调用方不传 0）。
    mem_history_ring: int = 200
    nav_penalty: float = 1.0
    recall_decay: bool = False
    recall_decay_tau: float = 7776000.0    # 与 lambda_decay 同基准（90 天）
    library_cache_ttl: float = 30.0

    # --- 模型与嵌入（server.py / core.py / embeddings.ENV_MODEL） ---
    emb_model: str = ""
    crl_model: str = ""
    allow_model_switch: bool = False

    # --- 写入策略（server.py 模块级） ---
    auto_save: bool = True
    auto_backfill: bool = True
    supersede: bool = True

    # --- 实体与块限（pure.py 函数内） ---
    entity_lexicon: str = ""
    daily_activity_days: str = ""
    knowledge_block_limit: str = ""
    pin_block_limit: str = ""

    # --- LNN（manager.py 函数内） ---
    lnn_ctx_max: int = LNN_CTX_MAX_DEFAULT

    # --- 漂移审计（core.py 函数内） ---
    source_root: str = ""
    ner_allow_torch: bool = False

    # --- 预热与握手（server.py 模块级，launcher 协作协议） ---
    warmup: str = "auto"
    warmup_timeout: float = 90.0
    warmup_wait: str = "raise"
    warmup_diag: bool = False
    handshake_window: float = 15.0
    preload: str = "1"
    preload_budget: float = 6.0

    # --- 工具面（server.py 模块级，剖面门控） ---
    tool_profile: str = "common"
    common_tools: str = ""

    # ------------------------------------------------------------------
    # 第二批收编的 28 个旋钮
    # ------------------------------------------------------------------
    # 来源四处：
    #   * common/constants 的**模块级求值**（3 个 ring + 写者 TTL +
    #     三个层开关）
    #   * brain/core.py 的函数内散点（状态卡矛盾信号 + 三条巡检链路）
    #   * brain/session.py 的 5 个预激活槽位开关（三态由来已久）
    #   * domain/topic.py 的 6 个（domain 不许 import config ⇒ 参数化注入）
    # 默认值一律写成**字面量**而不是 import 常量：config 是 L0 纯叶
    # （依赖图 ``_ALLOWED["config"] == {"config"}``），不许反向依赖 common
    # 或 domain —— 两侧同值的等价性由 ``test_config_parity`` 的 P7 钉住。

    # 环形缓冲与写入纪律（原 constants 模块级求值）
    tag_ops_ring: int = 50
    save_ledger_ring: int = 200
    l0_autosave_every: int = 20
    #: 写者心跳过期窗口。**正值门**：解析出 <=0 一律回落默认（现状语义，
    #: 见原 ``constants.writer_ttl`` 的 ``if v > 0`` 分支——它不是配错兜底，
    #: 是「0 秒过期」这种无意义取值要被挡掉）
    writer_ttl: float = 3600.0
    layer_fp: bool = True
    layer_del: bool = True
    source_sig: bool = True

    # 巡检链路与状态卡（原 brain/core.py 函数内散点）
    status_contradictions: bool = True
    map_auto_verify: bool = True
    map_auto_interval_h: float = 6.0
    map_auto_budget_s: float = 1.0
    map_mention: bool = True
    map_mention_interval_h: float = 6.0
    map_mention_budget_s: float = 1.0
    map_mention_max_records: int = 2000
    file_registry_off: bool = False
    file_registry_interval_h: int = 6

    # session_start 预激活槽位（原 _block_switch 的 env 分支；三态语义本身
    # 留在 session.py，Config 只提供 **param==0 时**的答案）
    overview_block: bool = True
    status_block: bool = True
    task_profile_block: bool = True
    map_index_block: bool = True
    map_detail_block: bool = True

    # 主题层（原 domain/topic.py；由调用方注入进纯函数）
    topic_layer: bool = True
    topic_join_sim: float = 0.55
    topic_max: int = 1000
    topic_max_members: int = 60
    topic_summary_topn: int = 3
    topic_low_info: bool = True

    # ------------------------------------------------------------------
    # 用户级记忆继承槽（新字段无迁移前语义，
    # 型别：条数三态 + 阈值档下限）
    # ------------------------------------------------------------------
    #: 用户级继承槽条数。三态表见 ``brain/pure.resolve_user_scope_limit``：
    #: param==0 时看这里——**正 = 条数、负 = 关**、0 / 空 / 非法
    #: 回落默认 3（注意与 pin_limit 的差别：pin 的 env 负值落默认，
    #: user_scope 的 env 负值是显式关）。
    user_scope_limit: int = 3
    #: 用户级任务目录槽条数。三态表在
    #: ``brain/pure.resolve_task_directory_limit``（与 user_scope 同构）：
    #: param==0 时看这里——**正 = 条数、负 = 关**、0 / 空 / 非法回落默认 8。
    task_directory_limit: int = 8
    #: 阈值档下限：非 pinned 条目 importance >= 该值才进继承槽。
    user_scope_importance: float = 0.7
    #: 提升候选稳定期限：≥N 个不同会话复现才成候选。
    promote_min_sessions: int = 2
    # 检索端过时窗口（天）：记忆 age > 窗口且未
    # pin → 不进 recall / get_memory_context 结果集。**零状态变更**：
    # 不删、不归档、不改 state，只挡输出。
    # **默认 0 = 关**：归档的老记忆也必须能被相似查询查到，不能被时间
    # 单方面判死；输出长度由 top_k / limit 控制，排名靠后的自然被截掉。
    # 需要"只看近期"时显式设天数（或查询时传 occurred_after/before）。
    # 时间基准＝_reference_time or 墙钟（约束 4）。
    recency_window_days: int = 0
    #: 敏感词表文件路径：一行一词，``#`` 注释行、空行
    #: 跳过；空 = 无部署词表（**模式级内置兜底仍在**——「任一不过
    #: 不写」纪律的第二道门不依赖词表存在）。
    sensitive_words: str = ""
    #: 逃生门：**False（默认）= 家库写边界生效**
    #: ——写类工具的目标不是本次连接的家库、也不是 user 受管通道时显式报错；
    #: True = 全局放开跨库写（运维搬库 / 批量迁移用）。
    #: 语义类型 B（**白名单**，与 recall_decay 同款）——逃生门必须是"说清楚了
    #: 才开"：空串、脏值（``maybe``）一律落默认 False。不能用 C 型（黑名单）：
    #: C 型的*"不是假就是真"*会让空 env 与拼错的取值把边界悄悄放开。
    cross_library_write: bool = False

    # ------------------------------------------------------------------
    # engine/brain 可直连面收编的 4 个旋钮
    # ------------------------------------------------------------------
    # 字符串运输、语义解析留在调用点（照抄原表达式，零语义漂移）：
    #   * onnx_no_arena 原文 `not in ("1", "true")`——大小写敏感、不含
    #     yes/on，走通用 bool 分派会漂移，故按 str 收；
    #   * emb_model / entity_lexicon 字段早已存在，本批只是杀掉
    #     embeddings.resolve_embedder_model 与 brain/entity 的第二读取点。
    # store/lnn/library/map/retrieval/domain 构造注入面 + 根层入口
    # 的收编见下方对应分组。
    onnx_backend: str = "auto"        # torch/onnx/auto，非法值回落 auto
    onnx_no_arena: str = ""           # "1"/"true" 才关 arena（默认开）
    shared_embedder: str = ""         # 0/false/off/no 才关共享池（默认开）
    emb_cache: str = ""               # 显式嵌入缓存目录
    # 按系统语言自动选「应用默认」嵌入模型（中文→BGE-zh，其余→MiniLM）。
    # 字符串运输、语义解析在 embeddings.auto_model_selection_enabled（照抄
    # shared_embedder 的同款口径：0/false/off/no 才关，未设即开）。
    # **只影响「什么都没配」这条路径**：显式 HIPPOCAMPUS_EMB_MODEL 永远优先。
    emb_model_auto: str = ""
    # 超长内容（超过嵌入模型 token 上限）的处置策略。
    # warn（默认，= 只自曝）/ split（拆成多张卡）/ reject（拒写）。
    # **白名单**：只有明确写 split|reject 才改变行为，其余一律 warn
    # （同款口径：写错等同于没写，绝不让开关被悄悄打开）。
    # 语义解析在 domain.long_content.parse_policy（纯函数，可单测）。
    long_content: str = ""
    # recall 返回体里**单条记忆 content 的字符上限**（截断护栏）。
    # 0 = 不截断（现状，长记忆一次召回能上万字符）。字符串运输，语义解析
    # 在 application/recall_service（`_content_cap`，非数字/负数按 0 处理
    # = 不截断，写错等同于没写）。**截断必须自曝**：被切的条目带
    # content_truncated=true + content_chars，绝不静默切。
    recall_content_max_chars: str = ""

    # ------------------------------------------------------------------
    # store/lnn/library/map/retrieval/domain 构造注入面 16 旋钮收编。
    # 全部**字符串运输**（口径：语义解析
    # 留消费点，零漂移），消费侧（store/lnn/library/map/retrieval/domain）
    # 受 G1 矩阵约束不 import config —— 由唯一生产构造方 **engine**
    # （manager import 时）apply_config 注入；消费点的 env 直读降级为
    # 「直连回退」（env 先、注入值回退——测试直接设 os.environ 的面
    # 原样保住；生产路径 env 未设时默认值单点走这里）。
    # 默认值全部＝迁移前 env 兜底值的原文。
    # ------------------------------------------------------------------
    snapshot_min_hours: str = "6"       # HIPPOCAMPUS_SNAPSHOT_MIN_HOURS
    multi_writer: str = "1"             # HIPPOCAMPUS_MULTI_WRITER
    save_lock_timeout: str = "60"       # HIPPOCAMPUS_SAVE_LOCK_TIMEOUT
    matrix_dtype: str = "fp32"          # HIPPOCAMPUS_MATRIX_DTYPE
    vectors_mmap: str = "0"             # HIPPOCAMPUS_VECTORS_MMAP
    allow_stale_save: str = ""          # HIPPOCAMPUS_ALLOW_STALE_SAVE
    store_backend: str = "json"         # HIPPOCAMPUS_STORE_BACKEND
    fts5: str = ""                      # HIPPOCAMPUS_FTS5
    lnn_backend: str = ""               # HIPPOCAMPUS_LNN_BACKEND
    library_roots: str = ""             # HIPPOCAMPUS_LIBRARY_ROOTS
    #: 默认驻留 2→3（实测容量权衡），此处按 3 落字段。
    max_resident_libs: str = "3"        # HIPPOCAMPUS_MAX_RESIDENT_LIBS
    file_registry_ignore: str = ""      # HIPPOCAMPUS_FILE_REGISTRY_IGNORE
    file_registry_max_files: str = ""   # HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES
    file_registry_budget_s: str = ""    # HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S
    intent_routing: str = "0"           # HIPPOCAMPUS_INTENT_ROUTING
    tag_schema: str = ""                # HIPPOCAMPUS_TAG_SCHEMA

    # ------------------------------------------------------------------
    # daemon 空闲「睡眠窗口」的脑侧旋钮。
    # **总开关不在 config**：``HIPPOCAMPUS_SLEEP_IDLE``（空闲多少秒开睡，
    # 0 = 关）是 daemon 进程拓扑决定，随 model-unload 同款在 daemon.py
    # 直读 env（口径：根层入口属启动拓扑 plumbing）。
    # 这里只收「睡眠批内容」的三个语义旋钮，由 brain.core.sleep_batch
    # 调用时现取（同 source_root / status_contradictions 的先例）。
    # 默认值不动生产行为：睡眠批本身默认关（SLEEP_IDLE=0），这里的
    # 旋钮只在开关打开后才被读。
    # ------------------------------------------------------------------
    #: 重放窗口（天）：只重放「窗口内写入」的 L1 记忆做共激活强化。
    #: 对齐生物事实——SWR 重放的是**近期**经历，不是全库倒带。
    sleep_replay_days: float = 2.0
    #: 单次睡眠批最多多少条 L1 记录参与重放（大库限幅；0 = 不限）。
    sleep_max_replay: int = 200
    #: 成簇下限：实体共现少于该数的记忆不构成重放簇（共激活至少要有
    #: 两个成员才有意义；无实体的记忆不参与）。
    sleep_min_cluster: int = 2

    # ------------------------------------------------------------------
    # 写侧·呈现线两个旋钮（编码门控 / 前瞻触发）
    # ------------------------------------------------------------------
    #: 新颖度/预测误差门控编码。True（默认）= ingest 分句判定时与
    #: 既有图式（L3 画像 / L2T 知识点摘要 / 高置信 L1）比对：印证→不加强
    #: （重复归 dedup 管）、矛盾→importance 提升 + 冲突对象回显、
    #: 全新实体→小幅加权。**门控只调编码强度（importance），不制造 pin**：
    #: base < 0.7 时提升后被钳在 0.69，绝不新跨越 PIN_IMPORTANCE——pin 的
    #: 进退权留给用户/反馈通道（pin 单调堆积是 policies.py 记录在案的教训）。
    #: 显式传入 importance 时门控整体旁路（宿主说了算）。
    #: 语义类型 A（黑名单默认 true）：0/false/no/off 显式关，脏值回落默认开。
    novelty_gate: bool = True
    #: 前瞻记忆情境触发。True（默认）= remember_intent 登记时从文本
    #: 派生触发条件（trigger_entities / trigger_topic），get_memory_context
    #: 的查询实体/实词与活跃待办的触发集相交时，该待办在 block_2 置前回带
    #: （cap 5 不膨胀）。关掉时**双端都关**：登记不再派生触发集、装配不再
    #: 触发回带（存量已带触发集的待办也一律不触发）。语义类型 A 同上。
    prospective_trigger: bool = True
    #: 访问驱动的画像转存。True（默认）= consolidate 刷新画像时把
    #: 「没人钉但反复被想起（检索强度 ≥ 3）且活得够久（≥ 7 天）」的
    #: L1 条目毕业进 L3 独立槽 ``recurring_facts``（展示层追加，不写记录、
    #: 不改状态、不碰用户库——与 U3 跨库提升流的边界见 domain/profile_ops
    #: 模块 docstring）。关掉只停**新增**毕业条目：存量条目的失效摘除
    #: （归档/删除后）仍由 prune 管理，不留僵尸。阈值常量在
    #: common/constants（PROFILE_STRENGTH_*）。语义类型 A（黑名单默认
    #: true）：0/false/no/off 显式关，脏值回落默认开。
    profile_promotion: bool = True

    def as_dict(self) -> dict[str, Any]:
        """字典视图（调试/自曝用；不参与行为）。"""
        return dict(self.__dict__)


#: env 名 → Config 字段名。一处对照表，防「改了 env 忘了字段」。
ENV_MAP: dict[str, str] = {
    "HIPPOCAMPUS_LOG_LEVEL": "log_level",
    "HIPPOCAMPUS_PERSIST_DIR": "persist_dir",
    "HIPPOCAMPUS_LAMBDA_DECAY": "lambda_decay",
    "HIPPOCAMPUS_MU_IMPORTANCE": "mu_importance",
    "HIPPOCAMPUS_W_SIM": "w_sim",
    "HIPPOCAMPUS_W_TIME": "w_time",
    "HIPPOCAMPUS_W_RECALL": "w_recall",
    # --- 检索强度权重（0 = 关）---
    "HIPPOCAMPUS_W_STRENGTH": "w_strength",
    # --- 改写历史账本 ring ---
    "HIPPOCAMPUS_MEM_HISTORY_RING": "mem_history_ring",
    "HIPPOCAMPUS_NAV_PENALTY": "nav_penalty",
    "HIPPOCAMPUS_RECALL_DECAY": "recall_decay",
    "HIPPOCAMPUS_RECALL_DECAY_TAU": "recall_decay_tau",
    "HIPPOCAMPUS_LIBRARY_CACHE_TTL": "library_cache_ttl",
    "HIPPOCAMPUS_EMB_MODEL": "emb_model",
    "HIPPOCAMPUS_CRL_MODEL": "crl_model",
    "HIPPOCAMPUS_ALLOW_MODEL_SWITCH": "allow_model_switch",
    "HIPPOCAMPUS_AUTO_SAVE": "auto_save",
    "HIPPOCAMPUS_AUTO_BACKFILL": "auto_backfill",
    "HIPPOCAMPUS_SUPERSEDE": "supersede",
    "HIPPOCAMPUS_ENTITY_LEXICON": "entity_lexicon",
    "HIPPOCAMPUS_DAILY_ACTIVITY_DAYS": "daily_activity_days",
    "HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT": "knowledge_block_limit",
    "HIPPOCAMPUS_PIN_BLOCK_LIMIT": "pin_block_limit",
    "HIPPOCAMPUS_LNN_CTX_MAX": "lnn_ctx_max",
    "HIPPOCAMPUS_SOURCE_ROOT": "source_root",
    "HIPPOCAMPUS_NER_ALLOW_TORCH": "ner_allow_torch",
    "HIPPOCAMPUS_WARMUP": "warmup",
    "HIPPOCAMPUS_WARMUP_TIMEOUT": "warmup_timeout",
    "HIPPOCAMPUS_WARMUP_WAIT": "warmup_wait",
    "HIPPOCAMPUS_WARMUP_DIAG": "warmup_diag",
    "HIPPOCAMPUS_HANDSHAKE_WINDOW": "handshake_window",
    "HIPPOCAMPUS_PRELOAD": "preload",
    "HIPPOCAMPUS_PRELOAD_BUDGET": "preload_budget",
    "HIPPOCAMPUS_TOOL_PROFILE": "tool_profile",
    "HIPPOCAMPUS_COMMON_TOOLS": "common_tools",

    # --- 检索端过时窗口（天；0 = 关）---
    "HIPPOCAMPUS_RECENCY_WINDOW_DAYS": "recency_window_days",

    # --- 第二批收编 28 个 ---
    "HIPPOCAMPUS_TAG_OPS_RING": "tag_ops_ring",
    "HIPPOCAMPUS_SAVE_LEDGER_RING": "save_ledger_ring",
    "HIPPOCAMPUS_L0_AUTOSAVE_EVERY": "l0_autosave_every",
    "HIPPOCAMPUS_WRITER_TTL": "writer_ttl",
    "HIPPOCAMPUS_LAYER_FP": "layer_fp",
    "HIPPOCAMPUS_LAYER_DEL_LEDGER": "layer_del",
    "HIPPOCAMPUS_SOURCE_SIG": "source_sig",
    "HIPPOCAMPUS_STATUS_CONTRADICTIONS": "status_contradictions",
    "HIPPOCAMPUS_MAP_AUTO_VERIFY": "map_auto_verify",
    "HIPPOCAMPUS_MAP_AUTO_INTERVAL": "map_auto_interval_h",
    "HIPPOCAMPUS_MAP_AUTO_BUDGET_S": "map_auto_budget_s",
    "HIPPOCAMPUS_MAP_MENTION": "map_mention",
    "HIPPOCAMPUS_MAP_MENTION_INTERVAL": "map_mention_interval_h",
    "HIPPOCAMPUS_MAP_MENTION_BUDGET_S": "map_mention_budget_s",
    "HIPPOCAMPUS_MAP_MENTION_MAX_RECORDS": "map_mention_max_records",
    "HIPPOCAMPUS_FILE_REGISTRY_OFF": "file_registry_off",
    "HIPPOCAMPUS_FILE_REGISTRY_INTERVAL": "file_registry_interval_h",
    "HIPPOCAMPUS_OVERVIEW_BLOCK": "overview_block",
    "HIPPOCAMPUS_STATUS_BLOCK": "status_block",
    "HIPPOCAMPUS_TASK_PROFILE_BLOCK": "task_profile_block",
    "HIPPOCAMPUS_MAP_INDEX_BLOCK": "map_index_block",
    "HIPPOCAMPUS_MAP_DETAIL_BLOCK": "map_detail_block",
    "HIPPOCAMPUS_TOPIC_LAYER": "topic_layer",
    "HIPPOCAMPUS_TOPIC_JOIN_SIM": "topic_join_sim",
    "HIPPOCAMPUS_TOPIC_MAX": "topic_max",
    "HIPPOCAMPUS_TOPIC_MAX_MEMBERS": "topic_max_members",
    "HIPPOCAMPUS_TOPIC_SUMMARY_TOPN": "topic_summary_topn",
    "HIPPOCAMPUS_TOPIC_LOW_INFO_FILTER": "topic_low_info",

    # --- 用户级记忆继承槽 ---
    "HIPPOCAMPUS_USER_SCOPE_LIMIT": "user_scope_limit",
    "HIPPOCAMPUS_USER_SCOPE_IMPORTANCE": "user_scope_importance",

    # --- 提升流 ---
    "HIPPOCAMPUS_PROMOTE_MIN_SESSIONS": "promote_min_sessions",
    "HIPPOCAMPUS_SENSITIVE_WORDS": "sensitive_words",

    # --- 家库写边界 ---
    "HIPPOCAMPUS_CROSS_LIBRARY_WRITE": "cross_library_write",

    # --- 用户级任务目录 ---
    "HIPPOCAMPUS_TASK_DIRECTORY_LIMIT": "task_directory_limit",

    # --- engine/brain 直连面 4 个 ---
    "HIPPOCAMPUS_EMB_BACKEND": "onnx_backend",
    "HIPPOCAMPUS_ONNX_NO_ARENA": "onnx_no_arena",
    "HIPPOCAMPUS_SHARED_EMBEDDER": "shared_embedder",
    "HIPPOCAMPUS_EMB_CACHE": "emb_cache",

    # --- 按系统语言自动选默认嵌入模型 ---
    "HIPPOCAMPUS_EMB_MODEL_AUTO": "emb_model_auto",

    # --- 超长内容处置（warn/split/reject，默认 warn）---
    "HIPPOCAMPUS_LONG_CONTENT": "long_content",

    # --- recall 单条 content 截断护栏（0 = 不截断）---
    "HIPPOCAMPUS_RECALL_CONTENT_MAX_CHARS": "recall_content_max_chars",

    # --- 睡眠窗口脑侧旋钮（总开关在 daemon）---
    "HIPPOCAMPUS_SLEEP_REPLAY_DAYS": "sleep_replay_days",
    "HIPPOCAMPUS_SLEEP_MAX_REPLAY": "sleep_max_replay",
    "HIPPOCAMPUS_SLEEP_MIN_CLUSTER": "sleep_min_cluster",

    # --- 写侧·呈现线（编码门控 / 前瞻触发）---
    "HIPPOCAMPUS_NOVELTY_GATE": "novelty_gate",
    "HIPPOCAMPUS_PROSPECTIVE_TRIGGER": "prospective_trigger",

    # --- 访问驱动的画像转存 ---
    "HIPPOCAMPUS_PROFILE_PROMOTION": "profile_promotion",

    # --- store/lnn/library/map/retrieval/domain 构造注入面 16 旋钮
    #     （消费点注入缝见各模块 apply_config）---
    "HIPPOCAMPUS_SNAPSHOT_MIN_HOURS": "snapshot_min_hours",
    "HIPPOCAMPUS_MULTI_WRITER": "multi_writer",
    "HIPPOCAMPUS_SAVE_LOCK_TIMEOUT": "save_lock_timeout",
    "HIPPOCAMPUS_MATRIX_DTYPE": "matrix_dtype",
    "HIPPOCAMPUS_VECTORS_MMAP": "vectors_mmap",
    "HIPPOCAMPUS_ALLOW_STALE_SAVE": "allow_stale_save",
    "HIPPOCAMPUS_STORE_BACKEND": "store_backend",
    "HIPPOCAMPUS_FTS5": "fts5",
    "HIPPOCAMPUS_LNN_BACKEND": "lnn_backend",
    "HIPPOCAMPUS_LIBRARY_ROOTS": "library_roots",
    "HIPPOCAMPUS_MAX_RESIDENT_LIBS": "max_resident_libs",
    "HIPPOCAMPUS_FILE_REGISTRY_IGNORE": "file_registry_ignore",
    "HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES": "file_registry_max_files",
    "HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S": "file_registry_budget_s",
    "HIPPOCAMPUS_INTENT_ROUTING": "intent_routing",
    "HIPPOCAMPUS_TAG_SCHEMA": "tag_schema",
}


# ── 根层入口 env 直读豁免清单（封闭）──────────
# 根层入口（launcher/daemon/cli/warmup/bridge/__init__）的 env 直读**刻意
# 不收编**进 load_config 字段——它们是**启动拓扑 plumbing**：多为「Config
# 加载之前的进程决定」（launcher 开关 / daemon socket·key / 自退超时 /
# 第三方库线程 env），收进 config 反而把加载顺序绕成环。清单**封闭**：
# tests/test_m1c_batch2_config.py 的 G7 守卫按文件钉住读取数，新增直读
# 必须先来这里登记理由，否则测试翻红。
#
# | 文件 | 读取数 | 名单与豁免理由 |
# |---|---|---|
# | launcher.py | 12 | DAEMON_READY_TIMEOUT（模块快照，等 daemon 就绪的进程决定）；RUNTIME_DIR（运行目录）；LAUNCHER 开关（是否走 daemon 拓扑）；`dict(os.environ)`（spawn 环境复制，非旋钮）；LAUNCHER_DEBUG；DIAG；RECONNECT；PERSIST_DIR / EMB_MODEL / DAEMON_LIBRARY / DAEMON_VERSION（向 daemon **announce 透传**——launcher 先于 config 存在，只做信使）；PERSIST_DIR **另有一次启动前可写性预检**（`preflight_persist_dir`：库目录写不进等于每次写入静默丢失，属配置坏了，必须 fail-fast 而非退回直连） |
# | daemon.py | 10 | DAEMON_HEARTBEAT（模块快照）；MODEL_UNLOAD / SLEEP_IDLE / SLEEP_BUDGET（哨兵阈值，进程拓扑决定——睡眠总开关按「根层入口属启动拓扑 plumbing」口径留在 daemon）；DAEMON_ECHO / FAULTHANDLER（诊断，Config 前生效）；DAEMON_KEY / LIBRARY / MODEL / IDLE_EXIT（socket/key/自退，进程拓扑） |
# | cli.py | 7 | APPDATA（系统 env，非本项目）；PERSIST_DIR / EMB_MODEL / EMB_CACHE / TOOL_PROFILE（人机输出/自省命令的**展示透传**——CLI 是读给人看的，值与 config 一致，不参与行为分支）；`dict(os.environ)`（setup 写后即验按刚写入的 entry 复制 spawn 环境做真实握手，非旋钮）；HF_ENDPOINT（模型预取走国内镜像，`setdefault` 的返回值直接复用，**不二次读取**） |
# | warmup.py | 4 | OMP/MKL_NUM_THREADS（第三方库 env，setdefault 预热线程上限）；LNN_BACKEND / EMB_BACKEND（warm 决策的早读——发生在 Config 消费前） |
# | bridge.py | 1 | BRIDGE_DEBUG（模块快照，调试面） |
# | __init__.py | 1 | OPENBLAS_NUM_THREADS（第三方库 env，setdefault——numpy 导入前必须定） |


#: bool 字段 → 语义类型（A/B/C/D，见上方注释）。迁移前四处散点各写各的，
#: 收编时必须按字段记住原语义，否则「HIPPOCAMPUS_RECALL_DECAY=maybe」
#: 这种边界值就从 False 变 True。
#: B 型（白名单）内部还分宽/窄名单：recall_decay 含 "on"、ner_allow_torch
#: 不含（core.py 实录）。_BOOL_NARROW 列窄名单字段。
_BOOL_SEMANTICS: dict[str, str] = {
    # A 黑名单默认 true（server.py 模块级）
    "auto_save": "A", "auto_backfill": "A", "supersede": "A",
    # 写侧·呈现线两旋钮同为 A 型：显式 0/false/no/off 才关，脏值回落默认开
    # （编码门控与前瞻触发都是"默认生效、说清楚才关"的机制）。
    "novelty_gate": "A", "prospective_trigger": "A",
    # 画像转存同为 A 型（默认生效的巩固期机制，说清楚才关）。
    "profile_promotion": "A",
    # B 白名单（manager/core 函数内）
    #   逃生门（cross_library_write）亦走白名单：**只有显式真值开**，
    #   空串/脏值（maybe）落默认 False——边界不能因为拼错的 env 悄悄失效。
    "recall_decay": "B", "ner_allow_torch": "B", "cross_library_write": "B",
    # C 黑名单默认 false（server.py _WARMUP_DIAG）
    "warmup_diag": "C",
    # D 非空非 0（manager ENV_ALLOW_MODEL_SWITCH）
    "allow_model_switch": "D",
    # E 三态（原 ``common/constants._env_flag``）**：0/false/no/off 关、
    #   1/true/yes/on 开、其余脏值回落到字段默认**。与 A 型的唯一差别就在
    #   ``X=maybe`` 这种输入上：A 判开、E 判默认——parity 探针
    #   实测过这一列，它是区分两种语义的唯一判据。
    "layer_fp": "E", "layer_del": "E", "source_sig": "E",
    "status_contradictions": "E", "map_auto_verify": "E", "map_mention": "E",
    "file_registry_off": "E",
    "overview_block": "E", "status_block": "E", "task_profile_block": "E",
    "map_index_block": "E", "map_detail_block": "E",
    "topic_layer": "E", "topic_low_info": "E",
}
_BOOL_NARROW: set[str] = {"ner_allow_torch"}

#: 特殊取值形态（不走通用类型分派）。登记的是**语义**，不是参数。
#: ``writer_ttl`` 的正值门见 Config 字段注释。
_POSITIVE_ONLY: set[str] = {"writer_ttl"}


def load_config() -> Config:
    """读 env 产出一份不可变配置。**每次调用重读**（无缓存，理由见 docstring）。"""
    c = Config()
    values: dict[str, Any] = {}
    for env, attr in ENV_MAP.items():
        default = getattr(c, attr)
        if isinstance(default, bool):
            sem = _BOOL_SEMANTICS.get(attr, "A")
            if sem == "A":
                values[attr] = _flag_deny_true(env, default)
            elif sem == "B":
                values[attr] = _flag_allow_only(
                    env, default, with_on=attr not in _BOOL_NARROW)
            elif sem == "C":
                values[attr] = _flag_deny_false(env, default)
            elif sem == "D":
                values[attr] = _flag_nonempty(env, default)
            else:
                values[attr] = _flag_tristate(env, default)
        elif isinstance(default, int) and not isinstance(default, bool):
            values[attr] = _int(env, default)
        elif attr in _POSITIVE_ONLY:
            values[attr] = _positive_float(env, default)
        elif isinstance(default, float):
            values[attr] = _float(env, default)
        else:
            values[attr] = _str(env, default)
    # recall_decay_tau 的现状语义：空串 = 默认 90 天（不是 0.0），
    # 显式 0 由调用方判「关闭」。_float 的空串回落已保证这一点。
    return Config(**values)
