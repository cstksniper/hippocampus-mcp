# -*- coding: utf-8 -*-
"""全包最底层的支撑模块：阈值常量表 + 无 ``self`` 的纯函数。

本模块**只 import 标准库**，不 import 包内任何模块 —— 这是它能被 ``brain`` /
``map`` / ``file_registry`` 等各层同时依赖的前提；一旦反向依赖，就会出现
层级倒挂或循环导入。本模块从包根层拆入 ``common/``（分层重构的产物）。

这些符号与 HippoBrain 的实例状态**完全无关**：它们在类出现之前就已求值
完毕，所以搬动不引入任何新的依赖方向。

``brain`` 侧 ``from ..common.constants import *`` 保持对外导入兼容 ——
包外有 20+ 个脚本直接 ``from hippocampus_mcp.brain import NAV_TAG_KEY``
这类常量，一行都不能断。

⚠️ ``__all__`` 必须显式列出**下划线开头**的符号（``_now`` / ``_nid`` /
``_STOPWORDS`` / ``_env_int`` …）：``import *`` 默认不导出它们，漏了就是
运行期 NameError，而且只在某些代码路径上炸。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

# 与拆分前 **同名**：拆分前这些函数里的 logger 就是 brain 模块的 logger，
# 硬编码名字才能让搬走后的日志行为逐位等价（不改 logger 名、不改 handler）。
logger = logging.getLogger("hippocampus_mcp.brain")

# ── 阈值常量（可直接调参，出处多为真库实测）──────────────────
RELEVANCE_HARD_FLOOR = 0.30   # 相似度硬门槛：低于此值直接淘汰
MMR_LAMBDA = 0.7              # MMR 相关性-多样性权衡
# MMR「字面强命中豁免」：BM25 单路排名前 N 的候选豁免多样性惩罚。
#
# 由来（中文小说库 9359 条实测）：
#   "顾父知情三拍是哪几章" 在 BM25 单路 top1 / 裸向量 top5 / 关掉 MMR 的完整路
#   都能命中含 ch43 的正解，唯独**开 MMR** 的完整路漏掉它——
#   MMR 把"与已选项近义、但带具体章号/数字的那条"当成冗余折叠了。
#   而这类"同一事实的多个表述版本"恰恰是强约束写作项目最需要看到的：
#   被折叠掉的往往就是带锚点/章号/数额的那一条。
#
# 只豁免 BM25 单路前 N 名（默认 3），不是全面关掉 MMR：
#   豁免面越大越接近"无多样性"，top_k 会被同一话题的近似表述占满。
MMR_LITERAL_EXEMPT_TOPN = 3
# ingest 回显"疑似相关"的相似度下限。
# 上界不用设——>= dedup 阈值（0.85）的已经**并入**同一条，由 ``merged_into`` 回显；
# 但已作废/已归档的记忆不参与合并，它们恰恰是"旧口径"的主要藏身处，
# 所以这里不设上界，让 superseded 的老版本也能被看见。
INGEST_NEIGHBOR_MIN_SIM = 0.60
# 未命中留痕阈值。**不要拍脑袋往上调**：
# 实测（中文小说库 9359 条）真命中的 top1 相似度只有 0.32~0.53
# （"ch43 的知情线锚点"命中时 sim=0.356），定 0.50 会把真命中记成未命中。
# 这里取召回自身的硬门槛——低于它意味着连硬门槛都没过，才算真的没找到。
MISS_LOG_THRESHOLD = RELEVANCE_HARD_FLOOR
# 留痕最多保留多少条（超出丢最旧的）。留痕是给归因分析用的，不是审计日志，
# 没有上限会在长期运行的库里无声膨胀。
MISS_LOG_MAX = 500

# 全局待办（open_topics）最多保留多少条（超出裁最旧的）。与 miss_log 同理：
# 纯追加 + 无上限会在长期运行的库里无声膨胀（实测反馈 2026-10-31）。
OPEN_TOPICS_MAX = 50

# --- 删除账本：派生层「真删除」的立碑日志 ---
# 并集表达不了减法。实体 / 画像这类**派生副本**靠「合并后按当前存活记忆重裁」
# 就够了，但 miss_log 这类**独立账本**被删了就是真删了，没有可重裁的源头。
# 实测复现：A 进程调 miss_log(clear=True)
# 清空并落盘 → B（手里还是旧快照）一次 save，3 条被完整并回来，全程无报错。
# 所以要有一份删除账本：谁删了哪一层的哪个键、什么时候删的，合并时据此拦。
LAYER_DEL_FILE = "brain_layerdel.json"
# 账本最多记多少个「层::键」。**不是**想删就能无限记：越界丢最旧的，
# 丢掉的墓碑理论上可能让更老的快照复活对应条目 —— 这是个**已知且接受**的
# 折中（30 天前的快照若此刻才落盘，它复活的不止 miss_log 一条）。
LAYER_DEL_MAX_KEYS = 5000
# 墓碑保留多久（秒）。留痕类条目本身有上限（miss_log 500 条），30 天
# 远超任何一份内存快照的合理寿命。
LAYER_DEL_TTL = 30 * 86400.0
ENV_LAYER_DEL = "HIPPOCAMPUS_LAYER_DEL_LEDGER"
# 按日活动计数保留多少天。这是个只增的计数器，不裁剪会无声膨胀；
# 趋势观察一般看最近一两周，60 天足够做趋势。
DAILY_COUNT_KEEP_DAYS = 60
# brain_stats 的 daily_activity 回显最近多少天。
# **存储上限（DAILY_COUNT_KEEP_DAYS=60）与回显窗口是两件事**：
# 报告里说"kept_days=1 太短"是误读——那是**存量只有 1 天**（功能刚上线），
# 不是上限设成 1。真正够不着「近 14 天活动曲线」的是回显窗口写死 7 天。
# 默认 14；用 env HIPPOCAMPUS_DAILY_ACTIVITY_DAYS 或 stats(daily_days=) 覆盖。
DAILY_ACTIVITY_DEFAULT_DAYS = 14
ENV_DAILY_ACTIVITY_DAYS = "HIPPOCAMPUS_DAILY_ACTIVITY_DAYS"
# 导航类内容的标签键：INDEX / 目录 / 表头这类"导航性"内容与任何
# 查询都沾点边（实测里首命中常被 INDEX 行占掉）。导入端给它们打上
# 这个标签，检索端按 :attr:`HippoBrain.nav_penalty` 打折。
# **只标记不删内容**——目录本身也是要能查到的，只是不该压过正文。
NAV_TAG_KEY = "_nav"
# 并行会话登记的默认存活时长：超过这个时间没结束的会话
# 不再出现在 active_sessions 里。**必须有过期**——本项目撞车教训累累，但
# 一个崩溃/强制关闭的会话若永久占位，登记就变成新的障碍而不是帮助。
# 只做"看得见"，不做锁：登记不阻止任何人写，只让人**写之前能看一眼**。
SESSION_TTL_SECONDS = 7200
# PIN_IMPORTANCE 从 store.memory 导入（唯一真相源，勿在此重定义）
SUPERSEDE_SIM = 0.72          # 疑似"同一事实被更新"的相似度下限（需同时满足实词重叠）
# 矛盾候选的相似度**上界**：≥ 此值的对通常是同一条事实的两个副本，交给
# 写入侧去重处理，不在这里当"矛盾"重复报给宿主。
CONTRADICTION_SIM_CEIL = 0.85
# 单个实体桶保留的矛盾候选数 / 最终回报给宿主的条数。
# 桶内两两比对是 O(k²)：候选数必须**独立于桶大小**，否则大库上光是攒候选
# 就能把内存和时间吃光（旧实现就是这么把 12,812 条全建 dict 再丢掉的）。
CONTRADICTION_BUCKET_MAX = 12
CONTRADICTION_REPORT_MAX = 5
# 实词重叠门槛。原为 0.3 —— 中文 bigram 口径下"同一份文档里的两段话"轻易超过，
# 导致 supersede 在批量导入场景大面积误杀（实测 28 对疑似取代，绝大多数是错的）。
SUPERSEDE_OVERLAP = 0.6       # 无共同实体时，需达到的实词重叠率
ENTITY_BONUS = 0.12           # 查询实体命中记忆实体的加成（跨越语义鸿沟）
DORMANT_DAYS = 180            # 多久未被调用 → 归档

# 批量改标签：一次可能动到上千条，且**改错了没有自动撤销**（标签是扁平
# dict，服务端不知道"改之前是什么"）。所以写操作默认关闭，且超过这个条数必须
# 显式 confirm —— 这两道闸门是刻意的：真实库里有大量完全无标签的条目，
# 「一行命令把全库补上域」听起来很诱人，一旦过滤条件写错就是上千条垃圾标签，
# 而 recall 按标签过滤时会**静默返回空**，宿主只会以为"这个域没内容"。
UPDATE_TAGS_CONFIRM_ABOVE = 100
UPDATE_TAGS_SAMPLE_MAX = 10

# update_tags 变更账本（借鉴前作"前后影像 + 账本"内核，**不引其审批状态机**）：
# 此前留的明确缺口是"改错了没有自动撤销"——
# 标签是扁平 dict、服务端不留改前快照，撤回只能靠 sample 里前 10 条的
# before 人工倒推。现在补上：每次**执行**记一条全量前后影像的 jsonl 条目，
# undo 按"当前值 == 记录的 after"才回滚（fail-closed：期间被第三方改过的
# 条目一律跳过并自曝，绝不盖掉后来的修改）。
# 为什么不做成层文件：它是 append-only 流水（同 save_ledger.jsonl 的定位），
# 不参与层合并；ring buffer 防无限增长。
TAG_OPS_FILE = "tag_ops.jsonl"
# 改写历史账本：L1 文本/重要度改写与合并丢弃文本的
# before 流水（sidecar，append-only，同 tag_ops 定位；形态逐字镜像
# store/tag_ops.py）。ring 大小的收编形态同上：config.mem_history_ring，
# 构造注入 manager（无 monkeypatch 面，不需要 core 快照）。
MEM_HISTORY_FILE = "mem_history.jsonl"
# ring 大小。这里只留**纯默认值**：env 读取收归 config.tag_ops_ring，
# 由 brain/core.py 在 import 时快照（``core.TAG_OPS_RING``）。为什么收货地
# 是 core 而不是本文件：本文件是 L0 纯叶，不许依赖 config；而 core 又必须
# 保住 ``TAG_OPS_RING`` 这个被测试 monkeypatch 的名字。这个值（连同 trim
# 阈值）唯一的定义处是 ``config.tag_ops_ring``，这里不留副本——两处同名
# 常量迟早漂移，而它已经没有潜在使用者。

# 检索时的**并存口径标注**：只增不删的库必然留下"同一事实的新旧两版"
# （supersede 关着 / 老数据早于该机制 / 当时没配对上的），命中时两条并列返回，
# 宿主看不出哪条更近就有取错值的风险。这里对结果集内的近邻**打标**——
# **绝不折叠、绝不改排序**：旧口径本身可能是被引用的历史事实，哪条对要由
# 权威来源决定，不由检索器替用户裁定——折叠是错的做法。
# 判据复用 _is_restatement（与写侧取代检测同一套语义）。
CONFLICT_ANNOTATE_MAX = 12    # 结果集大于此值不做两两检测（O(n²) 保护，非正确性开关）

# 删除操作的**相关性下限** = 召回自身的相似度硬门槛。
# 它的作用只有一个：把 recall 放行的 **BM25 单独召回**条目剔除掉
# ——这些条目根本没进向量路的 top_k，similarity 恰好是 0.0。
# 实测（中文小说库 9359 条）：
#   短查询下非首位候选里 sim=0 的占 44%，长句查询下占 58%。它们是"词面沾边"的主力。
#
# **不要调高这个值来过滤语义噪声——那套做法不成立**：真命中的相似度随查询长度整体漂移，
#   长句查询 band：0.632–0.781   同期噪声（有向量支撑）：0.453–0.542
#   短查询   band：0.461–0.580   同期噪声（有向量支撑）：0.276–0.387
# 两边的"真命中 vs 噪声"间隔不同（长句 ~0.09，短句 ~0.07），
# 定值门槛注定一边误杀、一边漏拦。真正稳定的是**相对优势**：靶子恒为 #1 且甩开第二名。
# 因此防误删的主力是下面的单条规则，而不是阈值。
FORGET_MIN_SIMILARITY = RELEVANCE_HARD_FLOOR
# 非显式授权时，一次 forget_by_query 最多删 1 条。
# "删掉那条记错的"在语义上几乎总是单数；top_k 里多出来的槽位正是误删的来源
# （实测复现：top_k=3 里混入 2 条几乎无关的真记忆并被全部物理删除）。
FORGET_SINGLE_BY_DEFAULT = True

# L3 画像容量上限：防止"钉住记忆上千条"把 stable_facts 灌成噪声。
# 钉住的记忆大多来自批量导入的误判，无上限时 consolidate 会把它们全塞进画像。
PROFILE_SLOT_MAX = 40         # 单个画像槽位（含 stable_facts）最多保留的条目数
PROFILE_ENTITY_MAX = 30       # 实体画像槽位数量上限（按 mention_count 取前 N）

# ── block_4（相关实体状态）的独立预算 ──────
# 老实现是"从**命中记忆**的实体出发"：top_k 越大，命中的实体越多，block_4 越吵。
# 实测（中文小说库，top_k=10）该块会把 10 条命中牵出的十几个实体全部铺开，
# 挤掉真正贴题的实体状态 —— 它拼进 system prompt 就是纯干扰。
# 收窄规则：
#   1) **查询词实体优先**（最贴题）；命中的实体只取前 BLOCK4_HIT_ENTITY_TOPN 条
#      命中作补充（查询里没提实体时的兜底）；
#   2) 整块有自己的上限（实体数 / 每实体事实数 / 总字符），
#      **不与全局 token_budget 混用** —— 否则它照样按"先到先占"吃满预算。
BLOCK4_MAX_ENTITIES = 3        # block_4 最多展示几个实体
BLOCK4_MAX_FACTS_PER_ENTITY = 3  # 每个实体最多带几条事实
BLOCK4_MAX_CHARS = 400         # block_4 全块字符上限（粗粒度闸门）
BLOCK4_HIT_ENTITY_TOPN = 3     # 只从命中的前 N 条里补实体

# ── 钉住槽 block_pinned_memories ──────────
# 两处毛病，实测都咬过人：
#   1) 条数写死 3。铁律/门禁/章节级事实同时被钉住时，后来提权的章节级卡
#      （importance 0.8）会把书级铁律挤出前三条——"钉住"等于没钉。
#   2) 排序纯按 importance。钉住条目的 importance 本就集中在高位且彼此接近，
#      这个排序实际排的是"谁先被提权"，不是"谁跟手头这件事有关"。
# 于是条数可配置（session_start(pin_limit=) > env > 默认），排序改为
# **任务相关优先**、同档内再看 importance——区分度只来自"对不对得上这次 task"。
PIN_BLOCK_LIMIT_DEFAULT = 3
ENV_PIN_BLOCK_LIMIT = "HIPPOCAMPUS_PIN_BLOCK_LIMIT"

# ── 知识地图槽 block_knowledge ──────────────────
# 开场包里唯一能回答「这个库到底装了些什么」的槽：L1 是平铺事实、L2 只有会话级
# 摘要且写完即沉睡、L3 是热点条目不是知识。
# 条数写死会两头不讨好：小库 8 条就全看见了，大库 8 条只看见最大的 8 个筐。
# 所以同样走 env 可配（与 PIN_BLOCK_LIMIT 同款语义，不新增 MCP 工具参数——
# 工具面每会话随 tools/list 注入宿主上下文，加一个参数就是长期 token 成本）。
KNOWLEDGE_BLOCK_LIMIT_DEFAULT = 8
ENV_KNOWLEDGE_BLOCK_LIMIT = "HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT"
KNOWLEDGE_BLOCK_CLIP = 70      # 单行字符上限（标题 + 规模 + 一句话）
# L3 画像里"钉住记忆"落在哪个槽（prune_stale_profile 按槽名分辨校验规则）
PROFILE_SLOT_STABLE = "stable_facts"
# 画像毕业条目的置信度校准（domain 只准 import common，故两系数留在这里；
# 槽名与三阈值是毕业校验专属，住 domain/constants.py）
PROFILE_STRENGTH_BASE = 0.3   # 置信度 = min(0.9, BASE + STEP·强度)：
PROFILE_STRENGTH_STEP = 0.06  #   强度3→0.48、5→0.60、10→0.90（槽内自洽排序）

# ingest_batch 单次上限：批量入口的意义是"一次调用省掉 N 次往返"，
# 但一次性塞几万条会把内存与单次落盘体积顶爆（落盘是整体覆写）。超限直接报错
# 让调用方分批，比"默默吃下然后 OOM"好。
INGEST_BATCH_MAX = 2000

# ── 项目概况与状态卡 ──────────────────────────────
# session_start 预激活包的两个槽：库级全貌 + 当前状态的紧凑投影。
# 立场：agent 每次开工时不只要「查询点的信息」，还要「这个库整体是什么、
# 现在什么状况」。两条硬纪律（对标前作）：
#   1. 纯派生、不检索、不聚类——热路径（每次开场）不付首次梳理那 2-3 秒；
#   2. 信号只放「推不出但错了会出事」的（FRAS 的 S 字段纪律），
#      可推断的量（总条数等）只作定位锚点，不展开。
OVERVIEW_TOP_ENTITIES = 3        # 全貌卡带几个热点实体
OVERVIEW_TOP_TAG_KEYS = 3        # 全貌卡带几个主标签键
OVERVIEW_RECENT_DAYS = 30        # 全貌卡「近 N 天增量」窗口
STATUS_SPARKLINE_DAYS = 14       # 状态卡 sparkline 窗口（天）
STATUS_MISS_DAYS = 7             # 状态卡「近 N 天未命中」窗口
STATUS_CONTRADICTION_CAP = 20    # 矛盾扫描预算上限（对）；达上限报「≥N」
SPARKLINE_MIN_BUCKETS = 4        # 少于此桶数不渲染 sparkline（避免假精度）
ENV_OVERVIEW_BLOCK = "HIPPOCAMPUS_OVERVIEW_BLOCK"          # 默认开；0/false/no/off 关
ENV_STATUS_BLOCK = "HIPPOCAMPUS_STATUS_BLOCK"
ENV_STATUS_CONTRADICTIONS = "HIPPOCAMPUS_STATUS_CONTRADICTIONS"  # 矛盾信号开关（默认开）

# ── 任务级「问题概况」──────────────────
# task_hits 是「点命中」，问题概况是「这个任务触及的结构」：
# 相关知识点 / 相关实体 / 匹配的未决待办 / 该领域近期 miss 缺口。
# 匹配纪律（全部确定性，不引模型——热路径不付推理代价）：
#   知识点/实体 —— task_hits 的 memory_id 与主题 member_ids / 实体
#     memory_ids **求交**（同一个 id 空间，精确，无阈值可调）；
#   待办/miss —— 文本 **token 相交**（任务词 + 锚点标签值 + 锚点实体名），
#     与 resolve_intent 的 keyword-in-text 同族语义。
TASK_TOPIC_MAX = 3           # 概况卡带几个相关知识点
TASK_ENTITY_MAX = 5          # 带几个相关实体
TASK_INTENT_MAX = 3          # 带几条匹配的未决待办
TASK_MISS_MAX = 3            # 带几条相关未命中
TASK_MISS_DAYS = 14          # miss 缺口的时间窗（比状态卡的 7 天长：
                            # 任务级要看「这个领域最近反复缺什么」）
TASK_TOKEN_MIN_LEN = 2       # 任务词最短长度（中文单字没有判别力）
TASK_PROFILE_CLIP = 28       # 概况卡单行标题截断（与知识地图槽同纪律：
                            # 绝不截规模数字，只截标题）
ENV_TASK_PROFILE_BLOCK = "HIPPOCAMPUS_TASK_PROFILE_BLOCK"  # 默认开；0/false/no/off 关

# sparkline 字符梯度（低→高）。块高递进，不用颜色——文本槽里没有颜色可用，
# 且「形状」才是要传达的信息（哪天活跃），不是精确值。
_SPARK_CHARS = "▁▃▅▇"


def sparkline(values: list[int]) -> str:
    """把一串计数映射成 ▁▃▅▇ 文本图（状态卡「活动」行）。

    空序列 / 全零 / 桶数 < :data:`SPARKLINE_MIN_BUCKETS` → 空串：
    调用方据此降级为纯数字。两三个点画不出形状，画了就是假精度
    （「全绿不等于每句话都对」的同型纪律）。

    归一化是**相对**的（min→▁, max→▇）：要传达的是「哪天相对更活跃」的
    形状；全同值时给中档 ▅，不假装有波动。
    """
    vals = [max(0, int(v)) for v in (values or [])]
    if len(vals) < SPARKLINE_MIN_BUCKETS or not any(vals):
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    top = len(_SPARK_CHARS) - 1
    out: list[str] = []
    for v in vals:
        idx = 2 if span == 0 else int(round((v - lo) / span * top))
        out.append(_SPARK_CHARS[max(0, min(top, idx))])
    return "".join(out)


def format_age(days: int | None) -> str:
    """把「距今天数」渲染成卡面用语（今天/昨天/N天前/N个月前）。

    None（没有时间戳可依据）⇒ 「时间未知」——卡片上不猜。月份用 30 天折算：
    这是新鲜度信号不是日历，写「2 个月前」只为省 token。
    """
    if days is None:
        return "时间未知"
    if days <= 0:
        return "今天"
    if days == 1:
        return "昨天"
    if days < 30:
        return "%d天前" % days
    return "%d个月前" % (days // 30)


# 任务 token 切分：拉丁数字串为整词；CJK 连续串切**二元组**——中文没有空格，
# 把「查朱由检煤山的事」当一个词去做 substring 匹配等于永远匹配不上，
# 而二元组（煤山/由检）是确定性切分里判别力与成本的最佳平衡点。
# 不引分词模型——匹配必须确定性可复现（同一任务两次开场得到同一张卡），
# 且热路径不付推理代价。
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

# 高频泛化二元组停用表：不拦的话「的一」「什么」这类词会把半个库的待办
# 都判成"相关"——卡片噪声比少看几个字糟得多（与知识地图槽的截断纪律同源）。
_TOKEN_STOP_BIGRAMS = frozenset({
    "的一", "了的", "是我", "是不", "我们", "你们", "他们", "这个", "那个",
    "什么", "怎么", "为什", "可以", "需要", "进行", "相关", "如下", "上述",
    "目前", "当前", "问题", "情况", "如下",
})


def _add_token(tokens: list[str], w: str) -> None:
    w = str(w).strip().lower()
    if len(w) >= TASK_TOKEN_MIN_LEN and w not in tokens and w not in _TOKEN_STOP_BIGRAMS:
        tokens.append(w)


def task_tokens(task: str, extra: list[str] | None = None) -> list[str]:
    """任务匹配用的 token 集合（确定性，不引模型）。

    两个来源：任务正文 + ``extra``（锚点记忆的标签值、锚点实体名）。
    全部 lower、去重、保序（保序让卡片输出稳定，同一任务两次开场逐字相同）。
    中文连续串额外产出全部二元组；单字（长度 < ``TASK_TOKEN_MIN_LEN``）
    丢弃——中文单字没有判别力。
    """
    tokens: list[str] = []

    def expand(chunk: str) -> None:
        cjk = _CJK_RUN_RE.findall(chunk)
        if cjk and "".join(cjk) == chunk:
            # 纯中文串：只出二元组（整串已无判别力，见模块注释）
            for run in cjk:
                for i in range(len(run) - 1):
                    _add_token(tokens, run[i:i + 2])
            return
        _add_token(tokens, chunk)
        for run in cjk:  # 中英混串：整词 + 中文部分的二元组都要
            for i in range(len(run) - 1):
                _add_token(tokens, run[i:i + 2])

    for chunk in _TOKEN_SPLIT_RE.split(task or ""):
        if chunk:
            expand(chunk)
    for w in extra or []:
        if w is None:
            continue
        _add_token(tokens, w)
        expand(str(w))  # 长实体名（如「煤山自缢事件」）同样补二元组
    return tokens


# ── 实体来源分级（origin）──────────────────────────────────────
# 实测（中文小说文档库）：中文启发式抽取的精度约 16%——
# 832 个候选里只有 112 个是真书名号命名，其余是"这笔账没法记"这类短语，
# 而其中 67% 由 ASCII 双引号模式（`"..."`）产生。中文里引号表示强调/引用/台词，
# 不表示专名；`《》` 才是中文唯一的"作品名"标记。故：
#   1) 中文文本下停用引号类模式，只保留《》（见 `_heuristic_entities`）；
#   2) 给每个实体记 origin，只有受信任来源参与「检索加成 / 画像 / 矛盾检测」，
#      启发式产物降级为"候选"，仍入库可查，但不污染决策链路。
ENTITY_ORIGIN_MANUAL = "manual"        # 宿主显式传入 entities=[...]（最可信）
ENTITY_ORIGIN_LEXICON = "lexicon"      # 命中项目专名词库
ENTITY_ORIGIN_TITLE = "title"          # 《书名号》内专名（中文唯一的作品名标记）
ENTITY_ORIGIN_SPACY = "spacy"          # spaCy NER（英文路径，质量尚可）
ENTITY_ORIGIN_HEURISTIC = "heuristic"  # 中文句式/引号启发式（噪声主要来源）

# 受信任来源：参与检索加成 / 画像 / 矛盾检测。
#   - 《》 单列一档：它是中文里明确的"作品名"分隔符，与"句式猜专名"不是一个可信度；
#     实测小说定稿范围里候选 224 个，其中 184 个来自《》（几何原本、某史料、
#     中国货币史 这类真书名），只有 40 个是裸模式噪声。若把《》 一并降级，
#     等于把最可靠的信号扔掉，所以单独给它受信任档。
#   - spaCy 保留在内——它的产物质量与中文启发式不是一个量级，且仅英文路径触发。
TRUSTED_ENTITY_ORIGINS = frozenset({
    ENTITY_ORIGIN_MANUAL, ENTITY_ORIGIN_LEXICON, ENTITY_ORIGIN_TITLE,
    ENTITY_ORIGIN_SPACY,
})
# 可信度排序：同一实体被多种来源命中时取最高（只升级不降级）。
_ORIGIN_RANK = {
    ENTITY_ORIGIN_HEURISTIC: 0,
    ENTITY_ORIGIN_SPACY: 1,
    ENTITY_ORIGIN_TITLE: 2,
    ENTITY_ORIGIN_LEXICON: 3,
    ENTITY_ORIGIN_MANUAL: 4,
}

# CJK 占比超过该值就跳过 spaCy —— en_core_web_sm 是英文模型，
# 对中文文档既抓不出实体，还白跑一趟。
_CJK_SKIP_SPACY_RATIO = 0.20

# 更新标记词表。校准用例（见 tests/test_i5_supersede_policy.py）：一组真更新
# 语句只含「迁移到/改到/提高到/升级为/延长为/切换到」——旧表全不认；
# 同族紧词一并补入，刻意不加的叙事高频词也见该测试文件。
UPDATE_MARKERS = (
    "now ", "no longer", "used to", "changed", "moved to", "switched",
    "recently", "these days", "nowadays", "现在", "不再", "改成", "换了",
    "以前", "之前是", "目前",
    "迁移到", "改到", "提高到", "升级为", "延长为", "切换到", "改为",
    "换到", "提升为", "延长到", "降到", "降为", "调整为", "调整到",
    "变更为", "更新为", "updated to", "upgraded to", "increased to",
    "extended to",
)

# markdown 结构字符：文档里海量表格行 / 树形目录会被启发式规则当成专名
# （实测 Top15 实体里 11 个是「├」「│  ├─ 伏笔表.md」这类碎片）。
_MD_TABLE_LINE = re.compile(r"^[ \t]*\|.*$", re.M)          # 表格行
_MD_CODE_FENCE = re.compile(r"```.*?```", re.S)              # 围栏代码块
_MD_TREE_CHARS = "│├└─┌┐┘┌┬┴┼═║╔╗╚╝■□●○◆◇→←↑↓"
_ENTITY_BAD_CHARS = set(
    "|`\\/*#<>[]{}()（）【】〈〉「」“”\"'=+~^$%&@:;,.!?，。！？、…—　"
    + _MD_TREE_CHARS
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "i",
    "my", "you", "your", "it", "that", "this", "to", "of", "in", "on", "at",
    "for", "and", "or", "we", "he", "she", "they", "his", "her", "their",
}

# ── lexicon_suggest 的内置停用词 ────────────────
# 只放**跨项目通用**的文档行话 / 协作套话——它们是"高频 + 跨文件"的完美候选，
# 却永远不是专名，会稳定霸占 lexicon_suggest 的头部。项目专名（人名/地名/商号）
# 一律不进来，那些写 `<词库同目录>/entity_lexicon_stop.txt`。
#
# 匹配语义：某候选 gram 只要**是**这里任一条的（含自身的）子串就丢弃。
# 所以只需写完整短语——"用户定案" 一条同时压掉 "用户定" / "案" 这些切片，
# 不必像外部停用词文件那样手工枚举前缀碎片。
_DEFAULT_LEXICON_STOP = (
    # 章节/文档自指
    "本章", "本节", "本卷", "上一章", "下一章", "前文", "后文", "全文",
    "正文", "原稿", "开篇", "结尾", "文末", "文中",
    # 元信息标签（标题/状态类）
    "备注", "说明", "摘要", "概要", "结论", "总述", "详见", "参见", "同上",
    "时间线", "待核清单", "核清单", "合订本", "要点版", "快照", "记录格式",
    "第一次", "第二次", "第三次", "第四次", "最后一次",
    # 人机协作流程套话
    "用户定案", "用户确认", "用户拍板", "用户批阅", "用户令", "已落正文",
    "已确认", "待确认", "已定案", "待定", "已完成",
    # 通用虚词/指代短语（跨项目高频，且绝不是专名）
    "为什么", "什么", "怎么", "这样", "那样", "一句话", "一件事", "本文件",
    "长出来", "一下子", "一会儿", "一样",
)

# 纯"中文数字(+第/量词)"的 gram（第五节 / 十二年 / 第一次 / 三段 / 一部分）：
# 天然高频、跨项目通用、永远不是专名。做成**规则而非枚举**，省得把量词逐个列进去。
# 只匹配"整串就是数字(+一个量词)"——"三宝""万历年"这类真专名不受影响。
_ORDINAL_RE = re.compile(
    r"^第?[一二三四五六七八九十百千万零两]{1,4}"
    r"(?:章|节|卷|次|年|月|日|版|轮|条|项|篇|段|部分|阶段|批|步)?$"
)
# 首字预筛：只有以数字或"第"开头的 gram 才值得跑正则（省掉绝大多数调用）。
_ORDINAL_HEADS = frozenset("第一二三四五六七八九十百千万零两")


def _now() -> float:
    return time.time()


def _nid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def strip_internal_fields(item: dict[str, Any]) -> dict[str, Any]:
    """剔除候选条目的**内部字段**（下划线开头），原地修改并返回该条。

    这些键是检索管道自己用的：``_emb``（当前模型向量，MMR 算多样性用）、
    ``_literal``（BM25 字面强命中，MMR 豁免惩罚用）、``_sel_sim``（MMR 复用
    的相似度缓存）。它们不该出现在工具返回里——既浪费体积，也暴露实现细节。

    早先这里是两个手写的 ``pop``，于是 ``_sel_sim`` 漏了出去
    （只有当次 MMR 选中过多条才触发，平时看不见）。改成**按前缀统一清理**，
    以后再加内部键也不会漏。公共输出字段（如 ``matched_by``）不带下划线，
    不受影响。
    """
    for k in [k for k in item if k.startswith("_")]:
        item.pop(k, None)
    return item


# ==================================================================
# 会话身份分路
# ==================================================================

# 没有 client 信息时的兜底路。所有未署名调用共享它 —— 与改造前等价。
DEFAULT_SESSION_KEY = "__default__"


def _session_key(client: str | None) -> str:
    """会话路 key。空 / 空白 -> 默认路；否则按 client 字符串。"""
    c = (client or "").strip()
    return c or DEFAULT_SESSION_KEY


# ==================================================================
# 写者可见化
# ==================================================================

WRITER_DIRNAME = "writers"
# 登记多久没心跳就算过期（秒）。崩溃/强退不会来注销，只能靠过期。
WRITER_TTL_DEFAULT = 3600.0


def pid_alive(pid) -> bool:
    """进程是否还在。**只读探测**，绝不给对方发信号。

    Windows 上不能用 ``os.kill(pid, 0)``——CPython 在 Windows 的 ``os.kill``
    是 TerminateProcess 的包装，传 0 等于用退出码 0 结束对方进程。
    这里改用 OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) 只开句柄看一眼。
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                k32.CloseHandle(h)
                return True
            return False
        except Exception:  # noqa: BLE001 —— ctypes 不可用时退回"当活着"
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def scan_writers(wdir: str, ttl: float) -> tuple[list[dict], int]:
    """扫写者登记目录，返回 (活跃登记, 清掉的过期登记数)。"""
    now = time.time()
    active: list[dict] = []
    stale = 0
    try:
        names = os.listdir(wdir)
    except OSError:
        return [], 0
    for nm in names:
        if not nm.endswith(".json"):
            continue
        path = os.path.join(wdir, nm)
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:  # noqa: BLE001 —— 读不出的登记一律当过期清掉
            stale += 1
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        if not isinstance(info, dict):
            stale += 1
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        seen = info.get("updated_at") or 0
        if (not pid_alive(info.get("pid"))
                or (now - float(seen)) > ttl):
            stale += 1
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        active.append(info)
    active.sort(key=lambda x: x.get("updated_at") or 0)
    return active, stale


# ---------------------------------------------------------------------------
# 保存账本（Save Ledger）
# ---------------------------------------------------------------------------
# 给「保存」本身建账：把事后靠 l0_dangling 反推丢了什么，换成当场看得见的
# 保存时间轴。口径：层文件白名单 / 四象限 /
# 双钟（at_mono 算间隔、at_wall 走 _reference_time）。
# 追加式 jsonl + ring buffer：绝不做成第二个写放大源（一次 save 账本 < 4KB）。
LEDGER_FILENAME = "save_ledger.jsonl"
# 这个值（连同 trim 阈值）唯一的定义处是 ``config.save_ledger_ring``，
#  运行时由 ``core.LEDGER_RING`` 暴露；本文件不再留副本。
# 启动对账在账本尾部回看的行数上限（fail 行找同 seq 的 begin 用）。
_LEDGER_TAIL_KEEP = 16

# 层内容指纹开关（＝派生层脏集）。默认开。
# 逃逸舱：万一某层的 json.dumps 成本压垮写路径（超大库），设 0 即回到
# "每次全量重写 10 层"的旧行为，语义不变、只是写放大回来。
# 开关实体在 ``config.layer_fp``：本文件不读 env（L0 纯叶，env 收归 config）。

# ---------------------------------------------------------------------------
# 源文件签名层 —— 漂移审计的事实源
# ---------------------------------------------------------------------------
# 借鉴前作「条目绑定源 SHA-256 + verify 四分类」，落到本项目的形态：
# 写入侧（ingest）给能解析到磁盘的 source 记一份 {mtime_ns, size, sha1}，
# drift_audit 拿它与磁盘现状比，回答"文件改了但记忆没更新"。
# 设计取舍（相对前作）：
#   * **按 source 标签记，不按记忆条目记** —— 一条 md 导入出几百条 L1，
#     逐条记签名是纯浪费；且 source 会消失（forget），按标签记才谈得上
#     "这个源整体过期了"；
#   * **内容事实没变就不动层** —— 签名条目带 sig_at 时间戳会让每次 ingest
#     都改层内容 ⇒ 每次 save 都写盘（写放大）。没变时保持旧条目不动，
#     "最近校验过"只留在内存缓存里；
#   * **sha1 流式读**（1MB 一块）—— 大文件不该整份进内存；
#   * 解不出路径的 source（会话 id / 自由文本）**不记** —— 硬记一份
#     "sha1=?" 的假签名只会把 unbaselined 池搅浑。
SOURCE_SIG_FILE = "brain_sourcesig.json"
# 签名计算缓存上限：(路径, mtime_ns, size) → sha1。批量导入 500 条同源文件
# 时只算一次哈希。纯性能缓存，不是事实源（事实源是层文件），超限 FIFO 丢。
SOURCE_SIG_CACHE_MAX = 256


# 开关实体在 ``config.source_sig``（同上：本文件不读 env）。

# L0 自动落盘的触发粒度 —— 累积多少条**新 raw** 落一次盘。
# 0 = 每条都落（最安全，写放大最大）；负数 = 关闭自动落盘，只自曝欠账。
# 取值在 ``config.l0_autosave_every``，实际常量见 core.py。
# ``_guess_importance`` 的上限。当前所有分支最高只到 0.75，
# 所以这行**实际是 no-op**（旧代码里写成字面量 min(score, 0.95)，被误读成
# "自动判定能给到 0.95"）。抽成常量并写明，是为了将来加更高档位时天花板
# 一眼可见 —— 0.95 以上属于"用户反复确认过"，词面规则不该给到。
_GUESS_IMPORTANCE_CAP = 0.95
# 只认下面这些层文件（**别写死数量**，加层就来改这里，写死的数一加层就变谎话）。
# 库目录里还有 .bak-* / writers/ / lexicon.json / 账本自己，
# 把它们算进来会把备份与外部工具的 touch 误报成「我漏写了」。
LEDGER_LAYER_FILES = (
    "brain_raw.json", "brain_entities.json", "brain_summaries.json",
    "brain_profile.json", "brain_sessions.json", "brain_intents.json",
    "brain_topics.json", "brain_tombstones.json", "brain_misslog.json",
    "brain_counters.json", "brain_knowledge.json", LAYER_DEL_FILE,
    SOURCE_SIG_FILE,
    "memories.json", "memories.vectors.npz",
    # LNN 状态主格式是 npz；旧库的 .pt 保留在白名单里（兼容读期间两者
    # 都可能存在于库目录，漏掉哪个都会把"别人的状态文件"误报成漏写）。
    "lnn_state.npz", "lnn_state.npz.meta.json", "lnn_state.npz.bak",
    "lnn_state.pt", "lnn_state.pt.meta.json",
)

__all__ = [
    "RELEVANCE_HARD_FLOOR",
    "MMR_LAMBDA",
    "MMR_LITERAL_EXEMPT_TOPN",
    "INGEST_NEIGHBOR_MIN_SIM",
    "MISS_LOG_THRESHOLD",
    "MISS_LOG_MAX",
    "OPEN_TOPICS_MAX",
    "LAYER_DEL_FILE",
    "LAYER_DEL_MAX_KEYS",
    "LAYER_DEL_TTL",
    "ENV_LAYER_DEL",
    "DAILY_COUNT_KEEP_DAYS",
    "DAILY_ACTIVITY_DEFAULT_DAYS",
    "ENV_DAILY_ACTIVITY_DAYS",
    "NAV_TAG_KEY",
    "SESSION_TTL_SECONDS",
    "SUPERSEDE_SIM",
    "CONTRADICTION_SIM_CEIL",
    "CONTRADICTION_BUCKET_MAX",
    "CONTRADICTION_REPORT_MAX",
    "SUPERSEDE_OVERLAP",
    "ENTITY_BONUS",
    "DORMANT_DAYS",
    "UPDATE_TAGS_CONFIRM_ABOVE",
    "UPDATE_TAGS_SAMPLE_MAX",
    "TAG_OPS_FILE",
    "MEM_HISTORY_FILE",
    "CONFLICT_ANNOTATE_MAX",
    "FORGET_MIN_SIMILARITY",
    "FORGET_SINGLE_BY_DEFAULT",
    "PROFILE_SLOT_MAX",
    "PROFILE_ENTITY_MAX",
    "BLOCK4_MAX_ENTITIES",
    "BLOCK4_MAX_FACTS_PER_ENTITY",
    "BLOCK4_MAX_CHARS",
    "BLOCK4_HIT_ENTITY_TOPN",
    "PIN_BLOCK_LIMIT_DEFAULT",
    "ENV_PIN_BLOCK_LIMIT",
    "KNOWLEDGE_BLOCK_LIMIT_DEFAULT",
    "ENV_KNOWLEDGE_BLOCK_LIMIT",
    "KNOWLEDGE_BLOCK_CLIP",
    "PROFILE_SLOT_STABLE",
    "PROFILE_STRENGTH_BASE",
    "PROFILE_STRENGTH_STEP",
    "INGEST_BATCH_MAX",
    "OVERVIEW_TOP_ENTITIES",
    "OVERVIEW_TOP_TAG_KEYS",
    "OVERVIEW_RECENT_DAYS",
    "STATUS_SPARKLINE_DAYS",
    "STATUS_MISS_DAYS",
    "STATUS_CONTRADICTION_CAP",
    "SPARKLINE_MIN_BUCKETS",
    "ENV_OVERVIEW_BLOCK",
    "ENV_STATUS_BLOCK",
    "ENV_STATUS_CONTRADICTIONS",
    "TASK_TOPIC_MAX",
    "TASK_ENTITY_MAX",
    "TASK_INTENT_MAX",
    "TASK_MISS_MAX",
    "TASK_MISS_DAYS",
    "TASK_TOKEN_MIN_LEN",
    "TASK_PROFILE_CLIP",
    "ENV_TASK_PROFILE_BLOCK",
    "format_age",
    "sparkline",
    "task_tokens",
    "ENTITY_ORIGIN_MANUAL",
    "ENTITY_ORIGIN_LEXICON",
    "ENTITY_ORIGIN_TITLE",
    "ENTITY_ORIGIN_SPACY",
    "ENTITY_ORIGIN_HEURISTIC",
    "TRUSTED_ENTITY_ORIGINS",
    "_ORIGIN_RANK",
    "_CJK_SKIP_SPACY_RATIO",
    "UPDATE_MARKERS",
    "_MD_TABLE_LINE",
    "_MD_CODE_FENCE",
    "_MD_TREE_CHARS",
    "_ENTITY_BAD_CHARS",
    "_STOPWORDS",
    "_DEFAULT_LEXICON_STOP",
    "_ORDINAL_RE",
    "_ORDINAL_HEADS",
    "_now",
    "_nid",
    "strip_internal_fields",
    "DEFAULT_SESSION_KEY",
    "_session_key",
    "WRITER_DIRNAME",
    "WRITER_TTL_DEFAULT",
    "pid_alive",
    "scan_writers",
    "LEDGER_FILENAME",
    "_LEDGER_TAIL_KEEP",
    "SOURCE_SIG_FILE",
    "SOURCE_SIG_CACHE_MAX",
    "_GUESS_IMPORTANCE_CAP",
    "LEDGER_LAYER_FILES",
]
