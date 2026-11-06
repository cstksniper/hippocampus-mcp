# -*- coding: utf-8 -*-
"""信息地图层与文件建档层的常量表。

从 ``common/constants.py`` 迁出：这些常量只服务
``map/`` 子包与 ``file_registry``，住在全包最底层的 constants 里会让
「地图层」与「通用层」的职责混在一起。

约束：本模块**不 import 包内任何东西**，只做常量定义，供 ``map/`` 内部与
``brain`` 编排层显式导入。
"""

# ── 地图索引行 + 文件建档 ─────────────────────────────────────
#
# 地图 = 带保留标签键的记忆：{"地图": "合约", "条目": "CN2601", "状态": "验证集"}。
# 一个键一条，更新走 update_tags——早期形态不新开 map_set，但约定
# 「一键一条」纪律：同一 条目 键写第二条就是旧状态与新状态竞争（设计文档 §2
# 的核心否命题），索引行的 ⚠️ 与 session_end 清单负责提醒。
# 设计评审收紧：ingest 侧加同键守卫（_guard_map_entry_dup）、
# update_tags 改地图条目时刷 timestamp（闸四只进必退的退路）、legacy 与
# registry 同键不同状态在索引行报冲突——纪律从「agent 自觉」升级为机制。
MAP_NAME_KEY = "地图"          # 保留标签键：这张记忆属于哪张地图
MAP_ENTRY_KEY = "条目"         # 保留标签键：条目键（CN2601 / chr1 / --use-avx2）
MAP_STATUS_KEY = "状态"        # 保留标签键：条目当前状态（验证集 / 开 / 已组装）
MAP_INDEX_MAX = 8              # 索引行最多带几张图（超出「等 N 张」自曝）
MAP_STALE_DAYS = 14            # 索引行 ⚠️ 阈值（闸四·陈旧面曝：只进必退，
                               # 标太勤等于没标——H3 stale 教训）

# ── 地图层侧车（map_set/get/list/verify 的存储）────────────────
# Map={name,desc,entries,updated_at}；
# Entry={key,status,fields,updated_at,updated_by,truth_source,source_sig,
#        drift,note}。条目不带向量不进 recall——地图是状态不是证据（§2）。
MAPS_VERSION = 1
MAP_DETAIL_MAX_ROWS = 20            # L1 相关图全文每图行数 cap（§6 默认 20 行；
                                    # 超出「等 N 条」自曝，绝不静默吞）
MAP_DETAIL_MAX_MAPS = 3             # L1 一次最多进包几张图（宽任务词可能
                                    # 命中多张，超了「等 N 张」自曝）

# ── 代码开关图全自动 verify ─────────────────────────────────────
# 图级 ``auto`` 配方声明真相源与提取器；没有 auto 的图自动 verify 一律不碰。
MAP_AUTO_BUDGET_S = 1.0          # 单次自动 verify 的墙钟预算；超了标
                                 # incomplete 自曝，不静默降级成「全 clean」
MAP_AUTO_MAX_SOURCE_BYTES = 1024 * 1024   # 单 source 1MB cap（构建脚本远小于此；
                                          # 太大不读——C1a 大文件不哈希同先例）

# ── 闸三：提及级漂移检测 ─────────────────────────────────────────
# 判据见 map/mention.py docstring：active 记忆晚于条目 updated_at，且**同一句**
# 里既提到条目键、又提到一个与当前 status 不同的状态词。状态词表来自该图
# 条目自己的 status 集合（不内置猜的词表）。
MAP_MENTION_BUDGET_S = 1.0       # 单次墙钟预算；超了 incomplete 自曝
MAP_MENTION_MAX_RECORDS = 2000   # 单轮最多看多少条记忆（超了自曝截断）

# 侧车写入等待（秒）。侧车临界区是毫秒级，
# 但「毫秒级」的对手方可能是 file_registry 的全量扫描（秒级）——
# 用户显式写操作值得等一小会儿，而不是立刻回 busy 让用户重试。
# 自动巡检（session_start 热路径）不适用：它抢不到就跳过本轮。
MAP_WRITE_WAIT_S = 2.0

# 文件建档：对项目文件建档，记录更新到 hippo 时的
# 文件签名；磁盘签名与建档签名不一致 ⇒ 该文件在 hippo 里的信息已过时。
# 三个时间戳分工（语义不同，不能混）：
#   first_seen  —— 首次扫到（建档时间）
#   last_synced —— hippo 内容反映该文件此签名的时刻（sync 时写入）
#   last_checked—— 上次扫描时刻（扫描本身不改变 synced 语义）
# 判定用**签名比对 + 哈希仲裁**（mtime_ns + size 失配时读一次内容哈希确认）
# 而非「文件 mtime > sync 墙钟」：
#   ① git checkout / robocopy / touch 会重置 mtime 但内容不变——哈希相同
#     即不误报（墙钟比对会把整个仓库判成过时）；
#   ② 不依赖两台机器时钟一致，也不怕同步时钟回拨。
#   签名没变的文件零读取（常态扫描只 stat）；哈希只在失配时付一次。
ENV_FILE_REGISTRY_INTERVAL = "HIPPOCAMPUS_FILE_REGISTRY_INTERVAL"  # 自动重扫间隔（小时）
ENV_FILE_REGISTRY_IGNORE = "HIPPOCAMPUS_FILE_REGISTRY_IGNORE"      # 额外忽略 glob（逗号分隔）
ENV_FILE_REGISTRY_MAX_FILES = "HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES"  # 覆盖文件数上限
ENV_FILE_REGISTRY_BUDGET_S = "HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S"    # 覆盖墙钟预算（秒）
# sync 是否拿源签名层验「hippo 真吃过当前内容」（默认开）。
# 关掉 = 回到信任制（说同步就同步）——留给「确认内容已进记忆但不是通过
# ingest 进的」这类场景；CLI 的 --no-verify 是它的单次等价物。
ENV_FR_SYNC_VERIFY = "HIPPOCAMPUS_FR_SYNC_VERIFY"
FILE_REGISTRY_CHECK_HOURS = 6   # 自动重扫间隔默认值（session_start 热路径上
                                # 不能每次开场都全量 stat——间隔门控 + 预算兜底）
FILE_REGISTRY_MAX_FILES = 20000  # 单次扫描文件数上限
FILE_REGISTRY_BUDGET_S = 1.0     # 单次扫描墙钟预算（秒）；超了标 scan_complete
                                 # =false 并在卡上自曝「扫描不完整」，不静默吞
FILE_REGISTRY_DEFAULT_IGNORES = (
    # 目录名（任一父目录命中即整枝剪掉）：版本库/依赖/缓存/IDE/本机环境
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "_probe_out", ".workbuddy",
    # 文件后缀 glob：二进制/瞬时/自指（库自己的数据文件）
    "*.pyc", "*.pyo", "*.db", "*.db-wal", "*.db-shm", "*.wal", "*.log",
    "*.tmp", ".DS_Store", "Thumbs.db",
)

__all__ = [
    "ENV_FILE_REGISTRY_BUDGET_S",
    "ENV_FILE_REGISTRY_IGNORE",
    "ENV_FILE_REGISTRY_INTERVAL",
    "ENV_FILE_REGISTRY_MAX_FILES",
    "ENV_FR_SYNC_VERIFY",
    "FILE_REGISTRY_BUDGET_S",
    "FILE_REGISTRY_CHECK_HOURS",
    "FILE_REGISTRY_DEFAULT_IGNORES",
    "FILE_REGISTRY_MAX_FILES",
    "MAPS_VERSION",
    "MAP_AUTO_BUDGET_S",
    "MAP_AUTO_MAX_SOURCE_BYTES",
    "MAP_DETAIL_MAX_MAPS",
    "MAP_DETAIL_MAX_ROWS",
    "MAP_ENTRY_KEY",
    "MAP_INDEX_MAX",
    "MAP_MENTION_BUDGET_S",
    "MAP_MENTION_MAX_RECORDS",
    "MAP_NAME_KEY",
    "MAP_STALE_DAYS",
    "MAP_STATUS_KEY",
    "MAP_WRITE_WAIT_S",
]
