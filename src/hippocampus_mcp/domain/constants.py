# -*- coding: utf-8 -*-
"""domain 层专属常量。

``common/constants`` 是**共享底座不是倾销场**：只被单一 domain 模块（及其
绑定壳）消费的常量不该挤在那里，住到本模块来。路线与 ``map/constants``
（地图层常量已迁出）同款。

首个住户是 O4「访问驱动的画像转存」——消费方 ``domain/profile_ops`` 与
``brain/core`` 的绑定壳（壳显式 import，以保住 brain 包再导出名基线 G5）。
"""
from __future__ import annotations

# ── 访问驱动的画像转存——L1→L3「毕业」槽 ──────────────────────────
# 生物对应：系统巩固的皮层化由**反复重激活**驱动（recall_strength
# 就是重激活计数）。与 stable_facts（钉住 = 用户显式说"这条重要"）互补：
# 本槽收的是**没人钉、但反复被想起且活得够久**的条目——访问频次自己投票。
# 与 U3「绝不自动提升」红线的关系：U3 管的是 project→user **跨库**提升流
# （显式确认才写用户库）；本槽是**库内** L3 画像的展示层刷新（L3 本就随
# consolidate 自动刷新），不写任何记录、不改状态、不碰用户库。
PROFILE_SLOT_RECURRING = "recurring_facts"  # 毕业条目落的画像槽名
PROFILE_STRENGTH_MIN = 3.0    # 反复重激活下限：recall_strength ≥ 此值才够格
PROFILE_STRENGTH_DAYS = 7.0   # 稳定期限：写入存活 ≥ 此天数（新痕不毕业）
PROFILE_STRENGTH_MAX = 20     # 单次巩固的候选上限（按强度降序取前 N）
# （置信度校准两系数 PROFILE_STRENGTH_BASE/STEP 住 common/constants——
#   domain 层同层禁互引，profile_ops 只能从 common 取）


__all__ = [
    "PROFILE_SLOT_RECURRING",
    "PROFILE_STRENGTH_MIN",
    "PROFILE_STRENGTH_DAYS",
    "PROFILE_STRENGTH_MAX",
]

# --- 会话/话题（验证反馈 B/C/D，2026-11-01；消费方 brain/session） ---

# 追加话题的相似度去重阈值（0~1）。相似度 = 归一（标点/分隔符/大小写）
# 后取 difflib 比值与字符 bigram 重叠系数的较大者（换措辞/换分隔符的
# 同义话题按验证方实测原话对校准：正例 0.63，无关对 ≤0.21，0.60 两侧
# 有余量）。≥ 阈值即跳过追加并回显命中（dedup_hits）；**0 = 关闭**。
# 设计：验证反馈B_C_D设计提案_20261101.md §D（可见的模糊）。
TOPIC_SIM_DEDUP = 0.60
ENV_TOPIC_SIM_DEDUP = "HIPPOCAMPUS_TOPIC_SIM_DEDUP"

# 会话空闲自动回收阈值（秒）：active 会话 idle 超此值，在
# active_sessions 展示时自动置 ended（标 recycled）——只动状态不动数据。
# **0/负 = 关闭**。设计：验证反馈B_C_D设计提案_20261101.md §C-3。
SESSION_IDLE_RECYCLE = 24 * 3600
ENV_SESSION_IDLE_RECYCLE = "HIPPOCAMPUS_SESSION_IDLE_RECYCLE"
