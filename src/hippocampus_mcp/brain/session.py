# -*- coding: utf-8 -*-
"""B2-3 P3：会话与写入者簇

本簇从 ``brain.py`` 逐字搬来。

为什么这一簇能搬：``_current_sessions`` / writer 登记的**写作者都在
本簇内**（``_touch_writer`` / ``_unregister_writer`` / ``session_*``），
与 P2 同一个内聚判据 —— 属性写者成簇，整簇搬走不会把一个属性的
写者拆到两个文件里。

⚠️ ``current_session_id`` 是 **property**，getter 与 setter **同名**，
必须同处一个类才成立 —— 搬运器按多节点处理，两者一起搬；
只搬一个会让 property 静默失效（读得到、写不进）。

``session_start`` / ``session_end`` 是 server.py 的直调入口，搬进
mixin 后靠 MRO 解析，``brain.session_start(...)`` 一行不变。
"""

from __future__ import annotations

from ..common.constants import *  # noqa: F401,F403  （常量表：见 common/constants.__all__）
from ..common.constants import logger
import difflib
import re
from ..common.constants import logger
from ..common.present import (
    render_map_detail_block,
    render_map_index_block,
    render_overview_block,
    render_status_block,
    render_task_profile_block,
)
from ..config import load_config
# 地图层常量（见 map/constants.py）
from ..map.constants import MAP_STALE_DAYS
# B/C/D 会话常量住 domain/constants（单一消费方不进共享底座，G4 守卫）。
from ..domain.constants import (ENV_SESSION_IDLE_RECYCLE,  # noqa: F401
                                ENV_TOPIC_SIM_DEDUP, SESSION_IDLE_RECYCLE,
                                TOPIC_SIM_DEDUP)
import json
import os
import time
from typing import Any
from ..store.memory import MemoryRecord, MemoryState
from ..domain import topic as topic_mod
from ..domain.user_scope import render_user_scope, select_user_scope
# 任务目录槽的纯数据层（选择/cap/渲染）。domain 只依赖
# 标准库，I/O 在 library.task_directory（下面一行）。
from ..domain.task_directory import (render_task_directory,
                                     select_task_directory)
# 用户级继承槽的轻量读取路径（brain → library + store 合法，
# 与 core.py 的 display_name 同款方向）。只读，不构 brain、不载嵌入模型。
# 任务目录 sidecar（``<user 库>/directory.json``）走同一条
# 轻量路径——读的是文件，不是 brain。
from ..library import (LibraryResolutionError, resolve_library,
                       scan_libraries)
from ..library import task_directory as task_directory_mod
from ..store.backend import get_backend
try:
    import socket as _socket
except ImportError:  # pragma: no cover
    _socket = None


def _topic_sim(a: str, b: str) -> float:
    """D：话题相似度（0~1），归一双指标 + 硬标识罚项。

    归一 = 拉丁转小写 + 中西标点/分隔符（，、；：/|()[]等）抹平——验证方
    实测的原话对「TA/I/J/…按月续传」vs「其余品种（TA、I、…）分期补齐」
    原样 difflib 只有 0.469（0.85 阈值形同虚设，首版即栽在这），归一 +
    双指标后 0.63；无关对 ≤0.21。阈值默认 0.60，两侧各有余量。

    硬标识罚项：数字/拉丁/中文数字（43、TA、卷五的"五"）词集不同的串
    **视为不同话题**——"第43章校对" vs "第44章校对"、"卷五" vs "卷六"
    这类单字之差在 bigram 下高达 0.6~0.85，不罚必误拦（本轮 T2/T3 即
    现行犯）。罚项 = 分数减半，把这类对压到阈值之下；同义改写不换硬
    标识（验证方原话对词集全同）不受影响。
    """
    def _norm(s: str) -> str:
        s = re.sub(r"[，、；：/|,;:（）()「」【】\[\]]", " ", s.lower())
        return re.sub(r"\s+", "", s)

    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    ba = {na[i:i + 2] for i in range(len(na) - 1)} if len(na) > 1 else {na}
    bb = {nb[i:i + 2] for i in range(len(nb) - 1)} if len(nb) > 1 else {nb}
    overlap = len(ba & bb) / min(len(ba), len(bb)) if ba and bb else 0.0
    score = max(ratio, overlap)
    tokens = re.compile(r"[a-z0-9]+|[一二三四五六七八九十百千万零两甲乙丙丁戊]+")
    ta = set(tokens.findall(na))
    tb = set(tokens.findall(nb))
    if ta != tb and (ta or tb):
        # 硬标识词集不同（顺序无关："43、44" 与 "44、43" 同义）⇒ 分数减半。
        score *= 0.5
    return score


def _slot_switch(param: int, cfg_value: bool) -> bool:
    """三态槽位开关：``param>0`` 开 / ``<0`` 关 / ``==0`` 用配置值。

    语义逐字搬自已退役的 ``common/constants._block_switch``（与
    ``pin_limit`` / ``knowledge_limit`` 的「0 = 用 env」同族约定）。

    为什么要搬家而不是留在 constants：它的第二操作数**不再是 env 名**，
    而是组合根已经算好的 Config 字段；而 constants 不许依赖 config
    （依赖图 ``_ALLOWED["common"] == {"common"}``）。三态本身是**工具面的
    UI 协议**（调用者用 0 表示"别替我决定"），属于 session 层而不是常量层。
    """
    if param > 0:
        return True
    if param < 0:
        return False
    return cfg_value


def _topic_defaults() -> topic_mod.TopicDefaults:
    """domain/topic 的默认值包：现取 Config 装配（domain 不许读 env）。

    **为什么是函数而不是模块级常量**：topic 六个旋钮的原实现是 call-time
    读 env（``topic_layer_enabled()`` 每次 ``os.environ.get``），测试与
    daemon 都靠「设 env → 调 brain」钉行为；做成 import-time 快照会让
    ``HIPPOCAMPUS_TOPIC_LAYER=0`` 这类设置静默失效（迁移当时实测：
    test_knowledge_topic_layer A8/B8 四个断言、test_topic_low_info L6
    全红）。ring 三件（``core.TAG_OPS_RING`` 等）刻意反过来保留快照——
    它们的原实现本就是 constants 的 import-time 求值。

    **为什么住 session.py**：core.py 行数触顶（行数守卫钉的 ≤6299），而这里是 brain
    包里既已 import ``topic_mod`` 又已 import ``load_config`` 的地方。
    """
    c = load_config()
    return topic_mod.TopicDefaults(
        join_sim=c.topic_join_sim, max_topics=c.topic_max,
        max_members=c.topic_max_members, summary_topn=c.topic_summary_topn,
        layer_enabled=c.topic_layer, low_info=c.topic_low_info)


def _health_snapshot(brain) -> dict:
    """Compact, opt-out health bill shown at every session_end.

    Reuses projections that already exist; no new scan pipeline and no new
    tool.  Governance debt was invisible between manual review batches, so it
    is surfaced here as five numbers rather than a prose reminder.
    """
    import os as _os
    out: dict = {}
    try:
        ledger = brain.save_ledger_view()
        out["unsaved_seconds"] = ledger.get("interval_s")
        out["unsaved_layers"] = len(ledger.get("pending_layers") or [])
        out["l1_dirty"] = bool(ledger.get("l1_dirty"))
        out["save_level"] = ledger.get("level")
    except Exception as exc:  # noqa: BLE001 -- health is a gift, not a failure
        out["save_level"] = "unknown:%s" % type(exc).__name__
    try:
        pin = brain._pin_stats()
        out["pinned_ratio"] = pin.get("pinned_ratio")
        out["pin_quota_warning"] = bool(pin.get("quota_warning"))
    except Exception:  # noqa: BLE001
        pass
    try:
        kinds = brain.knowledge_stats().get("summary_kinds") or {}
        out["knowledge_stale"] = int(kinds.get("host_stale") or 0)
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..domain import stats_projection
        raw_ids = {r.get("id") for r in brain.raw_log}
        dangling, recs = stats_projection.dangling_report(
            brain.manager.store.iter_records(), raw_ids)
        out["l0_dangling_ids"] = len(dangling)
        out["l0_dangling_memories"] = recs
    except Exception:  # noqa: BLE001
        pass
    try:
        out["open_topics"] = len(brain.open_topics)
    except Exception:  # noqa: BLE001
        pass
    out["enabled"] = _os.environ.get(
        "HIPPOCAMPUS_HEALTH_REPORT", "1").strip().lower() not in (
            "0", "false", "no", "off")
    return out


#: ``session_start`` 的 L0 健康行阈值（悬空记忆条数）。0 = 有任何一条就报。
L0_HEALTH_DANGLING_THRESHOLD = 0


def _l0_health_line(brain) -> dict | None:
    """Opening-pack L0 health bill — **only when it is actually off**.

    A field nobody reads is not observability: the dangling refs in a
    7000-record library were found by an external project, not by us.
    Returns ``None`` when healthy so the default opening pack stays
    byte-identical (its key order is pinned by golden tests); set
    ``HIPPOCAMPUS_L0_HEALTH=0`` to silence it entirely.

    两个信号，都只在**确实异常**时出现：
      * ``dangling_memories > 0``：已有 L1 引用盘上不存在的 raw（不可恢复）；
      * ``l0_pending`` 达到自动落盘阈值却还没落：说明那条路被堵住或关掉了
        （阈值以下的 pending 是正常中间态，报出来只是噪声）。
    """
    import os as _os
    if _os.environ.get("HIPPOCAMPUS_L0_HEALTH", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    pending = int(getattr(brain, "_l0_pending", 0) or 0)
    every = int(getattr(brain, "_l0_autosave_every", 0) or 0)
    dangling_ids = 0
    dangling_memories = 0
    try:
        from ..domain import stats_projection
        raw_ids = {r.get("id") for r in brain.raw_log}
        dangling, dangling_memories = stats_projection.dangling_report(
            brain.manager.store.iter_records(), raw_ids)
        dangling_ids = len(dangling)
    except Exception:  # noqa: BLE001 —— 体检失败不该拦住开场
        dangling_memories = 0
    pending_stuck = bool(pending) and (
        every <= 0 or pending >= every)
    if dangling_memories <= L0_HEALTH_DANGLING_THRESHOLD and not pending_stuck:
        return None
    notes = []
    if dangling_memories > 0:
        notes.append(
            f"{dangling_memories} memories cite {dangling_ids} raw entries that "
            f"are missing from this library's L0 (source text unrecoverable)")
    if pending_stuck:
        notes.append(
            f"{pending} raw entries are memory-only (autosave threshold "
            f"{every} not reached or failed)")
    return {
        "level": "warn",
        "l0_pending": pending,
        "dangling_ids": dangling_ids,
        "dangling_memories": dangling_memories,
        "note": ("; ".join(notes) + ". Run scripts/audit_l0_dangling.py "
                 "--persist-dir <library> for the breakdown."),
    }


#: Cap on the topic ids named in a session-end nudge. A library with hundreds
#: of pending clusters must not turn the closing receipt into a dump.
KNOWLEDGE_TODO_MAX_IDS = 5


def _knowledge_todo(brain, session_memory_ids: list[str]) -> dict | None:
    """Name the knowledge topics a session should tidy, with one runnable step.

    The service deliberately does not summarise on its own (its production
    path embeds no LLM), but "leave it to the host's initiative" turned out to
    mean "never happens": real-library runs ended with only 3 summaries
    applied. This is a nudge, not an execution: it reuses the layer already
    refreshed by ``consolidate`` and never triggers first-time clustering on
    the hot path. Returns ``None`` when nothing is pending, so a healthy
    library keeps the exact same response shape and key order. Disable with
    ``HIPPOCAMPUS_KNOWLEDGE_TODO=0``.
    """
    import os as _os
    if _os.environ.get("HIPPOCAMPUS_KNOWLEDGE_TODO", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    try:
        td = _topic_defaults()
        if not td.layer_enabled:
            return None
        topics = getattr(brain, "knowledge", None) or {}
        if not topics:
            return None
        pending = topic_mod.pending_summary_ids(topics, defaults=td)
        if not pending:
            return None
        stats = topic_mod.host_summary_stats(topics)
        written = {str(m) for m in (session_memory_ids or [])}
        touched = [tid for tid in pending
                   if written & {str(m) for m in
                                 (topics.get(tid) or {}).get("member_ids") or []}]
        return {
            "pending_count": len(pending),
            "from_this_session": len(touched),
            "topic_ids": (touched or pending)[:KNOWLEDGE_TODO_MAX_IDS],
            "host_summaries": int(stats.get("host", 0)),
            "host_stale": int(stats.get("host_stale", 0)),
            "action": ("""hippo_extra(action="knowledge_topic", args='{"action":"brief_batch","batch":5}')"""),
            "then": ("""Write one conclusion per topic, then hippo_extra(action="knowledge_topic", args='{"action":"summarize_batch","items":"[...]"}')"""),
            "done_when": "pending_count == 0",
            "note": ("Summarising intelligence stays on the host; this "
                     "service only prepares material, validates and stores."),
        }
    except Exception:  # noqa: BLE001 - a nudge must never break session_end
        return None


class _SessionMixin:
    """B2-3 P3：会话与写入者簇"""

    # ------------------------------------------------------------------
    # 会话身份（署名归因）分路
    # ------------------------------------------------------------------

    @property
    def current_session_id(self) -> str | None:
        """默认路的当前会话 id。

        共享连接下服务端自造的 sess_xxx 标识的是**进程**而不是调用方，
        必然指错人。改造后每位 client 一路，本 property 只暴露默认路，
        保证未接线调用方行为逐位不变。
        """
        return self._current_sessions.get(DEFAULT_SESSION_KEY)



    @current_session_id.setter
    def current_session_id(self, value: str | None) -> None:
        if value is None:
            self._current_sessions.pop(DEFAULT_SESSION_KEY, None)
        else:
            self._current_sessions[DEFAULT_SESSION_KEY] = value



    def current_session_of(self, client: str | None = None) -> str | None:
        """某位 client 的当前会话 id（没有则 None）。"""
        return self._current_sessions.get(_session_key(client))



    def end_session_of(self, client: str | None = None) -> str | None:
        """摘掉某位 client 的当前会话（返回被摘掉的 id）。只动这一路。"""
        return self._current_sessions.pop(_session_key(client), None)



    def session_routes(self) -> dict[str, str]:
        """当前活跃的所有会话路（自检 / 自曝用）。"""
        return dict(self._current_sessions)



    def _note_session_memory(self, client: str | None,
                             memory_ids: list[str]) -> None:
        """把本次 ingest 写入的记忆 id 累积到当前会话。

        session_end 的 L2 摘要此前是纯自由文本，"上次的结论是从哪几条来的"
        不可查。这里在写入 L1 后把新 id 记到当前 client 的会话上，session_end
        收尾时整体挂到该条摘要，使结论可回指来源记忆。

        只记**写入**，不记检索引用——recall 的返回体已自带 memory_id，宿主侧
        能自行追溯；把内部间接召回（装配上下文、钉住槽）也算上只会灌入噪声。
        没有活跃会话（历史宿主从不调 session_start）时静默跳过，行为不变。
        """
        sid = self.current_session_of(client)
        if not sid or not memory_ids:
            return
        sess = self.sessions.get(sid)
        if sess is None:
            return
        bucket = sess.setdefault("memory_ids", [])
        for mid in memory_ids:
            if mid not in bucket:
                bucket.append(mid)




    def session_start(self, task: str = "", recent_limit: int = 10,
                      working_on: str = "", task_top_k: int = 5,
                      pin_limit: int = 0,
                      knowledge_limit: int = 0,
                      filter_tags: dict[str, Any] | None = None,
                      client: str | None = None,
                      overview: int = 0, status: int = 0,
                      task_profile: int = 0,
                      map_index: int = 0,
                      map_detail: int = 0,
                      user_scope: int = 0,
                      task_directory: int = 0,
                      user_access: str = "full") -> dict[str, Any]:
        """开始一个会话：返回预激活包（画像 + 最近摘要 + 未完结话题 + 待办 + 最近记忆）。

        Args:
            task: 本次会话要做的任务描述。**非空时会用它做一次召回**
                ——此前 task 只被登记进会话记录、
                从不参与检索，于是宿主开场就把任务告诉服务端，服务端却回一堆
                "最近写过的记忆"，与手头任务无关。
            working_on: 声明"本次我在改什么"（文件 / 模块 / 章节，自由文本）。
                写入会话登记后，别的会话可以调 :meth:`active_sessions` 在
                动手前看一眼有没有人在同一处作业（本项目的撞车教训累累）。
                **只是登记，不是锁**——不阻止任何写入。
            recent_limit: 预激活包带最近多少条记忆，0 关闭。
            pin_limit: 钉住槽带几条（zcode 追加 19①）。**0 = 按 env
                ``HIPPOCAMPUS_PIN_BLOCK_LIMIT`` / 默认 3；负数 = 关闭**。
                钉住条目按"与 task 相关"优先排序，同档内按 importance。
            task_top_k: task 预召回带多少条，默认 5；0 关闭（退回旧行为）。
            knowledge_limit: 知识地图槽带几个知识点。**0 = 用 env
                ``HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT``**，都没设则 8；
                负数 = 不带该槽。
            filter_tags: 结构化标签过滤，如 ``{"章": 43}``。
                把 **task 预召回**（``task_hits``）收窄到某个标签域，与
                :meth:`recall` 的 ``filter_tags`` 同一套语义（全键 AND、单键
                多值 OR、``str()`` 归一）。**只作用于 task 预召回**；recent /
                pinned / knowledge 槽是固定视角，不受它影响。留空 = 不限域。
                标签无命中时 ``task_hits`` 为空（recall 按未命中留痕）。
            overview: 项目全貌卡槽。
                **0 = 按 env ``HIPPOCAMPUS_OVERVIEW_BLOCK``（都没设则开）；
                正 = 开；负 = 关**。与 pin_limit 同款三态语义。
            status: 当前状态卡槽。三态语义同 overview，
                env 为 ``HIPPOCAMPUS_STATUS_BLOCK``。
            map_index: 地图索引行槽。三态语义同 overview，
                env 为 ``HIPPOCAMPUS_MAP_INDEX_BLOCK``。
            map_detail: 相关地图全文槽（L1 递送）。三态语义同
                overview，env 为 ``HIPPOCAMPUS_MAP_DETAIL_BLOCK``。
                **只在 task 非空时可能出现内容**——匹配判据是任务
                token/锚点实体 ∩ 图 desc/条目名，不命中
                就不递（反例约束由匹配自然保证）。
            user_scope: 用户级继承槽：带几条**用户级库**
                （约定名 ``user``）的偏好类记忆。**0 = 按 env
                ``HIPPOCAMPUS_USER_SCOPE_LIMIT``（都没设则 3）；负 = 关**
                （与 pin_limit 同款参数三态；env 负值也是关，见
                :meth:`resolve_user_scope_limit`）。选择判据：
                pinned 优先（不看阈值）→ 阈值档（importance ≥
                ``Config.user_scope_importance``，默认 0.7）→ 组内
                ``(-importance, -timestamp)`` 降序 → cap。用户库失联/未建
                **不炸开场**：结构化 ``user_scope`` 键自曝
                ``ready=false`` + ``reason``。
            task_directory: 用户级任务目录槽：带几条
                ``任务 → 库`` 映射（用户级库的 ``directory.json`` 侧车）。
                **0 = 按 env ``HIPPOCAMPUS_TASK_DIRECTORY_LIMIT``
                （都没设则 8）；负 = 关**（三态同 user_scope；关态零返回键、
                零 sidecar I/O）。按 ``updated_at`` 降序 cap，超出的老任务由
                ``task_directory.total`` 自曝。用户库未建/失联、sidecar 坏都
                **不炸开场**：``ready=false`` + ``reason`` 自曝（同款）。

        Returns:
            ``block_task_hits`` 是拼 prompt 用的文本槽；``task_hits`` 是同一批
            命中的**完整条目**（含 ``memory_id`` / ``matched_by`` / ``tags``），
            供宿主回指与调试——"我要找的那条到底有没有被捞到"看这个字段。
            ``block_project_overview`` / ``block_project_status`` 是库级全貌与
            当前状态的紧凑投影（纯派生、不检索）；``project_overview`` /
            ``project_status`` 是它们的结构化形态（供程序消费与未来 UI 渲染）。
            ``block_task_profile`` 是任务级「问题概况」：
            任务触及的知识点/实体/匹配的未决待办/该领域近期 miss 缺口，
            锚点是已算好的 ``task_hits``（不重新检索）；``task_profile`` 是
            它的结构化形态。task 为空时该槽为空串。
            ``block_map_index`` 是地图索引行：
            库里有哪几地图、各多少条、多久没更新（⚠️ 标超期）。**只报名和
            新鲜度，零正文**——地图条目是带保留标签键（``地图``/``条目``）的
            记忆，正文走 L1 任务命中 / L2 ``recall(filter_tags=...)``。
            ``block_map_detail`` 是 L1 递送：任务命中的
            侧车地图（maps.json，map_set 写入）整图进包，每图 cap 20 行 +
            「等 N 条」自曝；``map_detail`` 是它的结构化形态。
        """
        sid = _nid("sess")
        now = _now()
        key = _session_key(client)
        # C-1（验证反馈 2026-11-01）：同 client 再开场 ⇒ 旧会话自动终结。
        # 此前旧会话永远 ended_at=None 挂在 active_sessions 装在线（泄漏，
        # 实测 user 库攒出僵尸），其 stale 态还参与落盘合并。选终结不选
        # 重用：重用会把两次对话的写入归因搅在一起。compaction 恢复场景
        # （工具 docstring 明说会再调 session_start）天然受益。
        old_sid = self._current_sessions.get(key)
        if (old_sid and old_sid in self.sessions
                and not self.sessions[old_sid].get("ended_at")):
            self.sessions[old_sid]["ended_at"] = now
            self.sessions[old_sid]["superseded_by"] = sid
            self._superseded_session = old_sid
        # 只登记到这一路。共享连接下别的 client 的 current 不受影响。
        self._current_sessions[key] = sid
        self.sessions[sid] = {
            "id": sid,
            "client": _session_key(client),
            "task": task,
            "working_on": working_on or "",
            "started_at": now,
            "last_seen_at": now,
            "ended_at": None,
            "summary": "",
            "open_topics": [],
            # 累积本次会话 ingest 写入的记忆 id，session_end 挂到 L2 摘要，
            # 使"结论从哪几条来的"可回查。老库加载的 session 无此字段，读取走 .get 兜底。
            "memory_ids": [],
        }
        # task 参与召回。放在会话登记之后做，这样 recall 的
        # miss_log / current_session_id 都归属本次会话（归因用得上）。
        task_hits: list[dict[str, Any]] = []
        if task and task_top_k > 0:
            task_hits = self.recall(task, top_k=task_top_k, log_miss=True,
                                    client=client, filter_tags=filter_tags)
        # 钉住槽（追加 19①）：条数可配置 + 任务相关优先。task 命中集复用
        # 上面的召回结果，不重复检索。
        pin_n = self.resolve_pin_limit(pin_limit)
        kn_n = self.resolve_knowledge_limit(knowledge_limit)
        us_n = self.resolve_user_scope_limit(user_scope)
        # 任务目录槽条数三态（与 user_scope 同族）。
        # 0 = 按 env HIPPOCAMPUS_TASK_DIRECTORY_LIMIT（都没设则 8）；负 = 关。
        td_n = self.resolve_task_directory_limit(task_directory)
        # M1b：整趟开场只取一次配置快照。五组槽位开关 + 主题层总开关都读它，
        # 避免同一条热路径里 load_config() 被调用十几次（每次都是一轮 env 解析）。
        cfg = load_config()
        # 归纳进度自曝：多少知识点已有宿主归纳、多少的依据成员已变。
        # 开场就摆出来，宿主才知道「梳理到什么程度」，不用额外调一次。
        kn_stats = (topic_mod.host_summary_stats(self.knowledge)
                    if (kn_n > 0 and cfg.topic_layer)
                    else {})
        pinned_block = self._pinned_brief(
            pin_n, task=task,
            task_hit_ids={h["memory_id"] for h in task_hits} if task_hits else None,
        )
        recent_block = self.recent_memories(limit=recent_limit)
        knowledge_block = self._knowledge_brief(kn_n)
        # 项目全貌卡 + 状态卡：库级定向信息。纯派生、不检索，
        # 与知识地图槽同纪律——只读已有派生层，首次梳理的 2-3 秒绝不进热路径。
        ov_on = _slot_switch(overview, cfg.overview_block)
        st_on = _slot_switch(status, cfg.status_block)
        ov = self.project_overview() if ov_on else None
        st = self.project_status() if st_on else None
        # 任务级「问题概况」。锚点就是上面已算好的
        # task_hits——不重新检索。task 为空时四样皆空，渲染器返回空串，
        # 所以 gate 再加一道 bool(task)：没任务就没有"任务轮廓"可言。
        tp_on = _slot_switch(task_profile, cfg.task_profile_block) \
            and bool(task)
        tp = self.task_profile(task, task_hits) if tp_on else None
        # 地图索引行（L0 递送）。只报名+条数+新鲜度，
        # 零正文——没有地图的库渲染为空串，非用户 token 零成本。
        mi_on = _slot_switch(map_index, cfg.map_index_block)
        # 代码开关图自动 verify，**必须在算索引行
        # 之前**——不然本轮索引行报的是上一轮的 drift/待核数。间隔门控
        # （默认 6h）+ 预算 + best-effort 落盘都在方法里；没 auto 配方的
        # 库零读盘零写盘。任何异常都不许炸开场。
        if mi_on:
            try:
                self.map_auto_verify()
            except Exception:  # noqa: BLE001 —— 巡检是赠品，开场不能失败
                pass
            # 闸三：更新之后被提到「另一个状态」的条目 ⇒ 疑
            # 过期。同样必须在算索引行之前（不然本轮索引行报的是上一轮数）。
            # 纯内存字符串匹配，不读文件；间隔门控/预算/锁都在方法里。
            try:
                self.map_mention_drift()
            except Exception:  # noqa: BLE001 —— 同上，不炸开场
                pass
        mi = self.map_index() if mi_on else None
        # L1 相关图全文。锚点复用 task_profile 的
        # 实体名（不重新检索）；task 为空时同 task_profile 的 gate——
        # 没任务就没有「任务相关的地图」可言。
        md_on = _slot_switch(map_detail, cfg.map_detail_block) \
            and bool(task)
        md = self.map_detail(task, tp) if md_on else None
        # 用户级继承槽。**关（us_n <= 0）= 零痕迹**：
        # 三个返回键与两个 receipt 键全部不出现，也不做任何用户库 I/O
        # （"注入可关"的字面实现）——所以下面连 resolve/scan 都不进。
        us_info: dict[str, Any] | None = None
        us_active: int | None = None
        us_block: list[str] = []
        if us_n > 0:
            try:
                us_info, us_active = self._user_scope_brief(
                    us_n, cfg.user_scope_importance,
                    access=str(user_access or "full"))
                us_block = render_user_scope(us_info["memories"])
            except Exception as e:  # noqa: BLE001 —— 继承槽是赠品，不炸开场
                logger.warning("user-scope slot assembly failed (opening unaffected): %r", e)
                us_info = {"ready": False, "library": None, "limit": us_n,
                           "count": 0, "memories": [],
                           "reason": "user_scope_error"}
                us_active = None
        # 用户级任务目录槽（任务 → 库 的导航）。
        # **关（td_n <= 0）= 零痕迹**：不解析用户库、不读 sidecar、返回体零键
        # （与 user_scope 同款「注入可关」语义）。
        td_info: dict[str, Any] | None = None
        td_block: list[str] = []
        if td_n > 0:
            try:
                td_info = self._task_directory_brief(td_n)
                td_block = render_task_directory(td_info["entries"])
            except Exception as e:  # noqa: BLE001 —— 目录是赠品，不炸开场
                logger.warning("task-directory slot assembly failed (opening unaffected): %r", e)
                td_info = {"ready": False, "library": None, "limit": td_n,
                           "count": 0, "total": 0, "entries": [],
                           "reason": "task_directory_error"}
        out = {
            "session_id": sid,
            "query": task,
            "working_on": working_on or "",
            # 任务相关记忆摆在最前：这是"你让我做的这件事，我手上有哪些料"
            "block_task_hits": [
                f"{h['content'][:200]}（{h['memory_id'][:8]}）" for h in task_hits
            ],
            "task_hits": task_hits,
            # 回显 task 预召回实际用的标签域（null = 没设，全库检索）。
            "filter_tags": dict(filter_tags) if filter_tags else None,
            "block_profile": self._profile_brief(),
            "block_recent_summaries": [s["text"] for s in self.summaries[-3:]],
            "block_open_topics": [t["text"] for t in self.open_topics[-5:]],
            "block_pending_intents": [
                f"[{i['status']}] {i['text']}" for i in self.intents if i["status"] == "open"
            ][:5],
            "block_pinned_memories": pinned_block,
            # 自曝实际条数：宿主据此知道"钉住的东西还有几条没进来"
            "pin_limit": pin_n,
            "pinned_total": self.pinned_total(),
            # "上次干到哪"的最便宜实现：最近写入的 N 条，带 source 可回指
            "block_recent_memories": recent_block,
            # 知识地图槽：这个库有哪些知识点。只列不检索——
            # 摘要类内容进检索的教训（V9f：F1 .311 → .200）在这儿同样成立。
            "block_knowledge": knowledge_block,
            "knowledge_stats": kn_stats,
            "knowledge_limit": kn_n,
            # 项目全貌卡 + 状态卡：agent 每次开工时
            # 「这个库整体是什么、现在什么状况」的紧凑投影。文本槽可直接拼
            # prompt；结构化 dict 供程序消费与未来人类 UI 渲染真信息图
            # （设计文档 §3.3 双轨）。关闭时槽位留空串——键始终在，
            # 宿主判真值即可，不用 try/except。
            "block_project_overview": render_overview_block(ov) if ov else "",
            "block_project_status": render_status_block(st) if st else "",
            "project_overview": ov,
            "project_status": st,
            # 二期：任务级「问题概况」——task_hits 是点命中，这里是这个任务
            # 触及的结构（相关知识点/实体/匹配的未决待办/该领域近期 miss 缺口）。
            # 与两卡同双轨：文本槽拼 prompt，dict 供宿主回指。
            "block_task_profile": render_task_profile_block(tp) if tp else "",
            "task_profile": tp,
            # 地图索引行。零正文——要哪张图的正文，
            # L1 靠任务命中（block_map_detail）、
            # L2 靠 map_get 精确查询。
            "block_map_index": render_map_index_block(mi) if mi else "",
            "map_index": mi,
            # 任务命中的侧车地图全文（L1）。与索引行
            # 同双轨：文本槽拼 prompt，dict 供宿主回指。不命中 = 空串，
            # 非用户 token 零成本。
            "block_map_detail": render_map_detail_block(md) if md else "",
            "map_detail": md,
            # 开新会话时顺手把「还有谁在作业」摆出来，省一次调用
            "active_sessions": self.active_sessions(),
            # **递送收据**（借鉴前作设计）。
            # 宿主 context compaction 之后的交接**只允许带这个收据的身份
            # 信息**（session_id / 装配时间 / 各槽条数 / 库的世代），不允许
            # 把开场包正文抄进压缩交接——抄过去的正文证明不了"我当前脑子里
            # 的系统图景还有效"，凭收据重新调一次本工具才是恢复的正路。
            # knowledge_epoch / memory_active 是"图景是否过期"的判据：
            # 库在宿主离线期间被别的会话写过，这两个数会对不上。
            "receipt": {
                "session_id": sid,
                "assembled_at": now,
                "blocks": {
                    "task_hits": len(task_hits),
                    "profile": len(self.profile),
                    "recent_summaries": len(self.summaries[-3:]),
                    "open_topics": len(self.open_topics[-5:]),
                    "pending_intents": sum(
                        1 for i in self.intents if i["status"] == "open"),
                    "pinned": len(pinned_block),
                    "recent_memories": len(recent_block),
                    "knowledge": len(knowledge_block),
                    "overview": 1 if ov else 0,
                    "status": 1 if st else 0,
                    "task_profile": 1 if tp else 0,
                    "map_index": 1 if mi else 0,
                    "map_detail": 1 if md else 0,
                },
                "memory_active": sum(
                    1 for r in self.manager.store.iter_records()
                    if r.state != MemoryState.ARCHIVED.value),
                "knowledge_epoch": (topic_mod.layer_epoch(self.knowledge)
                                    if cfg.topic_layer else None),
                "knowledge_topics": (len(self.knowledge)
                                     if cfg.topic_layer else None),
                # 🔴 恢复协议的**正文**已挪到 server 的 instructions（一次性递送），
                # 这里只留指针：session_start 每次返回都要付 token，而这段说明
                # 对宿主是一次性的。键位与"指向本工具 + 提到 session_id"的语义
                # 保持不变（tests/test_s4_session_receipt.py R6a/R6b/R6c）。
                "resume_hint": (
                    "After context compaction, call session_start again with session_id to rebuild the opening pack"
                    "(full recovery protocol in the session instructions)."),
            },
        }
        l0_health = _l0_health_line(self)
        if l0_health:
            # 只在**确实有欠账/悬空**时出现：健康状态下零痕迹，键序与
            # 基线逐字一致（既有 golden 断言依赖这一点）。
            out["l0_health"] = l0_health
        if us_n > 0:
            # D4：三个返回键 + 两个 receipt 键只在"开"时出现（关态零痕迹）。
            # 放返回体末尾——关态的键序与基线逐字节一致是硬约束，
            # 开态新增键的键位没有约束，落尾最不易扰动既有键序。
            out["block_user_scope"] = us_block
            out["user_scope"] = us_info
            out["user_scope_limit"] = us_n
            out["receipt"]["blocks"]["user_scope"] = len(us_block)
            out["receipt"]["user_scope_active"] = us_active
        if td_n > 0:
            # 三返回键 + 两收据键只在"开"时出现（关态零痕迹）。
            # 同样落尾：关态的键序与 U1 基线逐字节一致仍是硬约束。
            out["block_task_directory"] = td_block
            out["task_directory"] = td_info
            out["task_directory_limit"] = td_n
            out["receipt"]["blocks"]["task_directory"] = len(td_block)
            # 目录总条数（cap 前）：宿主据此判断"还有几条老任务没进包"。
            out["receipt"]["task_directory_total"] = td_info.get("total", 0)
        if getattr(self, "_superseded_session", None):
            # C-1 自曝（零痕迹：没发生终结就无此键）。
            out["superseded_session"] = self._superseded_session
            self._superseded_session = None
        return out



    def _user_scope_brief(self, limit: int,
                          importance_min: float,
                          access: str = "full"
                          ) -> tuple[dict[str, Any], int | None]:
        """用户级继承槽的轻量读取路径（U1/D2，U5 的 directory.json 侧车
        将复用同款路径——**别在这里加 brain 构造**）。

        路径：``resolve_library("user")`` → ``get_backend(path).load_memories()``
        → ``MemoryRecord.from_dict`` → :func:`domain.user_scope.select_user_scope`。
        **不构造 MemoryManager、不载嵌入模型**（零 25–33s 首载、零 1.2GB
        常驻；sqlite env 下 get_backend 首开 manifest-only 用户库会干净
        初始化 hippocampus.db——U0 建库报备已认可该行为）。

        D2b 读侧档位（``access``）：``none`` → ``ready=false`` +
        ``reason="user_scope_access_none"``，**零用户库 I/O**（连 resolve
        都不进——档位拒绝不该有旁路读写）；``masked`` → sensitive:* 卡
        在进选择器前滤除；``full``（默认）逐位不变。

        失败一律不炸开场（继承流是赠品），结构化自曝：
        - 解析不到用户库 → ``ready=false`` + ``reason="user_library_not_found"``
          （覆盖 ``HIPPOCAMPUS_LIBRARY_ROOTS`` 的部署失联要**看得见**）；
        - 库在但读不出/记录坏 → ``ready=false`` + ``reason="user_library_read_failed"``；
        - 意外异常由调用点的兜底网收（``reason="user_scope_error"``）。

        Returns:
            ``(info, active)``。info 按固定形状 ``{ready, library, limit,
            count, memories[, reason]}``（``library = {id, name, path}``，
            解析失败时为 None）；active = 用户库**非归档**条数（与
            receipt 的 ``memory_active`` 同口径的世代判据），
            库不可用时 None。
        """
        def _not_ready(reason: str,
                       lib: dict[str, Any] | None = None
                       ) -> tuple[dict[str, Any], int | None]:
            return ({"ready": False, "library": lib, "limit": limit,
                     "count": 0, "memories": [], "reason": reason}, None)

        # D2b：none 档零用户库 I/O（连 resolve 都不进）——拒绝不该有旁路。
        if access == "none":
            return _not_ready("user_scope_access_none")
        try:
            lib = resolve_library("user", libs=scan_libraries())
        except LibraryResolutionError:
            return _not_ready("user_library_not_found")
        if lib is None:  # "user" 非空必走解析；None = 真失联（防御同上）
            return _not_ready("user_library_not_found")
        meta = {"id": lib.library_id, "name": lib.name, "path": lib.path}
        try:
            data, _issue = get_backend(lib.path).load_memories()
            records = [MemoryRecord.from_dict(r)
                       for r in (data or {}).get("records") or []]
        except Exception:  # noqa: BLE001 —— 坏库文件不炸开场，自曝读失败
            return _not_ready("user_library_read_failed", meta)
        if access == "masked":
            # D2b masked：sensitive:* 卡不进继承槽（伪造偏好即使待审失效
            # 也不该经槽注入——与 recall 的 exclude_sensitive 同口径）。
            records = [r for r in records
                       if not str(getattr(r, "sensitivity", "") or "")
                       .startswith("sensitive:")]
        entries = select_user_scope(
            records, limit=limit, importance_min=importance_min,
            active_state=MemoryState.ACTIVE.value)
        active = sum(
            1 for r in records if r.state != MemoryState.ARCHIVED.value)
        return ({"ready": True, "library": meta, "limit": limit,
                 "count": len(entries), "memories": entries}, active)



    def _task_directory_brief(self, limit: int) -> dict[str, Any]:
        """用户级任务目录槽的轻量读取路径。

        路径：``resolve_library("user")`` → ``library.task_directory
        .load_directory(user.path)`` → :func:`domain.task_directory
        .select_task_directory`。**不构造 brain、不载嵌入模型**——与
        :meth:`_user_scope_brief` 同一条路径（目录是 sidecar 文件，不是记忆，
        更不该为它付 25–33 秒或 1.2GB）。

        失败一律不炸开场（目录是导航赠品），结构化自曝：

        - 用户库解析不到 → ``reason="user_library_not_found"``（覆盖
          ``HIPPOCAMPUS_LIBRARY_ROOTS`` 的部署失联要**看得见**）；
        - sidecar 坏/不可读 → ``reason="task_directory_read_failed"``
          （**不当空目录**：空目录＝「没登记过」，文件坏了是另一回事）；
        - 目录里没有条目**不是失败**：``ready=true`` + ``count=0``。

        Returns:
            ``{ready, library, limit, count, total, entries[, reason]}``——
            ``library = {id, name, path}``（解析失败时 None）；``total`` 是
            **cap 前**的有效条目数（``count < total`` 即「还有没进包的老任务」）。
        """
        def _not_ready(reason: str,
                       lib: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"ready": False, "library": lib, "limit": limit,
                    "count": 0, "total": 0, "entries": [], "reason": reason}

        try:
            lib = resolve_library("user", libs=scan_libraries())
        except LibraryResolutionError:
            return _not_ready("user_library_not_found")
        if lib is None:  # "user" 非空必走解析；None = 真失联（防御同上）
            return _not_ready("user_library_not_found")
        meta = {"id": lib.library_id, "name": lib.name, "path": lib.path}
        data, issue = task_directory_mod.load_directory(lib.path)
        if issue:
            return _not_ready(issue, meta)
        entries = select_task_directory(data.get("entries") or [], limit=limit)
        return {"ready": True, "library": meta, "limit": limit,
                "count": len(entries),
                "total": len(data.get("entries") or []),
                "entries": entries}

    def active_sessions(self, ttl: float = SESSION_TTL_SECONDS,
                        include_self: bool = True,
                        client: str | None = None) -> list[dict[str, Any]]:
        """列出仍在作业的会话（B1）——"改文件前先看一眼"。

        Args:
            ttl: 存活窗口（秒）。以 ``last_seen_at``（缺省回退 ``started_at``）
                为准，超过即视为已过期，不再列出。默认
                :data:`SESSION_TTL_SECONDS`（2 小时）。
            include_self: 是否包含当前会话，默认包含（便于确认自己登记成功）。

        这是**可见性**工具，不是锁：它不阻止任何写入，也不做仲裁。
        """
        now = _now()
        out: list[dict[str, Any]] = []
        try:
            recycle_after = float(os.environ.get(
                ENV_SESSION_IDLE_RECYCLE, "") or SESSION_IDLE_RECYCLE)
        except ValueError:
            recycle_after = SESSION_IDLE_RECYCLE
        for s in self.sessions.values():
            if s.get("ended_at"):
                continue
            seen = s.get("last_seen_at") or s.get("started_at") or 0.0
            age = now - seen
            if recycle_after > 0 and age > recycle_after:
                # C-3（验证反馈 2026-11-01）：idle 超时的「活跃」会话自动
                # 关门——只动状态不动数据（标 recycled 供归因），并摘掉
                # current 指针。此前只展示不回收，僵尸会话越攒越多。
                s["ended_at"] = now
                s["recycled"] = True
                if self._current_sessions.get(s.get("client")) == s.get("id"):
                    self._current_sessions.pop(s.get("client"), None)
                continue
            if age > ttl:
                continue
            if not include_self and s.get("id") == self.current_session_of(client):
                continue
            out.append({
                "session_id": s.get("id"),
                # 自曝署名，宿主能一眼看出"这条是谁的"
                "client": s.get("client") or DEFAULT_SESSION_KEY,
                "task": s.get("task") or "",
                "working_on": s.get("working_on") or "",
                "started_at": s.get("started_at"),
                "idle_seconds": round(age, 1),
                "is_self": s.get("id") == self.current_session_of(client),
            })
        out.sort(key=lambda x: x["idle_seconds"])
        return out



    def touch_session(self, session_id: str = "",
                      client: str | None = None) -> bool:
        """刷新某个会话的存活时间（心跳）。不存在则返回 False。"""
        sid = session_id or self.current_session_of(client)
        if not sid or sid not in self.sessions:
            return False
        s = self.sessions[sid]
        if s.get("ended_at"):
            return False
        s["last_seen_at"] = _now()
        return True




    def session_end(
        self,
        summary: str = "",
        open_topics: list[str] | None = None,
        client: str | None = None,
        close_topics: list[str] | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        """结束会话：写 L2 摘要、登记未完结话题、保守更新 L3 画像、触发巩固。

        实测反馈 2026-10-31：open_topics 此前纯追加——重复登记不挡、
        已收口话题永远残留、列表无限增长。现在先关后开：``close_topics``
        按去空白**精确**匹配移除（模糊相似会误关，不做）并**立删除墓碑**
        （删除必立碑纪律——不立碑落盘合并会把已关话题从磁盘复活，验证
        报告 2026-11-01 的 P0）；append 前做精确 + 相似度去重（阈值
        ``ENV_TOPIC_SIM_DEDUP``，相似命中回显 ``dedup_hits``，供调用方
        决定是否 close 旧条）。全局列表还有落盘侧 cap
        （``OPEN_TOPICS_MAX``，裁最旧），触顶时返回 ``topics_cap_reached``。

        C-2（验证反馈 2026-11-01）：``session_id`` 精确收尾——只许收
        **本 client** 的会话（不许代收，错配报错且零副作用），目标不是
        本路 current 时 current 不动。
        """
        if session_id:
            rec = self.sessions.get(session_id)
            if rec is None or rec.get("ended_at"):
                return {"error": f"session_id {session_id!r} does not exist or is already closed; "
                                 "no change was made"}
            if rec.get("client") != _session_key(client):
                return {"error": "session_id does not belong to this client (collecting on behalf of others is not allowed); "
                                 "no change was made"}
            sid = session_id
            # 目标恰为本路 current 才摘指针；收历史会话不动 current。
            if self._current_sessions.get(_session_key(client)) == session_id:
                self._current_sessions.pop(_session_key(client), None)
        else:
            # 只摘这一路的 current，别的 client 的会话不动。
            sid = self.end_session_of(client)
        session_mem_ids: list[str] = []
        if sid and sid in self.sessions:
            self.sessions[sid]["ended_at"] = _now()
            self.sessions[sid]["summary"] = summary
            self.sessions[sid]["open_topics"] = open_topics or []
            # 摘出本次会话 ingest 累积的写入记忆 id，挂到下面的 L2 摘要。
            session_mem_ids = list(self.sessions[sid].get("memory_ids") or [])
        if summary:
            self.summaries.append(
                {"id": _nid("sum"), "text": summary, "timestamp": _now(),
                 "session_id": sid, "memory_ids": session_mem_ids}
            )
        def _norm_topic(t: Any) -> str:
            return str(t or "").strip()

        closed = 0
        removed: list[dict[str, Any]] = []
        for t in close_topics or []:
            key = _norm_topic(t)
            if not key:
                continue
            before = len(self.open_topics)
            kept = [x for x in self.open_topics
                    if _norm_topic(x.get("text")) != key]
            if len(kept) < before:
                removed.extend(x for x in self.open_topics
                               if _norm_topic(x.get("text")) == key)
                self.open_topics = kept
                closed += before - len(kept)
        if removed:
            # 删除必立碑（miss_log 同款纪律）：不立碑的话，save 的「磁盘
            # 打底并集」会把刚关的话题从磁盘原样并回来——内存关了、盘上
            # 没关、返回体还说 saved:true（验证报告 2026-11-01 的 P0）。
            # 墓碑随 save 落盘后，其他进程的合并也被 _apply_layer_deletions
            # 拦住，跨进程复活一并闭合。
            self.note_layer_deletion_items("brain_topics.json", removed)
        try:
            sim = float(os.environ.get(ENV_TOPIC_SIM_DEDUP, "")
                        or TOPIC_SIM_DEDUP)
        except ValueError:
            sim = TOPIC_SIM_DEDUP
        sim = max(0.0, min(1.0, sim))
        deduped = 0
        dedup_hits: list[dict[str, Any]] = []
        for t in open_topics or []:
            key = _norm_topic(t)
            if not key:
                continue
            # D（验证反馈 2026-11-01）：精确命中照旧跳过；再加相似度去重
            # ——「可见的模糊」：命中必须回显 dedup_hits（含旧条原文），
            # 逃生门 = close 旧条后重新登记。不设强制登记旗标。
            dup = next((x for x in self.open_topics
                        if _norm_topic(x.get("text")) == key), None)
            if dup is None and sim > 0:
                for x in self.open_topics:
                    ratio = _topic_sim(key, _norm_topic(x.get("text")))
                    if ratio >= sim:
                        dup = x
                        dedup_hits.append({"input": key,
                                           "matched": x.get("text"),
                                           "sim": round(ratio, 3)})
                        break
            if dup is not None:
                deduped += 1
                continue
            self.open_topics.append({"text": t, "timestamp": _now(),
                                     "session_id": sid})

        # **全局收尾按最后退出者触发**。本路会话之外还有活跃 client
        # （登记过 session_start/touch 的）时，画像提炼 + 全库巩固推迟——
        # 它们扫全库、动派生层，daemon 下会踩到 B 正在进行的会话。
        # 没有任何登记过会话的路（历史宿主从不调 session_start）⇒ others=0，
        # 行为与改造前逐位一致。save() **不推迟**（追加 4 的落盘保证）。
        others_active = len(self._current_sessions)
        profile_updates: list[dict[str, Any]] = []
        pruned: list[dict[str, Any]] = []
        report: dict[str, Any] | None = None
        if others_active > 0:
            self._pending_consolidate = True
        else:
            had_pending = self._pending_consolidate
            profile_updates = self._update_profile()
            # consolidate() 内部会再刷一次画像，届时摘除清单已被清空——
            # 这里先留一份，session_end 返回的是"本次收尾总共摘了什么"。
            pruned = list(self.last_profile_pruned)
            report = self.consolidate()
            if had_pending:
                report = dict(report or {})
                report["covered_pending"] = True
            self._pending_consolidate = False

        # 落盘（覆盖报告·追加 4）：agent 语义上把 session_end 当"收尾"，
        # 而此前它**完全不落盘**——实测宿主脚本 ingest 2 条 → session_end →
        # 进程退出 → 磁盘条数回退，2 条蒸发（只能靠 save_brain 补救）。
        # 这里改成末尾强制保存，并且**失败不能让 session_end 失败**：
        # 多进程下并发写冲突是常态，把异常抛给宿主只会让它以为会话没结束，
        # 于是重试 → 又冲突。所以吞掉异常、在返回里自曝。
        saved = True
        save_error = ""
        try:
            self.save()
        except Exception as exc:  # noqa: BLE001 —— 见上：必须吞
            saved = False
            save_error = f"{type(exc).__name__}: {exc}"

        # 收尾维护清单。闸一（写时更新）的配套——
        # 把「状态改了你同步图了吗」嵌进 agent 既有工作流尾部，不新增动作。
        # 只列**确实有问题**的（文件过时 / 地图超期），没问题就是空列表，
        # 非用户 token 零成本。两个信号都是纯派生，失败不影响收尾本身。
        reminders: list[str] = []
        try:
            fr = self.file_registry_summary()
            if fr and (fr.get("stale") or fr.get("missing")):
                bits = []
                if fr.get("stale"):
                    bits.append("%d file-registry updates not synced (oldest %d days ago)"
                                % (fr["stale"], fr["oldest_stale_days"]))
                if fr.get("missing"):
                    bits.append("%d registered files have disappeared" % fr["missing"])
                reminders.append(
                    "File registry: %s. After re-ingesting the relevant content, run "
                    "scripts/file_registry.py sync <path> to mark it synced"
                    % " · ".join(bits))
            # 地图有两个来源，提醒分源给动作（legacy→update_tags，
            # registry→map_set）；registry 图另有闸二产物 drift 提醒。
            mi = self.map_index()
            legacy_stale = [m["name"] for m in mi["maps"]
                            if m.get("stale")
                            and m.get("source") != "registry"]
            registry_stale = [m["name"] for m in mi["maps"]
                              if m.get("stale")
                              and m.get("source") in ("registry", "both")]
            drifted = [(m["name"], m["drifted"]) for m in mi["maps"]
                       if m.get("drifted")]
            if legacy_stale:
                reminders.append(
                    "Map(s) %s not updated for over %d days -- when a status changes, sync via "
                    "update_tags (one entry per write; do not write a second entry)"
                    % ("、".join(legacy_stale), MAP_STALE_DAYS))
            if registry_stale:
                reminders.append(
                    "Map(s) %s not updated for over %d days -- update per key via "
                    "map_set(name, key, status) (one entry per write; "
                    "do not write a second entry)"
                    % ("、".join(registry_stale), MAP_STALE_DAYS))
            if drifted:
                bits = "、".join("%s(%d)" % (n, c) for n, c in drifted)
                reminders.append(
                    "Map(s) %s have entries whose source of truth changed (drift detected by map_verify) -- "
                    "re-verify, then refresh with map_set to clear" % bits)
            # 闸三：更新之后有记忆提到「另一个状态」。这是
            # 三个信号里最弱的一个（有人说 ≠ 真相是），所以动作写成两步：
            # 先看证据，再决定 ack（没变）还是 map_set（变了）。判不了的
            # 图（状态词表不足）不在这里 nag——那是配置问题，nag 等于让
            # 人天天读一条改不动的提示（闸四「标太勤等于没标」同纪律）。
            mentioned = [(m["name"], m["mentioned"]) for m in mi["maps"]
                         if m.get("mentioned")]
            if mentioned:
                reminders.append(
                    "Map(s) %s have %d entries mentioned with a different status after their last update (possibly stale, "
                    "evidence in entry mention.quote) -- if verified unchanged use scripts/maps.py ack "
                    "<map> <entry>; if truly changed use map_set"
                    % ("、".join("%s(%d)" % (n, c) for n, c in mentioned),
                       sum(c for _, c in mentioned)))
            # C 期 C1：自动 verify 的两种「说不清」。它们都不是 drift——
            # 一个是条目值解析不出（动态值/坏格式），一个是整图源读不了。
            # 只报名不指路，因为修法取决于具体情况（补 extract_args /
            # 改 auto 配方 / 把图改成人工维护），机器不替人做这个决定。
            unresolved = [(m["name"], m["failed"]) for m in mi["maps"]
                          if m.get("failed")]
            if unresolved:
                reminders.append(
                    "Map(s) %s have %d entries auto-verify could not decide (dynamic values / unrecognized format) "
                    "-- see the entry note for reasons; to verify automatically, add an auto recipe; "
                    "otherwise maintain manually via map_set"
                    % ("、".join("%s(%d)" % (n, c) for n, c in unresolved),
                       sum(c for _, c in unresolved)))
            broken_auto = [(m["name"], m["auto_error"]) for m in mi["maps"]
                           if m.get("auto_error")]
            if broken_auto:
                # 措辞不写死「读不到真相源」：C2 的失败面还有「没建档/对账
                # 目录不存在/扫描不完整」，它们同样不该被当成已核对。
                reminders.append(
                    "Map %r: auto-verify could not complete -- %s"
                    % ("、".join(n for n, _ in broken_auto),
                       broken_auto[0][1]))
            # 目录对账的三类信号。与自动 verify 的 drift 分开报，
            # 因为处置动作不同：缺文件要确认真删，错位要找实际位置，漏登
            # 要补条目——合成一句「有 N 条漂移」等于什么也没说。
            dir_miss = [(m["name"], m["dir_missing"]) for m in mi["maps"]
                        if m.get("dir_missing")]
            if dir_miss:
                reminders.append(
                    "地图%s有条目在数据目录里找不到（已删或改了名）——"
                    "确认真删就 map_set 改状态；整体搬过家就把 auto.root "
                    "指对再核对"
                    % "、".join("%s(%d)" % (n, c) for n, c in dir_miss))
            dir_conf = [(m["name"], m["dir_conflict"]) for m in mi["maps"]
                        if m.get("dir_conflict")]
            if dir_conf:
                reminders.append(
                    "地图%s有条目实际所在子目录与 status 不符——"
                    "map_get 看 drift.observed 是实际位置，改状态或搬文件"
                    % "、".join("%s(%d)" % (n, c) for n, c in dir_conf))
            unmatched = [(m["name"], m["unmatched"]) for m in mi["maps"]
                         if m.get("unmatched")]
            if unmatched:
                reminders.append(
                    "地图%s有目录里存在但没登记的键——核对后 map_set 补条目"
                    "（对账只报不填：世界多了东西，得有人告诉地图）"
                    % "、".join("%s(%d)" % (n, c) for n, c in unmatched))
            # 预算超限 / 部分源没核完：同样是「这轮没核完」，不能当全 clean
            truncated = [m["name"] for m in mi["maps"]
                         if m.get("auto_incomplete")
                         and not m.get("auto_error")]
            if truncated:
                reminders.append(
                    "地图「%s」的自动 verify 没核完（预算超限）——"
                    "下轮巡检会补，急的话 scripts/maps.py verify --auto"
                    % "、".join(truncated))
            # 评审收紧：双源同键冲突（legacy 旧值 vs registry
            # 新值）。索引行·冲突N 的清单版——不指路的话两条状态会一直
            # 无声竞争下去（设计 §2 核否命题的遗留形态）。
            conflicted = [m["name"] for m in mi["maps"] if m.get("conflicts")]
            if conflicted:
                reminders.append(
                    "地图「%s」有条目在 legacy 记忆与 registry 侧车里状态"
                    "不一致——以 registry 为准：把 legacy 那条 update_tags "
                    "改成一致，或 scripts/maps.py import-legacy 迁移后退役 "
                    "legacy" % "、".join(conflicted))
        except Exception:  # noqa: BLE001 —— 提醒是赠品，不能炸收尾
            pass

        out = {
            "session_id": sid,
            # 本次会话写入、已挂到 L2 摘要的记忆 id（可回指"结论从哪几条来的"）。
            "memory_ids": session_mem_ids,
            "summaries": len(self.summaries),
            "open_topics": len(self.open_topics),
            # 实测反馈 2026-10-31：本次收尾关闭 / 去重跳过的话题数
            # （去重命中不算错误，但要让 agent 看得见，不然像静默吞输入）。
            "topics_closed": closed,
            "topics_deduped": deduped,
            "profile_updates": profile_updates,
            "profile_pruned": pruned,
            "consolidate": report,
            # 自曝：本路之外还有活跃会话 ⇒ 全局收尾被推迟（诚实回显）
            "finalized": report is not None,
            "deferred_consolidate": others_active > 0,
            "active_clients_remaining": others_active,
            # 收尾维护清单（空列表 = 没什么要同步的）。文件过时 /
            # 地图超期时各一条，带下一步动作——不新增工具，只提醒。
            "maintenance_reminders": reminders,
            # 落盘自曝（追加 4）：saved=false 时宿主应显式调 save_brain /
            # retry_save，而不是以为已经存好了
            "saved": saved,
            "save_error": save_error,
        }
        if dedup_hits:
            # D：相似度去重的命中回显（可见的模糊——拦了谁、撞了哪条）。
            out["dedup_hits"] = dedup_hits
        if len(self.open_topics) >= OPEN_TOPICS_MAX:
            # D：cap 触顶给提示，别让登记侧静默依赖合并侧的"裁最旧"兜底。
            out["topics_cap_reached"] = True
        # B（验证反馈 2026-11-01）：层级明细。saved 仍是"没抛异常"，
        # 失败细节看 failed 层 + save_error；skipped 是合法优化如实回显。
        out["saved_layers"] = dict(getattr(self, "last_save_layers", {}) or {})
        # R2: passive health bill; opt out with HIPPOCAMPUS_HEALTH_REPORT=0
        import os as _os
        if _os.environ.get("HIPPOCAMPUS_HEALTH_REPORT", "1").strip().lower() not in (
                "0", "false", "no", "off"):
            out["health"] = _health_snapshot(self)
        # A4: name the topics this session touched and hand the host one
        # copy-paste command; absent when nothing is pending (zero noise).
        knowledge_todo = _knowledge_todo(self, session_mem_ids)
        if knowledge_todo:
            out["knowledge_todo"] = knowledge_todo
        return out




    def _touch_writer(self) -> None:
        """把「我是谁、我在哪、我改过几次」写进库，并清掉过期登记。

        共享连接 / 多进程共库时，"到底有几个进程在动这个库"此前完全靠猜。
        C11 那次数据丢失之所以只能考古，就是因为没有任何一方留名。

        崩溃 / 强退不会来注销，所以每个写者在自己落盘时顺手扫一遍目录，
        把**进程已不在**或**心跳过期**的登记清掉——清理是机会主义的，
        不做独立线程。
        """
        if not self.persist_dir:
            return
        wdir = os.path.join(self.persist_dir, WRITER_DIRNAME)
        try:
            os.makedirs(wdir, exist_ok=True)
        except OSError:
            return
        try:
            host = _socket.gethostname() if _socket is not None else ""
        except Exception:  # noqa: BLE001 —— 主机名拿不到不影响留名
            host = ""
        info = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "host": host,
            "cwd": os.getcwd(),
            "started_at": self._created_at,
            "updated_at": time.time(),
            "saves": self._save_seq,
        }
        path = os.path.join(wdir, f"{os.getpid()}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except OSError:
            return
        self.writers_active, self.writers_stale_cleaned = scan_writers(
            wdir, load_config().writer_ttl)



    def _unregister_writer(self) -> None:
        """进程退出时抹掉自己的写者登记。

        正常退出都要注销，否则别人 scans 时会看到一个已经不存在的 pid——
        虽然 :func:`scan_writers` 用 pid 存活校验兜底，但那是**事后**判活，
        不如本人主动退场干净。崩溃 / SIGKILL 不会走到这里，那种情况就只能
        靠 pid 校验。

        同 pid 的多个 Brain 实例会删同一份文件，第二个删到空是预期的，
        静默返回即可。
        """
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, WRITER_DIRNAME,
                            f"{os.getpid()}.json")
        try:
            os.remove(path)
        except OSError:
            return
