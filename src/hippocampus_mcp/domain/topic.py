# -*- coding: utf-8 -*-
"""主题层（L2T）：把平铺的 L1 事实收敛成主题，产出可回指的上层表述。

为什么要有这个模块
==================

L1 是检索主战场，但它的组织方式只有「平铺 + 实体倒排」——没有任何工具能
回答「这个库有哪些知识点」。L2 只有会话级一条摘要且**不入检索路径**（写完
即沉睡），L3 画像是热点条目不是知识；``ARCHITECTURE.md`` 里那句「RAPTOR
递归上卷未实现」说的就是这一层。

本模块只做**确定性**的部分：聚类、成员维护、抽取式代表句、统计。真正的
「归纳」（把 N 条事实写成一句上层结论）需要智能，而生产路径不内嵌 LLM 是
本项目的边界——那部分由宿主完成、经写回通道落库，本模块为它预留
``summary`` / ``summary_kind`` / ``summary_evidence`` 三个字段。

硬约束（改动前务必读）
======================

* **不进检索主路径**。L2 的 ``RELEVANCE_HARD_FLOOR`` 就是防「摘要层和什么都
  相似而霸榜」；V9f 实体图扩散把 F1 从 .311 打到 .200。主题层的出口是
  ``knowledge_map`` 与 session_start，**不是** ``recall``。
* **它是 L1 的派生层**。成员失效（删除 / 作废 / 归档）必须重裁；落盘走
  ``_LAYER_ATTRS`` 的读-合并-写，删除立碑走删除账本。只改 L1 不改主题层
  等于假删除。
* **确定性优先**。遍历顺序固定、不引入随机源（``hash()`` 受 PYTHONHASHSEED
  影响，一律用 sha1）、**不存质心**——质心可由成员向量重算，存下来会随换
  模型失效，且让层文件随维度膨胀。
* **不在写路径上跑**。聚类只在显式调用与 consolidate 时触发，``ingest``
  不因此变慢——写路径延迟是硬约束。
* **宁缺勿滥**。没算过向量的记录不进主题（进 ``unassigned`` 自曝），不猜。

用法::

    items = [{"id": ..., "vec": ndarray, "content": ..., "timestamp": ...,
              "entities": [...], "tags": {...}, "importance": ...}]
    topics = build(items)                       # 全量/增量（传入 topics 即增量）
    reclip(topics, alive_ids)                   # 成员失效后重裁
    refresh(topics, items_by_id, topn=3)        # 重算 span/标题/代表句
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# 常量（env 可覆盖，便于不动代码调参）
# ----------------------------------------------------------------------

ENV_TOPIC_LAYER = "HIPPOCAMPUS_TOPIC_LAYER"
ENV_TOPIC_JOIN_SIM = "HIPPOCAMPUS_TOPIC_JOIN_SIM"
ENV_TOPIC_MAX = "HIPPOCAMPUS_TOPIC_MAX"
ENV_TOPIC_SUMMARY_TOPN = "HIPPOCAMPUS_TOPIC_SUMMARY_TOPN"

# 归入已有主题的余弦门槛。0.55 是中文句级嵌入的经验起点——太低会把全库并成
# 一个筐，太高会让每条记录各成一簇（负对照：调到 0.95 时主题数应逼近
# 记录数，否则说明相似度根本没参与判定）。
TOPIC_JOIN_SIM_DEFAULT = 0.55
# 主题数上限。超过后**强制归入最相近的主题**而不是继续新建——触顶次数
# 在 stats 里自曝。**内部**主题数≠地图显示条数（那个由 `knowledge_map(limit=)`
# 控制），所以这里可以给得比「能一眼看完」多。
# 取值依据（两个独立真库的网格扫描）：cap=200/400 都撞顶；600 对一个
# 9 千条级中文语料真库够用（550 簇不撞顶），但另一个 3.6 千条级真库的
# 自然簇数就有 652，取 600 会让那个库重新撞顶——两个真库都不撞顶的
# 最小整档是 **1000**（550 / 652 都留有余量），故定 1000。
# 成本侧：1000/60 全量重建 2.6–2.9s（质心归一化缓存落地前 8.6s，见 build()）。
TOPIC_MAX_DEFAULT = 1000
# 单个主题的成员上限。**巨型筐的直接克星**：真库实测不加这个限制时，
# 第一个簇会一路吸附到 **2647 条（占全库 27%）**——中文句向量有各向异性
# （任意两句余弦普遍偏高），绝对阈值挡不住质心漂移，唯一的解是硬上限。
# 取值依据（同轮网格扫描）：mem=300 时最大簇贴着 300 长、top5 集中度 16%；
# mem=60 把最大簇压到 60、top5 压到 3.3%，中位簇 3–5 条。代价是单例占比
# 多约 1 个点——换掉"300 条的筐里啥都有"是划算的。两个独立真库都指向
# 60，不是单库巧合。
TOPIC_MAX_MEMBERS_DEFAULT = 60
# 分批矩阵乘的批大小：批内交给 BLAS，批间保持顺序（顺序是确定性的来源）。
TOPIC_CHUNK = 512
# 抽取式代表句条数。
TOPIC_SUMMARY_TOPN_DEFAULT = 3
# 代表句每条截断长度（字符）。
TOPIC_SUMMARY_CLIP = 60
# 少于这么多成员的主题**不值得归纳**（1-2 条事实本身就说清楚了，
# 再写一句结论只是把原文换个说法）。归纳成本是宿主出的，别浪费在碎渣上。
TOPIC_MIN_SIZE_FOR_SUMMARY = 3
# 失效成员占比超过它 → 标 stale（提示"这个知识点的内容已经变过了"）。
TOPIC_STALE_RATIO = 0.5
# 主题标题里实体的最大个数。
TOPIC_TITLE_ENTITIES = 2
# 标题兜底：无实体无标签时取最早成员正文的前多少字。
TOPIC_TITLE_CLIP = 12
# 主题「世代」字段名。重建（force）时 +1；层合并时**世代高的一方整代覆盖**。
# 没有它，「重建」只是键并集意义上的叠加：磁盘旧簇会被并回来（实测
# 652 + 698 → 1050 个簇），而且对面旧写者下次保存还会再并一次——永远清不掉。
TOPIC_EPOCH_FIELD = "_epoch"
# 假簇过滤：**有信息量的成员占比**低于此值 ⇒ 判为
# 假簇。格式相似≠一件事——一堆"好的/收到/嗯"在句向量上彼此很近（都短、都
# 是应答），会被聚成一簇，但那不是知识点，让宿主归纳它是纯浪费。
# 取 0.5 而不是更高的理由：判得越严越容易把真主题误判掉，而误判是**静默**
# 的（主题从 pending 里消失，没人知道）；宁可漏筛，不可误杀。
TOPIC_LOW_INFO_RATIO = 0.5
# 正文信息量的及格线（>=1 才算"有信息量"）。见 :func:`info_score`。
TOPIC_LOW_INFO_MIN_SCORE = 1
# 开关（默认开）：0/off/false/no 关闭假簇过滤。
ENV_TOPIC_LOW_INFO = "HIPPOCAMPUS_TOPIC_LOW_INFO_FILTER"
# 正文被判为"套话"的最长长度：超过这个长度还全是套话的，本身也算是内容。
TOPIC_FILLER_MAX_LEN = 6

STATUS_ACTIVE = "active"
STATUS_STALE = "stale"
STATUS_ARCHIVED = "archived"

SUMMARY_KIND_EXTRACTIVE = "extractive"
SUMMARY_KIND_HOST = "host"


@dataclass(frozen=True)
class TopicDefaults:
    """域默认值容器——注入接缝。

    为什么需要它：``domain`` 不许 import ``config``（依赖图
    ``_ALLOWED["domain"] == {"common"}``），所以这一层**不能自己去读 env**；
    而主题层的默认值又确实是可配的。解法是依赖倒置——把默认值做成一份纯
    数据，谁调用谁提供：

      * 不传 ⇒ 用 :class:`TopicDefaults` 的字段默认（等于「清空 env」的情形），
        domain 单独可测，不依赖任何环境；
      * brain 组合根从 :class:`hippocampus_mcp.config.Config` 装配一份传进来，
        env 旋钮才接上。

    字段名与 Config 的 ``topic_*`` 一一对应，由 ``test_config_parity`` P7 钉住。
    """
    join_sim: float = TOPIC_JOIN_SIM_DEFAULT
    max_topics: int = TOPIC_MAX_DEFAULT
    max_members: int = TOPIC_MAX_MEMBERS_DEFAULT
    summary_topn: int = TOPIC_SUMMARY_TOPN_DEFAULT
    layer_enabled: bool = True
    low_info: bool = True


def _def(d: TopicDefaults | None) -> TopicDefaults:
    return d if d is not None else TopicDefaults()


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

def _unit(vec: Any) -> np.ndarray | None:
    """归一化；零向量返回 None（零向量没有方向，进聚类只会制造噪声簇）。"""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return None
    return arr / n


def _topic_id(seed: str) -> str:
    """确定性主题 id（**不用** uuid4/hash()：那会让测试与落盘不可复现）。"""
    return "tpc_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


def _centroid(member_ids: Sequence[str], norms: dict[str, np.ndarray]) -> np.ndarray | None:
    """成员单位向量的均值（再归一化）。空/全缺失返回 None。"""
    acc: np.ndarray | None = None
    cnt = 0
    for mid in member_ids:
        v = norms.get(mid)
        if v is None:
            continue
        acc = v.copy() if acc is None else acc + v
        cnt += 1
    if acc is None or cnt == 0:
        return None
    return _unit(acc / cnt)


def new_topic(member_id: str, epoch: int = 0) -> dict[str, Any]:
    """新主题的骨架（其余字段由 :func:`refresh` 填）。

    ``epoch`` 是这一簇出生的世代：日常增量为 0（与既有簇同代，合并走
    键并集）；重建时整层统一 +1（合并时整代覆盖，见 :func:`layer_epoch`）。
    """
    return {
        "id": _topic_id(member_id),
        "title": "",
        "status": STATUS_ACTIVE,
        "member_ids": [member_id],
        "size": 1,
        "summary": "",
        "summary_kind": "",
        "summary_evidence": [],
        "entities": [],
        "tags": [],
        "span": [0.0, 0.0],
        "created_at": 0.0,
        "updated_at": 0.0,
        "recall_hits": 0,
        TOPIC_EPOCH_FIELD: int(epoch),
    }


# ----------------------------------------------------------------------
# 聚类
# ----------------------------------------------------------------------

def build(
    items: Sequence[dict[str, Any]],
    topics: dict[str, dict[str, Any]] | None = None,
    join_sim: float | None = None,
    max_topics: int | None = None,
    max_members: int | None = None,
    alive_ids: Iterable[str] | None = None,
    epoch: int | None = None,
    defaults: TopicDefaults | None = None,
) -> dict[str, dict[str, Any]]:
    """把记录归入主题（传入 ``topics`` 即**增量**：只处理还没归进去的）。

    算法是 **leader 型贪心扫描**，不是层次聚类：按 ``(timestamp, id)`` 固定
    顺序逐条决定「归到最像的那个主题 / 自己开一个」。选它而不是凝聚式的三个
    理由：

      1. 凝聚式要 O(n²) 相似度矩阵（万条库 8900 万对）；这里是 O(n·k)，
         k ≤ ``max_topics``，万条库约 47 万次点积，秒级。
      2. **天然支持增量**——新记录只需和 k 个质心比一次，不必重算全库，
         所以它能挂进 ``consolidate`` 而不碰写路径。
      3. 结果对顺序敏感（先到的决定中心），但**顺序是确定的**（按时间戳+id
         排序），所以同一份输入两次跑出来的结果逐字节相同——可测、可复现。

    质心在过程中**增量维护**（``sum_vec / count``），不对每条记录重算全部
    主题的质心（那样是 O(n·k·m)，实测会慢两个量级）。

    Args:
        items: ``{"id", "vec", "content", "timestamp", "entities", "tags",
            "importance"}``；``vec`` 缺失/零向量的记录**不进主题**（进
            ``unassigned``）。
        topics: 已有主题（增量模式）。会被**原地修改**并返回。
        join_sim: 归入门槛，默认取 env / 0.55。
        max_topics: 主题数上限，默认取 env / :data:`TOPIC_MAX_DEFAULT`。
        d: :class:`TopicDefaults` 注入包（domain 不许读 env ⇒ 由调用方给）。
            省略则用纯数值默认，等价于清空 env。
        alive_ids: 当前**存活**（active）的记录 id。与「有向量」是两件事：
            存活但还没算向量的记录必须留在成员里（见下方「存活≠有向量」
            注释）。省略则退回旧口径（只认有向量的），仅为兼容既有调用。

    Returns:
        ``{topic_id: topic}``。
    """
    thr = _def(defaults).join_sim if join_sim is None else float(join_sim)
    ep = 0 if epoch is None else int(epoch)
    cap = _def(defaults).max_topics if max_topics is None else int(max_topics)
    mm = _def(defaults).max_members if max_members is None else int(max_members)
    out: dict[str, dict[str, Any]] = dict(topics or {})

    by_id = {str(it.get("id")): it for it in items}
    # 维度必须一致才比得了余弦。生产里同一 model_key 下必然同维，这里防御
    # 是为了**不崩**：混维时按首个有效向量的维度为准，其余不进主题——
    # 宁缺勿滥，它们会落到 coverage 的 unassigned 里自曝，而不是被猜进某个簇。
    dim: int | None = None
    norms: dict[str, np.ndarray] = {}
    for it in items:
        u = _unit(it.get("vec"))
        if u is None:
            continue
        if dim is None:
            dim = int(u.size)
        elif int(u.size) != dim:
            continue
        norms[str(it.get("id"))] = u

    # 已有主题：先把**失效**成员摘掉（调用方可能刚删过记忆），再算初始质心。
    #
    # 「存活」不等于「有向量」。active 但还没算向量的记录必须留在成员里
    # ——否则每刷新一次就把它摘掉、向量补齐后又重新归入，成员集合反复抖动。
    # 抖动的代价不只是数字难看：宿主归纳记的是**成员签名**，成员一变签名就
    # 对不上，于是每次刷新都报"归纳已过期"（假 stale），宿主会把它当噪声，
    # 真的过期那天反而不看了。
    alive: set[str] | None = None if alive_ids is None else {str(x) for x in alive_ids}
    assigned: set[str] = set()
    for t in out.values():
        kept = [m for m in (t.get("member_ids") or [])
                if (m in norms if alive is None else m in alive)]
        t["member_ids"] = kept
        assigned.update(kept)

    # 质心的增量维护状态：tid -> [sum_vec, count]。**归档主题不建质心**——
    # 它们是被摘空才归档的，建了也不会选中；更怕的是反过来被新成员吸附，
    # 让「已归档」变成假象。
    cent: dict[str, list[Any]] = {}
    for tid, t in out.items():
        if t.get("status") == STATUS_ARCHIVED:
            continue
        acc: np.ndarray | None = None
        cnt = 0
        for m in t["member_ids"]:
            v = norms.get(m)
            if v is None:
                continue
            acc = v.copy() if acc is None else acc + v
            cnt += 1
        if acc is not None and cnt:
            cent[tid] = [acc, cnt]
    # 主题数上限只算**未归档**的。归档主题永久占名额时，删过内容的库
    # 会一直卡在 cap 上再也建不出新簇，而没有任何地方报这件事（静默）。
    live_n = sum(1 for t in out.values()
                 if t.get("status") != STATUS_ARCHIVED)

    order = sorted(
        (it for it in items if str(it.get("id")) in norms),
        key=lambda x: (float(x.get("timestamp") or 0.0), str(x.get("id"))),
    )

    # 分批矩阵乘：把「每条 × 每个质心」的点积整批交给 BLAS，而不是逐条
    # Python 循环。真实大库实测（50 簇）逐条版能慢到秒级，
    # 挂在 consolidate 里随时会踩宿主超时（session_end 曾因
    # _find_contradictions 慢到 206s 踩过一次，同一招降到 0.34s）；
    # 改成 (b×d) @ (d×k) 后降到秒级。批内并行、批间保持顺序——顺序是
    # 结果确定性的来源，不能整批一次性算完（那样质心就不随归入更新了）。
    #
    # 质心单位向量按 tid **缓存**，``cent[tid]`` 一变即失效。缓存前实测
    # 大库里 ``_unit`` 会被重复调用大量次数，其中很大一部分来自
    # 下方「批内新建簇补算」的 O(批大小×pending) 循环——每个新建簇的质心
    # 对批内**每一条**后续记录重算一遍。缓存后同一质心只在真的变了才算。
    # 缓存不改数值：取的还是同一个 ``_unit(cent[tid][0]/cent[tid][1])``。
    _UNSET = object()
    unit_cache: dict[str, Any] = {}

    def _cent_unit(tid: str) -> np.ndarray | None:
        u = unit_cache.get(tid, _UNSET)
        if u is not _UNSET:
            return u
        s, c = cent[tid]
        u = _unit(s / c)
        unit_cache[tid] = u
        return u

    pending: list[str] = []          # 本批内新建、还没进快照矩阵的簇
    for start in range(0, len(order), TOPIC_CHUNK):
        batch = order[start:start + TOPIC_CHUNK]
        snap = [t for t in cent if t not in pending]
        if snap:
            cent_mat = np.vstack([_cent_unit(t) for t in snap])
            # 已满的簇不再吸附：巨型筐的克星（见 TOPIC_MAX_MEMBERS 的注释）
            full = np.array(
                [len(out[t].get("member_ids") or []) >= mm for t in snap],
                dtype=bool)
        else:
            cent_mat = np.zeros((0, 1), dtype=np.float64)
            full = np.zeros((0,), dtype=bool)
        mat = np.vstack([norms[str(it.get("id"))] for it in batch])
        sims = (mat @ cent_mat.T) if cent_mat.shape[0] else \
            np.zeros((len(batch), 0), dtype=np.float64)
        if full.any():
            sims[:, full] = -np.inf
        for row, it in enumerate(batch):
            mid = str(it.get("id"))
            if mid in assigned:
                continue
            nv = norms[mid]
            best_tid, best_sim = "", -2.0
            if sims.shape[1]:
                j = int(np.argmax(sims[row]))
                v = float(sims[row][j])
                if np.isfinite(v):
                    best_tid, best_sim = snap[j], v
            for tid in pending:          # 批内新建的簇：补算（数量很少）
                c = _cent_unit(tid)
                if c is None:
                    continue
                s = float(np.dot(c, nv))
                if (s > best_sim
                        and len(out[tid].get("member_ids") or []) < mm):
                    best_tid, best_sim = tid, s
            # 再查一次上限：``full`` 掩码只在**批开始**算过一次，同一批内
            # 一个簇能从 mm 一路长到 mm+批大小（实测 300 → 563）。
            if (best_tid and best_sim >= thr
                    and len(out[best_tid].get("member_ids") or []) < mm):
                out[best_tid]["member_ids"].append(mid)
                cent[best_tid] = [cent[best_tid][0] + nv, cent[best_tid][1] + 1]
                unit_cache.pop(best_tid, None)   # 质心变了，缓存作废
                assigned.add(mid)
            elif live_n < cap:
                t = new_topic(mid, ep)
                out[t["id"]] = t
                cent[t["id"]] = [nv.copy(), 1]
                unit_cache[t["id"]] = _unit(nv.copy())   # 已知值，省一次重算
                pending.append(t["id"])
                assigned.add(mid)
                live_n += 1
            elif best_tid:
                # 触顶：不新建，强制归入最相近的（触顶次数由 stats 自曝）
                out[best_tid]["member_ids"].append(mid)
                cent[best_tid] = [cent[best_tid][0] + nv, cent[best_tid][1] + 1]
                unit_cache.pop(best_tid, None)   # 质心变了，缓存作废
                assigned.add(mid)
            # cap <= 0 且无簇可归 → 留在 unassigned，不静默新建
        pending = []
    return out


def reclip(topics: dict[str, dict[str, Any]], alive_ids: Iterable[str]) -> int:
    """按**当前存活**的记录重裁成员（派生层纪律：并集只会做加法）。

    L1 里被删除 / 作废 / 归档的记录必须从主题里摘掉，否则"知识点"会继续
    引用一条已经不算数的记忆——而它又不进检索路径，用户根本看不见这个
    不一致（静默是最坏的结果）。

    Args:
        topics: 主题表（原地修改）。
        alive_ids: 当前仍应计入的成员 id（由调用方按 active 状态筛出）。

    Returns:
        摘掉的成员引用条数。
    """
    alive = set(alive_ids)
    dropped = 0
    for t in topics.values():
        before = list(t.get("member_ids") or [])
        after = [m for m in before if m in alive]
        dropped += len(before) - len(after)
        t["member_ids"] = after
        t["size"] = len(after)
        if not after:
            t["status"] = STATUS_ARCHIVED
    return dropped


# ----------------------------------------------------------------------
# 刷新：标题 / 跨度 / 代表句 / 状态
# ----------------------------------------------------------------------

def _top_entities(members: Sequence[dict[str, Any]], limit: int) -> list[str]:
    cnt: dict[str, int] = {}
    for m in members:
        for e in (m.get("entities") or []):
            if not e:
                continue
            cnt[str(e)] = cnt.get(str(e), 0) + 1
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:limit]]


def _top_tags(members: Sequence[dict[str, Any]], limit: int) -> list[str]:
    cnt: dict[str, int] = {}
    for m in members:
        tags = m.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        for k, v in tags.items():
            vals = v if isinstance(v, (list, tuple)) else [v]
            for one in vals:
                key = f"{k}={one}"
                cnt[key] = cnt.get(key, 0) + 1
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:limit]]


def _title_of(members: Sequence[dict[str, Any]]) -> str:
    """确定性标题：受信任实体 > 结构化标签 > 最早成员正文片段。

    刻意**不做**"取最长的句子当标题"那类启发式：它会让标题随成员增删抖动，
    而标题是地图里唯一被人眼反复读的字段，抖动比朴素更伤。
    """
    ents = _top_entities(members, TOPIC_TITLE_ENTITIES)
    if ents:
        return "·".join(ents)
    tags = _top_tags(members, 1)
    if tags:
        return tags[0]
    if members:
        oldest = min(members, key=lambda m: (float(m.get("timestamp") or 0.0),
                                             str(m.get("id"))))
        return (str(oldest.get("content") or "").strip()[:TOPIC_TITLE_CLIP]
                or "(未命名)")
    return "(未命名)"


def _collision_suffix(topic: dict[str, Any],
                      members: Sequence[dict[str, Any]]) -> str:
    """撞名时给标题补的区分后缀。

    真库实测痛点：多个簇标题逐字相同（实测库里有两条同名簇），
    宿主在地图上**分不清指的是哪个**，也没法点名让它整理。

    后缀优先级（都取确定性来源，不引入随机/模型）：
      1. 原标题没用到的**下一个受信任实体** ⇒ 「sv·fitness·P-43」
      2. 结构标签里排第一且原标题里没有的 ⇒ 「sv·fitness·类型=实验」
      3. 都没有 ⇒ 序号「#2」（至少可指名，不至于两个同名条目）

    刻意**不改**宿主命名过的标题（``title_kind=host``）：那是人起的名字，
    擅自加后缀是"最伤信任的静默改写"那一类——它只拿 ``title_collision`` 提示。
    """
    base = str(topic.get("title") or "")
    for ent in _top_entities(members, TOPIC_TITLE_ENTITIES + 3):
        if ent and ent not in base:
            return f"·{ent}"
    for tag in _top_tags(members, 3):
        if tag and tag not in base:
            return f"·{tag}"
    return ""


def disambiguate_titles(topics: dict[str, dict[str, Any]],
                        items_by_id: dict[str, dict[str, Any]] | None = None) -> int:
    """给同名主题补区分后缀，返回**被改过标题**的条数。

    - 分组键＝逐字标题；只有组内 >1 条才处理（单条不加后缀，避免无谓抖动）。
    - 组内顺序按 ``(size 降序, topic id 升序)`` ⇒ 确定性，**第一条保留原标题**
      （大簇不动，小簇改名，成员增删时改动面最小）。
    - 宿主命名的标题**不改**，只标 ``title_collision``（见 :func:`_collision_suffix`）。
    - 同名数写进 ``title_collision`` 自曝：宿主看到 ``=3`` 就知道"还有两个同名的"，
      即便它手头这条是被改名后的那个。
    """
    groups: dict[str, list[str]] = {}
    for tid, t in topics.items():
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "")
        if not title or title == "(未命名)":
            continue
        groups.setdefault(title, []).append(str(tid))

    changed = 0
    for title, tids in groups.items():
        if len(tids) < 2:
            continue
        tids.sort(key=lambda x: (-(int(topics[x].get("size") or 0)), x))
        for idx, tid in enumerate(tids):
            t = topics[tid]
            t["title_collision"] = len(tids)
            if idx == 0 or str(t.get("title_kind") or "") == TITLE_KIND_HOST:
                continue
            members = []
            if items_by_id:
                members = [items_by_id[m] for m in (t.get("member_ids") or [])
                           if m in items_by_id]
            suffix = _collision_suffix(t, members) or f"#{idx + 1}"
            t["title"] = f"{title}{suffix}"
            changed += 1
    return changed


def _extractive_summary(
    topic: dict[str, Any],
    members: Sequence[dict[str, Any]],
    norms: dict[str, np.ndarray],
    topn: int,
) -> tuple[str, list[str]]:
    """抽取式"代表句"：取离质心最近的 N 条原文，不是归纳。

    命名与实现都刻意保守——它回答的是"这个主题大概在说什么"，**不是**
    "这几条能推出什么结论"。后者要等宿主写回（``summary_kind=host``）。
    把摘录当成归纳展示出去，是这一层最容易发生的过度承诺。
    """
    ids = [m for m in (topic.get("member_ids") or []) if m in norms]
    if not ids or topn <= 0:
        return "", []
    c = _centroid(ids, norms)
    scored: list[tuple[float, str]] = []
    for mid in ids:
        v = norms.get(mid)
        if v is None:
            continue
        s = float(np.dot(c, v)) if c is not None else 0.0
        scored.append((s, mid))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    picked = scored[:topn]
    content_by_id = {str(m.get("id")): str(m.get("content") or "") for m in members}
    parts: list[str] = []
    for _s, mid in picked:
        txt = " ".join(content_by_id.get(mid, "").split())[:TOPIC_SUMMARY_CLIP]
        if txt:
            parts.append(txt)
    return " ｜ ".join(parts), [mid for _s, mid in picked]


# ----------------------------------------------------------------------
# 假簇过滤：正文信息量判据
# ----------------------------------------------------------------------

# 应答/套话。它们聚在一起是因为**都短且都是应答**，不是因为讲同一件事。
# 全是 ASCII 小写的归一形（去标点、去空白后比较）。
_FILLER_PHRASES = frozenset({
    "好的", "好", "收到", "收到收到", "明白", "了解", "知道了", "嗯", "嗯嗯",
    "哦", "行", "可以", "是的", "对", "对对", "不错", "谢谢", "多谢", "感谢",
    "继续", "稍等", "等等", "等等看", "ok", "okay", "yes", "no", "hi", "hello",
    "哈哈", "呵呵", "嗯好", "好的好的", "行吧", "没问题", "没问题的", "okok",
})

_PUNCT_RE = re.compile(r"[^\w一-鿿]+")
_DIGIT_RE = re.compile(r"\d")


def _fold_repeat(s: str) -> str:
    """折叠整串重复：「好的好的」→「好的」、「okok」→「ok」。

    套话最常见的形态就是重复（附和时会连打两遍），不折叠就漏判。
    """
    for n in (2, 3, 4):
        if len(s) > n and len(s) % n == 0:
            part = s[:n]
            if s == part * (len(s) // n):
                return part
    return s


def info_score(content: Any, entities: Any = None, tags: Any = None) -> int:
    """一条正文的**信息量分**（0 = 没有可归纳的东西）。

    判据全确定性、不引模型——这一层是给宿主挑"值得归纳什么"的，判据本身
    要是黑箱，就没法解释为什么某个主题没被挑上。

    加分项（可累加）：
      * **保底 1 分**：只要不是套话、不是空的，就算"有内容"；
      * 归一化后 ≥20 字 +1；
      * 含阿拉伯数字 +1（硬信息；中文数字刻意不算——中文库里太常见，
        算进来就没有区分度了）；
      * 带实体或标签 +1（讲到了具体对象）。

    直接判 0 的：空、套话表命中（含重复折叠后命中）。

    🔴 **为什么给保底分（实测踩出来的，不能改回去）**：第一版按"≥8 字才
    +1"判，结果 `test_knowledge_tool_face` 的样本文本「青蒿 工艺 要点」
    （6 字）被判 0 分 ⇒ 所有真主题被判成假簇 ⇒ **pending 直接归零**。
    **短 ≠ 没有信息量**——知识点标题天生就短。判"有没有信息量"和判
    "是不是套话"是两件事，只有后者该判 0。
    """
    text = " ".join(str(content or "").split())
    if not text:
        return 0
    norm = _PUNCT_RE.sub("", text).lower()
    if not norm:
        return 0
    folded = _fold_repeat(norm)
    if folded in _FILLER_PHRASES or norm in _FILLER_PHRASES:
        return 0
    if len(norm) <= 1:
        return 0
    # 保底 1 分：不是套话就算有内容（见 docstring 那条实测教训）
    score = 1
    if len(norm) >= 20:
        score += 1
    if _DIGIT_RE.search(text):
        score += 1
    if entities or tags:
        score += 1
    return score


def info_ratio_of(members: Sequence[dict[str, Any]]) -> float:
    """簇内**有信息量成员的占比**（0.0–1.0）。空簇返回 0.0。"""
    if not members:
        return 0.0
    ok = 0
    for m in members:
        if info_score(m.get("content"), m.get("entities"), m.get("tags")) \
                >= TOPIC_LOW_INFO_MIN_SCORE:
            ok += 1
    return round(ok / len(members), 4)


def low_info_topic_ids(topics: dict[str, dict[str, Any]]) -> list[str]:
    """被判为假簇（低信息量）的主题 id，按地图顺序。

    给 stats 自曝用。🔴 **必须自曝**：这些簇被排除出 pending 了，不说就是
    静默消失——宿主会以为"都归纳完了"。
    """
    return [str(t.get("id")) for t in ordered(topics)
            if t.get("low_info")]


def layer_epoch(topics: dict[str, Any] | None) -> int:
    """这一层属于第几代 = 层内所有簇的最大 ``_epoch``（没有则 0）。

    合并时用它判「谁更新鲜」：**高的整代覆盖，同代才走键并集**。
    取 max 而不是取「层顶字段」，是因为层本身是 ``{topic_id: topic}``，
    往层里塞一个非簇的顶层键会被当成一簇（``len(topics)``、地图、
    ``host_summary_stats`` 全都遍历 values）——污染面太大。
    """
    if not isinstance(topics, dict) or not topics:
        return 0
    best = 0
    for t in topics.values():
        if not isinstance(t, dict):
            continue
        try:
            e = int(t.get(TOPIC_EPOCH_FIELD) or 0)
        except (TypeError, ValueError):
            e = 0
        if e > best:
            best = e
    return best


def migrate_host_summaries(
    old: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    min_overlap: float = 0.5,
) -> list[dict[str, Any]]:
    """重建时把旧的**宿主归纳**按成员重叠搬到新簇上。

    归纳是宿主花力气写的、本系统永远不会自动生成——重建把簇边界重画
    一遍就把它们全丢了，是最伤的一类沉默。所以哪怕边界变了也要搬，
    搬过去之后**故意不重算签名**：成员数与签名对不上 ⇒ 地图把它标成
    stale（⚠️）⇒ 提醒宿主「这条的依据变过，需要复核」。这比假装它
    还是原来那条要诚实。

    Returns:
        ``[{"from", "to", "overlap"}]``，没搬成的**不在列表里**——
        调用方要用它自曝"丢了几条"，不能静默。
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return []
    moved: list[dict[str, Any]] = []
    taken: set[str] = set()
    for oid, ot in old.items():
        if not isinstance(ot, dict):
            continue
        if str(ot.get("summary_kind") or "") != SUMMARY_KIND_HOST:
            continue
        if not str(ot.get("summary") or "").strip():
            continue
        om = {str(x) for x in (ot.get("member_ids") or [])}
        if not om:
            continue
        best_id, best_score = "", 0.0
        for nid, nt in new.items():
            if nid in taken or not isinstance(nt, dict):
                continue
            # 新簇自己已有归纳就不抢：旧归纳的边界更可疑
            if str(nt.get("summary_kind") or "") == SUMMARY_KIND_HOST:
                continue
            nm = {str(x) for x in (nt.get("member_ids") or [])}
            if not nm:
                continue
            score = len(om & nm) / float(len(om))
            if score > best_score:
                best_id, best_score = nid, score
        if not best_id or best_score < float(min_overlap):
            continue
        nt = new[best_id]
        nt["summary"] = ot.get("summary", "")
        nt["summary_kind"] = SUMMARY_KIND_HOST
        nt["summary_at"] = ot.get("summary_at", 0.0)
        nt["summary_evidence"] = list(ot.get("summary_evidence") or [])
        # 签名/规模**照抄旧的**：与新簇成员对不上 ⇒ 判 stale ⇒ 提示复核。
        nt["summary_signature"] = ot.get("summary_signature", "")
        nt["summary_member_count"] = ot.get("summary_member_count", 0)
        taken.add(best_id)
        moved.append({"from": str(oid), "to": str(best_id),
                      "overlap": round(best_score, 3)})
    return moved


def refresh(
    topics: dict[str, dict[str, Any]],
    items_by_id: dict[str, dict[str, Any]],
    topn: int | None = None,
    now: float | None = None,
    defaults: TopicDefaults | None = None,
) -> None:
    """重算派生字段：标题 / 跨度 / 实体 / 标签 / 代表句 / 状态。

    **只覆盖抽取式摘要**：宿主写回的（``summary_kind=host``）不动，否则
    一次巩固就把人花力气写的归纳冲掉了——那是最伤信任的一类静默改写。

    Args:
        topics: 主题表（原地修改）。
        items_by_id: ``id -> item``（只要存活的成员）。
        topn: 代表句条数，默认取 env / 3。
        now: 时间戳基准（测试可注入固定值）。
    """
    n = _def(defaults).summary_topn if topn is None else int(topn)
    stamp = now if now is not None else 0.0
    norms: dict[str, np.ndarray] = {}
    for k, it in items_by_id.items():
        u = _unit(it.get("vec"))
        if u is not None:
            norms[str(k)] = u
    for t in topics.values():
        before = list(t.get("member_ids") or [])
        ids = [m for m in before if m in items_by_id]
        members = [items_by_id[m] for m in ids]
        # 成员表随真相**收缩**：这是 reclip/build 之外的第三条路径（调用方
        # 可能直接传了只含存活项的 items_by_id）。不收缩的话，失效 id 会永久
        # 留在成员里——覆盖率算高、签名算歪、地图上的 size 也偏大。
        t["member_ids"] = ids
        t["size"] = len(ids)
        lost = len(before) - len(ids)
        if not ids:
            t["status"] = STATUS_ARCHIVED
            t["summary"] = ""
            t["summary_evidence"] = []
            t["summary_at"] = 0.0
            t["summary_signature"] = ""
            t["summary_member_count"] = 0
            continue
        # stale 描述的是「刚发现成员变了」，不是终身标签。每次刷新都从
        # active 重判——否则一次抖动会让这个主题**永久** stale，而地图上
        # stale 是要提醒人去看的，狼来了两次就没人看了。
        t["status"] = STATUS_ACTIVE
        # 宿主命名过的标题不重算。兜底标题（最早成员前 12 字）是实测
        # 痛点——多簇同名宿主分不清；而宿主在 fras.f 里起过名字的话，一次
        # 巩固就把它抹回兜底，正是"最伤信任的静默改写"那一类。
        if str(t.get("title_kind") or "") != TITLE_KIND_HOST:
            t["title"] = _title_of(members)
        ts = [float(m.get("timestamp") or 0.0) for m in members]
        t["span"] = [min(ts), max(ts)]
        t["entities"] = _top_entities(members, 3)
        t["tags"] = _top_tags(members, 3)
        # 假簇判据：**正文信息量**。只标不删——删了会让成员重组成
        # 新簇（宿主已写归纳的签名随之对不上），且删是真删除、没有墓碑。
        # 标了之后由 :func:`pending_summary_ids` 决定要不要挑它去归纳。
        ir = info_ratio_of(members)
        t["info_ratio"] = ir
        t["low_info"] = bool(ir < TOPIC_LOW_INFO_RATIO)
        if not t.get("created_at"):
            t["created_at"] = min(ts)
        t["updated_at"] = stamp or max(ts)
        # 只填抽取式：宿主写回的归纳不覆盖（见 docstring）
        if t.get("summary_kind") != SUMMARY_KIND_HOST:
            text, ev = _extractive_summary(t, members, norms, n)
            t["summary"] = text
            t["summary_evidence"] = ev
            t["summary_kind"] = SUMMARY_KIND_EXTRACTIVE if text else ""
        # 失效占比：这次刷新发现有多少成员已经不算数了（before 与 ids 的差）
        total = lost + len(ids)
        if lost and total and (lost / total) > TOPIC_STALE_RATIO:
            t["status"] = STATUS_STALE

    # 撞名消歧放在**所有标题都算完之后**——逐条处理时看不到别的簇
    # 用了什么名字，边算边改会让结果依赖遍历顺序。
    disambiguate_titles(topics, items_by_id)


# ----------------------------------------------------------------------
# 统计（给 brain_stats / knowledge_map 用）
# ----------------------------------------------------------------------

def coverage_of(topics: dict[str, dict[str, Any]], active_total: int) -> dict[str, Any]:
    """覆盖率自曝。

    空库报 ``None`` + ``unknown``，**不报 0**——"0 条"和"无从判断"是两件事，
    报 0 会让宿主以为"这个库真的什么都没归进去"。
    """
    assigned = len({m for t in topics.values() for m in (t.get("member_ids") or [])})
    if not active_total:
        return {
            "topics": len(topics),
            "assigned": assigned,
            "active_total": 0,
            "coverage": None,
            "coverage_level": "unknown",
            "unassigned": None,
            "note": "库内没有 active 记忆，覆盖率无意义",
        }
    cov = assigned / active_total
    note = ""
    if cov > 1.0:
        # 分子超过分母 ⇒ 有主题仍引用非 active 记录（重裁没跑 / 跑晚了）。
        # 报 >100% 把人当傻子，静默截断又藏住真问题 ⇒ 截断 + 自曝。
        note = ("assigned 超过 active 总数：有主题仍引用非 active 记录"
                "（重裁未跑或跑晚了）")
        cov = 1.0
    level = "ok" if cov >= 0.8 else ("low" if cov >= 0.5 else "very_low")
    return {
        "topics": len(topics),
        "assigned": assigned,
        "active_total": active_total,
        "coverage": round(cov, 4),
        "coverage_level": level,
        "unassigned": active_total - min(assigned, active_total),
        "note": note,
    }


def by_status(topics: dict[str, dict[str, Any]]) -> dict[str, int]:
    out = {STATUS_ACTIVE: 0, STATUS_STALE: 0, STATUS_ARCHIVED: 0}
    for t in topics.values():
        key = str(t.get("status") or STATUS_ACTIVE)
        out[key] = out.get(key, 0) + 1
    return out


def ordered(topics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """地图排序：规模 × 新近 × 被召回次数（确定性，同分时按 id 兜底）。"""
    return sorted(
        topics.values(),
        key=lambda t: (-int(t.get("size") or 0),
                       -float(t.get("updated_at") or 0.0),
                       -int(t.get("recall_hits") or 0),
                       str(t.get("id"))),
    )

# ----------------------------------------------------------------------
# 宿主归纳写回
# ----------------------------------------------------------------------
#
# 本模块自身只给得出「代表句」——离质心最近的几条原文摘录，它回答的是「这个主题
# 大概在说什么」。用户要的「上层知识」是**归纳**：把 N 条事实写成一句结论。
# 那一步需要智能，而生产路径不内嵌 LLM 是本项目的边界，
# 所以做法是一个**写回通道**：hippo 出素材 → 宿主归纳 → 写回落库。
#
# 这一层因此有两种 summary，必须分得清、且不能被互相覆盖：
#   * extractive —— 本模块随时可重算，丢了不心疼；
#   * host       —— 宿主花力气写的，**任何自动流程都不许覆盖它**。
# 所以这里只提供「写 / 撤 / 查状态」，不提供「自动改写」。

# 归纳文本长度上限。层文件随它膨胀，且过长的"归纳"通常说明宿主把素材
# 原样粘回来了——那还不如不写（直接读代表句即可）。
SUMMARY_MAX_LEN = 2000
# 出处条数上限。这一层的价值在于**能回指 L1**，所以出处必须精简到"够验证"。
SUMMARY_EVIDENCE_MAX = 20
# 素材（brief）每条正文截断长度。
TOPIC_EVIDENCE_CLIP = 200

# ----------------------------------------------------------------------
# 宿主归纳的轻量 FRAS 四字段契约
# ----------------------------------------------------------------------
# 借鉴前作创作契约的 FRAS 四字段（F 职责 / R 强关系 / A 契约 /
# S 不可推断约束），按主题层形态裁剪为四键：
#
#   f —— 范畴：这个主题是什么（一句话，直接当标题用）
#   r —— 关联：必须一起看的实体 / 相关主题（list）
#   a —— 可引用的结论：这一条的本体（maps/开场包显示的那句话）
#   s —— 不可推断的约束：a 推不出来、但搞错就出事的例外与边界
#
# 两个设计纪律：
#   ① **S 的准入标准是"推不出来 + 防错"**，不许复述 a——确实没有也要显式
#      写"无"，空着会被当成没填而整条拒绝；
#   ② **结构不过就整条拒绝**（一个字段都不改）——改一半比不改更难查。
# 为什么值得加：实测的两个痛点——标题兜底取"最早成员前 12 字"导致
# 多簇同名宿主分不清；簇的"纯度"会被写法相似骗过。逼宿主写出 f/s 是已知
# 最便宜的过滤器：**写不出 s 的簇大概率不是真主题**。
FRAS_KEYS = ("f", "r", "a", "s")
FRAS_F_MAX = 80       # 范畴是标题级：一行放得下
FRAS_R_MAX = 10       # 关联条目数上限
FRAS_S_MAX = 400      # 约束字段上限（a 复用 SUMMARY_MAX_LEN）
#: 标题被宿主命名过的标记。没有它，refresh 每次都会用 _title_of 重算标题，
#: 宿主起的名字（以及撞名区分）会被一次巩固静默抹掉。
TITLE_KIND_HOST = "host"


def validate_fras(fras: Any) -> tuple[dict[str, Any] | None, str]:
    """校验四字段结构。返回 ``(parsed, error)``；error 非空即整条拒绝。

    ``fras=None`` 是合法输入（老宿主只写自由文本），返回 ``(None, "")``。
    """
    if fras is None:
        return None, ""
    if not isinstance(fras, dict):
        return None, "fras 必须是对象（f / r / a / s 四个键）"
    missing = [k for k in FRAS_KEYS if k not in fras]
    if missing:
        return None, (f"fras 缺字段 {missing}：要么四键齐全，要么整个不传"
                      "（自由文本 summary 仍然支持）")
    f = " ".join(str(fras.get("f") or "").split())
    a = " ".join(str(fras.get("a") or "").split())
    s = " ".join(str(fras.get("s") or "").split())
    r_raw = fras.get("r")
    if isinstance(r_raw, str):
        r = [x.strip() for x in r_raw.split(",") if x.strip()]
    elif isinstance(r_raw, (list, tuple)):
        r = [str(x).strip() for x in r_raw if str(x).strip()]
    else:
        r = []
    if not f:
        return None, "fras.f 为空：范畴是这一条的标题，不能省"
    if len(f) > FRAS_F_MAX:
        return None, f"fras.f 过长（{len(f)} > {FRAS_F_MAX}），范畴要一行放得下"
    if not a:
        return None, "fras.a 为空：可引用的结论是这一条的本体"
    if len(a) > SUMMARY_MAX_LEN:
        return None, f"fras.a 过长（{len(a)} > {SUMMARY_MAX_LEN}），请压成结论"
    if not s:
        return None, ("fras.s 为空：准入标准是「推不出来 + 防错」的约束，"
                      "确实没有也要显式写「无」——空着与没填分不开")
    if len(s) > FRAS_S_MAX:
        return None, f"fras.s 过长（{len(s)} > {FRAS_S_MAX}）"
    if len(r) > FRAS_R_MAX:
        return None, f"fras.r 过多（{len(r)} > {FRAS_R_MAX}），只留强关联"
    return {"f": f, "r": r, "a": a, "s": s}, ""


def signature_of(member_ids: Any) -> str:
    """成员集合指纹：记录「这条归纳是在哪一批成员上做的」。

    为什么需要它：宿主写的归纳是对**当时**那批成员的判断。成员后来变了
    （新增 / 删除 / 作废），归纳可能已经不再成立。但它不能因此被自动改写
    ——refresh 不覆盖 host 是同一条纪律。所以做法是**只标注不覆盖**：签名
    对不上就在读取时报 ``stale=True``，由宿主自己决定要不要重写。
    """
    ids = sorted({str(m) for m in (member_ids or [])})
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:12]


def apply_host_summary(
    topic: dict[str, Any],
    text: str,
    evidence: Sequence[str] | None = None,
    member_ids: Sequence[str] | None = None,
    now: float = 0.0,
    fras: Any = None,
) -> tuple[bool, str]:
    """把宿主的归纳写回主题。**校验失败一律拒绝写**，不静默降级。

    四条硬校验（任一条不满足就整条退回并给出原因）：

      1. 文本非空且不超过 ``SUMMARY_MAX_LEN`` —— 空写回会把「没归纳」伪装成
         「归纳过了」，超长则会把层文件撑爆（它每次 save 都要读-合并-写）；
      2. ``evidence`` 里的 id 必须是本主题的**成员** —— 凭空引用等于伪造出
         处，而这一层相对 L2 的全部优势就是能回指到 L1；
      3. 出处条数不超过 ``SUMMARY_EVIDENCE_MAX``；
      4. （FRAS）``fras`` 给了就四键齐全（见 :func:`validate_fras`）——
         结构不过整条拒绝，一个字段都不改。

    Args:
        topic: 主题（原地修改）。
        text: 归纳文本。**只传 fras 时可留空**——那时用 ``fras.a``
            （可引用的结论）作为一句话，少让宿主写一遍。
        evidence: 出处（L1 记忆 id）。留空表示"这条结论不单挂某几条"。
        member_ids: 当前成员集合（省略则用 topic 自己的）。
        now: 写回时间戳。
        fras: 可选的 FRAS 四字段结构。给了就把标题交给 ``fras.f``
            （``title_kind=host``，refresh 不再重算标题），并整体存进
            ``summary_fras``。

    Returns:
        ``(ok, error)``。ok 为 False 时 topic **一个字段都没改**。
    """
    parsed, fras_err = validate_fras(fras)
    if fras_err:
        return False, fras_err
    txt = " ".join(str(text or "").split())
    if not txt and parsed:
        # 只给四字段：一句话结论就是 a（friction 降到最低，且口径唯一）
        txt = parsed["a"]
    if not txt:
        return False, "summary 为空：空写回会把「没归纳」伪装成「归纳过了」"
    if len(txt) > SUMMARY_MAX_LEN:
        return False, (f"summary 过长（{len(txt)} > {SUMMARY_MAX_LEN}），"
                       "请压缩成结论而不是复述素材")
    members = {str(m) for m in (member_ids if member_ids is not None
                                else (topic.get("member_ids") or []))}
    ev: list[str] = []
    for e in (evidence or []):
        s = str(e)
        if s not in members:
            return False, f"evidence 不是本主题成员：{s}"
        if s not in ev:
            ev.append(s)
    if len(ev) > SUMMARY_EVIDENCE_MAX:
        return False, f"evidence 过多（{len(ev)} > {SUMMARY_EVIDENCE_MAX}）"
    topic["summary"] = txt
    topic["summary_kind"] = SUMMARY_KIND_HOST
    topic["summary_evidence"] = ev
    topic["summary_at"] = float(now or 0.0)
    topic["summary_signature"] = signature_of(members)
    topic["summary_member_count"] = len(members)
    if parsed:
        topic["summary_fras"] = parsed
        # 标题交给宿主：兜底标题（最早成员前 12 字）是实测踩过的痛点
        # （多簇同名、宿主分不清）。title_kind 让 refresh 不再重算它。
        topic["title"] = parsed["f"]
        topic["title_kind"] = TITLE_KIND_HOST
    if now:
        topic["updated_at"] = float(now)
    return True, ""


def revert_host_summary(topic: dict[str, Any]) -> tuple[bool, str]:
    """撤销宿主归纳（回退到抽取式代表句）。

    写错了要能撤——不然一个坏归纳会永久占着这个主题，而自动流程又不敢动它。
    清空字段后**立即生效**，下次 refresh 会填回抽取式。

    宿主命名过的标题（``title_kind=host``）跟着归零——标题是归纳的
    一部分，撤了归纳却留着宿主起的名字，下次 refresh 又不会重算它，
    主题就顶着一个谁也对应不上的标题。自动标题不动。
    """
    if str(topic.get("summary_kind") or "") != SUMMARY_KIND_HOST:
        return False, "这个主题没有宿主归纳可撤销（当前不是 host 归纳）"
    topic["summary"] = ""
    topic["summary_kind"] = ""
    topic["summary_evidence"] = []
    topic["summary_at"] = 0.0
    topic["summary_signature"] = ""
    topic["summary_member_count"] = 0
    topic.pop("summary_fras", None)
    if str(topic.get("title_kind") or "") == TITLE_KIND_HOST:
        topic.pop("title_kind", None)
        topic["title"] = ""   # 下次 refresh 用 _title_of 重填
    return True, ""


def summary_state(
    topic: dict[str, Any],
    current_member_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """这条归纳现在是什么状态（供宿主判断要不要重写）。

    ``stale=True`` 的含义是**成员集合变了**，不是"内容错了"。本模块判断不了
    内容，只负责把这个事实摆出来。
    """
    kind = str(topic.get("summary_kind") or "")
    now_ids = list(current_member_ids if current_member_ids is not None
                   else (topic.get("member_ids") or []))
    sig_now = signature_of(now_ids)
    sig_at = str(topic.get("summary_signature") or "")
    return {
        "kind": kind or "none",
        "is_host": kind == SUMMARY_KIND_HOST,
        "written_at": topic.get("summary_at") or None,
        "evidence": list(topic.get("summary_evidence") or []),
        "stale": bool(kind == SUMMARY_KIND_HOST and sig_at and sig_at != sig_now),
        "member_count_at_write": int(topic.get("summary_member_count") or 0),
        "member_count_now": len(set(str(m) for m in now_ids)),
        "signature": sig_now,
        # 四字段契约（FRAS）的回显（没写就是 None）——宿主据此确认结构落库了，
        # 也知道这条归纳有没有把"范畴/关联/结论/约束"分开记。
        "has_fras": bool(topic.get("summary_fras")),
        "fras": topic.get("summary_fras") or None,
        "title_kind": str(topic.get("title_kind") or ""),
    }


def representative_ids(
    topic: dict[str, Any],
    items_by_id: dict[str, dict[str, Any]],
    limit: int,
) -> list[str]:
    """归纳素材的顺序：**离质心最近**优先。

    为什么不是按时间：素材服务的是"这个主题在说什么"，代表性问题用代表性
    排序直接回答；时间序会把最早一批（可能已被后续推翻）顶在最前面，而它们
    恰恰最占 token。需要看演进时宿主可以自己按 span 去 recall。

    只返回**当前还活着**的成员（传进来的 items_by_id 就是 active 的）。
    """
    ids = [m for m in (topic.get("member_ids") or []) if m in items_by_id]
    if not ids or limit <= 0:
        return []
    norms: dict[str, np.ndarray] = {}
    for mid in ids:
        u = _unit(items_by_id[mid].get("vec"))
        if u is not None:
            norms[mid] = u
    c = _centroid([m for m in ids if m in norms], norms)
    scored: list[tuple[float, str]] = []
    for mid in ids:
        v = norms.get(mid)
        if v is None:
            continue
        s = float(np.dot(c, v)) if c is not None else 0.0
        scored.append((s, mid))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    return [mid for _s, mid in scored[:limit]]

def pending_summary_ids(
    topics: dict[str, dict[str, Any]],
    min_size: int | None = None,
    include_stale: bool = True,
    exclude_low_info: bool | None = None,
    defaults: TopicDefaults | None = None,
) -> list[str]:
    """「还值得归纳」的主题 id（按地图顺序，确定性）。

    判定四连，全部为真才算 pending：
      1) 未归档（成员被删空的主题不算）；
      2) ``size >= min_size``（默认 :data:`TOPIC_MIN_SIZE_FOR_SUMMARY`）；
      3) 还没有 host 归纳，**或者**已有归纳但成员集合变了（``stale``）；
      4) **不是假簇**（``low_info`` 为真则跳过，见 :func:`info_score`）。

    第 3 条是「续跑」的关键：归纳是稀缺资产，已经写好且依据没变的**不该重做**
    ——重做只会把一条确认过的好结论换成一次模型抽风。

    🔴 **第 4 条必须配自曝**：被它挡掉的主题从待归纳列表里**消失**了，宿主
    看不见就不知道有这一批 ⇒ 调用方要把
    :func:`low_info_topic_ids` 的数报出去（``pending_excluded_low_info``）。
    ``low_info`` 字段缺失的旧层**一律放行**（没判过就不判，不误杀）。

    Args:
        exclude_low_info: 是否排除假簇。None = 按 :data:`ENV_TOPIC_LOW_INFO`
            （默认开），可由 ``defaults`` 的 :attr:`TopicDefaults.low_info` 覆盖。
        d: :class:`TopicDefaults` 注入包（brain 组合根装配）。
    """
    lo = TOPIC_MIN_SIZE_FOR_SUMMARY if min_size is None else int(min_size)
    flag = _def(defaults).low_info if exclude_low_info is None \
        else bool(exclude_low_info)
    out: list[str] = []
    for t in ordered(topics):
        if str(t.get("status") or "") == STATUS_ARCHIVED:
            continue
        if len(t.get("member_ids") or []) < lo:
            continue
        if flag and t.get("low_info"):
            continue
        kind = str(t.get("summary_kind") or "")
        if kind == SUMMARY_KIND_HOST:
            sig_now = signature_of(t.get("member_ids") or [])
            sig_at = str(t.get("summary_signature") or "")
            if not include_stale or not (sig_at and sig_at != sig_now):
                continue
        out.append(str(t.get("id")))
    return out


def host_summary_stats(topics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """归纳进度自曝：宿主归纳了几个主题、几个的依据成员已经变了。

    这两个数是「梳理进度」的验收口径——没有它们，只能靠感觉。
    ``host_stale`` 尤其重要：它数的是"写过归纳、但成员后来变了"的主题，
    这些是最该拿回宿主重写的一批。
    """
    kinds: dict[str, int] = {"host": 0, "extractive": 0, "none": 0}
    stale = 0
    with_fras = 0
    low_info = 0
    for t in topics.values():
        k = str(t.get("summary_kind") or "none")
        kinds[k] = kinds.get(k, 0) + 1
        if t.get("summary_fras"):
            with_fras += 1
        if t.get("low_info"):
            low_info += 1
        if (k == SUMMARY_KIND_HOST
                and t.get("summary_signature")
                and t.get("summary_signature")
                != signature_of(t.get("member_ids") or [])):
            stale += 1
    kinds["host_stale"] = stale
    # 假簇数。它进 stats 是**硬需求**不是装饰：这些簇被挡在
    # pending 之外，不报出来宿主会以为"全部归纳完了"。
    kinds["low_info"] = low_info
    # 带四字段结构的 host 归纳数。自由文本仍然合法，这一格只是让
    # "结构覆盖率"看得见——将来要按域机器处理归纳时，没有结构就没法处理。
    kinds["with_fras"] = with_fras
    return kinds
