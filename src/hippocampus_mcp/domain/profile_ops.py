"""画像层操作族：日计数 / 画像刷新 / 失效摘除。

从 ``brain.core`` 整搬三件。**状态容器不搬**：``profile`` 与
``daily_counts`` 由壳注入本体（就地修改）；store 读、信任判据、记录流
全部**调用时现取**注入（``records_fn`` 惰性——钉住扫描在实体扫描之后
才物化）；``base_fn`` 惰性——``bump_daily`` 的 kind/n 门禁在时间基准
计算**之前**，开关关掉时一次都不调（惰性注入纪律）。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * bump_daily：kind 只认 ``"recall"`` / ``"ingest"`` 且 n>0，否则
    **直接返回不碰表**；日键是 ``%Y-%m-%d`` 本地时区；槽缺省
    ``{"recall":0,"ingest":0}``；超 ``keep_days`` 时按**键名升序**裁掉
    最老的（键是 ISO 日期，升序=最旧）；裁剪在**追加之后**做；
  * prune_stale_profile：只对 ``stable_facts`` 与 ``entity:*`` 两类槽
    下手（其它自定义槽不动）；只对 **evidence 恰好 1 条**的条目判据；
    定位不到记忆一律保留（宁可漏摘不可误摘）；reason 三态
    memory_missing / memory_inactive / unpinned（unpinned 只判
    stable_facts）；槽摘空整个 pop，非空回写 kept；
  * update_profile：**先摘失效再入库**（prune 前置）；三个来源都只收
    active——实体按 mention_count 降序取前 entity_max 个（只认信任
    origin），mention_count<3 跳过，逐条记忆建条目取最近 3 条、
    confidence=0.3*mention_count；钉住按 (importance, timestamp) 降序；
    O4 毕业来源（recurring_slot 非空才启用）＝**没人钉但反复被想起**
    （检索强度 ≥ strength_min）**且活得够久**（age ≥ strength_days）
    的条目，按 (强度, timestamp) 降序取前 strength_max 个落独立槽，
    confidence = min(0.9, BASE + STEP·强度)（校准常量在 common.constants，
    强度 3→0.48、5→0.60、10→0.90——槽内自洽排序，与钉住/实体槽互不竞争）；
    条目去重按 text 全等；槽满丢**置信度最低**的（新条目参与比较，
    打平不挤旧）；confidence 上限 0.9 且 round2；evidence 截前 3。

O4 与 U3「绝不自动提升」红线的边界：U3 管的是 project→user **跨库**
提升流（显式确认才写用户库）；毕业来源只往**库内** L3 画像的展示层
追加条目，不写记录、不改状态、不碰用户库——L3 本就随 consolidate
自动刷新，两条流判据独立、通道隔离。
"""
from __future__ import annotations

import time as _time
from typing import Any, Callable

from ..common.constants import PROFILE_STRENGTH_BASE, PROFILE_STRENGTH_STEP


def bump_daily(
    kind: str, n: int, *, daily_counts: dict, base_fn: Callable[[], float],
    keep_days: int,
) -> None:
    """按日累计一次活动（追加五）：``kind`` 取 ``"recall"`` / ``"ingest"``。

    **时间基准必须跟 consolidate 的 now 一致**（由壳经 ``base_fn``
    惰性提供 ``manager._reference_time or now``）：写裸 ``time.time()``
    在数据时间戳是过去的评测下会把计数全归到今天。
    """
    if kind not in ("recall", "ingest") or n <= 0:
        return
    base = base_fn()
    day = _time.strftime("%Y-%m-%d", _time.localtime(base))
    slot = daily_counts.setdefault(day, {"recall": 0, "ingest": 0})
    slot[kind] = int(slot.get(kind, 0)) + int(n)
    if len(daily_counts) > keep_days:
        for old in sorted(daily_counts)[
            : len(daily_counts) - keep_days
        ]:
            daily_counts.pop(old, None)


def prune_stale_profile(
    profile: dict, *, store_get: Callable[[str], Any],
    stable_slot: str, active_state: str,
    recurring_slot: str = "",
) -> list[dict[str, str]]:
    """摘除**已失效**的 L3 画像条目（zcode 追加 19②）。

    判据（只对 evidence 恰好 1 条、且能定位到那条记忆的条目下手；
    定位不到的一律保留——**宁可漏摘，不可误摘**）：``stable_facts``
    该记忆仍 pin 且 active 才留；``entity:*`` 该记忆仍 active 才留；
    O4 毕业槽（``recurring_slot``）同 entity:* 口径——仍 active 才留
    （毕业条目从未 pin，**不吃** unpinned 判据）；其它自定义槽不动。
    返回被摘除清单——**画像少了什么必须看得见**。
    """
    dropped: list[dict[str, str]] = []
    for slot, items in list(profile.items()):
        managed = (
            slot == stable_slot
            or (bool(recurring_slot) and slot == recurring_slot)
            or slot.startswith("entity:")
        )
        if not managed:
            continue
        kept: list[dict[str, Any]] = []
        for it in items:
            ev = it.get("evidence") or []
            reason = ""
            if len(ev) == 1:
                rec = store_get(ev[0])
                if rec is None:
                    reason = "memory_missing"
                elif rec.state != active_state:
                    reason = "memory_inactive"
                elif slot == stable_slot and not rec.pin:
                    # 追加 19② 的主诉：取消钉住 / importance 降级退出钉住
                    reason = "unpinned"
            if reason:
                dropped.append({
                    "slot": slot,
                    "text": (it.get("text") or "")[:80],
                    "memory_id": ev[0] if len(ev) == 1 else "",
                    "reason": reason,
                })
                continue
            kept.append(it)
        if kept:
            profile[slot] = kept
        else:
            profile.pop(slot, None)
    return dropped


def update_profile(
    profile: dict, *, prune_fn: Callable[[], list],
    entities: dict, entity_trusted: Callable[[dict], bool],
    store_get: Callable[[str], Any], records_fn: Callable[[], Any],
    now_fn: Callable[[], float], active_state: str,
    slot_max: int, entity_max: int, stable_slot: str,
    recurring_slot: str = "",
    strength_min: float = 0.0, strength_days: float = 0.0,
    strength_max: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """提炼 L3 画像：只收高置信条目，冲突只追加不覆盖。

    返回 ``(updates, dropped)``；``dropped`` 由壳落到
    ``last_profile_pruned``（刷新前先摘失效，prune 前置）。
    """
    dropped = prune_fn()
    updates: list[dict[str, Any]] = []

    def _append(slot: str, text: str, evidence: list[str],
                confidence: float) -> None:
        existing = profile.setdefault(slot, [])
        if any(f["text"] == text for f in existing):
            return
        # 槽位容量上限（覆盖报告·高 5）：超限时丢弃置信度最低的那条
        # （新条目也参与比较）。
        entry = {
            "text": text, "since": now_fn(), "evidence": evidence[:3],
            "confidence": round(min(confidence, 0.9), 2),
        }
        if len(existing) >= slot_max:
            weakest = min(range(len(existing)),
                          key=lambda i: existing[i].get("confidence", 0.0))
            if existing[weakest].get("confidence", 0.0) >= entry["confidence"]:
                return
            existing.pop(weakest)
        existing.append(entry)
        updates.append({"slot": slot, "text": text,
                        "confidence": entry["confidence"]})

    # 实体画像只取 mention_count 最高的若干个；只认受信任 origin。
    ranked_ents = sorted(
        ((k, e) for k, e in entities.items() if entity_trusted(e)),
        key=lambda kv: -int(kv[1].get("mention_count", 0)),
    )[:entity_max]
    for key, ent in ranked_ents:
        if ent["mention_count"] < 3:
            continue
        # 逐条记忆建画像条目，evidence 精确到「支撑该条目的那一条记忆」。
        records = [store_get(m) for m in ent["memory_ids"]]
        records = [
            r for r in records
            if r is not None and r.state == active_state
        ]
        records.sort(key=lambda r: -r.timestamp)
        for rec in records[:3]:
            _append(f"entity:{key}", rec.content[:120], [rec.id],
                    0.3 * ent["mention_count"])

    # 记录流只物化一次：钉住与 O4 毕业两个来源共用同一份快照
    # （records_fn 惰性——实体扫描之后才调，纪律原样）。
    recs = list(records_fn())
    # 钉住记忆 = 长期稳定的偏好 / 禁忌 / 硬约束。
    # 按 importance 降序入队：槽位有容量上限，先到先得会让"插入顺序"
    # 而不是"重要性"决定谁留下。
    pinned = [
        r for r in recs
        if r.pin and r.state == active_state
    ]
    pinned.sort(key=lambda r: (-r.importance, -r.timestamp))
    for rec in pinned:
        _append(stable_slot, rec.content[:120], [rec.id], rec.importance)

    # 访问驱动的「毕业」来源——没人钉、但反复被想起（检索强度）
    # 且活得够久（稳定期限）的 L1 条目。recurring_slot 为空 = 来源整体
    # 关闭（关闸零行为变化）；不收 pin 条目（它们已有 stable_facts 通道）。
    # 独立落槽：confidence 校准只跟同类竞争，不与钉住/实体条目互挤。
    if recurring_slot:
        now = now_fn()
        grads = [
            r for r in recs
            if not r.pin and r.state == active_state
            and float(r.recall_strength or 0.0) >= strength_min
            and (now - r.timestamp) / 86400.0 >= strength_days
        ]
        grads.sort(key=lambda r: (-float(r.recall_strength or 0.0),
                                  -r.timestamp))
        for rec in grads[:max(int(strength_max), 0)]:
            conf = (PROFILE_STRENGTH_BASE
                    + PROFILE_STRENGTH_STEP * float(rec.recall_strength or 0.0))
            _append(recurring_slot, rec.content[:120], [rec.id], conf)
    return updates, dropped
