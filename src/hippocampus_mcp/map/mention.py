"""闸三：提及级漂移检测（信息地图层）。

原设想是一句
「实体 ``last_seen`` 比条目 ``updated_at`` 新 ⇒ 标待核」，评审指出两个
必须先回答的问题（假阳性、无清除规则），否则它只会把索引行涂红。本模块
是那两个问题的答案，也是闸三从 🔲 升级的依据。

判据（三条同时成立才算信号）
--------------------------
1. **时间晚于地图**：记忆是 active 的（归档/作废不算），且
   ``timestamp`` **严格大于**条目 ``updated_at``。同时刻不算——正常流程
   是「先写事实记忆、再 map_set」，把它算成漂移会在每次正常更新后自吹
   一次警。
2. **同句共现**：条目键与某个**状态词**出现在**同一句**里（按
   ``。！？!?；;\\n`` 切句）。正文某处提到条目、隔几段说另一件事的另一
   个状态，不算——「被提到」本身不是「状态变了」的证据（评审第一条）。
3. **说另一个状态**：该状态词 ≠ 条目当前 ``status``。说「训练集」而地图
   也写训练集，那是**复述**，不是漂移。

状态词表不内置：取**该图条目自己的 status 集合**（``∪`` 图级
``mention.state_words`` 配置）。理由和 C1 的 4+4 词表同源——词表是猜的
开始，而地图里已有的词是这张图自己声明过的语义。

**单字 CJK 状态词默认不收**：``开`` 会命中「开始」、``关`` 会命中「关注」，
单字中文没有词边界，靠它判漂移等于制造假阳性。过滤后不足 2 个不同词 ⇒
该图**跳过并自曝理由**（判不了就说判不了，不判 clean）。图级
``mention.state_words`` 里显式配的词不受此限（用户声明过的语义由用户负责）。

清除规则（三条，全部有定义）
--------------------------
1. **重写即确认**：``map_set`` 同键再写、或 ``maps.py ack <图> <条目>``
   把 ``updated_at`` 推到最新 ⇒ 这些提及不再满足判据 1 ⇒ 下一轮自动清。
   「我核过了，没变」在数据上就是「把确认时刻写进 updated_at」。
2. **记忆侧消失**：提及被归档/作废（只认 active）⇒ 下一轮清。
3. **截断时只增不减**：本轮看了 N 条上限或预算用尽 ⇒ 无法证明「没有别的
   提及」，因此**不清任何旧标记**，并标 ``truncated``。给半个数会被读成
   全量，与 C2「扫描不完整不判缺失」同一纪律。

为什么另开 ``ent["mention"]`` 而不复用 ``ent["drift"]``
----------------------------------------------------
那个槽已被三个写入者争用（源校验 / 值级 verify / 目录对账），各自靠
``reason`` 前缀认领。闸三的信号强度最弱（是「有人这么说」，不是「真相
是」），**弱信号不许覆盖强事实**：文件真的变了这件事不能被一条闲聊盖掉，
反过来闲聊也不该被文件漂移的清除逻辑顺手抹掉。所以独立成键，互不踩。
"""

from __future__ import annotations

import re
import time

from .constants import (
    MAP_MENTION_BUDGET_S,
    MAP_MENTION_MAX_RECORDS,
    MAP_NAME_KEY,
)

# 句子边界：中文标点 + 英文句读 + 换行（不用正则 split 的 lookbehind，
# 这里只需要「切段」，标点本身没有语义）
_SENT_SPLIT = re.compile(r"[。！？!?；;\r\n]+")
# 状态词最大长度（防止某个 note 当成状态词塞进来，整段变成「状态词」）
_STATE_WORD_MAX = 16


# ── 记录适配（MemoryRecord 与 dict 都吃，探针不必造记录对象） ──────

def _rec_get(rec, name: str, default=None):
    if isinstance(rec, dict):
        return rec.get(name, default)
    return getattr(rec, name, default)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _key_pattern(key: str) -> re.Pattern | None:
    """ASCII 键用边界正则，CJK 键返回 None（走子串）。

    ``CN2601`` 不许命中 ``CN26010``／``xCN2601``——与 C2 键抽取一律
    fullmatch 同一条纪律：前缀/子串匹配会让「另一个键」冒充本键。
    """
    if not key.isascii():
        return None
    return re.compile(r"(?<![0-9a-z_])" + re.escape(key) + r"(?![0-9a-z_])")


def _mentions(low_text: str, key: str, rx: re.Pattern | None) -> bool:
    if rx is None:
        return key in low_text
    return rx.search(low_text) is not None


def _state_words(m: dict) -> tuple[list[str], str]:
    """该图的状态词表 + 不可判定的理由（空串 = 可用）。

    来源：条目 status 集合 ∪ ``mention.state_words``。默认丢掉单字 CJK
    （歧义），配置词不丢（显式声明由声明者负责）。不足 2 个词 ⇒ 无法
    判「说了另一个状态」，返回理由让调用方自曝跳过。
    """
    cfg = m.get("mention") if isinstance(m.get("mention"), dict) else {}
    words: list[str] = []
    seen: set[str] = set()

    def _add(w: object, from_cfg: bool) -> None:
        s = str(w or "").strip()
        if not s or len(s) > _STATE_WORD_MAX:
            return
        if not from_cfg and len(s) == 1 and not s.isascii():
            return                      # 单字中文：开→开始、关→关注
        k = s.lower()
        if k in seen:
            return
        seen.add(k)
        words.append(s)

    for ent in (m.get("entries") or {}).values():
        if isinstance(ent, dict):
            _add(ent.get("status"), False)
    for w in (cfg.get("state_words") or []) if isinstance(cfg, dict) else []:
        _add(w, True)
    if len(words) < 2:
        return words, ("状态词表不足（%d 个可用状态词）——该图条目状态里没有"
                       "两个以上可区分的词，闸三判不出「说了另一个状态」；"
                       "单字中文状态词歧义太大（开/关），请在 map 级 "
                       "mention.state_words 里补多字状态词" % len(words))
    return words, ""


def _find_mention(content: str, key: str, status: str,
                  words: list[str]) -> tuple[str, str] | None:
    """返回 ``(状态词, 所在句)``；不构成信号返回 None。"""
    low = content.lower()
    klow = key.lower()
    rx = _key_pattern(klow)
    if not _mentions(low, klow, rx):
        return None
    cur = status.strip().lower()
    other = [w for w in words if w.lower() != cur]
    if not other:
        return None
    for sent in _sentences(content):
        s_low = sent.lower()
        if not _mentions(s_low, klow, rx):
            continue                    # 键不在这句里：同句共现不成立
        for w in other:
            if w.lower() in s_low:
                return w, sent
    return None


# ── 主流程 ──────────────────────────────────────────────────────

def run(data: dict, records, now: float | None = None,
        budget_s: float | None = None, max_records: int | None = None) -> dict:
    """跑一遍闸三。**只改 data（内存）**，落盘由调用方在锁内决定。

    ``records`` 是记忆记录序列（``MemoryRecord`` 或同形 dict）。只读，
    不检索、不引模型——闸三的全部成本是字符串匹配。
    """
    now = time.time() if now is None else now
    budget = MAP_MENTION_BUDGET_S if budget_s is None else float(budget_s)
    max_rec = (MAP_MENTION_MAX_RECORDS if max_records is None
               else int(max_records))
    t0 = time.perf_counter()
    deadline = t0 + budget

    maps = data.get("maps") or {}
    res: dict = {"maps": 0, "entries": 0, "checked": 0, "flagged": 0,
                 "cleared": 0, "scanned": 0, "truncated": False,
                 "skipped": [], "flagged_list": [], "budget_s": budget,
                 "max_records": max_rec, "at": now}

    # 先挑出参与本轮的记忆：active、不是地图条目本身（那是地图数据不是
    # 自由文本提及，否则 legacy 图会自己举报自己）、时间晚于最早的
    # updated_at（更早的记不可能是任何条目的「更新后的说法」）。
    cutoffs = [float((e or {}).get("updated_at") or 0.0)
               for m in maps.values()
               for e in ((m or {}).get("entries") or {}).values()]
    if not cutoffs:
        res["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return res
    cutoff = min(cutoffs)
    cands: list[tuple[float, str, str]] = []
    total_cands = 0
    for rec in records or ():
        if str(_rec_get(rec, "state", "active")) != "active":
            continue
        tags = _rec_get(rec, "tags") or {}
        if isinstance(tags, dict) and tags.get(MAP_NAME_KEY):
            continue                    # 地图条目本体，不是提及
        ts = float(_rec_get(rec, "timestamp", 0.0) or 0.0)
        if ts <= cutoff:
            continue
        total_cands += 1
        cands.append((ts, str(_rec_get(rec, "id", "") or ""),
                      str(_rec_get(rec, "content", "") or "")))
    cands.sort(key=lambda x: x[0])
    if len(cands) > max_rec:
        # 只留最近的 max_rec 条：旧提及的价值低于「本轮看了什么」的
        # 可解释性；截断即自曝，且不清旧标记（见 docstring 清除规则 3）
        cands = cands[-max_rec:]
        res["truncated"] = True
    res["scanned"] = len(cands)

    for mname in sorted(maps):
        m = maps.get(mname) or {}
        ents = m.get("entries") or {}
        if not ents:
            continue
        res["maps"] += 1
        words, why = _state_words(m)
        state = {"at": now, "scanned": len(cands),
                 "truncated": bool(res["truncated"])}
        if not map_enabled(m):
            state.update({"checked": 0, "flagged": 0, "cleared": 0,
                          "skipped": "图级关闭（mention.enabled=false）"})
            m["mention_state"] = state
            res["skipped"].append({"map": mname,
                                   "reason": "图级关闭（mention.enabled=false）"})
            continue
        if why:
            # 判不了就明说：不判 clean、也不判漂移
            state.update({"checked": 0, "flagged": 0, "cleared": 0,
                          "skipped": why})
            m["mention_state"] = state
            res["skipped"].append({"map": mname, "reason": why})
            continue

        checked = flagged = cleared = 0
        for key in sorted(ents):
            ent = ents.get(key)
            if not isinstance(ent, dict):
                continue
            status = str(ent.get("status") or "").strip()
            if not status:
                continue                # 没有状态就没有「另一个状态」可言
            if time.perf_counter() > deadline:
                res["truncated"] = True
                state["truncated"] = True
                break
            checked += 1
            res["entries"] += 1
            since = float(ent.get("updated_at") or 0.0)
            hit = None
            for i, (ts, rid, content) in enumerate(cands):
                if ts <= since:
                    continue
                if i % 200 == 0 and time.perf_counter() > deadline:
                    res["truncated"] = True
                    state["truncated"] = True
                    break
                got = _find_mention(content, key, status, words)
                if got:
                    hit = (ts, rid, got[0], got[1], content)
                    break
            if hit:
                ts, rid, word, sent, content = hit
                prev = ent.get("mention") or {}
                ent["mention"] = {
                    "state": "drifted",
                    "expect": status,
                    "observed": word,
                    # 句子截断，够人判断又不会把整条记忆塞进索引/提醒
                    "quote": sent[:80],
                    "memory_id": rid,
                    "memory_ts": ts,
                    # 只进必退：同一个提及连续几轮都成立时不重置 since，
                    # 否则「多久没人管」永远显示 0 天
                    "since": float(prev.get("since") or ts),
                    "at": now,
                }
                flagged += 1
                res["flagged"] += 1
                if len(res["flagged_list"]) < 20:
                    res["flagged_list"].append(
                        {"map": mname, "key": key, "expect": status,
                         "observed": word, "quote": sent[:80],
                         "memory_id": rid, "memory_ts": ts})
            elif ent.get("mention"):
                if res["truncated"]:
                    continue            # 截断：无法证明「没有」，保留旧标记
                ent["mention"] = None
                cleared += 1
                res["cleared"] += 1
        state.update({"checked": checked, "flagged": flagged,
                      "cleared": cleared})
        if not why:
            m["mention_state"] = state
        res["checked"] += checked

    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def set_config(m: dict, state_words: list[str] | None = None,
               enabled: bool | None = None) -> dict:
    """写图级 ``mention`` 配置（CLI 用）。``None`` = 不动该字段。

    配置只影响**词表与开关**，不改判据——判据是代码而不是参数，这是
    闸三能被信任的前提（可配置的判据等于没有判据）。
    """
    cfg = m.get("mention")
    cfg = dict(cfg) if isinstance(cfg, dict) else {}
    if state_words is not None:
        words = [str(w).strip() for w in state_words if str(w).strip()]
        if words:
            cfg["state_words"] = words
        else:
            cfg.pop("state_words", None)
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if cfg:
        m["mention"] = cfg
    else:
        m.pop("mention", None)
    # 配置变了：清掉图级统计，下一轮按新词表重算（旧统计的 checked/flagged
    # 是旧词表的结论，留着会被读成新词表的结果）
    m.pop("mention_state", None)
    return m


def map_enabled(m: dict) -> bool:
    cfg = m.get("mention")
    if isinstance(cfg, dict) and cfg.get("enabled") is False:
        return False
    return True


def needs_run(data: dict) -> bool:
    """有没有图需要跑闸三（无条目/显式关掉的图不算）。"""
    for m in (data.get("maps") or {}).values():
        if (m or {}).get("entries") and map_enabled(m):
            return True
    return False


def counts(m: dict) -> dict:
    """图级统计（给索引行 / CLI 用）：flagged = 有 mention 标记的条目数。"""
    ents = (m or {}).get("entries") or {}
    flagged = sum(1 for e in ents.values() if (e or {}).get("mention"))
    st = (m or {}).get("mention_state") or {}
    return {"flagged": flagged,
            "skipped": bool(st.get("skipped")),
            "truncated": bool(st.get("truncated")),
            "state": st}
