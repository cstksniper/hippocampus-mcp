"""Gate 3: mention-level drift detection (information map layer).

The original sketch was one sentence: "entity ``last_seen`` newer than the
entry's ``updated_at`` ⇒ flag for review". Review pointed out two questions
that had to be answered first (false positives, no clearing rule); otherwise it
would only paint index rows red. This module is the answer to those two
questions, and the basis for Gate 3's upgrade from 🔲.

Criteria (all three must hold to count as a signal)
---------------------------------------------------
1. **Newer than the map**: the memory is active (archived/superseded don't
   count), and its ``timestamp`` is **strictly greater than** the entry's
   ``updated_at``. The same instant does not count — the normal flow is
   "write the fact memory first, then map_set"; counting that as drift would
   cry wolf once after every normal update.
2. **Same-sentence co-occurrence**: the entry key and some **status word**
   appear in the **same sentence** (split on ``。！？!?；;\\n``). The entry
   being mentioned somewhere in the body while a few paragraphs later another
   thing's state is discussed does not count — "being mentioned" is not by
   itself evidence that "the state changed" (review point 1).
3. **Saying another state**: that status word ≠ the entry's current
   ``status``. Saying 「训练集」 when the map also says 训练集 is
   **restatement**, not drift.

The status word table is not built in: it takes **the map entry's own status
set** (``∪`` the map-level ``mention.state_words`` config). The rationale
comes from the same source as C1's 4+4 word tables — a word table is a
guessing start, while words already in the map are semantics this map declared
itself.

**Single-character CJK status words are excluded by default**: ``开`` would
hit 「开始」, ``关`` would hit 「关注」; single-character Chinese has no word
boundaries, so using it to judge drift manufactures false positives. Fewer
than 2 distinct words after filtering ⇒ the map is **skipped with a
self-reported reason** (if it can't be judged, say so; don't judge clean).
Words explicitly configured in map-level ``mention.state_words`` are exempt
(semantics declared by the user are the user's responsibility).

Clearing rules (three, all defined)
-----------------------------------
1. **Rewrite is confirmation**: ``map_set`` rewriting the same key, or
   ``maps.py ack <图> <条目>`` pushing ``updated_at`` to the latest ⇒ those
   mentions no longer meet criterion 1 ⇒ auto-cleared next round. "I checked,
   nothing changed" is, in data terms, "write the confirmation moment into
   updated_at".
2. **The memory side disappears**: the mention gets archived/superseded (only
   active counts) ⇒ cleared next round.
3. **On truncation, add-only, never remove**: this round hit the N-record cap
   or ran out of budget ⇒ "there are no other mentions" cannot be proven, so
   **no old flags are cleared** and ``truncated`` is set. Reporting half a
   count would be read as the full picture — the same discipline as C2's
   "an incomplete scan never judges absence".

Why a separate ``ent["mention"]`` instead of reusing ``ent["drift"]``
---------------------------------------------------------------------
That slot is already contended by three writers (source verification /
value-level verify / directory reconciliation), each claiming via ``reason``
prefixes. Gate 3's signal is the weakest ("someone said so", not "the truth
is"), and **weak signals must not overwrite strong facts**: the fact that a
file really changed must not be buried by a piece of chat, and conversely chat
should not be wiped away by the file-drift clearing logic. Hence an
independent key; neither steps on the other.
"""

from __future__ import annotations

import re
import time

from .constants import (
    MAP_MENTION_BUDGET_S,
    MAP_MENTION_MAX_RECORDS,
    MAP_NAME_KEY,
)

# Sentence boundaries: Chinese punctuation + English sentence marks + newlines
# (no regex-split lookbehind; here we only need "segment cutting", the
# punctuation itself carries no semantics)
_SENT_SPLIT = re.compile(r"[。！？!?；;\r\n]+")
# Max status-word length (prevents some note from being stuffed in as a status
# word, turning a whole paragraph into a "status word")
_STATE_WORD_MAX = 16


# ── Record adaptation (accepts both MemoryRecord and dict; probes need not
# build record objects) ──────

def _rec_get(rec, name: str, default=None):
    if isinstance(rec, dict):
        return rec.get(name, default)
    return getattr(rec, name, default)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _key_pattern(key: str) -> re.Pattern | None:
    """ASCII keys use a boundary regex; CJK keys return None (substring path).

    ``CN2601`` must not hit ``CN26010``／``xCN2601`` — the same discipline as
    C2's key extraction always using fullmatch: prefix/substring matching lets
    "another key" impersonate this one.
    """
    if not key.isascii():
        return None
    return re.compile(r"(?<![0-9a-z_])" + re.escape(key) + r"(?![0-9a-z_])")


def _mentions(low_text: str, key: str, rx: re.Pattern | None) -> bool:
    if rx is None:
        return key in low_text
    return rx.search(low_text) is not None


def _state_words(m: dict) -> tuple[list[str], str]:
    """This map's status word table + the reason it is undecidable (empty string = usable).

    Sources: entry status set ∪ ``mention.state_words``. Single-character CJK
    is dropped by default (ambiguous); configured words are not dropped (an
    explicit declaration is the declarer's responsibility). Fewer than 2 words
    ⇒ cannot judge "said another state"; return the reason so the caller can
    self-report the skip.
    """
    cfg = m.get("mention") if isinstance(m.get("mention"), dict) else {}
    words: list[str] = []
    seen: set[str] = set()

    def _add(w: object, from_cfg: bool) -> None:
        s = str(w or "").strip()
        if not s or len(s) > _STATE_WORD_MAX:
            return
        if not from_cfg and len(s) == 1 and not s.isascii():
            return                      # single-char Chinese: 开→开始, 关→关注
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
    """Return ``(status word, containing sentence)``; None when it is not a signal."""
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
            continue                    # key not in this sentence: same-sentence co-occurrence fails
        for w in other:
            if w.lower() in s_low:
                return w, sent
    return None


# ── Main flow ──────────────────────────────────────────────────────

def run(data: dict, records, now: float | None = None,
        budget_s: float | None = None, max_records: int | None = None) -> dict:
    """Run Gate 3 once. **Only mutates data (in memory)**; persisting is decided by the caller under lock.

    ``records`` is a sequence of memory records (``MemoryRecord`` or dicts of
    the same shape). Read-only: no recall, no model — Gate 3's entire cost is
    string matching.
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

    # First pick the memories participating in this round: active, not map
    # entries themselves (those are map data, not free-text mentions; otherwise
    # a legacy map would report itself), and newer than the earliest
    # updated_at (an older record cannot be any entry's "post-update statement").
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
            continue                    # a map entry body, not a mention
        ts = float(_rec_get(rec, "timestamp", 0.0) or 0.0)
        if ts <= cutoff:
            continue
        total_cands += 1
        cands.append((ts, str(_rec_get(rec, "id", "") or ""),
                      str(_rec_get(rec, "content", "") or "")))
    cands.sort(key=lambda x: x[0])
    if len(cands) > max_rec:
        # Keep only the most recent max_rec records: old mentions are worth
        # less than the explainability of "what this round examined"; truncation
        # self-reports, and old flags are not cleared (see docstring clearing
        # rule 3)
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
                          "skipped": "disabled at map level (mention.enabled=false)"})
            m["mention_state"] = state
            res["skipped"].append({"map": mname,
                                   "reason": "disabled at map level (mention.enabled=false)"})
            continue
        if why:
            # If it cannot be judged, say so: judge neither clean nor drift
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
                continue                # no status means no "other state" to speak of
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
                    # Sentence truncated: enough for a human to judge without
                    # stuffing the whole memory into index/reminder
                    "quote": sent[:80],
                    "memory_id": rid,
                    "memory_ts": ts,
                    # Only-in must-retreat: when the same mention keeps holding
                    # across several rounds, do not reset since, otherwise
                    # "how long unattended" would always show 0 days
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
                    continue            # truncated: "none" cannot be proven; keep old flags
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
    """Write the map-level ``mention`` config (for the CLI). ``None`` = leave that field alone.

    Config affects only the **word table and switch**, never the criteria —
    the criteria are code, not parameters; that is the premise for Gate 3 being
    trustworthy (a configurable criterion is no criterion).
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
    # Config changed: drop the map-level stats; next round recomputes with the
    # new word table (the old stats' checked/flagged were conclusions of the
    # old table; keeping them would be read as results of the new one)
    m.pop("mention_state", None)
    return m


def map_enabled(m: dict) -> bool:
    cfg = m.get("mention")
    if isinstance(cfg, dict) and cfg.get("enabled") is False:
        return False
    return True


def needs_run(data: dict) -> bool:
    """Whether any map needs Gate 3 run (maps without entries or explicitly disabled don't count)."""
    for m in (data.get("maps") or {}).values():
        if (m or {}).get("entries") and map_enabled(m):
            return True
    return False


def counts(m: dict) -> dict:
    """Map-level stats (for index rows / CLI): flagged = number of entries with a mention flag."""
    ents = (m or {}).get("entries") or {}
    flagged = sum(1 for e in ents.values() if (e or {}).get("mention"))
    st = (m or {}).get("mention_state") or {}
    return {"flagged": flagged,
            "skipped": bool(st.get("skipped")),
            "truncated": bool(st.get("truncated")),
            "state": st}
