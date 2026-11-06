# -*- coding: utf-8 -*-
"""Generic alias discovery: find "multiple names for the same entity", delivering
only candidates and evidence, never auto-merging.

**Why this step exists**: the alias unification mechanism
(``lexicon_add("别名 => 主名")``) can already fix the data, but **discovery** is
still entirely manual — when the same entity appears in three spellings across the
corpus, a person must first read them before thinking of registering them. This
step takes the "read the whole library hunting for names" work back from humans.

**Why candidates only**: once merged, an alias **changes entity identity**; a wrong
guess blends the memories of two different objects into one bucket and is very hard
to untangle afterwards. So the output is always "candidate + signal + evidence +
confidence"; persisting is confirmed by the caller via ``lexicon_add`` — the same
stance as ``heuristic`` entities being candidates only.

Four signals, all **domain-agnostic** (the code recognizes neither "人" (person)
nor "小说" (novel), only Chinese naming conventions and statistics):

  1. ``cue``    Appellation cue words: ``甲，字乙`` / ``甲，又名乙`` / ``甲，简称乙``.
     This is a Chinese naming convention, equally valid for people, organizations,
     products, and places; the lexicon is extensible
     (``<词库同目录>/alias_cue_words.txt``, one per line, ``#`` comments).
     **The separator must come before the cue word**: cue words are mostly common
     characters (字/号/名/称), and the character inside "句号" (period) or "编号"
     (number) is not an appellation — the difference lies in where the separator
     sits (see :func:`mine_patterns`).
  2. ``paren``  Parenthetical apposition: ``甲（乙）``. One tier below 1 — the
     parentheses often hold an **explanation** rather than an alternative name
     ("顾言（作者）"), so only consider parentheses whose content is
     **name-shaped** (or that carry an appellation cue themselves, like
     "甲（字乙）").
  3. ``cooc``   Co-occurrence + lift: ``lift = P(a,b) / (P(a)·P(b))``. Two names
     that always appear together, far beyond the independence assumption, usually
     denote the same object ("abbreviation + full name" is the most common case).
  4. ``abbrev`` Containment: ``北大`` ⊂ ``北京大学``. Abbreviations / full names /
     suffix-stripped spellings all go through this one.

Confidence is an **interpretable weighted sum** (weights in :data:`SIGNAL_WEIGHT`),
not a black-box score: every candidate can answer "because of which signal, which
piece of source text".

Usage::

    from hippocampus_mcp import aliases
    cue, stat = aliases.load_cue_words("<词库目录>/alias_cue_words.txt")
    hits = aliases.mine_patterns(text, cue, known={"顾言"})   # pattern-based
    pair, docs, n = aliases.cooccurrence(records)             # statistics-based

This module **does not depend on brain / storage** — pure functions + dataclasses,
easy to unit-test.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "ALIAS_CUE_WORDS",
    "ALIAS_CUE_FILENAME",
    "ALIAS_STOP_WORDS",
    "SIGNAL_WEIGHT",
    "Pair",
    "load_cue_words",
    "name_like",
    "mine_patterns",
    "cooccurrence",
    "lift_of",
    "confidence_of",
    "substring_relation",
    "abbrev_relation",
    "is_cjk_only",
    "snippet_around",
    "is_alias_noise",
]

# ---------------------------------------------------------------- Lexicon
# Appellation cue words: **generic Chinese naming vocabulary**, independent of the
# corpus's subject matter. Useful for people (字/号/小名/笔名), organizations
# (简称/全称/旧称), products (代号/型号), places (旧称/别称), so they ship in the
# built-in layer, working out of the box; project-specific spellings go into a
# supplementary file.
#
# Note: **do not** mix in structural words like "年/月/日/章/卷" — they belong to
# the tag extractor (tagging) domain and would only produce fake aliases like
# "本章 => 正文" here.
ALIAS_CUE_WORDS: tuple[str, ...] = (
    # People: appellations beyond the bare name
    "字", "号", "别号", "法号", "道号", "谥号", "庙号", "绰号", "浑号",
    "小名", "乳名", "本名", "原名", "旧名", "曾用名", "笔名", "艺名",
    "化名", "自号", "雅号", "封号", "帝号",
    # Generic alternative names
    "别称", "别名", "又名", "又称", "亦名", "亦称", "也叫", "外号",
    "简称", "缩写", "全称", "俗称", "通称", "尊称", "代号", "昵称",
    "译名", "英文名", "学名", "旧称",
    # English (same mechanism; no need for a separate path for English)
    "aka", "also known as", "formerly known as",
)

ALIAS_CUE_FILENAME = "alias_cue_words.txt"

# Generic words used only for **alias gating**: name-shaped (2 CJK characters)
# but semantically roles / relations / generic nouns, never an entity's
# "alternative name" ("作者" (author) inside "顾言（作者）" is the classic case).
# Matching is **whole-word equality**, no substring suppression — this table has
# a single purpose and is deliberately kept separate from ``lexicon_suggest``'s
# stop-word table: that one does substring suppression, where adding "人物"
# would also suppress real titles like "人物志".
ALIAS_STOP_WORDS: frozenset[str] = frozenset({
    "作者", "读者", "编者", "译者", "主角", "配角", "反派", "人物", "角色",
    "客户", "用户", "公司", "团队", "项目", "产品", "服务", "系统", "平台",
    "数据", "内容", "问题", "方法", "方案", "结果", "原因", "目的", "过程",
    "时间", "地方", "事情", "东西", "大家", "我们", "他们", "别人", "自己",
    "对方", "双方", "上面", "下面", "期间", "以后", "之前", "当时", "目前",
    "第一", "第二", "第三", "最后", "全部", "所有", "部分", "其他", "别的",
})

# Each signal's contribution to confidence. **Composed upper bound 1.0**;
# ``cooc`` is further scaled by "lift strength", so co-occurrence alone can
# contribute at most 0.30 — co-occurrence-only candidates naturally rank below
# pattern-based ones.
SIGNAL_WEIGHT: dict[str, float] = {
    "cue": 0.50,
    "paren": 0.30,
    "abbrev": 0.25,
    "cooc": 0.30,
}

# Lift reaching this value counts as "full strength" (lift=5 gives the full
# score, lift=2 gives 1/4).
_LIFT_FULL = 5.0
# Cap on the support bonus (more co-occurrence records = more believable, but
# don't let it drown out the differences between signals).
_SUPPORT_BONUS_MAX = 0.10

# ---------------------------------------------------------------- Character classes
_CJK = "\u4e00-\u9fa5"
# Characters allowed in a name: CJK, Latin letters, digits, middle dot
# ("阿·伊万诺夫"). Deliberately **excludes** full-width punctuation and
# parentheses — they are name boundaries, not part of the name.
_NAME_CHARS = rf"{_CJK}A-Za-z0-9·"
_RE_NAME_RUN = re.compile(rf"[{_NAME_CHARS}]+")
_CJK_NAME_RE = re.compile(rf"^[{_CJK}][{_CJK}·]{{1,11}}$")
_ASCII_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{1,29}$")
# Separators that may appear around a name (including Chinese quotes/title marks)
_SEP_CHARS = "，,、：:；;　 \t\"'“”‘’「」『』《》〈〉【】()（）"
# Characters used by numerals/ordinal words. Including "第" makes ordinal words
# like "第三/第二十" be rejected wholesale as non-names — otherwise they would
# slip into the candidates with the shape "2 CJK characters" ("第" itself is
# also common inside 《》, but there other characters sit around it, so it is
# not touched by this rule).
_NUMERIC_CHARS = set("0123456789０１２３４５６７８９〇零一二三四五六七八九十百千万亿两第")
_RE_PAREN = re.compile(rf"([{_NAME_CHARS}]{{2,16}})\s*[（(]([^（()）\n]{{1,24}})[）)]")

# Length cap (characters) for adopting a whole run when neither side is
# registered. Beyond it, better to give up: unpunctuated spellings like
# "此人是顾言，字子安" yield whole runs mixed with predicates, and only short
# runs are worth the gamble; long runs should be registered in the lexicon
# first, after which they get recognized naturally.
_MAX_RUN_UNKNOWN = 4


@dataclass
class Pair:
    """Raw evidence for one candidate pair (unscored, direction undecided)."""

    a: str                      # the name earlier in the pattern
    b: str                      # the name later in the pattern
    signal: str                 # cue / paren / abbrev / cooc
    snippet: str                # source fragment at the hit (for humans)
    cue: str = ""               # the matched cue word (set when signal=cue)
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str]:
        return self.a.lower(), self.b.lower()


# ---------------------------------------------------------------- Lexicon loading
def load_cue_words(
    path: str | None = None,
    extra: Sequence[str] | None = None,
) -> tuple[set[str], dict[str, int]]:
    """Return ``(cue word set, per-layer counts)``.

    A union of three layers, **add-only, never overwrite** relative to the
    built-in prefix: **built-in generic** (in code, cross-project out of the
    box) ∪ **project file** (``path``, one per line) ∪ **call-site extras**
    (``extra``). Project-specific spellings (e.g. a genre-specific "化名"
    habit) go into the file; don't modify the built-in layer.
    """
    builtin = {w.lower() for w in ALIAS_CUE_WORDS if w}
    file_words: set[str] = set()
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    file_words.add(line.lower())
        except OSError:
            file_words = set()
    ex = {str(w).strip().lower() for w in (extra or []) if str(w).strip()}
    return builtin | file_words | ex, {
        "builtin": len(builtin), "file": len(file_words), "extra": len(ex),
    }


def _cue_re(cue_words: Iterable[str]) -> re.Pattern[str] | None:
    """Cue-word regex (**longest first**, otherwise "又名" gets truncated by short words like "名")."""
    words = sorted({w for w in cue_words if w}, key=len, reverse=True)
    if not words:
        return None
    return re.compile("|".join(re.escape(w) for w in words), re.I)


# ---------------------------------------------------------------- Name shape
def name_like(s: str, min_len: int = 2, max_len: int = 12) -> bool:
    """Whether the string is "name-shaped".

    A **shape** judgment only, not semantic (that is the stop words' and
    confidence's job): pure CJK (2–12 characters, middle dot allowed) or pure
    Latin (starts with a letter, 2–30 characters). Pure numerals / numeral
    words ("第三" "一百二十") never count — they are almost always quantifier
    phrases.
    """
    t = (s or "").strip().strip(_SEP_CHARS)
    if not t or len(t) < min_len or len(t) > max_len:
        return False
    if set(t) <= _NUMERIC_CHARS:
        return False
    if _CJK_NAME_RE.match(t):
        return True
    return bool(_ASCII_NAME_RE.match(t))


def _prefer_known_suffix(
    run: str, known: set[str], min_len: int, max_len: int
) -> str | None:
    """Take the **tail** name from a run of consecutive name characters, known words first.

    Why "known words first": the corpus often has unpunctuated spellings like
    "此人是顾言，字子安", and the rule itself cannot tell whether "此人是" is a
    predicate or part of the name. But **known words** (lexicon entries /
    registered entities) can: search suffixes from long to short, take the
    first hit, and "顾言" is cut out correctly. With no known word available,
    adopt only if the whole run is short enough (<= 4 characters) — prefer
    missing over garbage.
    """
    if known:
        top = min(len(run), max_len)
        for ln in range(top, min_len - 1, -1):
            cand = run[-ln:]
            if cand.lower() in known and name_like(cand, min_len, max_len):
                return cand
    if len(run) <= _MAX_RUN_UNKNOWN and name_like(run, min_len, max_len):
        return run
    return None


def _prefer_known_prefix(
    run: str, known: set[str], min_len: int, max_len: int
) -> str | None:
    """Same as :func:`_prefer_known_suffix`, but taking the **head** name."""
    if known:
        top = min(len(run), max_len)
        for ln in range(top, min_len - 1, -1):
            cand = run[:ln]
            if cand.lower() in known and name_like(cand, min_len, max_len):
                return cand
    if len(run) <= _MAX_RUN_UNKNOWN and name_like(run, min_len, max_len):
        return run
    return None


def _has_sep(text: str, idx: int, before: bool, span: int = 2) -> bool:
    """Whether the ``span`` characters before/after position ``idx`` contain a separator (to judge whether a cue word stands alone)."""
    if before:
        seg = text[max(0, idx - span):idx]
    else:
        seg = text[idx:idx + span]
    return any(ch in _SEP_CHARS for ch in seg)


def is_alias_noise(word: str, extra: Iterable[str] | None = None) -> bool:
    """Whether this is **whole-word**-level alias noise (roles/relations/generic nouns; see :data:`ALIAS_STOP_WORDS`).

    Deliberately whole-word equality only: an alias is "another name for some
    entity", so a generic noun appearing whole in a candidate slot is simply
    wrong; but its appearing as a **substring** inside a real name
    ("人物" ⊂ "人物志") is fine, so the substring suppression used for
    proper-noun candidates must not be applied here.
    """
    w = (word or "").strip()
    if not w:
        return True
    if w in ALIAS_STOP_WORDS:
        return True
    return any(w == str(x).strip() for x in (extra or []) if str(x).strip())


def snippet_around(text: str, start: int, end: int, width: int = 36) -> str:
    """Take a context fragment around the hit (single line, extra whitespace collapsed)."""
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    frag = " ".join(text[lo:hi].split())
    return ("…" if lo > 0 else "") + frag + ("…" if hi < len(text) else "")


# ---------------------------------------------------------------- Signals 1/2
def mine_patterns(
    text: str,
    cue_words: Iterable[str],
    known: set[str] | None = None,
    min_len: int = 2,
    max_len: int = 12,
    with_paren: bool = True,
    stats: dict[str, int] | None = None,
) -> list[Pair]:
    """Mine two classes of candidates from a text: "appellation cues" and "parenthetical appositions".

    Args:
        cue_words: Cue-word table (see :func:`load_cue_words`).
        known: **Known-word set** (lowercased) — lexicon entries + trusted
            entity names. It is used only to **assist segmentation**, not to
            judge right or wrong: with it, the correct name can be cut out of
            glued-together text.
        with_paren: Whether to also scan parenthetical appositions.
        stats: Optional counter (updated in place), for the caller to judge
            "is the gate set wrong". Currently records ``cue_no_sep_before``
            (dropped because no separator before the cue word) and
            ``cue_sep_after_only`` (the variant where "the separator is only
            after the cue word" — the typical false-hit shape; see below).

    Returns:
        Raw candidate :class:`Pair` list (may contain duplicates; the caller
        aggregates).
    """
    out: list[Pair] = []
    if not text:
        return out
    known = known or set()
    cre = _cue_re(cue_words)

    # ---- Signal 1: appellation cue words -------------------------------
    if cre is not None:
        for m in cre.finditer(text):
            cue = m.group(0)
            # The cue word must **stand alone with the separator before it**
            # ("，字子安" / "（字子安）"), or both sides are known words
            # (punctuation-free spellings like "顾言字子安"). Otherwise the
            # "字" inside "名字" "文字" "数字" would be taken as an appellation
            # cue, fabricating pairs like "这个名 => 很有意思" out of thin air.
            sep_before = _has_sep(text, m.start(), before=True)
            # Left side: cut to after "the previous cue word", so that
            # "字子安号雪芹" doesn't string both sides together
            left_seg = text[max(0, m.start() - 40):m.start()]
            prev = list(cre.finditer(left_seg))
            if prev:
                left_seg = left_seg[prev[-1].end():]
            run_text = left_seg.rstrip(_SEP_CHARS)
            tail = re.search(rf"[{_NAME_CHARS}]+$", run_text)
            if not tail:
                continue
            a = _prefer_known_suffix(tail.group(0), known, min_len, max_len)
            if not a:
                continue
            # Right side: cut to before "the next cue word"
            right_seg = text[m.end():m.end() + 40]
            nxt = cre.search(right_seg)
            if nxt:
                right_seg = right_seg[:nxt.start()]
            right_seg = right_seg.lstrip(_SEP_CHARS)
            head = re.match(rf"[{_NAME_CHARS}]+", right_seg)
            if not head:
                continue
            b = _prefer_known_prefix(head.group(0), known, min_len, max_len)
            if not b or a.lower() == b.lower():
                continue
            # The separator must come **before** the cue word. Measured on a
            # real library (9359 Chinese-novel records), this is far stronger
            # than "a separator on either side": every false hit had the same
            # shape — the cue word was exactly the **next character** of some
            # ordinary word:
            # 「碎句**号**、长段落」（句号）「全量换**号**、活跃文档」（编号）
            # 「否定排比、破折**号**」「同章全称后简**称**」（简称）
            # 「何时取**字**，另行定」. Their common trait is a separator
            # **after** the cue word; genuine appellation spellings ("甲，字乙")
            # always have the separator **before** it. When relaxed to "a
            # separator on either side", such pairs made up over half of the
            # candidates (16 of 25).
            if not sep_before and (a.lower() not in known or b.lower() not in known):
                if stats is not None:
                    stats["cue_no_sep_before"] = stats.get("cue_no_sep_before", 0) + 1
                    if _has_sep(text, m.end(), before=False):
                        stats["cue_sep_after_only"] = (
                            stats.get("cue_sep_after_only", 0) + 1
                        )
                continue
            out.append(Pair(
                a=a, b=b, signal="cue", cue=cue,
                snippet=snippet_around(text, m.start(), m.end()),
            ))

    # ---- Signal 2: parenthetical apposition ----------------------------
    if with_paren and ("（" in text or "(" in text):
        for m in _RE_PAREN.finditer(text):
            outer_run, inner = m.group(1), m.group(2).strip()
            outer = _prefer_known_suffix(outer_run, known, min_len, max_len)
            if not outer:
                continue
            inner_cue = cre.search(inner) if cre is not None else None
            if inner_cue:
                # "顾言（字子安）" → the appellation comes after the cue word
                seg = inner[inner_cue.end():].lstrip(_SEP_CHARS)
                head = re.match(rf"[{_NAME_CHARS}]+", seg)
                if not head:
                    continue
                b = _prefer_known_prefix(head.group(0), known, min_len, max_len)
                cue = inner_cue.group(0)
            else:
                # The whole parenthetical is name-shaped → weak candidate
                # (parentheses often hold explanations too)
                b = inner if name_like(inner, min_len, max_len) else None
                cue = ""
            if not b or b.lower() == outer.lower():
                continue
            out.append(Pair(
                a=outer, b=b, signal="paren", cue=cue,
                snippet=snippet_around(text, m.start(), m.end()),
            ))
    return out


# ---------------------------------------------------------------- Signals 3/4
def cooccurrence(
    record_names: Iterable[tuple[str, set[str]]],
    max_names_per_record: int = 30,
) -> tuple[dict[tuple[str, str], int], dict[str, int], int]:
    """Count "same-record co-occurrence": ``(pair_counts, name_record_counts, n_records)``.

    The input is a stream of ``(memory id, set of names appearing in it)``. The
    **source** of the names is the caller's decision (registered entities +
    lexicon hits suffice) — this function only does the counting and does not
    guess provenance.

    ``max_names_per_record`` is a **cost gate**: when a single memory carries
    dozens of names, pairwise matching is O(n²). Past the cap, truncate (take
    the first N after sorting names) so that one "entity list"-style long text
    cannot drag down the whole mining round.
    """
    pair: dict[tuple[str, str], int] = {}
    docs: dict[str, int] = {}
    n = 0
    for _rid, names in record_names:
        n += 1
        ns = sorted({(x or "").strip().lower() for x in names if (x or "").strip()})
        if not ns:
            continue
        if len(ns) > max_names_per_record:
            ns = ns[:max_names_per_record]
        for x in ns:
            docs[x] = docs.get(x, 0) + 1
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                k = (ns[i], ns[j])
                pair[k] = pair.get(k, 0) + 1
    return pair, docs, n


def lift_of(count_ab: int, count_a: int, count_b: int, n_records: int) -> float:
    """Lift ``P(a,b) / (P(a)·P(b))``; ≈1 when independent; well above 1 indicates co-reference."""
    if count_ab <= 0 or count_a <= 0 or count_b <= 0 or n_records <= 0:
        return 0.0
    expected = count_a * count_b / n_records
    return count_ab / expected if expected > 0 else 0.0


def confidence_of(
    signals: Iterable[str],
    support: int = 0,
    lift: float = 0.0,
) -> float:
    """Fold a signal set into a 0–1 confidence (**weighted sum, interpretable**).

    The ``cooc`` contribution scales with lift: ``lift`` at :data:`_LIFT_FULL`
    (5) gets the full weight, ``lift=2`` only 1/5. Support adds a small bonus
    (capped at :data:`_SUPPORT_BONUS_MAX`), because "holds across multiple
    memories" resists noise better than "appears in only one".
    """
    sig = set(signals)
    total = 0.0
    for s in sig:
        w = SIGNAL_WEIGHT.get(s, 0.0)
        if s == "cooc":
            total += w * max(0.0, min(1.0, (lift - 1.0) / (_LIFT_FULL - 1.0)))
        else:
            total += w
    total += min(_SUPPORT_BONUS_MAX, 0.02 * max(0, support))
    return round(min(1.0, total), 4)


def substring_relation(a: str, b: str) -> str:
    """Return ``"a<b"`` / ``"b<a"`` / ``""`` (containment relation, for abbreviations/full names)."""
    la, lb = a.lower(), b.lower()
    if la == lb:
        return ""
    if la in lb:
        return "a<b"
    if lb in la:
        return "b<a"
    return ""


def abbrev_relation(a: str, b: str) -> str:
    """Count "abbreviation ⊂ full name" only when there is containment **and** both
    sides are CJK strings; otherwise return ``""``.

    Why this restriction (learned the hard way on a real library):
    substring relations among Latin strings are overwhelmingly
    **numbers/model codes**, not abbreviations — ``ch4`` ⊂ ``ch49``,
    ``v1`` ⊂ ``v12``; they denote completely different objects in the library
    yet were pushed into candidates by "containment + co-occurrence" (one run
    produced a dozen-plus garbage pairs like ``ch3x => ch3``). Chinese
    abbreviation/full-name pairs are the ones that are genuinely contiguous
    substrings (``北大`` ⊂ ``北京大学``、``清华`` ⊂ ``清华大学``), so this
    rule is not "guarding against English" but "guarding against mistaking
    model codes for abbreviations".
    """
    if not (is_cjk_only(a) and is_cjk_only(b)):
        return ""
    return substring_relation(a, b)


def is_cjk_only(s: str) -> bool:
    t = (s or "").strip()
    return bool(t) and all(
        ("\u4e00" <= ch <= "\u9fa5") or ch == "·" for ch in t
    )
