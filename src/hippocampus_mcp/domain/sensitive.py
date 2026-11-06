# -*- coding: utf-8 -*-
"""Sensitive-content matching (the criteria layer of the first gate in the sensitive double gate).

A **pure function** over pattern-level built-in fallbacks plus deployment
lexicon matching — the env lexicon content and the (replaceable) pattern set
are both injected by the caller; this module reads no env and does no I/O.

Two consumers (both at the service layer):

* ``OrganizeService.consolidate``'s promotion-candidate annotation (blocked
  entries must be annotated with the hit word and position, never silently);
* ``WriteService.store_memory``'s **dedicated sensitive filtering** when the
  target library name is ``user`` (plugging the direct-write bypass;
  **behavior of existing libraries unchanged** — non-user libraries see zero
  change; the home-library write boundary will land its guard at the factory
  layer with role-based exemption for the user library, composing orthogonally
  with this filter).

Matching semantics (narrowed to two kinds; no guessing a third):

* Lexicon entries → **substring** matching (case-sensitive — the lexicon is
  deployment-owned; what you write is what matches; for case-insensitivity
  provide both spellings in the lexicon);
* Patterns → **regex** (built-ins: 18-digit ID card / 11-digit phone number /
  common secret formats).

When the lexicon is absent, the **pattern-level fallback still holds**: the
second gate of red line 1 "any failure blocks the write" (explicit
confirmation) does not depend on the lexicon existing, and neither does the
first gate's fallback.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

__all__ = [
    "BUILTIN_PATTERNS",
    "SENSITIVITY_CATEGORIES",
    "match_sensitive",
    "classify_sensitivity",
]

#: Sensitive-category keyword table (D1: the category layer of privacy tagging;
#: a module-level managed constant, settled by Q-9).
#: Same rule as BUILTIN_PATTERNS: a module constant, injectable/replaceable
#: wholesale by the caller; keywords are **substring** matches (case-sensitive)
#: — it is a tagging hint, not an enforcement verdict; misses are the norm,
#: backed by periodic access_log audits (design doc §4.1 miss criteria).
#: Adding categories goes through explicit change (edit here + add tests); no
#: runtime injection surface.
SENSITIVITY_CATEGORIES: dict[str, tuple[str, ...]] = {
    "行程": (
        "机票", "航班", "火车票", "高铁票", "订酒店", "入住",
        "行程", "出行", "护照", "签证", "登机", "值机",
    ),
    "健康": (
        "确诊", "症状", "诊断", "处方", "用药", "复诊",
        "体检报告", "体检", "病历", "心理咨询", "复健",
    ),
    "财务": (
        "工资", "月薪", "年薪", "存款", "贷款", "负债",
        "欠款", "报销", "银行卡号", "纳税", "欠薪",
    ),
    "人际": (
        "离婚", "分手", "出轨", "恋爱关系", "相亲",
        "遗产", "抚养权", "家庭矛盾",
    ),
}

#: Pattern-level built-in fallback (module-level constant, not env —
#: replaceable/injectable wholesale by the caller).
#: (kind, compiled); kind goes into the hit details so the caller can
#: self-report "which format class hit".
BUILTIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 18-digit ID card number (last digit may be x/X; must not be adjacent to digits, to avoid swallowing long digit runs)
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    # 11-digit phone number (13-19 prefixes; must not be adjacent to digits)
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    # Common secret formats: OpenAI sk- / GitHub token / generic api_key= assignment
    ("secret_key", re.compile(
        r"(?i)\b(sk-[a-z0-9]{16,}|gh[pousr]_[a-z0-9]{30,}"
        r"|api[_-]?key[\"']?\s*[:=]\s*[\"']?[a-z0-9_\-]{16,})")),
)


def match_sensitive(
    content: str,
    *,
    words: Iterable[str] = (),
    patterns: Iterable[tuple[str, re.Pattern[str]]] | None = None,
) -> list[dict[str, Any]]:
    """Scan content and return the hit list (``[{word, start, end, kind}]``, ordered by position).

    Args:
        content: text to inspect.
        words: deployment lexicon entries (read from the env file by the caller
            and passed in; substring matching).
        patterns: a sequence of ``(kind, compiled)``; ``None`` = the built-in
            :data:`BUILTIN_PATTERNS`. An empty sequence = pattern channel off
            (lexicon only).

    Returns:
        One entry per occurrence: ``{"word": hit text, "start": start,
        "end": end, "kind": "word" or the pattern kind}``, ascending by
        ``start``. **Empty list = clean**.
    """
    text = content or ""
    hits: list[dict[str, Any]] = []
    for w in words:
        if not w:
            continue
        start = text.find(w)
        while start != -1:
            hits.append({"word": w, "start": start,
                         "end": start + len(w), "kind": "word"})
            start = text.find(w, start + 1)
    for kind, pat in (BUILTIN_PATTERNS if patterns is None else patterns):
        for m in pat.finditer(text):
            hits.append({"word": m.group(0), "start": m.start(),
                         "end": m.end(), "kind": kind})
    hits.sort(key=lambda h: (h["start"], h["end"], h["kind"]))
    return hits


def classify_sensitivity(
    content: str,
    *,
    categories: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Tag content with sensitive-category labels via category keywords (D1 pure function; classifies, never enforces).

    Division of labor with :func:`match_sensitive`: that one is the red line for
    **credential-class** content (a hit blocks the write); this one tags
    **category-level** privacy (travel/health/finance/relationships) — a hit
    does not block the write, it only decides the ``sensitivity`` label on the
    quarantine card (consumed by the D2a write-side gate), for drift_audit's
    manual-review grouping. Keyword table: :data:`SENSITIVITY_CATEGORIES`.

    Args:
        content: text to inspect.
        categories: category keyword table; ``None`` = the built-in
            :data:`SENSITIVITY_CATEGORIES`. An empty dict = category channel off.

    Returns:
        ``{"level": "sensitive"|"clean", "categories": [hit category names,
        sorted], "hits": [{category, word, start, end}]}``. level and categories
        are consistent: any hit means sensitive, no hit means clean (single
        source of truth; the caller need not re-judge).
    """
    text = content or ""
    table = SENSITIVITY_CATEGORIES if categories is None else categories
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cat, words in table.items():
        for w in words or ():
            if not w:
                continue
            start = text.find(w)
            if start == -1:
                continue
            seen.add(cat)
            # Record only the first hit position per category: tagging wants
            # "which class", not an exhaustive list of occurrences
            # (exhaustive positions are match_sensitive's enforcement job; save tokens here).
            hits.append({"category": cat, "word": w,
                         "start": start, "end": start + len(w)})
            break
    hits.sort(key=lambda h: (h["start"], h["category"]))
    cats = sorted(seen)
    return {"level": "sensitive" if cats else "clean",
            "categories": cats, "hits": hits}
