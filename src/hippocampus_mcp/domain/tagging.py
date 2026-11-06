# -*- coding: utf-8 -*-
"""Structured tags: a **configurable** structural-marker extractor.

Why this module exists
======================

The **filter side** of tags (``recall(filter_tags=...)``) has been generic
from the start -- any keys, AND/OR, ``str()`` normalization; the code never
had any shadow of "章/卷/年" in it.

The **extraction side** was not: three regexes (``第N章回`` / ``第N卷部`` /
``干支``) were hard-coded in the import script and existed only there. That
left two generic gaps:

1. **New corpus means code changes**. Contracts use 「第N条」, medical
   records use 「第N次化疗」, logs use 「2026-09」 -- all the same need
   ("structural markers carried by the document → filterable tags"), yet
   the only way was editing Python source.
2. **Server-side writes had no extraction ability at all**.
   ``ingest``/``ingest_batch`` only accept ``tags`` the host extracted
   itself; if the host did not extract, those entries stayed forever in the
   "not findable by tag" bucket.

This module turns "structural marker → tag" into an **extractor + config**:

* The built-ins are only **domain-agnostic Chinese document structure
  words** -- the numbering family (章/回/卷/部) and the time family (ganzhi
  years, Gregorian years, months). **Targeted at no specific corpus**;
* Key names (``keys``) and recognized suffixes (``suffixes``) can both be
  renamed/extended via config;
* Any project-specific structural marker goes through ``patterns`` config,
  **never into code**.

Config source (priority, high to low)::

    explicit schema param → HIPPOCAMPUS_TAG_SCHEMA (file path)
    → <persist_dir>/tag_schema.json → built-in defaults

Config format (JSON; all fields optional; write only what you override)::

    {
      "keys":     {"章": "chapter", "月": "month"},
      "suffixes": {"章": "chapter", "回": "chapter", "条": "clause"},
      "patterns": [
        {"key": "化疗次", "regex": "第\\\\s*([一二三四五六七八九十\\\\d]{1,3})\\\\s*次化疗",
         "type": "int", "min": 1, "max": 99}
      ],
      "month": true,
      "gregorian_year": true,
      "cap": 6
    }

Hard constraints (read before changing)
=======================================

* **Key names are the caller's contract**. If the host queries with
  ``filter_tags={"月": "闰四月"}``, this side must produce ``月``. Renaming
  must be synchronized on both ends -- **key mismatch = silent filter
  failure**, no error.
* **Prefer missing over wrong**. Only recognize unambiguous structural
  markers; never guess. A wrong tag is worse than no tag: it makes
  filtering give the **wrong answer** (rather than no answer).
* Values are uniformly ``int`` / ``str``; multiple values are stored as a
  list (the filter side expands it as OR), a single value degrades to a
  scalar.
* Extraction order does not matter, but **when one memory involves multiple
  values under the same key, all must be kept** -- ``{"章": [5, 12]}``
  means "this entry involves both chapters"; keeping only one makes it
  disappear under the other chapter.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

ENV_TAG_SCHEMA = "HIPPOCAMPUS_TAG_SCHEMA"

# Construction-time injection seam: the knobs are registered in config
# (tag_schema) and filled in by engine at import on the production path;
# for the resolution order (env first, injected value as fallback) see
# common/config_bridge.
from ..common.config_bridge import apply_config, resolve as _resolve
SCHEMA_FILENAME = "tag_schema.json"

# ----------------------------------------------------------------------
# Chinese numerals
# ----------------------------------------------------------------------
CN_NUM_CHARS = "零〇一二三四五六七八九十百千两"
_CN_DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn_num_to_int(text: str) -> int | None:
    """Chinese numeral to int ("四十三" → 43, "十" → 10, "43" → 43). Returns None on failure.

    Tag values are uniformly int: at filter time both ``{"章": 43}`` and
    ``{"章": "43"}`` match (``str()`` normalization before comparison), but
    storing int is friendlier for humans and for sorting.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    current = 0
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[ch]
            current = 0
        else:
            return None
    return total + current


# ----------------------------------------------------------------------
# Built-in patterns
# ----------------------------------------------------------------------
# Ganzhi year naming: 60 legal combinations = heavenly stems (10) × earthly
# branches (12) paired positionally, least common multiple 60
_TIANGAN = "甲乙丙丁戊己庚辛壬癸"
_DIZHI = "子丑寅卯辰巳午未申酉戌亥"
_GANZHI_60 = [_TIANGAN[i % 10] + _DIZHI[i % 12] for i in range(60)]
_GANZHI_ALT = "|".join(_GANZHI_60)

# Default key names. **These four are generic Chinese document structure
# words**, not any single project's private convention.
DEFAULT_KEYS: dict[str, str] = {"章": "章", "卷": "卷", "年": "年", "月": "月"}

# Structural suffix → key name. Defaults contain only the two families that
# match historical behavior; to recognize project-specific structures like
# 「第N条 / 第N节 / 第N幕」, add them via config, **do not change this**.
DEFAULT_SUFFIXES: dict[str, str] = {"章": "章", "回": "章", "卷": "卷", "部": "卷"}

# Special-call months (common lunar-calendar spellings). Only the three most
# generic are collected; no enumerating aliases like "菊月/荷月" -- an
# enumerated wordlist is exactly the old disease of "designing against one
# corpus".
LUNAR_MONTH_NAMES = ("正月", "冬月", "腊月")

_FEATURE = re.compile(r"[\u4e00-\u9fa5]")
_BAD_SUFFIX_CHARS = set("的了是在和与我你他她它")


class TagPattern:
    """One custom extraction rule (an item of ``patterns`` in the config).

    ``regex`` must **have one capture group**, whose content becomes the tag
    value; with ``type="int"`` it is converted via :func:`cn_num_to_int`,
    and on failure the match is dropped (no second-guessing).
    """

    __slots__ = ("key", "regex", "type", "min", "max", "multi")

    def __init__(self, key: str, regex: str, type: str = "str",
                 min: Any = None, max: Any = None, multi: bool = True):
        if not key:
            raise ValueError("TagPattern 缺 key")
        self.key = str(key)
        self.type = "int" if str(type).lower() in ("int", "number", "num") else "str"
        self.min = min
        self.max = max
        self.multi = bool(multi)
        try:
            self.regex = re.compile(regex)
        except re.error as e:
            raise ValueError(f"标签正则非法 {regex!r}: {e}") from e
        if self.regex.groups < 1:
            raise ValueError(f"标签正则必须有捕获组: {regex!r}")

    def findall(self, text: str) -> list[Any]:
        vals: list[Any] = []
        for m in self.regex.finditer(text):
            raw = m.group(1)
            if self.type == "int":
                v = cn_num_to_int(raw)
                if v is None:
                    continue
                if self.min is not None and v < int(self.min):
                    continue
                if self.max is not None and v > int(self.max):
                    continue
                vals.append(v)
            else:
                v = (raw or "").strip()
                if v:
                    vals.append(v)
        return vals

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "regex": self.regex.pattern, "type": self.type,
                "min": self.min, "max": self.max, "multi": self.multi}


def _tag_value_sort_key(v: Any) -> tuple:
    """Sort key for tag values: numbers by value, strings lexicographically, numbers first.

    You cannot sort via ``str(v)`` directly -- 12 would sort before 5
    (``"12" < "5"``), and the chapter list comes out scrambled (actually
    hit: ``[12, 5]`` after a merge).
    """
    if isinstance(v, bool):  # bool is an int subclass; given its own branch into the number group
        return (0, float(v), "")
    if isinstance(v, (int, float)):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def merge_tag_values(a: Any, b: Any) -> Any:
    """Merge two tag values (safe for scalar/list mixes); after dedup a single value degrades to a scalar."""
    vals: list[Any] = []
    for v in (a, b):
        if v is None:
            continue
        vals.extend(v if isinstance(v, list) else [v])
    uniq = sorted(set(vals), key=_tag_value_sort_key)
    return uniq[0] if len(uniq) == 1 else uniq


class TagExtractor:
    """Extract structured tags according to the configuration.

    Args:
        keys: override mapping built-in family → key name (e.g. ``{"章": "chapter"}``).
        suffixes: structural suffix → key name (e.g. ``{"条": "条"}``).
            **Extending it adds newly recognizable structures**.
        patterns: extra custom rules (:class:`TagPattern` or config dicts).
        month: whether to extract months (ganzhi month / numeric month /
            special-call month). Default True.
        gregorian_year: whether to extract Gregorian years (``2026年``).
            Default True.
        cap: per-key multi-value cap.
        num_max: numbering cap (guards against obvious mis-extractions like
            "第 9999 章").
        filter_keys: **filter-side whitelist**. Only when declared are
            `filter_tags` keys validated; undeclared = no validation (keeps
            the "any key" status quo). See :func:`validate_filter_keys`.
        filter_degrade: whether invalid keys **soft-degrade** to "ignore
            that key and continue". Default False = explicit error. See the
            trade-off notes in :func:`validate_filter_keys`.
        source: config source label, for logging/self-description only.
    """

    def __init__(
        self,
        keys: dict[str, str] | None = None,
        suffixes: dict[str, str] | None = None,
        patterns: Iterable[Any] | None = None,
        month: bool = True,
        gregorian_year: bool = True,
        cap: int = 6,
        num_max: int = 999,
        filter_keys: Iterable[str] | None = None,
        filter_degrade: bool = False,
        source: str = "builtin",
    ):
        self.keys = {**DEFAULT_KEYS, **(keys or {})}
        self.suffixes = {**DEFAULT_SUFFIXES, **(suffixes or {})}
        self.month = bool(month)
        self.gregorian_year = bool(gregorian_year)
        self.cap = max(int(cap or 1), 1)
        self.num_max = int(num_max)
        # An empty list counts as "undeclared" (writing `[]` would mark every key invalid, almost certainly a misconfiguration)
        fk = [str(k) for k in (filter_keys or []) if str(k)]
        self.filter_keys: tuple[str, ...] | None = tuple(fk) if fk else None
        self.filter_degrade = bool(filter_degrade)
        self.source = source
        self.patterns = [
            p if isinstance(p, TagPattern) else TagPattern(**p)
            for p in (patterns or [])
        ]

        # Suffixes sorted by length descending so 「单元」 wins over prefixes like 「元」 (no overlap today, sorted by convention anyway)
        sufs = sorted(self.suffixes, key=len, reverse=True)
        suf_alt = "|".join(re.escape(s) for s in sufs)
        self._re_suffix = (
            re.compile(rf"第\s*([{CN_NUM_CHARS}\d]{{1,6}})\s*({suf_alt})") if sufs else None
        )
        self._re_ch_num = re.compile(
            r"(?:^|[\s（(【])ch(?:apter)?\.?\s*(\d{1,4})(?![0-9])", re.I
        )
        self._re_bare_volume = (
            re.compile(rf"^\s*(卷)\s*([{CN_NUM_CHARS}\d]{{1,6}})")
            if "卷" in self.suffixes else None
        )
        self._re_ganzhi = re.compile(_GANZHI_ALT)
        self._re_ganzhi_month = re.compile(rf"(?:{_GANZHI_ALT})\s*月")
        self._re_gyear = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})\s*年")
        self._re_month_num = re.compile(r"(?<!\d)(\d{1,2})\s*月份?")
        # ISO dates: ``2026-09-01`` / ``2026/9/1`` / ``2026.09``.
        # Why a separate pattern is required: ``_re_gyear`` demands a 「年」
        # suffix, while engineering corpora write ISO dates throughout (most
        # unlabeled entries in the real library contain ``YYYY-MM-DD``, yet
        # the built-in extractor matched none of them). The day part is
        # optional: year-month only (``2026.09``) must also be recognized.
        # The lookahead ``(?!\d)`` blocks prefix false hits on long digit
        # strings like ``2026-09-011``; months limited to 1–12, days to
        # 1–31, so version numbers / model numbers are not taken for dates.
        self._re_iso_date = re.compile(
            r"(?<!\d)(1[0-9]{3}|20[0-9]{2})[-/.]"
            r"(0?[1-9]|1[0-2])"
            r"(?:[-/.](0?[1-9]|[12]\d|3[01]))?(?!\d)"
        )
        # Special-call months (正月/冬月/腊月) and Chinese-numeral months are written separately:
        # 「正/冬/腊」 are not in the Chinese numeral charset, the two do not overlap, and each capture group stays simpler.
        lunar_alt = "|".join(LUNAR_MONTH_NAMES)
        self._re_month_lunar = re.compile(rf"(闰?(?:{lunar_alt}))")
        self._re_month_cn = re.compile(r"(闰?[一二三四五六七八九十]{1,3}月)")

    # ------------------------------------------------------------------
    @property
    def chapter_key(self) -> str:
        return self.keys.get("章", "章")

    @property
    def volume_key(self) -> str:
        return self.keys.get("卷", "卷")

    @property
    def year_key(self) -> str:
        return self.keys.get("年", "年")

    @property
    def month_key(self) -> str:
        return self.keys.get("月", "月")

    def describe(self) -> dict[str, Any]:
        """Describe the extraction rules currently in effect (written into tool return values so the host can check key names)."""
        return {
            "source": self.source,
            "keys": dict(self.keys),
            "suffixes": dict(self.suffixes),
            "month": self.month,
            "gregorian_year": self.gregorian_year,
            "cap": self.cap,
            "custom_patterns": [p.to_dict() for p in self.patterns],
            # Filter-side whitelist. None = undeclared = no validation (the host may use any key)
            "filter_keys": list(self.filter_keys) if self.filter_keys else None,
            "filter_degrade": self.filter_degrade,
        }

    # ------------------------------------------------------------------
    def _family_key(self, family: str | None) -> str | None:
        """Family name → output key name.

        **The two-step mapping cannot be skipped**: ``suffixes`` maps a
        「suffix」 to a **family** (条 → the 条 family), then ``keys`` maps
        the **family** to the **output key name**. When there was only one
        step (suffix directly to key name), a config change to ``keys`` got
        overridden by the default ``suffixes`` -- manifesting as "the key
        rename did not take effect", with no error.
        """
        if not family:
            return None
        return self.keys.get(family, family)

    def _numbered(self, text: str) -> list[tuple[str, Any]]:
        """Numbering family: ``第N<suffix>`` / ``chN`` (the latter only as the English spelling of "章")."""
        out: list[tuple[str, Any]] = []
        if self._re_suffix is not None:
            for m in self._re_suffix.finditer(text):
                key = self._family_key(self.suffixes.get(m.group(2)))
                n = cn_num_to_int(m.group(1))
                if key is None or n is None or not (1 <= n <= self.num_max):
                    continue
                out.append((key, n))
        for m in self._re_ch_num.finditer(text):
            n = int(m.group(1))
            if 1 <= n <= self.num_max:
                out.append((self.chapter_key, n))
        return out

    def _bare_volume(self, text: str, scan_all: bool = False) -> list[tuple[str, Any]]:
        """Bare 「卷三」 spelling (no 「第」).

        Recognized only at the **start of a heading**: in body text "卷三"
        may be mere narration ("卷三之后"); without 「第」 the only reliable
        anchor is the line start.
        """
        out: list[tuple[str, Any]] = []
        if self._re_bare_volume is None:
            return out
        m = self._re_bare_volume.search(text)
        if m:
            n = cn_num_to_int(m.group(2))
            key = self._family_key(self.suffixes.get(m.group(1)))
            if n is not None and key and 1 <= n <= self.num_max:
                out.append((key, n))
        return out

    def _years(self, text: str) -> list[tuple[str, Any]]:
        """Time family, years: ganzhi + Gregorian. Ganzhi first, keeping the historical priority order."""
        out: list[tuple[str, Any]] = []
        for m in self._re_ganzhi.finditer(text):
            out.append((self.year_key, m.group(0)))
        if self.gregorian_year:
            for m in self._re_gyear.finditer(text):
                out.append((self.year_key, int(m.group(1))))
            # The year inside ISO dates (``2026-09-01`` → 2026). The historical
            # priority order -- ganzhi first, Gregorian after -- is unchanged;
            # duplicates between ISO and 「2026年」 are deduped by ``_assemble``.
            for m in self._re_iso_date.finditer(text):
                out.append((self.year_key, int(m.group(1))))
        return out

    def _months(self, text: str) -> list[tuple[str, Any]]:
        """Time family, months: ganzhi months (戊子月) / special-call months (正月·闰四月) / numeric months (3月·3月份)."""
        out: list[tuple[str, Any]] = []
        if not self.month:
            return out
        for m in self._re_ganzhi_month.finditer(text):
            out.append((self.month_key, re.sub(r"\s+", "", m.group(0))))
        for m in self._re_month_lunar.finditer(text):
            out.append((self.month_key, m.group(1)))
        for m in self._re_month_cn.finditer(text):
            out.append((self.month_key, m.group(1)))
        for m in self._re_month_num.finditer(text):
            n = int(m.group(1))
            if 1 <= n <= 12:
                out.append((self.month_key, n))
        # Months inside ISO dates. Placed last -- Chinese months (正月/闰四月/10月)
        # have higher priority, and the heading mode's existing "take the
        # first hit" behavior is unaffected. The regex already limits months
        # to 1–12, but ``09`` must be normalized to int 9, otherwise the
        # filter side's ``filter_tags={"月": 9}`` would not match ``"09"``
        # (the tag side normalizes via ``str()`` anyway, so the int-ification
        # here only aligns with the 「2026年9月」 path).
        for m in self._re_iso_date.finditer(text):
            out.append((self.month_key, int(m.group(2))))
        return out

    def _custom(self, text: str) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for p in self.patterns:
            for v in p.findall(text):
                out.append((p.key, v))
        return out

    # ------------------------------------------------------------------
    def extract(self, text: str, heading: bool = False) -> dict[str, Any]:
        """Extract tags.

        Args:
            heading: heading mode. A heading line holds only one structural
                value (``### 第43章``), so each key takes **only the first**
                hit, and the bare 「卷三」 spelling is recognized only here.
                Body mode (default) **collects all**: one memory may involve
                several chapters ("参见第 12 章…本章对应第 5 章"); keeping
                only one makes it disappear under the other chapter.
        """
        if not text:
            return {}
        pairs: list[tuple[str, Any]] = []
        pairs.extend(self._numbered(text))
        if heading:
            pairs.extend(self._bare_volume(text))
        pairs.extend(self._years(text))
        pairs.extend(self._months(text))
        pairs.extend(self._custom(text))
        return self._assemble(pairs, heading=heading)

    def _assemble(self, pairs: list[tuple[str, Any]], heading: bool) -> dict[str, Any]:
        grouped: dict[str, list[Any]] = {}
        for key, val in pairs:
            if not key:
                continue
            bucket = grouped.setdefault(str(key), [])
            if val not in bucket:
                bucket.append(val)
        out: dict[str, Any] = {}
        for key, vals in grouped.items():
            if heading:
                # Heading mode: **take the first by order of appearance**, no
                # sorting. The first structural value in a heading is its
                # subject ("卷三 己丑年" → 卷=3, 年=己丑); sorting would float
                # the int up and the ganzhi year would be crowded out by the
                # Gregorian year.
                out[key] = vals[0]
                continue
            # Body mode: numbers by numeric order, strings lexicographically -- cannot sort via str()
            # (12 would jump ahead of 5).
            vals = sorted(vals, key=_tag_value_sort_key)[: self.cap]
            out[key] = vals[0] if len(vals) == 1 else vals
        return out


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------
def resolve_schema_path(explicit: str | None = None,
                        persist_dir: str | None = None) -> str | None:
    """Resolve the tag schema file path: explicit > env > ``<persist_dir>/tag_schema.json``.

    Same convention as the proper-noun lexicon (``_resolve_lexicon_path``):
    **computed fresh each time, never cached** -- placing the schema file in
    place only after service startup is a common order of operations, and a
    cached None would make it unreadable forever.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    env_path = _resolve(ENV_TAG_SCHEMA, "tag_schema", "") or None
    if env_path and os.path.isfile(env_path):
        return env_path
    if persist_dir:
        default = os.path.join(persist_dir, SCHEMA_FILENAME)
        if os.path.isfile(default):
            return default
    return None


def extractor_from_config(cfg: dict[str, Any], source: str = "config") -> TagExtractor:
    """Build an extractor from a config dict. Invalid fields raise immediately -- **no silent degradation to default rules**.

    Silent degradation turns "I definitely configured it" into "it actually
    never took effect", the hardest class of fault to debug.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"标签 schema 必须是 JSON 对象，得到 {type(cfg).__name__}")
    known = {"keys", "suffixes", "patterns", "month", "gregorian_year",
             "cap", "num_max", "filter_keys", "filter_degrade", "observed"}
    # underscore-prefixed entries are metadata left **for humans reading the
    # file** (e.g. the `_generated` written by build_tag_schema.py); the
    # extractor never reads them -- they should not trigger a warning on
    # every load.
    unknown = {k for k in cfg if not str(k).startswith("_")} - known
    if unknown:
        logger.warning("标签 schema 含未知字段（已忽略）: %s", sorted(unknown))
    return TagExtractor(
        keys=cfg.get("keys"),
        suffixes=cfg.get("suffixes"),
        patterns=cfg.get("patterns"),
        month=cfg.get("month", True),
        gregorian_year=cfg.get("gregorian_year", True),
        cap=cfg.get("cap", 6),
        num_max=cfg.get("num_max", 999),
        filter_keys=cfg.get("filter_keys"),
        filter_degrade=cfg.get("filter_degrade", False),
        source=source,
    )


def validate_filter_keys(tags: dict[str, Any] | None,
                         extractor: TagExtractor,
                         ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate that the **keys** of ``filter_tags`` are within the whitelist.

    Returns ``(effective_tags, err)``, exactly one of which is ``None``
    (**never both non-empty**):

    * ``(tags, None)`` -- pass-through. Two cases: the whitelist is
      undeclared, or all keys are valid.
    * ``(None, err)``  -- blocked (``degrade=False``) or fully stripped
      (``degrade=True``).

    Why validation only happens when a **whitelist is declared**
    -----------------------------------------------------------
    ``filter_tags`` has been generic from the start (any keys / AND-OR /
    ``str()`` normalization). Turning validation on while the library does
    not yet have a ``tag_schema.json`` would mark every existing key in the
    library invalid ⇒ "a typo silently returns empty" becomes "every call
    errors out" -- pure regression. So the order must be **build the schema
    first, then turn on validation** -- never the reverse.

    Why **no soft degradation** by default
    --------------------------------------
    Degrading (throwing away invalid keys and continuing) makes "a typo in
    the key name" silent again: with the invalid keys removed, retrieval
    still returns results, and the caller reads "this is all the library
    has" -- identical to not validating at all. Key validation exists to
    plug exactly this hole, so the default ``degrade=False`` errors out
    explicitly; when degradation is genuinely needed, ``"filter_degrade":
    true`` must be written **explicitly** in the schema, and the return
    body must carry ``degraded: true`` and ``degraded_from`` so the caller
    can see the change.

    Only keys are validated, not values: a value's domain is decided by the
    corpus (``章`` can be 43 or "闰四月"); blocking values would hurt
    legitimate queries; and a wrong value is already surfaced by the
    ``matched=0`` echo.
    """
    if not tags:
        return tags, None
    allowed = extractor.filter_keys
    if not allowed:                      # No whitelist declared ⇒ no validation (bit-for-bit unchanged)
        return tags, None
    unknown = [str(k) for k in tags if str(k) not in allowed]
    if not unknown:
        return tags, None
    info = {
        "unknown_keys": unknown,
        "available_keys": list(allowed),
        "schema_source": extractor.source,
    }
    if not extractor.filter_degrade:
        info["error"] = (f"filter_tags 的键不在标签白名单里: {unknown}；"
                         f"可用键见 available_keys（库内从未出现过的键需先建进 "
                         f"tag_schema.json 的 filter_keys）")
        info["degraded"] = False
        return None, info
    # Soft degradation: strip invalid keys and keep running with the rest; if nothing remains it equals "no filtering", still echoed
    kept = {k: v for k, v in tags.items() if str(k) not in unknown}
    info["degraded"] = True
    info["degraded_from"] = dict(tags)
    info["dropped_keys"] = unknown
    if not kept:
        info["error"] = (f"filter_tags 所有键都不在白名单里（已降级为不过滤）: "
                         f"{unknown}")
        return None, info
    return kept, info


def load_extractor(schema: str | None = None,
                   persist_dir: str | None = None) -> TagExtractor:
    """Load the extractor by priority; fall back to built-in default rules when no config is found.

    Args:
        schema: explicit schema file path.
        persist_dir: memory library directory (to find ``tag_schema.json``).
    """
    path = resolve_schema_path(schema, persist_dir)
    if not path:
        return TagExtractor(source="builtin")
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # A broken file must be shouted about: silently using built-in rules makes people believe the config took effect
        logger.error("标签 schema 读取失败，回退内置规则: %s (%s)", path, e)
        return TagExtractor(source=f"builtin(读取失败:{path})")
    ext = extractor_from_config(cfg, source=path)
    logger.info("标签 schema 已加载: %s（keys=%s）", path, sorted(ext.keys.values()))
    return ext


# ----------------------------------------------------------------------
# Compatibility entry points (historical function names, signatures unchanged)
# ----------------------------------------------------------------------
_DEFAULT: TagExtractor | None = None


def default_extractor() -> TagExtractor:
    """Process-level default extractor (lazily built, reads env).

    Same reasoning as ``_resolve_lexicon_path``: **never cache the external
    file's contents**, only cache the one judgment "no explicit config" --
    call :func:`reset_default` when a hot update is needed.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_extractor()
    return _DEFAULT


def reset_default(schema: str | None = None, persist_dir: str | None = None) -> TagExtractor:
    """Rebuild the default extractor (config changes need no process restart)."""
    global _DEFAULT
    _DEFAULT = load_extractor(schema, persist_dir)
    return _DEFAULT


def extract_heading_tags(heading: str) -> dict[str, Any]:
    """Extract structured tags from a heading: ``第43章`` → ``{"章": 43}``.

    Key names default to ``章`` / ``卷`` / ``年`` / ``月`` (renamable via
    schema config; **renames must stay in sync with the caller's
    ``filter_tags``, otherwise filtering silently fails**).
    Only unambiguous structural markers are extracted; returns an empty
    dict when nothing matches.
    """
    return default_extractor().extract(heading, heading=True)


def extract_text_tags(text: str, cap: int = 6) -> dict[str, Any]:
    """Scan **body text** for structured tags (for backfilling existing entries; no source file needed).

    Difference from :func:`extract_heading_tags`: a heading has one
    structural value, while the body text of one memory may mention several
    ("参见第 12 章…本章对应第 5 章"). Here **all are collected**, multiple
    values stored as a list -- the filter side expands lists as OR, whose
    semantics are exactly "this memory involves these chapters".
    """
    ext = default_extractor()
    if cap != ext.cap:
        ext = TagExtractor(keys=ext.keys, suffixes=ext.suffixes,
                           patterns=ext.patterns, month=ext.month,
                           gregorian_year=ext.gregorian_year, cap=cap,
                           num_max=ext.num_max, source=ext.source)
    return ext.extract(text, heading=False)
