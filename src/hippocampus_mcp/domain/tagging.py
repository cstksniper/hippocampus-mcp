# -*- coding: utf-8 -*-
"""结构化标签：**可配置**的结构标记抽取器。

为什么要有这个模块
==================

标签的**过滤侧**（``recall(filter_tags=...)``）从一开始就是通用的——任意键、
AND/OR、``str()`` 归一，代码里没有任何"章/卷/年"的影子。

但**抽取侧**不是：三条正则（``第N章回`` / ``第N卷部`` / ``干支``）写死在导入
脚本里，且只存在于导入脚本中。于是出现两个普遍性缺口：

1. **换语料就要改代码**。合同按「第N条」、病历按「第N次化疗」、日志按
   「2026-09」，全都是同一类需求（"文档自带的结构标记 → 可过滤的标签"），
   却只能改 Python 源码。
2. **服务端写入根本没有抽取能力**。``ingest``/``ingest_batch`` 只接受宿主
   自己抽好的 ``tags``；宿主没抽，库里的条目就永远是"按标签查不到"的那部分。

本模块把"结构标记 → 标签"做成 **抽取器 + 配置**：

* 内置的只是**与领域无关的中文文档结构词**——编号族（章/回/卷/部）与时间族
  （干支纪年、公元年、月份）。**不针对任何具体语料**；
* 键名（``keys``）与被识别的后缀（``suffixes``）都能被配置改名/扩充；
* 任何项目特有的结构标记一律走 ``patterns`` 配置，**不进代码**。

配置来源（优先级从高到低）::

    显式 schema 参数 → HIPPOCAMPUS_TAG_SCHEMA（文件路径）
    → <persist_dir>/tag_schema.json → 内置默认

配置格式（JSON，字段全部可选，只写要覆盖的）::

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

硬约束（改动前务必读）
======================

* **键名是调用方契约**。宿主按 ``filter_tags={"月": "闰四月"}`` 查，这里就得产
  ``月``。改名必须双端同步——**键名不一致 = 过滤静默失效**，不报错。
* **宁缺勿滥**。只认明确的结构性标记，不做任何猜测。错误的标签比没有标签
  更糟：它会让过滤给出**错误答案**（而不是查不到）。
* 值统一 ``int`` / ``str``；多值记列表（过滤侧按 OR 展开），单值降为标量。
* 抽取顺序无关紧要，但**同一条记忆涉及多个同键值时必须全部保留**——
  ``{"章": [5, 12]}`` 的语义是"这条涉及这两章"，只留一个会让它在另一章下消失。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

ENV_TAG_SCHEMA = "HIPPOCAMPUS_TAG_SCHEMA"

# 构造注入缝：旋钮登记在 config（tag_schema），生产路径由 engine
# import 时灌入；解析序（env 先、注入值回退）见 common/config_bridge。
from ..common.config_bridge import apply_config, resolve as _resolve
SCHEMA_FILENAME = "tag_schema.json"

# ----------------------------------------------------------------------
# 中文数字
# ----------------------------------------------------------------------
CN_NUM_CHARS = "零〇一二三四五六七八九十百千两"
_CN_DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn_num_to_int(text: str) -> int | None:
    """中文数字转 int（"四十三" → 43，"十" → 10，"43" → 43）。失败返回 None。

    标签值统一用 int：过滤时 ``{"章": 43}`` 与 ``{"章": "43"}`` 都能对上
    （比较前 ``str()`` 归一），但存 int 更便于人读与排序。
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
# 内置模式
# ----------------------------------------------------------------------
# 干支纪年：60 组合法组合 = 天干(10) × 地支(12) 按位配对，最小公倍数 60
_TIANGAN = "甲乙丙丁戊己庚辛壬癸"
_DIZHI = "子丑寅卯辰巳午未申酉戌亥"
_GANZHI_60 = [_TIANGAN[i % 10] + _DIZHI[i % 12] for i in range(60)]
_GANZHI_ALT = "|".join(_GANZHI_60)

# 键名默认值。**这四个是通用中文文档结构词**，不是某个项目的私有约定。
DEFAULT_KEYS: dict[str, str] = {"章": "章", "卷": "卷", "年": "年", "月": "月"}

# 结构后缀 → 键名。默认只含与历史行为一致的两族；
# 想认「第N条 / 第N节 / 第N幕」等项目特有结构，用配置加，**不要改这里**。
DEFAULT_SUFFIXES: dict[str, str] = {"章": "章", "回": "章", "卷": "卷", "部": "卷"}

# 特称月（农历常见写法）。只收最通用的三个，不枚举"菊月/荷月"这类别称——
# 枚举式词表正是"对着一个语料定方案"的老毛病。
LUNAR_MONTH_NAMES = ("正月", "冬月", "腊月")

_FEATURE = re.compile(r"[\u4e00-\u9fa5]")
_BAD_SUFFIX_CHARS = set("的了是在和与我你他她它")


class TagPattern:
    """一条自定义抽取规则（配置里 ``patterns`` 的项）。

    ``regex`` 必须**有一个捕获组**，其内容作为标签值；``type="int"`` 时经
    :func:`cn_num_to_int` 转换，失败即丢弃（不做二次猜测）。
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
    """标签值排序键：数字按数值排、字符串按字典序，数字在前。

    不能用 ``str(v)`` 直接排——那样 12 会排在 5 前面（``"12" < "5"``），
    章号列表就变成乱的（实际踩过：合并后 ``[12, 5]``）。
    """
    if isinstance(v, bool):  # bool 是 int 子类，单独归到数字前
        return (0, float(v), "")
    if isinstance(v, (int, float)):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def merge_tag_values(a: Any, b: Any) -> Any:
    """把两个标签值合并（标量/列表混用安全），去重后单值降为标量。"""
    vals: list[Any] = []
    for v in (a, b):
        if v is None:
            continue
        vals.extend(v if isinstance(v, list) else [v])
    uniq = sorted(set(vals), key=_tag_value_sort_key)
    return uniq[0] if len(uniq) == 1 else uniq


class TagExtractor:
    """按配置抽取结构化标签。

    Args:
        keys: 内置族 → 键名的覆盖映射（如 ``{"章": "chapter"}``）。
        suffixes: 结构后缀 → 键名（如 ``{"条": "条"}``）。**扩充即新增可识别结构**。
        patterns: 额外的自定义规则（:class:`TagPattern` 或配置 dict）。
        month: 是否抽月份（干支月 / 数字月 / 特称月）。默认 True。
        gregorian_year: 是否抽公元年（``2026年``）。默认 True。
        cap: 同键多值上限。
        num_max: 编号上限（防"第 9999 章"这类明显误抽）。
        filter_keys: **过滤侧白名单**。声明了才校验 `filter_tags` 的键，
            未声明 = 不校验（保持"任意键"现状）。见 :func:`validate_filter_keys`。
        filter_degrade: 非法键是否**软降级**为"忽略该键继续跑"。默认 False
            ＝显式报错。见 :func:`validate_filter_keys` 的取舍说明。
        source: 配置来源标签，仅用于日志/自述。
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
        # 空列表视同"未声明"（写了 `[]` 等于把所有键都判非法，几乎必然是误配）
        fk = [str(k) for k in (filter_keys or []) if str(k)]
        self.filter_keys: tuple[str, ...] | None = tuple(fk) if fk else None
        self.filter_degrade = bool(filter_degrade)
        self.source = source
        self.patterns = [
            p if isinstance(p, TagPattern) else TagPattern(**p)
            for p in (patterns or [])
        ]

        # 后缀按长度降序，保证「单元」优先于「元」这类前缀（目前无重叠，仍按惯例排）
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
        # ISO 日期：``2026-09-01`` / ``2026/9/1`` / ``2026.09``。
        # 为什么必须单独一条：``_re_gyear`` 要求「年」字后缀，而工程类语料通篇写
        # ISO 日期（真实库的无标签条目里大部分含 ``YYYY-MM-DD``，内置抽取器
        # 却一条都匹配不上）。日部分可选：只给到年月（``2026.09``）也要认。
        # 前视 ``(?!\d)`` 拦住 ``2026-09-011`` 这类长数字串的前缀误命中；
        # 月份限 1–12、日限 1–31，避免把版本号 / 模型编号当日期。
        self._re_iso_date = re.compile(
            r"(?<!\d)(1[0-9]{3}|20[0-9]{2})[-/.]"
            r"(0?[1-9]|1[0-2])"
            r"(?:[-/.](0?[1-9]|[12]\d|3[01]))?(?!\d)"
        )
        # 特称月（正月/冬月/腊月）与中文数字月分开写：
        # 「正/冬/腊」不在中文数字字符集里，两者不重叠，各自的捕获组也更简洁。
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
        """自述当前生效的抽取规则（写进工具返回值，便于宿主核对键名）。"""
        return {
            "source": self.source,
            "keys": dict(self.keys),
            "suffixes": dict(self.suffixes),
            "month": self.month,
            "gregorian_year": self.gregorian_year,
            "cap": self.cap,
            "custom_patterns": [p.to_dict() for p in self.patterns],
            # 过滤侧白名单。None = 未声明 = 不校验（宿主可用任意键）
            "filter_keys": list(self.filter_keys) if self.filter_keys else None,
            "filter_degrade": self.filter_degrade,
        }

    # ------------------------------------------------------------------
    def _family_key(self, family: str | None) -> str | None:
        """族名 → 输出键名。

        **两步映射，不能省**：``suffixes`` 把「后缀」映到**族**（条 → 条族），
        ``keys`` 再把「族」映到**输出键名**。早先只有一步（后缀直接给键名）时，
        配置改 ``keys`` 会被默认 ``suffixes`` 覆盖掉——表现为"键名改了没生效"，
        且不报错。
        """
        if not family:
            return None
        return self.keys.get(family, family)

    def _numbered(self, text: str) -> list[tuple[str, Any]]:
        """编号族：``第N<后缀>`` / ``chN``（后者只作"章"的英文写法）。"""
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
        """裸「卷三」写法（无「第」）。

        只在**标题开头**认：正文里"卷三"可能只是叙述（"卷三之后"），
        无「第」时唯一可靠的锚点就是行首。
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
        """时间族·年：干支 + 公元。干支在前，保持历史优先序。"""
        out: list[tuple[str, Any]] = []
        for m in self._re_ganzhi.finditer(text):
            out.append((self.year_key, m.group(0)))
        if self.gregorian_year:
            for m in self._re_gyear.finditer(text):
                out.append((self.year_key, int(m.group(1))))
            # ISO 日期里的年（``2026-09-01`` → 2026）。干支在前、公元在后
            # 的历史优先序不变；ISO 与「2026年」重复时由 ``_assemble`` 去重。
            for m in self._re_iso_date.finditer(text):
                out.append((self.year_key, int(m.group(1))))
        return out

    def _months(self, text: str) -> list[tuple[str, Any]]:
        """时间族·月：干支月（戊子月）/ 特称月（正月·闰四月）/ 数字月（3月·3月份）。"""
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
        # ISO 日期里的月。放在最后——中文月（正月/闰四月/10月）优先级更高，
        # 标题模式「按第一个命中取值」的既有行为不受影响。
        # 月份已在正则里限 1–12，但 ``09`` 要归一成 int 9，否则过滤侧
        # ``filter_tags={"月": 9}`` 会与 ``"09"`` 对不上（不过标签侧统一走
        # ``str()`` 归一，所以这里 int 化只为与「2026年9月」那条路同口径）。
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
        """抽标签。

        Args:
            heading: 标题模式。标题一行只有一个结构值（``### 第43章``），
                所以每个键**只取第一个**命中，且裸「卷三」写法也在此时才认。
                正文模式（默认）**全部收集**：一条记忆可能涉及多章
                （"参见第 12 章…本章对应第 5 章"），只留一个会让它在另一章下消失。
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
                # 标题模式：**按出现顺序取第一个**，不排序。标题里的第一个结构值
                # 才是它的主题（"卷三 己丑年" → 卷=3，年=己丑）；排序会把 int
                # 提前，干支年就被公元年挤掉了。
                out[key] = vals[0]
                continue
            # 正文模式：数值按数值序、字符串按字典序——不能用 str() 排
            # （12 会跑到 5 前面）。
            vals = sorted(vals, key=_tag_value_sort_key)[: self.cap]
            out[key] = vals[0] if len(vals) == 1 else vals
        return out


# ----------------------------------------------------------------------
# 配置加载
# ----------------------------------------------------------------------
def resolve_schema_path(explicit: str | None = None,
                        persist_dir: str | None = None) -> str | None:
    """解析标签 schema 文件路径：显式 > env > ``<persist_dir>/tag_schema.json``。

    与专名词库同款约定（``_resolve_lexicon_path``）：**每次现算不缓存**——
    服务启动后才放进 schema 文件是常见操作顺序，缓存 None 会让它永远读不到。
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
    """从配置 dict 造抽取器。字段非法即抛错——**不静默降级成默认规则**。

    静默降级会让"我明明配了"变成"其实没生效"，这是最难查的一类故障。
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"标签 schema 必须是 JSON 对象，得到 {type(cfg).__name__}")
    known = {"keys", "suffixes", "patterns", "month", "gregorian_year",
             "cap", "num_max", "filter_keys", "filter_degrade", "observed"}
    # 下划线开头的是**给看文件的人**留的元数据（如 build_tag_schema.py 写的
    # `_generated`），抽取器本就不读——不该每加载一次就 warning 一遍。
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
    """校验 ``filter_tags`` 的**键**是否在白名单里。

    返回 ``(effective_tags, err)``，两者必有其一为 ``None``（**不同时非空**）：

    * ``(tags, None)`` —— 放行。含两种情况：白名单未声明、或键全部合法。
    * ``(None, err)``  —— 拦截（``degrade=False``）或全被剥光（``degrade=True``）。

    为什么只在**声明了白名单**时才校验
    --------------------------------
    ``filter_tags`` 从一开始就是通用的（任意键 / AND-OR / ``str()`` 归一）。
    在库里**还没有** ``tag_schema.json`` 时开校验，等于把库内现有键全部判成
    非法 ⇒ 从"键名写错静默返回空"变成"每次调用都报错"，是纯粹的倒退。
    故顺序必须是**先建档，再开校验**——次序不能反。

    为什么默认**不软降级**
    ----------------------
    降级（把非法键扔掉继续跑）会把"键名写错"重新变回静默：去掉非法键后检索
    照样返回结果，调用方读到的仍是"库里就这些"，与不校验时一模一样。键校验
    要堵的正是这个洞，所以缺省 ``degrade=False`` 显式报错；确实需要降级时必须
    在 schema 里**显式**写 ``"filter_degrade": true``，且返回体必带
    ``degraded: true`` 与 ``degraded_from``，让调用方看得见被改过。

    只校验键，不校验值：值的取值域由语料决定（``章`` 可以是 43 也可以是"闰四月"），
    拦值会误伤合法查询；且值写错已经被 ``matched=0`` 回显覆盖了。
    """
    if not tags:
        return tags, None
    allowed = extractor.filter_keys
    if not allowed:                      # 未声明白名单 ⇒ 不校验（逐位不变）
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
    # 软降级：剥掉非法键，剩下的继续跑；剩下的为空则等同于"不过滤"，仍要回显
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
    """按优先级加载抽取器；找不到配置就用内置默认规则。

    Args:
        schema: 显式 schema 文件路径。
        persist_dir: 记忆库目录（找 ``tag_schema.json``）。
    """
    path = resolve_schema_path(schema, persist_dir)
    if not path:
        return TagExtractor(source="builtin")
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # 文件坏了必须吼一声：静默用内置规则会让人以为配置生效了
        logger.error("标签 schema 读取失败，回退内置规则: %s (%s)", path, e)
        return TagExtractor(source=f"builtin(读取失败:{path})")
    ext = extractor_from_config(cfg, source=path)
    logger.info("标签 schema 已加载: %s（keys=%s）", path, sorted(ext.keys.values()))
    return ext


# ----------------------------------------------------------------------
# 兼容入口（历史函数名，签名不变）
# ----------------------------------------------------------------------
_DEFAULT: TagExtractor | None = None


def default_extractor() -> TagExtractor:
    """进程级默认抽取器（惰性构造，读 env）。

    与 ``_resolve_lexicon_path`` 同理：**不缓存外部文件的内容**，只缓存
    "没有显式配置"这一个判断——需要热更新时调用 :func:`reset_default`。
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_extractor()
    return _DEFAULT


def reset_default(schema: str | None = None, persist_dir: str | None = None) -> TagExtractor:
    """重建默认抽取器（配置改了不用重启进程）。"""
    global _DEFAULT
    _DEFAULT = load_extractor(schema, persist_dir)
    return _DEFAULT


def extract_heading_tags(heading: str) -> dict[str, Any]:
    """从标题抽结构化标签：``第43章`` → ``{"章": 43}``。

    键名默认 ``章`` / ``卷`` / ``年`` / ``月``（可经 schema 配置改名，
    **改名必须与调用方的 ``filter_tags`` 同步**，否则过滤静默失效）。
    只抽明确的结构性标记，抽不到就返回空字典。
    """
    return default_extractor().extract(heading, heading=True)


def extract_text_tags(text: str, cap: int = 6) -> dict[str, Any]:
    """扫**正文**抽结构化标签（存量回填用，不需要源文件）。

    与 :func:`extract_heading_tags` 的区别：标题只有一个结构值，正文里一条
    记忆可能同时提到多个（"参见第 12 章…本章对应第 5 章"）。这里**全部收集**，
    多值记列表——过滤侧对列表走 OR，语义正是"这条记忆涉及这几章"。
    """
    ext = default_extractor()
    if cap != ext.cap:
        ext = TagExtractor(keys=ext.keys, suffixes=ext.suffixes,
                           patterns=ext.patterns, month=ext.month,
                           gregorian_year=ext.gregorian_year, cap=cap,
                           num_max=ext.num_max, source=ext.source)
    return ext.extract(text, heading=False)
