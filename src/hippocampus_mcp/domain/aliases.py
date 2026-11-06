# -*- coding: utf-8 -*-
"""通用别名发现：找出"同一实体的多种称法"，只交付候选与证据，不自动归并。

**为什么需要这一步**：别名归一的机制（``lexicon_add("别名 => 主名")``）已经能修
数据，但**发现**仍然全靠人——同一个实体在语料里换了三种写法，人得先读到才会
想到去登记。这一步把"读全库找称法"的活从人手里拿回来。

**为什么只做候选**：别名一旦合并就**改变实体身份**，猜错等于把两个不同对象的
记忆混成一档，而且事后很难拆。所以输出永远是"候选 + 信号 + 证据 + 置信度"，
落盘由调用方用 ``lexicon_add`` 确认——与 ``heuristic`` 实体只做候选是同一立场。

四条信号，全部**领域无关**（代码不认"人"也不认"小说"，只认中文命名惯例与统计）：

  1. ``cue``    称谓提示词：``甲，字乙`` / ``甲，又名乙`` / ``甲，简称乙``。
     这是中文命名惯例，对人物、机构、产品、地名同样成立；词表可扩展
     （``<词库同目录>/alias_cue_words.txt``，一行一个，``#`` 注释）。
     **分隔符必须在提示词之前**：提示词本身多是常用字（字/号/名/称），
     "句号""编号""简称"里的那个字不是称谓——区别就在分隔符的位置
     （详见 :func:`mine_patterns`）。
  2. ``paren``  括号同位语：``甲（乙）``。比 1 低一档——括号里也常是**解释**
     而不是称法（"顾言（作者）"），所以只看括号内是**名字形状**的（或括号内
     自带称谓提示词，如"甲（字乙）"）。
  3. ``cooc``   共现 + 提升度：``lift = P(a,b) / (P(a)·P(b))``。总是一起出现、
     远超独立假设的两个名字，多半指同一个对象（"简称 + 全称"最常见）。
  4. ``abbrev`` 包含关系：``北大`` ⊂ ``北京大学``。缩写 / 全称 / 去掉后缀的写法
     都走这条。

置信度是**可解释的加权和**（权重见 :data:`SIGNAL_WEIGHT`），不是黑盒打分：
每个候选都能回答"因为哪条信号、哪段原文"。

用法::

    from hippocampus_mcp import aliases
    cue, stat = aliases.load_cue_words("<词库目录>/alias_cue_words.txt")
    hits = aliases.mine_patterns(text, cue, known={"顾言"})   # 模式类
    pair, docs, n = aliases.cooccurrence(records)             # 统计类

本模块**不依赖 brain / 存储**，纯函数 + 数据类，便于单测。
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

# ---------------------------------------------------------------- 词表
# 称谓提示词：**通用的中文命名词汇**，与语料题材无关。人（字/号/小名/笔名）、
# 机构（简称/全称/旧称）、产品（代号/型号）、地名（旧称/别称）都用得上，
# 所以放内置层开箱即用；项目特有的写法再写文件补充。
#
# 注意：**不要**把"年/月/日/章/卷"这类结构词混进来——它们是标签抽取器（tagging）
# 的领域，在这里只会产出"本章 => 正文"这种假别名。
ALIAS_CUE_WORDS: tuple[str, ...] = (
    # 人：名以外的称法
    "字", "号", "别号", "法号", "道号", "谥号", "庙号", "绰号", "浑号",
    "小名", "乳名", "本名", "原名", "旧名", "曾用名", "笔名", "艺名",
    "化名", "自号", "雅号", "封号", "帝号",
    # 通用称法
    "别称", "别名", "又名", "又称", "亦名", "亦称", "也叫", "外号",
    "简称", "缩写", "全称", "俗称", "通称", "尊称", "代号", "昵称",
    "译名", "英文名", "学名", "旧称",
    # 英文（同一个机制，不必为英文单开一条路）
    "aka", "also known as", "formerly known as",
)

ALIAS_CUE_FILENAME = "alias_cue_words.txt"

# 只用于**别名门控**的通用词：形状上像名字（2 字纯汉字），语义上是角色 /
# 关系 / 通用名词，绝不会是某个实体的"另一种称法"（"顾言（作者）"里的"作者"
# 就是典型）。匹配是**整词相等**，不做子串压制——这份表的用途单一，
# 与 ``lexicon_suggest`` 的停用词表刻意不同源：那份要做子串压制，
# 把"人物"放进去会连带压掉"人物志"这类真书名。
ALIAS_STOP_WORDS: frozenset[str] = frozenset({
    "作者", "读者", "编者", "译者", "主角", "配角", "反派", "人物", "角色",
    "客户", "用户", "公司", "团队", "项目", "产品", "服务", "系统", "平台",
    "数据", "内容", "问题", "方法", "方案", "结果", "原因", "目的", "过程",
    "时间", "地方", "事情", "东西", "大家", "我们", "他们", "别人", "自己",
    "对方", "双方", "上面", "下面", "期间", "以后", "之前", "当时", "目前",
    "第一", "第二", "第三", "最后", "全部", "所有", "部分", "其他", "别的",
})

# 各信号对置信度的贡献。**合成上界 1.0**；``cooc`` 会再乘"提升度强度"，
# 所以单靠共现最多拿 0.30 —— 单靠共现的候选天然排在模式类后面。
SIGNAL_WEIGHT: dict[str, float] = {
    "cue": 0.50,
    "paren": 0.30,
    "abbrev": 0.25,
    "cooc": 0.30,
}

# 提升度到达该值即算"满强度"（lift=5 给满分，lift=2 给 1/4）。
_LIFT_FULL = 5.0
# 支持度加成上限（共现条数越多越可信，但别让它盖过信号本身的差异）。
_SUPPORT_BONUS_MAX = 0.10

# ---------------------------------------------------------------- 字符类
_CJK = "\u4e00-\u9fa5"
# 名字允许的字符：汉字、拉丁字母、数字、间隔号（"阿·伊万诺夫"）。
# 刻意**不含**全角标点与括号——它们是名字边界，不是名字的一部分。
_NAME_CHARS = rf"{_CJK}A-Za-z0-9·"
_RE_NAME_RUN = re.compile(rf"[{_NAME_CHARS}]+")
_CJK_NAME_RE = re.compile(rf"^[{_CJK}][{_CJK}·]{{1,11}}$")
_ASCII_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{1,29}$")
# 名字两侧可出现的分隔符（含中文引号/书名号）
_SEP_CHARS = "，,、：:；;　 \t\"'“”‘’「」『』《》〈〉【】()（）"
# 纯数字/序数词用到的字符。带上"第"是为了让"第三/第二十"这类序数词整体被判非名字
# ——否则它会以"2 个汉字"的形状混进候选（"第"本身在《》里也常见，但那时两侧还有
# 别的字，不会被这条规则波及）。
_NUMERIC_CHARS = set("0123456789０１２３４５６７８９〇零一二三四五六七八九十百千万亿两第")
_RE_PAREN = re.compile(rf"([{_NAME_CHARS}]{{2,16}})\s*[（(]([^（()）\n]{{1,24}})[）)]")

# 未登记两侧的整串采纳上限（字）。超过就宁可放弃：
# "此人是顾言，字子安"这种没有分隔符的写法，整串取出来混着谓语，
# 只有短串才值得赌一把；长串先去词库里登记，随后自然会被认出。
_MAX_RUN_UNKNOWN = 4


@dataclass
class Pair:
    """一条候选对的原始证据（未打分、未定方向）。"""

    a: str                      # 模式里在前的那个名字
    b: str                      # 模式里在后的那个名字
    signal: str                 # cue / paren / abbrev / cooc
    snippet: str                # 命中处的原文片段（给人看）
    cue: str = ""               # 命中的提示词（signal=cue 时有值）
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str]:
        return self.a.lower(), self.b.lower()


# ---------------------------------------------------------------- 词表加载
def load_cue_words(
    path: str | None = None,
    extra: Sequence[str] | None = None,
) -> tuple[set[str], dict[str, int]]:
    """返回 ``(提示词集合, 各层条数)``。

    三层并集，与前缀**只增不覆盖**：**内置通用**（代码里，跨项目开箱即用）∪
    **项目文件**（``path``，一行一个）∪ **本次调用扩展**（``extra``）。
    项目专有的写法（例如某题材特有的"化名"习惯）写文件，别改内置层。
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
    """提示词正则（**长的优先**，否则"又名"会被"名"式的短词截断）。"""
    words = sorted({w for w in cue_words if w}, key=len, reverse=True)
    if not words:
        return None
    return re.compile("|".join(re.escape(w) for w in words), re.I)


# ---------------------------------------------------------------- 名字形状
def name_like(s: str, min_len: int = 2, max_len: int = 12) -> bool:
    """是否是"名字形状"的字符串。

    只做**形状**判断，不做语义判断（那是停用词与置信度的事）：
    纯汉字（2~12 字，允许间隔号）或纯拉丁（以字母开头，2~30 字符）。
    纯数字/纯数字词（"第三""一百二十"）一律不算——它们几乎总是量词短语。
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
    """从一段连续名字字符里取**尾部**的名字，已知词优先。

    为什么要"已知词优先"：语料里常有"此人是顾言，字子安"这种没分隔符的写法，
    规则本身分不出"此人是"是谓语还是名字的一部分。但**已知词**（词库/已建档
    实体）能：从长到短找后缀，命中即可，于是正确切出"顾言"。
    没有已知词可用时，只有整串够短（<= 4 字）才采纳——宁缺毋滥。
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
    """同 :func:`_prefer_known_suffix`，取**头部**的名字。"""
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
    """位置 ``idx`` 的前/后 ``span`` 个字符里是否含分隔符（用于判断提示词是否自成一段）。"""
    if before:
        seg = text[max(0, idx - span):idx]
    else:
        seg = text[idx:idx + span]
    return any(ch in _SEP_CHARS for ch in seg)


def is_alias_noise(word: str, extra: Iterable[str] | None = None) -> bool:
    """是否是**整词**级别的别名噪声（角色/关系/通用名词，见 :data:`ALIAS_STOP_WORDS`）。

    刻意只做整词相等：别名是"某个实体的另一种称法"，通用名词整词出现在
    候选位就是错的；但它作为**子串**出现在真名里（"人物" ⊂ "人物志"）没关系，
    所以不能像候选专名那样做子串压制。
    """
    w = (word or "").strip()
    if not w:
        return True
    if w in ALIAS_STOP_WORDS:
        return True
    return any(w == str(x).strip() for x in (extra or []) if str(x).strip())


def snippet_around(text: str, start: int, end: int, width: int = 36) -> str:
    """取命中处的上下文片段（单行、去多余空白）。"""
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    frag = " ".join(text[lo:hi].split())
    return ("…" if lo > 0 else "") + frag + ("…" if hi < len(text) else "")


# ---------------------------------------------------------------- 信号 1/2
def mine_patterns(
    text: str,
    cue_words: Iterable[str],
    known: set[str] | None = None,
    min_len: int = 2,
    max_len: int = 12,
    with_paren: bool = True,
    stats: dict[str, int] | None = None,
) -> list[Pair]:
    """从一段文本里挖"称谓提示词"与"括号同位语"两类候选。

    Args:
        cue_words: 提示词表（见 :func:`load_cue_words`）。
        known: **已知词集合**（小写）——词库词条 + 受信任实体名。它只用于
            **切分辅助**，不参与对错判断：有它就能从粘连文本里切出正确的名字。
        with_paren: 是否顺带扫括号同位语。
        stats: 可选计数器（就地累加），供调用方判断"门槛是不是设错了"。
            目前记 ``cue_no_sep_before``（提示词前面没有分隔符而丢弃）与
            ``cue_sep_after_only``（其中"分隔符只在提示词之后"的那种——
            假命中的典型形态，见下）。

    Returns:
        原始候选 :class:`Pair` 列表（可能含重复，由调用方汇总）。
    """
    out: list[Pair] = []
    if not text:
        return out
    known = known or set()
    cre = _cue_re(cue_words)

    # ---- 信号 1：称谓提示词 -------------------------------------------
    if cre is not None:
        for m in cre.finditer(text):
            cue = m.group(0)
            # 提示词必须**自成一段，且分隔符在它之前**（"，字子安" / "（字子安）"），
            # 或两侧都是已知词（"顾言字子安"这种省标点的写法）。否则"名字""文字"
            # "数字"里的"字"会被当成称谓提示词，凭空造出"这个名 => 很有意思"。
            sep_before = _has_sep(text, m.start(), before=True)
            # 左侧：截到"上一个提示词"之后，免得"字子安号雪芹"把两边串起来
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
            # 右侧：截到"下一个提示词"之前
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
            # 分隔符必须在提示词**之前**。真实库实测（中文小说
            # 9359 条）这条比"任意一侧有分隔符"强得多：跑出来的假命中全是同一个
            # 形态——提示词恰好是某个普通词的**后一个字**：
            # 「碎句**号**、长段落」（句号）「全量换**号**、活跃文档」（编号）
            # 「否定排比、破折**号**」「同章全称后简**称**」（简称）
            # 「何时取**字**，另行定」。它们的共同点是分隔符在提示词**之后**；
            # 而真正的称法写法（"甲，字乙"）分隔符一定在**之前**。
            # 放宽成"任一侧有分隔符"时，这类对占了候选的一半以上（25 条里 16 条）。
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

    # ---- 信号 2：括号同位语 -------------------------------------------
    if with_paren and ("（" in text or "(" in text):
        for m in _RE_PAREN.finditer(text):
            outer_run, inner = m.group(1), m.group(2).strip()
            outer = _prefer_known_suffix(outer_run, known, min_len, max_len)
            if not outer:
                continue
            inner_cue = cre.search(inner) if cre is not None else None
            if inner_cue:
                # "顾言（字子安）" → 提示词之后才是称法
                seg = inner[inner_cue.end():].lstrip(_SEP_CHARS)
                head = re.match(rf"[{_NAME_CHARS}]+", seg)
                if not head:
                    continue
                b = _prefer_known_prefix(head.group(0), known, min_len, max_len)
                cue = inner_cue.group(0)
            else:
                # 括号内整段就是名字形状 → 弱候选（括号里也常是解释）
                b = inner if name_like(inner, min_len, max_len) else None
                cue = ""
            if not b or b.lower() == outer.lower():
                continue
            out.append(Pair(
                a=outer, b=b, signal="paren", cue=cue,
                snippet=snippet_around(text, m.start(), m.end()),
            ))
    return out


# ---------------------------------------------------------------- 信号 3/4
def cooccurrence(
    record_names: Iterable[tuple[str, set[str]]],
    max_names_per_record: int = 30,
) -> tuple[dict[tuple[str, str], int], dict[str, int], int]:
    """统计"同名共现"：``(pair_counts, name_record_counts, n_records)``。

    输入是一串 ``(记忆 id, 该条里出现的名字集合)``。名字的**来源**由调用方
    决定（已建档实体 + 词库命中即可）——本函数只管统计，不猜来源。

    ``max_names_per_record`` 是**成本闸门**：一条记忆里挂了几十个名字时，
    两两配对是 O(n²)。超过上限就截断（按名字排序后取前 N），避免一条
    "实体清单"式长文本把整轮挖掘拖垮。
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
    """提升度 ``P(a,b) / (P(a)·P(b))``；独立时 ≈1，成正比于 1 时说明是同指。"""
    if count_ab <= 0 or count_a <= 0 or count_b <= 0 or n_records <= 0:
        return 0.0
    expected = count_a * count_b / n_records
    return count_ab / expected if expected > 0 else 0.0


def confidence_of(
    signals: Iterable[str],
    support: int = 0,
    lift: float = 0.0,
) -> float:
    """把信号集合折算成 0~1 的置信度（**加权和，可解释**）。

    ``cooc`` 的贡献随提升度缩放：``lift`` 到 :data:`_LIFT_FULL` (5) 拿满，
    ``lift=2`` 只拿 1/5。支持度再给一点加成（上限
    :data:`_SUPPORT_BONUS_MAX`），因为"在多条记忆里都成立"比"只在一条里
    出现"抗噪。
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
    """返回 ``"a<b"`` / ``"b<a"`` / ``""``（判断包含关系，用于缩写/全称）。"""
    la, lb = a.lower(), b.lower()
    if la == lb:
        return ""
    if la in lb:
        return "a<b"
    if lb in la:
        return "b<a"
    return ""


def abbrev_relation(a: str, b: str) -> str:
    """包含关系**且**两侧都是汉字串时才算"简称 ⊂ 全称"，否则返回 ``""``。

    为什么加这条限制（真实库实测踩出来的）：
    拉丁串里的子串关系绝大多数是**编号/型号**，不是简称——``ch4`` ⊂ ``ch49``、
    ``v1`` ⊂ ``v12``，它们在库里各自代表完全不同的对象，却被"包含 + 共现"
    推成了候选（实测一次跑出十几条 ``ch3x => ch3`` 这种垃圾）。
    中文的简称/全称才是连续子串关系（``北大`` ⊂ ``北京大学``、``清华`` ⊂ ``清华大学``），
    所以这条规则不是"防英文"，而是"防把编号当简称"。
    """
    if not (is_cjk_only(a) and is_cjk_only(b)):
        return ""
    return substring_relation(a, b)


def is_cjk_only(s: str) -> bool:
    t = (s or "").strip()
    return bool(t) and all(
        ("\u4e00" <= ch <= "\u9fa5") or ch == "·" for ch in t
    )
