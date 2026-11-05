# -*- coding: utf-8 -*-
"""敏感内容匹配（敏感双门禁第一道门禁的判据层）。

模式级内置兜底 + 部署词表匹配的**纯函数**——env 词表内容与
（可替换的）模式集都由调用方注入，本模块不读 env、不做 I/O。

消费方两处（都是服务层）：

* ``OrganizeService.consolidate`` 的提升候选标注（被拦条目必须
  标注命中词与位置，不静默）；
* ``WriteService.store_memory`` 对目标库名为 ``user`` 的**专属敏感过滤**
  （堵直写绕路；**存量库行为不变**——非 user 库零变化；
  家库写边界将来在工厂层落守卫、user 库按角色豁免，与本过滤正交组合）。

匹配语义（收窄为两种，不猜第三种）：

* 词表条目 → **子串**匹配（大小写敏感——词表是部署自管的，写什么匹配
  什么；要大小写不敏感请在词表里同时给两种写法）；
* 模式 → **正则**（内置四条：18 位身份证 / 11 位手机号 / 常见密钥格式）。

词表缺省时**模式级兜底仍在**：红线 1"任一不过不写"的第二道门（显式
确认）不依赖词表存在，第一道门的兜底也不依赖。
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

#: 敏感类别关键词表（D1：隐私打标的类别层，模块级受管常量，Q-9 拍板件）。
#: 与 BUILTIN_PATTERNS 同款口径：模块常量、可被调用方整体替换注入；
#: 关键词是**子串**匹配（大小写敏感）——它是打标提示不是执法判定，
#: 漏检是常态，靠 access_log 周期审计兜底（设计稿 §4.1 漏检口径）。
#: 扩类别走显式变更（改这里 + 补测试），不做运行时注入面。
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

#: 模式级内置兜底（模块级常量，非 env——可被调用方整体替换注入）。
#: (kind, compiled)；kind 进命中详情，供调用方自曝"哪类格式命中"。
BUILTIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 18 位身份证号（末位可 x/X；前后不能是数字，避免吃进长数字串）
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    # 11 位手机号（13-19 号段；前后不能是数字）
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    # 常见密钥格式：OpenAI sk- / GitHub token / 通用 api_key= 赋值
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
    """扫描 content，返回命中清单（``[{word, start, end, kind}]``，按位置序）。

    Args:
        content: 待检文本。
        words: 部署词表条目（调用方从 env 文件读好传入；子串匹配）。
        patterns: ``(kind, compiled)`` 序列；``None`` = 内置
            :data:`BUILTIN_PATTERNS`。传空序列 = 关模式通道（只跑词表）。

    Returns:
        每次出现一条：``{"word": 命中文本, "start": 起始, "end": 终止,
        "kind": "word" 或模式 kind}``，按 ``start`` 升序。**空列表 = 干净**。
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
    """按类别关键词给内容打敏感类别标签（D1 纯函数，只分类不执法）。

    与 :func:`match_sensitive` 的分工：那是对**凭据类**的红线（命中即拒写），
    本函数是对**类别级**隐私的打标（行程/健康/财务/人际）——命中不拒写，
    只决定 quarantine 卡片上的 ``sensitivity`` 标签（D2a 写侧门禁消费），
    供 drift_audit 人工复审分组。关键词表见 :data:`SENSITIVITY_CATEGORIES`。

    Args:
        content: 待检文本。
        categories: 类别关键词表；``None`` = 内置
            :data:`SENSITIVITY_CATEGORIES`。传空 dict = 关类别通道。

    Returns:
        ``{"level": "sensitive"|"clean", "categories": [命中类别名，排序],
        "hits": [{category, word, start, end}]}``。level 与 categories
        一致：有命中必 sensitive，无命中必 clean（单一真相，调用方不必自己再判）。
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
            # 每类别只记首个命中位置：打标要的是"哪类"，不是穷举所有出现处
            #（穷举交给需要执法位置的 match_sensitive，这里省 token）。
            hits.append({"category": cat, "word": w,
                         "start": start, "end": start + len(w)})
            break
    hits.sort(key=lambda h: (h["start"], h["category"]))
    cats = sorted(seen)
    return {"level": "sensitive" if cats else "clean",
            "categories": cats, "hits": hits}
