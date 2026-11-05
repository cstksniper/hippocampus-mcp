# -*- coding: utf-8 -*-
"""MCP 工具参数的纯解析助手（L0：纯标准库、零 I/O，谁都能 import）。

从 server.py 抽出（工具层/用例层共享的解析件）：这几个函数工具层（L4）
与用例层（L3）都要用，而 L3 按架构纪律不准 import L4——放 L0 共享底座
是唯一不制造环的位置。三个函数**逐字搬迁**（含 docstring），行为零变化；
server.py 改为别名 import，原有调用点一处不动。

两个 ``parse_str_list`` 是**刻意并存的两档严格度**（各自有主，别再合并）：

* :func:`parse_str_list`（严格版）——坏 JSON **显式报错**
  （``(None, err)`` 元组）。update_tags 的 clear_keys / missing_keys /
  memory_ids 用它：静默回退会让调用方以为"没有要删的键"，整次操作
  变成一次安静的空跑。
* :func:`parse_str_list_lenient`（lenient 版）——JSON/逗号都收，解析
  失败**回退逗号切分**。knowledge_topic 的 evidence/topics、词库三
  工具（lexicon_add / lexicon_suggest / lexicon_alias_suggest）用它：
  那些参数的语义是"顺手传几个词"，静默会表现成"参数传了没生效"，
  比直接报错难查得多。

与 server.py 版的一处刻意差异：``check_tag_keys`` **不在这里**——它要调
``domain.tagging.validate_filter_keys``（L2），common 不准依赖 domain
（依赖矩阵），故放 ``application/tags.py``。
"""
from __future__ import annotations

import json
from typing import Any


def parse_tags_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """把 MCP 传进来的 JSON 对象字符串解析成 dict。

    MCP 工具参数走简单类型，标签以 JSON 字符串传入。解析失败必须**显式报错**，
    不能静默当空——否则宿主以为挂了标签，实际一条都没挂，过滤时才发现查不到。
    """
    text = (raw or "").strip()
    if not text:
        return None, None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"不是合法 JSON: {exc}"
    if not isinstance(loaded, dict):
        return None, '必须是 JSON 对象，如 {"章": 43}'
    return loaded, None


def parse_str_list(raw: str) -> tuple[list[str] | None, str | None]:
    """把 MCP 传进来的 JSON 数组字符串解析成字符串列表（严格版）。

    与 :func:`parse_tags_json` 同一条纪律：解析失败**显式报错**，不能静默当空
    ——空列表会让调用方以为"没有要删的键"，于是整次操作变成一次安静的空跑。
    """
    text = (raw or "").strip()
    if not text:
        return None, None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"不是合法 JSON: {exc}"
    if not isinstance(loaded, list):
        return None, '必须是 JSON 数组，如 ["域", "章"]'
    return [str(x) for x in loaded], None


def parse_str_list_lenient(value: Any) -> list[str]:
    """把"列表类参数"解析成字符串列表（JSON 数组或逗号分隔都收）。

    协议上这类参数一律是字符串：``["同安","顾父"]`` 与 ``"同安, 顾父"`` 等价。
    解析失败**回退到逗号切分**而不是静默当空——静默会表现成"参数传了没生效"，
    比直接报错难查得多。
    """
    if not isinstance(value, str):
        # 直接以 Python 调用（测试/脚本）时也接受序列
        if isinstance(value, (list, tuple, set)):
            return [str(x).strip() for x in value if str(x).strip()]
        return []
    s = value.strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            return [t.strip() for t in s.strip("[]").split(",") if t.strip()]
    return [t.strip() for t in s.split(",") if t.strip()]


def parse_tagops_args(tags: str, clear_keys: str, missing_keys: str,
                      memory_ids: str) -> tuple[dict[str, Any] | None, str | None]:
    """update_tags 四段 JSON 协议解析（行为逐位不变）。

    四段解析（tags / clear_keys /
    missing_keys / memory_ids）逐段严格解析，**第一处错误即停**——错误消息
    带字段名前缀（``"tags 解析失败：..."``），与抽助手前的工具体逐字一致。

    为什么四段必须留在适配层而不是下沉服务：
    这些解析现行排在库解析（``_brain_for``）**之前**，「坏库名 + 坏 JSON」
    必须先报语法错；且错误体**不带** library 首键（库还没解析）。下沉服务
    （工厂调用即库解析）会同时翻转错误次序与信封形态——那是产品决策，
    不是重构副产物。

    两档严格度分工见模块 docstring：本函数四段全部走严格版
    （:func:`parse_str_list`），坏 JSON 显式报错，不与 lenient 版混名。
    """
    parsed, err = parse_tags_json(tags)
    if err:
        return None, f"tags 解析失败：{err}"
    keys, err = parse_str_list(clear_keys)
    if err:
        return None, f"clear_keys 解析失败：{err}"
    mkeys, err = parse_str_list(missing_keys)
    if err:
        return None, f"missing_keys 解析失败：{err}"
    ids, err = parse_str_list(memory_ids)
    if err:
        return None, f"memory_ids 解析失败：{err}"
    return {"tags": parsed, "clear_keys": keys,
            "missing_keys": mkeys, "memory_ids": ids}, None


def filter_applied(parsed_tags: dict[str, Any] | None,
                   matched: int | None,
                   degraded: dict[str, Any] | None = None) -> dict[str, Any]:
    """``filter_applied`` 回显：本次用的标签域 + 过滤后候选条数。

    存在的理由：``filter_tags`` 键名写错 / 值不匹配时，检索会**静默**返回空
    ——不报错也不提示，调用方只能当成"库里真没有"。把候选条数显式回显：

        ``tags=None``    没设过滤（全库检索）
        ``matched=None`` 同上（无过滤时没有"候选数"这个概念）
        ``matched=0``    过滤**没命中任何条目**（键名/值对不上，静默失败）
        ``matched=N``    收窄到 N 条后再检索

    ``matched=0`` 与"检索后为空但库里确有内容"由此可区分。候选数由调用方提供
    （各自路径上已算过白名单，避免重复遍历）。

    ``degraded`` 信息由 :func:`hippocampus_mcp.application.tags.check_tag_keys`
    在软降级时提供——只有 schema 显式开了 ``filter_degrade`` 才会有；缺省不开，
    见 ``tagging.validate_filter_keys``。
    """
    out: dict[str, Any] = {
        "tags": dict(parsed_tags) if parsed_tags else None,
        "matched": matched,
    }
    if degraded:
        out["degraded"] = True
        out["degraded_from"] = degraded.get("degraded_from")
        out["dropped_keys"] = degraded.get("dropped_keys")
    return out
