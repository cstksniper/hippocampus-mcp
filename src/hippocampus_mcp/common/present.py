# -*- coding: utf-8 -*-
"""预渲染文本块：结构化 dict → 文本槽（从 brain/core.py 提取的纯函数层）。

这五个函数是**双轨制**的文本轨（设计文档 §3.3）：session_start 的
结构化 dict 供程序消费与未来人类 UI 渲染真信息图，这里的文本槽
供直接拼 prompt。两边同源同口径，改渲染规则只改这一处。

为什么放 ``common/`` 而不是 ``application/``：现调用方是 brain 的
``_SessionMixin``（session.py），而依赖方向是「用例逻辑只从 brain
单向流向 application，不回头」——brain 不许 import application。
放 common 两层都能用；等 session.py 的组装体搬进
``BootstrapService`` 后，这一层归属不变。

全部是纯函数：输入 dict、输出 str，零 I/O、零 brain 状态。
"""
from __future__ import annotations

import time
from typing import Any

from .constants import TASK_PROFILE_CLIP, format_age, sparkline


def render_overview_block(ov: dict[str, Any]) -> str:
    """全貌卡的预渲染文本（≤180 token 目标；空库给占位不报错）。"""
    if not ov.get("memories"):
        return "（空库：还没有任何记忆）"
    lines = []
    head = "【库】%s · %s 条记忆" % (ov["name"], f"{ov['memories']:,}")
    if ov.get("knowledge_topics"):
        head += " · %d 知识点" % ov["knowledge_topics"]
    if ov.get("entities"):
        head += " · %d 实体" % ov["entities"]
    lines.append(head)
    if ov.get("first_write_ts") and ov.get("last_write_ts"):
        first = time.strftime("%Y-%m", time.localtime(ov["first_write_ts"]))
        last = time.strftime("%Y-%m", time.localtime(ov["last_write_ts"]))
        months = max(1, round((ov["last_write_ts"] - ov["first_write_ts"]) / 2592000))
        span = "【跨度】%s 至 %s（%d 个月）" % (first, last, months)
        if ov.get("recent_ingest"):
            span += " · 近 %d 天 +%d 条" % (ov["recent_days"], ov["recent_ingest"])
        lines.append(span)
    struct = "【结构】"
    if ov.get("tag_coverage") is not None:
        struct += "标签覆盖 %.1f%%" % (ov["tag_coverage"] * 100)
        if ov.get("tag_top_keys"):
            struct += "（主键：%s）" % "/".join(ov["tag_top_keys"])
    if ov.get("knowledge_topics"):
        struct += " · 知识点归纳 %d/%d" % (ov.get("knowledge_host") or 0,
                                         ov["knowledge_topics"])
        if ov.get("knowledge_stale"):
            struct += "（%d 个待重写 ⚠️）" % ov["knowledge_stale"]
    if struct != "【结构】":
        lines.append(struct)
    if ov.get("top_entities"):
        tops = " ".join("%s(%d)" % (e["name"], e["memories"])
                        for e in ov["top_entities"])
        lines.append("【实体】Top：%s" % tops)
    return "\n".join(lines)


def render_status_block(st: dict[str, Any]) -> str:
    """状态卡的预渲染文本（≤150 token 目标）。矛盾关闭时该段整体省略。"""
    lines = []
    pend = "【未决】"
    parts = []
    if st.get("open_topics"):
        parts.append("%d 个开放话题" % st["open_topics"])
    if st.get("open_intents"):
        parts.append("%d 条待办" % st["open_intents"])
    parts.append("巩固待触发" if st.get("pending_consolidate") else "巩固无需触发")
    lines.append(pend + " · ".join(parts))
    health = "【健康】"
    hparts = []
    hparts.append("L0 悬空 %s" % ("有" if st.get("l0_dangling") else 0))
    if st.get("contradictions") is not None:
        mark = "≥%d" % st["contradictions"] if st.get("contradictions_capped") \
            else str(st["contradictions"])
        hparts.append("矛盾 %s 处" % mark)
    if st.get("recent_miss"):
        hparts.append("近 %d 天未命中 %d 次（缺口信号）"
                      % (st["recent_miss_days"], st["recent_miss"]))
    lines.append(health + " · ".join(hparts))
    n_sess = st.get("active_sessions") or 0
    lines.append("【协作】另有 %d 个会话在作业" % max(0, n_sess - 1)
                 if n_sess > 1 else "【协作】无其他会话在作业")
    spark = sparkline(st.get("spark_series") or [])
    if spark:
        lines.append("【活动】近 %d 天写入 %s" % (st["spark_days"], spark))
    else:
        recent = sum(st.get("spark_series") or [])
        lines.append("【活动】近 %d 天写入 %d 条" % (st["spark_days"], recent))
    # 文件建档：项目文件与 hippo 的同步态。
    # 没启用建档（无侧车）时整行省略——opt-in 的东西不占非用户的 token。
    fr = st.get("file_registry")
    if fr:
        line = "【建档】%s 个文件" % f"{fr['total']:,}"
        if fr["stale"] or fr["missing"]:
            bits = []
            if fr["stale"]:
                bits.append("%d 个磁盘更新未同步（最旧 %d 天前）"
                            % (fr["stale"], fr["oldest_stale_days"]))
            if fr["missing"]:
                bits.append("%d 个消失" % fr["missing"])
            line += " · " + " · ".join(bits)
        else:
            line += " · 全部同步 ✓"
        if not fr["scan_complete"]:
            line += " · 扫描不完整"
        lines.append(line)
    return "\n".join(lines)


def render_map_index_block(mi: dict[str, Any] | None) -> str:
    """索引行文本（§5 样张）：合约(42条·3天前) · 基因组(18条·⚠️12天)。

    漂移标记：``合约(42条·3天前·漂移2)``——真相源变过的条目数
    （自动 verify 的产出）。只进必退同纪律：没有漂移就不占字符。
    冲突标记：``合约(42条·3天前·冲突1)``——
    legacy 记忆与 registry 侧车同键不同状态（双源竞争）。
    漏登标记：``数据集(12条·2天前·漏登3)``——目录里有 3 个键
    地图没登记（对账的另一路方向）。
    """
    if not mi or not mi.get("maps"):
        return ""
    parts = []
    for m in mi["maps"]:
        flag = "⚠️" if m["stale"] else ""
        seg = "%s(%d条·%s%s)" % (m["name"], m["entries"],
                                 flag, format_age(m["age_days"]))
        if m.get("drifted"):
            seg += "·漂移%d" % m["drifted"]
        if m.get("failed"):
            # 自动 verify 说不清的条目（动态值/坏格式）。既不是
            # clean 也不是 drift——单独用「待核」报名，不许静默吞。
            seg += "·待核%d" % m["failed"]
        if m.get("conflicts"):
            seg += "·冲突%d" % m["conflicts"]
        if m.get("mentioned"):
            # 提及信号：更新之后有记忆提到「另一个状态」。措辞用「疑变」
            # 而不是「漂移」——它不是真相源比对的结果，只是有人说了
            # 不一样的话（证据在条目 mention.quote 里）。
            seg += "·疑变%d" % m["mentioned"]
        if m.get("unmatched"):
            # 目录里有、地图没登记的键（漏登记清单的长度）。
            # 它是唯一一个「漂移」表达不了的方向——漂移只说真相变了，
            # 说不出「世界多了东西没人告诉地图」。
            seg += "·漏登%d" % m["unmatched"]
        parts.append(seg)
    line = "【地图】" + " · ".join(parts)
    if mi.get("hidden"):
        line += " · 等 %d 张" % mi["hidden"]
    return line


def render_map_detail_block(md: dict[str, Any] | None) -> str:
    """L1 图全文块：每图一行 desc + 至多 20 行条目（key=status）。"""
    if not md or not md.get("maps"):
        return ""
    lines = []
    for m in md["maps"]:
        head = "【地图·%s】%s" % (m["name"], m["desc"])
        if m.get("drifted"):
            head += "（⚠️%d 条真相源已变）" % m["drifted"]
        lines.append(head)
        lines.append(" · ".join(m["rows"]))
        if m.get("hidden_rows"):
            lines.append("……等 %d 条（超 %d 行预算，点名单条走 "
                         "map_get）" % (m["hidden_rows"], md["row_cap"]))
    if md.get("hidden"):
        lines.append("……等 %d 张相关图未进包" % md["hidden"])
    return "\n".join(lines)


def render_task_profile_block(tp: dict[str, Any] | None) -> str:
    """问题概况卡的预渲染文本（≤200 token 目标；全空返回空串）。

    只列非空行——空行对 agent 是纯噪声。数字（命中数/规模）绝不截断，
    只截标题（与知识地图槽同纪律：半截标题比少看几个字糟得多）。
    """
    if not tp:
        return ""
    lines: list[str] = []
    if tp.get("topics"):
        parts = []
        for t in tp["topics"]:
            # 只截标题、规模数字后缀完整保留（docstring 契约，不许整头
            # 截断——长标题时「（12345条·中9）」会被截成「（1…」）。
            # 后缀本身超预算时保数字、允许整行超 CLIP。
            suffix = "（%d条·中%d）" % (t["size"], t["hits"])
            title = t["title"]
            if len(title) + len(suffix) > TASK_PROFILE_CLIP:
                keep = max(TASK_PROFILE_CLIP - len(suffix) - 1, 0)
                title = title[:keep] + "…"
            parts.append(title + suffix)
        more = tp["topics_total"] - len(tp["topics"])
        lines.append("【知识点】" + " · ".join(parts)
                     + ("（等 %d 个）" % more if more > 0 else ""))
    if tp.get("entities"):
        parts = ["%s（中%d/共%d）" % (e["name"], e["hits"], e["mentions"])
                 for e in tp["entities"]]
        more = tp["entities_total"] - len(tp["entities"])
        lines.append("【实体】" + " · ".join(parts)
                     + ("（等 %d 个）" % more if more > 0 else ""))
    if tp.get("intents"):
        parts = []
        for it in tp["intents"]:
            text = " ".join(it["text"].split())
            if len(text) > TASK_PROFILE_CLIP:
                text = text[:TASK_PROFILE_CLIP - 1] + "…"
            if it.get("due"):
                text += "（%s）" % it["due"]
            parts.append(text)
        more = tp["intents_total"] - len(tp["intents"])
        lines.append("【待办】" + " · ".join(parts)
                     + ("（等 %d 条）" % more if more > 0 else ""))
    if tp.get("misses"):
        parts = ["「%s」%.2f" % (
            " ".join(m["query"].split())[:TASK_PROFILE_CLIP],
            m["top_similarity"]) for m in tp["misses"]]
        more = tp["misses_total"] - len(tp["misses"])
        lines.append("【缺口】近 %d 天未命中 %d 次相关查询：%s%s"
                     % (tp["miss_days"], tp["misses_total"], " · ".join(parts),
                        "（等 %d 次）" % more if more > 0 else ""))
    return "\n".join(lines)
