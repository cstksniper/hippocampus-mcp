# -*- coding: utf-8 -*-
"""开场包的「知识地图」槽（从 brain/core.py 提取）。

几条一行的知识点（标题 + 规模 + 一句话）。

🔴 **刻意不在这里自动梳理**。开场包在热路径上（每会话一次），而首次梳理是
全库聚类——真实大库实测 2-3 秒。把这几秒塞进
每次开场，换来的只是「万一宿主想看地图」。所以这里只读盘上已有的层，空就
明说「调 knowledge_map 会梳理一次」，把一次性成本留给用户显式调用。

一句话优先用**宿主归纳**（那才是上层知识），没有才退回抽取式代表句。

为什么放 domain/：纯派生投影。🔴 但主题口径（``ordered`` / 归档状态 /
``summary_state``）**走注入不 import** —— arch G1 的矩阵是
``domain → 仅 common``，同层互引都不行（因此吃过亏）。同理
``store.count()`` 也由调用方注入（只有"有料但没梳理过"这一条分支需要它，
注入成 callable 才能保住惰性）。

端口纪律：``limit`` / ``clip`` / 主题层开关全部显式参数，不读 Config。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping


def knowledge_brief(
    knowledge: Mapping[str, Any],
    limit: int = 8,
    *,
    topic_layer_enabled: bool,
    ordered_topics: Callable[[Any], list[dict[str, Any]]],
    archived_status: str,
    summary_state: Callable[[dict[str, Any], Any], dict[str, Any]],
    store_count: Callable[[], int],
    clip: int,
) -> list[str]:
    """开场包的知识地图槽：几条一行的知识点。"""
    if limit <= 0 or not topic_layer_enabled:
        return []
    if not knowledge:
        # 有料但从没梳理过 ≠ 没有知识点：把出路讲清楚，别让宿主以为库是空的
        if store_count():
            return ["（还没梳理过知识点；调 knowledge_map 会梳理一次）"]
        return []
    out: list[str] = []
    for t in ordered_topics(knowledge):
        if str(t.get("status") or "") == archived_status:
            continue
        title = str(t.get("title") or "(未命名)")
        size = len(t.get("member_ids") or [])
        # summary 这一格装的是**当前生效**的那句话：写了 host 归纳就是归纳，
        # 没写就是 refresh 填的抽取式代表句（两者互斥，见 topic.refresh）。
        line = str(t.get("summary") or "").strip() or "（尚无摘要）"
        mark = ""
        st = summary_state(t, t.get("member_ids"))
        if st.get("is_host"):
            # ⚠️ = 写过归纳但依据成员后来变了（最该拿回宿主重写的一批）
            mark = "" if not st.get("stale") else "⚠️"
        head = "%s（%d条）%s" % (title, size, mark)
        # 只截句子、**绝不截标题和规模**：真库实测整行硬截会把两者一起切掉，
        # 剩下半截标题，宿主认不出这是哪个知识点（比少看几个字糟得多）。
        budget = clip - len(head)
        if budget < 8:
            head = head[:clip - 1] + "…"
            body = ""
        else:
            line = " ".join(line.split())
            body = line if len(line) <= budget else line[:budget - 1] + "…"
        out.append(head + body)
        if len(out) >= limit:
            break
    return out
