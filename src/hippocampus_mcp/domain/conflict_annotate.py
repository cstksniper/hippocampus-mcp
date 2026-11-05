"""并存口径标注（读侧）：结果集内「疑似同一事实的不同口径」打标。

从 ``brain/core.py`` 的 ``_annotate_conflicts`` 提取。

**只做标注**：给配对的条目加 ``conflict_group``（同组同号）/
``conflict_with``（对方 id 列表）/ ``newer``（组内时间最新的那条）。
分数、顺序、条数一律不动——旧口径可能是被引用的历史事实，也可能才是对
的那条；哪条为准由权威来源（docs）决定，检索器不替用户裁定。

落点理由：这是**纯派生加工**——输入是结果集（list[dict]）与两个纯函数
能力（分词、判据），输出是就地打标 + 配对数。零 I/O、零模型、零配置读数，
符合 domain「只准依赖 common」的矩阵（``numpy`` 是第三方数值库，
``domain/topic.py`` 已在用，不属于包内依赖）。

🔴 判据**不在这里**：``is_restatement`` 由调用方注入（brain 侧是
``HippoBrain._is_restatement``，它委托 ``domain.policies.is_restatement``）。
写侧（``store/supersede.py``）用同一个回调，两边共用一份判据——若各写
一套阈值，迟早出现「写时判为同一事实、读时判为不同事实」的自相矛盾。
本模块不 import 它，是为了保住测试对 ``brain._is_restatement`` 的
monkeypatch 面。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..common.constants import CONFLICT_ANNOTATE_MAX

__all__ = ["annotate_conflicts"]


def annotate_conflicts(
    items: list[dict[str, Any]],
    tokens_of: Callable[[str], set[str]],
    is_restatement: Callable[..., bool],
    max_items: int = CONFLICT_ANNOTATE_MAX,
) -> int:
    """标注 ``items`` 内的并存口径，返回配对数（就地改 ``items``）。

    Args:
        items: 检索结果条目。每条需有 ``memory_id`` / ``content``；
            ``_emb``（当前模型的向量）缺失时该对不参与判定——BM25 单路
            命中的 ``similarity`` 恒为 0.0，不能拿来当相似度用。
        tokens_of: 文本 → 实词集合（预先算好，不在 O(n²) 里重复分词）。
        is_restatement: 同一事实判据，签名见
            ``domain.policies.is_restatement``。
        max_items: O(n²) 保护上限。超出直接跳过——**只影响标注，
            不影响检索结果本身**。

    Returns:
        配成的对数。

    预存行为（与原实现逐条对齐，不要「顺手修好」）：

    * 条目数 ``<2`` 或 ``>max_items`` ⇒ 直接返回 0，一条都不打标；
    * 已判定过的对（``b`` 已在 ``a["conflict_with"]`` 里）跳过，但
      **并查集仍然连通**（A~B、B~C 则三者同族），朴素「同组就跳过」
      会把第三条漏掉或把组号分裂；
    * 组号只在**连通分量 ≥2 条**时才分配，且从 1 开始递增；
    * ``newer`` 只在组内时间戳**不全相等**时才打，且只打在第一条
      时间最大的上（``break``）——时间戳全同则整组都不打。
    """
    if len(items) < 2 or len(items) > max_items:
        return 0
    toks = {it["memory_id"]: tokens_of(it["content"]) for it in items}
    by_id = {it["memory_id"]: it for it in items}
    # 并查集：冲突关系是**传递**的（A~B、B~C 则 A/B/C 同族），
    # 用"同组就跳过"这类朴素去重会把第三条漏掉或把组号分裂成两个。
    parent: dict[str, str] = {mid: mid for mid in by_id}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # 路径压缩
            parent[x], x = root, parent[x]
        return root

    pairs = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if b["memory_id"] in (a.get("conflict_with") or []):
                continue               # 已判定过这一对
            va, vb = a.get("_emb"), b.get("_emb")
            if va is None or vb is None:
                # 缺当前模型向量（BM25 单路命中）→ 无法判定，跳过。
                # 不能用"分数"充当相似度：BM25-only 的 similarity 恒为 0.0。
                continue
            denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-12
            sim = float(va @ vb / denom)
            if not is_restatement(
                sim, a["content"], b["content"], a.get("entity_ids"),
                b.get("entity_ids"), toks[a["memory_id"]], toks[b["memory_id"]],
            ):
                continue
            ra, rb = find(a["memory_id"]), find(b["memory_id"])
            if ra != rb:
                parent[rb] = ra
            for x, other in ((a, b["memory_id"]), (b, a["memory_id"])):
                ids = x.setdefault("conflict_with", [])
                if other not in ids:
                    ids.append(other)
            pairs += 1

    # 收尾：按连通分量编号，并给每条同族里**时间最新**的那条打 newer——
    # "取错了值"的典型形态就是取了旧口径。
    groups: dict[str, list[dict[str, Any]]] = {}
    for mid in by_id:
        groups.setdefault(find(mid), []).append(by_id[mid])
    gid = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        gid += 1
        for m in members:
            m["conflict_group"] = gid
        stamps = [m.get("timestamp") or 0.0 for m in members]
        if len(set(stamps)) > 1:
            for m in members:
                if (m.get("timestamp") or 0.0) == max(stamps):
                    m["newer"] = True
                    break
    return pairs
