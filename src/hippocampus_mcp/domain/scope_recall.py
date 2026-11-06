# -*- coding: utf-8 -*-
"""跨库检索 scope：值校验与双库合并的纯函数。

落点：I/O 编排在
``application/recall_service.py``（经组合根注入的用户库 brain 跑第二流），
本模块只做**纯数据**判据（domain 只准依赖 common）：

* ``validate_scope``——scope 值校验（非法值显式报错；错误文案在此
  单点维护，调用方拿去装裸错误体）；
* ``merge_scope_hits``——scope=all 的硬分段合并（项目库全前、
  用户级全后，top_k 是**总数**；跨库**精确内容**匹配才标 ``duplicate_of``
  ——两库独立嵌入模型分数不可比，不猜相似）。

边界注记：scope 只有三值枚举，不泛化为任意多库——
``libraries="A,B"`` 式跨任务检索是后续增量（任务目录落地后
以其为寻址合法值来源）。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SCOPE_ALL", "SCOPE_PROJECT", "SCOPE_USER", "VALID_SCOPES",
    "MAX_LIBRARY_SPEC",
    "merge_scope_hits", "merge_multi_library", "parse_library_spec",
    "rrf_merge_streams", "validate_scope",
]

SCOPE_PROJECT = "project"
SCOPE_USER = "user"
SCOPE_ALL = "all"

VALID_SCOPES = (SCOPE_PROJECT, SCOPE_USER, SCOPE_ALL)

#: ``libraries=`` 单次最多扇出几个外围库（B1 成本上限：每库一份 brain，
#: 冷载 25–33s / 边际内存；扇出过多 = 互相换出抖动）。超出显式报错，
#: 不静默截断——调用方要打几个库是明确意图，砍掉哪半个都不是。
MAX_LIBRARY_SPEC = 4

#: RRF 融合常数（标准值）。rank 从 1 计：score = Σ 1/(k + rank)。
RRF_K = 60


def validate_scope(scope: str, *, allow_all: bool = True) -> str | None:
    """校验 scope 值。合法返回 ``None``；非法返回**错误文案**（不抛）。

    Args:
        scope: 调用方给的 scope（recall/get_memory_context 的新参数）。
        allow_all: ``False`` = 本入口不支持 all（get_memory_context 的
            5 槽位装配没有"项目排前用户殿后"的自然落点，收窄；
            错误文案单独说清楚）。

    为什么不抛异常：错误体形态由调用方定（承 recall 的 filter_tags
    解析错误先例——裸 dict ``{"error": ..., "input": ...}``，不带任何
    自曝键，"错误体一律不带"）。
    """
    s = str(scope or "").strip()
    if s in VALID_SCOPES:
        if s == SCOPE_ALL and not allow_all:
            return ("scope=all is only supported by recall; get_memory_context is a 5-slot "
                    "assembly structure with no place for a project-first, user-second ordering. "
                    "For single-library assembly across libraries, call scope=project / scope=user separately.")
        return None
    return (f"scope accepts only {'/'.join(VALID_SCOPES)}, got {scope!r}. "
            "project=the current library (default); user=the user-level library; all=project library first, "
            "user library second (recall only).")


def merge_scope_hits(project_hits: list[dict[str, Any]],
                     user_hits: list[dict[str, Any]],
                     *, top_k: int) -> list[dict[str, Any]]:
    """scope=all 的硬分段合并。

    规则：
    1. **硬分段**：项目库结果全前、用户级全后（各自保持原序——分数
       跨库不可比，不做任何融合排序）；
    2. **top_k 是总数**：两库竞争名额，项目段先占（"我要 5 条"的调用方
       语义不变）；
    3. **duplicate_of 只标注不丢弃**（承"只标注不裁决"先例）：
       用户级条目与**已入列的**项目条目 content **精确相等**时，标
       ``duplicate_of = <项目条目 memory_id>``。两个刻意的窄口径——
       只比精确内容（不猜相似度），只对可见的重复标（项目孪生被 cap
       截掉时，用户级那条单独在场不算"双现"）。

    Args:
        project_hits: 项目流的结果条目（已是最终 dict 形态）。
        user_hits: 用户库流的结果条目。
        top_k: 总数 cap（``<= 0`` 返回空，与 recall 的 top_k 语义一致）。
    """
    if top_k <= 0:
        return []
    kept_project = list(project_hits)[:top_k]
    room = max(int(top_k) - len(kept_project), 0)
    # 可见重复的判据表：入列的项目条目 content -> memory_id（首个）
    project_ids = {}
    for h in kept_project:
        c = h.get("content")
        if c is not None and c not in project_ids:
            project_ids[c] = h.get("memory_id")
    out: list[dict[str, Any]] = [dict(h) for h in kept_project]
    for h in user_hits[:room]:
        entry = dict(h)
        dup = project_ids.get(entry.get("content"))
        if dup is not None:
            entry["duplicate_of"] = dup
        out.append(entry)
    return out


def parse_library_spec(spec: str) -> tuple[list[str], str | None]:
    """解析 ``libraries="甲,乙"``（B1）。返回 ``(names, error_text)``。

    * 空串 / 纯空白 = **未指定**（``[], None``），不是错误；
    * 逗号分段（**全角「，」同权**——中文输入法的常态，不为难调用方）、
      去空白、**空段忽略**、重复值**去重保序**（调用方笔误不是错误，
      但去重让"打几个库"的账可数）；
    * 超出 :data:`MAX_LIBRARY_SPEC` → 错误文案（不静默截断）。

    值本身**在这里不校验合法性**（有没有这个库是寻址层的事，三态解析
    失败自会报错）；本函数只管语法。
    """
    raw = str(spec or "").replace("，", ",")
    names: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p and p not in names:
            names.append(p)
    if not names:
        return [], None
    if len(names) > MAX_LIBRARY_SPEC:
        return [], (f"libraries= accepts at most {MAX_LIBRARY_SPEC} libraries per call "
                    f"(got {len(names)}: {names!r}). "
                    "Fan-out cost grows linearly with library count (one brain per library); split multi-library queries into separate calls.")
    return names, None


def rrf_merge_streams(
    streams: list[tuple[str, list[dict[str, Any]]]],
    *, top_k: int, k: int = RRF_K,
) -> list[dict[str, Any]]:
    """外围库扇出的 RRF 排名融合（拍板②a：只在外围集合**内部**融合）。

    ``score(d) = Σ 1/(k + rank_i(d))``（rank 从 1 计，各流独立名次）——
    排名融合与各库的分数体系无关（BM25 IDF 库内统计、向量余弦分分布
    随语料漂移，跨库比分数没有意义，比名次才有）。

    每条带 ``library``（**首现流**的库名；同 id 跨流重复时首现赢）与
    ``rrf_score``。同分 tie-break 按 ``memory_id`` 升序（确定性）。
    """
    if top_k <= 0:
        return []
    scores: dict[str, float] = {}
    first_seen: dict[str, dict[str, Any]] = {}
    lib_by_id: dict[str, str] = {}
    for lib_name, hits in streams:
        for rank, h in enumerate(hits, start=1):
            mid = str(h.get("memory_id"))
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
            if mid not in first_seen:
                first_seen[mid] = h
                lib_by_id[mid] = lib_name
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[dict[str, Any]] = []
    for mid, score in ranked[:top_k]:
        entry = dict(first_seen[mid])
        entry.setdefault("library", lib_by_id[mid])
        # 不舍入：rrf_score 是排名依据，截位会让同分判定漂移
        entry["rrf_score"] = score
        out.append(entry)
    return out


def merge_multi_library(
    home_hits: list[dict[str, Any]],
    periphery: list[tuple[str, list[dict[str, Any]]]],
    *, top_k: int, home_name: str,
) -> list[dict[str, Any]]:
    """层次化合并（拍板②a）：**家库段全前（原序），外围 RRF 竞争剩余名额**。

    与 :func:`merge_scope_hits` 同一硬分段精神——家库/任务库的既有排序
    语义与测试逐位保留；RRF 只用于外围集合内部，绝不跨段重排。

    Returns:
        合并列表（每条带 ``library``：家库段 = ``home_name``，外围 = 各流名）。
        ``top_k`` 是总额（家库段先占，外围竞争 ``top_k - len(home段)``）。
    """
    if top_k <= 0:
        return []
    kept_home = [dict(h, library=home_name) for h in list(home_hits)[:top_k]]
    room = max(int(top_k) - len(kept_home), 0)
    if room == 0 or not periphery:
        return kept_home
    # 外围流各自 cap 到剩余名额再融合：每流多带的条目在 RRF 里也排不上
    # （名额就这么多），提前 cap 省一次全量打分。
    capped = [(name, hits[:room]) for name, hits in periphery]
    return kept_home + rrf_merge_streams(capped, top_k=room)
