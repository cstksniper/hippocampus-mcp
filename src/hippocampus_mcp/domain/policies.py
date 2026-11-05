# -*- coding: utf-8 -*-
"""M2：判定策略纯函数（不碰 I/O、不碰实例状态的「怎么判」）

从 brain 层（pure.py / core.py）逐字搬来。搬动**零行为变化**的保障：
1. 本模块函数体逐字复制，只把 ``cls.`` 前缀改成模块级直调；
2. ``tests/golden/policies_golden.json`` 是搬迁前的现状行为快照
   （44 例，含阈值边界），``tests/test_policies_golden.py`` 逐例断言；
3. 原位置保留同名委托（pure.py 的 staticmethod/classmethod、core.py 的
   ``_is_restatement`` wrapper），调用点一行不动。

为什么值得搬：这些判定核混在 6299 行的 core.py / Mixin 里，策略逻辑
（阈值、词表、门条件）无法单测——要测就得构造整个 HippoBrain + 嵌入器。
搬到 domain 后它们与基础设施零依赖，golden 测试毫秒级跑完。

依赖方向：本模块只 import common 常量（L0），符合目标架构 §3 分层。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..common.constants import (
    _GUESS_IMPORTANCE_CAP,
    _ORIGIN_RANK,
    _STOPWORDS,
    ENTITY_ORIGIN_MANUAL,
    SUPERSEDE_OVERLAP,
    SUPERSEDE_SIM,
)


def guess_importance(text: str) -> float:
    """启发式重要性：强情绪/承诺/否定变更 → 高分。

    🔴 **更正（读代码得出，不是推测）**：``min(score, 0.95)``
    是**死代码** —— 下面各分支只能给出 0.5 / 0.6 / 0.7 / 0.75 四个值，
    上限永远不生效。⇒ **自动判定给不出 >0.75 的重要性**，库里那些 0.95
    只能来自 ``report_feedback(useful=True)`` 的 +0.05 累加（9 次）或
    调用方显式传入。此前把「孤句拿到 0.95」归因为本函数是**错的**。

    保留上限是为了将来加更高档位时不越过 :data:`_GUESS_IMPORTANCE_CAP`
    ——0.95 以上是"用户反复确认过"的区间，不该由词面规则给出去。

    ⚠️ 另一个已知后果（**是否调整留待后续决定**）：0.7 ≥
    ``PIN_IMPORTANCE``(0.7) ⇒ 命中「现在 / 不再 / changed / ！」的记忆会
    **自动 pin**，而 pin 只进不退 ⇒ 长库里 pin 会单调堆积（实测某库 78%）。
    """
    t = text.lower()
    score = 0.5
    if any(m in t for m in ("always", "never", "must", "important", "remember", "记住", "永远", "绝不", "重要")):
        score = 0.75
    if any(m in t for m in ("love", "hate", "崩溃", "狂喜", "amazing", "terrible", "！", "!")):
        score = max(score, 0.7)
    if any(m in t for m in ("no longer", "used to", "changed", "现在", "不再")):
        score = max(score, 0.7)
    if len(text) > 200:
        score = max(score, 0.6)
    return min(score, _GUESS_IMPORTANCE_CAP)


def tokens_of(s: str) -> set[str]:
    """中文 bigram + 英文词（去停用词）的实词集合。"""
    zh = set(re.findall(r"[\u4e00-\u9fa5]{2}", s))
    en = {w for w in re.findall(r"[A-Za-z]{3,}", s.lower()) if w not in _STOPWORDS}
    return zh | en


def overlap_of_sets(ta: set[str], tb: set[str]) -> float:
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def is_restatement(
    sim: float,
    text_a: str,
    text_b: str,
    ents_a: Any = None,
    ents_b: Any = None,
    toks_a: set[str] | None = None,
    toks_b: set[str] | None = None,
) -> bool:
    """两条记忆是否"在讲同一件事"（疑似同一事实的不同口径）。

    **判据只此一处**：写入侧（``_detect_supersede`` 判"谁取代谁"）与
    读取侧（``_annotate_conflicts`` 判"谁与谁并存冲突"）共用它。

    为什么要共用一个判据：写侧受限于时机——``supersede`` 关着（文档导入
    的默认）、老数据早于该机制、或当时没配对上的，都会留下**并存的新旧
    口径**；读侧拿到同一判据就能把这些补标出来。若两边各写一套阈值，
    迟早出现"写时判为同一事实、读时判为不同事实"的自相矛盾。

    条件：相似度达标（``sim >= SUPERSEDE_SIM``）**且**（共享实体
    **或** 实词重叠达标（``>= SUPERSEDE_OVERLAP``））。单一条件不够——
    中文 bigram 口径下同文档两段话很容易越过重叠门槛，只看重叠会把
    沿革叙述大面积误判（覆盖报告·高 4 实测 28 对误杀）。
    """
    if sim < SUPERSEDE_SIM:
        return False
    if set(ents_a or []) & set(ents_b or []):
        return True
    ta = toks_a if toks_a is not None else tokens_of(text_a)
    tb = toks_b if toks_b is not None else tokens_of(text_b)
    return overlap_of_sets(ta, tb) >= SUPERSEDE_OVERLAP


def merge_entity_origins(
    explicit: list[str] | None,
    extracted: dict[str, str],
    *,
    canonical: Callable[[str], str],
) -> tuple[list[str], dict[str, str]]:
    """显式实体（manual 起源）与抽取实体（各自起源）合并为 names/origins。

    从 ``brain.core.ingest`` 内联抽取。三条保留语义：

    * 显式传入的实体**同样过别名归一**（``canonical``，词库
      ``别名 => 主名``）——不过的话宿主按别名传会凭空建出别名实体条目，
      正是"同一实体被拆成多份档案"的源头；
    * 同一实体（小写名）多次命中时**只升级不降级**：按
      ``_ORIGIN_RANK`` 取更高可信度来源，先启发式入库的实体日后被
      词库/宿主命中也能"转正"；
    * names 保留**出现顺序且不去重**（``_link_entities`` 内部按
      ``dict.fromkeys`` 去重），与本函数抽取前一致。

    Args:
        explicit: 宿主显式传入的 ``entities=[...]``（可为 None）。
        extracted: ``extract_entities_with_origin(text)`` 的产物
            ``{实体名: origin}``。
        canonical: 别名归一函数（brain 的 ``canonical_entity``）。
    """
    names: list[str] = []
    origins: dict[str, str] = {}
    for e in (explicit or []):
        e = (e or "").strip()
        if not e:
            continue
        # 显式传入的实体**同样要过别名归一**：不过的话，宿主按别名传
        # （entities=["子安"]）会凭空建出一个别名实体条目，正是"同一实体
        # 被拆成多份档案"这个问题的源头。归一只在抽取侧做是不够的。
        e = canonical(e)
        names.append(e)
        k = e.lower()
        prev = origins.get(k)
        if prev is None or _ORIGIN_RANK[ENTITY_ORIGIN_MANUAL] > _ORIGIN_RANK[prev]:
            origins[k] = ENTITY_ORIGIN_MANUAL
    for name, origin in extracted.items():
        names.append(name)
        k = name.lower()
        prev = origins.get(k)
        if prev is None or _ORIGIN_RANK[origin] > _ORIGIN_RANK[prev]:
            origins[k] = origin
    return names, origins
