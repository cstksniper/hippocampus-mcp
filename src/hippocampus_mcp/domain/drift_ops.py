"""杂项纯逻辑族：实体补挂编排 + 停用词三层展开。

本模块收的是没有更合适归属的两件纯逻辑（``source_drift`` /
``forget_by_query`` 在基线上已是既有薄壳，分别由
``store.source_sig.drift_audit`` 与 ``domain/forget_flow`` 承接，
无可抽内联逻辑），实抽 ``relink_entities`` 与 ``_stop_grams`` 两件。

零项目内 import（架构约束：domain → 仅 common）。两件的外部能力全部由壳注入：

  * ``relink_entities``：``extract_fn`` / ``link_fn``（brain 的实体抽取
    与挂链，惰性注入——不开关、逐记录现调）、``merge_fn``（store 层的
    id 列表并集，domain 不许 import store）、``update_fn``（store 写回）、
    ``archived_state``（state 字符串值，未 patch 常量按值传）；
  * ``stop_grams``：``lexicon_path``（词库路径值）、``load_lexicon``
    （词文件读取回调）、``builtin_stop``（内置停用词值）、``exclude``
    （本次排除项）。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * relink：``list(records)`` 先物化再遍历（循环内会写回 store，防迭代
    器失效）；archived 跳过；``only_ids`` 假值视作不限定；空 origins
    跳过；``set(merged) != set(rec.entity_ids or [])`` 才写回——计数
    分别是「动了几条」与「新增了几个链接」（并集减旧集）；
  * stop_grams：三层只增不覆盖（内置 ∪ 词库文件 ∪ exclude），词与文件
    词统一 ``lower()``；子串展开是**全前缀**（长度 1..len 全部切片），
    登记一条完整短语即压掉它的全部切片；计数分母带各层条数。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable


def relink_entities(
    records: Iterable[Any],
    *,
    extract_fn: Callable[[str], dict],
    link_fn: Callable[[list, str, float, dict], list],
    merge_fn: Callable[[Any, Any], list],
    update_fn: Callable[..., Any],
    archived_state: str,
    only_ids: list[str] | None = None,
) -> dict[str, int]:
    """对已有记忆重跑实体抽取并**补挂**（词库扩充的必要配套）。

    词库新增词后，存量记忆的 ``entity_ids`` 不会自动变化。不重挂就是
    "加了词却查不到"，词库扩充闭环断在最后一环。补挂用并集，不覆盖
    已有的实体关联（与 ingest 的合并语义一致）。
    """
    only = set(only_ids) if only_ids else None
    touched = 0
    new_links = 0
    for rec in list(records):
        if rec.state == archived_state:
            continue
        if only is not None and rec.id not in only:
            continue
        origins = extract_fn(rec.content)
        if not origins:
            continue
        eids = link_fn(list(origins.keys()), rec.id, rec.timestamp, origins)
        merged = merge_fn(rec.entity_ids, eids)
        if set(merged) != set(rec.entity_ids or []):
            update_fn(rec.id, entity_ids=merged)
            touched += 1
            new_links += len(set(merged) - set(rec.entity_ids or []))
    return {"records_touched": touched, "new_links": new_links}


def stop_grams(
    lexicon_path: str | None,
    load_lexicon: Callable[[str], Iterable[str]],
    *,
    builtin_stop: Iterable[str],
    exclude: list[str] | None = None,
) -> tuple[set[str], set[str], dict[str, int]]:
    """停用词三层 → ``(原词集合, 展开后的子串集合, 各层条数)``。

    三层**只增不覆盖**：① 内置通用行话 ∪ ② 项目停用词文件
    ``<词库同目录>/entity_lexicon_stop.txt`` ∪ ③ 本次 ``exclude``。

    "展开成子串集合"是为了让**登记一条完整短语即压掉它的全部切片**
    （写"用户定案"同时挡掉"户定案"），不必手工枚举前缀碎片。
    """
    stop: set[str] = {t.lower() for t in builtin_stop}
    file_stop: set[str] = set()
    if lexicon_path:
        stop_path = os.path.join(
            os.path.dirname(lexicon_path), "entity_lexicon_stop.txt"
        )
        if os.path.exists(stop_path):
            file_stop = {t.lower() for t in load_lexicon(stop_path)}
            stop |= file_stop
    ex = {t.lower() for t in (exclude or [])}
    stop |= ex
    grams: set[str] = set()
    for s in stop:
        for m in range(1, len(s) + 1):
            for i in range(len(s) - m + 1):
                grams.add(s[i : i + m])
    return stop, grams, {
        "builtin": len(builtin_stop),
        "file": len(file_stop),
        "extra": len(ex),
    }
