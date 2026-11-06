# -*- coding: utf-8 -*-
"""未命中留痕写流（从 core 抽取）。

**为什么值得搬**：两段的共同点是**删除必立碑**（删除账本纪律）——上限
裁剪与清空都是真删除，不立碑就会被别的进程手里的旧快照落盘并回
（"裁了又长回来"/"已清空是假的"）。这段纪律逻辑与 entry 构造（brain
态读取）分离后，壳只剩「构造 entry → 委托」一条直线。

**依赖方向**：本模块零 import——``miss_log`` 列表由壳直传（原地
append/del/clear，与原实现同对象语义）；上限 ``max_n`` 由壳从本命名
空间现取（``MISS_LOG_MAX``，patch 调用方即生效）；立碑经
``note_deletions`` 回调注入（brain 侧是 ``note_layer_deletion_items``，
删除账本写面留在 brain）。

**复刻的怪癖（动一个就是行为变化，parity 会抓）**：

1. 裁剪判据 ``if max_n and len(...) > max_n``——``max_n=0`` = 关闭
   上限（真值判定，与原 ``MISS_LOG_MAX`` 假值语义一致）；
2. 裁剪量 ``drop_n = len - max_n``，立碑的是**被裁掉的头部切片**
   ``miss_log[:drop_n]``（最老的先走），``del`` 与立碑同批；
3. 立碑后才删——回调先于 ``del`` 执行（顺序有语义：账本没写成
   就删，等于白删）；
4. 清空：空表**不调** ``note_deletions``（0 座碑），返回 0；
   非空先整批立碑再 ``clear``，返回条数；
5. 两段都**原样返回/无返回设计**：append 返回 entry 本体（调用方
   拿到的是刚写入的那条），clear 返回条数。
"""
from __future__ import annotations

from typing import Any, Callable


def append_miss_entry(
    *,
    miss_log: list[dict[str, Any]],
    entry: dict[str, Any],
    max_n: int,
    note_deletions: Callable[[str, list[dict[str, Any]]], Any],
    ledger_name: str,
) -> dict[str, Any]:
    """追加一条未命中留痕 + 上限裁剪（删除必立碑纪律）。

    Args:
        miss_log: 留痕列表（**原地修改**，与原实现同对象语义）。
        entry: 壳构造好的条目（ts/截断/归一都在壳，brain 态读取）。
        max_n: 留痕上限（壳从本命名空间现取 ``MISS_LOG_MAX``；
            **0 = 关闭上限**，真值判定）。
        note_deletions: 立碑回调 ``(ledger_name, items)``（删除账本）。
        ledger_name: 账本层名（``"brain_misslog.json"``）。

    Returns:
        刚写入的 entry 本体。
    """
    miss_log.append(entry)
    if max_n and len(miss_log) > max_n:
        # 这也是一次**真删除**，必须立碑。不立的话别的进程手里
        # 那份还没裁的旧快照一落盘，这些被裁掉的条目又被并回来，
        # 上限就成了摆设（"裁了又长回来"最容易被当成裁剪没生效）。
        drop_n = len(miss_log) - max_n
        note_deletions(ledger_name, miss_log[:drop_n])
        del miss_log[:drop_n]
    return entry


def clear_miss_log(
    *,
    miss_log: list[dict[str, Any]],
    note_deletions: Callable[[str, list[dict[str, Any]]], Any],
    ledger_name: str,
) -> int:
    """清空未命中留痕（先整批立碑再清）。

    此前直接 ``miss_log.clear()`` 只在"我是最后一个写的进程"时有效——
    别的进程手里那份旧快照一落盘，整批就被并回来，用户看到的"已清空"
    是假的。空表不立碑也不清（0 座碑、返回 0）。

    Returns:
        清掉几条。
    """
    n = len(miss_log)
    if n:
        note_deletions(ledger_name, miss_log)
        miss_log.clear()
    return n
