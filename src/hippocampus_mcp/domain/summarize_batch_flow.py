# -*- coding: utf-8 -*-
"""批量写回主题归纳（``action="summarize_batch"``）的域内实现。

从 ``HippoBrain._knowledge_summarize_batch`` 抽出：本模块
只含纯逻辑与编排，零 import——``topic`` 模块函数（apply_host_summary /
pending_summary_ids）、``_knowledge_items``、``_persist_knowledge``、
``_topic_defaults`` 全部由壳调用时现取后注入（domain 同层禁互引）。

行为怪癖（逐条钉在 tests/test_m5_batch31_summarize_batch_flow.py）：

  1. 双守卫有先后：``not items`` 先于 ``isinstance``——空 dict / 0 /
     False / 空元组都报「items 是空的」，只有**非空**的非 list 才报
     「items 必须是数组」；
  2. 逐条独立：一条校验不过只退那一条（记进 ``failed`` 带原因），
     同批其他条照常写——否则一次手抖写错一个 id，整批几十条归纳全丢；
  3. 存活全集**逐条现取**（``items_of(require_vec=False)``，且只对
     **通过主题查找**的条目取）：调用次数语义与原实现一致，防猴子
     补丁计数面；
  4. ``topic_id`` 经 ``str(... or "")`` + strip：0 / False / None 全
     归一成空串，走「没有这个主题：」（拼的是 strip 后的 tid）；
  5. ``alive`` = ``member_ids`` ∩ 存活全集，保 ``member_ids`` 序；
  6. ``apply_host_summary`` 失败时 topic 零改动（域函数自身保证），
     本模块不兜底、不重试；
  7. ``written`` 条目的 ``title`` 兜底 ``(未命名)``；``has_fras`` 读
     apply **之后**的 ``summary_fras``——旧 fras 不清 ⇒ 不传 fras
     重写也报 True；
  8. 落盘条件：只有 ``written`` 非空才调 ``persist``——全失败的批次
     不写盘；
  9. ``ok = not failed or bool(written)``：部分成功算 ok；
  10. ``pending_summary_ids`` 调**两次**（``pending_count`` 与 note
      各一次）；``defaults`` 由壳一次现取后注入；
  11. note 三段式，有失败时中段插「，%d 条被退回（见 failed）」；
  12. 回显十键固定序：ok, enabled, action, written, failed,
      failed_count, results, saved, pending_count, note——注意
      ``written`` 是**计数**、``results`` 才是列表。
"""
from __future__ import annotations

from typing import Any, Callable

ACTION = "summarize_batch"


def guard(items: Any) -> tuple[Any, dict[str, Any] | None]:
    """双守卫（怪癖 1）。返回 ``(items, None)`` 或 ``(None, 错误体)``。

    错误体四键固定序：ok=False / enabled=True / action / error。两条
    消息逐字保留——工具面测试对它们做子串断言。
    """
    if not items:
        return None, {"ok": False, "enabled": True, "action": ACTION,
                      "error": "items 是空的（需要 [{topic_id, summary, "
                               "evidence}]）"}
    if not isinstance(items, list):
        return None, {"ok": False, "enabled": True, "action": ACTION,
                      "error": "items 必须是数组"}
    return items, None


def apply_items(
    items: list[Any],
    *,
    knowledge: dict[str, Any],
    items_of: Callable[..., list[dict[str, Any]]],
    apply_summary: Callable[..., tuple[bool, str]],
    now: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐条独立校验与写回（怪癖 2-7）。返回 ``(written, failed)``。

    ``items_of`` 逐条现取（怪癖 3，固定收 ``require_vec=False``）；
    ``apply_summary`` 签名与 ``topic.apply_host_summary`` 一致（topic,
    text, evidence, member_ids, now=, fras=），校验不过返回
    ``(False, 原因)`` 且**一个字段都不改**。
    """
    written: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            failed.append({"topic_id": "", "error": "条目不是对象"})
            continue
        tid = str(raw.get("topic_id") or "").strip()
        t = knowledge.get(tid)
        if t is None:
            failed.append({"topic_id": tid, "error": "没有这个主题：" + tid})
            continue
        info = {str(it["id"]): it for it in items_of(require_vec=False)}
        alive = [m for m in (t.get("member_ids") or []) if m in info]
        ok, err = apply_summary(t, raw.get("summary"), raw.get("evidence"),
                                alive, now=now, fras=raw.get("fras"))
        if not ok:
            failed.append({"topic_id": tid, "error": err})
            continue
        written.append({"topic_id": tid,
                        "title": t.get("title") or "(未命名)",
                        "size": len(alive),
                        "has_fras": bool(t.get("summary_fras"))})
    return written, failed


def assemble_report(
    *,
    written: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    knowledge: dict[str, Any],
    defaults: Any,
    pending_summary_ids: Callable[..., list[str]],
) -> dict[str, Any]:
    """十键回显体（怪癖 9-12）。

    ``pending_summary_ids`` 调两次——与原实现「pending_count 与 note
    各算一次」对齐，调用次数是行为面（猴子补丁计数可观测）。
    """
    pending = len(pending_summary_ids(knowledge, defaults=defaults))
    return {
        "ok": not failed or bool(written),
        "enabled": True,
        "action": ACTION,
        "written": len(written),
        "failed": failed,
        "failed_count": len(failed),
        "results": written,
        "saved": bool(written),
        "pending_count": pending,
        "note": ("已写回 %d 条%s；写回即落盘。剩余待归纳 %d 个。"
                 % (len(written),
                    "" if not failed else "，%d 条被退回（见 failed）"
                    % len(failed),
                    len(pending_summary_ids(knowledge, defaults=defaults)))),
    }


def collect(
    items: Any,
    *,
    knowledge: dict[str, Any],
    items_of: Callable[..., list[dict[str, Any]]],
    apply_summary: Callable[..., tuple[bool, str]],
    now: float,
    defaults: Any,
    pending_summary_ids: Callable[..., list[str]],
    persist: Callable[[], Any],
) -> dict[str, Any]:
    """编排入口（壳注入八面）。

    守卫不过直接回错误体——不落盘、不算 pending（怪癖 1/8）。
    """
    items, err = guard(items)
    if err is not None:
        return err
    written, failed = apply_items(
        items, knowledge=knowledge, items_of=items_of,
        apply_summary=apply_summary, now=now)
    if written:
        persist()
    return assemble_report(written=written, failed=failed,
                           knowledge=knowledge, defaults=defaults,
                           pending_summary_ids=pending_summary_ids)
