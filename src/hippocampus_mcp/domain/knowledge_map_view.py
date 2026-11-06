# -*- coding: utf-8 -*-
"""Three pieces of logic from ``brain.core.knowledge_map`` (extracted from core).

**Why worth moving**: ``knowledge_map`` is the knowledge-map entry point; of
its 75 lines, the row-projection loop (22 lines), the pending/excluded dual
standards, and the echo-body assembly are all pure logic. After the move the
shell is just one straight line: "gate → auto-build → inject → orchestrate".

**Dependency direction**: this module has zero imports — the five
topic-package symbols ``ordered`` / ``STATUS_ARCHIVED`` /
``SUMMARY_KIND_HOST`` / ``signature_of`` / ``pending_summary_ids`` are all
fetched by the shell at call time and injected (same-layer imports are banned
inside domain; topic lives in domain but may only depend on common; following
the knowledge_batch precedent). ``stats`` / ``open_topics`` / ``open_intents``
/ ``playbook`` are brain-side capabilities, injected by value or as lazy
callbacks.

**Replicated quirks (changing any one is a behavior change; parity will catch it)**:

1. ``project_rows``'s ``limit`` uses truthiness: ``limit=0`` = unlimited (all
   rows); ``limit`` counts **rows already added** (archived ones skipped by
   continue don't take quota), and the break check is **after** the append
   (the limit-th row enters the list, then it breaks).
2. ``summary_stale`` is a three-clause and: ``kind==HOST and signature and
   signature != signature_of(member_ids)`` — host-summarized but empty
   signature ⇒ not marked stale (better to miss a mark than mis-mark; the
   rewrite right belongs to the host).
3. ``entities`` / ``tags`` each truncated to the first 3; ``span`` copied (not
   the same object as the topic layer); ``size`` / ``recall_hits`` via
   ``int(x or 0)``; empty ``title`` becomes ``(untitled)``.
4. ``pending_counts``' excluded standard is the same function with
   ``exclude_low_info=False`` (false-cluster self-report difference set);
   ``excluded_n = max(0, diff)`` — normally excluded >= pending, so max(0) is
   defensive (文案钉, pinned for display).
5. ``assemble_report``'s fixed 13-key order (key order is signal);
   ``playbook`` is fetched via callback only when ``pending`` is non-empty
   (empty pending ⇒ no call — lazy injection, call counts aligned with the
   original implementation one by one); ``note`` is the operating-manual text
   (deliberately in the response body, not the docstring, saving the
   tools/list token tax).
6. This method **does not check layer_enabled** — the gate is in the shell;
   auto-build (empty layer + records present) is also in the shell; the
   injected knowledge is always the **current-layer snapshot**.
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# Block 1: topic row projection (old shell 3021-3049)
# ======================================================================
def project_rows(
    knowledge: dict[str, Any],
    *,
    include_archived: bool,
    limit: int,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    status_archived: str,
    summary_kind_host: str,
    signature_of: Callable[[list[str]], str],
) -> list[dict[str, Any]]:
    """Project the topic layer into map rows (14 keys in fixed order, including stale self-report).

    Args:
        knowledge: The topic-layer dict (the shell passes the current-layer
            snapshot straight through).
        include_archived: Whether to include archived topics.
        limit: Row cap; **0 = unlimited** (truthiness check; see quirk 1).
        ordered: ``topic_mod.ordered`` (layer-order enumeration).
        status_archived: The ``topic_mod.STATUS_ARCHIVED`` constant.
        summary_kind_host: The ``topic_mod.SUMMARY_KIND_HOST`` constant.
        signature_of: ``topic_mod.signature_of`` (member signature function).

    Returns:
        Map row list (layer order, at most limit rows).
    """
    rows: list[dict[str, Any]] = []
    for t in ordered(knowledge):
        if not include_archived and t.get("status") == status_archived:
            continue
        rows.append({
            "id": t.get("id"),
            "title": t.get("title") or "(untitled)",
            "status": t.get("status"),
            "size": int(t.get("size") or 0),
            "summary": t.get("summary") or "",
            "summary_kind": t.get("summary_kind") or "",
            # The member set this summary was based on has changed ⇒ this
            # higher-level conclusion may no longer hold. No rewriting, only
            # annotating (the rewrite right belongs to the host), but it must
            # be visible.
            "summary_stale": bool(
                t.get("summary_kind") == summary_kind_host
                and t.get("summary_signature")
                and t.get("summary_signature")
                != signature_of(t.get("member_ids") or [])),
            "entities": list(t.get("entities") or [])[:3],
            "tags": list(t.get("tags") or [])[:3],
            "span": list(t.get("span") or []),
            "updated_at": t.get("updated_at"),
            "recall_hits": int(t.get("recall_hits") or 0),
            # S3: whether this summary carries the four-field structure
            # (fras). When machines need to process summaries by domain
            # (contradiction detection / domain cards), no structure means no
            # handle — make coverage visible first.
            "has_fras": bool(t.get("summary_fras")),
        })
        if limit and len(rows) >= limit:
            break
    return rows


# ======================================================================
# Block 2: pending / excluded dual standards (old shell 3050-3056)
# ======================================================================
def pending_counts(
    knowledge: dict[str, Any],
    *,
    defaults: Any,
    pending_summary_ids: Callable[..., list[str]],
) -> tuple[list[str], int]:
    """Pending list + false-cluster self-report difference set (the difference from the same function under two standards).

    Args:
        knowledge: The topic-layer dict.
        defaults: A fresh snapshot of ``_topic_defaults()``.
        pending_summary_ids: ``topic_mod.pending_summary_ids``.

    Returns:
        ``(pending, excluded_low_info_n)``.
    """
    pending = pending_summary_ids(knowledge, defaults=defaults)
    # False-cluster self-report: the batch within pending blocked by the
    # low-information criterion; without reporting it the host would believe
    # "everything is summarized". A difference set computed under the same
    # standard as pending.
    excluded = pending_summary_ids(knowledge, exclude_low_info=False,
                                   defaults=defaults)
    excluded_n = max(0, len(excluded) - len(pending))
    return pending, excluded_n


# ======================================================================
# Block 3: echo-body assembly (old shell 3057-3078)
# ======================================================================
def assemble_report(
    *,
    rows: list[dict[str, Any]],
    total: int,
    stats: dict[str, Any],
    open_topics: list[Any],
    open_intents: list[Any],
    auto_built: bool,
    pending: list[str],
    excluded_low_info_n: int,
    playbook: Callable[[int, int], Any],
) -> dict[str, Any]:
    """Assemble the knowledge-map echo body (13 keys in fixed order; key order is signal).

    Quirk replicated: ``playbook`` is fetched via callback and **called only
    once, only when pending is non-empty** (empty pending ⇒ None and no call —
    lazy injection, call counts aligned).
    """
    return {
        "enabled": True,
        "topics": rows,
        "shown": len(rows),
        "total": total,
        "stats": stats,
        "open_topics": open_topics,
        "open_intents": open_intents,
        "auto_built": auto_built,
        "note": ("代表句是抽取式摘录（离质心最近的几条原文），不是归纳；"
                 "真正的上层结论要宿主总结后写回"),
        "pending_excluded_low_info": excluded_low_info_n,
        # ── The "tidy this library" operating manual ─────────────────────
        # Deliberately in the **response body**, not the docstring: a
        # docstring is injected once per session with tools/list, so writing
        # it there is a long-term token tax; the response body only pays when
        # tidying is actually wanted, and can be generated dynamically from
        # the current library's actual state (how many pending, whether the
        # cap was hit).
        "pending": pending,
        "pending_count": len(pending),
        "playbook": (playbook(len(pending), len(rows)) if pending else None),
    }


# ======================================================================
# Orchestration entry (the shell's only call site)
# ======================================================================
def collect(
    knowledge: dict[str, Any],
    *,
    include_archived: bool,
    limit: int,
    defaults: Any,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    status_archived: str,
    summary_kind_host: str,
    signature_of: Callable[[list[str]], str],
    pending_summary_ids: Callable[..., list[str]],
    stats: dict[str, Any],
    open_topics: list[Any],
    open_intents: list[Any],
    playbook: Callable[[int, int], Any],
    auto_built: bool,
) -> dict[str, Any]:
    """The full knowledge_map flow: projection → dual-standard pending → echo (the order carries semantics).

    Call order aligned line by line with the original implementation: the row
    loop first; the two pending/excluded lists after it (excluded after
    pending); stats / open_* / playbook fetched by value/callback during echo
    assembly.
    """
    rows = project_rows(knowledge, include_archived=include_archived,
                        limit=limit, ordered=ordered,
                        status_archived=status_archived,
                        summary_kind_host=summary_kind_host,
                        signature_of=signature_of)
    pending, excluded_n = pending_counts(
        knowledge, defaults=defaults,
        pending_summary_ids=pending_summary_ids)
    return assemble_report(rows=rows, total=len(knowledge), stats=stats,
                           open_topics=open_topics,
                           open_intents=open_intents,
                           auto_built=auto_built, pending=pending,
                           excluded_low_info_n=excluded_n,
                           playbook=playbook)
