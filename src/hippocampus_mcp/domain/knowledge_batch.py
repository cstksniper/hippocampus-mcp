# -*- coding: utf-8 -*-
"""The four logic pieces of ``brain.core._knowledge_brief_batch`` (extracted from core).

**Why this is worth moving**: ``_knowledge_brief_batch`` is the landing point of
``action="brief_batch"``; within its 63 lines, the four pieces — limit clamping,
candidate picking, material projection, echo assembly — are all pure logic, yet they
live on the self surface and cannot be unit-tested. After the move, the shell is a
single straight line of "fetch data → inject → orchestrate".

**Dependency direction**: this module has zero imports — the four callbacks
``ordered`` / ``pending_summary_ids`` / ``representative_ids`` / ``summary_state`` and
the ``clip`` constant (``TOPIC_EVIDENCE_CLIP``) are all fetched fresh by the shell at
call time and injected (even same-layer cross-references inside domain are not
allowed; the topic package lives in domain but may only depend on common; this
follows the injection precedent of topic_material). ``defaults`` (``_topic_defaults()``
fetched fresh) belongs to the injection surface too.

**Replicated quirks (changing any one is a behavior change; parity will catch it)**:

1. ``clamp_batch``'s ``int(batch or 10)`` — when batch is falsy (0/None/""), the result
   is the **default 10**, not the clamp lower bound 1; only negatives and
   over-upper-bound values are truly clamped.
   Same for per_topic (default 5, clamped to [1,30]; batch clamped to [1,50]).
2. ``pick_candidates``'s want set is stripped and drops empty strings; **explicit
   topic_ids bypass the pending account** (even fragmented topics with size<3 must be
   fetchable — "named explicitly means I want this one"); topic_ids None/empty ⇒
   auto-pick by pending_summary_ids (the same account as ``knowledge_map``'s pending).
3. The return order for explicitly named topics follows the layer order of
   ``ordered(knowledge)``, **not the request order** (cand is a filter, not a reorder).
4. ``build_items``'s ``alive`` = member_ids ∩ info (the require_vec=False full set) —
   ``size`` reports the alive count, not the length of member_ids (dead members do not
   count); content is whitespace-normalized first (``" ".join(str(...).split())``) then
   ``[:clip]``; ``truncated`` is strictly greater (exactly equal to clip does not count
   as truncated).
5. ``summary_state(t, alive)`` computes the summarization state over alive members.
6. ``assemble_report``'s fixed 11-key order (key order is signal); ``pending``
   truncated to the first 50; ``done = not pending_all`` (empty list ⇒ True, bool()).
7. This method **does not check layer_enabled** — the gate is in the caller
   (at the top of ``knowledge_topic``); with the layer off, this returns an ok=True
   echo with empty items (that path is actually unreachable; recorded for the record).
"""
from __future__ import annotations

from typing import Any, Callable


# ======================================================================
# Piece 1: batch parameter clamping (old shell 3243-3247)
# ======================================================================
def clamp_batch(batch: Any, per_topic: Any) -> tuple[int, int]:
    """Normalize batch parameters: ``int(x or default)`` then clamp to [1,50] / [1,30].

    Quirk replication: ``batch or 10`` makes 0/None/"" all take 10 (not the lower bound
    1); only negatives and over-upper-bound values are truly clamped. Non-numeric
    strings raise ValueError (same as the original implementation).
    """
    n = int(batch or 10)
    n = 1 if n < 1 else (50 if n > 50 else n)
    per = int(per_topic or 5)
    per = 1 if per < 1 else (30 if per > 30 else per)
    return n, per


# ======================================================================
# Piece 2: candidate topic picking (old shell 3252-3260)
# ======================================================================
def pick_candidates(
    knowledge: dict[str, Any],
    topic_ids: list[str] | None,
    n: int,
    *,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    pending_summary_ids: Callable[..., list[str]],
    defaults: Any,
) -> list[dict[str, Any]]:
    """Candidate topic picking: explicit names or auto-pick by the pending account, then truncate to the first n.

    Args:
        knowledge: the topic layer dict (``self.knowledge``, passed straight through
            by the shell).
        topic_ids: explicitly named topic id list; None/empty ⇒ auto-pick.
        n: batch cap (already clamped by :func:`clamp_batch`).
        ordered: ``topic_mod.ordered`` (layer-order enumeration).
        pending_summary_ids: ``topic_mod.pending_summary_ids``.
        defaults: snapshot fetched fresh from ``_topic_defaults()``.

    Returns:
        The candidate topic list (layer order, truncated to the first n).
    """
    if topic_ids:
        want = {str(x).strip() for x in topic_ids if str(x).strip()}
        cand = [t for t in ordered(knowledge)
                if str(t.get("id")) in want]
    else:
        pend = set(pending_summary_ids(knowledge, defaults=defaults))
        cand = [t for t in ordered(knowledge)
                if str(t.get("id")) in pend]
    return cand[:n]


# ======================================================================
# Piece 3: material item projection (old shell 3262-3282)
# ======================================================================
def build_items(
    picked: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    info: dict[str, dict[str, Any]],
    *,
    per: int,
    representative_ids: Callable[..., list[str]],
    summary_state: Callable[..., Any],
    clip: int,
) -> list[dict[str, Any]]:
    """Project the picked topics into material items (fixed six-key order).

    Args:
        picked: candidate topics (the output of :func:`pick_candidates`).
        by_id: the with-vector memory index (built by the shell via
            ``_knowledge_items()``).
        info: the alive full-set index (built by the shell via
            ``_knowledge_items(require_vec=False)``) — alive determination uses only
            this; dead members do not count into size.
        per: number of material items per topic (already clamped).
        representative_ids: ``topic_mod.representative_ids``.
        summary_state: ``topic_mod.summary_state``.
        clip: the ``TOPIC_EVIDENCE_CLIP`` constant (fetched fresh and injected by the
            shell at call time).

    Returns:
        The material item list (same length as picked, six keys per topic).
    """
    out_items: list[dict[str, Any]] = []
    for t in picked:
        tid = str(t.get("id"))
        alive = [m for m in (t.get("member_ids") or []) if m in info]
        reps = representative_ids(t, by_id, per)
        members = []
        for mid in reps:
            it = by_id[mid]
            txt = " ".join(str(it.get("content") or "").split())
            members.append({
                "id": mid,
                "content": txt[:clip],
                "truncated": len(txt) > clip,
            })
        out_items.append({
            "topic_id": tid,
            "title": t.get("title") or "(untitled)",
            "size": len(alive),
            "summary": summary_state(t, alive),
            "members": members,
            "shown": len(members),
        })
    return out_items


# ======================================================================
# Piece 4: echo body assembly (old shell 3283-3296)
# ======================================================================
def assemble_report(
    *,
    out_items: list[dict[str, Any]],
    picked_count: int,
    per: int,
    pending_all: list[str],
    auto_built: bool,
) -> dict[str, Any]:
    """Assemble the brief_batch echo body (fixed 11-key order; key order is signal).

    Quirk replication: ``pending`` truncated to the first 50 (the full list can be long);
    ``done = not pending_all`` (empty list ⇒ True); ``auto_built`` passed straight
    through by the shell (the knowledge-map auto-build flag; this method does not know
    whether the build happened).
    """
    return {
        "ok": True, "enabled": True, "action": "brief_batch",
        "items": out_items,
        "picked": picked_count,
        "per_topic": per,
        "pending_count": len(pending_all),
        "pending": pending_all[:50],
        "done": not pending_all,
        "auto_built": auto_built,
        "note": ("素材按「离质心最近」排序——**不是时间序**，最早那批往往"
                 "正是后来被推翻的。读完用 action=summarize_batch 写回，"
                 "evidence 只填上面列出的 id。"),
    }


# ======================================================================
# Orchestration entry (the shell's only call site)
# ======================================================================
def collect(
    topic_ids: list[str] | None,
    *,
    batch: Any,
    per_topic: Any,
    auto_built: bool,
    knowledge: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    info: dict[str, dict[str, Any]],
    defaults: Any,
    ordered: Callable[[dict[str, Any]], list[dict[str, Any]]],
    pending_summary_ids: Callable[..., list[str]],
    representative_ids: Callable[..., list[str]],
    summary_state: Callable[..., Any],
    clip: int,
) -> dict[str, Any]:
    """The full brief_batch flow: clamp → pick → project → echo (the order is semantic).

    Line-by-line aligned with the original implementation's call order: clamp comes
    first; by_id/info are fetched by the shell before making this call (the order of
    the two ``_knowledge_items`` calls is also unchanged); the candidate pend set is
    computed once before projection; the pending_all used by the echo is computed once
    after projection (this path is read-only, so the two are equivalent, but the
    structure keeps two calls).
    """
    n, per = clamp_batch(batch, per_topic)
    picked = pick_candidates(knowledge, topic_ids, n,
                             ordered=ordered,
                             pending_summary_ids=pending_summary_ids,
                             defaults=defaults)
    out_items = build_items(picked, by_id, info, per=per,
                            representative_ids=representative_ids,
                            summary_state=summary_state, clip=clip)
    pending_all = pending_summary_ids(knowledge, defaults=defaults)
    return assemble_report(out_items=out_items, picked_count=len(picked),
                           per=per, pending_all=pending_all,
                           auto_built=auto_built)
