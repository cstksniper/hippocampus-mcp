# -*- coding: utf-8 -*-
"""In-domain implementation of the batch write-back of topic summaries
(``action="summarize_batch"``).

Extracted from ``HippoBrain._knowledge_summarize_batch``: this module contains only pure
logic and orchestration, zero imports -- the ``topic`` module functions
(apply_host_summary / pending_summary_ids), ``_knowledge_items``, ``_persist_knowledge``,
and ``_topic_defaults`` are all fetched and injected by the shell at call time
(same-layer cross-imports banned within domain).

Behavior quirks (each pinned in tests/test_m5_batch31_summarize_batch_flow.py):

  1. The two guards are ordered: ``not items`` runs before ``isinstance`` -- an empty
     dict / 0 / False / empty tuple all report "items is empty (expects
     [{topic_id, summary, evidence}])"; only a **non-empty** non-list reports
     "items must be an array";
  2. Per-item independence: one item failing validation rejects only that item (recorded
     in ``failed`` with a reason); other items in the same batch are written as usual --
     otherwise one mistyped id would lose dozens of summaries in the batch;
  3. The alive set is **fetched per item** (``items_of(require_vec=False)``, and only
     for items that **passed the topic lookup**): the call-count semantics match the
     original implementation, guarding the monkeypatch-counting surface;
  4. ``topic_id`` goes through ``str(... or "")`` + strip: 0 / False / None all
     normalize to the empty string and take the "no such topic: " path (concatenating
     the stripped tid);
  5. ``alive`` = ``member_ids`` ∩ alive set, preserving ``member_ids`` order;
  6. When ``apply_host_summary`` fails, the topic is left unchanged (the domain function
     guarantees that itself); this module neither patches up nor retries;
  7. ``written`` entries' ``title`` falls back to "(untitled)"; ``has_fras`` reads
     ``summary_fras`` **after** apply -- a stale fras not cleared ⇒ rewriting without
     passing fras also reports True;
  8. Persist condition: ``persist`` is called only when ``written`` is non-empty -- a
     batch that failed entirely does not write to disk;
  9. ``ok = not failed or bool(written)``: partial success counts as ok;
  10. ``pending_summary_ids`` is called **twice** (once each for ``pending_count`` and
      note); ``defaults`` is fetched once by the shell and injected;
  11. The note has three segments; with failures the middle segment inserts
      "; %d item(s) rejected (see failed)";
  12. The echo has a fixed ten-key order: ok, enabled, action, written, failed,
      failed_count, results, saved, pending_count, note -- note that ``written`` is a
      **count** while ``results`` is the list.
"""
from __future__ import annotations

from typing import Any, Callable

ACTION = "summarize_batch"


def guard(items: Any) -> tuple[Any, dict[str, Any] | None]:
    """The two guards (quirk 1). Returns ``(items, None)`` or ``(None, error body)``.

    The error body has a fixed four-key order: ok=False / enabled=True / action / error.
    The two messages are preserved verbatim -- tool-surface tests assert substrings on
    them.
    """
    if not items:
        return None, {"ok": False, "enabled": True, "action": ACTION,
                      "error": "items is empty (expects [{topic_id, summary, "
                               "evidence}])"}
    if not isinstance(items, list):
        return None, {"ok": False, "enabled": True, "action": ACTION,
                      "error": "items must be an array"}
    return items, None


def apply_items(
    items: list[Any],
    *,
    knowledge: dict[str, Any],
    items_of: Callable[..., list[dict[str, Any]]],
    apply_summary: Callable[..., tuple[bool, str]],
    now: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-item independent validation and write-back (quirks 2-7). Returns ``(written, failed)``.

    ``items_of`` is fetched per item (quirk 3, always given ``require_vec=False``);
    ``apply_summary``'s signature matches ``topic.apply_host_summary`` (topic,
    text, evidence, member_ids, now=, fras=); on failed validation it returns
    ``(False, reason)`` and **changes not a single field**.
    """
    written: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            failed.append({"topic_id": "", "error": "item is not an object"})
            continue
        tid = str(raw.get("topic_id") or "").strip()
        t = knowledge.get(tid)
        if t is None:
            failed.append({"topic_id": tid, "error": "no such topic: " + tid})
            continue
        info = {str(it["id"]): it for it in items_of(require_vec=False)}
        alive = [m for m in (t.get("member_ids") or []) if m in info]
        ok, err = apply_summary(t, raw.get("summary"), raw.get("evidence"),
                                alive, now=now, fras=raw.get("fras"))
        if not ok:
            failed.append({"topic_id": tid, "error": err})
            continue
        written.append({"topic_id": tid,
                        "title": t.get("title") or "(untitled)",
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
    """The ten-key echo body (quirks 9-12).

    ``pending_summary_ids`` is called twice -- matching the original implementation's
    "once for pending_count and once for note"; the call count is part of the behavior
    surface (observable via monkeypatch counting).
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
        "note": ("Wrote back %d item(s)%s; write-back persists to disk. %d topic(s) still pending."
                 % (len(written),
                    "" if not failed else "; %d item(s) rejected (see failed)"
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
    """Orchestration entry point (eight things injected by the shell).

    A guard failure returns the error body directly -- nothing persisted to disk, no
    pending counted (quirks 1/8).
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
