# -*- coding: utf-8 -*-
"""Pre-rendered text blocks: structured dict → text slots (pure-function layer extracted from brain/core.py).

These five functions are the text track of the **dual-track design** (design doc §3.3):
session_start's structured dict serves programmatic consumption and future human UI
rendering of real infographics, while the text slots here serve direct prompt splicing.
Both sides share the same source and the same account; changing the rendering rules
happens in exactly one place.

Why ``common/`` rather than ``application/``: the current caller is brain's
``_SessionMixin`` (session.py), and the dependency direction is "use-case logic flows
one way from brain to application and never turns back" — brain must not import
application. Putting it in common makes it usable by both layers; once session.py's
assembly moves into ``BootstrapService``, this layer's ownership does not change.

All are pure functions: dict in, str out, zero I/O, zero brain state.
"""
from __future__ import annotations

import time
from typing import Any

from .constants import TASK_PROFILE_CLIP, format_age, sparkline


def render_overview_block(ov: dict[str, Any]) -> str:
    """Pre-render the library overview block (≤180 token target; empty library gets a placeholder)."""
    if not ov.get("memories"):
        return "(Empty library: no memories yet)"
    lines = []
    head = "[Library] %s · %s memories" % (ov["name"], f"{ov['memories']:,}")
    if ov.get("knowledge_topics"):
        head += " · %d knowledge topics" % ov["knowledge_topics"]
    if ov.get("entities"):
        head += " · %d entities" % ov["entities"]
    lines.append(head)
    if ov.get("first_write_ts") and ov.get("last_write_ts"):
        first = time.strftime("%Y-%m", time.localtime(ov["first_write_ts"]))
        last = time.strftime("%Y-%m", time.localtime(ov["last_write_ts"]))
        months = max(1, round((ov["last_write_ts"] - ov["first_write_ts"]) / 2592000))
        span = "[Span] %s to %s (%d month%s)" % (first, last, months, "" if months == 1 else "s")
        if ov.get("recent_ingest"):
            span += " · last %d days +%d entries" % (ov["recent_days"], ov["recent_ingest"])
        lines.append(span)
    struct = "[Structure]"
    if ov.get("tag_coverage") is not None:
        struct += " tag coverage %.1f%%" % (ov["tag_coverage"] * 100)
        if ov.get("tag_top_keys"):
            struct += " (keys: %s)" % "/".join(ov["tag_top_keys"])
    if ov.get("knowledge_topics"):
        struct += " · knowledge summaries %d/%d" % (ov.get("knowledge_host") or 0,
                                                    ov["knowledge_topics"])
        if ov.get("knowledge_stale"):
            struct += " (%d need rewrite ⚠️)" % ov["knowledge_stale"]
    if struct != "[Structure]":
        lines.append(struct)
    if ov.get("top_entities"):
        tops = " ".join("%s(%d)" % (e["name"], e["memories"])
                        for e in ov["top_entities"])
        lines.append("[Entities] Top: %s" % tops)
    return "\n".join(lines)


def render_status_block(st: dict[str, Any]) -> str:
    """Pre-render the status block (≤150 token target). Omitted when contradiction checking is disabled."""
    lines = []
    pend = "[Pending] "
    parts = []
    if st.get("open_topics"):
        parts.append("%d open topics" % st["open_topics"])
    if st.get("open_intents"):
        parts.append("%d todos" % st["open_intents"])
    parts.append("consolidation pending" if st.get("pending_consolidate") else "consolidation not needed")
    lines.append(pend + " · ".join(parts))
    health = "[Health] "
    hparts = []
    hparts.append("L0 dangling %s" % ("yes" if st.get("l0_dangling") else 0))
    if st.get("contradictions") is not None:
        mark = ">=%d" % st["contradictions"] if st.get("contradictions_capped") \
            else str(st["contradictions"])
        hparts.append("%s contradictions" % mark)
    if st.get("recent_miss"):
        hparts.append("no hits in last %d days (%d misses; gap signal)"
                      % (st["recent_miss_days"], st["recent_miss"]))
    lines.append(health + " · ".join(hparts))
    n_sess = st.get("active_sessions") or 0
    lines.append("[Collaboration] %d other active sessions" % max(0, n_sess - 1)
                 if n_sess > 1 else "[Collaboration] no other active sessions")
    spark = sparkline(st.get("spark_series") or [])
    if spark:
        lines.append("[Activity] writes in last %d days %s" % (st["spark_days"], spark))
    else:
        recent = sum(st.get("spark_series") or [])
        lines.append("[Activity] %d writes in last %d days" % (recent, st["spark_days"]))
    # File registry: sync state between the project files and hippo.
    # When the registry is not enabled (no sidecar) the whole line is omitted — an
    # opt-in feature does not spend tokens on non-users.
    fr = st.get("file_registry")
    if fr:
        line = "[File registry] %s files" % f"{fr['total']:,}"
        if fr["stale"] or fr["missing"]:
            bits = []
            if fr["stale"]:
                bits.append("%d disk updates unsynced (oldest %d days ago)"
                            % (fr["stale"], fr["oldest_stale_days"]))
            if fr["missing"]:
                bits.append("%d missing" % fr["missing"])
            line += " · " + " · ".join(bits)
        else:
            line += " · all synced ✓"
        if not fr["scan_complete"]:
            line += " · scan incomplete"
        lines.append(line)
    return "\n".join(lines)


def render_map_index_block(mi: dict[str, Any] | None) -> str:
    """Pre-render the map index row (§5 sample): contract(42 entries · 3d ago) · genome(18 entries · ⚠️12d).

    Drift marker: ``contract(42 entries · 3d ago · drift 2)``——number of entries
    whose truth source changed (produced by auto verify).
    Conflict marker: ``contract(42 entries · 3d ago · conflict 1)``——
    legacy memory and registry sidecar disagree on the same key.
    Missing registration marker: ``dataset(12 entries · 2d ago · unregistered 3)``——
    3 keys exist in the directory but are not registered in the map.
    """
    if not mi or not mi.get("maps"):
        return ""
    parts = []
    for m in mi["maps"]:
        flag = "⚠️" if m["stale"] else ""
        seg = "%s(%d %s · %s%s)" % (m["name"], m["entries"],
                                         "entry" if m["entries"] == 1 else "entries",
                                         flag, format_age(m["age_days"]))
        if m.get("drifted"):
            seg += " · drift %d" % m["drifted"]
        if m.get("failed"):
            # Entries auto verify cannot classify (dynamic values / bad format).
            # Reported separately as unverified rather than silently swallowed.
            seg += " · unverified %d" % m["failed"]
        if m.get("conflicts"):
            seg += " · conflict %d" % m["conflicts"]
        if m.get("mentioned"):
            # Mention signal: after an update, some memory mentioned another state.
            # This is not a truth-source comparison result, only a hint.
            seg += " · possibly changed %d" % m["mentioned"]
        if m.get("unmatched"):
            # Keys present in the directory but not registered in the map.
            seg += " · unregistered %d" % m["unmatched"]
        parts.append(seg)
    line = "[Maps] " + " · ".join(parts)
    if mi.get("hidden"):
        line += " · +%d more" % mi["hidden"]
    return line


def render_map_detail_block(md: dict[str, Any] | None) -> str:
    """L1 map detail block: one desc line per map + up to 20 rows (key=status)."""
    if not md or not md.get("maps"):
        return ""
    lines = []
    for m in md["maps"]:
        head = "[Map · %s] %s" % (m["name"], m["desc"])
        if m.get("drifted"):
            head += " (⚠️ %d source changes)" % m["drifted"]
        lines.append(head)
        lines.append(" · ".join(m["rows"]))
        if m.get("hidden_rows"):
            lines.append("... %d more entries (row budget %d; use map_get for details)"
                         % (m["hidden_rows"], md["row_cap"]))
    if md.get("hidden"):
        lines.append("... %d related maps not included" % md["hidden"])
    return "\n".join(lines)


def render_task_profile_block(tp: dict[str, Any] | None) -> str:
    """Pre-render the task profile block (≤200 token target; empty input returns the empty string).

    Only non-empty lines are emitted—empty lines are pure noise for the agent.
    Numbers (hit counts / sizes) are never truncated; only titles are clipped.
    """
    if not tp:
        return ""
    lines: list[str] = []
    if tp.get("topics"):
        parts = []
        for t in tp["topics"]:
            # Only clip the title; keep the numeric suffix intact.
            suffix = " (%d entries · %d hits)" % (t["size"], t["hits"])
            title = t["title"]
            if len(title) + len(suffix) > TASK_PROFILE_CLIP:
                keep = max(TASK_PROFILE_CLIP - len(suffix) - 1, 0)
                title = title[:keep] + "…"
            parts.append(title + suffix)
        more = tp["topics_total"] - len(tp["topics"])
        lines.append("[Knowledge] " + " · ".join(parts)
                     + (" (+%d more)" % more if more > 0 else ""))
    if tp.get("entities"):
        parts = ["%s (%d hits / %d mentions)" % (e["name"], e["hits"], e["mentions"])
                 for e in tp["entities"]]
        more = tp["entities_total"] - len(tp["entities"])
        lines.append("[Entities] " + " · ".join(parts)
                     + (" (+%d more)" % more if more > 0 else ""))
    if tp.get("intents"):
        parts = []
        for it in tp["intents"]:
            text = " ".join(it["text"].split())
            if len(text) > TASK_PROFILE_CLIP:
                text = text[:TASK_PROFILE_CLIP - 1] + "…"
            if it.get("due"):
                text += " (%s)" % it["due"]
            parts.append(text)
        more = tp["intents_total"] - len(tp["intents"])
        lines.append("[Todos] " + " · ".join(parts)
                     + (" (+%d more)" % more if more > 0 else ""))
    if tp.get("misses"):
        parts = ["\"%s\" %.2f" % (
            " ".join(m["query"].split())[:TASK_PROFILE_CLIP],
            m["top_similarity"]) for m in tp["misses"]]
        more = tp["misses_total"] - len(tp["misses"])
        lines.append("[Gap] %d misses in last %d days: %s%s"
                     % (tp["misses_total"], tp["miss_days"], " · ".join(parts),
                        " (+%d more)" % more if more > 0 else ""))
    return "\n".join(lines)
