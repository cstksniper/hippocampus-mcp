# -*- coding: utf-8 -*-
"""The session-start package's "knowledge map" slot (extracted from
brain/core.py).

A few one-line knowledge points (title + size + one sentence).

🔴 **Deliberately no auto-sorting here**. The session-start package is on the
hot path (once per session), and the first sorting run is a whole-library
clustering -- measured 2-3 seconds on a real large library. Stuffing those
seconds into every session start buys only "in case the host wants to see the
map". So this reads only the layer already on disk; if empty, say plainly
"calling knowledge_map will sort once", leaving the one-time cost to an
explicit user call.

The one sentence prefers the **host summary** (that is the upper-layer
knowledge) and only falls back to the extractive representative sentence
without it.

Why domain/: pure derived projection. 🔴 But the topic semantics
(``ordered`` / archived status / ``summary_state``) **go through injection,
not import** -- arch G1's matrix is ``domain -> common only``; same-layer
mutual imports are not allowed at all (bitten by this before). Likewise
``store.count()`` is injected by the caller (only the "has data but never
sorted" branch needs it; injecting it as a callable preserves laziness).

Port discipline: ``limit`` / ``clip`` / the topic-layer switch are all
explicit parameters; no Config reads.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping


def knowledge_brief(
    knowledge: Mapping[str, Any],
    limit: int = 8,
    *,
    topic_layer_enabled: bool,
    ordered_topics: Callable[[Any], list[dict[str, Any]]],
    archived_status: str,
    summary_state: Callable[[dict[str, Any], Any], dict[str, Any]],
    store_count: Callable[[], int],
    clip: int,
) -> list[str]:
    """The session-start package's knowledge map slot: a few one-line
    knowledge points."""
    if limit <= 0 or not topic_layer_enabled:
        return []
    if not knowledge:
        # Has data but never sorted != no knowledge points: state the way out
        # clearly; do not let the host believe the library is empty
        if store_count():
            return ["(no knowledge points sorted yet; calling knowledge_map will sort them once)"]
        return []
    out: list[str] = []
    for t in ordered_topics(knowledge):
        if str(t.get("status") or "") == archived_status:
            continue
        title = str(t.get("title") or "(untitled)")
        size = len(t.get("member_ids") or [])
        # The summary slot holds the **currently effective** sentence: if a
        # host summary was written, it is the summary; otherwise it is the
        # extractive representative sentence filled by refresh (the two are
        # mutually exclusive, see topic.refresh).
        line = str(t.get("summary") or "").strip() or "(no summary yet)"
        mark = ""
        st = summary_state(t, t.get("member_ids"))
        if st.get("is_host"):
            # ⚠️ = a summary was written but its underlying members changed
            # later (the batch most deserving of being taken back to the host
            # for a rewrite)
            mark = "" if not st.get("stale") else "⚠️"
        head = "%s (%d records) %s" % (title, size, mark)
        # Truncate only the sentence, **never the title and size**: measured on
        # a real library, a hard cut of the whole line slices both together,
        # leaving half a title the host cannot match to any knowledge point
        # (far worse than seeing a few characters fewer).
        budget = clip - len(head)
        if budget < 8:
            head = head[:clip - 1] + "…"
            body = ""
        else:
            line = " ".join(line.split())
            body = line if len(line) <= budget else line[:budget - 1] + "…"
        out.append(head + body)
        if len(out) >= limit:
            break
    return out
