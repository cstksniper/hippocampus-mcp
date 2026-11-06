# -*- coding: utf-8 -*-
"""Read-only library tree: snapshot loading, aggregation, markdown + HTML rendering.

V3 of the OpenViking borrowing plan
(``docs/design/OpenViking借鉴点执行计划书_20261106.md`` §4). One aggregation
core feeds two output channels:

* **markdown tree** — for the terminal and for agents;
* **single-file HTML viewer** (``--html``) — self-contained (inline CSS/JS +
  embedded JSON payload), opened via ``file://``. No port, no server, no
  new dependency, no build chain, no external assets.

Read-only guarantees (mirrors ``scripts/audit_l0_dangling.py``):

* sqlite opened with ``file:...?mode=ro`` — no schema init, no WAL writes;
* json backend: plain reads, no temp files;
* this module never writes anything into the library directory.

Privacy red lines (plan §4.2):

* default preview is the first ``PREVIEW_CHARS`` characters of the content;
  ``full=True`` opts into whole content;
* a **user library** (``library.json`` declares ``"name": "user"``) is
  rendered with **zero content in every channel**, and ``full=True`` raises
  :class:`TreePrivacyError` — the narrowed privacy criterion gets no bypass;
* L0 raw text never enters any output — only counts do.

Layering: stdlib + ``domain.stats_views`` only. Records and layer payloads
enter as plain dicts; ``brain`` / ``store`` are never imported (the loader
reads the on-disk formats directly, exactly like the audit script).
"""
from __future__ import annotations

import html as _html
import json
import sqlite3
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from .domain.stats_views import pin_stats

__all__ = [
    "LibraryNotFoundError",
    "TreePrivacyError",
    "aggregate",
    "is_user_library",
    "l0_health",
    "library_kind",
    "render_html",
    "render_markdown",
    "snapshot_library",
]

DB_FILENAME = "hippocampus.db"
MEMORIES_FILENAME = "memories.json"
LIBRARY_META = "library.json"
RAW_LAYER = "brain_raw.json"
ENTITIES_LAYER = "brain_entities.json"
KNOWLEDGE_LAYER = "brain_knowledge.json"
MISSLOG_LAYER = "brain_misslog.json"
COUNTERS_LAYER = "brain_counters.json"

USER_LIBRARY_NAME = "user"
PREVIEW_CHARS = 60
PIN_THRESHOLD_DEFAULT = 0.7  # store.PIN_IMPORTANCE, injected (domain must not import store)

_STATE_ORDER = ("active", "superseded", "archived")


class LibraryNotFoundError(Exception):
    """The directory does not look like a hippocampus library."""


class TreePrivacyError(Exception):
    """A privacy red line was hit (e.g. ``--full`` on the user library)."""


# ---------------------------------------------------------------- loading
def _sqlite_ro(path: str) -> sqlite3.Connection:
    uri = "file:%s?mode=ro" % path.replace("\\", "/")
    return sqlite3.connect(uri, uri=True)


def library_kind(directory: str) -> str | None:
    """Return ``"sqlite"`` / ``"json"`` when *directory* looks like a library."""
    import os

    if os.path.isfile(os.path.join(directory, DB_FILENAME)):
        return "sqlite"
    if os.path.isfile(os.path.join(directory, MEMORIES_FILENAME)):
        return "json"
    return None


def is_user_library(directory: str) -> bool:
    """A library is *the* user library when its ``library.json`` says so.

    The name ``"user"`` is the reserved task/library name for the user-level
    preference library (see the task directory in ``session_start``), so the
    privacy red line follows the self-declared name, not a hard-coded path.
    """
    import os

    meta_path = os.path.join(directory, LIBRARY_META)
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and meta.get("name") == USER_LIBRARY_NAME


def snapshot_library(directory: str) -> dict[str, Any]:
    """Read records + layers from either backend without writing a byte."""
    import os

    kind = library_kind(directory)
    if kind is None:
        raise LibraryNotFoundError(
            f"not a hippocampus library (no {DB_FILENAME} / {MEMORIES_FILENAME}): "
            f"{directory}")
    layers: dict[str, Any] = {}
    mems: "OrderedDict[str, dict]" = OrderedDict()
    refs: dict[str, list[str]] = {}

    if kind == "sqlite":
        con = _sqlite_ro(os.path.join(directory, DB_FILENAME))
        try:
            for name, data in con.execute("select name, data from layers"):
                try:
                    layers[name] = json.loads(data)
                except (TypeError, ValueError):
                    layers[name] = None
            for mid, content, data in con.execute(
                    "select id, content, data from memories"):
                try:
                    rec = json.loads(data)
                except (TypeError, ValueError):
                    rec = {}
                rec.setdefault("id", mid)
                rec.setdefault("content", content)
                mems[mid] = rec
            for mid, sid in con.execute(
                    "select memory_id, source_id from memory_sources"):
                refs.setdefault(mid, []).append(str(sid))
        finally:
            con.close()
    else:
        for fname in os.listdir(directory):
            if not (fname.startswith("brain_") and fname.endswith(".json")):
                continue
            path = os.path.join(directory, fname)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    layers[fname] = json.load(fh)
            except (OSError, ValueError):
                layers[fname] = None
        mem_path = os.path.join(directory, MEMORIES_FILENAME)
        data = json.load(open(mem_path, encoding="utf-8"))
        recs = data.get("records") if isinstance(data, dict) else data
        if isinstance(recs, dict):
            recs = list(recs.values())
        for rec in recs or []:
            if isinstance(rec, dict) and rec.get("id"):
                mems[str(rec["id"])] = rec
        # json libraries keep source_ids embedded (no relation table)
        for mid, rec in mems.items():
            refs[mid] = [str(s) for s in (rec.get("source_ids") or [])]

    # belt: fill refs from embedded ids where the relation table had nothing
    for mid, rec in mems.items():
        if mid not in refs:
            refs[mid] = [str(s) for s in (rec.get("source_ids") or [])]

    return {"directory": directory, "kind": kind,
            "user_library": is_user_library(directory),
            "records": list(mems.values()), "layers": layers,
            "refs": refs}


# ---------------------------------------------------------------- helpers
def _day(ts: Any) -> str | None:
    """UTC ``YYYY-MM-DD`` for a positive epoch, else ``None`` (never 1970)."""
    try:
        val = float(ts)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 1e11:  # milliseconds
        val /= 1000.0
    try:
        return datetime.fromtimestamp(val, timezone.utc).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return None


def _preview(text: Any, n: int) -> str:
    return str(text or "").replace("\r", " ").replace("\n", " ")[:max(int(n), 0)]


def _topic_of(rec: dict) -> str:
    tags = rec.get("tags")
    if isinstance(tags, dict):
        val = tags.get("topic")
        if val not in (None, ""):
            return str(val)
    return "(ungrouped)"


def _entities_index(layers: dict) -> tuple[list[dict], dict[str, list[str]]]:
    """Top-entity rows + a memory_id → [entity name] reverse map."""
    raw = layers.get(ENTITIES_LAYER) or {}
    if not isinstance(raw, dict):
        return [], {}
    rows: list[dict] = []
    reverse: dict[str, list[str]] = {}
    for ent in raw.values():
        if not isinstance(ent, dict):
            continue
        mids = [str(m) for m in (ent.get("memory_ids") or [])]
        rows.append({"name": str(ent.get("name") or ent.get("key") or "?"),
                     "mention_count": int(ent.get("mention_count") or 0),
                     "memory_count": len(mids)})
        for mid in mids:
            reverse.setdefault(mid, []).append(str(ent.get("name") or "?"))
    rows.sort(key=lambda r: (-r["mention_count"], r["name"]))
    return rows, reverse


# ------------------------------------------------------------- aggregation
def l0_health(snapshot: dict[str, Any]) -> dict[str, int]:
    """L0 citation health over ALL records (library-level facts, read-only).

    The same numbers ``scripts/audit_l0_dangling.py`` reports: raw ids cited
    by L1 but absent from the raw layer (dangling), how many memories they
    affect, and orphan raws (present but never cited — the opposite drift).
    """
    layers = snapshot.get("layers") or {}
    refs = snapshot.get("refs") or {}
    raw_layer = layers.get(RAW_LAYER) or []
    raw_ids = ({str(r.get("id")) for r in raw_layer
                if isinstance(r, dict) and r.get("id")}
               if isinstance(raw_layer, list) else set())
    referenced = {sid for sids in refs.values() for sid in sids}
    dangling = {sid for sids in refs.values()
                for sid in sids if sid not in raw_ids}
    affected = sum(1 for sids in refs.values()
                   if any(sid not in raw_ids for sid in sids))
    return {"l0_raw_total": len(raw_ids),
            "refs_total": sum(len(v) for v in refs.values()),
            "dangling_ids": len(dangling),
            "affected_memories": affected,
            "orphan_raw": len(raw_ids - referenced)}


def aggregate(
    snapshot: dict[str, Any],
    *,
    include_states: tuple[str, ...] = ("active",),
    full: bool = False,
    preview_chars: int = PREVIEW_CHARS,
    limit: int | None = None,
    miss_limit: int = 20,
    top_entities: int = 10,
    pin_threshold: float = PIN_THRESHOLD_DEFAULT,
) -> dict[str, Any]:
    """Build the one aggregated structure both renderers consume.

    Counts are computed **once** here; markdown and HTML must render from
    this same dict so the two channels can never disagree (plan §4.2).

    Raises:
        TreePrivacyError: ``full=True`` on the user library (no bypass for
            the narrowed privacy criterion).
    """
    user_library = bool(snapshot.get("user_library"))
    if user_library and full:
        raise TreePrivacyError(
            "--full is refused on the user library (privacy: zero content)")

    records = snapshot.get("records") or []
    layers = snapshot.get("layers") or {}
    refs = snapshot.get("refs") or {}

    by_state = Counter(str(r.get("state") or "active") for r in records)
    included = [r for r in records
                if str(r.get("state") or "active") in include_states]

    _, entity_reverse = _entities_index(layers)

    entries: list[dict] = []
    for rec in included:
        mid = str(rec.get("id") or "")
        state = str(rec.get("state") or "active")
        entry: dict[str, Any] = {
            "id": mid,
            "id8": mid[:8],
            "date": _day(rec.get("timestamp")) or "-",
            "state": state,
            "cognitive_type": str(rec.get("cognitive_type") or "(untyped)"),
            "importance": rec.get("importance"),
            "pin": bool(rec.get("pin")),
            "pin_source": str(rec.get("pin_source") or ""),
            "quarantine": bool(rec.get("quarantine")),
            "topic": _topic_of(rec),
            "preview": "",
            "entities": entity_reverse.get(mid, [])[:3],
        }
        if not user_library:
            entry["preview"] = _preview(rec.get("content"), preview_chars)
            if full:
                entry["content"] = str(rec.get("content") or "")
        entries.append(entry)
    # newest first; ties broken by importance desc; undated last
    dated_e = [e for e in entries if e["date"] != "-"]
    undated_e = [e for e in entries if e["date"] == "-"]
    dated_e.sort(key=lambda e: -float(e["importance"] or 0.0))
    dated_e.sort(key=lambda e: e["date"], reverse=True)  # stable: keeps imp order
    undated_e.sort(key=lambda e: -float(e["importance"] or 0.0))
    entries = dated_e + undated_e

    groups: "OrderedDict[str, dict]" = OrderedDict()
    for entry in entries:
        topic = entry["topic"]
        group = groups.setdefault(topic, {"topic": topic, "count": 0,
                                          "by_type": Counter(), "entries": []})
        group["count"] += 1
        group["by_type"][entry["cognitive_type"]] += 1
        group["entries"].append(entry)
    group_list = sorted(groups.values(),
                        key=lambda g: (g["topic"] == "(ungrouped)",
                                       -g["count"], g["topic"]))
    for group in group_list:
        group["by_type"] = dict(group["by_type"])

    # L0 health (library-level facts, computed over ALL records like the audit)
    l0 = l0_health(snapshot)

    ps = pin_stats([SimpleNamespace(**{k: r.get(k)
                                       for k in ("state", "pin", "importance",
                                                 "pin_source")})
                    for r in records], pin_threshold=pin_threshold)

    dated = [e["date"] for e in entries if e["date"] != "-"]

    misses_raw = layers.get(MISSLOG_LAYER) or []
    misses: list[dict] = []
    if isinstance(misses_raw, list):
        recent = sorted(misses_raw, key=lambda m: float(m.get("ts") or 0.0),
                        reverse=True)[:max(int(miss_limit), 0)]
        for m in recent:
            misses.append({"date": _day(m.get("ts")) or "-",
                           "query": str(m.get("query") or ""),
                           "top_similarity": m.get("top_similarity"),
                           "path": str(m.get("path") or "")})

    knowledge_raw = layers.get(KNOWLEDGE_LAYER) or {}
    topics: list[dict] = []
    if isinstance(knowledge_raw, dict):
        for t in knowledge_raw.values():
            if isinstance(t, dict) and t.get("status") == "active":
                topics.append({"title": str(t.get("title") or t.get("id") or "?"),
                               "size": len(t.get("member_ids") or [])})
        topics.sort(key=lambda t: (-t["size"], t["title"]))

    entity_rows, _ = _entities_index(layers)

    counters = layers.get(COUNTERS_LAYER) or {}
    daily: dict[str, dict] = {}
    if isinstance(counters, dict):
        for day, row in sorted(counters.items())[-30:]:
            if isinstance(row, dict):
                daily[str(day)] = {k: int(v) for k, v in row.items()
                                   if isinstance(v, (int, float))}

    return {
        "library": {"directory": snapshot.get("directory") or "",
                    "backend": snapshot.get("kind") or "?",
                    "user_library": user_library},
        "health": {
            "records_total": len(records),
            "by_state": {s: by_state.get(s, 0) for s in _STATE_ORDER}
                        | {"other": sum(v for k, v in by_state.items()
                                        if k not in _STATE_ORDER)},
            "included_total": len(included),
            "include_states": list(include_states),
            "pin": {k: ps.get(k) for k in
                    ("pinned_total", "pinned_ratio", "suspected_auto_pinned",
                     "pin_source_breakdown", "level")},
            "l0_raw_total": l0["l0_raw_total"],
            "refs_total": l0["refs_total"],
            "dangling_ids": l0["dangling_ids"],
            "affected_memories": l0["affected_memories"],
            "orphan_raw": l0["orphan_raw"],
            "first_seen": min(dated) if dated else None,
            "last_seen": max(dated) if dated else None,
        },
        "entries": entries,
        "groups": group_list,
        "entities_top": entity_rows[:max(int(top_entities), 0)],
        "knowledge_topics": topics,
        "timeline": dict(sorted(Counter(e["date"] for e in entries
                                        if e["date"] != "-").items())),
        "daily": daily,
        "gaps": {"miss_total": len(misses_raw) if isinstance(misses_raw, list) else 0,
                 "misses": misses},
        "options": {"full": full, "preview_chars": preview_chars,
                    "limit": limit, "miss_limit": miss_limit,
                    "top_entities": top_entities},
    }


# ---------------------------------------------------------------- markdown
def render_markdown(aggregated: dict[str, Any]) -> str:
    """Render the aggregated structure as a markdown tree (terminal/agent)."""
    lib = aggregated["library"]
    health = aggregated["health"]
    options = aggregated["options"]
    user_library = lib["user_library"]

    pin = health["pin"] or {}
    breakdown = pin.get("pin_source_breakdown") or {}
    lines: list[str] = []
    lines.append("# Library tree — %s (%s backend)" % (lib["directory"], lib["backend"]))
    if user_library:
        lines.append("")
        lines.append("PRIVACY: user library — content omitted in every channel")
    lines.append("")
    lines.append("## Health")
    lines.append("- records: %d total (%s)" % (
        health["records_total"],
        ", ".join("%s %d" % (s, n) for s, n in health["by_state"].items())))
    lines.append("- included: %d (states: %s)" % (
        health["included_total"], ", ".join(health["include_states"])))
    lines.append("- pin: %d (%.1f%%, suspected auto %d; sources %s)" % (
        pin.get("pinned_total") or 0,
        100.0 * (pin.get("pinned_ratio") or 0.0),
        pin.get("suspected_auto_pinned") or 0,
        ", ".join("%s=%d" % kv for kv in breakdown.items())))
    lines.append("- L0: raw %d, refs %d, **dangling %d** (affecting %d memories), orphan raw %d" % (
        health["l0_raw_total"], health["refs_total"], health["dangling_ids"],
        health["affected_memories"], health["orphan_raw"]))
    lines.append("- span: %s → %s" % (health["first_seen"] or "-", health["last_seen"] or "-"))

    topics = aggregated["knowledge_topics"]
    if topics:
        lines.append("- knowledge topics (%d): %s" % (
            len(topics), ", ".join("%s(%d)" % (t["title"], t["size"])
                                   for t in topics[:12])))

    entities = aggregated["entities_top"]
    if entities:
        lines.append("- entities (top %d): %s" % (
            len(entities), ", ".join("%s(%d)" % (e["name"], e["mention_count"])
                                     for e in entities)))

    gaps = aggregated["gaps"]
    lines.append("- retrieval gaps: %d misses recorded" % gaps["miss_total"])
    for m in gaps["misses"][:10]:
        lines.append("  - [%s] sim=%.4f path=%s \"%s\"" % (
            m["date"], float(m["top_similarity"] or 0.0), m["path"], m["query"]))

    lines.append("")
    lines.append("## Groups (%d entries)" % aggregated["health"]["included_total"])
    shown = 0
    limit = options.get("limit")
    for group in aggregated["groups"]:
        type_note = ", ".join("%s %d" % kv for kv in group["by_type"].items())
        lines.append("")
        lines.append("### %s — %d entries (%s)" % (group["topic"], group["count"], type_note))
        rendered = 0
        for entry in group["entries"]:
            if limit is not None and shown >= int(limit):
                remaining = group["count"] - rendered
                if remaining > 0:
                    lines.append("  - … + %d more not shown (limit %d)" % (remaining, limit))
                break
            marks = []
            if entry["pin"]:
                marks.append("pin")
            if entry["state"] != "active":
                marks.append(entry["state"])
            if entry["quarantine"]:
                marks.append("quarantine")
            mark = "".join("[%s]" % m for m in marks)
            text = entry["content"] if "content" in entry else entry["preview"]
            body = ("text omitted (user library)" if user_library
                    else (text or "(empty)"))
            ent = (" @" + ",".join(entry["entities"])) if entry["entities"] else ""
            lines.append("  - %s %s %s (%s, imp %s)%s %s" % (
                mark, entry["date"], entry["id8"], entry["cognitive_type"],
                _fmt_imp(entry["importance"]), ent, body))
            shown += 1
            rendered += 1
    if limit is not None and shown < aggregated["health"]["included_total"]:
        lines.append("")
        lines.append("(entries shown: %d / %d — raise --limit for more)"
                     % (shown, aggregated["health"]["included_total"]))
    return "\n".join(lines)


def _fmt_imp(value: Any) -> str:
    try:
        return "%.2f" % float(value)
    except (TypeError, ValueError):
        return "-"


# -------------------------------------------------------------------- html
_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--ink:#d7dce3;--dim:#8b93a1;--line:#242a35;
--pin:#e8b93e;--sup:#7c8394;--qtn:#d9776b;--acc:#6ea8fe;--ok:#5fbf8f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
header{padding:18px 22px 8px}h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--dim);font-size:12px}
nav{position:sticky;top:0;background:var(--bg);padding:10px 22px;z-index:2;
border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
nav input{flex:1;background:var(--card);border:1px solid var(--line);
color:var(--ink);border-radius:8px;padding:8px 12px;font-size:13px;outline:none}
nav input:focus{border-color:var(--acc)}
#count{color:var(--dim);font-size:12px;white-space:nowrap}
main{padding:14px 22px 60px;max-width:1100px;margin:0 auto}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:14px 16px;margin:12px 0}
h2{font-size:13px;margin:0 0 10px;color:var(--acc);letter-spacing:.04em;
text-transform:uppercase}
.badge{display:inline-block;font-size:10.5px;border-radius:6px;padding:1px 6px;
margin-right:4px;border:1px solid var(--line);color:var(--dim)}
.badge.pin{color:var(--pin);border-color:var(--pin)}
.badge.superseded,.badge.archived{color:var(--sup)}
.badge.quarantine{color:var(--qtn);border-color:var(--qtn)}
.row{padding:5px 0;border-bottom:1px solid var(--line);display:flex;gap:8px;
align-items:baseline;flex-wrap:wrap}
.row:last-child{border-bottom:0}
.row .date{color:var(--dim);font-size:12px;min-width:78px}
.row .id{color:var(--acc);font-family:Consolas,monospace;font-size:12px}
.row .txt{flex:1;min-width:200px}
.row button{background:none;border:1px solid var(--line);color:var(--acc);
border-radius:6px;font-size:11px;cursor:pointer;padding:1px 8px}
.full{display:none;white-space:pre-wrap;background:var(--bg);
border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-top:4px;
font-size:12.5px;color:var(--ink)}
.chip{display:inline-block;background:var(--bg);border:1px solid var(--line);
border-radius:14px;padding:2px 10px;margin:3px 4px 3px 0;font-size:12px}
.chip b{color:var(--acc);font-weight:600}
.bar{height:10px;background:var(--acc);border-radius:5px;opacity:.75}
.tl{display:grid;grid-template-columns:86px 1fr 40px;gap:6px;align-items:center;
margin:3px 0;font-size:12px}
.tl .n{color:var(--dim);text-align:right}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
gap:10px}
.kv{background:var(--bg);border:1px solid var(--line);border-radius:10px;
padding:8px 10px}.kv .v{font-size:17px;font-weight:600}.kv .k{color:var(--dim);
font-size:11px}.kv.warn .v{color:var(--qtn)}
.miss{padding:4px 0;border-bottom:1px dashed var(--line);font-size:12.5px}
.miss .sim{color:var(--qtn);font-family:Consolas,monospace}
.miss .q{color:var(--ink)}.miss .meta{color:var(--dim)}
h3.g{font-size:13.5px;margin:16px 0 4px}
h4.t{font-size:12px;color:var(--dim);margin:10px 0 2px;font-weight:500}
</style></head>
<body>
<header><h1 id="h"></h1><div class="sub" id="sub"></div></header>
<nav><input id="q" type="search" placeholder="Search id / text / topic / entity …"
autocomplete="off"><span id="count"></span></nav>
<main>
<section id="health"><h2>Health</h2><div class="grid" id="health-cards"></div></section>
<section id="gaps"><h2>Retrieval gaps (misses)</h2><div id="gap-list"></div></section>
<section id="timeline"><h2>Timeline (entries per day)</h2><div id="tl"></div></section>
<section id="topics"><h2>Knowledge topics</h2><div id="topic-chips"></div></section>
<section id="entities"><h2>Entities (top)</h2><div id="ent-chips"></div></section>
<section id="groups"><h2>Groups</h2><div id="group-list"></div></section>
</main>
<script id="tree-data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
var DATA = JSON.parse(document.getElementById("tree-data").textContent);
function el(tag, cls, text){var n=document.createElement(tag);
if(cls)n.className=cls;if(text!==undefined&&text!==null)n.textContent=String(text);
return n;}
function badge(label, cls){var b=el("span","badge "+(cls||label),label);return b;}
/* health cards */
(function(){
  var h=DATA.health, cards=[
    {v:h.records_total, k:"records total"},
    {v:h.included_total, k:"included ("+h.include_states.join(",")+")"},
    {v:(h.pin&&h.pin.pinned_total)||0, k:"pinned"},
    {v:h.l0_raw_total, k:"L0 raw"},
    {v:h.dangling_ids, k:"dangling ids", warn:h.dangling_ids>0},
    {v:h.orphan_raw, k:"orphan raw"}
  ];
  var box=document.getElementById("health-cards");
  cards.forEach(function(c){var d=el("div","kv"+(c.warn?" warn":""));
  d.appendChild(el("div","v",c.v));d.appendChild(el("div","k",c.k));
  box.appendChild(d);});
  var sub=DATA.library.directory+" · "+DATA.library.backend+" backend"+
    (DATA.library.user_library?" · USER LIBRARY (content omitted)":"")+
    " · span "+(h.first_seen||"-")+" → "+(h.last_seen||"-")+
    (DATA.options.full?"":" · preview-only export (regenerate with --full for expandable full text)");
  document.getElementById("sub").textContent=sub;
  document.getElementById("h").textContent="Library tree";
})();
/* gaps */
(function(){
  var box=document.getElementById("gap-list");
  var ms=DATA.gaps.misses||[];
  if(!ms.length){box.appendChild(el("div","dim","no misses recorded"));return;}
  ms.forEach(function(m){var d=el("div","miss");
  d.appendChild(el("span","sim","sim="+(m.top_similarity===null||m.top_similarity===undefined?"-":Number(m.top_similarity).toFixed(4))+" "));
  d.appendChild(el("span","q",'"'+m.query+'" '));
  d.appendChild(el("span","meta","["+m.date+"] "+m.path));
  box.appendChild(d);});
})();
/* timeline + daily counters */
(function(){
  var tl=document.getElementById("tl"), tlData=DATA.timeline||{};
  var keys=Object.keys(tlData), max=1;
  keys.forEach(function(k){if(tlData[k]>max)max=tlData[k];});
  if(!keys.length){tl.appendChild(el("div","dim","no dated entries"));return;}
  keys.forEach(function(day){var row=el("div","tl");
  row.appendChild(el("span",null,day));
  var w=el("div");var b=el("div","bar");b.style.width=(60*tlData[day]/max)+"%";
  w.appendChild(b);row.appendChild(w);
  row.appendChild(el("span","n",tlData[day]));tl.appendChild(row);});
})();
/* chips */
function chips(id, rows, fmt){var box=document.getElementById(id);
  if(!rows.length){box.appendChild(el("span","dim","none"));return;}
  rows.forEach(function(r){box.appendChild(chip(r,fmt));});}
function chip(r,fmt){var c=el("span","chip");var parts=fmt(r);
  c.appendChild(el("b",null,parts[0]));if(parts[1])c.appendChild(document.createTextNode(" "+parts[1]));
  return c;}
chips("topic-chips", DATA.knowledge_topics||[], function(t){return [t.title, "("+t.size+")"];});
chips("ent-chips", DATA.entities_top||[], function(e){return [e.name, "("+e.mention_count+")"];});
/* groups + rows (searchable) */
var ROWS=[];
(function(){
  var box=document.getElementById("group-list");
  (DATA.groups||[]).forEach(function(g){
    var h3=el("h3","g",g.topic+" — "+g.count+" entries ("+
      Object.keys(g.by_type).map(function(k){return k+" "+g.by_type[k];}).join(", ")+")");
    box.appendChild(h3);
    g.entries.forEach(function(e,idx){
      var row=el("div","row");row.setAttribute("data-idx",String(ROWS.length));
      var searchable=((e.id||"")+" "+(e.preview||"")+" "+(e.content||"")+" "+
        e.topic+" "+e.cognitive_type+" "+(e.entities||[]).join(" ")).toLowerCase();
      ROWS.push({row:row,search:searchable});
      if(e.pin)row.appendChild(badge("pin"));
      if(e.state!=="active")row.appendChild(badge(e.state));
      if(e.quarantine)row.appendChild(badge("quarantine"));
      row.appendChild(el("span","date",e.date));
      row.appendChild(el("span","id",e.id8));
      var txt=DATA.library.user_library?"(content omitted)":
        (e.content||e.preview||"(empty)");
      var span=el("span","txt",e.content?e.preview||e.content:txt);
      row.appendChild(span);
      if(e.entities&&e.entities.length)row.appendChild(el("span","dim","@"+e.entities.join(",@")));
      if(e.content){var btn=el("button",null,"full");
        var pre=el("div","full",e.content);
        btn.addEventListener("click",function(){
          pre.style.display=(pre.style.display==="block")?"none":"block";});
        row.appendChild(btn);row.appendChild(pre);}
      box.appendChild(row);
    });
  });
})();
/* search */
(function(){
  var input=document.getElementById("q"),count=document.getElementById("count");
  function apply(){var q=input.value.trim().toLowerCase(),shown=0;
    ROWS.forEach(function(r){var hit=!q||r.search.indexOf(q)>=0;
      r.row.style.display=hit?"":"none";if(hit)shown++;});
    count.textContent=shown+" / "+ROWS.length+" entries";}
  input.addEventListener("input",apply);apply();
})();
</script>
</body></html>
"""


def render_html(aggregated: dict[str, Any], *, title: str = "Library tree") -> str:
    """Render the aggregated structure as one self-contained HTML file.

    The payload is embedded as ``application/json`` with ``</`` escaped to
    ``<\\/`` so record content can never close the surrounding script tag;
    the JS builds the DOM exclusively via ``textContent`` (no innerHTML),
    so a ``<script>`` inside memory content can never execute.
    """
    payload = json.dumps(aggregated, ensure_ascii=False,
                         separators=(",", ":")).replace("</", "<\\/")
    return (_HTML_TEMPLATE
            .replace("__TITLE__", _html.escape(title, quote=True))
            .replace("__PAYLOAD__", payload))
