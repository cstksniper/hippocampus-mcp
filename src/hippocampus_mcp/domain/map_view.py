"""Read-side projections for the map index (L0) and task-relevant map full
text (L1).

Neither entry point searches, imports models, or writes to disk -- pure
read-side projections:

- :func:`map_index` merges the two sources (legacy reserved-tag memories +
  the ``maps.json`` sidecar), reporting only names + entry counts + freshness
  + drift/conflict counts, with no content (zero content is the balance point
  between "necessary information" and "must not be too long");
- :func:`map_detail` uses task tokens to hit a map's desc / entry keys and
  delivers the whole map within the cap (L1).

**Why these capabilities live in domain**: they are pure function projections
(input records + registry, output dict), touching neither I/O nor models. But
**what is read** is injected by the caller:

- ``records`` / ``registry``: the caller fetches them from the store and the
  map sidecar and passes them in -- domain must not import store nor map (arch
  matrix: domain may depend only on common); "where to read" is the assembly
  layer's business;
- ``list_maps``: the map package's row projection function itself, injected as
  a callback;
- ``limits``: all thresholds / reserved tag keys (``地图`` / ``条目`` /
  ``状态``) are fetched live by the caller from ``map.constants`` -- the
  constants live in the map package, invisible to domain; a hardcoded copy
  would sooner or later diverge from the map package;
- ``now`` / ``task_tokens_fn``: clock and tokenizer, injectable stand-ins
  (tests can pin time windows and lexicon behavior without waiting on a real
  clock).

Behavioral rules (verbatim aligned with the original brain methods, evidenced
by the parity probe):

- Archived memories do not enter the index (consolidate archives dormant map
  entries; counting them would inflate the index rows);
- Same key, different states across the two sources => ``conflicts`` count
  (same "enter-only, must-exit" discipline: it disappears as soon as either
  side becomes consistent);
- Conflict relations are not merged into one count: drifted (confirmed
  changed) and mentioned (someone said so) are accumulated and rendered
  separately -- two kinds of credibility;
- Group numbers / newer and the like are out of scope here (that is the
  coexistence semantics, see conflict_annotate);
- The index sorts by entry count descending, ties by name; details sort by
  hit-key count descending, ⚠️ rows pinned on top.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..common.constants import _now, task_tokens


@dataclass(frozen=True)
class IndexLimits:
    """Index row truncation and reserved tag keys (all injected, fetched live
    by the caller)."""

    max_maps: int          # MAP_INDEX_MAX: at most how many maps an index row carries
    stale_days: int        # MAP_STALE_DAYS: the ⚠️ staleness threshold
    name_key: str          # MAP_NAME_KEY: which map this memory belongs to
    entry_key: str         # MAP_ENTRY_KEY: the entry key
    status_key: str        # MAP_STATUS_KEY: the entry's current status


@dataclass(frozen=True)
class DetailLimits:
    """The two caps for L1 delivery (injected, fetched live by the caller)."""

    max_maps: int          # MAP_DETAIL_MAX_MAPS: at most how many maps enter the package at once
    max_rows: int          # MAP_DETAIL_MAX_ROWS: per-map row cap


def map_index(
    records: Iterable[Any],
    registry: dict[str, Any] | None,
    *,
    list_maps: Callable[[dict[str, Any]], list[dict[str, Any]]],
    now: Callable[[], float] | None = None,
    archived_state: str = "",
    limits: IndexLimits,
) -> dict[str, Any]:
    """Map index (L0 delivery): which maps the library has, how many entries
    each has, and how long since each was updated.

    Two sources merged:

    - **legacy**: memories carrying reserved tag keys (``{"地图": "合约", "条目":
      "CN2601", "状态": "验证集"}``), one entry per key;
    - **registry**: the ``maps.json`` sidecar (the formal map layer written by
      map_set).

    Same-named maps add up (entry counts summed, newest taken, source marked
    ``both``) -- old data is not lost, new data goes the new way. Pure O(n)
    reading of existing layers; no retrieval, no clustering.

    Two points tightened after design review:

    - **Archived entries do not enter the index** (consolidate archives
      dormant map-entry memories; counting them would inflate the index rows,
      and archived entries are unreachable by recall);
    - **Dual-source same-key conflict detection**: when the same (map, entry
      key) has inconsistent states between legacy memories and the registry
      sidecar => ``conflicts`` count + the index row's "conflicts N".
      Summing only solved count merging, not "two states competing" -- before
      the fix it could live silently inside a both map.
    """
    maps: dict[str, dict[str, Any]] = {}
    legacy_keys: dict[str, dict[str, str]] = {}
    registry_keys: dict[str, dict[str, str]] = {}
    # legacy: the tag namespace
    for rec in records:
        if rec.state == archived_state:
            continue
        tags = rec.tags or {}
        name = tags.get(limits.name_key)
        if not name:
            continue
        m = maps.setdefault(str(name), {"entries": 0, "newest": 0.0,
                                        "drifted": 0, "legacy": False,
                                        "registry": False, "failed": 0,
                                        "unmatched": 0, "dir_missing": 0,
                                        "dir_conflict": 0, "mentioned": 0})
        m["entries"] += 1
        m["legacy"] = True
        m["newest"] = max(m["newest"], float(rec.timestamp or 0.0))
        k = str(tags.get(limits.entry_key) or "").strip()
        if k:
            legacy_keys.setdefault(str(name), {})[k] = \
                str(tags.get(limits.status_key) or "").strip()
    # registry: the maps.json sidecar
    data = registry
    if data:
        for item in list_maps(data):
            m = maps.setdefault(item["name"],
                                {"entries": 0, "newest": 0.0,
                                 "drifted": 0, "legacy": False,
                                 "registry": False, "failed": 0,
                                 "unmatched": 0, "dir_missing": 0,
                                 "dir_conflict": 0, "mentioned": 0})
            m["entries"] += item["entries"]
            m["drifted"] += item["drifted"]
            m["failed"] += item.get("failed") or 0
            m["unmatched"] += item.get("unmatched") or 0
            m["dir_missing"] += item.get("dir_missing") or 0
            m["dir_conflict"] += item.get("dir_conflict") or 0
            # Mention signal: the number of entries mentioned at another status
            # after an update. Accumulated and rendered separately from
            # drifted, never merged into one number --
            # "the truth changed" and "someone said so" are two kinds of
            # credibility.
            m["mentioned"] += item.get("mentioned") or 0
            if item.get("mention_state"):
                m["mention_state"] = item["mention_state"]
            if item.get("auto_error"):
                m["auto_error"] = item["auto_error"]
            if item.get("auto_incomplete"):
                m["auto_incomplete"] = True
            m["registry"] = True
            m["newest"] = max(m["newest"], item["newest_ts"])
            for k, e in ((data["maps"].get(item["name"]) or {})
                         .get("entries") or {}).items():
                registry_keys.setdefault(item["name"], {})[str(k)] = \
                    str((e or {}).get("status") or "").strip()
    now_ts = (now or _now)()
    out = []
    for name, m in maps.items():
        age = int((now_ts - m["newest"]) // 86400) if m["newest"] else None
        # Same key, different states across the two sources = conflict (legacy
        # old value vs registry new value). Same "enter-only, must-exit"
        # discipline: it disappears as soon as either side becomes consistent.
        lk = legacy_keys.get(name) or {}
        rk = registry_keys.get(name) or {}
        conflict_keys = sorted(k for k in set(lk) & set(rk)
                               if lk[k] != rk[k])
        out.append({
            "name": name, "entries": m["entries"],
            "drifted": m["drifted"],
            "failed": m["failed"],
            "unmatched": m["unmatched"],
            "dir_missing": m["dir_missing"],
            "dir_conflict": m["dir_conflict"],
            "mentioned": m["mentioned"],
            "mention_state": m.get("mention_state") or {},
            "conflicts": len(conflict_keys),
            "conflict_keys": conflict_keys[:5],
            "auto_error": m.get("auto_error") or "",
            "auto_incomplete": bool(m.get("auto_incomplete")),
            "source": ("both" if m["legacy"] and m["registry"]
                       else "registry" if m["registry"] else "legacy"),
            "newest_ts": m["newest"], "age_days": age,
            "stale": bool(age is not None and age > limits.stale_days),
        })
    out.sort(key=lambda x: (-x["entries"], x["name"]))
    return {
        "maps": out[:limits.max_maps],
        "maps_total": len(out),
        "hidden": max(0, len(out) - limits.max_maps),
        "stale_days_threshold": limits.stale_days,
    }


def map_detail(
    task: str,
    tp: dict[str, Any] | None,
    registry: dict[str, Any] | None,
    *,
    list_maps: Callable[[dict[str, Any]], list[dict[str, Any]]],
    task_tokens_fn: Callable[[str, list[str] | None], list[str]] = task_tokens,
    limits: DetailLimits,
) -> dict[str, Any]:
    """Full-text projection of task-relevant maps (L1 delivery). Pure sidecar
    reading; no retrieval, no models.

    token = the task text + the anchor entity names in tp (the same
    :func:`task_tokens` as the task profile card; **no re-retrieval**). Hit
    criterion: a map's desc contains any token, or any of its entry keys
    contains a token.
    """
    if not (task or "").strip():
        return {"task": "", "maps": [], "maps_total": 0, "hidden": 0,
                "row_cap": limits.max_rows}
    data = registry
    if not data:
        return {"task": task, "maps": [], "maps_total": 0, "hidden": 0,
                "row_cap": limits.max_rows}
    extras = [e.get("name") for e in (tp or {}).get("entities", [])
              if e.get("name")]
    tokens = [t for t in task_tokens_fn(task, extras) if t]
    if not tokens:
        return {"task": task, "maps": [], "maps_total": 0, "hidden": 0,
                "row_cap": limits.max_rows}
    matched: list[dict[str, Any]] = []
    for item in list_maps(data):
        m = (data.get("maps") or {}).get(item["name"]) or {}
        desc = str(m.get("desc") or item["name"]).lower()
        entries = m.get("entries") or {}
        hit_desc = any(t in desc for t in tokens)
        hit_keys = [k for k in entries
                    if any(t in str(k).lower() for t in tokens)]
        if not (hit_desc or hit_keys):
            continue
        rows = []
        for k in sorted(entries,
                        key=lambda k: -float(
                            (entries[k] or {}).get("updated_at") or 0.0)):
            e = entries[k] or {}
            row = "%s=%s" % (k, e.get("status") or "?")
            extra_bits = []
            for fk, fv in (e.get("fields") or {}).items():
                extra_bits.append("%s:%s" % (fk, fv))
            if e.get("note"):
                extra_bits.append(str(e["note"]))
            if extra_bits:
                row += "（%s）" % "；".join(extra_bits)
            if e.get("drift") or e.get("mention"):
                # Mentioned drift also gets the ⚠️ prefix. Its direction
                # is the opposite of drift ("possibly stale" rather than
                # "confirmed changed"), but the delivery action is the same
                # "take a look at this one"; the evidence is in mention.
                row = "⚠️" + row
            rows.append(row[:80])
        rows.sort(key=lambda r: r.startswith("⚠️"), reverse=True)
        matched.append({
            "name": item["name"], "desc": m.get("desc") or item["name"],
            "entries": item["entries"], "drifted": item["drifted"],
            "age_days": item["age_days"],
            "matched_by": "desc" if hit_desc else "key",
            "matched_keys": hit_keys[:5],
            "rows": rows[:limits.max_rows],
            "hidden_rows": max(0, len(rows) - limits.max_rows),
        })
    matched.sort(key=lambda x: (-len(x["matched_keys"]),
                                -x["entries"], x["name"]))
    return {
        "task": task,
        "maps": matched[:limits.max_maps],
        "maps_total": len(matched),
        "hidden": max(0, len(matched) - limits.max_maps),
        "row_cap": limits.max_rows,
    }
