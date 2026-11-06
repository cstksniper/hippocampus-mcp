# -*- coding: utf-8 -*-
"""Source-file signatures and drift audit (extracted from brain/core.py).

Houses the implementation bodies of four brain methods (``_resolve_source_path`` /
``_file_sig`` / ``record_source_sig`` / ``source_drift``) and one pure helper
(``_PureMixin._drift_row``); the brain side keeps only thin wrappers.

Why store/ rather than brain / domain / application:
- This is a **query combination over inventory records** (``drift_audit`` walks store
  records to compute reference facts, then checks each source against the disk's
  current state) — the same placement criterion as contradictions;
- It needs ``store.memory.MemoryState``'s state machine to judge active; domain is
  constrained by the dependency matrix (may only depend on common) and cannot touch
  it, and application is downstream of brain ("use-case logic flows one way from
  brain to application and never turns back");
- File I/O (os.stat / open / hashlib) has precedent in the store layer (backend).

**Knob discipline** (inherited from tag_ops / ledger): the two Config knobs
``source_root`` / ``sig_enabled`` are fetched fresh from the brain side by the
wrapper layer at call time — this module does not import config (store is bound by
the matrix and may not import config either). The only directly imported constant is
``SOURCE_SIG_CACHE_MAX`` (the FIFO cache cap of ``_file_sig``; no monkeypatch
precedent anywhere in the repo, the same exception as tag_ops's TAG_OPS_FILE).

State semantics (verbatim identical to the original brain-side implementation):
- ``file_sig``'s cache key includes ``(path, mtime_ns, size)`` — a changed file is
  naturally rehashed, an unchanged one is hashed only once per process no matter how
  many times it is ingested; on OSError (file being deleted / no permission) it
  returns None, the caller treats it as "could not resolve", and the write path
  does not blow up;
- Cache eviction is FIFO (dicts preserve order), dropping the oldest; it is a pure
  performance cache and does not chase optimal hit rate;
- ``record_source_sig`` leaves the layer untouched when the content facts have not
  changed (does not even touch the dict) — a layer fingerprint changing on every
  ingest would make every save write to disk (a 500-entry bulk import = 500 disk
  writes); ``first_seen`` is inherited from an existing entry; only a first sighting
  records now;
- ``drift_audit`` is read-only: it does not modify ``source_sigs``, does not write
  files, and computes the disk facts (mtime/size/sha1) on the spot.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Iterable

from ..common.constants import SOURCE_SIG_CACHE_MAX, _now
from .memory import MemoryState


def resolve_source_path(source: str, persist_dir: str | None,
                        source_root: str) -> str | None:
    """Resolve a source label into an absolute path on disk; return None when unresolvable.

    Resolution order: absolute path → memory-library directory → ``source_root`` →
    process CWD. **Unresolvable is the norm, not an error**: source also carries
    session ids and free-text origins ("group chat" / "verbal"), which have no
    corresponding file to begin with — recording a fake signature for them would only
    muddy the unbaselined pool.
    """
    s = str(source or "").strip()
    if not s:
        return None
    candidates = [s] if os.path.isabs(s) else [
        os.path.join(persist_dir or "", s),
        os.path.join(source_root, s) if source_root else s,
        os.path.abspath(s),
    ]
    for c in candidates:
        try:
            if c and os.path.isfile(c):
                return os.path.abspath(c)
        except OSError:
            continue
    return None


def file_sig(path: str, cache: dict[tuple[str, int, int], str],
             cache_max: int = SOURCE_SIG_CACHE_MAX) -> dict[str, Any] | None:
    """Compute a file's {mtime_ns, size, sha1}; cached by (path, mtime, size).

    The cache key includes mtime/size: a changed file (mtime changed) is naturally
    recomputed; unchanged, repeated ingests of the same file within one process hash
    it only once. On OSError (file being deleted / no permission) it returns None —
    the caller treats it as "could not resolve", and the write path does not blow up.

    Eviction is FIFO (dicts preserve order), dropping the oldest: a pure performance
    cache that does not chase optimal hit rate.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    cached = cache.get(key)
    if cached is None:
        h = hashlib.sha1()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return None
        cached = h.hexdigest()
        if len(cache) >= cache_max:
            # FIFO drops the oldest (dicts preserve order): a pure performance cache, not chasing optimal hit rate
            cache.pop(next(iter(cache)))
        cache[key] = cached
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "sha1": cached}


def record_source_sig(source: str, source_sigs: dict[str, dict[str, Any]],
                      persist_dir: str | None, source_root: str,
                      sig_cache: dict[tuple[str, int, int], str], *,
                      sig_enabled: bool) -> dict[str, Any] | None:
    """Write side: record a source-file signature for a source label (called by ingest).

    **Content facts unchanged means the layer is not touched**: the signature entry
    carries sig_at; if every ingest re-stamped it, the layer fingerprint would change
    every time ⇒ every save would write to disk (a 500-entry bulk import = 500 disk
    writes). When nothing changed, not even the dict is touched and the layer stays
    clean.

    Args:
        source: the source label.
        source_sigs: the brain's signature layer (this function modifies it in place).
        persist_dir / source_root: path-resolution inputs (fetched fresh on the brain
            side).
        sig_cache: ``_file_sig``'s sha1 cache (a brain instance attribute, modified
            in place).
        sig_enabled: the Config knob source_sig (when off, return None directly).

    Returns:
        The currently effective signature entry (newly recorded or existing); returns
        None when the path cannot be resolved / the switch is off.
    """
    if not sig_enabled:
        return None
    s = str(source or "").strip()
    if not s:
        return None
    path = resolve_source_path(s, persist_dir, source_root)
    if not path:
        return None
    sig = file_sig(path, sig_cache)
    if sig is None:
        return None
    prev = source_sigs.get(s)
    if (isinstance(prev, dict)
            and prev.get("path") == path
            and prev.get("mtime_ns") == sig["mtime_ns"]
            and prev.get("size") == sig["size"]
            and prev.get("sha1") == sig["sha1"]):
        return prev  # unchanged: do not touch the layer
    now = _now()
    entry = dict(sig)
    entry["path"] = path
    entry["sig_at"] = now
    entry["first_seen"] = (float(prev.get("first_seen") or now)
                           if isinstance(prev, dict) else now)
    source_sigs[s] = entry
    return entry


def drift_row(source: str, category: str,
              ref: dict[str, int] | None,
              sig: dict[str, Any] | None,
              cur: dict[str, Any] | None,
              diff: list[str] | None = None) -> dict[str, Any]:
    """One row of the drift audit: reference facts + recorded signature vs current disk state (sha1 shows only the first 8 hex digits)."""
    row: dict[str, Any] = {
        "source": source,
        "category": category,
        "memories": (ref or {}).get("memories", 0),
        "active": (ref or {}).get("active", 0),
    }
    if sig:
        row["recorded"] = {
            "sig_at": sig.get("sig_at"),
            "sha1": str(sig.get("sha1") or "")[:8],
            "size": sig.get("size"),
            "path": sig.get("path"),
        }
    if cur:
        row["current"] = {
            "sha1": str(cur.get("sha1") or "")[:8],
            "size": cur.get("size"),
            "mtime_ns": cur.get("mtime_ns"),
        }
    if diff:
        row["diff_fields"] = diff
    return row


def drift_audit(records: Iterable[Any],
                source_sigs: dict[str, dict[str, Any]], *,
                source: str = "", limit: int = 50, include_ok: bool = False,
                persist_dir: str | None = None, source_root: str = "",
                sig_enabled: bool = True,
                sig_cache: dict[tuple[str, int, int], str] | None = None,
                ) -> dict[str, Any]:
    """C1b: source-file drift audit — the server-side answer to "the file changed but the memory did not".

    Borrowing the predecessor's verify four-way classification, grounded in the memory
    library's actual shape as:

    ============== ======================================================
    Category       Meaning
    ============== ======================================================
    ``stale``      A signature was recorded and the file still exists, but at
                   least one of mtime/size/sha1 differs ⇒ the memories under
                   this source may be outdated (the file is the newer truth)
    ``missing``    A signature was recorded but the file is gone ⇒ the memories
                   remain but the source is lost (imported_but_deleted); the
                   chain back to the original text is broken
    ``unbaselined``The file resolves, but **a signature was never recorded**
                   (stock imported before this feature shipped, or written
                   during ``HIPPOCAMPUS_SOURCE_SIG=0``) ⇒ not "no drift" but
                   "cannot judge" — its own category, not mixed into ok
    ``orphan``     A signature was recorded but no memory in the library
                   references this source ⇒ the signature entry itself is
                   stale (memories fully forgotten / re-imported under a new
                   label)
    ``unresolvable``The source is not a file (session id / free text) ⇒  informational,
                   does not participate in drift judgment
    ``ok``         A signature was recorded, the file still exists, content facts match
    ============== ======================================================

    Every item carries its **denominator** (per-category counts + total sources +
    referenced memories): ``count:0`` has two possible truths (truly no drift vs
    never collected at all); without the denominator the caller always reads it as
    the former — the same discipline as contradictions's ``candidate_pool``.

    What is **not** done: ``never_ingested`` (files under some directory were never
    imported) requires knowing what the "watched directory" is; that is an input for
    project-side scripts (``export_memories(source=)`` can already compute it), and
    this tool assumes no watched directory.

    Args:
        records: an iterable of inventory records (brain passes
            ``store.iter_records()``).
        source_sigs: the signature layer (read-only).
        source: audit only this one source (empty = whole library).
        limit: at most how many rows to return per category (0 = counts only).
        include_ok: whether to also list ``ok`` sources (by default only problematic
            ones).
        persist_dir / source_root: path-resolution inputs (fetched fresh on the brain
            side).
        sig_enabled: the Config knob source_sig (self-reported inside denominators).
        sig_cache: the sha1 cache (a brain instance attribute, modified in place; a
            local empty cache is used by default).
    """
    cache = sig_cache if sig_cache is not None else {}
    # 1) Reference facts: source label → {memories, active} (the denominators come from here)
    refs: dict[str, dict[str, int]] = {}
    for rec in records:
        s = str(rec.source or "").strip()
        if not s:
            continue
        row = refs.setdefault(s, {"memories": 0, "active": 0})
        row["memories"] += 1
        if rec.state != MemoryState.ARCHIVED.value:
            row["active"] += 1
    # 2) Judgment: signature layer ∪ reference set (sources unique to either side must appear);
    #    when a source is named, **narrow to just that one** (not union-plus-one — that
    #    would make "query only a" return whole-library results, and the caller would
    #    believe the filter took effect).
    want = source.strip()
    labels = ([want] if want
              else sorted(set(source_sigs) | set(refs)))
    buckets: dict[str, list[dict[str, Any]]] = {
        "stale": [], "missing": [], "unbaselined": [],
        "orphan": [], "unresolvable": [], "ok": [],
    }
    for s in labels:
        sig = source_sigs.get(s)
        ref = refs.get(s)
        if sig is None:
            # No signature ever recorded: it is unbaselined (cannot judge) only if the
            # file resolves; otherwise "this source is not a file at all" (informational)
            if resolve_source_path(s, persist_dir, source_root):
                buckets["unbaselined"].append(drift_row(
                    s, "unbaselined", ref, None, None))
            else:
                buckets["unresolvable"].append(drift_row(
                    s, "unresolvable", ref, None, None))
            continue
        path = resolve_source_path(s, persist_dir, source_root)
        if path is None:
            buckets["missing"].append(drift_row(
                s, "missing", ref, sig, None))
            continue
        cur = file_sig(path, cache)
        if cur is None:
            buckets["missing"].append(drift_row(
                s, "missing", ref, sig, None))
            continue
        if ref is None:
            buckets["orphan"].append(drift_row(s, "orphan", None, sig, cur))
            continue
        diff = [k for k in ("mtime_ns", "size", "sha1")
                if sig.get(k) != cur.get(k)]
        if diff:
            buckets["stale"].append(drift_row(
                s, "stale", ref, sig, cur, diff))
        else:
            buckets["ok"].append(drift_row(s, "ok", ref, sig, cur))
    # 3) Counts + truncation. orphan by definition has ref None, so its "referenced memory count" is 0.
    counts = {k: len(v) for k, v in buckets.items()}
    out_rows: dict[str, list[dict[str, Any]]] = {}
    for k, rows in buckets.items():
        if k == "ok" and not include_ok:
            continue
        out_rows[k] = rows[:limit] if limit and limit > 0 else rows
    actionable = counts["stale"] + counts["missing"] + counts["unbaselined"]
    return {
        "ok": True,
        "scope": source.strip() or "(whole library)",
        "counts": counts,
        # Sum of the three actionable (worth a host's glance) categories — "0" is what
        # "no drift" means, but when unbaselined>0 it says "cannot judge" and must be
        # read separately.
        "actionable": actionable,
        "verdict": ("clean" if actionable == 0 else
                    ("unknown" if counts["unbaselined"] and
                     counts["stale"] == counts["missing"] == 0 else
                     "drifted")),
        "denominators": {
            "sources_total": len(labels),
            "recorded_sigs": len(source_sigs),
            "referenced_sources": len(refs),
            "memories_with_source": sum(r["memories"] for r in refs.values()),
            "sig_enabled": sig_enabled,
        },
        "rows": out_rows,
        "truncated": {k: max(0, len(v) - len(out_rows.get(k, [])))
                      for k, v in buckets.items()
                      if k in out_rows},
        "note": (
            "stale=文件已变、记忆未更新（文件是较新真相）；"
            "missing=文件没了、记忆还在（回查原话链路已断）；"
            "unbaselined=有文件但从没记过签名（无法判断，不是没漂移）；"
            "orphan=签名还在但没有记忆引用它；"
            "unresolvable=source 不是文件（会话 id 等），不参与判定。"
            "never_ingested（某目录从没导入过）不在本工具范围——"
            "那需要监视目录，请用 export_memories(source=...) 项目侧算。"
        ),
    }
