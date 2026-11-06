"""Map-layer sidecar storage (information map layer).

Core judgment: **map != memory** -- the map answers "what is the state
now", memory answers "what happened". So map entries do not enter the
recall retrieval path (no vectors, not memories); they are updated by key
(the same key written repeatedly, the old value replaced rather than
competing with the new), and continuously expire (unchanged means wrong).

Sidecar JSON (``<persist_dir>/maps.json``)::

    {
      "version": 1,
      "maps": {
        "合约": {
          "desc": "各合约分到哪个数据集",
          "created_at": 1710000000.0,
          "updated_at": 1710000000.0,
          "entries": {
            "CN2601": {
              "status": "验证集",
              "fields": {"来源": "2026-10 重划分"},
              "updated_at": ..., "updated_by": "",
              "truth_source": "config/contracts.yaml",  # 可选
              "source_sig": {"size":..,"mtime_ns":..,"hash":..} | null,
              "drift": null | {"reason": "changed"|"missing", "since": ..},
              "note": ""
            }
          }
        }
      }
    }

Four disciplines (same family as file registration; see the design doc's
four gates for detail):
1. **Gate 1 - update at write time**: status goes through
   :func:`set_entry`, a single entry point (the bottom layer of the
   map_set tool); once set, the key's old value is replaced wholesale --
   "one key, one entry" is guaranteed by the data model, not by caller
   discipline.
2. **Gate 2 - truth-source verification**: an entry may declare
   ``truth_source`` (a file path); on set, its (size, mtime_ns) + blake2b
   signature is recorded; :func:`verify` re-reads and compares -- on
   signature mismatch, read the content hash once to arbitrate (touch /
   git checkout do not false-positive); a real change marks ``drift``.
   Same idea as file_registry's hash arbitration (an entry whose
   truth_source points at a file path inherits this mechanism
   automatically).
3. **opt-in**: no sidecar, none of the map layer's behavior; missing
   returns None without crashing the caller (same contract as
   file_registry.load).
4. **Failure must not blow up the hot path**: disk exceptions are the
   caller's to handle; this module raises ValueError only for "usage
   errors" (overlong keys/values / unknown maps, etc.).
5. **Concurrent writes inside the lock** (see :mod:`sidecar_lock`): all
   read-modify-write goes through :func:`update`; calling :func:`save`
   directly only guarantees "not writing it broken", not against lost
   updates. Failing to get the lock raises :class:`SidecarBusy`, and the
   caller decides to wait or skip -- **never silently drop a write** (the
   real disease of the tmp+rename era).
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from ..common import sidecar_lock as sl
from .constants import MAPS_VERSION
from .file_registry import HASH_MAX_BYTES, _hash_file
from ..common.sidecar_lock import SidecarBusy, SidecarUnreadable

# Size guardrails (against pathological JSON: an entry is not an article)
MAX_KEY_LEN = 100        # entry key (CN2601 / chr1 / --use-avx2)
MAX_STATUS_LEN = 200     # status value
MAX_DESC_LEN = 200       # map description (relevance matching relies on it; one sentence)
MAX_NOTE_LEN = 500       # note
MAX_FIELDS = 20          # number of extra fields
MAX_FIELD_LEN = 200      # length of a single extra field value
MAX_MAPS = 200           # max maps per library (guards against misuse as a memory store)


# ── Read/write ────────────────────────────────────────────────────────

def maps_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "maps.json")


def load(persist_dir: str) -> dict | None:
    """Read the sidecar; missing/corrupt both return None (corrupt is treated as not enabled, never crashing the caller)."""
    try:
        with open(maps_path(persist_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("maps"), dict):
        return None
    return data


def _write_atomic(persist_dir: str, data: dict) -> None:
    """Atomic write (tmp + rename): concurrent multi-session map edits must
    not leave half a JSON.

    **No locking here** -- called by :func:`update` inside an already-held
    lock critical section, or by :func:`save` after taking the lock itself.
    Neither may use it directly without holding the lock.
    """
    p = maps_path(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".maps_", dir=persist_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save(persist_dir: str, data: dict, wait: float = 0.0) -> None:
    """Persist **already-in-hand** data to disk (atomic write + sidecar lock).

    Warning: the lock only guarantees "not writing it broken, two writes not
    interleaving"; it does **not** prevent lost updates: if the read happened
    outside the lock, others' changes in the meantime get overwritten by
    this whole-snapshot write. To prevent lost updates use :func:`update` --
    there, read and write share one critical section.

    This signature exists because ``data`` is often computed elsewhere (CLI
    rehearsal, test fixtures); forcing a callback would be roundabout.
    Failing to get the lock raises :class:`SidecarBusy`; a write is never
    silently dropped.
    """
    lock = sl.lock_path(maps_path(persist_dir))
    tok = sl.acquire(lock, wait=wait)
    if tok is None:
        raise SidecarBusy(lock, sl.holder(lock))
    try:
        _write_atomic(persist_dir, data)
    finally:
        sl.release(lock, tok)


def update(persist_dir: str, fn, create: bool = False,
           wait: float = 0.0, reset_unreadable: bool = False):
    """Do "read -> modify -> write" inside the sidecar lock, **the only
    correct way to prevent lost updates**.

    ``fn(data)`` receives the **latest** data inside the lock (not the
    caller's stale snapshot), mutates it in place, and returns
    ``(save: bool, result)``:

    - ``save=True`` => persist;
    - ``save=False`` => compute only, no write (e.g. the patrol interval has
      not elapsed, no recipe to run) -- this is the norm, not an exception;
    - ``create=True`` and the sidecar is absent => start one from
      :func:`empty`; ``create=False`` and the sidecar is absent => **do not
      call fn**, return ``(None, False)`` directly (opt-in: with the map
      layer not enabled there should be no write).

    Red alert: **"unreadable" is not "nonexistent"**: the file exists but
    ``load`` returns None (corrupt/exclusively held) => raise
    :class:`SidecarUnreadable`; never overwrite with a fresh empty sidecar.
    Only ``reset_unreadable=True`` (explicit rebuild like init --force)
    renames the bad file into quarantine and rebuilds.

    Returns ``(result, saved)``. Failing to get the lock raises
    :class:`SidecarBusy` -- the caller decides to wait (``wait``) / raise /
    skip this round; it must not be treated as a successful write.
    """
    lock = sl.lock_path(maps_path(persist_dir))
    tok = sl.acquire(lock, wait=wait)
    if tok is None:
        raise SidecarBusy(lock, sl.holder(lock))
    try:
        data = load(persist_dir)
        if data is None:
            p = maps_path(persist_dir)
            exists = os.path.isfile(p)
            if exists and not reset_unreadable:
                raise sl.SidecarUnreadable(p)
            if exists:
                sl.quarantine(p)          # explicit rebuild: quarantine, do not delete
            if not create:
                return None, False
            data = empty()
        save_flag, result = fn(data)
        if save_flag:
            _write_atomic(persist_dir, data)
        return result, bool(save_flag)
    finally:
        sl.release(lock, tok)


def empty() -> dict:
    return {"version": MAPS_VERSION, "maps": {}}


# ── Write (gate 1: the only entry) ──────────────────────────────────────

def _clip(s: object, n: int, what: str) -> str:
    s = "" if s is None else str(s)
    if len(s) > n:
        raise ValueError("%s too long (%d > %d chars): map entries are not articles; "
                         "put long content into memories instead" % (what, len(s), n))
    return s


def resolve_source(path: str, root: str | None = None) -> str:
    """truth_source resolution: absolute paths as-is; relative paths try the
    registration project root first (the root explicitly given to
    file_registry.init), then the process cwd.

    Returns the **first existing candidate**; when none exists, returns the
    primary candidate (the caller records no signature; verify treats it as
    missing once the file appears).
    """
    path = str(path).strip()
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    cands = []
    if root and os.path.isdir(root):
        cands.append(os.path.join(root, path))
    cands.append(os.path.abspath(path))
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def _record_source_sig(full: str) -> dict | None:
    """The truth_source's current signature (size/mtime_ns + content hash).

    File missing / too large to hash => None (treated as missing or as
    no-baseline at verify time). Same arbitration signature as file_registry
    (inherits this mechanism automatically).
    """
    try:
        if not os.path.isfile(full) or os.path.getsize(full) > HASH_MAX_BYTES:
            return None
        st = os.stat(full)
        return {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                "hash": _hash_file(full)}
    except OSError:
        return None


def set_entry(data: dict, name: str, key: str, status: str | None = None,
              fields: dict | None = None, desc: str | None = None,
              truth_source: str | None = None, note: str | None = None,
              updated_by: str = "", root: str | None = None) -> tuple[dict, dict, list[str]]:
    """Upsert one entry by key (gate 1's sole write entry). Returns (map,
    entry, notes).

    - A missing map is created (desc defaults to name -- desc is the sole
      basis for L1 relevance matching, see the design doc §3/§6);
    - Writing the same key again **replaces wholesale** the old value (old
      status vs new status competing is excluded at the data-model level);
    - ``truth_source``: non-empty => set it and record the current signature
      ("has the truth changed" is measured from the moment of set); None
      (not passed) => **keep** the original value and its signature baseline
      (a local status update must not casually drop the truth-source
      declaration); empty string => explicit clear;
    - Re-setting automatically clears the entry's drift (a new baseline was
      recorded);
    - notes are non-fatal warnings (e.g. truth_source not on disk) --
      returned to the host; the write is not interrupted.
    """
    name = _clip(name, MAX_DESC_LEN, "map name")
    key = _clip(key, MAX_KEY_LEN, "entry key")
    if not name or not key:
        raise ValueError("map name and entry key must not be empty")
    maps = data.setdefault("maps", {})
    if name not in maps and len(maps) >= MAX_MAPS:
        raise ValueError("map count reached the limit %d -- maps are not a memory store; clean up first"
                         % MAX_MAPS)
    now = time.time()
    m = maps.setdefault(name, {"desc": name, "created_at": now,
                               "updated_at": now, "entries": {}})
    if desc is not None:
        m["desc"] = _clip(desc, MAX_DESC_LEN, "map description")
    ent = m["entries"].get(key) or {}
    new_ent: dict = {
        "status": _clip(status if status is not None else ent.get("status", ""),
                        MAX_STATUS_LEN, "status value"),
        "fields": ent.get("fields") or {},
        "updated_at": now,
        "updated_by": _clip(updated_by, 50, "updated_by"),
        "truth_source": str(truth_source or ""),
        "source_sig": None,
        "drift": None,
        "note": _clip(note if note is not None else ent.get("note", ""),
                      MAX_NOTE_LEN, "note"),
    }
    if fields:
        if not isinstance(fields, dict):
            raise ValueError("fields must be a dict")
        if len(fields) > MAX_FIELDS:
            raise ValueError("more than %d extra fields" % MAX_FIELDS)
        new_ent["fields"] = {str(k): _clip(v, MAX_FIELD_LEN, "field value")
                             for k, v in fields.items()}
    notes: list[str] = []
    old_ts = str(ent.get("truth_source") or "")
    old_sig = ent.get("source_sig")
    if truth_source is None:
        new_ent["truth_source"] = old_ts          # not passed => keep
        new_ent["source_sig"] = old_sig          # baseline untouched (verify will
    elif str(truth_source).strip():               # expose any change in between truthfully)
        new_ent["truth_source"] = _clip(str(truth_source).strip(), 400,
                                        "truth_source")
        full = resolve_source(new_ent["truth_source"], root)
        sig = _record_source_sig(full)
        if sig is None:
            notes.append("truth_source missing from disk or too large (no recorded signature): %s"
                         % new_ent["truth_source"])
        new_ent["source_sig"] = sig
    # truth_source == "" (explicit empty string) => clear: source_sig stays None
    m["entries"][key] = new_ent
    m["updated_at"] = now
    return m, new_ent, notes


# ── Query ────────────────────────────────────────────────────────

def get_entry(data: dict, name: str, key: str) -> dict | None:
    m = (data.get("maps") or {}).get(str(name)) or {}
    return (m.get("entries") or {}).get(str(key))


def map_names(data: dict) -> list[str]:
    return sorted((data.get("maps") or {}).keys())


def ack_entry(data: dict, name: str, key: str,
              now: float | None = None) -> dict:
    """"I re-checked it; the status is unchanged": only advances ``updated_at``,
    leaving status/fields untouched.

    Gate 3's (mention-level drift) **only explicit clearing gesture** (the
    clearing rule 1 in ``map/mention.py``): the trigger is "a later memory
    says another status"; writing the confirmation moment into
    ``updated_at`` makes it no longer satisfied; also clears
    ``ent["mention"]``.

    Why not reuse ``set_entry``: rewriting would make a human copy
    status/fields again, and a copying mistake turns "confirmed unchanged"
    into "changed to something else" -- the place that most needs safety
    should not carry transcription risk. ``updated_by`` is also untouched
    (the status was not written by them; they only confirmed it); the
    confirmation trace persists separately in ``acked_at``.
    """
    now = time.time() if now is None else now
    m = (data.get("maps") or {}).get(str(name))
    if not m:
        raise ValueError("no such map: %s" % name)
    ents = m.get("entries") or {}
    if str(key) not in ents:
        raise ValueError("map %r has no entry %r" % (name, key))
    ent = ents[str(key)]
    ent["updated_at"] = now
    ent["acked_at"] = now
    ent["mention"] = None            # gate-3 mark cleared with the confirmation (new ones get recomputed)
    m["updated_at"] = now
    return ent


def list_maps(data: dict, now: float | None = None) -> list[dict]:
    """Library-wide map summary (for the L0 index lines and the map_list tool)."""
    now = time.time() if now is None else now
    out = []
    for name, m in (data.get("maps") or {}).items():
        ents = m.get("entries") or {}
        drifted = sum(1 for e in ents.values() if e.get("drift"))
        # In auto-verify's three states, "cannot tell" must also surface --
        # extract_failed is not drift (the value did not change; we refused
        # to guess), but it likewise must not be hidden as clean (design §8
        # C1 "failure is not consistency").
        failed = sum(1 for e in ents.values()
                     if (e or {}).get("auto_state", {}).get("state") == "failed")
        # The two directions of directory reconciliation. drift is counted
        # separately by reason -- "value changed" and "file gone" need
        # completely different handling; merging into one number says
        # nothing.
        dir_missing = sum(
            1 for e in ents.values()
            if ((e or {}).get("drift") or {}).get("reason") == "dir_missing")
        dir_conflict = sum(
            1 for e in ents.values()
            if ((e or {}).get("drift") or {}).get("reason") == "dir_conflict")
        auto_state = m.get("auto_state") or {}
        # Gate 3: entries mentioned with "another status" after an update.
        # It is a **hint**, not fact -- counted separately from drift (the
        # truth-source comparison result) and written separately on the page
        # (`·疑变N` vs `·漂移N`), never merged into one big number.
        mentioned = sum(1 for e in ents.values() if (e or {}).get("mention"))
        newest = max([float(e.get("updated_at") or 0.0)
                      for e in ents.values()] or
                     [float(m.get("updated_at") or 0.0)])
        out.append({
            "name": name,
            "desc": m.get("desc") or name,
            "entries": len(ents),
            "drifted": drifted,
            "failed": failed,
            "dir_missing": dir_missing,
            "dir_conflict": dir_conflict,
            # directory -> map direction (the path structurally unreachable via truth_source)
            "unmatched": int(auto_state.get("unmatched_total") or 0),
            "unmatched_keys": list(auto_state.get("unmatched") or [])[:5],
            "unmatched_complete": auto_state.get("unmatched_complete") is not False,
            "auto": bool(m.get("auto")),
            "auto_error": auto_state.get("error") or "",
            "auto_incomplete": bool(auto_state.get("incomplete")),
            "mentioned": mentioned,
            "mention_state": m.get("mention_state") or {},
            "newest_ts": newest,
            "age_days": int((now - newest) // 86400) if newest else None,
        })
    out.sort(key=lambda x: (-x["entries"], x["name"]))
    return out


# ── Verify (gate 2) ────────────────────────────────────────────────

def verify(data: dict, name: str | None = None,
           root: str | None = None) -> dict:
    """Re-read truth sources, comparing signatures entry by entry (the
    mechanized version of §4 gate 2).

    For each entry carrying a ``source_sig`` baseline:
      - stat signature unchanged => clean, zero reads;
      - mismatch => read the content hash once: same => pure touch, refresh
        the signature; different => ``drift = {"reason": "changed"}``;
      - file gone => ``drift = {"reason": "missing"}``;
      - content back at the baseline hash (changed and changed back /
        deleted and restored) => clear drift.

    Entries without truth_source are skipped (purely manual statuses fall
    back to gate 3/gate 4, design doc §4 coverage matrix). Returns counts +
    the drift list (the host learns which entries to re-check).
    """
    now = time.time()
    res = {"checked": 0, "clean": 0, "touched": 0, "skipped": 0,
           "drifted": []}
    maps = data.get("maps") or {}
    targets = [name] if name else sorted(maps.keys())
    for mname in targets:
        m = maps.get(mname)
        if not m:
            continue
        for key, ent in (m.get("entries") or {}).items():
            ts = ent.get("truth_source")
            sig = ent.get("source_sig")
            if not ts or not sig:
                res["skipped"] += 1
                continue
            res["checked"] += 1
            full = resolve_source(ts, root)
            try:
                if not os.path.isfile(full):
                    raise OSError("missing")
                st = os.stat(full)
                cur = (st.st_size, st.st_mtime_ns)
                base = (sig.get("size"), sig.get("mtime_ns"))
                if cur == base:
                    ent["drift"] = None
                    res["clean"] += 1
                    continue
                digest = _hash_file(full)
            except OSError:
                if not (ent.get("drift") or {}).get("reason") == "missing":
                    ent["drift"] = {"reason": "missing", "since": now}
                    res["drifted"].append({"map": mname, "key": key,
                                           "reason": "missing"})
                else:
                    ent["drift"]["since"] = now
                continue
            if digest is None or digest == sig.get("hash"):
                # Pure touch (git checkout / robocopy resetting mtime) or
                # content back at the baseline: refresh signature, clear drift
                sig["size"], sig["mtime_ns"] = st.st_size, st.st_mtime_ns
                if digest is not None:
                    sig["hash"] = digest
                was_drifted = bool(ent.get("drift"))
                ent["drift"] = None
                res["touched" if was_drifted or cur != base else "clean"] += 1
                continue
            # Content really changed: keep the old baseline hash (a change-back
            # must be recognizable), mark drift
            ent["drift"] = {"reason": "changed", "since": now}
            res["drifted"].append({"map": mname, "key": key,
                                   "reason": "changed",
                                   "truth_source": ts})
    return res
