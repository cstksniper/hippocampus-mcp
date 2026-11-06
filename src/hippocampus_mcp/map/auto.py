"""Fully automatic verify of code-switch maps (the information-map layer).

One sentence: previously verify required a human to remember to call it; auto verify
hooks it onto signals — only when a truth file's signature changes does it re-read
and re-compare, and the result still only lands in ``drift`` / ``auto_state``,
**no auto-fix** (changing the map is a semantic decision that still goes through
map_set).

Map-level recipe (the map's ``auto`` key in ``maps.json``; a single entry or a list
are both legal)::

    "auto": {
      "kind": "code_flag",
      "source": "CMakeLists.txt",   # relative to the file_registry root
      "extract": "cmake_option",    # extractor
      "extract_args": {},           # regex uses pattern / yaml_path uses path
      "normalize": {"ON": "开"}     # optional: config value → map wording
    }

Extractors (**static literals only**):

===============  ==========================================  ==================
extract          what it reads                               example
===============  ==========================================  ==================
cmake_option     ``option(NAME ON/OFF)`` / ``set(NAME ON)``   option(USE_AVX2 ON)
c_define         ``#define NAME 0/1`` / ``#define NAME``      config.h
regex            custom regex with named groups key/value    Makefile ``?= 1``
yaml_path        yaml/json path lookup (``{key}`` placeholder)  build.avx2
===============  ==========================================  ==================

The four "no guessing" disciplines (each one corresponds to a real false-clean scenario):

1. **Dynamic values are skipped**: variable interpolation ``${BASE}``, non-literal
   values, unknown status words — marked extract_failed, judged neither clean nor
   drift (judging clean is an under-report, judging drift is a false report; both
   are worse than "cannot say").
2. **When shadowed by a dynamic block, falling back to the static declaration is not
   allowed**: ``option(USE_AVX2 ON)`` followed by ``if(WIN32) set(USE_AVX2 OFF)
   endif()`` — taking only the static ON would report clean on Windows; the rule is:
   a name ever appeared inside a dynamic block ⇒ that entry is extract_failed.
   The staleness system's worst incident is not an under-report, it is a **false
   guarantee**.
3. **Same-name multiple declarations take the last unconditional declaration**
   (existing CMake/Make semantics); if it was ever shadowed by a dynamic block in
   between, the whole thing fails — the implementation may not freely pick
   first/last.
4. **The root is not guessed**: ``source`` resolves relative to the **file_registry
   root**; when there is no registered root, only an **absolute path** in
   ``auto.root`` is accepted, otherwise execution is refused with a note — the
   daemon's cwd is the startup directory, and falling back to cwd would resolve to a
   different CMakeLists.txt and produce a seemingly valid conclusion.

Hash arbitration: if the source's (size, mtime_ns) is unchanged ⇒ zero reads, reuse
the last conclusion (``res["reused"]``); the file is read and extracted only when
the signature changed or on first run. The total wall-clock budget defaults to 1s;
when exceeded, mark ``incomplete`` to self-report.

Directory reconciliation (same module)::

    "auto": {
      "kind": "dir_reconcile",
      "root": "data/",                 # reconciliation directory, relative to the registered root (default = project root)
      "key_pattern": "CN[0-9]+",       # fullmatch; with (?P<key>...) the group content is taken
      "status_map": {"训练集": "train"},  # optional: status → the subdirectory it should live in
      "file_ext": ".fastq"             # optional: count only this kind of file
    }

The fundamental difference from C1 is the **direction of information**: C1 answers
"did the value in the truth file change" (map→source); C2 adds a second channel,
"the directory holds things the map does not know about" (source→map) — the latter
is structurally unreachable by truth_source, which can only answer "did the truth
change and did the map keep up". Four output channels:

====================  ============================================  ==========
Direction             Criterion                                     Result
====================  ============================================  ==========
map→directory         the entry key has no corresponding directory/file under  dir_missing
                      the reconciliation directory
map→directory (w/ map)  the subdirectory the status should live in ≠ the     dir_conflict
                      subdirectory it actually sits in
directory→map         the directory has the key, the map lacks the entry      unmatched
recovery              the cause of either of the two cases disappeared        drift cleared
====================  ============================================  ==========

Three implementation disciplines:
1. **Zero repeated IO**: no disk scanning; set operations run directly on the
   already-registered ``files`` of file_registry (reconciliation only reads the
   sidecar; the single ``isdir`` on whether ``auto.root`` exists is the exception,
   used to distinguish "the directory was renamed" from "all entries are gone").
2. **Consuming registry output has preconditions**: without the file_registry
   sidecar, **refuse to execute** (no degraded live scan, no cwd fallback) — a
   degraded live scan would create a second directory snapshot inconsistent with
   the registry's account, and the two sets of answers would fight each other.
3. **When the registered scan is incomplete (``scan_complete=false``), do not judge
   missing/unregistered**: keys in the truncated region were never seen in the
   first place; judging missing is a pure false positive (the same discipline as
   file_registry itself). In that case the map is flagged ``incomplete`` to
   self-report, and entries land in failed rather than clean.
"""

from __future__ import annotations

import json
import os
import re
import time

from .constants import (
    MAP_AUTO_BUDGET_S,
    MAP_AUTO_MAX_SOURCE_BYTES,
)

# Normalization word lists for status and config values respectively (design §8 C1:
# only these 4+4 words count as certain; everything else is failed — judging clean
# from words like 「启用」 (enabled) / 「部分」 (partial) is guessing, and so is judging drift)
ON_WORDS = {"开", "on", "true", "1"}
OFF_WORDS = {"关", "off", "false", "0"}

# CMake literals the extractor accepts (case-insensitive); not in the table ⇒ non-literal ⇒ dynamic
_CMAKE_LITERALS = {"ON", "OFF", "TRUE", "FALSE", "1", "0", "YES", "NO"}
_CMAKE_OPEN = {"if", "foreach", "while", "function", "macro", "block"}
_CMAKE_CLOSE = {"endif", "endforeach", "endwhile", "endfunction", "endmacro",
                "endblock"}
_C_LITERALS = {"0", "1", "true", "false", "TRUE", "FALSE"}


# ── Normalization ──────────────────────────────────────────────────────

def norm_flag(value: object) -> str | None:
    """Normalize a word into ``"on"`` / ``"off"``; return None when unrecognized (the caller judges it failed)."""
    v = "" if value is None else str(value).strip().lower()
    if v in ON_WORDS:
        return "on"
    if v in OFF_WORDS:
        return "off"
    return None


def apply_normalize(value: str, normalize: dict | None) -> str:
    """The map-level ``normalize`` applies first to the **config-side** value (``{"ON": "开"}``).

    The direction is pinned by design §8 C1: config value → map wording; opposite to
    C2's status_map (status → directory name). A case-insensitive fallback avoids
    missing an ``"on"`` that was not configured.
    """
    s = "" if value is None else str(value)
    if not normalize:
        return s
    if s in normalize:
        return str(normalize[s])
    low = {str(k).strip().lower(): v for k, v in normalize.items()}
    return str(low.get(s.strip().lower(), s))


# ── Extractors (return (values, error, lookup)) ──────────────────────
#
# values: {name: (value|None, dynamic_reason|None)} — None+reason means
#         "this name appeared but cannot be pinned down", never treated as absent.
# lookup: used by extractors like yaml_path whose "path carries {key}", computed
#         per entry key on the fly.

def _record(vals: dict, name: str, value: str | None,
            reason: str | None) -> None:
    """Record one declaration: once a dynamic appears it is never washed away by later
    static values (discipline 2); for statics, **later writes overwrite earlier
    writes** — same-name multiple declarations take the last one (discipline 3)."""
    cur = vals.get(name)
    if cur is not None and cur[1]:
        return
    vals[name] = (None if reason else value, reason)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _cmake_strip_comment(line: str) -> str:
    out: list[str] = []
    quote = ""
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out)


def _cmake_statements(text: str):
    """Split into "command name + args": handles line continuations ``\\`` and multi-line parentheses (the norm inside if blocks)."""
    buf: list[str] = []
    for raw in text.splitlines():
        line = _cmake_strip_comment(raw).rstrip()
        if not line.strip() and not buf:
            continue
        if line.endswith("\\"):
            buf.append(line[:-1])
            continue
        buf.append(line)
        joined = "\n".join(buf)
        if joined.count("(") > joined.count(")"):
            continue
        stmt = " ".join(b.strip() for b in buf)
        buf = []
        if stmt.strip():
            yield stmt
    if buf:
        yield " ".join(b.strip() for b in buf)


def extract_cmake(text: str, args: dict) -> tuple[dict, str | None, None]:
    vals: dict = {}
    depth = 0
    for stmt in _cmake_statements(text):
        mt = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", stmt,
                      re.S)
        if not mt:
            continue
        cmd = mt.group(1).lower()
        inner = mt.group(2)
        if cmd in _CMAKE_CLOSE:
            depth = max(0, depth - 1)
            continue
        if cmd in _CMAKE_OPEN:
            depth += 1
            continue
        if cmd not in ("option", "set"):
            continue
        parts = [_strip_quotes(t) for t in re.findall(r'"[^"]*"|\S+', inner)]
        if not parts:
            continue
        name = parts[0]
        # option(NAME "help" ON)'s value is the last argument; set(NAME VALUE ...)'s
        # value is the second (CACHE BOOL "help" may follow)
        value = parts[-1] if cmd == "option" else (
            parts[1] if len(parts) > 1 else "")
        reason = None
        if depth > 0:
            reason = "inside an if/foreach conditional block (shadowed by a dynamic block)"
        elif not value:
            reason = "missing value"
        elif ("$" in value or "(" in value or "," in value
              or value.upper() not in _CMAKE_LITERALS):
            reason = "dynamic value %s" % value
        _record(vals, name, value, reason)
    return vals, None, None


def extract_c_define(text: str, args: dict) -> tuple[dict, str | None, None]:
    vals: dict = {}
    depth = 0
    for raw in text.splitlines():
        line = raw.strip()
        mt = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b", line)
        if mt:
            cmd = mt.group(1)
            if cmd in ("if", "ifdef", "ifndef"):
                depth += 1
            elif cmd == "endif":
                depth = max(0, depth - 1)
            continue
        mt = re.match(r"#\s*define\s+([A-Za-z_]\w*)(.*)$", line)
        if mt:
            name, rest = mt.group(1), mt.group(2)
            rest = rest.split("//")[0].split("/*")[0].strip()
            value = rest or "1"
            reason = None
            if depth > 0:
                reason = "inside an #if/#ifdef conditional block (platform branches are not guessed)"
            elif value not in _C_LITERALS:
                reason = "non-literal value %s" % value
            _record(vals, name, value, reason)
            continue
        mt = re.match(r"#\s*undef\s+([A-Za-z_]\w*)", line)
        if mt:
            _record(vals, mt.group(1), None, "state is indeterminate after #undef")
    return vals, None, None


def extract_regex(text: str, args: dict) -> tuple[dict, str | None, None]:
    pattern = str((args or {}).get("pattern") or "")
    if not pattern:
        return {}, "regex extractor is missing extract_args.pattern", None
    try:
        rx = re.compile(pattern, re.M)
    except re.error as exc:
        return {}, "invalid regex: %s" % exc, None
    if "key" not in rx.groupindex:
        return {}, "regex is missing the named group (?P<key>...)", None
    vals: dict = {}
    for mt in rx.finditer(text):
        key = mt.group("key")
        if key is None:
            continue
        value = mt.groupdict().get("value")
        reason = None
        if value is None:
            reason = "regex provides no (?P<value>...) group"
        elif "$" in str(value):
            reason = "dynamic value %s" % value
        _record(vals, str(key), None if value is None else str(value), reason)
    return vals, None, None


def _load_struct(text: str) -> tuple[object, str | None]:
    """Parse as JSON first; if not JSON, try PyYAML (this repo adds no new dependencies — report when absent)."""
    try:
        return json.loads(text), None
    except ValueError:
        pass
    try:
        import yaml  # type: ignore
    except ImportError:
        return None, ("yaml parser unavailable (no new dependencies in this repo) -- convert the config to JSON, "
                      "or switch to the regex extractor")
    try:
        return yaml.safe_load(text), None
    except Exception as exc:  # noqa: BLE001 —— third-party parser exception types are unpredictable
        return None, "yaml parse failed: %s" % exc


def extract_yaml_path(text: str, args: dict):
    path = str((args or {}).get("path") or "")
    if not path:
        return {}, "yaml_path extractor is missing extract_args.path", None
    doc, err = _load_struct(text)
    if err:
        return {}, err, None

    def lookup(key: str) -> tuple[str | None, str | None]:
        cur = doc
        for seg in path.replace("{key}", str(key)).split("."):
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
            else:
                return None, None          # path does not exist ⇒ the key "did not appear"
        if isinstance(cur, (dict, list)):
            return None, "path %s is a container, not a flag value" % path
        if cur is None:
            return None, "path %s is empty" % path
        if "$" in str(cur):
            return None, "dynamic value %s" % cur
        return str(cur), None

    return {}, None, lookup


EXTRACTORS = {
    "cmake_option": extract_cmake,
    "c_define": extract_c_define,
    "regex": extract_regex,
    "yaml_path": extract_yaml_path,
}

# Two map-level recipe kinds: C1 value-level comparison (reads truth files) / C2 directory reconciliation (reads registry output)
KINDS = ("code_flag", "dir_reconcile")

# drift reasons written by directory reconciliation itself (cleanup clears only these
# two; never touch the hand-written map's changed/missing or the source verification's
# value_changed — all three channels share the single ent["drift"] channel)
RECONCILE_REASONS = ("dir_missing", "dir_conflict")


def _kind(rule: dict) -> str:
    """Recipe kind (defaults to code_flag, the same default as validate_rule)."""
    return str((rule or {}).get("kind") or "code_flag").strip()


# ── Config validation (used when the CLI writes auto) ────────────────────────────────

def _validate_reconcile(out: dict) -> dict:
    """C2 recipe validation (usage errors raise ValueError — a mis-configuration is a fake clean)."""
    for dead in ("source", "extract", "extract_args", "normalize"):
        if out.get(dead):
            raise ValueError("dir_reconcile does not need %s (reconciliation consumes file_registry "
                             "output, not truth files) -- remove it, or "
                             "switch kind back to code_flag" % dead)
        out.pop(dead, None)
    pat = str(out.get("key_pattern") or "").strip()
    if not pat:
        raise ValueError("dir_reconcile requires key_pattern (extracts the entry key from directory/file "
                         "names); for full-string equality use ^CN[0-9]+$, to extract use "
                         "^(?P<key>CN[0-9]+)\\.fastq$")
    try:
        re.compile(pat)
    except re.error as exc:
        raise ValueError("key_pattern is not a valid regex: %s" % exc)
    out["key_pattern"] = pat
    root = str(out.get("root") or "").strip()
    out["root"] = root if os.path.isabs(root) else _norm_rel(root)
    sm = out.get("status_map")
    if sm is not None:
        if not isinstance(sm, dict):
            raise ValueError("status_map must be a JSON object (status -> directory name)")
        for k, v in sm.items():
            if isinstance(v, (dict, list)) or v is None:
                raise ValueError("status_map entry %r is not a directory name (subdirectories are "
                                 "reconciled, not objects)" % k)
        out["status_map"] = {str(k): str(v) for k, v in sm.items()}
    ext = str(out.get("file_ext") or "").strip()
    if ext:
        out["file_ext"] = ext if ext.startswith(".") else "." + ext
    else:
        out.pop("file_ext", None)
    return out


def validate_rule(rule: dict) -> dict:
    """Validate one auto recipe and complete it with defaults; usage errors raise ValueError."""
    if not isinstance(rule, dict):
        raise ValueError("the auto recipe must be a JSON object")
    out = dict(rule)
    kind = _kind(out)
    if kind not in KINDS:
        raise ValueError("unsupported auto.kind %r -- available: %s"
                         % (kind, "、".join(KINDS)))
    out["kind"] = kind
    if kind == "dir_reconcile":
        return _validate_reconcile(out)
    source = str(out.get("source") or "").strip()
    if not source:
        raise ValueError("auto.source must not be empty (truth-file path relative to the "
                         "file_registry root)")
    out["source"] = source
    extract = str(out.get("extract") or "").strip()
    if extract not in EXTRACTORS:
        raise ValueError("unknown auto.extract %r -- available: %s"
                         % (extract, "、".join(EXTRACTORS)))
    out["extract"] = extract
    out.setdefault("extract_args", {})
    if not isinstance(out["extract_args"], dict):
        raise ValueError("auto.extract_args must be a JSON object")
    if extract == "regex" and not out["extract_args"].get("pattern"):
        raise ValueError("the regex extractor requires extract_args.pattern")
    if extract == "yaml_path" and not out["extract_args"].get("path"):
        raise ValueError("the yaml_path extractor requires extract_args.path")
    norm = out.get("normalize")
    if norm is not None and not isinstance(norm, dict):
        raise ValueError("auto.normalize must be a JSON object (config value -> map wording)")
    return out


def _rules(auto: object) -> list[dict]:
    """``auto`` single entry or list ⇒ list of rules (dirty data without kind is ignored outright)."""
    if isinstance(auto, dict):
        return [auto] if auto.get("kind") else []
    if isinstance(auto, list):
        return [r for r in auto if isinstance(r, dict) and r.get("kind")]
    return []


# ── Main flow ──────────────────────────────────────────────────────

def _base_root(rule: dict, registry_root: str | None) -> str | None:
    """Root resolution: registered root > ``auto.root`` (must be absolute) > None (= no guessing, refuse to execute)."""
    if registry_root:
        return registry_root
    r = str(rule.get("root") or "").strip()
    if r and os.path.isabs(r):
        return r
    return None


def _stat_sig(full: str) -> dict | None:
    try:
        if not os.path.isfile(full):
            return None
        st = os.stat(full)
    except OSError:
        return None
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _read_text(full: str) -> tuple[str | None, str | None]:
    try:
        if os.path.getsize(full) > MAP_AUTO_MAX_SOURCE_BYTES:
            return None, ("source file exceeds the %d-byte limit (a build script should not be this large)"
                          % MAP_AUTO_MAX_SOURCE_BYTES)
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), None
    except OSError as exc:
        return None, "cannot read the source file: %s" % exc


def _mark_failed(ent: dict, now: float, reason: str, **extra) -> None:
    state = {"state": "failed", "reason": reason, "at": now}
    state.update(extra)
    ent["auto_state"] = state


def _verify_flags(mname: str, m: dict, rules: list[dict],
                  registry_root: str | None, res: dict, now: float,
                  force: bool, deadline: float, state: dict) -> None:
    """C1: compare the config values in the truth file entry by entry (state is created by the caller and written back onto the map)."""
    bases = {}
    for rule in rules:
        base = _base_root(rule, registry_root)
        if base is None:
            state["error"] = ("no root to anchor on; not verifying (no registered root and "
                              "auto.root is not absolute)")
            state["at"] = now
            res["no_root"].append(mname)
            return
        bases[str(rule.get("source") or "")] = base

    sigs: dict = {}
    src_err: dict[str, str] = {}
    for rule in rules:
        src = str(rule.get("source") or "")
        sig = _stat_sig(os.path.join(bases[src], src))
        sigs[src] = sig
        if sig is None:
            src_err[src] = "source file does not exist"

    # Hash-arbitration fast path: source signature unchanged ⇒ zero reads, reuse the last conclusion (same discipline)
    prev = state.get("source_sigs")
    if (not force) and prev and prev == sigs and state.get("checked"):
        res["reused"] += 1
        return

    extracted = []
    for rule in rules:
        src = str(rule.get("source") or "")
        if src in src_err:
            extracted.append((rule, {}, src_err[src], None))
            continue
        text, err = _read_text(os.path.join(bases[src], src))
        if err:
            src_err[src] = err
            extracted.append((rule, {}, err, None))
            continue
        fn = EXTRACTORS.get(str(rule.get("extract") or ""))
        values, error, lookup = fn(text, rule.get("extract_args") or {})
        if error:
            src_err[src] = error
        extracted.append((rule, values, error, lookup))

    checked = clean = drifted = failed = 0
    for key, ent in (m.get("entries") or {}).items():
        if not isinstance(ent, dict):
            continue
        # Check the budget per entry (not per map): one map may hold hundreds of switches,
        # and the entry loop is not free either; when exceeded, stop in place — the ones
        # already checked count, the rest are left for the next round
        if time.perf_counter() > deadline:
            res["incomplete"] = True
            break
        checked += 1
        expect = norm_flag(ent.get("status"))
        if expect is None:
            failed += 1
            reason = ("status %r is not in {on,true,1,off,false,0} "
                      "-- not guessing" % (ent.get("status") or "empty"))
            _mark_failed(ent, now, reason)
            res["failed"].append({"map": mname, "key": key, "reason": reason})
            continue

        got_raw = got_norm = src_used = None
        fail_reason = None
        for rule, values, error, lookup in extracted:
            if error:
                continue
            if lookup is not None:
                value, why = lookup(key)
            else:
                value, why = values.get(key, (None, None))
            if why is not None:
                fail_reason = "%s（%s）" % (why, rule.get("source"))
                break
            if value is None:
                continue
            got_raw = value
            src_used = str(rule.get("source") or "")
            got_norm = norm_flag(apply_normalize(value, rule.get("normalize")))
            if got_norm is None:
                fail_reason = ("config value %r cannot be normalized to on/off (%s)"
                               % (value, src_used))
            break

        if fail_reason:
            failed += 1
            _mark_failed(ent, now, fail_reason)
            res["failed"].append({"map": mname, "key": key,
                                  "reason": fail_reason})
            continue
        if got_raw is None:
            failed += 1
            errs = [e for _, _, e, _ in extracted if e]
            if errs and len(errs) == len(extracted):
                # All sources failed to read: report the source errors themselves, more useful than "not found"
                reason = "；".join(dict.fromkeys(errs))
            else:
                reason = ("the flag was not found in the source (%s)"
                          % ("、".join(str(r.get("source") or "")
                                      for r, _, _, _ in extracted)
                             or "no available source"))
            _mark_failed(ent, now, reason)
            res["failed"].append({"map": mname, "key": key, "reason": reason})
            continue

        if got_norm == expect:
            clean += 1
            ent["auto_state"] = {"state": "clean", "observed": got_raw,
                                 "source": src_used, "at": now}
            # only clear the drift we wrote ourselves (truth_source's changed/missing is handled by the base patrol)
            if (ent.get("drift") or {}).get("reason") == "value_changed":
                ent["drift"] = None
            continue

        drifted += 1
        since = now
        if (ent.get("drift") or {}).get("reason") == "value_changed":
            since = (ent.get("drift") or {}).get("since") or now   # 只进必退: since is set on first entry and not reset while the condition persists
        ent["drift"] = {"reason": "value_changed", "since": since,
                        "observed": got_raw, "expect": ent.get("status"),
                        "source": src_used}
        ent["auto_state"] = {"state": "drifted", "observed": got_raw,
                             "expect": ent.get("status"),
                             "source": src_used, "at": now}
        res["drifted"].append({"map": mname, "key": key,
                               "expect": ent.get("status"),
                               "observed": got_raw, "source": src_used})

    state.update({"source_sigs": sigs, "at": now, "checked": checked,
                  "clean": clean, "drifted": drifted, "failed": failed,
                  "source_errors": src_err,
                  # exceeding the budget also counts as "this round did not finish checking" —
                  # without the flag, neither the index row nor the session-end sees it,
                  # silently treating a truncated round as all-clean
                  "incomplete": bool(src_err) or bool(res.get("incomplete"))})
    if src_err:
        # some sources failed to parse ⇒ mark the whole map incomplete (design §8 spec
        # revision 2), not "partially passed" — a map reporting clean must have read
        # every source
        state["error"] = "；".join("%s: %s" % (k, v) for k, v in src_err.items())
    else:
        state.pop("error", None)
    res["checked"] += checked
    res["clean"] += clean


# ── C2: data directory reconciliation ────────────────────────────────────────────

def _add_error(state: dict, msg: str) -> None:
    """Errors on the map are **accumulative** (source errors / not registered / the
    reconciliation directory does not exist): later arrivals must not wipe out
    earlier ones, otherwise the session-end only sees the last one."""
    old = str(state.get("error") or "")
    state["error"] = msg if (not old or msg in old) else "%s；%s" % (old, msg)


def _norm_rel(p: object) -> str:
    """Normalize a relative path: ``\\``→``/``, strip ``./`` and leading/trailing slashes; root directory → ``""``."""
    s = str(p or "").strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    s = s.strip("/")
    return "" if s in ("", ".") else s


def _match_key(rx: re.Pattern, name: str) -> str | None:
    """Extract a key from a single path segment: with the named group ``(?P<key>...)`` take the group content, otherwise take the whole string.

    **Always fullmatch** (design §8 spec revision 6): ``^CN[0-9]+`` matching
    ``CN2601_old`` must not extract CN2601 (that would make a deleted contract show
    as "still there", an under-reported missing); to extract, write a named group;
    to mean equality, use the whole string — there is no third semantic.
    """
    mt = rx.fullmatch(name)
    if not mt:
        return None
    if "key" in rx.groupindex:
        k = (mt.group("key") or "").strip()
        return k or None
    return name.strip() or None


def _scan_keys(files: dict, prefix: str, rx: re.Pattern,
               file_ext: str | None) -> dict[str, set[str]]:
    """Extract keys over the **already-registered output** (zero disk IO): ``{key: {subdirectories it sits in}}``.

    Both directory keys and file keys are recognized: in ``训练集/CN2601/raw/a.fastq``
    CN2601 is a directory key; in ``训练集/CN2601.fastq`` it is a file key.
    "Subdirectory it sits in" = the parent directory of the key entity (relative to
    the reconciliation directory), for status_map to judge "which subdirectory it
    should live in".
    """
    found: dict[str, set[str]] = {}
    pre = prefix + "/" if prefix else ""
    for rel, ent in (files or {}).items():
        if not isinstance(ent, dict) or ent.get("status") == "missing":
            continue              # the registry already judged it missing ⇒ truly absent on disk
        r = str(rel).replace("\\", "/")
        if pre:
            if not r.startswith(pre):
                continue
            sub = r[len(pre):]
        else:
            sub = r
        if not sub:
            continue
        segs = sub.split("/")
        base = segs[-1]
        if file_ext:
            if not base.lower().endswith(file_ext):
                continue          # count only this kind of file
            base = base[:len(base) - len(file_ext)]
        # every intermediate segment is a "directory key" candidate (a directory key's subdirectory = its parent directory)
        for i, seg in enumerate(segs[:-1]):
            k = _match_key(rx, seg)
            if k:
                found.setdefault(k, set()).add("/".join(segs[:i]))
        k = _match_key(rx, base)
        if k:
            found.setdefault(k, set()).add("/".join(segs[:-1]))
    return found


def _fmt_locs(locs: set[str]) -> str:
    items = sorted("(repo root)" if not x else x for x in locs)
    tail = " and %d more location(s)" % len(items) if len(items) > 3 else ""
    return "、".join(items[:3]) + tail


def _reconcile_root(rule: dict, base: str) -> tuple[str, str | None]:
    """Reconciliation-directory resolution: ``auto.root`` is relative to the **registered root**; absolute paths must sit under it."""
    raw = str(rule.get("root") or "").strip()
    if not raw:
        return "", None
    if os.path.isabs(raw):
        rel = os.path.relpath(raw, base).replace("\\", "/")
        if rel == ".." or rel.startswith("../"):
            return "", ("auto.root is outside the registered root (%s) -- directory reconciliation only "
                        "trusts the subtree under the registered root (re-root file_registry first to scan elsewhere)"
                        % raw)
        return _norm_rel(rel), None
    return _norm_rel(raw), None


def _verify_reconcile(mname: str, m: dict, rules: list[dict],
                      registry_root: str | None, res: dict, now: float,
                      deadline: float, state: dict,
                      registry: dict | None) -> None:
    """C2: two-way reconciliation between map entries ↔ data directories (reads only registry output, produces only signals).

    The four output channels are in the module docstring's table; the core judgment
    order is **"is it there" first, "where is it" second** — when the file is not
    there at all, talking about subdirectories is meaningless; and when status_map is
    missing, only the former is done (safe degradation, design §8 spec revision 7).
    """
    # Only the **registered sidecar itself** counts: the bare registry_root string is not
    # enough (without files there is no reconciliation data source). If missing, refuse
    # to execute — no degraded live scan, no cwd fallback.
    if not isinstance(registry, dict) \
            or not isinstance(registry.get("files"), dict):
        _add_error(state, "not registered (no file_registry sidecar), not verifying -- "
                          "directory reconciliation only consumes registry output, no live-scan fallback")
        state["at"] = now
        res["no_registry"].append(mname)
        return
    base = str(registry.get("root") or registry_root or "")
    if not base or not os.path.isdir(base):
        _add_error(state, "registered root does not exist or was never recorded (%s); not verifying"
                   % (base or "empty"))
        state["at"] = now
        res["no_registry"].append(mname)
        return
    files = (registry or {}).get("files") or {}
    complete = bool((registry or {}).get("scan_complete"))
    merged: dict[str, set[str]] = {}
    status_map: dict[str, str] = {}
    roots: list[str] = []
    ok_rules = 0
    for rule in rules:
        prefix, err = _reconcile_root(rule, base)
        rx = None
        if not err:
            pat = str(rule.get("key_pattern") or "").strip()
            if not pat:
                err = "dir_reconcile is missing key_pattern (no guessing; refusing to execute)"
            else:
                try:
                    rx = re.compile(pat)
                except re.error as exc:
                    err = "key_pattern is not a valid regex: %s" % exc
        if err:
            _add_error(state, err)
            res["bad_root"].append({"map": mname, "error": err})
            continue
        target = os.path.join(base, prefix) if prefix else base
        if not os.path.isdir(target):
            # The whole reconciliation subtree is absent: report "the reconciliation
            # directory does not exist" instead of letting every entry go missing —
            # the latter would narrate one rename as N data losses (a false alarm
            # erodes trust more than an under-report)
            err = "reconciliation directory does not exist: %s (relative to registered root %s)" % (prefix or ".", base)
            _add_error(state, err)
            res["bad_root"].append({"map": mname, "error": err})
            continue
        ok_rules += 1
        roots.append(prefix or "(project root)")
        for k, locs in _scan_keys(files, prefix, rx,
                                  rule.get("file_ext") or None).items():
            merged.setdefault(k, set()).update(locs)
        for st, d in (rule.get("status_map") or {}).items():
            status_map[str(st)] = _norm_rel(d)
    if not ok_rules:
        state["at"] = now
        return

    where = "、".join(roots)
    checked = clean = drifted = failed = 0
    truncated = False
    entries = {k: v for k, v in (m.get("entries") or {}).items()
               if isinstance(v, dict)}
    for key, ent in entries.items():
        if time.perf_counter() > deadline:
            res["incomplete"] = True
            truncated = True
            break
        checked += 1
        locs = merged.get(str(key))
        if not locs:
            if not complete:
                # keys in the truncated region were never seen in the first place: judging missing is a pure false positive
                failed += 1
                reason = ("registered scan incomplete (scan_complete=false) -- not scanned "
                          "does not mean absent; missing-entries are not judged")
                _mark_failed(ent, now, reason)
                res["failed"].append({"map": mname, "key": key,
                                      "reason": reason})
                continue
            drifted += 1
            prev = ent.get("drift") or {}
            since = ((prev.get("since") or now)
                     if prev.get("reason") == "dir_missing" else now)
            ent["drift"] = {"reason": "dir_missing", "since": since,
                            "observed": "not found under %s" % where,
                            "expect": ent.get("status") or "",
                            "source": "file_registry"}
            ent["auto_state"] = {"state": "drifted",
                                 "observed": "not under %s" % where,
                                 "expect": ent.get("status") or "",
                                 "source": "file_registry", "at": now}
            res["drifted"].append({"map": mname, "key": key,
                                   "reason": "dir_missing",
                                   "expect": ent.get("status") or "",
                                   "observed": "not under %s" % where,
                                   "source": "file_registry"})
            continue

        status = str(ent.get("status") or "").strip()
        expect_dir = None
        if status_map:
            expect_dir = status_map.get(status)
            if expect_dir is None:
                low = {k.strip().lower(): v for k, v in status_map.items()}
                expect_dir = low.get(status.lower())
            if expect_dir is None:
                failed += 1
                reason = ("status %r is not in status_map (%s) -- not guessing "
                          "which subdirectory it belongs to" % (status or "empty",
                                              ", ".join(status_map)))
                _mark_failed(ent, now, reason)
                res["failed"].append({"map": mname, "key": key,
                                      "reason": reason})
                continue

        ok = (expect_dir is None) or (locs == {expect_dir})
        if ok:
            clean += 1
            ent["auto_state"] = {
                "state": "clean",
                "observed": _fmt_locs(locs),
                "expect": status,
                "source": "file_registry", "at": now}
            # only clear the drift we wrote ourselves (the hand-written map's changed/missing and the source's value_changed are untouched)
            if (ent.get("drift") or {}).get("reason") in RECONCILE_REASONS:
                ent["drift"] = None
            continue

        drifted += 1
        prev = ent.get("drift") or {}
        since = ((prev.get("since") or now)
                 if prev.get("reason") == "dir_conflict" else now)
        ent["drift"] = {"reason": "dir_conflict", "since": since,
                        "observed": _fmt_locs(locs),
                        "expect": expect_dir or "(repo root)",
                        "source": "file_registry"}
        ent["auto_state"] = {"state": "drifted",
                             "observed": _fmt_locs(locs),
                             "expect": expect_dir or "(repo root)",
                             "source": "file_registry", "at": now}
        res["drifted"].append({"map": mname, "key": key,
                               "reason": "dir_conflict",
                               "expect": expect_dir or "(repo root)",
                               "observed": _fmt_locs(locs),
                               "source": "file_registry"})

    # Counts use **independent keys** (rec_*): C1's fast path reuses the last conclusion
    # for the whole round without touching counters; with both counter sets sharing the
    # same keys, a reused round would add the previous round's numbers again (inflated).
    # Entry-level conclusions are already written into each ent["auto_state"]; this is
    # just the summary for the CLI/diagnostics.
    state.update({"rec_checked": checked, "rec_clean": clean,
                  "rec_drifted": drifted, "rec_failed": failed})
    state["at"] = now
    res["checked"] += checked
    res["clean"] += clean

    # directory→map: the extra keys (the direction truth_source structurally cannot see).
    # When the scan is incomplete / truncated by budget, **do not write** this set of
    # numbers (rather no numbers than half numbers), and set unmatched_complete to False
    # so the read side knows the previous round's numbers are history.
    if complete and not truncated:
        extra = sorted(k for k in merged if k not in entries)
        state["unmatched"] = extra[:20]
        state["unmatched_total"] = len(extra)
        state["unmatched_complete"] = True
        for k in extra[:50]:
            res["unmatched"].append({"map": mname, "key": k,
                                     "where": _fmt_locs(merged[k])})
        res["unmatched_total"] += len(extra)
    else:
        state["unmatched"] = []
        state["unmatched_total"] = 0
        state["unmatched_complete"] = False
        # The whole round's conclusion covers only "the part already scanned": self-report
        # at the run level too, otherwise a caller seeing res["incomplete"]=False will
        # treat it as a complete round
        res["incomplete"] = True
        state["incomplete"] = True
        if not complete:
            _add_error(state, "registered scan incomplete (scan_complete=false) -- "
                              "this round skips missing/absent judgments and only checks what was scanned")


def _verify_map(mname: str, m: dict, rules: list[dict],
                registry_root: str | None, res: dict, now: float,
                force: bool, deadline: float,
                registry: dict | None = None) -> None:
    """Dispatch by recipe kind: C1 value-level comparison / C2 directory reconciliation, sharing the same ``auto_state``."""
    prev_state = m.get("auto_state")
    state: dict = dict(prev_state) if isinstance(prev_state, dict) else {}
    m["auto_state"] = state
    flag_rules = [r for r in rules if _kind(r) == "code_flag"]
    rec_rules = [r for r in rules if _kind(r) == "dir_reconcile"]
    if flag_rules:
        _verify_flags(mname, m, flag_rules, registry_root, res, now, force,
                      deadline, state)
    if rec_rules:
        _verify_reconcile(mname, m, rec_rules, registry_root, res, now,
                          deadline, state, registry)


def needs_registry(data: dict) -> bool:
    """Whether there is a dir_reconcile recipe (decides whether the file_registry sidecar needs reading).

    The brain side uses it to avoid pointlessly reading a multi-MB registry JSON for
    C1-only libraries.
    """
    for m in (data.get("maps") or {}).values():
        for rule in _rules((m or {}).get("auto")):
            if _kind(rule) == "dir_reconcile":
                return True
    return False


def run(data: dict, registry_root: str | None = None,
        budget_s: float | None = None, force: bool = False,
        now: float | None = None, registry: dict | None = None) -> dict:
    """Run over every map carrying an ``auto`` recipe. **Only modifies data (in memory)**; persisting to disk is the caller's job.

    Returns counts + the three-state lists; ``incomplete`` means the budget was
    exceeded or the registered scan is incomplete (no silent degradation to
    all-clean). ``registry`` is needed only when some map has a dir_reconcile recipe
    (pass the file_registry sidecar dict; C2's zero repeated IO depends entirely on
    it).
    """
    now = time.time() if now is None else now
    budget = MAP_AUTO_BUDGET_S if budget_s is None else float(budget_s)
    t0 = time.perf_counter()
    deadline = t0 + budget
    res: dict = {"maps": 0, "checked": 0, "clean": 0, "drifted": [],
                 "failed": [], "reused": 0, "no_root": [], "no_registry": [],
                 "bad_root": [], "unmatched": [], "unmatched_total": 0,
                 "incomplete": False, "budget_s": budget, "at": now}
    for mname in sorted(data.get("maps") or {}):
        m = data["maps"].get(mname) or {}
        rules = _rules(m.get("auto"))
        if not rules:
            continue
        res["maps"] += 1
        if time.perf_counter() > deadline:
            res["incomplete"] = True
            break
        _verify_map(mname, m, rules, registry_root, res, now, force, deadline,
                    registry)
    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res
