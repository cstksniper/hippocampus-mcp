"""代码开关图全自动 verify（信息地图层）。

一句话：此前的 verify 要人记得调，auto verify 把它挂到信号上——真相文件
签名变了才重读、重比，结果仍只落 ``drift`` / ``auto_state``，**不 auto-fix**
（改地图是有语义的决定，仍走 map_set）。

图级配方（``maps.json`` 里那张图的 ``auto`` 键；单条或列表都合法）::

    "auto": {
      "kind": "code_flag",
      "source": "CMakeLists.txt",   # 相对 file_registry root
      "extract": "cmake_option",    # 提取器
      "extract_args": {},           # regex 用 pattern / yaml_path 用 path
      "normalize": {"ON": "开"}     # 可选：配置值 → 地图用词
    }

提取器（**只认静态字面量**）：

===============  ==========================================  ==================
extract          读什么                                      例子
===============  ==========================================  ==================
cmake_option     ``option(NAME ON/OFF)`` / ``set(NAME ON)``   option(USE_AVX2 ON)
c_define         ``#define NAME 0/1`` / ``#define NAME``      config.h
regex            自定义正则，命名组 key/value                 Makefile ``?= 1``
yaml_path        yaml/json 路径取值（``{key}`` 占位）          build.avx2
===============  ==========================================  ==================

四条「不猜」纪律（每条都对应一个真实的假 clean 场景）：

1. **动态值跳过**：变量插值 ``${BASE}``、非字面量值、未知 status 词——
   标 extract_failed，既不判 clean 也不判 drift（判 clean 是漏报、判
   drift 是假报，两种都比「说不清」更糟）。
2. **被动态块遮蔽不许退回静态声明**：``option(USE_AVX2 ON)`` 后面跟着
   ``if(WIN32) set(USE_AVX2 OFF) endif()``，只取静态 ON 会在 Windows 上
   报 clean；规则是某名字在动态块内出现过 ⇒ 该条目 extract_failed。
   staleness 系统最重的事故不是漏报，是**假保证**。
3. **同名多声明取最后一个无条件声明**（CMake/Make 既有语义）；中途被
   动态块遮蔽过就整体 failed，不许实现随意选 first/last。
4. **根不猜**：``source`` 相对 **file_registry root** 解析；没建档 root
   时只接受 ``auto.root`` 的**绝对路径**，否则拒绝执行并 note——daemon
   的 cwd 是启动目录，回落 cwd 会解析到另一个 CMakeLists.txt 并给出
   貌似合法的结论。

哈希仲裁：source 的 (size, mtime_ns) 没变 ⇒ 零读取直接用上次
结论（``res["reused"]``）；签名变或首次才读文件提取。总墙钟预算默认 1s，
超了标 ``incomplete`` 自曝。

目录对账（同一个模块）::

    "auto": {
      "kind": "dir_reconcile",
      "root": "data/",                 # 对账目录，相对建档 root（默认=项目根）
      "key_pattern": "CN[0-9]+",       # fullmatch；有 (?P<key>...) 取组内容
      "status_map": {"训练集": "train"},  # 可选：status → 应住的子目录
      "file_ext": ".fastq"             # 可选：只数这类文件
    }

与 C1 的根本差别是**信息方向**：C1 答「真相文件里的值变了没」（地图→源），
C2 多了一路「目录里多了地图不知道的东西」（源→地图）——后者 truth_source
结构上抓不到，因为它只能答「真相变了地图跟上没」。四路输出：

====================  ============================================  ==========
方向                  判据                                          结果
====================  ============================================  ==========
地图→目录              条目键在对账目录下没有任何对应目录/文件        dir_missing
地图→目录（配了 map）  status 应住的子目录 ≠ 实际所在子目录          dir_conflict
目录→地图              目录里有键、地图没这条                        unmatched
恢复                   上两种情况的原因消失                          drift 清空
====================  ============================================  ==========

三条实现纪律：
1. **零重复 IO**：不扫盘，直接在 file_registry 已建档的 ``files`` 上做集合
   运算（对账只读侧车；``auto.root`` 是否存在的``isdir`` 那一次除外，用来
   区分「目录被改名」与「条目全丢」）。
2. **消费建档产物有前提**：没有 file_registry 侧车就**拒绝执行**（不降级
   现扫、不回落 cwd）——降级现扫会造出与建档口径不同的第二份目录快照，
   两套答案会打架。
3. **建档扫描不完整（``scan_complete=false``）时不判缺失/漏登**：截断区域
   里的键本来就没被看到，判 missing 是纯误报（与 file_registry 自己同一
   纪律）。此时图标 ``incomplete`` 自曝，条目落 failed 而不是 clean。
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

# status 与配置值各自的归一化词表（设计 §8 C1：只有这 4+4 个词算确定，
# 其余一律 failed——「启用」「部分」这类词判 clean 是猜，判 drift 也是猜）
ON_WORDS = {"开", "on", "true", "1"}
OFF_WORDS = {"关", "off", "false", "0"}

# 提取器认的 cmake 字面量（大小写不敏感）；不在表里 ⇒ 非字面量 ⇒ 动态
_CMAKE_LITERALS = {"ON", "OFF", "TRUE", "FALSE", "1", "0", "YES", "NO"}
_CMAKE_OPEN = {"if", "foreach", "while", "function", "macro", "block"}
_CMAKE_CLOSE = {"endif", "endforeach", "endwhile", "endfunction", "endmacro",
                "endblock"}
_C_LITERALS = {"0", "1", "true", "false", "TRUE", "FALSE"}


# ── 归一化 ──────────────────────────────────────────────────────

def norm_flag(value: object) -> str | None:
    """把词归一成 ``"on"`` / ``"off"``；认不出返回 None（调用方判 failed）。"""
    v = "" if value is None else str(value).strip().lower()
    if v in ON_WORDS:
        return "on"
    if v in OFF_WORDS:
        return "off"
    return None


def apply_normalize(value: str, normalize: dict | None) -> str:
    """图级 ``normalize`` 先作用于**配置侧**取值（``{"ON": "开"}``）。

    方向是设计 §8 C1 钉死的：配置值 → 地图用词；与 C2 的 status_map
    （status → 目录名）方向相反。大小写不敏感兜底，避免 ``"on"`` 漏配。
    """
    s = "" if value is None else str(value)
    if not normalize:
        return s
    if s in normalize:
        return str(normalize[s])
    low = {str(k).strip().lower(): v for k, v in normalize.items()}
    return str(low.get(s.strip().lower(), s))


# ── 提取器（返回 (values, error, lookup)） ──────────────────────
#
# values: {name: (value|None, dynamic_reason|None)}——None+reason 即
#         「这个名字出现过但说不清」，绝不当成没出现。
# lookup: yaml_path 这类「路径里带 {key}」的提取器用，按条目键现算。

def _record(vals: dict, name: str, value: str | None,
            reason: str | None) -> None:
    """登记一次声明：动态一旦出现就永不被后续静态值冲掉（纪律 2）；
    静态则**后写覆盖前写**——同名多声明取最后一个（纪律 3）。"""
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
    """切「命令名 + 参数」：处理续行 ``\\`` 与跨行括号（if 块内是常态）。"""
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
        # option(NAME "help" ON) 的值是最后一个参数；set(NAME VALUE ...)
        # 的值是第二个（后面可能跟 CACHE BOOL "help"）
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
    """先按 JSON 解；不是 JSON 再试 PyYAML（本仓不新增依赖，没有就报）。"""
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
    except Exception as exc:  # noqa: BLE001 —— 第三方解析器异常类型不定
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
                return None, None          # 路径不存在 ⇒ 该键「没出现」
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

# 两种图级配方：C1 值级比对（读真相文件）/ C2 目录对账（读建档产物）
KINDS = ("code_flag", "dir_reconcile")

# 目录对账自己写的 drift 原因（清理时只清这两种，别碰手写地图的 changed/missing
# 与源校验的 value_changed——三条链路共用 ent["drift"] 这一个通道）
RECONCILE_REASONS = ("dir_missing", "dir_conflict")


def _kind(rule: dict) -> str:
    """配方类型（缺省 code_flag，与 validate_rule 同默认）。"""
    return str((rule or {}).get("kind") or "code_flag").strip()


# ── 配置校验（CLI 写 auto 时用） ────────────────────────────────

def _validate_reconcile(out: dict) -> dict:
    """C2 配方校验（用法错误抛 ValueError——配错了就是假的 clean）。"""
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
    """校验并按默认值补全一条 auto 配方；用法错误抛 ValueError。"""
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
    """``auto`` 单条或列表 ⇒ 规则列表（无 kind 的脏数据直接忽略）。"""
    if isinstance(auto, dict):
        return [auto] if auto.get("kind") else []
    if isinstance(auto, list):
        return [r for r in auto if isinstance(r, dict) and r.get("kind")]
    return []


# ── 主流程 ──────────────────────────────────────────────────────

def _base_root(rule: dict, registry_root: str | None) -> str | None:
    """根解析：建档 root > ``auto.root``（必须绝对）> None（= 不猜，拒绝执行）。"""
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
    """C1：逐条比对真相文件里的配置值（state 由调用方建好并落回图上）。"""
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

    # 哈希仲裁快路径：源签名没变 ⇒ 零读取，用上次结论（同纪律）
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
        # 逐条查预算（不是逐图）：一张图可能几百个开关，条目循环也不是
        # 免费；超了就地停，已核的算数、剩下的留给下轮
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
                # 所有源都读不了：报源错误本身，比「没找到」有用
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
            # 只清自己写的漂移（truth_source 的 changed/missing 归基础巡检管）
            if (ent.get("drift") or {}).get("reason") == "value_changed":
                ent["drift"] = None
            continue

        drifted += 1
        since = now
        if (ent.get("drift") or {}).get("reason") == "value_changed":
            since = (ent.get("drift") or {}).get("since") or now   # 只进必退
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
                  # 预算超限也算「这轮没核完」——不标的话索引行与收尾
                  # 都看不到，等于把一次截断悄悄当成全 clean
                  "incomplete": bool(src_err) or bool(res.get("incomplete"))})
    if src_err:
        # 部分源解析失败 ⇒ 图整体标 incomplete（设计 §8 规格修订 2），
        # 不当「部分通过」——报 clean 的图必须每个源都读到了
        state["error"] = "；".join("%s: %s" % (k, v) for k, v in src_err.items())
    else:
        state.pop("error", None)
    res["checked"] += checked
    res["clean"] += clean


# ── C2：数据目录对账 ────────────────────────────────────────────

def _add_error(state: dict, msg: str) -> None:
    """图上错误是**累积**的（源错误 / 无建档 / 对账目录不存在）：
    后到的不能把先到的抹掉，否则收尾只看到最后一条。"""
    old = str(state.get("error") or "")
    state["error"] = msg if (not old or msg in old) else "%s；%s" % (old, msg)


def _norm_rel(p: object) -> str:
    """相对路径规范化：``\\``→``/``、去 ``./`` 与首尾斜杠；根目录 → ``""``。"""
    s = str(p or "").strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    s = s.strip("/")
    return "" if s in ("", ".") else s


def _match_key(rx: re.Pattern, name: str) -> str | None:
    """从单个路径段抽键：命名组 ``(?P<key>...)`` 取组内容，否则取整串。

    **一律 fullmatch**（设计 §8 规格修订 6）：``^CN[0-9]+`` 配 ``CN2601_old``
    不许抽出 CN2601（那样删除的合约会显示「还在」，missing 漏报）；要抽取
    就写命名组，要相等就整串——没有第三种语义词。
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
    """在**已建档产物**上抽键（零磁盘 IO）：``{key: {所在子目录}}``。

    目录键与文件键都认：``训练集/CN2601/raw/a.fastq`` 里 CN2601 是目录键，
    ``训练集/CN2601.fastq`` 里 CN2601 是文件键。``所在子目录`` = 键实体的
    父目录（相对对账目录），供 status_map 判「该住在哪个子目录」。
    """
    found: dict[str, set[str]] = {}
    pre = prefix + "/" if prefix else ""
    for rel, ent in (files or {}).items():
        if not isinstance(ent, dict) or ent.get("status") == "missing":
            continue              # 建档已判 missing ⇒ 磁盘上确实没有
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
                continue          # 只数这类文件
            base = base[:len(base) - len(file_ext)]
        # 中间每一段都是「目录键」候选（目录键的所在子目录 = 它的父目录）
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
    """对账目录解析：``auto.root`` 相对**建档 root**；绝对路径必须在其下。"""
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
    """C2：地图条目 ↔ 数据目录双向对账（只读建档产物，只产信号）。

    四路输出见模块 docstring 的表；核心判断顺序是**先「在不在」再「在哪」**
    ——文件根本不在时谈子目录没意义，而 status_map 缺失时只做前者
    （降级安全，设计 §8 规格修订 7）。
    """
    # 只认**建档侧车本身**：只有 registry_root 字符串不够（拿不到 files 就
    # 等于没有对账数据源）。缺了就拒绝执行，不降级现扫、不回落 cwd。
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
            # 目录整枝不在：报「对账目录不存在」而不是让每个条目都 missing
            # ——后者会把一次改名说成 N 条数据丢失（假告警比漏报更伤信任）
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
                # 截断区域里的键本来就没被看到：判 missing 是纯误报
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
            # 只清自己写的漂移（手写地图 changed/missing、源 value_changed 不碰）
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

    # 计数用**独立键**（rec_*）：C1 的快路径会整轮复用上次结论、不改计数，
    # 两套计数共用同名键时，复用轮次会把上一轮的数再加一遍（虚高）。条目级
    # 结论本来就写在每条 ent["auto_state"] 里，这里只是给 CLI/诊断看的汇总。
    state.update({"rec_checked": checked, "rec_clean": clean,
                  "rec_drifted": drifted, "rec_failed": failed})
    state["at"] = now
    res["checked"] += checked
    res["clean"] += clean

    # 目录→地图：多出来的键（truth_source 结构上抓不到的那个方向）。
    # 扫描不完整 / 预算截断时**不写**这组数（宁可没有数，也不给半个数），
    # 并把 unmatched_complete 置 False 让读侧知道上一轮的数是历史。
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
        # 整轮结论只是「已扫到的那部分」：run 级也要自曝，否则调用方看到
        # res["incomplete"]=False 会把它当完整一轮
        res["incomplete"] = True
        state["incomplete"] = True
        if not complete:
            _add_error(state, "registered scan incomplete (scan_complete=false) -- "
                              "this round skips missing/absent judgments and only checks what was scanned")


def _verify_map(mname: str, m: dict, rules: list[dict],
                registry_root: str | None, res: dict, now: float,
                force: bool, deadline: float,
                registry: dict | None = None) -> None:
    """按配方类型分派：C1 值级比对 / C2 目录对账，共用同一份 ``auto_state``。"""
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
    """有没有 dir_reconcile 配方（决定要不要读 file_registry 侧车）。

    brain 侧用它避免为 C1-only 的库白读一份几 MB 的建档 JSON。
    """
    for m in (data.get("maps") or {}).values():
        for rule in _rules((m or {}).get("auto")):
            if _kind(rule) == "dir_reconcile":
                return True
    return False


def run(data: dict, registry_root: str | None = None,
        budget_s: float | None = None, force: bool = False,
        now: float | None = None, registry: dict | None = None) -> dict:
    """跑一遍所有带 ``auto`` 配方的图。**只改 data（内存）**，落盘由调用方管。

    返回计数 + 三态清单；``incomplete`` 表示预算超限或建档扫描不完整
    （不静默降级成全 clean）。``registry`` 只在图里有 dir_reconcile 配方时
    需要（传 file_registry 的侧车 dict，C2 零重复 IO 全靠它）。
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
