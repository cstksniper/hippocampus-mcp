"""地图层侧车存储（信息地图层）。

核心判断：**地图 ≠ 记忆**——地图答「现在是什么」，记忆答「发生过
什么」。所以地图条目不进 recall 检索路径（不带向量、不是记忆），按键
更新（同一键反复写，旧值被替换而不是与新值竞争），持续过期（不改就是错）。

侧车 JSON（``<persist_dir>/maps.json``）::

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

四条纪律（与文件建档同族，详见设计文档四道闸）：
1. **闸一·写时更新**：状态只走 :func:`set_entry` 一个入口（map_set 工具
   的底层），一经 set 即整体替换该键的旧值——「一键一条」是数据模型保证
   的，不靠调用方自觉。
2. **闸二·真相源验证**：条目可声明 ``truth_source``（文件路径），set 时
   记录其（size, mtime_ns）+ blake2b 签名；:func:`verify` 重读比对——
   签名失匹读一次内容哈希仲裁（touch / git checkout 不误报），真变了
   标 ``drift``。与 file_registry 的哈希仲裁同一思路
   （条目 truth_source 指向文件路径即自动继承本机制）。
3. **opt-in**：没有侧车就没有地图层一切行为；不存在返回 None，不炸
   调用方（与 file_registry.load 同约定）。
4. **失败不炸热路径**：磁盘异常由调用方兜底；本模块只对「用法错误」
   （超长键值 / 未知 map 等）抛 ValueError。
5. **并发写在锁内**（见 :mod:`sidecar_lock`）：所有
   读-改-写一律走 :func:`update`；直接 :func:`save` 只保证「不写坏」，
   不防丢更新。抢不到锁抛 :class:`SidecarBusy`，由调用方决定等或跳过
   ——**绝不静默丢写**（这是 tmp+rename 时代的真实病灶）。
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

# 体量护栏（防病态 JSON：一个条目不是一段文章）
MAX_KEY_LEN = 100        # 条目键（CN2601 / chr1 / --use-avx2）
MAX_STATUS_LEN = 200     # 状态值
MAX_DESC_LEN = 200       # 图描述（相关性匹配靠它，一句话）
MAX_NOTE_LEN = 500       # note
MAX_FIELDS = 20          # 附加字段个数
MAX_FIELD_LEN = 200      # 单个附加字段值长度
MAX_MAPS = 200           # 一个库的地图数上限（防误用成记忆库）


# ── 读写 ────────────────────────────────────────────────────────

def maps_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "maps.json")


def load(persist_dir: str) -> dict | None:
    """读侧车；不存在/损坏都返回 None（损坏按未启用处理，不炸调用方）。"""
    try:
        with open(maps_path(persist_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("maps"), dict):
        return None
    return data


def _write_atomic(persist_dir: str, data: dict) -> None:
    """原子写（tmp + rename）：多会话并发改地图不能留半个 JSON。

    **不含锁**——由 :func:`update` 在已持锁的临界区里调用，或由
    :func:`save` 自己拿锁后调用。两者都不许在未持锁时直接用它。
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
    """给**已在手**的 data 落盘（原子写 + 侧车锁）。

    ⚠️ 锁只保证「不写坏、两次写不交错」，**不防丢更新**：读若发生在锁
    之外，别人这期间的改动会被这份整量快照覆盖。要防丢更新用
    :func:`update`——那里读和写在同一个临界区里。

    这个签名留着是因为 ``data`` 常是从别处算出来的（CLI 预演、测试
    造数据），强行要求回调反而绕。抢不到锁抛 :class:`SidecarBusy`，
    绝不静默丢写。
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
    """在侧车锁内做「读 → 改 → 写」，**防丢更新的唯一正确姿势**。

    ``fn(data)`` 在锁内拿到**最新的** data（不是调用方手里的旧快照），
    原地修改后返回 ``(save: bool, result)``：

    - ``save=True`` ⇒ 落盘；
    - ``save=False`` ⇒ 只算不写（例如巡检间隔未到、无配方可跑）——
      这是常态，不是异常；
    - ``create=True`` 且侧车不存在 ⇒ 用 :func:`empty` 起一份；
      ``create=False`` 且侧车不存在 ⇒ **不调 fn**，直接返回 ``(None, False)``
      （opt-in：没启用地图层就不该被写入）。

    🔴 **「读不出来」不等于「不存在」**：文件在但 ``load`` 返回 None
    （损坏/被独占）⇒ 抛 :class:`SidecarUnreadable`，绝不起空侧车覆盖。
    只有 ``reset_unreadable=True``（init --force 这种显式重建）才把坏
    文件改名隔离后重建。

    返回 ``(result, saved)``。抢不到锁抛 :class:`SidecarBusy`——由调用方
    决定等（``wait``）/ 报错 / 跳过本轮，不允许当成写成功。
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
                sl.quarantine(p)          # 显式重建：隔离，不删
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


# ── 写入（闸一：唯一入口） ──────────────────────────────────────

def _clip(s: object, n: int, what: str) -> str:
    s = "" if s is None else str(s)
    if len(s) > n:
        raise ValueError("%s too long (%d > %d chars): map entries are not articles; "
                         "put long content into memories instead" % (what, len(s), n))
    return s


def resolve_source(path: str, root: str | None = None) -> str:
    """truth_source 解析：绝对路径原样；相对路径先试建档项目根
    （file_registry.init 显式指定的 root），再试进程 cwd。

    返回**首个存在的候选**；都不存在时返回主候选（调用方记录不到签名，
    由 verify 在文件出现后按 missing 处理）。
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
    """truth_source 当前签名（size/mtime_ns + 内容哈希）。

    文件不存在 / 太大不哈希 ⇒ None（verify 时按 missing 或无基线处理）。
    与 file_registry 同一套仲裁签名（自动继承本机制）。
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
    """按键 upsert 一个条目（闸一唯一写入口）。返回 (map, entry, notes)。

    - map 不存在则新建（desc 缺省用 name——desc 是 L1 相关性匹配的
      唯一依据，见设计文档 §3/§6）；
    - 同键再写**整体替换**旧值（旧状态与新状态竞争在数据模型层面被排除）；
    - ``truth_source``：非空 ⇒ 设置并记录当前签名（set 的那一刻起算「真相
      变没变」）；None（没传）⇒ **保留**原值与原签名基线（局部更新状态
      不该顺手丢掉真相源声明）；空串 ⇒ 显式清除；
    - 重新 set 自动清除该条目的 drift（重新记录了基线）；
    - notes 是非致命警告（如 truth_source 不在磁盘上）——返回给宿主，
      不打断写入。
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
        new_ent["truth_source"] = old_ts          # 没传 ⇒ 保留
        new_ent["source_sig"] = old_sig          # 基线不动（verify 会
    elif str(truth_source).strip():               # 如实暴露这期间的变化）
        new_ent["truth_source"] = _clip(str(truth_source).strip(), 400,
                                        "truth_source")
        full = resolve_source(new_ent["truth_source"], root)
        sig = _record_source_sig(full)
        if sig is None:
            notes.append("truth_source missing from disk or too large (no recorded signature): %s"
                         % new_ent["truth_source"])
        new_ent["source_sig"] = sig
    # truth_source == ""（显式空串）⇒ 清除：source_sig 保持 None
    m["entries"][key] = new_ent
    m["updated_at"] = now
    return m, new_ent, notes


# ── 查询 ────────────────────────────────────────────────────────

def get_entry(data: dict, name: str, key: str) -> dict | None:
    m = (data.get("maps") or {}).get(str(name)) or {}
    return (m.get("entries") or {}).get(str(key))


def map_names(data: dict) -> list[str]:
    return sorted((data.get("maps") or {}).keys())


def ack_entry(data: dict, name: str, key: str,
              now: float | None = None) -> dict:
    """「我核过了，状态没变」：只前移 ``updated_at``，不动 status/fields。

    闸三（提及级漂移）的**唯一显式清除手势**（``map/mention.py`` 清除
    规则 1）：判据是「有更晚的记忆说了另一个状态」，把确认时刻写进
    ``updated_at`` 就让它不再满足；同时清掉 ``ent["mention"]``。

    为什么不复用 ``set_entry``：重写要人把 status/fields 再抄一遍，
    抄错就是把「确认没变」写成「改成了别的」——最需要安全的地方不该有
    抄写风险。也不动 ``updated_by``（状态不是他写的，只是确认过），
    确认痕迹单独落 ``acked_at``。
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
    ent["mention"] = None            # 闸三标记随确认一起清（新增的会重算）
    m["updated_at"] = now
    return ent


def list_maps(data: dict, now: float | None = None) -> list[dict]:
    """全库地图摘要（给 L0 索引行与 map_list 工具用）。"""
    now = time.time() if now is None else now
    out = []
    for name, m in (data.get("maps") or {}).items():
        ents = m.get("entries") or {}
        drifted = sum(1 for e in ents.values() if e.get("drift"))
        # 自动 verify 的三态里「说不清」也要上浮——
        # extract_failed 不是 drift（值没变，是我们拒绝猜），但同样不能
        # 当成 clean 藏起来（设计 §8 C1「失败不当一致」）。
        failed = sum(1 for e in ents.values()
                     if (e or {}).get("auto_state", {}).get("state") == "failed")
        # 目录对账的两个方向。drift 按原因分开数——
        # 「值变了」与「文件没了」的处置完全不同，混成一个数等于没说。
        dir_missing = sum(
            1 for e in ents.values()
            if ((e or {}).get("drift") or {}).get("reason") == "dir_missing")
        dir_conflict = sum(
            1 for e in ents.values()
            if ((e or {}).get("drift") or {}).get("reason") == "dir_conflict")
        auto_state = m.get("auto_state") or {}
        # 闸三：更新后被提到「另一个状态」的条目数。它是
        # **提示**不是事实——与 drift（真相源比对结果）分开计数，页面上
        # 也分开写（`·疑变N` vs `·漂移N`），不合并成一个大数。
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
            # 目录→地图方向（truth_source 结构上抓不到的那一路）
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


# ── 验证（闸二） ────────────────────────────────────────────────

def verify(data: dict, name: str | None = None,
           root: str | None = None) -> dict:
    """重读真相源，逐条比对签名（§4 闸二的机械化版）。

    对每个带 ``source_sig`` 基线的条目：
      - stat 签名没变 ⇒ clean，零读取；
      - 失配 ⇒ 读一次内容哈希：相同 ⇒ 纯 touch，刷新签名；不同 ⇒
        ``drift = {"reason": "changed"}``；
      - 文件消失 ⇒ ``drift = {"reason": "missing"}``；
      - 内容回到基线哈希（改了又改回 / 删了又恢复）⇒ 清除 drift。

    没有 truth_source 的条目跳过（纯人工状态退到闸三/闸四，设计文档
    §4 覆盖矩阵）。返回计数 + 漂移清单（宿主据此知道该重核哪些条目）。
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
                # 纯 touch（git checkout / robocopy 重置 mtime）或内容
                # 回到了基线：刷新签名、清除 drift
                sig["size"], sig["mtime_ns"] = st.st_size, st.st_mtime_ns
                if digest is not None:
                    sig["hash"] = digest
                was_drifted = bool(ent.get("drift"))
                ent["drift"] = None
                res["touched" if was_drifted or cur != base else "clean"] += 1
                continue
            # 内容真的变了：保留旧基线哈希（改回来要能识别），标 drift
            ent["drift"] = {"reason": "changed", "since": now}
            res["drifted"].append({"map": mname, "key": key,
                                   "reason": "changed",
                                   "truth_source": ts})
    return res
