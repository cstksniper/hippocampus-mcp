"""项目文件建档：hippo 与项目磁盘之间的过时检测。

需求：「对项目所有文件建档，记录文件更新到 hippo 的时间；
文件在此之后被更新过，就知道此文件的信息是过时的」。本模块把它实现为
**签名比对**而非墙钟比对（理由见 common/constants 常量区注释：git checkout /
robocopy 批量重置 mtime 但不改内容时，墙钟比对会把整个仓库误判成过时）。

侧车 JSON（``<persist_dir>/file_registry.json``）::

    {
      "version": 1,
      "root": "/absolute/path/to/project",  # init 时显式指定，不猜
      "ignores": [...],                   # 生效的忽略规则快照
      "last_checked": 1710000000.0,
      "scan_complete": true,
      "files": {
        "src/main.py": {
          "size": 1234, "mtime_ns": 1710000000000000000,
          "hash": "b3a1...",              # blake2b-128，见 _hash_file
          "first_seen": ..., "last_synced": ..., "last_checked": ...,
          "status": "synced"              # synced | stale | missing
        }
      }
    }

四条硬纪律：
1. **opt-in**：没有侧车就没有一切行为（不扫、不报、不占卡面一行）——
   启用方式是显式 ``init``，杜绝「猜一个项目根然后扫错目录」。
2. **哈希仲裁**：stale 判定不是「mtime 变了就算变」。git checkout /
   robocopy / touch 会重置 mtime 但内容不变，(mtime, size) 签名失配时
   **读一次内容哈希**确认：哈希相同 ⇒ 只是 touch，更新签名保持 synced。
   反过来，mtime 保留式篡改（``touch -r`` 回填时间）不在范围内——
   与 C1a source_sig 同一性质，热路径不为对抗性场景付全量读取的代价。
3. **预算兜底**：扫描带文件数 + 墙钟双重上限，超限标 ``scan_complete=false``
   并在状态卡自曝「扫描不完整」——绝不静默吞（与全项目「自曝」纪律一致）。
4. **失败不炸热路径**：所有磁盘操作异常由调用方兜底；本模块自己该抛的
   只有「用法错误」（如重复 init、sync 不存在的路径）。
5. **读-改-写走锁**：daemon 的周期扫描与 CLI
   ``sync`` / ``init`` 会撞在一起，裸 tmp+rename 是**静默丢更新**。
   一律用 :func:`update`；抢不到锁抛 :class:`SidecarBusy`，不静默写。
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tempfile
import time

from ..common import sidecar_lock as sl
from ..common.constants import SOURCE_SIG_FILE
from .constants import (
    ENV_FILE_REGISTRY_BUDGET_S,
    ENV_FILE_REGISTRY_IGNORE,
    ENV_FILE_REGISTRY_MAX_FILES,
    FILE_REGISTRY_BUDGET_S,
    FILE_REGISTRY_DEFAULT_IGNORES,
    FILE_REGISTRY_MAX_FILES,
)
from ..common.sidecar_lock import SidecarBusy, SidecarUnreadable

# 构造注入缝：旋钮登记在 config，生产路径由 engine import 时灌入；
# 解析序（env 先、注入值回退）见 common/config_bridge。
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

REGISTRY_VERSION = 1
IGNORE_FILE_NAME = ".hippoignore"
HASH_MAX_BYTES = 64 * 1024 * 1024   # 超过此大小不哈希（数据文件）：
                                    # stat 签名失配即保守判 stale，宁可多报


# ── 读写 ────────────────────────────────────────────────────────

def registry_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "file_registry.json")


def load(persist_dir: str) -> dict | None:
    """读侧车；不存在/损坏都返回 None（损坏按未启用处理，不炸调用方）。"""
    p = registry_path(persist_dir)
    try:
        with open(p, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(reg, dict) or "files" not in reg:
        return None
    return reg


def _write_atomic(persist_dir: str, reg: dict) -> None:
    """原子写（tmp + rename）：扫描可能被并发会话打断，不能留半个 JSON。

    **不含锁**——只由 :func:`update`（已持锁）或 :func:`save`（自己拿锁）
    调用。
    """
    p = registry_path(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".file_registry_", dir=persist_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save(persist_dir: str, reg: dict, wait: float = 0.0) -> None:
    """落盘（原子写 + 侧车锁）。**不防丢更新**，理由同 map/store.save。"""
    lock = sl.lock_path(registry_path(persist_dir))
    tok = sl.acquire(lock, wait=wait)
    if tok is None:
        raise SidecarBusy(lock, sl.holder(lock))
    try:
        _write_atomic(persist_dir, reg)
    finally:
        sl.release(lock, tok)


def update(persist_dir: str, fn, create: bool = False, wait: float = 0.0,
           reset_unreadable: bool = False):
    """锁内「读 → 改 → 写」；``fn(reg) -> (save, result)``。

    与 :func:`map/store.update` 同约定（含 ``create`` / 无侧车不调 fn /
    「读不出来 ≠ 不存在」/ 抢不到锁抛 :class:`SidecarBusy`）。**这是建档
    侧车唯一安全的写姿势**：周期扫描（可能几秒）与 CLI ``sync`` 撞车时，
    后写的不会抹掉先写的。
    """
    lock = sl.lock_path(registry_path(persist_dir))
    tok = sl.acquire(lock, wait=wait)
    if tok is None:
        raise SidecarBusy(lock, sl.holder(lock))
    try:
        reg = load(persist_dir)
        if reg is None:
            p = registry_path(persist_dir)
            exists = os.path.isfile(p)
            if exists and not reset_unreadable:
                raise sl.SidecarUnreadable(p)
            if exists:
                sl.quarantine(p)
            if not create:
                return None, False
            reg = {"version": REGISTRY_VERSION, "root": "",
                   "ignores": [], "last_checked": 0.0,
                   "scan_complete": False, "files": {}}
        save_flag, result = fn(reg)
        if save_flag:
            _write_atomic(persist_dir, reg)
        return result, bool(save_flag)
    finally:
        sl.release(lock, tok)


def init(persist_dir: str, root: str, force: bool = False) -> dict:
    """建档初始化：显式指定项目根（不猜——猜错就是扫错目录）。

    已存在时拒绝（除非 force）：重复 init 会丢掉全部建档历史，
    这种操作必须是显式的。

    锁内「判存在 → 建」：两个会话同时 init 时，后到的那次必须看到
    先到的结果并走「已建档」分支，而不是各写一份把对方的历史抹掉。
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError("the project root does not exist: %s" % root)

    def _do(reg):
        if (reg.get("root") or reg.get("files")) and not force:
            raise FileExistsError(
                "already registered (root=%s, %d files); re-init would lose history -- pass force to confirm"
                % (reg.get("root"), len(reg.get("files") or {})))
        reg.clear()
        reg.update({
            "version": REGISTRY_VERSION,
            "root": root,
            "ignores": effective_ignores(root),
            "last_checked": 0.0,
            "scan_complete": False,
            "files": {},
        })
        return True, reg

    # force 同时表示「允许隔离读不出来的旧侧车后重建」——重建本来就是
    # 显式破坏性操作，把「读到坏文件」和「已建档」两种拒绝统一到 force 上。
    reg, _saved = update(persist_dir, _do, create=True,
                         reset_unreadable=force)
    return reg


# ── 忽略规则 ────────────────────────────────────────────────────

def effective_ignores(root: str) -> list[str]:
    """默认忽略 + 根下 .hippoignore + env 额外 glob（快照进侧车）。

    .hippoignore 用 gitignore 的子集语法：``#`` 注释、空行忽略、
    ``*`` glob（fnmatch）、结尾 ``/`` 表示只匹配目录、其余按
    路径片段匹配（任一父目录或完整相对路径命中即忽略）。
    """
    patterns = list(FILE_REGISTRY_DEFAULT_IGNORES)
    p = os.path.join(root, IGNORE_FILE_NAME)
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    except OSError:
        pass
    extra = _resolve(ENV_FILE_REGISTRY_IGNORE, "file_registry_ignore", "")
    for pat in extra.split(","):
        pat = pat.strip()
        if pat:
            patterns.append(pat)
    return patterns


def _match_pattern(rel: str, is_dir: bool, pat: str) -> bool:
    """单条模式匹配（子集语法，见 effective_ignores 注释）。"""
    if pat.endswith("/"):
        if not is_dir:
            return False
        pat = pat[:-1]
    rel_n = rel.replace(os.sep, "/")
    parts = rel_n.split("/")
    # 目录名/路径片段：任一父目录或自身 basename 命中
    if not any(ch in pat for ch in "*?["):
        if pat in parts:
            return True
    # glob：完整相对路径或 basename
    if fnmatch.fnmatch(rel_n, pat):
        return True
    if fnmatch.fnmatch(parts[-1], pat):
        return True
    return False


def is_ignored(rel: str, is_dir: bool, patterns: list[str]) -> bool:
    """**参考实现**（逐模式 fnmatch）。语义基准，供测试/CLI 对拍；
    扫描热路径走 :class:`_IgnoreIndex`（同一语义的编译态，见其 docstring）。
    """
    return any(_match_pattern(rel, is_dir, p) for p in patterns)


class _IgnoreIndex:
    """忽略规则的编译态（每次 scan 编译一次，26 模式 × 2 万文件的重复匹配
    从循环里挪到循环外）。

    动机是实测（NTFS，2 万文件量级）：
    **热扫描 21.6s，其中 81% 是 is_ignored**——每文件对 26 条模式各做一次
    ``str.replace/split`` + 两次 ``fnmatch``，而 fnmatch 在 Windows 上每次
    都带 ``normcase``（Win32 LCMapStringEx）。纯 Python 的模式匹配本来就是
    CPU 密集，不需要换语言：预编译就能解决。

    语义与 :func:`is_ignored` **逐字符对齐**（探针对拍全仓 362 个路径）：
    - 无 glob 字符的模式 → 路径片段集合（任一父目录或 basename 命中即中），
      另加整路径全等（fnmatch 对无 glob 模式退化为全等）；
    - 带 glob 的模式 → 预编译正则，匹配完整相对路径或 basename；
    - 结尾 ``/`` 只匹配目录；
    - 大小写在 Windows 上折叠（normcase，历史行为不变）。
    """

    __slots__ = ("_plain", "_dir_plain", "_globs")

    def __init__(self, patterns: list[str]):
        plain: set[str] = set()
        dir_plain: set[str] = set()
        globs: list[tuple[re.Pattern, bool]] = []
        for raw in patterns:
            pat = raw
            dir_only = pat.endswith("/")
            if dir_only:
                pat = pat[:-1]
            if not pat:
                continue
            npat = os.path.normcase(pat)
            if any(ch in pat for ch in "*?["):
                globs.append((re.compile(fnmatch.translate(npat)), dir_only))
            else:
                (dir_plain if dir_only else plain).add(npat)
        self._plain = plain
        self._dir_plain = dir_plain
        self._globs = globs

    def is_ignored(self, rel: str, is_dir: bool) -> bool:
        # 顺序要紧：先按 "/" 切段，再 normcase。Windows 上 normcase 会把
        # "/" 转 "\\" 并小写——先 normcase 就再也切不开父目录段（实测假阴性
        # 4 万路径的教训：与参考实现 _match_pattern 的 replace(sep,"/")
        # 必须同序）。参考实现的净语义：**任一**路径段 normcase 相等即中
        # （父目录命中即整枝忽略），或整路径 normcase 相等；glob 走完整
        # 路径或 basename。
        rel_f = rel.replace(os.sep, "/") if os.sep != "/" else rel
        segs = rel_f.split("/")
        rel_n = os.path.normcase(rel_f)
        for s in segs:
            ns = os.path.normcase(s)
            if ns in self._plain:
                return True
            if is_dir and ns in self._dir_plain:
                return True
        if rel_n in self._plain:
            return True
        if is_dir and rel_n in self._dir_plain:
            return True
        base = os.path.normcase(segs[-1])
        for rx, dir_only in self._globs:
            if dir_only and not is_dir:
                continue
            if rx.match(rel_n) or rx.match(base):
                return True
        return False


def scan_limits() -> tuple[int, float]:
    """双预算的生效值（env 可覆盖，供 brain 自动巡检与 CLI 共用）。

    默认 2 万文件 / 1s。量化项目若远超此量级（十万级文件），不必改代码：
    ``HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES`` / ``HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S``
    直接覆盖；超限的降级行为不变（scan_complete=false 自曝，不判 missing）。
    """
    def _int_env(name: str, default: int, lo: int, hi: int) -> int:
        try:
            raw = _resolve(name, "file_registry_max_files", str(default))
            v = int(float(raw or default))
        except ValueError:
            v = default
        return max(lo, min(hi, v))

    def _float_env(name: str, default: float, lo: float, hi: float) -> float:
        try:
            raw = _resolve(name, "file_registry_budget_s", str(default))
            v = float(raw or default)
        except ValueError:
            v = default
        return max(lo, min(hi, v))

    return (_int_env(ENV_FILE_REGISTRY_MAX_FILES, FILE_REGISTRY_MAX_FILES, 1, 10 ** 7),
            _float_env(ENV_FILE_REGISTRY_BUDGET_S, FILE_REGISTRY_BUDGET_S, 0.05, 3600.0))


# ── 扫描 ────────────────────────────────────────────────────────

def _sig(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def _hash_file(path: str) -> str | None:
    """blake2b-128 内容哈希；太大/读不了返回 None（调用方保守判 stale）。

    只在 stat 签名失配时才调（touch / git checkout 的仲裁），
    签名没变的文件一次都不读——这是「quick check + 哈希仲裁」的
    rsync --checksum 思路，常态扫描零内容读取。
    """
    try:
        if os.path.getsize(path) > HASH_MAX_BYTES:
            return None
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def scan(reg: dict, max_files: int, budget_s: float) -> dict:
    """带预算地重扫项目根，更新每个文件的 status（原地改并返回 reg）。

    status 语义：
      synced  —— hippo 的信息反映当前磁盘内容
      stale   —— 内容变过且 hippo 未同步（或文件大到不哈希、签名失配时
                保守判 stale）；从未 sync 过的新文件一进场就是 synced
                （「看到」不等于「过时」，只有改过才算）
      missing —— 建档在册但磁盘上没有（删除/移动 ⇒ 同样要知道）

    判定分两路（哈希仲裁，见模块 docstring 纪律 2）：
      stat 签名（size, mtime_ns）没变 ⇒ synced，零读取；
      签名失配 ⇒ 读内容哈希：相同 ⇒ 纯 touch，更新签名仍 synced；
      不同（或太大不哈希）⇒ stale。

    性能（NTFS 实测，2 万文件量级）：
    老写法 2 万文件热扫 21.6s——81% 耗在 is_ignored 的逐模式 fnmatch
    （Windows 上每次带 normcase/LCMapStringEx），其余是每文件一次
    os.stat（≈0.29ms，NTFS 过滤驱动价）。现在：忽略规则编译成
    :class:`_IgnoreIndex`（一次编译，集合+正则匹配）+ scandir 递归用
    ``DirEntry.stat()`` 复用枚举缓存（Windows 上 FindFirstFile 已带回
    size/mtime，零额外系统调用；实测 5.59s → 0.16s）。
    """
    root = reg.get("root") or ""
    if not root or not os.path.isdir(root):
        reg["scan_complete"] = False
        return reg
    # 每次重读忽略规则：.hippoignore / env 改了要生效（init 时的快照会过期）
    patterns = effective_ignores(root)
    reg["ignores"] = patterns
    idx = _IgnoreIndex(patterns)
    now = time.time()
    files: dict = reg.setdefault("files", {})
    seen: set[str] = set()
    scanned = 0
    complete = True
    deadline = now + budget_s

    def _walk(dirpath: str, rel_dir: str) -> bool:
        """递归扫一层（scandir：拿得到 DirEntry，stat 复用枚举缓存）。
        返回 False = 双预算用尽，停止递归（与 os.walk 提前跳出等价）。"""
        nonlocal scanned
        try:
            it = os.scandir(dirpath)
        except OSError:
            return True  # 目录读不了（权限/瞬删）：跳过这枝，不算预算耗尽
        with it:
            for entry in it:
                # 双预算逐文件检查（不只在大目录结束后）：墙钟超限时最多多扫
                # 一个文件，不会把整个目录吃完
                if scanned >= max_files or time.time() > deadline:
                    return False
                name = entry.name
                rel = "%s/%s" % (rel_dir, name) if rel_dir else name
                try:
                    is_link = entry.is_symlink()
                    if entry.is_dir(follow_symlinks=False):
                        # 剪枝：忽略的目录整枝不进（.git 有几十万对象）。
                        if idx.is_ignored(rel, True):
                            continue
                        if not _walk(entry.path, rel):
                            return False
                        continue
                    if is_link and entry.is_dir():
                        # 符号链接→目录：os.walk(followlinks=False) 同语义——
                        # 不递归（防环）也不登记，与老行为逐字符一致。
                        continue
                except OSError:
                    continue
                if idx.is_ignored(rel, False):
                    continue
                scanned += 1
                seen.add(rel)
                try:
                    st = entry.stat()  # 默认跟随符号链接 = os.stat 语义
                except OSError:
                    st = None
                ent = files.get(rel)
                if st is None:
                    if ent is not None:
                        ent["status"] = "missing"
                        ent["last_checked"] = now
                    continue
                size, mtime_ns = st.st_size, st.st_mtime_ns
                if ent is None:
                    files[rel] = {
                        "size": size, "mtime_ns": mtime_ns,
                        "hash": _hash_file(entry.path),
                        "first_seen": now, "last_synced": now,
                        "last_checked": now, "status": "synced",
                    }
                    continue
                ent["last_checked"] = now
                if (size, mtime_ns) == (ent.get("size"), ent.get("mtime_ns")):
                    # 签名没变 ⇒ 内容没再变。但 stale 是「hippo 未同步」的
                    # 状态机，**只能被 mark_synced 清除**（重新 ingest 过），
                    # 不能被扫描自愈——否则改完一次，下次扫描就悄悄变绿，
                    # 「过时」信号最多活一个扫描间隔，等于没有。
                    if ent.get("status") == "missing":
                        ent["status"] = "synced"  # 回来了且签名没变（删了又恢复）
                    continue
                # 签名失配：哈希仲裁（touch / git checkout 不误报）
                digest = _hash_file(entry.path)
                ent["size"], ent["mtime_ns"] = size, mtime_ns
                if digest is not None and digest == ent.get("hash"):
                    ent["status"] = "synced"   # 纯 touch：刷新签名，不算过时
                else:
                    if digest is not None:
                        ent["hash"] = digest
                    ent["status"] = "stale"
        return True

    if not _walk(root, ""):
        complete = False
    # 建档在册但本次扫描没看到的（被 break 截断的区域不能判 missing——
    # 那正是 scan_complete=false 要防的误报）
    if complete:
        for rel, ent in files.items():
            if rel not in seen and ent.get("status") != "missing":
                ent["status"] = "missing"
                ent["last_checked"] = now
    reg["last_checked"] = now
    reg["scan_complete"] = complete
    reg["scanned"] = scanned
    return reg


def load_source_sigs(persist_dir: str) -> dict | None:
    """读 C1a 源签名层（brain 的 ``source_sigs``，ingest 时回填）。

    不存在 / 读不出来 ⇒ None。这是**证据缺失**，与「层在、但这个文件没
    记录过」（＝事实：没 ingest 过）**不是一回事**——调用方必须分开处理
    （见 :func:`mark_synced` 的 ``source_sigs`` 参数）。

    sqlite 后端下这份文件不存在（层在库里）⇒ 同样返回 None，CLI 会自曝
    「无法校验」而不是静默按信任制放行。

    已知边界：读的是**盘上**的层，daemon 内存里还没落盘的那次 ingest 读
    不到 ⇒ 校验会比 brain 内存层旧一拍。方向是保守的（旧 ⇒ 更可能拒，
    不会放行不该放的），且 CLI 与 daemon 本来就是两个进程，取一致快照要
    额外的跨进程协议，不值当。
    """
    p = os.path.join(persist_dir, SOURCE_SIG_FILE)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def source_sig_index(sigs: dict | None) -> dict[str, dict]:
    """源签名层 → **绝对路径**索引；同一路径多条取 ``sig_at`` 最新者。

    与层合并口径一致（``_LAYER_MERGE_RULES["sourcesig"]``：同键取较新）
    ——两个进程都 ingest 过同一个文件时，后校验的那份才是磁盘现状。

    纯函数：brain 传内存层、CLI 传读盘结果，共用一份路径归一化，避免两处
    各写一遍然后漂移。
    """
    out: dict[str, dict] = {}
    for e in (sigs or {}).values():
        if not isinstance(e, dict):
            continue
        path = e.get("path")
        if not path:
            continue
        key = os.path.normcase(os.path.abspath(str(path)))
        prev = out.get(key)
        if prev is None or float(e.get("sig_at") or 0.0) >= float(
                prev.get("sig_at") or 0.0):
            out[key] = e
    return out


def _sha1_file(path: str) -> str | None:
    """sha1 内容哈希（与 brain ``_file_sig`` 同算法，用于对得上它的值）。

    只在 (mtime_ns, size) 对不上时兜底读一次，且 sync 是显式命令、路径数
    有限——热扫描路径不碰（那里是 blake2b，见 :func:`_hash_file`）。
    """
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sync_verify(full: str, disk: tuple[int, int],
                 idx: dict[str, dict]) -> str | None:
    """「hippo 真吃过当前内容」的判据：None = 通过，否则是拒绝原因码。

    * ``not_ingested``：签名层里没有这个文件 ⇒ 无从证明，拒绝
    * ``sig_stale``   ：记过，但记的那版不是磁盘当前这版（ingest 完又改了）

    两档放行：(mtime_ns, size) 双等 ⇒ 那次 ingest 之后文件没再变；
    对不上但 **sha1 相同** ⇒ touch / git checkout，内容还是吃过的那版
    （与 :func:`scan` 的哈希仲裁同一精神：签名失配不直接等于内容变了）。
    """
    sig = idx.get(os.path.normcase(os.path.abspath(full)))
    if not sig:
        return "not_ingested"
    size, mtime_ns = disk
    try:
        if (int(sig.get("mtime_ns") or 0) == int(mtime_ns)
                and int(sig.get("size") or -1) == int(size)):
            return None
    except (TypeError, ValueError):
        pass
    want = str(sig.get("sha1") or "")
    if want:
        got = _sha1_file(full)
        if got and got == want:
            return None
    return "sig_stale"


def mark_synced(reg: dict, rel_paths: list[str],
                source_sigs: dict | None = None
                ) -> tuple[list[str], list[str], list[dict]]:
    """把指定路径标记为「hippo 已反映当前磁盘内容」（更新建档签名 + 哈希）。

    返回 ``(ok, not_found, rejected)``：

    * ``ok``       已标记（相对项目根的在册路径）
    * ``not_found``不在册，或磁盘上已经没有这个文件（顺带标 missing）
    * ``rejected`` **校验没过，一个都没标**（``{"path","reason"}``，
      reason 见 :func:`_sync_verify`）

    **信任制与本函数的默认**：不传 ``source_sigs`` ⇒ 老行为（说同步就同步）。
    传了 ⇒ 逐条拿源签名层验「hippo 真吃过当前内容」，没 ingest 过的文件
    **不能标 synced**——否则「改完文件、一行 sync 就变绿」的过时信号形同
    虚设（设计评审后补上）。

    校验是**证据判定，不是内容搬运**：放行只说明签名层认得这版内容，内容
    怎么进记忆仍是宿主/agent 的动作（没有自动重导入）。
    """
    now = time.time()
    ok: list[str] = []
    miss: list[str] = []
    rejected: list[dict] = []
    files = reg.get("files") or {}
    idx = source_sig_index(source_sigs) if source_sigs is not None else None
    for raw in rel_paths:
        rel = str(raw).replace(os.sep, "/").lstrip("./")
        ent = files.get(rel)
        if ent is None:
            # 容忍调用方传绝对路径或在册路径的各种写法
            cand = [k for k in files if k.endswith("/" + rel) or rel.endswith("/" + k)]
            if not cand:
                miss.append(rel)
                continue
            rel = cand[0]
            ent = files[rel]
        full = os.path.join(reg["root"], rel)
        sig = _sig(full)
        if sig is None:
            ent["status"] = "missing"
            ent["last_checked"] = now
            miss.append(rel)
            continue
        if idx is not None:
            why = _sync_verify(full, sig, idx)
            if why:
                rejected.append({"path": rel, "reason": why})
                continue
        size, mtime_ns = sig
        ent["size"], ent["mtime_ns"] = size, mtime_ns
        ent["hash"] = _hash_file(full)
        ent["last_synced"] = now
        ent["last_checked"] = now
        ent["status"] = "synced"
        ok.append(rel)
    return ok, miss, rejected


# ── 摘要 ────────────────────────────────────────────────────────

def _stale_days(ent: dict, now: float) -> int:
    """过时多久了：自「变更发生」（磁盘 mtime）或「上次同步」起算，取早者。

    用 max(last_synced, mtime)：文件 30 天前同步、2 天前被改 ⇒ 过时 2 天，
    不是 30 天——「多久没同步」会高估，「多久前变的」才是agent 想听的。
    mtime 在未来（时钟漂移）时 clamp 到 0。
    """
    ref = max(float(ent.get("last_synced") or 0.0),
              float(ent.get("mtime_ns") or 0) / 1e9)
    return max(0, int((now - ref) // 86400))


def summarize(reg: dict, now: float | None = None) -> dict:
    """给状态卡/收尾清单的紧凑摘要（O(册内文件数)，纯内存）。"""
    now = time.time() if now is None else now
    files = reg.get("files") or {}
    stale, missing, oldest = [], [], 0
    for rel, ent in files.items():
        status = ent.get("status")
        if status == "stale":
            stale.append(rel)
            oldest = max(oldest, _stale_days(ent, now))
        elif status == "missing":
            missing.append(rel)
    return {
        "root": reg.get("root"),
        "total": len(files),
        "stale": len(stale),
        "missing": len(missing),
        "oldest_stale_days": oldest if stale else 0,
        "stale_paths": sorted(stale)[:20],
        "missing_paths": sorted(missing)[:20],
        "scan_complete": bool(reg.get("scan_complete")),
        "last_checked": float(reg.get("last_checked") or 0.0),
    }
