"""存储后端抽象（TODO G1）。

把"读盘 / 落盘"这一层从 :class:`HippoBrain` 与 :class:`VectorStore` 里剥出来，
让 **JSON 多文件** 与 **SQLite 单库** 两种实现可以互换。

为什么只抽象读写、不动内存结构
------------------------------
brain 与 VectorStore 的业务逻辑（检索、合并、去重、裁剪、脏集）全部建立在
内存对象上。把它们一起搬进 SQL，会让一次改动同时横跨"进程内存模型"和
"磁盘格式"两层 —— 出问题分不清是哪层（TODO 纪律：一次只动一层）。
所以这里只替换最外层的 read / write，内存里的 dict / 矩阵 / 脏集一概不动。

由此得到的收益
--------------
* **事务**：一次保存 = 一个事务（JSON 方案在文件系统层没有多文件原子提交的原语，
  10 个层文件只能逐个 ``os.replace``，中途失败必然层间不一致）。
* **行级写**：复用 G0 已经建好的 ``_dirty_ids`` / ``_deleted_ids``，改 1 条只写 1 行，
  取代"改 1 条重写 5.28MB memories.json + 32.7MB npz"。

后端选择
--------
``HIPPOCAMPUS_STORE_BACKEND=json|sqlite``（默认 **json**，与改动前行为完全一致）。
``HIPPOCAMPUS_FTS5=1`` 额外启用 FTS5 全文索引（H4：把 BM25 索引挪上盘）。

⚠️ 老库**不自动迁移**：目录里有 ``memories.json`` 而没有 ``hippocampus.db`` 时，
sqlite 后端会把它当成新库（空），并留一条 warning 提示跑 ``migrate-store``。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# env 开关名（唯一定义处；测试与文档都引用这里的常量）
BACKEND_ENV = "HIPPOCAMPUS_STORE_BACKEND"
FTS5_ENV = "HIPPOCAMPUS_FTS5"

# 构造注入缝：旋钮登记在 config（store_backend / fts5），生产路径由
# engine import 时灌入；解析序（env 先、注入值回退）见 common/config_bridge。
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

DB_FILENAME = "hippocampus.db"
MEMORIES_FILENAME = "memories.json"
VECTORS_SUFFIX = ".vectors.npz"

# G2：WAL 的写串行化 = 协议级的"一个库同一时刻只有一个写者"，
# 比自研 filelock 可靠。busy_timeout 让并发写排队而不是直接抛 SQLITE_BUSY。
_BUSY_TIMEOUT_MS = 60000
# 并发首次建库的重试参数（见 SqliteBackend._init_resilient）。
# 5 次 × 递增间隔 ≈ 0.75s，足够让先到的那个进程把库建完，又不至于
# 在真失败时把调用方挂住。
_INIT_RETRIES = 5
_INIT_RETRY_SLEEP = 0.05


def _is_readonly_error(exc: BaseException) -> bool:
    """是不是"被降级成只读连接"那类错误。

    判据只认 SQLite 的原话，不认我们自己起的名 —— 误判会把真错误（比如
    disk I/O error）也吞进重试循环，那比重试本身更糟。
    """
    return isinstance(exc, sqlite3.OperationalError) and (
        "readonly" in str(exc).lower())


def _is_transient_init_error(exc: BaseException) -> bool:
    """首次建库阶段**可以靠重开连接解决**的错误。

    两类都实测出现过（并发首次建库压测 160 进程）：
      * `attempt to write a readonly database` —— 见 `_init_resilient` 注释；
      * `database is locked` —— 撞在 `PRAGMA journal_mode=WAL` 上：切 WAL 需要
        短暂的独占，另一个进程正建库时它会直接抛，而不是等 busy_timeout
        （PRAGMA 不吃 busy_timeout 这一套）。
    其它错误一律不认，原样抛给调用方。
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "readonly" in msg or "is locked" in msg


def backend_kind() -> str:
    """当前后端种类（默认 json）。未知取值**显式报错**，不静默回退。"""
    raw = (_resolve(BACKEND_ENV, "store_backend", "") or "json").strip().lower()
    if raw not in ("json", "sqlite"):
        raise ValueError(
            f"{BACKEND_ENV} only accepts json or sqlite, got {raw!r}"
        )
    return raw


def fts5_enabled() -> bool:
    """是否启用 FTS5 全文索引（H4）。默认关——检索语义不变，避免影响评测分。"""
    return (_resolve(FTS5_ENV, "fts5", "") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def needs_migration(directory: str) -> bool:
    """目录是 JSON 老库、但当前要按 sqlite 打开 ⇒ 需要先跑 migrate-store。"""
    return (
        backend_kind() == "sqlite"
        and os.path.isfile(os.path.join(directory, MEMORIES_FILENAME))
        and not os.path.isfile(os.path.join(directory, DB_FILENAME))
    )


class StoreBackend:
    """落盘层抽象。

    子类只需实现下面这些原语；业务语义（合并 / 去重 / 裁剪）仍在上层。
    """

    name = "?"
    #: 是否支持行级增量写（json 不支持：它只能整份重写文件）
    supports_incremental = False

    # ------------------------------------------------------------------
    # 派生层（L0 / L2 / L3 …，整块 JSON 语义）
    # ------------------------------------------------------------------
    def read_layer(self, name: str) -> Any | None:
        raise NotImplementedError

    def has_layer(self, name: str) -> bool:
        raise NotImplementedError

    def layer_stamp(self, name: str) -> tuple[int, int] | None:
        """这一层的"版本戳"（判断有没有被别的进程改过）。

        json 用 ``(mtime_ns, size)``；sqlite 没有文件，用 ``(updated_at_ns, 长度)``
        顶上。返回 ``None`` 表示"这一层还不存在"。
        """
        raise NotImplementedError

    def write_layers(self, layers: dict[str, Any]) -> None:
        """写入若干派生层。**要么全成功要么全不写**（事务）。"""
        raise NotImplementedError

    def store_stamp(self) -> tuple[int, int] | None:
        """整库的"磁盘戳"：判断上次落盘之后有没有被别的进程写过。

        json 用 ``memories.json`` 的 ``(mtime_ns, size)``；sqlite 没有这个文件，
        用 ``(写序号, 记录数)`` 顶上 —— 写序号在每个写事务提交前 +1。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # L1 记忆本体
    # ------------------------------------------------------------------
    def load_memories(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """返回 ``(data, issue)``。

        ``data`` 与旧版 memories.json 同构：``{"format","dim","model_key",
        "models","records"}``；目录全新时返回 ``(None, None)``。
        ``issue`` 是加载异常说明（供上层自曝 / 隔离现场），正常时为 ``None``。
        """
        raise NotImplementedError

    def save_memories_full(self, data: dict[str, Any]) -> None:
        """整份覆写 L1（首次落盘、或后端不支持增量时走这条）。"""
        raise NotImplementedError

    def upsert_memories(self, records: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def delete_memories(self, ids: list[str]) -> None:
        raise NotImplementedError

    def save_meta(self, meta: dict[str, Any]) -> None:
        """只更新元数据（format / dim / model_key / models），不动记录。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 向量
    # ------------------------------------------------------------------
    def load_vectors(self) -> dict[str, np.ndarray] | None:
        """返回 ``{vec_key: ndarray}``（vec_key 形如 ``"m0::<id>"``）。"""
        raise NotImplementedError

    def save_vectors_full(self, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def upsert_vectors(self, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def delete_vectors_for(self, memory_ids: list[str]) -> None:
        """删除这些记忆的全部向量（跨所有模型槽位）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 事务 / 生命周期
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """默认：无事务（json 后端逐文件 replace，本来就没有跨文件原子性）。"""
        yield

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 在线快照（可选；只有 sqlite 支持）
    # ------------------------------------------------------------------
    supports_snapshot = False

    def snapshot_to(self, dst: str) -> None:
        """产出一份在线一致快照到 ``dst``。仅 ``supports_snapshot`` 为真的后端实现。

        ``dst`` 必须不存在——SQLite 的 ``VACUUM INTO`` 不覆盖已存在的文件，
        所以调用方要用时间戳命名，撞名就跳过。
        """
        raise NotImplementedError("the current backend does not support online snapshots")

    # ------------------------------------------------------------------
    # FTS5（H4，可选）
    # ------------------------------------------------------------------
    def fts5_supported(self) -> bool:
        return False

    def fts5_sync(self, records: list[dict[str, Any]]) -> None:
        pass

    def fts5_search(self, query: str, top_n: int = 50) -> list[tuple[str, float]]:
        return []


# ======================================================================
# JSON 后端：与改动前完全一致的文件布局
# ======================================================================
class JsonBackend(StoreBackend):
    """多文件 JSON + npz。**默认后端**，行为与 G1 之前逐字节等价。"""

    name = "json"
    supports_incremental = False

    def __init__(self, directory: str) -> None:
        self.dir = directory
        self.memories_path = os.path.join(directory, MEMORIES_FILENAME)
        base, _ = os.path.splitext(self.memories_path)
        self.vectors_path = base + VECTORS_SUFFIX

    # -- 派生层 --------------------------------------------------------
    def read_layer(self, name: str) -> Any | None:
        path = os.path.join(self.dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:      # noqa: BLE001 —— 读不到不阻塞保存
            return None

    def has_layer(self, name: str) -> bool:
        return os.path.isfile(os.path.join(self.dir, name))

    def layer_stamp(self, name: str) -> tuple[int, int] | None:
        path = os.path.join(self.dir, name)
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def store_stamp(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self.memories_path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def write_layers(self, layers: dict[str, Any]) -> None:
        for name, data in layers.items():
            path = os.path.join(self.dir, name)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)

    # -- L1 ------------------------------------------------------------
    def load_memories(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """主文件 → ``.bak`` 回退 → 都读不出则隔离现场并显式报错（缺陷 N1）。

        这段回退逻辑原本在 ``VectorStore._load_impl`` 里，搬到 backend 是为了让
        sqlite 后端也有自己对应的"打不开"处理；json 分支必须**原样保留**
        ——它是"上次 save 崩在备份与替换之间"的唯一自救路径。
        """
        bak_path = self.memories_path + ".bak"
        data: dict[str, Any] | None = None
        main_err: str | None = None
        bak_err: str | None = None

        if os.path.isfile(self.memories_path):
            try:
                with open(self.memories_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                main_err = str(e)

        if data is None and os.path.isfile(bak_path):
            try:
                with open(bak_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.warning("memory data restored from backup %s", bak_path)
            except (json.JSONDecodeError, OSError) as e:
                bak_err = str(e)
                data = None

        if data is not None:
            if main_err is not None:
                # 从备份救回来了：坏的主文件留着，下次 save 会用它盖掉好备份
                q = self._quarantine(self.memories_path)
                return data, {
                    "kind": "recovered",
                    "detail": {"main": main_err},
                    "quarantined": [q] if q else [],
                }
            return data, None

        # 两个文件都不存在 = 全新目录（正常首启）
        if main_err is None and bak_err is None:
            return None, None

        quarantined = []
        if main_err:
            q = self._quarantine(self.memories_path)
            if q:
                quarantined.append(q)
        if bak_err:
            q = self._quarantine(bak_path)
            if q:
                quarantined.append(q)
        logger.error(
            "Memory data is unreadable and no backup is available (main file: %s; backup: %s). "
            "The corrupt file has been quarantined as %s; the original file is still on disk for manual recovery.",
            main_err or "缺失", bak_err or "缺失", quarantined or "无",
        )
        return None, {
            "kind": "unrecoverable",
            "detail": {"main": main_err, "bak": bak_err},
            "quarantined": quarantined,
        }

    def _quarantine(self, path: str) -> str | None:
        """把坏文件改名保留现场（不删）。"""
        if not os.path.isfile(path):
            return None
        dst = f"{path}.corrupt-{int(time.time())}"
        try:
            os.replace(path, dst)
            return dst
        except OSError:
            return None

    def save_memories_full(self, data: dict[str, Any]) -> None:
        directory = os.path.dirname(self.memories_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = self.memories_path + ".tmp"
        bak_path = self.memories_path + ".bak"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(self.memories_path):
            os.replace(self.memories_path, bak_path)
        os.replace(tmp_path, self.memories_path)

    def upsert_memories(self, records: list[dict[str, Any]]) -> None:
        raise NotImplementedError("the json backend can only be rewritten as a whole")

    def delete_memories(self, ids: list[str]) -> None:
        raise NotImplementedError("the json backend can only be rewritten as a whole")

    def save_meta(self, meta: dict[str, Any]) -> None:
        # 元数据与记录同文件 ⇒ 只能整份重写；调用方应改走 save_memories_full
        raise NotImplementedError("the json backend rewrites metadata together with the records as a whole")

    # -- 向量 ----------------------------------------------------------
    def load_vectors(self) -> dict[str, np.ndarray] | None:
        if not os.path.isfile(self.vectors_path):
            return None
        try:
            with np.load(self.vectors_path) as npz:
                return {k: np.asarray(npz[k]) for k in npz.files}
        except Exception as e:      # noqa: BLE001
            logger.warning("failed to load vector file %s: %s", self.vectors_path, e)
            return None

    def save_vectors_full(self, arrays: dict[str, np.ndarray]) -> None:
        tmp = self.vectors_path + ".tmp.npz"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, self.vectors_path)

    def upsert_vectors(self, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError("the json backend can only rewrite npz as a whole")

    def delete_vectors_for(self, memory_ids: list[str]) -> None:
        raise NotImplementedError("the json backend can only rewrite npz as a whole")


# ======================================================================
# SQLite 后端：每库一个 hippocampus.db
# ======================================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS layers (
    name       TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY,
    content    TEXT NOT NULL DEFAULT '',
    data       TEXT NOT NULL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, source_id)
);
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE TABLE IF NOT EXISTS vectors (
    vec_key    TEXT PRIMARY KEY,
    memory_id  TEXT NOT NULL,
    model_slug TEXT NOT NULL DEFAULT '',
    dim        INTEGER NOT NULL DEFAULT 0,
    blob       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_memory ON vectors(memory_id);
CREATE INDEX IF NOT EXISTS idx_sources_src    ON memory_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_entities_ent   ON memory_entities(entity_id);
"""

# H4：BM25 索引挪上盘。三个决定都不是随便定的：
#  1. **trigram 分词**（不是 unicode61）——unicode61 按空格/标点切词，中文
#     整句没有空格 ⇒ 整段被当成一个 token，召回基本失效。trigram 对中文
#     （以及子串匹配）才有效，实测 SQLite 3.45 支持。
#  2. **external content**（content='memories'）——FTS 表不另存正文，省一半
#     空间，且天然与 memories 表同源。
#  3. **触发器维护**——写入/更新/删除自动同步索引，调用方不用记得调 sync。
#     external content 表删除时必须提供**旧值**，所以 ad/au 里要把 old.content
#     一并塞进 'delete' 命令。
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(content, content='memories', content_rowid='rowid',
           tokenize='trigram');
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

#: trigram 分词器要求 term 至少 3 个字符，短于此查不出东西
_FTS_MIN_TERM = 3
# MATCH 表达式里短语数的上限（长查询 3-gram 展开的封顶）
_FTS_MAX_TERMS = 32


class SqliteBackend(StoreBackend):
    """单文件 SQLite。事务 + 行级 upsert，取代多文件整份重写。"""

    name = "sqlite"
    supports_incremental = True
    supports_snapshot = True

    def __init__(self, directory: str) -> None:
        self.dir = directory
        self.db_path = os.path.join(directory, DB_FILENAME)
        os.makedirs(directory, exist_ok=True)
        self._fts = False
        self._fts_tried = False
        # daemon 模式下同一进程有 N 条连接 = N 个线程共享这一个 backend 实例，
        # 而 sqlite3 的 Connection 本身不是线程安全的 ⇒ 连接放开线程检查，
        # 由这把锁把并发串住（与 embeddings.py 的 _ENCODE_LOCK 同一个道理）。
        self._lock = threading.RLock()
        # 事务嵌套深度：brain.save() 会开一个跨 L1 / 派生层的外层事务，
        # 内部的 upsert 各自也会开事务 —— 只有最外层负责 commit / rollback。
        self._txn_depth = 0
        # 写序号：每个写事务提交前 +1，是 sqlite 侧的"磁盘戳"来源
        self._seq = 0
        self._conn = None  # type: ignore[assignment]
        self._init_resilient()

    # -- 连接 ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
        )
        try:
            # WAL：读不阻塞写、写串行化。synchronous=NORMAL 在 WAL 下断电只会丢
            # 最后一个事务，不会损坏库文件（FULL 是每次都 fsync，慢很多）。
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys=OFF")
        except BaseException:
            # PRAGMA 阶段抛错时这条连接已经建好了：不关就是每次重试漏一个句柄。
            # 最典型的就是并发建库时 `journal_mode=WAL` 抛 database is locked
            # （PRAGMA 不吃 busy_timeout，只能重试）—— 见 `_init_resilient`。
            with contextlib.suppress(Exception):
                conn.close()
            raise
        return conn

    def _init_schema(self) -> None:
        with self.transaction():
            self._conn.executescript(_SCHEMA)

    def _init_resilient(self) -> None:
        """连接 + 建表的重试着陆点（Windows 真缺陷，不是抖动）。

        现象（全量回归里 `test_g2_sqlite_concurrency` 偶发红，
        拿到完整堆栈才定位）：两个进程**同时**给一个全新的库目录建库时，后到的
        那个会撞两种错误之一，两种都发生在**建库阶段**、且堆栈都指向别处：

        1. `attempt to write a readonly database` —— Windows 上拿不到写句柄时
           SQLite 不开失败，而是**静默降级成只读连接**；于是报错时机被推迟到
           **第一次写**，而那次写恰好是事务收尾的 `_bump_seq`：

               SqliteBackend.__init__ -> _init_schema -> transaction().__exit__
               -> _bump_seq -> sqlite3.OperationalError

           堆栈里一个"建库"的字样都没有，极易被误判成 flaky 或磁盘问题。

        2. `database is locked` —— 撞在 `_connect` 的
           `PRAGMA journal_mode=WAL` 上：切 WAL 需要短暂独占，另一个进程正建库
           时它直接抛，而且 **PRAGMA 不吃 busy_timeout**（所以把 timeout 调大
           没用，只能重试）。

        取舍：**只**对这两类错误重试（判据见 `_is_transient_init_error`），其它
        错误原样抛出（掩盖真实错误比偶发失败更糟）；每次重试**重开连接**（旧连接
        已经是只读的 / 已经错过 WAL 窗口，在同一条连接上重试多少次都没用）；
        次数用尽就把最后一个异常抛给调用方。schema 全是 `IF NOT EXISTS`，
        重复执行幂等，重开连接不会写坏东西。
        """
        for attempt in range(_INIT_RETRIES):
            try:
                self._conn = self._connect()
                self._init_schema()
                return
            except sqlite3.OperationalError as exc:
                if not _is_transient_init_error(exc):
                    raise
                if attempt >= _INIT_RETRIES - 1:
                    raise
                logger.warning(
                    "sqlite first-time creation hit concurrency (attempt %d: %s); reopening connection and retrying",
                    attempt + 1, exc,
                )
                time.sleep(_INIT_RETRY_SLEEP * (attempt + 1))
                if self._conn is not None:
                    with contextlib.suppress(sqlite3.Error):
                        self._conn.close()
                    self._conn = None

    @property
    def _fts_ok(self) -> bool:
        if self._fts_tried:
            return self._fts
        self._fts_tried = True
        if not fts5_enabled():
            self._fts = False
            return False
        try:
            self._conn.executescript(_FTS_SCHEMA)
            self._conn.executescript(_FTS_TRIGGERS)
            # 存量回填：FTS 表可能是后建的，此时 memories 里已有数据。
            # rebuild 是 FTS5 的内置命令，幂等，只在首次建表时真正干活。
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self._conn.commit()
            self._fts = True
        except sqlite3.Error as e:
            # FTS5 不可用（裁剪版 sqlite 没有该模块）不该让整个库打不开
            logger.warning("FTS5 unavailable; BM25 still built in memory: %s", e)
            self._fts = False
        return self._fts

    # -- FTS5 全文检索（H4）--------------------------------------------
    def fts5_sync(self, records: list[dict[str, Any]]) -> None:
        """索引由触发器自动维护，这里只做**可用性与存量回填**检查。

        保留这个方法是因为上层需要一个"我现在能不能用 FTS"的显式入口；
        它不是同步动作，误以为它是同步动作会导致"写了但没索引"的错觉。
        """
        if self._fts_ok:
            self._conn.execute(
                "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"
            )
            self._conn.commit()

    def fts5_query_terms(self, query: str) -> str | None:
        """把用户查询转成 FTS5 MATCH 表达式；**不能用就返回 None**。

        返回 None 的场景必须由调用方回退到内存 BM25，绝不能静默空召回：
        * trigram 分词器要求 term ≥ 3 字符，短词查不出东西；
        * 空查询 / 全是短词。

        中文 3-gram 展开：中文连续段不再整串当一个短语——
        无空格整句（`崇祯皇帝煤山自缢`）在库里没有逐字同串 ⇒ FTS 恒 `[]`，
        而调用方按设计**不回退** ⇒ 字面通道整个丢失。现在把每个含中文的段
        切成**滑动 3-gram OR 展开**（`煤山自缢` → `"煤山自" OR "山自缢"`），
        trigram 索引可直接吃；任一 gram 命中即候选，精度由 bm25 排序 +
        融合打分兜底。纯 ASCII 段保持整段短语（原行为）。2 字中文词依旧
        进不了 trigram（物理限制），全短词查询仍回退内存 BM25。
        """
        if not query or not self._fts_ok:
            return None
        # 去掉 FTS5 语法敏感字符，避免用户一句带引号的话直接把查询打成语法错误
        cleaned = "".join(
            ch if (ch.isalnum() or ch.isspace() or "\u4e00" <= ch <= "\u9fff") else " "
            for ch in query
        )
        # 每个 ≥3 字符的片段产出一组短语；含中文的段做滑动 3-gram 展开。
        terms: list[str] = []
        seen: set[str] = set()
        for seg in cleaned.split():
            if len(seg) < _FTS_MIN_TERM:
                continue
            if any("\u4e00" <= ch <= "\u9fff" for ch in seg):
                grams = [seg[i: i + _FTS_MIN_TERM]
                         for i in range(len(seg) - _FTS_MIN_TERM + 1)]
            else:
                grams = [seg]
            for g in grams:
                if g and g not in seen:
                    seen.add(g)
                    terms.append(g)
        if not terms:
            return None
        # 封顶：超长查询的 gram 数线性膨胀，MATCH 表达式太长既慢也无增益
        terms = terms[:_FTS_MAX_TERMS]
        # 每个 term 加引号当短语查（中文整串本身就是一个短语）
        return " OR ".join('"%s"' % t.replace('"', "") for t in terms)

    def fts5_search(
        self, query: str, top_n: int = 50
    ) -> list[tuple[str, float]] | None:
        """全文检索，返回 ``[(memory_id, score), ...]``（按相关度降序）。

        **返回值要区分两种情况**，这个区别决定了内存能不能真省下来：
        * ``None`` = 这条路走不了（未启用 / 词太短 / 语法错误）⇒ 调用方
          **必须**回退内存 BM25；
        * ``[]`` = 索引可用、查询合法，就是没有命中 ⇒ 调用方**不要**回退
          （回退就会去建内存索引，省内存这件事当场作废）。
        """
        match = self.fts5_query_terms(query)
        if not match:
            return None
        try:
            rows = self._conn.execute(
                "SELECT m.id, bm25(memories_fts) AS s "
                "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                "WHERE memories_fts MATCH ? ORDER BY s LIMIT ?",
                (match, int(top_n)),
            ).fetchall()
        except sqlite3.Error as e:
            # 语法/分词器错误不该让召回整条路挂掉：回退内存 BM25
            logger.warning("FTS5 query failed (falling back to in-memory BM25): %s | %s", query, e)
            return None
        # bm25() 返回的是**负的**相关度（越小越相关）⇒ 取反还原
        return [(r[0], -float(r[1])) for r in rows]

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """写事务：``BEGIN IMMEDIATE`` 立刻取写锁，避免升级锁时被动等。

        可重入：嵌套调用时不新开事务，交给最外层 commit / rollback —— 这样
        ``brain.save()`` 才能把 L1（manager.store）与 10 个派生层包进
        **同一个**事务里，做到"四层同生共死"。
        """
        with self._lock:
            if self._txn_depth > 0:
                self._txn_depth += 1
                try:
                    yield
                finally:
                    self._txn_depth -= 1
                return
            self._txn_depth = 1
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield
                self._bump_seq()
                self._conn.commit()
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.rollback()
                raise
            finally:
                self._txn_depth = 0

    # -- 派生层 --------------------------------------------------------
    def read_layer(self, name: str) -> Any | None:
        row = self._conn.execute(
            "SELECT data FROM layers WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def has_layer(self, name: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM layers WHERE name=?", (name,)
        ).fetchone() is not None

    def layer_stamp(self, name: str) -> tuple[int, int] | None:
        row = self._conn.execute(
            "SELECT updated_at, data FROM layers WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        # 纳秒戳 + 长度，语义对齐 json 的 (mtime_ns, size)
        return (int(float(row[0]) * 1e9), len(row[1] or ""))

    def _bump_seq(self) -> None:
        """写序号 +1。每个写事务提交前调用一次（读事务不调用）。"""
        self._seq += 1
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('save_seq', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(self._seq),),
        )

    def store_stamp(self) -> tuple[int, int] | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='save_seq'"
        ).fetchone()
        try:
            seq = int(json.loads(row[0])) if row else 0
        except (ValueError, TypeError):
            seq = 0
        cnt = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return (seq, int(cnt))

    def snapshot_to(self, dst: str) -> None:
        """``VACUUM INTO``：在一个读事务里产出一致副本，不停服、不用拷文件。

        为什么用**独立连接**：``VACUUM`` 不能在活跃事务里执行，而 ``self._conn``
        此刻可能正被 ``brain.save()`` 的外层事务持有；独立连接还能让读者与写者
        互不阻塞（WAL 下读不阻塞写）。

        ``dst`` 已存在时 SQLite 会报错 —— 由调用方用时间戳命名避开。
        """
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        try:
            conn.execute("VACUUM INTO ?", (dst,))
        finally:
            conn.close()

    def write_layers(self, layers: dict[str, Any]) -> None:
        """只写内容变了的层（逐层比对），整批仍在**一个事务**里。"""
        now = time.time()
        with self.transaction():
            for name, data in layers.items():
                blob = json.dumps(data, ensure_ascii=False)
                row = self._conn.execute(
                    "SELECT data FROM layers WHERE name=?", (name,)
                ).fetchone()
                if row is not None and row[0] == blob:
                    continue      # 没变，不写
                self._conn.execute(
                    "INSERT INTO layers(name, data, updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(name) DO UPDATE SET data=excluded.data, "
                    "updated_at=excluded.updated_at",
                    (name, blob, now),
                )

    # -- L1 ------------------------------------------------------------
    def load_memories(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        try:
            meta = {
                k: json.loads(v)
                for k, v in self._conn.execute(
                    "SELECT key, value FROM meta"
                ).fetchall()
            }
        except (sqlite3.Error, json.JSONDecodeError) as e:
            return None, {"kind": "unrecoverable", "detail": {"db": str(e)},
                          "quarantined": []}
        if not meta and not self._conn.execute(
            "SELECT 1 FROM memories LIMIT 1"
        ).fetchone():
            return None, None      # 全新库
        rows = self._conn.execute("SELECT data FROM memories").fetchall()
        records = []
        for (blob,) in rows:
            try:
                records.append(json.loads(blob))
            except json.JSONDecodeError:
                continue
        data = {
            "format": meta.get("format", 2),
            "dim": meta.get("dim", 0),
            "model_key": meta.get("model_key", ""),
            "models": meta.get("models", {}),
            "records": records,
        }
        return data, None

    def save_memories_full(self, data: dict[str, Any]) -> None:
        records = data.get("records") or []
        with self.transaction():
            self._write_meta({
                "format": data.get("format", 2),
                "dim": data.get("dim", 0),
                "model_key": data.get("model_key", ""),
                "models": data.get("models", {}),
            })
            self._conn.execute("DELETE FROM memories")
            self._conn.execute("DELETE FROM memory_sources")
            self._conn.execute("DELETE FROM memory_entities")
            self._replace_records(records)

    def _write_meta(self, meta: dict[str, Any]) -> None:
        for k, v in meta.items():
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (k, json.dumps(v, ensure_ascii=False)),
            )

    def _replace_records(self, records: list[dict[str, Any]]) -> None:
        now = time.time()
        for r in records:
            rid = r.get("id")
            if not rid:
                continue
            self._conn.execute(
                "INSERT INTO memories(id, content, data, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET content=excluded.content, "
                "data=excluded.data, updated_at=excluded.updated_at",
                (rid, r.get("content", ""),
                 json.dumps(r, ensure_ascii=False), now),
            )
            self._conn.execute(
                "DELETE FROM memory_sources WHERE memory_id=?", (rid,)
            )
            for sid in r.get("source_ids") or []:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_sources(memory_id, source_id) "
                    "VALUES(?,?)", (rid, sid),
                )
            self._conn.execute(
                "DELETE FROM memory_entities WHERE memory_id=?", (rid,)
            )
            for eid in r.get("entity_ids") or []:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_entities(memory_id, entity_id) "
                    "VALUES(?,?)", (rid, eid),
                )
        # 🔴 不要在这里手工同步 FTS 索引：memories_fts 是 external content
        # 表（只有 content 列），索引由 memories_fts_ai/au/ad 三个触发器
        # 随 memories 的写入自动维护。手工同步既多余又会抛
        # "no such column: memory_id"。

    def upsert_memories(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self.transaction():
            self._replace_records(records)

    def delete_memories(self, ids: list[str]) -> None:
        if not ids:
            return
        with self.transaction():
            for rid in ids:
                self._conn.execute("DELETE FROM memories WHERE id=?", (rid,))
                self._conn.execute(
                    "DELETE FROM memory_sources WHERE memory_id=?", (rid,)
                )
                self._conn.execute(
                    "DELETE FROM memory_entities WHERE memory_id=?", (rid,)
                )
                self._conn.execute(
                    "DELETE FROM vectors WHERE memory_id=?", (rid,)
                )
                # 🔴 FTS 索引**不要**在这里手动删：H4 之后 memories_fts 是
                # external content 表，列只有 content（没有 memory_id），上面
                # 那行旧代码会抛 "no such column: memory_id"。索引由
                # memories_fts_ad 触发器随 memories 的删除自动维护。

    def save_meta(self, meta: dict[str, Any]) -> None:
        with self.transaction():
            self._write_meta(meta)

    # -- 向量 ----------------------------------------------------------
    def load_vectors(self) -> dict[str, np.ndarray] | None:
        rows = self._conn.execute(
            "SELECT vec_key, blob FROM vectors"
        ).fetchall()
        if not rows:
            return None
        out: dict[str, np.ndarray] = {}
        for key, blob in rows:
            out[key] = np.frombuffer(blob, dtype=np.float32).copy()
        return out

    def save_vectors_full(self, arrays: dict[str, np.ndarray]) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM vectors")
            self._put_vectors(arrays)

    def upsert_vectors(self, arrays: dict[str, np.ndarray]) -> None:
        if not arrays:
            return
        with self.transaction():
            self._put_vectors(arrays)

    def _put_vectors(self, arrays: dict[str, np.ndarray]) -> None:
        for key, arr in arrays.items():
            vec = np.asarray(arr, dtype=np.float32)
            # vec_key 形如 "m0::<id>"（与 npz 的键一致，保证两条路径同构）
            memory_id = key.split("::", 1)[1] if "::" in key else key
            slug = key.split("::", 1)[0] if "::" in key else ""
            self._conn.execute(
                "INSERT INTO vectors(vec_key, memory_id, model_slug, dim, blob) "
                "VALUES(?,?,?,?,?) ON CONFLICT(vec_key) DO UPDATE SET "
                "memory_id=excluded.memory_id, model_slug=excluded.model_slug, "
                "dim=excluded.dim, blob=excluded.blob",
                (key, memory_id, slug, int(vec.shape[0]),
                 vec.tobytes()),
            )

    def delete_vectors_for(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with self.transaction():
            for rid in memory_ids:
                self._conn.execute(
                    "DELETE FROM vectors WHERE memory_id=?", (rid,)
                )

    # -- FTS5 ----------------------------------------------------------
    # 🔴 本类里 FTS5 的**唯一实现在 _fts_ok 之后**（trigram + external
    # content + 触发器）。这里只留一个可用性查询入口。
    # 历史上这里还放着一份"手工维护 memory_id 列"的旧实现，它按 unicode61
    # 建表、手动 delete/insert 索引；因为**后定义覆盖前定义**，那份旧代码
    # 会把新实现整个盖掉（表现为 trigram 一直不生效、删索引抛
    # "no such column: memory_id"）。改 FTS5 相关代码时先确认只有一份。
    def fts5_supported(self) -> bool:
        return self._fts_ok

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


# 进程内按目录缓存。**必须**让 brain 与 manager.store 拿到同一个实例：
# 否则它们各持一条连接、各开各的事务，"一次 save = 一个事务"就无从
# 谈起 —— 崩溃时必然出现"L1 写了、派生层没写"这种夹层状态。
_BACKEND_CACHE: dict[str, StoreBackend] = {}
_CACHE_LOCK = threading.Lock()


def reset_backend_cache(directory: str | None = None) -> None:
    """丢弃缓存（测试用：临时库被删掉重建后必须调，否则会拿到指向已删文件的连接）。"""
    with _CACHE_LOCK:
        if directory is None:
            _BACKEND_CACHE.clear()
        else:
            _BACKEND_CACHE.pop(os.path.abspath(directory), None)


def get_backend(directory: str) -> StoreBackend:
    """按 env 构造后端（同目录复用同一实例）。老库按 sqlite 打开时只 warning。"""
    key = os.path.abspath(directory)
    with _CACHE_LOCK:
        cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    if needs_migration(directory):
        logger.warning(
            "Directory %s is an old JSON library without %s: opening it as sqlite yields an empty library. "
            "Run migrate-store first; old libraries are not migrated automatically.",
            directory, DB_FILENAME,
        )
    backend: StoreBackend
    if backend_kind() == "sqlite":
        backend = SqliteBackend(directory)
    else:
        backend = JsonBackend(directory)
    with _CACHE_LOCK:
        _BACKEND_CACHE[key] = backend
    return backend
