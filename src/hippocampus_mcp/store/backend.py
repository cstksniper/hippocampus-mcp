"""Storage backend abstraction (TODO G1).

Peels the "read from disk / persist to disk" layer out of :class:`HippoBrain` and
:class:`VectorStore`, so that the **multi-file JSON** and **single-file SQLite**
implementations are interchangeable.

Why only read/write is abstracted, not the in-memory structures
---------------------------------------------------------------
The business logic of brain and VectorStore (retrieval, merging, dedup, trimming,
dirty sets) is entirely built on in-memory objects. Moving those into SQL too would
make one change straddle two layers at once — "process memory model" and
"disk format" — and when something breaks you cannot tell which layer (TODO
discipline: change one layer at a time).
So only the outermost read / write is replaced; the in-memory dicts / matrices /
dirty sets are untouched.

Benefits this yields
--------------------
* **Transactions**: one save = one transaction (the JSON approach has no
  multi-file atomic-commit primitive at the filesystem level; the 10 layer files
  can only be ``os.replace``-ed one by one, and a mid-way failure inevitably
  leaves layers inconsistent).
* **Row-level writes**: reuses the ``_dirty_ids`` / ``_deleted_ids`` already built
  in G0; changing 1 record writes 1 row, replacing "change 1 record and rewrite
  the 5.28MB memories.json + 32.7MB npz".

Backend selection
-----------------
``HIPPOCAMPUS_STORE_BACKEND=json|sqlite`` (default **json**, behavior identical to
before the change).
``HIPPOCAMPUS_FTS5=1`` additionally enables the FTS5 full-text index (H4: moving
the BM25 index onto disk).

⚠️ Old libraries are **not migrated automatically**: when the directory contains
``memories.json`` but no ``hippocampus.db``, the sqlite backend treats it as a new
(empty) library and leaves a warning suggesting you run ``migrate-store``.
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

# env switch names (the single definition site; tests and docs both reference these constants)
BACKEND_ENV = "HIPPOCAMPUS_STORE_BACKEND"
FTS5_ENV = "HIPPOCAMPUS_FTS5"

# Construction-time injection seam: the knobs are registered in config
# (store_backend / fts5) and filled in by the engine at import time in the
# production path; the resolution order (env first, injected value as fallback)
# is in common/config_bridge.
from ..common.config_bridge import apply_config, resolve as _resolve  # noqa: E402

DB_FILENAME = "hippocampus.db"
MEMORIES_FILENAME = "memories.json"
VECTORS_SUFFIX = ".vectors.npz"

# G2: WAL's write serialization = protocol-level "one library has at most one
# writer at a time", more reliable than a home-grown filelock. busy_timeout makes
# concurrent writes queue up instead of directly raising SQLITE_BUSY.
_BUSY_TIMEOUT_MS = 60000
# Retry parameters for concurrent first-time DB creation (see SqliteBackend._init_resilient).
# 5 tries × increasing intervals ≈ 0.75s — enough for the first-arriving process
# to finish creating the library, without hanging the caller on a real failure.
_INIT_RETRIES = 5
_INIT_RETRY_SLEEP = 0.05


def _is_readonly_error(exc: BaseException) -> bool:
    """Is this the "silently downgraded to a read-only connection" kind of error.

    The criterion trusts only SQLite's own words, not names we made up — a
    misjudgment would swallow real errors (e.g. disk I/O error) into the retry
    loop too, which is worse than the retrying itself.
    """
    return isinstance(exc, sqlite3.OperationalError) and (
        "readonly" in str(exc).lower())


def _is_transient_init_error(exc: BaseException) -> bool:
    """Errors during first-time DB creation that **can be fixed by reopening the connection**.

    Both kinds actually occurred (160-process stress test of concurrent first-time
    creation):
      * `attempt to write a readonly database` —— see the `_init_resilient` comment;
      * `database is locked` —— hits `PRAGMA journal_mode=WAL`: switching to WAL
        needs brief exclusivity; when another process is creating the library it
        raises directly instead of waiting for busy_timeout
        (PRAGMA does not honor busy_timeout).
    All other errors are not recognized and are raised to the caller as-is.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "readonly" in msg or "is locked" in msg


def backend_kind() -> str:
    """The current backend kind (default json). Unknown values **raise explicitly**, no silent fallback."""
    raw = (_resolve(BACKEND_ENV, "store_backend", "") or "json").strip().lower()
    if raw not in ("json", "sqlite"):
        raise ValueError(
            f"{BACKEND_ENV} only accepts json or sqlite, got {raw!r}"
        )
    return raw


def fts5_enabled() -> bool:
    """Whether the FTS5 full-text index is enabled (H4). Off by default — retrieval semantics unchanged, to avoid affecting benchmark scores."""
    return (_resolve(FTS5_ENV, "fts5", "") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def needs_migration(directory: str) -> bool:
    """The directory is an old JSON library but would be opened as sqlite ⇒ migrate-store must run first."""
    return (
        backend_kind() == "sqlite"
        and os.path.isfile(os.path.join(directory, MEMORIES_FILENAME))
        and not os.path.isfile(os.path.join(directory, DB_FILENAME))
    )


class StoreBackend:
    """The persistence-layer abstraction.

    Subclasses only need to implement the primitives below; business semantics
    (merge / dedup / trim) remain upstream.
    """

    name = "?"
    #: whether row-level incremental writes are supported (json does not: it can only rewrite whole files)
    supports_incremental = False

    # ------------------------------------------------------------------
    # Derived layers (L0 / L2 / L3 …, whole-blob JSON semantics)
    # ------------------------------------------------------------------
    def read_layer(self, name: str) -> Any | None:
        raise NotImplementedError

    def has_layer(self, name: str) -> bool:
        raise NotImplementedError

    def layer_stamp(self, name: str) -> tuple[int, int] | None:
        """This layer's "version stamp" (to judge whether another process changed it).

        json uses ``(mtime_ns, size)``; sqlite has no file and substitutes
        ``(updated_at_ns, length)``. Returning ``None`` means "this layer does not
        exist yet".
        """
        raise NotImplementedError

    def write_layers(self, layers: dict[str, Any]) -> None:
        """Write several derived layers. **All succeed or none is written** (transactional)."""
        raise NotImplementedError

    def store_stamp(self) -> tuple[int, int] | None:
        """The whole library's "disk stamp": to judge whether another process wrote after the last persist.

        json uses ``memories.json``'s ``(mtime_ns, size)``; sqlite has no such file
        and substitutes ``(write sequence number, record count)`` — the sequence
        number is +1'd before every write transaction commits.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # L1 memory bodies
    # ------------------------------------------------------------------
    def load_memories(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return ``(data, issue)``.

        ``data`` is isomorphic to the old memories.json: ``{"format","dim","model_key",
        "models","records"}``; a brand-new directory returns ``(None, None)``.
        ``issue`` describes a load anomaly (for the upper layer to self-report /
        quarantine the scene) and is ``None`` when normal.
        """
        raise NotImplementedError

    def save_memories_full(self, data: dict[str, Any]) -> None:
        """Overwrite L1 in full (used for the first persist, or when the backend does not support incremental writes)."""
        raise NotImplementedError

    def upsert_memories(self, records: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def delete_memories(self, ids: list[str]) -> None:
        raise NotImplementedError

    def save_meta(self, meta: dict[str, Any]) -> None:
        """Update only metadata (format / dim / model_key / models), leaving records untouched."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Vectors
    # ------------------------------------------------------------------
    def load_vectors(self) -> dict[str, np.ndarray] | None:
        """Return ``{vec_key: ndarray}`` (vec_key looks like ``"m0::<id>"``)."""
        raise NotImplementedError

    def save_vectors_full(self, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def upsert_vectors(self, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def delete_vectors_for(self, memory_ids: list[str]) -> None:
        """Delete all vectors of these memories (across all model slots)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Transactions / lifecycle
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Default: no transaction (the json backend replaces file by file; there is no cross-file atomicity to begin with)."""
        yield

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Online snapshot (optional; only sqlite supports it)
    # ------------------------------------------------------------------
    supports_snapshot = False

    def snapshot_to(self, dst: str) -> None:
        """Produce an online consistent snapshot at ``dst``. Implemented only by backends with ``supports_snapshot`` true.

        ``dst`` must not exist — SQLite's ``VACUUM INTO`` does not overwrite an
        existing file, so callers should name it with a timestamp and skip on
        collision.
        """
        raise NotImplementedError("the current backend does not support online snapshots")

    # ------------------------------------------------------------------
    # FTS5 (H4, optional)
    # ------------------------------------------------------------------
    def fts5_supported(self) -> bool:
        return False

    def fts5_sync(self, records: list[dict[str, Any]]) -> None:
        pass

    def fts5_search(self, query: str, top_n: int = 50) -> list[tuple[str, float]]:
        return []


# ======================================================================
# JSON backend: file layout identical to before the change
# ======================================================================
class JsonBackend(StoreBackend):
    """Multi-file JSON + npz. The **default backend**; behavior is byte-for-byte equivalent to pre-G1."""

    name = "json"
    supports_incremental = False

    def __init__(self, directory: str) -> None:
        self.dir = directory
        self.memories_path = os.path.join(directory, MEMORIES_FILENAME)
        base, _ = os.path.splitext(self.memories_path)
        self.vectors_path = base + VECTORS_SUFFIX

    # -- Derived layers --------------------------------------------------------
    def read_layer(self, name: str) -> Any | None:
        path = os.path.join(self.dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:      # noqa: BLE001 —— an unreadable layer does not block saving
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
        """Main file → ``.bak`` fallback → if neither can be read, quarantine the scene and raise explicitly (defect N1).

        This fallback logic originally lived in ``VectorStore._load_impl``; moving it
        into the backend gives the sqlite backend its own corresponding "cannot
        open" handling. The json branch must be **preserved as-is** — it is the
        only self-rescue path for "the last save crashed between backup and
        replace".
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
                # recovered from the backup: the corrupt main file must not stay in place — the next save would use it to overwrite the good backup
                q = self._quarantine(self.memories_path)
                return data, {
                    "kind": "recovered",
                    "detail": {"main": main_err},
                    "quarantined": [q] if q else [],
                }
            return data, None

        # neither file exists = brand-new directory (normal first start)
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
        """Rename the corrupt file to preserve the scene (do not delete)."""
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
        # metadata lives in the same file as the records ⇒ only a full rewrite is possible; callers should switch to save_memories_full
        raise NotImplementedError("the json backend rewrites metadata together with the records as a whole")

    # -- Vectors ----------------------------------------------------------
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
# SQLite backend: one hippocampus.db per library
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

# H4: moving the BM25 index onto disk. None of the three decisions is arbitrary:
#  1. **trigram tokenizer** (not unicode61) — unicode61 splits on spaces/punctuation;
#     a Chinese sentence has no spaces ⇒ a whole paragraph becomes one token and
#     recall basically dies. trigram works for Chinese (and for substring matching);
#     verified on SQLite 3.45.
#  2. **external content** (content='memories') — the FTS table does not store the
#     body text separately, halving the space, and it is naturally same-source with
#     the memories table.
#  3. **Trigger maintenance** — inserts/updates/deletes sync the index automatically;
#     callers need not remember to call sync.
#     An external content table requires the **old value** on delete, so the ad/au
#     triggers must also feed old.content into the 'delete' command.
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

#: the trigram tokenizer requires a term of at least 3 characters; shorter terms find nothing
_FTS_MIN_TERM = 3
# cap on the number of phrases in a MATCH expression (upper bound for long queries' 3-gram expansion)
_FTS_MAX_TERMS = 32


class SqliteBackend(StoreBackend):
    """Single-file SQLite. Transactions + row-level upsert, replacing multi-file full rewrites."""

    name = "sqlite"
    supports_incremental = True
    supports_snapshot = True

    def __init__(self, directory: str) -> None:
        self.dir = directory
        self.db_path = os.path.join(directory, DB_FILENAME)
        os.makedirs(directory, exist_ok=True)
        self._fts = False
        self._fts_tried = False
        # in daemon mode one process holds N connections = N threads sharing this
        # single backend instance, and sqlite3's Connection is not thread-safe by
        # itself ⇒ the connection disables the thread check, and this lock
        # serializes the concurrency (same reasoning as embeddings.py's _ENCODE_LOCK).
        self._lock = threading.RLock()
        # transaction nesting depth: brain.save() opens an outer transaction spanning
        # L1 / derived layers, and the inner upserts open transactions of their own —
        # only the outermost level is responsible for commit / rollback.
        self._txn_depth = 0
        # write sequence number: +1 before every write transaction commits; the
        # source of sqlite's "disk stamp"
        self._seq = 0
        self._conn = None  # type: ignore[assignment]
        self._init_resilient()

    # -- Connection ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
        )
        try:
            # WAL: reads do not block writes, writes are serialized. Under WAL,
            # synchronous=NORMAL loses only the last transaction on power loss and
            # does not corrupt the library file (FULL fsyncs every time, much slower).
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys=OFF")
        except BaseException:
            # when the PRAGMA stage raises, this connection is already open: not
            # closing it leaks one handle per retry. The most typical case is
            # `journal_mode=WAL` raising database is locked during concurrent
            # first-time creation (PRAGMA does not honor busy_timeout; only retrying
            # works) —— see `_init_resilient`.
            with contextlib.suppress(Exception):
                conn.close()
            raise
        return conn

    def _init_schema(self) -> None:
        with self.transaction():
            self._conn.executescript(_SCHEMA)

    def _init_resilient(self) -> None:
        """The retry landing point for connect + schema creation (a real Windows defect, not jitter).

        Symptom (sporadic red for `test_g2_sqlite_concurrency` in the full
        regression; only localized once the full stack was captured): when two
        processes **simultaneously** create the library in a brand-new directory,
        the later-arriving one hits one of two errors; both occur in the
        **creation phase** and both stacks point somewhere else:

        1. `attempt to write a readonly database` —— on Windows, when SQLite cannot
           get a write handle it does not fail to open; instead it **silently
           downgrades to a read-only connection**. The error is therefore deferred
           to the **first write**, and that write happens to be the
           transaction-finalizing `_bump_seq`:

               SqliteBackend.__init__ -> _init_schema -> transaction().__exit__
               -> _bump_seq -> sqlite3.OperationalError

           Nowhere in the stack does "creating the library" appear, so it is easily
           misjudged as flaky or a disk problem.

        2. `database is locked` —— hits `_connect`'s
           `PRAGMA journal_mode=WAL`: switching to WAL needs brief exclusivity;
           when another process is creating the library it raises directly, and
           **PRAGMA does not honor busy_timeout** (so raising the timeout does not
           help; only retrying does).

        Trade-offs: retry **only** these two error kinds (criterion in
        `_is_transient_init_error`); every other error is raised as-is (masking
        real errors is worse than a sporadic failure); each retry **reopens the
        connection** (the old one is already read-only / has already missed the WAL
        window — retrying on the same connection is useless no matter how many
        times); when retries are exhausted, the last exception goes to the caller.
        The schema is all `IF NOT EXISTS`; repeated execution is idempotent, and
        reopening the connection cannot corrupt anything.
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
            # backfill for existing rows: the FTS table may have been created later,
            # when memories already held data. rebuild is an FTS5 built-in command,
            # idempotent, doing real work only on first table creation.
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self._conn.commit()
            self._fts = True
        except sqlite3.Error as e:
            # FTS5 being unavailable (a trimmed-down sqlite lacking the module) must not keep the whole library from opening
            logger.warning("FTS5 unavailable; BM25 still built in memory: %s", e)
            self._fts = False
        return self._fts

    # -- FTS5 full-text search (H4)--------------------------------------------
    def fts5_sync(self, records: list[dict[str, Any]]) -> None:
        """The index is maintained automatically by triggers; this only performs the
        **availability and backfill** check.

        The method is kept because the upper layer needs an explicit "can I use FTS
        right now" entry point; it is not a sync action — mistaking it for one
        creates the illusion of "written but not indexed".
        """
        if self._fts_ok:
            self._conn.execute(
                "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"
            )
            self._conn.commit()

    def fts5_query_terms(self, query: str) -> str | None:
        """Convert a user query into an FTS5 MATCH expression; **return None when unusable**.

        When None is returned the caller must fall back to in-memory BM25 — never a
        silent empty recall:
        * the trigram tokenizer requires terms ≥ 3 chars; short words find nothing;
        * empty query / nothing but short words.

        Chinese 3-gram expansion: a continuous Chinese segment is no longer treated
        as one whole-string phrase — a space-free sentence (`崇祯皇帝煤山自缢`) has
        no verbatim identical string in the library ⇒ FTS returns `[]` forever, and
        the caller by design **does not fall back** ⇒ the literal channel is lost
        entirely. Now every Chinese-containing segment is cut into a **sliding
        3-gram OR expansion** (`煤山自缢` → `"煤山自" OR "山自缢"`) that the trigram
        index can consume directly; any gram hit becomes a candidate, with precision
        backstopped by bm25 ranking + fusion scoring. Pure-ASCII segments keep the
        whole-segment phrase (original behavior). 2-char Chinese words still cannot
        enter trigram (a physical limitation), and all-short-word queries still fall
        back to in-memory BM25.
        """
        if not query or not self._fts_ok:
            return None
        # strip FTS5 syntax-sensitive characters so a user sentence with quotes does not turn the query into a syntax error outright
        cleaned = "".join(
            ch if (ch.isalnum() or ch.isspace() or "\u4e00" <= ch <= "\u9fff") else " "
            for ch in query
        )
        # every segment ≥3 chars produces a group of phrases; Chinese-containing segments get the sliding 3-gram expansion.
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
        # cap: gram counts of over-long queries grow linearly; an over-long MATCH expression is both slow and pointless
        terms = terms[:_FTS_MAX_TERMS]
        # quote each term and search it as a phrase (a whole Chinese string is itself a phrase)
        return " OR ".join('"%s"' % t.replace('"', "") for t in terms)

    def fts5_search(
        self, query: str, top_n: int = 50
    ) -> list[tuple[str, float]] | None:
        """Full-text search, returning ``[(memory_id, score), ...]`` (relevance descending).

        **The return value distinguishes two cases**, and this distinction decides
        whether memory can actually be saved:
        * ``None`` = this path is unusable (not enabled / terms too short / syntax
          error) ⇒ the caller **must** fall back to in-memory BM25;
        * ``[]`` = the index is usable and the query is legal, just no hits ⇒ the
          caller must **not** fall back (falling back would build the in-memory
          index, voiding the memory savings on the spot).
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
            # syntax/tokenizer errors must not take down the whole recall path: fall back to in-memory BM25
            logger.warning("FTS5 query failed (falling back to in-memory BM25): %s | %s", query, e)
            return None
        # bm25() returns **negative** relevance (smaller = more relevant) ⇒ negate to restore
        return [(r[0], -float(r[1])) for r in rows]

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Write transaction: ``BEGIN IMMEDIATE`` takes the write lock right away, avoiding passive waiting on lock upgrade.

        Re-entrant: nested calls do not open a new transaction; the outermost level
        does commit / rollback — this is what lets ``brain.save()`` wrap L1
        (manager.store) and the 10 derived layers into **one** transaction, making
        "the four layers live and die together".
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

    # -- Derived layers --------------------------------------------------------
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
        # nanosecond stamp + length, semantically aligned with json's (mtime_ns, size)
        return (int(float(row[0]) * 1e9), len(row[1] or ""))

    def _bump_seq(self) -> None:
        """Write sequence number +1. Called once before each write transaction commits (read transactions do not call it)."""
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
        """``VACUUM INTO``: produce a consistent copy inside a read transaction, without stopping service or copying files.

        Why an **independent connection**: ``VACUUM`` cannot run inside an active
        transaction, and ``self._conn`` may at this moment be held by
        ``brain.save()``'s outer transaction; an independent connection also lets
        reader and writer avoid blocking each other (under WAL, reads do not block
        writes).

        SQLite raises when ``dst`` already exists —— callers avoid it by naming
        with a timestamp.
        """
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        try:
            conn.execute("VACUUM INTO ?", (dst,))
        finally:
            conn.close()

    def write_layers(self, layers: dict[str, Any]) -> None:
        """Write only the layers whose content changed (compared layer by layer); the whole batch still runs in **one transaction**."""
        now = time.time()
        with self.transaction():
            for name, data in layers.items():
                blob = json.dumps(data, ensure_ascii=False)
                row = self._conn.execute(
                    "SELECT data FROM layers WHERE name=?", (name,)
                ).fetchone()
                if row is not None and row[0] == blob:
                    continue      # unchanged, do not write
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
            return None, None      # brand-new library
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
        # 🔴 Do not manually sync the FTS index here: memories_fts is an external
        # content table (the only column is content); the index is maintained
        # automatically by the three triggers memories_fts_ai/au/ad as memories is
        # written. Manual syncing is both redundant and raises
        # "no such column: memory_id".

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
                # 🔴 Do **not** manually delete from the FTS index here: after H4,
                # memories_fts is an external content table whose only column is
                # content (no memory_id); the old code line above would raise
                # "no such column: memory_id". The index is maintained automatically
                # by the memories_fts_ad trigger as memories rows are deleted.

    def save_meta(self, meta: dict[str, Any]) -> None:
        with self.transaction():
            self._write_meta(meta)

    # -- Vectors ----------------------------------------------------------
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
            # vec_key looks like "m0::<id>" (same keys as the npz, keeping the two paths isomorphic)
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
    # 🔴 In this class, FTS5's **only implementation is the one after _fts_ok**
    # (trigram + external content + triggers). What remains here is just an
    # availability query entry point.
    # Historically an older implementation ("manually maintaining a memory_id
    # column") also sat here, creating the table with unicode61 and manually
    # delete/insert-ing the index; because **later definitions override earlier
    # ones**, that old code would clobber the new implementation entirely
    # (manifesting as trigram never taking effect and index deletion raising
    # "no such column: memory_id"). Before touching FTS5-related code, confirm
    # there is only one implementation.
    def fts5_supported(self) -> bool:
        return self._fts_ok

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


# In-process cache keyed by directory. brain and manager.store **must** get the
# same instance: otherwise each holds its own connection and opens its own
# transactions, and "one save = one transaction" becomes impossible — a crash
# would inevitably produce the sandwich state of "L1 written, derived layers not
# written".
_BACKEND_CACHE: dict[str, StoreBackend] = {}
_CACHE_LOCK = threading.Lock()


def reset_backend_cache(directory: str | None = None) -> None:
    """Drop the cache (for tests: must be called after a temporary library is deleted and recreated, otherwise you get a connection pointing at deleted files)."""
    with _CACHE_LOCK:
        if directory is None:
            _BACKEND_CACHE.clear()
        else:
            _BACKEND_CACHE.pop(os.path.abspath(directory), None)


def get_backend(directory: str) -> StoreBackend:
    """Build the backend per env (the same directory reuses the same instance). Opening an old library as sqlite only warns."""
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
