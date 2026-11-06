"""Numpy in-memory array based vector store (multi-model version).

Design points (v0.4):
    - **Vectors are derivatives**: the memory bodies (text + structure) live
      in JSON; embedding vectors are stored separately in a binary .npz.
      Deleting the .npz only triggers recomputation at the next retrieval;
      no memory is lost.
    - **Multiple models coexist**: each record holds ``{model key: vector}``.
      When switching embedding models, records that already have that model's
      vector are usable directly; the rest are backfilled by the layer above
      (no need to re-import documents).
    - Retrieval only uses vectors of the **currently active model**; records
      missing vectors are exposed to the layer above via ``missing_ids()``
      for backfill.

Later this can be smoothly replaced by a dedicated vector database such as
chromadb, as long as the external interface stays unchanged.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np

from .memory import LEGACY_MODEL_KEY, PIN_IMPORTANCE, MemoryRecord
from .backend import DB_FILENAME, StoreBackend, get_backend


# Number of rolling backup copies to keep (zcode requirement C2). 0 = off
# (only the fixed .bak remains).
# Why keep rolling copies at all: the ``.bak`` gets overwritten on every
# save, so effectively only "the previous save" can be rolled back. When
# things really break, what you want back is "the library as it was before
# this afternoon's bad edits", not the state right before the last save.
BACKUP_ROTATE_KEEP = 3


# ── Construction injection seam ──────────────────────────────────
# This module (and backend / lnn) is constrained by the G1 dependency matrix
# to **not import config**; the registry of knobs and the single env read
# point live in config (ENV_MAP). In production the sole constructor,
# **engine** (at manager import time), feeds config string transport values
# in via :func:`apply_config`; for the parse order (env first, injected
# value as fallback) and the zero-drift argument see the
# ``common/config_bridge.py`` module docstring.
from ..common.config_bridge import apply_config as _bridge_apply_config
from ..common.config_bridge import resolve as _resolve


def apply_config(**values: str) -> None:
    """Constructor injection (wraps config_bridge; ``save_lock_timeout``
    additionally refreshes the module snapshot :data:`SAVE_LOCK_TIMEOUT` --
    the M1b ring trio precedent: a module attribute is the test monkeypatch
    surface; a dirty value raising in float() here = the original
    import-time crash semantics carried over verbatim)."""
    _bridge_apply_config(**values)
    if "save_lock_timeout" in values:
        global SAVE_LOCK_TIMEOUT
        SAVE_LOCK_TIMEOUT = float(values["save_lock_timeout"])


def snapshot_min_hours() -> float:
    """Minimum interval (hours) between sqlite rolling snapshots; ``0`` = do
    one on every save (debug only).

    Why only sqlite needs throttling: ``VACUUM INTO`` measures about
    **1 second per 64MB library**, while ``save()`` triggers it once every
    10 steps; the json backend's copy is just a few MB via ``copy2``, so it
    keeps the existing "copy one on every save" semantics.
    """
    raw = _resolve("HIPPOCAMPUS_SNAPSHOT_MIN_HOURS", "snapshot_min_hours", "6")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 6.0


# Write policy when multiple processes share one library (two ZCode
# workspaces each spawning an MCP subprocess):
#   1 / true (default) → take a cross-process lock + replay this process's
#       changes on top of the other side's latest state (merge)
#   0 / false      → fall back to the old optimistic-lock semantics: raise
#       ConcurrentWriteError as soon as another writer is detected
# Why the default changed to merging: the old "error out on detected
# conflict" was **guaranteed to happen** with two windows --
# each side loads a snapshot once, after which any save collides, and the
# user only ever sees save failures.
def multi_writer_enabled() -> bool:
    """Whether multi-process merge saving is enabled (reads the env var on
    every call, easing tests and runtime switching)."""
    return _resolve("HIPPOCAMPUS_MULTI_WRITER", "multi_writer", "1").lower() not in (
        "0", "false", "no", "off",
    )


def _serialized(func):
    """In-process write mutual exclusion: runs the whole method body under
    ``self._io_lock``.

    Why it is needed (it only became a necessity after daemonization): the
    cross-process protection (filelock + dirty-set replay) assumes "the
    other side is **another process** holding an independent in-memory
    copy". In daemon mode N connections in one process = N threads **sharing
    one VectorStore object**, so ``add()`` and ``save()`` interleave, and
    ``_write_to_disk``'s "take snapshot → clear dirty set" is not atomic --
    a record the other side adds in between is neither in the written file
    nor recorded as "changed", and the next ``_merge_with_disk`` silently
    drops it.

    Only write operations are wrapped (add / update / delete / save); reads
    (recall / count) take no lock, otherwise the daemon's concurrent reads
    would be serialized, throwing away the throughput a shared process
    provides.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._io_lock:
            return func(self, *args, **kwargs)
    return wrapper


# Lock wait upper bound (seconds). Normal writes are millisecond-scale; if
# the wait takes this long, the other side is stuck (e.g. doing a full
# overwrite of a 9000+ record library) -- better to fail explicitly than to
# wait forever.
# The default value is registered in config (``save_lock_timeout``); the
# actual value is injected and refreshed by engine at import time via
# :func:`apply_config` -- the module attribute is the monkeypatch surface
# (ring snapshot semantics); the float() parsing and dirty-value raise kept
# here preserve their original semantics.
SAVE_LOCK_TIMEOUT: float = 60.0


def _matrix_dtype() -> np.dtype:
    """dtype of the retrieval buffer matrix (default fp32; env is read on
    every call to ease test/probe switching).

    Why fp32 is enough: vectors are stored as float32 anyway (both npz and
    sqlite blobs serialize as fp32), and ``record.embeddings`` is already
    normalized to ``array('f')`` (dual-store fix) -- the retrieval buffer is
    a derivative of these two, and fp64 is "pretending precision on already
    rounded data" at double the memory (real library 9859×768 ⇒ 58MB vs
    29MB). Relative error ~1e-7 can only flip the ranking of two near-tied
    adjacent entries; A/B acceptance anchored top-k agreement at 100%.
    Escape hatch: ``HIPPOCAMPUS_MATRIX_DTYPE=fp64`` restores the old
    behavior.
    """
    raw = _resolve("HIPPOCAMPUS_MATRIX_DTYPE", "matrix_dtype", "fp32")
    raw = raw.strip().lower()
    if raw in ("fp64", "float64", "double"):
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _vectors_mmap_enabled() -> bool:
    """Vector file-mapping load switch (default off; env is read on every
    call to ease test/probe switching).

    When enabled, ``record.embeddings`` changes from an anonymous
    ``array('f')`` (~32.6MB in the real library) to a read-only np.memmap
    slice -- file-backed, reclaimable by the OS, page cache shared across
    processes. Judge the benefit by **Private Bytes**, not by instantaneous
    WS (file-backed pages still stay resident in WS after being touched, but
    are not the "held onto forever" kind of memory; see h3_selective_quant
    REPORT §2.3).
    """
    return _resolve(
        "HIPPOCAMPUS_VECTORS_MMAP", "vectors_mmap", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


class ConcurrentWriteError(RuntimeError):
    """The on-disk memory file has been rewritten by **another process**, and
    this save **cannot be merged safely**.

    Trigger scenario: the same ``persist_dir`` is held by two processes at
    once (two ZCode windows each spawning an MCP subprocess; or an import
    script coexisting with a running MCP service). Each side ``load()``s a
    snapshot, mutates its own memory, then **fully overwrites** the file;
    the later saver makes the earlier saver's writes vanish entirely, and
    both print "saved" -- **silent data loss**.

    Current handling (default, see ``multi_writer_enabled``): ``load()``
    records the file version stamp (mtime_ns + size), and ``save()`` first
    verifies it under the **cross-process write lock**; on mismatch this
    process's changes are replayed on top of the disk's latest state
    (``_merge_with_disk``) -- neither side's additions nor deletions are
    lost. So with two windows coexisting the default **no longer raises**.

    Cases that still raise this exception: "the JSON on disk is unreadable
    (better to fail explicitly than merge with half the data)" /
    "HIPPOCAMPUS_MULTI_WRITER=0 explicitly turns merging off" / "the write
    lock cannot be acquired, usually meaning filelock is missing from the
    environment".

    Remedy: the caller should ``load()`` again (MCP tool ``load_state``) to
    get the latest state and redo the change; or make sure the other process
    has exited. If you truly need to force the overwrite (e.g. you know the
    other process is read-only), set ``HIPPOCAMPUS_ALLOW_STALE_SAVE=1`` to
    downgrade to a warning.
    """


class VectorStore:
    """Multi-model vector store based on numpy in-memory arrays.

    Internally keeps one ``list[MemoryRecord]`` (records carry each model's
    vectors); at retrieval time the ``(N, dim)`` matrix for the active model
    is built lazily and cached.

    Args:
        dim: vector dimension of the currently active model.
        persist_path: persistence path for the memory bodies (JSON);
            ``None`` means no persistence to disk.
        model_key: key of the currently active model (e.g.
            ``"BAAI/bge-base-zh-v1.5"``).
    """

    # Persistence format version: 1 = legacy single-vector format, 2 = multi-model + separate .npz
    FORMAT_VERSION = 2

    def __init__(
        self,
        dim: int = 32,
        persist_path: str | None = None,
        model_key: str = "",
    ) -> None:
        self.dim = dim
        self.persist_path = persist_path
        self.model_key = model_key or LEGACY_MODEL_KEY

        # id -> row index within _records
        self._id_to_index: dict[str, int] = {}
        # Record list preserving insertion order
        self._records: list[MemoryRecord] = []
        # Matrix cache for the active model: **fixed-size buffer + used row
        # count**, built lazily. Appends go through incremental maintenance
        # (``_sync_row``) instead of a full rebuild -- this is the root-cause
        # fix for O(N²) writes (defect D2): add() used to invalidate
        # unconditionally and the dedup lookup triggered a full vstack every
        # time, making "write the k-th record" cost O(k).
        self._matrix: np.ndarray | None = None      # buffer, rows >= _matrix_n
        self._matrix_n: int = 0                      # used row count
        self._matrix_ids: list[str] = []             # row -> id (len == _matrix_n)
        self._id_to_row: dict[str, int] = {}          # id -> row number

        # Known model dimensions: model_key -> dim
        self.model_dims: dict[str, int] = {}
        # Model key -> short name inside npz (avoids "/" in npz keys)
        self._model_slugs: dict[str, str] = {}
        # Active model recorded at persist time (the JSON's "model_key"
        # field). This is the authoritative record of "which model this
        # library last **ran on**", restored on load, used for "letting the
        # embedding model follow the data" (see preferred_model_key).
        self.persisted_model_key: str | None = None

        # Issue report from the last load (None = clean).
        # On corruption the state is not silently cleared; the scene is
        # recorded here for upper-layer alerts / test assertions.
        self.last_load_issue: dict[str, Any] | None = None

        # Concurrent write protection: file version stamp at the last
        # "synced with disk". None = the file is not on disk yet (or never
        # synced).
        self._disk_stamp: tuple[int, int] | None = None
        # Rolling backup copies to keep (see BACKUP_ROTATE_KEEP). 0 disables.
        self.backup_rotate_keep: int = BACKUP_ROTATE_KEEP
        # Most recent intercepted concurrent write conflict (for upper-layer
        # alerts / test assertions)
        self.last_conflict: dict[str, Any] | None = None

        # ---- Multiple processes sharing one persist_dir (see save /
        # _merge_with_disk) ----
        # Ids this process has "changed / deleted" since the last successful
        # persist. If a save finds the disk was rewritten by another process,
        # these two sets replay this process's changes **on top of** the
        # other side's latest state rather than fully overwriting -- the
        # latter would silently erase the other side's writes.
        # Records carry no updated_at (timestamp is when the memory happened,
        # not when it was modified), so record-level LWW is impossible;
        # merging can only rely on "whoever changed it replays it".
        self._dirty_ids: set[str] = set()
        self._deleted_ids: set[str] = set()
        # Stats of the most recent merge (for logs / test assertions)
        self.last_merge: dict[str, Any] | None = None

        # ---- In-process write mutual exclusion (a necessity that only
        # appeared after daemonization) ----
        # The cross-process set (filelock + dirty-set replay) assumes "the
        # other side is another process with an independent in-memory copy".
        # In daemon mode one process has N connections = N threads **sharing
        # one VectorStore object**, so:
        #   * A takes a _records snapshot in _write_to_disk and has not yet
        #     cleared _dirty_ids;
        #   * B add()s a new record at that moment (same process, same list;
        #     it is not in A's snapshot);
        #   * A clears _dirty_ids -- B's record is neither in the file A
        #     wrote nor recorded as "changed";
        #   * the next save by either side triggers _merge_with_disk, where
        #     merged only trusts the disk version + dirty-set replay ⇒ that
        #     record **silently disappears**.
        # Measured (tests/test_f9_4b_daemon_mcp_e2e.py E2): two connections
        # writing 15 records each persisted only 20~25 records before the
        # fix; losses depended on timing (non-reproducible sporadic data
        # loss). So add / update / delete / save must share this one lock.
        self._io_lock = threading.RLock()

        # ---- G1: persistence backend (json multi-file / sqlite single db) ----
        # Only the "read disk / write disk" layer is swapped; the in-memory
        # structures (_records / matrix / dirty sets) stay untouched, so the
        # retrieval / merge / dedup business logic needs zero changes. See
        # store/backend.py.
        self._backend: StoreBackend | None = None
        if persist_path:
            self._backend = get_backend(
                os.path.dirname(os.path.abspath(persist_path))
            )
        # Whether a full baseline already exists on disk. The first save must
        # be a full write: incremental writing presupposes the library
        # already holds a complete copy, otherwise "upsert only dirty rows"
        # would produce a library with just a few records.
        self._full_saved = False
        # Vector mmap (HIPPOCAMPUS_VECTORS_MMAP=1): holds the mapping object
        self._vectors_mm = None

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def vectors_path(self) -> str | None:
        """Binary vector file path (same directory as ``persist_path``)."""
        if not self.persist_path:
            return None
        base, _ = os.path.splitext(self.persist_path)
        return base + ".vectors.npz"

    def _slug_for(self, model_key: str) -> str:
        """Assign a stable short name for a model key (used as the npz key
        prefix)."""
        existing = self._model_slugs.get(model_key)
        if existing:
            return existing
        used = {s for s in self._model_slugs.values() if s}
        i = 0
        while f"m{i}" in used:
            i += 1
        slug = f"m{i}"
        self._model_slugs[model_key] = slug
        return slug

    def _key_for(self, model_key: str, record_id: str) -> str:
        return f"{self._slug_for(model_key)}::{record_id}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invalidate(self) -> None:
        """Invalidate the whole matrix cache; rebuild fully on the next
        search.

        Called only on structural changes (delete / model switch / model
        prune / load). Added records and in-place vector updates go through
        ``_sync_row`` incremental maintenance, not a full rebuild.
        """
        self._matrix = None
        self._matrix_n = 0
        self._matrix_ids = []
        self._id_to_row = {}

    def _record_dim(self, vector: Any, model_key: str) -> int:
        emb = np.asarray(vector, dtype=np.float64).reshape(-1)
        dim = int(emb.shape[0])
        known = self.model_dims.get(model_key)
        if known is not None and known != dim:
            raise ValueError(
                f"model {model_key} has inconsistent vector dimensions: recorded {known}, now {dim}"
            )
        self.model_dims[model_key] = dim
        return dim

    def _validate_embedding(self, embedding: Any, model_key: str | None = None) -> np.ndarray:
        """Validate and convert to a 1D float64 vector."""
        key = model_key or self.model_key
        emb = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if key == self.model_key and emb.shape[0] != self.dim:
            raise ValueError(
                f"embedding dimension {emb.shape[0]} 与当前model {key}'s dimension {self.dim} -- mismatch"
            )
        self._record_dim(emb, key)
        return emb

    def _active_matrix(self) -> tuple[list[str], np.ndarray]:
        """Return (ids, matrix) for the active model, rebuilding on demand.

        Returns:
            (ids, matrix): ids and matrix rows correspond one to one, in
            record insertion order. matrix is a **view** of the internal
            buffer (row count == len(ids)); callers must treat it as
            read-only.
        """
        if self._matrix is None:
            ids: list[str] = []
            rows: list[np.ndarray] = []
            for rec in self._records:
                vec = rec.vector_for(self.model_key)
                if vec is None:
                    continue
                ids.append(rec.id)
                rows.append(np.asarray(vec, dtype=np.float64).reshape(-1))
            dim = rows[0].shape[0] if rows else self.dim
            dt = _matrix_dtype()
            buf = np.zeros((max(len(rows), 8), dim), dtype=dt)
            for i, row in enumerate(rows):
                buf[i] = row
            self._matrix = buf
            self._matrix_n = len(rows)
            self._matrix_ids = ids
            self._id_to_row = {rid: i for i, rid in enumerate(ids)}
        return self._matrix_ids, self._matrix[: self._matrix_n]

    def _grow_matrix(self) -> None:
        """Grow the matrix buffer (doubling), amortized O(1)."""
        old = self._matrix
        if old is None:
            return
        new = np.zeros((max(old.shape[0] * 2, 8), old.shape[1]), dtype=old.dtype)
        new[: self._matrix_n] = old[: self._matrix_n]
        self._matrix = new

    def _sync_row(self, memory_id: str, vector: Any | None) -> None:
        """Incrementally sync one record's active-model vector into the
        matrix cache.

        Takes effect only when the matrix is already built; if not built it
        stays lazy and the next retrieval naturally includes it. When a row
        must be **removed** (the record lost the current model's vector),
        numpy cannot delete a row in O(1), so it degrades to a full
        invalidation -- this path never occurs in normal writes.
        """
        if self._matrix is None:
            return
        row = self._id_to_row.get(memory_id)
        if vector is None:
            if row is not None:
                self._invalidate()
            return
        vec = np.asarray(vector, dtype=self._matrix.dtype).reshape(-1)
        if vec.shape[0] != self._matrix.shape[1]:
            # Dimension anomaly (theoretically blocked by _validate_embedding); degrade safely
            self._invalidate()
            return
        if row is not None:
            self._matrix[row] = vec
            return
        if self._matrix_n >= self._matrix.shape[0]:
            self._grow_matrix()
        self._matrix[self._matrix_n] = vec
        self._id_to_row[memory_id] = self._matrix_n
        self._matrix_ids.append(memory_id)
        self._matrix_n += 1

    def _cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        """Batch-compute cosine similarity between the query vector and all
        stored vectors of the active model.

        query is cast to the **matrix's dtype** before the multiply: an fp32
        matrix @ fp64 vector makes numpy temporarily promote the whole
        matrix to fp64 (a 58MB-scale temporary array, undoing the fp32
        conversion). The result is still cast back to float64, keeping the
        score semantics callers see unchanged.
        """
        _, matrix = self._active_matrix()
        if matrix.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        q = np.asarray(query, dtype=matrix.dtype).reshape(-1)
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(q)
        denom = norms * q_norm
        denom[denom == 0.0] = 1e-12
        return (matrix @ q).astype(np.float64) / denom.astype(np.float64)

    def pairwise_similarities(
        self,
        ids: Sequence[str],
        low: float,
        high: float,
        limit: int = 12,
        chunk: int = 512,
        normalize: bool = True,
    ) -> list[tuple[str, str, float]]:
        """Find pairs within an id subset whose similarity falls in
        ``[low, high)``, returned in descending similarity order.

        Why this separate entry point exists: **pairwise comparison is an
        order of magnitude more expensive than "one against all"**, and
        ``consolidate``'s contradiction detection is exactly pairwise within
        a bucket -- it wants "who resembles whom among the memories of the
        same entity", not "which memory best matches the query".

        The implementation is **blocked matrix multiplication**, not a
        per-pair Python loop. The two give **numerically identical** results
        (both are dot products of normalized rows), but the cost differs by
        two orders of magnitude: on a real large library (thousands of
        records / hundreds of entity buckets) measured to drop from minutes
        to seconds.
        The per-pair version's cost is not in the dot product at all -- it
        is in **converting a Python list to a numpy array for every pair**
        (``manager.vector_of`` returns ``record.embeddings[key]``, an
        ``np.asarray`` over 768 elements each time) -- this must be written
        in a comment, otherwise people later assume "dot products are just
        slow" and give up on optimizing.

        Args:
            ids: the subset to compare (one entity bucket).
            low / high: similarity interval ``[low, high)``. **The upper
                bound is deliberate**: pairs with similarity ≥ high are
                usually two copies of the same fact, not a "contradiction";
                they are left to write-side dedup and not re-reported here.
            limit: max pairs returned per subset (descending by similarity).
            chunk: block row count. Controls peak memory of the temporary
                matrix (``chunk × n × 8`` bytes).
            normalize: whether to normalize rows first. Default True
                (returns true cosine values); False filters by raw dot
                product, only for callers that already know the vectors are
                normalized.

        Returns:
            ``[(id_a, id_b, sim), ...]``, descending by sim; only the upper
            triangle (``i < j``), no self-comparison. ids missing vectors
            are skipped automatically.
        """
        if len(ids) < 2:
            return []
        _, matrix = self._active_matrix()
        if matrix.shape[0] == 0:
            return []
        row_of = self._id_to_row
        kept: list[str] = []
        rows: list[int] = []
        for mid in ids:
            r = row_of.get(mid)
            if r is not None:
                kept.append(mid)
                rows.append(r)
        if len(rows) < 2:
            return []

        sub = np.asarray(matrix[rows], dtype=np.float64)
        if normalize:
            norms = np.linalg.norm(sub, axis=1)
            norms[norms == 0.0] = 1e-12
            sub = sub / norms[:, None]

        best: list[tuple[float, int, int]] = []
        for s in range(0, len(rows), chunk):
            blk = sub[s:s + chunk]
            sims = blk @ sub.T
            for i in range(blk.shape[0]):
                lo = s + i + 1          # upper triangle only
                if lo >= len(rows):
                    break
                row = sims[i, lo:]
                hit = np.nonzero((row >= low) & (row < high))[0]
                if hit.size:
                    for g in hit:
                        best.append((float(row[g]), s + i, lo + int(g)))
            # Prune while computing: shrink candidates back to limit so large buckets do not eat all the memory
            if len(best) > limit * 4:
                best.sort(key=lambda x: -x[0])
                del best[limit:]

        best.sort(key=lambda x: -x[0])
        return [(kept[i], kept[j], sim) for sim, i, j in best[:limit]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @_serialized
    def add(self, record: MemoryRecord) -> str:
        """Add a memory and return its id.

        If ``record.id`` is empty a UUID is generated; if the id already
        exists an in-place update is performed.
        """
        if not record.id:
            record.id = str(uuid.uuid4())

        for key, vec in record.embeddings.items():
            self._validate_embedding(vec, key)

        if record.id in self._id_to_index:
            idx = self._id_to_index[record.id]
            self._records[idx] = record
        else:
            self._records.append(record)
            self._id_to_index[record.id] = len(self._records) - 1

        # Incrementally maintain the matrix cache (stays lazy when unbuilt; the next retrieval rebuild includes this record)
        self._sync_row(record.id, record.vector_for(self.model_key))
        self._mark_dirty(record.id)
        return record.id

    def set_vector(
        self, memory_id: str, model_key: str, vector: Any
    ) -> bool:
        """Write a model's vector for the given memory (used by the backfill
        path)."""
        rec = self.get(memory_id)
        if rec is None:
            return False
        emb = self._validate_embedding(vector, model_key)
        rec.set_vector(model_key, emb)
        if model_key == self.model_key:
            self._sync_row(memory_id, emb)
        self._mark_dirty(memory_id)
        return True

    def search(
        self, query: np.ndarray, top_k: int = 10,
        allow_ids: set[str] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Cosine similarity search (active model only); returns a (record,
        similarity) list in descending order.

        Args:
            allow_ids: only memories with these ids may be returned (None =
                no restriction). Tag filtering goes through here: narrow the
                candidate set **before retrieval** rather than filtering
                after the fact -- the latter misses recalls when everything
                matching lies outside top_k.
        """
        if len(self._records) == 0:
            return []
        if allow_ids is not None and not allow_ids:
            return []

        query = np.asarray(query, dtype=np.float64).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(
                f"query vector dimension {query.shape[0]} 与当前model {self.model_key}'s dimension "
                f"{self.dim} -- mismatch"
            )

        ids, _ = self._active_matrix()
        if not ids:
            return []

        sims = self._cosine_similarity(query)
        if allow_ids is not None:
            # Mask rows not in the allowlist (-inf rather than 0: 0 can still be picked as "the tallest among the short")
            keep = np.fromiter(
                (rid in allow_ids for rid in ids), dtype=bool, count=len(ids)
            )
            sims = np.where(keep, sims, -np.inf)
        top_k = min(top_k, len(ids))
        top_idx = np.argsort(-sims)[:top_k]
        return [
            (self._records[self._id_to_index[ids[int(i)]]], float(sims[int(i)]))
            for i in top_idx
            if np.isfinite(sims[int(i)])
        ]

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a memory record by id; returns None if absent."""
        idx = self._id_to_index.get(memory_id)
        if idx is None:
            return None
        return self._records[idx]

    @_serialized
    def update(self, memory_id: str, **fields: Any) -> bool:
        """Update fields of the given memory (e.g. recall_count). Returns
        whether the update succeeded.

        Accepts any dataclass field; ``embedding`` is kept for backward
        compatibility with old calls and is equivalent to writing the active
        model's vector.
        """
        idx = self._id_to_index.get(memory_id)
        if idx is None:
            return False

        record = self._records[idx]
        for key, value in fields.items():
            if key == "embedding":
                self.set_vector(memory_id, self.model_key, value)
                continue
            if key == "embeddings":
                if isinstance(value, dict):
                    for mk, mv in value.items():
                        self.set_vector(memory_id, mk, mv)
                continue
            if not hasattr(record, key):
                continue
            setattr(record, key, value)

        # Defect D4: pin was originally set from importance only in
        # MemoryRecord.__post_init__, and setattr here bypasses it -- so
        # "importance>=0.7 pins automatically" only worked for newly created
        # records; when the merge (dedup) path raised importance, pin stayed
        # False and stats().pinned stayed 0.
        # Recompute only when importance actually changed this time and the
        # caller did not pass pin explicitly.
        if "importance" in fields and "pin" not in fields:
            record.sync_pin()
        self._mark_dirty(memory_id)
        return True

    @_serialized
    def delete(self, memory_id: str) -> bool:
        """Delete the given memory; returns whether deletion succeeded."""
        idx = self._id_to_index.get(memory_id)
        if idx is None:
            return False

        del self._records[idx]
        # Rebuild the index mapping, keeping the list compact
        self._id_to_index.clear()
        for i, rec in enumerate(self._records):
            self._id_to_index[rec.id] = i
        self._invalidate()
        # Deletions must be booked too: during a merge this id must be
        # removed from the other side's latest state, otherwise its on-disk
        # copy gets "kept", manifesting as a record that comes back after
        # deletion.
        self._dirty_ids.discard(memory_id)
        self._deleted_ids.add(memory_id)
        return True

    def count(self) -> int:
        """Return the number of memories currently stored."""
        return len(self._records)

    def iter_records(self) -> Iterator[MemoryRecord]:
        """M3: **public** entry point for read-only iteration over all
        records.

        Before this migration the upper layers (brain/core, engine/manager,
        45 call sites) reached directly into ``self._records`` -- that made
        VectorStore's internal representation (list + insertion order) a de
        facto API, and any storage-structure change would blow up a layer.
        This method folds "iteration" into the port contract: callers get an
        iterator, not the container.

        Semantics are identical to a direct ``iter(self._records)`` (same
        list object, same order, lazy evaluation) -- so this is a
        **zero-behavior-change** consolidation, not a semantic enhancement.
        The threading-safety boundary is unchanged: in-process write mutual
        exclusion is the caller's held lock's job; this method takes no lock
        and should not.
        """
        return iter(self._records)

    # ------------------------------------------------------------------
    # Multi-model API
    # ------------------------------------------------------------------
    def set_model(self, model_key: str, dim: int | None = None) -> None:
        """Switch the active model (retrieval will use that model's
        vectors)."""
        self.model_key = model_key
        if dim is not None:
            self.dim = dim
        elif model_key in self.model_dims:
            self.dim = self.model_dims[model_key]
        self._invalidate()

    def missing_ids(self, model_key: str | None = None) -> list[str]:
        """Return ids of memories missing vectors for the given (default:
        active) model."""
        key = model_key or self.model_key
        return [r.id for r in self._records if not r.has_vector(key)]

    def model_stats(self) -> dict[str, dict[str, int]]:
        """Report per-model vector coverage.

        Returns:
            ``{model key: {"vectors": records holding a vector, "dim":
            dimension}}``.
        """
        stats: dict[str, dict[str, int]] = {}
        for rec in self._records:
            for key in rec.embeddings:
                entry = stats.setdefault(
                    key, {"vectors": 0, "dim": self.model_dims.get(key, 0)}
                )
                entry["vectors"] += 1
        return stats

    def preferred_model_key(self) -> str | None:
        """Determine "which embedding model this library actually runs on",
        for letting the embedding model follow the data.

        Priority:
            1. the persisted ``model_key`` (top-level JSON field, = the
               active model at last save) -- but only if it **really has
               vector coverage**; otherwise it is dirty residue left by
               "some run with the wrong model that got persisted", not to be
               trusted blindly;
            2. the non-``__legacy__`` slot with the highest coverage (where
               the vectors are, the model is);
            3. neither exists (empty library / only a legacy single vector)
               → ``None``.

        ``__legacy__`` is never a candidate: it is the placeholder slot for
        the "legacy single-vector format"; claiming it requires an explicit
        :meth:`adopt_legacy`, it must not be followed as if it were a real
        embedding model.

        ``None`` means "no followable model in the library"; the caller
        decides what default to use.
        """
        stats = self.model_stats()   # {model_key: {"vectors": n, "dim": d}}
        counted = {
            k: int(v.get("vectors", 0))
            for k, v in stats.items()
            if v.get("vectors") and k != LEGACY_MODEL_KEY
        }
        if not counted:
            return None
        pk = self.persisted_model_key
        if pk and pk != LEGACY_MODEL_KEY and counted.get(pk):
            return pk
        # Highest coverage wins; ties broken by lexicographically smallest key for reproducibility
        return sorted(counted.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def prune_models(self, keep: set[str]) -> int:
        """Drop model vectors not in ``keep``; returns the total number of
        dropped vectors."""
        dropped = 0
        for rec in self._records:
            dropped += rec.drop_vectors(keep)
        for key in list(self.model_dims):
            if key not in keep:
                self.model_dims.pop(key, None)
        for key in list(self._model_slugs):
            if key not in keep:
                self._model_slugs.pop(key, None)
        if self.model_key not in keep:
            self.model_key = next(iter(keep)) if keep else LEGACY_MODEL_KEY
        self._invalidate()
        return dropped

    def adopt_legacy(self, model_key: str) -> int:
        """Reclaim vectors in the ``__legacy__`` slot as ``model_key``
        (dimensions must match).

        Returns:
            Number of vectors reclaimed.
        """
        moved = 0
        legacy_dim = self.model_dims.get(LEGACY_MODEL_KEY, self.dim)
        for rec in self._records:
            vec = rec.vector_for(LEGACY_MODEL_KEY)
            if vec is None:
                continue
            if len(vec) != legacy_dim:
                continue
            rec.set_vector(model_key, vec)
            del rec.embeddings[LEGACY_MODEL_KEY]
            moved += 1
        if moved:
            self.model_dims[model_key] = legacy_dim
            self.model_dims.pop(LEGACY_MODEL_KEY, None)
            self._model_slugs.pop(LEGACY_MODEL_KEY, None)
            self._invalidate()
        return moved

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @_serialized
    def save(self, check_conflict: bool = True) -> None:
        """Persist: memory bodies go to JSON, vectors to .npz.

        Both use the "write temp file then atomically replace" strategy to
        avoid corruption from interrupted writes. Vectors are stored as
        float32 (half the size, no material impact on retrieval precision).

        Args:
            check_conflict: whether to perform the cross-process check
                (default on). When on: take the cross-process write lock
                first; if the disk was changed by another process, either
                merge-save per ``HIPPOCAMPUS_MULTI_WRITER`` (default 1) or
                fail explicitly (=0). Pass False only when you positively
                know the other process will not write and you want to force
                the overwrite.

        Raises:
            ConcurrentWriteError: on a conflict **that cannot be merged
                safely** (disk unreadable / multi-process merge explicitly
                disabled / write lock unavailable) -- see the class
                docstring.
        """
        if self.persist_path is None:
            return
        if not check_conflict:
            self._write_to_disk()
            return
        # With multiple processes coexisting (two ZCode workspaces each
        # spawning a subprocess): take the cross-process write lock first --
        # two processes would trample the same ``memories.json.tmp``, a
        # corruption worse than "last write wins"; then check whether the
        # disk changed: if so, **replay** this process's changes on top
        # instead of erroring out.
        with self._save_lock() as locked:
            self._resolve_conflict(locked)
            self._write_to_disk()

    def _write_to_disk(self) -> None:
        """Persist to disk (memory bodies + vectors); clears the dirty sets
        on success.

        The disk write goes through :attr:`_backend`. When the **disk
        already has a baseline**, the sqlite backend only upserts dirty rows
        and deletes removed rows -- editing 1 record no longer rewrites the
        5.28MB memories.json and the 32.7MB npz (measured bill). The first
        persist (no baseline yet) is still a full write.
        """
        if not self.persist_path:
            return
        backend = self._backend
        if backend is None:
            backend = get_backend(
                os.path.dirname(os.path.abspath(self.persist_path))
            )
            self._backend = backend

        meta = {
            "format": self.FORMAT_VERSION,
            "dim": self.dim,
            "model_key": self.model_key,
            "models": {
                key: {"dim": self.model_dims.get(key, 0), "slug": self._slug_for(key)}
                for key in self.model_dims
            },
        }

        # ---- 1. Memory bodies ----
        incremental = backend.supports_incremental and self._full_saved
        try:
            if incremental:
                dirty = [r.to_dict(with_embeddings=False)
                         for r in self._records if r.id in self._dirty_ids]
                backend.upsert_memories(dirty)
                backend.delete_memories(list(self._deleted_ids))
                backend.save_meta(meta)
            else:
                backend.save_memories_full(dict(meta, records=[
                    r.to_dict(with_embeddings=False) for r in self._records
                ]))
                self._full_saved = True
        except Exception as e:
            logging.warning("保存记忆到 %s 失败: %s", self.persist_path, e)
            # The baseline may be half-written ⇒ fall back to full save, so we do not keep incrementally patching a crippled library
            self._full_saved = False
            return

        # After persisting, sync the disk stamp (json = file mtime; sqlite =
        # write sequence number + record count). The next save uses it to
        # detect "was it changed by another process", triggering a merge
        # rather than an overwrite.
        self._disk_stamp = backend.store_stamp()
        self.last_conflict = None

        # Rolling snapshot (json copies the .bak / sqlite uses VACUUM INTO).
        # Placed after the L1 persist and before vectors: what it protects
        # is the memory bodies, and it keeps a single call site (the vector
        # branch has an early return; hanging this at the end would miss
        # it).
        self._rotate_after_save(backend)

        # ---- 2. Vectors ----
        if not self.vectors_path:
            self._dirty_ids.clear()
            self._deleted_ids.clear()
            return
        try:
            if incremental:
                dirty_arrays = {
                    self._key_for(mk, r.id): np.asarray(vec, dtype=np.float32)
                    for r in self._records if r.id in self._dirty_ids
                    for mk, vec in r.embeddings.items()
                }
                backend.upsert_vectors(dirty_arrays)
                # Deleted memories: their vectors were cleared with delete_memories; sweep once more here as a backstop
                backend.delete_vectors_for(list(self._deleted_ids))
            else:
                arrays = {
                    self._key_for(mk, r.id): np.asarray(vec, dtype=np.float32)
                    for r in self._records
                    for mk, vec in r.embeddings.items()
                }
                backend.save_vectors_full(arrays)
        except Exception as e:
            logging.warning("保存向量到 %s 失败: %s", self.vectors_path, e)
            self._full_saved = False
            return

        self._dirty_ids.clear()
        self._deleted_ids.clear()

    def _rotate_backup(self, bak_path: str) -> None:
        """Save an extra timestamped copy of the just-vacated ``.bak``,
        keeping only the most recent few.

        The ``.bak`` itself is always kept untouched (load's fallback path
        depends on it); what rotates is its copies. Timestamps use
        ``YYYYMMDD_HHMMSS`` -- lexicographic order is time order, so pruning
        just sorts and trims the tail.
        """
        keep = int(getattr(self, "backup_rotate_keep", BACKUP_ROTATE_KEEP) or 0)
        if keep <= 0:
            return
        directory = os.path.dirname(bak_path) or "."
        prefix = os.path.basename(bak_path) + "-"
        if not os.path.isfile(bak_path):
            return  # first save: the main file did not exist before, no .bak was vacated, nothing to copy
        dst = os.path.join(directory, prefix + time.strftime("%Y%m%d_%H%M%S"))
        try:
            shutil.copy2(bak_path, dst)
        except OSError as e:
            logging.warning("滚动备份失败 %s: %s", dst, e)
            return
        self._prune_snapshots(directory, prefix, keep)

    def _rotate_snapshot_db(self, backend: Any, keep: int) -> None:
        """sqlite backend: ``VACUUM INTO`` a full-library snapshot, keeping
        the most recent few.

        Under sqlite the whole library is one file (memories + vectors +
        derived layer + entities + sources all inside), so this one file is
        the complete snapshot -- simpler and more consistent than the json
        era's loose-file backups.

        Throttling: see :func:`snapshot_min_hours` -- a snapshot costs about
        1 second per 64MB, so it cannot run on every ``save()``.
        """
        directory = os.path.dirname(os.path.abspath(self.persist_path))
        prefix = DB_FILENAME + ".bak-"
        if not self._snapshot_due(directory, prefix):
            return
        dst = os.path.join(directory, prefix + time.strftime("%Y%m%d_%H%M%S"))
        if os.path.exists(dst):
            return  # timestamps have second precision: multiple saves within the same second share one snapshot, skip
        backend.snapshot_to(dst)
        self._prune_snapshots(directory, prefix, keep)

    def _snapshot_due(self, directory: str, prefix: str) -> bool:
        """Whether the minimum interval since the newest snapshot has
        elapsed (empty / unreadable directory ⇒ do it now)."""
        hours = snapshot_min_hours()
        if hours <= 0:
            return True
        try:
            olds = sorted(f for f in os.listdir(directory) if f.startswith(prefix))
        except OSError:
            return True
        if not olds:
            return True
        try:
            age = time.time() - os.path.getmtime(os.path.join(directory, olds[-1]))
        except OSError:
            return True
        return age >= hours * 3600.0

    def _prune_snapshots(self, directory: str, prefix: str, keep: int) -> None:
        """Keep only the most recent ``keep`` timestamped copies
        (lexicographic order is time order)."""
        try:
            olds = sorted(f for f in os.listdir(directory) if f.startswith(prefix))
        except OSError:
            return
        for name in olds[:-keep]:
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass

    def _rotate_after_save(self, backend: Any) -> None:
        """Produce a historical copy after a successful save. **Never raises
        on failure** (same contract as auto_save).

        Backups must not break the save: when things go wrong, better no
        backup than a failed save. json uses the existing ``.bak`` copy;
        sqlite uses ``VACUUM INTO``.
        """
        keep = int(getattr(self, "backup_rotate_keep", BACKUP_ROTATE_KEEP) or 0)
        if keep <= 0:
            return
        try:
            if getattr(backend, "supports_snapshot", False):
                self._rotate_snapshot_db(backend, keep)
            else:
                self._rotate_backup(self.persist_path + ".bak")
        except Exception as e:
            logging.warning("滚动快照失败（不影响本次保存）: %s", e)

    def _stamp_of(self, path: str | None) -> tuple[int, int] | None:
        """The file's version stamp ``(mtime_ns, size)``; ``None`` if it
        does not exist.

        mtime+size instead of a content hash: a ten-thousand-record
        library's JSON is several MB, and re-hashing before every save is
        not worth it, while "was it touched by another process" is reliably
        judged from metadata (a rewrite within the same second with exactly
        the same size would have to be deliberately contrived in practice).
        """
        if not path:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    @property
    def disk_stamp(self) -> tuple[int, int] | None:
        """Version stamp at the most recent disk sync (successful load /
        save)."""
        return self._disk_stamp

    def _check_write_conflict(self) -> None:
        """Verify the disk version before saving; raise
        :class:`ConcurrentWriteError` if another process changed it.

        Verified even when there are **no in-memory records**: an empty
        library overwriting a non-empty file is just as dangerous, so that
        is blocked too.
        If the file on disk was deleted (another process wiped the library),
        that also counts as a conflict.
        """
        if self.persist_path is None:
            return
        current = self._stamp_of(self.persist_path)
        if current == self._disk_stamp:
            return

        # Never synced (stamp is None) and no file on disk → first write, allow
        if self._disk_stamp is None and current is None:
            return

        detail = {
            "persist_path": self.persist_path,
            "expected": self._disk_stamp,
            "found": current,
            "in_memory_records": len(self._records),
        }
        self.last_conflict = detail

        if _resolve("HIPPOCAMPUS_ALLOW_STALE_SAVE",
                    "allow_stale_save", "") in ("1", "true", "True", "yes"):
            logging.warning(
                "Detected that %s was rewritten by another process (expected version %s, actual %s); "
                "HIPPOCAMPUS_ALLOW_STALE_SAVE=1 is set, so overwriting anyway -- "
                "the other process's writes will be lost.",
                self.persist_path, self._disk_stamp, current,
            )
            return

        raise ConcurrentWriteError(
            f"Save refused: {self.persist_path} was rewritten by another process; "
            f"overwriting now would silently discard that process's writes.\n"
            f"  expected version (at load): {self._disk_stamp}\n"
            f"  actual version on disk    : {current}\n"
            f"  in-memory record count    : {len(self._records)}\n"
            f"Remedies (pick one):\n"
            f"  1) reload the latest state, then retry (MCP tool load_state);\n"
            f"  2) make sure the other process (another host window / import script) has exited, then retry;\n"
            f"  3) if you know the other process is read-only, set HIPPOCAMPUS_ALLOW_STALE_SAVE=1 to force the overwrite."
        )

    # ------------------------------------------------------------------
    # Multiple processes sharing one library: cross-process lock + dirty-set
    # replay merge
    # ------------------------------------------------------------------
    def _mark_dirty(self, memory_id: str) -> None:
        """Book-keeping: this process changed this memory; on save it must
        be replayed on top of the other side's latest state."""
        if not memory_id:
            return
        self._dirty_ids.add(memory_id)
        self._deleted_ids.discard(memory_id)

    def _rebuild_index(self) -> None:
        """Rebuild ``_id_to_index`` and invalidate the matrix cache (called
        after a merge)."""
        self._id_to_index = {r.id: i for i, r in enumerate(self._records)}
        self._invalidate()

    @property
    def lock_path(self) -> str | None:
        """File path of the cross-process write lock (same directory as
        ``memories.json``)."""
        if not self.persist_path:
            return None
        return self.persist_path + ".lock"

    @contextmanager
    def _save_lock(self):
        """Cross-process mutual exclusion (best-effort): falls back to the
        original optimistic-lock behavior when filelock is missing from the
        environment.

        Without this lock, two processes would write into the **same**
        ``memories.json.tmp`` simultaneously; ``os.replace`` is atomic, but
        the temp files trample each other -- a class of corruption worse
        than "last write wins". So the lock is the floor, not merely a way
        to make conflict detection more accurate.
        """
        path = self.lock_path
        if path is None:
            yield False
            return
        try:
            from filelock import FileLock, Timeout
        except Exception:  # noqa: BLE001 a missing dependency must not block saving
            logging.warning(
                "未找到 filelock，多进程写保护降级为乐观锁（冲突时仍会明确失败）"
            )
            yield False
            return
        lock = FileLock(path, timeout=SAVE_LOCK_TIMEOUT)
        try:
            lock.acquire()
        except Timeout as e:
            raise ConcurrentWriteError(
                f"等待写锁超时（{SAVE_LOCK_TIMEOUT:.0f}s）：{path}\n"
                f"  通常是另一个进程正在整体覆写同一个库，或持锁进程卡住了。\n"
                f"  可用 HIPPOCAMPUS_SAVE_LOCK_TIMEOUT 调大等待上限。"
            ) from e
        try:
            yield True
        finally:
            try:
                lock.release()
            except Exception:  # noqa: BLE001 a release failure does not affect this save's result
                pass

    def _read_records_from_disk(self) -> list[MemoryRecord] | None:
        """Read-only parse of the memories on disk (with vectors); **does
        not touch in-memory state**.

        A ``None`` return means the data is there but unreadable -- in that
        case, better to fall back to "explicit failure" than to merge with
        half the data (a merge would write the bad data back, worse than an
        error).

        G2: disk reads must go through :attr:`_backend`. Under sqlite the
        records live in the db and there is no memories.json to read;
        without this, multi-process merging would see an "empty disk" and
        this process's full overwrite would erase everything the other side
        just wrote.
        """
        backend = self._backend
        if backend is None:
            return []
        data, _ = backend.load_memories()
        if data is None:
            return []
        if not isinstance(data, dict):
            return None
        try:
            recs = [MemoryRecord.from_dict(d) for d in data.get("records", [])]
        except Exception as e:  # noqa: BLE001
            logging.warning("合并前解析磁盘记录失败（将退回明确失败）: %s", e)
            return None

        # Vectors: keys look like "<slug>::<id>". The slug → model key
        # mapping follows the models section on disk; this process's known
        # mapping is a supplement.
        slug_to_model = {
            str(meta["slug"]): str(key)
            for key, meta in (data.get("models") or {}).items()
            if isinstance(meta, dict) and meta.get("slug")
        }
        for key, slug in self._model_slugs.items():
            slug_to_model.setdefault(slug, key)

        arrays = backend.load_vectors() or {}
        by_id = {r.id: r for r in recs}
        for vkey, vec in arrays.items():
            if "::" not in vkey:
                continue
            slug, _, rec_id = vkey.partition("::")
            model = slug_to_model.get(slug)
            rec = by_id.get(rec_id)
            if model is None or rec is None:
                continue
            rec.set_vector(model, vec)
        return recs

    def _merge_with_disk(self) -> bool:
        """**Replay** this process's changes on top of the disk's latest
        state (multi-process merge save).

        The rules depend only on the fact of "who changed it" -- records
        carry no ``updated_at`` (``timestamp`` is when the memory happened,
        not when it was modified), so record-level LWW is impossible:

        * on disk, untouched by this process → kept as-is (the other side's
          write is absorbed into memory)
        * changed by this process (``_dirty_ids``) → overwritten with this
          process's in-memory version
        * deleted by this process (``_deleted_ids``) → removed from the
          result
        * records newly added by the other side → enter this process's
          memory (visible here afterwards)

        Side effect: when the same record was changed by both sides, this
        process wins (last write wins) -- an unavoidable trade-off without
        version numbers; but **neither side's additions/deletions are ever
        lost**.

        Returns:
            True = safe to continue persisting (memory now holds the merged
            result);
            False = disk state unreadable, the caller should fall back to
            "explicit failure".
        """
        disk_records = self._read_records_from_disk()
        if disk_records is None:
            return False

        mine = {r.id: r for r in self._records}
        merged = {r.id: r for r in disk_records}
        absorbed = sum(1 for rid in merged if rid not in mine)

        replayed = 0
        for rid in self._dirty_ids:
            rec = mine.get(rid)
            if rec is not None:
                merged[rid] = rec
                replayed += 1

        dropped = 0
        for rid in self._deleted_ids:
            if rid in merged:
                dropped += 1
            merged.pop(rid, None)

        self._records = list(merged.values())
        self._rebuild_index()
        self.last_merge = {
            "absorbed_from_disk": absorbed,
            "replayed_local_edits": replayed,
            "applied_local_deletes": dropped,
            "total": len(self._records),
        }
        logging.info(
            "multi-process merge save: absorbed %d records from the other process, replayed %d local changes / %d local deletions (%d records total)",
            absorbed, replayed, dropped, len(self._records),
        )
        return True

    def _resolve_conflict(self, locked: bool) -> None:
        """Decide whether this save is safe; merge first when needed. Raise
        if unsafe.

        Args:
            locked: whether the cross-process write lock is currently held
                (without the lock, merging is too risky -- if a third
                process writes during the merge, the results still overwrite
                each other).
        """
        # G2: sqlite has no memories.json to stat; the disk stamp is
        # provided by the backend itself (write sequence number + record
        # count). Otherwise it would always judge "unchanged", and
        # concurrent writes would directly overwrite the other side.
        current = (self._backend.store_stamp() if self._backend
                   else self._stamp_of(self.persist_path))
        if current == self._disk_stamp:
            return
        # Never synced (stamp None) and no file on disk → first write, allow
        if self._disk_stamp is None and current is None:
            return

        detail = {
            "persist_path": self.persist_path,
            "expected": self._disk_stamp,
            "found": current,
            "in_memory_records": len(self._records),
        }
        self.last_conflict = detail

        if _resolve("HIPPOCAMPUS_ALLOW_STALE_SAVE",
                    "allow_stale_save", "") in ("1", "true", "True", "yes"):
            logging.warning(
                "Detected that %s was rewritten by another process (expected version %s, actual %s); "
                "HIPPOCAMPUS_ALLOW_STALE_SAVE=1 is set, so overwriting anyway -- "
                "the other process's writes will be lost.",
                self.persist_path, self._disk_stamp, current,
            )
            return

        if multi_writer_enabled() and locked:
            if self._merge_with_disk():
                return
            reason = "the on-disk file could not be read, so a safe merge is impossible"
        elif not multi_writer_enabled():
            reason = "HIPPOCAMPUS_MULTI_WRITER=0 (multi-process merge disabled)"
        else:
            reason = "could not acquire the cross-process write lock (filelock missing); refusing to merge without a lock"

        raise ConcurrentWriteError(
            f"Save refused: {self.persist_path} was rewritten by another process; "
            f"overwriting now would silently discard that process's writes.\n"
            f"  reason                    : {reason}\n"
            f"  expected version (at load): {self._disk_stamp}\n"
            f"  actual version on disk    : {current}\n"
            f"  in-memory record count    : {len(self._records)}\n"
            f"Remedies (pick one):\n"
            f"  1) reload the latest state, then retry (MCP tool load_state);\n"
            f"  2) make sure the other process (another host window / import script) has exited, then retry;\n"
            f"  3) if you know the other process is read-only, set HIPPOCAMPUS_ALLOW_STALE_SAVE=1 to force the overwrite."
        )

    def load(self) -> None:
        """Load memories and vectors from disk, replacing the current
        in-memory state.

        This is the public entry point: whether the internal load succeeds,
        fails, or quarantines a bad file, the disk version stamp is synced
        into memory on exit (for concurrent-write protection). The real
        loading logic is in :meth:`_load_impl`.
        """
        try:
            self._load_impl()
        finally:
            self._disk_stamp = (
                self._backend.store_stamp() if self._backend
                else self._stamp_of(self.persist_path)
            )
        # Only a **clean** load counts as "memory aligned with disk"; at that
        # point this process's prior changes are all persisted and need no
        # replay. When the load had problems (bad file quarantined,
        # structure unparseable), keep the dirty sets so we do not also lose
        # the fact "these changes still need replaying".
        if self.last_load_issue is None:
            self._dirty_ids.clear()
            self._deleted_ids.clear()

    def _load_impl(self) -> None:
        """Load memories and vectors from disk, replacing the current
        in-memory state.

        Legacy format compatibility (v1, a single inline ``embedding`` per
        record): legacy vectors go into the ``__legacy__`` slot -- not lost,
        but excluded from the active model's retrieval; the layer above
        backfills the current model's vectors, or ``adopt_legacy()`` claims
        them explicitly.

        Corruption handling (defect N1): **a file that exists but cannot be
        read** no longer silently clears state. The bad file is quarantined,
        renamed to ``<path>.corrupt-<timestamp>`` to preserve the scene, the
        exception details are recorded in :attr:`last_load_issue` with an
        explicit warning; records already in memory are not wiped. Only "the
        file never existed" (fresh directory) is the normal silent-return
        path.

        Also supports recovery when "the main file is missing but the
        ``.bak`` is still there" -- the previous ``save()`` crashed between
        "backing up the old file" and "replacing with the new file", leaving
        that state, where the backup is the only valid copy.
        """
        self.last_load_issue = None
        self.persisted_model_key = None
        if self.persist_path is None:
            return

        # G1: disk reads go through the backend. The json flow's "main file
        # -> .bak fallback -> quarantine the scene" logic has moved into
        # JsonBackend; sqlite reports "library won't open" itself. The data
        # both return is isomorphic to the legacy memories.json, so the
        # parsing below needed zero changes.
        backend = self._backend or get_backend(
            os.path.dirname(os.path.abspath(self.persist_path))
        )
        data, issue = backend.load_memories()
        if issue is not None:
            self.last_load_issue = issue
        if data is None:
            # Fresh directory (issue None, silent return) or unrecoverable (issue already recorded)
            if issue is not None:
                self._invalidate()
            return
        if issue is not None and issue.get("kind") == "recovered":
            logging.warning(
                "the main file was corrupt and has been restored from backup; the corrupt file quarantined as %s",
                issue.get("quarantined"),
            )
        # The disk already has a full baseline ⇒ later saves can use row-level increments
        self._full_saved = True

        try:
            # Model metadata (slug mapping + per-model dimensions)
            self._model_slugs = {}
            self.model_dims = {}
            for key, meta in (data.get("models") or {}).items():
                key = str(key)
                if isinstance(meta, dict):
                    slug = str(meta.get("slug") or "")
                    if slug:
                        self._model_slugs[key] = slug
                    self.model_dims[key] = int(meta.get("dim", 0))

            # The active model follows the constructor (current embedder); the dimension comes from that model's recorded dimension
            if self.model_key in self.model_dims and self.model_dims[self.model_key]:
                self.dim = self.model_dims[self.model_key]

            self._records = []
            self._id_to_index.clear()
            for rec_dict in data.get("records", []):
                rec = MemoryRecord.from_dict(rec_dict)
                # Inline record vectors (legacy format, or JSON exported with_embeddings)
                for key, vec in rec.embeddings.items():
                    self._record_dim(vec, key)
                self._records.append(rec)
                self._id_to_index[rec.id] = len(self._records) - 1

            # Active model at last persist (for preferred_model_key; left None if missing)
            self.persisted_model_key = (str(data.get("model_key") or "") or None)

            # Inject vectors from the .npz (file mapping when mmap is enabled and viable, otherwise the original path)
            if _vectors_mmap_enabled() and self._load_vectors_mmap():
                pass
            else:
                self._load_vectors()
            self._invalidate()
        except Exception as e:
            # JSON parses but the structure is wrong (truncated fields,
            # scrambled types) -- quarantine the scene here too, otherwise
            # the next save would overwrite the still-readable backup with
            # this half-built data.
            q = self._quarantine(self.persist_path)
            self.last_load_issue = {
                "kind": "unparseable",
                "detail": {"error": str(e)},
                "quarantined": [q] if q else [],
            }
            logging.error(
                "解析记忆数据 %s 失败: %s；已隔离为 %s，内存重置为空状态",
                self.persist_path, e, q,
            )
            self._records = []
            self._id_to_index.clear()
            self._invalidate()

    def _quarantine(self, path: str) -> str | None:
        """Rename a corrupted file to ``<path>.corrupt-<timestamp>``,
        preserving the incident scene.

        No deletion: the bad file is the only incident evidence, and a human
        may still be able to salvage records from it. Multiple corruptions
        within the same second append a sequence number to avoid overwriting
        each other.

        Returns:
            The new quarantined path; ``None`` if the file does not exist or
            the rename fails.
        """
        if not os.path.isfile(path):
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = f"{path}.corrupt-{stamp}"
        n = 1
        while os.path.exists(target):
            n += 1
            target = f"{path}.corrupt-{stamp}-{n}"
        try:
            os.replace(path, target)
            return target
        except OSError as e:
            logging.error("隔离损坏文件 %s 失败: %s", path, e)
            return None

    def _load_vectors(self) -> None:
        """Inject vectors into records (via the backend: the npz file or the
        sqlite vectors table)."""
        backend = self._backend
        if backend is None:
            return
        arrays = backend.load_vectors()
        if not arrays:
            return
        slug_to_model = {slug: key for key, slug in self._model_slugs.items() if slug}
        for key, vec in arrays.items():
            if "::" not in key:
                continue
            slug, _, rec_id = key.partition("::")
            model_key = slug_to_model.get(slug)
            if model_key is None:
                continue
            rec = self.get(rec_id)
            if rec is None:
                continue
            rec.set_vector(model_key, vec)
            self._record_dim(vec, model_key)

    def _load_vectors_mmap(self) -> bool:
        """HIPPOCAMPUS_VECTORS_MMAP=1: inject vectors via file mapping.

        Sidecar: ``<base>.vectors.bin`` (fp32 sequential blob) +
        ``<base>.vectors.bin.json`` (``npz_stamp`` + ``keys: {slug::id: [offset, nelem]}``).
        Missing bin or a changed npz stamp (a save happened) → one-shot
        sequential rebuild (temp file + ``os.replace`` to prevent concurrent
        file tearing). All records share **one** memmap mapping; each record
        holds only a slice view.

        Returns:
            True = injected via mmap; False = not applicable / failed (the
            caller falls back to the original npz path). Any exception is
            swallowed inside this method -- an mmap problem must never
            become a load failure.
        """
        try:
            npz_path = self.vectors_path
            if not npz_path or not os.path.isfile(npz_path):
                return False          # sqlite backend / no npz: not applicable
            # Note: vectors_path = splitext(persist_path)[0] + ".vectors.npz";
            # splitext again only strips ".npz" ⇒ the sidecar base must be
            # taken from persist_path, otherwise you get double-suffixed
            # names like memories.vectors.vectors.bin.
            base = os.path.splitext(self.persist_path)[0]
            bin_path = base + ".vectors.bin"
            idx_path = base + ".vectors.bin.json"
            stamp = list(self._stamp_of(npz_path) or ())
            index: dict | None = None
            if os.path.isfile(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and loaded.get("keys") \
                        and list(loaded.get("npz_stamp") or ()) == stamp \
                        and os.path.isfile(bin_path):
                    index = loaded
            if index is None:
                entries: dict[str, list[int]] = {}
                with open(bin_path + ".tmp", "wb") as out:
                    offset = 0
                    for key, arr in np.load(npz_path).items():
                        v = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
                        out.write(v.tobytes())
                        entries[key] = [offset, int(v.size)]
                        offset += int(v.size)
                bin_used = os.path.basename(bin_path)
                try:
                    os.replace(bin_path + ".tmp", bin_path)
                except OSError:
                    # Windows: replace fails on the file lock while the old
                    # memmap is still open (same-process load_state / an
                    # unreleased old store in tests both hit this).
                    # Fallback = write a uniquely named bin; the index
                    # records the actual filename.
                    bin_used = (f"{os.path.basename(bin_path)}."
                                f"{os.getpid()}-{int(time.time() * 1000) % 10 ** 9}")
                    os.replace(bin_path + ".tmp",
                               os.path.join(os.path.dirname(bin_path), bin_used))
                index = {"npz_stamp": stamp, "keys": entries, "bin": bin_used}
                with open(idx_path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(index, f, ensure_ascii=False)
                os.replace(idx_path + ".tmp", idx_path)

            keys = index["keys"]
            bin_file = index.get("bin") or os.path.basename(bin_path)
            mm = np.memmap(os.path.join(os.path.dirname(bin_path), bin_file),
                           dtype=np.float32, mode="r")
            self._vectors_mm = mm      # hold the mapping to prevent Windows handle drift after GC
            slug_to_model = {slug: key for key, slug in self._model_slugs.items() if slug}
            n_set = 0
            for vkey, (off, ne) in keys.items():
                slug, _, rec_id = str(vkey).partition("::")
                model_key = slug_to_model.get(slug)
                rec = self.get(rec_id)
                if model_key is None or rec is None:
                    continue
                vec = mm[off:off + ne]      # read-only slice view, 0 bytes of anonymous memory
                rec.embeddings[model_key] = vec
                self._record_dim(vec, model_key)
                n_set += 1
            return n_set > 0 or not keys
        except Exception as e:  # noqa: BLE001 -- mmap is always just an optimization, never a failure
            logging.warning("向量 mmap 加载失败，回退 npz 路径: %s", e)
            return False
