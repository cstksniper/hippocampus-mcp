"""Introspection and reclamation of the local embedding-model cache
(read-only scanning + explicit reclamation).

**Why this module is needed**: the model cache is the only thing in this
project that keeps growing. A Chinese engine, weights plus ONNX
artifacts, can reach 1.3 GB, and often more than half of that is
redundancy that is **never read** (see below). Users should neither
delete directories by hand in ``~/.cache/huggingface`` (delete the wrong
one and nothing starts) nor guess which file is deletable to save space.

**Four disciplines** (each corresponds to one real mis-deletion risk):

1. **Read-only first**: :func:`scan_models` only measures, never touches
   the disk; :func:`plan_prune` only produces a plan. Actual deletion
   requires explicitly calling :func:`apply_prune`, and the path must
   lie inside a cache root and the directory name must be ``models--*``.
2. **Referenced models are not deleted**: any model pointed to by an
   ``embedder_model`` in any ``library.json`` is protected. Delete it
   and that library cannot load the model on next startup (offline there
   is no remedy, and the vector dims would not match either). An empty
   string means "not pinned", not a reference.
3. **The current default model is not deleted**: otherwise the next
   startup needs to re-download it online; offline environments simply
   cannot start.
4. **Redundancy tiers**: ``onnx_int8/`` is the quantized output of
   ``scripts/export_onnx.py``; **runtime code never reads it**
   (``ONNX_SUBDIR`` only accepts ``"onnx"``) ⇒ pure redundancy,
   deletable in any case; ``onnx/`` is the ONNX fast path, used by the
   ``auto`` backend, and is redundant only when
   ``HIPPOCAMPUS_EMB_BACKEND=torch`` is set explicitly.

**Why the size is not the repo's nominal value**: the
``BAAI/bge-base-zh-v1.5`` repo is 410 MB, yet locally it is often
1.3 GB -- the entire difference is in ``onnx/`` (389 MB) and
``onnx_int8/`` (558 MB). The former is the price of the ONNX fast path
(measured 2.3x faster encoding, 7x faster cold start); the latter is
pure dead weight.

Watch out for symlinks: the HF cache links ``snapshots/<rev>/xxx`` to
``blobs/<hash>``; counting both sides doubles the size. So measurement
**deduplicates by realpath**, walking ``snapshots`` first (only there
are semantically meaningful paths like ``onnx/``), then the remaining
files.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

#: Directory prefix of each model in the HF cache (``org/name`` → ``models--org--name``)
MODEL_DIR_PREFIX = "models--"

#: Weight-like file suffixes. Used to also classify orphan files in blobs into the 「weights」 tier.
_WEIGHT_SUFFIXES = (".bin", ".safetensors", ".h5", ".msgpack", ".ckpt", ".npz")

#: Kinds of reclamation action
KIND_MODEL = "model"
KIND_ONNX_INT8 = "onnx_int8"
KIND_ONNX = "onnx"


@dataclass(frozen=True)
class CachedModel:
    """A model directory in the cache.

    Attributes:
        model_id: in ``org/name`` form (recovered from the directory name).
        path: absolute path of the model directory.
        total_bytes: total occupancy of the directory (after dedup).
        weight_bytes: weight files (``.bin`` / ``.safetensors`` ...).
        onnx_bytes: ONNX artifacts under ``onnx/`` (fast path, used by ``auto``).
        int8_bytes: quantized artifacts under ``onnx_int8/`` (**never read at runtime**).
        other_bytes: the remaining odds and ends (configs, vocab, README...).
    """

    model_id: str
    path: str
    total_bytes: int = 0
    weight_bytes: int = 0
    onnx_bytes: int = 0
    int8_bytes: int = 0
    other_bytes: int = 0

    @property
    def engine_bytes(self) -> int:
        """The part that actually gets loaded into memory (weights + ONNX fast path)."""
        return self.weight_bytes + self.onnx_bytes


@dataclass(frozen=True)
class PruneAction:
    """One pending reclamation action.

    When ``kind`` is :data:`KIND_MODEL`, the whole model directory is
    deleted; otherwise a subdirectory under it.
    """

    kind: str
    model_id: str
    path: str
    bytes_: int
    reason: str


@dataclass
class PrunePlan:
    """Reclamation plan: executable actions + notes on what was skipped as protected."""

    actions: list[PruneAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes_ for a in self.actions)


# ======================================================================
# 1. Size accounting
# ======================================================================
def _slug_to_model_id(slug: str) -> str:
    """``models--org--name`` → ``org/name``.

    HF only replaces ``/`` with ``--``, so the reverse replacement
    suffices; but the prefix must be **stripped only once**, otherwise
    models whose names inherently contain ``--`` would be rewritten.
    """
    body = slug[len(MODEL_DIR_PREFIX):] if slug.startswith(MODEL_DIR_PREFIX) else slug
    return body.replace("--", "/")


def _categorize(rel_path: str) -> str:
    """Tier by relative path: onnx_int8 > onnx > weight suffix > other."""
    norm = rel_path.replace("\\", "/").lower()
    parts = norm.split("/")
    if "onnx_int8" in parts:
        return "int8"
    if "onnx" in parts:
        return "onnx"
    if norm.endswith(_WEIGHT_SUFFIXES):
        return "weight"
    return "other"


def _iter_files(root_dir: str):
    """List files recursively (does not follow directory symlinks)."""
    for dirpath, _dirnames, filenames in os.walk(root_dir, followlinks=False):
        for name in filenames:
            yield os.path.join(dirpath, name)


def scan_model_dir(path: str) -> CachedModel:
    """Measure the occupancy breakdown of a single model directory (dedup by realpath; does not follow directory links)."""
    buckets = {"weight": 0, "onnx": 0, "int8": 0, "other": 0}
    seen: set[str] = set()

    def _add(file_path: str, rel: str) -> None:
        try:
            real = os.path.realpath(file_path)
        except OSError:
            return
        if real in seen:
            return
        try:
            size = os.path.getsize(file_path)      # follows symlinks to take the target's size
        except OSError:
            return
        seen.add(real)
        buckets[_categorize(rel)] += size

    # Walk snapshots first: only paths there carry onnx/ onnx_int8/ semantics
    snapshots = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots):
        for f in _iter_files(snapshots):
            _add(f, os.path.relpath(f, path))
    # Then the rest (blobs / refs / trees). Files linked above are already deduped by realpath.
    sep_snap = os.sep + "snapshots" + os.sep
    for f in _iter_files(path):
        if sep_snap in f:
            continue
        _add(f, os.path.relpath(f, path))

    total = sum(buckets.values())
    return CachedModel(
        model_id=_slug_to_model_id(os.path.basename(path)),
        path=path,
        total_bytes=total,
        weight_bytes=buckets["weight"],
        onnx_bytes=buckets["onnx"],
        int8_bytes=buckets["int8"],
        other_bytes=buckets["other"],
    )


def scan_models(cache_roots: list[str]) -> list[CachedModel]:
    """Scan multiple cache roots, returning the model list sorted by size
    descending (**read-only**).

    When the same model appears under multiple roots, account for each
    ``(model_id, path)`` separately -- they really are two disk
    occupancies; merging the display would mislead about how much can be
    saved.
    """
    out: list[CachedModel] = []
    seen_paths: set[str] = set()
    for root in cache_roots:
        root = os.path.abspath(os.path.expanduser(root)) if root else ""
        if not root or not os.path.isdir(root):
            continue
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for name in names:
            if not name.startswith(MODEL_DIR_PREFIX):
                continue
            path = os.path.join(root, name)
            if not os.path.isdir(path) or path in seen_paths:
                continue
            seen_paths.add(path)
            out.append(scan_model_dir(path))
    out.sort(key=lambda m: (-m.total_bytes, m.model_id))
    return out


# ======================================================================
# 2. Reclamation plan (pure function, never touches disk)
# ======================================================================
def plan_prune(
    models: list[CachedModel],
    *,
    in_use: dict[str, list[str]] | None = None,
    protected: set[str] | None = None,
    keep_onnx: bool = True,
) -> PrunePlan:
    """Work out what can be safely reclaimed.

    Args:
        models: the result of :func:`scan_models`.
        in_use: ``model_id → list of library names``, from each library's
            ``library.json``.
        protected: models to keep even when unreferenced (current default /
            explicitly specified via env).
        keep_onnx: True = keep ``onnx/`` (the ``auto``/``onnx`` backends
            need it). False (only with an explicit ``torch`` backend)
            counts ``onnx/`` as redundant too.

    Returns:
        :class:`PrunePlan`. **Describes only; does not execute.**
    """
    in_use = in_use or {}
    protected = protected or set()
    plan = PrunePlan()

    for m in models:
        refs = in_use.get(m.model_id) or []
        is_protected = m.model_id in protected

        # int8 artifacts are never read at runtime; reclaimable whether or not the model is in use
        if m.int8_bytes:
            int8_dirs = _subdirs(m.path, "onnx_int8")
            plan.actions.append(PruneAction(
                kind=KIND_ONNX_INT8,
                model_id=m.model_id,
                path=int8_dirs[0] if int8_dirs else os.path.join(m.path, "onnx_int8"),
                bytes_=m.int8_bytes,
                reason="onnx_int8/ is not read at runtime (ONNX_SUBDIR only accepts onnx; the quantized artifact is an export intermediate)",
            ))

        # Note: a flag is used here instead of continue inside the branch --
        # a referenced model may still warrant deleting onnx/ (with an
        # explicit torch backend); an early continue would skip that path
        # entirely, presenting as 「torch was set yet not a byte is
        # reclaimed」.
        keep_model = bool(refs) or is_protected
        if refs:
            plan.notes.append(
                f"{m.model_id}: referenced by libraries {', '.join(sorted(refs))}; kept whole")
        elif is_protected:
            plan.notes.append(
                f"{m.model_id}: current default / explicitly specified model; kept whole")

        if keep_model:
            # Only for kept models is 「whether onnx is redundant」 even a question
            if m.onnx_bytes and not keep_onnx:
                onnx_dirs = _subdirs(m.path, "onnx")
                plan.actions.append(PruneAction(
                    kind=KIND_ONNX,
                    model_id=m.model_id,
                    path=onnx_dirs[0] if onnx_dirs else os.path.join(m.path, "onnx"),
                    bytes_=m.onnx_bytes,
                    reason="explicit torch backend: the ONNX fast path is unused (can be regenerated via export_onnx.py)",
                ))
            continue

        # Neither referenced nor the default ⇒ the whole directory is deletable
        plan.actions.append(PruneAction(
            kind=KIND_MODEL,
            model_id=m.model_id,
            path=m.path,
            bytes_=m.total_bytes,
            reason="no library references it, and it is not the current default model",
        ))

    plan.actions.sort(key=lambda a: -a.bytes_)
    return plan


def _subdirs(model_path: str, name: str) -> list[str]:
    """Find real directories under ``snapshots/*/<name>`` (possibly multiple revisions)."""
    out: list[str] = []
    snapshots = os.path.join(model_path, "snapshots")
    if not os.path.isdir(snapshots):
        return out
    for rev in sorted(os.listdir(snapshots)):
        cand = os.path.join(snapshots, rev, name)
        if os.path.isdir(cand):
            out.append(cand)
    return out


# ======================================================================
# 3. Execution (the only place that touches disk)
# ======================================================================
def _is_safe_target(path: str, cache_roots: list[str]) -> bool:
    """Guard rail before deletion: the path must lie inside one of the
    cache roots, and the outermost model directory name must be compliant.

    The point of this check is to turn 「accidentally passing an empty
    string / a parent directory」 into an explicit error, instead of one
    ``rmtree`` wiping the entire ``~/.cache``.
    """
    target = os.path.abspath(path)
    if not target or not os.path.isdir(target):
        return False
    for root in cache_roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not root or target == root:
            continue
        if not target.startswith(root + os.sep):
            continue
        rel = os.path.relpath(target, root)
        first = rel.split(os.sep)[0]
        if first.startswith(MODEL_DIR_PREFIX):
            return True
    return False


def apply_prune(actions: list[PruneAction], cache_roots: list[str]) -> tuple[int, list[str]]:
    """Execute reclamation, returning ``(bytes freed, error list)``.

    A single action failing **does not abort** the rest: partial deletion
    is more common than none at all (on Windows, an occupied file raises
    PermissionError); per-item reporting is the only way to handle it.
    """
    freed = 0
    errors: list[str] = []
    for act in actions:
        if not _is_safe_target(act.path, cache_roots):
            errors.append(f"{act.kind} {act.model_id}: path outside the cache roots, skipped {act.path}")
            continue
        try:
            before = _dir_size(act.path)
            shutil.rmtree(act.path)
        except OSError as e:
            errors.append(f"{act.kind} {act.model_id}：{e}")
            continue
        freed += before
    return freed, errors


def _dir_size(path: str) -> int:
    total = 0
    seen: set[str] = set()
    for f in _iter_files(path):
        real = os.path.realpath(f)
        if real in seen:
            continue
        seen.add(real)
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    return total


# ======================================================================
# 4. Display
# ======================================================================
def human_size(n: int) -> str:
    """Human-readable size (1 KB = 1024 B, close to ``du``'s convention).

    Bytes carry no decimals; from KB up, always one decimal place --
    mixing them (``2 KB`` next to ``2.0 MB``) looks like two different
    conventions.
    """
    size = float(max(0, n))
    if size < 1024:
        return f"{size:.0f} B"
    size /= 1024
    for unit in ("KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_report(
    models: list[CachedModel],
    *,
    in_use: dict[str, list[str]] | None = None,
    protected: set[str] | None = None,
    cache_roots: list[str] | None = None,
) -> str:
    """Render a human-readable report (width kept within 78 columns)."""
    in_use = in_use or {}
    protected = protected or set()
    lines: list[str] = []
    lines.append("Local embedding-model cache")
    for root in (cache_roots or []):
        lines.append(f"  roots: {root}")
    if not models:
        lines.append("  (no models--* directories found in the scan)")
        return "\n".join(lines)

    lines.append("-" * 74)
    for m in models:
        lines.append(f"{m.model_id}   {human_size(m.total_bytes)}")
        detail = (f"weights {human_size(m.weight_bytes)}"
                  f" + ONNX {human_size(m.onnx_bytes)}"
                  f" + int8 {human_size(m.int8_bytes)}"
                  f" + other {human_size(m.other_bytes)}")
        lines.append(f"    {detail}")
        refs = in_use.get(m.model_id) or []
        flags = []
        flags.append("in use: " + (", ".join(sorted(refs)) if refs else "(no library references)"))
        if m.model_id in protected:
            flags.append("current default")
        lines.append("    " + "   ".join(flags))
        lines.append(f"    path: {m.path}")
    lines.append("-" * 74)
    lines.append(f"total {human_size(sum(m.total_bytes for m in models))} "
                 f"(the engine portions that must be kept "
                 f"{human_size(sum(m.engine_bytes for m in models))})")
    return "\n".join(lines)


def format_plan(plan: PrunePlan) -> str:
    """Render the reclamation plan (the last confirmation before ``--yes``)."""
    if not plan.actions and not plan.notes:
        return "Nothing reclaimable."
    lines: list[str] = []
    if plan.actions:
        lines.append(f"Reclaimable {human_size(plan.total_bytes)}:")
        for a in plan.actions:
            label = {"model": "[whole model]", "onnx_int8": "[int8]     ", "onnx": "[ONNX]     "}.get(a.kind, a.kind)
            lines.append(f"  {label} {a.model_id}  {human_size(a.bytes_)}")
            lines.append(f"           {a.reason}")
            lines.append(f"           {a.path}")
    if plan.notes:
        if lines:
            lines.append("")
        lines.append("Keep notes:")
        for n in plan.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)
