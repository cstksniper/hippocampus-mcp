# -*- coding: utf-8 -*-
"""State persistence (StateCheckpoint).

The primary format is **npz** (``lnn_state.npz``): ``MemoryCore.get_state()``
produces numpy arrays uniformly, so persisting/reading back needs **no torch at
all** — part of the "zero torch in the production process" goal. ``lnn_state.pt``
from historical libraries remains readable (lazy ``import torch``; when torch is
absent from the environment, warn and safely reset, consistent with the existing
"corrupt → reset" semantics).

Writes use atomic replacement (write ``.tmp`` first, then ``os.replace``) to
prevent a mid-way crash from corrupting data; the optional backup mechanism
(``.bak``) keeps the previous version for rollback recovery.
"""

import json
import os
import sys
import time
from typing import Any

import numpy as np

# npz payload key names (format v1)
_KEY_HAS_HX = "has_hx"
_KEY_HX = "hx"
_KEY_LAST = "last_output"
_KEY_WINDOW = "window"
_KEY_WINDOW_K = "window_k"
_KEY_SCALARS = "scalars_json"
_KEY_VERSION = "format_version"

_NPZ_VERSION = 1


def _stderr_print(*args, **kwargs) -> None:
    """Diagnostic output always goes to stderr.

    **In MCP stdio mode, stdout is the JSON-RPC protocol channel**: anything
    written to stdout gets parsed by the client as a protocol message, raising
    "Invalid JSON" and possibly dropping the connection outright (measured,
    triggered by the LNN state dimension-mismatch notice; see
    memory_core.load_state). Hence diagnostics go to stderr instead.
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _to_array(value: Any) -> np.ndarray:
    """torch tensor / numpy / scalar → numpy array (the unified shape for checkpoint payloads)."""
    if hasattr(value, "detach"):  # torch tensor (may appear on the legacy .pt path)
        return value.detach().cpu().numpy()
    return np.asarray(value)


def payload_from_state(state: dict) -> dict:
    """Compress ``MemoryCore.get_state()`` (+ scalars appended by the manager) into the npz payload.

    Module-level function: ``_write_npz`` and the migration subprocess
    ``pt_migrate`` share the same encoding, preventing format drift between the
    two sites.
    """
    hx = state.get("hx")
    window = list(state.get("window", []) or [])
    scalars = {}
    for k, v in state.items():
        if k in ("hx", "last_output", "window"):
            continue
        scalars[k] = v
    return {
        _KEY_VERSION: np.array(_NPZ_VERSION),
        _KEY_HAS_HX: np.array(hx is not None),
        _KEY_HX: _to_array(hx) if hx is not None else np.zeros((0,), np.float32),
        _KEY_LAST: _to_array(state.get("last_output")),
        _KEY_WINDOW: (
            np.stack([_to_array(t) for t in window])
            if window
            else np.zeros((0,), np.float32)
        ),
        _KEY_WINDOW_K: np.array(len(window)),
        _KEY_SCALARS: np.array(json.dumps(scalars, ensure_ascii=False)),
    }


class StateCheckpoint:
    """Disk persistence of the memory core's state (npz primary format + legacy .pt compatibility read).

    Args:
        path: checkpoint file path (``lnn_state.npz``).
    """

    def __init__(self, path: str):
        self.path = path

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    @property
    def tmp_path(self) -> str:
        """Temporary file path used for atomic writes."""
        return self.path + ".tmp"

    @property
    def bak_path(self) -> str:
        """Backup file path of the previous version."""
        return self.path + ".bak"

    @property
    def legacy_pt_path(self) -> str:
        """The historical-format ``lnn_state.pt`` (same directory, same basename)."""
        root, _ = os.path.splitext(self.path)
        return root + ".pt"

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    def save(self, state: dict) -> None:
        """Serialize the state dict to disk (convenience interface).

        Internally delegates to :meth:`save_with_backup`, i.e. **atomic write +
        backup of the previous version**. (Earlier versions used the
        non-backing :meth:`save_atomic` here, so the ``.bak`` fallback branch
        never had a backup available — defect N2. Defaults to backing up, to
        avoid stepping on the same pit again.)

        On persistence failure only a warning is printed, no exception is
        raised, guaranteeing the caller does not crash because of it.

        Args:
            state: state dict to save (usually from MemoryCore.get_state()).
        """
        if not self.save_with_backup(state):
            _stderr_print("[StateCheckpoint] save failed; state was not persisted (see the error output above)")

    def save_atomic(self, state: dict) -> bool:
        """Atomic write: write ``.tmp`` first, then ``os.replace`` onto the target file on success.

        Does not back up the old version; use :meth:`save_with_backup` if you
        need to keep the previous version. Even if the process crashes mid-write,
        the target file can only be the old version or the new version, never
        half-written.

        Args:
            state: state dict to save.

        Returns:
            True on successful write, False on failure.
        """
        tmp = self.tmp_path
        try:
            self._ensure_dir()
            payload = payload_from_state(state)
            target_eff = tmp if tmp.endswith(".npz") else tmp + ".npz"
            np.savez(target_eff, **payload)
            if target_eff != tmp:
                os.replace(target_eff, tmp)
        except Exception as exc:
            _stderr_print(f"[StateCheckpoint] save_atomic write failed: {exc}")
            self._safe_remove(tmp)
            return False
        try:
            os.replace(tmp, self.path)
            return True
        except Exception as exc:
            _stderr_print(f"[StateCheckpoint] save_atomic replace failed: {exc}")
            self._safe_remove(tmp)
            return False

    def save_with_backup(self, state: dict) -> bool:
        """Atomic write with a backup of the previous version.

        Full flow: write ``.tmp`` -> back up the current target file as ``.bak``
        -> atomically replace the target. The old file is backed up before the
        replacement, so even if the new checkpoint is corrupt, the previous
        version can be restored from ``.bak``.

        Args:
            state: state dict to save.

        Returns:
            True on successful write, False on failure.
        """
        tmp = self.tmp_path
        bak = self.bak_path
        try:
            self._ensure_dir()
            # 1. Write the temporary file
            payload = payload_from_state(state)
            target_eff = tmp if tmp.endswith(".npz") else tmp + ".npz"
            np.savez(target_eff, **payload)
            if target_eff != tmp:
                os.replace(target_eff, tmp)
            # 2. Back up the current target file (if present)
            if os.path.isfile(self.path):
                os.replace(self.path, bak)
            # 3. Atomically replace the target file
            os.replace(tmp, self.path)
            return True
        except Exception as exc:
            _stderr_print(f"[StateCheckpoint] save_with_backup write failed: {exc}")
            self._safe_remove(tmp)
            return False

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    def load(self) -> dict:
        """Restore the state dict from disk.

        Loads the npz primary file first; on failure tries to restore from
        ``.bak``; if neither exists, tries the historical legacy ``.pt`` (lazy
        torch). If all fail, returns an empty dict. State arrays are returned
        uniformly as **numpy** (torch tensors are converted at the exit).

        A corrupt file is **quarantined** by renaming to ``.corrupt-<timestamp>``
        instead of being left in place: otherwise the next
        :meth:`save_with_backup` would "back up" that bad file over the ``.bak``
        that could have been lifesaving.

        Returns:
            The deserialized state dict; ``{}`` when no usable file exists.
        """
        main_err: Exception | None = None
        if os.path.isfile(self.path):
            try:
                return self._read_npz(self.path)
            except Exception as exc:
                main_err = exc
                _stderr_print(f"[StateCheckpoint] main file load failed ({self.path}): {exc}")
        # 2. Fall back to the backup file (reached whether the main file is missing or corrupt)
        data: dict | None = None
        if os.path.isfile(self.bak_path):
            try:
                _stderr_print(f"[StateCheckpoint] attempting restore from backup ({self.bak_path})")
                data = self._read_npz(self.bak_path)
            except Exception as exc:
                _stderr_print(f"[StateCheckpoint] backup file load failed ({self.bak_path}): {exc}")
        if data is not None:
            if main_err is not None:
                q = self._quarantine(self.path)
                if q:
                    _stderr_print(f"[StateCheckpoint] quarantined corrupt main file -> {q}")
            return data
        # 3. legacy .pt (historical-format libraries; lazy torch — safely give up when absent from the environment)
        legacy = self._load_legacy_pt()
        if legacy is not None:
            if main_err is not None:
                q = self._quarantine(self.path)
                if q:
                    _stderr_print(f"[StateCheckpoint] quarantined corrupt main file -> {q}")
            return legacy
        # 4. All failed: quarantine the scene (preserve evidence, avoid repeated failures)
        if main_err is not None:
            q = self._quarantine(self.path)
            if q:
                _stderr_print(f"[StateCheckpoint] quarantined corrupt main file -> {q}")
        if os.path.isfile(self.bak_path):
            q = self._quarantine(self.bak_path)
            if q:
                _stderr_print(f"[StateCheckpoint] quarantined corrupt backup file -> {q}")
        return {}

    def _read_npz(self, path: str) -> dict:
        with np.load(path, allow_pickle=False) as z:
            version = int(z[_KEY_VERSION])
            if version > _NPZ_VERSION:
                raise ValueError(f"npz format version is too new: {version} > {_NPZ_VERSION}")
            has_hx = bool(z[_KEY_HAS_HX])
            hx = np.asarray(z[_KEY_HX], dtype=np.float32) if has_hx else None
            last = np.asarray(z[_KEY_LAST], dtype=np.float32)
            window_arr = np.asarray(z[_KEY_WINDOW], dtype=np.float32)
            k = int(z[_KEY_WINDOW_K])
            import json

            scalars = json.loads(str(z[_KEY_SCALARS]))
        if k > 0:
            dim = window_arr.shape[1]
            window = [window_arr[i].copy() for i in range(k)]
            _ = dim  # (k, dim) already carries the dimension info
        else:
            window = []
        out: dict[str, Any] = dict(scalars)
        out["hx"] = hx
        out["last_output"] = last
        out["window"] = window
        return out

    def _load_legacy_pt(self) -> dict | None:
        """Read the historical-version ``lnn_state.pt``.

        🔴 **Read in place only in processes where torch is already present**
        (evaluation scripts etc.) — the main process (MCP server) aims for
        **zero resident torch**, and lazily spawning the migration subprocess
        **deterministically hangs** in this machine's server parent-process
        context (measured: the subprocess burns 31ms of CPU then blocks on a
        native call during interpreter startup, py-spy shows no Python stack;
        running standalone for 4.5s works fine. A native-level issue, not dug
        into further). When torch is missing: warn once + return None (safe
        reset; ``reference_time`` is recovered via the meta JSON fallback, hx
        re-accumulates — weights are fixed (the production path does not
        train), so re-accumulation has clean semantics).
        To move legacy state into the new format: run the migration script once
        offline in a **separate process** (not inside the main process, avoiding
        the hang above).
        """
        for cand in (self.legacy_pt_path, self.legacy_pt_path + ".bak"):
            if not os.path.isfile(cand):
                continue
            # Already migrated (npz present) should never reach here; defense: if the npz exists, just read it
            if os.path.isfile(self.path):
                try:
                    return self._read_npz(self.path)
                except Exception as exc:
                    _stderr_print(f"[StateCheckpoint] npz read failed ({self.path}): {exc}")
                    return None
            if "torch" not in sys.modules:
                _stderr_print(
                    f"[StateCheckpoint] found legacy LNN state {cand}. The torch-free main process does not read it -- "
                    "hx/window will re-accumulate (reference_time falls back to the meta JSON). "
                    "To keep the old temporal state, run offline: python scripts/migrate_lnn_state.py"
                )
                return None
            try:
                import torch

                data = torch.load(cand, map_location="cpu", weights_only=False)
            except Exception as exc:
                _stderr_print(f"[StateCheckpoint] legacy state read failed ({cand}): {exc}")
                continue
            out: dict[str, Any] = {}
            for k, v in (data or {}).items():
                if k in ("hx", "last_output", "window"):
                    continue
                out[k] = v
            out["hx"] = _to_array(data["hx"]) if data.get("hx") is not None else None
            out["last_output"] = (
                _to_array(data["last_output"])
                if data.get("last_output") is not None
                else None
            )
            out["window"] = [_to_array(t) for t in (data.get("window") or [])]
            return out
        return None

    def exists(self) -> bool:
        """Whether a **recoverable** checkpoint exists.

        The main file being missing while ``.bak`` / legacy ``.pt`` remain (the
        last save crashed between "back up the old file" and "replace with the
        new file") also counts as existing — otherwise that backup would never
        be read and the state would be lost for nothing.

        Returns:
            True if any of npz / backup / legacy .pt exists, otherwise False.
        """
        return (
            os.path.isfile(self.path)
            or os.path.isfile(self.bak_path)
            or os.path.isfile(self.legacy_pt_path)
            or os.path.isfile(self.legacy_pt_path + ".bak")
        )

    def _quarantine(self, path: str) -> str | None:
        """Rename the corrupt file to ``<path>.corrupt-<timestamp>``, preserving the incident scene.

        Returns:
            The new path after quarantine; ``None`` when the file does not exist or the rename fails.
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
        except OSError as exc:
            _stderr_print(f"[StateCheckpoint] failed to quarantine corrupt file {path} : {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_dir(self) -> None:
        """Ensure the directory containing the target file exists."""
        dir_path = os.path.dirname(os.path.abspath(self.path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def _safe_remove(self, path: str) -> None:
        """Delete a file safely, ignoring missing files or permission errors."""
        try:
            for cand in (path, path + ".npz"):
                if os.path.isfile(cand):
                    os.remove(cand)
        except OSError:
            pass
