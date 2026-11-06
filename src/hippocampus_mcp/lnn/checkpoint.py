# -*- coding: utf-8 -*-
"""状态持久化 (StateCheckpoint).

主格式为 **npz**（``lnn_state.npz``）：``MemoryCore.get_state()``
统一产出 numpy 数组，落盘/读回**全程不需要 torch**——这是"生产进程零 torch"目标
的一部分。历史库里的 ``lnn_state.pt`` 仍可读（惰性 ``import torch``；环境里没有
torch 时告警并安全重置，与"损坏→重置"既有语义一致）。

写入采用原子替换 (先写 ``.tmp`` 再 ``os.replace``) 防止中途崩溃损坏数据;
可选的备份机制 (``.bak``) 保留上一版本以便回滚恢复.
"""

import json
import os
import sys
import time
from typing import Any

import numpy as np

# npz 载荷键名（格式 v1）
_KEY_HAS_HX = "has_hx"
_KEY_HX = "hx"
_KEY_LAST = "last_output"
_KEY_WINDOW = "window"
_KEY_WINDOW_K = "window_k"
_KEY_SCALARS = "scalars_json"
_KEY_VERSION = "format_version"

_NPZ_VERSION = 1


def _stderr_print(*args, **kwargs) -> None:
    """诊断输出一律走 stderr。

    **MCP stdio 模式下 stdout 是 JSON-RPC 协议通道**：任何写到 stdout 的内容都会被
    客户端当成协议消息解析，报 "Invalid JSON" 并可能直接断连（实测由 LNN 状态
    维度不匹配的提示触发，见 memory_core.load_state）。故诊断信息改走 stderr。
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _to_array(value: Any) -> np.ndarray:
    """torch 张量 / numpy / 标量 → numpy 数组（checkpoint 载荷统一形态）。"""
    if hasattr(value, "detach"):  # torch tensor（旧 .pt 路径可能出现）
        return value.detach().cpu().numpy()
    return np.asarray(value)


def payload_from_state(state: dict) -> dict:
    """把 ``MemoryCore.get_state()``（+ manager 追加的标量）压成 npz 载荷。

    模块级函数：``_write_npz`` 与迁移子进程 ``pt_migrate`` 共用同一份编码，
    防止两处格式漂移。
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
    """记忆核状态的磁盘持久化（npz 主格式 + legacy .pt 兼容读）.

    Args:
        path: checkpoint 文件路径（``lnn_state.npz``）。
    """

    def __init__(self, path: str):
        self.path = path

    # ------------------------------------------------------------------ #
    # 路径辅助
    # ------------------------------------------------------------------ #
    @property
    def tmp_path(self) -> str:
        """原子写入使用的临时文件路径."""
        return self.path + ".tmp"

    @property
    def bak_path(self) -> str:
        """上一版本备份文件路径."""
        return self.path + ".bak"

    @property
    def legacy_pt_path(self) -> str:
        """历史格式的 ``lnn_state.pt``（同目录同基名）。"""
        root, _ = os.path.splitext(self.path)
        return root + ".pt"

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #
    def save(self, state: dict) -> None:
        """将状态字典序列化到磁盘 (便捷接口).

        内部委托至 :meth:`save_with_backup`，即**原子写 + 备份上一版本**。
        （早期版本这里走的是不备份的 :meth:`save_atomic`，于是 ``.bak`` 回退
        分支永远无备份可用——缺陷 N2。默认带备份，避免再次踩同一个坑。）

        持久化失败时仅打印警告, 不抛出异常, 保证调用方不会因此崩溃.

        Args:
            state: 待保存的状态字典 (通常来自 MemoryCore.get_state()).
        """
        if not self.save_with_backup(state):
            _stderr_print("[StateCheckpoint] save failed; state was not persisted (see the error output above)")

    def save_atomic(self, state: dict) -> bool:
        """原子写入: 先写 ``.tmp``, 成功后 ``os.replace`` 替换目标文件.

        不备份旧版本; 如需保留上一版本请使用 :meth:`save_with_backup`.
        即使写入过程中崩溃, 目标文件也只可能是旧版本或新版本, 不会被写坏.

        Args:
            state: 待保存的状态字典.

        Returns:
            写入成功返回 True, 失败返回 False.
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
        """原子写入并备份上一版本.

        完整流程: 写 ``.tmp`` -> 将当前目标文件备份为 ``.bak`` -> 原子替换目标.
        替换前备份旧文件, 这样即使新 checkpoint 损坏, 也能从 ``.bak`` 恢复上一版本.

        Args:
            state: 待保存的状态字典.

        Returns:
            写入成功返回 True, 失败返回 False.
        """
        tmp = self.tmp_path
        bak = self.bak_path
        try:
            self._ensure_dir()
            # 1. 写入临时文件
            payload = payload_from_state(state)
            target_eff = tmp if tmp.endswith(".npz") else tmp + ".npz"
            np.savez(target_eff, **payload)
            if target_eff != tmp:
                os.replace(target_eff, tmp)
            # 2. 备份当前目标文件 (若存在)
            if os.path.isfile(self.path):
                os.replace(self.path, bak)
            # 3. 原子替换目标文件
            os.replace(tmp, self.path)
            return True
        except Exception as exc:
            _stderr_print(f"[StateCheckpoint] save_with_backup write failed: {exc}")
            self._safe_remove(tmp)
            return False

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def load(self) -> dict:
        """从磁盘恢复状态字典.

        优先加载 npz 主文件; 失败则尝试从 ``.bak`` 恢复; 都没有再试历史
        legacy ``.pt``（惰性 torch）。仍失败则返回空字典.
        状态数组统一以 **numpy** 形态返回（torch 张量在出口处转换）。

        损坏的文件会被**隔离**重命名为 ``.corrupt-<时间戳>`` 而不是留在原地:
        否则下一次 :meth:`save_with_backup` 会把这个坏文件"备份"过去，
        覆盖掉本可救命的 ``.bak``。

        Returns:
            反序列化后的状态字典; 无可用文件时返回 ``{}``.
        """
        main_err: Exception | None = None
        if os.path.isfile(self.path):
            try:
                return self._read_npz(self.path)
            except Exception as exc:
                main_err = exc
                _stderr_print(f"[StateCheckpoint] main file load failed ({self.path}): {exc}")
        # 2. 回退到备份文件（主文件缺失或损坏都走这里）
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
        # 3. legacy .pt（历史格式的库；惰性 torch——环境里没有就安全放弃）
        legacy = self._load_legacy_pt()
        if legacy is not None:
            if main_err is not None:
                q = self._quarantine(self.path)
                if q:
                    _stderr_print(f"[StateCheckpoint] quarantined corrupt main file -> {q}")
            return legacy
        # 4. 全部失败: 隔离现场（保留证据，避免反复失败）
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
            _ = dim  # (k, dim) 已含维度信息
        else:
            window = []
        out: dict[str, Any] = dict(scalars)
        out["hx"] = hx
        out["last_output"] = last
        out["window"] = window
        return out

    def _load_legacy_pt(self) -> dict | None:
        """读历史版本的 ``lnn_state.pt``。

        🔴 **只在 torch 已在场的进程里就地读**（评测脚本等）——主进程（MCP
        server）的目标是**零 torch 常驻**，而惰性拉迁移子进程在本机 server
        父进程上下文里**确定性挂死**（实测：子进程 31ms CPU 后
        阻塞在解释器启动期的原生调用，py-spy 无 Python 栈；独立运行 4.5s
        正常。native 级问题，不再深挖）。缺 torch 时告警一次 + 返回 None
        （安全重置；``reference_time`` 由 meta JSON 兜底恢复，hx 重新积累
        ——权重固定（生产路径不训练），重积累的语义是干净的）。
        想把 legacy 状态搬进新格式：在**独立进程**里离线跑一次迁移脚本
        （不进主进程，避开上面的挂死）。
        """
        for cand in (self.legacy_pt_path, self.legacy_pt_path + ".bak"):
            if not os.path.isfile(cand):
                continue
            # 已迁移过（npz 在）就不该再走到这里；防御：npz 存在则直接读它
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
        """判断是否存在**可恢复**的 checkpoint.

        主文件缺失但 ``.bak`` / legacy ``.pt`` 仍在（上一次保存在「备份旧文件」
        与「替换新文件」之间崩溃）也算存在——否则那份备份永远不会被读到，
        状态白丢。

        Returns:
            npz / 备份 / legacy .pt 任一存在返回 True, 否则 False.
        """
        return (
            os.path.isfile(self.path)
            or os.path.isfile(self.bak_path)
            or os.path.isfile(self.legacy_pt_path)
            or os.path.isfile(self.legacy_pt_path + ".bak")
        )

    def _quarantine(self, path: str) -> str | None:
        """把损坏文件重命名为 ``<path>.corrupt-<时间戳>``，保留事故现场.

        Returns:
            隔离后的新路径; 文件不存在或重命名失败时返回 ``None``.
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
    # 内部工具
    # ------------------------------------------------------------------ #
    def _ensure_dir(self) -> None:
        """确保目标文件所在目录存在."""
        dir_path = os.path.dirname(os.path.abspath(self.path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def _safe_remove(self, path: str) -> None:
        """安全删除文件, 忽略不存在或权限错误."""
        try:
            for cand in (path, path + ".npz"):
                if os.path.isfile(cand):
                    os.remove(cand)
        except OSError:
            pass
