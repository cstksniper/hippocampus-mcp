"""基于 numpy 内存数组的向量存储（多模型版）。

设计要点（v0.4）：
    - **向量是派生物**：记忆本体（文本 + 结构）存在 JSON，嵌入向量单独存
      二进制 .npz。删除 .npz 只会让下次检索时补算，不会丢记忆。
    - **多模型并存**：每条记录持有 ``{模型标识: 向量}``。切换嵌入模型时，
      已有该模型向量的记录直接可用；没有的由上层补算（无需重新导入文档）。
    - 检索只使用**当前活跃模型**的向量；缺向量的记录通过 ``missing_ids()``
      暴露给上层补算。

后续可平滑替换为 chromadb 等专业向量数据库，只要保持对外接口不变即可。
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


# 滚动备份保留份数（zcode 需求 C2）。0 = 关闭（只留固定的 .bak）。
# 之所以还要滚动副本：``.bak`` 每次保存都被覆盖，等于只有"上一次保存"能回退。
# 真出事时想找回的是"今天下午改坏之前的库"，而不是最后一次保存前的状态。
BACKUP_ROTATE_KEEP = 3


# ── 构造注入缝 ────────────────────────────────────────────────────
# 本模块（及 backend / lnn）受 G1 依赖矩阵约束**不 import config**；旋钮的
# 登记处与唯一 env 读取点在 config（ENV_MAP）。生产路径由唯一构造方
# **engine**（manager import 时）经 :func:`apply_config` 灌入 config 字符串
# 运输值；解析序（env 先、注入值回退）与零漂移论证见
# ``common/config_bridge.py`` 模块 docstring。
from ..common.config_bridge import apply_config as _bridge_apply_config
from ..common.config_bridge import resolve as _resolve


def apply_config(**values: str) -> None:
    """构造方注入（包装 config_bridge；``save_lock_timeout`` 额外刷新模块
    快照 :data:`SAVE_LOCK_TIMEOUT`——M1b ring 三件先例：模块属性即测试
    monkeypatch 面；float() 脏值在此抛错＝原 import 期崩溃语义平移）。"""
    _bridge_apply_config(**values)
    if "save_lock_timeout" in values:
        global SAVE_LOCK_TIMEOUT
        SAVE_LOCK_TIMEOUT = float(values["save_lock_timeout"])


def snapshot_min_hours() -> float:
    """sqlite 滚动快照的最短间隔（小时）；``0`` = 每次 save 都做（仅调试用）。

    为什么只有 sqlite 需要节流：``VACUUM INTO`` 实测约 **1 秒 / 64MB 库**，
    而 ``save()`` 每 10 步就走一次；json 后端那份只是几 MB 的 ``copy2``，
    沿用"每次保存都转一份"的既有语义。
    """
    raw = _resolve("HIPPOCAMPUS_SNAPSHOT_MIN_HOURS", "snapshot_min_hours", "6")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 6.0


# 多进程共用一个库（两个 ZCode 工作区各起一个 MCP 子进程）时的写策略：
#   1 / true（默认）→ 加跨进程锁 + 把本进程的改动重放到对方最新状态之上（合并）
#   0 / false      → 退回旧的乐观锁语义：检测到别人写过就抛 ConcurrentWriteError
# 为什么默认改成合并：旧的"检测到冲突就报错"在双窗口下是**必然发生**的 ——
# 两边各自 load 一次快照，之后任何一次保存都会撞上，用户侧只会看到保存失败。
def multi_writer_enabled() -> bool:
    """是否启用多进程合并保存（每次调用时读环境变量，便于测试与运行时切换）。"""
    return _resolve("HIPPOCAMPUS_MULTI_WRITER", "multi_writer", "1").lower() not in (
        "0", "false", "no", "off",
    )


def _serialized(func):
    """进程内写互斥：把方法体整体放进 ``self._io_lock``。

    为什么需要它（daemon 化之后才成为必需件）：跨进程那套保护（filelock +
    脏集合重放）假设"另一方是**另一个进程**、持有独立的内存副本"。daemon 模式
    下同一进程内 N 条连接 = N 个线程**共享同一个 VectorStore 对象**，于是
    ``add()`` 与 ``save()`` 会交错，而 ``_write_to_disk`` 的"取快照 → 清脏集"
    不是原子的 —— 另一方在中间 add 的记录既不在写出的文件里、也不再被记为
    "改过"，下一次 ``_merge_with_disk`` 就把它静默丢掉。

    只包写操作（add / update / delete / save），读操作（recall / count）不加锁，
    否则 daemon 的并发读会被串行化，白丢掉共享进程带来的吞吐。
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._io_lock:
            return func(self, *args, **kwargs)
    return wrapper


# 等锁上限（秒）。正常写入是毫秒级；等这么久还没拿到，说明对方卡住了
# （例如正在整体覆写 9000+ 条的大库），宁可明确失败也不要无限等下去。
# 默认值登记进 config（``save_lock_timeout``），实际值由 engine
# import 时经 :func:`apply_config` 注入刷新——模块属性即 monkeypatch 面
# （ring 快照语义）；这里保留的 float() 解析与脏值抛错的语义原样。
SAVE_LOCK_TIMEOUT: float = 60.0


def _matrix_dtype() -> np.dtype:
    """检索缓冲矩阵的 dtype（默认 fp32，每次读 env 便于测试/探针切换）。

    为什么 fp32 够：向量落盘本来就是 float32（npz / sqlite blob 均按 fp32
    序列化），``record.embeddings`` 也已归一为 ``array('f')``（双存修复）
    ——检索缓冲是这两者的派生物，用 fp64 是"对已舍入的数据假装精确"，
    白付一倍内存（真库 9859×768 ⇒ 58MB vs 29MB）。相对误差 ~1e-7，只会
    在近平局的相邻两条上可能翻转排序，A/B 验收锚定 top-k 一致率 100%。
    逃逸舱：``HIPPOCAMPUS_MATRIX_DTYPE=fp64`` 回旧行为。
    """
    raw = _resolve("HIPPOCAMPUS_MATRIX_DTYPE", "matrix_dtype", "fp32")
    raw = raw.strip().lower()
    if raw in ("fp64", "float64", "double"):
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _vectors_mmap_enabled() -> bool:
    """向量文件映射加载开关（默认关；每次读 env 便于测试/探针切换）。

    开启后 ``record.embeddings`` 从匿名 ``array('f')``（真库 ~32.6MB）变成
    np.memmap 只读切片——file-backed、可被系统回收、多进程共享页缓存。
    判定收益看 **Private Bytes** 不看瞬时 WS（file-backed 页被触碰后仍会
    驻留 WS，但不是"占住不放"的那类内存，见 h3_selective_quant REPORT §2.3）。
    """
    return _resolve(
        "HIPPOCAMPUS_VECTORS_MMAP", "vectors_mmap", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


class ConcurrentWriteError(RuntimeError):
    """磁盘上的记忆文件已被**其他进程**改写，且本次保存**无法安全合并**。

    触发场景：同一个 ``persist_dir`` 被两个进程同时持有（两个 ZCode 窗口各起
    一个 MCP 子进程；或导入脚本与正在运行的 MCP 服务并存）。两边各自 ``load()``
    拿到快照、各自改内存、再各自**整体覆写**文件，后保存的一方会让先保存方的
    写入整段消失，且双方都打印"已保存"——**静默数据丢失**。

    当前处理（默认，见 ``multi_writer_enabled``）：``load()`` 记下文件版本戳
    （mtime_ns + size），``save()`` 前先在**跨进程写锁**内核对；不一致就把本
    进程的改动重放到磁盘最新状态之上（``_merge_with_disk``）—— 双方的新增与
    删除都不会丢。所以两个窗口并存时**默认不再报错**。

    仍然抛本异常的情形：「磁盘上的 JSON 读不出来（宁可明确失败，也不拿半截
    数据去合并）」·「HIPPOCAMPUS_MULTI_WRITER=0 显式关掉合并」·「拿不到写锁，
    通常意味着环境里没有 filelock」。

    处置：调用方应重新 ``load()``（MCP 工具 ``load_state``）拿到最新状态后重做
    本次修改；或者确认另一进程已退出。若确实要强制覆盖（例如明知另一进程是
    只读的），设 ``HIPPOCAMPUS_ALLOW_STALE_SAVE=1`` 降级为告警。
    """


class VectorStore:
    """基于 numpy 内存数组的多模型向量存储。

    内部维护一份 ``list[MemoryRecord]``（记录自带各模型向量），
    检索时按活跃模型惰性构建 ``(N, dim)`` 矩阵并缓存。

    Args:
        dim: 当前活跃模型的向量维度。
        persist_path: 记忆本体（JSON）持久化路径；为 ``None`` 时不落盘。
        model_key: 当前活跃模型标识（如 ``"BAAI/bge-base-zh-v1.5"``）。
    """

    # 持久化格式版本：1 = 单向量旧格式，2 = 多model + 独立 .npz
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

        # id -> 在 _records 中的行索引
        self._id_to_index: dict[str, int] = {}
        # 保持插入顺序的记录列表
        self._records: list[MemoryRecord] = []
        # 活跃模型的矩阵缓存：**定长缓冲 + 已用行数**，惰性构建。
        # 追加写入走增量维护（``_sync_row``），不再整体重建 —— 这是写入
        # O(N²) 的根因修复（缺陷 D2）：原先 add() 无条件失效，去重检索又
        # 每次触发全量 vstack，于是"写第 k 条"的代价是 O(k)。
        self._matrix: np.ndarray | None = None      # 缓冲，行数 >= _matrix_n
        self._matrix_n: int = 0                      # 已用行数
        self._matrix_ids: list[str] = []             # 行 -> id（长度 == _matrix_n）
        self._id_to_row: dict[str, int] = {}          # id -> 行号

        # 已知模型的维度：model_key -> dim
        self.model_dims: dict[str, int] = {}
        # 模型标识 -> npz 内短名（避免 "/" 出现在 npz key 中）
        self._model_slugs: dict[str, str] = {}
        # 落盘时记录的活跃模型（JSON 的 "model_key" 字段）。这是"这个库上次
        # 是**用哪个模型**在跑"的权威记录，load 时恢复，供"让嵌入模型跟随
        # 数据"用（见 preferred_model_key）。
        self.persisted_model_key: str | None = None

        # 上一次 load 的异常报告（None = 干净）。
        # 损坏时不会静默清空，而是把现场记录在这里供上层告警/测试断言。
        self.last_load_issue: dict[str, Any] | None = None

        # 并发写保护：最近一次"与磁盘同步"时的文件版本戳。
        # None = 磁盘上还没有本文件（或从未同步过）。
        self._disk_stamp: tuple[int, int] | None = None
        # 滚动备份保留份数（见 BACKUP_ROTATE_KEEP）。置 0 可关闭。
        self.backup_rotate_keep: int = BACKUP_ROTATE_KEEP
        # 最近一次被拦截的并发写冲突（供上层告警/测试断言）
        self.last_conflict: dict[str, Any] | None = None

        # ---- 多进程共享同一个 persist_dir（见 save / _merge_with_disk）----
        # 本进程自上次成功落盘以来"改过 / 删过"的 id。save 时若发现磁盘已被
        # 别的进程改写，就用这两个集合把本进程的改动**重放**到对方的最新状态
        # 之上，而不是整体覆盖 —— 后者会静默抹掉对方的写入。
        # 记录里没有 updated_at（timestamp 是记忆发生时间，不是修改时间），
        # 所以做不了记录级 LWW，只能靠"谁改过谁重放"来合并。
        self._dirty_ids: set[str] = set()
        self._deleted_ids: set[str] = set()
        # 最近一次合并的统计（供日志 / 测试断言）
        self.last_merge: dict[str, Any] | None = None

        # ---- 进程内写互斥（daemon 化后才出现的必需件）----
        # 跨进程那套（filelock + 脏集合重放）假设"另一方是另一个进程、有独立的
        # 内存副本"。daemon 模式下同一进程内有 N 条连接 = N 个线程**共享同一个
        # VectorStore 对象**，于是：
        #   * A 在 _write_to_disk 里取了 _records 快照、还没清 _dirty_ids；
        #   * B 此时 add() 了新记录（同进程同列表，A 的快照里没有它）；
        #   * A 把 _dirty_ids 清空 —— B 那条既不在 A 写出的文件里，也不再被
        #     记为"改过"；
        #   * 下次任何一方的 save 触发 _merge_with_disk，merged 只认磁盘版本 +
        #     脏集重放 ⇒ 那条记录**静默消失**。
        # 实测（tests/test_f9_4b_daemon_mcp_e2e.py E2）：两连接各写 15 条，
        # 修复前落盘 20~25 条，丢失数与时序相关（不可复现的偶发丢数据）。
        # 所以 add / update / delete / save 必须共用这一把锁。
        self._io_lock = threading.RLock()

        # ---- G1：落盘后端（json 多文件 / sqlite 单库）----
        # 只替换"读盘 / 落盘"这一层，内存结构（_records / 矩阵 / 脏集）不动，
        # 于是检索 / 合并 / 去重那些业务逻辑一行都不用改。见 store/backend.py。
        self._backend: StoreBackend | None = None
        if persist_path:
            self._backend = get_backend(
                os.path.dirname(os.path.abspath(persist_path))
            )
        # 磁盘上是否已有完整基线。首次必须全量写：增量写的前提是库里已经有
        # 一份完整数据，"只 upsert 脏行"否则会写出一个只有几条记录的库。
        self._full_saved = False
        # 向量 mmap（HIPPOCAMPUS_VECTORS_MMAP=1）：挂住映射对象
        self._vectors_mm = None

    # ------------------------------------------------------------------
    # 派生路径
    # ------------------------------------------------------------------
    @property
    def vectors_path(self) -> str | None:
        """二进制向量文件路径（与 ``persist_path`` 同目录）。"""
        if not self.persist_path:
            return None
        base, _ = os.path.splitext(self.persist_path)
        return base + ".vectors.npz"

    def _slug_for(self, model_key: str) -> str:
        """为模型标识分配稳定的短名（用于 npz 键前缀）。"""
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
    # 内部辅助方法
    # ------------------------------------------------------------------
    def _invalidate(self) -> None:
        """整体失效矩阵缓存，下次搜索时全量重建。

        只在结构性变更（删除 / 换model / 裁剪model / 加载）时调用。
        新增记录与原地改向量走 ``_sync_row`` 增量维护，不整体重建。
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
        """校验并转换为 1D float64 向量。"""
        key = model_key or self.model_key
        emb = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if key == self.model_key and emb.shape[0] != self.dim:
            raise ValueError(
                f"embedding dimension {emb.shape[0]} 与当前model {key}'s dimension {self.dim} -- mismatch"
            )
        self._record_dim(emb, key)
        return emb

    def _active_matrix(self) -> tuple[list[str], np.ndarray]:
        """取活跃模型的 (ids, matrix)，按需重建。

        Returns:
            (ids, matrix)：ids 与 matrix 行一一对应，顺序与记录插入顺序一致。
            matrix 是内部缓冲的**视图**（行数 == len(ids)），调用方只读。
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
        """矩阵缓冲扩容（翻倍），均摊 O(1)。"""
        old = self._matrix
        if old is None:
            return
        new = np.zeros((max(old.shape[0] * 2, 8), old.shape[1]), dtype=old.dtype)
        new[: self._matrix_n] = old[: self._matrix_n]
        self._matrix = new

    def _sync_row(self, memory_id: str, vector: Any | None) -> None:
        """把一条记录的活跃模型向量增量同步进矩阵缓存。

        仅当矩阵已构建时生效；矩阵未构建则保持惰性，下次检索自然包含。
        行需要被**移除**时（记录丢了当前模型的向量）numpy 无法 O(1) 删行，
        退化为整体失效 —— 这条路径在正常写入中不会出现。
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
            # 维度异常（理论上被 _validate_embedding 拦住），安全退化
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
        """批量计算查询向量与活跃模型所有存储向量的余弦相似度。

        query 转成**矩阵同 dtype** 再乘：fp32 矩阵 @ fp64 向量会让 numpy
        临时把整份矩阵升维 fp64（一份 58MB 级临时数组，fp32 化就白改了）。
        返回仍转回 float64，保持调用方拿到的分数口径不变。
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
        """在 id 子集内找相似度落在 ``[low, high)`` 的对，按相似度降序返回。

        为什么要单独开这个口子：**两两比对比"一条对全部"贵一个量级**，而
        ``consolidate`` 的矛盾检测正是桶内两两比对——它要的是"同一个实体下的
        记忆里谁和谁像"，不是"哪条最像查询"。

        实现是**分块矩阵乘**而不是逐对 Python 循环。两者的**数值完全一致**
        （都是归一化后的点积），但代价差两个数量级：真实大库（数千条 /
        数百个实体桶）实测从分钟级降到秒级。
        逐对版的开销根本不在点积上，而在**每对都要把 Python list 转成
        numpy 数组**（``manager.vector_of`` 返回 ``record.embeddings[key]``，
        每次 ``np.asarray`` 768 个元素）——这一点必须写在注释里，否则后人
        很容易以为"点积本来就慢"而放弃优化。

        Args:
            ids: 待比对的子集（同一个实体桶）。
            low / high: 相似度区间 ``[low, high)``。**上界是刻意的**：
                ≥ high 的对通常是同一条事实的两个副本，不构成"矛盾"，
                交由写入侧去重处理，这里不重复报告。
            limit: 每个子集最多返回多少对（按相似度降序）。
            chunk: 分块行数。控制临时矩阵峰值内存（``chunk × n × 8`` 字节）。
            normalize: 是否先按行归一化。默认 True（返回真余弦值）；
                传 False 时按点积筛选，仅在调用方已知向量已归一化时用。

        Returns:
            ``[(id_a, id_b, sim), ...]``，按 sim 降序；只含上三角（``i < j``），
            不自我比对。缺失向量的 id 自动跳过。
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
                lo = s + i + 1          # 只算上三角
                if lo >= len(rows):
                    break
                row = sims[i, lo:]
                hit = np.nonzero((row >= low) & (row < high))[0]
                if hit.size:
                    for g in hit:
                        best.append((float(row[g]), s + i, lo + int(g)))
            # 边算边剪：把候选压回 limit，避免大桶把内存吃满
            if len(best) > limit * 4:
                best.sort(key=lambda x: -x[0])
                del best[limit:]

        best.sort(key=lambda x: -x[0])
        return [(kept[i], kept[j], sim) for sim, i, j in best[:limit]]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    @_serialized
    def add(self, record: MemoryRecord) -> str:
        """添加一条记忆，返回其 id。

        若 ``record.id`` 为空则自动生成 UUID；若 id 已存在则执行原地更新。
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

        # 增量维护矩阵缓存（缓存未构建时保持惰性，下次检索重建即包含本条）
        self._sync_row(record.id, record.vector_for(self.model_key))
        self._mark_dirty(record.id)
        return record.id

    def set_vector(
        self, memory_id: str, model_key: str, vector: Any
    ) -> bool:
        """为指定记忆写入某个模型的向量（补算路径使用）。"""
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
        """余弦相似度检索（仅活跃模型），返回 (record, similarity) 降序列表。

        Args:
            allow_ids: 只允许返回这些 id 的记忆（None = 不限）。标签过滤
                走这里：先在**检索前**把候选集收窄，而不是检索完再筛——
                后者会在"满足条件的都在 top_k 之外"时漏召回。
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
            # 屏蔽不在白名单里的行（-inf 而不是 0：0 仍可能被"矮子里拔将军"选中）
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
        """根据 id 获取记忆记录，不存在则返回 None。"""
        idx = self._id_to_index.get(memory_id)
        if idx is None:
            return None
        return self._records[idx]

    @_serialized
    def update(self, memory_id: str, **fields: Any) -> bool:
        """更新指定记忆的字段（如 recall_count）。返回是否更新成功。

        支持任意 dataclass 字段；``embedding`` 为兼容旧调用保留，
        等价于写入当前活跃模型的向量。
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

        # 缺陷 D4：pin 原先只在 MemoryRecord.__post_init__ 里按 importance 设定，
        # 而这里走 setattr 会绕过它 —— 于是"importance>=0.7 自动钉住"只对新建生效，
        # 合并（dedup）路径提升 importance 时 pin 仍是 False、stats().pinned 也是 0。
        # 仅在本次确实改了 importance、且调用方没有显式指定 pin 时才重算。
        if "importance" in fields and "pin" not in fields:
            record.sync_pin()
        self._mark_dirty(memory_id)
        return True

    @_serialized
    def delete(self, memory_id: str) -> bool:
        """删除指定记忆，返回是否删除成功。"""
        idx = self._id_to_index.get(memory_id)
        if idx is None:
            return False

        del self._records[idx]
        # 重建索引映射，保持列表紧凑
        self._id_to_index.clear()
        for i, rec in enumerate(self._records):
            self._id_to_index[rec.id] = i
        self._invalidate()
        # 删除也要记账：合并时要把这条从对方的最新状态里去掉，
        # 否则对方磁盘上的副本会被"保留下来"，表现为删了又活。
        self._dirty_ids.discard(memory_id)
        self._deleted_ids.add(memory_id)
        return True

    def count(self) -> int:
        """返回当前存储的记忆数量。"""
        return len(self._records)

    def iter_records(self) -> Iterator[MemoryRecord]:
        """M3：只读遍历全部记录的**公开**入口。

        迁移前上层（brain/core、engine/manager 等 45 处）直接摸
        ``self._records``——那让 VectorStore 的内部表示（list + 插入序）
        成了事实上 API，改存储结构就会炸一层。本方法把「遍历」收归
        端口契约：调用方拿迭代器，不拿容器。

        语义与直接 ``iter(self._records)`` 完全一致（同一 list 对象、
        同一顺序、惰性求值）——所以这是**零行为变化**的收编，不是
        语义增强。线程安全边界与原来相同：进程内写互斥由调用方
        持有的锁负责，本方法不加锁也不该加。
        """
        return iter(self._records)

    # ------------------------------------------------------------------
    # 多模型相关 API
    # ------------------------------------------------------------------
    def set_model(self, model_key: str, dim: int | None = None) -> None:
        """切换活跃模型（检索将使用该模型的向量）。"""
        self.model_key = model_key
        if dim is not None:
            self.dim = dim
        elif model_key in self.model_dims:
            self.dim = self.model_dims[model_key]
        self._invalidate()

    def missing_ids(self, model_key: str | None = None) -> list[str]:
        """返回缺少指定（默认活跃）模型向量的记忆 id 列表。"""
        key = model_key or self.model_key
        return [r.id for r in self._records if not r.has_vector(key)]

    def model_stats(self) -> dict[str, dict[str, int]]:
        """统计各模型的向量覆盖情况。

        Returns:
            ``{模型标识: {"vectors": 有向量的条数, "dim": 维度}}``。
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
        """判定"这个库实际跑在哪个嵌入模型上"，供嵌入模型跟随数据。

        优先级：
            1. 落盘记录的 ``model_key``（JSON 顶层字段，= 上次保存时的活跃
               模型）——但要求它**确实有向量覆盖**，否则是"某次用错模型跑
               过并落了盘"留下的脏值，不能盲从；
            2. 覆盖率最高的非 ``__legacy__`` 槽位（向量在哪，模型就是谁）；
            3. 都没有（空库 / 只有旧格式单向量）→ ``None``。

        ``__legacy__`` 永远不作为候选：它是"旧格式单向量"的占位槽，认领它
        需要显式的 :meth:`adopt_legacy`，不能被当成一个真实的嵌入模型跟随。

        返回 ``None`` 表示"库内没有可跟随的模型"，由调用方决定用什么默认值。
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
        # 覆盖率最高者胜；并列时取字典序最小，保证可复现
        return sorted(counted.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def prune_models(self, keep: set[str]) -> int:
        """丢弃不在 ``keep`` 中的模型向量，返回丢弃的向量总数。"""
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
        """把 ``__legacy__`` 槽位的向量认领为 ``model_key``（维度需一致）。

        Returns:
            被认领的向量条数。
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
    # 持久化
    # ------------------------------------------------------------------
    @_serialized
    def save(self, check_conflict: bool = True) -> None:
        """持久化：记忆本体写 JSON，向量写 .npz。

        两者都采用"先写临时文件再原子替换"策略，避免写入中断导致损坏。
        向量以 float32 存储（体积减半，检索精度无实质影响）。

        Args:
            check_conflict: 是否做跨进程校验（默认开）。开着时：先取跨进程写锁，
                若磁盘已被别的进程改过，则按 ``HIPPOCAMPUS_MULTI_WRITER``
                （默认 1）合并保存、或（=0 时）明确失败。仅在明确知道另一进程
                不会写、且要强制覆写时才传 False。

        Raises:
            ConcurrentWriteError: 冲突**且无法安全合并**时（磁盘读不出来 /
                显式关闭了多进程合并 / 拿不到写锁）——见类文档。
        """
        if self.persist_path is None:
            return
        if not check_conflict:
            self._write_to_disk()
            return
        # 多进程共存（两个 ZCode 工作区各起一个子进程）时：先拿跨进程写锁——
        # 两个进程会踩同一个 ``memories.json.tmp``，那是比"后写覆盖"更糟的损坏；
        # 再看磁盘是否被改过：改过就把本进程的改动**重放**上去，而不是报错。
        with self._save_lock() as locked:
            self._resolve_conflict(locked)
            self._write_to_disk()

    def _write_to_disk(self) -> None:
        """落盘（记忆本体 + 向量）；成功后清空脏集合。

        落盘动作走 :attr:`_backend`。sqlite 后端在**磁盘已有基线**时只
        upsert 脏行、删掉删过的行 —— 改 1 条不再重写 5.28MB 的 memories.json
        和 32.7MB 的 npz（实测账单）。首次落盘（还没有基线）仍走全量。
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

        # ---- 1. 记忆本体 ----
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
            # 基线可能只写了一半 ⇒ 退回全量，免得之后一直增量补一个残缺库
            self._full_saved = False
            return

        # 落盘后同步磁盘戳（json = 文件 mtime；sqlite = 写序号 + 记录数）。
        # 下一次 save 靠它判断"有没有被别的进程改过"，从而触发合并而非覆盖。
        self._disk_stamp = backend.store_stamp()
        self.last_conflict = None

        # 滚动快照（json 复制 .bak 副本 / sqlite 用 VACUUM INTO）。
        # 放在 L1 落盘后、向量前：保护的是记忆本体，且只留一个调用点
        # （向量分支有早退 return，挂到末尾会漏掉）。
        self._rotate_after_save(backend)

        # ---- 2. 向量 ----
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
                # 删掉的记忆：它的向量已随 delete_memories 清掉，这里再兜一次
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
        """把刚腾出来的 ``.bak`` 另存一份带时间戳的副本，只保留最近若干份。

        ``.bak`` 始终保留不动（load 的回退路径依赖它），滚动的是它的副本。
        时间戳用 ``YYYYMMDD_HHMMSS``——字典序即时间序，清理时直接排序取尾部。
        """
        keep = int(getattr(self, "backup_rotate_keep", BACKUP_ROTATE_KEEP) or 0)
        if keep <= 0:
            return
        directory = os.path.dirname(bak_path) or "."
        prefix = os.path.basename(bak_path) + "-"
        if not os.path.isfile(bak_path):
            return  # 首次保存：主文件原本不存在，没腾出 .bak，无需转存
        dst = os.path.join(directory, prefix + time.strftime("%Y%m%d_%H%M%S"))
        try:
            shutil.copy2(bak_path, dst)
        except OSError as e:
            logging.warning("滚动备份失败 %s: %s", dst, e)
            return
        self._prune_snapshots(directory, prefix, keep)

    def _rotate_snapshot_db(self, backend: Any, keep: int) -> None:
        """sqlite 后端：``VACUUM INTO`` 一份全库快照，保留最近若干份。

        sqlite 下全库就是一个文件（记忆 + 向量 + 派生层 + 实体 + 来源都在里面），
        所以这一份就是完整快照 —— 比 json 时代散装文件的备份更简单也更一致。

        节流见 :func:`snapshot_min_hours`：快照代价约 1 秒 / 64MB，
        不能跟着 ``save()`` 每次都做。
        """
        directory = os.path.dirname(os.path.abspath(self.persist_path))
        prefix = DB_FILENAME + ".bak-"
        if not self._snapshot_due(directory, prefix):
            return
        dst = os.path.join(directory, prefix + time.strftime("%Y%m%d_%H%M%S"))
        if os.path.exists(dst):
            return  # 时间戳精度到秒：同一秒内的多次保存共用一份，跳过
        backend.snapshot_to(dst)
        self._prune_snapshots(directory, prefix, keep)

    def _snapshot_due(self, directory: str, prefix: str) -> bool:
        """距最近一份快照是否已超过最短间隔（目录为空 / 读不到 ⇒ 立刻做）。"""
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
        """只留最近 ``keep`` 份带时间戳的副本（字典序即时间序）。"""
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
        """落盘成功后产出一份历史副本。**失败一律不抛**（与 auto_save 同约定）。

        备份不该把保存搞挂：真出问题时宁可没备份，也不能让这次保存失败。
        json 走既有的 ``.bak`` 复制，sqlite 走 ``VACUUM INTO``。
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
        """文件的版本戳 ``(mtime_ns, size)``；不存在则返回 ``None``。

        用 mtime+size 而不是内容 hash：万条库的 JSON 有几 MB，每次 save 前
        重算 hash 不划算，而"被别的进程动过"这件事用元数据判足够可靠
        （同一秒内改写且体积恰好相同的情况在实践中需要刻意构造）。
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
        """最近一次与磁盘同步（load / save 成功）时的版本戳。"""
        return self._disk_stamp

    def _check_write_conflict(self) -> None:
        """保存前核对磁盘版本；被别的进程改过就抛 :class:`ConcurrentWriteError`。

        仅在**当前有内存记录**时校验：空库覆盖一份非空文件同样危险，所以也拦。
        若磁盘上的文件已被删除（另一进程清了库），同样视为冲突。
        """
        if self.persist_path is None:
            return
        current = self._stamp_of(self.persist_path)
        if current == self._disk_stamp:
            return

        # 从未同步过（stamp is None）且磁盘无文件 → 首次写入，放行
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
    # 多进程共享同一个库：跨进程锁 + 脏集合重放合并
    # ------------------------------------------------------------------
    def _mark_dirty(self, memory_id: str) -> None:
        """记账：本进程改动过这条记忆，落盘时要把它重放到对方的最新状态之上。"""
        if not memory_id:
            return
        self._dirty_ids.add(memory_id)
        self._deleted_ids.discard(memory_id)

    def _rebuild_index(self) -> None:
        """重排 ``_id_to_index`` 并让矩阵缓存失效（合并后调用）。"""
        self._id_to_index = {r.id: i for i, r in enumerate(self._records)}
        self._invalidate()

    @property
    def lock_path(self) -> str | None:
        """跨进程写锁的文件路径（与 ``memories.json`` 同目录）。"""
        if not self.persist_path:
            return None
        return self.persist_path + ".lock"

    @contextmanager
    def _save_lock(self):
        """跨进程互斥（best-effort）：环境里没有 filelock 时退回原乐观锁行为。

        没有这把锁时，两个进程会同时往**同一个** ``memories.json.tmp`` 里写；
        ``os.replace`` 本身是原子的，但临时文件会互相踩 —— 这是比"后写覆盖"
        更糟的一类损坏。所以锁是保底，不只是为了让冲突检测更准。
        """
        path = self.lock_path
        if path is None:
            yield False
            return
        try:
            from filelock import FileLock, Timeout
        except Exception:  # noqa: BLE001 缺依赖不该阻断保存
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
            except Exception:  # noqa: BLE001 释放失败不影响本次保存结果
                pass

    def _read_records_from_disk(self) -> list[MemoryRecord] | None:
        """只读地解析磁盘上的记忆（含向量），**不动内存状态**。

        返回 ``None`` 表示数据在但读不出来 —— 那种情况宁可回到"明确失败"，
        也不要拿一份半截数据去合并（合并会把坏数据写回去，比报错更糟）。

        G2：读盘必须走 :attr:`_backend`。sqlite 下记录在 db 里、没有
        memories.json 可读；不改这里，多进程合并会读到一份"空磁盘"，
        于是本进程的全量覆写把对方刚写的记录全部抹掉。
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

        # 向量：key 形如 "<slug>::<id>"。slug → 模型标识的映射以磁盘上的
        # models 段为准，本进程已知的映射作补充。
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
        """把本进程的改动**重放**到磁盘最新状态之上（多进程合并保存）。

        规则只依赖"谁改过"这一事实 —— 记录里没有 ``updated_at``
        （``timestamp`` 是记忆发生时间，不是修改时间），做不了记录级 LWW：

        * 磁盘有、本进程没碰过 → 原样保留（对方的写入被吸收进内存）
        * 本进程改过（``_dirty_ids``）→ 用本进程内存里的版本覆盖
        * 本进程删过（``_deleted_ids``）→ 从结果中移除
        * 对方新增的记录 → 进入本进程内存（此后本进程也能看到）

        副作用：同一条被两边同时改过时以本进程为准（后写胜），这是没有版本号
        时的必然取舍；但**任何一方的新增/删除都不会丢**。

        Returns:
            True = 可以继续落盘（内存已换成合并结果）；
            False = 磁盘状态读不出来，调用方应退回"明确失败"。
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
        """判定本次保存是否安全；需要时先做合并。不安全则抛异常。

        Args:
            locked: 当前是否已持有跨进程写锁（没拿到锁就不敢合并 —— 合并
                期间若有第三个进程写入，结果仍会互相覆盖）。
        """
        # G2：sqlite 没有 memories.json 可 stat，磁盘戳由后端自己给
        # （写序号 + 记录数）。否则永远判定"没变过"，并发写会直接覆盖对方。
        current = (self._backend.store_stamp() if self._backend
                   else self._stamp_of(self.persist_path))
        if current == self._disk_stamp:
            return
        # 从未同步过（stamp 为 None）且磁盘也没文件 → 首次写入，放行
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
        """从磁盘加载记忆与向量，覆盖当前内存状态。

        本方法是公开入口：无论内部加载成功、失败还是隔离了坏文件，退出时都会
        把磁盘版本戳同步到内存（供并发写保护使用）。真正的加载逻辑见
        :meth:`_load_impl`。
        """
        try:
            self._load_impl()
        finally:
            self._disk_stamp = (
                self._backend.store_stamp() if self._backend
                else self._stamp_of(self.persist_path)
            )
        # 只有**干净**加载才算"内存与磁盘已对齐"，此时此前的本进程改动都已落盘、
        # 无需再重放。加载出过问题（坏文件被隔离、结构不可解析）时保留脏集合，
        # 免得把"这些改动还需要重放"这件事一起丢掉。
        if self.last_load_issue is None:
            self._dirty_ids.clear()
            self._deleted_ids.clear()

    def _load_impl(self) -> None:
        """从磁盘加载记忆与向量，覆盖当前内存状态。

        兼容旧格式（v1，每条记录内嵌单个 ``embedding``）：旧向量归入
        ``__legacy__`` 槽位，不会丢失，但也不参与当前模型的检索——
        由上层补算当前模型向量，或由 ``adopt_legacy()`` 显式认领。

        损坏处理（缺陷 N1）：**文件存在但读不出来**时不再静默清空。坏文件被
        隔离重命名为 ``<path>.corrupt-<时间戳>`` 保留现场，异常详情记入
        :attr:`last_load_issue`，并显式告警；内存中已有记录也不会被抹掉。
        只有"文件本就不存在"（全新目录）才是静默返回的正常路径。

        另外支持"主文件缺失但 ``.bak`` 尚在"的恢复——上一次 ``save()`` 在
        「备份旧文件」与「替换新文件」之间崩溃就会留下这个状态，此时备份是
        唯一的有效副本。
        """
        self.last_load_issue = None
        self.persisted_model_key = None
        if self.persist_path is None:
            return

        # G1：读盘走后端。json 的「主文件 -> .bak 回退 -> 隔离现场」整段逻辑
        # 已搬进 JsonBackend，sqlite 则由它自己报告"库打不开"。两者返回的 data
        # 与旧版 memories.json 同构，下面的解析逻辑因此一行都不用改。
        backend = self._backend or get_backend(
            os.path.dirname(os.path.abspath(self.persist_path))
        )
        data, issue = backend.load_memories()
        if issue is not None:
            self.last_load_issue = issue
        if data is None:
            # 全新目录（issue 为 None，静默返回）或不可恢复（issue 已记好）
            if issue is not None:
                self._invalidate()
            return
        if issue is not None and issue.get("kind") == "recovered":
            logging.warning(
                "the main file was corrupt and has been restored from backup; the corrupt file quarantined as %s",
                issue.get("quarantined"),
            )
        # 磁盘已有完整基线 ⇒ 之后的保存可以走行级增量
        self._full_saved = True

        try:
            # 模型元数据（slug 映射 + 各模型维度）
            self._model_slugs = {}
            self.model_dims = {}
            for key, meta in (data.get("models") or {}).items():
                key = str(key)
                if isinstance(meta, dict):
                    slug = str(meta.get("slug") or "")
                    if slug:
                        self._model_slugs[key] = slug
                    self.model_dims[key] = int(meta.get("dim", 0))

            # 活跃模型以构造方（当前嵌入器）为准；维度取该模型已记录的维度
            if self.model_key in self.model_dims and self.model_dims[self.model_key]:
                self.dim = self.model_dims[self.model_key]

            self._records = []
            self._id_to_index.clear()
            for rec_dict in data.get("records", []):
                rec = MemoryRecord.from_dict(rec_dict)
                # 记录内嵌向量（旧格式，或 with_embeddings 导出的 JSON）
                for key, vec in rec.embeddings.items():
                    self._record_dim(vec, key)
                self._records.append(rec)
                self._id_to_index[rec.id] = len(self._records) - 1

            # 落盘时的活跃模型（供 preferred_model_key 用；缺失留 None）
            self.persisted_model_key = (str(data.get("model_key") or "") or None)

            # 从 .npz 注入向量（mmap 开启且可行时走文件映射，否则原路径）
            if _vectors_mmap_enabled() and self._load_vectors_mmap():
                pass
            else:
                self._load_vectors()
            self._invalidate()
        except Exception as e:
            # JSON 能解析但结构不对（半截字段、类型错乱）——同样隔离现场，
            # 否则下一次 save 会用这份半成品覆盖掉还能读的备份。
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
        """把损坏文件重命名为 ``<path>.corrupt-<时间戳>``，保留事故现场。

        不做删除：坏文件是唯一的事故证据，人工可能还能从里面捞出记录。
        同一秒内多次损坏时追加序号，避免互相覆盖。

        Returns:
            隔离后的新路径；文件不存在或重命名失败时返回 ``None``。
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
        """把向量注入到记录里（经后端：npz 文件 或 sqlite 的 vectors 表）。"""
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
        """HIPPOCAMPUS_VECTORS_MMAP=1：向量走文件映射注入。

        侧车：``<base>.vectors.bin``（fp32 顺序 blob）+
        ``<base>.vectors.bin.json``（``npz_stamp`` + ``keys: {slug::id: [offset, nelem]}``）。
        bin 缺失或 npz 戳变了（save 过）→ 一次性顺序重建（临时文件 +
        ``os.replace`` 防并发撕文件）。所有记录共享**一个** memmap 映射，
        每条记录只持一个切片视图。

        Returns:
            True = 已按 mmap 注入；False = 不适用/失败（调用方回退原 npz 路径）。
            任何异常都在本方法内消化——绝不把 mmap 的问题变成加载失败。
        """
        try:
            npz_path = self.vectors_path
            if not npz_path or not os.path.isfile(npz_path):
                return False          # sqlite 后端 / 无 npz：不适用
            # 注意：vectors_path = splitext(persist_path)[0] + ".vectors.npz"，
            # 再 splitext 只能剥掉 ".npz" ⇒ 侧车 base 必须从 persist_path 取，
            # 否则会拼出 memories.vectors.vectors.bin 这种双缀名。
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
                    # Windows：旧 memmap 还开着时 replace 会因文件锁失败
                    # （同进程 load_state / 测试里旧 store 未释放都会踩）。
                    # 退路 = 写唯一名 bin，index 记录实际文件名。
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
            self._vectors_mm = mm      # 挂住映射，防 GC 后 Windows 句柄漂移
            slug_to_model = {slug: key for key, slug in self._model_slugs.items() if slug}
            n_set = 0
            for vkey, (off, ne) in keys.items():
                slug, _, rec_id = str(vkey).partition("::")
                model_key = slug_to_model.get(slug)
                rec = self.get(rec_id)
                if model_key is None or rec is None:
                    continue
                vec = mm[off:off + ne]      # 只读切片视图，匿名内存 0 字节
                rec.embeddings[model_key] = vec
                self._record_dim(vec, model_key)
                n_set += 1
            return n_set > 0 or not keys
        except Exception as e:  # noqa: BLE001 —— mmap 永远只是优化，不能变成故障
            logging.warning("向量 mmap 加载失败，回退 npz 路径: %s", e)
            return False
