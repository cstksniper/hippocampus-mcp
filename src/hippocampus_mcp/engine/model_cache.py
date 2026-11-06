"""本地嵌入模型缓存的自省与回收（只读扫描 + 显式回收）。

**为什么需要这个模块**：模型缓存是本项目唯一会持续变大的东西。一个中文
引擎连权重带 ONNX 产物能到 1.3 GB，而其中往往有一大半是**永远读不到**的
冗余（见下）。用户既不该手工去 ``~/.cache/huggingface`` 里删目录（删错
就起不来），也不该为了省空间去猜哪个文件能删。

**四条纪律**（每条都对应一次真实的误删风险）：

1. **只读优先**：:func:`scan_models` 只统计、不改盘；:func:`plan_prune`
   只出计划。真正删盘要显式调 :func:`apply_prune`，且路径必须落在缓存
   根之内、目录名必须是 ``models--*``。
2. **被引用的模型不删**：任何 ``library.json`` 的 ``embedder_model`` 指向
   的模型一律保护。删掉它，那个库下次启动就加载不了模型（离线时没有任何
   补救办法，向量维度还对不上）。空字符串表示"没钉死"，不算引用。
3. **当前默认模型不删**：否则下次启动要联网重下，离线环境直接起不来。
4. **冗余分档**：``onnx_int8/`` 是 ``scripts/export_onnx.py`` 的量化产物，
   **运行时代码从不读它**（``ONNX_SUBDIR`` 只有 ``"onnx"``）⇒ 纯冗余，
   任何情况下可删；``onnx/`` 是 ONNX 快路径，``auto`` 后端会用，只有显式
   ``HIPPOCAMPUS_EMB_BACKEND=torch`` 时才算冗余。

**体积为什么不是仓库标称值**：``BAAI/bge-base-zh-v1.5`` 仓库 410 MB，本地
却常是 1.3 GB —— 差额全在 ``onnx/``(389 MB) 与 ``onnx_int8/``(558 MB)。
前者是 ONNX 快路径的代价（实测编码快 2.3x、冷启动快 7x），后者纯属白占。

符号链接要当心：HF 缓存把 ``snapshots/<rev>/xxx`` 链到 ``blobs/<hash>``，
两边都算一遍会把体积翻倍。所以统计时**按 realpath 去重**，且先走
``snapshots``（那里才有 ``onnx/`` 这类有语义的路径）、再走剩余文件。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

#: HF 缓存里每个模型的目录前缀（``org/name`` → ``models--org--name``）
MODEL_DIR_PREFIX = "models--"

#: 权重类文件后缀。用于把 blobs 里的孤儿文件也归到「权重」档。
_WEIGHT_SUFFIXES = (".bin", ".safetensors", ".h5", ".msgpack", ".ckpt", ".npz")

#: 回收动作种类
KIND_MODEL = "model"
KIND_ONNX_INT8 = "onnx_int8"
KIND_ONNX = "onnx"


@dataclass(frozen=True)
class CachedModel:
    """缓存里的一个模型目录。

    Attributes:
        model_id: ``org/name`` 形式（由目录名反推）。
        path: 模型目录绝对路径。
        total_bytes: 该目录总占用（去重后）。
        weight_bytes: 权重文件（``.bin`` / ``.safetensors`` …）。
        onnx_bytes: ``onnx/`` 下的 ONNX 产物（快路径，``auto`` 会用）。
        int8_bytes: ``onnx_int8/`` 下的量化产物（**运行时从不读**）。
        other_bytes: 其余零碎（配置、词表、README…）。
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
        """真正会被加载进内存的那部分（权重 + ONNX 快路径）。"""
        return self.weight_bytes + self.onnx_bytes


@dataclass(frozen=True)
class PruneAction:
    """一条待执行的回收动作。

    ``kind`` 为 :data:`KIND_MODEL` 时删整个模型目录，否则删其下的子目录。
    """

    kind: str
    model_id: str
    path: str
    bytes_: int
    reason: str


@dataclass
class PrunePlan:
    """回收计划：可执行动作 + 被保护而跳过的说明。"""

    actions: list[PruneAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes_ for a in self.actions)


# ======================================================================
# 一、体积统计
# ======================================================================
def _slug_to_model_id(slug: str) -> str:
    """``models--org--name`` → ``org/name``。

    HF 只用 ``--`` 替换 ``/``，所以反向替换即可；但必须**只剥一次前缀**，
    否则名字里本来就带 ``--`` 的模型会被改写。
    """
    body = slug[len(MODEL_DIR_PREFIX):] if slug.startswith(MODEL_DIR_PREFIX) else slug
    return body.replace("--", "/")


def _categorize(rel_path: str) -> str:
    """按相对路径判档：onnx_int8 > onnx > 权重后缀 > 其他。"""
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
    """递归列出文件（不跟随目录符号链接）。"""
    for dirpath, _dirnames, filenames in os.walk(root_dir, followlinks=False):
        for name in filenames:
            yield os.path.join(dirpath, name)


def scan_model_dir(path: str) -> CachedModel:
    """统计单个模型目录的占用构成（按 realpath 去重，不跟随目录链接）。"""
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
            size = os.path.getsize(file_path)      # 跟随符号链接取目标大小
        except OSError:
            return
        seen.add(real)
        buckets[_categorize(rel)] += size

    # 先走 snapshots：只有那里的路径带 onnx/ onnx_int8/ 语义
    snapshots = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots):
        for f in _iter_files(snapshots):
            _add(f, os.path.relpath(f, path))
    # 再走其余（blobs / refs / trees）。被上面链过的文件 realpath 已去重。
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
    """扫描多个缓存根，返回按体积降序的模型列表（**只读**）。

    同一个模型在多个根下出现时按 ``(model_id, path)`` 各自记账 —— 它们
    确实是两份磁盘占用，合并显示反而会让人误判能省多少。
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
# 二、回收计划（纯函数，不碰盘）
# ======================================================================
def plan_prune(
    models: list[CachedModel],
    *,
    in_use: dict[str, list[str]] | None = None,
    protected: set[str] | None = None,
    keep_onnx: bool = True,
) -> PrunePlan:
    """算出能安全回收什么。

    Args:
        models: :func:`scan_models` 的结果。
        in_use: ``model_id → 库名列表``，来自各库 ``library.json``。
        protected: 即便没人引用也要留着的模型（当前默认 / 显式 env 指定）。
        keep_onnx: True = 保留 ``onnx/``（``auto``/``onnx`` 后端要用它）。
            False（仅当显式 ``torch`` 后端）才把 ``onnx/`` 也算冗余。

    Returns:
        :class:`PrunePlan`。**只描述，不执行。**
    """
    in_use = in_use or {}
    protected = protected or set()
    plan = PrunePlan()

    for m in models:
        refs = in_use.get(m.model_id) or []
        is_protected = m.model_id in protected

        # int8 产物运行时不读，无论该模型是否在用都可回收
        if m.int8_bytes:
            int8_dirs = _subdirs(m.path, "onnx_int8")
            plan.actions.append(PruneAction(
                kind=KIND_ONNX_INT8,
                model_id=m.model_id,
                path=int8_dirs[0] if int8_dirs else os.path.join(m.path, "onnx_int8"),
                bytes_=m.int8_bytes,
                reason="onnx_int8/ is not read at runtime (ONNX_SUBDIR only accepts onnx; the quantized artifact is an export intermediate)",
            ))

        # 注意：这里用标志位而不是在分支里 continue —— 被引用的模型仍然
        # 可能该删 onnx/（显式 torch 后端时），提前 continue 会把那条路
        # 整个跳过，表现为「明明设了 torch 却一分空间都不回收」。
        keep_model = bool(refs) or is_protected
        if refs:
            plan.notes.append(
                f"{m.model_id}: referenced by libraries {', '.join(sorted(refs))}; kept whole")
        elif is_protected:
            plan.notes.append(
                f"{m.model_id}: current default / explicitly specified model; kept whole")

        if keep_model:
            # 保留的模型才谈得上「onnx 是否冗余」
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

        # 既没人引用、也不是默认 ⇒ 整目录可删
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
    """找 ``snapshots/*/<name>`` 下的实体目录（可能多版本）。"""
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
# 三、执行（唯一会改盘的地方）
# ======================================================================
def _is_safe_target(path: str, cache_roots: list[str]) -> bool:
    """删盘前的护栏：必须落在某个缓存根之内，且最外层的模型目录名合规。

    这道检查的意义是把「手滑传了个空串 / 传了个上级目录」变成显式错误，
    而不是一次 ``rmtree`` 把整个 ``~/.cache`` 清掉。
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
    """执行回收，返回 ``(释放字节数, 错误列表)``。

    单个动作失败**不中断**其余动作：一次删不干净比全都不删更常见
    （Windows 上文件被占用就会 PermissionError），逐条报告才好处理。
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
# 四、展示
# ======================================================================
def human_size(n: int) -> str:
    """人类可读体积（1 KB = 1024 B，与 ``du`` 口径接近）。

    字节不带小数，KB 起统一一位小数 —— 混着来（``2 KB`` 旁边跟着
    ``2.0 MB``）看着像两套口径。
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
    """渲染一张人看的清单（宽度控制在 78 列内）。"""
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
    """渲染回收计划（``--yes`` 之前的最后一道确认）。"""
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
