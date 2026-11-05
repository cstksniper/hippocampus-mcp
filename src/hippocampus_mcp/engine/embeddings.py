# -*- coding: utf-8 -*-
"""文本嵌入工具.

提供两种嵌入器:
    - SentenceTransformerEmbedder: 基于 sentence-transformers 的语义嵌入器 (推荐).
      模型名留空时按 ``HIPPOCAMPUS_EMB_MODEL`` → ``all-MiniLM-L6-v2`` 依次回落，
      回落会告警；中文语料请显式指定 ``BAAI/bge-base-zh-v1.5``。
    - HashEmbedder: 基于哈希投影的确定性嵌入器 (MVP/后备).
      无外部依赖, 但无法捕获语义相似性.

两者实现统一接口:
    embed(text)        -> 文档侧嵌入，写入记忆时用
    embed_query(text)  -> 查询侧嵌入，召回时用（BGE 等模型此处加指令前缀）
    embed_batch(texts) -> 批量嵌入
    model_key          -> 多模型向量库的槽位键

该接口由 ``create_embedder()`` 强制校验，新增后端缺任何一个方法都会在
初始化时直接报错，而不是等到召回时抛 AttributeError。

模型缓存目录解析（覆盖报告·阻断 2 的修复）：
    本地模型目录不再只认 ``<项目>/libs/models``，而是按候选列表依次探测，
    可用 ``HIPPOCAMPUS_EMB_CACHE`` 环境变量显式覆盖。候选顺序：

        1. ``HIPPOCAMPUS_EMB_CACHE``（若设置，则只认它）
        2. ``<项目>/libs/models``
        3. ``<项目上级>/libs/models``   ← 本机 BGE 实际所在处
        4. ``<项目上级>/libs``
        5. ``~/.cache/huggingface/hub``（HF 标准缓存，含 cache_folder 结构）

    离线（``HF_HUB_OFFLINE=1``）且全部候选都找不到模型时，**立刻抛出**一份
    列出已尝试路径的错误，而不是丢给 SentenceTransformer 去联网等超时。
"""

from __future__ import annotations

import functools
import hashlib
import locale
import logging
import os
import sys
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# H2：嵌入后端选择（torch / onnx / auto）。这里只导入一个读环境变量的函数，
# 真正的重依赖（onnxruntime / tokenizers）在 _attach_onnx 里延迟导入 ——
# 没开 ONNX 的进程不该付任何代价。
from .onnx_embedder import (  # noqa: E402
    backend_kind,
    get_shared_onnx,
    onnx_pool_stats,
    shared_embedder_enabled,
)
from ..config import load_config  # noqa: E402  （env 读收编 config）

# 模型生命周期（延迟加载 / 空闲卸载）。放在**独立模块**里是因为它要被
# torch 侧与 onnx 侧共用，塞进任一方都会绕成环形导入。
from . import model_lifecycle as _ML


def _with_model_use(fn):
    """给会用到模型的方法包一层「租约」。

    租约存在的意义：daemon 是一进程 N 线程共享一个 brain，卸载时若某条请求
    正跑到一半，它会拿到半截模型。登记之后 :func:`_ML.release_models` 在租约
    非 0 时会**拒绝执行**并回报持有者，而不是把内存撂给别人收拾。
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with _ML.model_use():
            return fn(self, *args, **kwargs)

    return wrapper


# 模型加载的**进程级串行锁**（背景见 SentenceTransformerEmbedder._ensure_model）。
# 用 RLock 而非 Lock：加载若因异常栈重入不会自锁死。
_MODEL_LOAD_LOCK = threading.RLock()

# 推理（encode）的**进程级串行锁**。
#
# 为什么要它：``SentenceTransformer.encode`` 每次调用都会走
# ``tokenization_utils_tokenizers.set_truncation_and_padding`` →
# ``self._tokenizer.enable_truncation(...)``，这是对 Rust 侧 tokenizer 的
# **可变借用**。两个线程同时 encode 就会撞上 Rust 的 RefCell 双重借用，抛
#
#     RuntimeError: Already borrowed
#
# 直连 stdio 时一进程一线程，永远不会撞；daemon 化之后一进程 N 个会话
# = N 个线程共享同一个 embedder，撞车就成了常态。实测（
# 两连接并发各写 15 条）：随机若干次 ``store_memory`` 抛这个错，工具返回
# isError，记录静默丢失（30 条里只剩 20~29 条，数量随时序变化）。
#
# 代价：并发 encode 被串行。daemon 的目标是省内存而不是提吞吐，
# 正确性优先；单条 encode 是几十毫秒级，串行不会成为瓶颈。
_ENCODE_LOCK = threading.RLock()


# ==================================================================
# F6 · 进程级共享池（torch 侧：SentenceTransformer 模型对象）
# ==================================================================
# 池化的是**模型对象**（占 ~400MB 的那一坨），不是嵌入器实例——理由见
# ``onnx_embedder`` 模块末尾的 F6 段：嵌入器实例身上带 ``model_source``
# 这种"决策元数据"，共享出去会让第二个库继承第一个库的语义，可能把
# 「跟随库内模型」判成「显式指定」而直接报错让库起不来。
#
# 值 = ``(model, dim)``：dim 一起存是因为命中池时要跳过 test encode
# （那份 encode 只是为了探维度，重复跑既慢又会撞 tokenizer 可变借用）。
_MODEL_POOL: "dict[tuple[str, str], tuple[object, int]]" = {}
_MODEL_POOL_LOCK = threading.Lock()

#: 池上限（同 onnx 侧的 POOL_MAX）
POOL_MAX = 4


def reset_model_pool() -> int:
    """清空 torch 侧模型池（**只给测试用**），返回清掉的份数。"""
    with _MODEL_POOL_LOCK:
        n = len(_MODEL_POOL)
        _MODEL_POOL.clear()
    return n


def embedder_pool_stats() -> dict:
    """自曝进程级共享池现状（``brain_stats`` / 测试用）。

    ``torch_size`` 与 ``onnx_size`` 分开报：只跑 ONNX 的生产进程里
    ``torch_size`` 恒为 0，看到非 0 就说明这条进程其实还在走 torch
    回退——这正是"我以为在省内存"那类静默失效的自曝口。
    """
    with _MODEL_POOL_LOCK:
        torch_size = len(_MODEL_POOL)
        keys = [f"{k[1]}@{k[0]}" for k in _MODEL_POOL]
    onnx = onnx_pool_stats()
    return {
        "enabled": shared_embedder_enabled(),
        "torch_size": torch_size,
        "torch_keys": keys,
        "onnx_size": onnx["size"],
        "onnx_keys": onnx["keys"],
        "max": POOL_MAX,
    }
# 环境变量名：此前这个字符串散落在 server.py / brain.py / 各脚本里，
# 每处各写一遍、各默认一遍，于是"某个入口忘了设"就会静默落到 MiniLM。
ENV_MODEL = "HIPPOCAMPUS_EMB_MODEL"

# 允许"显式指定模型与库内向量不符"时强行切换（离线对比不同模型的实验用）。
# 不设时：不符就**报错并说清怎么办**，绝不静默换模型重算整库。
ENV_ALLOW_MODEL_SWITCH = "HIPPOCAMPUS_ALLOW_MODEL_SWITCH"

# 应用层默认模型（未显式指定、且库内也没有可跟随的模型时才会用到）。
# 之所以是 MiniLM：它是英文语义检索的通用基线，且 LoCoMo（英文）评测与
# Mem0 对比评测都固定用它（见 scripts/eval_mem0_*.py、bench_bge.py）。
# **中文语料必须显式指定 BGE**——这个默认值不是给中文生产库用的。
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

#: 中文语料的默认模型（BGE-base-zh-v1.5，768 维）。
#: 语义是**分流**不是替换：英文/其它语言仍走 :data:`DEFAULT_MODEL_NAME`。
#: 只在「系统语言是中文」且用户**没有**显式配置模型时作为应用默认值生效。
ZH_DEFAULT_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

#: 关掉「按系统语言自动选默认模型」的开关（默认开）。
#: 设 ``0/false/no/off`` 关闭 → 恒用 MiniLM。
#: 显式 ``HIPPOCAMPUS_EMB_MODEL`` 永远优先于本开关，不受影响。
ENV_MODEL_AUTO = "HIPPOCAMPUS_EMB_MODEL_AUTO"


def _default_libs_path() -> str:
    """从本文件位置推断项目 ``libs/``（与构造函数内的推断口径一致）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return os.path.join(project_root, "libs")


def _cache_roots(libs_path: str = "", cache_folder: str = "") -> list[str]:
    """按优先级返回模型缓存根目录候选（``_resolve_cache_candidates`` 的真相源）。

    抽到模块级是为了让「按语言选默认模型」能在**不构造嵌入器**的前提下
    判断中文模型在不在本地（构造嵌入器会牵进 torch / ONNX，代价太高）。
    """
    explicit = cache_folder or load_config().emb_cache or ""
    if explicit:
        return [os.path.abspath(explicit)]

    project_libs = libs_path or ""
    parent_libs = (
        os.path.join(os.path.dirname(os.path.dirname(project_libs)), "libs")
        if project_libs
        else ""
    )
    candidates = [
        os.path.join(project_libs, "models") if project_libs else "",
        os.path.join(parent_libs, "models") if parent_libs else "",
        parent_libs,
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ]
    out: list[str] = []
    for c in candidates:
        if c and os.path.abspath(c) not in out:
            out.append(os.path.abspath(c))
    return out


def local_model_dir(model_name: str, roots: list[str]) -> str | None:
    """在 ``roots`` 里找 ``model_name`` 的本地副本，找不到返回 ``None``。

    ``SentenceTransformerEmbedder._find_local_model`` 的真相源（后者已改为
    转调本函数，行为逐位不变）。三种命中形态见该方法的 docstring。
    """
    short = model_name.split("/")[-1]
    cls = SentenceTransformerEmbedder
    for root in roots:
        cand = os.path.join(root, short)
        if os.path.isdir(cand) and cls._looks_like_model(cand):
            return cand
        if os.path.basename(root) == short and cls._looks_like_model(root):
            return root
        hit = cls._find_hf_snapshot(root, short)
        if hit:
            return hit
    return None


def _win_ui_language_id() -> int | None:
    """Windows 上取「系统 UI 语言」LANGID；非 Windows / 取不到返回 ``None``。

    🔴 **不能用 ``LANG`` 判断**：Git Bash 会把 ``LANG`` 设成 ``en_US.UTF-8``，
    而系统其实是中文 —— 实测本机就是这样。以 env 为准会把中文系统判成英文。
    """
    if os.name != "nt":  # pragma: no cover - 非 Windows 走 LANG/locale
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
    except Exception:  # noqa: BLE001 - 检测失败不该连累主流程
        return None


def _is_zh_language_id(langid: int) -> bool:
    """LANGID 主语言是否为中文（``LANG_CHINESE == 0x04``，含繁简）。"""
    return (langid & 0x03FF) == 0x04


def _is_zh_locale(tag: str) -> bool:
    """``zh_CN.UTF-8`` / ``Chinese (Simplified)_China`` / ``zh-TW`` 都算中文。"""
    t = (tag or "").strip().lower()
    return t.startswith("zh") or t.startswith("chinese")


def detect_system_language() -> str:
    """检测系统语言，中文返回 ``"zh"``，其余返回 ``""``。

    探测顺序（先系统级、再进程环境，避免 shell 环境把系统语言盖掉）：

        1. Windows ``GetUserDefaultUILanguage()`` 的 LANGID 主语言；
        2. POSIX 的 ``LC_ALL`` / ``LC_MESSAGES`` / ``LANG``；
        3. ``locale.getlocale()[0]`` 兜底（Windows 上也能给出
           ``"Chinese (Simplified)_China"``）。

    任一步失败都**继续往下试**，全部失败按非中文处理（回落英文基线，
    这是本函数唯一安全的默认方向）。
    """
    langid = _win_ui_language_id()
    if langid is not None:
        return "zh" if _is_zh_language_id(langid) else ""

    for env_key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        val = os.environ.get(env_key, "").strip()
        if val and val.upper() not in ("C", "POSIX"):
            return "zh" if _is_zh_locale(val) else ""

    try:
        tag = (locale.getlocale() or ("", ""))[0] or ""
    except Exception:  # noqa: BLE001
        tag = ""
    return "zh" if _is_zh_locale(tag) else ""


def auto_model_selection_enabled() -> bool:
    """「按系统语言自动选默认模型」开关（默认开；``HIPPOCAMPUS_EMB_MODEL_AUTO=0`` 关闭）。

    与 ``onnx_embedder.shared_embedder_enabled`` 同款口径：黑名单语义，
    只有 ``0/false/off/no`` 才关，未设/空串/怪异值都算开。
    """
    raw = (load_config().emb_model_auto or "").strip().lower()
    return raw not in ("0", "false", "off", "no")


def default_model_for_system() -> tuple[str, str]:
    """按系统语言给出「应用默认模型」，返回 ``(model_name, reason)``。

    **只影响「什么都没配」这条路径**：显式参数 > ``HIPPOCAMPUS_EMB_MODEL``
    > 本函数。返回的 ``source`` 仍是 ``"default"``（不是 ``"env"``），这一点
    **不能改** —— ``MemoryManager._sync_embedder_with_store`` 靠 ``source ==
    "default"`` 判定「可以跟随库内已有模型」，改成 ``"env"`` 会让所有存量
    中文库在升级后直接抛 :class:`EmbedderModelMismatchError`。

    ``reason`` 取值：

        * ``auto_off``          —— 开关关闭，恒为英文基线；
        * ``not_zh``            —— 系统非中文，走英文基线；
        * ``zh_local``          —— 中文且中文模型本地已有（直接用）；
        * ``zh_download``       —— 中文且需首次下载（在线，能用但慢）；
        * ``zh_unavailable``    —— 中文但离线且本地没有 → **回落英文基线**
          （宁可检索差一点，也不能让服务起不来）。
    """
    if not auto_model_selection_enabled():
        return DEFAULT_MODEL_NAME, "auto_off"
    if detect_system_language() != "zh":
        return DEFAULT_MODEL_NAME, "not_zh"

    roots = _cache_roots(_default_libs_path())
    if local_model_dir(ZH_DEFAULT_MODEL_NAME, roots) is not None:
        return ZH_DEFAULT_MODEL_NAME, "zh_local"

    offline = os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False")
    if offline:
        return DEFAULT_MODEL_NAME, "zh_unavailable"
    return ZH_DEFAULT_MODEL_NAME, "zh_download"


def _emit_warning(msg: str) -> None:
    """告警同时走 logging 与 stderr。

    MCP 服务是 stdio 协议，日志走 stderr 才不会被当成协议帧；而脚本场景
    常常只配了 logging。两边都发，保证"这句警告一定有人看见"。
    """
    logging.warning(msg)
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:  # pragma: no cover - stderr 被关掉时不该连累主流程
        pass


def _warn_default_model(model_name: str) -> None:
    """回落到应用默认模型时说清「为什么是这个模型」（防"静默用错模型"复发）。

    reason 由 :func:`default_model_for_system` 现取 —— 警告文案与选型决策
    共用同一处判断，避免"文案说英文基线、实际跑的是中文模型"这种自相矛盾。
    """
    _, reason = default_model_for_system()
    if reason == "zh_unavailable":
        _emit_warning(
            f"[embeddings] 系统语言为中文，但中文模型 {ZH_DEFAULT_MODEL_NAME} 在本地"
            f"不存在且当前处于离线模式，已回落英文基线 {model_name}（384D）——"
            f"中文语义检索会明显偏弱。修复：把 {ZH_DEFAULT_MODEL_NAME} 放到模型缓存"
            f"目录（或设 {ENV_CACHE}），或去掉 HF_HUB_OFFLINE 让它联网下载。"
        )
        return
    if reason == "zh_download":
        _emit_warning(
            f"[embeddings] 系统语言为中文，默认使用 {ZH_DEFAULT_MODEL_NAME}（768D）。"
            f"首次运行需联网下载约 400MB；离线环境会失败，建议提前预置，"
            f"或设 HIPPOCAMPUS_EMB_MODEL={DEFAULT_MODEL_NAME} 显式使用英文基线。"
        )
        return
    if reason == "zh_local":
        logging.info(
            "[embeddings] 系统语言为中文，默认使用 %s（768D，本地已有副本）",
            model_name,
        )
        return
    _emit_warning(
        f"[embeddings] 未显式指定嵌入模型，回落默认 {model_name}"
        f"（英文通用基线，384D）。中文语料请设 {ENV_MODEL}=BAAI/bge-base-zh-v1.5，"
        f"或构造时显式传 model_name。"
    )


def resolve_embedder_model(explicit: str | None = None) -> tuple[str, str]:
    """解析生效嵌入模型名，返回 ``(model_name, source)``。

    ``source`` 取值：

        - ``"explicit"`` — 调用方显式传入（参数或 ``--model``）；
        - ``"env"``      — 来自 :data:`ENV_MODEL` 环境变量；
        - ``"default"``  — 都没给，落到 :func:`default_model_for_system`：
          **系统语言为中文且中文模型可用 → BGE-zh，其余 → MiniLM**。
          不管选中哪个，来源都记 ``"default"``（供"跟随库内模型"判定）。

    **这是解析模型的唯一真相源。** 以前每个入口各写一遍
    ``os.environ.get("HIPPOCAMPUS_EMB_MODEL", "")`` 再各自默认，
    漏一处就静默跑成 MiniLM；集中到这里之后，调用方只需要把 source
    记进日志，"我现在到底在用哪个模型"就再也不需要猜。
    """
    name = (explicit or "").strip()
    if name:
        return name, "explicit"
    name = (load_config().emb_model or "").strip()
    if name:
        return name, "env"
    # 都没有 → 按**系统语言**选应用默认（中文 → BGE-zh，其余 → MiniLM）。
    # 🔴 source 必须仍标 "default"，不能标 "env"：
    # MemoryManager._sync_embedder_with_store 靠 `source == "default"` 判定
    # 「可以跟随库内已有模型」。标成 "env" 会让所有存量中文库（MiniLM 384D）
    # 在升级后直接抛 EmbedderModelMismatchError —— 也就是说，这个自动选型
    # 对老库必须"透明"，只对**新建库**生效。
    default_name, _reason = default_model_for_system()
    return default_name, "default"


# ==================================================================
# HashEmbedder — 无依赖的后备嵌入器
# ==================================================================

class HashEmbedder:
    """基于哈希投影的确定性文本嵌入器.

    无外部依赖, 保证相同文本始终产生相同嵌入.
    但无法捕获语义相似性, 仅适用于快速测试.

    Args:
        dim: 输出嵌入维度, 默认 32.
        salt: 哈希盐值, 用于区分不同嵌入空间.
    """

    def __init__(self, dim: int = 32, salt: str = "hippocampus_mcp") -> None:
        self.dim = dim
        self.salt = salt
        # 模型标识：作为多模型向量库中的槽位键
        self.model_key = f"hash:{salt}:{dim}"

    def embed(self, text: str) -> np.ndarray:
        """将文本嵌入为固定维度的归一化向量.

        Args:
            text: 待嵌入的文本.

        Returns:
            L2 归一化的 numpy 向量, shape = (dim,).
        """
        if not text:
            return np.zeros(self.dim, dtype=np.float64)

        vec = np.zeros(self.dim, dtype=np.float64)

        words = text.strip().lower().split()
        if not words:
            words = [text.strip().lower()]

        for n in (1, 2, 3):
            for i in range(max(len(words) - n + 1, 1)):
                if n == 1 and len(words) > 1:
                    gram = words[i]
                elif len(words) >= n:
                    gram = " ".join(words[i : i + n])
                else:
                    gram = " ".join(words)

                h = hashlib.sha256(
                    f"{self.salt}:{gram}".encode("utf-8")
                ).digest()

                for j in range(self.dim):
                    byte_idx = j % len(h)
                    val = (h[byte_idx] / 127.5) - 1.0
                    vec[j] += val

        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec = np.ones(self.dim, dtype=np.float64)
            vec = vec / np.linalg.norm(vec)

        return vec

    def embed_query(self, text: str) -> np.ndarray:
        """嵌入查询文本.

        哈希嵌入器无指令前缀概念, 与 :meth:`embed` 完全等价。此方法的存在
        是为了满足嵌入器协议——``MemoryManager.recall()`` 一律走
        ``embed_query``，缺了它回退后端会在召回时直接 ``AttributeError``
        （覆盖报告 D5，已修）。

        Args:
            text: 查询文本.

        Returns:
            L2 归一化的 numpy 向量, shape = (dim,).
        """
        return self.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量嵌入文本.

        Args:
            texts: 待嵌入的文本列表.

        Returns:
            (N, dim) 的 numpy 矩阵.
        """
        return np.vstack([self.embed(t) for t in texts])


    @property
    def max_seq_length(self) -> int:
        """hash 后端按字符 n-gram 哈希，不看 token，因此没有截断上限（0 = 无上限）。"""
        return 0

    def embedding_coverage(self, text: str) -> dict:
        """hash 后端对整段文本都可见，恒不截断。"""
        total = len(text or "")
        return {
            "truncated": False,
            "total_chars": total,
            "embedded_chars": total,
            "tokens": total,
            "max_seq_length": 0,
            "dropped_ratio": 0.0,
        }

# ==================================================================
# SentenceTransformerEmbedder — 语义嵌入器 (推荐)
# ==================================================================

class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的语义文本嵌入器.

    默认 all-MiniLM-L6-v2 (384 维, 英文)。中文场景推荐 BGE 系列:
        - BAAI/bge-small-zh-v1.5 (512 维, 24M, CPU 快)
        - BAAI/bge-base-zh-v1.5 (768 维, 102M, CPU 可跑, 中文检索 ndcg@10 ~70)

    BGE 系列遵循官方用法: 查询侧加指令前缀, 文档侧不加。
    embed() = 文档侧 (默认), embed_query() = 查询侧 (自动加前缀)。

    模型会在首次调用 embed() 时懒加载 (避免启动延迟).
    首次加载需要下载模型权重, 后续从本地缓存加载.

    Args:
        model_name: 模型名称。**推荐显式传**；留空时依次取
            ``HIPPOCAMPUS_EMB_MODEL`` 环境变量、再回落 ``all-MiniLM-L6-v2``
            （回落时会打警告，见 :func:`resolve_embedder_model`）。
            中文场景建议 "BAAI/bge-base-zh-v1.5" 或 "BAAI/bge-small-zh-v1.5"。
        cache_folder: 模型缓存目录. 默认为项目 libs/ 下的 models 目录.
        libs_path: 项目 libs/ 目录路径, 用于添加到 sys.path 以加载依赖.
    """

    # 默认模型和维度。注意：这里是**应用兜底默认值**，不是"中文生产库的推荐值"。
    # 中文语料请显式指定 BAAI/bge-*-zh-v1.5（见 resolve_embedder_model）。
    DEFAULT_MODEL = DEFAULT_MODEL_NAME
    DEFAULT_DIM = 384

    # BGE zh 系列官方查询指令前缀 (只加在查询侧)
    BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    # 已知模型的维度 (用于初始化前预估, 实际维度以模型输出为准)
    KNOWN_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "BAAI/bge-small-zh-v1.5": 512,
        "BAAI/bge-base-zh-v1.5": 768,
        "BAAI/bge-large-zh-v1.5": 1024,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    }

    # 环境变量：显式指定模型缓存根目录
    ENV_CACHE = "HIPPOCAMPUS_EMB_CACHE"

    # 模型目录的必备文件（用于判定"这个目录真的是一份模型"）
    _MODEL_MARKERS = ("config.json", "modules.json")

    def __init__(
        self,
        model_name: str = "",
        cache_folder: Optional[str] = None,
        libs_path: Optional[str] = None,
    ) -> None:
        # 未显式指定 → 依次看 env → 默认值，并把**来源**记下来。
        # `model_source == "default"` 时会大声告警：这条路径就是历史上
        # "服务静默回落 MiniLM"的入口，必须让它可见。
        resolved, source = resolve_embedder_model(model_name)
        self.model_name = resolved
        self.model_source = source
        self.model_explicit = source != "default"
        if source == "default":
            _warn_default_model(resolved)
        model_name = resolved
        # BGE 系列写全名 (BAAI/xxx), 短名兼容
        if model_name in ("bge-base-zh-v1.5", "bge-base-zh"):
            model_name = "BAAI/bge-base-zh-v1.5"
        elif model_name in ("bge-small-zh-v1.5", "bge-small-zh"):
            model_name = "BAAI/bge-small-zh-v1.5"
        elif model_name in ("bge-large-zh-v1.5", "bge-large-zh"):
            model_name = "BAAI/bge-large-zh-v1.5"
        self.model_name = model_name
        self.dim = self.KNOWN_DIMS.get(model_name, self.DEFAULT_DIM)
        self._model = None
        #: 模型**完全就绪**（含维度探测的 test encode 也跑完）才算 True。
        #: 只有 ``_model`` 非空就放行会让别的线程与未结束的 test encode 并发。
        self._model_ready = False
        self._cache_folder = cache_folder
        self._libs_path = libs_path
        # 实际加载用到的模型目录（诊断用：加载后才知道，失败时是 None）
        self.resolved_model_path: str | None = None

        # 模型标识：作为多模型向量库中的槽位键（换模型即换槽位，互不覆盖）
        self.model_key = self.model_name

        # BGE 系列在查询侧需要加指令前缀
        self.is_bge = model_name.startswith("BAAI/bge")

        # 推断默认 libs 路径 (项目根目录/libs)
        if self._libs_path is None:
            # 从当前文件位置推断项目根目录
            # embeddings.py 位于 src/hippocampus_mcp/embeddings.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # src/hippocampus_mcp -> src -> project_root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self._libs_path = os.path.join(project_root, "libs")

        # 解析缓存候选目录（缺陷·阻断 2）
        self._cache_candidates = self._resolve_cache_candidates(cache_folder)
        # 兼容旧属性：默认取第一个候选（存在的优先），供 os.makedirs / 日志用
        self._cache_folder = next(
            (d for d in self._cache_candidates if os.path.isdir(d)),
            self._cache_candidates[0],
        )

        # H2：ONNX 后端（组合）。挂载成功则本实例退化成薄壳，编码动作全部
        # 转发给 _onnx，torch / sentence-transformers **一次都不会被 import**
        # ——这正是省 ~280MB 的机制所在。
        self._onnx: object | None = None
        #: F6：本实例的重对象是否来自进程级共享池（同模型的别的库也在用）
        self.shared_from_pool = False
        self._attach_onnx()
        # 登记到进程级生命周期。卸载要知道"该去通知谁"，而这里不能用
        # 强引用（否则嵌入器永远不会被回收，卸载也就没了意义）。
        _ML.register(self)

    def release_model(self) -> bool:
        """卸载模型本体，**保留本实例与它的后端决策**。

        刻意**不**把 ``_onnx`` 置成 None：``_ensure_model`` 的快路径靠
        ``self._onnx is not None`` 走 ONNX 分支；置空会让下一次加载退回
        torch 路径（慢且多吃 ~280MB），等于自伤。让 OnnxEmbedder 自己卸掉
        session/tokenizer、把 ``_model_ready`` 置 False，重载时自然回到原后端。

        Returns:
            是否做了「已加载 → 未加载」的状态复位（本来就没加载返回 False）。
        """
        # 🔴 不能用子对象的返回值判断自己要不要复位：共享池里那份可能已经被
        # 别的嵌入器先卸过一轮（此时它返回 False），于是本实例明明已经没有
        # 模型，``_model_ready`` 却还停在 True。torch 路径下这就是真事故——
        # ``embed()`` 的快路径会因为 ``_model is not None and _model_ready``
        # 不成立而走到 ``self._model.encode(...)``，而 _model 已经是 None。
        was_loaded = bool(
            self._model_ready or self._model is not None
            or self._onnx is not None)
        released_any = False
        if self._onnx is not None:
            try:
                released_any = bool(self._onnx.release_model()) or released_any
            except Exception as exc:  # noqa: BLE001
                logger.warning("释放 ONNX 会话失败: %s: %s",
                               type(exc).__name__, exc)
        if self._model is not None:
            self._model = None
            released_any = True
        if was_loaded:
            self._model_ready = False
            self.shared_from_pool = False
            # ⚠️ 别清 resolved_model_path：ONNX 路径下重载走的是
            # ``_onnx._ensure_model``，不会再进 ``_load_model``，清了就永久
            # 丢掉"模型实际用的是哪个目录"这条自曝信息。
        return bool(was_loaded or released_any)

    def _pool_key(self, load_path: str) -> tuple[str, str]:
        """torch 侧池键 = （后端 kind, 绝对路径/模型名）。

        带 kind 是因为 ``auto`` 回退 torch 与显式 ``torch`` 拿到的是同一份
        权重，本可共用；但同一进程内 kind 不变，带上它只为让自曝更可读。
        """
        p = os.path.abspath(load_path) if os.path.isdir(load_path) else load_path
        return (backend_kind(), p)

    # ------------------------------------------------------------------
    # 缓存目录解析
    # ------------------------------------------------------------------
    def _resolve_cache_candidates(self, cache_folder: Optional[str]) -> list[str]:
        """返回按优先级排列的缓存根目录候选列表。

        显式参数或 ``HIPPOCAMPUS_EMB_CACHE`` 存在时**只认它**（尊重显式配置，
        避免"我明明指了路却还是去别处找"的困惑）；否则用默认候选链。

        实现已移到模块级 :func:`_cache_roots`（供"按语言选默认模型"
        在不构造嵌入器的前提下探测本地模型），本方法保留为薄壳。
        """
        return _cache_roots(self._libs_path or "", cache_folder or "")

    def _find_local_model(self) -> str | None:
        """在候选目录里找本地模型目录，找不到返回 ``None``。

        三种命中形态：
            - ``<候选根>/<短名>``（如 ``libs/models/bge-base-zh-v1.5``）——
              本机 BGE 就是这个形态；
            - ``<候选根>`` 本身就是模型目录（basename == 短名，且有 config.json）；
            - **HF hub 布局** ``<候选根>/models--<org>--<名>/snapshots/<hash>/``
              —— sentence-transformers 从 HuggingFace 下载后落盘就是这个形态。
              少了这一条，本机只有 hub 布局的模型（如 all-MiniLM-L6-v2）在离线时
              会报 FileNotFoundError，而**实际缓存是齐的**，只是没被认出来。
        """
        return local_model_dir(self.model_name, self._cache_candidates)

    @classmethod
    def _find_hf_snapshot(cls, root: str, short: str) -> str | None:
        """按 HF hub 布局在 ``root`` 下找模型，返回 snapshot 目录或 ``None``。

        目标形态：``<root>/models--<org>--<short>/snapshots/<hash>/``。

        从 ``model_name`` 反推 org 不可靠（``"all-MiniLM-L6-v2"`` 根本没带 org，
        实际目录却叫 ``models--sentence-transformers--all-MiniLM-L6-v2``），
        所以这里**按目录名后缀模糊匹配**，不构造 org 前缀。

        目录名大小写在 Windows 上不敏感、在 Linux 上敏感，故两端都 lower 比较。
        多个 snapshot 时取排序最大的一个（并用 ``_looks_like_model`` 滤掉空壳）。
        """
        if not os.path.isdir(root):
            return None
        suffix = "--" + short.lower()
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("models--") or not entry.lower().endswith(suffix):
                continue
            snaps = os.path.join(root, entry, "snapshots")
            if not os.path.isdir(snaps):
                continue
            for h in sorted(os.listdir(snaps), reverse=True):
                cand = os.path.join(snaps, h)
                if os.path.isdir(cand) and cls._looks_like_model(cand):
                    return cand
        return None

    @classmethod
    def _looks_like_model(cls, path: str) -> bool:
        """目录里是否有模型必需文件（缺了说明只是个空壳/半截下载）。"""
        return os.path.isfile(os.path.join(path, "config.json")) or os.path.isfile(
            os.path.join(path, "modules.json")
        )

    def describe_source(self) -> str:
        """一句话说明"这个嵌入器最终会从哪加载、加载了没"。"""
        where = self.resolved_model_path or "（尚未加载）"
        return f"{self.model_name} @ {where}"

    def _attach_onnx(self, *, from_store: bool = False) -> bool:
        """尝试挂载 ONNX 后端，返回**是否挂上**。

        🔴 **调用时机是本方法的全部难点**（修过的一个坑）：
        构造时 ``self.model_name`` 可能只是**回落的默认值**，而真正生效的
        模型要到 ``manager._follow_store_model()``（加载库之后）才确定。
        所以本方法**必须能被重复调用**，且"模型名还没定"时**不许抛错**——
        否则显式 ``onnx`` 会在装配阶段就炸掉，而 ``auto`` 会静默回退 torch
        （表现为"我以为在用 ONNX，其实没有"）。

        三态语义：

        * ``torch`` → 直接返回 False（今天的行为，零变化）
        * ``auto``  → 有产物就用，没有就安静走 torch
        * ``onnx``  → 缺产物/缺依赖**抛错**，但下面两种情况例外（都不算
          用户显式指定模型，抛错只会误伤）：
          - ``model_source == "default"``（模型名是回落出来的）
          - ``from_store=True``（模型名是跟随库数据来的）
        """
        kind = backend_kind()
        if kind == "torch":
            return False
        # 模型名是否"用户显式指定"：只有显式时才允许严格抛错
        strict = kind == "onnx" and not from_store and self.model_source != "default"
        try:
            from .onnx_embedder import OnnxEmbedder, find_onnx_artifacts
        except ImportError as e:
            if strict:
                raise ImportError(
                    "HIPPOCAMPUS_EMB_BACKEND=onnx 但 onnxruntime/tokenizers 不可用：%s" % e
                ) from e
            logger.info(
                "嵌入后端：torch（ONNX 依赖不可用：%s）", e,
            )
            return False
        # ONNX 路只认本地目录（它本就不联网、不碰 torch）
        model_dir = self._find_local_model()
        if not model_dir and os.path.isdir(self.model_name):
            model_dir = self.model_name
        if not model_dir:
            if strict:
                raise FileNotFoundError(
                    "HIPPOCAMPUS_EMB_BACKEND=onnx 但找不到本地模型目录 %r"
                    % self.model_name
                )
            logger.info(
                "嵌入后端：torch（%s 无本地模型目录，待模型名确定后再试）",
                self.model_name,
            )
            return False
        if find_onnx_artifacts(model_dir) is None:
            if strict:
                raise FileNotFoundError(
                    "HIPPOCAMPUS_EMB_BACKEND=onnx 但 %s 下没有 ONNX 产物。"
                    "先跑：python scripts/export_onnx.py --model %s"
                    % (model_dir, model_dir)
                )
            logger.info(
                "嵌入后端：torch（%s 下没有 ONNX 产物%s）",
                model_dir,
                "，且模型名尚未最终确定" if not from_store else "",
            )
            return False
        # F6：命中共享池就不新建 session / 不重读 tokenizer.json。
        # ``OnnxEmbedder`` 只装 session + tokenizer，无随库变化的状态，
        # 可以安全共用；本实例自己的 ``model_source`` / ``model_explicit``
        # 不受影响（那是"模型名怎么来的"，必须每个库各算各的）。
        self._onnx = get_shared_onnx(
            model_dir=model_dir, model_name=self.model_name
        )
        self.shared_from_pool = bool(getattr(self._onnx, "shared_from_pool", False))
        self.dim = getattr(self._onnx, "dim", self.dim)
        # 池里的实例可能已被别的库加载过；路径自曝取实例自己的（未加载时
        # 是 None，与"本实例还没 encode"的事实一致）。
        self.resolved_model_path = getattr(self._onnx, "resolved_model_path", None)
        logger.info("嵌入后端：ONNX（%s）%s，运行期不加载 torch/ST",
                    model_dir, "（复用进程内共享实例）" if self.shared_from_pool else "")
        return True

    def _ensure_model(self) -> None:
        """懒加载模型. 首次调用时加载, 后续直接复用.

        **必须跨线程串行**（见 ``_MODEL_LOAD_LOCK``）：模型加载会顺带
        ``import sentence_transformers`` → ``scipy.special`` → 加载
        ``_ufuncs.pyd`` 这个 C 扩展。两个线程同时导入同一批模块，
        CPython 的 per-module 锁会让两边互相等（实测：预热线程与工具线程
        同时进 ``_ensure_model`` 后，两边都停在 importlib 的模块锁上不动）。
        预热已挪到后台线程，所以这种并发是常态而非边缘情况。
        """
        # 快路径：已加载则完全无锁开销。**必须看 _model_ready**：模型对象赋值
        # 之后还要跑一次 test encode 才算真正可用，只看 _model 会让并发线程
        # 提前放行（见 _load_model 末尾的注释）。
        if self._onnx is not None:
            # ONNX 路：模型就是 session + Rust tokenizer，加载极快，且没有
            # ST 那个"Rust 可变借用"并发问题。
            self._onnx._ensure_model()
            self.dim = self._onnx.dim
            self._model_ready = True
            return
        if self._model is not None and self._model_ready:
            return
        with _MODEL_LOAD_LOCK:
            if self._model is not None and self._model_ready:
                return               # 双重检查：等锁期间别人可能已加载完
            # 加载本身也算"正在使用"：否则刚好跨过空闲阈值的这次加载，
            # 会被另一边的卸载把 session 抽掉。
            _ML.touch()
            self._load_model()

    def _load_model(self) -> None:
        """真正构造模型（只在持锁状态下被 ``_ensure_model`` 调用）。"""
        # 使用 HuggingFace 国内镜像避免连接超时
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 将 libs/ 作为**兜底**追加到 sys.path（不是覆盖系统已装依赖）。
        # 🔴 2026-11-03 实测：这里原先是 `sys.path.insert(0, libs)`，libs 里
        # 残留的 `regex`（cp310 的 `_regex.cp310-win_amd64.pyd`）在 3.11 上
        # 加载失败，抢在 site-packages 之前把好包遮蔽，报出来的却是
        # "cannot import name '_regex' … circular import"，看上去像依赖没装。
        # 追加到末尾才是「项目内便携依赖」的正确语义：系统有的先用系统，
        # 没有才吃项目内——insert(0) 会让一个过期轮子废掉整条嵌入链路。
        if self._libs_path and self._libs_path not in sys.path:
            sys.path.append(self._libs_path)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                f"无法导入 sentence_transformers。请把依赖装到当前解释器"
                f"（pip install -e .）或项目内 libs/ 目录。\n"
                f"错误: {e}\n"
                f"libs_path: {self._libs_path}"
            ) from e

        # 本地目录优先：在候选链里找现成的模型目录
        load_path = self._find_local_model()
        offline = os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False")

        if load_path is None:
            if os.path.isdir(self.model_name):
                # model_name 本身就是一个目录路径
                load_path = self.model_name
            elif offline:
                # 离线且本地没有 → 明确报错，别让 SentenceTransformer 去联网等超时
                tried = "\n".join(f"    - {c}" for c in self._cache_candidates)
                raise FileNotFoundError(
                    f"离线模式（HF_HUB_OFFLINE=1）下找不到模型 {self.model_name!r} 的本地副本。\n"
                    f"已尝试的缓存根目录：\n{tried}\n"
                    f"  每个根目录下都找过 <根>/"
                    f"{self.model_name.split('/')[-1]}/\n"
                    f"修复方式（任选）：\n"
                    f"  1) 把模型放到上述任一目录下（推荐 <项目上级>/libs/models/…）；\n"
                    f"  2) 设置环境变量 {self.ENV_CACHE}=<模型所在目录>；\n"
                    f"  3) 去掉 HF_HUB_OFFLINE 让它可以联网下载。"
                )
            else:
                # 在线：交给 SentenceTransformer 按常规方式解析/下载
                load_path = self.model_name

        if os.path.dirname(load_path):
            os.makedirs(os.path.dirname(load_path), exist_ok=True)
        else:
            os.makedirs(self._cache_folder, exist_ok=True)

        # F6：命中共享池 ⇒ 直接用别人已加载好的模型对象，不 import
        # sentence_transformers、不读权重、不跑 test encode。多库同时开时
        # 省下的就是第二份权重（实测 ≈404MB）与第二次 ~30s 加载。
        pool_key = self._pool_key(load_path)
        if shared_embedder_enabled():
            with _MODEL_POOL_LOCK:
                hit = _MODEL_POOL.get(pool_key)
            if hit is not None:
                self._model, self.dim = hit
                self.resolved_model_path = load_path
                self._model_ready = True
                self.shared_from_pool = True
                logger.info(
                    "嵌入模型复用进程内共享实例：%s（%d 维），跳过加载",
                    load_path, self.dim,
                )
                return

        # 加载模型
        # 加载要几十秒，先 flush 一行 stderr 状态 —— 让"慢"与"坏"可区分
        # （stdout 是 JSON-RPC 专用通道，状态只能走 stderr，同 launcher._diag）。
        print(f"[hippocampus] 正在加载嵌入模型：{load_path}"
              f"（冷启动数十秒属正常；反复卡死看 daemon 日志）",
              file=sys.stderr, flush=True)
        self._model = SentenceTransformer(
            load_path,
            cache_folder=self._cache_folder,
        )
        self.resolved_model_path = load_path

        # H7：_prepare_torch 不再无条件导入 torch（numpy LNN 生产路径零 torch）。
        # 这条 ST 回退路是 torch 真正进场的另一扇门，单线程限制在这里补上
        # （Windows 下 OpenMP 多线程偶发 access violation，见 server._prepare_torch）。
        try:
            import torch as _torch

            _torch.set_num_threads(1)
        except Exception:  # noqa: BLE001 - 线程限制失败不该让模型加载失败
            pass

        # 更新实际维度 (以防模型与预期不同)。
        # 同样要过 _ENCODE_LOCK：这次 encode 与任何业务侧 encode 撞上都会抛
        # "Already borrowed"（Rust tokenizer 的可变借用）。
        with _ENCODE_LOCK:
            test_vec = self._model.encode("test", normalize_embeddings=True)
        self.dim = len(test_vec)
        print(f"[hippocampus] 嵌入模型就绪：{load_path}（{self.dim} 维）",
              file=sys.stderr, flush=True)
        # **最后**才置位：在此之前 ``self._model`` 已经非空，若没有这个标志，
        # 另一个线程会在 ``_ensure_model`` 的快路径上直接放行，与本线程尚未
        # 结束的 test encode 并发 —— 实测正是 daemon 首个工具调用就炸的原因。
        self._model_ready = True

        # F6：入池，供同进程内同模型的其它库复用（超过上限就不入池，
        # 不报错——只是这份不再被复用）。
        if shared_embedder_enabled():
            with _MODEL_POOL_LOCK:
                if len(_MODEL_POOL) < POOL_MAX:
                    _MODEL_POOL[pool_key] = (self._model, self.dim)
                    self.shared_from_pool = True

    @_with_model_use
    def embed(self, text: str) -> np.ndarray:
        """将文本嵌入为固定维度的归一化语义向量 (文档侧, 无指令前缀).

        Args:
            text: 待嵌入的文本.

        Returns:
            L2 归一化的 numpy 向量, shape = (dim,).
        """
        if self._onnx is not None:
            return self._onnx.embed(text)

        if not text:
            return np.zeros(self.dim, dtype=np.float64)

        self._ensure_model()

        # encode 返回 numpy 数组, normalize_embeddings=True 做 L2 归一化。
        # 加锁原因见 _ENCODE_LOCK：tokenizer 的 enable_truncation 是可变借用，
        # 并发调用会抛 "Already borrowed"，异常被工具层吃掉后表现为**静默丢记录**。
        with _ENCODE_LOCK:
            vec = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vec, dtype=np.float64)

    @_with_model_use
    def embed_query(self, text: str) -> np.ndarray:
        """嵌入查询文本 (查询侧).

        BGE zh 系列官方要求查询加指令前缀, 文档侧不加;
        其他模型的行为与 embed() 相同.

        Args:
            text: 查询文本.

        Returns:
            L2 归一化的 numpy 向量, shape = (dim,).
        """
        if self.is_bge:
            text = self.BGE_ZH_QUERY_PREFIX + text
        return self.embed(text)

    @_with_model_use
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量嵌入文本, 利用 GPU 批处理加速.

        Args:
            texts: 待嵌入的文本列表.

        Returns:
            (N, dim) 的 numpy 矩阵, 每行已 L2 归一化.
        """
        if self._onnx is not None:
            return self._onnx.embed_batch(texts)

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)

        self._ensure_model()

        with _ENCODE_LOCK:
            vecs = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
        return np.asarray(vecs, dtype=np.float64)

    @property
    def max_seq_length(self) -> int:
        """嵌入模型能吃下的最大 token 数（0 = 取不到，按"未知"处理）。

        BGE / MiniLM 一系 transformer 后端是 **512**，这是**模型硬上限**：
        超出的部分在 encode 时被静默丢掉，对向量**零贡献**。实测（中文库
        主卡 1374 token）``cos(embed(前600字), embed(全文)) = 1.000000``
        ——第 600 字之后完全不可检索。写入侧必须拿它做截断自曝，否则
        "写进去了但检索看不见"这件事永远没有人知道。
        """
        # torch 路径下这里会（间接触发）加载模型，也要记租约，否则可能与
        # 空闲卸载撞上；ONNX 路径只读元数据，包一层开销可忽略。
        _ML.touch()
        if self._onnx is not None:
            return self._onnx.max_seq_length
        try:
            self._ensure_model()
            limit = getattr(self._model, "max_seq_length", 0)
            if not limit:
                tok = getattr(self._model, "tokenizer", None)
                limit = getattr(tok, "model_max_length", 0) if tok is not None else 0
        except Exception:
            return 0
        try:
            return int(limit)
        except (TypeError, ValueError):
            return 0

    @_with_model_use
    def embedding_coverage(self, text: str) -> dict:
        """自曝：这段文本**真正进入向量**的可见比例（截断自曝）。

        超过 ``max_seq_length`` 的内容被静默截断——写入成功、回读在库、
        ``brain_stats`` 正常，**只有检索悄悄差一截**。这里把"到底纳入了多少"
        显式算出来给调用方看。

        Returns:
            ``{"truncated": bool, "total_chars": int, "embedded_chars": int,
             "tokens": int, "max_seq_length": int, "dropped_ratio": float}``。
            ``truncated`` 为 True 时，``embedded_chars`` 之后的原文**检索不到**。
        """
        if self._onnx is not None:
            return self._onnx.embedding_coverage(text)

        total = len(text or "")
        out = {
            "truncated": False,
            "total_chars": total,
            "embedded_chars": total,
            "tokens": 0,
            "max_seq_length": int(self.max_seq_length),
            "dropped_ratio": 0.0,
        }
        if total == 0 or out["max_seq_length"] <= 0:
            return out
        try:
            self._ensure_model()
            tok = getattr(self._model, "tokenizer", None)
            if tok is None:
                return out
            fast = bool(getattr(tok, "is_fast", False))
            # 同一把 _ENCODE_LOCK：``tokenizer.__call__`` 内部同样会走
            # ``set_truncation_and_padding`` → ``enable_truncation``（哪怕
            # truncation=False 也会），与并发的 encode 撞上就是
            # "Already borrowed"。自曝绝不能反过来把写入搞挂，更不能把
            # 别人的写入搞挂。
            with _ENCODE_LOCK:
                enc = tok(
                    text,
                    truncation=False,
                    add_special_tokens=True,
                    return_offsets_mapping=fast,
                )
            ids = enc.get("input_ids") or []
            n = len(ids)
            out["tokens"] = n
            limit = out["max_seq_length"]
            if n <= limit:
                return out
            out["truncated"] = True
            kept = 0
            offs = enc.get("offset_mapping") if fast else None
            if offs:
                idx = min(limit, len(offs)) - 1
                end = int(offs[idx][1])
                if 0 < end <= total:
                    kept = end
            if kept <= 0:
                # 拿不到 offset（非 fast 分词器 / 特例 token）→ 按比例估，
                # 宁可给一个近似值也不要回退成"没截断"。
                kept = int(total * limit / max(n, 1))
            out["embedded_chars"] = max(0, min(total, kept))
            out["dropped_ratio"] = round(1.0 - out["embedded_chars"] / total, 4)
        except Exception as exc:  # 自曝绝不能反过来把写入搞挂
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out



def embedding_coverage(embedder: object, text: str) -> dict:
    """任意嵌入器的截断自曝。

    后端没有实现 ``embedding_coverage`` 时**不静默跳过**，而是返回
    ``truncated=None`` 的"显式未知"字典——按 B2-1 的同一条判据：
    静默降级比显式报错危险，调用方必须能分辨"确定没截断"与"没查"。
    """
    fn = getattr(embedder, "embedding_coverage", None)
    if callable(fn):
        try:
            out = fn(text)
            if isinstance(out, dict):
                return out
        except Exception as exc:
            return {
                "truncated": None, "total_chars": len(text or ""),
                "embedded_chars": None, "tokens": None,
                "max_seq_length": None, "dropped_ratio": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "truncated": None, "total_chars": len(text or ""),
        "embedded_chars": None, "tokens": None,
        "max_seq_length": None, "dropped_ratio": None,
        "error": "嵌入器未提供 embedding_coverage，无法判定是否截断",
    }

# ==================================================================
# 工厂函数 — 根据配置选择嵌入器
# ==================================================================

def create_embedder(
    backend: str = "sentence_transformer",
    dim: int = 384,
    **kwargs,
) -> "HashEmbedder | SentenceTransformerEmbedder":
    """创建嵌入器实例.

    Args:
        backend: 嵌入后端类型.
            "sentence_transformer" — 使用 SentenceTransformerEmbedder (推荐).
            "hash" — 使用 HashEmbedder (无依赖, 快速测试).
        dim: 嵌入维度 (仅对 HashEmbedder 有效, SentenceTransformerEmbedder
             维度由模型决定).
        **kwargs: 传递给嵌入器构造函数的额外参数.

    Returns:
        嵌入器实例, 实现 embed/embed_query/embed_batch 接口.

    Raises:
        TypeError: 嵌入器缺少协议要求的方法。任何后端都必须同时提供
            ``embed`` / ``embed_query`` / ``embed_batch`` 与 ``model_key``：
            ``recall()`` 走的是 ``embed_query``，缺了它只会在**召回时**
            才炸（覆盖报告 D5），因此这里提前拦截。
    """
    if backend == "sentence_transformer":
        embedder = SentenceTransformerEmbedder(**kwargs)
    elif backend == "hash":
        embedder = HashEmbedder(dim=dim, **kwargs)
    else:
        raise ValueError(
            f"未知的嵌入后端: {backend}. "
            f"支持: 'sentence_transformer', 'hash'"
        )

    missing = [
        name
        for name in ("embed", "embed_query", "embed_batch", "model_key")
        if not hasattr(embedder, name)
    ]
    if missing:
        raise TypeError(
            f"嵌入后端 {backend!r} 不满足嵌入器协议，缺少: {', '.join(missing)}"
        )
    return embedder
