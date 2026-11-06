"""ONNX Runtime 嵌入器：运行期**不加载 torch / sentence-transformers**。

为什么要有这个模块
------------------
生产进程单份稳态 ~1.25GB，其中 torch 运行时 ~160MB + sentence-transformers
~210MB 是**纯推理开销**（实测分段账单）。
本模块用 ``onnxruntime`` + ``tokenizers``（Rust，都很轻）复刻同一套向量，
把这两块换成 ONNX 推理，实测可省 ~280MB。

它和 SentenceTransformerEmbedder 的关系
---------------------------------------
**接口完全对齐**（``embed`` / ``embed_query`` / ``embed_batch`` /
``max_seq_length`` / ``embedding_coverage`` / ``dim`` / ``model_key``），
所以上层一行不用改。切换靠环境变量 ``HIPPOCAMPUS_EMB_BACKEND``：

* ``torch`` —— 旧的 sentence-transformers 路径（行为与今天完全一致）
* ``onnx``  —— 本模块；**必须有导出产物**，缺了就报错，不静默回落
* ``auto``  —— 有产物走 onnx，没有走 torch（默认）

四个容易踩的坑（都踩过或差点踩）
--------------------------------
1. **pooling 不能猜**：bge-base-zh-v1.5 在 sentence-transformers 里是
   **CLS pooling**（``1_Pooling/config.json: pooling_mode_cls_token=true``），
   不是常见的 mean。猜成 mean 的话向量完全对不上，而且**不报错**。
   本模块只信 ``onnx_meta.json`` 里写死的 ``pooling`` 字段。
2. **不要静默回落**：ONNX 产物缺失时若悄悄回 torch，省内存这件事就永远
   没人知道没生效。所以 ``onnx`` 模式下缺产物必须抛错。
3. **维度也不能猜**（踩过的缺陷）：``word_embedding_dimension``
   早先只认顶层字段、读不到就 ``or 768`` 兜底，而导出器曾把维度写在
   ``pooling_config`` 嵌套里 ⇒ 非 bge 模型（MiniLM 384）挂 ONNX 后
   ``dim`` 恒报 768，LNN 核按 768 建、向量实出 384，直到首次 ingest 才在
   ``matmul`` 上炸（x=386 vs w=770）——报错点离病根十万八千里。现在两种
   嵌套形态都认，都没有则显式抛错，见 :func:`_resolve_embedding_dim`。
4. **并发**（🔴 早期注释曾写错，已更正）：
   早期注释这里写的是「onnxruntime 与 Rust tokenizer 都没有 ST 那个可变借用
   问题，因此**不加全局编码锁**」——**这是错的，且后果是进程崩溃**。

   本类实测一直在用 ``enable_truncation`` / ``no_truncation``（下面 185 行、
   ``embedding_coverage`` 里都有），它们与 ST 侧踩的是**同一个**
   ``tokenizers`` Rust 可变借用：并发调用会 panic。区别在于 ST 侧抛的是
   Python 层 ``RuntimeError: Already borrowed``（能进日志），而本路径在
   daemon 的 worker 线程里直接把**进程**带走 —— 没有 traceback、没有
   faulthandler dump，表现就是「daemon 无声消失」。

   复现场景：两个工作区复用同一个 daemon，两个 brain 并发首次
   encode ⇒ daemon 崩，两个宿主会话一起挂住（等一个已经死了的连接）。
   这正是「两个任务同时开」这个**正常用法**，不是边缘场景。

   锁只包 tokenizer 那段（``session.run`` 仍可并发，不牺牲吞吐）。
   不用 ``embeddings._ENCODE_LOCK`` 是因为依赖方向：``embeddings`` 会 import
   本模块（取 ``backend_kind`` / ``OnnxEmbedder``），反向 import 会成环。
   两个嵌入器各自护各自的 tokenizer 实例，语义上是对的。
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import threading
from typing import Optional

import numpy as np

#: tokenizer 的**可变借用**互斥锁（理由见本模块顶部第 3 条）。
#: 只护 tokenizer，不护 ``session.run``。
_ENCODE_LOCK = threading.RLock()

logger = logging.getLogger(__name__)

#: 环境变量：选择嵌入后端（torch / onnx / auto）
ENV_BACKEND = "HIPPOCAMPUS_EMB_BACKEND"

#: ONNX 产物在模型目录下的子目录名
ONNX_SUBDIR = "onnx"
ONNX_MODEL_FILE = "model.onnx"
ONNX_META_FILE = "onnx_meta.json"

# BGE zh 系列官方查询指令前缀（与 SentenceTransformerEmbedder 保持一致）
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 模型加载串行锁（同 ST 版：并发 import / 建 session 会互相等）
_MODEL_LOAD_LOCK = threading.RLock()

# 模型生命周期（延迟加载 / 空闲卸载）。与 torch 侧共用同一份状态。
from . import model_lifecycle as _ML  # noqa: E402
from ..config import load_config  # noqa: E402  （env 读收编 config）


def _with_model_use(fn):
    """给会用到 session/tokenizer 的方法包租约（理由同 ST 版装饰器）。"""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with _ML.model_use():
            return fn(self, *args, **kwargs)

    return wrapper


def backend_kind() -> str:
    """读 ``HIPPOCAMPUS_EMB_BACKEND``，未设置或非法值一律按 ``auto``。"""
    raw = (load_config().onnx_backend or "").strip().lower()
    return raw if raw in ("torch", "onnx", "auto") else "auto"


def find_onnx_artifacts(model_dir: str) -> tuple[str, dict] | None:
    """在模型目录下找 ONNX 产物。找到返回 ``(model_path, meta)``，否则 None。"""
    d = os.path.join(model_dir, ONNX_SUBDIR)
    model_path = os.path.join(d, ONNX_MODEL_FILE)
    meta_path = os.path.join(d, ONNX_META_FILE)
    if not (os.path.isfile(model_path) and os.path.isfile(meta_path)):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model_path, meta


def _resolve_embedding_dim(meta: dict, meta_file: str) -> int:
    """从 ``onnx_meta.json`` 解析向量维度；解析不到**显式报错**，不猜。

    实测踩中的缺陷：这里原先写作
    ``int(meta.get("word_embedding_dimension", 0)) or 768``，而
    ``scripts/export_onnx.py`` 早期把维度写在 ``pooling_config`` 嵌套里 ⇒
    非 bge 模型（MiniLM 384）挂 ONNX 后 ``dim`` 恒报 768 ⇒ LNN 核按 768
    建、向量实出 384 ⇒ 首次 ingest 才在 ``matmul`` 上炸（x=386 vs w=770），
    报错点离病根十万八千里。

    依次认两种形态：顶层 ``word_embedding_dimension``（现行导出器）与
    ``pooling_config.word_embedding_dimension``（早期导出器）；两处都没有
    或不是正数 ⇒ 抛错，并指明重导出的命令。
    """
    raw = meta.get("word_embedding_dimension")
    if raw in (None, "", 0):
        nested = meta.get("pooling_config")
        if isinstance(nested, dict):
            raw = nested.get("word_embedding_dimension")
    try:
        dim = int(raw)
    except (TypeError, ValueError):
        dim = 0
    if dim <= 0:
        raise ValueError(
            f"{meta_file} 里没有可用的 word_embedding_dimension"
            f"（顶层与 pooling_config 嵌套两处都没有）。不猜维度——猜错会让"
            f"向量与 torch 版完全对不上且不报错。请重跑 "
            f"python scripts/export_onnx.py --model <model dir>, "
            f"或改回 {ENV_BACKEND}=torch。"
        )
    return dim


class OnnxEmbedder:
    """用 onnxruntime 跑 sentence-transformers 导出的模型。

    Args:
        model_dir: **本地模型目录**（如 ``.../libs/models/bge-base-zh-v1.5``）。
            本类不接受 Hub 名称——它设计的前提就是"运行期不联网、不碰 torch"。
        model_name: 对外报告的模型名（用于向量槽位键 ``model_key``），
            必须与 torch 版一致，否则同一份语料会被当成两个模型的向量。
    """

    def __init__(
        self,
        model_dir: str,
        model_name: str = "",
        libs_path: Optional[str] = None,
    ) -> None:
        self.model_dir = os.path.abspath(model_dir)
        self.model_name = model_name or os.path.basename(self.model_dir.rstrip("/\\"))
        self.model_key = self.model_name
        self.is_bge = "bge" in self.model_name.lower()
        self.model_source = "onnx"
        self.model_explicit = True
        self.resolved_model_path: str | None = None
        self._libs_path = libs_path
        #: F6：本实例是否被进程级共享池持有（True = 同模型的另一个库也在用
        #: 这一份）。打在**入池的那一刻**，见 ``get_shared_onnx``。
        self.shared_from_pool = False

        found = find_onnx_artifacts(self.model_dir)
        if found is None:
            raise FileNotFoundError(
                f"The ONNX backend was selected explicitly ({ENV_BACKEND}=onnx), but {self.model_dir} has no "
                f"{ONNX_SUBDIR}/{ONNX_MODEL_FILE} + {ONNX_META_FILE}.\n"
                f"Run first: python scripts/export_onnx.py --model {self.model_dir}\n"
                f"or switch back to {ENV_BACKEND}=torch."
            )
        self._onnx_path, self._meta = found

        self._pooling = self._meta.get("pooling") or ""
        if self._pooling not in ("cls", "mean", "max"):
            raise ValueError(
                f"{self._onnx_path}: invalid pooling field in onnx_meta.json at "
                f"{self._pooling!r}（只认 cls/mean/max）。不猜 pooling —— "
                f"猜错会让向量与 torch 版完全对不上且不报错。"
            )
        self._max_seq_length = int(self._meta.get("max_seq_length", 512))
        self._meta_file = os.path.join(os.path.dirname(self._onnx_path), ONNX_META_FILE)
        self.dim = _resolve_embedding_dim(self._meta, self._meta_file)

        self._session = None
        self._tok = None
        self._model_ready = False
        # 登记到进程级生命周期（弱引用，不阻碍回收）
        _ML.register(self)

    def release_model(self) -> bool:
        """卸掉 session + tokenizer，把 ~400MB 还给操作系统。

        与 ST 版同理：**保留本实例**（它的 ``model_dir`` / ``_meta`` / pooling
        配置还要复用），只丢掉占内存的那两样，并把 ``_model_ready`` 置回
        False —— 下一次 :meth:`_ensure_model` 会原样重载。

        Returns:
            是否真的卸掉了东西（本来就没加载返回 False）。
        """
        did = False
        if self._session is not None:
            # ort.InferenceSession 没有 close()：断开引用即可，它的 native
            # 内存在 refcount 归零时释放（本机实测卸载后 PrivateUsage 从
            # 537MB 回落到 138MB，见 mem_lifecycle 探针）。
            self._session = None
            did = True
        if self._tok is not None:
            self._tok = None
            did = True
        if did:
            self._input_names = []
            self._output_name = ""
            self._model_ready = False
        return did

    def __repr__(self) -> str:
        where = self.resolved_model_path or "(not yet loaded)"
        return f"{self.model_name} @ {where} [onnx]"

    # -- 加载 ------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model_ready:
            return
        with _MODEL_LOAD_LOCK:
            if self._model_ready:
                return
            self._load_model()

    def _load_model(self) -> None:
        """加载 onnxruntime session 与 Rust tokenizer（**都不需要 torch**）。"""
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - 依赖缺失分支
            raise ImportError(
                "ONNX 后端需要 onnxruntime：`pip install onnxruntime`（"
                "注意：它不依赖 torch，装它不会把 torch 拉回来）。\n"
                f"错误: {e}"
            ) from e
        try:
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ONNX 后端需要 tokenizers（HuggingFace 的 Rust 分词库）："
                "`pip install tokenizers`。\n"
                f"错误: {e}"
            ) from e

        # 加载状态先行 flush 到 stderr（stdout 是 JSON-RPC 专用通道）：
        # 让"慢"与"坏"可区分 —— 冷启动几秒属正常，反复卡死看 daemon 日志。
        print(f"[hippocampus] loading embedding model (ONNX): {self._onnx_path}",
              file=sys.stderr, flush=True)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 🔴 批量 embed 的内存/延迟权衡（实测依据）：
        # * arena/pattern **开**（默认）：单句延迟 ~34ms（recall 路径的生命线）；
        #   但特定批量内容下首次大 batch 会一次性分配 ~2GB 且被 arena 常驻
        #   （无泄漏、不随次数增长；精确触发条件待查，生产单句路径不受影响）。
        # * **关**（HIPPOCAMPUS_ONNX_NO_ARENA=1）：批量内存完全平稳（820MB 恒定），
        #   但单句延迟 ~1048ms（30×）——只给离线批量脚本用（reembed 等）。
        if (load_config().onnx_no_arena or "").strip() not in ("1", "true"):
            so.enable_mem_pattern = True
            so.enable_cpu_mem_arena = True
        else:
            so.enable_mem_pattern = False
            so.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(
            self._onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name

        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"the model directory lacks tokenizer.json: {tok_path}")
        self._tok = Tokenizer.from_file(tok_path)
        # 截断到模型硬上限：与 ST 的 max_seq_length 行为对齐（超出静默丢弃）
        # 这两个调用改的是 tokenizer 的**可变状态**，同样要过锁
        with _ENCODE_LOCK:
            self._tok.enable_truncation(max_length=self._max_seq_length)
            # 🔴 实测：padding **不能**恒定到 max_seq_length。
            # 原 `enable_padding(length=512)` 让每个查询都按 512 token 跑前向——
            # 单句延迟 480ms（batch-longest 动态 padding 只要 ~30ms，**16 倍**），
            # 且批量 36 句时 attention 中间量 (36×12×512×512) 直接把 ORT arena
            # 撑出 ~2GB 常驻。CLS pooling + attention mask 下 padding 与数值无关
            # （cos=1.0 实测验证），改 batch-longest 纯赚。
            self._tok.enable_padding()

        self.resolved_model_path = self._onnx_path
        # 维度以 session 输出为准（meta 里的只是预估）
        shape = self._session.get_outputs()[0].shape
        if len(shape) == 3 and isinstance(shape[2], int):
            self.dim = int(shape[2])
        # 与 ST 版同纪律：跑通一次再置 ready，避免别的线程提前进场。
        # 🔴 这里必须走 ``_forward`` 而不是 ``_encode``：``_encode`` 会调
        # ``_ensure_model``，而此时 ``_model_ready`` 还是 False ⇒ 无限递归
        # （实测：RecursionError，且它把真正的加载错误一起掩盖了）。
        _ = self._forward(["warmup"])
        print(f"[hippocampus] embedding model ready (ONNX): {self.model_name}"
              f"（{self.dim}D)", file=sys.stderr, flush=True)
        self._model_ready = True

    # -- 编码 ------------------------------------------------------------
    def _forward(self, texts: list[str]) -> np.ndarray:
        """纯前向：**不**做加载检查（调用方保证已加载）。

        与 :meth:`_encode` 分开的唯一理由就是上面那条递归红线——加载收尾
        时的 warmup 只能走这里。
        """
        # 🔴 tokenizer 的可变借用必须串行（见模块顶部第 3 条）：并发 encode
        # 会让 Rust 侧 panic，在 daemon 的 worker 线程里表现为**整个进程
        # 无声消失**。``session.run`` 留在锁外——它才是耗时的部分。
        with _ENCODE_LOCK:
            encodings = self._tok.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # 有些模型没有 token_type_ids 输入（导出时会记在 meta.input_names 里）
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = token_type_ids

        hidden = self._session.run([self._output_name], feed)[0]  # (B, L, H)

        if self._pooling == "cls":
            pooled = hidden[:, 0, :]
        elif self._pooling == "mean":
            mask = attention_mask[:, :, None].astype(np.float32)
            summed = (hidden * mask).sum(axis=1)
            counts = np.maximum(mask.sum(axis=1), 1e-9)
            pooled = summed / counts
        else:  # max
            mask = attention_mask[:, :, None].astype(np.float32)
            pooled = (hidden * mask + (1 - mask) * (-1e9)).max(axis=1)

        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return (pooled / norm).astype(np.float64)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """tokenize → onnx 前向 → pooling → L2 归一化（含懒加载）。"""
        self._ensure_model()
        return self._forward(texts)

    @_with_model_use
    def embed(self, text: str) -> np.ndarray:
        """文档侧嵌入（不加查询前缀）。返回 L2 归一化向量。"""
        if not text:
            self._ensure_model()
            return np.zeros(self.dim, dtype=np.float64)
        return self._encode([text])[0]

    @_with_model_use
    def embed_query(self, text: str) -> np.ndarray:
        """查询侧嵌入（BGE 系加官方指令前缀）。"""
        if self.is_bge:
            text = BGE_ZH_QUERY_PREFIX + text
        return self.embed(text)

    @_with_model_use
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """批量嵌入。返回 ``(N, dim)``，每行已 L2 归一化。

        🔴 **空串不要跳过**：sentence-transformers 的 ``encode(["", ...])``
        对空串照样跑一次前向（[CLS][SEP]），得到的是**非零**向量；而单条
        :meth:`embed("")` 按既有约定返回**零**向量。这两个不一致是 torch 版
        的既有行为，本类必须**逐字复刻**——早期版本我在这里把空串短路成零
        向量，结果批量路径 cos=0.000000（验收直接挂）。
        """
        if not texts:
            self._ensure_model()
            return np.zeros((0, self.dim), dtype=np.float64)
        # 分块：一次塞几万条会把内存打爆，且与 ST 的 batch_size 语义对齐
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        step = max(1, int(batch_size))
        for start in range(0, len(texts), step):
            chunk = texts[start : start + step]
            out[start : start + len(chunk)] = self._encode(chunk)
        return out

    @property
    def max_seq_length(self) -> int:
        """模型硬上限 token 数（超出部分被静默丢弃，与 torch 版同语义）。"""
        return self._max_seq_length

    @_with_model_use
    def embedding_coverage(self, text: str) -> dict:
        """截断自曝——**返回键必须与 torch 版逐字一致**。

        上层（``brain`` / ``brain_stats``）读的是 ``truncated`` /
        ``embedded_chars`` / ``dropped_ratio``。少一个键就是**静默降级**：
        写入照样成功、库里照样有，只是"第 N 字之后检索不到"这件事没人知道
        ——这条红线比向量算错更隐蔽。

        ``no_truncation()`` 是必须的：本类的 tokenizer 常驻着
        ``enable_truncation(512)``，不关掉的话 ids 长度永远 ≤ 512，
        ``truncated`` 恒为 False —— 自曝直接失去意义。
        """
        total = len(text or "")
        out = {
            "truncated": False,
            "total_chars": total,
            "embedded_chars": total,
            "tokens": 0,
            "max_seq_length": int(self._max_seq_length),
            "dropped_ratio": 0.0,
        }
        if total == 0 or out["max_seq_length"] <= 0:
            return out
        try:
            self._ensure_model()
            # 关截断是**临时**改 tokenizer 状态，与正在 encode 的
            # 别的线程撞上同样是可变借用冲突 —— 必须与 encode 同一把锁。
            # 🔴 no_truncation() 是普通方法（永久关闭、返回 None），不是上下文
            # 管理器——`with self._tok.no_truncation():` 会在关掉截断后立刻抛
            # TypeError 且被下方 except 吞掉，截断从此再也回不来，超 512 token
            # 的文本全部打穿到 ORT 报广播错。必须显式关、finally 里显式恢复。
            with _ENCODE_LOCK:
                self._tok.no_truncation()
                try:
                    enc = self._tok.encode(text, add_special_tokens=True)
                finally:
                    self._tok.enable_truncation(max_length=self._max_seq_length)
            ids = enc.ids
            n = len(ids)
            out["tokens"] = n
            limit = out["max_seq_length"]
            if n <= limit:
                return out
            out["truncated"] = True
            kept = 0
            offs = enc.offsets
            if offs:
                idx = min(limit, len(offs)) - 1
                end = int(offs[idx][1])
                if 0 < end <= total:
                    kept = end
            if kept <= 0:
                # 与 torch 版同兜底：拿不到 offset 就按比例估，宁可给近似值
                # 也不要回退成"没截断"。
                kept = int(total * limit / max(n, 1))
            out["embedded_chars"] = max(0, min(total, kept))
            out["dropped_ratio"] = round(1.0 - out["embedded_chars"] / total, 4)
        except Exception as exc:      # 自曝绝不能反过来把写入搞挂
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out


# ==================================================================
# F6 · 进程级共享池（同模型只驻留一份）
# ==================================================================
# 为什么不共享"整个嵌入器实例"而只共享"重的那一层"：
#   ``OnnxEmbedder`` 只装 session + Rust tokenizer，没有任何随库变化的状态，
#   可以安全共享；而上层（``SentenceTransformerEmbedder`` / ``MemoryManager``）
#   身上带着 **决策元数据** —— ``model_source``（模型名是显式给的还是回落出来的）
#   决定"库内模型与当前模型冲突时是报错还是跟随库数据"（见
#   ``manager._sync_embedder_with_store``）。把整个实例共享出去，第二个库会
#   继承第一个库的 source，本该"跟随库数据"的库可能被判成"显式指定"而直接抛
#   ``EmbedderModelMismatchError`` ⇒ **库起不来**。所以只共享重对象，
#   每个库仍持有自己的薄壳与自己的元数据。
#
# 收益（实测基线：单库 daemon 543.5MB、双库 947.8MB，第二库增量
# ≈404MB = 第二份权重）：共享后同模型的第二库不再加载权重。

#: 逃逸舱：``HIPPOCAMPUS_SHARED_EMBEDDER=0`` 关闭共享（行为回到 F6 之前）。
ENV_SHARED = "HIPPOCAMPUS_SHARED_EMBEDDER"

#: 池上限（保险丝）。正常只用得上 1 份；留余量是给"库内模型不同"的场景
#: （如一个 bge 库 + 一个 MiniLM 库 ⇒ 2 份，本来就该 2 份）。超过上限的
#: 不再入池，**不报错**——新建的照常工作，只是不复用。防的是换模型实验把
#: N 份权重全攥在池里永不释放。
POOL_MAX = 4

_ONNX_POOL: "dict[tuple[str, str], OnnxEmbedder]" = {}
_ONNX_POOL_LOCK = threading.Lock()


def shared_embedder_enabled() -> bool:
    """共享嵌入器总开关（默认开；``HIPPOCAMPUS_SHARED_EMBEDDER=0`` 关闭）。"""
    raw = (load_config().shared_embedder or "").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _pool_key(model_dir: str, model_name: str) -> tuple[str, str]:
    """池键 = （绝对路径, 对外模型名）。

    带上 ``model_name`` 是因为同一个目录可以被不同名字引用，而 ``model_key``
    （向量槽位键）取的是名字——名字不同就必须算两个槽位，不能共用一份实例
    报告同一个 ``model_key``。
    """
    return (os.path.abspath(model_dir), model_name or "")


def get_shared_onnx(
    model_dir: str,
    model_name: str = "",
    libs_path: Optional[str] = None,
) -> "OnnxEmbedder":
    """取一份 ``OnnxEmbedder``：**同（目录, 名字）复用同一实例**。

    命中池时不重新建 session、不重新读 tokenizer.json——这就是 F6 省下的
    那 ~400MB 与那 30s 加载。返回的实例可能**尚未** ``_ensure_model()``
    （构造是轻的，加载在首次 encode），调用方照常调 ``_ensure_model`` 即可，
    它内部有双重检查锁。

    ``shared_from_pool`` 会被打在入池的实例上：拿到 True 就说明这份权重是
    进程级共享的（第二个库拿到时必然是 True）。
    """
    key = _pool_key(model_dir, model_name)
    if shared_embedder_enabled():
        with _ONNX_POOL_LOCK:
            hit = _ONNX_POOL.get(key)
        if hit is not None:
            return hit
    emb = OnnxEmbedder(model_dir=key[0], model_name=model_name, libs_path=libs_path)
    if shared_embedder_enabled():
        with _ONNX_POOL_LOCK:
            if len(_ONNX_POOL) < POOL_MAX:
                _ONNX_POOL[key] = emb
                emb.shared_from_pool = True
    return emb


def onnx_pool_stats() -> dict:
    """自曝池现状（给 ``brain_stats`` 与测试用）。"""
    with _ONNX_POOL_LOCK:
        return {
            "enabled": shared_embedder_enabled(),
            "size": len(_ONNX_POOL),
            "max": POOL_MAX,
            "keys": [f"{k[1]}@{k[0]}" for k in _ONNX_POOL],
        }


def reset_onnx_pool() -> int:
    """清空 ONNX 池（**只给测试用**），返回清掉的份数。

    测试要在同一进程里验证"关闭共享后确实各建各的"，不重置会被上一个用例
    留在池里的实例污染，变成假绿。
    """
    with _ONNX_POOL_LOCK:
        n = len(_ONNX_POOL)
        _ONNX_POOL.clear()
    return n
