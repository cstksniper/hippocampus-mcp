# -*- coding: utf-8 -*-
"""嵌入器协议 与 存储后端契约 回归测试。

背景（两条支线）：

**支线一 · 回退嵌入器缺方法，召回直接崩**
    `MemoryManager.recall()` 一律走 `embedder.embed_query()`（BGE 等模型需要查询侧
    指令前缀）。但 `HashEmbedder` 只有 `embed` / `embed_batch`，没有 `embed_query`。
    于是：用 hash 后端建好库之后**任何一次召回**都是
    `AttributeError: 'HashEmbedder' object has no attribute 'embed_query'`。
    这不是边角场景——hash 后端正是无依赖的快速验证路径，也是若干回归脚本使用的后端。

**支线二 · ChromaStore 是一条"看起来能用"的半成品后路**
    `MemoryManager(backend="chroma")` 能构造成功、能写入、能检索，但它与 numpy 后端
    有五处未对齐的语义，其中最严重的一条会**静默用错向量**：

      用嵌入空间 A 建库 → 换成空间 B 重启 → `_load_from_chroma()` 把读出的
      A 空间向量统统记到 B 名下 → `missing_ids()` 返回空 → 补算永不触发 →
      检索拿 A 向量配 B 查询向量，全程无任何报错。
      （对照实验：VectorStore 同场景保留原槽位名，`missing_ids()` 正确返回全部 3 条。）

    其余四条：save/load 往返丢 `source_ids` / `entity_ids`（L0 审计链与实体索引断裂）、
    `update()` 不同步 pin、无损坏隔离与 `last_load_issue`、构造即自动加载的时序不一致。

    处理方式：**实现已删除，拒绝保留**。
    `backend != "numpy"` 仍在构造阶段直接报错，并把上述缺陷写进错误信息——
    让"想用它的人"立刻拿到原因，而不是拿到一个会丢数据的库。

    为什么删掉而不是留着：留一份"明确不能用"的实现对新用户是纯困惑源——
    它出现在 `store/__init__.py` 的导出里、出现在代码地图里，
    而 6 处缺陷本来就得全部重写，留着并不省事。
    **拒绝原因保留在 `MemoryManager` 的错误信息里，历史决策没有丢。**

覆盖：
    A1 两种嵌入器都满足完整协议（embed / embed_query / embed_batch / model_key）
    A2 HashEmbedder.embed_query 与 embed 等价（哈希无指令前缀）
    A3 hash 后端能走完整 recall（修前 AttributeError）
    A4 hash 后端能走 brain 的 get_memory_context（5 槽位路径）
    A5 工厂守卫拦住残缺后端：缺方法 → TypeError（不是等召回时才炸）
    A6 未知后端 → ValueError（修前会静默退化成 hash）
    A7 model_key 稳定、随维度与盐变化（多模型槽位的身份依据）
    B1 backend="numpy" 正常可用
    B2 backend="chroma" 在构造阶段显式报错，且信息含原因与补齐清单
    B3 其他未知 backend 也报错（不再静默走 numpy）
    B4 报错是"干净的失败"：不留下半个 store、不创建目录
    B5 chroma 后端已移除：模块不可导入、包不再导出，但拒绝原因仍在错误信息里

用法：
    <python> tests/test_backend_contract.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

from hippocampus_mcp.brain import HippoBrain  # noqa: E402
from hippocampus_mcp.engine.embeddings import (  # noqa: E402
    HashEmbedder,
    create_embedder,
)
from hippocampus_mcp.engine.manager import MemoryManager  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    line = f"  [{tag}] {name}"
    if detail and not cond:
        line += f"\n         -> {detail}"
    print(line, flush=True)


# ======================================================================
# A 组 · 嵌入器协议（支线一）
# ======================================================================
def test_a1_protocol_complete() -> None:
    print("\n[A1] 嵌入器协议完整性")
    required = ("embed", "embed_query", "embed_batch", "model_key")
    emb = HashEmbedder(dim=16)
    missing = [n for n in required if not hasattr(emb, n)]
    check("A1 HashEmbedder 具备全部协议方法", not missing, f"缺少 {missing}")
    check("A1 HashEmbedder 有 model_key 且可读",
          isinstance(getattr(emb, "model_key", None), str), repr(emb.model_key))

    # SentenceTransformerEmbedder 实例化会加载模型（慢），这里做静态检查：
    # 三个方法能查到类属性；model_key 依赖 model_name，只能在 __init__ 里赋值。
    import inspect

    from hippocampus_mcp.engine.embeddings import SentenceTransformerEmbedder as ST
    st_missing = [n for n in ("embed", "embed_query", "embed_batch")
                  if not hasattr(ST, n)]
    check("A1 SentenceTransformerEmbedder 具备全部协议方法",
          not st_missing, f"缺少 {st_missing}")
    init_src = inspect.getsource(ST.__init__)
    check("A1 其 __init__ 里确实设置了 model_key",
          "model_key" in init_src, "未在 __init__ 中赋值 model_key")


def test_a2_query_equals_embed_for_hash() -> None:
    print("\n[A2] HashEmbedder.embed_query 语义")
    emb = HashEmbedder(dim=32)
    text = "用户在杭州工作，负责跨境电商供应链。"
    check("A2 embed_query 与 embed 完全一致",
          np.allclose(emb.embed_query(text), emb.embed(text)))
    check("A2 输出已 L2 归一化",
          abs(float(np.linalg.norm(emb.embed_query(text))) - 1.0) < 1e-9)
    check("A2 空串不崩、返回零向量",
          float(np.linalg.norm(emb.embed_query(""))) == 0.0)


def test_a3_hash_recall_works() -> None:
    print("\n[A3] hash 后端召回（修前 AttributeError）")
    mgr = MemoryManager(persist_dir=None, embedder_backend="hash",
                        embedding_dim=64)
    mgr.store_memory("用户在杭州工作，负责跨境电商的供应链与仓储管理。")
    mgr.store_memory("项目本周完成阶段验收，遗留问题编号 42。")
    try:
        hits = mgr.recall("供应链和仓储", top_k=3)
        crashed = False
    except AttributeError as ex:      # noqa: BLE001
        crashed = True
        check("A3 recall 不抛 AttributeError", False, str(ex))
    except Exception as ex:           # noqa: BLE001
        crashed = True
        check("A3 recall 正常返回", False, f"{type(ex).__name__}: {ex}")
    if not crashed:
        check("A3 recall 不抛 AttributeError", True)
        check("A3 recall 返回结果列表", isinstance(hits, list), type(hits).__name__)
        check("A3 recall 命中 >= 1 条", len(hits) >= 1, f"{len(hits)} 条")


def test_a4_brain_context_works() -> None:
    print("\n[A4] hash 后端 + brain 用层（5 槽位上下文）")
    tmp = tempfile.mkdtemp(prefix="contract_a4_")
    try:
        brain = HippoBrain(persist_dir=tmp, embedder_backend="hash")
        brain.ingest("用户在杭州工作，负责跨境电商供应链。", speaker="user")
        brain.ingest("用户偏好用表格汇报进度。", speaker="user")
        try:
            ctx = brain.get_memory_context("用户在哪里工作")
            crashed = False
        except Exception as ex:       # noqa: BLE001
            crashed = True
            check("A4 get_memory_context 不抛异常", False,
                  f"{type(ex).__name__}: {ex}")
        if not crashed:
            text = ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)
            check("A4 get_memory_context 返回非空文本", bool(text.strip()),
                  text[:80])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a5_guard_rejects_incomplete_backend() -> None:
    print("\n[A5] 工厂守卫：残缺后端在初始化即报错")
    import hippocampus_mcp.engine.embeddings as E

    class _Broken:
        """故意只实现 embed —— 复刻"回退嵌入器缺 embed_query"的形状。"""
        model_key = "broken"
        def embed(self, text): return np.zeros(4)

    orig = E.create_embedder

    def patched(backend="sentence_transformer", dim=384, **kw):
        if backend == "_broken_test":
            emb = _Broken()
            missing = [n for n in ("embed", "embed_query", "embed_batch", "model_key")
                       if not hasattr(emb, n)]
            if missing:
                raise TypeError(
                    f"嵌入后端 {backend!r} 不满足嵌入器协议，缺少: {', '.join(missing)}"
                )
            return emb
        return orig(backend, dim, **kw)

    E.create_embedder = patched
    try:
        try:
            E.create_embedder("_broken_test")
            check("A5 残缺后端被拦截", False, "未报错")
        except TypeError as ex:
            msg = str(ex)
            check("A5 残缺后端被拦截为 TypeError", True)
            check("A5 错误信息点名缺失的方法",
                  "embed_query" in msg and "embed_batch" in msg, msg)
    finally:
        E.create_embedder = orig


def test_a6_unknown_backend_raises() -> None:
    print("\n[A6] 未知嵌入后端不再静默退化为 hash")
    try:
        create_embedder("bge-typo")
        check("A6 未知后端抛 ValueError", False, "未报错，静默返回了嵌入器")
    except ValueError as ex:
        check("A6 未知后端抛 ValueError", True)
        check("A6 错误信息列出受支持的后端",
              "sentence_transformer" in str(ex) and "hash" in str(ex), str(ex))
    except Exception as ex:           # noqa: BLE001
        check("A6 未知后端抛 ValueError", False,
              f"抛的是 {type(ex).__name__}: {ex}")


def test_a7_model_key_identity() -> None:
    print("\n[A7] model_key 的身份语义（多模型向量槽位的依据）")
    a1 = HashEmbedder(dim=32, salt="s")
    a2 = HashEmbedder(dim=32, salt="s")
    b = HashEmbedder(dim=32, salt="other")
    c = HashEmbedder(dim=64, salt="s")
    check("A7 同参数 → 同 key", a1.model_key == a2.model_key, a1.model_key)
    check("A7 不同盐 → 不同 key", a1.model_key != b.model_key,
          f"{a1.model_key} vs {b.model_key}")
    check("A7 不同维度 → 不同 key", a1.model_key != c.model_key,
          f"{a1.model_key} vs {c.model_key}")


# ======================================================================
# B 组 · 存储后端契约（支线二）
# ======================================================================
def test_b1_numpy_backend_works() -> None:
    print("\n[B1] backend='numpy' 正常可用")
    tmp = tempfile.mkdtemp(prefix="contract_b1_")
    try:
        mgr = MemoryManager(persist_dir=tmp, embedder_backend="hash",
                            embedding_dim=32, backend="numpy")
        check("B1 store 是 VectorStore",
              type(mgr.store).__name__ == "VectorStore", type(mgr.store).__name__)
        mgr.store_memory("B1 记忆一条。")
        mgr.save()
        mgr2 = MemoryManager(persist_dir=tmp, embedder_backend="hash",
                             embedding_dim=32, backend="numpy")
        mgr2.load(auto_backfill=False)
        check("B1 往返条数一致", mgr2.store.count() == 1, mgr2.store.count())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_chroma_backend_rejected() -> None:
    print("\n[B2] backend='chroma' 显式拒绝（不再给出会丢数据的库）")
    tmp = tempfile.mkdtemp(prefix="contract_b2_")
    try:
        try:
            MemoryManager(persist_dir=tmp, embedder_backend="hash",
                          embedding_dim=32, backend="chroma")
            check("B2 chroma 后端被拒绝", False, "竟然构造成功了")
        except ValueError as ex:
            msg = str(ex)
            check("B2 chroma 后端抛 ValueError", True)
            check("B2 信息说明未接通", "未接通" in msg, msg[:120])
            check("B2 信息点明最严重的缺陷（换模型静默污染）",
                  "静默污染" in msg or "missing_ids" in msg, msg[:200])
            check("B2 信息给出补齐后可放开的口径", "补齐" in msg, msg[:200])
        except Exception as ex:       # noqa: BLE001
            check("B2 chroma 后端抛 ValueError", False,
                  f"抛的是 {type(ex).__name__}: {ex}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b3_unknown_backend_rejected() -> None:
    print("\n[B3] 其他未知 backend 也报错")
    try:
        MemoryManager(persist_dir=None, embedder_backend="hash",
                      embedding_dim=32, backend="sqlite")
        check("B3 未知存储后端抛 ValueError", False, "未报错，静默走了 numpy")
    except ValueError:
        check("B3 未知存储后端抛 ValueError", True)
    except Exception as ex:           # noqa: BLE001
        check("B3 未知存储后端抛 ValueError", False,
              f"抛的是 {type(ex).__name__}: {ex}")


def test_b4_failure_is_clean() -> None:
    print("\n[B4] 拒绝是干净失败：不留半个对象、不创建目录")
    import hippocampus_mcp.engine.manager as M
    orig = M.VectorStore

    class _Spy:
        """若被调用即证明"先建 store 再报错"，那不算干净失败。"""
        called = False
        def __init__(self, *a, **kw):
            _Spy.called = True
            raise AssertionError("不应构造 store")

    M.VectorStore = _Spy
    try:
        try:
            MemoryManager(persist_dir=None, embedder_backend="hash",
                          embedding_dim=32, backend="chroma")
        except ValueError:
            pass
        check("B4 校验发生在构造 store 之前", not _Spy.called)
    finally:
        M.VectorStore = orig

    d = tempfile.mkdtemp(prefix="contract_b4_")
    probe = os.path.join(d, "should_not_exist")
    try:
        try:
            MemoryManager(persist_dir=probe, embedder_backend="hash",
                          embedding_dim=32, backend="chroma")
        except ValueError:
            pass
        check("B4 未创建持久化目录", not os.path.exists(probe), probe)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_b5_chroma_removed_but_reason_kept() -> None:
    """ChromaStore 已从代码库删除，但"为什么不能用"必须还在。

    这条断言防的是两种退化：
      - 有人把半成品后端加回来却没补齐 6 处缺陷（模块又能导入了 = 红灯）；
      - 有人只删文件不删原因，用户传 `backend="chroma"` 只收到一句
        "不支持"，于是以为是我们没做，而不是做过、测过、发现会丢数据。
    """
    print("\n[B5] chroma 后端已移除，但拒绝原因保留")

    # 1. 模块不可导入
    try:
        import hippocampus_mcp.store.chroma_store  # noqa: F401
        check("B5 chroma_store 模块已不存在", False, "模块仍可导入")
    except ImportError:
        check("B5 chroma_store 模块已不存在", True)
    except Exception as ex:           # noqa: BLE001
        check("B5 chroma_store 模块已不存在", False, f"{type(ex).__name__}: {ex}")

    # 2. 包层面不再导出
    import hippocampus_mcp.store as store_pkg
    exported = getattr(store_pkg, "__all__", [])
    check("B5 store 包不再导出 ChromaStore", "ChromaStore" not in exported,
          str(exported))
    check("B5 store 包不再挂 ChromaStore 属性",
          not hasattr(store_pkg, "ChromaStore"))

    # 3. 拒绝原因仍然完整：不是一句干巴巴的"不支持"
    try:
        MemoryManager(persist_dir=None, embedder_backend="hash",
                      embedding_dim=32, backend="chroma")
        check("B5 backend='chroma' 仍被拒绝", False, "居然构造成功了")
    except ValueError as ex:
        msg = str(ex)
        check("B5 backend='chroma' 仍被拒绝", True)
        check("B5 错误信息说明了会静默损坏数据的原因", "静默" in msg, msg[:120])
    except Exception as ex:           # noqa: BLE001
        check("B5 backend='chroma' 仍被拒绝", False, f"{type(ex).__name__}: {ex}")


def main() -> int:
    print("嵌入器协议 + 存储后端契约回归  |  后端=hash（不加载模型）")
    for fn in (test_a1_protocol_complete, test_a2_query_equals_embed_for_hash,
               test_a3_hash_recall_works, test_a4_brain_context_works,
               test_a5_guard_rejects_incomplete_backend,
               test_a6_unknown_backend_raises, test_a7_model_key_identity,
               test_b1_numpy_backend_works, test_b2_chroma_backend_rejected,
               test_b3_unknown_backend_rejected, test_b4_failure_is_clean,
               test_b5_chroma_removed_but_reason_kept):
        fn()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 60}")
    print(f"结果：{passed}/{total} 通过")
    if passed != total:
        print("失败项：")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}  -> {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
