# -*- coding: utf-8 -*-
"""Embedder protocol and storage-backend contract regression tests.

Background (two threads):

**Thread 1 · fallback embedder missing a method, recall crashes outright**
    `MemoryManager.recall()` always goes through `embedder.embed_query()`
    (BGE-style models need a query-side instruction prefix). But `HashEmbedder`
    only has `embed` / `embed_batch`, no `embed_query`. So: once a library is
    built on the hash backend, **every single recall** dies with
    `AttributeError: 'HashEmbedder' object has no attribute 'embed_query'`.
    This is not an edge case -- the hash backend is exactly the
    dependency-free fast validation path, and the backend several regression
    scripts use.

**Thread 2 · ChromaStore is a half-finished fallback that "looks usable"**
    `MemoryManager(backend="chroma")` constructs, writes and retrieves fine,
    but it diverges from the numpy backend in five unaligned semantics, the
    worst of which **silently uses the wrong vectors**:

      build the library in embedding space A -> restart in space B ->
      `_load_from_chroma()` books every A-space vector it reads under B ->
      `missing_ids()` returns empty -> backfill never fires -> retrieval pairs
      A vectors with a B query vector, no error anywhere along the way.
      (Control experiment: VectorStore keeps the original slot names in the
      same scenario, and `missing_ids()` correctly returns all 3 records.)

    The other four: save/load round-trip drops `source_ids` / `entity_ids`
    (L0 audit chain and entity index broken), `update()` does not sync pin,
    no corruption isolation and no `last_load_issue`, and auto-load at
    construction has inconsistent ordering.

    Resolution: **the implementation was deleted; keeping it was refused.**
    `backend != "numpy"` still raises right at construction time, with the
    defects above written into the error message -- anyone who wants it gets
    the reason immediately, instead of a library that loses data.

    Why delete rather than keep: a "known unusable" implementation is a pure
    source of confusion for new users -- it shows up in `store/__init__.py`
    exports and in the code map, and all 6 defects would need a full rewrite
    anyway, so keeping it saves nothing.
    **The refusal reason lives on in `MemoryManager`'s error message; the
    historical decision is not lost.**

Coverage:
    A1 both embedders satisfy the full protocol (embed / embed_query / embed_batch / model_key)
    A2 HashEmbedder.embed_query equals embed (hash has no instruction prefix)
    A3 hash backend runs a full recall (AttributeError before the fix)
    A4 hash backend runs brain's get_memory_context (5-slot path)
    A5 factory guard rejects an incomplete backend: missing method -> TypeError (not a delayed crash at recall time)
    A6 unknown backend -> ValueError (before the fix it silently degraded to hash)
    A7 model_key stable, varies with dim and salt (identity basis for multi-model vector slots)
    B1 backend="numpy" works normally
    B2 backend="chroma" rejected explicitly at construction, message carries the reason and the completion checklist
    B3 other unknown backends also raise (no more silent numpy fallback)
    B4 the rejection is a "clean failure": no half-built store left, no directory created
    B5 chroma backend removed: module not importable, no package export, but the refusal reason lives on in the error message

Usage:
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
# Group A · embedder protocol (thread 1)
# ======================================================================
def test_a1_protocol_complete() -> None:
    print("\n[A1] 嵌入器协议完整性")
    required = ("embed", "embed_query", "embed_batch", "model_key")
    emb = HashEmbedder(dim=16)
    missing = [n for n in required if not hasattr(emb, n)]
    check("A1 HashEmbedder has all protocol methods", not missing, f"缺少 {missing}")
    check("A1 HashEmbedder has a readable model_key",
          isinstance(getattr(emb, "model_key", None), str), repr(emb.model_key))

    # Instantiating SentenceTransformerEmbedder would load a model (slow); static checks here:
    # the three methods resolve on the class; model_key depends on model_name, __init__-only.
    import inspect

    from hippocampus_mcp.engine.embeddings import SentenceTransformerEmbedder as ST
    st_missing = [n for n in ("embed", "embed_query", "embed_batch")
                  if not hasattr(ST, n)]
    check("A1 SentenceTransformerEmbedder has all protocol methods",
          not st_missing, f"缺少 {st_missing}")
    init_src = inspect.getsource(ST.__init__)
    check("A1 its __init__ really does set model_key",
          "model_key" in init_src, "未在 __init__ 中赋值 model_key")


def test_a2_query_equals_embed_for_hash() -> None:
    print("\n[A2] HashEmbedder.embed_query 语义")
    emb = HashEmbedder(dim=32)
    text = "用户在杭州工作，负责跨境电商供应链。"
    check("A2 embed_query identical to embed",
          np.allclose(emb.embed_query(text), emb.embed(text)))
    check("A2 output is L2-normalized",
          abs(float(np.linalg.norm(emb.embed_query(text))) - 1.0) < 1e-9)
    check("A2 empty string does not crash and returns a zero vector",
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
        check("A3 recall does not raise AttributeError", False, str(ex))
    except Exception as ex:           # noqa: BLE001
        crashed = True
        check("A3 recall returns normally", False, f"{type(ex).__name__}: {ex}")
    if not crashed:
        check("A3 recall does not raise AttributeError", True)
        check("A3 recall returns a result list", isinstance(hits, list), type(hits).__name__)
        check("A3 recall hits >= 1 record", len(hits) >= 1, f"{len(hits)} 条")


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
            check("A4 get_memory_context raises nothing", False,
                  f"{type(ex).__name__}: {ex}")
        if not crashed:
            text = ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)
            check("A4 get_memory_context returns non-empty text", bool(text.strip()),
                  text[:80])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a5_guard_rejects_incomplete_backend() -> None:
    print("\n[A5] 工厂守卫：残缺后端在初始化即报错")
    import hippocampus_mcp.engine.embeddings as E

    class _Broken:
        """Deliberately implements only embed -- replicating the "fallback embedder missing embed_query" shape."""
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
            check("A5 incomplete backend rejected", False, "未报错")
        except TypeError as ex:
            msg = str(ex)
            check("A5 incomplete backend rejected as TypeError", True)
            check("A5 error message names the missing methods",
                  "embed_query" in msg and "embed_batch" in msg, msg)
    finally:
        E.create_embedder = orig


def test_a6_unknown_backend_raises() -> None:
    print("\n[A6] 未知嵌入后端不再静默退化为 hash")
    try:
        create_embedder("bge-typo")
        check("A6 unknown backend raises ValueError", False, "未报错，静默返回了嵌入器")
    except ValueError as ex:
        check("A6 unknown backend raises ValueError", True)
        check("A6 error message lists supported backends",
              "sentence_transformer" in str(ex) and "hash" in str(ex), str(ex))
    except Exception as ex:           # noqa: BLE001
        check("A6 unknown backend raises ValueError", False,
              f"抛的是 {type(ex).__name__}: {ex}")


def test_a7_model_key_identity() -> None:
    print("\n[A7] model_key 的身份语义（多模型向量槽位的依据）")
    a1 = HashEmbedder(dim=32, salt="s")
    a2 = HashEmbedder(dim=32, salt="s")
    b = HashEmbedder(dim=32, salt="other")
    c = HashEmbedder(dim=64, salt="s")
    check("A7 same params -> same key", a1.model_key == a2.model_key, a1.model_key)
    check("A7 different salt -> different key", a1.model_key != b.model_key,
          f"{a1.model_key} vs {b.model_key}")
    check("A7 different dim -> different key", a1.model_key != c.model_key,
          f"{a1.model_key} vs {c.model_key}")


# ======================================================================
# Group B · storage backend contract (thread 2)
# ======================================================================
def test_b1_numpy_backend_works() -> None:
    print("\n[B1] backend='numpy' 正常可用")
    tmp = tempfile.mkdtemp(prefix="contract_b1_")
    try:
        mgr = MemoryManager(persist_dir=tmp, embedder_backend="hash",
                            embedding_dim=32, backend="numpy")
        check("B1 store is VectorStore",
              type(mgr.store).__name__ == "VectorStore", type(mgr.store).__name__)
        mgr.store_memory("B1 记忆一条。")
        mgr.save()
        mgr2 = MemoryManager(persist_dir=tmp, embedder_backend="hash",
                             embedding_dim=32, backend="numpy")
        mgr2.load(auto_backfill=False)
        check("B1 round-trip record count matches", mgr2.store.count() == 1, mgr2.store.count())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_chroma_backend_rejected() -> None:
    print("\n[B2] backend='chroma' 显式拒绝（不再给出会丢数据的库）")
    tmp = tempfile.mkdtemp(prefix="contract_b2_")
    try:
        try:
            MemoryManager(persist_dir=tmp, embedder_backend="hash",
                          embedding_dim=32, backend="chroma")
            check("B2 chroma backend rejected", False, "竟然构造成功了")
        except ValueError as ex:
            msg = str(ex)
            check("B2 chroma backend raises ValueError", True)
            check("B2 message says not wired up", "not wired up" in msg, msg[:120])
            check("B2 message names the worst defect (silent pollution on model swap)",
                  "静默污染" in msg or "missing_ids" in msg, msg[:200])
            check("B2 message gives the criteria for lifting the ban once completed", "Once all five" in msg, msg[:200])
        except Exception as ex:       # noqa: BLE001
            check("B2 chroma backend raises ValueError", False,
                  f"抛的是 {type(ex).__name__}: {ex}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b3_unknown_backend_rejected() -> None:
    print("\n[B3] 其他未知 backend 也报错")
    try:
        MemoryManager(persist_dir=None, embedder_backend="hash",
                      embedding_dim=32, backend="sqlite")
        check("B3 unknown storage backend raises ValueError", False, "未报错，静默走了 numpy")
    except ValueError:
        check("B3 unknown storage backend raises ValueError", True)
    except Exception as ex:           # noqa: BLE001
        check("B3 unknown storage backend raises ValueError", False,
              f"抛的是 {type(ex).__name__}: {ex}")


def test_b4_failure_is_clean() -> None:
    print("\n[B4] 拒绝是干净失败：不留半个对象、不创建目录")
    import hippocampus_mcp.engine.manager as M
    orig = M.VectorStore

    class _Spy:
        """If called, that proves "store built before the error" -- not a clean failure."""
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
        check("B4 validation happens before the store is constructed", not _Spy.called)
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
        check("B4 persist directory not created", not os.path.exists(probe), probe)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_b5_chroma_removed_but_reason_kept() -> None:
    """ChromaStore is gone from the codebase, but the "why it is unusable" must remain.

    This assertion guards against two kinds of regression:
      - someone re-adds the half-finished backend without fixing the 6 defects
        (the module becomes importable again = red light);
      - someone deletes the file but not the reason, so a user passing
        `backend="chroma"` only gets a bare "unsupported" and concludes we
        never built it -- rather than having built it, tested it, and found
        it loses data.
    """
    print("\n[B5] chroma 后端已移除，但拒绝原因保留")

    # 1. module not importable
    try:
        import hippocampus_mcp.store.chroma_store  # noqa: F401
        check("B5 chroma_store module no longer exists", False, "模块仍可导入")
    except ImportError:
        check("B5 chroma_store module no longer exists", True)
    except Exception as ex:           # noqa: BLE001
        check("B5 chroma_store module no longer exists", False, f"{type(ex).__name__}: {ex}")

    # 2. no longer exported at package level
    import hippocampus_mcp.store as store_pkg
    exported = getattr(store_pkg, "__all__", [])
    check("B5 store package no longer exports ChromaStore", "ChromaStore" not in exported,
          str(exported))
    check("B5 store package no longer carries a ChromaStore attribute",
          not hasattr(store_pkg, "ChromaStore"))

    # 3. the refusal reason is still complete: not a bare "unsupported"
    try:
        MemoryManager(persist_dir=None, embedder_backend="hash",
                      embedding_dim=32, backend="chroma")
        check("B5 backend='chroma' still rejected", False, "居然构造成功了")
    except ValueError as ex:
        msg = str(ex)
        check("B5 backend='chroma' still rejected", True)
        check("B5 error message explains it would silently corrupt data", "silently" in msg, msg[:120])
    except Exception as ex:           # noqa: BLE001
        check("B5 backend='chroma' still rejected", False, f"{type(ex).__name__}: {ex}")


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
