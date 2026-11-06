# -*- coding: utf-8 -*-
"""Dedup-merge regression tests.

Background:
    When similarity >= `dedup_threshold`, `MemoryManager.store_memory` creates
    no new record and instead merges the content into the existing memory. That
    branch previously **only updated content/embedding/timestamp/importance**,
    dropping every other field wholesale (defect D3):

      - `source_ids` dropped -> this run's L0 raw text lost its back-reference
        and became a **permanent orphan**: the "raw text kept verbatim and
        traceable back" audit chain silently broke, and since it appears in no
        memory's source_ids, even forget's cascade cleanup cannot reach it.
      - `entity_ids` dropped -> entities mentioned this run lost their link to
        this memory; the entity inverted index broke.
      - `memory_type` / `cognitive_type` dropped -> type info regressed after a
        duplicate write.
      - It even merged into superseded / archived memories, so new content went
        invisible the moment it landed.

    Also `brain.ingest` calls `store.update(mid, entity_ids=eids)` after
    writing, **overwriting** existing entity links -- on the normal path eids
    match what was just written (idempotent), but on the merge path `mid` is
    the existing memory, so the overwrite drops all entity links it had
    accumulated.

    Adjacent defect D4: `pin` is derived from importance only in
    `MemoryRecord.__post_init__`, and `VectorStore.update` bypasses it via
    `setattr`, so "auto-pin at importance>=0.7" only worked on creation --
    while the merge path raises importance precisely through update.

Coverage:
    M1 merge keeps source_ids -- this run's L0 raw text no longer orphaned (audit chain intact)
    M2 merge unions entity_ids -- existing entity links kept (incl. brain.ingest path)
    M3 type fields: replaced by the new value only when the old value is the default
    M4 no merging into superseded / archived memories (a new record is created)
    M5 D4: update syncs pin when importance changes; an explicit pin is left untouched
    M6 D3+D4 crossover: pin synced when a merge raises importance
    M7 after a merge, forget cascade-marks **both** L0 raw texts
    M8 original merge semantics unchanged: longer content / newest timestamp / max importance

Usage:
    <py311> tests/test_dedup_merge.py
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
from hippocampus_mcp.engine.manager import MemoryManager  # noqa: E402
from hippocampus_mcp.store.memory import MemoryState  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    line = f"  [{tag}] {name}"
    if detail and not cond:
        line += f"\n         -> {detail}"
    print(line, flush=True)


class ConstEmbedder:
    """Returns the same vector for every text -> cosine similarity is always 1.0.

    Used to trigger merging **deterministically**: under HashEmbedder only
    byte-identical texts reach the 0.85 threshold, so "similar but different"
    cases (e.g. which of two contents is longer) cannot be constructed.
    """

    model_key = "const:dedup_test"
    dim = 8

    def embed(self, text: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float64) / np.sqrt(self.dim)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(text)


def new_mgr(always_merge: bool = False, **kw) -> MemoryManager:
    """MemoryManager on the HashEmbedder (no model load, runs in seconds).

    With always_merge=True the embedder is swapped for the constant one
    (model_key / dim kept unchanged, so vector slots and dim checks are
    unaffected).
    """
    mgr = MemoryManager(
        persist_dir=None,
        embedder_backend="hash",
        embedding_dim=8,
        **kw,
    )
    if always_merge:
        mgr.embedder = ConstEmbedder()
    # After the D5 fix HashEmbedder ships embed_query; no manual patch needed
    assert hasattr(mgr.embedder, "embed_query"), "嵌入器缺 embed_query（D5 回归）"
    return mgr


def new_brain(tag: str) -> tuple[HippoBrain, str]:
    tmp = tempfile.mkdtemp(prefix=f"dedup_{tag}_")
    # HippoBrain has no embedding_dim parameter (dim is decided by the embedder backend)
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    assert hasattr(b.manager.embedder, "embed_query"), "嵌入器缺 embed_query（D5 回归）"
    return b, tmp


def ent_id(brain: HippoBrain, name: str) -> str | None:
    ent = brain.entities.get(name.lower())
    return ent["id"] if ent else None


# ======================================================================
# M1 · merge keeps source_ids (L0 no longer orphaned)
# ======================================================================
def test_m1_merge_keeps_source_ids() -> None:
    print("\n[M1] 合并保留 source_ids —— L0 原文不再成为孤儿")
    b, tmp = new_brain("m1")
    text = "用户在杭州工作，负责跨境电商的供应链与仓储管理。"
    r1 = b.ingest(text)
    check("M1 first ingest produces 1 L1 record", len(r1["memory_ids"]) == 1,
          f"{r1['memory_ids']}")
    mid = r1["memory_ids"][0]

    r2 = b.ingest(text)
    check("M1 second ingest hits merge (no new record)",
          r2["memory_ids"] == [mid], f"{r2['memory_ids']} vs {[mid]}")
    check("M1 L1 count still 1 after merge", b.manager.store.count() == 1,
          f"{b.manager.store.count()}")

    rec = b.manager.store.get(mid)
    check("M1 source_ids after merge contain both L0 raw texts",
          set(rec.source_ids or []) == {r1["raw_id"], r2["raw_id"]},
          f"{rec.source_ids} vs {[r1['raw_id'], r2['raw_id']]}")

    referenced = {s for r in b.manager.store._records for s in (r.source_ids or [])}
    orphans = [r["id"] for r in b.raw_log if r["id"] not in referenced]
    check("M1 every L0 raw text has an L1 back-reference (no orphans)", not orphans, f"孤儿 {orphans}")
    shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# M2 · merge unions entity_ids (existing entity links kept)
# ======================================================================
def test_m2_merge_keeps_entity_ids() -> None:
    print("\n[M2] 合并并集 entity_ids —— 不丢既存实体关联")
    b, tmp = new_brain("m2")
    text = "项目代号青鸾，负责团队是设计组。"
    r1 = b.ingest(text, entities=["青鸾"])
    mid = r1["memory_ids"][0]
    qing = ent_id(b, "青鸾")
    check("M2 first ingest linked entity Qingluan",
          qing is not None and qing in (b.manager.store.get(mid).entity_ids or []),
          f"{b.manager.store.get(mid).entity_ids}")

    r2 = b.ingest(text, entities=["设计组"])
    check("M2 second ingest hits merge", r2["memory_ids"] == [mid],
          f"{r2['memory_ids']} vs {[mid]}")
    sheji = ent_id(b, "设计组")
    check("M2 new entity DesignTeam indexed", sheji is not None, f"{list(b.entities)}")

    eids = b.manager.store.get(mid).entity_ids or []
    check("M2 existing entity Qingluan not overwritten after merge", qing in eids, f"{eids}")
    check("M2 new entity DesignTeam also linked after merge", sheji in eids, f"{eids}")

    # Reverse check: memory_ids in the entity inverted index must not dangle
    dangling = sum(
        1 for e in b.entities.values()
        for m in e["memory_ids"] if b.manager.store.get(m) is None
    )
    check("M2 entity inverted index has no dangling references", dangling == 0, f"{dangling} 条悬挂")
    shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# M3 · type fields: only default values get overwritten
# ======================================================================
def test_m3_type_fields() -> None:
    print("\n[M3] memory_type / cognitive_type：仅原值为默认值时采纳新值")
    mgr = new_mgr(always_merge=True)
    content = "用户偏好深色主题界面，也接受浅色。"
    mid = mgr.store_memory(content, memory_type="general", cognitive_type="world")

    mgr.store_memory(content, memory_type="episodic", cognitive_type="experience")
    rec = mgr.store.get(mid)
    check("M3 new type adopted when old value is the default",
          rec.memory_type == "episodic" and rec.cognitive_type == "experience",
          f"{rec.memory_type} / {rec.cognitive_type}")

    mgr.store_memory(content, memory_type="general", cognitive_type="world")
    rec = mgr.store.get(mid)
    check("M3 specific old value not clobbered by the default",
          rec.memory_type == "episodic" and rec.cognitive_type == "experience",
          f"{rec.memory_type} / {rec.cognitive_type}")
    check("M3 exactly one memory throughout", mgr.store.count() == 1, f"{mgr.store.count()}")


# ======================================================================
# M4 · no merging into superseded / archived
# ======================================================================
def test_m4_no_merge_into_inactive() -> None:
    print("\n[M4] 不合并进 superseded / archived 记忆（改为新建）")
    for state in (MemoryState.SUPERSEDED.value, MemoryState.ARCHIVED.value):
        mgr = new_mgr(always_merge=True)
        content = "用户住在杭州。"
        id1 = mgr.store_memory(content)
        mgr.store.update(id1, state=state)

        id2 = mgr.store_memory(content)
        check(f"M4 does not merge into the {state} memory (new one created)",
              id2 != id1, f"{id2} == {id1}")
        rec2 = mgr.store.get(id2)
        check(f"M4 the newly created one is active", rec2 is not None
              and rec2.state == MemoryState.ACTIVE.value,
              f"{rec2.state if rec2 else '无'}")
        check(f"M4 old record state untouched",
              mgr.store.get(id1).state == state,
              f"{mgr.store.get(id1).state}")
        check(f"M4 total count is 2", mgr.store.count() == 2, f"{mgr.store.count()}")


# ======================================================================
# M5 · D4: update syncs pin when importance changes
# ======================================================================
def test_m5_pin_sync_on_update() -> None:
    print("\n[M5] D4：update 改 importance 时同步 pin")
    mgr = new_mgr()
    mid = mgr.store_memory("用户有花生过敏，点餐需避开。", importance=0.5)
    check("M5 on creation importance=0.5 -> pin=False",
          mgr.store.get(mid).pin is False, f"{mgr.store.get(mid).pin}")

    mgr.store.update(mid, importance=0.95)
    check("M5 after update raises importance, pin auto-set True",
          mgr.store.get(mid).pin is True, f"{mgr.store.get(mid).pin}")
    pinned = sum(1 for r in mgr.store._records if r.pin)
    check("M5 tally agrees (1 record pinned)", pinned == 1, f"{pinned}")

    mgr.store.update(mid, importance=0.2)
    check("M5 after update lowers importance, pin auto-set False",
          mgr.store.get(mid).pin is False, f"{mgr.store.get(mid).pin}")

    mgr.store.update(mid, importance=0.9, pin=False)
    check("M5 explicit pin not recomputed automatically", mgr.store.get(mid).pin is False,
          f"{mgr.store.get(mid).pin}")

    # When only pin is passed (importance untouched), pin must not be overwritten in reverse
    mgr.store.update(mid, pin=True)
    check("M5 pin-only change takes the passed value", mgr.store.get(mid).pin is True,
          f"{mgr.store.get(mid).pin}")


# ======================================================================
# M6 · pin synced when merge raises importance (D3 x D4 crossover)
# ======================================================================
def test_m6_merge_pin_sync() -> None:
    print("\n[M6] 合并提升 importance 时 pin 同步")
    mgr = new_mgr(always_merge=True)
    content = "用户偏好每周一上午做周报。"
    mid = mgr.store_memory(content, importance=0.3)
    mid2 = mgr.store_memory(content, importance=0.9)
    check("M6 merge hit", mid2 == mid, f"{mid2} vs {mid}")
    rec = mgr.store.get(mid)
    check("M6 merge takes the larger importance", rec.importance == 0.9, f"{rec.importance}")
    check("M6 pin synced to True after merge", rec.pin is True, f"{rec.pin}")


# ======================================================================
# M7 · after merge, forget cascade-marks both L0 texts
# ======================================================================
def test_m7_forget_after_merge() -> None:
    print("\n[M7] 合并后 forget 能级联标记两条 L0 原文")
    b, tmp = new_brain("m7")
    text = "星轨项目由林晚负责，三月开始。"
    r1 = b.ingest(text)
    r2 = b.ingest(text)
    mid = r1["memory_ids"][0]
    check("M7 precondition: merged into one",
          r2["memory_ids"] == [mid] and b.manager.store.count() == 1)

    rep = b.forget(mid, reason="M7 测试")
    marked = set(rep.get("raw_marked") or [])
    check("M7 forget marked both L0 raw texts (audit chain fully covered)",
          marked == {r1["raw_id"], r2["raw_id"]},
          f"{marked} vs {[r1['raw_id'], r2['raw_id']]}")
    check("M7 L0 raw texts themselves kept on disk (traceable)",
          len(b.raw_log) == 2, f"{len(b.raw_log)}")
    shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# M8 · original merge semantics unchanged
# ======================================================================
def test_m8_merge_semantics_unchanged() -> None:
    print("\n[M8] 原有合并语义不变：content 取更长 / timestamp 取最新 / importance 取最大")
    mgr = new_mgr(always_merge=True)
    short = "短句。"
    long_text = "短句。附带更长的补充说明，用来验证合并时取更长的一方。"
    mid = mgr.store_memory(short, importance=0.2, timestamp=100.0)

    mid2 = mgr.store_memory(long_text, importance=0.6, timestamp=200.0)
    check("M8 merge hit under constant similarity", mid2 == mid, f"{mid2} vs {mid}")
    rec = mgr.store.get(mid)
    check("M8 content takes the longer one", rec.content == long_text, f"{rec.content!r}")
    check("M8 timestamp takes the newest", rec.timestamp == 200.0, f"{rec.timestamp}")
    check("M8 importance takes the max", rec.importance == 0.6, f"{rec.importance}")

    mid3 = mgr.store_memory(short, importance=0.1, timestamp=50.0)
    rec = mgr.store.get(mid)
    check("M8 second merge hits the same record", mid3 == mid, f"{mid3} vs {mid}")
    check("M8 shorter content does not overwrite the longer", rec.content == long_text, f"{rec.content!r}")
    check("M8 older timestamp does not overwrite the newest", rec.timestamp == 200.0, f"{rec.timestamp}")
    check("M8 smaller importance does not overwrite the max", rec.importance == 0.6, f"{rec.importance}")


def main() -> int:
    print("去重合并回归  |  后端=hash（M3~M8 用恒定嵌入器确定性触发合并）")
    for fn in (test_m1_merge_keeps_source_ids, test_m2_merge_keeps_entity_ids,
               test_m3_type_fields, test_m4_no_merge_into_inactive,
               test_m5_pin_sync_on_update, test_m6_merge_pin_sync,
               test_m7_forget_after_merge, test_m8_merge_semantics_unchanged):
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
