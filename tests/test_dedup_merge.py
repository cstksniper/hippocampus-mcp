# -*- coding: utf-8 -*-
"""去重合并（dedup merge）回归测试。

背景：
    `MemoryManager.store_memory` 在相似度 >= `dedup_threshold` 时不新建，而是把
    内容合并进既存记忆。这条分支此前**只更新 content/embedding/timestamp/importance**，
    其余字段被整体丢弃（缺陷 D3）：

      - `source_ids` 丢掉 → 本次写入的 L0 原文失去回指，成为**永久孤儿**：
        "原文一字不改、可回指核对"的审计链静默断裂，而且因为不在任何记忆的
        source_ids 里，连 forget 的级联清理都覆盖不到它。
      - `entity_ids` 丢掉 → 本次提到的实体与这条记忆失去关联，实体倒排索引断裂。
      - `memory_type` / `cognitive_type` 丢掉 → 类型信息在重复写入后回退。
      - 还会合并进 superseded / archived 的记忆，让新内容一进来就隐身。

    另外 `brain.ingest` 在写完记忆后会 `store.update(mid, entity_ids=eids)`
    **覆盖**既有实体关联——普通路径下 eids 与刚写入的相同（幂等），但合并路径下
    `mid` 是既存记忆，覆盖会把它此前积累的实体关联全部丢掉。

    与之相邻的缺陷 D4：`pin` 只在 `MemoryRecord.__post_init__` 里按 importance 设定，
    而 `VectorStore.update` 走 `setattr` 绕过它，于是"importance>=0.7 自动钉住"
    只对新建生效。而合并路径恰恰是通过 update 提升 importance 的。

覆盖：
    M1 合并保留 source_ids —— 本次 L0 原文不再成为孤儿（审计链完整）
    M2 合并并集 entity_ids —— 不丢既存实体关联（含 brain.ingest 路径）
    M3 类型字段：仅当原值为默认值时才被新值替换
    M4 不合并进 superseded / archived 记忆（改为新建）
    M5 D4：update 改 importance 时同步 pin，显式传 pin 时不干预
    M6 D3+D4 交叉：合并提升 importance 时 pin 同步
    M7 合并后 forget 能级联标记**两条** L0 原文
    M8 原有合并语义不变：content 取更长、timestamp 取最新、importance 取最大

用法：
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
    """所有文本返回同一向量 → 余弦相似度恒为 1.0。

    用于**确定性**地触发合并：HashEmbedder 下只有逐字相同的文本才会命中
    0.85 的阈值，无法构造"近似但不同"的用例（比如比较两条内容谁更长）。
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
    """HashEmbedder 管理器（不加载模型，秒级）。

    always_merge=True 时替换为恒定嵌入器（保持 model_key / dim 不变，
    因此不影响向量槽位与维度校验）。
    """
    mgr = MemoryManager(
        persist_dir=None,
        embedder_backend="hash",
        embedding_dim=8,
        **kw,
    )
    if always_merge:
        mgr.embedder = ConstEmbedder()
    # D5 修好后 HashEmbedder 自带 embed_query，无需再手动补
    assert hasattr(mgr.embedder, "embed_query"), "嵌入器缺 embed_query（D5 回归）"
    return mgr


def new_brain(tag: str) -> tuple[HippoBrain, str]:
    tmp = tempfile.mkdtemp(prefix=f"dedup_{tag}_")
    # HippoBrain 没有 embedding_dim 参数（维度由嵌入后端决定）
    b = HippoBrain(persist_dir=tmp, embedder_backend="hash")
    assert hasattr(b.manager.embedder, "embed_query"), "嵌入器缺 embed_query（D5 回归）"
    return b, tmp


def ent_id(brain: HippoBrain, name: str) -> str | None:
    ent = brain.entities.get(name.lower())
    return ent["id"] if ent else None


# ======================================================================
# M1 · 合并保留 source_ids（L0 不再孤儿）
# ======================================================================
def test_m1_merge_keeps_source_ids() -> None:
    print("\n[M1] 合并保留 source_ids —— L0 原文不再成为孤儿")
    b, tmp = new_brain("m1")
    text = "用户在杭州工作，负责跨境电商的供应链与仓储管理。"
    r1 = b.ingest(text)
    check("M1 第一次 ingest 产出 1 条 L1", len(r1["memory_ids"]) == 1,
          f"{r1['memory_ids']}")
    mid = r1["memory_ids"][0]

    r2 = b.ingest(text)
    check("M1 第二次 ingest 命中合并（不新建）",
          r2["memory_ids"] == [mid], f"{r2['memory_ids']} vs {[mid]}")
    check("M1 合并后 L1 总数仍为 1", b.manager.store.count() == 1,
          f"{b.manager.store.count()}")

    rec = b.manager.store.get(mid)
    check("M1 合并后 source_ids 含两条 L0 原文",
          set(rec.source_ids or []) == {r1["raw_id"], r2["raw_id"]},
          f"{rec.source_ids} vs {[r1['raw_id'], r2['raw_id']]}")

    referenced = {s for r in b.manager.store._records for s in (r.source_ids or [])}
    orphans = [r["id"] for r in b.raw_log if r["id"] not in referenced]
    check("M1 所有 L0 原文都有 L1 回指（无孤儿）", not orphans, f"孤儿 {orphans}")
    shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# M2 · 合并并集 entity_ids（不丢既存实体关联）
# ======================================================================
def test_m2_merge_keeps_entity_ids() -> None:
    print("\n[M2] 合并并集 entity_ids —— 不丢既存实体关联")
    b, tmp = new_brain("m2")
    text = "项目代号青鸾，负责团队是设计组。"
    r1 = b.ingest(text, entities=["青鸾"])
    mid = r1["memory_ids"][0]
    qing = ent_id(b, "青鸾")
    check("M2 第一次 ingest 关联了实体「青鸾」",
          qing is not None and qing in (b.manager.store.get(mid).entity_ids or []),
          f"{b.manager.store.get(mid).entity_ids}")

    r2 = b.ingest(text, entities=["设计组"])
    check("M2 第二次 ingest 命中合并", r2["memory_ids"] == [mid],
          f"{r2['memory_ids']} vs {[mid]}")
    sheji = ent_id(b, "设计组")
    check("M2 新实体「设计组」已建索引", sheji is not None, f"{list(b.entities)}")

    eids = b.manager.store.get(mid).entity_ids or []
    check("M2 合并后既有实体「青鸾」未被覆盖掉", qing in eids, f"{eids}")
    check("M2 合并后新实体「设计组」也已关联", sheji in eids, f"{eids}")

    # 反向：实体倒排索引里的 memory_ids 不应有悬挂
    dangling = sum(
        1 for e in b.entities.values()
        for m in e["memory_ids"] if b.manager.store.get(m) is None
    )
    check("M2 实体倒排索引无悬挂引用", dangling == 0, f"{dangling} 条悬挂")
    shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# M3 · 类型字段：仅默认值被覆盖
# ======================================================================
def test_m3_type_fields() -> None:
    print("\n[M3] memory_type / cognitive_type：仅原值为默认值时采纳新值")
    mgr = new_mgr(always_merge=True)
    content = "用户偏好深色主题界面，也接受浅色。"
    mid = mgr.store_memory(content, memory_type="general", cognitive_type="world")

    mgr.store_memory(content, memory_type="episodic", cognitive_type="experience")
    rec = mgr.store.get(mid)
    check("M3 原值为默认时采纳新类型",
          rec.memory_type == "episodic" and rec.cognitive_type == "experience",
          f"{rec.memory_type} / {rec.cognitive_type}")

    mgr.store_memory(content, memory_type="general", cognitive_type="world")
    rec = mgr.store.get(mid)
    check("M3 原值已具体时不被默认值冲掉",
          rec.memory_type == "episodic" and rec.cognitive_type == "experience",
          f"{rec.memory_type} / {rec.cognitive_type}")
    check("M3 全程只此一条记忆", mgr.store.count() == 1, f"{mgr.store.count()}")


# ======================================================================
# M4 · 不合并进 superseded / archived
# ======================================================================
def test_m4_no_merge_into_inactive() -> None:
    print("\n[M4] 不合并进 superseded / archived 记忆（改为新建）")
    for state in (MemoryState.SUPERSEDED.value, MemoryState.ARCHIVED.value):
        mgr = new_mgr(always_merge=True)
        content = "用户住在杭州。"
        id1 = mgr.store_memory(content)
        mgr.store.update(id1, state=state)

        id2 = mgr.store_memory(content)
        check(f"M4 不合并进 {state} 记忆（新建）", id2 != id1, f"{id2} == {id1}")
        rec2 = mgr.store.get(id2)
        check(f"M4 新建的那条是 active", rec2 is not None
              and rec2.state == MemoryState.ACTIVE.value,
              f"{rec2.state if rec2 else '无'}")
        check(f"M4 旧记录状态未被改动",
              mgr.store.get(id1).state == state,
              f"{mgr.store.get(id1).state}")
        check(f"M4 总数为 2", mgr.store.count() == 2, f"{mgr.store.count()}")


# ======================================================================
# M5 · D4：update 改 importance 时同步 pin
# ======================================================================
def test_m5_pin_sync_on_update() -> None:
    print("\n[M5] D4：update 改 importance 时同步 pin")
    mgr = new_mgr()
    mid = mgr.store_memory("用户有花生过敏，点餐需避开。", importance=0.5)
    check("M5 新建 importance=0.5 -> pin=False",
          mgr.store.get(mid).pin is False, f"{mgr.store.get(mid).pin}")

    mgr.store.update(mid, importance=0.95)
    check("M5 update 提升 importance 后 pin 自动为 True",
          mgr.store.get(mid).pin is True, f"{mgr.store.get(mid).pin}")
    pinned = sum(1 for r in mgr.store._records if r.pin)
    check("M5 统计口径一致（1 条被钉住）", pinned == 1, f"{pinned}")

    mgr.store.update(mid, importance=0.2)
    check("M5 update 降低 importance 后 pin 自动为 False",
          mgr.store.get(mid).pin is False, f"{mgr.store.get(mid).pin}")

    mgr.store.update(mid, importance=0.9, pin=False)
    check("M5 显式传 pin 时不被自动重算", mgr.store.get(mid).pin is False,
          f"{mgr.store.get(mid).pin}")

    # 只改 pin 不改 importance 时，pin 也不该被反向覆盖
    mgr.store.update(mid, pin=True)
    check("M5 只改 pin 时按传入值生效", mgr.store.get(mid).pin is True,
          f"{mgr.store.get(mid).pin}")


# ======================================================================
# M6 · 合并提升 importance 时 pin 同步（D3 与 D4 的交叉点）
# ======================================================================
def test_m6_merge_pin_sync() -> None:
    print("\n[M6] 合并提升 importance 时 pin 同步")
    mgr = new_mgr(always_merge=True)
    content = "用户偏好每周一上午做周报。"
    mid = mgr.store_memory(content, importance=0.3)
    mid2 = mgr.store_memory(content, importance=0.9)
    check("M6 命中合并", mid2 == mid, f"{mid2} vs {mid}")
    rec = mgr.store.get(mid)
    check("M6 合并取较大 importance", rec.importance == 0.9, f"{rec.importance}")
    check("M6 合并后 pin 同步为 True", rec.pin is True, f"{rec.pin}")


# ======================================================================
# M7 · 合并后 forget 能级联标记两条 L0
# ======================================================================
def test_m7_forget_after_merge() -> None:
    print("\n[M7] 合并后 forget 能级联标记两条 L0 原文")
    b, tmp = new_brain("m7")
    text = "星轨项目由林晚负责，三月开始。"
    r1 = b.ingest(text)
    r2 = b.ingest(text)
    mid = r1["memory_ids"][0]
    check("M7 前置条件：已合并为一条",
          r2["memory_ids"] == [mid] and b.manager.store.count() == 1)

    rep = b.forget(mid, reason="M7 测试")
    marked = set(rep.get("raw_marked") or [])
    check("M7 forget 标记了两条 L0 原文（审计链覆盖完整）",
          marked == {r1["raw_id"], r2["raw_id"]},
          f"{marked} vs {[r1['raw_id'], r2['raw_id']]}")
    check("M7 L0 原文本身仍保留在磁盘（可追溯）",
          len(b.raw_log) == 2, f"{len(b.raw_log)}")
    shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# M8 · 原有合并语义不变
# ======================================================================
def test_m8_merge_semantics_unchanged() -> None:
    print("\n[M8] 原有合并语义不变：content 取更长 / timestamp 取最新 / importance 取最大")
    mgr = new_mgr(always_merge=True)
    short = "短句。"
    long_text = "短句。附带更长的补充说明，用来验证合并时取更长的一方。"
    mid = mgr.store_memory(short, importance=0.2, timestamp=100.0)

    mid2 = mgr.store_memory(long_text, importance=0.6, timestamp=200.0)
    check("M8 恒定相似度下命中合并", mid2 == mid, f"{mid2} vs {mid}")
    rec = mgr.store.get(mid)
    check("M8 content 取更长者", rec.content == long_text, f"{rec.content!r}")
    check("M8 timestamp 取最新", rec.timestamp == 200.0, f"{rec.timestamp}")
    check("M8 importance 取最大", rec.importance == 0.6, f"{rec.importance}")

    mid3 = mgr.store_memory(short, importance=0.1, timestamp=50.0)
    rec = mgr.store.get(mid)
    check("M8 再次合并命中同一条", mid3 == mid, f"{mid3} vs {mid}")
    check("M8 更短内容不覆盖更长内容", rec.content == long_text, f"{rec.content!r}")
    check("M8 更旧 timestamp 不覆盖最新", rec.timestamp == 200.0, f"{rec.timestamp}")
    check("M8 更小 importance 不覆盖最大", rec.importance == 0.6, f"{rec.importance}")


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
