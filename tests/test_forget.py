"""forget cascade-deletion regression tests.

Verifies that "delete one memory" is cleaned out of every derived structure.
This is the foundation of error correction: if any layer keeps a copy, the
deleted wrong information still enters context via the profile or entity state.

Coverage:
  1. profile layer keeps no entries derived from that memory (incl. after restart)
  2. entity inverted index removes references and recomputes mention_count, no danglers
  3. an emptied entity is removed and stripped from other memories' entity_ids (reverse danglers)
  4. the old memory superseded by the deleted one is restored to active
  5. no-longer-referenced sources in the L0 raw layer are marked forgotten
  6. forget_by_query only previews by default, deletes only on explicit execute
  7. tombstones persist, so deletions stay traceable

Usage:
    <py311> tests/test_forget.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from hippocampus_mcp.brain import HippoBrain
from hippocampus_mcp.store.memory import MemoryState

MODEL = os.environ.get("HIPPOCAMPUS_EMB_MODEL", "")
EMBEDDER_BACKEND = "sentence_transformer" if MODEL else "hash"


def _brain_kwargs() -> dict:
    """Defaults to the hash backend so this suite needs no model download; real embeddings only when a model is set explicitly."""
    return {"embedder_model": MODEL} if MODEL else {"embedder_backend": "hash"}


_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    line = f"  [{tag}] {name}"
    if detail and not cond:
        line += f"\n         → {detail}"
    print(line)


def new_brain() -> tuple[HippoBrain, str]:
    tmp = tempfile.mkdtemp(prefix="forget_test_")
    return HippoBrain(persist_dir=tmp, **_brain_kwargs()), tmp


def profile_texts(brain: HippoBrain) -> list[str]:
    return [i["text"] for items in brain.profile.values() for i in items]


def dangling(brain: HippoBrain) -> int:
    n = 0
    for ent in brain.entities.values():
        n += sum(1 for m in ent["memory_ids"] if brain.manager.store.get(m) is None)
    return n


# ---------------------------------------------------------------- cases 1 & 4 & 7
def test_cascade_cleanup() -> None:
    print("\n[1] 级联清理：画像 / 实体索引 / L0 来源 / 墓碑")
    b, tmp = new_brain()
    for t in ["星轨项目由林晚负责。", "星轨项目计划三月上线。", "星轨项目将使用 React 框架。"]:
        b.ingest(t, speaker="user", entities=["星轨"])
    b.session_end(summary="讨论了星轨项目安排")

    ent = b.entities.get("星轨")
    check("mention_count semantics correct (3 memories -> 3, not the old inflated value)",
          ent is not None and ent["mention_count"] == 3,
          f"实际 {ent['mention_count'] if ent else '实体不存在'}")

    before = profile_texts(b)
    check("profile generated with 3 entries", len(before) == 3, f"实际 {before}")
    # evidence must be exact to a single memory, else deleting one would hit other entries of the same entity
    gran = [len(i["evidence"]) for items in b.profile.values() for i in items]
    check("profile evidence granularity = single memory", all(g == 1 for g in gran), f"实际 {gran}")

    victim = next(r for r in b.manager.store._records if "三月上线" in r.content)
    rep = b.forget(victim.id, reason="用户纠正：上线时间记错了")
    check("forget reports success", rep.get("ok") is True, str(rep))

    after = profile_texts(b)
    check("wrong entry gone from every profile slot", not any("三月上线" in t for t in after),
          f"仍残留 {after}")
    check("other profile entries of the same entity unharmed",
          any("React" in t for t in after) and any("林晚" in t for t in after),
          f"剩余 {after}")
    check("entity inverted index has no dangling references", dangling(b) == 0, f"悬挂 {dangling(b)} 条")
    check("mention_count recomputed to 2", b.entities["星轨"]["mention_count"] == 2,
          str(b.entities["星轨"]["mention_count"]))
    check("orphaned L0 source marked forgotten", len(rep["raw_marked"]) == 1,
          str(rep["raw_marked"]))
    check("tombstone written and carries a reason", b.tombstones and b.tombstones[-1]["reason"].startswith("用户纠正"),
          str(b.tombstones[-1:] or "空"))

    b2 = HippoBrain(persist_dir=tmp, **_brain_kwargs())
    check("no profile residue after restart",
          not any("三月上线" in t for t in profile_texts(b2)),
          f"残留 {profile_texts(b2)}")
    check("tombstone still there after restart (deletion traceable)",
          len(b2.tombstones) == 1 and b2.tombstones[0]["content"] == victim.content,
          str(b2.tombstones))
    check("entity index has no danglers after restart", dangling(b2) == 0, f"悬挂 {dangling(b2)} 条")


# ---------------------------------------------------------------- case 3
def test_empty_entity_removed() -> None:
    print("\n[2] 实体清空后被移除，并剔除其他记忆上的反向悬挂")
    b, _ = new_brain()
    b.ingest("银河号项目已经立项了。", speaker="user", entities=["银河号"])
    b.ingest("潮汐项目是我负责的。", speaker="user", entities=["潮汐"])
    b.ingest("潮汐项目进展顺利，准备验收。", speaker="user", entities=["潮汐"])

    # make one memory carry two entities, to verify reverse stripping
    mixed = b.ingest("银河号和潮汐项目由同一个团队负责。", speaker="user",
                     entities=["银河号", "潮汐"])
    check("Yinhe entity exists", "银河号" in b.entities, str(list(b.entities)))

    # delete the two memories containing 「银河号」
    for mid in mixed["memory_ids"]:
        b.forget(mid, reason="测试：清除银河号相关")
    lone = [r for r in b.manager.store._records if "银河号" in r.content]
    for r in lone:
        b.forget(r.id, reason="测试：清除银河号相关")

    check("Yinhe entity removed once its memories are gone", "银河号" not in b.entities,
          f"仍存在 {list(b.entities)}")
    check("Chaoxi entity retained", "潮汐" in b.entities, str(list(b.entities)))
    check("no dangling entity references library-wide", dangling(b) == 0, f"悬挂 {dangling(b)}")

    ghost = [e for e in b.entities.values() for _ in [0]]
    ghost_ids = {e["id"] for e in ghost}
    stray = [
        r.content for r in b.manager.store._records
        if any(e not in ghost_ids for e in (r.entity_ids or []))
    ]
    check("other memories' entity_ids stripped of the dead entity", not stray, f"残留 {stray}")


# ---------------------------------------------------------------- case 4
def test_restore_superseded() -> None:
    print("\n[3] 删除取代者后，被取代的旧记忆恢复活跃")
    b, _ = new_brain()
    b.ingest("我们主要做家居品类。", speaker="user", entities=["品类"])
    b.ingest("我们不再做家居品类了，现在改成宠物用品。", speaker="user",
             entities=["品类"])

    superseded = [r for r in b.manager.store._records
                  if r.state == MemoryState.SUPERSEDED.value]
    if not superseded:
        print("  [SKIP] 本例未触发 supersede（相似度/实词重叠未达阈值），跳过")
        return
    old = superseded[0]
    new_id = old.superseded_by
    check("old memory was marked superseded", bool(new_id), str(new_id))

    rep = b.forget(new_id, reason="测试：取代者是错的")
    restored = b.manager.store.get(old.id)
    check("superseded memory restored to active", restored.state == MemoryState.ACTIVE.value,
          f"实际 {restored.state}")
    check("back-reference cleared", restored.superseded_by is None,
          f"实际 {restored.superseded_by}")
    check("restoration reported", old.id in rep["restored_superseded"],
          str(rep["restored_superseded"]))


# ---------------------------------------------------------------- case 6
def test_forget_by_query() -> None:
    print("\n[4] forget_by_query：默认预览，显式执行才删")
    b, _ = new_brain()
    for t in ["星轨项目计划三月上线。", "星轨项目预算为十万。", "星轨项目由林晚负责。"]:
        b.ingest(t, speaker="user", entities=["星轨"])

    preview = b.forget_by_query("星轨项目计划三月上线。", top_k=3)
    check("dry_run default returns candidates", preview.get("dry_run") is True, str(preview)[:100])
    check("preview deletes nothing", b.manager.store.count() == 3,
          f"剩余 {b.manager.store.count()}")
    check("candidates carry memory_id and content",
          bool(preview["candidates"]) and "memory_id" in preview["candidates"][0],
          str(preview["candidates"][:1]))

    target = preview["candidates"][0]["memory_id"]
    done = b.forget_by_query("星轨项目计划三月上线。", top_k=1, dry_run=False)
    check("explicit execute really deletes", b.manager.store.count() == 2,
          f"剩余 {b.manager.store.count()}")
    check("the deleted one is the target record", b.manager.store.get(target) is None, target)
    check("removed count reported", done.get("removed_count") == 1, str(done.get("removed_count")))


# ---------------------------------------------------------------- case 5
def test_missing_id() -> None:
    print("\n[5] 边界：不存在的 ID 不抛异常")
    b, _ = new_brain()
    rep = b.forget("不存在的-id")
    check("returns ok=False instead of raising", rep.get("ok") is False, str(rep))
    check("tombstones not polluted", not b.tombstones, str(b.tombstones))


def main() -> int:
    print(f"forget 级联删除回归测试  |  嵌入: {MODEL or EMBEDDER_BACKEND}")
    test_cascade_cleanup()
    test_empty_entity_removed()
    test_restore_superseded()
    test_forget_by_query()
    test_missing_id()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 52}")
    print(f"结果：{passed}/{total} 通过")
    if passed != total:
        print("失败项：")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}  → {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
