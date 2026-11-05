"""S2 撤销用例：update_tags 的 fail-closed 回滚（undo_tag_ops 方法体）。

为什么放 store/ 而不是 brain：操作对象全是库存记录（``store.get`` /
``store.update``）与 tag_ops 账本（本家 I/O 在 ``store/tag_ops.py``，由包装层
注入），判定逻辑本身零领域知识——"当前值仍等于记录的 after 才改回 before"
是一条纯数据比对规则。

**为什么不是 application**：brain 不许 import application（依赖矩阵：用例
逻辑只从 brain 单向流向 application，不回头），而撤销入口 ``undo_tag_ops``
是 brain 公开方法、由 ``brain.update_tags(undo=...)`` 分发——M6 随
update_tags 一起上收 application 时，本模块整体搬走，接口不用改。

**注入而非 import 的四件事**（测试 monkeypatch 面的保法，见各批「兼容面」
纪律）：账本读取/追加、摘要、覆盖率、落盘都由包装层传入；两个闸门常量
（``sample_max`` / ``confirm_above``）由包装层**调用时**从 brain 命名空间现
取——``core.UPDATE_TAGS_CONFIRM_ABOVE`` 被测试 monkeypatch，快照进本模块会
让闸门断言静默失效。``_now`` 无 monkeypatch 先例，直接 import（同
``source_sig.py`` 例外）。
"""
from __future__ import annotations

from typing import Any, Callable

from ..common.constants import _now



def undo_tag_ops(
    ops: list[dict[str, Any]],
    target: str,
    dry_run: bool,
    confirm: bool,
    *,
    store: Any,
    append_op: Callable[[dict[str, Any]], "str | None"],
    brief: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    coverage: Callable[[], dict[str, Any]],
    save: Callable[[], None],
    sample_max: int,
    confirm_above: int,
) -> dict[str, Any]:
    """撤销一次 update_tags 执行。

    **fail-closed 回滚**：只把"当前值仍等于记录的 after"的条目改回
    before。期间被第三方（另一次 update_tags / update_memory）改过的
    条目一律跳过并进 ``conflicts`` 自曝——盖掉后来的修改比不回滚更糟。

    撤销本身也记帐（op="undo"），所以撤错了可以再撤一次（对称恢复）。

    Args:
        ops: 账本全部条目（包装层已读，坏行已跳过）。
        target: op id（``tagops_3``）或 ``"last"``（最近一次）。
        dry_run: 默认 True —— 只出回滚计划。
        confirm: 回滚条数超过 ``confirm_above`` 时须为真
            （回滚和写入一样能造成大面积破坏，闸门同级）。
        store: 库存实例（``manager.store``），只用 ``get`` / ``update``。
        append_op: 记账回调（包装层 ``_tag_ops_append``），返回 op id。
        brief: 摘要回调（包装层 ``_tag_ops_brief``），给报错带路。
        coverage: 覆盖率回调（brain ``tag_coverage``）。
        save: 落盘回调（brain ``save``），失败由本函数捕获进返回体。
        sample_max: ``conflicts`` / ``gone`` / ``sample`` 的截断上限。
        confirm_above: 回滚条数闸门（与写入侧同常量）。
    """
    if not ops:
        return {"ok": False, "error": "账本是空的：没有可撤销的 update_tags 执行",
                "recent_ops": []}
    want = str(target or "last").strip()
    if want in ("", "last"):
        op = ops[-1]
    else:
        op = next((o for o in ops if str(o.get("op_id")) == want), None)
        if op is None:
            return {"ok": False,
                    "error": f"没有这个 op：{want}",
                    "recent_ops": brief(ops)}
    changes = [c for c in (op.get("changes") or []) if isinstance(c, dict)]
    if not changes:
        return {"ok": False, "error": "这个 op 没有记录任何变更",
                "op_id": op.get("op_id")}

    plan: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    conflicts: list[dict[str, Any]] = []
    gone: list[str] = []
    for c in changes:
        mid = str(c.get("id") or "")
        rec = store.get(mid)
        if rec is None:
            gone.append(mid)
            continue
        before = dict(c.get("before") or {})
        after = dict(c.get("after") or {})
        if dict(rec.tags or {}) != after:
            conflicts.append({
                "id": mid, "expected_after": after,
                "current": dict(rec.tags or {}),
                "note": "当前值与记录的 after 不一致（期间被改过）⇒ 跳过",
            })
            continue
        if before == after:
            continue
        plan.append((mid, before, after))

    payload: dict[str, Any] = {
        "ok": True,
        "action": "undo",
        "dry_run": bool(dry_run),
        "target_op": {
            "op_id": op.get("op_id"),
            "ts": op.get("ts"),
            "filter": op.get("filter"),
            "mode": op.get("mode"),
            "to_change": op.get("to_change"),
            "applied": op.get("applied"),
        },
        "to_revert": len(plan),
        "conflicts": conflicts[:sample_max],
        "conflicts_n": len(conflicts),
        "gone": gone[:sample_max],
        "gone_n": len(gone),
        "sample": [
            {"id": mid, "after": {k: v for k, v in after.items()
                                  if k not in before or before.get(k) != v},
             "before": {k: v for k, v in before.items()
                        if k not in after or after.get(k) != v}}
            for mid, before, after in plan[:sample_max]
        ],
    }
    if dry_run:
        payload["hint"] = ("这是回滚计划，没有写入。确认后用 dry_run=false 执行"
                           + (f"；影响 {len(plan)} 条 > "
                              f"{confirm_above}，还需 confirm=true"
                              if len(plan) > confirm_above else ""))
        return payload
    if len(plan) > confirm_above and not confirm:
        payload.update({
            "ok": False, "applied": False,
            "error": (f"回滚 {len(plan)} 条 > {confirm_above}，"
                      f"拒绝执行：请先 dry_run=true 看计划，确认后加 confirm=true"),
        })
        return payload

    reverted = 0
    failed: list[str] = []
    undo_changes: list[dict[str, Any]] = []
    for mid, before, after in plan:
        try:
            if store.update(mid, tags=dict(before)):
                reverted += 1
                undo_changes.append({"id": mid, "before": after,
                                     "after": before})
            else:
                failed.append(mid)
        except Exception as exc:  # noqa: BLE001 —— 单条失败不该吞掉整批
            failed.append(f"{mid}: {type(exc).__name__}: {exc}")
    payload["applied"] = True
    payload["reverted"] = reverted
    if failed:
        payload["failed"] = failed[:sample_max]
        payload["failed_n"] = len(failed)
    # 记账：撤销也是一次变更（对称记录 ⇒ 可再撤一次）。先记账后落盘，
    # 与 update_tags 同序——内存已改，账本必须留下"改之前是什么"。
    op_id = append_op({
        "op": "undo",
        "target": op.get("op_id"),
        "ts": _now(),
        "to_change": len(plan),
        "applied": reverted,
        "changes": undo_changes,
    })
    payload["undo_op_id"] = op_id
    try:
        save()
        payload["saved"] = True
    except Exception as exc:  # noqa: BLE001
        payload["saved"] = False
        payload["save_error"] = f"{type(exc).__name__}: {exc}"
        payload["hint"] = "标签已在内存回滚但未落盘，请调 save_state 重试"
    payload["coverage_after"] = coverage()
    return payload
