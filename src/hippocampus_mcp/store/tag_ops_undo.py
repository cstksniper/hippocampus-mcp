"""S2 undo use case: update_tags' fail-closed rollback (the undo_tag_ops
method body).

Why store/ and not brain: the objects operated on are all inventory records
(``store.get`` / ``store.update``) and the tag_ops ledger (its home I/O is
in ``store/tag_ops.py``, injected by the wrapper layer); the decision logic
itself carries zero domain knowledge -- "revert to before only if the
current value still equals the recorded after" is a pure data comparison
rule.

**Why not application**: brain must not import application (dependency
matrix: use-case logic flows one way from brain to application, never
back), while the undo entry point ``undo_tag_ops`` is a brain public method
dispatched by ``brain.update_tags(undo=...)`` -- when M6 lifts update_tags
into application, this module moves wholesale and the interface needs no
change.

**The four things injected rather than imported** (preserving the test
monkeypatch surface; see each batch's "compatibility surface" discipline):
ledger read/append, brief, coverage, and save are all passed in by the
wrapper layer; the two gate constants (``sample_max`` / ``confirm_above``)
are fetched fresh from the brain namespace **at call time** by the wrapper
layer -- ``core.UPDATE_TAGS_CONFIRM_ABOVE`` is monkeypatched by tests;
snapshotting it into this module would silently invalidate the gate
assertions. ``_now`` has no monkeypatch precedent, so it is imported
directly (the same exception as ``source_sig.py``).
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
    """Undo one update_tags execution.

    **fail-closed rollback**: only entries whose "current value still equals
    the recorded after" are reverted to before. Entries changed in the
    meantime by a third party (another update_tags / update_memory) are all
    skipped and self-exposed in ``conflicts`` -- overwriting a later change
    is worse than not rolling back.

    The undo itself is also booked (op="undo"), so a mistaken undo can be
    undone once more (symmetric recovery).

    Args:
        ops: all ledger entries (already read by the wrapper layer; bad
            lines skipped).
        target: an op id (``tagops_3``) or ``"last"`` (the most recent).
        dry_run: default True -- produce the rollback plan only.
        confirm: must be true when the rollback count exceeds
            ``confirm_above`` (a rollback can cause damage on the same scale
            as a write; the gates are peers).
        store: the inventory instance (``manager.store``); only ``get`` /
            ``update`` are used.
        append_op: bookkeeping callback (the wrapper layer's
            ``_tag_ops_append``); returns an op id.
        brief: brief callback (the wrapper layer's ``_tag_ops_brief``);
            gives error messages a path to follow.
        coverage: coverage callback (brain ``tag_coverage``).
        save: persist callback (brain ``save``); failures are caught by
            this function into the return body.
        sample_max: truncation cap for ``conflicts`` / ``gone`` / ``sample``.
        confirm_above: rollback-count gate (same constant as the write
            side).
    """
    if not ops:
        return {"ok": False, "error": "the ledger is empty: no reversible update_tags execution",
                "recent_ops": []}
    want = str(target or "last").strip()
    if want in ("", "last"):
        op = ops[-1]
    else:
        op = next((o for o in ops if str(o.get("op_id")) == want), None)
        if op is None:
            return {"ok": False,
                    "error": f"no such op: {want}",
                    "recent_ops": brief(ops)}
    changes = [c for c in (op.get("changes") or []) if isinstance(c, dict)]
    if not changes:
        return {"ok": False, "error": "this op recorded no changes",
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
                "note": "current value differs from the recorded after (changed in between) => skipped",
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
        payload["hint"] = ("this is a rollback plan; nothing was written. Execute with dry_run=false once confirmed"
                           + (f"; it would affect {len(plan)} entries > "
                              f"{confirm_above}; confirm=true is also required"
                              if len(plan) > confirm_above else ""))
        return payload
    if len(plan) > confirm_above and not confirm:
        payload.update({
            "ok": False, "applied": False,
            "error": (f"rollback would affect {len(plan)} entries > {confirm_above}，"
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
        except Exception as exc:  # noqa: BLE001 -- one entry failing must not swallow the whole batch
            failed.append(f"{mid}: {type(exc).__name__}: {exc}")
    payload["applied"] = True
    payload["reverted"] = reverted
    if failed:
        payload["failed"] = failed[:sample_max]
        payload["failed_n"] = len(failed)
    # Bookkeeping: the undo is itself a change (symmetric recording => it
    # can be undone once more). Book first, persist after, same order as
    # update_tags -- memory is already changed; the ledger must record
    # "what it was before the change".
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
        payload["hint"] = "tags were rolled back in memory but not persisted; call save_state and retry"
    payload["coverage_after"] = coverage()
    return payload
