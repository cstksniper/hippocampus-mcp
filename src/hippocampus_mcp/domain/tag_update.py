# -*- coding: utf-8 -*-
"""The plan / echo / write three sections of ``update_tags`` (extracted
inline from ``brain.core``).

Why it is worth moving
======================

``update_tags`` is the **only channel** for backfilling tags on existing
records (``update_memory`` does not accept ``tags``), and it is also the bulk
tool that once touched 2,010 records on a real library. It was originally one
140-line entry method with four kinds of logic tangled together:

1. **Input normalization and validation** (mode case/invalid values, tags and
   clear_keys both empty) -- a pure function, yet testing it required
   constructing a whole HippoBrain + embedder;
2. **Plan building** (before/after computed per record, unchanged values
   skipped) -- the entirety of idempotency semantics;
3. **Echo assembly** (the payload's eleven keys + sample projection + the
   prompts of the dry_run/confirm two gates) -- pure string and number
   assembly;
4. **Write loop** (per-record ``store.update``, map entries refresh
   timestamp, single-record failures do not swallow the batch) -- tangled with
   the L1 storage port and the clock.

After moving into this module, all four sections unit-test in milliseconds;
``brain.core.update_tags`` keeps only a thin shell (dispatch + selection + cap
+ ledger + persist + coverage).

Guarantees that the move is **zero behavior change**:
1. The four sections are copied verbatim, with only ``self.`` / module-level
   constants replaced by injected parameters;
2. The two gate constants (``UPDATE_TAGS_SAMPLE_MAX`` /
   ``UPDATE_TAGS_CONFIRM_ABOVE``) are **not imported**; the shell fetches them
   live from the core namespace at call time and passes them in -- the J1 test
   monkeypatches ``core.UPDATE_TAGS_CONFIRM_ABOVE``, and snapshotting them
   into this module would make the gate silently stop working (same
   discipline as ``store.tag_ops_undo``);
3. The clock is injected as a **callable** (``now=_now``): it is not called at
   all when there are no map entries; passing a value would still pay the cost
   when the switch is off (and could not freeze time in tests);
4. The parity probe (``_probe_out/m5_batch20_parity.py``) runs the HEAD old
   ``update_tags`` (exec'd into the ``brain_core.__dict__`` body, hung off a
   subclass) and the new shell against two seeded libraries, comparing 15
   scenarios' complete response bodies + library state + ledger files field
   by field; DRIFT must be 0.

Dependency direction: this module has **zero imports** (not even common is
used) -- record objects are used duck-typed (``.id`` / ``.tags``), and
storage/clock/constants are all injected, satisfying the "domain -> common
only" layering discipline (stricter than the discipline, here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Plan entry: (record id, tags before, tags after). before/after are both
# **new dicts** (``dict(...)`` copies) -- the ledger and sample must hold them
# long-term.
PlanEntry = tuple[str, dict[str, Any], dict[str, Any]]


def normalize_request(
    mode: str,
    tags: dict[str, Any] | None,
    clear_keys: list[str] | None,
) -> tuple[str, dict[str, Any], list[str], dict[str, Any] | None]:
    """Input normalization and validation (the first three sections of the
    original ``core.update_tags``, preserved verbatim).

    Returns:
        ``(mode, new_tags, ckeys, err)``. When ``err`` is non-None the shell
        returns it directly (explicit error, never a silent no-op run); when
        valid, the first three items are the normalized values.

    Three preserved semantics:

    * after normalizing mode with ``strip().lower()``, only ``merge`` /
      ``replace`` are accepted;
    * ``clear_keys`` filters out empty-string keys (``["", "章"]`` ->
      ``["章"]``) -- ``pop`` with an empty key name is a silent no-op, but it
      would pollute the ``clear_keys`` in the echo;
    * tags and clear_keys both empty = "nothing to change", an explicit error.
    """
    mode = (mode or "merge").strip().lower()
    if mode not in ("merge", "replace"):
        return mode, {}, [], {
            "ok": False, "error": f"invalid mode: {mode!r}",
            "allowed_modes": ["merge", "replace"]}
    new_tags = dict(tags or {})
    ckeys = [str(k) for k in (clear_keys or []) if str(k)]
    if not new_tags and not ckeys:
        return mode, new_tags, ckeys, {
            "ok": False,
            "error": "tags and clear_keys cannot both be empty (nothing to change)"}
    return mode, new_tags, ckeys, None


def build_plan(
    recs: list[Any],
    *,
    mode: str,
    new_tags: dict[str, Any],
    ckeys: list[str],
) -> list[PlanEntry]:
    """Compute before/after per record; unchanged values stay out of the plan
    (the entirety of idempotency semantics).

    **Why "unchanged values are not counted" is the precondition of
    idempotency**: the original requirement assumed merge semantics alone
    would bring idempotency -- on a second run every key already exists, but
    **the values may differ** (the first run wrote ``域=F``, the host manually
    changed it to ``域=Z`` midway, and a re-run would change it back to F).
    Only when the whole before/after pair is equal is the record skipped, so
    that a re-run yields ``to_change=0``.

    ``replace`` mode **ignores before**: ``after = dict(new_tags)`` then delete
    clear_keys -- clearing old keys wholesale is the definition of replace.
    """
    plan: list[PlanEntry] = []
    for rec in recs:
        before = dict(rec.tags or {})
        if mode == "replace":
            after = dict(new_tags)
        else:
            after = dict(before)
            after.update(new_tags)
        for k in ckeys:
            after.pop(k, None)
        if after == before:
            continue  # values unchanged -> idempotent: not counted here on a re-run
        plan.append((rec.id, before, after))
    return plan


def project_sample(
    plan: list[PlanEntry],
    *,
    new_tags: dict[str, Any],
    ckeys: list[str],
    sample_max: int,
) -> list[dict[str, Any]]:
    """Sample projection: only the **before/after** of the first
    ``sample_max`` entries.

    Only the keys **involved this time** are projected (the keys of
    ``new_tags`` + ``ckeys``) -- a memory may carry dozens of historical tags;
    echoing them all would explode the response body, while the caller only
    needs to verify what changed this time.
    """
    return [
        {
            "id": rid,
            "before": {k: v for k, v in before.items()
                       if k in new_tags or k in ckeys},
            "after": {k: v for k, v in after.items()
                      if k in new_tags or k in ckeys},
        }
        for rid, before, after in plan[:sample_max]
    ]


def build_payload(
    *,
    dry_run: bool,
    confirm: bool,
    mode: str,
    new_tags: dict[str, Any],
    ckeys: list[str],
    filt: dict[str, Any],
    matched: int,
    considered: int,
    capped: bool,
    to_change: int,
    plan: list[PlanEntry],
    sample_max: int,
    confirm_above: int,
) -> tuple[dict[str, Any], bool]:
    """Assemble the plan/execution echo body and enforce the dry_run and
    confirm gates.

    Returns:
        ``(payload, proceed)``. When ``proceed=False`` the shell returns the
        payload directly (a dry_run plan, or a gate refusal); only ``True``
        enters the write loop.

    Two gates (both added after stepping on the same kind of pit):

    * **Default ``dry_run=True``**: bulk tag changes have no transaction; a
      wrong run raises no error and only shows up at filter time as "nothing
      found";
    * **Above ``confirm_above`` entries, ``confirm=True`` is mandatory**: a
      real library was touched 2,010 records in one go; one wrong character in
      the filter is 2,010 pieces of garbage tags.

    The gate constants are fetched live by the shell from the core namespace
    at call time and passed in (tests monkeypatch
    ``core.UPDATE_TAGS_CONFIRM_ABOVE``) -- this module does not import them.
    """
    payload: dict[str, Any] = {
        "ok": True,
        "dry_run": bool(dry_run),
        "mode": mode,
        "tags": new_tags,
        "clear_keys": ckeys,
        "filter": filt,
        "matched": matched,
        "considered": considered,
        "capped": capped,
        "to_change": to_change,
        "skipped_unchanged": considered - to_change,
        "sample": project_sample(plan, new_tags=new_tags, ckeys=ckeys,
                                 sample_max=sample_max),
    }
    payload["reverse_hint"] = (
        "Reversible: the server keeps a full before/after ledger; the result carries op_id. "
        "update_tags(undo=op_id or \"last\", dry_run=false) performs a fail-closed "
        "rollback (restores before only if the current value still equals after; changed entries are skipped and self-reported)"
    )

    if dry_run:
        payload["hint"] = (
            "This is a plan; nothing was written. Execute with dry_run=false once confirmed"
            + (f"; it would affect {to_change} entries > {confirm_above}, "
               f"confirm=true is also required" if to_change > confirm_above else "")
        )
        return payload, False

    if to_change > confirm_above and not confirm:
        payload.update({
            "ok": False,
            "applied": False,
            "error": (f"would affect {to_change} entries > {confirm_above}; "
                      f"refusing to execute: run dry_run=true first to inspect the sample, then add confirm=true"),
        })
        return payload, False
    return payload, True


@dataclass(frozen=True)
class ApplyPorts:
    """All dependencies of the write loop (injected by the brain side; domain
    imports no upper layer).

    Attributes:
        store: the L1 storage port. Only one surface is used,
            ``update(rid, **fields)`` (a truthy return counts as success;
            falsy/raised exceptions both go into ``failed``).
        map_name_key: the map reserved tag key
            (``map.constants.MAP_NAME_KEY``). Changing tags on an entry with
            this key = a maintenance action, so ``timestamp`` must be
            refreshed along with it -- the index row / gate four computes "N
            days since update" from timestamp; without the refresh, an agent
            that dutifully "changes status via update_tags" every time would
            still see the index row flagged ⚠️. It lives in the map package,
            invisible to domain, hence injected.
        now: the clock (brain's ``_now``). Injected as a **callable** -- not
            called at all when there are no map entries.
    """
    store: Any
    map_name_key: str
    now: Callable[[], float]


@dataclass(frozen=True)
class ApplyResult:
    """The three outputs of one write loop (raw material for the execution echo body)."""
    changed: int = 0
    failed: list[str] = field(default_factory=list)
    applied_changes: list[dict[str, Any]] = field(default_factory=list)


def apply_plan(plan: list[PlanEntry], ports: ApplyPorts) -> ApplyResult:
    """Run ``store.update`` per record; a single failure does not swallow the
    batch.

    Three preserved semantics:

    * map entries (``map_name_key in after``) refresh ``timestamp`` along the
      way -- a good side effect: map entries are from then on never archived
      by consolidate as dormant memories;
    * a falsy return from ``store.update`` = that record did not land, goes
      into ``failed`` (not raising still counts as failure);
    * an exception on a single record only records ``"id: ExcType: msg"`` into
      ``failed`` and keeps going -- for a bulk tool, quitting halfway is harder
      to clean up than finishing with errors recorded (``noqa: BLE001`` same
      as the original site).
    """
    changed = 0
    failed: list[str] = []
    applied_changes: list[dict[str, Any]] = []
    for rid, before, after in plan:
        try:
            fields = {"tags": dict(after)}
            if ports.map_name_key in after:
                fields["timestamp"] = ports.now()
            if ports.store.update(rid, **fields):
                changed += 1
                applied_changes.append({"id": rid, "before": before,
                                        "after": after})
            else:
                failed.append(rid)
        except Exception as exc:  # noqa: BLE001 -- a single-record failure must not swallow the whole batch
            failed.append(f"{rid}: {type(exc).__name__}: {exc}")
    return ApplyResult(changed=changed, failed=failed,
                       applied_changes=applied_changes)
