# -*- coding: utf-8 -*-
"""Write-side quarantine (D2a: pending-review queue criteria and audit event
construction for the user library's managed write channel).

Pure-function layer of decision ① (〔review report 20261028〕P1-1): from a
non-home workspace, managed-write to the user library is **downgraded to
pending review** -- the write carries a quarantine marker, does not enter
the recall pool, does not enter the ``user_scope`` slot; after the
drift_audit report is reviewed it is explicitly promoted or deleted.

Why this module exists: ``write_boundary.classify_write_permission`` grants
``MANAGED_WRITE`` to any target library named ``user`` (U4's managed-channel
exemption, originally meant for the U3 promotion flow), but a bare
``store_memory(library="user")`` write goes through **the same exemption** --
an injector from an untrusted repository could directly write a forged "user
preference" and, via the ``user_scope`` slot, inject it into every
subsequent workspace session. This module separates "direct writes on the
exempted channel" from "managed promotion": **the exemption stays; direct
writes are downgraded to pending review**.

All pure functions: no env reads, no I/O, no imports from other modules in
this package -- the three criteria inputs (target name / is home library /
escape hatch) are computed by the caller (composition-root factory) and
passed in.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

__all__ = [
    "USER_LIBRARY_NAME",
    "ACCESS_LOG_FILENAME",
    "should_quarantine",
    "sensitivity_label",
    "build_access_event",
]

#: Conventional name of the user-level library (same name, same origin as
#: write_boundary's user role exemption -- decision wording: the exempted
#: **channel** is unchanged; **direct writes** on that channel are downgraded
#: to pending review).
USER_LIBRARY_NAME = "user"

#: Sidecar filename for write-side audit events (lives in the user library
#: directory, same-level same-pattern precedent as directory.json; JSONL
#: append, one event per line).
ACCESS_LOG_FILENAME = "access_log.jsonl"


def should_quarantine(
    *, target_name: str, same_as_home: bool, escape_open: bool
) -> bool:
    """Pending-review criterion (single source of truth, shared by the
    composition-root factory and tests).

    True if and only if: the target is the user library (role-exempt
    channel) **and** the target is not the home library **and** the escape
    hatch is closed. The home library = the daily-assistant host of the user
    library (decision ③'s full tier) writing to its own home is not
    quarantined; the escape hatch (``HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1``) is
    an explicit ops window -- migration batch operations must not land in
    the pending-review queue.
    """
    return (
        (target_name or "").strip() == USER_LIBRARY_NAME
        and not same_as_home
        and not escape_open
    )


def sensitivity_label(
    classification: dict[str, Any], *, personal_default: bool = True
) -> str:
    """Map the D1 classification result to the card's ``sensitivity`` string
    label.

    Hit categories → ``sensitive:cat1+cat2`` (joined with ``+``, **sorted
    internally**: the label is independent of input order; the same content
    always gets the same label);
    no hits → ``personal`` (better to over-mark than under-mark: automatic
    ingestion into the user library defaults to personal when no category
    hits; public is reserved for explicit declaration -- 〔review〕P2-3
    adopted).
    ``personal_default=False`` leaves room for a future "explicitly declared
    public" channel; current callers do not pass it.
    """
    cats = sorted(classification.get("categories") or [])
    if cats:
        return "sensitive:" + "+".join(cats)
    return "personal" if personal_default else ""


def build_access_event(
    *,
    kind: str,
    tool: str,
    home: str,
    target_name: str,
    target_path: str,
    quarantined: bool,
    sensitivity: str = "",
    content: str = "",
    ts: float | None = None,
) -> dict[str, Any]:
    """Build one write-side audit event (one line of access_log.jsonl).

    **Never persists the raw text**: content leaves only a 16-char
    fingerprint (first 16 hex of sha256) -- the audit wants "who touched the
    user library when, and whether it was quarantined"; plaintext review goes
    through the memory card itself.
    ``kind``: ``store_memory`` / ``ingest`` / ``ingest_batch`` /
    ``quarantine_clear``.
    """
    digest = hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:16]
    return {
        "ts": float(time.time() if ts is None else ts),
        "kind": kind,
        "tool": tool or "",
        "home": home or "",
        "target": target_name or "",
        "target_path": target_path or "",
        "quarantined": bool(quarantined),
        "sensitivity": sensitivity or "",
        "content_sha16": digest,
    }
