# -*- coding: utf-8 -*-
"""The home-library write boundary for cross-task collaboration: **pure
decisions**, zero I/O.

Semantics in one sentence:

    The home library is writable, other libraries are read-only, and
    cross-library writes go only through the explicit managed channel;
    ``library_use`` only affects addressing (where reads land); **writes are
    always pinned to the home library**.

This module only makes **pure decisions** like "are these two paths the
same library" and "is this write allowed", with the same wording as
:mod:`domain.scope_recall` / :mod:`domain.user_scope`:
unit tests can pin down boundary behavior with zero I/O, and policy changes
never need to touch the registry or disk. Path resolution and other dirty
work stay in the composition root (``server_runtime/composition.py``), which
feeds the results in -- this module does **not import library**
(architectural constraint: domain may depend only on common).

Three pieces of foreign baggage, carried over as-is under the move-only
change-nothing discipline:

* **Normalization uses ``os.path.normcase``** (the registry's same wording,
  see ``library.registry``'s handling of drive-letter case): on Windows
  ``D:\\x`` and ``d:\\x`` are the same library; without normalization you
  get "two brains pointing at the same memories.json".
* **The user library is the only role exemption** (§2.4 transition rule): a
  target library that is the user-level library under the conventional name
  ``user`` ⇒ ``managed-write`` is allowed; all other libraries are
  read-only. What is exempted is the **channel role**, not the tools -- U3's
  sensitivity filtering will later be added on this path.
* **The escape hatch is an input to the three-valued decision**:
  ``escape_open=True`` ⇒ everything is allowed
  (``HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1``, for ops library moves), read from
  config by the composition root and fed in; this module does not know env
  exists (domain must not import config, the P6 ledger discipline).
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "CrossLibraryWriteDenied",
    "MANAGED_WRITE",
    "READ_ONLY",
    "READ_WRITE",
    "USER_LIBRARY_NAME",
    "WRITE_CLASS_TOOLS",
    "boundary_error_message",
    "caller_tool_name",
    "classify_write_permission",
    "library_display",
    "same_library",
]

#: Conventional name of the user-level library. The role exemption is judged
#: by the **roster name**, not by the ``lib_`` id -- ids are randomly derived
#: at library creation and the same user-level library gets a new id in a new
#: deployment, while the name is the convention.
USER_LIBRARY_NAME = "user"

READ_WRITE = "read-write"
MANAGED_WRITE = "managed-write"
READ_ONLY = "read-only"


class CrossLibraryWriteDenied(ValueError):
    """Raised when a write-class tool tries to write a library that is
    neither the home library nor the managed channel (§2.2).

    It inherits ``ValueError`` to stay in the same family as
    :class:`library.LibraryResolutionError` -- both are explicit refusals of
    "this call cannot proceed as-is", and neither is a system fault. The two
    are told apart by **class name** (anyone can see whether it is ambiguous
    resolution or a boundary crossing), so the guard text must go through
    :func:`boundary_error_message` instead of raising a bare one-liner.
    """

#: Single source of truth for write-class tools (re-verified one by one at
#: implementation time).
#:
#: The verification ran over all 46 tools: first the "tool → use-case
#: factory" mapping was exported in full (AST direct read of ``server.py``),
#: then each was checked for whether it really writes to disk at the
#: application layer. Compared with the design draft's 19, **verification
#: added 2**:
#:
#: * ``miss_log`` -- ``clear=True`` goes through ``brain.clear_miss_log()``
#:   (``maintain_service.py:70``); **clearing = deletion**, a real write, not
#:   a read-only view;
#: * ``reset_context`` -- does not delete memories, but changes another
#:   library's brain in-memory sequence state (``manager.reset_context()``),
#:   which with auto_save on may persist to disk along the way. This one was
#:   collapsed to the conservative side: better a rare cross-library
#:   diagnostic call errors out explicitly than leaving a backdoor that can
#:   mutate another library's state ("the design draft did not name it" is
#:   not a reason to let it pass).
#:
#: The remaining 12 / unlisted 25 tools are pure reads or disk-irrelevant
#: (the five ``library_*``, the 11 recall-surface tools, ``get_stats`` /
#: ``brain_stats`` / ``export_memories`` /
#: ``map_get`` / ``map_list`` / ``map_verify`` / ``lexicon_suggest`` /
#: ``lexicon_alias_suggest`` / ``session_start``).
#:
#: Every name in the table must be a tool body that really exists in
#: ``server.py`` -- pinned by the AST assertions in
#: ``tests/test_u4_write_boundary.py`` (so a misspelled name cannot silently
#: strip that tool of its guard, and a removed tool cannot leave a zombie
#: entry behind).
WRITE_CLASS_TOOLS: frozenset[str] = frozenset({
    # -- Write surface, 9 (including two persisting ones) --
    "store_memory", "update_memory", "update_tags",
    "forget_memory", "forget_by_query",
    "ingest", "ingest_batch",
    "save_state", "retry_save",
    # -- Organize surface, 4 --
    "consolidate", "relink_entities", "lexicon_add", "map_set",
    # -- Bootstrap surface, 4 --
    "session_end", "remember_intent", "complete_intent", "report_feedback",
    # -- Maintain surface, 4 (persisting kind; §2.5 says pin these too) --
    "save_brain", "load_state", "miss_log", "reset_context",
})


def same_library(a: str, b: str) -> bool:
    """Whether two paths are the same library (case-insensitive normalized
    comparison).

    ``normpath`` first eats ``./`` and repeated separators; ``normcase`` then
    eats drive-letter/suffix case differences -- the "specified but ambiguous
    resolution always errors" discipline governs resolution, while this one
    governs **whether two already-resolved paths can be treated as the same
    target**. Empty strings do not fall back to a False-value ambiguity:
    empty paths are pairwise unequal (a True from ``"" == ""`` would let an
    empty target slip past the guard).
    """
    if not a or not b:
        return False
    return os.path.normcase(os.path.normpath(a)) == \
        os.path.normcase(os.path.normpath(b))


def classify_write_permission(*, target_path: str, home_path: str,
                              target_name: str = "",
                              escape_open: bool = False) -> str:
    """The permission tier a write to ``target_path`` lands in. Three-valued
    semantics in §2.4.

    Args:
        target_path: Resolved absolute path of the target library (required;
            empty = ambiguous resolution, the caller should have raised
            ``LibraryResolutionError`` further upstream).
        home_path: Absolute path of the home library (the connection's
            default library, ``default_library_path()``).
        target_name: The target library's roster name (for user role
            recognition; may be empty).
        escape_open: Whether the escape hatch is open
            (``HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1``).

    Returns:
        ``read-write`` (home library) / ``managed-write`` (user managed
        channel) / ``read-only`` (any other library ⇒ the caller must error
        out explicitly).

    The decision order must not be rearranged: **escape hatch → home
    library → user role → fallback read-only**. The escape hatch comes first
    because it is the "open everything" ops switch; home before user because
    when the user library happens to be the home library the answer should be
    ``read-write`` -- that is a plain write to this library and must not be
    displayed as the managed channel.
    """
    if escape_open:
        return READ_WRITE
    if same_library(target_path, home_path):
        return READ_WRITE
    if (target_name or "").strip() == USER_LIBRARY_NAME:
        return MANAGED_WRITE
    return READ_ONLY


def library_display(name: str, path: str) -> str:
    """One-line reference for error messages: with a name, include it;
    without one, at least give the absolute path.

    .path is always included -- two libraries sharing a name (different
    paths) is the norm; giving only the name would make "which library is
    the target" ambiguous again.
    """
    name = (name or "").strip()
    return f"{name}（{path}）" if name else (path or "(unknown)")


def boundary_error_message(*, tool: str, target_display: str,
                           home_display: str) -> str:
    """The block message (§2.2: must say clearly which library is the
    target, which is home, and what to do).

    All three are indispensable -- an error that only says "not allowed"
    sends the agent off trying other parameters on a gamble, and a
    cross-library incident costs data written into the wrong library that is
    very hard to recover.
    """
    return (
        f"{tool}'s write was blocked by the home-library write boundary: the target library {target_display}」不是本次"
        f"连接的家库（home，当前为「{home_display}'). \n"
        f"library_use 只影响读落在哪，**写永远钉在家库**；跨库写只走受管"
        f"通道（user 库受管提升 / 目录登记 / 进程内部动作）。\n"
        f"要写这个库：去它自己的工作区开会话（那里的家库就是它），或联系"
        f"用户开逃生门 env HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1（运维搬库用，"
        f"默认关）。"
    )


def caller_tool_name(depth: int = 2) -> str:
    """Get the function name at frame ``depth`` of the call chain (so the
    guard can tell "who is writing").

    ``depth=3`` corresponds to the four-frame chain "tool body → use-case
    factory → guard → this function" (0 = this function), matching design
    §2.2's "guards live in the service factory layer, not spread across each
    tool body": the four write-related factories each call the guard once and
    the 21 tool bodies keep unchanged. Note the guard is **one extra frame**
    -- writing 2 would retreat to the factory name; see the depth assertions
    in ``tests/test_u4_write_boundary.py``.

    Returns ``""`` when unobtainable (the stack is not that deep, e.g. a
    script calling the factory directly) -- the composition root lets it
    pass on that basis, because such a call never came in through the tool
    surface anyway. This fail-open **is watched on its behalf by AST
    assertions** (see the :data:`WRITE_CLASS_TOOLS` comment): as long as
    M4's "single call path" discipline holds (tool bodies call the factory
    directly, no wrappers in between), ``depth=2`` holds identically.
    """
    try:
        return sys._getframe(depth).f_code.co_name
    except ValueError:  # pragma: no cover -- the stack is not that deep
        return ""
