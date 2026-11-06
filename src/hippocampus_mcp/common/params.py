# -*- coding: utf-8 -*-
"""Pure parsing helpers for MCP tool parameters (L0: pure stdlib, zero
I/O; anyone may import).

Extracted from server.py (parsing pieces shared by the tool layer and
the use-case layer): these functions are needed by both the tool layer
(L4) and the use-case layer (L3), and L3 is forbidden by architecture
discipline from importing L4 -- putting them on the L0 shared base is
the only position that creates no cycle. The three functions were
**moved verbatim** (docstrings included); behavior unchanged;
server.py switched to alias imports, with no call site touched.

The two ``parse_str_list`` functions are **deliberately coexisting
strictness tiers** (each has its own owner; do not merge them again):

* :func:`parse_str_list` (strict) -- bad JSON **raises explicitly**
  (``(None, err)`` tuple). update_tags' clear_keys / missing_keys /
  memory_ids use it: a silent fallback would make the caller believe
  "there are no keys to delete", turning the whole operation into one
  quiet no-op.
* :func:`parse_str_list_lenient` (lenient) -- accepts JSON or commas;
  on parse failure **falls back to comma splitting**. knowledge_topic's
  evidence/topics and the three lexicon tools (lexicon_add /
  lexicon_suggest / lexicon_alias_suggest) use it: those parameters
  mean "casually pass a few words", and silence would present as "the
  parameter was passed but did nothing", far harder to debug than an
  outright error.

One deliberate difference from the server.py version: ``check_tag_keys``
**is not here** -- it must call
``domain.tagging.validate_filter_keys`` (L2), and common may not depend
on domain (dependency matrix), so it lives in
``application/tags.py``.
"""
from __future__ import annotations

import json
from typing import Any


def parse_tags_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the JSON-object string passed in by MCP into a dict.

    MCP tool parameters use simple types; tags are passed as a JSON
    string. A parse failure must **raise explicitly**, never silently
    become empty -- otherwise the host believes tags were attached while
    in fact none were, only to discover at filter time that nothing is
    found.
    """
    text = (raw or "").strip()
    if not text:
        return None, None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"is not valid JSON: {exc}"
    if not isinstance(loaded, dict):
        return None, 'must be a JSON object, e.g. {"ch": 43}'
    return loaded, None


def parse_str_list(raw: str) -> tuple[list[str] | None, str | None]:
    """Parse the JSON-array string passed in by MCP into a list of
    strings (strict version).

    Same discipline as :func:`parse_tags_json`: a parse failure **raises
    explicitly**, never silently becomes empty -- an empty list would
    make the caller believe "there are no keys to delete", turning the
    whole operation into one quiet no-op.
    """
    text = (raw or "").strip()
    if not text:
        return None, None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"is not valid JSON: {exc}"
    if not isinstance(loaded, list):
        return None, 'must be a JSON array, e.g. ["domain", "ch"]'
    return [str(x) for x in loaded], None


def parse_str_list_lenient(value: Any) -> list[str]:
    """Parse a "list-like parameter" into a list of strings (JSON array
    or comma-separated, both accepted).

    On the wire such parameters are always strings: ``["同安","顾父"]``
    and ``"同安, 顾父"`` are equivalent.
    A parse failure **falls back to comma splitting** instead of silently
    becoming empty -- silence presents as "the parameter was passed but
    did nothing", far harder to debug than an outright error.
    """
    if not isinstance(value, str):
        # direct Python calls (tests/scripts) also accept sequences
        if isinstance(value, (list, tuple, set)):
            return [str(x).strip() for x in value if str(x).strip()]
        return []
    s = value.strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            return [t.strip() for t in s.strip("[]").split(",") if t.strip()]
    return [t.strip() for t in s.split(",") if t.strip()]


def parse_tagops_args(tags: str, clear_keys: str, missing_keys: str,
                      memory_ids: str) -> tuple[dict[str, Any] | None, str | None]:
    """update_tags' four-segment JSON protocol parsing (behavior
    bit-identical).

    The four segments (tags / clear_keys /
    missing_keys / memory_ids) are parsed strictly one by one, **stopping
    at the first error** -- error messages carry a field-name prefix
    (``"tags 解析失败：..."``), verbatim identical to the tool body
    before the helper extraction.

    Why the four segments must stay in the adapter layer rather than
    sinking into a service:
    these parses currently run **before** library resolution
    (``_brain_for``); 「bad library name + bad JSON」 must report the
    syntax error first; and the error body carries **no** library first
    key (the library is not yet resolved). Sinking into a service (where
    the factory call itself resolves the library) would flip both the
    error order and the envelope shape -- that is a product decision,
    not a refactor by-product.

    For the strictness-tier division see the module docstring: all four
    segments here go through the strict version
    (:func:`parse_str_list`), bad JSON raises explicitly, and names are
    not mixed with the lenient version.
    """
    parsed, err = parse_tags_json(tags)
    if err:
        return None, f"tags parse failed: {err}"
    keys, err = parse_str_list(clear_keys)
    if err:
        return None, f"clear_keys parse failed: {err}"
    mkeys, err = parse_str_list(missing_keys)
    if err:
        return None, f"missing_keys parse failed: {err}"
    ids, err = parse_str_list(memory_ids)
    if err:
        return None, f"memory_ids parse failed: {err}"
    return {"tags": parsed, "clear_keys": keys,
            "missing_keys": mkeys, "memory_ids": ids}, None


def filter_applied(parsed_tags: dict[str, Any] | None,
                   matched: int | None,
                   degraded: dict[str, Any] | None = None) -> dict[str, Any]:
    """``filter_applied`` self-report: the tag domain used this time +
    the candidate count after filtering.

    Reason to exist: when a ``filter_tags`` key is misspelled / its value
    mismatches, retrieval **silently** returns empty -- no error, no
    hint; the caller can only conclude "the library really has nothing".
    Explicitly self-reporting the candidate count:

        ``tags=None``    no filter set (whole-library retrieval)
        ``matched=None`` same as above (with no filter there is no
                         "candidate count" concept)
        ``matched=0``    the filter **matched no entries** (key/value
                         mismatch, silent failure)
        ``matched=N``    narrowed to N entries before retrieval

    ``matched=0`` is thereby distinguishable from "retrieval came back
    empty while the library does have content". The candidate count is
    supplied by the caller (each path has already computed the
    whitelist, avoiding a duplicate traversal).

    The ``degraded`` info is provided by
    :func:`hippocampus_mcp.application.tags.check_tag_keys` on soft
    degradation -- present only when the schema explicitly enabled
    ``filter_degrade``; off by default; see
    ``tagging.validate_filter_keys``.
    """
    out: dict[str, Any] = {
        "tags": dict(parsed_tags) if parsed_tags else None,
        "matched": matched,
    }
    if degraded:
        out["degraded"] = True
        out["degraded_from"] = degraded.get("degraded_from")
        out["dropped_keys"] = degraded.get("dropped_keys")
    return out
