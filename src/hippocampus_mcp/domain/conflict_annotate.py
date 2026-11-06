"""Coexisting-version annotation (read side): flag "suspected different versions of the same fact" within a result set.

Extracted from ``brain/core.py``'s ``_annotate_conflicts``.

**Annotation only**: adds ``conflict_group`` (same group, same number) /
``conflict_with`` (list of counterpart ids) / ``newer`` (the member with the
latest timestamp in the group) to the paired entries. Scores, order, and
counts are never touched — an old version may be a referenced historical fact,
or may be the correct one; which version wins is decided by the authority
(docs), not arbitrated by the retriever on the user's behalf.

Placement rationale: this is **pure derived processing** — the input is a
result set (list[dict]) plus two pure-function capabilities (tokenization,
criterion), the output is in-place labeling + a pair count. Zero I/O, zero
model, zero config reads, matching domain's "may only depend on common"
matrix (``numpy`` is a third-party numeric library already used by
``domain/topic.py``; it is not an in-package dependency).

🔴 The criterion is **not here**: ``is_restatement`` is injected by the caller
(on the brain side it is ``HippoBrain._is_restatement``, which delegates to
``domain.policies.is_restatement``). The write side (``store/supersede.py``)
uses the same callback; both sides share one criterion — if each wrote its own
thresholds, sooner or later you would get the self-contradiction of "judged
the same fact at write time, judged different facts at read time". This module
does not import it in order to preserve the monkeypatch surface of tests
against ``brain._is_restatement``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..common.constants import CONFLICT_ANNOTATE_MAX

__all__ = ["annotate_conflicts"]


def annotate_conflicts(
    items: list[dict[str, Any]],
    tokens_of: Callable[[str], set[str]],
    is_restatement: Callable[..., bool],
    max_items: int = CONFLICT_ANNOTATE_MAX,
) -> int:
    """Annotate coexisting versions within ``items`` and return the pair count (mutates ``items`` in place).

    Args:
        items: Recall-result entries. Each needs ``memory_id`` / ``content``;
            when ``_emb`` (the current model's vector) is missing, that pair
            is excluded from judgment — a BM25-only hit's ``similarity`` is
            always 0.0 and must not be used as a similarity.
        tokens_of: text → content-word set (computed up front, no repeated
            tokenization inside the O(n²)).
        is_restatement: The same-fact criterion; see the signature of
            ``domain.policies.is_restatement``.
        max_items: O(n²) protection cap. Past it, skip outright — **affects
            only annotation, not the recall results themselves**.

    Returns:
        The number of pairs formed.

    Preserved behaviors (aligned item by item with the original; do not
    "fix them in passing"):

    * Entry count ``<2`` or ``>max_items`` ⇒ return 0 directly, not one label
      applied;
    * Already-judged pairs (``b`` already in ``a["conflict_with"]``) are
      skipped, but the **union-find stays connected** (A~B, B~C ⇒ all three
      same family); naive "same group, skip" would miss the third one or
      split the group number;
    * Group numbers are assigned only for **connected components of >=2
      members**, incrementing from 1;
    * ``newer`` is applied only when the group's timestamps are **not all
      equal**, and only on the first one with the maximum timestamp
      (``break``) — with identical timestamps the whole group gets nothing.
    """
    if len(items) < 2 or len(items) > max_items:
        return 0
    toks = {it["memory_id"]: tokens_of(it["content"]) for it in items}
    by_id = {it["memory_id"]: it for it in items}
    # Union-find: the conflict relation is **transitive** (A~B, B~C ⇒ A/B/C
    # same family); naive dedup like "same group, skip" would miss the third
    # one or split the group number in two.
    parent: dict[str, str] = {mid: mid for mid in by_id}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    pairs = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if b["memory_id"] in (a.get("conflict_with") or []):
                continue               # this pair was already judged
            va, vb = a.get("_emb"), b.get("_emb")
            if va is None or vb is None:
                # Missing the current model's vector (BM25-only hit) →
                # undecidable, skip. The "score" must not stand in for
                # similarity: BM25-only similarity is always 0.0.
                continue
            denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-12
            sim = float(va @ vb / denom)
            if not is_restatement(
                sim, a["content"], b["content"], a.get("entity_ids"),
                b.get("entity_ids"), toks[a["memory_id"]], toks[b["memory_id"]],
            ):
                continue
            ra, rb = find(a["memory_id"]), find(b["memory_id"])
            if ra != rb:
                parent[rb] = ra
            for x, other in ((a, b["memory_id"]), (b, a["memory_id"])):
                ids = x.setdefault("conflict_with", [])
                if other not in ids:
                    ids.append(other)
            pairs += 1

    # Wrap-up: number by connected component, and mark ``newer`` on the member
    # with the **latest** timestamp in each family — the classic shape of
    # "took the wrong value" is having taken the old version.
    groups: dict[str, list[dict[str, Any]]] = {}
    for mid in by_id:
        groups.setdefault(find(mid), []).append(by_id[mid])
    gid = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        gid += 1
        for m in members:
            m["conflict_group"] = gid
        stamps = [m.get("timestamp") or 0.0 for m in members]
        if len(set(stamps)) > 1:
            for m in members:
                if (m.get("timestamp") or 0.0) == max(stamps):
                    m["newer"] = True
                    break
    return pairs
