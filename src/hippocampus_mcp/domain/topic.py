# -*- coding: utf-8 -*-
"""Topic layer (L2T): converge flat L1 facts into topics, producing
higher-level statements that can be referred back to.

Why this module exists
======================

L1 is the main retrieval battlefield, but it is organized only as "flat +
entity inverted index" -- no tool can answer "what knowledge points does
this library have". L2 has only one session-level summary and **stays out
of the retrieval path** (it goes dormant once written), and the L3 profile
is hot entries, not knowledge; the "RAPTOR recursive aggregation not
implemented" line in ``ARCHITECTURE.md`` refers to exactly this layer.

This module does only the **deterministic** part: clustering, membership
maintenance, extractive representative sentences, statistics. Real
"summarization" (writing N facts into one higher-level conclusion) needs
intelligence, and the production path not embedding an LLM is this
project's boundary -- that part is done by the host and persisted through
the write-back channel; this module reserves the ``summary`` /
``summary_kind`` / ``summary_evidence`` fields for it.

Hard constraints (read before changing anything)
================================================

* **Never enters the main retrieval path**. L2's ``RELEVANCE_HARD_FLOOR``
  exists precisely to prevent "a summary layer similar to everything
  dominating the rankings"; V9f entity-graph diffusion dropped F1 from .311
  to .200. The topic layer's exits are ``knowledge_map`` and session_start,
  **not** ``recall``.
* **It is a derived layer of L1**. Member invalidation (delete / supersede /
  archive) must be re-adjudicated; persistence goes through the
  read-merge-write of ``_LAYER_ATTRS``, deletions lay tombstones via the
  deletion ledger. Changing L1 without changing the topic layer equals a
  fake deletion.
* **Determinism first**. Traversal order is fixed, no random sources
  (``hash()`` is affected by PYTHONHASHSEED; always use sha1), **no stored
  centroids** -- a centroid can be recomputed from member vectors; storing
  it breaks on a model change and bloats the layer file with dimensions.
* **Never runs on the write path**. Clustering triggers only on explicit
  calls and consolidate; ``ingest`` does not slow down because of it --
  write-path latency is a hard constraint.
* **Better missing than wrong**. Records without a computed vector do not
  enter topics (they self-report into ``unassigned``); no guessing.

Usage::

    items = [{"id": ..., "vec": ndarray, "content": ..., "timestamp": ...,
              "entities": [...], "tags": {...}, "importance": ...}]
    topics = build(items)                       # full/incremental (passing topics = incremental)
    reclip(topics, alive_ids)                   # re-adjudicate after member invalidation
    refresh(topics, items_by_id, topn=3)        # recompute span/title/representative sentences
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Constants (env-overridable, for tuning without touching code)
# ----------------------------------------------------------------------

ENV_TOPIC_LAYER = "HIPPOCAMPUS_TOPIC_LAYER"
ENV_TOPIC_JOIN_SIM = "HIPPOCAMPUS_TOPIC_JOIN_SIM"
ENV_TOPIC_MAX = "HIPPOCAMPUS_TOPIC_MAX"
ENV_TOPIC_SUMMARY_TOPN = "HIPPOCAMPUS_TOPIC_SUMMARY_TOPN"

# The cosine gate for joining an existing topic. 0.55 is the empirical
# starting point for Chinese sentence-level embeddings -- too low merges the
# whole library into one basket, too high gives every record its own cluster
# (negative control: at 0.95 the topic count should approach the record
# count; otherwise similarity is not really participating in the decision).
TOPIC_JOIN_SIM_DEFAULT = 0.55
# Upper bound on topic count. Beyond it, **force-join the nearest topic**
# instead of creating new ones -- the cap-hit count is self-reported in
# stats. The **internal** topic count ≠ the map's displayed entry count
# (that is controlled by `knowledge_map(limit=)`), so this can be set higher
# than "readable at a glance".
# Basis for the value (grid scans on two independent real libraries):
# cap=200/400 both hit the cap; 600 suffices for one real library at the
# 9k-record Chinese-corpus scale (550 clusters, no cap hit), but the other
# 3.6k-record real library has 652 natural clusters, so 600 would push that
# library back into the cap -- the smallest whole step where neither real
# library hits the cap is **1000** (550 / 652 both leave headroom), hence
# 1000.
# Cost side: full rebuild at 1000/60 takes 2.6-2.9s (8.6s before the
# centroid normalization cache landed; see build()).
TOPIC_MAX_DEFAULT = 1000
# Per-topic member cap. **The direct nemesis of the giant basket**: measured
# on a real library without this limit, the first cluster keeps absorbing up
# to **2647 records (27% of the whole library)** -- Chinese sentence vectors
# are anisotropic (cosines between arbitrary sentences run universally
# high), absolute thresholds cannot stop centroid drift, and the only fix is
# a hard cap.
# Basis for the value (same-round grid scan): at mem=300 the largest cluster
# hugs 300 and top-5 concentration is 16%; mem=60 presses the largest
# cluster to 60 and top-5 to 3.3%, with median clusters of 3-5 records. The
# cost is roughly 1 more point of singleton share -- trading away "a
# 300-record basket with everything in it" is worth it. Both independent
# real libraries point to 60; not a single-library coincidence.
TOPIC_MAX_MEMBERS_DEFAULT = 60
# Batch size for the batched matrix multiply: within a batch it goes to
# BLAS, order is preserved across batches (the order is the source of
# determinism).
TOPIC_CHUNK = 512
# Number of extractive representative sentences.
TOPIC_SUMMARY_TOPN_DEFAULT = 3
# Per-sentence clip length for representative sentences (chars).
TOPIC_SUMMARY_CLIP = 60
# Topics with fewer members than this are **not worth summarizing** (1-2
# facts say it by themselves; another concluding sentence is just the
# original rephrased). Summarization costs the host effort -- do not waste
# it on crumbs.
TOPIC_MIN_SIZE_FOR_SUMMARY = 3
# Invalidated-member ratio above this → mark stale (a hint that "this
# knowledge point's content has changed").
TOPIC_STALE_RATIO = 0.5
# Maximum number of entities in a topic title.
TOPIC_TITLE_ENTITIES = 2
# Title fallback: with no entities and no tags, take this many leading chars
# of the earliest member's content.
TOPIC_TITLE_CLIP = 12
# Field name for the topic "generation". +1 on rebuild (force); on layer
# merge, **the higher generation overwrites the whole generation**.
# Without it, "rebuild" is only an additive key-union: old clusters on disk
# get merged back (measured: 652 + 698 → 1050 clusters), and the other
# side's old writer merges them in again on its next save -- never gets
# cleaned up.
TOPIC_EPOCH_FIELD = "_epoch"
# Fake-cluster filter: when the **share of informative members** falls below
# this ⇒ judged a fake cluster. Similar formatting ≠ the same subject -- a
# pile of "好的/收到/嗯" sits close together in sentence-vector space (all
# short, all acknowledgments) and gets clustered together, but that is not a
# knowledge point; having the host summarize it is pure waste.
# Why 0.5 and not higher: the stricter the judgment, the easier to misjudge
# real topics, and a misjudgment is **silent** (the topic disappears from
# pending and nobody knows); better to miss a filter than to kill a real
# one.
TOPIC_LOW_INFO_RATIO = 0.5
# Passing line for content informativeness (>=1 counts as "informative").
# See :func:`info_score`.
TOPIC_LOW_INFO_MIN_SCORE = 1
# Switch (default on): 0/off/false/no disables the fake-cluster filter.
ENV_TOPIC_LOW_INFO = "HIPPOCAMPUS_TOPIC_LOW_INFO_FILTER"
# The longest a body may be and still be judged "filler": beyond this
# length, all-filler text itself counts as content.
TOPIC_FILLER_MAX_LEN = 6

STATUS_ACTIVE = "active"
STATUS_STALE = "stale"
STATUS_ARCHIVED = "archived"

SUMMARY_KIND_EXTRACTIVE = "extractive"
SUMMARY_KIND_HOST = "host"


@dataclass(frozen=True)
class TopicDefaults:
    """Domain defaults container -- the injection seam.

    Why it is needed: ``domain`` must not import ``config`` (dependency
    graph ``_ALLOWED["domain"] == {"common"}``), so this layer **cannot read
    env by itself**; yet the topic layer's defaults are genuinely
    configurable. The solution is dependency inversion -- make the defaults
    a piece of pure data; whoever calls, provides:

      * Not passed ⇒ use :class:`TopicDefaults`' field defaults (equivalent
        to "env cleared"); domain is testable standalone, depending on no
        environment;
      * The brain composition root assembles one from
        :class:`hippocampus_mcp.config.Config` and passes it in; only then
        do the env knobs connect.

    Field names correspond one-to-one with Config's ``topic_*``, pinned by
    ``test_config_parity`` P7.
    """
    join_sim: float = TOPIC_JOIN_SIM_DEFAULT
    max_topics: int = TOPIC_MAX_DEFAULT
    max_members: int = TOPIC_MAX_MEMBERS_DEFAULT
    summary_topn: int = TOPIC_SUMMARY_TOPN_DEFAULT
    layer_enabled: bool = True
    low_info: bool = True


def _def(d: TopicDefaults | None) -> TopicDefaults:
    return d if d is not None else TopicDefaults()


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def _unit(vec: Any) -> np.ndarray | None:
    """Normalize; returns None for the zero vector (a zero vector has no
    direction and would only create noise clusters in clustering)."""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return None
    return arr / n


def _topic_id(seed: str) -> str:
    """Deterministic topic id (**no** uuid4/hash(): those make tests and
    persistence unreproducible)."""
    return "tpc_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


def _centroid(member_ids: Sequence[str], norms: dict[str, np.ndarray]) -> np.ndarray | None:
    """Mean of the members' unit vectors (re-normalized). Returns None when
    empty or all missing."""
    acc: np.ndarray | None = None
    cnt = 0
    for mid in member_ids:
        v = norms.get(mid)
        if v is None:
            continue
        acc = v.copy() if acc is None else acc + v
        cnt += 1
    if acc is None or cnt == 0:
        return None
    return _unit(acc / cnt)


def new_topic(member_id: str, epoch: int = 0) -> dict[str, Any]:
    """The skeleton of a new topic (the remaining fields are filled by
    :func:`refresh`).

    ``epoch`` is the generation this cluster was born into: 0 for daily
    increments (same generation as existing clusters; merges go through key
    union); on rebuild the whole layer increments uniformly (+1; merges then
    overwrite by whole generation, see :func:`layer_epoch`).
    """
    return {
        "id": _topic_id(member_id),
        "title": "",
        "status": STATUS_ACTIVE,
        "member_ids": [member_id],
        "size": 1,
        "summary": "",
        "summary_kind": "",
        "summary_evidence": [],
        "entities": [],
        "tags": [],
        "span": [0.0, 0.0],
        "created_at": 0.0,
        "updated_at": 0.0,
        "recall_hits": 0,
        TOPIC_EPOCH_FIELD: int(epoch),
    }


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------

def build(
    items: Sequence[dict[str, Any]],
    topics: dict[str, dict[str, Any]] | None = None,
    join_sim: float | None = None,
    max_topics: int | None = None,
    max_members: int | None = None,
    alive_ids: Iterable[str] | None = None,
    epoch: int | None = None,
    defaults: TopicDefaults | None = None,
) -> dict[str, dict[str, Any]]:
    """Assign records to topics (passing ``topics`` makes it **incremental**:
    only records not yet assigned are processed).

    The algorithm is a **leader-style greedy scan**, not hierarchical
    clustering: in the fixed ``(timestamp, id)`` order, each record decides
    "join the most similar topic / start its own". Three reasons to pick it
    over agglomerative:

      1. Agglomerative needs an O(n²) similarity matrix (an 89-million-pair
         library at 10k records); this is O(n·k) with k ≤ ``max_topics`` --
         about 470k dot products for a 10k library, seconds.
      2. **Naturally incremental** -- a new record only needs one comparison
         against k centroids, no full-library recompute, so it can hang off
         ``consolidate`` without touching the write path.
      3. The result is order-sensitive (earliest arrivals decide centers),
         but **the order is deterministic** (sorted by timestamp+id), so the
         same input run twice is byte-identical -- testable, reproducible.

    Centroids are **maintained incrementally** along the way
    (``sum_vec / count``); centroids of all topics are not recomputed per
    record (that would be O(n·k·m) -- measured two orders of magnitude
    slower).

    Args:
        items: ``{"id", "vec", "content", "timestamp", "entities", "tags",
            "importance"}``; records with a missing/zero ``vec`` **do not
            enter topics** (they go to ``unassigned``).
        topics: Existing topics (incremental mode). Modified **in place** and
            returned.
        join_sim: Join gate; defaults to env / 0.55.
        max_topics: Topic count cap; defaults to env /
            :data:`TOPIC_MAX_DEFAULT`.
        d: The :class:`TopicDefaults` injection pack (domain must not read
            env ⇒ the caller provides it). Omitted, pure numeric defaults
            apply, equivalent to clearing env.
        alive_ids: Ids of records currently **alive** (active). Different
            from "has a vector": records that are alive but have no vector
            yet must stay in the membership (see the "alive ≠ has vector"
            comment below). Omitted, the old wording applies (only records
            with vectors count), purely for compatibility with existing
            callers.

    Returns:
        ``{topic_id: topic}``.
    """
    thr = _def(defaults).join_sim if join_sim is None else float(join_sim)
    ep = 0 if epoch is None else int(epoch)
    cap = _def(defaults).max_topics if max_topics is None else int(max_topics)
    mm = _def(defaults).max_members if max_members is None else int(max_members)
    out: dict[str, dict[str, Any]] = dict(topics or {})

    by_id = {str(it.get("id")): it for it in items}
    # Cosine comparison needs consistent dimensions. In production, one
    # model_key implies one dimension; this defense is for **not crashing**:
    # with mixed dimensions, the first valid vector's dimension wins and the
    # rest do not enter topics -- better missing than wrong, they land in
    # coverage's unassigned and self-report instead of being guessed into a
    # cluster.
    dim: int | None = None
    norms: dict[str, np.ndarray] = {}
    for it in items:
        u = _unit(it.get("vec"))
        if u is None:
            continue
        if dim is None:
            dim = int(u.size)
        elif int(u.size) != dim:
            continue
        norms[str(it.get("id"))] = u

    # Existing topics: strip **invalidated** members first (the caller may
    # have just deleted memories), then compute initial centroids.
    #
    # "Alive" is not the same as "has a vector". Records that are active but
    # have no vector yet must stay in the membership -- otherwise every
    # refresh strips them out and re-adds them once vectors are back, and
    # the member set flaps forever. The cost of flapping is not just ugly
    # numbers: host summaries record a **member signature**, and once
    # members change the signature stops matching, so every refresh reports
    # "summary is stale" (a false stale); the host learns to treat it as
    # noise, and stops looking on the day it is genuinely stale.
    alive: set[str] | None = None if alive_ids is None else {str(x) for x in alive_ids}
    assigned: set[str] = set()
    for t in out.values():
        kept = [m for m in (t.get("member_ids") or [])
                if (m in norms if alive is None else m in alive)]
        t["member_ids"] = kept
        assigned.update(kept)

    # Incremental centroid state: tid -> [sum_vec, count]. **Archived topics
    # get no centroid** -- they were archived precisely for being emptied
    # out; a centroid would never win selection, and what is really feared
    # is the opposite: new members getting absorbed into it, making
    # "archived" a facade.
    cent: dict[str, list[Any]] = {}
    for tid, t in out.items():
        if t.get("status") == STATUS_ARCHIVED:
            continue
        acc: np.ndarray | None = None
        cnt = 0
        for m in t["member_ids"]:
            v = norms.get(m)
            if v is None:
                continue
            acc = v.copy() if acc is None else acc + v
            cnt += 1
        if acc is not None and cnt:
            cent[tid] = [acc, cnt]
    # The topic cap counts only **non-archived** topics. If archived topics
    # occupied slots permanently, a library that has had deletions would
    # stay stuck at the cap, never forming a new cluster, with nothing
    # reporting it (silent).
    live_n = sum(1 for t in out.values()
                 if t.get("status") != STATUS_ARCHIVED)

    order = sorted(
        (it for it in items if str(it.get("id")) in norms),
        key=lambda x: (float(x.get("timestamp") or 0.0), str(x.get("id"))),
    )

    # Batched matrix multiply: hand the "every record × every centroid" dot
    # products to BLAS in bulk instead of a per-record Python loop. Measured
    # on a real large library (50 clusters), the per-record version slows to
    # the order of seconds; hung inside consolidate it would trip the host
    # timeout at any moment (session_end once hit 206s because of
    # _find_contradictions; the same trick brought it to 0.34s);
    # switching to (b×d) @ (d×k) brought it back down. Within a batch it is
    # parallel; order is preserved across batches -- the order is the source
    # of result determinism, and it cannot be computed in one shot (then
    # centroids would stop updating as records get assigned).
    #
    # Centroid unit vectors are **cached** by tid and invalidated the moment
    # ``cent[tid]`` changes. Before the cache, measurements on a large
    # library showed ``_unit`` being called a huge number of times, a large
    # share of it from the "supplementary scoring of clusters created within
    # the batch" O(batch size×pending) loop below -- each new cluster's
    # centroid was recomputed for **every** subsequent record in the batch.
    # With the cache, the same centroid is computed only when it really
    # changed. The cache does not change values: it still fetches the same
    # ``_unit(cent[tid][0]/cent[tid][1])``.
    _UNSET = object()
    unit_cache: dict[str, Any] = {}

    def _cent_unit(tid: str) -> np.ndarray | None:
        u = unit_cache.get(tid, _UNSET)
        if u is not _UNSET:
            return u
        s, c = cent[tid]
        u = _unit(s / c)
        unit_cache[tid] = u
        return u

    pending: list[str] = []          # clusters created within this batch, not yet in the snapshot matrix
    for start in range(0, len(order), TOPIC_CHUNK):
        batch = order[start:start + TOPIC_CHUNK]
        snap = [t for t in cent if t not in pending]
        if snap:
            cent_mat = np.vstack([_cent_unit(t) for t in snap])
            # Full clusters stop absorbing: the giant-basket nemesis (see the TOPIC_MAX_MEMBERS comment)
            full = np.array(
                [len(out[t].get("member_ids") or []) >= mm for t in snap],
                dtype=bool)
        else:
            cent_mat = np.zeros((0, 1), dtype=np.float64)
            full = np.zeros((0,), dtype=bool)
        mat = np.vstack([norms[str(it.get("id"))] for it in batch])
        sims = (mat @ cent_mat.T) if cent_mat.shape[0] else \
            np.zeros((len(batch), 0), dtype=np.float64)
        if full.any():
            sims[:, full] = -np.inf
        for row, it in enumerate(batch):
            mid = str(it.get("id"))
            if mid in assigned:
                continue
            nv = norms[mid]
            best_tid, best_sim = "", -2.0
            if sims.shape[1]:
                j = int(np.argmax(sims[row]))
                v = float(sims[row][j])
                if np.isfinite(v):
                    best_tid, best_sim = snap[j], v
            for tid in pending:          # clusters created within this batch: score them too (few)
                c = _cent_unit(tid)
                if c is None:
                    continue
                s = float(np.dot(c, nv))
                if (s > best_sim
                        and len(out[tid].get("member_ids") or []) < mm):
                    best_tid, best_sim = tid, s
            # Re-check the cap: the ``full`` mask was computed only once at
            # **batch start**; within one batch a cluster can grow from mm
            # all the way to mm+batch size (measured 300 → 563).
            if (best_tid and best_sim >= thr
                    and len(out[best_tid].get("member_ids") or []) < mm):
                out[best_tid]["member_ids"].append(mid)
                cent[best_tid] = [cent[best_tid][0] + nv, cent[best_tid][1] + 1]
                unit_cache.pop(best_tid, None)   # centroid changed; cache invalidated
                assigned.add(mid)
            elif live_n < cap:
                t = new_topic(mid, ep)
                out[t["id"]] = t
                cent[t["id"]] = [nv.copy(), 1]
                unit_cache[t["id"]] = _unit(nv.copy())   # known value; saves a recompute
                pending.append(t["id"])
                assigned.add(mid)
                live_n += 1
            elif best_tid:
                # Cap hit: no new topic; force-join the nearest (cap hits self-reported by stats)
                out[best_tid]["member_ids"].append(mid)
                cent[best_tid] = [cent[best_tid][0] + nv, cent[best_tid][1] + 1]
                unit_cache.pop(best_tid, None)   # centroid changed; cache invalidated
                assigned.add(mid)
            # cap <= 0 and no cluster to join → stays in unassigned; never silently creates
        pending = []
    return out


def reclip(topics: dict[str, dict[str, Any]], alive_ids: Iterable[str]) -> int:
    """Re-adjudicate membership against the **currently alive** records
    (derived-layer discipline: union only ever adds).

    Records deleted / superseded / archived in L1 must be stripped from
    topics, otherwise a "knowledge point" keeps citing a memory that no
    longer counts -- and since it never enters the retrieval path, the user
    can never see the inconsistency (silence is the worst outcome).

    Args:
        topics: The topic table (modified in place).
        alive_ids: Member ids that should still count (filtered by the
            caller on the active state).

    Returns:
        Number of member references removed.
    """
    alive = set(alive_ids)
    dropped = 0
    for t in topics.values():
        before = list(t.get("member_ids") or [])
        after = [m for m in before if m in alive]
        dropped += len(before) - len(after)
        t["member_ids"] = after
        t["size"] = len(after)
        if not after:
            t["status"] = STATUS_ARCHIVED
    return dropped


# ----------------------------------------------------------------------
# Refresh: title / span / representative sentences / status
# ----------------------------------------------------------------------

def _top_entities(members: Sequence[dict[str, Any]], limit: int) -> list[str]:
    cnt: dict[str, int] = {}
    for m in members:
        for e in (m.get("entities") or []):
            if not e:
                continue
            cnt[str(e)] = cnt.get(str(e), 0) + 1
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:limit]]


def _top_tags(members: Sequence[dict[str, Any]], limit: int) -> list[str]:
    cnt: dict[str, int] = {}
    for m in members:
        tags = m.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        for k, v in tags.items():
            vals = v if isinstance(v, (list, tuple)) else [v]
            for one in vals:
                key = f"{k}={one}"
                cnt[key] = cnt.get(key, 0) + 1
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:limit]]


def _title_of(members: Sequence[dict[str, Any]]) -> str:
    """Deterministic title: trusted entities > structured tags > earliest
    member content fragment.

    Deliberately **no** "take the longest sentence as title" style
    heuristics: they make the title flap as members come and go, and the
    title is the one field in the map read over and over by human eyes --
    flapping hurts more than being plain.
    """
    ents = _top_entities(members, TOPIC_TITLE_ENTITIES)
    if ents:
        return "·".join(ents)
    tags = _top_tags(members, 1)
    if tags:
        return tags[0]
    if members:
        oldest = min(members, key=lambda m: (float(m.get("timestamp") or 0.0),
                                             str(m.get("id"))))
        return (str(oldest.get("content") or "").strip()[:TOPIC_TITLE_CLIP]
                or "(未命名)")
    return "(未命名)"


def _collision_suffix(topic: dict[str, Any],
                      members: Sequence[dict[str, Any]]) -> str:
    """The disambiguating suffix appended to a title on name collision.

    Pain point measured on a real library: several clusters with
    character-identical titles (the measured library had two same-named
    clusters); the host **cannot tell which one the map means**, and cannot
    name one to have it tidied up.

    Suffix priority (all deterministic sources, no randomness/models):
      1. The **next trusted entity** unused by the original title ⇒
         「sv·fitness·P-43」
      2. The first structured tag absent from the original title ⇒
         「sv·fitness·类型=实验」
      3. Neither exists ⇒ a sequence number 「#2」 (at least nameable, not
         two indistinguishable same-named entries)

    Deliberately **does not touch** host-named titles (``title_kind=host``):
    those are names given by a person, and appending a suffix without asking
    belongs to the "most trust-damaging silent rewrite" class -- it only
    gets the ``title_collision`` hint.
    """
    base = str(topic.get("title") or "")
    for ent in _top_entities(members, TOPIC_TITLE_ENTITIES + 3):
        if ent and ent not in base:
            return f"·{ent}"
    for tag in _top_tags(members, 3):
        if tag and tag not in base:
            return f"·{tag}"
    return ""


def disambiguate_titles(topics: dict[str, dict[str, Any]],
                        items_by_id: dict[str, dict[str, Any]] | None = None) -> int:
    """Append disambiguating suffixes to same-named topics; returns the
    number of **titles that were changed**.

    - Grouping key = character-identical title; only groups with >1 entry
      are processed (a single entry gets no suffix, avoiding needless
      flapping).
    - Within a group, order by ``(size descending, topic id ascending)`` ⇒
      deterministic; the **first keeps its original title** (big clusters
      stay put, small ones rename -- the smallest change surface as members
      come and go).
    - Host-named titles are **not changed**; only ``title_collision`` is set
      (see :func:`_collision_suffix`).
    - The same-name count goes into ``title_collision`` as a self-report:
      seeing ``=3`` the host knows "two more share this name", even when the
      one at hand is the renamed entry.
    """
    groups: dict[str, list[str]] = {}
    for tid, t in topics.items():
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "")
        if not title or title == "(未命名)":
            continue
        groups.setdefault(title, []).append(str(tid))

    changed = 0
    for title, tids in groups.items():
        if len(tids) < 2:
            continue
        tids.sort(key=lambda x: (-(int(topics[x].get("size") or 0)), x))
        for idx, tid in enumerate(tids):
            t = topics[tid]
            t["title_collision"] = len(tids)
            if idx == 0 or str(t.get("title_kind") or "") == TITLE_KIND_HOST:
                continue
            members = []
            if items_by_id:
                members = [items_by_id[m] for m in (t.get("member_ids") or [])
                           if m in items_by_id]
            suffix = _collision_suffix(t, members) or f"#{idx + 1}"
            t["title"] = f"{title}{suffix}"
            changed += 1
    return changed


def _extractive_summary(
    topic: dict[str, Any],
    members: Sequence[dict[str, Any]],
    norms: dict[str, np.ndarray],
    topn: int,
) -> tuple[str, list[str]]:
    """Extractive "representative sentences": the N originals closest to the
    centroid, not a summary.

    Both the name and the implementation are deliberately conservative -- it
    answers "what is this topic roughly about", **not** "what conclusion do
    these records support". The latter waits for the host to write back
    (``summary_kind=host``). Presenting excerpts as if they were summaries
    is the over-promise this layer is most prone to.
    """
    ids = [m for m in (topic.get("member_ids") or []) if m in norms]
    if not ids or topn <= 0:
        return "", []
    c = _centroid(ids, norms)
    scored: list[tuple[float, str]] = []
    for mid in ids:
        v = norms.get(mid)
        if v is None:
            continue
        s = float(np.dot(c, v)) if c is not None else 0.0
        scored.append((s, mid))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    picked = scored[:topn]
    content_by_id = {str(m.get("id")): str(m.get("content") or "") for m in members}
    parts: list[str] = []
    for _s, mid in picked:
        txt = " ".join(content_by_id.get(mid, "").split())[:TOPIC_SUMMARY_CLIP]
        if txt:
            parts.append(txt)
    return " ｜ ".join(parts), [mid for _s, mid in picked]


# ----------------------------------------------------------------------
# Fake-cluster filter: the content-informativeness criterion
# ----------------------------------------------------------------------

# Acknowledgments/filler phrases. They cluster together because **they are
# all short and all acknowledgments**, not because they discuss the same
# thing. Normalized forms, all ASCII lowercase (compared after stripping
# punctuation and whitespace).
_FILLER_PHRASES = frozenset({
    "好的", "好", "收到", "收到收到", "明白", "了解", "知道了", "嗯", "嗯嗯",
    "哦", "行", "可以", "是的", "对", "对对", "不错", "谢谢", "多谢", "感谢",
    "继续", "稍等", "等等", "等等看", "ok", "okay", "yes", "no", "hi", "hello",
    "哈哈", "呵呵", "嗯好", "好的好的", "行吧", "没问题", "没问题的", "okok",
})

_PUNCT_RE = re.compile(r"[^\w一-鿿]+")
_DIGIT_RE = re.compile(r"\d")


def _fold_repeat(s: str) -> str:
    """Fold whole-string repetition: 「好的好的」→「好的」, 「okok」→「ok」.

    The most common filler shape is repetition (people type it twice when
    agreeing); without folding, detection misses it.
    """
    for n in (2, 3, 4):
        if len(s) > n and len(s) % n == 0:
            part = s[:n]
            if s == part * (len(s) // n):
                return part
    return s


def info_score(content: Any, entities: Any = None, tags: Any = None) -> int:
    """The **informativeness score** of one content body (0 = nothing worth
    summarizing).

    The criteria are fully deterministic, no models -- this layer exists so
    the host can pick "what is worth summarizing"; if the criteria
    themselves were a black box, there would be no explaining why some topic
    was not picked.

    Scoring additions (cumulative):
      * **A floor of 1 point**: as long as it is not filler and not empty,
        it counts as "has content";
      * ≥20 chars after normalization +1;
      * contains Arabic numerals +1 (hard information; Chinese numerals are
        deliberately not counted -- too common in Chinese libraries, counting
        them would destroy discriminability);
      * has entities or tags +1 (it touches concrete objects).

    Scored 0 outright: empty, or a filler-table hit (including after
    repetition folding).

    🔴 **Why the floor point exists (learned the hard way; must not be
    reverted)**: the first version gave +1 only at "≥8 chars", and the
    `test_knowledge_tool_face` sample text 「青蒿 工艺 要点」 (6 chars) scored
    0 ⇒ every real topic was judged a fake cluster ⇒ **pending dropped
    straight to zero**. **Short ≠ uninformative** -- knowledge-point titles
    are short by nature. "Has information?" and "is filler?" are two
    different judgments; only the latter deserves a 0.
    """
    text = " ".join(str(content or "").split())
    if not text:
        return 0
    norm = _PUNCT_RE.sub("", text).lower()
    if not norm:
        return 0
    folded = _fold_repeat(norm)
    if folded in _FILLER_PHRASES or norm in _FILLER_PHRASES:
        return 0
    if len(norm) <= 1:
        return 0
    # Floor of 1 point: not filler means it has content (see the hard-won lesson in the docstring)
    score = 1
    if len(norm) >= 20:
        score += 1
    if _DIGIT_RE.search(text):
        score += 1
    if entities or tags:
        score += 1
    return score


def info_ratio_of(members: Sequence[dict[str, Any]]) -> float:
    """The **share of informative members** in the cluster (0.0-1.0).
    Returns 0.0 for an empty cluster."""
    if not members:
        return 0.0
    ok = 0
    for m in members:
        if info_score(m.get("content"), m.get("entities"), m.get("tags")) \
                >= TOPIC_LOW_INFO_MIN_SCORE:
            ok += 1
    return round(ok / len(members), 4)


def low_info_topic_ids(topics: dict[str, dict[str, Any]]) -> list[str]:
    """Ids of topics judged fake clusters (low informativeness), in map
    order.

    For stats self-reporting. 🔴 **Must be self-reported**: these clusters
    are excluded from pending; not saying so means they vanish silently --
    the host would believe "everything is summarized".
    """
    return [str(t.get("id")) for t in ordered(topics)
            if t.get("low_info")]


def layer_epoch(topics: dict[str, Any] | None) -> int:
    """Which generation this layer belongs to = the maximum ``_epoch`` over
    all clusters in the layer (0 if absent).

    Merges use it to decide "who is fresher": **the higher generation
    overwrites wholesale; only the same generation merges by key union**.
    Taking the max instead of a "top-of-layer field" is because the layer
    itself is ``{topic_id: topic}`` -- stuffing a non-cluster top-level key
    into the layer would be counted as a cluster (``len(topics)``, the map,
    and ``host_summary_stats`` all iterate values) -- far too much pollution
    surface.
    """
    if not isinstance(topics, dict) or not topics:
        return 0
    best = 0
    for t in topics.values():
        if not isinstance(t, dict):
            continue
        try:
            e = int(t.get(TOPIC_EPOCH_FIELD) or 0)
        except (TypeError, ValueError):
            e = 0
        if e > best:
            best = e
    return best


def migrate_host_summaries(
    old: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    min_overlap: float = 0.5,
) -> list[dict[str, Any]]:
    """On rebuild, carry old **host summaries** onto new clusters by member
    overlap.

    Summaries are written with real host effort and this system will never
    generate them automatically -- a rebuild redraws cluster boundaries and
    would drop every one of them, the most damaging kind of silence. So they
    are carried even across changed boundaries, and after carrying, the
    signature is **deliberately not recomputed**: member count vs signature
    mismatch ⇒ the map marks it stale (⚠️) ⇒ reminding the host that "the
    basis for this one has changed; re-verify". That is more honest than
    pretending it is still the original entry.

    Returns:
        ``[{"from", "to", "overlap"}]``; failed carries are **not in the
        list** -- the caller must use it to self-report "how many were
        lost", never silently.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return []
    moved: list[dict[str, Any]] = []
    taken: set[str] = set()
    for oid, ot in old.items():
        if not isinstance(ot, dict):
            continue
        if str(ot.get("summary_kind") or "") != SUMMARY_KIND_HOST:
            continue
        if not str(ot.get("summary") or "").strip():
            continue
        om = {str(x) for x in (ot.get("member_ids") or [])}
        if not om:
            continue
        best_id, best_score = "", 0.0
        for nid, nt in new.items():
            if nid in taken or not isinstance(nt, dict):
                continue
            # If the new cluster already has a summary, do not seize it: the old summary's boundaries are more suspect
            if str(nt.get("summary_kind") or "") == SUMMARY_KIND_HOST:
                continue
            nm = {str(x) for x in (nt.get("member_ids") or [])}
            if not nm:
                continue
            score = len(om & nm) / float(len(om))
            if score > best_score:
                best_id, best_score = nid, score
        if not best_id or best_score < float(min_overlap):
            continue
        nt = new[best_id]
        nt["summary"] = ot.get("summary", "")
        nt["summary_kind"] = SUMMARY_KIND_HOST
        nt["summary_at"] = ot.get("summary_at", 0.0)
        nt["summary_evidence"] = list(ot.get("summary_evidence") or [])
        # Signature/scale **copied verbatim from the old**: mismatch with the new cluster's members ⇒ judged stale ⇒ prompts re-verification.
        nt["summary_signature"] = ot.get("summary_signature", "")
        nt["summary_member_count"] = ot.get("summary_member_count", 0)
        taken.add(best_id)
        moved.append({"from": str(oid), "to": str(best_id),
                      "overlap": round(best_score, 3)})
    return moved


def refresh(
    topics: dict[str, dict[str, Any]],
    items_by_id: dict[str, dict[str, Any]],
    topn: int | None = None,
    now: float | None = None,
    defaults: TopicDefaults | None = None,
) -> None:
    """Recompute derived fields: title / span / entities / tags /
    representative sentences / status.

    **Only overwrites extractive summaries**: host-written ones
    (``summary_kind=host``) are untouched, otherwise a single consolidation
    would wash away the summary a person put effort into -- the most
    trust-damaging class of silent rewrite.

    Args:
        topics: The topic table (modified in place).
        items_by_id: ``id -> item`` (only the surviving members).
        topn: Number of representative sentences; defaults to env / 3.
        now: Timestamp basis (tests may inject a fixed value).
    """
    n = _def(defaults).summary_topn if topn is None else int(topn)
    stamp = now if now is not None else 0.0
    norms: dict[str, np.ndarray] = {}
    for k, it in items_by_id.items():
        u = _unit(it.get("vec"))
        if u is not None:
            norms[str(k)] = u
    for t in topics.values():
        before = list(t.get("member_ids") or [])
        ids = [m for m in before if m in items_by_id]
        members = [items_by_id[m] for m in ids]
        # The member list **shrinks** with the truth: this is a third path
        # besides reclip/build (the caller may pass an items_by_id that
        # contains only surviving items). Without shrinking, invalidated ids
        # stay in the membership forever -- coverage computes high,
        # signatures compute crooked, and the map's size runs large.
        t["member_ids"] = ids
        t["size"] = len(ids)
        lost = len(before) - len(ids)
        if not ids:
            t["status"] = STATUS_ARCHIVED
            t["summary"] = ""
            t["summary_evidence"] = []
            t["summary_at"] = 0.0
            t["summary_signature"] = ""
            t["summary_member_count"] = 0
            continue
        # stale describes "just discovered the members changed", not a
        # lifelong label. Every refresh re-judges from active -- otherwise
        # one flap would make this topic **permanently** stale, and on the
        # map stale exists to make people look; cry wolf twice and nobody
        # looks.
        t["status"] = STATUS_ACTIVE
        # Host-named titles are not recomputed. The fallback title (first 12
        # chars of the earliest member) is a measured pain point -- multiple
        # clusters with the same name the host cannot tell apart; and if the
        # host named it in fras.f, one consolidation would wipe it back to
        # the fallback -- exactly the "most trust-damaging silent rewrite"
        # class.
        if str(t.get("title_kind") or "") != TITLE_KIND_HOST:
            t["title"] = _title_of(members)
        ts = [float(m.get("timestamp") or 0.0) for m in members]
        t["span"] = [min(ts), max(ts)]
        t["entities"] = _top_entities(members, 3)
        t["tags"] = _top_tags(members, 3)
        # Fake-cluster criterion: **content informativeness**. Mark only,
        # never delete -- deleting would let members re-form into new
        # clusters (host-written summary signatures would stop matching),
        # and deletion is a real deletion with no tombstone.
        # After marking, :func:`pending_summary_ids` decides whether to pick
        # it for summarization.
        ir = info_ratio_of(members)
        t["info_ratio"] = ir
        t["low_info"] = bool(ir < TOPIC_LOW_INFO_RATIO)
        if not t.get("created_at"):
            t["created_at"] = min(ts)
        t["updated_at"] = stamp or max(ts)
        # Fill extractive only: host-written summaries are not overwritten (see the docstring)
        if t.get("summary_kind") != SUMMARY_KIND_HOST:
            text, ev = _extractive_summary(t, members, norms, n)
            t["summary"] = text
            t["summary_evidence"] = ev
            t["summary_kind"] = SUMMARY_KIND_EXTRACTIVE if text else ""
        # Invalidated share: how many members this refresh found no longer count (the before-vs-ids difference)
        total = lost + len(ids)
        if lost and total and (lost / total) > TOPIC_STALE_RATIO:
            t["status"] = STATUS_STALE

    # Name-collision disambiguation runs **after all titles are computed** --
    # processing one by one cannot see what names other clusters took, and
    # renaming while computing would make results depend on traversal order.
    disambiguate_titles(topics, items_by_id)


# ----------------------------------------------------------------------
# Stats (for brain_stats / knowledge_map)
# ----------------------------------------------------------------------

def coverage_of(topics: dict[str, dict[str, Any]], active_total: int) -> dict[str, Any]:
    """Coverage self-report.

    An empty library reports ``None`` + ``unknown``, **not 0** -- "0
    records" and "cannot tell" are two different things; reporting 0 would
    make the host believe "this library really has nothing assigned".
    """
    assigned = len({m for t in topics.values() for m in (t.get("member_ids") or [])})
    if not active_total:
        return {
            "topics": len(topics),
            "assigned": assigned,
            "active_total": 0,
            "coverage": None,
            "coverage_level": "unknown",
            "unassigned": None,
            "note": "库内没有 active 记忆，覆盖率无意义",
        }
    cov = assigned / active_total
    note = ""
    if cov > 1.0:
        # Numerator exceeds denominator ⇒ some topic still cites non-active
        # records (re-adjudication not run / run late). Reporting >100%
        # insults the reader, silently truncating hides the real problem ⇒
        # truncate + self-report.
        note = ("assigned 超过 active 总数：有主题仍引用非 active 记录"
                "（重裁未跑或跑晚了）")
        cov = 1.0
    level = "ok" if cov >= 0.8 else ("low" if cov >= 0.5 else "very_low")
    return {
        "topics": len(topics),
        "assigned": assigned,
        "active_total": active_total,
        "coverage": round(cov, 4),
        "coverage_level": level,
        "unassigned": active_total - min(assigned, active_total),
        "note": note,
    }


def by_status(topics: dict[str, dict[str, Any]]) -> dict[str, int]:
    out = {STATUS_ACTIVE: 0, STATUS_STALE: 0, STATUS_ARCHIVED: 0}
    for t in topics.values():
        key = str(t.get("status") or STATUS_ACTIVE)
        out[key] = out.get(key, 0) + 1
    return out


def ordered(topics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ordering: size × recency × recall hits (deterministic; id as the
    tiebreaker)."""
    return sorted(
        topics.values(),
        key=lambda t: (-int(t.get("size") or 0),
                       -float(t.get("updated_at") or 0.0),
                       -int(t.get("recall_hits") or 0),
                       str(t.get("id"))),
    )

# ----------------------------------------------------------------------
# Host summary write-back
# ----------------------------------------------------------------------
#
# This module by itself can only produce "representative sentences" -- a few
# original excerpts closest to the centroid, answering "what is this topic
# roughly about". The "higher-level knowledge" the user wants is a
# **summary**: writing N facts into one conclusion. That step needs
# intelligence, and the production path not embedding an LLM is this
# project's boundary, so the approach is a **write-back channel**: hippo
# produces the material → the host summarizes → write back and persist.
#
# This layer therefore has two kinds of summary that must be kept distinct
# and must never overwrite each other:
#   * extractive -- this module can recompute it any time; losing it costs
#     nothing;
#   * host       -- written with real host effort; **no automatic flow may
#     ever overwrite it**.
# So this module offers only "write / revert / query state", never
# "automatic rewriting".

# Length cap for summary text. The layer file grows with it, and an
# over-long "summary" usually means the host pasted the material back
# verbatim -- which is worse than not writing one (just read the
# representative sentences).
SUMMARY_MAX_LEN = 2000
# Cap on evidence entries. This layer's value is **being able to point back
# to L1**, so evidence must stay trimmed to "enough to verify".
SUMMARY_EVIDENCE_MAX = 20
# Per-entry content clip length for the brief material.
TOPIC_EVIDENCE_CLIP = 200

# ----------------------------------------------------------------------
# The lightweight FRAS four-field contract for host summaries
# ----------------------------------------------------------------------
# Borrowed from the previous work's creation contract: the FRAS four fields
# (F role / R strong relations / A contract / S non-inferable constraints),
# trimmed to four keys for the topic layer's shape:
#
#   f -- category: what this topic is (one sentence, used directly as the
#        title)
#   r -- relations: entities / related topics that must be read together
#        (list)
#   a -- citable conclusion: the substance of this entry (the sentence shown
#        on maps / the opening package)
#   s -- non-inferable constraints: exceptions and boundaries that cannot be
#        derived from a, but getting them wrong causes damage
#
# Two design disciplines:
#   ① **S's admission bar is "not derivable + prevents errors"**; repeating
#      a is not allowed -- if there genuinely is none, write "无" (none)
#      explicitly; leaving it empty is treated as unfilled and the whole
#      entry is refused;
#   ② **Structure failure refuses the whole entry** (not one field changed)
#      -- half a change is harder to audit than no change.
# Why it is worth adding: two measured pain points -- the title fallback
# taking "the earliest member's first 12 chars" left multiple same-named
# clusters the host could not tell apart; a cluster's "purity" could be
# fooled by similar writing style. Forcing the host to write f/s is the
# cheapest known filter: **a cluster that cannot produce an s is probably
# not a real topic**.
FRAS_KEYS = ("f", "r", "a", "s")
FRAS_F_MAX = 80       # the category is title-grade: fits on one line
FRAS_R_MAX = 10       # cap on relation entries
FRAS_S_MAX = 400      # cap on the constraints field (a reuses SUMMARY_MAX_LEN)
#: Marker that a title was named by the host. Without it, refresh would
#: recompute the title with _title_of every time, and the host's chosen name
#: (plus collision disambiguation) would be silently wiped by one
#: consolidation.
TITLE_KIND_HOST = "host"


def validate_fras(fras: Any) -> tuple[dict[str, Any] | None, str]:
    """Validate the four-field structure. Returns ``(parsed, error)``; a
    non-empty error means the whole entry is refused.

    ``fras=None`` is a legal input (older hosts write only free text) and
    returns ``(None, "")``.
    """
    if fras is None:
        return None, ""
    if not isinstance(fras, dict):
        return None, "fras 必须是对象（f / r / a / s 四个键）"
    missing = [k for k in FRAS_KEYS if k not in fras]
    if missing:
        return None, (f"fras 缺字段 {missing}：要么四键齐全，要么整个不传"
                      "（自由文本 summary 仍然支持）")
    f = " ".join(str(fras.get("f") or "").split())
    a = " ".join(str(fras.get("a") or "").split())
    s = " ".join(str(fras.get("s") or "").split())
    r_raw = fras.get("r")
    if isinstance(r_raw, str):
        r = [x.strip() for x in r_raw.split(",") if x.strip()]
    elif isinstance(r_raw, (list, tuple)):
        r = [str(x).strip() for x in r_raw if str(x).strip()]
    else:
        r = []
    if not f:
        return None, "fras.f 为空：范畴是这一条的标题，不能省"
    if len(f) > FRAS_F_MAX:
        return None, f"fras.f 过长（{len(f)} > {FRAS_F_MAX}），范畴要一行放得下"
    if not a:
        return None, "fras.a 为空：可引用的结论是这一条的本体"
    if len(a) > SUMMARY_MAX_LEN:
        return None, f"fras.a 过长（{len(a)} > {SUMMARY_MAX_LEN}），请压成结论"
    if not s:
        return None, ("fras.s 为空：准入标准是「推不出来 + 防错」的约束，"
                      "确实没有也要显式写「无」——空着与没填分不开")
    if len(s) > FRAS_S_MAX:
        return None, f"fras.s 过长（{len(s)} > {FRAS_S_MAX}）"
    if len(r) > FRAS_R_MAX:
        return None, f"fras.r 过多（{len(r)} > {FRAS_R_MAX}），只留强关联"
    return {"f": f, "r": r, "a": a, "s": s}, ""


def signature_of(member_ids: Any) -> str:
    """Member-set fingerprint: records "which batch of members this summary
    was made on".

    Why it is needed: a host summary is a judgment about the members **as
    they were**. Members later change (additions / deletions / supersedes)
    and the summary may no longer hold. But it must not be automatically
    rewritten for that -- refresh not overwriting host is the same
    discipline. So the approach is **annotate only, never overwrite**: on a
    signature mismatch, reads report ``stale=True`` and the host decides
    whether to rewrite.
    """
    ids = sorted({str(m) for m in (member_ids or [])})
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:12]


def apply_host_summary(
    topic: dict[str, Any],
    text: str,
    evidence: Sequence[str] | None = None,
    member_ids: Sequence[str] | None = None,
    now: float = 0.0,
    fras: Any = None,
) -> tuple[bool, str]:
    """Write the host's summary back onto the topic. **Validation failure
    always refuses the write**; no silent degradation.

    Four hard validations (failing any one returns the whole entry with a
    reason):

      1. Text non-empty and not over ``SUMMARY_MAX_LEN`` -- an empty write
         back disguises "not summarized" as "summarized"; over-long text
         bloats the layer file (every save re-reads and re-merges it);
      2. Ids in ``evidence`` must be **members** of this topic -- citing out
         of thin air equals forging provenance, and this layer's entire
         advantage over L2 is being able to point back into L1;
      3. Evidence count not over ``SUMMARY_EVIDENCE_MAX``;
      4. (FRAS) when ``fras`` is given, all four keys must be present (see
         :func:`validate_fras`) -- structure failure refuses the whole
         entry, not one field changed.

    Args:
        topic: The topic (modified in place).
        text: Summary text. **May be left empty when only fras is passed** --
            ``fras.a`` (the citable conclusion) then serves as the sentence,
            sparing the host from writing it twice.
        evidence: Provenance (L1 memory ids). Empty means "this conclusion
            is not pinned to specific records".
        member_ids: The current member set (omitted, the topic's own is
            used).
        now: Write-back timestamp.
        fras: The optional FRAS four-field structure. When given, the title
            is handed to ``fras.f`` (``title_kind=host``; refresh stops
            recomputing the title) and the whole structure is stored in
            ``summary_fras``.

    Returns:
        ``(ok, error)``. When ok is False, **not one field** of topic was
        changed.
    """
    parsed, fras_err = validate_fras(fras)
    if fras_err:
        return False, fras_err
    txt = " ".join(str(text or "").split())
    if not txt and parsed:
        # Four fields only: the one-sentence conclusion is a (minimal friction, and a single canonical wording)
        txt = parsed["a"]
    if not txt:
        return False, "summary 为空：空写回会把「没归纳」伪装成「归纳过了」"
    if len(txt) > SUMMARY_MAX_LEN:
        return False, (f"summary 过长（{len(txt)} > {SUMMARY_MAX_LEN}），"
                       "请压缩成结论而不是复述素材")
    members = {str(m) for m in (member_ids if member_ids is not None
                                else (topic.get("member_ids") or []))}
    ev: list[str] = []
    for e in (evidence or []):
        s = str(e)
        if s not in members:
            return False, f"evidence 不是本主题成员：{s}"
        if s not in ev:
            ev.append(s)
    if len(ev) > SUMMARY_EVIDENCE_MAX:
        return False, f"evidence 过多（{len(ev)} > {SUMMARY_EVIDENCE_MAX}）"
    topic["summary"] = txt
    topic["summary_kind"] = SUMMARY_KIND_HOST
    topic["summary_evidence"] = ev
    topic["summary_at"] = float(now or 0.0)
    topic["summary_signature"] = signature_of(members)
    topic["summary_member_count"] = len(members)
    if parsed:
        topic["summary_fras"] = parsed
        # Hand the title to the host: the fallback title (the earliest
        # member's first 12 chars) is a measured pain point (multiple
        # same-named clusters, host cannot tell them apart). title_kind
        # stops refresh from recomputing it.
        topic["title"] = parsed["f"]
        topic["title_kind"] = TITLE_KIND_HOST
    if now:
        topic["updated_at"] = float(now)
    return True, ""


def revert_host_summary(topic: dict[str, Any]) -> tuple[bool, str]:
    """Revert the host summary (fall back to extractive representative
    sentences).

    A mistake must be revertible -- otherwise one bad summary occupies the
    topic forever while automatic flows dare not touch it. Clearing the
    fields takes effect **immediately**; the next refresh refills the
    extractive form.

    A host-named title (``title_kind=host``) is zeroed along with it -- the
    title is part of the summary; reverting the summary while keeping the
    host's chosen name, which the next refresh will not recompute, leaves
    the topic wearing a title nothing corresponds to. Automatic titles are
    untouched.
    """
    if str(topic.get("summary_kind") or "") != SUMMARY_KIND_HOST:
        return False, "这个主题没有宿主归纳可撤销（当前不是 host 归纳）"
    topic["summary"] = ""
    topic["summary_kind"] = ""
    topic["summary_evidence"] = []
    topic["summary_at"] = 0.0
    topic["summary_signature"] = ""
    topic["summary_member_count"] = 0
    topic.pop("summary_fras", None)
    if str(topic.get("title_kind") or "") == TITLE_KIND_HOST:
        topic.pop("title_kind", None)
        topic["title"] = ""   # refilled by _title_of on the next refresh
    return True, ""


def summary_state(
    topic: dict[str, Any],
    current_member_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The current state of this summary (for the host to decide whether to
    rewrite).

    ``stale=True`` means **the member set changed**, not "the content is
    wrong". This module cannot judge content; it only surfaces that fact.
    """
    kind = str(topic.get("summary_kind") or "")
    now_ids = list(current_member_ids if current_member_ids is not None
                   else (topic.get("member_ids") or []))
    sig_now = signature_of(now_ids)
    sig_at = str(topic.get("summary_signature") or "")
    return {
        "kind": kind or "none",
        "is_host": kind == SUMMARY_KIND_HOST,
        "written_at": topic.get("summary_at") or None,
        "evidence": list(topic.get("summary_evidence") or []),
        "stale": bool(kind == SUMMARY_KIND_HOST and sig_at and sig_at != sig_now),
        "member_count_at_write": int(topic.get("summary_member_count") or 0),
        "member_count_now": len(set(str(m) for m in now_ids)),
        "signature": sig_now,
        # Echo of the four-field contract (FRAS) (None if not written) -- the
        # host uses it to confirm the structure persisted, and to know
        # whether this summary recorded "category/relations/conclusion/
        # constraints" separately.
        "has_fras": bool(topic.get("summary_fras")),
        "fras": topic.get("summary_fras") or None,
        "title_kind": str(topic.get("title_kind") or ""),
    }


def representative_ids(
    topic: dict[str, Any],
    items_by_id: dict[str, dict[str, Any]],
    limit: int,
) -> list[str]:
    """Ordering of summary material: **closest to the centroid** first.

    Why not by time: the material serves "what is this topic about"; a
    representativeness question is answered directly by a representativeness
    ordering. Time order would push the earliest batch (possibly already
    overturned by later records) to the front, and they are exactly the most
    token-hungry. When evolution matters, the host can recall by span
    itself.

    Only members **currently alive** are returned (the items_by_id passed in
    is already active-only).
    """
    ids = [m for m in (topic.get("member_ids") or []) if m in items_by_id]
    if not ids or limit <= 0:
        return []
    norms: dict[str, np.ndarray] = {}
    for mid in ids:
        u = _unit(items_by_id[mid].get("vec"))
        if u is not None:
            norms[mid] = u
    c = _centroid([m for m in ids if m in norms], norms)
    scored: list[tuple[float, str]] = []
    for mid in ids:
        v = norms.get(mid)
        if v is None:
            continue
        s = float(np.dot(c, v)) if c is not None else 0.0
        scored.append((s, mid))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    return [mid for _s, mid in scored[:limit]]

def pending_summary_ids(
    topics: dict[str, dict[str, Any]],
    min_size: int | None = None,
    include_stale: bool = True,
    exclude_low_info: bool | None = None,
    defaults: TopicDefaults | None = None,
) -> list[str]:
    """Ids of topics that are "still worth summarizing" (map order,
    deterministic).

    A four-part test; all must hold to count as pending:
      1) not archived (topics whose members were all deleted do not count);
      2) ``size >= min_size`` (default :data:`TOPIC_MIN_SIZE_FOR_SUMMARY`);
      3) no host summary yet, **or** a summary exists but the member set
         changed (``stale``);
      4) **not a fake cluster** (skip when ``low_info`` is true; see
         :func:`info_score`).

    Part 3 is the key to "resuming": summaries are a scarce asset; one
    already written on an unchanged basis **must not be redone** -- redoing
    only trades a confirmed good conclusion for one bout of model noise.

    🔴 **Part 4 must come with self-reporting**: the topics it blocks
    **disappear** from the to-summarize list; what the host cannot see, it
    does not know exists ⇒ the caller must report
    :func:`low_info_topic_ids`'s count (``pending_excluded_low_info``).
    Old layers missing the ``low_info`` field are **always allowed through**
    (never judged, never misjudged).

    Args:
        exclude_low_info: Whether to exclude fake clusters. None = per
            :data:`ENV_TOPIC_LOW_INFO` (default on), overridable by
            ``defaults``'s :attr:`TopicDefaults.low_info`.
        d: The :class:`TopicDefaults` injection pack (assembled by the brain
            composition root).
    """
    lo = TOPIC_MIN_SIZE_FOR_SUMMARY if min_size is None else int(min_size)
    flag = _def(defaults).low_info if exclude_low_info is None \
        else bool(exclude_low_info)
    out: list[str] = []
    for t in ordered(topics):
        if str(t.get("status") or "") == STATUS_ARCHIVED:
            continue
        if len(t.get("member_ids") or []) < lo:
            continue
        if flag and t.get("low_info"):
            continue
        kind = str(t.get("summary_kind") or "")
        if kind == SUMMARY_KIND_HOST:
            sig_now = signature_of(t.get("member_ids") or [])
            sig_at = str(t.get("summary_signature") or "")
            if not include_stale or not (sig_at and sig_at != sig_now):
                continue
        out.append(str(t.get("id")))
    return out


def host_summary_stats(topics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarization progress self-report: how many topics the host
    summarized, how many have a changed member basis.

    These two numbers are the acceptance wording of "curation progress" --
    without them it is all gut feeling. ``host_stale`` matters most: it
    counts the topics "summarized, but members changed later" -- the batch
    that most needs to go back to the host for a rewrite.
    """
    kinds: dict[str, int] = {"host": 0, "extractive": 0, "none": 0}
    stale = 0
    with_fras = 0
    low_info = 0
    for t in topics.values():
        k = str(t.get("summary_kind") or "none")
        kinds[k] = kinds.get(k, 0) + 1
        if t.get("summary_fras"):
            with_fras += 1
        if t.get("low_info"):
            low_info += 1
        if (k == SUMMARY_KIND_HOST
                and t.get("summary_signature")
                and t.get("summary_signature")
                != signature_of(t.get("member_ids") or [])):
            stale += 1
    kinds["host_stale"] = stale
    # Fake-cluster count. It is in stats as a **hard requirement**, not
    # decoration: these clusters are kept out of pending; unreported, the
    # host would believe "everything is summarized".
    kinds["low_info"] = low_info
    # Count of host summaries carrying the four-field structure. Free text
    # remains legal; this cell only makes the "structure coverage" visible --
    # when summaries need machine processing per domain someday, no structure
    # means no processing.
    kinds["with_fras"] = with_fras
    return kinds
