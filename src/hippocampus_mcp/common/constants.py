# -*- coding: utf-8 -*-
"""The lowest-level support module in the package: threshold constants + pure functions with no ``self``.

**Imports only the standard library**, nothing else from the package -- the precondition for being depended on at the
same time by ``brain`` / ``map`` / ``file_registry`` etc.; any reverse dependency produces an inverted hierarchy or a
circular import. Split out of the package root into ``common/`` (a product of the layering refactor).

Symbols are **completely independent** of HippoBrain instance state: fully evaluated before the class exists, so moving
them introduces no new dependency direction.

``brain`` keeps ``from ..common.constants import *`` for external import compatibility -- 20+ scripts outside the
package do ``from hippocampus_mcp.brain import NAV_TAG_KEY`` and the like; not a single line of that may break.

⚠️ ``__all__`` must explicitly list the **underscore-prefixed** symbols (``_now`` / ``_nid`` / ``_STOPWORDS`` /
``_env_int`` ...): ``import *`` does not export them by default; a missing one is a runtime NameError, exploding only
on certain code paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

# **Same name** as before the split: the logger inside these functions was the brain module's logger; hardcoding the
# name keeps the log behavior bit-for-bit equivalent after the move (no logger rename, no handler change).
logger = logging.getLogger("hippocampus_mcp.brain")

# ── Threshold constants (directly tunable; most values measured on real libraries) ──────────────────
RELEVANCE_HARD_FLOOR = 0.30   # hard similarity gate: below this, candidates are dropped outright
MMR_LAMBDA = 0.7              # MMR relevance-diversity tradeoff
# MMR "literal strong-hit exemption": candidates ranked in the BM25 single-channel top N are exempt from the diversity
# penalty. Origin (measured on a Chinese novel library, 9359 entries): "顾父知情三拍是哪几章" hits the ch43-bearing
# correct answer via BM25 top1 / raw-vector top5 / full pipeline with MMR off, but the **full pipeline with MMR on**
# misses it -- MMR folded the entry near-synonymous with the already-selected ones but carrying a concrete chapter
# number / figure as redundant; yet multiple phrasings of one fact are exactly what a strongly-constrained writing
# project most needs (what gets folded away is usually the entry with the anchor / chapter number / amount). Only the
# BM25 top N (default 3) are exempt -- MMR is not switched off wholesale: the wider the exemption, the closer to
# "no diversity", and top_k fills up with near-identical phrasings of one topic.
MMR_LITERAL_EXEMPT_TOPN = 3
# R1-B4 (2026-11-08, decided route "score scaling") dimension-scale fix for
# ``_mmr_select``: the relevance it receives is a fused-domain total
# (RRF + 0.1*vec + entity bonus + UR boost) spanning ~0.057-0.227 on a real
# library, while the penalty term's max_sim is a native 0-1 cosine. Without
# rescaling, 0.7*relevance loses to 0.3*max_sim by ~3.5x and sim=0 literal
# rescues systematically displace semantically similar candidates (measured:
# 1.73 sim=0 placeholders per query, -34% mean top-5 similarity vs the legacy
# additive path; C=10 recovers 84% of the gap). The fix multiplies relevance
# by this constant:
#   val = MMR_LAMBDA * (MMR_REL_SCALE * score) - (1 - MMR_LAMBDA) * max_sim * penalty
# Calibration anchor -- recalibrate with ONE paired probe when the upstream
# score scale moves (RRF k=60 in retrieval/bm25.py, the brain's bm25_weight,
# the fetch_k = 4*top_k pool multiplier, or the entity-bonus / UR-boost
# magnitudes): per-query median of (top-pool cosine / top relevance) ~= 6.9;
# sensitivity inflection at C ~= 7, flat within [10, 25] (±25% band spread
# <= 0.006), so 10 keeps margin above the knee. Gate: HIPPOCAMPUS_MMR_SCALE_FIX
# (semantic A, default on); with it off the raw score is used verbatim.
# Design doc: docs/design/R1-B4_MMR量纲缺陷_拍板材料_20261108.md (repo-only).
MMR_REL_SCALE = 10.0
# Similarity floor for ingest's "possibly related" echo. No ceiling needed -- entries at or above the dedup threshold
# (0.85) have already been **merged into** the same entry and echoed via ``merged_into``; superseded/archived memories
# do not merge, and they are precisely where "old framings" hide -- so no ceiling: older superseded versions stay visible.
INGEST_NEIGHBOR_MIN_SIM = 0.60
# Miss-log threshold. **Do not raise it on a hunch**: measured on the Chinese novel library (9359 entries), true hits'
# top1 similarity is only 0.32~0.53 ("ch43 的知情线锚点" hit at sim=0.356); setting 0.50 would record true hits as misses.
# This takes recall's own hard gate -- below it means even the hard gate was not passed; only that truly counts as "not found".
MISS_LOG_THRESHOLD = RELEVANCE_HARD_FLOOR
# How many miss-log entries to keep at most (oldest dropped beyond that). The miss log is for attribution analysis,
# not an audit log; without a cap it bloats silently in a long-running library.
MISS_LOG_MAX = 500

# How many global open intents (open_topics) to keep at most (oldest trimmed beyond that). Same reasoning as miss_log:
# append-only + uncapped bloats silently in a long-running library (field feedback 2026-10-31).
OPEN_TOPICS_MAX = 50

# --- Deletion ledger: a tombstone log for "true deletions" in the derived layer ---
# Union cannot express subtraction. Derived copies (entities / profiles) can rely on "re-deriving from currently-alive
# memories after merge", but an **independent ledger** such as miss_log, once deleted, is truly deleted -- no source
# to re-derive from. Reproduced: process A calls miss_log(clear=True), empties it and persists to disk -> process B
# (still holding an old snapshot) does one save, and the 3 entries come back in full, with no error anywhere along the way.
# Hence a deletion ledger: who deleted which key in which layer and when, so merges can block on it.
LAYER_DEL_FILE = "brain_layerdel.json"
# How many "layer::key" tombstones the ledger keeps at most. **Not** unlimited: beyond the cap the oldest are dropped,
# and a dropped tombstone can in theory let an even older snapshot resurrect the corresponding entry -- a **known and
# accepted** tradeoff (a 30-day-old snapshot persisted only now would resurrect more than just miss_log).
LAYER_DEL_MAX_KEYS = 5000
# How long tombstones are kept (seconds). Trace-like entries are capped anyway
# (miss_log: 500 entries); 30 days far exceeds the reasonable lifetime of any in-memory snapshot.
LAYER_DEL_TTL = 30 * 86400.0
ENV_LAYER_DEL = "HIPPOCAMPUS_LAYER_DEL_LEDGER"
# How many days of per-day activity counts to keep. Append-only counter, bloats silently if never trimmed; trend
# watching generally looks at the last week or two, so 60 days is enough for trends.
DAILY_COUNT_KEEP_DAYS = 60
# How many days of daily_activity brain_stats echoes back. **The storage cap (DAILY_COUNT_KEEP_DAYS=60) and the echo window
# are two different things**: the "kept_days=1 is too short" report was a misreading -- that was **only 1 day of data in
# stock** (the feature had just shipped), not a cap of 1; what made the "last 14 days activity curve" unreachable was the
# echo window being hardcoded to 7 days. Default 14; override via env HIPPOCAMPUS_DAILY_ACTIVITY_DAYS or stats(daily_days=).
DAILY_ACTIVITY_DEFAULT_DAYS = 14
ENV_DAILY_ACTIVITY_DAYS = "HIPPOCAMPUS_DAILY_ACTIVITY_DAYS"
# Tag key for navigation-like content: INDEX / tables of contents / header rows relate only tangentially to any query
# (in measurements the first hit was often taken by an INDEX row). The import side tags such content with this key;
# the retrieval side discounts it via :attr:`HippoBrain.nav_penalty`. **Tagging only, never deleting content** -- the
# table of contents itself must stay findable, it just must not outrank the body text.
NAV_TAG_KEY = "_nav"
# Default time-to-live for parallel-session registrations: sessions not ended within this window no longer appear in
# active_sessions. **An expiry is mandatory** -- this project has plenty of collision scars, but a crashed /
# force-closed session holding its slot forever would make the registry a new obstacle instead of a help. Visibility
# only, no locking: registering stops no one from writing; it only lets people **take a look before writing**.
SESSION_TTL_SECONDS = 7200
# PIN_IMPORTANCE is imported from store.memory (single source of truth; do not redefine here)
SUPERSEDE_SIM = 0.72          # similarity floor for "same fact probably updated" (must also satisfy content-word overlap)
# Similarity **ceiling** for contradiction candidates: pairs at or above this are usually two copies of the same fact,
# left to write-side dedup rather than re-reported to the host as "contradictions" here.
CONTRADICTION_SIM_CEIL = 0.85
# Contradiction candidates kept per entity bucket / entries finally reported to the host. Pairwise comparison within a
# bucket is O(k²): the candidate count must be **independent of bucket size**, else on a large library merely accumulating
# candidates eats memory and time (the old implementation built dicts for all 12,812 entries, then threw them away).
CONTRADICTION_BUCKET_MAX = 12
CONTRADICTION_REPORT_MAX = 5
# Content-word overlap gate. Originally 0.3 -- under the Chinese bigram metric "two passages
# from the same document" easily exceeds it, making supersede misfire massively during bulk
# imports (28 suspected supersede pairs measured, the overwhelming majority wrong).
SUPERSEDE_OVERLAP = 0.6       # content-word overlap rate required when there is no shared entity
ENTITY_BONUS = 0.12           # boost when a query entity hits a memory entity (bridges the semantic gap)
DORMANT_DAYS = 180            # days uncalled -> archive

# Bulk tag updates: one call can touch thousands of entries, and **a wrong change has no automatic undo** (tags are a
# flat dict; the server does not know "what it was before"). So write operations are disabled by default, and above
# this entry count an explicit confirm is required -- both gates are deliberate: real libraries hold many entries with
# no tags at all, and "one command to stamp a domain onto the whole library" sounds tempting, but one wrong filter
# condition means thousands of junk tags, and recall filtering by tag then **silently returns empty**, so the host
# concludes "this domain has no content".
UPDATE_TAGS_CONFIRM_ABOVE = 100
UPDATE_TAGS_SAMPLE_MAX = 10

# update_tags change ledger (borrows the prior work's "before/after images + ledger" core, **without adopting its
# approval state machine**): the explicitly acknowledged gap was "a wrong change has no automatic undo" -- tags are
# a flat dict, the server keeps no pre-change snapshot, so undoing meant manually reverse-engineering from the before
# images of the first 10 sample entries. Now closed: every **execution** records one jsonl entry with full
# before/after images, and undo rolls back only when "current value == recorded after" (fail-closed: entries modified
# by a third party in between are skipped and self-reported, never clobbering later edits). Why not a layer file: it
# is an append-only stream (same positioning as save_ledger.jsonl), takes no part in layer merging; a ring buffer
# prevents unbounded growth.
TAG_OPS_FILE = "tag_ops.jsonl"
# Rewrite-history ledger: a before-image stream for L1 text/importance rewrites and text dropped by merges (sidecar,
# append-only, same positioning as tag_ops; the shape mirrors store/tag_ops.py verbatim). Ring size is wired up the
# same way as above: config.mem_history_ring, injected into the manager at construction (no monkeypatch surface, no
# core snapshot needed).
MEM_HISTORY_FILE = "mem_history.jsonl"
# Ring size. Only the **pure default** lives here: env reading is consolidated into config.tag_ops_ring, snapshotted
# by brain/core.py at import time (``core.TAG_OPS_RING``). Why the consumer is core and not this file: this file is
# an L0 pure leaf and must not depend on config, while core must preserve the name ``TAG_OPS_RING`` that tests
# monkeypatch. The single definition site of this value (together with the trim threshold) is ``config.tag_ops_ring``;
# no copy is kept here -- same-named constants in two places will drift sooner or later, and it no longer has any
# potential consumer.

# **Coexisting-framing annotation** at retrieval time: an append-only, never-delete library inevitably retains "an old
# and a new version of the same fact" (supersede off / older data predating that mechanism / pairs that never matched
# back then); on a hit both are returned side by side, and if the host cannot tell which is more recent it risks
# picking the wrong value. Near neighbors inside the result set get **annotated** -- **never folded, never
# re-ordered**: an old framing may itself be a historical fact being cited; which one is correct must be decided by
# the authoritative source, not adjudicated on the user's behalf by the retriever -- folding is simply the wrong
# approach. The criterion reuses _is_restatement (the same semantics as write-side supersede detection).
CONFLICT_ANNOTATE_MAX = 12    # skip pairwise detection above this result-set size (O(n²) protection, not a correctness switch)

# **Relevance floor** for deletion = recall's own similarity hard gate. It serves exactly one purpose: filtering out
# the entries recall admitted via **BM25-only recall** -- those never entered the vector channel's top_k and their
# similarity is exactly 0.0. Measured (Chinese novel library, 9359 entries): with short queries, 44% of non-first
# candidates have sim=0; with long-sentence queries, 58%. They are the main force of "surface-word overlap only".
# **Do not raise this value to filter semantic noise -- that approach does not hold**: true-hit similarity drifts
# wholesale with query length.
#   long-sentence query band: 0.632–0.781   contemporaneous noise (vector-backed): 0.453–0.542
#   short query        band: 0.461–0.580   contemporaneous noise (vector-backed): 0.276–0.387
# The "true hit vs noise" gap differs on the two sides (~0.09 long, ~0.07 short); a fixed threshold is doomed to
# over-kill one side and under-block the other. What is actually stable is **relative dominance**: the target is
# always #1 and well clear of #2. So the main defense against wrong deletion is the single-entry rule below, not the
# threshold.
FORGET_MIN_SIMILARITY = RELEVANCE_HARD_FLOOR
# Without explicit authorization, one forget_by_query deletes at most 1 entry. "Delete the entry that was recorded
# wrong" is almost always semantically singular; the extra slots in top_k are exactly where wrong deletions come
# from (reproduced: top_k=3 mixed in 2 nearly-unrelated real memories and all were physically deleted).
FORGET_SINGLE_BY_DEFAULT = True

# L3 profile capacity caps: prevent "thousands of pinned memories" from flooding stable_facts into noise. Pinned
# memories mostly come from bulk-import misjudgments; uncapped, consolidate would stuff them all into the profile.
PROFILE_SLOT_MAX = 40         # max entries kept in a single profile slot (including stable_facts)
PROFILE_ENTITY_MAX = 30       # max number of entity profile slots (top N by mention_count)

# ── Independent budget for block_4 (related entity states) ──────
# The old implementation started "from the entities of **hit memories**": the larger the top_k, the more entities get
# hit, the noisier block_4. Measured (Chinese novel library, top_k=10): this block spread out the dozen-plus entities
# dragged in by 10 hits, crowding out the entity states that actually mattered -- spliced into the system prompt it
# was pure noise. Narrowing rules: 1) **Query-text entities first** (most on-topic); hit entities are taken only from
# the top BLOCK4_HIT_ENTITY_TOPN hits as a supplement (the fallback when the query names no entity); 2) the block has
# its own caps (entity count / facts per entity / total chars), **not mixed with the global token_budget** --
# otherwise it still eats the whole budget on a first-come-first-served basis.
BLOCK4_MAX_ENTITIES = 3        # how many entities block_4 shows at most
BLOCK4_MAX_FACTS_PER_ENTITY = 3  # how many facts each entity carries at most
BLOCK4_MAX_CHARS = 400         # character cap for the whole block_4 (coarse gate)
BLOCK4_HIT_ENTITY_TOPN = 3     # supplement entities only from the top N hits

# ── Pinned slot block_pinned_memories ──────────
# Two defects, both of which bit in practice: 1) count hardcoded to 3 -- when iron rules / gates / chapter-level
# facts are all pinned, a chapter-level card elevated later (importance 0.8) squeezes book-level iron rules out of
# the top three -- "pinned" ends up meaning not pinned; 2) ordering purely by importance -- pinned entries'
# importance is already concentrated at the high end and close together, so that ordering actually ranks "who got
# elevated first", not "who relates to the task at hand". Hence the count is configurable
# (session_start(pin_limit=) > env > default) and ordering became **task-relevance first**, importance as tiebreaker
# within a tier -- the only discrimination comes from "does it match this task".
PIN_BLOCK_LIMIT_DEFAULT = 3
ENV_PIN_BLOCK_LIMIT = "HIPPOCAMPUS_PIN_BLOCK_LIMIT"

# ── Knowledge map slot block_knowledge ──────────────────
# The only slot in the preactivation package that can answer "what does this library actually contain": L1 is flat
# facts, L2 only holds session-level summaries that go dormant right after being written, and L3 is hot entries, not
# knowledge. A hardcoded count fails both ways: a small library is fully visible in 8 entries, while in a large
# library 8 entries show only the 8 biggest baskets. So this is also env-configurable (same semantics as
# PIN_BLOCK_LIMIT; no new MCP tool parameter -- the tool surface is injected into host context with tools/list each
# session, and one more parameter is a permanent token cost).
KNOWLEDGE_BLOCK_LIMIT_DEFAULT = 8
ENV_KNOWLEDGE_BLOCK_LIMIT = "HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT"
KNOWLEDGE_BLOCK_CLIP = 70      # per-line character cap (title + scale + one-liner)
# Which slot "pinned memories" land in within the L3 profile (prune_stale_profile tells
# the validation rules apart by slot name)
PROFILE_SLOT_STABLE = "stable_facts"
# Confidence calibration for profile graduation entries (domain may only import common, so the two coefficients live
# here; the slot name and the three thresholds are graduation-validation specific and reside in domain/constants.py)
PROFILE_STRENGTH_BASE = 0.3   # confidence = min(0.9, BASE + STEP·strength):
PROFILE_STRENGTH_STEP = 0.06  #   strength 3→0.48, 5→0.60, 10→0.90 (self-consistent ordering within the slot)

# ingest_batch per-call cap: the point of a bulk entry point is "one call saves N round trips", but stuffing tens
# of thousands of entries in at once blows up memory and the size of a single persist (persisting is a full
# overwrite). Erroring out beyond the cap so the caller splits into batches is better than "swallow it quietly, then OOM".
INGEST_BATCH_MAX = 2000

# ── Project overview and status card ──────────────────────────────
# The two slots of the session_start preactivation package: a library-level overview + a compact projection of the
# current state. Position: when starting work, an agent needs not only "information at the query point" but also
# "what this library is as a whole and what state it is in now". Two hard disciplines (mirroring the prior work):
# 1. pure derivation, no retrieval, no clustering -- the hot path (every session start) does not pay the 2-3 seconds
# of first-time combing; 2. only signals that "cannot be derived but hurt when wrong" go in (FRAS's S-field
# discipline); derivable quantities (total entry counts etc.) serve only as anchoring landmarks, not expanded.
OVERVIEW_TOP_ENTITIES = 3        # how many hot entities the overview card carries
OVERVIEW_TOP_TAG_KEYS = 3        # how many main tag keys the overview card carries
OVERVIEW_RECENT_DAYS = 30        # overview card "last N days delta" window
STATUS_SPARKLINE_DAYS = 14       # status card sparkline window (days)
STATUS_MISS_DAYS = 7             # status card "last N days misses" window
STATUS_CONTRADICTION_CAP = 20    # contradiction scan budget cap (pairs); at the cap, report "≥N"
SPARKLINE_MIN_BUCKETS = 4        # no sparkline below this bucket count (avoids false precision)
ENV_OVERVIEW_BLOCK = "HIPPOCAMPUS_OVERVIEW_BLOCK"          # on by default; 0/false/no/off disables
ENV_STATUS_BLOCK = "HIPPOCAMPUS_STATUS_BLOCK"
ENV_STATUS_CONTRADICTIONS = "HIPPOCAMPUS_STATUS_CONTRADICTIONS"  # contradiction signal switch (on by default)

# ── Task-level "issue profile" ──────────────────
# task_hits are point hits; the issue profile is "the structure this task touches": related knowledge points /
# related entities / matching open intents / recent miss gaps in that area. Matching discipline (all deterministic,
# no model -- the hot path pays no inference cost): knowledge points / entities -- **intersect** task_hits'
# memory_id with topic member_ids / entity memory_ids (same id space; exact; no tunable threshold); open intents /
# misses -- text **token intersection** (task words + anchor tag values + anchor entity names), same family of
# semantics as resolve_intent's keyword-in-text.
TASK_TOPIC_MAX = 3           # how many related knowledge points the profile card carries
TASK_ENTITY_MAX = 5          # how many related entities it carries
TASK_INTENT_MAX = 3          # how many matching open intents it carries
TASK_MISS_MAX = 3            # how many related misses it carries
TASK_MISS_DAYS = 14          # time window for miss gaps (longer than the status card's 7 days:
                            # at task level you want "what this area keeps lacking lately")
TASK_TOKEN_MIN_LEN = 2       # minimum task-word length (single Chinese characters carry no discriminative power)
TASK_PROFILE_CLIP = 28       # per-line title truncation for the profile card (same discipline as the
                            # knowledge map slot: never clip scale numbers, only titles)
ENV_TASK_PROFILE_BLOCK = "HIPPOCAMPUS_TASK_PROFILE_BLOCK"  # on by default; 0/false/no/off disables

# Sparkline character gradient (low→high). Block heights progress; no color is used -- a text slot has no color
# available, and it is the "shape" that carries the message (which days were active), not exact values.
_SPARK_CHARS = "▁▃▅▇"


def sparkline(values: list[int]) -> str:
    """Map a series of counts onto a ▁▃▅▇ text chart (status card "activity" row).

    Empty sequence / all zeros / fewer than :data:`SPARKLINE_MIN_BUCKETS` buckets -> empty string: callers degrade to
    plain numbers on that. Two or three points cannot draw a shape, and drawing one anyway is false precision (same
    discipline as "all green does not mean every sentence is right").

    Normalization is **relative** (min→▁, max→▇): what it conveys is the shape of "which days were relatively more
    active"; with all-equal values it gives the mid-level ▅ rather than pretending there was fluctuation.
    """
    vals = [max(0, int(v)) for v in (values or [])]
    if len(vals) < SPARKLINE_MIN_BUCKETS or not any(vals):
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    top = len(_SPARK_CHARS) - 1
    out: list[str] = []
    for v in vals:
        idx = 2 if span == 0 else int(round((v - lo) / span * top))
        out.append(_SPARK_CHARS[max(0, min(top, idx))])
    return "".join(out)


def format_age(days: int | None) -> str:
    """Render days-since-update as card-facing text (today/yesterday/Nd ago/Nmo ago).

    ``None`` (no usable timestamp) becomes ``"unknown"``; the card must not guess. Months use a 30-day approximation:
    this is a freshness signal, not a calendar.
    """
    if days is None:
        return "unknown"
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 30:
        return "%dd ago" % days
    return "%dmo ago" % (days // 30)


# Task token splitting: Latin/digit runs are whole words; CJK runs are split into **bigrams** -- Chinese has no
# spaces, and treating "查朱由检煤山的事" as one word for substring matching means it never matches anything, while
# bigrams (煤山/由检) are the best tradeoff between discriminative power and cost within deterministic splitting.
# No segmentation model is introduced -- matching must be deterministic and reproducible (the same task yields the
# same card on two session starts), and the hot path pays no inference cost.
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

# Stoplist of high-frequency generic bigrams: without blocking, bigrams like 「的一」「什么」
# would mark half the library's open intents as "related" -- card noise is far worse than
# seeing a few characters fewer (same root as the knowledge map slot's truncation discipline).
_TOKEN_STOP_BIGRAMS = frozenset({
    "的一", "了的", "是我", "是不", "我们", "你们", "他们", "这个", "那个",
    "什么", "怎么", "为什", "可以", "需要", "进行", "相关", "如下", "上述",
    "目前", "当前", "问题", "情况", "如下",
})


def _add_token(tokens: list[str], w: str) -> None:
    w = str(w).strip().lower()
    if len(w) >= TASK_TOKEN_MIN_LEN and w not in tokens and w not in _TOKEN_STOP_BIGRAMS:
        tokens.append(w)


def task_tokens(task: str, extra: list[str] | None = None) -> list[str]:
    """Token set for task matching (deterministic, no model).

    Two sources: the task text + ``extra`` (anchor memories' tag values, anchor entity names). Everything is
    lowercased, deduplicated, and order-preserving (stable order keeps card output stable: the same task yields a
    character-identical card on two session starts). CJK runs additionally produce all bigrams; single characters
    (length < ``TASK_TOKEN_MIN_LEN``) are dropped -- single Chinese characters carry no discriminative power.
    """
    tokens: list[str] = []

    def expand(chunk: str) -> None:
        cjk = _CJK_RUN_RE.findall(chunk)
        if cjk and "".join(cjk) == chunk:
            # Pure CJK run: bigrams only (the whole run has no discriminative power; see module comment)
            for run in cjk:
                for i in range(len(run) - 1):
                    _add_token(tokens, run[i:i + 2])
            return
        _add_token(tokens, chunk)
        for run in cjk:  # mixed Latin/CJK run: need both the whole words and bigrams of the CJK part
            for i in range(len(run) - 1):
                _add_token(tokens, run[i:i + 2])

    for chunk in _TOKEN_SPLIT_RE.split(task or ""):
        if chunk:
            expand(chunk)
    for w in extra or []:
        if w is None:
            continue
        _add_token(tokens, w)
        expand(str(w))  # long entity names (e.g. 「煤山自缢事件」) also get bigram supplementation
    return tokens


# ── Entity origin tiers ──────────────────────────────────────
# Measured (Chinese novel document library): the precision of Chinese heuristic extraction is about 16% -- of 832
# candidates only 112 were genuine 《》 title-bracket names; the rest were phrases like "这笔账没法记", and 67% of
# those came from the ASCII double-quote pattern (`"..."`). In Chinese, quotation marks express emphasis / quotation
# / dialogue, not proper names; `《》` is Chinese's only "work title" marker. Therefore: 1) for Chinese text,
# quote-based patterns are disabled and only 《》 is kept (see `_heuristic_entities`); 2) every entity records an
# origin; only trusted origins participate in "retrieval boost / profile / contradiction detection", while heuristic
# output is demoted to "candidates" -- still stored and queryable, but kept out of decision chains.
ENTITY_ORIGIN_MANUAL = "manual"        # host explicitly passed entities=[...] (most trusted)
ENTITY_ORIGIN_LEXICON = "lexicon"      # hit the project's proper-noun lexicon
ENTITY_ORIGIN_TITLE = "title"          # proper name inside 《title brackets》 (Chinese's only work-title marker)
ENTITY_ORIGIN_SPACY = "spacy"          # spaCy NER (English path, decent quality)
ENTITY_ORIGIN_HEURISTIC = "heuristic"  # Chinese syntax/quote heuristics (the main noise source)

# Trusted origins: participate in retrieval boost / profile / contradiction detection.
#   - 《》 gets its own tier: it is Chinese's unambiguous "work title" delimiter, not in the same trust class as
#     "guessing proper names from sentence patterns"; measured over the novel finalization corpus, of 224 candidates
#     184 came from 《》 (real book titles such as 几何原本、某史料、中国货币史), and only 40 were bare-pattern noise.
#     Demoting 《》 as well would throw away the most reliable signal, so it alone gets the trusted tier.
#   - spaCy stays trusted -- its output quality is in a different league from the Chinese heuristics, and it only
#     triggers on the English path.
TRUSTED_ENTITY_ORIGINS = frozenset({
    ENTITY_ORIGIN_MANUAL, ENTITY_ORIGIN_LEXICON, ENTITY_ORIGIN_TITLE,
    ENTITY_ORIGIN_SPACY,
})
# Credibility ranking: when one entity is hit by multiple origins, take the highest (upgrade only, never downgrade).
_ORIGIN_RANK = {
    ENTITY_ORIGIN_HEURISTIC: 0,
    ENTITY_ORIGIN_SPACY: 1,
    ENTITY_ORIGIN_TITLE: 2,
    ENTITY_ORIGIN_LEXICON: 3,
    ENTITY_ORIGIN_MANUAL: 4,
}

# Skip spaCy once the CJK share exceeds this -- en_core_web_sm is an English model;
# on Chinese documents it neither finds entities nor justifies the run.
_CJK_SKIP_SPACY_RATIO = 0.20

# Update-marker vocabulary. Calibration case (see tests/test_i5_supersede_policy.py): a set of genuine update sentences
# contained only 「迁移到/改到/提高到/升级为/延长为/切换到」 -- none of which the old table recognized; closely related terms
# were added along with them; narrative high-frequency words deliberately left out are documented in that test file too.
UPDATE_MARKERS = (
    "now ", "no longer", "used to", "changed", "moved to", "switched",
    "recently", "these days", "nowadays", "现在", "不再", "改成", "换了",
    "以前", "之前是", "目前",
    "迁移到", "改到", "提高到", "升级为", "延长为", "切换到", "改为",
    "换到", "提升为", "延长到", "降到", "降为", "调整为", "调整到",
    "变更为", "更新为", "updated to", "upgraded to", "increased to",
    "extended to",
)

# Markdown structure characters: masses of table rows / tree-directory lines in documents
# get mistaken for proper names by the heuristic rules
# (measured: 11 of the Top15 entities were fragments like 「├」「│  ├─ 伏笔表.md」).
_MD_TABLE_LINE = re.compile(r"^[ \t]*\|.*$", re.M)          # table rows
_MD_CODE_FENCE = re.compile(r"```.*?```", re.S)              # fenced code blocks
_MD_TREE_CHARS = "│├└─┌┐┘┌┬┴┼═║╔╗╚╝■□●○◆◇→←↑↓"
_ENTITY_BAD_CHARS = set(
    "|`\\/*#<>[]{}()（）【】〈〉「」“”\"'=+~^$%&@:;,.!?，。！？、…—　"
    + _MD_TREE_CHARS
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "i",
    "my", "you", "your", "it", "that", "this", "to", "of", "in", "on", "at",
    "for", "and", "or", "we", "he", "she", "they", "his", "her", "their",
}

# ── lexicon_suggest's built-in stopwords ────────────────
# Only **cross-project generic** document jargon / collaboration boilerplate goes here -- they are perfect
# "high-frequency + cross-file" candidates yet never proper names, and they stably occupy the top of
# lexicon_suggest. Project proper names (person / place / brand names) never go in; those belong in
# `<lexicon directory>/entity_lexicon_stop.txt`.
# Matching semantics: a candidate gram is dropped if it **is a substring of** any entry here (including itself). So
# only complete phrases need to be written -- one entry "用户定案" simultaneously suppresses slices like "用户定" /
# "案", without having to hand-enumerate prefix fragments the way the external stopword file does.
_DEFAULT_LEXICON_STOP = (
    # chapter/document self-reference
    "本章", "本节", "本卷", "上一章", "下一章", "前文", "后文", "全文",
    "正文", "原稿", "开篇", "结尾", "文末", "文中",
    # metadata tags (title/status kind)
    "备注", "说明", "摘要", "概要", "结论", "总述", "详见", "参见", "同上",
    "时间线", "待核清单", "核清单", "合订本", "要点版", "快照", "记录格式",
    "第一次", "第二次", "第三次", "第四次", "最后一次",
    # human-AI collaboration boilerplate
    "用户定案", "用户确认", "用户拍板", "用户批阅", "用户令", "已落正文",
    "已确认", "待确认", "已定案", "待定", "已完成",
    # generic function words / deictic phrases (cross-project high frequency, never proper names)
    "为什么", "什么", "怎么", "这样", "那样", "一句话", "一件事", "本文件",
    "长出来", "一下子", "一会儿", "一样",
)

# Grams that are purely "Chinese numerals (+ optional 第/measure word)" (第五节 / 十二年 / 第一次 / 三段 / 一部分):
# naturally high-frequency, cross-project generic, never proper names. Implemented as a **rule, not an enumeration**,
# sparing the listing of every measure word. Only matches "the whole string is numerals (+ one measure word)" --
# real proper names like "三宝" / "万历年" are unaffected.
_ORDINAL_RE = re.compile(
    r"^第?[一二三四五六七八九十百千万零两]{1,4}"
    r"(?:章|节|卷|次|年|月|日|版|轮|条|项|篇|段|部分|阶段|批|步)?$"
)
# First-character pre-filter: only grams starting with a digit or "第" are worth running
# through the regex (skips the vast majority of calls).
_ORDINAL_HEADS = frozenset("第一二三四五六七八九十百千万零两")


def _now() -> float:
    return time.time()


def _nid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def strip_internal_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Strip the candidate entry's **internal fields** (underscore-prefixed), modifying it in place and returning it.

    These keys are for the retrieval pipeline's own use: ``_emb`` (current model vector, used for MMR diversity),
    ``_literal`` (BM25 literal strong hit, used for the MMR penalty exemption), ``_sel_sim`` (similarity cache
    reused by MMR). They must not appear in tool returns -- they waste payload and leak implementation detail.

    This used to be two hand-written ``pop`` calls, and ``_sel_sim`` leaked out (only triggered when that MMR run
    over-selected; invisible otherwise). Now it is a **unified prefix-based cleanup**, so any internal key added
    later is covered too. Public output fields (such as ``matched_by``) carry no underscore and are unaffected.
    """
    for k in [k for k in item if k.startswith("_")]:
        item.pop(k, None)
    return item


# ==================================================================
# Session identity routing
# ==================================================================

# Fallback route when no client info is present; all unsigned calls share it -- equivalent to before the rework.
DEFAULT_SESSION_KEY = "__default__"


def _session_key(client: str | None) -> str:
    """Session-route key. Empty / blank -> default route; otherwise keyed by the client string."""
    c = (client or "").strip()
    return c or DEFAULT_SESSION_KEY


# ==================================================================
# Writer visibility
# ==================================================================

WRITER_DIRNAME = "writers"
# How long a registration may go without a heartbeat before expiring (seconds). Crashes /
# force-kills never deregister, so expiry is the only cleanup.
WRITER_TTL_DEFAULT = 3600.0


def pid_alive(pid) -> bool:
    """Whether the process is still alive. **Read-only probe**; never signals the other process.

    ``os.kill(pid, 0)`` cannot be used on Windows -- CPython's Windows ``os.kill`` wraps
    TerminateProcess, so passing 0 means ending the other process with exit code 0.
    Instead this opens a handle with OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) to peek.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                k32.CloseHandle(h)
                return True
            return False
        except Exception:  # noqa: BLE001 -- fall back to "assume alive" when ctypes is unavailable
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def scan_writers(wdir: str, ttl: float) -> tuple[list[dict], int]:
    """Scan the writer-registration directory; return (active registrations, stale ones removed)."""
    now = time.time()
    active: list[dict] = []
    stale = 0
    try:
        names = os.listdir(wdir)
    except OSError:
        return [], 0
    for nm in names:
        if not nm.endswith(".json"):
            continue
        path = os.path.join(wdir, nm)
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:  # noqa: BLE001 -- unreadable registrations are always treated as stale and removed
            stale += 1
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        if not isinstance(info, dict):
            stale += 1
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        seen = info.get("updated_at") or 0
        if (not pid_alive(info.get("pid"))
                or (now - float(seen)) > ttl):
            stale += 1
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        active.append(info)
    active.sort(key=lambda x: x.get("updated_at") or 0)
    return active, stale


# ---------------------------------------------------------------------------
# Save ledger
# ---------------------------------------------------------------------------
# Keep a ledger of "saving" itself: instead of reverse-engineering afterwards via l0_dangling what got lost, provide
# a save timeline visible on the spot. Coverage: the layer-file whitelist / four quadrants / dual clocks
# (at_mono for intervals, at_wall via _reference_time).
# Append-only jsonl + ring buffer: never a second write-amplification source (one save's ledger < 4KB).
LEDGER_FILENAME = "save_ledger.jsonl"
# The single definition site of this value (together with the trim threshold) is ``config.save_ledger_ring``; exposed
# at runtime as ``core.LEDGER_RING``; this file keeps no copy. Cap on lines re-examined from the ledger tail during
# startup reconciliation (used by fail lines to find the begin with the same seq).
_LEDGER_TAIL_KEEP = 16

# Layer content fingerprint switch (= the derived layer's dirty set). On by default. Escape hatch: if some layer's
# json.dumps cost crushes the write path (very large library), setting 0 returns to the old behavior of "rewriting
# all 10 layers in full every time" -- same semantics, just the write amplification comes back. The switch itself
# lives in ``config.layer_fp``: this file reads no env (L0 pure leaf; env reading is consolidated into config).

# ---------------------------------------------------------------------------
# Source-file signature layer -- the fact source for drift auditing
# ---------------------------------------------------------------------------
# Borrows the prior work's "entry-bound source SHA-256 + four-way verify classification", adapted to this project's
# shape: the write side (ingest) records {mtime_ns, size, sha1} for any source that resolves to disk, and
# drift_audit compares it against the disk's current state, answering "the file changed but the memory was not
# updated". Design tradeoffs (relative to the prior work):
#   * **Keyed by source label, not by memory entry** -- one md import yields hundreds of L1 entries; recording a
#     signature per entry is pure waste; and sources disappear (forget), so only keying by label makes "this source
#     as a whole went stale" expressible;
#   * **If the content facts did not change, do not touch the layer** -- a sig_at timestamp on the signature entry
#     would make every ingest change layer content => every save writes to disk (write amplification). When
#     unchanged, the old entry stays put, and "verified recently" lives only in the in-memory cache;
#   * **sha1 streamed** (1MB chunks) -- large files must not be read whole into memory;
#   * sources that do not resolve to a path (session ids / free text) are **not recorded** -- forcing a fake
#     "sha1=?" signature would only muddy the unbaselined pool.
SOURCE_SIG_FILE = "brain_sourcesig.json"
# Signature-computation cache cap: (path, mtime_ns, size) -> sha1. Bulk-importing 500 entries from the same source file
# hashes it only once. Pure performance cache, not a fact source (the fact source is the layer file); FIFO eviction beyond the cap.
SOURCE_SIG_CACHE_MAX = 256


# The switch itself lives in ``config.source_sig`` (same as above: this file reads no env).

# L0 auto-persist trigger granularity -- how many **new raw** entries accumulate before a write to disk. 0 = persist
# every entry (safest, most write amplification); negative = auto persist off, only self-reporting the backlog. The
# value lives in ``config.l0_autosave_every``; the actual constant is in core.py.
# Cap for ``_guess_importance``. Every current branch tops out at 0.75, so this line is **effectively a no-op** (old
# code wrote the literal min(score, 0.95), which was misread as "auto-judgment can reach 0.95"). Extracting it as a
# named constant with this note makes the ceiling obvious when a higher tier is added later -- anything above 0.95
# belongs to "repeatedly confirmed by the user", which surface-word rules must not award.
_GUESS_IMPORTANCE_CAP = 0.95
# Only the layer files below count (**never hardcode the count** -- when a layer is added, edit here; a hardcoded
# number becomes a lie the moment a layer is added). The library directory also holds .bak-* / writers/ / lexicon.json
# / the ledgers themselves; counting those would misreport backups and external tools' touches as "I failed to write".
LEDGER_LAYER_FILES = (
    "brain_raw.json", "brain_entities.json", "brain_summaries.json",
    "brain_profile.json", "brain_sessions.json", "brain_intents.json",
    "brain_topics.json", "brain_tombstones.json", "brain_misslog.json",
    "brain_counters.json", "brain_knowledge.json", LAYER_DEL_FILE,
    SOURCE_SIG_FILE,
    "memories.json", "memories.vectors.npz",
    # LNN state's primary format is npz; old libraries' .pt stays on the whitelist (during the compatibility-read window
    # both may exist in the library directory; missing either would misreport "someone else's state file" as an unwritten layer).
    "lnn_state.npz", "lnn_state.npz.meta.json", "lnn_state.npz.bak",
    "lnn_state.pt", "lnn_state.pt.meta.json",
)

__all__ = [
    "RELEVANCE_HARD_FLOOR",
    "MMR_LAMBDA",
    "MMR_LITERAL_EXEMPT_TOPN",
    "MMR_REL_SCALE",
    "INGEST_NEIGHBOR_MIN_SIM",
    "MISS_LOG_THRESHOLD",
    "MISS_LOG_MAX",
    "OPEN_TOPICS_MAX",
    "LAYER_DEL_FILE",
    "LAYER_DEL_MAX_KEYS",
    "LAYER_DEL_TTL",
    "ENV_LAYER_DEL",
    "DAILY_COUNT_KEEP_DAYS",
    "DAILY_ACTIVITY_DEFAULT_DAYS",
    "ENV_DAILY_ACTIVITY_DAYS",
    "NAV_TAG_KEY",
    "SESSION_TTL_SECONDS",
    "SUPERSEDE_SIM",
    "CONTRADICTION_SIM_CEIL",
    "CONTRADICTION_BUCKET_MAX",
    "CONTRADICTION_REPORT_MAX",
    "SUPERSEDE_OVERLAP",
    "ENTITY_BONUS",
    "DORMANT_DAYS",
    "UPDATE_TAGS_CONFIRM_ABOVE",
    "UPDATE_TAGS_SAMPLE_MAX",
    "TAG_OPS_FILE",
    "MEM_HISTORY_FILE",
    "CONFLICT_ANNOTATE_MAX",
    "FORGET_MIN_SIMILARITY",
    "FORGET_SINGLE_BY_DEFAULT",
    "PROFILE_SLOT_MAX",
    "PROFILE_ENTITY_MAX",
    "BLOCK4_MAX_ENTITIES",
    "BLOCK4_MAX_FACTS_PER_ENTITY",
    "BLOCK4_MAX_CHARS",
    "BLOCK4_HIT_ENTITY_TOPN",
    "PIN_BLOCK_LIMIT_DEFAULT",
    "ENV_PIN_BLOCK_LIMIT",
    "KNOWLEDGE_BLOCK_LIMIT_DEFAULT",
    "ENV_KNOWLEDGE_BLOCK_LIMIT",
    "KNOWLEDGE_BLOCK_CLIP",
    "PROFILE_SLOT_STABLE",
    "PROFILE_STRENGTH_BASE",
    "PROFILE_STRENGTH_STEP",
    "INGEST_BATCH_MAX",
    "OVERVIEW_TOP_ENTITIES",
    "OVERVIEW_TOP_TAG_KEYS",
    "OVERVIEW_RECENT_DAYS",
    "STATUS_SPARKLINE_DAYS",
    "STATUS_MISS_DAYS",
    "STATUS_CONTRADICTION_CAP",
    "SPARKLINE_MIN_BUCKETS",
    "ENV_OVERVIEW_BLOCK",
    "ENV_STATUS_BLOCK",
    "ENV_STATUS_CONTRADICTIONS",
    "TASK_TOPIC_MAX",
    "TASK_ENTITY_MAX",
    "TASK_INTENT_MAX",
    "TASK_MISS_MAX",
    "TASK_MISS_DAYS",
    "TASK_TOKEN_MIN_LEN",
    "TASK_PROFILE_CLIP",
    "ENV_TASK_PROFILE_BLOCK",
    "format_age",
    "sparkline",
    "task_tokens",
    "ENTITY_ORIGIN_MANUAL",
    "ENTITY_ORIGIN_LEXICON",
    "ENTITY_ORIGIN_TITLE",
    "ENTITY_ORIGIN_SPACY",
    "ENTITY_ORIGIN_HEURISTIC",
    "TRUSTED_ENTITY_ORIGINS",
    "_ORIGIN_RANK",
    "_CJK_SKIP_SPACY_RATIO",
    "UPDATE_MARKERS",
    "_MD_TABLE_LINE",
    "_MD_CODE_FENCE",
    "_MD_TREE_CHARS",
    "_ENTITY_BAD_CHARS",
    "_STOPWORDS",
    "_DEFAULT_LEXICON_STOP",
    "_ORDINAL_RE",
    "_ORDINAL_HEADS",
    "_now",
    "_nid",
    "strip_internal_fields",
    "DEFAULT_SESSION_KEY",
    "_session_key",
    "WRITER_DIRNAME",
    "WRITER_TTL_DEFAULT",
    "pid_alive",
    "scan_writers",
    "LEDGER_FILENAME",
    "_LEDGER_TAIL_KEEP",
    "SOURCE_SIG_FILE",
    "SOURCE_SIG_CACHE_MAX",
    "_GUESS_IMPORTANCE_CAP",
    "LEDGER_LAYER_FILES",
]
