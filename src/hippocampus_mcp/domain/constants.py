# -*- coding: utf-8 -*-
"""Constants specific to the domain layer.

``common/constants`` is a **shared foundation, not a dumping ground**: constants consumed
only by a single domain module (and its binding shell) do not belong there; they live in
this module. Same route as ``map/constants`` (map-layer constants already migrated out).

The first resident is O4 "access-driven profile promotion" -- consumers are
``domain/profile_ops`` and ``brain/core``'s binding shell (the shell imports explicitly,
to preserve the brain package's re-export name baseline G5).
"""
from __future__ import annotations

# ── Access-driven profile promotion -- the L1→L3 "graduation" slot ──────────────────────────
# Biological analogy: system-consolidation corticalization is driven by **repeated
# reactivation** (recall_strength is the reactivation count). Complementary to
# stable_facts (pin = the user explicitly said "this entry matters"): this slot collects
# entries that **nobody pinned but that keep being recalled and have lived long enough**
# -- access frequency votes on its own.
# Relation to the U3 "never auto-promote" red line: U3 governs the project→user
# **cross-library** promotion flow (explicit confirmation required before writing the
# user library); this slot is an **in-library** refresh of the L3 profile's display layer
# (L3 already refreshes automatically with consolidate) -- it writes no records, changes
# no state, never touches the user library.
PROFILE_SLOT_RECURRING = "recurring_facts"  # profile slot name where graduation entries land
PROFILE_STRENGTH_MIN = 3.0    # repeated-reactivation floor: recall_strength must reach this to qualify
PROFILE_STRENGTH_DAYS = 7.0   # stabilization period: alive >= this many days since write (new traces do not graduate)
PROFILE_STRENGTH_MAX = 20     # candidate cap per consolidation (top N by strength, descending)
# (The two confidence-calibration coefficients PROFILE_STRENGTH_BASE/STEP live in
#   common/constants -- same-layer cross-imports are banned within domain, so profile_ops
#   can only take them from common.)


__all__ = [
    "PROFILE_SLOT_RECURRING",
    "PROFILE_STRENGTH_MIN",
    "PROFILE_STRENGTH_DAYS",
    "PROFILE_STRENGTH_MAX",
]

# --- Sessions/topics (validation feedback B/C/D, 2026-11-01; consumer brain/session) ---

# Similarity dedup threshold (0~1) for appending topics. Similarity = the larger of the
# difflib ratio and the character-bigram overlap coefficient, computed after
# normalization (punctuation/separators/case); calibrated against the validator's
# verbatim pairs of reworded/re-delimited synonymous topics: positive examples 0.63,
# unrelated pairs ≤0.21, so 0.60 leaves margin on both sides. At or above the threshold
# the append is skipped and the hit is echoed (dedup_hits); **0 = off**.
# Design: 验证反馈B_C_D设计提案_20261101.md §D (visible fuzziness).
TOPIC_SIM_DEDUP = 0.60
ENV_TOPIC_SIM_DEDUP = "HIPPOCAMPUS_TOPIC_SIM_DEDUP"

# Session idle auto-recycle threshold (seconds): an active session idle beyond this value
# is automatically marked ended (tagged recycled) when shown in active_sessions --
# only the status moves, never the data. **0/negative = off**.
# Design: 验证反馈B_C_D设计提案_20261101.md §C-3.
SESSION_IDLE_RECYCLE = 24 * 3600
ENV_SESSION_IDLE_RECYCLE = "HIPPOCAMPUS_SESSION_IDLE_RECYCLE"
