# -*- coding: utf-8 -*-
"""Constants for the information map layer and the file registry layer.

Moved out of ``common/constants.py``: these constants serve only the
``map/`` subpackage and ``file_registry``; keeping them in the bottom-most
package-wide constants would blur the responsibilities of the "map layer"
and the "common layer".

Constraint: this module **imports nothing from the package**; it only
defines constants, for explicit import by ``map/`` internals and the
``brain`` orchestration layer.
"""

# ── Map index line + file registry ─────────────────────────────────────
#
# A map = a memory carrying retained tag keys:
# {"地图": "合约", "条目": "CN2601", "状态": "验证集"}.
# One key, one entry; updates go through update_tags -- the early form did
# not open a new map_set, but the "one key, one entry" discipline is the
# convention: writing a second entry under the same 条目 key means the old
# state and the new state compete (the core negation proposition of design
# doc §2); the index line's ⚠️ and the session_end checklist do the
# reminding.
# Design-review tightening: a same-key guard on the ingest side
# (_guard_map_entry_dup), update_tags refreshing the timestamp when it
# modifies a map entry (the retreat path for Gate 4's only-in-must-exit
# rule), and legacy vs registry same-key-different-status conflicts reported
# on the index line -- the discipline upgrades from "agent self-restraint"
# to mechanism.
MAP_NAME_KEY = "地图"          # retained tag key: which map this memory belongs to
MAP_ENTRY_KEY = "条目"         # retained tag key: the entry key (CN2601 / chr1 / --use-avx2)
MAP_STATUS_KEY = "状态"        # retained tag key: the entry's current status (验证集 / 开 / 已组装)
MAP_INDEX_MAX = 8              # how many maps the index line may carry at most (beyond that, a "wait for N maps" self-report)
MAP_STALE_DAYS = 14            # index-line ⚠️ threshold (Gate 4 · stale-surface
                               # exposure: only-in-must-exit; marking too
                               # eagerly equals not marking -- the H3 stale lesson)

# ── Map layer sidecar (storage for map_set/get/list/verify) ────────────────
# Map={name,desc,entries,updated_at}；
# Entry={key,status,fields,updated_at,updated_by,truth_source,source_sig,
#        drift,note}. Entries carry no vector and never enter recall -- a map
# is state, not evidence (§2).
MAPS_VERSION = 1
MAP_DETAIL_MAX_ROWS = 20            # per-map row cap for full text of L1-relevant
                                    # maps (§6 default 20 rows; beyond that, a
                                    # "wait for N entries" self-report, never a
                                    # silent swallow)
MAP_DETAIL_MAX_MAPS = 3             # how many maps L1 may pack per call (broad
                                    # task words may hit several; beyond that, a
                                    # "wait for N maps" self-report)

# ── Fully automatic verify for code-switch maps ─────────────────────────────
# The map-level ``auto`` recipe declares the truth source and extractor;
# maps without auto are never touched by automatic verify.
MAP_AUTO_BUDGET_S = 1.0          # wall-clock budget for one automatic verify; when
                                 # exceeded, mark incomplete and self-report,
                                 # never silently degrade to "all clean"
MAP_AUTO_MAX_SOURCE_BYTES = 1024 * 1024   # 1MB cap per source (build scripts are far
                                          # smaller; anything bigger is not read --
                                          # same precedent as C1a not hashing big files)

# ── Gate 3: mention-level drift detection ─────────────────────────────────
# Criteria in the map/mention.py docstring: an active memory newer than the
# entry's updated_at, and **the same sentence** mentions both the entry key
# and a status word different from the current status. The status word list
# comes from the map's own entries' status sets (no built-in guessed
# lexicon).
MAP_MENTION_BUDGET_S = 1.0       # wall-clock budget per run; when exceeded, an
                                 # incomplete self-report
MAP_MENTION_MAX_RECORDS = 2000   # at most this many memories per round (over that,
                                 # self-report the truncation)

# Sidecar write wait (seconds). The sidecar critical section is milliseconds,
# but the counterpart to those "milliseconds" may be file_registry's full
# scan (seconds) -- a user's explicit write is worth waiting a little for,
# instead of immediately returning busy and making the user retry.
# Automatic patrols (the session_start hot path) do not get this: if they
# cannot grab it they skip the round.
MAP_WRITE_WAIT_S = 2.0

# File registry: register project files, recording the file signature at the
# time of the last update into hippo; if the on-disk signature and the
# registered signature disagree ⇒ the file's information in hippo is stale.
# Division of labor among the three timestamps (different semantics, must
# not be mixed):
#   first_seen  -- first scanned (registration time)
#   last_synced -- when hippo's content came to reflect this signature of
#                  the file (written at sync time)
#   last_checked-- time of the last scan (scanning itself does not change
#                  the synced semantics)
# The verdict uses **signature comparison + hash arbitration** (on an
# mtime_ns + size mismatch, read the content hash once to confirm)
# rather than "file mtime > sync wall clock":
#   ① git checkout / robocopy / touch reset mtime but not content -- equal
#     hashes mean no false positive (wall-clock comparison would condemn an
#     entire repository as stale);
#   ② no dependence on two machines' clocks agreeing, and no fear of synced
#     clocks rolling back.
#   Files whose signature did not change cost zero reads (the normal scan
#   only stats); the hash is paid once only on mismatch.
ENV_FILE_REGISTRY_INTERVAL = "HIPPOCAMPUS_FILE_REGISTRY_INTERVAL"  # auto rescan interval (hours)
ENV_FILE_REGISTRY_IGNORE = "HIPPOCAMPUS_FILE_REGISTRY_IGNORE"      # extra ignore globs (comma-separated)
ENV_FILE_REGISTRY_MAX_FILES = "HIPPOCAMPUS_FILE_REGISTRY_MAX_FILES"  # max files covered
ENV_FILE_REGISTRY_BUDGET_S = "HIPPOCAMPUS_FILE_REGISTRY_BUDGET_S"    # wall-clock budget for coverage (seconds)
# Whether sync verifies against the source-signature layer that "hippo
# really ingested the current content" (default on).
# Turning it off = back to the trust system (synced because it says so) --
# reserved for cases like "confirmed the content is in memory but did not
# get there via ingest"; the CLI's --no-verify is its one-shot equivalent.
ENV_FR_SYNC_VERIFY = "HIPPOCAMPUS_FR_SYNC_VERIFY"
FILE_REGISTRY_CHECK_HOURS = 6   # default auto rescan interval (the session_start
                                # hot path cannot full-stat on every opening --
                                # interval gating + budget fallback)
FILE_REGISTRY_MAX_FILES = 20000  # max files per scan
FILE_REGISTRY_BUDGET_S = 1.0     # wall-clock budget per scan (seconds); when
                                 # exceeded, mark scan_complete
                                 # =false and self-report "scan incomplete" on
                                 # the card, never a silent swallow
FILE_REGISTRY_DEFAULT_IGNORES = (
    # Directory names (any parent-directory hit prunes the whole branch):
    # VCS/dependencies/caches/IDE/local environments
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "_probe_out", ".workbuddy",
    # File-suffix globs: binary/ephemeral/self-referential (the library's own
    # data files)
    "*.pyc", "*.pyo", "*.db", "*.db-wal", "*.db-shm", "*.wal", "*.log",
    "*.tmp", ".DS_Store", "Thumbs.db",
)

__all__ = [
    "ENV_FILE_REGISTRY_BUDGET_S",
    "ENV_FILE_REGISTRY_IGNORE",
    "ENV_FILE_REGISTRY_INTERVAL",
    "ENV_FILE_REGISTRY_MAX_FILES",
    "ENV_FR_SYNC_VERIFY",
    "FILE_REGISTRY_BUDGET_S",
    "FILE_REGISTRY_CHECK_HOURS",
    "FILE_REGISTRY_DEFAULT_IGNORES",
    "FILE_REGISTRY_MAX_FILES",
    "MAPS_VERSION",
    "MAP_AUTO_BUDGET_S",
    "MAP_AUTO_MAX_SOURCE_BYTES",
    "MAP_DETAIL_MAX_MAPS",
    "MAP_DETAIL_MAX_ROWS",
    "MAP_ENTRY_KEY",
    "MAP_INDEX_MAX",
    "MAP_MENTION_BUDGET_S",
    "MAP_MENTION_MAX_RECORDS",
    "MAP_NAME_KEY",
    "MAP_STALE_DAYS",
    "MAP_STATUS_KEY",
    "MAP_WRITE_WAIT_S",
]
