# -*- coding: utf-8 -*-
"""B2-3 P1: pure methods of HippoBrain (never touch instance state)

This cluster was moved over from ``brain.py`` verbatim.

The criterion is AST-verified, not gut feeling: **neither reads/writes
``self.*`` nor calls any ``self`` method** -- they have nothing to do with
HippoBrain's instance state, so extracting them is necessarily equivalent.

A Mixin instead of "pass the manager in": ``self.manager`` has 62 read
sites; any signature change would trigger mass call-site edits, which is
the classic way a big refactor spirals out of control. The Mixin still
writes ``self``, resolved via the MRO, so call sites don't change a line.

⚠️ ``_is_restatement`` is **deliberately left** in brain.py:
``scripts/test_batch2.py`` locates it via
``src.split("def _is_restatement")[1]`` as a structural guard; moving it
away raises IndexError. That guard is the "single source of truth for the
criterion", worth more than the location of 32 lines of code.
"""

from __future__ import annotations

from ..common.constants import *  # noqa: F401,F403  (constants table: see common/constants.__all__)
from ..common.constants import logger
from ..config import load_config
import hashlib
import json
import os
import re
from typing import Any
import numpy as np
from ..domain import policies as _policies
from ..domain import topic as topic_mod
from ..store.memory import CognitiveType
from ..store import source_sig as _source_sig

#: Default count for the user-scope inheritance slot: fallback when
#: param==0 and env is empty. Deliberately lives with its consumer
#: (resolve_user_scope_limit), not common/constants -- the architecture
#: guard pins a hard line-count ceiling on constants, so new constants
#: follow the guard's guidance into domain/config/consumer modules. Same
#: literal default as ``Config.user_scope_limit``; parity pinned by
#: tests/test_u1_user_scope.py C6 (test_config_parity P7, same criterion).
USER_SCOPE_LIMIT_DEFAULT = 3

#: Default count for the task-directory slot: fallback when param==0 and
#: env is empty. The directory is an **index**, not the body (design §3.4);
#: 8 entries cover "the one at hand + the recent few". Older tasks beyond
#: that are truncated after ``task_directory.total`` self-reports the
#: total, and the host can query more as needed. Same literal default as
#: ``Config.task_directory_limit``; parity pinned by
#: tests/test_u5_task_directory.py C6 (test_config_parity P7, same criterion).
TASK_DIRECTORY_LIMIT_DEFAULT = 8


class _PureMixin:
    """B2-3 P1: pure methods of HippoBrain (never touch instance state)"""

    # ==================================================================
    # Project proper-noun lexicon (coverage report, High 6)
    # ==================================================================
    @staticmethod
    def _resolve_lexicon_path(explicit: str | None, persist_dir: str | None) -> str | None:
        """Resolve the lexicon file path: explicit param > env > ``<memory library>/entity_lexicon.txt``.

        The default path is **computed fresh each time, never cached** -- if
        entity_lexicon.txt is placed into the memory library only after
        service startup (a very common order of operations), a cached "None"
        would make ``reload_lexicon()`` read an empty lexicon forever, with
        no warning at all.
        """
        if explicit:
            return explicit
        env_path = load_config().entity_lexicon or None
        if env_path:
            return env_path
        if persist_dir:
            default_lex = os.path.join(persist_dir, "entity_lexicon.txt")
            if os.path.isfile(default_lex):
                return default_lex
        return None




    @staticmethod
    def _lexicon_mtime_of(path: str | None) -> float | None:
        """The lexicon file's mtime; returns None if missing / unreadable (**never raised as an error**)."""
        try:
            return os.path.getmtime(path) if path and os.path.isfile(path) else None
        except OSError:
            return None




    @classmethod
    def _parse_lexicon_text(cls, text: str) -> tuple[set[str], dict[str, str], list[str]]:
        """Parse lexicon text -> ``(term set, alias map, problem lines)``.

        Line syntax (UTF-8, ``#`` starts a comment, blank lines ignored)::

            顾言                     # plain term
            子安 => 顾言              # alias line: left=alias, right=canonical name
            陆暄 -> 暄儿             # arrow form is equivalent (-> / → / ⇒)

        **Both aliases and canonical names go into the term set** (both must
        participate in substring hits; if "the alias is never even
        recognized" there is nothing to normalize). The alias map is stored
        separately as ``{alias_lower: canonical_verbatim}`` for
        normalization on both the extraction side and the query side.

        Why aliases: **multiple names for the same entity exist in any
        knowledge base** (courtesy name / abbreviation / full name / old
        name / codename). Novel characters are just the most visible case
        -- querying "子安" missing the entity boost for "顾言", with
        profiles stored separately and unconnected, is a generic defect,
        not any single project's private matter.
        """
        terms: set[str] = set()
        aliases: dict[str, str] = {}
        problems: list[str] = []
        for lineno, raw in enumerate((text or "").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = cls._ALIAS_SEP.search(line)
            if not m:
                terms.add(line.lower())
                continue
            alias = line[:m.start()].strip().lstrip("#").strip()
            canon = line[m.end():].strip()
            if not alias or not canon:
                # Not silently dropped: a malformed line must be made known (otherwise it shows up as "the alias never took effect")
                problems.append(f"line {lineno} alias syntax incomplete: {line!r}")
                continue
            if alias.lower() == canon.lower():
                terms.add(canon.lower())
                continue
            terms.add(alias.lower())
            terms.add(canon.lower())
            prev = aliases.get(alias.lower())
            if prev is not None and prev.lower() != canon.lower():
                problems.append(
                    f"第 {lineno} line alias {alias!r} 重复指向不同规范名"
                    f"（已保留 {prev!r}, ignored {canon!r}）"
                )
                continue
            aliases[alias.lower()] = canon
        return terms, aliases, problems



    @classmethod
    def _load_lexicon(cls, path: str | None) -> set[str]:
        """Read the lexicon file, return the lowercased term set (aliases included).

        For document corpora, guessing Chinese proper nouns via heuristic
        rules is doomed to drown in noise; instead have the project supply
        its own proper-noun list (person names / place names / trade names /
        book titles) -- a hit links immediately, both accurate and cheap.
        """
        return cls._read_lexicon(path)[0]



    @classmethod
    def _read_lexicon(cls, path: str | None) -> tuple[set[str], dict[str, str]]:
        """Read the lexicon file, return ``(term set, alias map)``. Unreadable means empty."""
        if not path or not os.path.isfile(path):
            return set(), {}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            logger.warning("lexicon read failed %s: %s", path, e)
            return set(), {}
        terms, aliases, problems = cls._parse_lexicon_text(text)
        for p in problems:
            logger.warning("lexicon %s %s", path, p)
        return terms, aliases




    @staticmethod
    def _preprocess_for_entities(text: str) -> str:
        """Strip markdown structure before extracting entities (coverage report, High 6).

        In document-like corpora, table rows (``| 卷末（1664 | ...``), tree
        listings (``│  ├─ 伏笔表.md``) and fenced code blocks get mistaken
        for proper nouns by the heuristic rules -- measured: 11 of the
        Top15 entities were such fragments. These characters carry no
        "entity" semantics; removing them wholesale before extraction
        removes the noise source.
        """
        if not text:
            return ""
        t = _MD_CODE_FENCE.sub(" ", text)
        lines = []
        for line in t.split("\n"):
            if _MD_TABLE_LINE.match(line):
                continue
            # Tree-drawing symbols inside a line are stripped too ("├─ 顾言" → "顾言")
            if any(ch in line for ch in _MD_TREE_CHARS):
                line = "".join(" " if ch in _MD_TREE_CHARS else ch for ch in line)
            lines.append(line)
        return "\n".join(lines)



    @staticmethod
    def _cjk_ratio(text: str) -> float:
        """Proportion of CJK characters among "non-whitespace" characters."""
        if not text:
            return 0.0
        dense = [c for c in text if not c.isspace()]
        if not dense:
            return 0.0
        cjk = sum(1 for c in dense if "\u4e00" <= c <= "\u9fa5")
        return cjk / len(dense)




    @classmethod
    def _clean_entity_name(cls, name: str) -> str | None:
        """Clean extracted candidate entities: drop function words, structural characters and code-like noise.

        The measured junk samples from coverage report High 6 (「├」「│  ├─ 伏笔表.md」「H05」
        「本文件（人读）」「章自检（2026」) are all intercepted here:
        structural characters go through ``_ENTITY_BAD_CHARS`` (markdown is
        already stripped in preprocessing; this catches the residue), and
        code-like names go through ``_CODE_LIKE``.
        """
        n = (name or "").strip()
        if not n:
            return None
        # Chinese names capped at 12 chars; pure-English names relaxed to 30 (New York City / SentenceTransformer)
        limit = 30 if n.isascii() else 12
        if not (1 < len(n) <= limit):
            return None
        # Contains punctuation / markdown structural chars -> noise (entity names should not carry these)
        if any(c in _ENTITY_BAD_CHARS for c in n):
            return None
        # Pure function words / phrases containing function words -> noise
        if any(c in n for c in cls._NOISE_CHARS):
            return None
        if re.fullmatch(r"[\d\W_]+", n):
            return None
        if cls._CODE_LIKE.match(n):
            return None
        return n




    @classmethod
    def _title_entities(cls, text: str) -> list[str]:
        """Extract proper names inside `《书名号》` (Chinese book-title brackets). The only explicit work-title marker in Chinese; given its own confidence tier."""
        names = []
        for m in cls._TITLE_RE.finditer(text):
            w = (m.group(1) or "").strip()
            if 1 < len(w) <= 12:
                names.append(w)
        return names



    @classmethod
    def _heuristic_entities(cls, text: str, cjk: bool = True) -> list[str]:
        """Heuristic entity extraction for Chinese and English (fallback when
        spaCy is unavailable or the text is not English).

        Chinese patterns: 我叫X / X品类 / X公司 (sentence-shape cues).
        English patterns: additionally enables quote patterns + runs of
        consecutive capitalized words.

        `《书名》` is not handled here -- it is extracted separately by
        ``_title_entities`` and tagged with the title tier.

        Args:
            cjk: whether the text is predominantly Chinese. When true,
                **quote-like patterns are disabled** (see the note on
                ``_QUOTE_PATTERNS`` -- Chinese quotation marks are emphasis
                markers, not proper-noun markers).
        """
        names: list[str] = []
        patterns = list(cls._STATEMENT_PATTERNS)
        if not cjk:
            patterns.extend(cls._QUOTE_PATTERNS)
        for pat in patterns:
            for m in re.finditer(pat, text):
                w = (m.group(1) or "").strip()
                if 1 < len(w) <= 12:
                    names.append(w)
        if not cjk:
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
                w = m.group(1).strip()
                if w.lower() not in _STOPWORDS and len(w) > 1:
                    names.append(w)
        return names




    @staticmethod
    def _entity_origin(ent: dict[str, Any]) -> str:
        """Read the entity's origin. **Defaults to heuristic** -- older library
        serializations lack this field; treat it as untrusted, never grant
        trust just because "the data is old"."""
        return ent.get("origin", ENTITY_ORIGIN_HEURISTIC)



    @classmethod
    def _entity_trusted(cls, ent: dict[str, Any]) -> bool:
        """Whether this entity is trusted (participates in retrieval boosts / profile / contradiction detection)."""
        return cls._entity_origin(ent) in TRUSTED_ENTITY_ORIGINS




    @staticmethod
    def _drift_row(source: str, category: str,
                   ref: dict[str, int] | None,
                   sig: dict[str, Any] | None,
                   cur: dict[str, Any] | None,
                   diff: list[str] | None = None) -> dict[str, Any]:
        """One row of the drift audit (implementation lives in
        ``store.source_sig.drift_row``; this is a same-name delegate -- the
        C1b drift audit moved to the store side as a cluster with
        source_drift, ``_PureMixin``'s surface unchanged, behavior pinned by
        tests/test_c1_source_drift.py)."""
        return _source_sig.drift_row(source, category, ref, sig, cur, diff)




    @staticmethod
    def _split_facts(text: str) -> list[str]:
        """Split a passage of text into short sentences usable as memories."""
        text = text.strip()
        if not text:
            return []
        parts = re.split(r"(?<=[。！？!?；;\n])|(?<=[.!?])\s+", text)
        facts, buf = [], ""
        for p in parts:
            p = (p or "").strip()
            if not p:
                continue
            buf = f"{buf} {p}".strip() if buf else p
            if len(buf) >= 24:
                facts.append(buf)
                buf = ""
        if buf:
            if facts and len(buf) < 18:
                facts[-1] = f"{facts[-1]} {buf}".strip()
            else:
                facts.append(buf)
        return facts or [text]



    @staticmethod
    def _guess_cognitive_type(text: str, speaker: str) -> str:
        t = text.lower()
        if speaker in ("user", "assistant"):
            if any(m in t for m in ("we ", "我们一起", "上次我们", "i did", "you did")):
                return CognitiveType.EXPERIENCE.value
        if any(m in t for m in ("i think", "i feel", "probably", "maybe", "我觉得", "我认为", "可能")):
            return CognitiveType.OPINION.value
        return CognitiveType.WORLD.value



    @staticmethod
    def _guess_importance(text: str) -> float:
        """Heuristic importance (the decision core moved to
        ``domain.policies.guess_importance``; this is a same-name delegate,
        behavior pinned by 44 golden test cases)."""
        return _policies.guess_importance(text)




    @staticmethod
    def _tokens_of(s: str) -> set[str]:
        """Chinese bigrams + English words (delegates to ``domain.policies.tokens_of``)."""
        return _policies.tokens_of(s)



    @classmethod
    def _overlap_of_sets(cls, ta: set[str], tb: set[str]) -> float:
        """M2: delegates to ``domain.policies.overlap_of_sets`` (the cls param is kept to guard against signature changes)."""
        return _policies.overlap_of_sets(ta, tb)



    @classmethod
    def _token_overlap(cls, a: str, b: str) -> float:
        """Content-word overlap ratio (used to judge whether two memories are talking about the same thing)."""
        return cls._overlap_of_sets(cls._tokens_of(a), cls._tokens_of(b))




    def _resolve_daily_days(self, daily_days: int = 0) -> int:
        """Daily-window entry count: explicit param > env
        ``HIPPOCAMPUS_DAILY_ACTIVITY_DAYS`` > 14.

        ``0`` = follow env / default. **No upper-bound validation** -- pass a
        big number to see everything; ``daily_counts`` itself holds at most
        :data:`DAILY_COUNT_KEEP_DAYS` days anyway.
        """
        if daily_days:
            return max(int(daily_days), 1)
        raw = load_config().daily_activity_days
        if raw:
            try:
                v = int(raw)
                if v > 0:
                    return v
            except ValueError:
                pass
        return DAILY_ACTIVITY_DEFAULT_DAYS




    def _mmr_select(self, items: list[dict[str, Any]], top_k: int,
                    scale_fix: bool = True) -> list[dict[str, Any]]:
        """MMR selection: balances relevance and diversity, suppressing duplicate content from dominating the ranking.

        ``scale_fix`` (R1-B4, 2026-11-08, default on = the shipped behavior):
        multiply the fused-domain relevance by :data:`MMR_REL_SCALE` so the
        ``MMR_LAMBDA * relevance`` term competes with the native 0-1 cosine
        penalty term on the same order of magnitude (calibration anchor in
        the constant's note). ``False`` restores the pre-fix raw-score
        formula bit-for-bit (rollback tier).
        """
        selected: list[dict[str, Any]] = []
        pool = list(items)
        while pool and len(selected) < top_k:
            best, best_val = None, -1e9
            for cand in pool:
                relevance = float(cand["score"])
                if scale_fix:
                    relevance *= MMR_REL_SCALE
                if not selected:
                    val = relevance
                else:
                    max_sim = 0.0
                    for sel in selected:
                        s = cand.get("_sel_sim", {}).get(sel["memory_id"])
                        if s is None:
                            a, b = cand.get("_emb"), sel.get("_emb")
                            if a is None or b is None:
                                # Candidates missing the current model's vector (BM25-only hits) get no diversity penalty
                                continue
                            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
                            s = float(a @ b / denom)
                            cand.setdefault("_sel_sim", {})[sel["memory_id"]] = s
                        max_sim = max(max_sim, s)
                    # Literal strong hits (top N via the BM25 single route) are exempt from the diversity penalty:
                    # otherwise "the entry carrying the chapter number/amount" gets folded away by near-synonyms (see the note at the constants).
                    penalty = 0.0 if cand.get("_literal") else 1.0
                    val = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim * penalty
                if val > best_val:
                    best, best_val = cand, val
            selected.append(best)  # type: ignore[arg-type]
            pool.remove(best)  # type: ignore[arg-type]
        return selected




    @staticmethod
    def _render_blocks(blocks: dict[str, list[str]], token_budget: int,
                       item_max_chars: int = 200) -> str:
        labels = {
            "block_1_current_hits": "📋 Current relevant memories",
            "block_2_pending": "❓ Open todos / questions",
            "block_3_timeline": "📍 Timeline (recent sessions)",
            "block_4_entity_state": "🧠 Related entity state",
            "block_5_open_topics": "🔄 Previously unfinished",
        }
        budget = max(int(token_budget) * 2, 200)  # Chinese is roughly 2 chars/token

        # Render whole blocks first, then pack by priority (block_1 → block_5).
        # Coverage report, Low 13: it used to be a hard `out[: budget*2]` truncation,
        # which cut a line into half a sentence -- the host received broken content
        # like "【📋 当前相关信息】\n  - 顾言表字是". Now **only whole blocks/whole
        # items are dropped**; never cut in the middle of a line.
        rendered: list[tuple[str, list[str]]] = []
        for k, label in labels.items():
            vals = [v for v in blocks.get(k, []) if v]
            if not vals:
                continue
            clip = item_max_chars if item_max_chars > 0 else None
            rendered.append((f"【{label}】",
                             [f"  - {v[:clip] if clip else v}" for v in vals]))

        parts: list[str] = []
        used = 0
        for header, lines in rendered:
            # 1) If the whole block fits, place it whole
            block = header + "\n" + "\n".join(lines)
            gap = 2 if parts else 0
            if used + gap + len(block) <= budget:
                parts.append(block)
                used += gap + len(block)
                continue
            # 2) Whole block doesn't fit → keep the header, pack item by item until full (items stay complete)
            kept: list[str] = []
            here = len(header) + gap
            if used + here <= budget:
                for line in lines:
                    if used + here + len(line) + 1 > budget:
                        break
                    kept.append(line)
                    here += len(line) + 1
            if kept:
                parts.append(header + "\n" + "\n".join(kept))
                used += here
            # 3) Even the header doesn't fit → drop the whole block (later blocks are lower priority, so naturally not placed either)
        return "\n\n".join(parts)




    def resolve_knowledge_limit(self, knowledge_limit: int = 0) -> int:
        """Knowledge-map slot count: explicit param > env
        ``HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT`` > default 8.

        ``0`` = follow env / default; **negative = close the slot** (same
        semantics as ``pin_limit``).
        """
        if knowledge_limit:
            return max(int(knowledge_limit), 0)
        raw = load_config().knowledge_block_limit
        if raw:
            try:
                v = int(raw)
                if v != 0:
                    return max(v, 0)
            except ValueError:
                pass
        return KNOWLEDGE_BLOCK_LIMIT_DEFAULT




    def resolve_pin_limit(self, pin_limit: int = 0) -> int:
        """Pin slot count: explicit param > env ``HIPPOCAMPUS_PIN_BLOCK_LIMIT`` > default 3.

        ``0`` = follow env / default; **negative = close the slot** (same
        semantics as ``recent_limit``).
        """
        if pin_limit:
            return max(int(pin_limit), 0)
        raw = load_config().pin_block_limit
        if raw:
            try:
                v = int(raw)
                if v > 0:
                    return v
            except ValueError:
                pass
        return PIN_BLOCK_LIMIT_DEFAULT




    def resolve_user_scope_limit(self, user_scope: int = 0) -> int:
        """User-scope inheritance slot count, three-state (table copied line by line):

        ====================  ==========================================
        Input                 resolve_user_scope_limit
        ====================  ==========================================
        param > 0             param
        param < 0             0 (off)
        param == 0, env > 0   env
        param == 0, env < 0   0 (off)
        param == 0, env 0/empty/invalid  3 (default)
        ====================  ==========================================

        The **only** difference from :meth:`resolve_pin_limit` is a negative
        env: pin falls back to the default (the ``v > 0`` gate), while
        user_scope is designed as an explicit off -- ``0`` (off) and "back
        to default" must stay distinguishable, or the "injection can be
        turned off" discipline would have no env-side way to turn off.
        Empty/invalid env fallback is handled by ``_int`` in
        :func:`config.load_config` (the fallback field defaults to 3, same
        value as :data:`USER_SCOPE_LIMIT_DEFAULT`).
        """
        if user_scope:
            return max(int(user_scope), 0)
        raw = load_config().user_scope_limit
        if raw:
            return max(int(raw), 0)
        return USER_SCOPE_LIMIT_DEFAULT

    def resolve_task_directory_limit(self, task_directory: int = 0) -> int:
        """Task-directory slot count, three-state (U5, design §3.4): same family as ``user_scope``.

        ====================  ==================================
        Input                 resolve_task_directory_limit
        ====================  ==================================
        param > 0             param
        param < 0             0 (off)
        param == 0, env > 0   env
        param == 0, env < 0   0 (off)
        param == 0, env 0/empty/invalid  8 (default)
        ====================  ==================================

        Line-for-line isomorphic with :meth:`resolve_user_scope_limit`
        (including negative env = explicit off); the only difference is the
        default of 8. Empty/invalid env fallback is handled by ``_int`` in
        ``config.load_config`` (the fallback field defaults to 8, same value
        as :data:`TASK_DIRECTORY_LIMIT_DEFAULT`).
        """
        if task_directory:
            return max(int(task_directory), 0)
        raw = load_config().task_directory_limit
        if raw:
            return max(int(raw), 0)
        return TASK_DIRECTORY_LIMIT_DEFAULT




    @staticmethod
    def _tag_ops_brief(ops: list[dict[str, Any]],
                       limit: int = 5) -> list[dict[str, Any]]:
        """Brief of the ledger (guides the 'no such op' error: the host learns which ids are available)."""
        return [
            {"op_id": o.get("op_id"), "op": o.get("op"), "ts": o.get("ts"),
             "to_change": o.get("to_change"), "applied": o.get("applied"),
             "filter": o.get("filter")}
            for o in ops[-limit:]
        ]




    def _knowledge_playbook(self, pending_n: int, shown_n: int) -> dict[str, Any]:
        """Operating manual for "tidy up this library" (shown to the host; following it completes one distillation round).

        Deliberately written like a recipe, not like documentation: the host
        executes step by step, and a vague "you could summarize a bit" buys
        no action. The three rules come from pits already stepped in --
        evidence citing fabricated ids gets the whole item rejected,
        existing summaries should not be redone, and a bad write must be
        undoable.
        """
        batch = min(10, max(1, pending_n))
        return {
            "goal": ("Distill %d knowledge points into higher-level conclusions. **The summarizing intelligence is on your side** -- "
                     "this service only produces material, validates, and stores; its production path embeds no LLM." % pending_n),
            "steps": [
                '1) Fetch material: knowledge_topic(action="brief_batch", batch=%d)'
                '(without topics, the %d most due topics are picked automatically)' % (batch, batch),
                "2) Read each topic's material and write one conclusion (<= 2000 chars, clear subject + facts)",
                "3) Write back: knowledge_topic(action=\"summarize_batch\", "
                "items='[{\"topic_id\":\"...\",\"summary\":\"...\","
                "\"evidence\":[\"...\"]}]') (items is a JSON array string)",
                "4) Repeat 1-3 until pending_count is 0 (each batch automatically skips already-summarized topics)",
            ],
            "rules": [
                "evidence must use ids listed in that topic's brief -- fabricated references cause the whole item to be **rejected**, "
                "changing no field (half-applied writes are harder to debug than none)",
                "Topics already summarized and not flagged ⚠️ are skipped automatically -- do not redo them; "
                "to rewrite, name them explicitly via topics",
                "If one write goes wrong: undo via knowledge_topic(action=\"revert\", topic_id=...)",
                "Write-back persists to disk (with a ledger); no separate save call needed",
                # S3: the four-field contract. Written into rules rather than
                # new parameter docs -- the host learns it by walking the
                # recipe once; the f/s slots are the cheapest known "fake-cluster filter".
                "Optional four fields (recommended): add \"fras\":{\"f\":\"category (used as title)\","
                "\"r\":[\"related entities/topics\"],\"a\":\"the citable conclusion\",\"s\":\"constraints not derivable "
                "but costly if wrong\"} to the item. All four keys are required; if s truly has none, write \"none\". "
                "When fras is passed without summary, a is the sentence itself",
            ],
            "done_when": "pending_count == 0",
        }




    # ------------------------------------------------------------------
    # Aligning L0 and L1 on the "merge on conflict" semantics
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_raw_log(disk: list[dict], mine: list[dict]) -> list[dict]:
        """Union of two L0 logs by id (**addition only: no overwrite, no deletion**).

        Why L0 needs this: L1 (``memories.json``) has long had the additive
        "merge on conflict" semantics (``manager.retry_save``: take the
        disk's newest version + add back ids the disk lacks), while L0
        (``brain_raw.json``) was previously **direct overwrite / direct
        replacement** -- when two processes each write the same library, you
        get "L1 is the union, L0 is the shorter copy", so L1's
        ``source_ids`` cannot point at the L0 originals. Measured on the
        Chinese novel library: 21 ids / 36 dangling L1 entries (addendum
        five, "L0 bookkeeping drift"); the root cause is exactly this
        semantic misalignment.

        Addition-only is deliberate: L0 is an append-only log, and deletion
        is soft too (a ``forgotten`` marker, the entry remains), so the
        union never resurrects anything deleted. When two entries share an
        id but differ in content, **the disk copy wins** (consistent with
        retry_save's choice for L1).
        """
        if not disk:
            return list(mine)
        by_id = {str(r.get("id")): r for r in disk if r.get("id")}
        merged = list(disk)
        for r in mine:
            rid = r.get("id")
            if rid and str(rid) not in by_id:
                merged.append(r)
                by_id[str(rid)] = r
        return merged




    @staticmethod
    def _layer_key(item: dict) -> str:
        """Merge key for list layers: id if present, else a content fingerprint (so an id-less layer is not wholesale clobbered)."""
        rid = item.get("id")
        if rid:
            return "id:" + str(rid)
        fp = {k: item.get(k) for k in
              ("text", "query", "ts", "timestamp", "session_id", "reason", "name")}
        return "fp:" + json.dumps(fp, sort_keys=True, ensure_ascii=False,
                                  default=str)



    @classmethod
    def _merge_list_layer(cls, disk, mine, cap: int = 0):
        """Union of list layers: disk as the base, entries this process has
        not seen get added in; on the same key **this process** wins.

        Addition-only is deliberate -- these layers all have log/ledger
        semantics; the union never resurrects anything deleted (deletion
        goes through tombstones or soft markers).
        """
        if not isinstance(disk, list):
            return list(mine) if isinstance(mine, list) else disk
        if not isinstance(mine, list):
            return list(disk)
        out: dict[str, dict] = {}
        order: list[str] = []
        for item in list(disk) + list(mine):
            if not isinstance(item, dict):
                continue
            k = cls._layer_key(item)
            if k not in out:
                order.append(k)
            out[k] = item
        merged = [out[k] for k in order]
        if cap and len(merged) > cap:
            merged = merged[-cap:]
        return merged



    @staticmethod
    def _merge_dict_layer(disk, mine):
        """Union of dict layers: disk as the base, this process's keys overwrite (local wins)."""
        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        out = dict(disk)
        out.update(mine)
        return out



    @staticmethod
    def _merge_entity_layer(disk, mine):
        """Entity layer: union of keys, ``memory_ids`` takes the **union**,
        ``mention_count`` takes the max.

        Taking the local copy directly would wipe out the memory ids the
        other side attached (measured phenomenon). mention_count is
        recomputed from live memories at load time anyway
        (``_recompute_entity_stats``); taking the max just prevents a
        spurious dip during the merge.
        """
        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        out: dict[str, Any] = {}
        for k in set(disk) | set(mine):
            a, b = disk.get(k), mine.get(k)
            if a is None:
                out[k] = b
                continue
            if b is None:
                out[k] = a
                continue
            if not (isinstance(a, dict) and isinstance(b, dict)):
                out[k] = b
                continue
            merged = dict(a)
            merged.update(b)
            if "memory_ids" in a or "memory_ids" in b:
                ids = list(a.get("memory_ids") or []) + list(b.get("memory_ids") or [])
                merged["memory_ids"] = sorted({str(x) for x in ids})
            mc = [x for x in (a.get("mention_count"), b.get("mention_count"))
                  if isinstance(x, (int, float))]
            if mc:
                merged["mention_count"] = max(mc)
            out[k] = merged
        return out



    @classmethod
    def _merge_intent_layer(cls, disk, mine):
        """Open-intent layer: union of keys + **status only moves forward**.

        Why a separate rule: generic list merging is "disk as base, local
        wins", fine for append-only ledgers (entries where order does not
        matter), but not for layers with terminal states -- A marks done,
        B still holds open, and the moment B persists, done is wiped back
        to open with no error on either side (reproduced repeatedly across
        two instances).

        So the trade-off here: **status takes the higher rank**, all other
        fields still follow the local copy. "Done" is the only irreversible
        part of this semantics.
        """
        merged = cls._merge_list_layer(disk, mine)
        if not isinstance(disk, list) or not isinstance(merged, list):
            return merged
        disk_by_key: dict[str, dict] = {}
        for r in disk:
            if isinstance(r, dict):
                disk_by_key[cls._layer_key(r)] = r
        rank = cls._INTENT_STATUS_RANK
        out: list[dict] = []
        for it in merged:
            if not isinstance(it, dict):
                out.append(it)
                continue
            other = disk_by_key.get(cls._layer_key(it))
            if isinstance(other, dict):
                mine_rank = rank.get(str(it.get("status") or ""), 0)
                disk_rank = rank.get(str(other.get("status") or ""), 0)
                if disk_rank > mine_rank:
                    fixed = dict(it)
                    fixed["status"] = other["status"]
                    # Carry over the completion-trace fields too; don't leave "done but with no timestamp"
                    for extra in ("done_at", "resolved_at", "completed_at"):
                        if other.get(extra) and not fixed.get(extra):
                            fixed[extra] = other[extra]
                    it = fixed
            out.append(it)
        return out



    @classmethod
    def _merge_session_layer(cls, disk, mine):
        """Session layer: union of keys + **terminal state wins**.

        When the other side has already ended (wrote ended_at) while this
        local route has not wrapped up yet, "local wins" would wipe
        ended_at -- the finished session resurrects inside
        ``active_sessions``.
        """
        out = cls._merge_dict_layer(disk, mine)
        if not isinstance(disk, dict) or not isinstance(out, dict):
            return out
        for k, v in disk.items():
            cur = out.get(k)
            if not (isinstance(cur, dict) and isinstance(v, dict)):
                continue
            if v.get("ended_at") and not cur.get("ended_at"):
                merged = dict(cur)
                merged["ended_at"] = v["ended_at"]
                if v.get("summary") and not merged.get("summary"):
                    merged["summary"] = v["summary"]
                out[k] = merged
        return out



    @staticmethod
    def _merge_counter_layer(disk, mine):
        """Per-day counters: take the **max** on the same key, deliberately not the sum.

        Both sides may carry the same baseline for one key (each read it at
        its own load); summing would inflate the denominator of the
        "should-have-recalled rate". Taking the max at worst under-counts;
        it never inflates.
        """
        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        out = dict(disk)
        for k, v in mine.items():
            old = out.get(k)
            if isinstance(old, dict) and isinstance(v, dict):
                merged = dict(old)
                for kk, vv in v.items():
                    if (isinstance(vv, (int, float))
                            and isinstance(old.get(kk), (int, float))):
                        merged[kk] = max(old[kk], vv)
                    else:
                        merged[kk] = vv
                out[k] = merged
            else:
                out[k] = v
        return out




    # ------------------------------------------------------------------
    # "Real deletion" ledger for derived layers
    #
    # A union only adds. The original fix was "after merging, re-derive
    # from the currently-live memories", which only saves **derived
    # copies** (entities / profile, re-computable from their L1 source).
    # miss_log / sessions / intents are **independent ledgers** with no
    # re-derivable source; once deleted, they are deleted.
    #
    # Measured (_probe_out/_f92_delete_gap_20260929.py): A clears miss_log
    # and persists to disk; B (an older snapshot) merges all 3 entries
    # right back with a single save, no error on either side. Hence we
    # need a ledger of "which keys were deleted, and when", and merging
    # intercepts by it.
    # ------------------------------------------------------------------

    @staticmethod
    def _del_ledger_key(layer: str, key: str) -> str:
        """Ledger key: ``<layer filename>::<entry key>`` (aligned with the value produced by :meth:`_layer_key`)."""
        return f"{layer}::{key}"




    @staticmethod
    def _merge_del_ledger_layer(disk, mine):
        """Deletion ledger: union of keys; on the same key take the **later** deletion time.

        Deliberately not the dict rule's "local wins" -- both sides are
        deletion records; only the later one avoids **under-recording** the
        other side's deletion (under-recorded = cannot block the old
        version, i.e. as good as not recorded).
        """
        if not isinstance(disk, dict):
            return dict(mine) if isinstance(mine, dict) else {}
        out = dict(disk)
        if isinstance(mine, dict):
            for k, v in mine.items():
                old = out.get(k)
                if isinstance(v, (int, float)) and isinstance(old, (int, float)):
                    out[k] = max(old, v)
                else:
                    out[k] = v
        return out



    @staticmethod
    def _entry_ts(item: dict) -> float:
        """The entry's own timestamp (used to judge "was it re-created after the deletion").

        Returns 0.0 when unavailable -- conservatively treated as "very
        old", so once a tombstone exists the merge is always blocked.
        """
        for k in ("ts", "timestamp", "created_at", "started_at", "done_at"):
            v = item.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0




    @staticmethod
    def _merge_topic_layer(disk, mine):
        """Topic layer: union of keys; ``member_ids`` takes the **union**,
        scalars take the newer side.

        Why not the dict rule's "local wins" directly: ``member_ids`` is
        the body of this layer -- taking the local copy would wipe the
        members the other side attached (same class of incident).
        ``summary`` is chosen with **host taking priority over timestamps**
        (see the in-method comment); only when neither side is host does
        ``updated_at`` pick the newer side.
        """
        def _span_pair(raw):
            """Collapse a span of any shape into two floats [lo, hi].

            ``[]`` / ``None`` / a single element / string elements -- all
            are defended against. Derived-layer merging is the last step of
            the save chain; it must not take down the whole save over one
            dirty entry (the original ``float(span)[0]`` raised TypeError
            outright on a list).
            """
            try:
                s = list(raw or [])
                if len(s) < 2:
                    return [0.0, 0.0]
                return [float(s[0]), float(s[1])]
            except (TypeError, ValueError):
                return [0.0, 0.0]

        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        # Generations: rebuild ⇒ +1, **the higher generation replaces the whole layer**.
        # Key union happens only within the same generation -- a cross-generation
        # union is exactly what resurrects old clusters: 652 new clusters on
        # disk ∪ 698 old clusters in the other side's memory = 1050. The older
        # writer's layer epoch is lower, so on its next save it **yields
        # voluntarily** to the disk copy and does not merge again -- that is
        # what makes clusters "actually clearable".
        e_disk = topic_mod.layer_epoch(disk)
        e_mine = topic_mod.layer_epoch(mine)
        if e_mine > e_disk:
            return dict(mine)
        if e_disk > e_mine:
            return dict(disk)
        out: dict[str, Any] = {}
        for k in set(disk) | set(mine):
            a, b = disk.get(k), mine.get(k)
            if a is None:
                out[k] = b
                continue
            if b is None:
                out[k] = a
                continue
            if not (isinstance(a, dict) and isinstance(b, dict)):
                out[k] = b
                continue
            merged = dict(a)
            merged.update(b)
            ids = list(a.get("member_ids") or []) + list(b.get("member_ids") or [])
            merged["member_ids"] = sorted({str(x) for x in ids})
            merged["size"] = len(merged["member_ids"])
            # Summary merging: **host takes priority over timestamps**.
            # Extracted representative sentences can be recomputed at the
            # next refresh (losing them costs nothing); a host summary is
            # written with real effort and this system will never
            # auto-regenerate it -- once lost, it is gone for good. So if
            # either side is host, take that side; only when neither (or
            # both) is host does updated_at pick the newer. The first
            # version took by time alone; once host summaries became the
            # norm, consolidation refreshes would silently flush summaries
            # back into extracts -- precisely "the silent rewrite that
            # hurts trust the most".
            ka = str(a.get("summary_kind") or "")
            kb = str(b.get("summary_kind") or "")
            if ka == topic_mod.SUMMARY_KIND_HOST and kb != topic_mod.SUMMARY_KIND_HOST:
                newer = a
            elif kb == topic_mod.SUMMARY_KIND_HOST and ka != topic_mod.SUMMARY_KIND_HOST:
                newer = b
            else:
                ta = float(a.get("updated_at") or 0.0)
                tb = float(b.get("updated_at") or 0.0)
                newer = b if tb > ta else a
            merged["summary"] = newer.get("summary", "")
            merged["summary_kind"] = newer.get("summary_kind", "")
            merged["summary_evidence"] = list(newer.get("summary_evidence") or [])
            # Write-back metadata follows the selected side (signature/time/size
            # must share the same origin as summary; otherwise the stale check
            # pairs A's summary with B's members and reports a false expiry).
            for f in ("summary_at", "summary_signature", "summary_member_count"):
                merged[f] = newer.get(f, a.get(f))
            # S3: the four fields and the host title likewise follow the
            # selected side -- they are one package with summary (fras.a is
            # often the summary body itself). Following the wrong side
            # means: the map shows A's conclusion while clicking through
            # reveals B's constraints, and the host decides based on the
            # wrong one.
            if newer.get("summary_fras"):
                merged["summary_fras"] = newer.get("summary_fras")
            else:
                merged.pop("summary_fras", None)
            if str(newer.get("title_kind") or "") == topic_mod.TITLE_KIND_HOST:
                merged["title"] = newer.get("title", merged.get("title"))
                merged["title_kind"] = topic_mod.TITLE_KIND_HOST
            else:
                # The selected side has no host title ⇒ clear the other side's
                # residual host title. Title and summary are one package:
                # keeping B's name attached to A's conclusion means "the map
                # shows A's conclusion topped by a name from B that matches
                # nothing". Same discipline as the fras pop above (this
                # residual path was verified by a negative control).
                merged.pop("title_kind", None)
                merged["title"] = newer.get("title", "")
            merged["recall_hits"] = max(int(a.get("recall_hits") or 0),
                                        int(b.get("recall_hits") or 0))
            # span takes the union (earliest, latest). **Never write float(span)[0]**:
            # span is itself a list, and float() on a list raises TypeError
            # outright -- not a "wrong value" but the whole save exploding
            # here (triggered only when multi-process shared-library writers
            # collide on the same topic).
            sa, sb = _span_pair(a.get("span")), _span_pair(b.get("span"))
            merged["span"] = [min(sa[0], sb[0]), max(sa[1], sb[1])]
            out[k] = merged
        return out



    @staticmethod
    def _merge_sourcesig_layer(disk, mine):
        """C1a: source-signature layer merge -- on the same key take the newer ``sig_at``.

        When two processes ingest the same file, both record a signature:
        the later-validated one is closer to the disk's current state. With
        the same sha1 the facts are identical, so either choice is right;
        with different sha1s, mistakenly taking the older one makes drift
        report an unchanged file as stale (cry wolf twice and nobody reads
        the audit anymore).
        """
        if not isinstance(disk, dict) or not isinstance(mine, dict):
            return mine if isinstance(mine, dict) else disk
        out = dict(disk)
        for k, b in mine.items():
            a = out.get(k)
            if not isinstance(b, dict):
                continue
            if not isinstance(a, dict) or float(b.get("sig_at") or 0.0) >= float(
                    a.get("sig_at") or 0.0):
                out[k] = b
        return out




    @staticmethod
    def _stamp_of(path: str):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)




    # -- F14-1 layer version numbers / G0 derived-layer dirty set ---------------------------------
    @staticmethod
    def _layer_fingerprint(value: Any) -> str:
        """Layer content fingerprint (= layer version number).

        Why a **content hash** instead of per-entry dirty flags: the 10
        layers have different shapes (list / dict / counter); per-entry
        dirty marking would require instrumenting dozens of modification
        points, and one miss is a silent lost write. A content hash has
        exactly one semantic -- "has it changed compared to what I last
        wrote down" -- leaving no room for missed instrumentation.

        ``sort_keys=True`` is mandatory: dict insertion order may differ
        across processes; without sorting, "the content is actually
        identical" gets judged as changed (a wasted rewrite).
        """
        try:
            blob = json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), default=str,
            )
        except (TypeError, ValueError):  # pragma: no cover - fallback
            blob = repr(value)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
