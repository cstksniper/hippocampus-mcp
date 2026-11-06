# -*- coding: utf-8 -*-
"""B2-3 P2: entity and proper-noun lexicon cluster

This cluster was moved over verbatim from ``brain.py``.

Why the whole cluster could move: among the writers of its 5 attributes
(``_alias_reverse`` / ``_lexicon_mtime`` / ``entity_lexicon`` /
``entity_lexicon_aliases`` / ``entity_lexicon_path``), **the only writer
outside the cluster is ``__init__``** -- writers travel as a cluster, so
moving it does not split one attribute's writers across two files. This
was counted via AST, not guessed from naming.

P1 already moved the 9 pure methods of this cluster (``_read_lexicon`` /
``_parse_lexicon_text`` / ``_clean_entity_name`` ...); this module only
holds the remaining ones that actually touch instance state.

Warning: after ``discover_aliases`` moved, the related structural guard
assertions (originally scanning only brain.py) were updated in step to
scan this file as well -- otherwise that structural guard would fail
falsely.
"""

from __future__ import annotations

from ..common.constants import *  # noqa: F401,F403  (constant table: see common/constants.__all__)
from ..common.constants import logger
import os
import re
from typing import Any
from ..domain import aliases as alias_mod
from ..store.memory import MemoryState
from ..config import load_config  # noqa: E402  (env reads folded into config)


class _EntityLexiconMixin:
    """B2-3 P2: entity and proper-noun lexicon cluster"""

    def reload_lexicon(self) -> int:
        """Reload the proper-noun lexicon (no service restart needed when the
        lexicon file changes). Returns the number of lexicon entries.

        The path is re-resolved: if the lexicon file did not exist at startup
        (``path=None``), adding the file afterwards and calling this method
        still takes effect.
        """
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        self.entity_lexicon_path = self._resolve_lexicon_path(
            self._explicit_lexicon_path, persist_dir
        )
        self.entity_lexicon, self.entity_lexicon_aliases = self._read_lexicon(
            self.entity_lexicon_path
        )
        self._alias_reverse = None      # reverse index invalidated (alias mapping changed)
        self._lexicon_mtime = self._lexicon_mtime_of(self.entity_lexicon_path)
        return len(self.entity_lexicon)



    def _ensure_default_lexicon_path(self) -> str | None:
        """Default lexicon path; **create the file if it does not exist** (breaks
        the "no lexicon without a lexicon" deadlock).

        Returns None = there is not even a persist_dir (pure in-memory library)
        or the directory is not writable -- only then is persisting truly
        impossible; hand the error to the caller (the error must state which
        three paths were searched).
        """
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        if not persist_dir:
            return None
        default_lex = os.path.join(persist_dir, "entity_lexicon.txt")
        if not os.path.isfile(default_lex):
            try:
                with open(default_lex, "a", encoding="utf-8"):
                    pass        # touch: create an empty lexicon first, then append via the normal path
            except OSError:
                return None
        return default_lex




    def _reload_lexicon_if_changed(self) -> int:
        """Hot-reload the lexicon when its file changed (**no new MCP tool**: done
        alongside ``stats()``).

        Why this is needed: the daemon is a long-lived process and the lexicon
        path is resolved only once at construction
        (:meth:`_resolve_lexicon_path`). If ``entity_lexicon.txt`` does not
        exist at service start => ``path=None``; when the user later drops the
        file into the memory library, this process will **not** re-resolve,
        ``entity_lexicon_size`` stays 0 forever, and ``lexicon_add`` raises
        directly because ``path=None`` -- a deadlock, only escapable by
        restarting the daemon.
        (Confirmed by testing: this is exactly the real shape of "lexicon hot
        reload broken", not a missing path fallback.)

        Returns the number of entries. Exceptions are always swallowed and the
        state left as-is: self-exposure aside, ``stats()`` must not die.

        Thread safety: only **replaces** references such as ``entity_lexicon``,
        never mutates the collections in place. Concurrent threads may briefly
        read stale references but will not crash (existing convention of the
        daemon's multi-threaded shared brain).
        """
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        path = self._resolve_lexicon_path(self._explicit_lexicon_path, persist_dir)
        mtime = self._lexicon_mtime_of(path)
        if path == self.entity_lexicon_path and mtime == self._lexicon_mtime:
            return len(self.entity_lexicon)
        self.entity_lexicon_path = path
        self.entity_lexicon, self.entity_lexicon_aliases = self._read_lexicon(path)
        self._alias_reverse = None
        self._lexicon_mtime = mtime
        return len(self.entity_lexicon)




    def canonical_entity(self, name: str) -> str:
        """Normalize an entity name to its canonical name (the ``别名 => 主名``
        declared by lexicon alias lines).

        Names not registered as aliases are returned as-is. **Both the
        extraction side and the query side go through here** -- normalizing
        only one side is the same as not normalizing: the library stores the
        main name, so querying the "alias" still gets no entity boost.
        """
        n = (name or "").strip()
        if not n:
            return n
        return self.entity_lexicon_aliases.get(n.lower(), n)




    def _canonicalize_map(self, mapping: dict[str, str]) -> dict[str, str]:
        """Fold aliases in ``{实体名: origin}`` to canonical names; for the same key keep the higher-trust origin."""
        if not getattr(self, "entity_lexicon_aliases", None):
            return mapping
        merged: dict[str, str] = {}
        disp_of: dict[str, str] = {}
        for name, origin in mapping.items():
            canon = self.canonical_entity(name)
            k = canon.lower()
            cur = disp_of.get(k)
            if cur is None:
                disp_of[k] = canon
                merged[canon] = origin
            elif _ORIGIN_RANK[origin] > _ORIGIN_RANK[merged[cur]]:
                merged[cur] = origin
        return merged




    def _trusted_entity_ids(self) -> set[str]:
        """Set of ids of trusted entities.

        Computed fresh each time (E is on the order of a thousand entries,
        well under 1ms). No caching, to avoid hard-to-spot stale bugs like
        "cache invalidation after the entity table is added to/removed
        from/modified" -- this is not a hot path.
        """
        return {e["id"] for e in self.entities.values() if self._entity_trusted(e)}



    def entity_origin_stats(self) -> dict[str, int]:
        """Count entities by origin, for observing extraction quality (trusted vs candidate)."""
        out: dict[str, int] = {}
        for ent in self.entities.values():
            o = self._entity_origin(ent)
            out[o] = out.get(o, 0) + 1
        return out



    def _recompute_entity_stats(self, keys: list[str] | None = None) -> int:
        """Recompute entity stats (mention_count / first_seen / last_seen).

        Based on "currently alive memories": ids of already-deleted memories
        are pruned from ``memory_ids``. Must be called after deleting or
        superseding memories, otherwise the inverted index keeps dangling
        references.

        Args:
            keys: entity keys to recompute; None means a full recompute
                (self-heal when loading an old library).

        Returns:
            Total number of dangling references pruned.
        """
        items = (
            list(self.entities.items()) if keys is None
            else [(k, self.entities[k]) for k in keys if k in self.entities]
        )
        dropped = 0
        for _, ent in items:
            alive: list[tuple[str, float]] = []
            for mid in ent.get("memory_ids") or []:
                rec = self.manager.store.get(mid)
                if rec is None:
                    dropped += 1
                    continue
                alive.append((mid, rec.timestamp))
            ent["memory_ids"] = [m for m, _ in alive]
            ent["mention_count"] = len(alive)
            if alive:
                stamps = [t for _, t in alive]
                ent["first_seen"] = min(stamps)
                ent["last_seen"] = max(stamps)
        return dropped




    def _task_entity_ids(self, text: str) -> set[str]:
        """Ids of **trusted** entities found in the text (shared by pin ordering / relevance decisions)."""
        if not (text or "").strip():
            return set()
        try:
            origins = self.extract_entities_with_origin(text)
        except Exception:  # noqa: BLE001 -- extraction failure must not fail session_start
            return set()
        out: set[str] = set()
        for name, origin in origins.items():
            if origin not in TRUSTED_ENTITY_ORIGINS:
                continue
            ent = self._resolve_entity(name)
            if ent is not None:
                out.add(ent["id"])
        return out




    def _alias_siblings(self, canon_key: str) -> list[str]:
        """Entity keys of the **aliases** pointing to this canonical name (lowercase). Lazily builds the reverse index; invalidated on reload."""
        if not self.entity_lexicon_aliases:
            return []
        rev = getattr(self, "_alias_reverse", None)
        if rev is None:
            rev = {}
            for alias, canon in self.entity_lexicon_aliases.items():
                rev.setdefault(canon.lower(), []).append(alias)
            self._alias_reverse = rev
        return rev.get((canon_key or "").lower(), [])



    def _alias_family(self, key: str) -> list[str]:
        """The **alias family** this entity key belongs to (itself, the canonical
        name, and all sibling aliases).

        ``key`` may itself be an alias (in historical data "子安" was archived
        independently); in that case we still need to find its canonical name
        and siblings -- so first look up the alias mapping for the canonical
        name, then expand.
        """
        k = (key or "").lower()
        if not k:
            return []
        canon = (self.entity_lexicon_aliases.get(k) or k).lower()
        fam = [canon]
        for a in self._alias_siblings(canon):
            if a not in fam:
                fam.append(a)
        if k not in fam:
            fam.append(k)
        return fam



    def _resolve_entity(self, entity_key: str) -> dict[str, Any] | None:
        """Find an entity entry by name (aliases included).

        Priority: the canonical-name entity -> the family member with the
        **most mentions** (in historical data sometimes only the alias was
        archived; in that case we must not report "no such entity").
        """
        canon = self.canonical_entity(entity_key or "")
        ent = self.entities.get(canon.lower())
        if ent is not None:
            return ent
        cands = [self.entities[k] for k in self._alias_family(entity_key)
                 if k in self.entities]
        if not cands:
            return None
        return max(cands, key=lambda e: e.get("mention_count", 0))



    def _entity_memory_ids(self, ent: dict[str, Any]) -> list[str]:
        """Memory ids of the entity **together with its alias family** (dedup,
        order-preserving).

        The read side merges immediately instead of waiting for
        ``relink_entities`` to re-link: in historical data the same entity is
        often already split into several archives by alias; recognizing only
        the canonical name would show someone who "just configured an alias"
        a partial archive. The lexicon is configuration, and configuration
        taking effect must not depend on a one-time data rewrite.
        """
        ids = list(ent.get("memory_ids", []))
        seen = set(ids)
        for k in self._alias_family(ent.get("key", "")):
            if k == ent.get("key"):
                continue
            s = self.entities.get(k)
            if not s:
                continue
            for mid in s.get("memory_ids", []):
                if mid not in seen:
                    seen.add(mid)
                    ids.append(mid)
        return ids



    def entity_facts(self, entity_key: str, limit: int = 5) -> list[str]:
        """List an entity's active facts (newest first). Aliases are normalized to the canonical name."""
        ent = self._resolve_entity(entity_key)
        if not ent:
            return []
        facts = []
        for mid in self._entity_memory_ids(ent):
            rec = self.manager.store.get(mid)
            if rec and rec.state == MemoryState.ACTIVE.value:
                facts.append((rec.timestamp, rec.content))
        facts.sort(key=lambda x: -x[0])
        return [c for _, c in facts[:limit]]



    # ==================================================================
    # Entity profile / lexicon expansion closed loop (usage-layer requirement)
    # ==================================================================
    def entity_profile(
        self,
        entity_key: str,
        limit: int = 20,
        order: str = "asc",
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        """Fetch an entity's full profile: pull all memories it participated in,
        in time order.

        Difference from :meth:`entity_facts`: the latter returns only content
        strings (for context assembly); this method returns structured entries
        with ``timestamp`` / ``source`` / ``memory_id`` / ``state`` -- when the
        host wants to "pull someone's entire set of finalized decisions in
        time order" before writing character scenes, it needs to verify by
        time, point back to the original file, and see which entries have
        been superseded.

        Args:
            order: ``"asc"`` (default, chronological, good for timelines) /
                ``"desc"`` (most recent first).
            include_superseded: whether to include superseded entries
                (default only active).
            limit: max entries returned; 0 means unlimited.
        """
        canon = self.canonical_entity(entity_key or "")
        ent = self._resolve_entity(entity_key)
        if not ent:
            return {"found": False, "entity": entity_key, "facts": [], "total": 0}
        rows: list[dict[str, Any]] = []
        for mid in self._entity_memory_ids(ent):
            rec = self.manager.store.get(mid)
            if rec is None:
                continue
            if not include_superseded and rec.state != MemoryState.ACTIVE.value:
                continue
            rows.append({
                "memory_id": rec.id,
                "content": rec.content,
                "timestamp": rec.timestamp,
                "source": rec.source or "",
                "state": rec.state,
                "importance": rec.importance,
                "superseded_by": rec.superseded_by,
            })
        rows.sort(key=lambda r: r["timestamp"], reverse=(order == "desc"))
        kept = rows[:limit] if limit > 0 else rows
        out = {
            "found": True,
            "entity": ent["name"],
            "key": ent["key"],
            "origin": self._entity_origin(ent),
            "trusted": self._entity_trusted(ent),
            "mention_count": ent.get("mention_count", 0),
            "total": len(rows),
            "count": len(kept),
            "facts": kept,
        }
        # Alias family: how many alias archives were merged, and which name the
        # query was folded in from -- if not stated explicitly, the host will
        # assume "this archive is all there is" (the rest hangs under an alias).
        sibs = [k for k in self._alias_family(ent.get("key", ""))
                if k != ent.get("key")]
        if sibs:
            out["aliases"] = sibs
            out["mention_count_merged"] = len(self._entity_memory_ids(ent))
        if canon != (entity_key or ""):
            out["resolved_from"] = entity_key
        return out



    def entity_list(
        self,
        prefix: str = "",
        limit: int = 50,
        min_mentions: int = 1,
        trusted_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List entities (descending by mention count), to discover "which
        entity keys actually exist in the library".

        Without this entry point the host can only guess entity names when
        calling :meth:`entity_profile` -- and entity names are extracted, so
        their spelling often differs from how they appear in the memories.
        """
        rows: list[dict[str, Any]] = []
        for ent in self.entities.values():
            if trusted_only and not self._entity_trusted(ent):
                continue
            if prefix and not ent["name"].lower().startswith(prefix.lower()):
                continue
            if ent.get("mention_count", 0) < min_mentions:
                continue
            rows.append({
                "name": ent["name"],
                "key": ent["key"],
                "origin": self._entity_origin(ent),
                "trusted": self._entity_trusted(ent),
                "mention_count": ent.get("mention_count", 0),
                "last_seen": ent.get("last_seen", 0.0),
            })
        rows.sort(key=lambda r: (-r["mention_count"], r["name"]))
        return rows[:limit] if limit > 0 else rows



    def lexicon_add(self, terms: list[str], relink: bool = True) -> dict[str, Any]:
        """Append proper nouns to the **currently effective** lexicon and
        hot-reload.

        Pitfall (found by testing): the lexicon is not necessarily in the
        memory library directory -- the host's actually effective path is
        ``data/entity_lexicon_<项目名>.txt`` (via env / explicit argument; the
        memory library directory has **no** entity_lexicon.txt). Writing back
        to ``<persist_dir>/entity_lexicon.txt`` would conjure up a new file
        that has no effect: the host believes the addition succeeded while
        retrieval changes not at all. Therefore always write back to
        ``self.entity_lexicon_path`` (the actually effective one).

        Args:
            terms: term list. **Alias lines supported** -- ``"子安 => 顾言"``
                (``->`` / ``→`` equivalent). Aliases are written into the
                lexicon file, and both extraction and query sides normalize
                accordingly. This is the general solution to "one entity,
                several names", not limited to person names (abbreviations/
                full forms/old names/codes likewise). Syntactically incomplete
                lines are **not persisted**; they go verbatim into the
                returned ``invalid``.
            relink: whether to re-link entities across the whole library after
                appending. **Must be done by default**: otherwise new terms
                only affect memories written later, existing memories remain
                unfindable, and the closed loop breaks here. When aliases were
                written, also runs :meth:`merge_alias_entities` along the way,
                merging archives that history split apart by alias back into
                the canonical name.

        Returns: ``added`` lines actually persisted - ``skipped_duplicate`` -
        ``invalid`` - ``alias_count`` total effective aliases (so the host can
        check normalization really took hold) - ``lexicon_size`` / ``path`` -
        optional ``relink`` / ``alias_merge``.
        """
        entries = list(dict.fromkeys(
            t.strip() for t in (terms or []) if t and t.strip()
        ))
        path = self.entity_lexicon_path
        if not entries:
            return {"added": [], "lexicon_size": len(self.entity_lexicon), "path": path}
        if not path:
            # Self-heal: when the file under the default path does not exist,
            # path is None (resolved once at construction). We should
            # **create** the default file here, not raise -- otherwise the
            # host deadlocks:
            # no lexicon -> this tool raises -> lexicon cannot be created ->
            # daemon restart is the only way out.
            path = self._ensure_default_lexicon_path()
        if not path:
            _dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
            _env = load_config().entity_lexicon or ""
            # State truthfully which three paths were searched -- the old
            # message only said "persist_dir unset", while the real cause is
            # usually "file under the default path missing", sending people
            # entirely the wrong way.
            raise ValueError(
                "cannot append to the lexicon: no writable lexicon path. Searched, in order -- "
                f"1) explicit entity_lexicon_path="
                f"{self._explicit_lexicon_path or 'unset'}; "
                f"2) env HIPPOCAMPUS_ENTITY_LEXICON={_env or 'unset'}; "
                f"3) default <memory library>/entity_lexicon.txt: persist_dir="
                f"{_dir or 'unset'} (creation failed or no permission)."
            )
        existing_terms, existing_aliases = self._read_lexicon(path)
        before = len(existing_terms)
        added: list[str] = []
        invalid: list[str] = []
        has_alias = False
        for e in entries:
            m = self._ALIAS_SEP.search(e)
            if not m:
                if e.lower() in existing_terms:
                    continue
                added.append(e)
                continue
            alias = e[:m.start()].strip().lstrip("#").strip()
            canon = e[m.end():].strip()
            if not alias or not canon:
                invalid.append(e)
                continue
            if alias.lower() == canon.lower():
                if alias.lower() not in existing_terms:
                    added.append(canon)
                continue
            if existing_aliases.get(alias.lower(), "").lower() == canon.lower():
                continue
            # Persist using the canonical spelling (`别名 => 主名`), avoiding three arrow styles mixed in the file
            added.append(f"{alias} => {canon}")
            has_alias = True
        if added:
            # Append at the end: preserve existing comments and grouping, do not rewrite the whole file
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(added) + "\n")
        size = self.reload_lexicon()
        out: dict[str, Any] = {
            "added": added,
            "skipped_duplicate": len(entries) - len(added) - len(invalid),
            "lexicon_size_before": before,
            "lexicon_size": size,
            "alias_count": len(self.entity_lexicon_aliases),
            "path": path,
        }
        if invalid:
            # Do not swallow silently: if a malformed alias line is returned
            # as success, the host will believe normalization took effect
            out["invalid"] = invalid
        if relink:
            out["relink"] = self.relink_entities()
        if has_alias:
            out["alias_merge"] = self.merge_alias_entities()
        return out




    def merge_alias_entities(self) -> dict[str, Any]:
        """Merge alias archives into the canonical-name archive (library cleanup
        after disambiguation).

        The read side (:meth:`entity_profile` / :meth:`entity_facts`) already
        merges alias families, so **not running this is also correct for
        queries**; running it is to truly clean up the library -- otherwise
        ``entity_list`` keeps showing "子安" and "顾言" as two parallel
        archives, and the profile slot, mention counts, and entity boosts are
        each counted separately.

        Actions: (1) merge the alias entry's ``memory_ids`` into the canonical
        entry (union; origin takes the higher); (2) in those memories'
        ``entity_ids``, replace the alias id with the canonical id;
        (3) delete the alias entry; (4) recompute all stats. When the
        canonical entry does not exist yet, simply **rename** (id and inverted
        index untouched).

        Difference from :meth:`forget`: this is not deletion semantics but
        **identity merging** -- not a single memory is lost; it just no longer
        hangs under two names.
        """
        if not self.entity_lexicon_aliases:
            return {"merged": 0, "renamed": 0, "remapped_memories": 0}
        merged = renamed = remapped = 0
        for alias, canon in list(self.entity_lexicon_aliases.items()):
            a = self.entities.get(alias)
            if not a:
                continue
            ck = canon.lower()
            c = self.entities.get(ck)
            if c is None:
                a["name"] = canon
                a["key"] = ck
                self.entities[ck] = a
                del self.entities[alias]
                renamed += 1
                continue
            if c is a:
                continue
            for mid in a.get("memory_ids") or []:
                rec = self.manager.store.get(mid)
                if rec is None:
                    continue
                old_ids = list(rec.entity_ids or [])
                new_ids: list[str] = []
                for e in old_ids:
                    v = c["id"] if e == a["id"] else e
                    if v not in new_ids:
                        new_ids.append(v)
                if new_ids != old_ids:
                    self.manager.store.update(mid, entity_ids=new_ids)
                    remapped += 1
                if mid not in c["memory_ids"]:
                    c["memory_ids"].append(mid)
            # Origin only goes up, never down: if the alias entry was confirmed
            # by the lexicon, the canonical name should receive that trust too
            if _ORIGIN_RANK.get(a.get("origin", ENTITY_ORIGIN_HEURISTIC), 0) > \
                    _ORIGIN_RANK.get(c.get("origin", ENTITY_ORIGIN_HEURISTIC), 0):
                c["origin"] = a["origin"]
            del self.entities[alias]
            merged += 1
        if merged or renamed:
            self._recompute_entity_stats()
            self._alias_reverse = None
        return {"merged": merged, "renamed": renamed,
                "remapped_memories": remapped}




    def _alias_direction(
        self,
        a: str,
        b: str,
        in_lexicon: set[str],
        docs: dict[str, int],
    ) -> tuple[str, str, str]:
        """Decide which of the pair is the alias and which the canonical name;
        returns ``(alias, canonical, basis)``.

        Rules ordered by trustworthiness; **each tier's reason must be
        stateable** (the basis appears in the candidate, because guessing the
        direction wrong means merging two archives wrongly -- a human must be
        able to re-check at a glance):

          1. One side already registered in the lexicon as an alias -> the
             canonical name is the one it points to (a human already decided)
          2. One side is in the lexicon (formal entry) -> it is the canonical
             name
          3. The side with more mentions -> canonical (max of archive
             mention_count and co-occurrence count)
          4. The longer side -> canonical (full form over abbreviation)
          5. Still tied -> lexicographic order, and the basis notes
             "direction needs manual confirmation"
        """
        ca = self.canonical_entity(a)
        cb = self.canonical_entity(b)
        a_is_alias = ca.lower() != a.lower()
        b_is_alias = cb.lower() != b.lower()
        if a_is_alias and not b_is_alias:
            return b, ca, f"{a} is already registered in the lexicon as an alias of {ca}"
        if b_is_alias and not a_is_alias:
            return a, cb, f"{b} is already registered in the lexicon as an alias of {cb}"
        a_lex, b_lex = a in in_lexicon, b in in_lexicon
        if a_lex != b_lex:
            return (b, a, "registered in lexicon") if a_lex else (a, b, "registered in lexicon")
        na = max(docs.get(a, 0), self.entities.get(a, {}).get("mention_count", 0))
        nb = max(docs.get(b, 0), self.entities.get(b, {}).get("mention_count", 0))
        if na != nb:
            return ((b, a, f"mentioned more often ({na} > {nb})") if na > nb
                    else (a, b, f"mentioned more often ({nb} > {na})"))
        if len(a) != len(b):
            return ((b, a, f"longer form wins (full name over abbreviation, {len(a)}/{len(b)} chars)")
                    if len(a) > len(b)
                    else (a, b, f"longer form wins (full name over abbreviation, {len(b)}/{len(a)} chars)"))
        if a <= b:
            return b, a, "cannot decide; ordered lexicographically -- confirm the direction manually"
        return a, b, "cannot decide; ordered lexicographically -- confirm the direction manually"



    def discover_aliases(
        self,
        top_n: int = 30,
        min_confidence: float = 0.35,
        min_support: int = 2,
        min_lift: float = 2.0,
        min_containment: float = 0.5,
        allow_stat_only: bool = False,
        max_evidence: int = 3,
        max_names_per_record: int = 12,
        exclude: list[str] | None = None,
        cue_words: list[str] | None = None,
        with_paren: bool = True,
        refine: bool = True,
        max_probe_names: int = 200,
    ) -> dict[str, Any]:
        """Mine candidates for "several names for the same entity". **Candidates
        and evidence only; nothing is written to disk.**

        Division of labor vs :meth:`lexicon_suggest`: that one asks "this word
        is not registered yet -- should it be registered as a proper noun";
        this one asks "these two names refer to the same thing" -- the
        question is a **relation**, not a word.

        Why discover automatically: the normalization **mechanism**
        (``lexicon_add("别名 => 主名")``) has existed for a long time; what
        was missing is **discovery** -- when the same entity appears in three
        spellings across the corpus, a human must first read them before
        thinking of registering them. This step takes the "read the whole
        library looking for names" work back from human hands.

        Why candidates only: once merged, an alias **changes entity
        identity**; guessing wrong means mixing memories of two different
        objects into one archive, hard to untangle afterwards. So this never
        writes to disk automatically; it only returns ``paste`` (a line
        directly pasteable into ``lexicon_add``) and ``evidence`` (original
        text).

        Signals and criteria are in :mod:`hippocampus_mcp.domain.aliases`;
        five hard gates:
          * **Shape**: both sides must be name-shaped (pure Chinese 2~12 /
            pure Latin) and must not be pure number words ("第三""一百二十");
          * **Noise**: neither side may hit the three stop-word layers (same
            source as ``lexicon_suggest``) or the role/relation-word table
            (``作者/读者/主角/用户…``);
          * **Cue position**: the separator must come **before** the cue word
            ("甲，字乙"). Cue words are common characters (字/号/名/称); the
            same character inside "句号""编号""简称" is not an appellation --
            real-library testing showed this is the **biggest noise source**:
            relaxing to "separator on either side" produced false hits like
            「碎句号、长段落」「全量换号、活跃文档」「同章全称后简称」
            accounting for more than half of candidates (16 of 25). Drop
            counts in ``skipped.cue_context`` / ``cue_sep_after_only``.
          * **Co-occurrence needs an anchor**: pure co-occurrence pairs
            (``cooc``) require at least one side already registered in the
            lexicon, or a containment relation between the two sides
            (abbreviation/full form) -- otherwise "two book titles always
            appearing together" would also become candidates.
            (Note: this was **not** extended to the pattern class. Tested;
            does not hold -- see below.)
          * **Statistical signals cannot stand alone**
            (``allow_stat_only=False``, default): ``cooc`` and ``abbrev``
            only **reinforce** pattern signals. Real-library testing: pairs
            like ``台湾外记 => 台湾`` / ``剡溪雪 => 剡溪`` /
            ``听女道士卞玉京弹琴歌 => 卞玉京`` have 100% containment and lifts
            in the hundreds, all perfect on "co-occurrence", yet they are not
            the same thing -- **a title containing someone's name does not
            make the title that person's alias**. Substring + co-occurrence
            cannot derive "same referent"; this is the round's most important
            tested conclusion.

        Why the pattern class does **not** get an "at least one end in the
        library" anchor (tried; rejected by the data): of 50 real-library
        candidates, 44 had "neither end in the entity index", and 8 of the 9
        judged true by a human **were also single-end-unknown** ("苏文 =>
        阿文""明远 => 于山""云锦 => 俞锦" were all absent from the entity
        index) -- while blocking noise, the anchor would delete true
        candidates along with them. What separates true from false is **cue
        position**, not "in the library or not".

        Args:
            top_n: max candidates returned, default 30 (descending by
                confidence).
            min_confidence: confidence gate, default 0.35. Cue-word hits
                (0.50) pass alone; parenthetical appositions (0.30) **cannot
                pass alone**, needing a few more pieces of evidence (support
                bonus) or another signal -- real-library testing found too
                many annotational parentheses like "（一手）""（作者）";
                letting parentheses stand alone would sink the whole table's
                usability.
            allow_stat_only: whether candidates may rest **only on
                statistical signals** (co-occurrence / containment), default
                False. See the hard-gates notes above -- substring +
                co-occurrence cannot derive "same referent".
            min_support: minimum shared-appearance count for pure
                co-occurrence pairs, default 2.
            min_lift: lift gate for pure co-occurrence pairs, default 2.0
                (independence is ~1).
            min_containment: containment gate for pure co-occurrence pairs,
                default 0.5 -- ``共同出现条数 / 两侧较少的那一方的条数``, i.e.
                "when the less-mentioned side appears, how likely the other
                side is present too". **Parallel to min_lift; either one
                suffices**: on small corpora with few names lift naturally
                approaches 1, and lift alone would miss whole batches of true
                candidates.
            max_evidence: max original-text evidence items per candidate,
                default 3.
            max_names_per_record: max names per memory entering pairwise
                matching, default 12 -- a cost gate, preventing one
                "entity list"-style long text from dragging the mining down.
            exclude: temporary exclusions for this run (not persisted), same
                meaning as ``lexicon_suggest``.
            cue_words: extra appellation cue words for this run (not
                persisted).
            with_paren: whether to scan parenthetical appositions, default
                True.
            refine: whether to run a **second whole-library sweep**, default
                True. Unregistered aliases are only "known" in the one memory
                the pattern caught in round one, badly undercounting support;
                round two takes round one's candidate names (usually a few
                dozen) and scans the library exactly once more to count
                "appears in multiple memories". On large libraries this is
                the main time cost; turn it off for speed (noticeably loses
                recall).
            max_probe_names: how many candidate names the second sweep may
                probe, **cost gate** (default 200).

        Returns:
            ``candidates`` (alias/canonical/confidence/**band**/signals/
            support/lift/direction_basis/paste/evidence) - ``scanned`` -
            ``cue_words`` - ``stop_words`` - ``skipped`` (how many each gate
            blocked, to judge whether a threshold is set wrong) - ``note``.
        """
        in_lexicon = {t.lower() for t in self.entity_lexicon}
        trusted_keys = {
            k.lower() for k, e in self.entities.items() if self._entity_trusted(e)
        }
        # Known terms serve only as **segmentation aid** (cutting "顾言" out
        # of "此人是顾言，字子安"), not as correctness judges -- prefer a
        # cleaner set over stuffing heuristic noise in.
        known = in_lexicon | trusted_keys
        name_disp = {k.lower(): e.get("name", k) for k, e in self.entities.items()}
        lex_disp = {t.lower(): t for t in self.entity_lexicon}

        cue_path = None
        if self.entity_lexicon_path:
            cue_path = os.path.join(
                os.path.dirname(self.entity_lexicon_path), alias_mod.ALIAS_CUE_FILENAME
            )
        cue_set, cue_stat = alias_mod.load_cue_words(cue_path, cue_words)
        stop, stop_grams, stop_stat = self._stop_grams(exclude)
        stop_list = sorted(stop)

        agg: dict[tuple[str, str], dict[str, Any]] = {}

        def _touch(ka: str, kb: str, a: str | None = None, b: str | None = None):
            key = (ka, kb) if ka < kb else (kb, ka)
            ent = agg.get(key)
            if ent is None:
                ent = {"signals": set(), "cues": set(), "evidence": [],
                       "support": 0, "rids": set(), "lift": 0.0, "disp": {}}
                agg[key] = ent
            ent["disp"].setdefault(ka, a or name_disp.get(ka) or lex_disp.get(ka) or ka)
            ent["disp"].setdefault(kb, b or name_disp.get(kb) or lex_disp.get(kb) or kb)
            return ent

        # ---- Scan the library: pattern-class signals + name set for co-occurrence
        # These two are **one and the same definition**: divergent assertions drift.
        #   Shape (name_like + not a role/relation word + no stop-word collision)
        #   -- only decides "is this name worth feeding into statistics"; it
        #   does not affect final candidate adjudication.
        def _probe_ok(w: str) -> bool:
            if not alias_mod.name_like(w):
                return False
            if alias_mod.is_alias_noise(w):
                return False
            return not (w in stop_grams or any(s and s in w for s in stop_list))

        id_to_key = {e["id"]: e["key"] for e in self.entities.values()}
        n_records = 0
        n_pattern_records = 0
        probe: set[str] = set()
        pass1: list[tuple[str, set[str]]] = []
        ctx_stat: dict[str, int] = {}
        for rec in self.manager.store.iter_records():
            if rec.state == MemoryState.ARCHIVED.value:
                continue
            n_records += 1
            # Name set = archived trusted entities ∪ **names the pattern mined
            # from this record**. The latter matters: patterns only prove
            # "they are listed together in this one record"; co-occurrence is
            # what proves "they always appear together across many records" --
            # only when both signals stack on the same candidate pair do
            # confidence and support reflect true strength.
            names = {
                id_to_key[eid] for eid in (rec.entity_ids or [])
                if id_to_key.get(eid) in trusted_keys
            }
            text = self._preprocess_for_entities(rec.content or "")
            hits = (
                alias_mod.mine_patterns(text, cue_set, known,
                                        with_paren=with_paren, stats=ctx_stat)
                if text.strip() else []
            )
            if hits:
                n_pattern_records += 1
            rec_probe: set[str] = set()
            for p in hits:
                ka, kb = p.a.lower(), p.b.lower()
                if ka == kb:
                    continue
                names.add(ka)
                names.add(kb)
                if p.signal == "cue":
                    # **Only cue-signal names become probe-sweep candidates**.
                    # Rationale: a cue is an explicit statement "this is
                    # another name for the same thing" ("甲，字乙"), while
                    # parentheses are mere juxtaposition -- "一手" in
                    # "《明季南略》（一手）" is an annotation, not an
                    # appellation. Feeding parenthesis products into the
                    # whole-library probe sweep **amplifies one misjudgment
                    # into dozens of support counts**.
                    rec_probe.add(ka)
                    rec_probe.add(kb)
                ent = _touch(ka, kb, p.a, p.b)
                ent["signals"].add(p.signal)
                if p.cue:
                    ent["cues"].add(p.cue)
                if rec.id not in ent["rids"]:
                    ent["rids"].add(rec.id)
                    ent["support"] += 1
                if len(ent["evidence"]) < max_evidence:
                    ent["evidence"].append({
                        "memory_id": rec.id, "signal": p.signal,
                        "cue": p.cue, "text": p.snippet,
                    })
            pass1.append((rec.id, names))
            probe |= {x for x in rec_probe if _probe_ok(x)}

        # ---- Round two: the candidate-name set is small; one exact whole-library
        # sweep yields true support -----------------------------------------
        # In round 1, **unrecognized aliases** ("子安"-style) are only caught
        # in the one record the pattern matched; when they appear in other
        # memories nobody recognizes them, badly undercounting support. The
        # candidate name set is small (usually a few dozen), so a second
        # library sweep to recognize them fully is worth it -- support is the
        # precondition for the co-occurrence signal to say anything; without
        # this round, "holds across many memories" is never seen.
        cooc_input = pass1
        probe_list: list[str] = []
        if refine and probe:
            probe_list = sorted(probe)[:max_probe_names]
            cooc_input = []
            for rec in self.manager.store.iter_records():
                if rec.state == MemoryState.ARCHIVED.value:
                    continue
                low = (rec.content or "").lower()
                names = {
                    id_to_key[eid] for eid in (rec.entity_ids or [])
                    if id_to_key.get(eid) in trusted_keys
                }
                names |= {n for n in probe_list if n in low}
                cooc_input.append((rec.id, names))

        # ---- Co-occurrence + lift -----------------------------------------
        pair_counts, docs, n_seen = alias_mod.cooccurrence(
            cooc_input, max_names_per_record=max_names_per_record
        )
        n_cooc_kept = 0
        for (ka, kb), cnt in pair_counts.items():
            if cnt < min_support:
                continue
            da, db = docs.get(ka, 0), docs.get(kb, 0)
            lf = alias_mod.lift_of(cnt, da, db, n_seen)
            # Two **parallel** criteria; either one suffices:
            #   (1) lift (exceeding the independence assumption) -- fits
            #       "abbreviation and full form appearing together" cases;
            #   (2) containment count_ab / min(count_a, count_b) -- "whenever
            #       a is mentioned, b is always mentioned too". On small
            #       corpora lift naturally approaches 1 (the library just has
            #       few names); lift alone would miss whole batches of true
            #       candidates.
            contain = cnt / min(da, db) if min(da, db) > 0 else 0.0
            if lf < min_lift and contain < min_containment:
                continue
            rel = alias_mod.abbrev_relation(ka, kb)
            if ka not in in_lexicon and kb not in in_lexicon and not rel:
                # Pure co-occurrence needs an anchor: otherwise "two book
                # titles always appearing together" would also become candidates
                continue
            n_cooc_kept += 1
            ent = _touch(ka, kb)
            ent["signals"].add("cooc")
            ent["lift"] = max(ent["lift"], lf)
            ent["support"] = max(ent["support"], cnt)
            if rel:
                ent["signals"].add("abbrev")
            if len(ent["evidence"]) < max_evidence:
                ent["evidence"].append({
                    "memory_id": "", "signal": "cooc", "cue": "",
                    "text": f"co-occur {cnt} times (lift {lf:.1f}x / containment {contain:.0%}）",
                })

        # ---- Gate -> direction -> scoring -> ordering ----------------------
        skipped = {"declared": 0, "shape": 0, "noise": 0, "stop": 0,
                   "stat_only": 0, "lowconf": 0,
                   # Hits dropped because no separator precedes the cue word;
                   # the subset "separator only after the cue" is the
                   # **typical shape of a false hit** (句号/编号/简称) --
                   # surfaced separately so one can judge whether this gate
                   # is over-killing.
                   "cue_context": ctx_stat.get("cue_no_sep_before", 0),
                   "cue_sep_after_only": ctx_stat.get("cue_sep_after_only", 0)}
        # Statistical signals (co-occurrence / containment) can **only
        # reinforce pattern signals; they cannot become candidates alone**.
        # Real-library testing: pairs like `台湾外记 => 台湾`, `剡溪雪 => 剡溪`,
        # `听女道士卞玉京弹琴歌 => 卞玉京` where a book title/heading contains
        # a person/place name have perfect containment and co-occurrence
        # (containment 100%, lift in the hundreds), yet they are **not the
        # same thing** -- a name appearing in a title does not make the title
        # that person's alias. Substring + co-occurrence cannot derive "same
        # referent".
        _STAT_SIGNALS = {"cooc", "abbrev"}

        def _bad_stop(w: str) -> bool:
            if w in stop_grams:
                return True
            return any(s and s in w for s in stop_list)

        candidates: list[dict[str, Any]] = []
        for (ka, kb), ent in agg.items():
            if not (alias_mod.name_like(ka) and alias_mod.name_like(kb)):
                skipped["shape"] += 1
                continue
            if alias_mod.is_alias_noise(ka) or alias_mod.is_alias_noise(kb):
                skipped["noise"] += 1
                continue
            if _bad_stop(ka) or _bad_stop(kb):
                skipped["stop"] += 1
                continue
            # Already declared the same family (both sides normalize to the
            # same canonical name) -> already normalized
            if self.canonical_entity(ka).lower() == self.canonical_entity(kb).lower():
                skipped["declared"] += 1
                continue
            if not allow_stat_only and ent["signals"] <= _STAT_SIGNALS:
                skipped["stat_only"] += 1
                continue
            conf = alias_mod.confidence_of(ent["signals"], ent["support"], ent["lift"])
            if conf < min_confidence:
                skipped["lowconf"] += 1
                continue
            alias, canon, basis = self._alias_direction(ka, kb, in_lexicon, docs)
            disp = ent["disp"]
            a_disp = disp.get(alias.lower(), alias)
            c_disp = disp.get(canon.lower(), canon)
            candidates.append({
                "alias": a_disp,
                "canonical": c_disp,
                "confidence": conf,
                # Three bands, giving the caller a triage basis it does not
                # have to recompute:
                #   high   >=0.60 multi-signal (or high support) -- worth
                #       reviewing directly;
                #   medium 0.45~0.60 single cue word -- **must read each
                #       evidence item**;
                #   low    <0.45 parenthesis/statistics only -- mostly
                #       annotational juxtaposition.
                # Why this band: on the real library single-character cue
                # words (字/号/名) are ambiguous; "明远 => 于山""云锦 =>
                # 俞锦" and "改为 => 同安""无事 => 自建" share the very same
                # 0.52 confidence -- **the score cannot separate them**; only
                # a human reading the evidence can.
                "band": ("high" if conf >= 0.60 else
                         "medium" if conf >= 0.45 else "low"),
                "signals": sorted(ent["signals"]),
                "cue_words": sorted(ent["cues"]),
                "support": ent["support"],
                "lift": round(ent["lift"], 2),
                "direction_basis": basis,
                "paste": f"{a_disp} => {c_disp}",
                "evidence": ent["evidence"][:max_evidence],
            })
        candidates.sort(key=lambda c: (-c["confidence"], -c["support"], c["alias"]))
        return {
            "candidates": candidates[:top_n] if top_n > 0 else candidates,
            "scanned": {
                "records": n_records,
                "records_with_pattern": n_pattern_records,
                "records_with_names": n_seen,
                "pairs": len(agg),
                "cooc_pairs_kept": n_cooc_kept,
                # Which names the round-two probe sweep used (from **cue**
                # signals only, see above). Exposed so that "were parenthesis
                # products amplified library-wide" is auditable.
                "probe_names": probe_list,
            },
            "cue_words": cue_stat,
            "stop_words": stop_stat,
            "skipped": skipped,
            "note": "candidates only, nothing written automatically; after confirming the direction, persist via lexicon_add(paste)",
        }



    def lexicon_suggest(
        self,
        top_n: int = 30,
        min_docs: int = 4,
        min_files: int = 2,
        min_len: int = 3,
        max_len: int = 4,
        exclude: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List candidate proper nouns that "appear frequently but are not yet
        registered in the lexicon", for human screening and later appending.

        **Heuristic extraction is not used as the signal source** (rejected by
        testing): Chinese has no word segmentation or NER;
        ``_heuristic_entities`` relies on quote/sentence-pattern regexes and
        on the real library's 9359 records produced only 34 distinct kinds
        with a top frequency of 2, all shredded fragments like "活人的制"
        "人成本逐" -- it is not a proper-noun finder; using it as the
        candidate source would deliver a pile of garbage.

        Instead uses **Chinese n-gram frequency statistics** (the simplest
        zero-dependency new-word discovery):
          1. Sliding windows of 2~4 characters count "in how many memories it
             appears" (deduped within one record, so a single short text
             cannot inflate a term) and "across how many source files";
          2. Drop terms containing noise characters, **hitting stop words**
             (built-in generic jargon ∪ project file ∪ exclude), already in
             the lexicon, or already trusted entities;
          3. **Fragment suppression**: if a short candidate is contained by a
             longer candidate at >=90% of the latter's frequency (e.g. "活人"
             always appearing inside "活人的制"), judge it a slicing fragment
             and drop it.

        Args:
            min_docs: minimum number of memories it appeared in, default 4
                (the main gate).
            min_files: minimum number of source files, default 2. Terms
                **recurring only within a single file are usually that file's
                section headings / terminology**, not proper nouns to
                register.
            min_len / max_len: candidate length, default 3~4. Does not start
                at 2 characters by default -- testing showed 2-char
                candidates are almost entirely generic words like
                "主角/用户/正文/定案/第一次".
            exclude: temporary exclusion words (one-shot demining, not
                persisted). **Stop words total three layers, added
                cumulatively**:
                (1) **built-in generic jargon** (`_DEFAULT_LEXICON_STOP`, in
                    code, works out of the box) -- cross-project boilerplate
                    like "本章/正文/时间线/待核清单/用户定案/第一次", plus a
                    "pure Chinese numeral + measure word" rule (第五节/十二年);
                (2) **project stop-word file** `<词库同目录>/entity_lexicon_stop.txt`
                    (one per line, # comments) -- for **project-specific**
                    structural labels/jargon (e.g. "感情线/伏笔表/制度篇").
                    The built-in layer handles only generic words; project
                    words go here;
                (3) this `exclude` argument -- words to exclude just this
                    once.
                Matching also **suppresses slices of already-registered
                terms** (registering "用户定案" also blocks 3-char fragments
                like "户定案/页定案"), so write the full phrase; no need to
                hand-enumerate prefix fragments.
        """
        in_lexicon = {t.lower() for t in self.entity_lexicon}
        trusted_keys = {
            k.lower() for k, e in self.entities.items() if self._entity_trusted(e)
        }
        # Three stop-word layers (additive only; later layers do not override
        # earlier ones):
        #   (1) **Built-in** generic jargon `_DEFAULT_LEXICON_STOP` -- works
        #       out of the box, needs no configuration. This is the root
        #       cause of the pre-D4 "high noise": old versions had only (2)
        #       and (3), so a new project without the file was inevitably
        #       dominated by words like "用户定案/时间线/第一次".
        #   (2) Project stop-word file `<词库同目录>/entity_lexicon_stop.txt`
        #   (3) this call's `exclude` argument (temporary, not persisted)
        stop, stop_grams, stop_stat = self._stop_grams(exclude)
        logger.info(
            "[lexicon_suggest] %d stopwords (built-in %d / project file %d / excluded this run %d)",
            len(stop), stop_stat["builtin"], stop_stat["file"], stop_stat["extra"],
        )
        noise = tuple(self._NOISE_CHARS)
        # Count one extra layer (not output): 4-char candidates need 5-char
        # frequencies to identify whether they are fragments -- otherwise
        # "天工开"/"工开物" would each sneak into candidates with 48 records,
        # though they are mere slices of "天工开物".
        stat_max = max_len + 1
        docs: dict[str, int] = {}
        # Full counts (**no** lexicon/trusted-entity filtering), used solely
        # for fragment suppression. Without it this absurdity happens:
        # "天工开物" is already in the lexicon -> excluded -> nothing can
        # prove "天工开" and "工开物" are only its slices -> both fragments
        # sneak into the candidate list with 48 records each.
        raw: dict[str, int] = {}
        files: dict[str, set[str]] = {}
        for rec in self.manager.store.iter_records():
            if rec.state == MemoryState.ARCHIVED.value:
                continue
            src = rec.source or ""
            seen: set[str] = set()
            for run in re.findall(r"[\u4e00-\u9fa5]{2,}", rec.content):
                for n in range(min_len, stat_max + 1):
                    if len(run) < n:
                        continue
                    for i in range(len(run) - n + 1):
                        g = run[i : i + n]
                        if g in seen:
                            continue
                        if any(ch in g for ch in noise) or g.lower() in stop_grams:
                            continue
                        if g[0] in _ORDINAL_HEADS and _ORDINAL_RE.match(g):
                            continue
                        seen.add(g)
                        raw[g] = raw.get(g, 0) + 1
                        if g.lower() in in_lexicon or g.lower() in trusted_keys:
                            continue
                        docs[g] = docs.get(g, 0) + 1
                        if n <= max_len:
                            files.setdefault(g, set()).add(src)

        # Fragment suppression: drop a short candidate "wrapped" by a longer
        # one (compared within the candidate set only, no library re-scan)
        pool = {t: c for t, c in docs.items() if c >= min_docs}
        # Fragment suppression: the criterion uses raw (long words already in
        # the lexicon can also suppress their own fragments).
        #
        # The approach is "long suppresses short", not "find superstrings for
        # each short candidate" -- the latter compares hundreds of short
        # candidates against millions of grams, measured at 63 seconds.
        # Instead iterate long grams, split out their substrings, and **first
        # prune low-frequency long grams that cannot suppress anything**
        # (candidates all have docs >= min_docs, so suppressing one needs at
        # least 0.9*min_docs frequency) -- measured back to seconds.
        short_set = {t for t in pool if len(t) < stat_max}
        floor = 0.9 * min_docs
        best: dict[str, int] = {}
        for g, c in raw.items():
            if c < floor or len(g) <= min_len:
                continue
            n = len(g)
            for m in range(min_len, n):
                for i in range(n - m + 1):
                    sub = g[i : i + m]
                    if sub in short_set and c > best.get(sub, 0):
                        best[sub] = c
        dropped = {s for s, c in best.items() if c >= 0.9 * raw.get(s, 0)}

        rows = [
            {
                "term": t,
                "docs": c,
                "files": len(files.get(t, ())),
                "sample_sources": sorted(files.get(t, ()))[:3],
            }
            for t, c in pool.items()
            if len(t) <= max_len and t not in dropped
            and len(files.get(t, ())) >= min_files
        ]
        rows.sort(key=lambda r: (-r["docs"], -r["files"], r["term"]))
        return rows[:top_n] if top_n > 0 else rows
