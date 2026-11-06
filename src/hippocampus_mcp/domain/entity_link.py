"""Entity-linkage family: entity extraction orchestration + entity table
writes.

Moved wholesale from ``brain.core``: ``extract_entities_with_origin``
(multi-channel extraction → cleaning/dedup merged by trust → alias
canonicalization) and ``_link_entities`` (entity table writes and inverted
index maintenance). All external capabilities of both are fetched by the
shell **at call time** and injected; this module has zero imports:

  * extract: ``preprocess`` / ``cjk_ratio`` / ``get_nlp`` (lazy -- spaCy is
    not loaded unless called) / ``title_entities`` / ``heuristic_entities``
    / ``clean_name`` / ``canonicalize_map`` are brain-side capability
    callbacks; ``lexicon`` is the lexicon sequence (hot reload swaps it
    wholesale; the shell fetches it every time); the four origin constants
    and ``origin_rank`` / ``cjk_skip_ratio`` are passed by value.

Semantics preserved verbatim (touching any one is a behavior change; the
parity tests will catch it):

  * extract's four channels have a fixed order: lexicon substrings
    (case-insensitive) → spaCy (**only when the English ratio is high**;
    exceptions swallowed, never fatal) → 《》 book-title marks → heuristic
    fallback;
  * the dedup key is ``lower().strip() after clean``; **single-character
    candidates are dropped** (``len(k) <= 1``); for the same key, a later
    candidate overwrites origin only when **strictly more trusted** (equal
    rank does not overwrite -- first come, first served, preserving the
    first-seen display name);
  * an empty prepared text (empty after ``strip()``) returns an empty dict
    directly; none of the four channels run;
  * link: names are deduped with ``dict.fromkeys`` first; a missing origin
    is treated as heuristic (untrusted); upgrades go **only up, never down**
    (written only on strictly greater); ``mention_count`` is a derived
    quantity always equal to ``len(memory_ids)`` (never accumulated); a
    falsy ``memory_id`` does not enter the inverted index; ``last_seen``
    takes the max, and ``first_seen`` resets when the inverted index is
    empty.
"""
from __future__ import annotations

from typing import Any, Callable


def extract_entities_with_origin(
    text: str,
    *,
    preprocess: Callable[[str], str],
    cjk_ratio: Callable[[str], float],
    cjk_skip_ratio: float,
    lexicon: Any,
    get_nlp: Callable[[], Any],
    title_entities: Callable[[str], list],
    heuristic_entities: Callable[..., list],
    clean_name: Callable[[str], str],
    canonicalize_map: Callable[[dict], dict],
    origin_rank: dict,
    origin_lexicon: str,
    origin_spacy: str,
    origin_title: str,
    origin_heuristic: str,
) -> dict[str, str]:
    """Same as ``extract_entities`` but returns ``{entity name: origin}``.

    When the same entity is hit by multiple channels, the **most trusted**
    origin wins (manual > lexicon > spacy > heuristic). The return order is
    descending by trust, matching the old ``extract_entities``'s ordering
    semantics. Channels with no output leave no placeholder keys.
    """
    prepared = preprocess(text)
    if not prepared.strip():
        return {}

    is_cjk = cjk_ratio(prepared) > cjk_skip_ratio
    candidates: list[tuple[str, str]] = []

    # 1) Proper-noun lexicon: substring hits (case-insensitive)
    if lexicon:
        lowered = prepared.lower()
        for term in lexicon:
            if term.lower() in lowered:
                candidates.append((term, origin_lexicon))

    # 2) spaCy: enabled only for text with a high English ratio
    if not is_cjk:
        nlp = get_nlp()
        if nlp is not None:
            try:
                doc = nlp(prepared)
                for ent in doc.ents:
                    if ent.label_ in (
                        "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                        "WORK_OF_ART", "NORP", "FAC",
                    ):
                        candidates.append((ent.text.strip(), origin_spacy))
            except Exception:
                pass

    # 3) 《》 book-title marks: Chinese's explicit "work title" marker, given
    # its own trusted title tier
    for n in title_entities(prepared):
        candidates.append((n, origin_title))

    # 4) Heuristic fallback (spaCy unavailable / no output / Chinese text all
    # land here)
    for n in heuristic_entities(prepared, cjk=is_cjk):
        candidates.append((n, origin_heuristic))

    out: dict[str, str] = {}        # display name (original casing kept) -> origin
    by_key: dict[str, str] = {}     # lower -> display name (dedup by lowercase)
    for n, origin in candidates:
        cleaned = clean_name(n)
        if not cleaned:
            continue
        k = cleaned.lower().strip()
        if not k or len(k) <= 1:
            continue
        disp = by_key.get(k)
        if disp is None:
            by_key[k] = cleaned
            out[cleaned] = origin
        elif origin_rank[origin] > origin_rank[out[disp]]:
            out[disp] = origin
    # 5) Alias canonicalization: when the lexicon says ``子安 => 顾言``, this
    #    folds "子安" into "顾言". Folding only one side equals folding
    #    neither -- without folding on the extraction side, the library
    #    splits one entity's multiple names into disconnected profiles;
    #    without folding on the query side, querying "子安" still misses the
    #    entity bonus of "顾言". Both sides go through this method
    #    (recall/block_4/ingest all call it), so they stay consistent by
    #    construction.
    return canonicalize_map(out)


def link_entities(
    names: list[str],
    memory_id: str,
    ts: float,
    origins: dict[str, str] | None = None,
    *,
    entities: dict,
    nid: Callable[[str], str],
    origin_rank: dict,
    origin_heuristic: str,
) -> list[str]:
    """Write entities into the entity table, maintain the inverted index,
    and return the list of entity ids.

    ``mention_count`` means "mentioned by how many memories" and always
    equals ``len(memory_ids)`` -- **derived, not accumulated**. The earlier
    accumulating implementation inflated the count (a single memory could
    push it above 3), rendering the profile admission gate effectively
    void, and it could not self-correct after a memory deletion. Names are
    deduped first to avoid counting the same entity twice.

    A missing ``origins`` entry is treated as ``heuristic`` (untrusted), so
    a caller that forgets to annotate does not get trust for free; when the
    same entity is hit multiple times from different sources, **upgrade
    only, never downgrade**. ``entities`` is brain's entity table proper
    (mutated in place; the container is not replaced).
    """
    origins = origins or {}
    ids = []
    for name in dict.fromkeys(names):
        key = name.lower()
        new_origin = origins.get(key, origin_heuristic)
        ent = entities.get(key)
        if ent is None:
            ent = {
                "id": nid("ent"),
                "name": name,
                "key": key,
                "mention_count": 0,
                "memory_ids": [],
                "first_seen": ts,
                "last_seen": ts,
                "origin": new_origin,
            }
            entities[key] = ent
        elif origin_rank.get(new_origin, 0) > origin_rank.get(
            ent.get("origin", origin_heuristic), 0
        ):
            # Upgrade: the candidate origin was confirmed by a more trusted one
            ent["origin"] = new_origin
        if memory_id and memory_id not in ent["memory_ids"]:
            ent["memory_ids"].append(memory_id)
        # Derived quantity: kept identical to memory_ids, so it stays correct
        # after memory deletions
        ent["mention_count"] = len(ent["memory_ids"])
        ent["last_seen"] = max(ent.get("last_seen", ts), ts)
        if not ent["memory_ids"]:
            ent["first_seen"] = ts
        ids.append(ent["id"])
    return ids
