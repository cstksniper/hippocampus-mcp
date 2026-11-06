"""Miscellaneous pure-logic family: entity relink orchestration +
three-layer stopword expansion.

This module collects the two pieces of pure logic without a better home
(``source_drift`` / ``forget_by_query`` were already existing thin
shells on the baseline, carried respectively by
``store.source_sig.drift_audit`` and ``domain/forget_flow``, with no
inlinable logic), actually extracting ``relink_entities`` and
``_stop_grams``.

Zero in-project imports (architecture constraint: domain → common
only). Both pieces' external capabilities are injected by the shell:

  * ``relink_entities``: ``extract_fn`` / ``link_fn`` (brain's entity
    extraction and linking, lazily injected -- no switch, called live
    per record), ``merge_fn`` (the store layer's id-list union; domain
    may not import store), ``update_fn`` (store write-back),
    ``archived_state`` (state string value; unpatchable constants
    passed by value);
  * ``stop_grams``: ``lexicon_path`` (lexicon path value),
    ``load_lexicon`` (lexicon-file read callback), ``builtin_stop``
    (built-in stopword values), ``exclude`` (this call's exclusions).

Semantics preserved verbatim (changing any one is a behavior change;
parity will catch it):

  * relink: ``list(records)`` materialized before iterating (the loop
    writes back to the store; guards against iterator invalidation);
    archived skipped; a falsy ``only_ids`` means unrestricted; empty
    origins skipped; write-back only when
    ``set(merged) != set(rec.entity_ids or [])`` -- the counts are
    「how many records were touched」 and 「how many links were added」
    (union minus old set);
  * stop_grams: three layers add-only, never overwrite (builtin ∪
    lexicon file ∪ exclude); lexicon and file words are uniformly
    ``lower()``; substring expansion is **all prefixes** (all slices of
    length 1..len); registering one full phrase suppresses all of its
    slices; the count denominators carry each layer's entry count.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable


def relink_entities(
    records: Iterable[Any],
    *,
    extract_fn: Callable[[str], dict],
    link_fn: Callable[[list, str, float, dict], list],
    merge_fn: Callable[[Any, Any], list],
    update_fn: Callable[..., Any],
    archived_state: str,
    only_ids: list[str] | None = None,
) -> dict[str, int]:
    """Re-run entity extraction on existing memories and **relink**
    (the necessary companion of lexicon expansion).

    After new words enter the lexicon, existing memories'
    ``entity_ids`` do not change automatically. Without relinking it is
    "added the words but nothing is found" -- the lexicon-expansion
    loop broken at its last link. Relinking merges by union and does
    not overwrite existing entity associations (same merge semantics as
    ingest).
    """
    only = set(only_ids) if only_ids else None
    touched = 0
    new_links = 0
    for rec in list(records):
        if rec.state == archived_state:
            continue
        if only is not None and rec.id not in only:
            continue
        origins = extract_fn(rec.content)
        if not origins:
            continue
        eids = link_fn(list(origins.keys()), rec.id, rec.timestamp, origins)
        merged = merge_fn(rec.entity_ids, eids)
        if set(merged) != set(rec.entity_ids or []):
            update_fn(rec.id, entity_ids=merged)
            touched += 1
            new_links += len(set(merged) - set(rec.entity_ids or []))
    return {"records_touched": touched, "new_links": new_links}


def stop_grams(
    lexicon_path: str | None,
    load_lexicon: Callable[[str], Iterable[str]],
    *,
    builtin_stop: Iterable[str],
    exclude: list[str] | None = None,
) -> tuple[set[str], set[str], dict[str, int]]:
    """Three stopword layers → ``(original-word set, expanded substring
    set, per-layer entry counts)``.

    The three layers **add only, never overwrite**: ① built-in generic
    jargon ∪ ② the project stopword file
    ``<lexicon dir>/entity_lexicon_stop.txt`` ∪ ③ this call's
    ``exclude``.

    "Expanding into a substring set" exists so that **registering one
    full phrase suppresses all of its slices** (writing 「用户定案」
    also blocks 「户定案」), with no need to hand-enumerate prefix
    fragments.
    """
    stop: set[str] = {t.lower() for t in builtin_stop}
    file_stop: set[str] = set()
    if lexicon_path:
        stop_path = os.path.join(
            os.path.dirname(lexicon_path), "entity_lexicon_stop.txt"
        )
        if os.path.exists(stop_path):
            file_stop = {t.lower() for t in load_lexicon(stop_path)}
            stop |= file_stop
    ex = {t.lower() for t in (exclude or [])}
    stop |= ex
    grams: set[str] = set()
    for s in stop:
        for m in range(1, len(s) + 1):
            for i in range(len(s) - m + 1):
                grams.add(s[i : i + m])
    return stop, grams, {
        "builtin": len(builtin_stop),
        "file": len(file_stop),
        "extra": len(ex),
    }
