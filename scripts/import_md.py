# -*- coding: utf-8 -*-
"""Bulk-import Markdown documents into a HippoMCP memory library.

Takes existing .md knowledge documents (conventions, progress logs, gotcha
notes, etc.), chunks them, and ingests them one by one — a one-shot import
tool for migrating from "reread the full text" to "memory retrieval".

Usage:
    python -u scripts/import_md.py --dir <md dir> --project <project name>
    python -u scripts/import_md.py --file <single .md> --project <project name>
    Add -u (unbuffered): otherwise per-file progress is invisible when piped
    or run in the background.

Common flags:
    --dry-run       count only, write nothing
    --model         embedding model (default: HIPPOCAMPUS_EMB_MODEL)
    --importance    uniform importance, default 0.5 (not pinned)
    --supersede     enable fact-update detection (**off by default**, see below)
    --granularity   sentence (default, split per sentence) / chunk (one per block)
    --save-every    persist every N records, default 300 (interruptible)
    --force         ignore the "already imported" skip logic, force re-import
    --lexicon       entity lexicon file (one entry per line)

Design notes:
    1. **Import model must match the serving model**: the script reads
       HIPPOCAMPUS_EMB_MODEL and passes it explicitly to HippoBrain, prints
       the effective model before importing, and warns on non-BGE models.
       Earlier revisions did not pass a model -> records landed in default
       MiniLM while the server ran BGE -> dimension mismatch, retrieval
       guaranteed to fail.
    2. **supersede off by default**: `_detect_supersede` was designed for fact
       updates in *conversation*, but an md document already IS the deduped
       current truth; historical narrative ("X was changed to Y") is normal in
       documents, and bulk import would mass-kill real facts.
    3. **source tracing**: every memory carries the relative file path, echoed
       back in recall hits, so "which file did this come from" is auditable.
    4. **uniform importance**: documents have no conversational context; the
       length/exclamation/emphasis heuristics tuned for dialogue would pin
       thousands of sentences at block level. A single constant is right.
    5. **visible progress + resumability + idempotence**: per-record flushed
       heartbeat lines, incremental persistence every N records, skip files
       already imported (matched by source).

Chunking strategy:
    - Split by markdown headings (##/###), paragraphs by blank lines within a
      section, overlong paragraphs further split by sentence
    - A code block is kept whole as one procedural memory
    - Fragments under MIN_CHUNK chars and pure table separators are skipped
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
import time
from typing import Any

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

MAX_CHUNK = 500   # max chars per memory record
MIN_CHUNK = 10    # paragraphs shorter than this are skipped
HEARTBEAT = 200   # print a heartbeat every N records (long imports must feel alive)

# Markdown table **separator rows** (`|---|---|`, `|:--|--:|`) carry no
# information; ingesting them only dilutes retrieval quality. Data rows are
# still imported as usual (tables are often the most useful content).
_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")

# ------------------------------------------------------------------
# Structural markers -> structured tags
# ------------------------------------------------------------------
# **Extraction logic does NOT live here**: everything goes through the
# configurable extractor in ``hippocampus_mcp.domain.tagging`` (numbered-item
# families chapter/volume/roll/part + time families sexagenary/gregorian
# year/month + custom patterns).
#
# Why the move: hardcoding three regexes (chapter/volume/sexagenary) into an
# import script pins the solution to one corpus — a new corpus (contract
# clauses, medical visit numbers) would require a Python change, while the
# serving-side write path (ingest) has no extraction at all. Import, backfill
# and serving-side writes now share one implementation.
#
# Default keys: 章 (chapter) / 卷 (volume) / 年 (year) / 月 (month). To rename
# keys or add markers, write ``<library>/tag_schema.json`` (or point
# HIPPOCAMPUS_TAG_SCHEMA at a file) — **do not change code**. Config format is
# documented at the top of tagging.py.
from hippocampus_mcp.domain.tagging import (  # noqa: E402  (sys.path set above)
    TagExtractor,
    cn_num_to_int,
    extract_heading_tags,
    extract_text_tags,
    load_extractor,
    merge_tag_values,
    reset_default,
)
from hippocampus_mcp.brain import NAV_TAG_KEY  # noqa: E402

def _p(*args, **kwargs) -> None:
    """print with flush — makes progress visible in pipes/background runs."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def split_section(text: str) -> list[tuple[str, str]]:
    """Split one section into (type, content) tuples. Types: text/code."""
    out: list[tuple[str, str]] = []
    # Extract code blocks first
    parts = re.split(r"(```.*?\n.*?```)", text, flags=re.S)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith("```"):
            lang = p.split("\n")[0].strip("`").strip() or "code"
            out.append(("procedural", f"[{lang}]\n{p}"))
            continue
        # Split into paragraphs; overlong ones are split further by sentence
        for para in re.split(r"\n\s*\n", p):
            para = para.strip()
            # Drop table separator lines within a paragraph (they usually sit
            # in the same paragraph as the table's data rows)
            if "\n" in para:
                para = "\n".join(
                    ln for ln in para.split("\n")
                    if not _TABLE_SEPARATOR.match(ln.strip())
                ).strip()
            if len(para) < MIN_CHUNK:
                continue
            if _TABLE_SEPARATOR.match(para):
                continue
            if len(para) <= MAX_CHUNK:
                out.append(("world", para))
            else:
                buf = ""
                # Sentence boundary lookahead includes CJK punctuation on
                # purpose: Chinese text has no spaces to split on
                for sent in re.split(r"(?<=[。！？!?\n])", para):
                    if len(buf) + len(sent) > MAX_CHUNK and buf:
                        out.append(("world", buf.strip()))
                        buf = sent
                    else:
                        buf += sent
                if buf.strip() and len(buf.strip()) >= MIN_CHUNK:
                    out.append(("world", buf.strip()))
    return out


def iter_chunks(md_path: str, source_label: str | None = None) -> list[tuple[str, str, str]]:
    """Return [(cognitive_type, content, source_label)].

    Kept for the old signature (``preflight_entities.py`` still uses it).
    For tags use :func:`iter_chunks_tagged`.
    """
    return [
        (ctype, content, source)
        for ctype, content, source, _tags in iter_chunks_tagged(md_path, source_label)
    ]


def iter_chunks_tagged(
    md_path: str,
    source_label: str | None = None,
    tags_from_heading: bool = False,
    tags_from_content: bool = False,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return [(cognitive_type, content, source_label, tags)].

    Two tag-extraction routes, usable separately or together (union):

    * ``tags_from_heading=True``: extract from markdown **headings**
      (``### Chapter 43`` -> ``{"章": 43}``); tags attach to every chunk under
      that heading. Requires well-formed headings.
    * ``tags_from_content=True``: extract from **chunk bodies**. For documents
      with irregular headings this is the route that actually has coverage (in
      practice only a small fraction of chunk bodies carry structural markers,
      while heading-tagged chunks can be single digits). Multi-values are
      stored as lists; the filter side ORs them.

    Both routes only recognize **explicit structural markers** — no guessing.
    Which markers are recognized and what the keys are called is decided by the
    tag schema (``--tag-schema`` / ``<library>/tag_schema.json`` / built-in
    generic rules).
    """
    with open(md_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    source = source_label or os.path.basename(md_path)
    chunks: list[tuple[str, str, str, dict[str, Any]]] = []
    # Split into sections at ##/### headings; heading becomes a context prefix
    sections = re.split(r"\n(?=#{1,3} )", raw)
    for sec in sections:
        lines = sec.split("\n", 1)
        heading = lines[0].lstrip("#").strip() if lines[0].startswith("#") else ""
        body = lines[1] if len(lines) > 1 else ""
        prefix = f"[{heading}] " if heading else ""
        head_tags = extract_heading_tags(heading) if tags_from_heading else {}
        for ctype, content in split_section(body if body else sec):
            text = f"{prefix}{content}"
            tags = dict(head_tags)
            if tags_from_content:
                # Union of heading tags and body tags: the heading says "which
                # chapter this section belongs to", the body says "which other
                # chapters this record references" — neither should be lost.
                for k, v in extract_text_tags(text).items():
                    tags[k] = merge_tag_values(tags.get(k), v)
            chunks.append((ctype, text, source, tags))
    return chunks


def collect_files(args) -> list[str]:
    if args.file:
        return [args.file]
    skip = set(args.skip_dirs.split(","))
    files: list[str] = []
    for root, dirs, names in os.walk(args.dir):
        dirs[:] = [d for d in dirs if d not in skip]
        files.extend(os.path.join(root, n) for n in names if n.endswith(".md"))
    return sorted(files)


def source_label_for(path: str, args) -> str:
    """Prefer a relative path (readable, auditable); fall back to basename."""
    base = args.source_root or args.dir or os.path.dirname(path)
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        rel = os.path.basename(path)
    return rel.replace("\\", "/")


def _report_scale(chunks: list[tuple[str, str, str]]) -> None:
    """Estimate the real L1 yield and time cost.

    Default granularity is sentence — ``ingest`` splits each chunk **by
    sentence again**, so L1 count is usually a multiple of chunk count and the
    time cost scales the same way. Without this estimate, a long import only
    reveals its true runtime after it starts. The sentence-splitting heuristic
    reuses ``HippoBrain._split_facts`` directly to avoid drift between the two.
    """
    n_chunk = len(chunks)
    chars = sum(len(c) for _t, c, _s in chunks)
    try:
        from hippocampus_mcp.brain import HippoBrain
        split = HippoBrain._split_facts
    except Exception as e:  # missing optional deps (e.g. torch): report chunks only
        _p(f"(skipping sentence-granularity estimate: {e})")
        return
    n_sent = sum(len(split(c)) for _t, c, _s in chunks)
    _p("")
    _p("Granularity comparison (--granularity):")
    _p(f"  chunk    -> {n_chunk} L1 records (~{chars / max(n_chunk, 1):.0f} chars/record)"
       f"  ~= {n_chunk * 0.2 / 60:.0f} min")
    _p(f"  sentence -> {n_sent} L1 records (~{chars / max(n_sent, 1):.0f} chars/record)"
       f"  ~= {n_sent * 0.2 / 60:.0f} min   <- default")
    _p("  (time estimated from an empirical ~0.2 s/record embedding; excludes "
       "model load and disk writes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="directory of md files (walked recursively)")
    ap.add_argument("--file", help="a single md file")
    ap.add_argument("--project", help="project name (determines the library directory)")
    ap.add_argument("--persist-dir", default="", help="library directory directly (overrides --project)")
    ap.add_argument("--data-root", default=os.path.join(PROJECT_DIR, "data"),
                    help="root directory of memory libraries")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-dirs", default=".git,node_modules,__pycache__,.workbuddy",
                    help="directory names to skip while walking, comma-separated")
    ap.add_argument("--model", default=None,
                    help="embedding model. When empty, priority: "
                         "HIPPOCAMPUS_EMB_MODEL > follow the model already stored in "
                         "the library > all-MiniLM-L6-v2 (empty libraries only, with a "
                         "warning). Forgetting the env no longer silently imports "
                         "under the English model.")
    ap.add_argument("--importance", type=float, default=0.5,
                    help="uniform importance, default 0.5 (below the 0.7 pin line, "
                         "never auto-pinned)")
    ap.add_argument("--supersede", action="store_true",
                    help="enable fact-update detection (off by default: md is already "
                         "current truth, enabling it mass-kills real facts)")
    ap.add_argument("--granularity", choices=["sentence", "chunk"], default="sentence",
                    help="sentence = split per sentence (default); chunk = one record per block")
    ap.add_argument("--tags-from-heading", action="store_true",
                    help="extract structured tags from markdown headings (default keys "
                         "章/卷/年/月), e.g. '### 第43章 知情线' -> {\"章\": 43}, enabling "
                         "recall filter_tags filtering")
    ap.add_argument("--tags-from-content", action="store_true",
                    help="extract structured tags from **chunk bodies** (one record may "
                         "reference several chapters; values become lists). This is the "
                         "route with real coverage on documents with irregular headings; "
                         "combined with --tags-from-heading it takes the union")
    ap.add_argument("--tag-schema", default=None,
                    help="tag extraction config (JSON); default: env HIPPOCAMPUS_TAG_SCHEMA, "
                         "then <library>/tag_schema.json, then the built-in generic rules "
                         "(numbered families 章/回/卷/部 + time families sexagenary/gregorian "
                         "year/month). Add your project's own markers in the config — "
                         "**no code changes needed**; keys must match the host's "
                         "filter_tags or filtering silently no-ops")
    ap.add_argument("--save-every", type=int, default=300,
                    help="persist every N records (0 = only at the end), default 300, "
                         "interruptible")
    ap.add_argument("--force", action="store_true",
                    help="ignore the already-imported check, force re-import")
    ap.add_argument("--lexicon", default=None,
                    help="entity lexicon file (one entry per line); default "
                         "<library>/entity_lexicon.txt")
    ap.add_argument("--source-root", default="",
                    help="base directory for relative source paths, defaults to --dir")
    ap.add_argument("--nav-files", default="",
                    help="globs for navigation files (comma-separated), e.g. "
                         "'INDEX*,*索引*,README*'. Their content (TOC/headers/navigation "
                         "lines) is vaguely similar to every query, and in practice "
                         "they often grab the top hit in live probes. Matches get the "
                         "--nav-tag tag and are discounted by --nav-penalty at recall "
                         "time")
    ap.add_argument("--nav-tag", default=NAV_TAG_KEY,
                    help=f"tag key applied to navigation files (default {NAV_TAG_KEY})")
    ap.add_argument("--nav-penalty", type=float, default=1.0,
                    help="recall discount for navigation-tagged memories, 1.0 = no "
                         "discount (default). **The tag is only recorded at import "
                         "time; the discount takes effect only if HIPPOCAMPUS_NAV_PENALTY "
                         "is set or nav_penalty is passed to the brain**")
    args = ap.parse_args()

    if not args.dir and not args.file:
        ap.error("exactly one of --dir or --file is required")
    if not args.persist_dir and not args.project:
        ap.error("exactly one of --project or --persist-dir is required")

    files = collect_files(args)
    if not files:
        _p("No md files found")
        return

    if args.persist_dir:
        persist_dir = os.path.abspath(args.persist_dir)
    else:
        persist_dir = os.path.join(args.data_root, f"hippo_{args.project}")
    labels = {fp: source_label_for(fp, args) for fp in files}

    # Tag extraction rules are configurable (explicit > env > <library>/tag_schema.json
    # > built-in generic). Must be rebuilt before any extraction happens —
    # default_extractor is lazily cached.
    _ext = reset_default(args.tag_schema, persist_dir)
    if args.tags_from_heading or args.tags_from_content:
        _p(f"Tag extraction rules: {_ext.source}"
           f" (keys={sorted(set(_ext.keys.values()))}; "
           f"source: {'heading' if args.tags_from_heading else ''}"
           f"{'+body' if args.tags_from_content else ''})")

    _p(f"Target library: {persist_dir}")
    _p(f"Files to import: {len(files)}")

    if args.dry_run:
        total = 0
        tagged = 0
        chunks: list[tuple[str, str, str]] = []
        for fp in files:
            cs = iter_chunks_tagged(fp, labels[fp], args.tags_from_heading,
                                    args.tags_from_content)
            total += len(cs)
            tagged += sum(1 for _t, _c, _s, tg in cs if tg)
            chunks.extend((t, c, s) for t, c, s, _tg in cs)
            _p(f"  {labels[fp]}: {len(cs)} chunks")
        _p(f"Total {total} (chunk-granularity L1 count; dry-run, nothing written)")
        if args.tags_from_heading or args.tags_from_content:
            pct = tagged / total * 100 if total else 0.0
            _p(f"Taggable: {tagged}/{total} = {pct:.1f}%"
               f" (untagged records are invisible to tag filters — a physical limit "
               f"of the extraction route; the only way past it is making the write "
               f"side always carry tags)")
        _report_scale(chunks)
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from hippocampus_mcp.brain import HippoBrain
    from hippocampus_mcp.store import ConcurrentWriteError

    model = args.model or os.environ.get("HIPPOCAMPUS_EMB_MODEL") or ""
    brain = HippoBrain(
        persist_dir=persist_dir,
        embedder_model=model,
        supersede_enabled=args.supersede,
        entity_lexicon_path=args.lexicon,
    )

    # The effective model must be printed in your face (the original bug was
    # precisely that nothing ever said which model was in use)
    effective = brain.manager.model_key
    _p(f"Effective embedding model: {effective} (dim {brain.manager.embedding_dim})")
    if "bge" not in effective.lower():
        _p("  ⚠️  Warning: this is not a BGE Chinese model. If the server is "
           "configured with HIPPOCAMPUS_EMB_MODEL=BAAI/bge-base-zh-v1.5, import and "
           "serving will use different models — retrieval will fail or silently "
           "return zero hits.")
        _p("  ⚠️  Fix: set the HIPPOCAMPUS_EMB_MODEL env var, or pass --model.")
    _p(f"supersede detection: {'on' if args.supersede else 'off (recommended)'}; "
       f"importance={args.importance}; granularity={args.granularity}; "
       f"heading tags={'on' if args.tags_from_heading else 'off'}")
    _p(f"Entity lexicon: {len(brain.entity_lexicon)} entries"
       f"{' (' + brain.entity_lexicon_path + ')' if brain.entity_lexicon_path else ''}")

    # Idempotence: sources already in the library
    existing_sources = brain.manager.sources()
    already = {s for s, n in existing_sources.items() if s}
    if already and not args.force:
        skipped = [fp for fp in files if labels[fp] in already]
        if skipped:
            _p(f"Skipping {len(skipped)} already-imported files (--force to re-import)")
            files = [fp for fp in files if labels[fp] not in already]
            if not files:
                _p("All files already imported, nothing to do.")
                return

    ok = fail = 0
    since_save = 0
    t0 = time.time()
    nav_globs = [g.strip() for g in (args.nav_files or "").split(",") if g.strip()]
    nav_tag = args.nav_tag.strip() or NAV_TAG_KEY
    if nav_globs:
        _p(f"Navigation files ({nav_globs}) -> tag {nav_tag}, "
           f"discounted {args.nav_penalty} at recall (--nav-penalty)")
    for fp in files:
        # Navigation content is discounted: INDEX / TOC / header-like content is
        # vaguely similar to every query and in practice often grabs the top hit
        # in live probes. Tag it rather than tweak scores — the retrieval-side
        # discount is explicit configuration (default 1.0 = off); the import side
        # only records provenance.
        is_nav = any(fnmatch.fnmatch(labels[fp], g) or fnmatch.fnmatch(
            os.path.basename(fp), g) for g in nav_globs)
        # Persist once before each file: the previous file's results survive
        # a failure in this one
        try:
            for ctype, content, source, tags in iter_chunks_tagged(
                fp, labels[fp], args.tags_from_heading, args.tags_from_content
            ):
                try:
                    chunk_tags = dict(tags or {})
                    if is_nav:
                        chunk_tags[nav_tag] = "1"
                    brain.ingest(
                        content, speaker="user", cognitive_type=ctype,
                        importance=args.importance,
                        source=source,
                        granularity=args.granularity,
                        tags=chunk_tags or None,
                    )
                    ok += 1
                except Exception as e:  # one record failing never stops the import
                    fail += 1
                    _p(f"  !! failed: {source}: {e}")
                if ok and ok % HEARTBEAT == 0:
                    _p(f"  … {ok} records processed ({time.time()-t0:.0f}s, "
                       f"L1={brain.manager.store.count()})")
                if args.save_every and ok and ok % args.save_every == 0:
                    brain.save()
                    since_save = 0
                    _p(f"  … persisted (cumulative {ok} records)")
        except ConcurrentWriteError as e:
            _p("\n!! Save blocked (another process is writing the same library)")
            _p(f"   {e}")
            _p("   Close ZCode / other MCP sessions first, then rerun; already-"
               "imported records are not lost.")
            raise SystemExit(2)
        _p(f"  Processed {labels[fp]} (cumulative {ok} ok / {fail} fail / "
           f"{time.time()-t0:.0f}s)")

    brain.save()
    stats = brain.stats()
    _p(f"\nDone: {ok} records written / {fail} failed / {time.time()-t0:.0f}s")
    _p(f"Stats: L1={stats['L1_memories']} entities={stats['entities']} "
       f"pinned={stats['pinned']} superseded={stats['L1_by_state'].get('superseded', 0)}")
    _p(f"Embedding model: {stats['embed_model']}; vector coverage "
       f"{stats['vectors_by_model']}; missing {stats['vectors_missing']}")
    _p(f"Source files: {stats['sources']}")
    if stats["vectors_missing"]:
        _p("  ⚠️  Vectors are still missing — recall will skip those memories. "
           "Check that the models match.")


if __name__ == "__main__":
    main()
