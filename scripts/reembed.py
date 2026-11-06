# -*- coding: utf-8 -*-
"""Multi-model vector store manager: inspect / backfill / adopt / prune vectors.

Background: the memory body (text + structure) is independent of the embedding
model; vectors are a rebuildable derivative. Switching the embedding model does
**not** require re-importing documents — this script backfills vectors for the
target model.

Usage:
    # 1. Inspect vector inventory of one library (default action)
    python scripts/reembed.py --dir data/hippo_myproj

    # 2. Backfill after switching models (recomputes vectors only; no LLM,
    #    no re-chunking)
    python scripts/reembed.py --dir data/hippo_myproj \
        --model BAAI/bge-base-zh-v1.5

    # 3. Legacy single-vector libraries: confirm the model and adopt without
    #    a recompute pass
    python scripts/reembed.py --dir data/old_lib \
        --assume-model all-MiniLM-L6-v2

    # 4. Prune vectors of other models, keep only the current one (reclaim disk)
    python scripts/reembed.py --dir data/hippo_myproj --prune

    # 5. Process every library under data/ in bulk
    python scripts/reembed.py --all --model BAAI/bge-base-zh-v1.5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Same convention as other tools in scripts/: resolve the repo root relative to
# __file__ instead of hardcoding absolute paths (public-layer rule — must run
# from any clone location).
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DATA_DIR = os.path.join(PROJECT_DIR, "data")


def find_libraries(args) -> list[str]:
    """Locate the library directories to process."""
    # Script-side tools uniformly go through _lib_access: backend-agnostic checks
    _scripts = os.path.join(PROJECT_DIR, "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from _lib_access import has_library_data, require_library
    if args.dir:
        # Explicit directory: refuse "looks like a library but has no data",
        # and never treat an existing sqlite library as missing
        require_library(args.dir, action="reembed")
        return [args.dir]
    libs = []
    for name in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, name)
        # Backend-agnostic check: the old heuristic tested memories.json
        # existence, but **sqlite libraries have no memories.json, so they
        # would be silently skipped** (reembed reports "no libraries found"
        # while the user's library is right there on disk).
        # IMPORTANT: **backups must be excluded**. `--all` is a bulk **write**
        # operation; without exclusion, backup directories get backfilled too,
        # rewriting them and destroying the snapshot semantics (rule: backups
        # never participate in automatic selection). Backup names look like
        # `<name>.bak-<timestamp>` / `<name>.snap-<tag>` — `.bak` in the middle.
        if os.path.isdir(d) and has_library_data(d) and ".bak" not in name and ".snap" not in name:
            libs.append(d)
    return libs


def process(lib: str, model: str, args) -> None:
    from hippocampus_mcp.engine.manager import MemoryManager

    # This script's whole job is switching/backfilling a library's model
    # vectors, so it explicitly allows the "model differs from library"
    # guard (otherwise MemoryManager.load() refuses to open). To prevent
    # mistakes, the target model must be given explicitly via --model or
    # HIPPOCAMPUS_EMB_MODEL (see main).
    os.environ["HIPPOCAMPUS_ALLOW_MODEL_SWITCH"] = "1"

    print(f"\n{'=' * 62}\n  Library: {lib}\n{'=' * 62}")
    manager = MemoryManager(
        persist_dir=lib,
        embedder_backend="sentence_transformer",
        embedder_model=model,
    )
    manager.load(auto_backfill=False)
    store = manager.store
    active = manager.model_key
    print(f"  Active model: {active} (dim={manager.embedding_dim})")
    print(f"  Memory records: {store.count()}")

    stats = store.model_stats()
    if stats:
        print("  Existing vectors:")
        for key, info in stats.items():
            mark = "  <- active model" if key == active else ""
            print(f"    {key}: {info['vectors']} vectors (dim={info['dim']}){mark}")
    else:
        print("  Existing vectors: none")

    if args.assume_model:
        moved = store.adopt_legacy(args.assume_model)
        print(f"  Adopted legacy vectors -> {args.assume_model}: {moved}")
        if moved and args.assume_model == active:
            store.set_model(args.assume_model, store.model_dims.get(args.assume_model))

    missing = store.missing_ids(active)
    print(f"  Missing vectors for active model: {len(missing)}")

    if args.prune:
        keep = {active}
        if args.assume_model:
            keep.add(args.assume_model)
        dropped = store.prune_models(keep)
        print(f"  Pruned vectors: dropped {dropped} (kept {sorted(keep)})")

    if args.dry_run:
        print("  [dry-run] nothing written")
        return

    if missing:
        print(f"  Backfilling {len(missing)} vectors with {active} ...")
        result = manager.ensure_vectors(verbose=True)
        print(f"  Backfill done: {result['filled']} vectors")
    store.save()
    print(f"  Saved: {store.persist_path}")
    print(f"  Vector file: {store.vectors_path}")

    after = store.model_stats()
    print("  Vector inventory after processing:")
    for key, info in after.items():
        mark = "  <- active model" if key == active else ""
        print(f"    {key}: {info['vectors']} vectors (dim={info['dim']}){mark}")
    print(f"  Still missing for active model: {len(store.missing_ids(active))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-model vector store manager")
    ap.add_argument("--dir", help="library directory (contains memories.json)")
    ap.add_argument("--all", action="store_true", help="process every library under data/")
    ap.add_argument("--model", default=os.environ.get("HIPPOCAMPUS_EMB_MODEL") or "",
                    help="target embedding model. **No default** — this is an offline "
                         "model-switch tool; silently falling back to MiniLM would risk "
                         "accidentally recomputing a Chinese library with an English "
                         "model. Must be set explicitly via this flag or "
                         "HIPPOCAMPUS_EMB_MODEL.")
    ap.add_argument("--assume-model", dest="assume_model", default="",
                    help="adopt legacy-format vectors as the given model (dims must match)")
    ap.add_argument("--prune", action="store_true", help="drop vectors of non-active models")
    ap.add_argument("--dry-run", action="store_true", help="inspect only, write nothing")
    args = ap.parse_args()

    if not args.dir and not args.all:
        ap.error("specify --dir <library dir> or --all")
    if not args.model:
        ap.error("target embedding model is required: pass --model BAAI/bge-base-zh-v1.5 "
                 "or set HIPPOCAMPUS_EMB_MODEL (this tool has no default model)")

    # An earlier revision used `libs = [DATA_DIR] if args.all else find_libraries(args)`
    # — treating the **data/ directory itself** as a library. But data/ root has
    # neither memories.json nor hippocampus.db (it is only the container), so that
    # branch always produced an empty list, and the memories.json filter afterwards
    # meant **sqlite libraries were skipped entirely**. Delegate to find_libraries
    # (it handles both --dir and scan modes internally).
    libs = find_libraries(args)
    # Second check is also backend-agnostic: either hippocampus.db or
    # memories.json counts as library data
    sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
    from _lib_access import has_library_data

    libs = [d for d in libs if has_library_data(d)]
    if not libs:
        print("No libraries found (neither hippocampus.db nor memories.json) — "
              "pass --dir to point at one explicitly")
        return

    for lib in libs:
        process(lib, args.model, args)
    print(f"\nProcessed {len(libs)} libraries")


if __name__ == "__main__":
    main()
