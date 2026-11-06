# -*- coding: utf-8 -*-
"""HippoMCP minimal runnable example -- no model downloads, no API key.

Demonstrates the four core actions of the memory layer:

    1. ingest()              write: raw text goes to L0, extracted facts to L1, entities attached
    2. recall()              retrieve: hybrid search (vector + BM25 + temporal) with conflict flags
    3. get_memory_context()  consume: assemble 5 fixed slots, ready to paste into a system prompt
    4. forget()              delete: cascade-clean L2/L3/index layers, no orphans left

Run::

    PYTHONPATH=src python examples/quickstart.py

Note: this example deliberately uses ``embedder_backend="hash"`` so it
**downloads no models and starts in seconds** -- good for verifying the whole
pipeline end to end. Hash embeddings have no semantic generalization; retrieval
matches surface tokens only. For real quality, switch to a sentence-embedding
model::

    PYTHONPATH=src python examples/quickstart.py --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile

from hippocampus_mcp.brain import HippoBrain


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(description="HippoMCP minimal runnable example")
    ap.add_argument(
        "--model", default="",
        help="sentence-embedding model name; empty uses the hash backend (zero download)",
    )
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="hippocampus_quickstart_")
    backend = "sentence_transformer" if args.model else "hash"
    print(f"Temp memory library : {workdir}")
    print(f"Embedding backend   : {backend}" + (f" ({args.model})" if args.model else ""))

    # Sample corpus: three fictional conversations from one project team.
    # Fully generic content, no private data.
    corpus = [
        ("The design review settled on PostgreSQL, not MongoDB, because of transactions and reporting.", "meeting-notes.md"),
        ("Engineer Zhang's email is zhang@example.com; they own the backend.", "team-roster.md"),
        ("The deploy window moved to every Wednesday early morning; ops said Tuesday has routine maintenance.", "meeting-notes.md"),
        ("The deploy window used to be Monday; it was later changed to Wednesday.", "meeting-notes.md"),
    ]

    brain = HippoBrain(
        persist_dir=workdir,
        embedder_backend=backend,
        embedder_model=args.model,
        auto_save=False,
    )

    try:
        # ---------------- 1. Write ----------------
        banner("1. ingest -- write and extract")
        for text, source in corpus:
            r = brain.ingest(text, source=source)
            n = len(r.get("memory_ids") or r.get("memories") or [])
            print(f"  + [{source}] {text[:34]}…  -> {n} new L1 facts")

        st = brain.stats()
        print(f"\n  L0 raw: {st.get('L0_raw')}   L1 facts: {st.get('L1_memories')}   "
              f"entities: {st.get('entities')}   embed model: {st.get('embed_model')}")

        # ---------------- 2. Recall ----------------
        banner("2. recall -- hybrid retrieval")
        for q in ("when is the deploy window", "who owns the backend"):
            hits = brain.recall(q, top_k=3)
            print(f"\n  query: {q}")
            if not hits:
                print("    (no hits)")
            for h in hits:
                score = h.get("score", h.get("similarity", 0.0))
                flags = " ".join(
                    k for k in ("conflict_group", "newer", "superseded") if k in h
                )
                print(f"    {score:.3f}  {str(h.get('content', ''))[:44]}  {flags}")

        # ---------------- 3. Context assembly ----------------
        banner("3. get_memory_context -- 5-slot assembly")
        ctx = brain.get_memory_context("preparing to write deployment docs", top_k=5)
        blocks = ctx.get("blocks", {})
        print("  slot fill:")
        for key, val in blocks.items():
            print(f"    {key:24s} {len(val)} items")
            for item in list(val)[:2]:
                print(f"        - {str(item)[:58]}")
        print(f"\n  estimated tokens: {ctx.get('token_estimate')}")
        print("  ---- text ready to paste into a system prompt ----")
        for line in (ctx.get("text") or "(empty)").splitlines()[:12]:
            print(f"    {line}")

        # ---------------- 4. Cascade delete ----------------
        banner("4. forget -- cascade delete")
        victim = brain.recall("zhang@example.com", top_k=1)
        if victim:
            mid = victim[0].get("id") or victim[0].get("memory_id")
            rep = brain.forget(mid, reason="quickstart demo: cascade delete")
            print(f"  deleted {mid}")
            print(f"  cascade report: {rep}")
            left = brain.recall("zhang@example.com", top_k=1)
            print(f"  hits for the same query after deletion: {len(left)}")

        banner("Done")
        print("  All steps completed. For real use, point persist_dir at your memory library directory.")
        print("  See docs/ARCHITECTURE.md")
        return 0

    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
