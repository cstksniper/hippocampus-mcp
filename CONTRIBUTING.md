# Contributing

Thank you for spending time on this project. This document explains **how to get
changes in safely**, plus a few hard constraints — most of them come from real
incidents, not style preferences.

---

## Development environment

```bash
git clone <your fork>
cd hippocampus-mcp

# Base dependencies
pip install -e .

# Full development chain (including NER / evaluation)
pip install -e ".[dev,ner]"
```

Python **>= 3.11** is required (verified baseline; 3.10 is untested and unsupported).

**Quick smoke run** (downloads no models):

```bash
PYTHONPATH=src python examples/quickstart.py
```

It uses `embedder_backend="hash"` — no sentence-transformers, no API key.

---

## Read before coding: five hard constraints

These are not suggestions. PRs that violate them get bounced — not for style,
but because they **cause silent errors**.

### 1. The embedding model has exactly one source of truth

Model-name resolution happens only in `embeddings.resolve_embedder_model()`,
with priority **argument > env > default**. Do not copy env literals elsewhere.

Reason: multiple entry points used to read the env independently; one forgot to
set it and a Chinese library silently ran on the English model, with no error
anywhere. `brain_stats()["embedder_model_source"]` now self-reports the source —
keep it valid when touching related code.

### 2. The only legal entry point for switching embedding models is `scripts/reembed.py`

Changing the model via raw config makes stored vectors mismatch the new model.
`MemoryManager._sync_embedder_with_store()` handles it:
explicitly specified but mismatched → error; default but mismatched → **follow the model already in the library**.

### 3. Deletion must cascade

L2 / L3 / index layers are derived copies of L1 — **deleting L1 alone is a fake
deletion**. The only entry point is `brain.forget(id, reason)`, which runs a
6-step cascade internally.

### 4. Filtering must happen before retrieval

Structured tag filtering must use `search(allow_ids=)`, masking out-of-whitelist
entries with `-inf`. **The BM25 path must filter explicitly**, otherwise the
filter is bypassed.

### 5. Every MCP tool must pass `structured_output=False`

All of them, no exceptions — otherwise the host receives duplicated returns.

More engineering conventions: the "engineering constraints" section of
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Testing

```bash
# Run all public suites (~16 min, ~155 suites / 4700+ assertions)
python scripts/run_all_tests.py

# Only suites whose name contains "forget"
python scripts/run_all_tests.py -k forget

# Also run private suites under scripts/
python scripts/run_all_tests.py --include-scripts
```

`scripts/run_all_tests.py` **uses only the standard library**; pytest is not
required. If pytest is installed, `pytest tests/` works too (`pytest.ini` is
configured).

**Bar for new tests**: discoverable by `run_all_tests.py`, no private data, no
real model downloads, no absolute paths written into the repo. Only tests
meeting all four go into `tests/`; everything else goes to `scripts/`
(the private layer).

---

## Commits and PRs

- Commit messages in **Chinese or English**, but they must say **why**, not just what
- One PR does one thing; keep refactors and feature changes separate
- `src/` changes must attach test results (which suites were run)
- Evaluation changes must state whether the numbers moved

---

## ⚠️ Public / private boundary

This repository is exported from a research repository containing private data,
and **the boundary is hard**:

| Layer | Content | Rule |
|---|---|---|
| Public layer | whitelisted `src/`, `tests/`, `examples/`, `docs/` | free to change |
| Private layer | `data/`, `.env`, personal corpora | **never commit**; mounted via `HIPPOCAMPUS_PERSIST_DIR` |
| Process layer | `eval_v*/`, `*_snap_*/`, `_probe_out/` | already in `.gitignore`, never enters the repo |

**Self-check before committing**:

```bash
python tools/export_public.py --audit
```

It scans the whole repository and reports hits for secrets, private data, and
absolute paths. Files that hit sensitive content are **refused at export time**
instead of being silently copied.

---

## Reporting issues

Use the templates in `.github/ISSUE_TEMPLATE/`. When reporting a bug, attach the
output of `brain_stats()` — its self-report fields (NER backend, embedding model
source, L0 dangling count, tag coverage) usually pinpoint the problem directly.

For security issues, private-data exposure, and anything unsuitable for public
discussion, email [cstksniper@qq.com](mailto:cstksniper@qq.com) directly instead
of opening an issue first.
