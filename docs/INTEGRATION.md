# Integration guide

> How to install HippoMCP into an MCP host, wire the session lifecycle, and operate it safely.
> The goal is not only “the server starts”, but “the agent uses memory correctly.”

**English** ｜ Chinese overview lives in [`README.zh-CN.md`](../README.zh-CN.md).

---

## 1. Requirements

- Python **3.11+**
- Git, if installing from source
- An MCP-compatible host
- Network access for the first embedding-model download
- Enough disk space for the selected embedding model

The core service does not require an LLM API key. A judge key is only needed for evaluation.

## 2. Install from source

```bash
git clone https://github.com/cstksniper/hippocampus-mcp.git
cd hippocampus-mcp
python -m pip install -e .
python examples/quickstart.py
```

`quickstart.py` is a pipeline smoke test using a hash embedder. It does not download an
embedding model and does not demonstrate semantic retrieval quality.

For real semantic retrieval:

```bash
python examples/quickstart.py --model sentence-transformers/all-MiniLM-L6-v2
```

## 3. Install into an MCP host

Until a PyPI release is available, install directly from GitHub:

```bash
python -m pip install "git+https://github.com/cstksniper/hippocampus-mcp.git"
hippocampus-mcp models --fetch
hippocampus-mcp setup
```

If host auto-detection is wrong, pass it explicitly:

```bash
hippocampus-mcp setup --host workbuddy
hippocampus-mcp setup --host zcode
hippocampus-mcp setup --host trae
hippocampus-mcp setup --host claude
```

`claude` refers to **Claude Desktop**. Claude Code, Cursor, and other MCP hosts require separate
manual configuration rather than this one-command setup path.

`setup`:

- backs up before writing;
- refuses to overwrite corrupt JSON;
- changes only its own MCP entry;
- performs a real MCP handshake after writing.

Then **restart the host** and start a new session.

## 4. Host configuration targets

| Host | Configuration file | One-command setup |
|---|---|---|
| WorkBuddy | `<workspace>/.mcp.json`, else `~/.workbuddy/mcp.json` | `--host workbuddy` |
| ZCode | `<workspace>/.zcode/config.json` — not `.mcp.json` | `--host zcode` |
| TRAE | `%APPDATA%\TRAE SOLO CN\User\mcp.json` | `--host trae` |
| Claude Desktop | `%APPDATA%\Claude\claude_desktop_config.json` | `--host claude` |

The production entry point is:

```text
python -m hippocampus_mcp.launcher
```

The launcher lets multiple hosts and sessions share one daemon and one embedding model. If the
launcher fails, it falls back to direct connection. `HIPPOCAMPUS_LAUNCHER=0` disables the
launcher path explicitly.

## 5. Verify the installation

After restart:

```bash
python -m hippocampus_mcp.cli doctor --host <host> --project .
hippocampus-mcp models --prune
```

In the new session:

```text
library_whoami()
```

If no `mcp__hippocampus__*` tools appear:

1. Run the module-form doctor, not an old executable entry point.
2. Fix the reported configuration or interpreter issue.
3. Restart the host or session again. Old sessions keep a frozen tool table.

## 6. Give the host operating rules

HippoMCP provides memory tools; the host agent decides when to call them. Add rules like these
to `AGENTS.md` or host instructions:

```text
At the start of a task, call session_start with the current task and working files.
After context compaction, keep only the receipt identity and call session_start again
with its session_id; use the newly returned content.
Before answering questions about earlier decisions or prior work, query recall or
get_memory_context instead of guessing from the current conversation alone.
Use get_memory_context when you need budgeted, assembled context.
Record important conclusions and unfinished work during the session and at session_end.
Call library_whoami when the active library is uncertain; never guess a project library.
Keep AGENTS.md and MEMORY.md as the lightweight fallback. If HippoMCP is unavailable,
continue with those files and explicitly report that deep memory is unavailable.
```

Recommended lifecycle:

```text
session_start(task)            → task-relevant memories, recent state, open topics, and more
ingest(conversation)           → L0 original input + L1 extracted facts + entity links
get_memory_context(query)      → five budgeted context slots
session_end(summary, topics)   → session summary + profile consolidation + open topics
forget_memory(id, reason)      → cascade deletion across derived copies
```

## 7. Multi-library and user scope

Use libraries to separate projects:

```text
library_list()
library_use("project-name")
library_whoami()
```

Priority is:

```text
explicit library= parameter > session-sticky library > default library
```

Unresolvable or ambiguous library names should fail loudly rather than silently rerouting.
The first key of write responses is `library`, so a wrong destination is visible.

A separate `user` library can hold long-lived user preferences. Project sessions can read
permission-filtered user-scope context, but non-home writes to the user library enter a review
queue by default. See [`PRIVACY.md`](PRIVACY.md).

## 8. Context compaction recovery

`session_start` returns a receipt containing identity information such as `session_id`,
`assembled_at`, `blocks`, `memory_active`, and `knowledge_epoch`.

After host-side context compaction:

1. Do not copy the full bootstrap content into the compacted context.
2. Keep only the receipt identity.
3. Call `session_start` again using that `session_id`.
4. Treat the newly returned content as authoritative.

If `memory_active` or `knowledge_epoch` does not match the offline expectation, the library has
been changed by another session; use the new result.

## 9. Failure fallback

HippoMCP is a local service and can be unavailable, slow, or misconfigured.

Recommended host policy:

```text
If HippoMCP tools are unavailable or time out, continue with AGENTS.md and MEMORY.md.
Do not block the main task on the memory service.
Explicitly tell the user that deep memory is unavailable.
Retry memory operations only when the user asks or when the service recovers.
```

This is a host-side integration policy, not a guarantee that the memory service itself can
never fail or block.

## 10. Diagnostics and maintenance

Common commands:

```bash
hippocampus-mcp doctor
python -m hippocampus_mcp.cli doctor --host zcode --project .
hippocampus-mcp models
hippocampus-mcp models --fetch
hippocampus-mcp models --prune
```

`models --prune` previews reclaimable model files. Add `--yes` only after reviewing the list.
For Chinese corpora, prefer `BAAI/bge-base-zh-v1.5`; for English corpora, the MiniLM baseline is
usually sufficient. Do not set `HF_HUB_OFFLINE=1` before the model has been downloaded.

## 11. Uninstall or move a library

Before removing HippoMCP:

1. Export memories if needed with `export_memories`.
2. Note the active `HIPPOCAMPUS_PERSIST_DIR` and model cache locations.
3. Remove the MCP entry with the host's normal editor or configuration flow.
4. Keep `AGENTS.md` and `MEMORY.md` as the surviving fallback layer.

The memory directory is ordinary local data. Do not delete it unless you intend to lose those
memories.
