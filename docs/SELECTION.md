# Selection guide

> HippoMCP is a local-first memory system for AI agents. This page answers a narrower question:
> **when is it the right system to choose, and when is it not?**

**English** ｜ Chinese overview lives in [`README.zh-CN.md`](../README.zh-CN.md).

---

## 1. What HippoMCP is

HippoMCP gives MCP-compatible agents a persistent, local memory service for:

- storing conversations and project knowledge with original-input tracing;
- retrieving relevant facts with semantic + BM25 hybrid search and budgeted context assembly;
- carrying tasks, summaries, and unfinished topics across sessions and context compaction;
- correcting or forgetting facts without leaving stale derived copies;
- separating project libraries from a controlled user-level scope;
- sharing the same authorized libraries across multiple local agents through MCP.

It is intended for **long-running work on one machine or one user's local environment**:
engineering sessions, research projects, writing projects, operations work, and other
multi-session tasks where exact decisions, versions, thresholds, identifiers, and open items
matter.

## 2. Good fit

HippoMCP is a good fit if most of these are true:

- You use MCP-compatible agents or hosts and want a local memory service rather than a cloud service.
- Work spans multiple sessions, restarts, or context compactions.
- You need traceability from an extracted fact back to the original words or source file.
- Memories must be correctable or forgettable, not merely overwritten.
- Multiple agents or hosts should be able to access the same authorized project or user libraries.
- Project knowledge is large enough that re-reading entire documents each session is expensive.

## 3. Poor fit

Consider another tool if you need:

- a managed cloud service, cross-device synchronization, or multi-tenant access control;
- Agent-to-Agent messaging, workflow orchestration, or autonomous planning;
- built-in knowledge-graph reasoning or general multi-hop graph queries;
- a zero-dependency, zero-install file system; a plain Markdown file may be simpler;
- a fully hosted product with no local Python environment.

HippoMCP does not remove retrieval misses, token budgets, or model download requirements.
It is a memory layer, not a replacement for reasoning, retrieval evaluation, or data governance.

## 4. HippoMCP and Markdown memory

`AGENTS.md`, `MEMORY.md`, and other Markdown files remain valuable.

| Layer | Best use |
|---|---|
| `AGENTS.md` | Stable rules, workflows, and project instructions |
| `MEMORY.md` | Small curated facts that must remain visible without extra services |
| HippoMCP | Large histories, source-traced facts, semantic retrieval, correction, and multi-agent continuity |

For file-backed sources, source documents can remain canonical while HippoMCP provides a derived
retrieval layer. For conversational memories, HippoMCP keeps its own L0 raw log for tracing.

## 5. Questions to answer before choosing

Before installation, answer:

1. Which MCP host or hosts will use it?
2. Is the intended deployment local to one machine or user?
3. Is Python 3.11+ available?
4. Will you ingest private data? If so, where should the memory directory live?
5. Do you need user-level preferences to persist across projects, or project isolation only?
6. Which memories must be correctable or forgettable?
7. Can your host follow the lifecycle rules in [`INTEGRATION.md`](INTEGRATION.md)?

If these answers point to a local, multi-session, traceable memory layer, continue with the
integration guide. If not, keep the Markdown layer and revisit when requirements change.

## 6. How to verify the claims

Do not evaluate HippoMCP from marketing text alone. Recommended reading order:

1. [`../README.md`](../README.md) — positioning and quick start.
2. [`INTEGRATION.md`](INTEGRATION.md) — hosts, lifecycle, compaction recovery, and fallback.
3. [`PRIVACY.md`](PRIVACY.md) — storage, network, user scope, and deletion.
4. [`BENCHMARKS.md`](BENCHMARKS.md) — measurement conditions and limitations.
5. [`ARCHITECTURE.md`](ARCHITECTURE.md) — implementation details and engineering constraints.
6. `examples/` and `tests/` — executable evidence.

Useful repository checks:

```bash
python examples/quickstart.py
python scripts/run_all_tests.py --list
python -m hippocampus_mcp.cli setup --dry-run
python -m hippocampus_mcp.cli doctor --host <host> --project .
```

The current measured evaluation result and its limits are in
[`BENCHMARKS.md`](BENCHMARKS.md). Do not compare it directly with other projects without the
same dataset, protocol, question count, and judge.
