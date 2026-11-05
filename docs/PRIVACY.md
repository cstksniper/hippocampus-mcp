# Privacy and data handling

> What HippoMCP stores, what leaves the machine, and what you should review before giving it
> private project or user data.

**English** ｜ Chinese overview lives in [`README.zh-CN.md`](../README.zh-CN.md).

---

## 1. Deployment model

HippoMCP is designed as a **local-first, single-machine** memory service.

- It speaks MCP over stdio from the host process.
- It is not designed to be exposed as a public network service.
- It is not a managed cloud, multi-tenant, or cross-device synchronization system.
- Multiple local agents can connect to the same service and authorized libraries.

If you need remote synchronization, encrypted shared storage, or organization-wide governance,
use an external storage policy around the local service or choose a different tool.

## 2. What is stored

With default settings, memory data lives under:

```text
~/.hippocampus_mcp
```

You can move it with:

```text
HIPPOCAMPUS_PERSIST_DIR
```

Depending on usage, the memory directory may contain:

- L0 raw conversation or document input;
- L1 extracted facts;
- L2 summaries and topic aggregates;
- L3 profile entries;
- entity and lexicon indexes;
- embedding vectors;
- session state, intents, and open topics;
- deletion tombstones and related maintenance records.

Treat the memory directory as sensitive: it may contain whatever was ingested, including
project decisions, user preferences, and private text.

## 3. Model and network behavior

The core memory service does not require an LLM API key and does not send memory content to an
inference provider as part of its ingestion path.

Network access is mainly for downloading the selected embedding model through
` sentence-transformers` / Hugging Face. You can prefetch it with:

```bash
hippocampus-mcp models --fetch
```

Evaluation can optionally use an LLM-as-judge through `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and
`JUDGE_MODEL`. The core memory service does not depend on those variables.

This repository does not add telemetry to the memory service. Third-party model libraries may
have their own download and caching behavior; review them if that matters for your environment.

## 4. User-level memory and review queue

HippoMCP separates project memory from user-level memory.

- Project libraries are workspace-bound.
- A `user` library can hold long-lived user preferences.
- Project sessions can receive permission-filtered user-scope context.
- Non-home writes to the user library enter a review queue by default.

Quarantined writes are marked `quarantined: true` and do not enter the normal recall pool or the
user-scope slot. They can be promoted only through the controlled home-library flow. Deletion
still uses `forget_memory` and its cascade.

This is designed to prevent another project from silently injecting new “user preferences” into
every future session.

## 5. Correction and deletion

Memory correction uses explicit state rather than silent overwriting:

- `active`
- `superseded`
- `archived`

Deletion is intended to clean derived copies, not only the main record:

```text
forget_memory(memory_id, reason)
```

`forget_by_query` is preview-first by default. Use it only after reviewing the candidate list.
Deletion tombstones are used to prevent old snapshots from resurrecting removed facts.

If you need a portable copy, use:

```text
export_memories
```

Exports contain memory content but not embedding vectors.

## 6. Backups, retention, and encryption

- The memory directory is local data. Protect it with normal filesystem permissions.
- There is no built-in encryption-at-rest in HippoMCP itself.
- Do not place the memory directory on a shared or publicly readable location.
- Use your own backup policy if the memories matter.
- If you store highly sensitive material, consider whether it should be ingested at all.

A safe rule: **do not put secrets, credentials, private keys, or regulated data into the memory
library unless you have a specific reason and a governance plan.**

## 7. What to check before private deployment

Before giving HippoMCP private data, review:

1. Where `HIPPOCAMPUS_PERSIST_DIR` points.
2. Which embedding model will be downloaded and from where.
3. Which hosts are allowed to connect to the same libraries.
4. Whether user-level quarantine fits your workflow.
5. Who can read the local memory directory.
6. How deletion, export, and backup will be handled.
7. Whether `AGENTS.md` and `MEMORY.md` should remain as the non-service fallback.

## 8. Operational recommendations

- Use one project library per workspace rather than one global library for everything.
- Keep the `user` library for durable user-level facts, not project-specific state.
- Call `library_whoami` when the destination is uncertain.
- Review quarantined user-library writes instead of blindly promoting them.
- Use `drift_audit` when source files have changed after ingestion.
- Prefer `export_memories` before major model, storage, or host migrations.

## 9. Boundary summary

| Question | Current answer |
|---|---|
| Cloud service? | No |
| Cross-device sync? | Not built in |
| Multi-tenant access control? | Not built in |
| Encrypted at rest? | Not built in |
| Local-first storage? | Yes |
| Requires LLM API key for core writes? | No |
| Designed for public network exposure? | No |
| Separate user/project scopes? | Yes |
| Reviewable cross-library user writes? | Yes, by default for non-home writes |
