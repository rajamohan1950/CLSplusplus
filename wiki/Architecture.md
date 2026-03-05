# Architecture

CLS++ implements a four-store memory hierarchy inspired by neuroscientific Complementary Learning Systems (CLS) theory.

---

## Four-Store Hierarchy

| Store | Brain Analog | Role |
|-------|--------------|------|
| **L0** | Prefrontal cortex (working memory) | Hot buffer for recent, high-salience items |
| **L1** | Hippocampus (episodic) | Indexed episodic memories, semantic search |
| **L2** | Neocortex (schema graph) | Consolidated knowledge, relationships |
| **L3** | Thalamus (deep recess) | Long-term, rarely accessed |

```
Write → L0 (buffer) → L1 (index) → Sleep cycle → L2/L3 (consolidation)
Read  → Query L0–L3, merge by confidence
```

---

## Plasticity Engine

Memories are not static. The plasticity engine applies:

- **Salience** — How important is this memory?
- **Usage** — How often is it retrieved?
- **Authority** — How trusted is the source?
- **Conflict** — Does it contradict existing beliefs?
- **Surprise** — How unexpected is it?

These signals drive **strengthening** (consolidation) and **decay** (forgetting).

---

## Sleep Cycle

Nightly maintenance (or on-demand via `POST /v1/memory/sleep`):

1. **N1** — Rank items by plasticity signals
2. **N2** — Strengthen high-value, decay low-value
3. **N3** — Deduplicate similar memories
4. **REM** — Promote L1 → L2 → L3 (consolidation)

---

## Reconsolidation Gate

Belief revision requires **evidence quorum**. A single conflicting input does not overwrite an established memory. Multiple corroborating inputs are required.

---

## Data Flow

```
Client (LLM) → POST /v1/memory/read (before inference)
                    ↓
              CLS++ Core
              - Embed query
              - Search L0, L1, L2, L3
              - Merge by confidence
              - Return top-k
                    ↓
Client (LLM) → POST /v1/memory/write (after inference)
                    ↓
              - Store in L0
              - Index in L1
              - Sleep cycle promotes to L2/L3
```

---

## Tech Stack

- **Redis** — L0 working buffer, rate limiting
- **PostgreSQL + pgvector** — L1, L2, L3 stores, embeddings
- **MinIO** (optional) — Large object storage
- **sentence-transformers** — Embeddings (all-MiniLM-L6-v2)

---

[← Home](Home) | [API Reference →](API-Reference)
