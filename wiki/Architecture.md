# Architecture

CLS++ implements a unified memory system grounded in two complementary theories that describe the **same biological process** from two different perspectives:

1. **CLS Theory** (Brain Architecture) — What structures exist: L0–L3 hierarchy mirroring cortical regions
2. **Thermodynamics** (Free Energy Dynamics) — Why and when things move: phase transitions driven by F(θ,Σ,ρ,τ)

These are not competing theories. The brain architecture describes the **storage tiers**. The thermodynamics describes the **promotion dynamics** between them.

---

## Single Code Path

All memory operations flow through one unified path:

```
Write: text → PhaseMemoryEngine.store() → L1 (persistence, fire-and-forget)
       ↳ 384-dim embedding attached to PhaseMemoryItem (EmbeddingService)
       ↳ Auto recall_long_tail() every 50 writes per namespace

Read:  query → PhaseMemoryEngine.search() → 6-layer TRR (in-memory, sub-ms)
       ↳ 384-dim semantic re-ranking applied on top of TRR results

Sleep: recall_long_tail() every 5 min (background) + schema export to L2 (REM)
```

**PhaseMemoryEngine** is the brain — pure Python, zero external dependencies, thermodynamic memory with gas→liquid→solid→glass phase transitions.

**L1/L2 PostgreSQL** is persistence — write-through for durability. If PostgreSQL is unavailable, the brain keeps working; data loads back on next startup.

---

## Phase Transitions → Store Mapping

| Phase | Consolidation Strength | Brain Analog | L-Store |
|-------|----------------------|--------------|---------|
| **Gas** | s < 0.05 | Prefrontal transient | L0 (volatile) |
| **Liquid** | s ≥ 0.05, no schema | Hippocampus (episodic) | L1 (PostgreSQL + pgvector) |
| **Solid** | schema_meta ≠ None | Neocortex (schema) | L2 (PostgreSQL graph) |
| **Glass** | solid + entropy converged | Long-term archive | L3 (deep recess) |

Phase transitions are thermodynamically derived — no manual thresholds:

```
F(θ, Σ, ρ, τ) = E_pred − Σ·S_model + λ·L_landauer

gas → liquid:   s rises above STRENGTH_FLOOR (0.05) as retrieval reinforces
liquid → solid: ΔF < 0 (crystallization — Landauer cost of abstraction < benefit)
solid → glass:  std(H_history[-3:]) / mean < 1% (entropy convergence = rigid memory)
```

---

## Free Energy Formula

```
s(t) = exp(−Δt/τ) · (1 + β·ln(1 + R)) − D   [consolidation strength]

F = E_pred − Σ·S_model + λ·L_landauer         [free energy per item]
  = (1 − s) − Σ·H·ρ + λ·kT·ln(2)·H/τ

ΔF_crystal = F_schema − Σ F_liquid + C_abs     [crystallization criterion]
When ΔF < 0: crystallize (thermodynamically favorable)
```

---

## Retrieval — TRR (Thermodynamic Resonance Retrieval)

All search goes through **PhaseMemoryEngine** with 6 in-memory layers (sub-ms):

1. **Schema-Aware Query Expansion** — add fixed-point tokens Φ* from matched schemas
2. **Morphological Kernel** — prefix expansion (4-char) bridges "medic" → "medication"
3. **BMX Score** — entropy-weighted BM25: IDF × H_weight per matched token
4. **PPMI-SVD Semantic Bonus** — cosine(query_vec, item_vec) × avg_idf (50-dim, in-memory)
5. **Thermodynamic Component** — −F(item)/kT (stable items ranked higher)
6. **Phase Susceptibility** — χ(phase) multiplier: gas=0.7, liquid=1.0, solid/glass=1.0+boost

**Post-TRR: 384-dim Semantic Re-ranking** (via EmbeddingService, in MemoryService layer):
- Query embedded with SentenceTransformer `all-MiniLM-L6-v2` (384-dim)
- Items re-scored: `final = 0.6 × ttr_score_norm + 0.4 × cosine_384(query, item)`
- Bridges vocabulary gaps ("relocated" ↔ "moved") that the morphological kernel misses

**CER (Cross-Entity Resonance)** fires for multi-entity queries via PESQD decomposition (Kuramoto coupled oscillators).

---

## Sleep Cycle — REM Only + Auto-Trigger

The N1/N2/N3 phases (rank/decay/deduplicate) are superseded by PhaseMemoryEngine's continuous thermodynamic recomputation at write time.

**Hippocampal Replay — Auto-triggered (no manual call needed):**
1. Every 50 writes per namespace — inline in `MemoryService.write()`
2. Every 5 minutes — asyncio background loop in `api.py:startup()`

**REM** (also on-demand via `POST /v1/memory/sleep`):
1. `recall_long_tail()` — rehearse low-retrieval items: s(t) kept above STRENGTH_FLOOR via β·ln(1+R)
2. Schema export — write crystallized (solid/glass) items to L2 PostgreSQL

---

## Reconsolidation Gate

Belief revision via `POST /v1/memory/adjudicate` requires **evidence quorum**. A single conflicting input does not overwrite an established memory.

PhaseMemoryEngine handles internal contradiction detection automatically via **surprise scoring** + **Landauer damage** (applied at write time, no explicit call needed).

---

## Vector Architecture

| System | Dims | Location | Persisted | Role |
|--------|------|----------|-----------|------|
| PPMI-SVD (`_token_vectors`) | 50 | In-memory dict | No (rebuilds from corpus) | TRR Layer 4 semantic bonus |
| SentenceTransformer embeddings | 384 | On `PhaseMemoryItem.embedding_dense` | Via L1 pgvector | Post-TRR semantic re-ranking |

The 50-dim PPMI-SVD captures **corpus-specific co-occurrence** (fast, no deps). The 384-dim ST captures **pre-trained language semantics** (bridges vocabulary gaps). Both are active in production.

---

## Tech Stack

| Component | Role |
|-----------|------|
| **PhaseMemoryEngine** | Thermodynamic brain — all store/search logic, zero external deps |
| **EmbeddingService** | 384-dim sentence-transformer embeddings for re-ranking + L1/L2 persistence |
| **PostgreSQL + pgvector** | L1 (episodic persistence), L2 (schema graph) |
| **sentence-transformers** | `all-MiniLM-L6-v2` — 384-dim dense embeddings |
| **Redis** | L0 working buffer (reserved — PhaseMemoryEngine handles in-memory) |
| **MinIO** | L3 deep archive (glass phase long-term storage) |

---

## End-to-End User Flow

```
User message
  → MemoryService.read()          # TRR + 384-dim re-rank → top-K context
  → LLM (Claude/OpenAI/Gemini)   # augmented with memory context
  → LLM response
  → MemoryService.write()         # store user message + LLM synthesis
    ↳ engine.store()              # thermodynamic ingestion
    ↳ embedding_service.embed()   # 384-dim attached to PhaseMemoryItem
    ↳ auto recall_long_tail()     # every 50 writes (hippocampal replay)
    ↳ L1.write() async            # fire-and-forget persistence
  → Display to user
```

---

## LoCoMo Benchmark — v0.6.0 Results

| Run | Architecture | Overall | Multi-hop | Temporal | Open-domain | Single-hop | Adversarial |
|-----|-------------|---------|-----------|---------|-------------|-----------|------------|
| Run 7 — v0.51 | TRR + LLM enhanced | 30.6 | 31.2 | 9.0 | 6.0 | 12.5 | 99.8 |
| Run 8 — v0.60 | + 384-dim re-rank + timeline + improved prompts | TBD | TBD | TBD | TBD | TBD | TBD |
| **Mem0 (target)** | Graph + LLM | **41.0** | **38.7** | **28.6** | **47.7** | — | — |

---

[← Home](Home) | [API Reference →](API-Reference)
