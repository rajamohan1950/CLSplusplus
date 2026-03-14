# Cat 1 (Multi-Hop) — Thermodynamic Semantic Field (TSF) Design

## Problem Analysis

Cat 1 (multi-hop) requires connecting facts across multiple entities/sessions.

Example failure: "Which city have both Jean and John visited?" → needs to find `jean visited rome` AND `john visited rome`, then intersect.

**Current F1: 0.057 (282 questions)**

### Root Causes (general, not benchmark-specific)

1. **Word-level Jaccard retrieval** — query "Which city have both Jean and John visited?" has poor overlap with fact `jean visited rome` because Jaccard dilutes shared words across all words.
2. **Single-fact extraction** — "I went to Rome and loved the pasta" produces ONE fact. The pasta preference is lost.
3. **No semantic bridging** — "food" in a query cannot find facts with relation "eat". No character-level method can bridge words with zero character overlap.
4. **No multi-entity query decomposition** — a question about "both Jean and John" should search for Jean's facts AND John's facts separately.

## Design Philosophy: 50-Year-Ahead Thermodynamic Retrieval

### Core Principle: Intelligence at Write-Time, Physics at Read-Time

Every retrieval system in 2025 puts intelligence at READ time (embedding models, neural rerankers, cross-encoders). We INVERT this:

- **Write path (ingest):** LLM already called for fact extraction → also generate the **semantic field** (all query forms that would seek this fact). This is FREE — same LLM call, ~30 extra tokens.
- **Read path (search):** Pure hash lookups + arithmetic. Zero ML. Sub-microsecond.

The LLM IS the semantic engine. Using a second model (sentence-transformer) to re-encode what the first model already understood is architecturally redundant.

### Phase-Theoretic Foundation

In field theory, each memory item has a **field** — a region of influence in query space. The field has a **radius** (correlation length ξ) that depends on the thermodynamic state:

```
ξ(s) = floor(N_forms × s^(1/3))
```

- **Liquid (s→1):** Long correlation length, memory is broadly discoverable
- **Near critical (s ≈ floor):** Correlation length contracts rapidly
- **Gas (s < floor):** Zero correlation length, memory vanishes from index entirely

The exponent 1/3 is the mean-field critical exponent ν from 3D thermodynamics:
```
ξ ∝ |T - T_c|^(-ν)    where ν = 1/3 in mean-field theory
```

**No existing retrieval system has index entries that phase-transition in and out of existence based on memory thermodynamics.**

## Algorithm #1: Thermodynamic Semantic Field (TSF)

### The Ranking Equation

```
rank(q, i) = (n_matched_slots - 1) - F_i / kT
```

Where:
- `n_matched_slots` ∈ {1, 2, 3} — how many (S, R, V) slots the query matched
- `F_i` = full free energy: `(1-s) - Σ·H·ρ + λ·kT·ln2·H/τ`
- `kT` = energy scale (existing config parameter)

Expanding:
```
rank = (n_slots - 1) - (1-s)/kT + Σ·H·ρ/kT - λ·ln2·H/τ
```

Each term has physical meaning for ranking:
- `(n_slots - 1)`: Structural coupling energy — degrees of freedom aligned
- `-(1-s)/kT`: Consolidation bonus — well-consolidated memories rank higher
- `+Σ·H·ρ/kT`: Surprise-information bonus — surprising, informative facts rank higher
- `-λ·ln2·H/τ`: Landauer penalty — expensive-to-maintain memories rank lower

### LLM-Generated Semantic Field (at ingest time)

The extraction prompt returns an EXTENDED format:

```json
{
  "subject": "raj",
  "relation": "eat",
  "value": "banana",
  "override": false,
  "query_field": {
    "subject_aliases": ["raj"],
    "relation_forms": ["eat", "eats", "eating", "ate", "eaten", "food", "diet", "meal", "snack", "cuisine", "favorite food"],
    "value_aliases": ["banana", "bananas"]
  }
}
```

- The LLM generates ALL morphological variants (no stemmer needed)
- The LLM generates semantic neighbors (food ↔ eat, city ↔ visit)
- Zero hand-curated lists. Language-agnostic (LLM can generate cross-lingual forms)
- ~30 extra tokens per fact. Same LLM call.

### Triple-Index Architecture

```python
_subject_index:  dict[str, list[PhaseMemoryItem]]   # O(1) lookup
_relation_index: dict[str, list[PhaseMemoryItem]]    # O(1) lookup
_value_index:    dict[str, list[PhaseMemoryItem]]     # O(1) lookup
```

At ingest: insert item under ALL query_field keys (canonical + aliases + forms).
At search: tokenize query → hash lookup against all 3 indexes → union candidates.

**Scaling at 10T facts:**
- Index size grows with UNIQUE KEYS (vocabulary), not fact count
- Relations: ~100K unique × 15 forms = 1.5M keys × 10 bytes = 15MB
- Subjects: ~1B unique × 3 aliases = 3B keys × 20 bytes = 60GB
- Values: ~10M unique × 3 forms = 30M keys × 15 bytes = 450MB
- Total: ~61GB. Fits in RAM on a single node.

### Phase-Modulated Field Radius

```python
R(s) = floor(N_forms × s^(1/3))
```

- s = 1.0 → R = N_forms (all query forms indexed, max discoverability)
- s = 0.5 → R = floor(N × 0.79) (most forms indexed)
- s = 0.1 → R = floor(N × 0.46) (about half indexed)
- s < floor → R = 0 (de-indexed entirely — GAS PHASE)

Query forms are ordered by priority (canonical first, then common, then rare). R determines how many are active in the index. When s changes, forms are indexed/de-indexed via lazy evaluation.

### Search Algorithm (sub-μs)

```
query → tokenize → unigrams + bigrams
      → hash lookup against subject/relation/value indexes
      → candidate set (matched items)
      → for each candidate:
           recompute s(t), F, R(s)
           if R changed: lazy-update index entries
           rank = (n_matched_slots - 1) - F/kT
      → return top-k
```

Fallback: if zero candidates, return top-k liquid items by pure `-F/kT` (maximum entropy response).

Stop-word set: ~50 common English words excluded from index lookup to prevent noise.

### Performance Budget

| Step | Time |
|------|------|
| Tokenize + stopword filter | ~0.01ms |
| Index lookup (3 hash maps) | ~0.001ms |
| Score candidates (~100 items) | ~0.001ms |
| Sort top-k | ~0.005ms |
| Build context string | ~0.01ms |
| **Total** | **~0.03ms** |

LLM call: ~500ms. Our overhead: 0.03ms. Retrieval is invisible.

## Algorithm #2: Relation Symmetry Groups (future)

LLM normalizes relations to canonical forms at extraction time. No stemmer. The LLM IS the symmetry detector.

## Algorithm #3: Multi-Fact Fission (implemented)

Already in place. Single message → multiple facts. Each fact gets independent thermodynamic state.

## Algorithm #4: Cross-Entity Resonance Search (future)

For multi-entity queries ("both Jean and John"), decompose into sub-queries per entity. Each gets its own TSF search. Results merged and re-ranked.

## Algorithm #5: Contradiction Cascade (implemented)

Surprise damage with sigmoid sharpening. Already working.

## Algorithm #6: Augmented Context Assembly (future refinement)

Present liquid-phase memories ranked by -F to the LLM. No "NEWEST FIRST" hack.

## Files Changed

| File | Action | What Changes |
|------|--------|-------------|
| `src/clsplusplus/memory_phase.py` | MODIFY | Remove embeddings, add triple-index, add query_field, rewrite search(), update extraction prompt |
| `src/clsplusplus/config.py` | MODIFY | Remove embedding_model |
| `src/clsplusplus/demo_local.py` | MODIFY | Remove embedding_model param from engine init |
| `tests/test_memory_phase.py` | MODIFY | Update tests for TSF, remove embedding tests |

## Dependencies Removed

- `sentence-transformers` — no longer needed
- `numpy` — no longer needed (no vector operations)
- `nltk` — never needed (no stemmer)

## What's Removed from Code

- `_embed()`, `_embed_fact()`, `_cosine_similarity()`, `_get_embedder()`
- `embedding` field on PhaseMemoryItem
- `embedding_model` parameter on PhaseMemoryEngine
- `_stem_word()`, `_normalize_relation()` (replaced by LLM normalization)
- `import numpy as np`

## Non-Goals

- Overfitting to benchmark data format
- Changing the thermodynamic equations (F, s, τ, Σ — all stay exactly as-is)
- Adding external databases or vector stores
- Using any transformer/embedding model at search time
- Hand-curated morphology or synonym lists
