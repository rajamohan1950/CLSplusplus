# Changelog

## v0.4.0 — Landauer Crystallization Engine (2026-03-15)

**Full Phase Diagram: Gas → Liquid → Solid → Glass**

### Landauer Crystallization Engine (Liquid → Solid)
- ONE threshold: `ΔF(G) = F_schema − Σ F_liquid(i) + C_abstraction < 0`
- RG soft fixed point (Φ*): tokens in ≥80% of group members = schema content
- Landauer hysteresis: form at ΔF < 0, melt only when ΔF > F_melt = kT·ln(2)·H_lost
- Glass detection: schema entropy convergence (std/mean < 1%)
- Surprise resistance: schemas resist damage ∝ 1/(1+|ΔF|), glass 10× more
- Schema absorption: new episodes matching ≥60% of schema tokens reinforce the schema
- SchemaMeta dataclass: member_ids, fixed_point_tokens, H_schema, H_sum_episodes, delta_F, H_history

### SRG (Semantic Renormalization Group)
- Punctuation stripping as UV lattice coarse-graining
- Auto-fact subject heuristic with verb/adverb suffix detection

### Contradiction Cascade Deep Audit
- 7 bugs found and fixed in Algorithm #5
- Override signal refinement (removed false positive triggers: "only", "always", "never", "actually", "changed", "longer")

### PESQD Deep Audit
- Comprehensive structural and mathematical fixes

### Test Suite
- **1010 tests**, all passing in ~20s
- **8 exhaustive testing agents** deployed across 2 waves
- **Zero engine bugs** found (3 test bugs fixed)
- Coverage: adversarial, state invariants, math/stress, concurrency, thermodynamic laws, cross-algorithm

### Files Changed
- `src/clsplusplus/memory_phase.py` — +1088 lines (crystallization engine, SRG, CC fixes)
- `tests/test_memory_phase.py` — +12346 lines (1010 tests)

---

## v0.3.0 — Cross-Entity Resonance & TSF (2026-03-10)

- Thermodynamic Semantic Field (TSF) — zero-ML sub-μs retrieval
- Cross-Entity Resonance (CER) — Kuramoto coupled oscillator model
- Self-sufficient memory engine — store/search/augment without external LLM
- Chat UI with cross-session memory

## v0.2.0 — Initial Phase Engine

- Gas → Liquid phase transition
- Free energy formulation
- Basic plasticity scoring

## v0.1.0 — Project Scaffold

- FastAPI skeleton
- Docker Compose with Redis, PostgreSQL
- Initial documentation
