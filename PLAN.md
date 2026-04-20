# Plan: Fix Integration Gaps → Run LoCoMo → Write HLD

## Critical Bug Found

**In batch mode (benchmark), SVD never fires.** The `_recompute_svd()` check on line 2242 skips when `_batch_mode=True`. After batch ingest sets `_batch_mode=False`, there's **no trigger** to recompute SVD. So `self._token_vectors` stays empty → `semantic_bonus = 0` for EVERY query → the entire PPMI-SVD layer is dead weight in benchmarks.

## Step 1: Fix batch-mode SVD gap in `memory_phase.py`

After `_batch_mode = False` in `_direct_ingest_conversation()`, trigger:
```python
engine._recompute_svd()
```

Also add a `finalize_batch()` public method to PhaseMemoryEngine that:
1. Recomputes SVD from accumulated co-occurrence
2. Recomputes free energies for all namespaces
3. Updates CER graph

This is the proper API for batch ingest → query transitions.

## Step 2: Run LoCoMo benchmark (post-fix)

Run: `python3 benchmarking_LoCoMo/run_clspp_benchmark.py --mode direct --out-file benchmarking_LoCoMo/outputs/clspp_v050_full.json`

This will validate whether PPMI-SVD semantic bonus improves retrieval over the stale TRR v1 run (0.2395 F1).

## Step 3: Write HLD document

Create `docs/HLD_USER_EMBEDDINGS.md` covering:
- The mathematical foundation (PPMI ≅ Word2Vec insight)
- 4-layer architecture with diagrams
- Scale engineering (sliding window, CMS, incremental SVD)
- Integration with PhaseMemoryEngine
- Privacy model (vectors only, never raw text)
- 50-year forward design (generational knowledge store)
- Benchmark results (before/after comparison)
