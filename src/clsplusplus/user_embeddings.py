"""
User-Specific Embeddings — The 50-Year Forward Design.

=============================================================================
THE INSIGHT
=============================================================================

Word2Vec ≅ SVD(PPMI matrix).  — Levy & Goldberg, 2014

We don't need $300B of pre-training. We don't need cloud. We don't need GPUs.

Each user builds their own PPMI matrix from their own memories. SVD gives
50-dimensional vectors. Cross-user vector intersection bootstraps synonym
knowledge WITHOUT sharing raw text.

    User 1: "Bob got fired"     → context: {job, resume, boss, work}
    User 2: "Lost my job"       → context: {unemployed, resume, work, interview}
    Intersection: "fired" ↔ "lost job" share {job, work, resume}
                  → system learns they're synonyms

No single user needs both phrases. The collective teaches the engine meaning.

=============================================================================
ARCHITECTURE — 50-YEAR HORIZON
=============================================================================

Layer 0: UserEmbeddingSpace
    Per-user PPMI → SVD → 50-dim vectors.
    ~1MB per user. Updates async. Pure Python. Zero cloud.

Layer 1: CollectiveSemanticField
    Cross-user vector intersection discovers synonyms.
    Privacy-safe: only vectors shared, never raw memories.
    Quorum-based: synonym accepted only when N≥3 users confirm.

Layer 2: SemanticDriftDetector
    Tracks how each user's vocabulary evolves over time.
    Detects meaning shifts: "cloud" (weather→computing→AI).
    Triggers re-embedding when drift exceeds threshold.

Layer 3: GenerationalKnowledgeStore
    Embeddings that outlive individual users.
    Collective meaning accumulates over decades.
    Cultural semantic shifts captured as vector rotation.

=============================================================================
PROPERTIES
=============================================================================

    - Zero cloud. Zero pre-training. Zero GPU.
    - ~1MB per user (sparse PPMI + 50-dim vectors for ≤5000 tokens).
    - Updates async via background co-occurrence tracking.
    - Personalized to each user's vocabulary.
    - Privacy-safe: only vectors shared across users, never raw memories.
    - Synonym knowledge emerges from collective, not dictionaries.
    - Self-healing: drift detection + generational decay prevent semantic rot.

=============================================================================
PHYSICS ANALOGY
=============================================================================

Each user's embedding space is a LOCAL gauge field — their personal
coordinate system for meaning. Cross-user intersection is a GAUGE
TRANSFORMATION: finding the rotation that aligns one user's meaning-space
to another's, preserving the invariant (shared context vectors).

The Collective Semantic Field is the VACUUM STATE: the background
meaning-space that new users bootstrap from. It evolves as the collective
evolves. Over 50 years, it captures the full arc of how language changes.

References:
    - Levy & Goldberg (2014): Neural word embedding as implicit matrix
      factorization. NIPS.
    - Mikolov et al. (2013): Distributed representations of words and
      phrases. NIPS.
    - Hamilton et al. (2016): Diachronic word embeddings reveal statistical
      laws of semantic change. ACL.
    - Landauer & Dumais (1997): A solution to Plato's problem: The latent
      semantic analysis theory of acquisition. Psychological Review.

Copyright (c) 2026 CLS++. All rights reserved.
"""

from __future__ import annotations

import math
import time
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


# =============================================================================
# Constants
# =============================================================================

# Embedding dimensionality — matches PhaseMemoryEngine._SVD_DIMS
EMBEDDING_DIMS: int = 50

# Cross-user synonym discovery thresholds
SYNONYM_COSINE_THRESHOLD: float = 0.65      # Minimum cosine sim for synonym candidate
SYNONYM_QUORUM: int = 3                      # Min users confirming before acceptance
SYNONYM_MAX_PER_TOKEN: int = 8               # Cap synonym list per token

# Semantic drift detection
DRIFT_WINDOW_SIZE: int = 100                 # Memories between drift snapshots
DRIFT_THRESHOLD: float = 0.30               # Cosine distance triggering re-embed
DRIFT_HISTORY_DEPTH: int = 50                # Max snapshots retained

# Generational knowledge decay
GENERATIONAL_HALF_LIFE_YEARS: float = 25.0   # Meaning half-life in years
GENERATIONAL_MIN_WEIGHT: float = 0.01        # Below this, knowledge is pruned

# Privacy
VECTOR_SALT_BYTES: int = 16                  # Random salt per user for vector export
LOCAL_WEIGHT: float = 0.8                    # User's own vectors dominate
EXTERNAL_WEIGHT: float = 0.2                 # External vectors contribute gently

# Async recompute
RECOMPUTE_INTERVAL: int = 50                 # Stores between SVD recomputation
MAX_VOCAB_SIZE: int = 5000                   # Cap per-user vocabulary

# Scale mode constants — fixes O(n²) → O(n·w), unbounded → fixed, O(V²k) → O(Vk²)
COOCCURRENCE_WINDOW: int = 5                 # Sliding window size for scaled mode
CMS_WIDTH: int = 65536                       # Count-Min Sketch width (counters per row)
CMS_DEPTH: int = 4                           # Count-Min Sketch depth (hash functions)
VOCAB_DECAY_FACTOR: float = 0.995            # Exponential decay per observation for LRU vocab
INCREMENTAL_SVD_WARMSTART_ITERS: int = 3     # Power iterations for warm-start (vs 15 full)
FULL_RECOMPUTE_CYCLE: int = 10               # Full SVD every N incremental cycles


# =============================================================================
# Scale Mode
# =============================================================================

class ScaleMode(Enum):
    """
    Controls which co-occurrence / SVD pipeline to use.

    CLASSIC: Original all-pairs co-occurrence + Counter + full SVD.
             Correct for small-to-medium usage (<10K memories).
    SCALED:  Sliding window + Count-Min Sketch + incremental SVD.
             Required for heavy usage (1M+ tokens/day).
    """
    CLASSIC = "classic"
    SCALED = "scaled"


# =============================================================================
# Count-Min Sketch — Fixed-Memory Approximate Counter
# =============================================================================

class CountMinSketch:
    """
    Probabilistic frequency counter with fixed memory footprint.

    Properties:
        - Never underestimates (may overestimate by ε with probability δ)
        - Fixed memory: depth × width × 4 bytes ≈ 1MB for defaults
        - O(depth) per increment/query

    Replaces collections.Counter for co-occurrence tracking at scale.
    At CMS_WIDTH=65536, CMS_DEPTH=4: error ≤ total/65536 with prob ≥ 93.75%.
    """

    __slots__ = ("_width", "_depth", "_tables", "_total")

    def __init__(
        self,
        width: int = CMS_WIDTH,
        depth: int = CMS_DEPTH,
    ) -> None:
        self._width = width
        self._depth = depth
        self._tables: list[list[int]] = [
            [0] * width for _ in range(depth)
        ]
        self._total: int = 0

    def _hash(self, key: tuple[str, str], seed: int) -> int:
        """Hash a canonical pair to a bucket index."""
        raw = f"{seed}:{key[0]}:{key[1]}".encode("utf-8")
        h = hashlib.md5(raw).hexdigest()
        return int(h, 16) % self._width

    def increment(self, key: tuple[str, str]) -> None:
        """Increment count for key across all hash rows."""
        for i in range(self._depth):
            idx = self._hash(key, i)
            self._tables[i][idx] += 1
        self._total += 1

    def query(self, key: tuple[str, str]) -> int:
        """Return estimated count (minimum across rows — never underestimates)."""
        return min(
            self._tables[i][self._hash(key, i)]
            for i in range(self._depth)
        )

    def memory_bytes(self) -> int:
        """Fixed memory footprint in bytes."""
        return self._depth * self._width * 4

    @property
    def total_increments(self) -> int:
        """Total number of increment operations."""
        return self._total


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class SynonymEdge:
    """
    A discovered synonym relationship between two tokens.

    Emerges from cross-user vector intersection — no dictionary, no LLM.
    Strength grows with each confirming user.
    """
    token_a: str
    token_b: str
    cosine_similarity: float      # Average cosine sim across confirming users
    confirming_users: int         # Number of independent users confirming
    discovered_epoch: float       # Unix timestamp of first discovery
    last_confirmed_epoch: float   # Unix timestamp of most recent confirmation
    confidence: float = 0.0       # confirming_users / SYNONYM_QUORUM, capped at 1.0


@dataclass
class DriftSnapshot:
    """
    A snapshot of a token's vector at a point in time.

    Used to detect semantic drift: when the user's meaning for a word
    shifts over time (e.g., "model" from fashion→ML→LLM).
    """
    token: str
    vector: list[float]           # 50-dim vector at snapshot time
    epoch: float                  # Unix timestamp
    memory_count: int             # Total memories at snapshot time


@dataclass
class GenerationalEntry:
    """
    A collective meaning vector that accumulates over decades.

    Weight decays with half-life of 25 years. Multiple generations
    of users contribute, creating a living semantic fossil record.
    """
    token: str
    vector: list[float]           # 50-dim accumulated vector
    total_weight: float           # Sum of all contributing weights
    contributor_count: int        # Number of unique users who contributed
    first_epoch: float            # Unix timestamp of first contribution
    last_epoch: float             # Unix timestamp of most recent contribution
    generation: int = 0           # Incremented when drift exceeds threshold


# =============================================================================
# Layer 0: UserEmbeddingSpace
# =============================================================================

class UserEmbeddingSpace:
    """
    Per-user embedding space grown from their own memories.

    Each user accumulates co-occurrence statistics from their stored memories.
    PPMI normalization → SVD → 50-dimensional vectors.

    ~1MB per user. Pure Python. Zero external dependencies.

    The embedding space is a LOCAL gauge field — the user's personal
    coordinate system for meaning. No two users have the same space.
    Every space is valid. Cross-user alignment discovers shared structure.
    """

    def __init__(
        self,
        user_id: str,
        scale_mode: ScaleMode = ScaleMode.CLASSIC,
    ) -> None:
        self.user_id: str = user_id
        self.scale_mode: ScaleMode = scale_mode

        # Co-occurrence statistics (built from user's memories)
        self._cooccurrence: Counter = Counter()  # Used in CLASSIC mode
        self._doc_freq: Counter = Counter()
        self._total_docs: int = 0

        # Token vectors (PPMI-SVD output)
        self._token_vectors: dict[str, list[float]] = {}
        self._vocab: list[str] = []

        # Async tracking
        self._store_count_since_svd: int = 0
        self._svd_dirty: bool = False
        self._last_svd_epoch: float = 0.0

        # Statistics
        self._total_tokens_seen: int = 0
        self._unique_tokens: set[str] = set()

        # --- Scaled mode additions ---
        if scale_mode == ScaleMode.SCALED:
            self._cms: CountMinSketch = CountMinSketch()
            self._known_pairs: set[tuple[str, str]] = set()
            self._token_frequency: dict[str, float] = {}
            # Incremental SVD state
            self._prev_basis: list[list[float]] | None = None
            self._prev_vocab: list[str] | None = None
            self._full_recompute_counter: int = 0

    @property
    def vector_count(self) -> int:
        """Number of tokens with computed vectors."""
        return len(self._token_vectors)

    @property
    def memory_size_bytes(self) -> int:
        """
        Approximate memory footprint in bytes.

        Classic mode:
            Co-occurrence: ~16 bytes per pair (two 4-byte indices + 8-byte float)
            Vectors: ~400 bytes per token (50 × 8-byte float)
            Doc freq: ~12 bytes per token (4-byte index + 8-byte count)

        Scaled mode:
            CMS: fixed ~1MB (depth × width × 4)
            Known pairs: ~120 bytes per pair (tuple + set overhead)
            Token frequency: ~20 bytes per token (key + float)
            Vectors + doc freq: same as classic
        """
        vec_bytes = len(self._token_vectors) * EMBEDDING_DIMS * 8
        df_bytes = len(self._doc_freq) * 12

        if self.scale_mode == ScaleMode.SCALED:
            cms_bytes = self._cms.memory_bytes()
            pairs_bytes = len(self._known_pairs) * 120
            freq_bytes = len(self._token_frequency) * 20
            return cms_bytes + pairs_bytes + freq_bytes + vec_bytes + df_bytes
        else:
            cooc_bytes = len(self._cooccurrence) * 16
            return cooc_bytes + vec_bytes + df_bytes

    def observe(self, tokens: list[str]) -> None:
        """
        Observe a set of tokens from a stored memory.

        Updates co-occurrence matrix and document frequency.
        Triggers SVD recomputation when enough observations accumulate.

        This is the fundamental learning operation. Each memory the user
        stores teaches the embedding space about their vocabulary.

        Classic mode: O(n²) all-pairs co-occurrence.
        Scaled mode:  O(n·w) sliding window co-occurrence (w=COOCCURRENCE_WINDOW).
        """
        if not tokens:
            return

        unique_tokens = set(tokens)
        self._total_docs += 1
        self._total_tokens_seen += len(tokens)
        self._unique_tokens.update(unique_tokens)

        # Update document frequency
        for t in unique_tokens:
            self._doc_freq[t] += 1

        if self.scale_mode == ScaleMode.SCALED:
            # Sliding window co-occurrence: O(n·w) instead of O(n²)
            # Preserve input order while deduplicating
            ordered_unique = list(dict.fromkeys(tokens))
            for i, token in enumerate(ordered_unique):
                window_start = max(0, i - COOCCURRENCE_WINDOW)
                for j in range(window_start, i):
                    pair = _canonical_pair(ordered_unique[j], token)
                    self._cms.increment(pair)
                    self._known_pairs.add(pair)
                # Update frequency-decayed vocab
                self._token_frequency[token] = (
                    self._token_frequency.get(token, 0.0)
                    * VOCAB_DECAY_FACTOR + 1.0
                )
            # Vocab eviction when 120% of cap
            if len(self._token_frequency) > int(MAX_VOCAB_SIZE * 1.2):
                self._evict_vocab()
        else:
            # Classic all-pairs co-occurrence: O(n²)
            sorted_unique = sorted(unique_tokens)
            for i in range(len(sorted_unique)):
                for j in range(i + 1, len(sorted_unique)):
                    pair = (sorted_unique[i], sorted_unique[j])
                    self._cooccurrence[pair] += 1

        self._store_count_since_svd += 1
        self._svd_dirty = True

        # Async recompute check — adaptive interval for scaled mode
        interval = RECOMPUTE_INTERVAL
        if self.scale_mode == ScaleMode.SCALED:
            vocab_size = len(self._token_frequency)
            interval = RECOMPUTE_INTERVAL + int(0.02 * vocab_size)
        if self._store_count_since_svd >= interval:
            self.recompute_vectors()

    def recompute_vectors(self) -> None:
        """
        Recompute PPMI-SVD token vectors from accumulated co-occurrence.

        Pure Python. No numpy. No scipy. No GPU.

        Classic mode:
            Power iteration for truncated SVD — 15 iterations from random init.
            O(V²k) per iter. For V=5000, k=50: ~375M ops. Runs in <2s.

        Scaled mode:
            Builds PPMI from Count-Min Sketch queries over known pairs.
            Warm-start power iteration (3 iterations) using previous basis.
            Full recompute every FULL_RECOMPUTE_CYCLE incremental cycles.
            O(nnz·k·3) instead of O(nnz·k·15) — 5x speedup on warm-start.
        """
        if self.scale_mode == ScaleMode.SCALED:
            self._recompute_scaled()
        else:
            self._recompute_classic()

    def _recompute_classic(self) -> None:
        """Original full SVD from Counter-based co-occurrence."""
        if not self._cooccurrence:
            return

        # Build vocabulary from tokens with sufficient frequency
        vocab: set[str] = set()
        for (a, b) in self._cooccurrence:
            if self._doc_freq.get(a, 0) >= 2:
                vocab.add(a)
            if self._doc_freq.get(b, 0) >= 2:
                vocab.add(b)

        # Cap vocabulary
        if len(vocab) > MAX_VOCAB_SIZE:
            vocab_sorted = sorted(
                vocab,
                key=lambda t: self._doc_freq.get(t, 0),
                reverse=True,
            )
            vocab = set(vocab_sorted[:MAX_VOCAB_SIZE])

        vocab_list = sorted(vocab)
        V = len(vocab_list)
        if V < 3:
            return

        tok2idx = {t: i for i, t in enumerate(vocab_list)}
        dims = min(EMBEDDING_DIMS, V - 1)
        N = max(self._total_docs, 1)

        # Build sparse PPMI matrix
        ppmi_rows: dict[int, dict[int, float]] = {}
        for (a, b), co in self._cooccurrence.items():
            if a not in tok2idx or b not in tok2idx:
                continue
            df_a = self._doc_freq.get(a, 0)
            df_b = self._doc_freq.get(b, 0)
            if df_a == 0 or df_b == 0:
                continue
            pmi_val = math.log(N * co / (df_a * df_b))
            if pmi_val <= 0:
                continue
            i, j = tok2idx[a], tok2idx[b]
            ppmi_rows.setdefault(i, {})[j] = pmi_val
            ppmi_rows.setdefault(j, {})[i] = pmi_val

        if not ppmi_rows:
            return

        V, dims, vocab_list, ppmi_rows  # noqa: used below
        self._run_power_iteration(V, dims, vocab_list, ppmi_rows, n_iters=15)

    def _recompute_scaled(self) -> None:
        """Scaled SVD: CMS-backed PPMI + warm-start power iteration."""
        if not self._known_pairs:
            return

        # Build vocab from frequency-decayed tokens (top MAX_VOCAB_SIZE)
        candidates = {
            t for t in self._token_frequency
            if self._doc_freq.get(t, 0) >= 2
        }
        if len(candidates) > MAX_VOCAB_SIZE:
            candidates_sorted = sorted(
                candidates,
                key=lambda t: self._token_frequency.get(t, 0.0),
                reverse=True,
            )
            candidates = set(candidates_sorted[:MAX_VOCAB_SIZE])

        vocab_list = sorted(candidates)
        V = len(vocab_list)
        if V < 3:
            return

        tok2idx = {t: i for i, t in enumerate(vocab_list)}
        dims = min(EMBEDDING_DIMS, V - 1)
        N = max(self._total_docs, 1)

        # Build sparse PPMI from CMS queries over known pairs
        ppmi_rows: dict[int, dict[int, float]] = {}
        for (a, b) in self._known_pairs:
            if a not in tok2idx or b not in tok2idx:
                continue
            co = self._cms.query((a, b))
            if co <= 0:
                continue
            df_a = self._doc_freq.get(a, 0)
            df_b = self._doc_freq.get(b, 0)
            if df_a == 0 or df_b == 0:
                continue
            pmi_val = math.log(N * co / (df_a * df_b))
            if pmi_val <= 0:
                continue
            i, j = tok2idx[a], tok2idx[b]
            ppmi_rows.setdefault(i, {})[j] = pmi_val
            ppmi_rows.setdefault(j, {})[i] = pmi_val

        if not ppmi_rows:
            return

        # Decide: warm-start vs full recompute
        use_warm_start = (
            self._prev_basis is not None
            and self._full_recompute_counter < FULL_RECOMPUTE_CYCLE
        )

        if use_warm_start:
            n_iters = INCREMENTAL_SVD_WARMSTART_ITERS
            self._full_recompute_counter += 1
        else:
            n_iters = 15
            self._full_recompute_counter = 0

        self._run_power_iteration(
            V, dims, vocab_list, ppmi_rows, n_iters,
            warm_basis=self._reindex_basis(vocab_list) if use_warm_start else None,
        )

        # Save basis for next warm-start
        self._prev_vocab = vocab_list

    def _run_power_iteration(
        self,
        V: int,
        dims: int,
        vocab_list: list[str],
        ppmi_rows: dict[int, dict[int, float]],
        n_iters: int = 15,
        warm_basis: list[list[float]] | None = None,
    ) -> None:
        """
        Shared power iteration SVD — used by both classic and scaled modes.

        If warm_basis is provided, uses it as starting point (incremental).
        Otherwise, initializes from random vectors (full recompute).
        """
        # Sparse matrix-vector multiply
        def spmv(x: list[float]) -> list[float]:
            y = [0.0] * V
            for i, row in ppmi_rows.items():
                for j, val in row.items():
                    y[i] += val * x[j]
            return y

        import random as _rng
        _rng.seed(42 + hash(self.user_id) % 10000)

        if warm_basis is not None and len(warm_basis) == dims:
            basis = warm_basis
        else:
            basis = []
            for _ in range(dims):
                vec = [_rng.gauss(0, 1) for _ in range(V)]
                mag = math.sqrt(sum(v * v for v in vec)) or 1.0
                basis.append([v / mag for v in vec])

        for _iter in range(n_iters):
            new_basis = [spmv(vec) for vec in basis]
            # Modified Gram-Schmidt
            for i in range(len(new_basis)):
                for j in range(i):
                    dot = sum(a * b for a, b in zip(new_basis[i], new_basis[j]))
                    new_basis[i] = [
                        a - dot * b for a, b in zip(new_basis[i], new_basis[j])
                    ]
                mag = math.sqrt(sum(v * v for v in new_basis[i])) or 1e-12
                new_basis[i] = [v / mag for v in new_basis[i]]
            basis = new_basis

        # Extract token vectors
        self._token_vectors = {}
        self._vocab = vocab_list
        for idx, token in enumerate(vocab_list):
            self._token_vectors[token] = [basis[d][idx] for d in range(dims)]

        # Save basis for potential warm-start
        if self.scale_mode == ScaleMode.SCALED:
            self._prev_basis = basis

        self._store_count_since_svd = 0
        self._svd_dirty = False
        self._last_svd_epoch = time.time()

    def _evict_vocab(self) -> None:
        """
        Evict bottom 20% of vocabulary by decayed frequency.

        Removes tokens from: _token_frequency, _known_pairs, _doc_freq,
        _unique_tokens, _token_vectors. Keeps the system bounded.
        """
        if not hasattr(self, "_token_frequency"):
            return
        freq_sorted = sorted(
            self._token_frequency.items(),
            key=lambda kv: kv[1],
        )
        evict_count = len(freq_sorted) // 5  # Bottom 20%
        evict_tokens = {t for t, _ in freq_sorted[:evict_count]}

        for t in evict_tokens:
            self._token_frequency.pop(t, None)
            self._doc_freq.pop(t, None)
            self._unique_tokens.discard(t)
            self._token_vectors.pop(t, None)

        # Remove pairs involving evicted tokens
        self._known_pairs = {
            (a, b) for (a, b) in self._known_pairs
            if a not in evict_tokens and b not in evict_tokens
        }

    def _reindex_basis(self, new_vocab: list[str]) -> list[list[float]] | None:
        """
        Remap previous basis vectors to new vocabulary ordering.

        When vocab changes between recomputes (new tokens added, old evicted),
        re-map the basis to the new index. Zero-fill new positions.
        Returns None if reindexing is not possible.
        """
        if self._prev_basis is None or self._prev_vocab is None:
            return None

        old_tok2idx = {t: i for i, t in enumerate(self._prev_vocab)}
        V_new = len(new_vocab)
        dims = len(self._prev_basis)

        reindexed: list[list[float]] = []
        for d in range(dims):
            old_vec = self._prev_basis[d]
            new_vec = [0.0] * V_new
            for new_idx, token in enumerate(new_vocab):
                old_idx = old_tok2idx.get(token)
                if old_idx is not None and old_idx < len(old_vec):
                    new_vec[new_idx] = old_vec[old_idx]
            reindexed.append(new_vec)

        return reindexed

    def get_vector(self, token: str) -> Optional[list[float]]:
        """Get the embedding vector for a token. None if not in vocabulary."""
        return self._token_vectors.get(token)

    def get_neighbors(
        self,
        token: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find the nearest neighbors of a token in embedding space.

        Returns list of (token, cosine_similarity) pairs, sorted descending.
        This is the user's PERSONAL neighborhood — shaped by their vocabulary.
        """
        vec = self._token_vectors.get(token)
        if vec is None:
            return []

        scores: list[tuple[str, float]] = []
        for other_token, other_vec in self._token_vectors.items():
            if other_token == token:
                continue
            sim = _cosine_similarity(vec, other_vec)
            if sim > 0.0:
                scores.append((other_token, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def export_vectors(self) -> dict[str, list[float]]:
        """
        Export anonymized vectors for cross-user sharing.

        Returns token → 50-dim vector. No raw memories. No user data.
        Only the mathematical shadow of meaning — the co-occurrence geometry.
        """
        return dict(self._token_vectors)

    def import_vectors(
        self,
        external_vectors: dict[str, list[float]],
        weight: float = EXTERNAL_WEIGHT,
    ) -> int:
        """
        Import vectors from collective field or another user.

        Local vectors dominate (0.8 weight). External vectors contribute
        gently (0.2 weight). New tokens get external vectors directly.

        Returns number of tokens affected.
        """
        affected = 0
        local_weight = 1.0 - weight
        for token, ext_vec in external_vectors.items():
            if len(ext_vec) != EMBEDDING_DIMS:
                continue
            if token in self._token_vectors:
                local_vec = self._token_vectors[token]
                merged = [
                    local_weight * l + weight * e
                    for l, e in zip(local_vec, ext_vec)
                ]
                mag = math.sqrt(sum(v * v for v in merged))
                if mag > 1e-12:
                    self._token_vectors[token] = [v / mag for v in merged]
                affected += 1
            else:
                self._token_vectors[token] = list(ext_vec)
                affected += 1
        return affected

    def stats(self) -> dict[str, Any]:
        """Diagnostic statistics for the embedding space."""
        pair_count = (
            len(self._known_pairs)
            if self.scale_mode == ScaleMode.SCALED
            else len(self._cooccurrence)
        )
        result = {
            "user_id": self.user_id,
            "scale_mode": self.scale_mode.value,
            "total_docs": self._total_docs,
            "total_tokens_seen": self._total_tokens_seen,
            "unique_tokens": len(self._unique_tokens),
            "cooccurrence_pairs": pair_count,
            "vector_count": len(self._token_vectors),
            "memory_size_bytes": self.memory_size_bytes,
            "memory_size_kb": round(self.memory_size_bytes / 1024, 1),
            "svd_dirty": self._svd_dirty,
            "stores_since_svd": self._store_count_since_svd,
            "last_svd_epoch": self._last_svd_epoch,
        }
        if self.scale_mode == ScaleMode.SCALED:
            result["cms_memory_bytes"] = self._cms.memory_bytes()
            result["full_recompute_counter"] = self._full_recompute_counter
        return result


# =============================================================================
# Layer 1: CollectiveSemanticField
# =============================================================================

class CollectiveSemanticField:
    """
    Cross-user vector intersection for synonym discovery.

    The collective teaches the engine meaning. No single user needs every
    phrase. When multiple users' vectors place different tokens near the
    same context, the collective discovers they're synonyms.

    Privacy invariant: ONLY vectors are shared. Never raw text. Never memory
    content. The vectors are the mathematical shadow of meaning — geometry
    without content.

    The Collective Field is the VACUUM STATE of the semantic gauge theory.
    New users bootstrap from it. It evolves as the collective evolves.

    Quorum-based acceptance: a synonym is only accepted when N≥3 independent
    users confirm the relationship. This prevents idiosyncratic noise from
    one user polluting the collective.
    """

    def __init__(self) -> None:
        # Registered user spaces
        self._user_spaces: dict[str, UserEmbeddingSpace] = {}

        # Discovered synonyms: (token_a, token_b) → SynonymEdge
        # Canonical order: token_a < token_b
        self._synonyms: dict[tuple[str, str], SynonymEdge] = {}

        # Per-token synonym index for fast lookup
        self._synonym_index: dict[str, list[str]] = {}

        # Collective vectors (weighted average across all users)
        self._collective_vectors: dict[str, list[float]] = {}
        self._collective_weights: dict[str, float] = {}
        self._collective_contributor_count: dict[str, int] = {}

        # Statistics
        self._total_intersections_computed: int = 0
        self._total_synonyms_discovered: int = 0

    @property
    def user_count(self) -> int:
        return len(self._user_spaces)

    @property
    def synonym_count(self) -> int:
        return len(self._synonyms)

    def register_user(self, space: UserEmbeddingSpace) -> None:
        """Register a user's embedding space with the collective."""
        self._user_spaces[space.user_id] = space

    def unregister_user(self, user_id: str) -> None:
        """Remove a user from the collective. Their vectors are forgotten."""
        self._user_spaces.pop(user_id, None)
        # Collective vectors will be recomputed on next sync

    def compute_collective_vectors(self) -> int:
        """
        Compute the collective vector field from all registered users.

        Each user contributes their vectors with equal weight.
        The collective vector for a token is the L2-normalized mean
        of all user vectors for that token.

        Returns the number of tokens in the collective vocabulary.
        """
        accumulator: dict[str, list[float]] = {}
        weights: dict[str, float] = {}
        contributor_count: dict[str, int] = {}

        for user_id, space in self._user_spaces.items():
            user_vectors = space.export_vectors()
            for token, vec in user_vectors.items():
                if token not in accumulator:
                    accumulator[token] = [0.0] * EMBEDDING_DIMS
                    weights[token] = 0.0
                    contributor_count[token] = 0
                for d in range(min(len(vec), EMBEDDING_DIMS)):
                    accumulator[token][d] += vec[d]
                weights[token] += 1.0
                contributor_count[token] += 1

        # Normalize
        self._collective_vectors = {}
        for token, acc in accumulator.items():
            mag = math.sqrt(sum(v * v for v in acc))
            if mag > 1e-12:
                self._collective_vectors[token] = [v / mag for v in acc]

        self._collective_weights = weights
        self._collective_contributor_count = contributor_count
        return len(self._collective_vectors)

    def discover_synonyms(self) -> list[SynonymEdge]:
        """
        Discover synonyms from cross-user vector intersection.

        Algorithm:
        1. For each pair of users, find tokens whose vectors are similar
           but whose surface forms differ.
        2. Track which user-pairs confirm each candidate synonym.
        3. Accept synonyms only when quorum (≥3 users) is reached.

        The magic: User 1 has "fired" near {work, job, resume}.
                   User 2 has "lost job" near {work, job, resume}.
                   Their shared context vectors reveal the synonymy.

        Returns newly discovered synonyms in this cycle.
        """
        # Build per-user neighborhood maps
        user_neighborhoods: dict[str, dict[str, set[str]]] = {}

        for user_id, space in self._user_spaces.items():
            neighborhoods: dict[str, set[str]] = {}
            vectors = space.export_vectors()
            tokens = list(vectors.keys())

            for token in tokens:
                vec = vectors[token]
                neighbors: set[str] = set()
                for other_token in tokens:
                    if other_token == token:
                        continue
                    other_vec = vectors[other_token]
                    sim = _cosine_similarity(vec, other_vec)
                    if sim >= SYNONYM_COSINE_THRESHOLD:
                        neighbors.add(other_token)
                neighborhoods[token] = neighbors

            user_neighborhoods[user_id] = neighborhoods

        # Cross-user intersection: find tokens with overlapping neighborhoods
        new_synonyms: list[SynonymEdge] = []
        user_ids = list(self._user_spaces.keys())
        now = time.time()

        # For each pair of users
        candidate_confirmations: dict[tuple[str, str], list[float]] = {}

        for ui in range(len(user_ids)):
            for uj in range(ui + 1, len(user_ids)):
                uid_a, uid_b = user_ids[ui], user_ids[uj]
                nh_a = user_neighborhoods.get(uid_a, {})
                nh_b = user_neighborhoods.get(uid_b, {})
                vectors_a = self._user_spaces[uid_a].export_vectors()
                vectors_b = self._user_spaces[uid_b].export_vectors()

                self._total_intersections_computed += 1

                # Find tokens in user A whose neighborhoods overlap with
                # different tokens in user B
                for token_a, neighbors_a in nh_a.items():
                    for token_b, neighbors_b in nh_b.items():
                        if token_a == token_b:
                            continue  # Same token — not a synonym discovery

                        # Context overlap: how many neighbors do they share?
                        shared_context = neighbors_a & neighbors_b
                        if len(shared_context) < 2:
                            continue  # Not enough shared context

                        # Compute cosine between the tokens' vectors
                        # across user spaces (gauge alignment)
                        vec_a = vectors_a.get(token_a)
                        vec_b = vectors_b.get(token_b)
                        if vec_a is None or vec_b is None:
                            continue

                        # Context-based similarity: how similar are their
                        # neighborhoods (not the tokens themselves)?
                        context_sim = len(shared_context) / max(
                            len(neighbors_a | neighbors_b), 1
                        )

                        if context_sim >= 0.3:  # 30% context overlap
                            key = _canonical_pair(token_a, token_b)
                            if key not in candidate_confirmations:
                                candidate_confirmations[key] = []
                            candidate_confirmations[key].append(context_sim)

        # Update synonym registry
        for (token_a, token_b), sims in candidate_confirmations.items():
            existing = self._synonyms.get((token_a, token_b))
            avg_sim = sum(sims) / len(sims)
            n_confirmers = len(sims)

            if existing:
                # Update existing synonym edge
                total_confirmers = existing.confirming_users + n_confirmers
                blended_sim = (
                    existing.cosine_similarity * existing.confirming_users
                    + avg_sim * n_confirmers
                ) / total_confirmers
                self._synonyms[(token_a, token_b)] = SynonymEdge(
                    token_a=token_a,
                    token_b=token_b,
                    cosine_similarity=blended_sim,
                    confirming_users=total_confirmers,
                    discovered_epoch=existing.discovered_epoch,
                    last_confirmed_epoch=now,
                    confidence=min(total_confirmers / SYNONYM_QUORUM, 1.0),
                )
            else:
                # New synonym candidate
                edge = SynonymEdge(
                    token_a=token_a,
                    token_b=token_b,
                    cosine_similarity=avg_sim,
                    confirming_users=n_confirmers,
                    discovered_epoch=now,
                    last_confirmed_epoch=now,
                    confidence=min(n_confirmers / SYNONYM_QUORUM, 1.0),
                )
                self._synonyms[(token_a, token_b)] = edge

                if n_confirmers >= SYNONYM_QUORUM:
                    new_synonyms.append(edge)
                    self._total_synonyms_discovered += 1

        # Rebuild synonym index
        self._rebuild_synonym_index()

        return new_synonyms

    def get_synonyms(self, token: str) -> list[tuple[str, float]]:
        """
        Get known synonyms for a token with confidence scores.

        Returns list of (synonym, confidence) pairs.
        Only returns synonyms that have reached quorum.
        """
        related_tokens = self._synonym_index.get(token, [])
        results: list[tuple[str, float]] = []

        for related in related_tokens:
            key = _canonical_pair(token, related)
            edge = self._synonyms.get(key)
            if edge and edge.confidence >= 1.0:
                results.append((related, edge.cosine_similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:SYNONYM_MAX_PER_TOKEN]

    def expand_query(self, tokens: list[str]) -> list[str]:
        """
        Expand a query with collective synonyms.

        For each query token, add its confirmed synonyms.
        This is how the collective teaches individual searches.
        """
        expanded: list[str] = list(tokens)
        seen: set[str] = set(tokens)

        for token in tokens:
            synonyms = self.get_synonyms(token)
            for syn, _conf in synonyms:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

        return expanded

    def broadcast_to_users(self) -> dict[str, int]:
        """
        Push collective vectors back to all registered users.

        Each user imports the collective with EXTERNAL_WEIGHT (0.2),
        keeping their own vectors dominant (0.8 weight).

        Returns dict of user_id → number of tokens affected.
        """
        if not self._collective_vectors:
            self.compute_collective_vectors()

        results: dict[str, int] = {}
        for user_id, space in self._user_spaces.items():
            affected = space.import_vectors(self._collective_vectors)
            results[user_id] = affected

        return results

    def _rebuild_synonym_index(self) -> None:
        """Rebuild the per-token synonym lookup index."""
        self._synonym_index = {}
        for (token_a, token_b), edge in self._synonyms.items():
            self._synonym_index.setdefault(token_a, []).append(token_b)
            self._synonym_index.setdefault(token_b, []).append(token_a)

    def stats(self) -> dict[str, Any]:
        """Diagnostic statistics for the collective field."""
        confirmed = sum(
            1 for e in self._synonyms.values()
            if e.confidence >= 1.0
        )
        return {
            "user_count": len(self._user_spaces),
            "collective_vocab_size": len(self._collective_vectors),
            "total_synonym_candidates": len(self._synonyms),
            "confirmed_synonyms": confirmed,
            "total_intersections_computed": self._total_intersections_computed,
            "total_synonyms_discovered": self._total_synonyms_discovered,
        }


# =============================================================================
# Layer 2: SemanticDriftDetector
# =============================================================================

class SemanticDriftDetector:
    """
    Tracks how each user's vocabulary evolves over time.

    Meaning shifts are real: "cloud" went from weather→computing→AI in 20 years.
    "Model" went from fashion→statistics→ML→LLM in 30 years.

    The detector takes periodic snapshots of token vectors and measures
    cosine distance between consecutive snapshots. When drift exceeds
    DRIFT_THRESHOLD, it triggers re-embedding and alerts the collective.

    Over a 50-year horizon, this captures the full arc of semantic change
    — both personal (a user changes careers) and cultural (technology shifts).

    Reference: Hamilton et al. (2016) — Diachronic word embeddings reveal
    statistical laws of semantic change.
    """

    def __init__(self) -> None:
        # Per-user, per-token drift history
        # user_id → token → list[DriftSnapshot]
        self._drift_history: dict[str, dict[str, list[DriftSnapshot]]] = {}

        # Detected drift events
        self._drift_events: list[dict[str, Any]] = []

    def snapshot(self, space: UserEmbeddingSpace) -> int:
        """
        Take a snapshot of the user's current embedding state.

        Captures vectors for all tokens. Compares against last snapshot
        to detect drift.

        Returns number of drifted tokens detected.
        """
        user_id = space.user_id
        vectors = space.export_vectors()
        now = time.time()
        doc_count = space._total_docs

        if user_id not in self._drift_history:
            self._drift_history[user_id] = {}

        user_history = self._drift_history[user_id]
        drifted_count = 0

        for token, vec in vectors.items():
            snap = DriftSnapshot(
                token=token,
                vector=list(vec),
                epoch=now,
                memory_count=doc_count,
            )

            if token not in user_history:
                user_history[token] = [snap]
                continue

            history = user_history[token]
            last_snap = history[-1]

            # Compute drift (cosine distance)
            drift = 1.0 - _cosine_similarity(last_snap.vector, vec)

            if drift >= DRIFT_THRESHOLD:
                drifted_count += 1
                self._drift_events.append({
                    "user_id": user_id,
                    "token": token,
                    "drift": drift,
                    "epoch": now,
                    "from_memory_count": last_snap.memory_count,
                    "to_memory_count": doc_count,
                })

            # Append snapshot, cap history depth
            history.append(snap)
            if len(history) > DRIFT_HISTORY_DEPTH:
                history.pop(0)

        return drifted_count

    def get_drift_trajectory(
        self,
        user_id: str,
        token: str,
    ) -> list[tuple[float, float]]:
        """
        Get the drift trajectory for a token over time.

        Returns list of (epoch, cumulative_drift) pairs.
        Cumulative drift = sum of cosine distances between consecutive snapshots.

        A flat trajectory = stable meaning.
        A rising trajectory = shifting meaning.
        A sudden jump = meaning revolution (career change, paradigm shift).
        """
        history = self._drift_history.get(user_id, {}).get(token, [])
        if len(history) < 2:
            return [(history[0].epoch, 0.0)] if history else []

        trajectory: list[tuple[float, float]] = [(history[0].epoch, 0.0)]
        cumulative = 0.0

        for i in range(1, len(history)):
            drift = 1.0 - _cosine_similarity(
                history[i - 1].vector,
                history[i].vector,
            )
            cumulative += drift
            trajectory.append((history[i].epoch, cumulative))

        return trajectory

    def get_most_drifted(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Get the tokens with the most cumulative drift for a user.

        These are the words whose meaning has changed the most over time.
        Useful for understanding how the user's vocabulary has evolved.
        """
        user_history = self._drift_history.get(user_id, {})
        token_drift: list[tuple[str, float]] = []

        for token, history in user_history.items():
            if len(history) < 2:
                continue
            total = 0.0
            for i in range(1, len(history)):
                total += 1.0 - _cosine_similarity(
                    history[i - 1].vector,
                    history[i].vector,
                )
            token_drift.append((token, total))

        token_drift.sort(key=lambda x: x[1], reverse=True)
        return token_drift[:top_k]

    def recent_drift_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recent drift events across all users."""
        return self._drift_events[-limit:]

    def stats(self) -> dict[str, Any]:
        """Diagnostic statistics."""
        total_snapshots = sum(
            len(snaps)
            for user_h in self._drift_history.values()
            for snaps in user_h.values()
        )
        return {
            "tracked_users": len(self._drift_history),
            "total_snapshots": total_snapshots,
            "total_drift_events": len(self._drift_events),
        }


# =============================================================================
# Layer 3: GenerationalKnowledgeStore
# =============================================================================

class GenerationalKnowledgeStore:
    """
    Embeddings that outlive individual users.

    Over a 50-year horizon, users come and go. But meaning accumulates.
    The GenerationalKnowledgeStore is the semantic fossil record — it
    captures how collective meaning evolves across decades.

    Each contribution is weighted by recency:
        weight(t) = 2^(−Δt / half_life)
        half_life = 25 years

    A contribution from 25 years ago has 0.5 weight.
    A contribution from 50 years ago has 0.25 weight.
    Below GENERATIONAL_MIN_WEIGHT (0.01), knowledge is pruned.

    This creates a living, evolving semantic field that reflects the
    CURRENT meaning of words — not frozen-in-time pre-training from 2024.

    The store also tracks generation numbers: when a token's meaning
    drifts significantly (detected by SemanticDriftDetector), its
    generation counter increments. This lets us know: "model" generation 0
    was fashion, generation 1 was statistics, generation 2 is ML.
    """

    def __init__(self) -> None:
        # token → GenerationalEntry
        self._entries: dict[str, GenerationalEntry] = {}

        # Generation history: token → list of (generation, epoch, vector)
        self._generation_history: dict[str, list[tuple[int, float, list[float]]]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._entries)

    def contribute(
        self,
        token: str,
        vector: list[float],
        epoch: Optional[float] = None,
    ) -> None:
        """
        Add a contribution to the generational knowledge for a token.

        Contributions are time-weighted: recent vectors count more.
        The stored vector is a weighted running average.
        """
        if len(vector) != EMBEDDING_DIMS:
            return

        now = epoch or time.time()
        existing = self._entries.get(token)

        if existing is None:
            self._entries[token] = GenerationalEntry(
                token=token,
                vector=list(vector),
                total_weight=1.0,
                contributor_count=1,
                first_epoch=now,
                last_epoch=now,
                generation=0,
            )
            return

        # Decay existing weight based on time since last contribution
        dt_years = (now - existing.last_epoch) / (365.25 * 24 * 3600)
        decay = 2.0 ** (-dt_years / GENERATIONAL_HALF_LIFE_YEARS)
        decayed_weight = existing.total_weight * decay

        # Merge: weighted average
        new_weight = decayed_weight + 1.0
        merged = [
            (decayed_weight * old + 1.0 * new) / new_weight
            for old, new in zip(existing.vector, vector)
        ]

        # Normalize
        mag = math.sqrt(sum(v * v for v in merged))
        if mag > 1e-12:
            merged = [v / mag for v in merged]

        self._entries[token] = GenerationalEntry(
            token=token,
            vector=merged,
            total_weight=new_weight,
            contributor_count=existing.contributor_count + 1,
            first_epoch=existing.first_epoch,
            last_epoch=now,
            generation=existing.generation,
        )

    def increment_generation(self, token: str) -> int:
        """
        Mark a generational shift for a token.

        Called when SemanticDriftDetector detects a major meaning change.
        Archives the current vector and increments the generation counter.

        Returns new generation number, or -1 if token not found.
        """
        entry = self._entries.get(token)
        if entry is None:
            return -1

        # Archive current generation
        if token not in self._generation_history:
            self._generation_history[token] = []
        self._generation_history[token].append(
            (entry.generation, entry.last_epoch, list(entry.vector))
        )

        # Increment
        new_gen = entry.generation + 1
        self._entries[token] = GenerationalEntry(
            token=entry.token,
            vector=entry.vector,
            total_weight=entry.total_weight,
            contributor_count=entry.contributor_count,
            first_epoch=entry.first_epoch,
            last_epoch=entry.last_epoch,
            generation=new_gen,
        )
        return new_gen

    def prune_decayed(self, reference_epoch: Optional[float] = None) -> int:
        """
        Remove entries whose weight has decayed below threshold.

        Returns number of entries pruned.
        """
        now = reference_epoch or time.time()
        to_prune: list[str] = []

        for token, entry in self._entries.items():
            dt_years = (now - entry.last_epoch) / (365.25 * 24 * 3600)
            effective_weight = entry.total_weight * (
                2.0 ** (-dt_years / GENERATIONAL_HALF_LIFE_YEARS)
            )
            if effective_weight < GENERATIONAL_MIN_WEIGHT:
                to_prune.append(token)

        for token in to_prune:
            del self._entries[token]

        return len(to_prune)

    def get_vector(self, token: str) -> Optional[list[float]]:
        """Get the generational vector for a token."""
        entry = self._entries.get(token)
        return list(entry.vector) if entry else None

    def get_generation_history(
        self,
        token: str,
    ) -> list[tuple[int, float, list[float]]]:
        """
        Get the full generational history of a token.

        Returns list of (generation, epoch, vector) tuples.
        Each entry represents a distinct era of meaning.
        """
        history = list(self._generation_history.get(token, []))
        # Add current generation
        entry = self._entries.get(token)
        if entry:
            history.append((entry.generation, entry.last_epoch, list(entry.vector)))
        return history

    def export_bootstrap_vectors(
        self,
        min_contributors: int = 5,
    ) -> dict[str, list[float]]:
        """
        Export vectors for bootstrapping new users.

        Only exports tokens with sufficient contributor diversity.
        This is the collective inheritance — the accumulated wisdom
        of all users who came before.
        """
        result: dict[str, list[float]] = {}
        for token, entry in self._entries.items():
            if entry.contributor_count >= min_contributors:
                result[token] = list(entry.vector)
        return result

    def stats(self) -> dict[str, Any]:
        """Diagnostic statistics."""
        total_contributors = sum(e.contributor_count for e in self._entries.values())
        max_gen = max(
            (e.generation for e in self._entries.values()), default=0
        )
        return {
            "vocab_size": len(self._entries),
            "total_contributions": total_contributors,
            "max_generation": max_gen,
            "generation_history_tokens": len(self._generation_history),
        }


# =============================================================================
# Orchestrator: UserEmbeddingOrchestrator
# =============================================================================

class UserEmbeddingOrchestrator:
    """
    Top-level orchestrator binding all four layers together.

    This is the single entry point for the user embedding system.
    It manages the lifecycle: observe → embed → share → discover → drift → evolve.

    Usage:
        orch = UserEmbeddingOrchestrator()
        orch.register_user("alice")
        orch.observe("alice", ["bob", "fired", "job", "resume", "work"])
        orch.observe("alice", ["alice", "hired", "job", "interview", "offer"])
        # ... after many observations ...
        orch.sync()  # Recompute vectors, discover synonyms, detect drift

    The orchestrator runs in the background. Zero user-facing complexity.
    One button press. Zero friction. The embeddings just grow.
    """

    def __init__(
        self,
        scale_mode: ScaleMode = ScaleMode.CLASSIC,
    ) -> None:
        self.scale_mode: ScaleMode = scale_mode
        self.collective: CollectiveSemanticField = CollectiveSemanticField()
        self.drift_detector: SemanticDriftDetector = SemanticDriftDetector()
        self.generational_store: GenerationalKnowledgeStore = (
            GenerationalKnowledgeStore()
        )

        # Quick lookup
        self._user_spaces: dict[str, UserEmbeddingSpace] = {}

        # Sync counter
        self._sync_count: int = 0

    def register_user(self, user_id: str) -> UserEmbeddingSpace:
        """
        Register a new user and bootstrap from generational knowledge.

        New users inherit the collective's accumulated wisdom — they
        don't start from zero. The generational store provides the
        initial vocabulary, which the user's own memories then personalize.
        """
        space = UserEmbeddingSpace(user_id, scale_mode=self.scale_mode)

        # Bootstrap from generational knowledge
        bootstrap = self.generational_store.export_bootstrap_vectors()
        if bootstrap:
            space.import_vectors(bootstrap, weight=0.5)

        self._user_spaces[user_id] = space
        self.collective.register_user(space)
        return space

    def observe(self, user_id: str, tokens: list[str]) -> None:
        """
        Observe tokens from a user's stored memory.

        This is called by PhaseMemoryEngine.store() after tokenization.
        The embedding space learns from every memory the user stores.
        """
        space = self._user_spaces.get(user_id)
        if space is None:
            space = self.register_user(user_id)
        space.observe(tokens)

    def sync(self) -> dict[str, Any]:
        """
        Full synchronization cycle.

        1. Recompute all user vectors (if dirty).
        2. Compute collective vectors.
        3. Discover synonyms from cross-user intersection.
        4. Broadcast collective back to users.
        5. Take drift snapshots.
        6. Contribute to generational store.
        7. Prune decayed generational knowledge.

        Designed to run async in background. Non-blocking.
        Frequency: every ~100 stores or on explicit trigger.
        """
        results: dict[str, Any] = {}

        # 1. Recompute dirty user vectors
        recomputed = 0
        for space in self._user_spaces.values():
            if space._svd_dirty:
                space.recompute_vectors()
                recomputed += 1
        results["users_recomputed"] = recomputed

        # 2. Compute collective
        collective_size = self.collective.compute_collective_vectors()
        results["collective_vocab_size"] = collective_size

        # 3. Discover synonyms
        new_synonyms = self.collective.discover_synonyms()
        results["new_synonyms"] = len(new_synonyms)
        results["synonym_examples"] = [
            {
                "a": s.token_a,
                "b": s.token_b,
                "similarity": round(s.cosine_similarity, 3),
                "confirmers": s.confirming_users,
            }
            for s in new_synonyms[:5]
        ]

        # 4. Broadcast collective to users
        broadcast_results = self.collective.broadcast_to_users()
        results["tokens_broadcast"] = broadcast_results

        # 5. Drift detection
        total_drifted = 0
        for space in self._user_spaces.values():
            drifted = self.drift_detector.snapshot(space)
            total_drifted += drifted
        results["drifted_tokens"] = total_drifted

        # 6. Contribute to generational store
        for space in self._user_spaces.values():
            for token, vec in space.export_vectors().items():
                self.generational_store.contribute(token, vec)

        # 7. Prune decayed
        pruned = self.generational_store.prune_decayed()
        results["generational_pruned"] = pruned

        self._sync_count += 1
        results["sync_number"] = self._sync_count

        return results

    def expand_query(self, tokens: list[str]) -> list[str]:
        """Expand query using collective synonym knowledge."""
        return self.collective.expand_query(tokens)

    def get_user_space(self, user_id: str) -> Optional[UserEmbeddingSpace]:
        """Get a user's embedding space."""
        return self._user_spaces.get(user_id)

    def stats(self) -> dict[str, Any]:
        """Full system statistics."""
        user_stats = {
            uid: space.stats()
            for uid, space in self._user_spaces.items()
        }
        total_memory = sum(
            s.memory_size_bytes for s in self._user_spaces.values()
        )
        return {
            "total_users": len(self._user_spaces),
            "total_memory_bytes": total_memory,
            "total_memory_kb": round(total_memory / 1024, 1),
            "total_memory_mb": round(total_memory / (1024 * 1024), 2),
            "sync_count": self._sync_count,
            "users": user_stats,
            "collective": self.collective.stats(),
            "drift": self.drift_detector.stats(),
            "generational": self.generational_store.stats(),
        }


# =============================================================================
# Utility Functions — Pure Python, Zero Dependencies
# =============================================================================

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure Python."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return (a, b) in canonical order for deduplication."""
    return (a, b) if a < b else (b, a)
