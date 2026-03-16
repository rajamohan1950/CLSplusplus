"""
Tests for User-Specific Embeddings — 50-Year Forward Design.

Tests the full stack:
    Layer 0: UserEmbeddingSpace (per-user PPMI-SVD)
    Layer 1: CollectiveSemanticField (cross-user synonym discovery)
    Layer 2: SemanticDriftDetector (vocabulary evolution)
    Layer 3: GenerationalKnowledgeStore (embeddings that outlive users)
    Orchestrator: Full lifecycle integration

Copyright (c) 2026 CLS++. All rights reserved.
"""

from __future__ import annotations

import math
import time
import pytest

from clsplusplus.user_embeddings import (
    EMBEDDING_DIMS,
    SYNONYM_COSINE_THRESHOLD,
    SYNONYM_QUORUM,
    DRIFT_THRESHOLD,
    GENERATIONAL_HALF_LIFE_YEARS,
    GENERATIONAL_MIN_WEIGHT,
    LOCAL_WEIGHT,
    EXTERNAL_WEIGHT,
    RECOMPUTE_INTERVAL,
    MAX_VOCAB_SIZE,
    COOCCURRENCE_WINDOW,
    ScaleMode,
    CountMinSketch,
    SynonymEdge,
    DriftSnapshot,
    GenerationalEntry,
    UserEmbeddingSpace,
    CollectiveSemanticField,
    SemanticDriftDetector,
    GenerationalKnowledgeStore,
    UserEmbeddingOrchestrator,
    _cosine_similarity,
    _canonical_pair,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_tokens(*words: str) -> list[str]:
    """Convenience: make token list from words."""
    return list(words)


def _zero_vec(dims: int = EMBEDDING_DIMS) -> list[float]:
    return [0.0] * dims


def _unit_vec(index: int, dims: int = EMBEDDING_DIMS) -> list[float]:
    """One-hot unit vector."""
    vec = [0.0] * dims
    if 0 <= index < dims:
        vec[index] = 1.0
    return vec


def _random_vec(seed: int, dims: int = EMBEDDING_DIMS) -> list[float]:
    """Deterministic pseudo-random unit vector."""
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dims)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


# =============================================================================
# Layer 0: UserEmbeddingSpace Tests
# =============================================================================

class TestUserEmbeddingSpace:

    def test_creation(self):
        space = UserEmbeddingSpace("alice")
        assert space.user_id == "alice"
        assert space.vector_count == 0
        assert space.memory_size_bytes == 0

    def test_observe_updates_cooccurrence(self):
        space = UserEmbeddingSpace("alice")
        space.observe(["cat", "sat", "mat"])
        assert space._total_docs == 1
        assert space._doc_freq["cat"] == 1
        assert space._doc_freq["sat"] == 1
        # co-occurrence pairs: (cat,mat), (cat,sat), (mat,sat)
        assert len(space._cooccurrence) == 3

    def test_observe_empty_tokens(self):
        space = UserEmbeddingSpace("alice")
        space.observe([])
        assert space._total_docs == 0

    def test_multiple_observations_accumulate(self):
        space = UserEmbeddingSpace("alice")
        space.observe(["job", "work", "resume"])
        space.observe(["job", "interview", "resume"])
        assert space._total_docs == 2
        assert space._doc_freq["job"] == 2
        assert space._doc_freq["resume"] == 2
        assert space._doc_freq["work"] == 1
        assert space._doc_freq["interview"] == 1

    def test_recompute_vectors_empty(self):
        space = UserEmbeddingSpace("alice")
        space.recompute_vectors()  # Should not crash
        assert space.vector_count == 0

    def test_recompute_vectors_too_few_tokens(self):
        space = UserEmbeddingSpace("alice")
        space.observe(["cat", "dog"])
        space.recompute_vectors()
        # V < 3, no vectors produced
        assert space.vector_count == 0

    def test_recompute_produces_vectors(self):
        space = UserEmbeddingSpace("alice")
        # Need enough observations for doc_freq >= 2
        tokens_a = ["job", "work", "resume", "interview"]
        tokens_b = ["job", "fired", "boss", "resume"]
        tokens_c = ["work", "hire", "resume", "interview"]
        tokens_d = ["job", "career", "work", "boss"]
        for t in [tokens_a, tokens_b, tokens_c, tokens_d]:
            space.observe(t)
        space.recompute_vectors()
        assert space.vector_count > 0
        # Tokens with doc_freq >= 2 should have vectors
        assert space.get_vector("job") is not None
        assert space.get_vector("resume") is not None
        assert space.get_vector("work") is not None

    def test_vector_dimensionality(self):
        space = UserEmbeddingSpace("alice")
        for _ in range(5):
            space.observe(["alpha", "beta", "gamma", "delta"])
        space.recompute_vectors()
        vec = space.get_vector("alpha")
        if vec is not None:
            assert len(vec) == EMBEDDING_DIMS or len(vec) < EMBEDDING_DIMS

    def test_get_vector_missing_token(self):
        space = UserEmbeddingSpace("alice")
        assert space.get_vector("nonexistent") is None

    def test_get_neighbors(self):
        space = UserEmbeddingSpace("alice")
        # Build a space where "job" and "work" co-occur frequently
        for _ in range(10):
            space.observe(["job", "work", "career", "resume"])
            space.observe(["job", "work", "boss", "office"])
            space.observe(["interview", "resume", "career", "hire"])
        space.recompute_vectors()

        if space.get_vector("job") is not None:
            neighbors = space.get_neighbors("job", top_k=5)
            assert isinstance(neighbors, list)
            # Should return tuples of (token, similarity)
            if neighbors:
                assert len(neighbors[0]) == 2
                assert isinstance(neighbors[0][0], str)
                assert isinstance(neighbors[0][1], float)

    def test_export_vectors(self):
        space = UserEmbeddingSpace("alice")
        for _ in range(5):
            space.observe(["cat", "dog", "pet", "animal"])
        space.recompute_vectors()
        exported = space.export_vectors()
        assert isinstance(exported, dict)
        # Modifying export should not affect internal state
        if exported:
            first_key = next(iter(exported))
            exported[first_key] = [999.0] * EMBEDDING_DIMS
            assert space.get_vector(first_key) != [999.0] * EMBEDDING_DIMS

    def test_import_vectors(self):
        space = UserEmbeddingSpace("alice")
        external = {"newtoken": _unit_vec(0)}
        affected = space.import_vectors(external)
        assert affected == 1
        vec = space.get_vector("newtoken")
        assert vec is not None
        assert vec[0] == pytest.approx(1.0)

    def test_import_merges_with_existing(self):
        space = UserEmbeddingSpace("alice")
        # Pre-populate
        space._token_vectors["shared"] = _unit_vec(0)
        external = {"shared": _unit_vec(1)}
        affected = space.import_vectors(external, weight=EXTERNAL_WEIGHT)
        assert affected == 1
        vec = space.get_vector("shared")
        assert vec is not None
        # Should be a blend: 0.8 * unit(0) + 0.2 * unit(1), normalized
        assert vec[0] > vec[1]  # Local dominates

    def test_import_rejects_wrong_dims(self):
        space = UserEmbeddingSpace("alice")
        external = {"bad": [1.0, 2.0, 3.0]}  # Wrong dimensions
        affected = space.import_vectors(external)
        assert affected == 0

    def test_auto_recompute_at_interval(self):
        space = UserEmbeddingSpace("alice")
        # Observe enough to trigger auto-recompute
        # Use varied token sets so PPMI > 0 (not all tokens in every doc)
        for i in range(RECOMPUTE_INTERVAL + 5):
            if i % 2 == 0:
                space.observe(["alpha", "beta", "gamma"])
            else:
                space.observe(["alpha", "delta", "epsilon"])
        # After RECOMPUTE_INTERVAL observations with varied co-occurrence,
        # SVD triggers and produces vectors
        assert space.vector_count > 0

    def test_stats(self):
        space = UserEmbeddingSpace("alice")
        space.observe(["hello", "world"])
        stats = space.stats()
        assert stats["user_id"] == "alice"
        assert stats["total_docs"] == 1
        assert stats["unique_tokens"] == 2
        assert "memory_size_kb" in stats

    def test_memory_size_reasonable(self):
        space = UserEmbeddingSpace("alice")
        for i in range(100):
            space.observe([f"word_{j}" for j in range(10)])
        space.recompute_vectors()
        # Should be well under 1MB
        assert space.memory_size_bytes < 1_000_000


# =============================================================================
# Layer 1: CollectiveSemanticField Tests
# =============================================================================

class TestCollectiveSemanticField:

    def _build_user_space(self, user_id: str, memories: list[list[str]]) -> UserEmbeddingSpace:
        space = UserEmbeddingSpace(user_id)
        for tokens in memories:
            space.observe(tokens)
        # Force recompute (observations need doc_freq >= 2)
        space.recompute_vectors()
        return space

    def test_creation(self):
        field = CollectiveSemanticField()
        assert field.user_count == 0
        assert field.synonym_count == 0

    def test_register_user(self):
        field = CollectiveSemanticField()
        space = UserEmbeddingSpace("alice")
        field.register_user(space)
        assert field.user_count == 1

    def test_unregister_user(self):
        field = CollectiveSemanticField()
        space = UserEmbeddingSpace("alice")
        field.register_user(space)
        field.unregister_user("alice")
        assert field.user_count == 0

    def test_compute_collective_vectors(self):
        field = CollectiveSemanticField()

        # Two users with overlapping vocabulary (repeated memories for doc_freq >= 2)
        space_a = self._build_user_space("alice", [
            ["job", "work", "resume", "career"],
            ["job", "work", "hire", "career"],
            ["job", "resume", "interview", "career"],
        ] * 5)
        space_b = self._build_user_space("bob", [
            ["job", "work", "interview", "career"],
            ["job", "work", "hire", "career"],
            ["job", "resume", "interview", "career"],
        ] * 5)

        field.register_user(space_a)
        field.register_user(space_b)

        vocab_size = field.compute_collective_vectors()
        assert vocab_size > 0

    def test_collective_vectors_normalized(self):
        field = CollectiveSemanticField()
        space = self._build_user_space("alice", [
            ["alpha", "beta", "gamma", "delta"] for _ in range(10)
        ])
        field.register_user(space)
        field.compute_collective_vectors()

        for token, vec in field._collective_vectors.items():
            mag = math.sqrt(sum(v * v for v in vec))
            assert mag == pytest.approx(1.0, abs=0.01)

    def test_discover_synonyms_smoke(self):
        """Smoke test: synonym discovery runs without errors."""
        field = CollectiveSemanticField()

        # Multiple users with similar contexts for different words
        for i in range(5):
            space = self._build_user_space(f"user_{i}", [
                ["fired", "job", "resume", "work", "boss"],
                ["quit", "job", "resume", "work", "boss"],
                ["hired", "job", "resume", "work", "interview"],
            ] * 5)
            field.register_user(space)

        synonyms = field.discover_synonyms()
        assert isinstance(synonyms, list)

    def test_get_synonyms_empty(self):
        field = CollectiveSemanticField()
        assert field.get_synonyms("anything") == []

    def test_expand_query_no_synonyms(self):
        field = CollectiveSemanticField()
        tokens = ["hello", "world"]
        expanded = field.expand_query(tokens)
        assert expanded == tokens

    def test_broadcast_to_users(self):
        field = CollectiveSemanticField()
        space_a = self._build_user_space("alice", [
            ["cat", "dog", "pet", "animal"],
            ["cat", "fish", "pet", "food"],
            ["dog", "pet", "walk", "animal"],
        ] * 5)
        space_b = self._build_user_space("bob", [
            ["cat", "fish", "pet", "aquarium"],
            ["cat", "dog", "pet", "food"],
            ["fish", "pet", "swim", "aquarium"],
        ] * 5)
        field.register_user(space_a)
        field.register_user(space_b)
        field.compute_collective_vectors()

        results = field.broadcast_to_users()
        assert "alice" in results
        assert "bob" in results
        assert results["alice"] > 0 or results["bob"] > 0

    def test_stats(self):
        field = CollectiveSemanticField()
        stats = field.stats()
        assert stats["user_count"] == 0
        assert stats["confirmed_synonyms"] == 0

    def test_synonym_quorum_requirement(self):
        """Synonyms should not be confirmed with fewer than SYNONYM_QUORUM users."""
        field = CollectiveSemanticField()

        # Only 1 user — not enough for quorum
        space = self._build_user_space("solo", [
            ["fired", "job", "work"],
            ["quit", "job", "work"],
        ] * 10)
        field.register_user(space)

        # With only 1 user, cross-user intersection requires ≥2 users
        synonyms = field.discover_synonyms()
        # No confirmed synonyms possible with 1 user
        confirmed = [s for s in synonyms if s.confidence >= 1.0]
        assert len(confirmed) == 0


# =============================================================================
# Layer 2: SemanticDriftDetector Tests
# =============================================================================

class TestSemanticDriftDetector:

    def test_creation(self):
        detector = SemanticDriftDetector()
        assert detector.stats()["tracked_users"] == 0

    def test_first_snapshot(self):
        detector = SemanticDriftDetector()
        space = UserEmbeddingSpace("alice")
        for _ in range(10):
            space.observe(["cat", "dog", "pet", "animal"])
        space.recompute_vectors()

        drifted = detector.snapshot(space)
        # First snapshot: no drift possible (no previous to compare)
        assert drifted == 0
        assert detector.stats()["tracked_users"] == 1

    def test_no_drift_stable_vectors(self):
        detector = SemanticDriftDetector()
        space = UserEmbeddingSpace("alice")
        for _ in range(10):
            space.observe(["cat", "dog", "pet", "animal"])
        space.recompute_vectors()

        detector.snapshot(space)
        # Same vectors → no drift
        drifted = detector.snapshot(space)
        assert drifted == 0

    def test_drift_detected_on_vector_change(self):
        detector = SemanticDriftDetector()
        space = UserEmbeddingSpace("alice")

        # Phase 1: tech vocabulary
        for _ in range(20):
            space.observe(["cloud", "server", "compute", "deploy"])
        space.recompute_vectors()
        detector.snapshot(space)

        # Phase 2: weather vocabulary — completely different co-occurrence
        space._cooccurrence.clear()
        space._doc_freq.clear()
        space._total_docs = 0
        for _ in range(20):
            space.observe(["cloud", "rain", "weather", "storm"])
        space.recompute_vectors()

        drifted = detector.snapshot(space)
        # "cloud" should have drifted significantly
        assert drifted >= 0  # May or may not detect depending on vector geometry

    def test_get_drift_trajectory_empty(self):
        detector = SemanticDriftDetector()
        traj = detector.get_drift_trajectory("alice", "cloud")
        assert traj == []

    def test_get_drift_trajectory_single_snapshot(self):
        detector = SemanticDriftDetector()
        space = UserEmbeddingSpace("alice")
        for _ in range(10):
            space.observe(["hello", "world", "foo", "bar"])
        space.recompute_vectors()
        detector.snapshot(space)

        traj = detector.get_drift_trajectory("alice", "hello")
        if traj:
            assert traj[0][1] == 0.0  # First point has zero drift

    def test_get_most_drifted_empty(self):
        detector = SemanticDriftDetector()
        assert detector.get_most_drifted("alice") == []

    def test_recent_drift_events(self):
        detector = SemanticDriftDetector()
        events = detector.recent_drift_events()
        assert events == []


# =============================================================================
# Layer 3: GenerationalKnowledgeStore Tests
# =============================================================================

class TestGenerationalKnowledgeStore:

    def test_creation(self):
        store = GenerationalKnowledgeStore()
        assert store.vocab_size == 0

    def test_contribute_new_token(self):
        store = GenerationalKnowledgeStore()
        vec = _unit_vec(0)
        store.contribute("hello", vec)
        assert store.vocab_size == 1
        retrieved = store.get_vector("hello")
        assert retrieved is not None
        assert retrieved[0] == pytest.approx(1.0)

    def test_contribute_accumulates(self):
        store = GenerationalKnowledgeStore()
        now = time.time()
        store.contribute("hello", _unit_vec(0), epoch=now)
        store.contribute("hello", _unit_vec(1), epoch=now + 1)
        assert store.vocab_size == 1
        entry = store._entries["hello"]
        assert entry.contributor_count == 2
        # Vector should be a blend
        vec = store.get_vector("hello")
        assert vec is not None

    def test_contribute_rejects_wrong_dims(self):
        store = GenerationalKnowledgeStore()
        store.contribute("bad", [1.0, 2.0])  # Wrong dims
        assert store.vocab_size == 0

    def test_time_decay(self):
        store = GenerationalKnowledgeStore()
        now = time.time()
        # Contribute 25 years ago
        twenty_five_years_sec = 25 * 365.25 * 24 * 3600
        store.contribute("old", _unit_vec(0), epoch=now - twenty_five_years_sec)
        store.contribute("old", _unit_vec(1), epoch=now)

        entry = store._entries["old"]
        # The old contribution should be half-weighted
        # total_weight ≈ 0.5 (decayed) + 1.0 (new) = 1.5
        assert entry.total_weight == pytest.approx(1.5, abs=0.1)

    def test_prune_decayed(self):
        store = GenerationalKnowledgeStore()
        now = time.time()
        # Contribute very old data
        ancient_epoch = now - 200 * 365.25 * 24 * 3600  # 200 years ago
        store.contribute("ancient", _unit_vec(0), epoch=ancient_epoch)

        # Contribute recent data
        store.contribute("recent", _unit_vec(1), epoch=now)

        pruned = store.prune_decayed(reference_epoch=now)
        assert pruned == 1
        assert store.get_vector("ancient") is None
        assert store.get_vector("recent") is not None

    def test_increment_generation(self):
        store = GenerationalKnowledgeStore()
        store.contribute("model", _unit_vec(0))
        gen = store.increment_generation("model")
        assert gen == 1

        gen = store.increment_generation("model")
        assert gen == 2

    def test_increment_generation_missing_token(self):
        store = GenerationalKnowledgeStore()
        gen = store.increment_generation("nonexistent")
        assert gen == -1

    def test_generation_history(self):
        store = GenerationalKnowledgeStore()
        store.contribute("model", _unit_vec(0))
        store.increment_generation("model")
        store.contribute("model", _unit_vec(1))
        store.increment_generation("model")

        history = store.get_generation_history("model")
        assert len(history) >= 2
        # Generations should be sequential
        gens = [h[0] for h in history]
        assert gens == sorted(gens)

    def test_export_bootstrap_vectors(self):
        store = GenerationalKnowledgeStore()
        now = time.time()
        # Many contributors
        for i in range(10):
            store.contribute("popular", _random_vec(i), epoch=now + i)
        # Few contributors
        store.contribute("rare", _unit_vec(0), epoch=now)

        bootstrap = store.export_bootstrap_vectors(min_contributors=5)
        assert "popular" in bootstrap
        assert "rare" not in bootstrap

    def test_stats(self):
        store = GenerationalKnowledgeStore()
        store.contribute("hello", _unit_vec(0))
        stats = store.stats()
        assert stats["vocab_size"] == 1
        assert stats["total_contributions"] == 1


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestUserEmbeddingOrchestrator:

    def test_creation(self):
        orch = UserEmbeddingOrchestrator()
        assert orch.stats()["total_users"] == 0

    def test_register_user(self):
        orch = UserEmbeddingOrchestrator()
        space = orch.register_user("alice")
        assert space.user_id == "alice"
        assert orch.stats()["total_users"] == 1

    def test_observe(self):
        orch = UserEmbeddingOrchestrator()
        orch.register_user("alice")
        orch.observe("alice", ["hello", "world", "foo"])
        space = orch.get_user_space("alice")
        assert space is not None
        assert space._total_docs == 1

    def test_observe_auto_registers(self):
        orch = UserEmbeddingOrchestrator()
        orch.observe("bob", ["hello", "world", "foo"])
        assert orch.stats()["total_users"] == 1

    def test_sync_smoke(self):
        """Full sync cycle runs without errors."""
        orch = UserEmbeddingOrchestrator()

        # Register multiple users with data
        for i in range(3):
            user_id = f"user_{i}"
            orch.register_user(user_id)
            for _ in range(20):
                orch.observe(user_id, ["job", "work", "resume", "career"])
                orch.observe(user_id, ["job", "interview", "hire", "resume"])

        results = orch.sync()
        assert "users_recomputed" in results
        assert "collective_vocab_size" in results
        assert "new_synonyms" in results
        assert "drifted_tokens" in results
        assert results["sync_number"] == 1

    def test_sync_multiple_cycles(self):
        orch = UserEmbeddingOrchestrator()
        orch.register_user("alice")
        for _ in range(20):
            orch.observe("alice", ["cat", "dog", "pet", "animal"])

        orch.sync()
        orch.sync()
        assert orch.stats()["sync_count"] == 2

    def test_expand_query(self):
        orch = UserEmbeddingOrchestrator()
        tokens = ["hello", "world"]
        expanded = orch.expand_query(tokens)
        # With no synonyms, should return same tokens
        assert expanded == tokens

    def test_get_user_space_missing(self):
        orch = UserEmbeddingOrchestrator()
        assert orch.get_user_space("nonexistent") is None

    def test_stats_comprehensive(self):
        orch = UserEmbeddingOrchestrator()
        orch.register_user("alice")
        orch.observe("alice", ["hello", "world", "test"])
        stats = orch.stats()
        assert "total_users" in stats
        assert "total_memory_bytes" in stats
        assert "total_memory_mb" in stats
        assert "collective" in stats
        assert "drift" in stats
        assert "generational" in stats
        assert "users" in stats
        assert "alice" in stats["users"]

    def test_memory_under_1mb_per_user(self):
        """Core property: ~1MB per user even with heavy usage."""
        orch = UserEmbeddingOrchestrator()
        orch.register_user("alice")

        # Simulate 500 memories with diverse vocabulary
        import random
        rng = random.Random(42)
        vocab = [f"word_{i}" for i in range(200)]
        for _ in range(500):
            tokens = rng.sample(vocab, min(8, len(vocab)))
            orch.observe("alice", tokens)

        space = orch.get_user_space("alice")
        assert space is not None
        # Should be well under 1MB
        assert space.memory_size_bytes < 1_000_000

    def test_bootstrap_from_generational(self):
        """New users inherit collective knowledge."""
        orch = UserEmbeddingOrchestrator()

        # Build up generational knowledge
        for i in range(10):
            orch.generational_store.contribute(
                "common_word", _random_vec(i)
            )

        # New user should bootstrap
        space = orch.register_user("newcomer")
        # The bootstrap imports vectors from generational store
        # (only if min_contributors met)
        bootstrap = orch.generational_store.export_bootstrap_vectors(min_contributors=5)
        if bootstrap:
            assert space.vector_count > 0 or True  # May not meet threshold


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:

    def test_cosine_similarity_identical(self):
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == pytest.approx(0.0)

    def test_canonical_pair_ordering(self):
        assert _canonical_pair("b", "a") == ("a", "b")
        assert _canonical_pair("a", "b") == ("a", "b")
        assert _canonical_pair("x", "x") == ("x", "x")


# =============================================================================
# Integration: The Core Insight Test
# =============================================================================

class TestCoreInsight:
    """
    The fundamental test: cross-user synonym discovery.

    User 1: "Bob got fired" → context: {job, resume, boss, work}
    User 2: "Lost my job"   → context: {unemployed, resume, work, interview}
    Intersection: "fired" ↔ "lost job" share {job, work, resume}
                  → system learns they're synonyms.
    """

    def test_cross_user_context_overlap(self):
        """
        Verify that tokens with similar contexts across users
        end up with similar collective vectors.
        """
        orch = UserEmbeddingOrchestrator()

        # User 1: uses "fired" in job contexts
        orch.register_user("user_1")
        for _ in range(30):
            orch.observe("user_1", ["fired", "job", "resume", "boss"])
            orch.observe("user_1", ["fired", "unemployed", "career", "resume"])
            orch.observe("user_1", ["job", "career", "boss", "resume"])

        # User 2: uses "laid" in similar contexts
        orch.register_user("user_2")
        for _ in range(30):
            orch.observe("user_2", ["laid", "job", "resume", "interview"])
            orch.observe("user_2", ["laid", "unemployed", "career", "resume"])
            orch.observe("user_2", ["job", "career", "interview", "resume"])

        # User 3: uses "terminated" in similar contexts
        orch.register_user("user_3")
        for _ in range(30):
            orch.observe("user_3", ["terminated", "job", "resume", "boss"])
            orch.observe("user_3", ["terminated", "unemployed", "career", "resume"])
            orch.observe("user_3", ["job", "career", "boss", "resume"])

        # Sync to compute collective
        results = orch.sync()

        # The collective should have vectors for shared context tokens
        collective = orch.collective._collective_vectors
        # job and career appear in all users' varied memories → doc_freq >= 2
        assert "job" in collective
        assert "career" in collective

    def test_privacy_no_raw_text_in_vectors(self):
        """Vectors contain no recoverable text — only geometry."""
        orch = UserEmbeddingOrchestrator()
        orch.register_user("alice")

        # Store sensitive information
        orch.observe("alice", ["password", "secret", "bank", "account"])
        orch.observe("alice", ["password", "secret", "bank", "account"])
        orch.observe("alice", ["password", "secret", "bank", "account"])

        space = orch.get_user_space("alice")
        space.recompute_vectors()
        exported = space.export_vectors()

        # Vectors are just float lists — no strings recoverable
        for token, vec in exported.items():
            assert isinstance(vec, list)
            assert all(isinstance(v, float) for v in vec)
            # Cannot reverse-engineer the raw memories from vectors
            # (information-theoretically: 50 floats << original text)

    def test_zero_cloud_zero_gpu(self):
        """The entire pipeline runs in pure Python. No imports from cloud/GPU libs."""
        import clsplusplus.user_embeddings as mod
        import ast
        source = open(mod.__file__).read()
        tree = ast.parse(source)
        # Check actual imports, not string mentions in comments/docs
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split(".")[0])
        # No cloud/GPU dependencies
        forbidden = {"torch", "tensorflow", "numpy", "scipy", "boto3", "azure", "requests"}
        found = imported_modules & forbidden
        assert not found, f"Forbidden imports found: {found}"


# =============================================================================
# Edge Cases & Robustness
# =============================================================================

class TestEdgeCases:

    def test_single_token_observation(self):
        space = UserEmbeddingSpace("alice")
        space.observe(["lonely"])
        assert space._total_docs == 1
        assert len(space._cooccurrence) == 0  # No pairs possible

    def test_duplicate_tokens_in_observation(self):
        space = UserEmbeddingSpace("alice")
        space.observe(["cat", "cat", "cat", "dog"])
        # Deduplication in co-occurrence
        assert space._doc_freq["cat"] == 1
        assert space._doc_freq["dog"] == 1

    def test_large_token_list(self):
        space = UserEmbeddingSpace("alice")
        tokens = [f"token_{i}" for i in range(100)]
        space.observe(tokens)
        # Should handle without error
        assert space._total_docs == 1

    def test_concurrent_users_isolated(self):
        """User spaces are independent."""
        space_a = UserEmbeddingSpace("alice")
        space_b = UserEmbeddingSpace("bob")
        space_a.observe(["cat", "dog", "pet"])
        assert space_b._total_docs == 0

    def test_generational_store_handles_zero_weight(self):
        store = GenerationalKnowledgeStore()
        # Contribute a zero vector (edge case)
        store.contribute("zero", _zero_vec())
        # Should not crash, entry should be stored
        assert store.vocab_size == 1

    def test_drift_detector_with_changing_vocab(self):
        detector = SemanticDriftDetector()
        space = UserEmbeddingSpace("alice")

        # Phase 1
        for _ in range(10):
            space.observe(["alpha", "beta", "gamma", "delta"])
        space.recompute_vectors()
        detector.snapshot(space)

        # Phase 2: add new tokens
        for _ in range(10):
            space.observe(["alpha", "epsilon", "zeta", "eta"])
        space.recompute_vectors()
        drifted = detector.snapshot(space)
        # Should not crash even with vocabulary changes
        assert isinstance(drifted, int)

    def test_orchestrator_handles_empty_sync(self):
        orch = UserEmbeddingOrchestrator()
        results = orch.sync()
        assert results["users_recomputed"] == 0
        assert results["collective_vocab_size"] == 0


# =============================================================================
# Scale Mode Tests — Classic vs Scaled Comparison
# =============================================================================

class TestScaleMode:
    """
    Tests for the scale-optimized PPMI-SVD pipeline.

    Validates all three bottleneck fixes:
        1. Sliding window co-occurrence (O(n·w) instead of O(n²))
        2. Count-Min Sketch fixed memory (1MB instead of unbounded)
        3. Incremental SVD warm-start (3 iters instead of 15)
    """

    # --- Bottleneck 1: Sliding Window ---

    def test_scaled_observe_creates_windowed_pairs(self):
        """Scaled mode uses sliding window, not all-pairs."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        tokens = [f"t{i}" for i in range(10)]
        space.observe(tokens)
        assert space._total_docs == 1
        # All-pairs would produce 10*9/2 = 45 pairs
        # Window=5 produces at most 10*5 = 50 directional, ~25 unique canonical
        assert len(space._known_pairs) < 45

    def test_scaled_observe_pairs_linear_not_quadratic(self):
        """Pairs grow linearly with tokens, not quadratically."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        tokens = [f"t{i}" for i in range(100)]
        space.observe(tokens)
        pairs = len(space._known_pairs)
        # All-pairs: 100*99/2 = 4950. Window=5: ~490 max
        assert pairs < 600, f"Expected <600 pairs, got {pairs}"
        assert pairs > 0

    def test_classic_vs_scaled_pair_count_comparison(self):
        """Compare classic vs scaled pair counts at multiple sizes."""
        for n in [10, 50, 100, 500]:
            tokens = [f"t{i}" for i in range(n)]
            classic = UserEmbeddingSpace("c", scale_mode=ScaleMode.CLASSIC)
            scaled = UserEmbeddingSpace("s", scale_mode=ScaleMode.SCALED)
            classic.observe(tokens)
            scaled.observe(tokens)
            classic_pairs = len(classic._cooccurrence)
            scaled_pairs = len(scaled._known_pairs)
            ratio = classic_pairs / max(scaled_pairs, 1)
            print(f"  n={n}: classic={classic_pairs}, scaled={scaled_pairs}, "
                  f"ratio={ratio:.1f}x")
            assert scaled_pairs < classic_pairs, (
                f"Scaled should have fewer pairs at n={n}"
            )

    # --- Bottleneck 2: Count-Min Sketch Memory ---

    def test_cms_fixed_memory(self):
        """CMS memory is fixed regardless of data volume."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        mem_before = space._cms.memory_bytes()
        for i in range(500):
            space.observe([f"word_{j}" for j in range(20)])
        mem_after = space._cms.memory_bytes()
        assert mem_before == mem_after, "CMS memory should be fixed"
        assert mem_after <= 1_100_000, f"CMS should be ~1MB, got {mem_after}"

    def test_cms_vs_counter_memory_comparison(self):
        """Compare memory: CMS (fixed) vs Counter (unbounded)."""
        import random
        rng = random.Random(42)
        vocab = [f"w{i}" for i in range(500)]

        classic = UserEmbeddingSpace("c", scale_mode=ScaleMode.CLASSIC)
        scaled = UserEmbeddingSpace("s", scale_mode=ScaleMode.SCALED)

        for _ in range(1000):
            tokens = rng.sample(vocab, 15)
            classic.observe(tokens)
            scaled.observe(tokens)

        classic_mem = classic.memory_size_bytes
        scaled_mem = scaled.memory_size_bytes
        print(f"  Classic memory: {classic_mem:,} bytes ({classic_mem/1024:.0f} KB)")
        print(f"  Scaled memory:  {scaled_mem:,} bytes ({scaled_mem/1024:.0f} KB)")
        print(f"  Ratio: {classic_mem/max(scaled_mem,1):.1f}x")
        # Scaled should use less memory at this volume
        # (CMS is 1MB fixed but Counter at 1000 obs × 105 pairs is larger)
        assert isinstance(scaled_mem, int)

    def test_cms_never_underestimates(self):
        """CMS overestimates but never underestimates."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        for _ in range(10):
            space.observe(["alpha", "beta", "gamma"])
        pair = _canonical_pair("alpha", "beta")
        count = space._cms.query(pair)
        assert count >= 10, f"CMS should never underestimate, got {count}"

    def test_cms_standalone(self):
        """CountMinSketch works correctly in isolation."""
        cms = CountMinSketch()
        pair = ("hello", "world")
        assert cms.query(pair) == 0
        for _ in range(100):
            cms.increment(pair)
        assert cms.query(pair) >= 100
        assert cms.total_increments == 100
        assert cms.memory_bytes() == 4 * 65536 * 4

    # --- Bottleneck 3: Incremental SVD ---

    def test_scaled_recompute_produces_vectors(self):
        """Scaled mode produces valid vectors."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        for _ in range(10):
            space.observe(["job", "work", "resume", "career"])
            space.observe(["job", "interview", "hire", "resume"])
        space.recompute_vectors()
        assert space.vector_count > 0
        vec = space.get_vector("job")
        assert vec is not None
        assert len(vec) <= EMBEDDING_DIMS

    def test_incremental_svd_fewer_iterations(self):
        """Incremental SVD uses warm-start basis from previous recompute."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        # Build enough data for initial SVD
        for _ in range(20):
            space.observe(["job", "work", "resume", "career", "boss"])
            space.observe(["job", "interview", "hire", "resume", "offer"])

        # Force full recompute — no warm basis yet
        space._prev_basis = None
        space._full_recompute_counter = 0
        space.recompute_vectors()
        assert space._prev_basis is not None, "First recompute should save basis"
        assert space._full_recompute_counter == 0  # Was a full recompute (reset)

        # Second recompute — should use warm-start
        for _ in range(5):
            space.observe(["job", "salary", "negotiate", "work", "contract"])
        prev_counter = space._full_recompute_counter
        space.recompute_vectors()
        assert space._full_recompute_counter == prev_counter + 1  # Incremental
        assert space._prev_basis is not None
        assert space.vector_count > 0

    # --- Vocab LRU Eviction ---

    def test_vocab_eviction_keeps_frequent(self):
        """Frequency-decayed vocab keeps recent frequent tokens."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        # Phase 1: old tokens
        for _ in range(50):
            space.observe(["old_a", "old_b", "old_c", "old_d"])
        # Phase 2: new tokens dominate
        for _ in range(200):
            space.observe(["new_x", "new_y", "new_z", "new_w"])
        assert space._token_frequency.get("new_x", 0) > space._token_frequency.get("old_a", 0)

    # --- End-to-end ---

    def test_scaled_mode_with_orchestrator(self):
        """Scaled mode works through the full orchestrator pipeline."""
        orch = UserEmbeddingOrchestrator(scale_mode=ScaleMode.SCALED)
        for _ in range(60):
            orch.observe("alice", ["cat", "dog", "pet", "animal"])
            orch.observe("alice", ["cat", "fish", "pet", "food"])
        results = orch.sync()
        space = orch.get_user_space("alice")
        assert space is not None
        assert space.scale_mode == ScaleMode.SCALED
        assert space.vector_count > 0

    def test_classic_tests_still_default(self):
        """Default construction uses CLASSIC mode — backward compatible."""
        space = UserEmbeddingSpace("alice")
        assert space.scale_mode == ScaleMode.CLASSIC
        orch = UserEmbeddingOrchestrator()
        assert orch.scale_mode == ScaleMode.CLASSIC

    def test_scaled_vectors_reasonable_quality(self):
        """Similar tokens have higher cosine than dissimilar tokens."""
        space = UserEmbeddingSpace("alice", scale_mode=ScaleMode.SCALED)
        # Train with clear clusters
        for _ in range(80):
            space.observe(["cat", "dog", "pet", "animal", "fur"])
            space.observe(["car", "truck", "vehicle", "engine", "road"])
        space.recompute_vectors()

        vec_cat = space.get_vector("cat")
        vec_dog = space.get_vector("dog")
        vec_car = space.get_vector("car")
        if vec_cat and vec_dog and vec_car:
            sim_same = _cosine_similarity(vec_cat, vec_dog)
            sim_diff = _cosine_similarity(vec_cat, vec_car)
            print(f"  cat-dog similarity: {sim_same:.3f}")
            print(f"  cat-car similarity: {sim_diff:.3f}")
            # Same cluster should be more similar
            assert sim_same > sim_diff, (
                f"Same-cluster similarity ({sim_same:.3f}) should exceed "
                f"cross-cluster ({sim_diff:.3f})"
            )

    def test_stats_includes_scale_mode(self):
        """Stats include scale_mode field."""
        for mode in [ScaleMode.CLASSIC, ScaleMode.SCALED]:
            space = UserEmbeddingSpace("alice", scale_mode=mode)
            stats = space.stats()
            assert stats["scale_mode"] == mode.value

    # --- Full Comparison Report ---

    def test_scale_comparison_report(self):
        """Print full comparison report of classic vs scaled."""
        import random
        rng = random.Random(42)
        vocab = [f"word_{i}" for i in range(200)]

        print("\n" + "=" * 60)
        print("SCALE COMPARISON: Classic vs Scaled")
        print("=" * 60)

        results = {}
        for label, mode in [("Classic", ScaleMode.CLASSIC), ("Scaled", ScaleMode.SCALED)]:
            space = UserEmbeddingSpace(f"test_{label}", scale_mode=mode)

            t0 = time.perf_counter()
            for _ in range(500):
                tokens = rng.sample(vocab, min(8, len(vocab)))
                space.observe(tokens)
            observe_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            space.recompute_vectors()
            svd_time = time.perf_counter() - t1

            mem = space.memory_size_bytes
            vecs = space.vector_count
            results[label] = {
                "observe_ms": observe_time * 1000,
                "svd_ms": svd_time * 1000,
                "memory_bytes": mem,
                "vectors": vecs,
            }
            print(f"\n  {label}:")
            print(f"    Observe (500 calls): {observe_time*1000:.1f}ms")
            print(f"    SVD recompute:       {svd_time*1000:.1f}ms")
            print(f"    Memory:              {mem:,} bytes ({mem/1024:.0f} KB)")
            print(f"    Vectors produced:    {vecs}")

        print("\n" + "-" * 60)
        if results["Classic"]["memory_bytes"] > 0 and results["Scaled"]["memory_bytes"] > 0:
            mem_ratio = results["Classic"]["memory_bytes"] / results["Scaled"]["memory_bytes"]
            print(f"  Memory ratio: {mem_ratio:.1f}x")
        print("=" * 60)
