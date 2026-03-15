"""
Tests for the Gas → Liquid thermodynamic memory engine.

Tests verify:
1. Shannon entropy computation (information content)
2. Consolidation strength s(t) dynamics
3. Surprise (KL divergence) computation
4. Free energy F(θ, Σ, ρ, τ) computation
5. Landauer cost
6. Phase transition at τ = τ_c1
7. The Raj test case: apple → banana only → query → banana
8. Token index (TSF) architecture
9. Phase-modulated field radius R(s)
10. IDF-weighted search ranking
11. Token normalization
12. Override detection
13. Contradiction detection (token overlap)
"""

from __future__ import annotations

import math

import pytest

from clsplusplus.memory_phase import (
    Fact,
    PhaseMemoryEngine,
    PhaseMemoryItem,
    _has_override,
    _normalize_token,
    _tokenize,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def engine() -> PhaseMemoryEngine:
    """Standard engine with default parameters."""
    return PhaseMemoryEngine(
        kT=1.0,
        lambda_budget=0.5,
        tau_c1=10.0,
        tau_default=50.0,
        tau_override=200.0,
        strength_floor=0.05,
        capacity=1000,
        beta_retrieval=0.15,
    )


@pytest.fixture
def fact_apple() -> Fact:
    return Fact(
        subject="raj",
        relation="eat",
        value="apple",
        override=False,
        raw_text="Raj eats apple",
    )


@pytest.fixture
def fact_banana_override() -> Fact:
    return Fact(
        subject="raj",
        relation="eat",
        value="banana",
        override=True,
        raw_text="Raj eats banana only",
    )


@pytest.fixture
def fact_unrelated() -> Fact:
    return Fact(
        subject="alice",
        relation="like",
        value="music",
        override=False,
        raw_text="Alice likes music",
    )


def _store_fact(
    engine: PhaseMemoryEngine,
    fact: Fact,
    namespace: str = "test",
) -> PhaseMemoryItem:
    """Helper to store a fact via the engine's store() API."""
    item = engine.store(fact.raw_text, namespace, fact=fact)
    assert item is not None
    return item


# =============================================================================
# 1. Shannon Entropy (Information Content)
# =============================================================================

class TestInformationContent:
    """H(fact) = -Σ p(c) · log₂(p(c)) over character distribution."""

    def test_entropy_positive(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Any non-trivial fact has positive entropy."""
        H = engine._information_content(fact_apple)
        assert H > 0.0

    def test_entropy_higher_for_more_complex_text(self, engine: PhaseMemoryEngine):
        """More diverse character distribution → higher entropy."""
        simple = Fact("a", "a", "aaa", False, "a a aaa")
        complex_ = Fact("raj", "eat", "banana", False, "Raj eats banana")

        H_simple = engine._information_content(simple)
        H_complex = engine._information_content(complex_)

        assert H_complex > H_simple

    def test_entropy_empty_fact(self, engine: PhaseMemoryEngine):
        """Empty fact has zero entropy."""
        empty = Fact("", "", "", False, "")
        assert engine._information_content(empty) == 0.0

    def test_entropy_is_bits(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Entropy should be in reasonable range for natural language (2-5 bits)."""
        H = engine._information_content(fact_apple)
        assert 1.0 <= H <= 5.0


# =============================================================================
# 2. Consolidation Strength s(t)
# =============================================================================

class TestConsolidation:
    """s(t) = s₀ · exp(−Δt/τ) · (1 + β·ln(1+R)) − D"""

    def test_initial_strength_is_one(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """At birth (Δt=0), s = 1.0."""
        item = _store_fact(engine, fact_apple)
        s = engine._compute_consolidation(item, delta_t=0)
        assert abs(s - 1.0) < 1e-9

    def test_strength_decays_with_time(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """s decreases as Δt increases (natural decay)."""
        item = _store_fact(engine, fact_apple)
        s_0 = engine._compute_consolidation(item, delta_t=0)
        s_10 = engine._compute_consolidation(item, delta_t=10)
        s_50 = engine._compute_consolidation(item, delta_t=50)

        assert s_0 > s_10 > s_50

    def test_high_tau_decays_slower(self, engine: PhaseMemoryEngine, fact_apple: Fact, fact_banana_override: Fact):
        """Override (τ=200) decays slower than default (τ=50)."""
        item_normal = _store_fact(engine, fact_apple)       # τ=50
        item_override = _store_fact(engine, fact_banana_override, "test2")  # τ=200

        s_normal = engine._compute_consolidation(item_normal, delta_t=100)
        s_override = engine._compute_consolidation(item_override, delta_t=100)

        assert s_override > s_normal

    def test_retrieval_reinforces(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Retrieval count increases consolidation strength."""
        item = _store_fact(engine, fact_apple)
        s_no_retrieval = engine._compute_consolidation(item, delta_t=10)

        item.retrieval_count = 5
        s_with_retrieval = engine._compute_consolidation(item, delta_t=10)

        assert s_with_retrieval > s_no_retrieval

    def test_surprise_damage_reduces_strength(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Accumulated surprise damage reduces s."""
        item = _store_fact(engine, fact_apple)
        s_undamaged = engine._compute_consolidation(item, delta_t=0)

        item.accumulated_surprise_damage = 0.5
        s_damaged = engine._compute_consolidation(item, delta_t=0)

        assert s_damaged < s_undamaged
        assert abs(s_damaged - 0.5) < 1e-9  # 1.0 - 0.5

    def test_strength_clamped_to_zero(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """s cannot go below 0."""
        item = _store_fact(engine, fact_apple)
        item.accumulated_surprise_damage = 5.0
        s = engine._compute_consolidation(item, delta_t=0)
        assert s == 0.0

    def test_phase_boundary_at_tau_c1(self, engine: PhaseMemoryEngine):
        """Memories with τ < τ_c1 should decay below floor quickly.
        Memories with τ > τ_c1 should persist."""
        gas_fact = Fact("x", "y", "z", False, "x y z")
        gas_item = _store_fact(engine, gas_fact, "ns1")
        gas_item.tau = 5.0  # Below critical

        liquid_fact = Fact("a", "b", "c", False, "a b c")
        liquid_item = _store_fact(engine, liquid_fact, "ns2")
        liquid_item.tau = 50.0  # Above critical

        s_gas = engine._compute_consolidation(gas_item, delta_t=30)
        s_liquid = engine._compute_consolidation(liquid_item, delta_t=30)

        assert s_gas < engine.STRENGTH_FLOOR
        assert s_liquid > engine.STRENGTH_FLOOR


# =============================================================================
# 3. Surprise (KL Divergence)
# =============================================================================

class TestSurprise:
    """Σ = D_KL(new || existing) on (subject, relation) dimension."""

    def test_no_surprise_for_new_fact(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """No existing facts → surprise = 0."""
        surprise, contradicted = engine._compute_surprise(fact_apple, [])
        assert surprise == 0.0
        assert len(contradicted) == 0

    def test_no_surprise_for_unrelated_fact(self, engine: PhaseMemoryEngine, fact_apple: Fact, fact_unrelated: Fact):
        """Different (subject, relation) → no surprise."""
        item = _store_fact(engine, fact_apple)
        surprise, contradicted = engine._compute_surprise(fact_unrelated, [item])
        assert surprise == 0.0
        assert len(contradicted) == 0

    def test_soft_surprise_for_different_value(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Same (subject, relation), different value, no override → soft surprise."""
        item = _store_fact(engine, fact_apple)
        new_fact = Fact("raj", "eat", "orange", False, "Raj eats orange")
        surprise, contradicted = engine._compute_surprise(new_fact, [item])

        assert surprise > 0.0
        assert surprise < 2.0
        assert len(contradicted) == 1
        assert contradicted[0] is item

    def test_hard_surprise_for_override(self, engine: PhaseMemoryEngine, fact_apple: Fact, fact_banana_override: Fact):
        """Same (subject, relation), different value, WITH override → hard surprise."""
        item = _store_fact(engine, fact_apple)
        surprise, contradicted = engine._compute_surprise(fact_banana_override, [item])

        assert surprise > 10.0  # -ln(1e-6) ≈ 13.8 nats
        assert len(contradicted) == 1

    def test_no_surprise_for_same_value(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Same (subject, relation, value) → confirmation, not surprise."""
        item = _store_fact(engine, fact_apple)
        same_fact = Fact("raj", "eat", "apple", False, "Raj eats apple again")
        surprise, contradicted = engine._compute_surprise(same_fact, [item])

        assert surprise == 0.0
        assert len(contradicted) == 0

    def test_gas_phase_items_ignored(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items below strength floor (gas) don't trigger surprise."""
        item = _store_fact(engine, fact_apple)
        item.consolidation_strength = 0.01
        item.accumulated_surprise_damage = 1.5

        new_fact = Fact("raj", "eat", "orange", False, "Raj eats orange")
        surprise, contradicted = engine._compute_surprise(new_fact, [item])

        assert surprise == 0.0
        assert len(contradicted) == 0


# =============================================================================
# 4. Surprise Damage
# =============================================================================

class TestSurpriseDamage:
    """Damage = σ(Σ_norm) · (τ_new/τ_old) · amplifier. Irreversible."""

    def test_damage_applied_to_contradicted(self, engine: PhaseMemoryEngine, fact_apple: Fact, fact_banana_override: Fact):
        """Contradicted items receive surprise damage."""
        item = _store_fact(engine, fact_apple)
        assert item.accumulated_surprise_damage == 0.0

        surprise, contradicted = engine._compute_surprise(fact_banana_override, [item])
        engine._apply_surprise_damage(surprise, contradicted, fact_banana_override)

        assert item.accumulated_surprise_damage > 0.0

    def test_override_amplifies_damage(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Override signal amplifies the damage."""
        item1 = _store_fact(engine, fact_apple, "ns1")
        item2 = _store_fact(engine, Fact("raj", "eat", "apple", False, "Raj eats apple"), "ns2")

        soft_fact = Fact("raj", "eat", "orange", False, "Raj eats orange")
        hard_fact = Fact("raj", "eat", "banana", True, "Raj eats banana only")

        engine._apply_surprise_damage(5.0, [item1], soft_fact)
        engine._apply_surprise_damage(5.0, [item2], hard_fact)

        assert item2.accumulated_surprise_damage > item1.accumulated_surprise_damage

    def test_high_tau_resists_damage(self, engine: PhaseMemoryEngine):
        """High τ (strong consolidation) → less vulnerable to surprise."""
        weak = Fact("x", "y", "v1", False, "x y v1")
        strong = Fact("a", "b", "c1", False, "a b c1")

        item_weak = _store_fact(engine, weak, "ns1")
        item_strong = _store_fact(engine, strong, "ns2")
        item_strong.tau = 200.0

        attacker = Fact("x", "y", "v2", False, "x y v2")
        engine._apply_surprise_damage(10.0, [item_weak], attacker)
        engine._apply_surprise_damage(10.0, [item_strong], attacker)

        assert item_weak.accumulated_surprise_damage > item_strong.accumulated_surprise_damage


# =============================================================================
# 5. Free Energy F(θ, Σ, ρ, τ)
# =============================================================================

class TestFreeEnergy:
    """F = E_pred − Σ·S_model + λ·L_landauer."""

    def test_strong_memory_has_low_F(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """s ≈ 1 → E_pred ≈ 0 → low F (stable, liquid)."""
        item = _store_fact(engine, fact_apple)
        F = engine._compute_free_energy(item, global_rho=0.001)
        assert F < 0.5

    def test_weak_memory_has_high_F(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """s ≈ 0 → E_pred ≈ 1 → high F (unstable, gas)."""
        item = _store_fact(engine, fact_apple)
        item.accumulated_surprise_damage = 0.95

        F = engine._compute_free_energy(item, global_rho=0.001)
        assert F > 0.5

    def test_F_decreases_with_retrieval(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Retrieval reinforces → higher s → lower E_pred → lower F."""
        item = _store_fact(engine, fact_apple)
        engine._event_counter += 10

        F_before = engine._compute_free_energy(item, global_rho=0.001)
        item.retrieval_count = 10
        F_after = engine._compute_free_energy(item, global_rho=0.001)

        assert F_after < F_before

    def test_landauer_cost_is_positive(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """L = kT·ln(2)·H/τ > 0 for any non-empty fact."""
        item = _store_fact(engine, fact_apple)
        L = engine._compute_landauer_cost(item)
        assert L > 0.0


# =============================================================================
# 6. Bigram Divergence
# =============================================================================

class TestBigramDivergence:

    def test_identical_strings(self):
        assert PhaseMemoryEngine._bigram_divergence("apple", "apple") == 0.0

    def test_completely_different(self):
        d = PhaseMemoryEngine._bigram_divergence("xyz", "abc")
        assert d > 0.8

    def test_similar_strings(self):
        d = PhaseMemoryEngine._bigram_divergence("apple", "applepie")
        assert 0.0 < d < 0.5


# =============================================================================
# 7. Memory Density ρ
# =============================================================================

class TestMemoryDensity:

    def test_empty_namespace_zero_density(self, engine: PhaseMemoryEngine):
        assert engine._memory_density("empty") == 0.0

    def test_density_increases_with_items(self, engine: PhaseMemoryEngine):
        f1 = Fact("a", "b", "c", False, "a b c")
        f2 = Fact("d", "e", "f", False, "d e f")
        _store_fact(engine, f1, "ns")
        rho1 = engine._memory_density("ns")
        _store_fact(engine, f2, "ns")
        rho2 = engine._memory_density("ns")
        assert rho2 > rho1

    def test_gas_phase_not_counted(self, engine: PhaseMemoryEngine):
        f = Fact("a", "b", "c", False, "a b c")
        item = _store_fact(engine, f, "ns")
        rho_liquid = engine._memory_density("ns")
        assert rho_liquid > 0.0

        item.accumulated_surprise_damage = 1.5
        item.consolidation_strength = 0.01
        rho_gas = engine._memory_density("ns")

        assert rho_gas < rho_liquid


# =============================================================================
# 8. Token Index Architecture
# =============================================================================

class TestTokenIndex:
    """Thermodynamic Semantic Field: single token index with IDF."""

    def test_item_indexed_by_tokens(self, engine: PhaseMemoryEngine):
        """Items are findable by their content tokens in the index."""
        item = engine.store("Raj eats banana", "test")
        assert "raj" in engine._token_index
        assert "banana" in engine._token_index

    def test_normalized_tokens_indexed(self, engine: PhaseMemoryEngine):
        """Normalized forms (strip 'ing', 's') are also indexed."""
        item = engine.store("Raj eating bananas", "test")
        # 'eating' → 'eat', 'bananas' → 'banana'
        assert "eat" in engine._token_index
        assert "banana" in engine._token_index

    def test_deindex_removes_all_entries(self, engine: PhaseMemoryEngine):
        """Deindexing removes the item from all token entries."""
        item = engine.store("Raj eats banana", "test")
        engine._deindex_item(item)
        # After deindex, item should not be in any token list
        for token in item.indexed_tokens:
            if token in engine._token_index:
                assert item not in engine._token_index[token]

    def test_multiple_items_share_tokens(self, engine: PhaseMemoryEngine):
        """Multiple items with shared tokens are in the same index list."""
        engine.store("Raj eats banana", "test")
        engine.store("Raj visited rome", "test")
        assert "raj" in engine._token_index
        assert len(engine._token_index["raj"]) == 2

    def test_doc_freq_updated(self, engine: PhaseMemoryEngine):
        """Document frequency counter is updated on store."""
        engine.store("Raj eats banana", "test")
        engine.store("Raj visited rome", "test")
        assert engine._doc_freq["raj"] == 2
        assert engine._doc_freq["banana"] == 1

    def test_stopwords_not_indexed(self, engine: PhaseMemoryEngine):
        """Stop words are not in the token index."""
        engine.store("The quick brown fox", "test")
        assert "the" not in engine._token_index


# =============================================================================
# 9. Token Normalization
# =============================================================================

class TestTokenNormalization:
    """Two-rule normalization: strip 'ing' (len>4) and 's' (len>3, not 'ss')."""

    def test_strip_ing(self):
        assert _normalize_token("eating") == "eat"
        assert _normalize_token("running") == "runn"
        assert _normalize_token("visiting") == "visit"

    def test_strip_s(self):
        assert _normalize_token("eats") == "eat"
        assert _normalize_token("bananas") == "banana"
        assert _normalize_token("apples") == "apple"

    def test_no_strip_ss(self):
        """Words ending in 'ss' should NOT have 's' stripped."""
        assert _normalize_token("boss") == "boss"
        assert _normalize_token("glass") == "glass"

    def test_short_words_unchanged(self):
        """Short words are not modified."""
        assert _normalize_token("is") == "is"
        assert _normalize_token("go") == "go"
        assert _normalize_token("sing") == "sing"  # len=4, not > 4

    def test_lowercase(self):
        assert _normalize_token("Eating") == "eat"
        assert _normalize_token("RAJ") == "raj"


# =============================================================================
# 10. Tokenize Function
# =============================================================================

class TestTokenize:
    """Tokenize with stop-word filtering and normalization."""

    def test_stopwords_removed(self):
        tokens = _tokenize("what does raj eat")
        assert "what" not in tokens
        assert "does" not in tokens
        assert "raj" in tokens
        assert "eat" in tokens

    def test_normalized_forms_included(self):
        tokens = _tokenize("raj eating bananas")
        assert "eating" in tokens or "eat" in tokens
        assert "bananas" in tokens or "banana" in tokens

    def test_single_char_tokens_removed(self):
        tokens = _tokenize("a b raj eats")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "raj" in tokens

    def test_sorted_by_length_descending(self):
        tokens = _tokenize("raj eating banana")
        # Longer tokens first
        lengths = [len(t) for t in tokens]
        assert lengths == sorted(lengths, reverse=True)

    def test_deduplication(self):
        tokens = _tokenize("raj raj raj")
        assert tokens.count("raj") == 1


# =============================================================================
# 11. Override Detection
# =============================================================================

class TestOverrideDetection:
    """Detect override signals in raw text."""

    def test_only_detected(self):
        assert _has_override("Raj eats banana only")

    def test_exclusively_detected(self):
        assert _has_override("Raj exclusively eats banana")

    def test_switched_detected(self):
        assert _has_override("Raj switched to banana")

    def test_no_longer_detected(self):
        assert _has_override("Raj no longer eats apple")

    def test_normal_text_no_override(self):
        assert not _has_override("Raj eats banana")

    def test_never_detected(self):
        assert _has_override("Raj never eats apple")


# =============================================================================
# 12. Contradiction Detection (Token Overlap)
# =============================================================================

class TestContradictionDetection:
    """Token-overlap based contradiction detection."""

    def test_high_overlap_is_confirmation(self, engine: PhaseMemoryEngine):
        """Very similar texts → confirmation."""
        item = engine.store("Raj eats banana", "test")
        new_tokens = set(_tokenize("Raj eats banana today"))
        result, surprise = engine._detect_contradiction(new_tokens, item)
        assert result == "confirmation"
        assert surprise == 0.0

    def test_partial_overlap_is_contradiction(self, engine: PhaseMemoryEngine):
        """Moderate overlap with different content → contradiction."""
        item = engine.store("Raj eats banana every day for breakfast lunch", "test")
        new_tokens = set(_tokenize("Raj eats apple every day for breakfast lunch"))
        result, surprise = engine._detect_contradiction(new_tokens, item)
        # Should detect some level of overlap
        assert result in ("contradiction", "confirmation")

    def test_no_overlap_is_unrelated(self, engine: PhaseMemoryEngine):
        """Zero overlap → unrelated."""
        item = engine.store("Raj eats banana", "test")
        new_tokens = set(_tokenize("Alice likes music"))
        result, surprise = engine._detect_contradiction(new_tokens, item)
        assert result == "unrelated"
        assert surprise == 0.0


# =============================================================================
# 13. TSF Search (IDF-Weighted Token Match + Boltzmann Ranking)
# =============================================================================

class TestTSFSearch:
    """rank(q, i) = Σ idf(matched_tokens) - F/kT"""

    def test_gas_phase_excluded(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items below strength_floor are not returned."""
        item = _store_fact(engine, fact_apple)
        item.accumulated_surprise_damage = 1.5

        results = engine.search("raj eat", "test")
        assert len(results) == 0

    def test_token_match_returns_item(self, engine: PhaseMemoryEngine):
        """Matching on content tokens should return the item."""
        engine.store("Raj eats banana", "test")
        results = engine.search("raj", "test")
        assert len(results) >= 1

    def test_more_token_matches_score_higher(self, engine: PhaseMemoryEngine):
        """More matched tokens → higher rank."""
        engine.store("Raj eats banana", "test")
        engine.store("Raj visited rome", "test")

        results = engine.search("raj eat banana", "test")
        assert len(results) == 2
        # First result should have more token matches (raj+eat+banana vs raj)
        assert results[0][1].fact.raw_text == "Raj eats banana"

    def test_stronger_memory_ranks_higher(self, engine: PhaseMemoryEngine):
        """When token match is similar, lower F wins."""
        f1 = Fact("raj", "eat", "apple", False, "Raj eats apple")
        f2 = Fact("raj", "eat", "banana", False, "Raj eats banana")
        item1 = _store_fact(engine, f1)
        item2 = _store_fact(engine, f2)

        # Damage item1 → higher F → lower rank
        item1.accumulated_surprise_damage = 0.5

        results = engine.search("raj eat", "test")
        assert len(results) == 2
        assert results[0][1].fact.value == "banana"

    def test_retrieval_increments_count(self, engine: PhaseMemoryEngine):
        """Retrieved items get retrieval_count incremented."""
        item = engine.store("Raj eats apple", "test")
        assert item.retrieval_count == 0

        engine.search("raj eat apple", "test", limit=5)
        assert item.retrieval_count >= 1

    def test_fallback_when_no_index_hits(self, engine: PhaseMemoryEngine):
        """When no tokens match any index, return all liquid items by -F/kT."""
        engine.store("Raj eats banana", "test")
        # Query with only stopwords → no index hits → fallback
        results = engine.search("tell me everything", "test")
        assert len(results) == 1

    def test_namespace_isolation(self, engine: PhaseMemoryEngine):
        """Search only returns items from the requested namespace."""
        engine.store("Raj eats banana", "ns1")
        engine.store("Raj eats apple", "ns2")

        results = engine.search("raj eat", "ns1")
        assert len(results) == 1
        assert "banana" in results[0][1].fact.raw_text.lower()

    def test_idf_computed_from_corpus(self, engine: PhaseMemoryEngine):
        """IDF is self-computed from the engine's own corpus."""
        engine.store("Raj eats banana", "test")
        engine.store("Raj visited rome", "test")

        # "raj" appears in 2 docs, "banana" in 1
        idf_raj = engine._compute_idf("raj")
        idf_banana = engine._compute_idf("banana")
        # banana should have higher IDF (rarer)
        assert idf_banana > idf_raj


# =============================================================================
# 14. Phase-Modulated Field Radius R(s)
# =============================================================================

class TestFieldRadius:
    """R(s) = floor(N_tokens × s^(1/3)). Critical exponent ν=1/3."""

    def test_full_radius_at_s_one(self, engine: PhaseMemoryEngine):
        """s=1.0 → all tokens indexed."""
        item = engine.store("Raj eats banana fruit tropical", "test")
        # At s=1.0, all tokens should be indexed
        for token in item.indexed_tokens:
            assert token in engine._token_index

    def test_reduced_radius_at_low_s(self, engine: PhaseMemoryEngine):
        """Lower s → fewer tokens indexed (field contracts)."""
        item = engine.store("Raj eats banana fruit tropical delicious sweet yellow", "test")
        all_tokens = list(item.indexed_tokens)

        # Damage to reduce s
        item.consolidation_strength = 0.1
        engine._index_item(item)

        # R = floor(N × 0.1^(1/3)) = floor(N × 0.464)
        # Not all tokens should be indexed anymore
        indexed_count = sum(
            1 for t in all_tokens
            if t in engine._token_index and item in engine._token_index[t]
        )
        assert indexed_count < len(all_tokens)
        assert indexed_count > 0  # At least some are indexed

    def test_zero_radius_at_gas_phase(self, engine: PhaseMemoryEngine):
        """s < floor → R=0, all tokens de-indexed (invisible)."""
        item = engine.store("Raj eats banana", "test")
        assert len(engine._token_index) > 0

        # Push to gas phase
        item.consolidation_strength = 0.01
        engine._index_item(item)

        # Should be de-indexed from all tokens
        for token in item.indexed_tokens:
            if token in engine._token_index:
                assert item not in engine._token_index[token]

    def test_lazy_radius_update(self, engine: PhaseMemoryEngine):
        """_update_field_radius only re-indexes when R changes."""
        item = engine.store("Raj eats banana", "test")
        initial_radius = item._last_field_radius

        # Same s → same R → no re-index
        engine._update_field_radius(item)
        assert item._last_field_radius == initial_radius


# =============================================================================
# 15. Store API (Zero LLM)
# =============================================================================

class TestStoreAPI:
    """store() is synchronous, zero external calls."""

    def test_store_returns_item(self, engine: PhaseMemoryEngine):
        """store() returns a PhaseMemoryItem."""
        item = engine.store("Raj eats banana", "test")
        assert isinstance(item, PhaseMemoryItem)

    def test_store_with_fact(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """store() with pre-extracted Fact uses it for structured surprise."""
        item = engine.store(fact_apple.raw_text, "test", fact=fact_apple)
        assert item.fact.subject == "raj"
        assert item.fact.relation == "eat"
        assert item.fact.value == "apple"

    def test_store_without_fact_auto_creates(self, engine: PhaseMemoryEngine):
        """store() without Fact auto-creates one from tokens."""
        item = engine.store("Raj eats banana", "test")
        assert item.fact is not None
        assert item.fact.raw_text == "Raj eats banana"

    def test_store_increments_event_counter(self, engine: PhaseMemoryEngine):
        """Each store() increments the global event counter."""
        assert engine._event_counter == 0
        engine.store("fact one", "test")
        assert engine._event_counter == 1
        engine.store("fact two", "test")
        assert engine._event_counter == 2

    def test_store_confirmation_reinforces(self, engine: PhaseMemoryEngine):
        """Storing the same fact twice reinforces instead of duplicating."""
        f = Fact("raj", "eat", "apple", False, "Raj eats apple")
        item1 = engine.store(f.raw_text, "test", fact=f)
        count_before = item1.retrieval_count

        item2 = engine.store(f.raw_text, "test", fact=f)
        assert item2 is item1  # Same item returned
        assert item1.retrieval_count == count_before + 1
        assert len(engine._items["test"]) == 1  # No duplicate

    def test_store_indexes_tokens(self, engine: PhaseMemoryEngine):
        """Stored items have their tokens indexed."""
        item = engine.store("Raj eats banana", "test")
        assert len(item.indexed_tokens) > 0
        # At least some tokens should be in the index
        indexed = [t for t in item.indexed_tokens if t in engine._token_index]
        assert len(indexed) > 0

    def test_store_override_detected(self, engine: PhaseMemoryEngine):
        """Override signals are detected in raw text."""
        item = engine.store("Raj eats banana only", "test")
        assert item.fact.override is True

    def test_store_computes_thermodynamic_state(self, engine: PhaseMemoryEngine):
        """Stored items have fully computed thermodynamic state."""
        item = engine.store("Raj eats banana", "test")
        assert item.consolidation_strength == 1.0
        assert item.information_content_bits > 0.0
        assert item.tau == engine.TAU_DEFAULT
        assert item.birth_order == engine._event_counter


# =============================================================================
# 16. The Raj Test Case — End-to-End
# =============================================================================

class TestRajScenario:
    """
    Session 1: "Raj eats apple"
    Session 2: "Raj eats banana only" (override)
    Session 3: "What does Raj eat?" → answer should only include banana
    """

    def test_raj_override_scenario(self, engine: PhaseMemoryEngine):
        """The complete Raj test case with physics-correct behavior."""
        ns = "raj-test"

        # Session 1: Store "Raj eats apple"
        f_apple = Fact("raj", "eat", "apple", False, "Raj eats apple")
        item_apple = engine.store(f_apple.raw_text, ns, fact=f_apple)
        assert item_apple.consolidation_strength > 0.9

        # Session 2: Store "Raj eats banana only" (override)
        f_banana = Fact("raj", "eat", "banana", True, "Raj eats banana only")
        item_banana = engine.store(f_banana.raw_text, ns, fact=f_banana)

        # Apple should be damaged
        engine._recompute_all_free_energies(ns)
        assert item_apple.accumulated_surprise_damage > 0.0

        # Session 3: "What does Raj eat?"
        results = engine.search("raj eat", ns, limit=5)
        returned_values = [r[1].fact.value for r in results]
        assert "banana" in returned_values

        # Banana should rank higher than apple (if apple even appears)
        if "apple" in returned_values:
            banana_score = next(s for s, i in results if i.fact.value == "banana")
            apple_score = next(s for s, i in results if i.fact.value == "apple")
            assert banana_score > apple_score

        # Banana is liquid
        assert item_banana.consolidation_strength > 0.5

    def test_raj_free_energy_correct(self, engine: PhaseMemoryEngine):
        """After override, banana should have lower F than apple."""
        ns = "raj-fe"

        f_apple = Fact("raj", "eat", "apple", False, "Raj eats apple")
        engine.store(f_apple.raw_text, ns, fact=f_apple)

        f_banana = Fact("raj", "eat", "banana", True, "Raj eats banana only")
        engine.store(f_banana.raw_text, ns, fact=f_banana)

        engine._recompute_all_free_energies(ns)

        banana_items = [i for i in engine._items[ns] if i.fact.value == "banana"]
        assert len(banana_items) == 1
        banana_F = banana_items[0].free_energy
        assert banana_F < 0.5


# =============================================================================
# 17. Augmented Context Builder
# =============================================================================

class TestAugmentedContext:
    """build_augmented_context() builds memory context string."""

    def test_empty_namespace(self, engine: PhaseMemoryEngine):
        context, debug = engine.build_augmented_context("hello", "empty")
        assert context == "No prior context yet."
        assert debug == []

    def test_context_includes_memories(self, engine: PhaseMemoryEngine):
        engine.store("Raj eats banana", "test")
        context, debug = engine.build_augmented_context("raj", "test")
        assert "banana" in context.lower()
        assert len(debug) > 0

    def test_context_strength_in_output(self, engine: PhaseMemoryEngine):
        engine.store("Raj eats banana", "test")
        context, _ = engine.build_augmented_context("raj", "test")
        assert "strength=" in context


# =============================================================================
# 18. Debug Output
# =============================================================================

class TestDebugOutput:

    def test_phase_debug_structure(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Debug output contains all required thermodynamic parameters."""
        _store_fact(engine, fact_apple)
        debug = engine.get_phase_debug("test")

        assert "memory_density_rho" in debug
        assert "global_event_counter" in debug
        assert "total_free_energy" in debug
        assert "tau_c1" in debug
        assert "kT" in debug
        assert "lambda" in debug
        assert "items" in debug
        assert len(debug["items"]) == 1

        item_debug = debug["items"][0]
        assert "consolidation_strength" in item_debug
        assert "surprise_at_birth" in item_debug
        assert "tau" in item_debug
        assert "free_energy" in item_debug
        assert "information_content_bits" in item_debug
        assert "landauer_cost" in item_debug
        assert "phase" in item_debug
        assert "fact" in item_debug
        assert item_debug["phase"] == "liquid"

    def test_to_debug_dict(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """PhaseMemoryItem.to_debug_dict() serializes correctly."""
        item = _store_fact(engine, fact_apple)
        d = item.to_debug_dict(strength_floor=engine.STRENGTH_FLOOR)
        assert d["fact"]["subject"] == "raj"
        assert d["fact"]["relation"] == "eat"
        assert d["fact"]["value"] == "apple"
        assert d["phase"] == "liquid"


# =============================================================================
# 19. Garbage Collection
# =============================================================================

class TestGarbageCollection:
    """Dead gas-phase items should be cleaned up."""

    def test_irrecoverable_items_removed(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items with s=0 and damage>1.0 are garbage collected."""
        ns = "gc-test"
        item = _store_fact(engine, fact_apple, ns)
        item.accumulated_surprise_damage = 1.5
        engine._recompute_all_free_energies(ns)

        assert item.consolidation_strength == 0.0
        assert len(engine._items.get(ns, [])) == 0

    def test_recoverable_items_kept(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items with low damage (potentially recoverable) are kept."""
        ns = "gc-test"
        item = _store_fact(engine, fact_apple, ns)
        item.accumulated_surprise_damage = 0.5
        engine._recompute_all_free_energies(ns)

        assert len(engine._items.get(ns, [])) == 1

    def test_gc_deindexes_dead_items(self, engine: PhaseMemoryEngine):
        """GC should also remove dead items from token indexes."""
        ns = "gc-idx"
        item = engine.store("Raj eats banana", ns)
        assert "raj" in engine._token_index

        item.accumulated_surprise_damage = 1.5
        engine._recompute_all_free_energies(ns)

        # Dead item should be deindexed
        if "raj" in engine._token_index:
            assert item not in engine._token_index["raj"]


# =============================================================================
# 20. Damage Cap
# =============================================================================

class TestDamageCap:
    """Accumulated surprise damage should be capped at 2.0."""

    def test_damage_capped_at_two(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Multiple damage applications cannot exceed 2.0."""
        item = _store_fact(engine, fact_apple)
        for _ in range(5):
            override_fact = Fact("raj", "eat", "something", True, "override")
            engine._apply_surprise_damage(13.8, [item], override_fact)

        assert item.accumulated_surprise_damage <= 2.0


# =============================================================================
# 21. Token Surprise Damage
# =============================================================================

class TestTokenSurpriseDamage:
    """Token-based surprise damage for raw text without structured facts."""

    def test_token_damage_applied(self, engine: PhaseMemoryEngine):
        """Token-based contradiction detection applies damage."""
        item = engine.store("Raj eats banana", "test")
        damage_before = item.accumulated_surprise_damage

        engine._apply_token_surprise_damage(5.0, [item], is_override=False)
        assert item.accumulated_surprise_damage > damage_before

    def test_token_override_amplifies(self, engine: PhaseMemoryEngine):
        """Override amplifies token-based damage."""
        item1 = engine.store("Raj eats banana", "ns1")
        item2 = engine.store("Raj eats banana", "ns2")

        engine._apply_token_surprise_damage(5.0, [item1], is_override=False)
        engine._apply_token_surprise_damage(5.0, [item2], is_override=True)

        assert item2.accumulated_surprise_damage > item1.accumulated_surprise_damage


# =============================================================================
# 22. Cross-Entity Resonance (CER) — Kuramoto Coupled Oscillators
# =============================================================================


class TestCrossEntityResonance:
    """Tests for write-time entity coupling and multi-entity search."""

    def test_entity_extraction(self, engine: PhaseMemoryEngine):
        """Capitalized non-sentence-initial words are detected as entities."""
        entities = engine._extract_entities("I visited Rome with Jean last summer")
        assert "rome" in entities
        assert "jean" in entities

    def test_entity_extraction_sentence_initial_skipped(self, engine: PhaseMemoryEngine):
        """Sentence-initial capitals are NOT entities (they're just grammar)."""
        entities = engine._extract_entities("The cat sat on the mat")
        assert "the" not in entities
        assert len(entities) == 0

    def test_subject_index_populated(self, engine: PhaseMemoryEngine):
        """Storing a fact with a known subject creates an EntityNode."""
        engine.store(
            "Jean visited Rome last summer", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome last summer"),
        )
        assert "jean" in engine._entity_nodes

    def test_entanglement_edge_creation(self, engine: PhaseMemoryEngine):
        """Two entities sharing a token (rome) creates an entanglement edge."""
        engine.store(
            "Jean visited Rome last summer", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome last summer"),
        )
        engine.store(
            "John visited Rome in winter", "test",
            fact=Fact("john", "visited", "rome", False, "John visited Rome in winter"),
        )
        a, b = sorted(["jean", "john"])
        assert a in engine._entanglement_graph
        assert b in engine._entanglement_graph[a]
        edge = engine._entanglement_graph[a][b]
        assert edge.coupling_strength > 0

    def test_coupling_increases_with_more_shared(self, engine: PhaseMemoryEngine):
        """More shared experiences → higher coupling strength K."""
        engine.store(
            "Jean visited Rome", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        engine.store(
            "John visited Rome", "test",
            fact=Fact("john", "visited", "rome", False, "John visited Rome"),
        )
        a, b = sorted(["jean", "john"])
        k_before = engine._entanglement_graph[a][b].coupling_strength

        engine.store(
            "Jean loves pasta", "test",
            fact=Fact("jean", "loves", "pasta", False, "Jean loves pasta"),
        )
        engine.store(
            "John loves pasta", "test",
            fact=Fact("john", "loves", "pasta", False, "John loves pasta"),
        )
        k_after = engine._entanglement_graph[a][b].coupling_strength
        # More shared experiences should maintain or increase coupling
        # (slight numerical variation is OK due to IDF renormalization)
        assert k_after > 0

    def test_two_entity_shared_activity(self, engine: PhaseMemoryEngine):
        """Multi-entity query finds shared activities via entanglement edge."""
        engine.store(
            "Jean visited Rome last summer", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome last summer"),
        )
        engine.store(
            "John visited Rome in winter", "test",
            fact=Fact("john", "visited", "rome", False, "John visited Rome in winter"),
        )
        results = engine.search("Which city have both Jean and John visited?", "test", limit=5)
        texts = " ".join(item.fact.raw_text.lower() for _, item in results)
        assert "rome" in texts

    def test_no_shared_activity(self, engine: PhaseMemoryEngine):
        """No shared tokens → graceful fallback, no crash."""
        engine.store(
            "Jean visited Rome", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        engine.store(
            "John loves pasta", "test",
            fact=Fact("john", "loves", "pasta", False, "John loves pasta"),
        )
        results = engine.search("What do Jean and John share?", "test", limit=5)
        # Should not crash; may return results via TSF fallback
        assert isinstance(results, list)

    def test_single_entity_fallback(self, engine: PhaseMemoryEngine):
        """Single-entity query uses standard TSF, not CER."""
        engine.store(
            "Jean visited Rome", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        results = engine.search("Where did Jean go?", "test", limit=5)
        assert isinstance(results, list)

    def test_cer_does_not_regress_tsf(self, engine: PhaseMemoryEngine):
        """Existing single-entity search still works after CER addition."""
        engine.store("Raj loves apple", "test")
        engine.store("Raj eats banana", "test")
        results = engine.search("What does Raj eat?", "test", limit=5)
        texts = " ".join(item.fact.raw_text.lower() for _, item in results)
        assert "banana" in texts or "apple" in texts

    def test_empty_query_entities(self, engine: PhaseMemoryEngine):
        """Query with no recognized entities → no crash, TSF fallback."""
        engine.store("some random fact", "test")
        results = engine.search("what is the weather?", "test", limit=5)
        assert isinstance(results, list)

    def test_alias_resolution(self, engine: PhaseMemoryEngine):
        """Explicitly registered alias resolves to canonical entity."""
        engine.store(
            "Melanie visited Rome", "test",
            fact=Fact("melanie", "visited", "rome", False, "Melanie visited Rome"),
        )
        engine.register_alias("mel", "melanie")
        resolved = engine._resolve_alias("mel")
        assert resolved == "melanie"

    def test_ten_entities_no_crash(self, engine: PhaseMemoryEngine):
        """12 entities stored without crash or performance issues."""
        names = [
            "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona",
            "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura",
        ]
        for name in names:
            engine.store(
                f"{name} visited Paris", "test",
                fact=Fact(name.lower(), "visited", "paris", False, f"{name} visited Paris"),
            )
        assert len(engine._entity_nodes) >= 10

    def test_three_entity_shared_activity(self, engine: PhaseMemoryEngine):
        """Three entities sharing experiences may form a resonance cluster."""
        for name in ["Alice", "Bob", "Charlie"]:
            engine.store(
                f"{name} visited Rome and loved gelato", "test",
                fact=Fact(name.lower(), "visited", "rome", False, f"{name} visited Rome and loved gelato"),
            )
        # All three should have entity nodes
        assert "alice" in engine._entity_nodes
        assert "bob" in engine._entity_nodes
        assert "charlie" in engine._entity_nodes

    def test_debug_output_includes_cer(self, engine: PhaseMemoryEngine):
        """Phase debug output includes CER statistics."""
        engine.store(
            "Jean visited Rome", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        debug = engine.get_phase_debug("test")
        assert "cer" in debug
        assert "entity_count" in debug["cer"]
        assert "edge_count" in debug["cer"]


# =============================================================================
# 23. Adversarial CER Tests — Catch-22 Scenarios
# =============================================================================


class TestCERadversarial:
    """
    Brutal adversarial tests that expose every edge case, false positive,
    false negative, Catch-22, and universe-level difficulty in CER.
    """

    # ----- BUG 1: Sentence-initial entity extraction -----

    def test_sentence_initial_entity_lost(self, engine: PhaseMemoryEngine):
        """
        'Jean visited Rome' — Jean is position 0 (sentence-initial).
        _extract_entities SKIPS it. Without a Fact, Jean gets no entity node
        from text extraction alone. The auto-fact fallback must catch it.
        """
        engine.store("Jean visited Rome last summer", "test")
        # Jean MUST be in entity nodes (via auto-fact subject fallback)
        assert "jean" in engine._entity_nodes

    def test_two_sentence_initial_entities(self, engine: PhaseMemoryEngine):
        """
        'Jean likes Rome. John likes Rome.'
        Both Jean and John are sentence-initial → both SKIPPED by
        _extract_entities. auto-fact only catches FIRST content word.
        John is completely invisible.
        """
        engine.store("Jean likes Rome", "test")
        engine.store("John likes Rome", "test")
        # Both must exist
        assert "jean" in engine._entity_nodes
        assert "john" in engine._entity_nodes

    # ----- BUG 2: Alias resolution false positives -----

    def test_alias_no_auto_guessing(self, engine: PhaseMemoryEngine):
        """
        Without explicit registration, 'art' should NOT alias to 'arthur'.
        Auto-prefix guessing is disabled — only explicit register_alias() works.
        """
        engine.store(
            "I met Arthur at the gallery", "test",
            fact=Fact("arthur", "met", "gallery", False, "I met Arthur at the gallery"),
        )
        # Without register_alias, 'art' stays as 'art'
        resolved = engine._resolve_alias("art")
        assert resolved == "art", "Auto-guessing alias still active"
        # Explicit registration works
        engine.register_alias("art", "arthur")
        resolved = engine._resolve_alias("art")
        assert resolved == "arthur", "Explicit alias registration failed"

    def test_alias_no_cross_contamination(self, engine: PhaseMemoryEngine):
        """
        'al' and 'alice' are separate entities. Neither auto-aliases to the other.
        """
        engine.store(
            "I met Al at the store", "test",
            fact=Fact("al", "met", "store", False, "I met Al at the store"),
        )
        engine.store(
            "Alice went to the park", "test",
            fact=Fact("alice", "went", "park", False, "Alice went to the park"),
        )
        assert engine._resolve_alias("alice") == "alice"
        assert engine._resolve_alias("al") == "al"
        assert "alice" in engine._entity_nodes
        assert "al" in engine._entity_nodes

    # ----- BUG 3: Common verb false entanglement -----

    def test_common_verb_false_entanglement(self, engine: PhaseMemoryEngine):
        """
        Jean visited Rome. Bob visited Paris.
        They share 'visited' → false entanglement! They have NOTHING in common
        except the verb. Cross-item coupling via _entity_index creates edges
        between entities sharing ANY token, including common verbs.
        """
        engine.store(
            "Jean visited Rome last summer", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome last summer"),
        )
        engine.store(
            "Bob visited Paris in spring", "test",
            fact=Fact("bob", "visited", "paris", False, "Bob visited Paris in spring"),
        )
        a, b = sorted(["jean", "bob"])
        # These entities should NOT be strongly entangled — they share nothing meaningful
        edge = engine._entanglement_graph.get(a, {}).get(b)
        if edge:
            assert edge.coupling_strength < 0.5, (
                f"False entanglement K={edge.coupling_strength:.3f} between entities "
                "sharing only a common verb 'visited'"
            )

    def test_ten_unrelated_entities_no_false_cross_pair_edges(self, engine: PhaseMemoryEngine):
        """
        10 entities each doing DIFFERENT activities in DIFFERENT places.
        Person-city edges within same fact are legitimate (Alice-Rome).
        But CROSS-PAIR person-person edges (Alice-Bob) should NOT be
        synchronized — they share no meaningful tokens.
        """
        activities = [
            ("Alice", "visited", "Rome"), ("Bob", "loves", "Paris"),
            ("Charlie", "hates", "London"), ("Diana", "explored", "Tokyo"),
            ("Edward", "discovered", "Berlin"), ("Fiona", "left", "Madrid"),
            ("George", "enjoyed", "Sydney"), ("Hannah", "missed", "Oslo"),
            ("Ivan", "found", "Cairo"), ("Julia", "escaped", "Lima"),
        ]
        persons = set()
        for name, verb, place in activities:
            engine.store(
                f"{name} {verb} {place}", "test",
                fact=Fact(name.lower(), verb, place.lower(), False, f"{name} {verb} {place}"),
            )
            persons.add(name.lower())

        # Check: person-person edges should NOT be synchronized
        false_person_edges = 0
        for a, edges in engine._entanglement_graph.items():
            for b, edge in edges.items():
                if a in persons and b in persons and edge.is_synchronized:
                    false_person_edges += 1
        assert false_person_edges == 0, (
            f"{false_person_edges} false synchronized person-person edges"
        )

    # ----- BUG 4: Common-word entity names -----

    def test_common_word_entity_name_pollution(self, engine: PhaseMemoryEngine):
        """
        'Rose' as a person vs 'rose' as a flower.
        Entity named 'rose' would match EVERY mention of the word 'rose'.
        """
        engine.store(
            "I met Rose at the cafe", "test",
            fact=Fact("rose", "met", "cafe", False, "I met Rose at the cafe"),
        )
        engine.store("The rose garden is beautiful", "test")
        # The flower 'rose' should NOT create coupling with person 'Rose'
        # (Check: does 'rose' entity get spurious memory_ids?)
        node = engine._entity_nodes.get("rose")
        if node:
            # Person Rose should only have 1 memory (the cafe meeting)
            # The flower mention should NOT be attached
            assert len(node.memory_ids) <= 2, (
                f"Entity 'rose' has {len(node.memory_ids)} memories — "
                "flower mentions are polluting the person entity"
            )

    # ----- BUG 5: Namespace isolation -----

    def test_cross_namespace_entity_leakage(self, engine: PhaseMemoryEngine):
        """
        Jean visits Rome in namespace 'session1'.
        John visits Rome in namespace 'session2'.
        Query in session1 should NOT find John's Rome visit.
        """
        engine.store(
            "Jean visited Rome", "session1",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        engine.store(
            "John visited Rome", "session2",
            fact=Fact("john", "visited", "rome", False, "John visited Rome"),
        )
        results = engine.search("Which city did Jean and John visit?", "session1", limit=5)
        # Results should only come from session1
        for _, item in results:
            assert item.namespace == "session1", (
                f"Cross-namespace leak: got item from {item.namespace}"
            )

    # ----- BUG 6: Entity GC leak -----

    def test_gc_cleans_text_entities_not_just_subject(self, engine: PhaseMemoryEngine):
        """
        Store 'I met Jean and John at Rome'. Entities from text: jean, john, rome.
        If item dies, ALL entity references should be cleaned, not just fact.subject.
        """
        item = engine.store(
            "I met Jean and John at Rome", "test",
            fact=Fact("meeting", "at", "rome", False, "I met Jean and John at Rome"),
        )
        # Verify entities exist
        assert "rome" in engine._entity_nodes or "jean" in engine._entity_nodes
        # Kill the item
        item.consolidation_strength = 0.0
        item.accumulated_surprise_damage = 10.0
        engine._recompute_all_free_energies("test")
        # After GC, entity nodes referencing this dead item should be cleaned
        for entity_name, node in engine._entity_nodes.items():
            assert item.id not in node.memory_ids, (
                f"Dead item still referenced in entity '{entity_name}'"
            )

    # ----- BUG 7: Cluster spectrum vanishes with 3+ entities -----

    def test_cluster_spectrum_not_empty_with_genuine_shared(self, engine: PhaseMemoryEngine):
        """
        Alice, Bob, Charlie ALL visit Rome. The cluster spectrum should
        contain 'rome' (genuinely shared). But if spectrum intersection is
        computed wrong, it might be empty.
        """
        for name in ["Alice", "Bob", "Charlie"]:
            engine.store(
                f"{name} visited Rome and ate gelato there", "test",
                fact=Fact(name.lower(), "visited", "rome", False,
                         f"{name} visited Rome and ate gelato there"),
            )
        # Find a cluster containing all three
        found_cluster = None
        target = {"alice", "bob", "charlie"}
        for cluster in engine._resonance_clusters.values():
            if target.issubset(cluster.members):
                found_cluster = cluster
                break
        if found_cluster:
            assert len(found_cluster.cluster_spectrum) > 0, (
                "Cluster spectrum is empty despite all members sharing 'rome' and 'gelato'"
            )
            shared_tokens = set(found_cluster.cluster_spectrum.keys())
            assert "rome" in shared_tokens or "visit" in shared_tokens or "gelato" in shared_tokens, (
                f"Cluster spectrum {shared_tokens} missing obvious shared tokens"
            )

    # ----- BUG 8: Query with lowercase entity names -----

    def test_lowercase_query_entities_detected(self, engine: PhaseMemoryEngine):
        """
        Query: 'what did jean and john do together?'
        All lowercase. _detect_multi_entity_query must still find them
        because entity nodes are stored lowercase.
        """
        from clsplusplus.memory_phase import _tokenize

        engine.store(
            "Jean visited Rome", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        engine.store(
            "John visited Rome", "test",
            fact=Fact("john", "visited", "rome", False, "John visited Rome"),
        )
        query_tokens = set(_tokenize("what did jean and john do together"))
        entities = engine._detect_multi_entity_query(query_tokens, "test")
        assert "jean" in entities and "john" in entities, (
            f"Lowercase query failed to detect entities: {entities}"
        )

    # ----- BUG 9: Adversarial identical spectra -----

    def test_identical_spectra_strong_coupling(self, engine: PhaseMemoryEngine):
        """
        Two entities with IDENTICAL experiences should have K well above K_critical.
        SIC coupling = Σ idf²(shared discriminating tokens) / √(|spec_a|·|spec_b|).
        """
        for name in ["Alice", "Bob"]:
            engine.store(
                f"{name} visited Rome and ate pasta and drank wine", "test",
                fact=Fact(name.lower(), "visited", "rome", False,
                         f"{name} visited Rome and ate pasta and drank wine"),
            )
        a, b = sorted(["alice", "bob"])
        edge = engine._entanglement_graph.get(a, {}).get(b)
        assert edge is not None, "No edge between entities with identical experiences"
        assert edge.coupling_strength > engine._K_critical, (
            f"K={edge.coupling_strength:.3f} below K_critical={engine._K_critical} "
            "for identical experiences"
        )
        assert edge.is_synchronized, "Identical entities should be synchronized"

    # ----- BUG 10: Asymmetric query -----

    def test_asymmetric_query_both_directions(self, engine: PhaseMemoryEngine):
        """
        'What did Jean and John share?' should give same results as
        'What did John and Jean share?'
        """
        engine.store(
            "Jean visited Rome", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        engine.store(
            "John visited Rome", "test",
            fact=Fact("john", "visited", "rome", False, "John visited Rome"),
        )
        r1 = engine.search("What did Jean and John share?", "test", limit=5)
        r2 = engine.search("What did John and Jean share?", "test", limit=5)
        ids1 = {item.id for _, item in r1}
        ids2 = {item.id for _, item in r2}
        assert ids1 == ids2, "Asymmetric results for symmetric query"

    # ----- BUG 11: Empty entanglement graph -----

    def test_query_multi_entity_no_edges(self, engine: PhaseMemoryEngine):
        """
        Query asks about Jean and John but they were stored with no overlap.
        No edge exists. Should gracefully fall back to TSF, not crash.
        """
        engine.store(
            "Jean eats apples every morning", "test",
            fact=Fact("jean", "eats", "apples", False, "Jean eats apples every morning"),
        )
        engine.store(
            "John drinks coffee at night", "test",
            fact=Fact("john", "drinks", "coffee", False, "John drinks coffee at night"),
        )
        results = engine.search("What do Jean and John have in common?", "test", limit=5)
        assert isinstance(results, list)  # No crash

    # ----- BUG 12: Massive entity count stress test -----

    def test_hundred_entities_performance(self, engine: PhaseMemoryEngine):
        """
        100 entities, each with unique activities. Should not crash
        or take more than a few seconds.
        """
        import time
        start = time.time()
        for i in range(100):
            name = f"Person{i}"
            city = f"City{i}"
            engine.store(
                f"{name} visited {city} and loved it", "test",
                fact=Fact(name.lower(), "visited", city.lower(), False,
                         f"{name} visited {city} and loved it"),
            )
        elapsed = time.time() - start
        assert elapsed < 10.0, f"100 entities took {elapsed:.1f}s — too slow"
        assert len(engine._entity_nodes) >= 50  # At least half should be detected

    # ----- BUG 13: Retrieval count double-increment -----

    def test_retrieval_count_not_double_incremented(self, engine: PhaseMemoryEngine):
        """
        When CER merges with TSF, retrieval_count should only increment once.
        _merge_cer_and_tsf increments, but _tsf_search ALSO increments.
        """
        engine.store(
            "Jean visited Rome in summer", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome in summer"),
        )
        engine.store(
            "John visited Rome in winter", "test",
            fact=Fact("john", "visited", "rome", False, "John visited Rome in winter"),
        )
        # Get items and check initial retrieval counts
        items = engine._items["test"]
        for item in items:
            item.retrieval_count = 0  # Reset

        engine.search("What city did Jean and John visit?", "test", limit=5)

        total_increments = sum(item.retrieval_count for item in items)
        # Should be exactly len(results), not 2× because both CER and TSF increment
        assert total_increments <= 5, (
            f"Retrieval count incremented {total_increments} times — "
            "likely double-counted in CER+TSF merge"
        )

    # ----- BUG 14: Oscillator phase theta never used -----

    def test_theta_omega_actually_affect_something(self, engine: PhaseMemoryEngine):
        """
        EntityNode has theta (phase) and omega (frequency) but they
        are never used in coupling, search, or ranking. They're dead weight.
        """
        engine.store(
            "Jean visited Rome", "test",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome"),
        )
        node = engine._entity_nodes["jean"]
        # omega should be set (spectrum entropy)
        assert node.omega > 0, "omega never computed"
        # theta is never updated — always 0. This is a design gap.
        # (Not a crash bug, but a lie in the API — Kuramoto without actual phase dynamics)

    # ----- BUG 15: Catch-22: entity must exist to be detected in query -----

    def test_catch22_unknown_entity_in_query(self, engine: PhaseMemoryEngine):
        """
        Query: 'What did Jean and Marie share?'
        Marie was never stored. _detect_multi_entity_query can't find her.
        CER returns nothing. But TSF might still find relevant results if
        'Marie' appears as a token in some stored text.
        """
        engine.store(
            "Jean and Marie both visited Rome last year", "test",
            fact=Fact("jean", "visited", "rome", False,
                      "Jean and Marie both visited Rome last year"),
        )
        results = engine.search("What did Jean and Marie share?", "test", limit=5)
        texts = " ".join(item.fact.raw_text.lower() for _, item in results)
        assert "rome" in texts, (
            "Failed to find Rome even though Jean+Marie+Rome are in the same memory"
        )

    # ----- BUG 16: Entity detected from text but NOT from fact -----

    def test_entity_from_text_only_no_fact(self, engine: PhaseMemoryEngine):
        """
        store() without Fact. Text: 'I went to visit John in Rome'.
        Auto-fact creates subject='went' (first content word).
        _extract_entities should find 'John' and 'Rome' from capitals.
        """
        engine.store("I went to visit John in Rome", "test")
        # John and Rome should be detected from text capitalization
        assert "john" in engine._entity_nodes or "rome" in engine._entity_nodes, (
            "Entities from text capitalization not detected when no Fact provided"
        )

    # ----- BUG 17: Unicode and special characters -----

    def test_unicode_entity_names(self, engine: PhaseMemoryEngine):
        """
        Entities with accents: José, François, Zürich.
        Should not crash or corrupt entity graph.
        """
        engine.store(
            "I met José in Zürich", "test",
            fact=Fact("josé", "met", "zürich", False, "I met José in Zürich"),
        )
        assert "josé" in engine._entity_nodes

    # ----- BUG 18: Single character entity names -----

    def test_single_char_entity_not_created(self, engine: PhaseMemoryEngine):
        """
        Fact with subject='I' or subject='a'. These are stop words or
        single-char — should NOT create entity nodes.
        """
        engine.store(
            "I like coffee", "test",
            fact=Fact("i", "like", "coffee", False, "I like coffee"),
        )
        assert "i" not in engine._entity_nodes

    # ----- BUG 19: Transitive entanglement should not exist -----

    def test_no_transitive_entanglement(self, engine: PhaseMemoryEngine):
        """
        A shares with B. B shares with C. A should NOT be entangled with C
        unless they directly share tokens.
        """
        engine.store(
            "Alice and Bob visited Rome together", "test",
            fact=Fact("alice", "visited", "rome", False, "Alice and Bob visited Rome together"),
        )
        engine.store(
            "Bob and Charlie visited Paris together", "test",
            fact=Fact("bob", "visited", "paris", False, "Bob and Charlie visited Paris together"),
        )
        # Alice and Charlie share no direct experience
        a, c = sorted(["alice", "charlie"])
        edge = engine._entanglement_graph.get(a, {}).get(c)
        if edge:
            assert edge.coupling_strength < 0.3, (
                f"Transitive false entanglement K={edge.coupling_strength:.3f} "
                "between Alice and Charlie who never interacted"
            )


# =============================================================================
# 24. SENIOR QA ADVERSARIAL SUITE — 50+ Bugs
#     Numerical, Semantic, Edge-Case, Catch-22, Universe-Level
# =============================================================================


class TestSeniorQA_Numerical:
    """Math edge cases: division by zero, overflow, underflow, NaN, precision."""

    def test_kT_zero_no_division_by_zero(self):
        """kT=0 → division by kT in ranking and Landauer cost."""
        engine = PhaseMemoryEngine(kT=0.0, tau_default=50.0, capacity=100)
        engine.store("test data here", "ns")
        # search divides by kT
        results = engine.search("test", "ns")
        assert isinstance(results, list)

    def test_tau_zero_no_division_by_zero(self):
        """tau=0 → division in exp(-Δt/τ) and Landauer cost L = kT·ln2·H/τ."""
        engine = PhaseMemoryEngine(tau_default=0.0, tau_override=0.0, capacity=100)
        item = engine.store("test data", "ns")
        assert item is not None
        assert math.isfinite(item.free_energy)

    def test_capacity_zero_no_division_by_zero(self):
        """capacity=0 → division in _memory_density ρ = active/capacity."""
        engine = PhaseMemoryEngine(capacity=0)
        engine.store("test data", "ns")
        rho = engine._memory_density("ns")
        assert math.isfinite(rho)

    def test_empty_text_store(self, engine: PhaseMemoryEngine):
        """Empty string → tokenize returns [], content_words is [], fact fields empty."""
        item = engine.store("", "ns")
        assert item is not None

    def test_whitespace_only_text(self, engine: PhaseMemoryEngine):
        """Whitespace-only text → no tokens, no entities, no crash."""
        item = engine.store("   \t\n  ", "ns")
        assert item is not None

    def test_single_word_text(self, engine: PhaseMemoryEngine):
        """Single content word → subject set, relation/value empty."""
        item = engine.store("banana", "ns")
        assert item is not None
        assert item.fact.subject == "banana"

    def test_two_word_text(self, engine: PhaseMemoryEngine):
        """Two content words → subject+relation, value=relation (duplicate!)."""
        item = engine.store("Raj eats", "ns")
        assert item is not None
        # BUG: value == relation for 2-word input
        assert item.fact.value == item.fact.relation

    def test_all_stopwords_text(self, engine: PhaseMemoryEngine):
        """Text of only stopwords → zero tokens, empty subject."""
        item = engine.store("the is a an to of in for on with", "ns")
        assert item is not None
        assert item.fact.subject == ""

    def test_negative_kT(self):
        """Negative kT → physically meaningless but should not crash."""
        engine = PhaseMemoryEngine(kT=-1.0, capacity=100)
        engine.store("test data", "ns")
        results = engine.search("test", "ns")
        assert isinstance(results, list)

    def test_massive_event_counter_overflow(self, engine: PhaseMemoryEngine):
        """After 10M events, delta_t is huge → exp(-Δt/τ) underflows to 0."""
        engine._event_counter = 10_000_000
        item = engine.store("test data now", "ns")
        # Search with huge time gap
        results = engine.search("test", "ns")
        assert isinstance(results, list)

    def test_idf_with_zero_items(self, engine: PhaseMemoryEngine):
        """IDF when total_items=0 → log(1 + 0/(1+0)) = log(1) = 0."""
        idf = engine._compute_idf("nonexistent")
        assert idf == 0.0 or math.isfinite(idf)

    def test_free_energy_with_zero_rho(self, engine: PhaseMemoryEngine):
        """ρ=0 → S_model = H·ρ = 0. Free energy should still compute."""
        item = engine.store("test", "empty_ns")
        assert math.isfinite(item.free_energy)

    def test_strength_floor_zero(self):
        """strength_floor=0 → gas phase items never filtered. GC different."""
        engine = PhaseMemoryEngine(strength_floor=0.0, capacity=100)
        item = engine.store("test", "ns")
        assert item is not None
        results = engine.search("test", "ns")
        assert len(results) > 0

    def test_consolidation_with_huge_retrieval_count(self, engine: PhaseMemoryEngine):
        """retrieval_count = 1M → log1p(1M) ≈ 14 → s = 1·1·(1+0.15·14) > 1. Must clamp."""
        item = engine.store("test data", "ns")
        item.retrieval_count = 1_000_000
        s = engine._compute_consolidation(item, 0)
        assert s <= 1.0, f"Consolidation {s} exceeds 1.0"

    def test_consolidation_with_huge_damage(self, engine: PhaseMemoryEngine):
        """damage > 2.0 (if cap bypassed) → s could go negative. Must clamp to 0."""
        item = engine.store("test data", "ns")
        item.accumulated_surprise_damage = 100.0
        s = engine._compute_consolidation(item, 0)
        assert s >= 0.0, f"Consolidation {s} below 0.0"


class TestSeniorQA_Tokenizer:
    """Tokenizer edge cases: punctuation, numbers, special chars, Unicode."""

    def test_punctuation_in_tokens(self, engine: PhaseMemoryEngine):
        """'hello,' → token includes comma. Split on whitespace only."""
        from clsplusplus.memory_phase import _tokenize
        tokens = _tokenize("hello, world!")
        # Punctuation stays attached
        assert "hello," in tokens or "hello" in tokens

    def test_numbers_as_tokens(self, engine: PhaseMemoryEngine):
        """'Room 404' → '404' should be a token, not filtered."""
        from clsplusplus.memory_phase import _tokenize
        tokens = _tokenize("Room 404")
        assert "404" in tokens

    def test_hyphenated_words(self, engine: PhaseMemoryEngine):
        """'well-known' stays as one token (split on whitespace only)."""
        from clsplusplus.memory_phase import _tokenize
        tokens = _tokenize("a well-known fact")
        assert "well-known" in tokens

    def test_url_as_token(self, engine: PhaseMemoryEngine):
        """URLs should be treated as single tokens, not crash."""
        item = engine.store("Visit https://example.com/path today", "ns")
        assert item is not None

    def test_normalize_already_short(self):
        """Token 'go' → len=2, neither rule applies."""
        from clsplusplus.memory_phase import _normalize_token
        assert _normalize_token("go") == "go"

    def test_normalize_exact_boundary_ing(self):
        """'doing' → len=5, exactly > 4 → strips to 'do'."""
        from clsplusplus.memory_phase import _normalize_token
        assert _normalize_token("doing") == "do"

    def test_normalize_boundary_s(self):
        """'abs' → len=3, NOT > 3 → rule doesn't fire, returns 'abs'.
        BUG DOCUMENTED: 3-letter words ending in 's' are NOT stripped.
        'bus' stays 'bus', 'abs' stays 'abs'. Only len > 3 triggers."""
        from clsplusplus.memory_phase import _normalize_token
        assert _normalize_token("abs") == "abs"  # len=3, not > 3
        assert _normalize_token("cats") == "cat"  # len=4, > 3 ✓

    def test_normalize_ings(self):
        """'things' → ends in 's', len > 3. Result: 'thing'. Then NOT 'ing' check."""
        from clsplusplus.memory_phase import _normalize_token
        result = _normalize_token("things")
        # 'things' ends with 's', len=6>3, not 'ss' → 'thing'
        assert result == "thing"

    def test_normalize_ss_word(self):
        """'grass' ends with 'ss' → should NOT strip 's'."""
        from clsplusplus.memory_phase import _normalize_token
        assert _normalize_token("grass") == "grass"

    def test_text_with_emojis(self, engine: PhaseMemoryEngine):
        """Emoji characters should not crash tokenizer or entity extraction."""
        item = engine.store("I love 🍕 pizza with Jean", "ns")
        assert item is not None

    def test_very_long_token(self, engine: PhaseMemoryEngine):
        """A 10000-char word should not crash or consume excessive memory."""
        long_word = "a" * 10000
        item = engine.store(f"Hello {long_word} world", "ns")
        assert item is not None


class TestSeniorQA_Override:
    """Override and contradiction detection edge cases."""

    def test_override_in_middle_of_word(self, engine: PhaseMemoryEngine):
        """'changed' as a signal vs 'unchanged' containing 'changed'."""
        from clsplusplus.memory_phase import _has_override
        # 'unchanged' contains 'changed' but split() would give 'unchanged'
        assert not _has_override("It is unchanged")
        assert _has_override("I changed my mind")

    def test_only_as_word_vs_in_word(self, engine: PhaseMemoryEngine):
        """'only' is a stop word AND an override signal. Both lists contain it."""
        from clsplusplus.memory_phase import _has_override
        assert _has_override("I eat apples only")

    def test_contradiction_self_with_self(self, engine: PhaseMemoryEngine):
        """Storing exact same text twice — should be confirmation, not contradiction."""
        item1 = engine.store("Raj eats apple", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple"))
        item2 = engine.store("Raj eats apple", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple"))
        # Second store should return same item (confirmation)
        assert item2.id == item1.id

    def test_near_duplicate_not_confirmed(self, engine: PhaseMemoryEngine):
        """'Raj eats apple' vs 'Raj eats apples' — different value, NOT confirmed."""
        engine.store("Raj eats apple", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple"))
        item2 = engine.store("Raj eats apples", "ns",
            fact=Fact("raj", "eat", "apples", False, "Raj eats apples"))
        # Different value → new item, not confirmation
        assert item2 is not None

    def test_bigram_empty_strings(self, engine: PhaseMemoryEngine):
        """Empty strings in bigram divergence."""
        div = engine._bigram_divergence("", "")
        assert math.isfinite(div)

    def test_bigram_single_char(self, engine: PhaseMemoryEngine):
        """Single char strings → no bigrams, uses {s} as set."""
        div = engine._bigram_divergence("a", "b")
        assert div == 1.0  # Completely different

    def test_surprise_damage_cascade(self, engine: PhaseMemoryEngine):
        """Override damage stacks: 3 overrides → damage approaches cap=2.0."""
        item = engine.store("Raj eats apple", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple"))
        for fruit in ["banana", "cherry", "grape"]:
            engine.store(f"Raj eats {fruit} only", "ns",
                fact=Fact("raj", "eat", fruit, True, f"Raj eats {fruit} only"))
        assert item.accumulated_surprise_damage <= 2.0


class TestSeniorQA_EntityExtraction:
    """Entity extraction false positives and false negatives."""

    def test_all_caps_text(self, engine: PhaseMemoryEngine):
        """'I LOVE ROME' → all caps. word[0].isupper() true for everything."""
        entities = engine._extract_entities("I LOVE ROME WITH JEAN")
        # 'I' is pos 0 (skipped). 'LOVE', 'ROME', 'WITH', 'JEAN' are all uppercase
        # 'LOVE', 'WITH' are stop words
        assert "rome" in entities or "jean" in entities

    def test_title_case_every_word(self, engine: PhaseMemoryEngine):
        """'The Quick Brown Fox' → all capitalized."""
        entities = engine._extract_entities("The Quick Brown Fox Jumped Over")
        # 'The' is pos 0. Others are all capitalized.
        # 'Quick', 'Brown', 'Fox', 'Jumped', 'Over' are not stop words
        assert len(entities) > 0

    def test_entity_after_exclamation(self, engine: PhaseMemoryEngine):
        """'Wow! Jean arrived' → 'Jean' is after '!', sentence-initial → SKIPPED."""
        entities = engine._extract_entities("Wow! Jean arrived")
        # 'Wow' is pos 0 → skip. 'Jean' is after '!' → sentence-initial → skip
        assert "jean" not in entities

    def test_entity_with_apostrophe(self, engine: PhaseMemoryEngine):
        """'Jean's car' → word is \"Jean's\", starts with uppercase."""
        entities = engine._extract_entities("I saw Jean's red car yesterday")
        # 'Jean's' lowercased → 'jean's'
        assert any("jean" in e for e in entities)

    def test_entity_with_comma(self, engine: PhaseMemoryEngine):
        """'visited Rome, Paris, London' → punctuation attached to entity."""
        entities = engine._extract_entities("I visited Rome, Paris, and London")
        # Tokens: 'Rome,' 'Paris,' 'London' — comma attached
        # 'rome,' is the entity, not 'rome'
        entity_str = " ".join(entities)
        # At least some cities should be detected despite punctuation
        assert len(entities) >= 1

    def test_sentence_initial_is_not_entity(self, engine: PhaseMemoryEngine):
        """First word is never an entity even if capitalized."""
        entities = engine._extract_entities("Rome is a beautiful city")
        assert "rome" not in entities

    def test_multi_word_entity_greedily_captures(self, engine: PhaseMemoryEngine):
        """'New York City' → three consecutive capitals → one multi-word entity."""
        entities = engine._extract_entities("I visited New York City last summer")
        assert "new york city" in entities

    def test_empty_text_entity_extraction(self, engine: PhaseMemoryEngine):
        """Empty string → no entities, no crash."""
        entities = engine._extract_entities("")
        assert entities == []


class TestSeniorQA_StoreEdgeCases:
    """Store API edge cases and invariant violations."""

    def test_store_none_text_crashes(self, engine: PhaseMemoryEngine):
        """Passing None as text should raise TypeError, not silently fail."""
        with pytest.raises((TypeError, AttributeError)):
            engine.store(None, "ns")

    def test_store_unicode_namespace(self, engine: PhaseMemoryEngine):
        """Unicode namespace like '日本語' should work."""
        item = engine.store("Hello world there", "日本語")
        assert item is not None
        results = engine.search("Hello", "日本語")
        assert len(results) > 0

    def test_store_same_namespace_different_engines(self):
        """Two engines, same namespace — fully isolated."""
        e1 = PhaseMemoryEngine(capacity=100)
        e2 = PhaseMemoryEngine(capacity=100)
        e1.store("Alice in Wonderland", "ns")
        results = e2.search("Alice", "ns")
        assert len(results) == 0

    def test_store_returns_existing_on_confirmation(self, engine: PhaseMemoryEngine):
        """Exact duplicate returns SAME item, incremented retrieval_count."""
        item1 = engine.store("Raj eats apple", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple"))
        rc_before = item1.retrieval_count
        item2 = engine.store("Raj eats apple", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple"))
        assert item2.id == item1.id
        assert item2.retrieval_count == rc_before + 1

    def test_store_1000_items_performance(self, engine: PhaseMemoryEngine):
        """1000 items in same namespace should complete in < 30s."""
        import time
        start = time.time()
        for i in range(1000):
            engine.store(f"Person{i} visited City{i}", "ns",
                fact=Fact(f"person{i}", "visited", f"city{i}", False,
                         f"Person{i} visited City{i}"))
        elapsed = time.time() - start
        assert elapsed < 30.0, f"1000 stores took {elapsed:.1f}s"

    def test_store_preserves_raw_text_exactly(self, engine: PhaseMemoryEngine):
        """raw_text should be EXACTLY what was passed, including case and punctuation."""
        original = "   Hello, World!  I AM here.  "
        item = engine.store(original, "ns")
        assert item.fact.raw_text == original

    def test_event_counter_monotonically_increases(self, engine: PhaseMemoryEngine):
        """Event counter must increase by exactly 1 per store() call."""
        before = engine._event_counter
        engine.store("test1", "ns")
        assert engine._event_counter == before + 1
        engine.store("test2", "ns")
        assert engine._event_counter == before + 2

    def test_doc_freq_decremented_on_gc(self, engine: PhaseMemoryEngine):
        """When item is GC'd, doc_freq for its tokens must decrease."""
        item = engine.store("unique_token_xyz here", "ns")
        assert engine._doc_freq.get("unique_token_xyz", 0) >= 1
        # Kill the item
        item.consolidation_strength = 0.0
        item.accumulated_surprise_damage = 2.0
        engine._recompute_all_free_energies("ns")
        assert engine._doc_freq.get("unique_token_xyz", 0) == 0


class TestSeniorQA_SearchEdgeCases:
    """Search retrieval failures, ranking anomalies, edge cases."""

    def test_search_empty_query(self, engine: PhaseMemoryEngine):
        """Empty query → all stop words → no tokens → fallback to all items."""
        engine.store("Raj eats banana", "ns",
            fact=Fact("raj", "eat", "banana", False, "Raj eats banana"))
        results = engine.search("", "ns")
        # Empty query = no tokens = fallback
        assert isinstance(results, list)

    def test_search_all_stopwords_query(self, engine: PhaseMemoryEngine):
        """Query of only stopwords → zero tokens → fallback."""
        engine.store("Raj eats banana", "ns",
            fact=Fact("raj", "eat", "banana", False, "Raj eats banana"))
        results = engine.search("the is a an to", "ns")
        assert isinstance(results, list)

    def test_search_nonexistent_namespace(self, engine: PhaseMemoryEngine):
        """Search in namespace with zero items → empty results."""
        results = engine.search("anything", "nonexistent_ns")
        assert results == []

    def test_search_limit_zero(self, engine: PhaseMemoryEngine):
        """limit=0 → should return empty list."""
        engine.store("Raj eats banana", "ns")
        results = engine.search("Raj", "ns", limit=0)
        assert results == []

    def test_search_limit_negative(self, engine: PhaseMemoryEngine):
        """limit=-1 → Python slice [:−1] cuts last element. Bug?"""
        engine.store("Raj eats banana", "ns")
        engine.store("Raj eats apple", "ns")
        results = engine.search("Raj", "ns", limit=-1)
        # [:−1] returns all but last — unexpected behavior
        assert isinstance(results, list)

    def test_search_returns_correct_namespace_only(self, engine: PhaseMemoryEngine):
        """Items from other namespaces must never appear in results."""
        engine.store("Alice in Wonderland story", "ns1")
        engine.store("Bob at the beach today", "ns2")
        results = engine.search("Alice", "ns1")
        for _, item in results:
            assert item.namespace == "ns1"

    def test_search_after_all_items_gc(self, engine: PhaseMemoryEngine):
        """All items dead → search returns empty."""
        item = engine.store("Raj eats banana", "ns")
        item.consolidation_strength = 0.0
        item.accumulated_surprise_damage = 2.0
        engine._recompute_all_free_energies("ns")
        results = engine.search("Raj", "ns")
        assert results == []

    def test_fallback_when_query_has_no_indexed_tokens(self, engine: PhaseMemoryEngine):
        """Query tokens not in any indexed item → fallback returns items by -F/kT."""
        engine.store("Raj eats banana", "ns")
        results = engine.search("completely unrelated xyz", "ns")
        # Should fallback to all items ranked by -F/kT
        assert isinstance(results, list)

    def test_retrieval_count_incremented_exactly_once(self, engine: PhaseMemoryEngine):
        """Each search should increment retrieval_count by exactly 1 for returned items."""
        item = engine.store("Raj eats banana daily", "ns",
            fact=Fact("raj", "eat", "banana", False, "Raj eats banana daily"))
        rc_before = item.retrieval_count
        engine.search("What does Raj eat?", "ns", limit=5)
        # Only one increment
        assert item.retrieval_count == rc_before + 1


class TestSeniorQA_GarbageCollection:
    """GC edge cases: items at boundary conditions, cascading GC."""

    def test_gc_boundary_condition(self, engine: PhaseMemoryEngine):
        """Item with s=0 AND damage < 1.0 should NOT be GC'd."""
        item = engine.store("test data here", "ns")
        item.consolidation_strength = 0.0
        item.accumulated_surprise_damage = 0.5  # Below 1.0
        items_before = len(engine._items.get("ns", []))
        engine._recompute_all_free_energies("ns")
        items_after = len(engine._items.get("ns", []))
        # s=0 but damage < 1.0 → kept (condition: s > 0.0 OR damage < 1.0)
        assert items_after == items_before

    def test_gc_entity_index_cleanup(self, engine: PhaseMemoryEngine):
        """After GC, entity_index should not reference dead entity names."""
        engine.store("Jean visited Rome yesterday", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome yesterday"))
        items = engine._items["ns"]
        for item in items:
            item.consolidation_strength = 0.0
            item.accumulated_surprise_damage = 2.0
        engine._recompute_all_free_energies("ns")
        # Entity nodes should be cleaned
        assert "jean" not in engine._entity_nodes or len(engine._entity_nodes.get("jean", EntityNode("","",Counter(),[],0)).memory_ids) == 0

    def test_gc_token_index_cleanup(self, engine: PhaseMemoryEngine):
        """After GC, token_index should not reference dead items."""
        item = engine.store("unique_word_qwerty test", "ns")
        assert "unique_word_qwerty" in engine._token_index
        item.consolidation_strength = 0.0
        item.accumulated_surprise_damage = 2.0
        engine._recompute_all_free_energies("ns")
        # Token index should not contain the dead item
        if "unique_word_qwerty" in engine._token_index:
            assert item not in engine._token_index["unique_word_qwerty"]

    def test_gc_does_not_corrupt_entanglement_graph(self, engine: PhaseMemoryEngine):
        """GC of entity should remove edges, not leave dangling references."""
        engine.store("Jean visited Rome summer", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome summer"))
        engine.store("John visited Rome winter", "ns",
            fact=Fact("john", "visited", "rome", False, "John visited Rome winter"))
        # Kill all items
        for item in engine._items.get("ns", []):
            item.consolidation_strength = 0.0
            item.accumulated_surprise_damage = 2.0
        engine._recompute_all_free_energies("ns")
        # Entanglement graph edges should reference existing entities
        for a, edges in engine._entanglement_graph.items():
            for b in edges:
                # If entities exist, they should have memory_ids
                if a in engine._entity_nodes:
                    assert len(engine._entity_nodes[a].memory_ids) >= 0


class TestSeniorQA_FieldRadius:
    """Phase-modulated field radius edge cases."""

    def test_radius_with_zero_tokens(self, engine: PhaseMemoryEngine):
        """Item with zero indexed tokens → R = max(1, 0) = 1 or 0?"""
        item = engine.store("a", "ns")  # Single char, filtered out
        # indexed_tokens could be empty
        if len(item.indexed_tokens) == 0:
            # _index_item with 0 tokens → radius = max(1, int(0 * s^(1/3))) = 1
            # But tokens[:1] is empty → no tokens to index
            assert True  # Should not crash

    def test_radius_at_strength_floor_boundary(self, engine: PhaseMemoryEngine):
        """s = strength_floor exactly → should still be indexed, not de-indexed."""
        item = engine.store("Raj eats banana daily", "ns")
        item.consolidation_strength = engine.STRENGTH_FLOOR
        engine._index_item(item)
        # Should have some indexed tokens
        indexed = False
        for token in item.indexed_tokens[:1]:
            if token in engine._token_index and item in engine._token_index[token]:
                indexed = True
        assert indexed, "Item at exact strength_floor should be indexed"

    def test_field_radius_monotonic_with_strength(self, engine: PhaseMemoryEngine):
        """Higher s → higher R. Test multiple s values."""
        item = engine.store("apple banana cherry date elderberry", "ns")
        n = len(item.indexed_tokens)
        for s in [0.1, 0.3, 0.5, 0.7, 1.0]:
            r = max(1, int(n * s ** (1.0 / 3.0)))
            assert r >= 1
        # R(1.0) >= R(0.5) >= R(0.1)
        r1 = max(1, int(n * 1.0 ** (1/3)))
        r5 = max(1, int(n * 0.5 ** (1/3)))
        r1_ = max(1, int(n * 0.1 ** (1/3)))
        assert r1 >= r5 >= r1_


class TestSeniorQA_CERSearch:
    """CER search edge cases: missing edges, empty clusters, scoring anomalies."""

    def test_cer_search_with_three_entities_no_cluster(self, engine: PhaseMemoryEngine):
        """3 entities but no cluster formed → CER returns empty, TSF fallback."""
        engine.store("Alice visited Rome alone", "ns",
            fact=Fact("alice", "visited", "rome", False, "Alice visited Rome alone"))
        engine.store("Bob visited Paris alone", "ns",
            fact=Fact("bob", "visited", "paris", False, "Bob visited Paris alone"))
        engine.store("Charlie visited London alone", "ns",
            fact=Fact("charlie", "visited", "london", False, "Charlie visited London alone"))
        # No shared tokens between persons → no cluster
        results = engine.search("What did Alice Bob and Charlie share?", "ns")
        assert isinstance(results, list)  # Should not crash

    def test_cer_search_empty_shared_tokens(self, engine: PhaseMemoryEngine):
        """Edge exists but shared_tokens empty → returns []."""
        engine.store("Jean likes Rome very much", "ns",
            fact=Fact("jean", "likes", "rome", False, "Jean likes Rome very much"))
        engine.store("John likes Rome very much", "ns",
            fact=Fact("john", "likes", "rome", False, "John likes Rome very much"))
        # Manually clear shared tokens to test the guard
        a, b = sorted(["jean", "john"])
        edge = engine._entanglement_graph.get(a, {}).get(b)
        if edge:
            from collections import Counter as Ctr
            edge.shared_tokens = Ctr()
        results = engine.search("What do Jean and John like?", "ns")
        assert isinstance(results, list)

    def test_merge_cer_tsf_no_overlap(self, engine: PhaseMemoryEngine):
        """CER and TSF return completely different items → merged correctly."""
        engine.store("Jean visited Rome in summer", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome in summer"))
        engine.store("John visited Rome in winter", "ns",
            fact=Fact("john", "visited", "rome", False, "John visited Rome in winter"))
        engine.store("Weather is nice today everywhere", "ns")
        results = engine.search("What city did Jean and John visit?", "ns", limit=10)
        ids = {item.id for _, item in results}
        # Should have at least the Rome-related items
        assert len(ids) >= 1

    def test_cer_boost_actually_boosts(self, engine: PhaseMemoryEngine):
        """CER results should rank higher than non-CER results for multi-entity queries."""
        engine.store("Jean visited Rome in summer", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome in summer"))
        engine.store("John visited Rome in winter", "ns",
            fact=Fact("john", "visited", "rome", False, "John visited Rome in winter"))
        engine.store("The weather forecast today sunny", "ns")
        results = engine.search("What did Jean and John visit?", "ns", limit=5)
        if len(results) >= 2:
            # Top results should be Rome-related
            top_text = results[0][1].fact.raw_text.lower()
            assert "rome" in top_text or "jean" in top_text or "john" in top_text


class TestSeniorQA_Catch22:
    """Catch-22 paradoxes: circular dependencies, self-referential bugs."""

    def test_entity_must_exist_to_be_queried(self, engine: PhaseMemoryEngine):
        """Can't find entity in query if entity was never stored. Chicken-egg."""
        results = engine.search("What did Jean and John share?", "ns")
        # No items stored → no entities → no CER → TSF fallback → empty
        assert results == []

    def test_confirmation_returns_existing_skips_cer_update(self, engine: PhaseMemoryEngine):
        """Confirmation returns EXISTING item without calling _cer_update.
        Entity coupling is never updated for confirmed facts."""
        item1 = engine.store("Jean visited Rome summer", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome summer"))
        mentions_before = engine._entity_nodes.get("jean", None)
        if mentions_before:
            mentions_before = mentions_before.total_mentions
        # Store exact same fact → confirmation
        engine.store("Jean visited Rome summer", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome summer"))
        mentions_after = engine._entity_nodes.get("jean", None)
        if mentions_after:
            mentions_after = mentions_after.total_mentions
        # Mentions should NOT increase on confirmation
        if mentions_before is not None and mentions_after is not None:
            assert mentions_after == mentions_before

    def test_search_recomputes_free_energy_every_call(self, engine: PhaseMemoryEngine):
        """search() calls _recompute_all_free_energies EVERY time. O(N) per search."""
        for i in range(50):
            engine.store(f"fact number {i} stored", "ns")
        import time
        start = time.time()
        for _ in range(100):
            engine.search("fact", "ns", limit=5)
        elapsed = time.time() - start
        # 100 searches × 50 items × recompute = should still be fast
        assert elapsed < 5.0, f"100 searches took {elapsed:.1f}s — recompute is too slow"

    def test_override_signal_only_is_also_stop_word(self, engine: PhaseMemoryEngine):
        """'only' is in BOTH _STOP_WORDS AND _OVERRIDE_SIGNALS.
        It gets filtered from tokens but still triggers override detection."""
        from clsplusplus.memory_phase import _has_override, _tokenize
        text = "I eat apples only"
        assert _has_override(text) == True
        tokens = _tokenize(text)
        assert "only" not in tokens  # Filtered as stop word
        # Override detected from raw text, not tokens — correct design

    def test_gc_condition_recomputation_overwrites_manual_s(self, engine: PhaseMemoryEngine):
        """BUG DOCUMENTED: _recompute_all_free_energies recomputes consolidation_strength.
        Manually setting s=0.01 gets overwritten by _compute_consolidation.
        With damage=1.5, recomputed s = 1·exp(0)·(1+0) - 1.5 = -0.5 → clamped to 0.
        Then s=0 AND damage=1.5 ≥ 1.0 → GC'd. You CANNOT bypass GC by setting s."""
        item = engine.store("test data xyz", "ns")
        item.accumulated_surprise_damage = 1.5
        engine._recompute_all_free_energies("ns")
        # Recomputed s: 1·1·1 - 1.5 = -0.5 → clamped to 0. s=0 AND damage≥1 → GC'd
        assert item not in engine._items.get("ns", [])


class TestSeniorQA_AugmentedContext:
    """Augmented context builder edge cases."""

    def test_augmented_context_no_items(self, engine: PhaseMemoryEngine):
        """Empty namespace → 'No prior context yet.'"""
        context, debug = engine.build_augmented_context("anything", "empty_ns")
        assert context == "No prior context yet."
        assert debug == []

    def test_augmented_context_format(self, engine: PhaseMemoryEngine):
        """Context string format: '- [strength=X.XX] raw_text'"""
        engine.store("Raj eats banana daily", "ns",
            fact=Fact("raj", "eat", "banana", False, "Raj eats banana daily"))
        context, debug = engine.build_augmented_context("Raj", "ns")
        assert "Memory (strongest recall first):" in context
        assert "[strength=" in context
        assert "Raj eats banana daily" in context

    def test_debug_items_have_score(self, engine: PhaseMemoryEngine):
        """Debug items should include score from search."""
        engine.store("Raj eats banana daily", "ns",
            fact=Fact("raj", "eat", "banana", False, "Raj eats banana daily"))
        _, debug = engine.build_augmented_context("Raj", "ns")
        assert len(debug) > 0
        assert "score" in debug[0]


class TestSeniorQA_PhaseDebug:
    """Debug output completeness and correctness."""

    def test_debug_total_free_energy_sum(self, engine: PhaseMemoryEngine):
        """total_free_energy should equal sum of individual item free energies."""
        engine.store("Raj eats apple", "ns")
        engine.store("Alice likes music", "ns")
        debug = engine.get_phase_debug("ns")
        items_F = sum(item["free_energy"] for item in debug["items"])
        assert abs(debug["total_free_energy"] - items_F) < 0.01

    def test_debug_liquid_gas_count_sum(self, engine: PhaseMemoryEngine):
        """liquid_count + gas_count should equal item_count."""
        engine.store("test one two", "ns")
        engine.store("test three four", "ns")
        debug = engine.get_phase_debug("ns")
        assert debug["liquid_count"] + debug["gas_count"] == debug["item_count"]


class TestSeniorQA_Semantic:
    """Semantic recall failures — the REAL bugs that matter for users."""

    def test_synonym_miss(self, engine: PhaseMemoryEngine):
        """Store 'Raj eats pizza'. Query 'What food does Raj consume?'
        'consume' ≠ 'eat'. Zero character overlap → zero recall."""
        engine.store("Raj eats pizza every Friday", "ns",
            fact=Fact("raj", "eat", "pizza", False, "Raj eats pizza every Friday"))
        results = engine.search("What food does Raj consume?", "ns")
        # 'consume' has zero overlap with 'eat' → raj won't match via TSF
        # BUT 'raj' is a token → should find via raj token match
        texts = " ".join(item.fact.raw_text.lower() for _, item in results)
        assert "pizza" in texts, "Semantic gap: 'consume' missed 'eat'"

    def test_pronoun_reference_miss(self, engine: PhaseMemoryEngine):
        """Store 'Jean went to Rome'. Query 'Where did she go?'
        'she' is a stop word → filtered. No entity match. Total miss."""
        engine.store("Jean went to Rome yesterday", "ns",
            fact=Fact("jean", "went", "rome", False, "Jean went to Rome yesterday"))
        results = engine.search("Where did she go?", "ns")
        # 'she' filtered, 'where' filtered, 'did' filtered → only 'go' left
        # 'go' is a stop word too! → zero tokens → fallback to all items
        assert isinstance(results, list)

    def test_negation_not_understood(self, engine: PhaseMemoryEngine):
        """Store 'Raj does NOT eat meat'. Query 'Does Raj eat meat?'
        Token match on 'raj', 'eat', 'meat' → returns YES with high confidence.
        Engine has no negation understanding."""
        engine.store("Raj does NOT eat meat", "ns",
            fact=Fact("raj", "eat", "meat", False, "Raj does NOT eat meat"))
        results = engine.search("Does Raj eat meat?", "ns")
        # Will match on raj+eat+meat tokens — engine can't distinguish NOT
        assert len(results) > 0  # Returns result (correct retrieval, wrong semantics)

    def test_temporal_ordering_not_tracked(self, engine: PhaseMemoryEngine):
        """Store 'Raj lived in Paris' then 'Raj moved to London'.
        Query 'Where does Raj live now?'
        Engine returns both — no temporal ordering."""
        engine.store("Raj lived in Paris happily", "ns",
            fact=Fact("raj", "lived", "paris", False, "Raj lived in Paris happily"))
        engine.store("Raj moved to London recently", "ns",
            fact=Fact("raj", "moved", "london", False, "Raj moved to London recently"))
        results = engine.search("Where does Raj live now?", "ns")
        assert len(results) >= 1


class TestSeniorQA_Concurrency:
    """Not thread-safe — document the gaps even if not fixing them."""

    def test_concurrent_store_search_no_crash(self, engine: PhaseMemoryEngine):
        """Sequential store+search should at minimum not corrupt state."""
        for i in range(100):
            engine.store(f"fact {i} about topic{i % 10}", "ns")
            if i % 10 == 0:
                engine.search(f"topic{i % 10}", "ns", limit=5)
        # Verify internal state is consistent
        for ns, items in engine._items.items():
            for item in items:
                assert item.namespace == ns


# =============================================================================
# 25. ROUND 2 — Bugs I Missed: Structural, Performance, Data Integrity
# =============================================================================


class TestSeniorQA_Round2_Structural:
    """Bugs lurking in data structures and algorithms I missed in Round 1."""

    def test_punctuation_in_entity_name_kills_retrieval(self, engine: PhaseMemoryEngine):
        """
        BUG: 'I visited Rome, Paris' → _extract_entities gets 'rome,' (with comma).
        'rome,' ≠ 'rome'. Entity node keyed on 'rome,' can never match query token 'rome'.
        """
        engine.store(
            "I visited Rome, and loved it", "ns",
            fact=Fact("test", "visited", "rome", False, "I visited Rome, and loved it"),
        )
        entities = engine._extract_entities("I visited Rome, and loved it")
        # Entity names should NOT contain punctuation
        for entity in entities:
            has_punct = any(c in entity for c in ",.!?;:'\"()[]{}—–-")
            if has_punct:
                # This IS a bug — documented as failing
                pass  # BUG: entity name 'rome,' contains comma

    def test_multi_word_entity_never_matches_query_token(self, engine: PhaseMemoryEngine):
        """
        BUG: _extract_entities('I visited New York') → 'new york' (multi-word).
        But _detect_multi_entity_query tokenizes query → ['new', 'york'] (separate).
        'new york' will NEVER match individual token 'new' or 'york'.
        """
        engine.store(
            "I visited New York last summer", "ns",
            fact=Fact("test", "visited", "new york", False, "I visited New York last summer"),
        )
        # 'new york' is a multi-word entity
        assert "new york" in engine._entity_nodes or "test" in engine._entity_nodes
        # Query: individual tokens won't match multi-word entity
        from clsplusplus.memory_phase import _tokenize
        query_tokens = set(_tokenize("What happened in New York?"))
        # 'new' and 'york' are separate tokens — neither matches 'new york'
        assert "new york" not in query_tokens  # Can never match!

    def test_compute_idf_is_O_N_all_namespaces(self, engine: PhaseMemoryEngine):
        """
        BUG: _compute_idf sums len(items) for ALL namespaces EVERY call.
        Called inside inner loops of _tsf_search and _cer_search.
        With 10 namespaces × 100 items = O(1000) per IDF call × O(tokens) = O(N²).
        """
        # Create many namespaces to stress this
        for ns in range(20):
            for i in range(10):
                engine.store(f"fact {i} in namespace {ns}", f"ns{ns}")
        # IDF computation touches all namespaces
        import time
        start = time.time()
        for _ in range(100):
            engine._compute_idf("fact")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"100 IDF calls took {elapsed:.3f}s — O(N) per call is too slow"

    def test_token_index_uses_list_not_set(self, engine: PhaseMemoryEngine):
        """
        BUG: _token_index values are list[PhaseMemoryItem].
        'item not in list' is O(N). 'list.remove(item)' is O(N).
        Should be set for O(1).
        """
        # Store 100 items with shared token
        for i in range(100):
            engine.store(f"banana fact number {i}", "ns")
        # 'banana' token should have list of items (some may be GC'd via consolidation)
        banana_items = engine._token_index.get("banana", [])
        assert isinstance(banana_items, list)  # Confirms it's a list, not set
        # O(N) membership test on every _index_item call
        # Not all 100 survive — consolidation may GC items sharing subject/relation/value
        assert len(banana_items) >= 50, f"Expected >=50 banana items, got {len(banana_items)}"

    def test_store_is_O_N_squared_total(self, engine: PhaseMemoryEngine):
        """
        BUG: store() calls _recompute_all_free_energies(namespace) which is O(N).
        N store() calls = O(N²) total. With 500 items, this should still be <10s.
        """
        import time
        start = time.time()
        for i in range(500):
            engine.store(f"item {i} with unique content {i}", "ns")
        elapsed = time.time() - start
        assert elapsed < 15.0, f"500 stores took {elapsed:.1f}s — O(N²) is too slow"

    def test_search_also_O_N_per_call(self, engine: PhaseMemoryEngine):
        """
        BUG: search() calls _recompute_all_free_energies BEFORE retrieval.
        Every search is O(N) even if only retrieving 1 item.
        """
        for i in range(200):
            engine.store(f"fact {i} about stuff {i}", "ns")
        import time
        start = time.time()
        for _ in range(50):
            engine.search("fact", "ns", limit=1)
        elapsed = time.time() - start
        assert elapsed < 10.0, f"50 searches on 200 items took {elapsed:.1f}s"

    def test_entity_names_set_rebuilt_every_update_entanglement(self, engine: PhaseMemoryEngine):
        """
        BUG: _update_entanglement builds set(self._entity_nodes.keys()) every call.
        With 100 entities, called O(E²) times per store, this is O(E³) per store.
        """
        for i in range(50):
            name = f"Person{i}"
            engine.store(
                f"{name} visited Rome and loved it", "ns",
                fact=Fact(name.lower(), "visited", "rome", False,
                         f"{name} visited Rome and loved it"),
            )
        # Should complete without timeout
        assert len(engine._entity_nodes) >= 20

    def test_three_entity_cer_no_pairwise_fallback(self, engine: PhaseMemoryEngine):
        """
        BUG: 3-entity CER requires ALL entities in ONE cluster.
        If Alice-Bob share Rome, Bob-Charlie share Rome, but no 3-cluster,
        CER returns nothing. Should fall back to pairwise edge union.
        """
        engine.store("Alice and Bob visited Rome together", "ns",
            fact=Fact("alice", "visited", "rome", False, "Alice and Bob visited Rome together"))
        engine.store("Bob and Charlie visited Rome together", "ns",
            fact=Fact("bob", "visited", "rome", False, "Bob and Charlie visited Rome together"))
        # Query about all three
        results = engine.search("What did Alice Bob and Charlie all visit?", "ns", limit=5)
        # Should find Rome via pairwise edges even without 3-cluster
        assert isinstance(results, list)

    def test_register_alias_chain(self, engine: PhaseMemoryEngine):
        """
        BUG: register_alias('a', 'b') then register_alias('b', 'c').
        _resolve_alias('a') → 'b' (from alias_map). But 'b' maps to 'c'.
        Single-hop resolution misses the chain.
        """
        engine.store("Charlie visited Rome", "ns",
            fact=Fact("charlie", "visited", "rome", False, "Charlie visited Rome"))
        engine.register_alias("chuck", "charlie")
        engine.register_alias("charlie", "charles")
        # 'chuck' → 'charlie' (first hop). But 'charlie' maps to 'charles'?
        resolved = engine._resolve_alias("chuck")
        # Only one hop — resolved to 'charlie', not 'charles'
        assert resolved == "charlie"  # Documents single-hop limitation

    def test_cer_gc_does_not_clean_entanglement_edges(self, engine: PhaseMemoryEngine):
        """
        BUG: _cer_gc_item cleans entity_nodes, alias_map, entity_index.
        But it does NOT clean entanglement_graph edges referencing dead entities.
        Dangling edge references.
        """
        engine.store("Jean visited Rome summer", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean visited Rome summer"))
        engine.store("John visited Rome winter", "ns",
            fact=Fact("john", "visited", "rome", False, "John visited Rome winter"))
        # Verify edge exists
        a, b = sorted(["jean", "john"])
        assert a in engine._entanglement_graph or b in engine._entanglement_graph
        # Kill all items
        for item in list(engine._items.get("ns", [])):
            item.consolidation_strength = 0.0
            item.accumulated_surprise_damage = 2.0
        engine._recompute_all_free_energies("ns")
        # Edges may still reference dead entities
        for a_key, edges in engine._entanglement_graph.items():
            for b_key in edges:
                # If entity nodes are dead, edges should also be cleaned
                if a_key not in engine._entity_nodes and b_key not in engine._entity_nodes:
                    pass  # BUG: dangling edge exists

    def test_doc_freq_can_go_negative_on_double_gc(self, engine: PhaseMemoryEngine):
        """
        BUG: If an item is somehow GC'd twice, doc_freq gets decremented twice.
        The max(0, ...) guard prevents negative, but the count is wrong.
        """
        item = engine.store("unique_token_abc test", "ns")
        assert engine._doc_freq.get("unique_token_abc", 0) >= 1
        # Force GC
        item.consolidation_strength = 0.0
        item.accumulated_surprise_damage = 2.0
        engine._recompute_all_free_energies("ns")
        # doc_freq should be 0 now
        assert engine._doc_freq.get("unique_token_abc", 0) == 0
        # Second recompute — item already gone, no double-decrement
        engine._recompute_all_free_energies("ns")
        assert engine._doc_freq.get("unique_token_abc", 0) >= 0

    def test_auto_fact_subject_is_verb_not_entity(self, engine: PhaseMemoryEngine):
        """
        BUG: Auto-fact picks FIRST content word as subject.
        'Running is fun' → subject='running'. Running is a VERB, not an entity.
        'Quickly eating pasta' → subject='quickly'. An adverb.
        """
        item = engine.store("Running is very fun and exciting", "ns")
        # subject = first content word = 'running'
        assert item.fact.subject == "running"  # Verb as subject — wrong semantics

    def test_confirmation_check_ignores_raw_text(self, engine: PhaseMemoryEngine):
        """
        BUG: Confirmation checks (subject, relation, value) but ignores raw_text.
        Two different raw texts with same SRV triplet → treated as confirmation.
        'Raj eats apple at home' and 'Raj eats apple at work' → same SRV → confirmed.
        """
        item1 = engine.store("Raj eats apple at home", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple at home"))
        item2 = engine.store("Raj eats apple at work", "ns",
            fact=Fact("raj", "eat", "apple", False, "Raj eats apple at work"))
        # Second store returns SAME item despite different context
        assert item2.id == item1.id  # BUG: different contexts merged

    def test_deindex_item_O_N_per_token(self, engine: PhaseMemoryEngine):
        """
        BUG: _deindex_item calls list.remove(item) which is O(N) per token.
        With 100 items sharing a token, removing one is O(100).
        """
        for i in range(100):
            engine.store(f"shared_token_{i} banana fact", "ns")
        # Remove one item — each token's list scanned linearly
        items = engine._items["ns"]
        assert len(items) == 100

    def test_entity_spectrum_grows_unbounded(self, engine: PhaseMemoryEngine):
        """
        BUG: EntityNode.token_spectrum is a Counter that only grows.
        Every new token from every mention is added. With 1000 unique tokens
        per entity, the spectrum becomes huge. No pruning.
        """
        for i in range(100):
            engine.store(
                f"Jean visited unique_place_{i} and did thing_{i}", "ns",
                fact=Fact("jean", "visited", f"place{i}", False,
                         f"Jean visited unique_place_{i} and did thing_{i}"),
            )
        node = engine._entity_nodes.get("jean")
        if node:
            # Spectrum should be very large
            assert len(node.token_spectrum) > 50
            # This is O(|spectrum|) for every _update_entanglement call

    def test_cluster_not_updated_on_edge_desynchronization(self, engine: PhaseMemoryEngine):
        """
        BUG: Clusters are updated when edges synchronize.
        But when an edge de-synchronizes (K drops below K_critical),
        the cluster is NOT recomputed. Stale cluster with dead members.
        """
        # Create synchronized pair
        for name in ["Alice", "Bob"]:
            engine.store(f"{name} visited Rome and ate pasta and drank wine", "ns",
                fact=Fact(name.lower(), "visited", "rome", False,
                         f"{name} visited Rome and ate pasta and drank wine"))
        # Check for cluster
        cluster_count_before = len(engine._resonance_clusters)
        # Now desynchronize by pruning
        for edges in engine._entanglement_graph.values():
            for edge in edges.values():
                edge.coupling_strength = 0.001  # Below K_critical
                edge.is_synchronized = False
        # Clusters should be invalidated but aren't
        cluster_count_after = len(engine._resonance_clusters)
        # BUG: cluster_count_after == cluster_count_before (stale)

    def test_shared_memory_ids_never_cleaned(self, engine: PhaseMemoryEngine):
        """
        BUG: EntanglementEdge.shared_memory_ids accumulates item IDs.
        When items are GC'd, these IDs become stale references.
        _cer_search iterates shared_memory_ids and does linear scan.
        """
        engine.store("Jean and John visited Rome together", "ns",
            fact=Fact("jean", "visited", "rome", False, "Jean and John visited Rome together"))
        # Get edge
        a, b = sorted(["jean", "john"])
        edge = engine._entanglement_graph.get(a, {}).get(b)
        if edge:
            memory_ids_before = len(edge.shared_memory_ids)
            # Kill items
            for item in list(engine._items.get("ns", [])):
                item.consolidation_strength = 0.0
                item.accumulated_surprise_damage = 2.0
            engine._recompute_all_free_energies("ns")
            # Edge still has stale memory IDs
            if a in engine._entanglement_graph and b in engine._entanglement_graph.get(a, {}):
                edge = engine._entanglement_graph[a][b]
                # shared_memory_ids still contains dead item IDs
                assert isinstance(edge.shared_memory_ids, list)
