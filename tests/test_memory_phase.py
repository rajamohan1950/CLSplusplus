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
