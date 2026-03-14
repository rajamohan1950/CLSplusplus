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
8. Thermodynamic Semantic Field (TSF) retrieval
9. Triple-index architecture
10. Phase-modulated field radius R(s)
11. Multi-fact extraction
12. LLM-generated query field
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import pytest_asyncio

from clsplusplus.memory_phase import (
    Fact,
    PhaseMemoryEngine,
    PhaseMemoryItem,
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


def _make_item(
    engine: PhaseMemoryEngine,
    fact: Fact,
    namespace: str = "test",
    subject_aliases: list[str] | None = None,
    relation_forms: list[str] | None = None,
    value_aliases: list[str] | None = None,
) -> PhaseMemoryItem:
    """Helper to create a PhaseMemoryItem with computed thermodynamic state and TSF index."""
    engine._event_counter += 1
    H = engine._information_content(fact)
    tau = engine.TAU_OVERRIDE if fact.override else engine.TAU_DEFAULT
    rho = engine._memory_density(namespace)
    L = (engine.kT * math.log(2) * H) / max(tau, 1e-6)

    # Default query field: canonical forms only
    sa = subject_aliases or [fact.subject]
    rf = relation_forms or [fact.relation]
    va = value_aliases or [fact.value]

    item = PhaseMemoryItem(
        id=f"test-{engine._event_counter}",
        fact=fact,
        namespace=namespace,
        consolidation_strength=1.0,
        surprise_at_birth=0.0,
        tau=tau,
        birth_order=engine._event_counter,
        rho_at_birth=rho,
        free_energy=0.0,
        information_content_bits=H,
        landauer_cost=L,
        query_field_subject=sa,
        query_field_relation=rf,
        query_field_value=va,
    )
    engine._items.setdefault(namespace, []).append(item)
    engine._index_item(item)
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
        assert 1.0 <= H <= 5.0  # character-level entropy of short text


# =============================================================================
# 2. Consolidation Strength s(t)
# =============================================================================

class TestConsolidation:
    """s(t) = s₀ · exp(−Δt/τ) · (1 + β·ln(1+R)) − D"""

    def test_initial_strength_is_one(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """At birth (Δt=0), s = 1.0."""
        item = _make_item(engine, fact_apple)
        s = engine._compute_consolidation(item, delta_t=0)
        assert abs(s - 1.0) < 1e-9

    def test_strength_decays_with_time(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """s decreases as Δt increases (natural decay)."""
        item = _make_item(engine, fact_apple)
        s_0 = engine._compute_consolidation(item, delta_t=0)
        s_10 = engine._compute_consolidation(item, delta_t=10)
        s_50 = engine._compute_consolidation(item, delta_t=50)

        assert s_0 > s_10 > s_50

    def test_high_tau_decays_slower(self, engine: PhaseMemoryEngine, fact_apple: Fact, fact_banana_override: Fact):
        """Override (τ=200) decays slower than default (τ=50)."""
        item_normal = _make_item(engine, fact_apple)       # τ=50
        item_override = _make_item(engine, fact_banana_override)  # τ=200

        s_normal = engine._compute_consolidation(item_normal, delta_t=100)
        s_override = engine._compute_consolidation(item_override, delta_t=100)

        assert s_override > s_normal

    def test_retrieval_reinforces(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Retrieval count increases consolidation strength."""
        item = _make_item(engine, fact_apple)
        s_no_retrieval = engine._compute_consolidation(item, delta_t=10)

        item.retrieval_count = 5
        s_with_retrieval = engine._compute_consolidation(item, delta_t=10)

        assert s_with_retrieval > s_no_retrieval

    def test_surprise_damage_reduces_strength(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Accumulated surprise damage reduces s."""
        item = _make_item(engine, fact_apple)
        s_undamaged = engine._compute_consolidation(item, delta_t=0)

        item.accumulated_surprise_damage = 0.5
        s_damaged = engine._compute_consolidation(item, delta_t=0)

        assert s_damaged < s_undamaged
        assert abs(s_damaged - 0.5) < 1e-9  # 1.0 - 0.5

    def test_strength_clamped_to_zero(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """s cannot go below 0."""
        item = _make_item(engine, fact_apple)
        item.accumulated_surprise_damage = 5.0  # Way more than s can handle
        s = engine._compute_consolidation(item, delta_t=0)
        assert s == 0.0

    def test_phase_boundary_at_tau_c1(self, engine: PhaseMemoryEngine):
        """Memories with τ < τ_c1 should decay below floor quickly.
        Memories with τ > τ_c1 should persist."""
        # τ = 5 (below τ_c1 = 10) — should decay fast
        gas_fact = Fact("x", "y", "z", False, "x y z")
        gas_item = _make_item(engine, gas_fact)
        gas_item.tau = 5.0  # Below critical

        # τ = 50 (above τ_c1 = 10) — should persist
        liquid_fact = Fact("a", "b", "c", False, "a b c")
        liquid_item = _make_item(engine, liquid_fact)
        liquid_item.tau = 50.0  # Above critical

        # After 30 events
        s_gas = engine._compute_consolidation(gas_item, delta_t=30)
        s_liquid = engine._compute_consolidation(liquid_item, delta_t=30)

        # Gas should be below floor, liquid should be well above
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
        item = _make_item(engine, fact_apple)
        # fact_unrelated has subject="alice", relation="like" — different from "raj","eat"
        surprise, contradicted = engine._compute_surprise(fact_unrelated, [item])
        assert surprise == 0.0
        assert len(contradicted) == 0

    def test_soft_surprise_for_different_value(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Same (subject, relation), different value, no override → soft surprise."""
        item = _make_item(engine, fact_apple)
        new_fact = Fact("raj", "eat", "orange", False, "Raj eats orange")
        surprise, contradicted = engine._compute_surprise(new_fact, [item])

        assert surprise > 0.0
        assert surprise < 2.0  # Soft, not hard
        assert len(contradicted) == 1
        assert contradicted[0] is item

    def test_hard_surprise_for_override(self, engine: PhaseMemoryEngine, fact_apple: Fact, fact_banana_override: Fact):
        """Same (subject, relation), different value, WITH override → hard surprise (near max)."""
        item = _make_item(engine, fact_apple)
        surprise, contradicted = engine._compute_surprise(fact_banana_override, [item])

        # -ln(1e-6) ≈ 13.8 nats
        assert surprise > 10.0
        assert len(contradicted) == 1

    def test_no_surprise_for_same_value(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Same (subject, relation, value) → confirmation, not surprise."""
        item = _make_item(engine, fact_apple)
        same_fact = Fact("raj", "eat", "apple", False, "Raj eats apple again")
        surprise, contradicted = engine._compute_surprise(same_fact, [item])

        assert surprise == 0.0
        assert len(contradicted) == 0

    def test_gas_phase_items_ignored(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items below strength floor (gas) don't trigger surprise."""
        item = _make_item(engine, fact_apple)
        item.consolidation_strength = 0.01  # Gas phase (direct set for this check)
        item.accumulated_surprise_damage = 1.5  # Ensure it stays gas

        new_fact = Fact("raj", "eat", "orange", False, "Raj eats orange")
        surprise, contradicted = engine._compute_surprise(new_fact, [item])

        assert surprise == 0.0
        assert len(contradicted) == 0


# =============================================================================
# 4. Surprise Damage
# =============================================================================

class TestSurpriseDamage:
    """Damage = Σ · (1/τ) · amplifier. Irreversible."""

    def test_damage_applied_to_contradicted(self, engine: PhaseMemoryEngine, fact_apple: Fact, fact_banana_override: Fact):
        """Contradicted items receive surprise damage."""
        item = _make_item(engine, fact_apple)
        assert item.accumulated_surprise_damage == 0.0

        surprise, contradicted = engine._compute_surprise(fact_banana_override, [item])
        engine._apply_surprise_damage(surprise, contradicted, fact_banana_override)

        assert item.accumulated_surprise_damage > 0.0

    def test_override_amplifies_damage(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Override signal amplifies the damage."""
        item1 = _make_item(engine, fact_apple)
        item2 = _make_item(engine, Fact("raj", "eat", "apple", False, "Raj eats apple"))

        soft_fact = Fact("raj", "eat", "orange", False, "Raj eats orange")
        hard_fact = Fact("raj", "eat", "banana", True, "Raj eats banana only")

        engine._apply_surprise_damage(5.0, [item1], soft_fact)   # amplifier=1.0
        engine._apply_surprise_damage(5.0, [item2], hard_fact)   # amplifier=1.5

        assert item2.accumulated_surprise_damage > item1.accumulated_surprise_damage

    def test_high_tau_resists_damage(self, engine: PhaseMemoryEngine):
        """High τ (strong consolidation) → less vulnerable to surprise."""
        weak = Fact("x", "y", "v1", False, "x y v1")
        strong = Fact("a", "b", "c1", False, "a b c1")

        item_weak = _make_item(engine, weak)    # τ=50
        item_strong = _make_item(engine, strong)  # τ=50
        item_strong.tau = 200.0  # Manually set to strong consolidation

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
        item = _make_item(engine, fact_apple)
        F = engine._compute_free_energy(item, global_rho=0.001)
        assert F < 0.5

    def test_weak_memory_has_high_F(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """s ≈ 0 → E_pred ≈ 1 → high F (unstable, gas)."""
        item = _make_item(engine, fact_apple)
        item.accumulated_surprise_damage = 0.95  # Nearly destroyed

        F = engine._compute_free_energy(item, global_rho=0.001)
        assert F > 0.5

    def test_F_decreases_with_retrieval(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Retrieval reinforces → higher s → lower E_pred → lower F."""
        item = _make_item(engine, fact_apple)
        engine._event_counter += 10  # Age the memory

        F_before = engine._compute_free_energy(item, global_rho=0.001)
        item.retrieval_count = 10
        F_after = engine._compute_free_energy(item, global_rho=0.001)

        assert F_after < F_before

    def test_landauer_cost_is_positive(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """L = kT·ln(2)·H/τ > 0 for any non-empty fact."""
        item = _make_item(engine, fact_apple)
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
        _make_item(engine, f1, "ns")
        rho1 = engine._memory_density("ns")
        _make_item(engine, f2, "ns")
        rho2 = engine._memory_density("ns")
        assert rho2 > rho1

    def test_gas_phase_not_counted(self, engine: PhaseMemoryEngine):
        f = Fact("a", "b", "c", False, "a b c")
        item = _make_item(engine, f, "ns")
        rho_liquid = engine._memory_density("ns")
        assert rho_liquid > 0.0

        item.accumulated_surprise_damage = 1.5
        item.consolidation_strength = 0.01
        rho_gas = engine._memory_density("ns")

        assert rho_gas < rho_liquid


# =============================================================================
# 8. TSF Triple-Index Architecture
# =============================================================================

class TestTripleIndex:
    """Thermodynamic Semantic Field: subject/relation/value hash indexes."""

    def test_item_indexed_by_subject(self, engine: PhaseMemoryEngine):
        """Items are findable by subject in the index."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        _make_item(engine, fact, "test", subject_aliases=["raj"])
        assert "raj" in engine._subject_index
        assert len(engine._subject_index["raj"]) == 1

    def test_item_indexed_by_relation_forms(self, engine: PhaseMemoryEngine):
        """Items are indexed under ALL relation forms from query field."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        _make_item(engine, fact, "test",
                   relation_forms=["eat", "eats", "eating", "food", "diet", "meal"])
        # All forms should be in the relation index
        assert "eat" in engine._relation_index
        assert "eats" in engine._relation_index
        assert "food" in engine._relation_index

    def test_item_indexed_by_value(self, engine: PhaseMemoryEngine):
        """Items are findable by value aliases."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        _make_item(engine, fact, "test", value_aliases=["banana", "bananas"])
        assert "banana" in engine._value_index
        assert "bananas" in engine._value_index

    def test_deindex_removes_all_entries(self, engine: PhaseMemoryEngine):
        """Deindexing removes the item from all indexes."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        item = _make_item(engine, fact, "test",
                          subject_aliases=["raj"],
                          relation_forms=["eat", "food"],
                          value_aliases=["banana"])
        engine._deindex_item(item)
        assert "raj" not in engine._subject_index
        assert "eat" not in engine._relation_index
        assert "food" not in engine._relation_index
        assert "banana" not in engine._value_index

    def test_multiple_items_same_subject(self, engine: PhaseMemoryEngine):
        """Multiple items for same subject share the index entry."""
        f1 = Fact("raj", "eat", "banana", False, "Raj eats banana")
        f2 = Fact("raj", "visit", "rome", False, "Raj visited rome")
        _make_item(engine, f1, "test", subject_aliases=["raj"])
        _make_item(engine, f2, "test", subject_aliases=["raj"])
        assert len(engine._subject_index["raj"]) == 2


# =============================================================================
# 9. TSF Search (Thermodynamic Ranking)
# =============================================================================

class TestTSFSearch:
    """rank(q, i) = (n_matched_slots - 1) - F/kT"""

    def test_gas_phase_excluded(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items below strength_floor are not returned."""
        item = _make_item(engine, fact_apple, subject_aliases=["raj"],
                          relation_forms=["eat"], value_aliases=["apple"])
        item.accumulated_surprise_damage = 1.5

        results = engine.search("raj eat", "test")
        assert len(results) == 0

    def test_subject_match_returns_item(self, engine: PhaseMemoryEngine):
        """Matching on subject alone should return the item."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        _make_item(engine, fact, "test", subject_aliases=["raj"],
                   relation_forms=["eat"], value_aliases=["banana"])
        results = engine.search("raj", "test")
        assert len(results) == 1

    def test_relation_form_match(self, engine: PhaseMemoryEngine):
        """Query token matching a relation form finds the item."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        _make_item(engine, fact, "test", subject_aliases=["raj"],
                   relation_forms=["eat", "food", "diet"],
                   value_aliases=["banana"])
        # 'food' is in relation_forms — should match
        results = engine.search("raj food", "test")
        assert len(results) == 1

    def test_multi_slot_scores_higher(self, engine: PhaseMemoryEngine):
        """More matched slots → higher rank."""
        f1 = Fact("raj", "eat", "banana", False, "Raj eats banana")
        f2 = Fact("raj", "visit", "rome", False, "Raj visited rome")
        _make_item(engine, f1, "test", subject_aliases=["raj"],
                   relation_forms=["eat", "food"], value_aliases=["banana"])
        _make_item(engine, f2, "test", subject_aliases=["raj"],
                   relation_forms=["visit"], value_aliases=["rome"])

        # Query "raj eat banana" matches f1 on 3 slots, f2 on 1 slot (subject only)
        results = engine.search("raj eat banana", "test")
        assert len(results) == 2
        assert results[0][1].fact.value == "banana"  # 3 slots > 1 slot

    def test_stronger_memory_ranks_higher_same_slots(self, engine: PhaseMemoryEngine):
        """When slot count is equal, lower F wins."""
        f1 = Fact("raj", "eat", "apple", False, "Raj eats apple")
        f2 = Fact("raj", "eat", "banana", False, "Raj eats banana")
        item1 = _make_item(engine, f1, "test", subject_aliases=["raj"],
                           relation_forms=["eat"], value_aliases=["apple"])
        item2 = _make_item(engine, f2, "test", subject_aliases=["raj"],
                           relation_forms=["eat"], value_aliases=["banana"])

        # Damage item1 → higher F → lower rank
        item1.accumulated_surprise_damage = 0.5

        results = engine.search("raj eat", "test")
        assert len(results) == 2
        assert results[0][1].fact.value == "banana"

    def test_retrieval_increments_count(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Retrieved items get retrieval_count incremented."""
        item = _make_item(engine, fact_apple, subject_aliases=["raj"],
                          relation_forms=["eat"], value_aliases=["apple"])
        assert item.retrieval_count == 0

        engine.search("raj eat apple", "test", limit=5)
        assert item.retrieval_count == 1

    def test_fallback_when_no_index_hits(self, engine: PhaseMemoryEngine):
        """When no tokens match any index, return all liquid items by -F/kT."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        _make_item(engine, fact, "test", subject_aliases=["raj"],
                   relation_forms=["eat"], value_aliases=["banana"])
        # Query with only stopwords → no index hits → fallback
        results = engine.search("tell me everything", "test")
        assert len(results) == 1

    def test_stopwords_filtered(self, engine: PhaseMemoryEngine):
        """Stop words in query don't cause spurious index hits."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        _make_item(engine, fact, "test", subject_aliases=["raj"],
                   relation_forms=["eat"], value_aliases=["banana"])
        # "what does" are stop words — only "raj" should match
        results = engine.search("what does raj eat", "test")
        assert len(results) == 1

    def test_bigram_matching(self, engine: PhaseMemoryEngine):
        """Bigrams in query match multi-word relation forms."""
        fact = Fact("raj", "favorite food", "banana", False, "Raj's favorite food is banana")
        _make_item(engine, fact, "test", subject_aliases=["raj"],
                   relation_forms=["favorite food", "food"],
                   value_aliases=["banana"])
        # "favorite food" as a bigram should match the relation form
        results = engine.search("raj favorite food", "test")
        assert len(results) == 1

    def test_namespace_isolation(self, engine: PhaseMemoryEngine):
        """Search only returns items from the requested namespace."""
        f1 = Fact("raj", "eat", "banana", False, "Raj eats banana")
        f2 = Fact("raj", "eat", "apple", False, "Raj eats apple")
        _make_item(engine, f1, "ns1", subject_aliases=["raj"],
                   relation_forms=["eat"], value_aliases=["banana"])
        _make_item(engine, f2, "ns2", subject_aliases=["raj"],
                   relation_forms=["eat"], value_aliases=["apple"])

        results = engine.search("raj eat", "ns1")
        assert len(results) == 1
        assert results[0][1].fact.value == "banana"


# =============================================================================
# 10. Phase-Modulated Field Radius R(s)
# =============================================================================

class TestFieldRadius:
    """R(s) = floor(N_forms × s^(1/3)). Critical exponent ν=1/3."""

    def test_full_radius_at_s_one(self, engine: PhaseMemoryEngine):
        """s=1.0 → all query forms indexed."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        item = _make_item(engine, fact, "test",
                          relation_forms=["eat", "eats", "eating", "food", "diet"])
        # s=1.0, cube_root(1.0)=1.0, R=5
        assert "eat" in engine._relation_index
        assert "diet" in engine._relation_index  # Last form should be indexed

    def test_reduced_radius_at_low_s(self, engine: PhaseMemoryEngine):
        """Lower s → fewer forms indexed (field contracts)."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        forms = ["eat", "eats", "eating", "ate", "eaten",
                 "food", "diet", "meal", "snack", "cuisine"]
        item = _make_item(engine, fact, "test", relation_forms=forms)

        # Damage to reduce s
        item.consolidation_strength = 0.1  # s=0.1
        engine._index_item(item)

        # R = floor(10 × 0.1^(1/3)) = floor(10 × 0.464) = floor(4.64) = 4
        # Only first 4 forms should be indexed
        assert "eat" in engine._relation_index      # 1st — indexed
        assert "eats" in engine._relation_index     # 2nd — indexed
        assert "eating" in engine._relation_index   # 3rd — indexed
        assert "ate" in engine._relation_index      # 4th — indexed
        # 5th and beyond should NOT be indexed
        assert "cuisine" not in engine._relation_index

    def test_zero_radius_at_gas_phase(self, engine: PhaseMemoryEngine):
        """s < floor → R=0, all forms de-indexed (invisible)."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        item = _make_item(engine, fact, "test",
                          relation_forms=["eat", "food"])
        assert "eat" in engine._relation_index

        # Push to gas phase
        item.consolidation_strength = 0.01
        engine._index_item(item)

        assert "eat" not in engine._relation_index
        assert "food" not in engine._relation_index

    def test_lazy_radius_update(self, engine: PhaseMemoryEngine):
        """_update_field_radius only re-indexes when R changes."""
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        item = _make_item(engine, fact, "test",
                          relation_forms=["eat", "food", "diet"])

        initial_radius = item._last_field_radius

        # Same s → same R → no re-index
        engine._update_field_radius(item)
        assert item._last_field_radius == initial_radius


# =============================================================================
# 11. The Raj Test Case — End-to-End
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

        # --- Session 1: "Raj eats apple" ---
        fact_apple = Fact("raj", "eat", "apple", False, "Raj eats apple")
        item_apple = _make_item(engine, fact_apple, ns,
                                subject_aliases=["raj"],
                                relation_forms=["eat", "eats", "food"],
                                value_aliases=["apple", "apples"])

        # Verify: apple is in liquid phase
        engine._recompute_all_free_energies(ns)
        assert item_apple.consolidation_strength > 0.9

        # --- Session 2: "Raj eats banana only" (override) ---
        fact_banana = Fact("raj", "eat", "banana", True, "Raj eats banana only")

        # Compute surprise
        surprise, contradicted = engine._compute_surprise(fact_banana, engine._items[ns])
        assert surprise > 10.0  # Hard override → near-max surprise
        assert len(contradicted) == 1
        assert contradicted[0] is item_apple

        # Apply damage
        engine._apply_surprise_damage(surprise, contradicted, fact_banana)
        assert item_apple.accumulated_surprise_damage > 0.0

        # Store banana
        item_banana = _make_item(engine, fact_banana, ns,
                                 subject_aliases=["raj"],
                                 relation_forms=["eat", "eats", "food"],
                                 value_aliases=["banana", "bananas"])

        # Recompute
        engine._recompute_all_free_energies(ns)

        # --- Session 3: "What does Raj eat?" ---
        results = engine.search("raj eat", ns, limit=5)

        returned_values = [r[1].fact.value for r in results]
        assert "banana" in returned_values

        if "apple" in returned_values:
            banana_score = next(s for s, i in results if i.fact.value == "banana")
            apple_score = next(s for s, i in results if i.fact.value == "apple")
            assert banana_score > apple_score * 5

        # Verify apple is in gas phase
        assert item_apple.consolidation_strength < engine.STRENGTH_FLOOR

        # Verify banana is in liquid phase
        assert item_banana.consolidation_strength > 0.9

    def test_raj_free_energy_correct(self, engine: PhaseMemoryEngine):
        """After override, banana should have lower F than apple."""
        ns = "raj-fe"

        f_apple = Fact("raj", "eat", "apple", False, "Raj eats apple")
        item_a = _make_item(engine, f_apple, ns,
                            subject_aliases=["raj"],
                            relation_forms=["eat"],
                            value_aliases=["apple"])
        engine._recompute_all_free_energies(ns)

        f_banana = Fact("raj", "eat", "banana", True, "Raj eats banana only")
        surprise, contradicted = engine._compute_surprise(f_banana, engine._items[ns])
        engine._apply_surprise_damage(surprise, contradicted, f_banana)

        _make_item(engine, f_banana, ns,
                   subject_aliases=["raj"],
                   relation_forms=["eat"],
                   value_aliases=["banana"])
        engine._recompute_all_free_energies(ns)

        banana_items = [i for i in engine._items[ns] if i.fact.value == "banana"]
        assert len(banana_items) == 1
        banana_F = banana_items[0].free_energy
        assert banana_F < 0.5


# =============================================================================
# 12. Debug Output
# =============================================================================

class TestDebugOutput:

    def test_phase_debug_structure(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Debug output contains all required thermodynamic parameters."""
        _make_item(engine, fact_apple)
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
        item = _make_item(engine, fact_apple)
        d = item.to_debug_dict(strength_floor=engine.STRENGTH_FLOOR)
        assert d["fact"]["subject"] == "raj"
        assert d["fact"]["relation"] == "eat"
        assert d["fact"]["value"] == "apple"
        assert d["phase"] == "liquid"


# =============================================================================
# 13. Confirmation Reinforcement
# =============================================================================

class TestConfirmationReinforcement:
    """Repeating the same fact should reinforce, not duplicate."""

    @pytest.mark.asyncio
    async def test_confirmation_reinforces_existing(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Same (subject, relation, value) twice → one item with retrieval_count=1."""
        ns = "confirm-test"
        item = _make_item(engine, fact_apple, ns,
                          subject_aliases=["raj"],
                          relation_forms=["eat"],
                          value_aliases=["apple"])
        assert item.retrieval_count == 0

        # Mock LLM returns same fact with query_field
        async def mock_caller(system, msg):
            return '[{"subject": "raj", "relation": "eat", "value": "apple", "override": false, "query_field": {"subject_aliases": ["raj"], "relation_forms": ["eat"], "value_aliases": ["apple"]}}]'

        results = await engine.ingest("Raj eats apple", ns, mock_caller)
        assert len(results) == 1
        assert results[0] is item
        assert item.retrieval_count == 1
        assert len(engine._items[ns]) == 1


# =============================================================================
# 14. Garbage Collection
# =============================================================================

class TestGarbageCollection:
    """Dead gas-phase items should be cleaned up."""

    def test_irrecoverable_items_removed(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items with s=0 and damage>1.0 are garbage collected."""
        ns = "gc-test"
        item = _make_item(engine, fact_apple, ns)
        item.accumulated_surprise_damage = 1.5
        engine._recompute_all_free_energies(ns)

        assert item.consolidation_strength == 0.0
        assert len(engine._items.get(ns, [])) == 0

    def test_recoverable_items_kept(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Items with low damage (potentially recoverable) are kept."""
        ns = "gc-test"
        item = _make_item(engine, fact_apple, ns)
        item.accumulated_surprise_damage = 0.5
        engine._recompute_all_free_energies(ns)

        assert len(engine._items.get(ns, [])) == 1

    def test_gc_deindexes_dead_items(self, engine: PhaseMemoryEngine):
        """GC should also remove dead items from indexes."""
        ns = "gc-idx"
        fact = Fact("raj", "eat", "banana", False, "Raj eats banana")
        item = _make_item(engine, fact, ns,
                          subject_aliases=["raj"],
                          relation_forms=["eat"],
                          value_aliases=["banana"])
        assert "raj" in engine._subject_index

        item.accumulated_surprise_damage = 1.5
        engine._recompute_all_free_energies(ns)

        # Dead item should be deindexed
        assert "raj" not in engine._subject_index


# =============================================================================
# 15. Damage Cap
# =============================================================================

class TestDamageCap:
    """Accumulated surprise damage should be capped at 2.0."""

    def test_damage_capped_at_two(self, engine: PhaseMemoryEngine, fact_apple: Fact):
        """Multiple damage applications cannot exceed 2.0."""
        item = _make_item(engine, fact_apple)
        for _ in range(5):
            override_fact = Fact("raj", "eat", "something", True, "override")
            engine._apply_surprise_damage(13.8, [item], override_fact)

        assert item.accumulated_surprise_damage <= 2.0


# =============================================================================
# 16. Multi-Fact Extraction
# =============================================================================

class TestMultiFactExtraction:
    """Single message can produce multiple facts."""

    @pytest.mark.asyncio
    async def test_multi_fact_ingest(self, engine: PhaseMemoryEngine):
        """A compound statement should produce multiple memory items."""
        ns = "multi-test"

        async def mock_caller(system, msg):
            return '''[
                {"subject": "raj", "relation": "visit", "value": "rome", "override": false,
                 "query_field": {"subject_aliases": ["raj"], "relation_forms": ["visit"], "value_aliases": ["rome"]}},
                {"subject": "raj", "relation": "like", "value": "pasta", "override": false,
                 "query_field": {"subject_aliases": ["raj"], "relation_forms": ["like"], "value_aliases": ["pasta"]}}
            ]'''

        results = await engine.ingest("I went to Rome and loved the pasta", ns, mock_caller)
        assert len(results) == 2
        values = {r.fact.value for r in results}
        assert "rome" in values
        assert "pasta" in values

    @pytest.mark.asyncio
    async def test_single_fact_backward_compat(self, engine: PhaseMemoryEngine):
        """Single-fact extraction still works (backward compatible)."""
        ns = "single-test"

        async def mock_caller(system, msg):
            return '{"subject": "raj", "relation": "eat", "value": "banana", "override": false}'

        results = await engine.ingest("Raj eats banana", ns, mock_caller)
        assert len(results) == 1
        assert results[0].fact.value == "banana"

    @pytest.mark.asyncio
    async def test_no_facts_returns_empty(self, engine: PhaseMemoryEngine):
        """Non-factual message returns empty list."""
        ns = "nofact-test"

        async def mock_caller(system, msg):
            return '[{"extract": false}]'

        results = await engine.ingest("Hello, how are you?", ns, mock_caller)
        assert results == []

    @pytest.mark.asyncio
    async def test_ingest_returns_list(self, engine: PhaseMemoryEngine):
        """ingest() always returns a list."""
        ns = "list-test"

        async def mock_caller(system, msg):
            return '[{"extract": false}]'

        results = await engine.ingest("Hi", ns, mock_caller)
        assert isinstance(results, list)


# =============================================================================
# 17. Query Field Generation
# =============================================================================

class TestQueryFieldGeneration:
    """LLM-generated query fields are parsed and stored correctly."""

    @pytest.mark.asyncio
    async def test_query_field_stored_on_item(self, engine: PhaseMemoryEngine):
        """Ingested items should have query field from LLM."""
        ns = "qf-test"

        async def mock_caller(system, msg):
            return '''[{"subject": "raj", "relation": "eat", "value": "banana", "override": false,
                "query_field": {
                    "subject_aliases": ["raj", "rajan"],
                    "relation_forms": ["eat", "eats", "eating", "food", "diet"],
                    "value_aliases": ["banana", "bananas"]
                }}]'''

        results = await engine.ingest("Raj eats banana", ns, mock_caller)
        assert len(results) == 1
        item = results[0]
        assert "raj" in item.query_field_subject
        assert "rajan" in item.query_field_subject
        assert "food" in item.query_field_relation
        assert "bananas" in item.query_field_value

    @pytest.mark.asyncio
    async def test_canonical_forms_always_present(self, engine: PhaseMemoryEngine):
        """Even if LLM omits canonical form from query_field, it's auto-added."""
        ns = "canon-test"

        async def mock_caller(system, msg):
            return '''[{"subject": "raj", "relation": "eat", "value": "banana", "override": false,
                "query_field": {
                    "subject_aliases": ["rajan"],
                    "relation_forms": ["food", "diet"],
                    "value_aliases": ["bananas"]
                }}]'''

        results = await engine.ingest("Raj eats banana", ns, mock_caller)
        item = results[0]
        # Canonical forms should be auto-inserted at position 0
        assert item.query_field_subject[0] == "raj"
        assert item.query_field_relation[0] == "eat"
        assert item.query_field_value[0] == "banana"

    @pytest.mark.asyncio
    async def test_missing_query_field_uses_defaults(self, engine: PhaseMemoryEngine):
        """If LLM doesn't return query_field, use canonical forms."""
        ns = "noqf-test"

        async def mock_caller(system, msg):
            return '[{"subject": "raj", "relation": "eat", "value": "banana", "override": false}]'

        results = await engine.ingest("Raj eats banana", ns, mock_caller)
        item = results[0]
        assert item.query_field_subject == ["raj"]
        assert item.query_field_relation == ["eat"]
        assert item.query_field_value == ["banana"]

    @pytest.mark.asyncio
    async def test_indexed_after_ingest(self, engine: PhaseMemoryEngine):
        """After ingest, items should be in the triple index."""
        ns = "idx-test"

        async def mock_caller(system, msg):
            return '''[{"subject": "raj", "relation": "eat", "value": "banana", "override": false,
                "query_field": {
                    "subject_aliases": ["raj"],
                    "relation_forms": ["eat", "food"],
                    "value_aliases": ["banana"]
                }}]'''

        await engine.ingest("Raj eats banana", ns, mock_caller)
        assert "raj" in engine._subject_index
        assert "eat" in engine._relation_index
        assert "food" in engine._relation_index
        assert "banana" in engine._value_index


# =============================================================================
# 18. Tokenize Query
# =============================================================================

class TestTokenizeQuery:
    """Query tokenization with stop-word filtering and bigrams."""

    def test_stopwords_removed(self, engine: PhaseMemoryEngine):
        tokens = engine._tokenize_query("what does raj eat")
        assert "what" not in tokens
        assert "does" not in tokens
        assert "raj" in tokens
        assert "eat" in tokens

    def test_bigrams_generated(self, engine: PhaseMemoryEngine):
        tokens = engine._tokenize_query("raj favorite food")
        assert "raj" in tokens
        assert "favorite" in tokens
        assert "food" in tokens
        assert "raj favorite" in tokens
        assert "favorite food" in tokens

    def test_single_char_tokens_removed(self, engine: PhaseMemoryEngine):
        tokens = engine._tokenize_query("a b raj")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "raj" in tokens
