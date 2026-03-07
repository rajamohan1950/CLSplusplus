"""Comprehensive plasticity engine tests - scoring, promotion, decay, pruning, edge cases."""

import math
from datetime import datetime, timedelta

import pytest

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.plasticity import PlasticityEngine


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

class TestComputeScore:

    def test_basic_score(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="test",
            salience=0.8,
            usage_count=10,
            authority=0.9,
            conflict_score=0.0,
            surprise=0.2,
        )
        score = engine.compute_score(item)
        assert score > 0
        assert item.promotion_score == score

    def test_score_formula_exact(self):
        s = Settings()
        engine = PlasticityEngine(s)
        item = MemoryItem(
            text="test",
            salience=0.5,
            usage_count=5,
            authority=0.5,
            conflict_score=0.1,
            surprise=0.3,
        )
        expected = (
            s.alpha_salience * 0.5
            + s.beta_usage * math.log1p(5)
            + s.gamma_authority * 0.5
            - s.lambda_conflict * 0.1
            + s.delta_surprise * 0.3
        )
        score = engine.compute_score(item)
        assert score == pytest.approx(max(0.0, expected), rel=1e-6)

    def test_zero_all_signals(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="test",
            salience=0.0,
            usage_count=0,
            authority=0.0,
            conflict_score=0.0,
            surprise=0.0,
        )
        score = engine.compute_score(item)
        assert score == 0.0

    def test_max_all_signals(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="test",
            salience=1.0,
            usage_count=1000,
            authority=1.0,
            conflict_score=0.0,
            surprise=1.0,
        )
        score = engine.compute_score(item)
        assert score > 0

    def test_high_conflict_reduces_score(self):
        engine = PlasticityEngine()
        low_conflict = MemoryItem(text="test", salience=0.5, authority=0.5, conflict_score=0.0)
        high_conflict = MemoryItem(text="test", salience=0.5, authority=0.5, conflict_score=1.0)
        engine.compute_score(low_conflict)
        engine.compute_score(high_conflict)
        assert low_conflict.promotion_score > high_conflict.promotion_score

    def test_score_never_negative(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="test",
            salience=0.0,
            usage_count=0,
            authority=0.0,
            conflict_score=1.0,
            surprise=0.0,
        )
        score = engine.compute_score(item)
        assert score >= 0.0

    def test_surprise_increases_score(self):
        engine = PlasticityEngine()
        no_surprise = MemoryItem(text="x", salience=0.5, authority=0.5, surprise=0.0)
        high_surprise = MemoryItem(text="x", salience=0.5, authority=0.5, surprise=1.0)
        engine.compute_score(no_surprise)
        engine.compute_score(high_surprise)
        assert high_surprise.promotion_score > no_surprise.promotion_score

    def test_usage_increases_score_logarithmically(self):
        engine = PlasticityEngine()
        low = MemoryItem(text="x", usage_count=1)
        med = MemoryItem(text="x", usage_count=10)
        high = MemoryItem(text="x", usage_count=100)
        s1 = engine.compute_score(low)
        s2 = engine.compute_score(med)
        s3 = engine.compute_score(high)
        assert s1 < s2 < s3

    def test_custom_coefficients(self):
        s = Settings(alpha_salience=2.0, beta_usage=0.0, gamma_authority=0.0, lambda_conflict=0.0, delta_surprise=0.0)
        engine = PlasticityEngine(s)
        item = MemoryItem(text="x", salience=0.5)
        score = engine.compute_score(item)
        assert score == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Promotion thresholds
# ---------------------------------------------------------------------------

class TestPromotionToL1:

    def test_low_score_no_promote(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.3, usage_count=0, authority=0.3)
        assert not engine.should_promote_to_l1(item)

    def test_high_score_promotes(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.9, usage_count=50, authority=0.9)
        assert engine.should_promote_to_l1(item)

    def test_custom_threshold(self):
        s = Settings(l1_promotion_threshold=0.1)
        engine = PlasticityEngine(s)
        item = MemoryItem(text="x", salience=0.5, authority=0.5)
        assert engine.should_promote_to_l1(item)


class TestPromotionToL2:

    def test_requires_high_confidence(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="x",
            salience=1.0,
            usage_count=100,
            authority=1.0,
            confidence=0.5,
            timestamp=datetime.utcnow() - timedelta(days=10),
        )
        assert not engine.should_promote_to_l2(item)

    def test_requires_age(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="x",
            salience=1.0,
            usage_count=100,
            authority=1.0,
            confidence=0.95,
            timestamp=datetime.utcnow(),
        )
        assert not engine.should_promote_to_l2(item)

    def test_all_criteria_met(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="x",
            salience=1.0,
            usage_count=100,
            authority=1.0,
            confidence=0.95,
            timestamp=datetime.utcnow() - timedelta(days=10),
        )
        assert engine.should_promote_to_l2(item)


class TestPromotionToL3:

    def test_delegates_to_l2_plus_confidence(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="x",
            salience=1.0,
            usage_count=100,
            authority=1.0,
            confidence=0.95,
            timestamp=datetime.utcnow() - timedelta(days=10),
        )
        assert engine.should_promote_to_l3(item)


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

class TestDecay:

    def test_reduces_salience(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.8, confidence=0.9)
        engine.apply_decay(item)
        assert item.salience < 0.8

    def test_reduces_confidence(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.8, confidence=0.9)
        engine.apply_decay(item)
        assert item.confidence < 0.9

    def test_confidence_decays_slower(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.8, confidence=0.8)
        orig_s, orig_c = item.salience, item.confidence
        engine.apply_decay(item)
        assert (orig_s - item.salience) > (orig_c - item.confidence)

    def test_never_below_zero(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.01, confidence=0.01)
        for _ in range(100):
            engine.apply_decay(item)
        assert item.salience >= 0.0
        assert item.confidence >= 0.0

    def test_custom_decay_factor(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=1.0)
        engine.apply_decay(item, decay_factor=0.5)
        assert item.salience == pytest.approx(0.5)

    def test_zero_decay_no_change(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.8, confidence=0.9)
        engine.apply_decay(item, decay_factor=0.0)
        assert item.salience == pytest.approx(0.8)
        assert item.confidence == pytest.approx(0.9)

    def test_full_decay(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.8, confidence=0.9)
        engine.apply_decay(item, decay_factor=1.0)
        assert item.salience == 0.0
        assert item.confidence == pytest.approx(0.45)

    def test_returns_item(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x")
        result = engine.apply_decay(item)
        assert result is item


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class TestPruning:

    def test_low_salience_pruned(self):
        engine = PlasticityEngine()
        assert engine.should_prune(MemoryItem(text="x", salience=0.1))

    def test_high_salience_not_pruned(self):
        engine = PlasticityEngine()
        assert not engine.should_prune(MemoryItem(text="x", salience=0.5))

    def test_at_threshold_not_pruned(self):
        engine = PlasticityEngine()
        assert not engine.should_prune(MemoryItem(text="x", salience=0.2))

    def test_just_below_threshold_pruned(self):
        engine = PlasticityEngine()
        assert engine.should_prune(MemoryItem(text="x", salience=0.199))

    def test_zero_salience_pruned(self):
        engine = PlasticityEngine()
        assert engine.should_prune(MemoryItem(text="x", salience=0.0))

    def test_custom_threshold(self):
        s = Settings(min_salience_prune=0.5)
        engine = PlasticityEngine(s)
        assert engine.should_prune(MemoryItem(text="x", salience=0.4))
        assert not engine.should_prune(MemoryItem(text="x", salience=0.6))
