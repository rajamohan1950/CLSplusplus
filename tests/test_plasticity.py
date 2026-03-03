"""Plasticity engine tests."""

from datetime import datetime, timedelta

import pytest

from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.plasticity import PlasticityEngine


def test_compute_score():
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


def test_should_promote_to_l1():
    engine = PlasticityEngine()
    low = MemoryItem(text="x", salience=0.3, usage_count=0, authority=0.3)
    high = MemoryItem(text="x", salience=0.9, usage_count=50, authority=0.9)
    assert not engine.should_promote_to_l1(low)
    assert engine.should_promote_to_l1(high)


def test_apply_decay():
    engine = PlasticityEngine()
    item = MemoryItem(text="x", salience=0.8, confidence=0.9)
    engine.apply_decay(item)
    assert item.salience < 0.8
    assert item.confidence < 0.9


def test_should_prune():
    engine = PlasticityEngine()
    assert engine.should_prune(MemoryItem(text="x", salience=0.1))
    assert not engine.should_prune(MemoryItem(text="x", salience=0.5))
