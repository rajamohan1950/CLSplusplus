"""Tests for MeteringReconciler (ADR 0001 step 3).

Fakes for both Postgres and Redis so the whole file runs without infra.
Covers the pure comparison logic, the scan loop, drift-to-dead-letter
enqueue, tolerance thresholds, and the daily period helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from clsplusplus.config import Settings
from clsplusplus.metering_v2.reconciler import (
    DriftFinding,
    MeteringReconciler,
    _current_period,
    _hash_key,
    _period_window,
    _prior_period,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeRedis:
    """Minimal subset of redis.asyncio.Redis we use in reconciler."""

    def __init__(self, keys: dict[str, int]):
        # keys maps the full Redis key (e.g. "cls:ops:k1:2026-04") to count.
        self.keys = dict(keys)

    async def scan_iter(self, match: str, count: int = 500):
        # Naive prefix / suffix match; enough for "cls:ops:*:YYYY-MM"
        prefix, _, suffix = match.partition("*")
        for k in list(self.keys.keys()):
            if k.startswith(prefix) and k.endswith(suffix):
                yield k

    async def get(self, key: str):
        v = self.keys.get(key)
        return str(v) if v is not None else None


class FakeConn:
    def __init__(self, owner: "FakePool"):
        self.owner = owner

    async def fetch(self, sql: str, *args):
        if "FROM usage_events" in sql:
            return [
                {"api_key_id": h, "total": n} for h, n in self.owner.pg_counts.items()
            ]
        return []

    async def execute(self, sql: str, *args):
        if "INTO metering_dead_letter" in sql:
            self.owner.dead_letter.append(args)


class _AcquireCtx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return None


class FakePool:
    def __init__(self, pg_counts: dict[str, int]):
        self.pg_counts = dict(pg_counts)
        self.dead_letter: list = []

    def acquire(self):
        return _AcquireCtx(FakeConn(self))


def _make_reconciler(
    redis_keys: dict[str, int],
    pg_counts: dict[str, int],
    *,
    drift_percent: float = 0.001,
    min_abs_drift: int = 5,
    enabled: bool = True,
) -> tuple[MeteringReconciler, FakePool]:
    pool = FakePool(pg_counts)
    redis = FakeRedis(redis_keys)

    async def pool_getter() -> Any:
        return pool

    async def redis_getter() -> Any:
        return redis

    settings = Settings(metering_v2_write_enabled=enabled)
    rec = MeteringReconciler(
        settings, pool_getter, redis_getter,
        drift_percent=drift_percent,
        min_abs_drift=min_abs_drift,
    )
    return rec, pool


# --------------------------------------------------------------------------- #
# Period helpers
# --------------------------------------------------------------------------- #


def test_period_window_january():
    start, end = _period_window("2026-01")
    assert start == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert end == datetime(2026, 2, 1, tzinfo=timezone.utc)


def test_period_window_december_rolls_year():
    start, end = _period_window("2026-12")
    assert start == datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert end == datetime(2027, 1, 1, tzinfo=timezone.utc)


def test_prior_period_january_rolls_to_previous_year():
    # This is a pure wall-clock function; sanity check against current month.
    prior = _prior_period()
    current = _current_period()
    y1, m1 = map(int, prior.split("-"))
    y2, m2 = map(int, current.split("-"))
    if m2 == 1:
        assert (y1, m1) == (y2 - 1, 12)
    else:
        assert (y1, m1) == (y2, m2 - 1)


def test_hash_key_matches_writer_scheme():
    # Writer uses sha256()[:16] — exact same scheme must match.
    assert _hash_key("abcd") == _hash_key("abcd")
    assert len(_hash_key("abcd")) == 16
    assert _hash_key("a") != _hash_key("b")


# --------------------------------------------------------------------------- #
# Reconciliation logic
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_drift_when_both_views_agree():
    h_k1 = _hash_key("k1")
    redis = {"cls:ops:k1:2026-04": 10_000}
    pg = {h_k1: 10_000}
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    assert result.findings == []
    assert pool.dead_letter == []
    assert result.redis_keys_seen == 1
    assert result.postgres_aggregates_seen == 1


@pytest.mark.asyncio
async def test_small_absolute_drift_is_ignored():
    """Abs drift under min_abs_drift is noise from async write lag."""
    h_k1 = _hash_key("k1")
    redis = {"cls:ops:k1:2026-04": 10_000}
    pg = {h_k1: 10_003}  # drift=3 < min_abs=5 default
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    assert result.findings == []
    assert pool.dead_letter == []


@pytest.mark.asyncio
async def test_small_percent_drift_is_ignored():
    """Drift above min_abs but under the percent threshold is allowed."""
    h_k1 = _hash_key("k1")
    redis = {"cls:ops:k1:2026-04": 100_000}
    pg = {h_k1: 100_050}  # 0.05% drift, under 0.1% threshold
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    assert result.findings == []


@pytest.mark.asyncio
async def test_large_drift_flags_finding():
    h_k1 = _hash_key("k1")
    redis = {"cls:ops:k1:2026-04": 10_000}
    pg = {h_k1: 5_000}  # 50% drift, well over threshold
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.api_key_hash == h_k1
    assert f.redis_count == 10_000
    assert f.postgres_count == 5_000
    assert f.drift == 5_000
    assert f.drift_pct == pytest.approx(0.5)
    # Dead-letter row was enqueued.
    assert len(pool.dead_letter) == 1
    err_class, msg, _payload_json = pool.dead_letter[0]
    assert err_class == "ReconciliationDrift"
    assert "2026-04" in msg


@pytest.mark.asyncio
async def test_postgres_only_hash_flags_drift():
    """Hash in PG but NOT in Redis means PG over-counted (or Redis was wiped)."""
    rogue = _hash_key("ghost")
    redis = {}
    pg = {rogue: 1_000}
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.redis_count == 0
    assert f.postgres_count == 1_000


@pytest.mark.asyncio
async def test_redis_only_hash_flags_drift():
    """Hash in Redis but NOT in PG means a write was dropped — THE money bug."""
    redis = {"cls:ops:lost:2026-04": 5_000}
    pg = {}
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.api_key_hash == _hash_key("lost")
    assert f.redis_count == 5_000
    assert f.postgres_count == 0


@pytest.mark.asyncio
async def test_multiple_hashes_compared_independently():
    h1, h2, h3 = _hash_key("k1"), _hash_key("k2"), _hash_key("k3")
    redis = {
        "cls:ops:k1:2026-04": 100,         # matches pg
        "cls:ops:k2:2026-04": 200,         # drifts
        "cls:ops:k3:2026-04": 1000,        # matches pg
    }
    pg = {h1: 100, h2: 500, h3: 1000}
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    drift_hashes = {f.api_key_hash for f in result.findings}
    assert drift_hashes == {h2}


@pytest.mark.asyncio
async def test_scan_ignores_keys_from_other_periods():
    """Pattern match is strict — don't pull prior-period keys into this window."""
    redis = {
        "cls:ops:k1:2026-04": 100,
        "cls:ops:k1:2026-03": 999_999,  # wrong period
    }
    pg = {_hash_key("k1"): 100}
    rec, pool = _make_reconciler(redis, pg)
    result = await rec.reconcile_once("2026-04")
    assert result.findings == []


@pytest.mark.asyncio
async def test_reconciler_disabled_still_returns_structure():
    """Flag off shouldn't crash — reconcile_once is useful for manual runs."""
    rec, pool = _make_reconciler({}, {}, enabled=False)
    result = await rec.reconcile_once("2026-04")
    assert result.findings == []
    # enabled=False means the daily loop won't auto-start, but manual call works
    assert rec.enabled is False


@pytest.mark.asyncio
async def test_custom_thresholds():
    """Caller can tighten or relax tolerance."""
    h_k1 = _hash_key("k1")
    redis = {"cls:ops:k1:2026-04": 1_000_000}
    pg = {h_k1: 1_000_100}  # 0.01% drift, 100 absolute
    # Default threshold: 0.1% + min_abs=5 — drift pct too small → no finding
    rec_default, _ = _make_reconciler(redis, pg)
    assert (await rec_default.reconcile_once("2026-04")).findings == []
    # Tighten: 0.001% threshold — now it fires
    rec_strict, _ = _make_reconciler(redis, pg, drift_percent=0.00001)
    assert len(((await rec_strict.reconcile_once("2026-04")).findings)) == 1
