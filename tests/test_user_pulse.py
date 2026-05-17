"""Tests for the live user-pulse frustration signal + CSAT aggregation.

Covers:
  * Happy path — a user with no negative signals reads as 'happy'.
  * The scoring contract — weights, saturation at 100, band thresholds.
  * Each behavioural signal — 4xx, 5xx, quota, rage-retry, thumbs-down.
  * The corruption edge case — Redis errors must fail OPEN (no raise,
    no blocked request), never lock a user out or crash the path.
  * Rage-retry detection — only fires after repeated same-op failures.
  * CSAT aggregation — % satisfied (4-5), avg score, daily trend.

Redis is faked in-process the same way test_window_limits.py mocks it:
the pulse logic is deterministic, so an in-memory fake is the right
unit-level boundary. Run just this file:

    pytest tests/test_user_pulse.py -v
"""

import time
from unittest.mock import patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.user_pulse import (
    _BAND_FRUSTRATED,
    _BAND_NEUTRAL,
    PULSE_WINDOW_SECONDS,
    get_user_pulse,
    list_frustrated_users,
    record_http_outcome,
    record_signal,
    score_from_counts,
)
from clsplusplus.stores.user_store import UserStore


def _settings():
    return Settings(redis_url="redis://localhost:6379")


# ---------------------------------------------------------------------------
# In-memory fake Redis — supports exactly the ops user_pulse.py uses.
# ---------------------------------------------------------------------------

class FakeRedis:
    """Minimal async Redis fake: string counters + one sorted set."""

    def __init__(self):
        self.strings: dict[str, int] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def incrby(self, key, amount=1):
        self.strings[key] = self.strings.get(key, 0) + amount
        return self.strings[key]

    async def incr(self, key):
        return await self.incrby(key, 1)

    async def get(self, key):
        v = self.strings.get(key)
        return str(v) if v is not None else None

    async def expire(self, key, ttl):
        return True

    async def zadd(self, key, mapping):
        z = self.zsets.setdefault(key, {})
        z.update(mapping)
        return len(mapping)

    async def zrange(self, key, start, stop):
        members = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        if stop == -1:
            stop = len(members)
        else:
            stop += 1
        return [m for m, _ in members[start:stop]]

    async def zremrangebyscore(self, key, lo, hi):
        z = self.zsets.get(key, {})
        hi_f = float("inf") if hi == "+inf" else float(hi)
        lo_f = float("-inf") if lo == "-inf" else float(lo)
        removed = [m for m, s in z.items() if lo_f <= s <= hi_f]
        for m in removed:
            del z[m]
        return len(removed)

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    """Buffers calls, replays them on execute() — matches redis-py asyncio."""

    def __init__(self, client: FakeRedis):
        self._client = client
        self._ops: list = []

    def incrby(self, key, amount=1):
        self._ops.append(("incrby", (key, amount)))
        return self

    def incr(self, key):
        self._ops.append(("incrby", (key, 1)))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", (key, ttl)))
        return self

    def get(self, key):
        self._ops.append(("get", (key,)))
        return self

    def zadd(self, key, mapping):
        self._ops.append(("zadd", (key, mapping)))
        return self

    async def execute(self):
        results = []
        for name, args in self._ops:
            results.append(await getattr(self._client, name)(*args))
        return results


class BrokenRedis:
    """Every operation raises — used to prove the fail-open contract."""

    def __getattr__(self, _name):
        raise ConnectionError("redis down")


# ---------------------------------------------------------------------------
# Pure scoring contract
# ---------------------------------------------------------------------------

class TestScoreFromCounts:

    def test_no_signals_is_happy(self):
        result = score_from_counts({})
        assert result["score"] == 0
        assert result["band"] == "happy"
        assert result["reasons"] == []

    def test_single_5xx_weighted(self):
        # error_5xx weight is 22 → one event = 22 points.
        result = score_from_counts({"error_5xx": 1})
        assert result["score"] == 22
        assert result["band"] == "happy"  # 22 < neutral threshold (25)

    def test_band_thresholds(self):
        # Two 5xx (44) lands in neutral; thumbs_down alone (30) is neutral.
        assert score_from_counts({"error_5xx": 2})["band"] == "neutral"
        assert score_from_counts({"thumbs_down": 1})["band"] == "neutral"
        # A thumbs-down + 5xx (52) is still neutral; add a 4xx → 61 frustrated.
        mixed = score_from_counts({"thumbs_down": 1, "error_5xx": 1, "error_4xx": 1})
        assert mixed["score"] == 61
        assert mixed["band"] == "frustrated"

    def test_score_saturates_at_100(self):
        # Twenty 5xx would be 440 raw — must clamp to 100.
        result = score_from_counts({"error_5xx": 20})
        assert result["score"] == 100
        assert result["band"] == "frustrated"

    def test_reasons_sorted_loudest_first(self):
        result = score_from_counts({"error_4xx": 1, "error_5xx": 1})
        # error_5xx (22pt) must come before error_4xx (9pt).
        assert result["reasons"][0]["signal"] == "error_5xx"
        assert result["reasons"][0]["points"] == 22
        assert result["reasons"][1]["signal"] == "error_4xx"

    def test_band_constants_ordered(self):
        # Sanity guard so a future tweak can't invert the bands.
        assert 0 < _BAND_NEUTRAL < _BAND_FRUSTRATED <= 100


# ---------------------------------------------------------------------------
# Recording behavioural signals through fake Redis
# ---------------------------------------------------------------------------

class TestRecordSignal:

    @pytest.mark.asyncio
    async def test_record_and_read_back(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            await record_signal("user-1", "error_5xx", _settings())
            await record_signal("user-1", "error_5xx", _settings())
            pulse = await get_user_pulse("user-1", _settings())
        # Two 5xx → 44 points, neutral band.
        assert pulse["score"] == 44
        assert pulse["band"] == "neutral"
        assert pulse["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_unknown_signal_ignored(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            await record_signal("user-1", "not_a_real_signal", _settings())
            pulse = await get_user_pulse("user-1", _settings())
        assert pulse["score"] == 0

    @pytest.mark.asyncio
    async def test_empty_user_id_is_noop(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            await record_signal("", "error_5xx", _settings())
        assert fake.strings == {}


# ---------------------------------------------------------------------------
# HTTP-outcome classification + rage-retry detection
# ---------------------------------------------------------------------------

class TestRecordHttpOutcome:

    @pytest.mark.asyncio
    async def test_2xx_records_nothing(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            await record_http_outcome("user-1", 200, "write", _settings())
            pulse = await get_user_pulse("user-1", _settings())
        assert pulse["score"] == 0
        assert pulse["band"] == "happy"

    @pytest.mark.asyncio
    async def test_quota_block_classified(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            await record_http_outcome("user-1", 429, "read", _settings())
            pulse = await get_user_pulse("user-1", _settings())
        # 429 → quota_block weight 14.
        assert pulse["score"] == 14
        assert any(r["signal"] == "quota_block" for r in pulse["reasons"])

    @pytest.mark.asyncio
    async def test_5xx_classified(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            await record_http_outcome("user-1", 503, "write", _settings())
            pulse = await get_user_pulse("user-1", _settings())
        assert any(r["signal"] == "error_5xx" for r in pulse["reasons"])

    @pytest.mark.asyncio
    async def test_rage_retry_fires_after_three_same_op_failures(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            # Three 404s on the SAME operation → rage-retry trips on the 3rd.
            for _ in range(3):
                await record_http_outcome("user-1", 404, "write", _settings())
            pulse = await get_user_pulse("user-1", _settings())
        signals = {r["signal"] for r in pulse["reasons"]}
        assert "rage_retry" in signals
        assert "error_4xx" in signals

    @pytest.mark.asyncio
    async def test_no_rage_retry_for_distinct_operations(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            # Three 404s on DIFFERENT operations — not rage, just bad luck.
            for op in ("write", "read", "delete"):
                await record_http_outcome("user-1", 404, op, _settings())
            pulse = await get_user_pulse("user-1", _settings())
        signals = {r["signal"] for r in pulse["reasons"]}
        assert "rage_retry" not in signals
        assert "error_4xx" in signals


# ---------------------------------------------------------------------------
# Fail-open invariant — a Redis outage must never raise or block
# ---------------------------------------------------------------------------

class TestFailOpen:

    @pytest.mark.asyncio
    async def test_record_signal_swallows_redis_error(self):
        with patch("clsplusplus.user_pulse._redis_client", return_value=BrokenRedis()):
            # Must not raise.
            await record_signal("user-1", "error_5xx", _settings())

    @pytest.mark.asyncio
    async def test_record_http_outcome_swallows_redis_error(self):
        with patch("clsplusplus.user_pulse._redis_client", return_value=BrokenRedis()):
            await record_http_outcome("user-1", 500, "write", _settings())

    @pytest.mark.asyncio
    async def test_get_pulse_fails_open_to_happy(self):
        with patch("clsplusplus.user_pulse._redis_client", return_value=BrokenRedis()):
            pulse = await get_user_pulse("user-1", _settings())
        assert pulse["score"] == 0
        assert pulse["band"] == "happy"

    @pytest.mark.asyncio
    async def test_list_frustrated_fails_open_to_empty(self):
        with patch("clsplusplus.user_pulse._redis_client",
                   side_effect=ConnectionError("down")):
            result = await list_frustrated_users(_settings())
        assert result == []


# ---------------------------------------------------------------------------
# Listing frustrated users from the active set
# ---------------------------------------------------------------------------

class TestListFrustratedUsers:

    @pytest.mark.asyncio
    async def test_only_frustrated_users_returned_worst_first(self):
        fake = FakeRedis()
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            # happy-user: nothing.
            # neutral-user: two 5xx (44) → neutral, must NOT appear.
            await record_signal("neutral-user", "error_5xx", _settings())
            await record_signal("neutral-user", "error_5xx", _settings())
            # frustrated-A: five 5xx (110→100) → frustrated.
            for _ in range(5):
                await record_signal("frustrated-A", "error_5xx", _settings())
            # frustrated-B: three 5xx (66) → frustrated, lower score.
            for _ in range(3):
                await record_signal("frustrated-B", "error_5xx", _settings())
            result = await list_frustrated_users(_settings())

        ids = [u["user_id"] for u in result]
        assert "neutral-user" not in ids
        assert "happy-user" not in ids
        assert ids == ["frustrated-A", "frustrated-B"]  # worst first
        assert result[0]["score"] >= result[1]["score"]

    @pytest.mark.asyncio
    async def test_stale_active_members_pruned(self):
        fake = FakeRedis()
        # Inject a stale member older than the window.
        fake.zsets["cls:pulse:active"] = {
            "stale-user": time.time() - PULSE_WINDOW_SECONDS - 100,
        }
        with patch("clsplusplus.user_pulse._redis_client", return_value=fake):
            await list_frustrated_users(_settings())
        assert "stale-user" not in fake.zsets["cls:pulse:active"]


# ---------------------------------------------------------------------------
# CSAT aggregation — pure-Python part of UserStore.get_feedback_summary
# ---------------------------------------------------------------------------

class TestCsatAggregation:
    """Exercises the CSAT math (% satisfied, avg, trend) via a fake DB
    connection — no live Postgres needed for the aggregation logic."""

    class _FakeConn:
        def __init__(self, totals, trend, recent):
            self._totals = totals
            self._trend = trend
            self._recent = recent

        async def fetchrow(self, query, *args):
            return self._totals

        async def fetch(self, query, *args):
            # The trend query groups by DATE; the recent query joins users.
            if "GROUP BY DATE" in query:
                return self._trend
            return self._recent

    class _FakePool:
        def __init__(self, conn):
            self._conn = conn

        def acquire(self):
            pool = self

            class _Ctx:
                async def __aenter__(self_):
                    return pool._conn

                async def __aexit__(self_, *a):
                    return False

            return _Ctx()

    @pytest.mark.asyncio
    async def test_csat_percent_and_trend(self, monkeypatch):
        totals = {"responses": 10, "satisfied": 7, "avg_score": 4.1,
                  "thumbs_down": 2}
        trend = [
            {"day": "2026-05-15", "responses": 4, "satisfied": 3, "avg_score": 4.0},
            {"day": "2026-05-16", "responses": 6, "satisfied": 4, "avg_score": 4.2},
        ]
        recent = []
        store = UserStore(_settings())
        fake_pool = self._FakePool(self._FakeConn(totals, trend, recent))

        async def _fake_get_pool():
            return fake_pool

        monkeypatch.setattr(store, "get_pool", _fake_get_pool)
        summary = await store.get_feedback_summary(days=30)

        # 7 of 10 satisfied → 70.0% CSAT.
        assert summary["csat_percent"] == 70.0
        assert summary["avg_score"] == 4.1
        assert summary["responses"] == 10
        assert summary["thumbs_down"] == 2
        assert len(summary["trend"]) == 2
        # Day 1: 3/4 = 75%, Day 2: 4/6 = 66.7%.
        assert summary["trend"][0]["csat"] == 75.0
        assert summary["trend"][1]["csat"] == 66.7

    @pytest.mark.asyncio
    async def test_csat_zero_responses_no_div_by_zero(self, monkeypatch):
        totals = {"responses": 0, "satisfied": 0, "avg_score": 0,
                  "thumbs_down": 0}
        store = UserStore(_settings())
        fake_pool = self._FakePool(self._FakeConn(totals, [], []))

        async def _fake_get_pool():
            return fake_pool

        monkeypatch.setattr(store, "get_pool", _fake_get_pool)
        summary = await store.get_feedback_summary(days=30)
        assert summary["csat_percent"] == 0.0
        assert summary["responses"] == 0
        assert summary["trend"] == []
