"""Tests for free-tier multi-window usage caps (window_limits.py).

Covers the happy path (under cap → allowed), each invariant (caps
enforced, shortest window reported first, paid tiers unaffected via the
middleware), the corruption edge case (Redis error must fail OPEN, not
lock users out), and a regression guard for bucket-string rollover.

Redis is mocked the way the existing suite (test_rate_limit.py) mocks
it — the real suite treats Redis as real in integration tests, but the
unit-level window logic is deterministic and best mocked here.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.window_limits import (
    _bucket,
    _window_key,
    check_window_limits,
    record_window_operation,
    retry_after_seconds,
)


def _settings(**overrides):
    base = dict(
        free_cap_per_hour=120,
        free_cap_per_day=1_000,
        free_cap_per_week=4_000,
        free_cap_per_month=8_000,
        redis_url="redis://localhost:6379",
    )
    base.update(overrides)
    return Settings(**base)


def _client_with_counts(counts: dict[str, int]):
    """Mock Redis client whose GET returns counts keyed by window period.

    `counts` maps a period name (hour/day/week/month) to its stored count.
    """
    client = MagicMock()

    async def _get(key):
        for period, val in counts.items():
            if f":win:{period}:" in key:
                return str(val)
        return None

    client.get = AsyncMock(side_effect=_get)
    client.incr = AsyncMock(return_value=1)
    client.expire = AsyncMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Bucket / key construction
# ---------------------------------------------------------------------------

class TestBucketStrings:

    def test_bucket_formats(self):
        now = datetime(2026, 5, 17, 14, 30, 0)  # a Sunday
        assert _bucket("hour", now) == "2026-05-17-14"
        assert _bucket("day", now) == "2026-05-17"
        assert _bucket("month", now) == "2026-05"
        # ISO week for 2026-05-17 is week 20
        assert _bucket("week", now) == "2026-W20"

    def test_window_key_includes_subject_and_period(self):
        now = datetime(2026, 5, 17, 14, 0, 0)
        key = _window_key("hour", "owner:abc123", now)
        assert key == "cls:win:hour:owner:abc123:2026-05-17-14"

    def test_unknown_period_raises(self):
        with pytest.raises(ValueError):
            _bucket("decade")

    def test_hour_bucket_rolls_over(self):
        """Regression guard: two timestamps in different hours must produce
        different bucket strings, else an hourly cap would never reset."""
        h1 = _bucket("hour", datetime(2026, 5, 17, 14, 59, 59))
        h2 = _bucket("hour", datetime(2026, 5, 17, 15, 0, 0))
        assert h1 != h2


# ---------------------------------------------------------------------------
# Happy path — under every cap
# ---------------------------------------------------------------------------

class TestUnderCap:

    @pytest.mark.asyncio
    async def test_under_all_caps_allowed(self):
        client = _client_with_counts({"hour": 10, "day": 50, "week": 100, "month": 200})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, used, limit = await check_window_limits("owner:x", _settings())
        assert allowed is True
        assert window is None
        assert used == 0 and limit == 0

    @pytest.mark.asyncio
    async def test_at_zero_allowed(self):
        """No counter set yet (GET → None) is treated as zero usage."""
        client = _client_with_counts({})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, _, _ = await check_window_limits("owner:new", _settings())
        assert allowed is True
        assert window is None


# ---------------------------------------------------------------------------
# Invariant — each window cap is enforced
# ---------------------------------------------------------------------------

class TestCapEnforced:

    @pytest.mark.asyncio
    async def test_hour_cap_blocks(self):
        client = _client_with_counts({"hour": 120, "day": 200, "week": 300, "month": 400})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, used, limit = await check_window_limits("owner:x", _settings())
        assert allowed is False
        assert window == "hour"
        assert used == 120 and limit == 120

    @pytest.mark.asyncio
    async def test_day_cap_blocks(self):
        client = _client_with_counts({"hour": 10, "day": 1_000, "week": 1_100, "month": 1_200})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, used, limit = await check_window_limits("owner:x", _settings())
        assert allowed is False
        assert window == "day"
        assert limit == 1_000

    @pytest.mark.asyncio
    async def test_month_cap_blocks(self):
        client = _client_with_counts({"hour": 10, "day": 50, "week": 100, "month": 8_000})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, used, limit = await check_window_limits("owner:x", _settings())
        assert allowed is False
        assert window == "month"
        assert limit == 8_000

    @pytest.mark.asyncio
    async def test_shortest_breached_window_reported_first(self):
        """When several windows are over cap, the shortest one wins so the
        Retry-After we hand back is the most actionable."""
        client = _client_with_counts({"hour": 999, "day": 9_999, "week": 9_999, "month": 9_999})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, _, _ = await check_window_limits("owner:x", _settings())
        assert allowed is False
        assert window == "hour"

    @pytest.mark.asyncio
    async def test_negative_cap_means_unlimited(self):
        """A negative env override disables that single window."""
        client = _client_with_counts({"hour": 10_000, "day": 5, "week": 5, "month": 5})
        s = _settings(free_cap_per_hour=-1)
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, _, _ = await check_window_limits("owner:x", s)
        assert allowed is True
        assert window is None

    @pytest.mark.asyncio
    async def test_at_cap_boundary_blocks(self):
        """used == limit is over cap (the limit-th op is the one rejected)."""
        client = _client_with_counts({"hour": 120, "day": 1, "week": 1, "month": 1})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, _, _ = await check_window_limits("owner:x", _settings())
        assert allowed is False
        assert window == "hour"

    @pytest.mark.asyncio
    async def test_one_below_cap_allowed(self):
        client = _client_with_counts({"hour": 119, "day": 1, "week": 1, "month": 1})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, _, _ = await check_window_limits("owner:x", _settings())
        assert allowed is True
        assert window is None


# ---------------------------------------------------------------------------
# Invariant — Redis errors must fail OPEN (never lock free users out)
# ---------------------------------------------------------------------------

class TestFailOpen:

    @pytest.mark.asyncio
    async def test_check_fails_open_on_redis_error(self):
        client = MagicMock()
        client.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            allowed, window, used, limit = await check_window_limits("owner:x", _settings())
        assert allowed is True
        assert window is None
        assert used == 0 and limit == 0

    @pytest.mark.asyncio
    async def test_check_fails_open_when_client_unavailable(self):
        with patch(
            "clsplusplus.window_limits._redis_client",
            side_effect=ConnectionError("no redis"),
        ):
            allowed, window, _, _ = await check_window_limits("owner:x", _settings())
        assert allowed is True
        assert window is None

    @pytest.mark.asyncio
    async def test_record_swallows_redis_error(self):
        """record_window_operation must never raise — metering can't block."""
        client = MagicMock()
        client.incr = AsyncMock(side_effect=ConnectionError("Redis down"))
        client.expire = AsyncMock()
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            await record_window_operation("owner:x", _settings())  # must not raise


# ---------------------------------------------------------------------------
# record_window_operation — increments all four windows
# ---------------------------------------------------------------------------

class TestRecord:

    @pytest.mark.asyncio
    async def test_record_increments_all_four_windows(self):
        client = _client_with_counts({})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            await record_window_operation("owner:x", _settings())
        assert client.incr.await_count == 4
        assert client.expire.await_count == 4
        incremented = {call.args[0] for call in client.incr.await_args_list}
        for period in ("hour", "day", "week", "month"):
            assert any(f":win:{period}:" in k for k in incremented)

    @pytest.mark.asyncio
    async def test_record_sets_ttl_longer_than_window(self):
        """Each TTL must exceed its window length so buckets self-evict
        only after the window they represent has fully elapsed."""
        client = _client_with_counts({})
        with patch("clsplusplus.window_limits._redis_client", return_value=client):
            await record_window_operation("owner:x", _settings())
        ttls = {}
        for call in client.expire.await_args_list:
            key, ttl = call.args[0], call.args[1]
            for period in ("hour", "day", "week", "month"):
                if f":win:{period}:" in key:
                    ttls[period] = ttl
        assert ttls["hour"] > 3600
        assert ttls["day"] > 86400
        assert ttls["week"] > 604800
        assert ttls["month"] > 86400 * 30


# ---------------------------------------------------------------------------
# retry_after_seconds
# ---------------------------------------------------------------------------

class TestRetryAfter:

    def test_retry_after_per_window(self):
        assert retry_after_seconds("hour") == 3600
        assert retry_after_seconds("day") == 86400
        assert retry_after_seconds("week") == 604800
        assert retry_after_seconds("month") == 2592000

    def test_retry_after_unknown_window_safe_default(self):
        assert retry_after_seconds(None) == 60
        assert retry_after_seconds("bogus") == 60


# ---------------------------------------------------------------------------
# Config — env-overridable free-tier caps
# ---------------------------------------------------------------------------

class TestConfigDefaults:

    def test_default_caps(self):
        s = Settings()
        assert s.free_cap_per_hour == 120
        assert s.free_cap_per_day == 1_000
        assert s.free_cap_per_week == 4_000
        assert s.free_cap_per_month == 8_000

    def test_caps_env_overridable(self, monkeypatch):
        monkeypatch.setenv("CLS_FREE_CAP_PER_HOUR", "5")
        monkeypatch.setenv("CLS_FREE_CAP_PER_DAY", "50")
        monkeypatch.setenv("CLS_FREE_CAP_PER_WEEK", "200")
        monkeypatch.setenv("CLS_FREE_CAP_PER_MONTH", "400")
        s = Settings()
        assert s.free_cap_per_hour == 5
        assert s.free_cap_per_day == 50
        assert s.free_cap_per_week == 200
        assert s.free_cap_per_month == 400


# ---------------------------------------------------------------------------
# Middleware integration — free tier gated, paid tiers untouched
# ---------------------------------------------------------------------------

class TestMiddlewareIntegration:

    @pytest.mark.asyncio
    async def test_free_tier_over_window_returns_429(self):
        from clsplusplus.middleware import QuotaMiddleware

        mw = QuotaMiddleware(app=MagicMock(), settings=_settings(), tier_resolver=None)

        request = MagicMock()
        request.url.path = "/v1/memory/write"
        request.method = "POST"
        request.state.api_key = "cls_live_freekey"

        async def _call_next(req):
            return MagicMock(status_code=200)

        with patch("clsplusplus.tiers.check_quota",
                   new=AsyncMock(return_value=(True, 5, 1_000))), \
             patch("clsplusplus.window_limits.check_window_limits",
                   new=AsyncMock(return_value=(False, "hour", 120, 120))), \
             patch("clsplusplus.usage.make_subject", return_value="key:abc"):
            resp = await mw.dispatch(request, _call_next)

        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "3600"

    @pytest.mark.asyncio
    async def test_free_tier_under_window_passes_and_records(self):
        from clsplusplus.middleware import QuotaMiddleware

        mw = QuotaMiddleware(app=MagicMock(), settings=_settings(), tier_resolver=None)

        request = MagicMock()
        request.url.path = "/v1/memory/write"
        request.method = "POST"
        request.state.api_key = "cls_live_freekey"

        async def _call_next(req):
            return MagicMock(status_code=201)

        record_mock = AsyncMock()
        with patch("clsplusplus.tiers.check_quota",
                   new=AsyncMock(return_value=(True, 5, 1_000))), \
             patch("clsplusplus.window_limits.check_window_limits",
                   new=AsyncMock(return_value=(True, None, 0, 0))), \
             patch("clsplusplus.window_limits.record_window_operation",
                   new=record_mock), \
             patch("clsplusplus.usage.make_subject", return_value="key:abc"):
            resp = await mw.dispatch(request, _call_next)

        assert resp.status_code == 201
        record_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_paid_tier_skips_window_check(self):
        """Pro tier must never consult the window limits — monthly cap only."""
        from clsplusplus.middleware import QuotaMiddleware
        from clsplusplus.tier_resolver import TierResolver

        store = MagicMock()
        store.resolve_tier_from_key = AsyncMock(return_value="pro")
        store.resolve_owner_email_from_key = AsyncMock(return_value="paid@example.com")
        resolver = TierResolver(store)

        mw = QuotaMiddleware(app=MagicMock(), settings=_settings(), tier_resolver=resolver)

        request = MagicMock()
        request.url.path = "/v1/memory/write"
        request.method = "POST"
        request.state.api_key = "cls_live_prokey"

        async def _call_next(req):
            return MagicMock(status_code=200)

        check_mock = AsyncMock(return_value=(False, "hour", 120, 120))
        with patch("clsplusplus.tiers.check_quota",
                   new=AsyncMock(return_value=(True, 5, 50_000))), \
             patch("clsplusplus.window_limits.check_window_limits", new=check_mock):
            resp = await mw.dispatch(request, _call_next)

        assert resp.status_code == 200
        check_mock.assert_not_awaited()
