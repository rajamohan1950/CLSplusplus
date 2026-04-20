"""Tests for tier definitions, quota enforcement, and QuotaMiddleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.config import Settings
from clsplusplus.tiers import Tier, TIER_LIMITS, get_tier, get_limits, check_quota, get_quota_status

VALID_API_KEY = "cls_live_test1234567890123456789012"


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

class TestTierDefinitions:
    def test_all_tiers_defined(self):
        assert set(TIER_LIMITS.keys()) == {Tier.free, Tier.pro, Tier.business, Tier.enterprise}

    def test_free_limits(self):
        lim = TIER_LIMITS[Tier.free]
        assert lim["ops_per_month"] == 1_000
        assert lim["max_items"] == 500
        assert lim["max_namespaces"] == 1
        assert lim["rate_limit_requests"] == 20

    def test_business_has_concrete_limits(self):
        lim = TIER_LIMITS[Tier.business]
        assert lim["ops_per_month"] == 200_000
        assert lim["max_namespaces"] == 50

    def test_enterprise_limits(self):
        lim = TIER_LIMITS[Tier.enterprise]
        assert lim["ops_per_month"] == 1_000_000
        assert lim["max_namespaces"] == 500

    def test_get_tier_valid(self):
        s = Settings(tier="pro")
        assert get_tier(s) == Tier.pro

    def test_get_tier_invalid_falls_back_to_free(self):
        s = Settings(tier="nonexistent")
        assert get_tier(s) == Tier.free

    def test_get_limits(self):
        lim = get_limits(Tier.pro)
        assert lim["ops_per_month"] == 50_000


# ---------------------------------------------------------------------------
# Quota check
# ---------------------------------------------------------------------------

class TestCheckQuota:
    @pytest.mark.asyncio
    async def test_enterprise_high_quota(self):
        s = Settings(track_usage=True)
        with patch("clsplusplus.usage.get_operation_count", new_callable=AsyncMock, return_value=500_000):
            allowed, usage, limit = await check_quota("any-key", Tier.enterprise, s)
            assert allowed is True
            assert limit == 1_000_000

    @pytest.mark.asyncio
    async def test_under_quota_allowed(self):
        s = Settings(track_usage=True)
        with patch("clsplusplus.usage.get_operation_count", new_callable=AsyncMock, return_value=500):
            allowed, usage, limit = await check_quota("key", Tier.free, s)
            assert allowed is True
            assert usage == 500
            assert limit == 1_000

    @pytest.mark.asyncio
    async def test_at_quota_blocked(self):
        s = Settings(track_usage=True)
        with patch("clsplusplus.usage.get_operation_count", new_callable=AsyncMock, return_value=1_000):
            allowed, usage, limit = await check_quota("key", Tier.free, s)
            assert allowed is False
            assert usage == 1_000

    @pytest.mark.asyncio
    async def test_over_quota_blocked(self):
        s = Settings(track_usage=True)
        with patch("clsplusplus.usage.get_operation_count", new_callable=AsyncMock, return_value=1_500):
            allowed, usage, limit = await check_quota("key", Tier.free, s)
            assert allowed is False


# ---------------------------------------------------------------------------
# Quota status (for /v1/usage response)
# ---------------------------------------------------------------------------

class TestGetQuotaStatus:
    @pytest.mark.asyncio
    async def test_returns_tier_info(self):
        s = Settings(track_usage=True)
        with patch("clsplusplus.usage.get_operation_count", new_callable=AsyncMock, return_value=42), \
             patch("clsplusplus.usage.get_usage", new_callable=AsyncMock, return_value={"writes": 20, "reads": 22, "period": "2026-04"}):
            status = await get_quota_status("key", Tier.free, s)
            assert status["tier"] == "free"
            assert status["operations"] == 42
            assert status["operations_limit"] == 1_000
            assert status["writes"] == 20
            assert status["reads"] == 22
            assert status["namespaces_limit"] == 1
            assert status["storage_limit"] == 500
            assert status["rate_limit"] == 20


# ---------------------------------------------------------------------------
# QuotaMiddleware integration
# ---------------------------------------------------------------------------

class TestQuotaMiddleware:
    @pytest.mark.asyncio
    async def test_quota_disabled_passes_through(self):
        """When enforce_quotas=False, no blocking happens."""
        from clsplusplus.api import create_app
        settings = Settings(require_api_key=False, enforce_quotas=False)
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_quota_exceeded_returns_402(self):
        """When quota is exceeded, POST to protected route returns 402."""
        from clsplusplus.api import create_app
        settings = Settings(
            require_api_key=True,
            api_keys=VALID_API_KEY,
            enforce_quotas=True,
            track_usage=True,
            tier="free",
        )
        app = create_app(settings)
        transport = ASGITransport(app=app)

        with patch("clsplusplus.usage.get_operation_count", new_callable=AsyncMock, return_value=2_000):
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            ) as ac:
                resp = await ac.post(
                    "/v1/memory/write",
                    json={"text": "test fact", "namespace": "default"},
                )
                assert resp.status_code == 402
                body = resp.json()
                assert body["tier"] == "free"
                assert body["limit"] == 1_000

    @pytest.mark.asyncio
    async def test_get_requests_not_metered(self):
        """GET requests bypass quota check even when enforcement is on."""
        from clsplusplus.api import create_app
        settings = Settings(
            require_api_key=True,
            api_keys=VALID_API_KEY,
            enforce_quotas=True,
            track_usage=True,
            tier="free",
        )
        app = create_app(settings)
        transport = ASGITransport(app=app)

        with patch("clsplusplus.usage.get_operation_count", new_callable=AsyncMock, return_value=2_000):
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            ) as ac:
                resp = await ac.get("/v1/usage")
                # Should not be 402 — GETs are not metered
                assert resp.status_code != 402


# ---------------------------------------------------------------------------
# Usage counter functions
# ---------------------------------------------------------------------------

class TestUsageCounter:
    @pytest.mark.asyncio
    async def test_record_and_get_operation(self, in_memory_store):
        """record_operation increments, get_operation_count reads it back."""
        from clsplusplus.usage import record_operation, get_operation_count

        settings = Settings(track_usage=True)
        with patch("clsplusplus.usage._redis_client", return_value=in_memory_store):
            await record_operation("test-key", settings)
            await record_operation("test-key", settings)
            await record_operation("test-key", settings)

            count = await get_operation_count("test-key", settings)
            assert count == 3

    @pytest.mark.asyncio
    async def test_record_operation_noop_when_tracking_disabled(self, in_memory_store):
        """When track_usage=False, nothing is recorded."""
        from clsplusplus.usage import record_operation, get_operation_count

        settings = Settings(track_usage=False)
        with patch("clsplusplus.usage._redis_client", return_value=in_memory_store):
            await record_operation("test-key", settings)
            count = await get_operation_count("test-key", settings)
            assert count == 0
