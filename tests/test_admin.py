"""Tests for admin dashboard routes and metrics."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.config import Settings
from clsplusplus.jwt_utils import create_token

JWT_SECRET = "test-secret-for-admin-tests-32bit"


def _admin_settings(**overrides) -> Settings:
    defaults = dict(
        require_api_key=False,
        jwt_secret=JWT_SECRET,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _admin_token():
    return create_token("admin-001", "admin@test.com", True, JWT_SECRET)


def _user_token():
    return create_token("user-001", "user@test.com", False, JWT_SECRET)


# ---------------------------------------------------------------------------
# Admin route protection
# ---------------------------------------------------------------------------

class TestAdminProtection:
    @pytest.mark.asyncio
    async def test_admin_route_401_without_auth(self):
        from clsplusplus.api import create_app
        settings = _admin_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/admin/metrics/summary")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_route_403_for_non_admin(self):
        from clsplusplus.api import create_app
        settings = _admin_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app)
        token = _user_token()
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"cls_session": token},
        ) as ac:
            resp = await ac.get("/admin/metrics/summary")
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_route_accessible_for_admin(self):
        from clsplusplus.api import create_app
        settings = _admin_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        token = _admin_token()
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"cls_session": token},
        ) as ac:
            resp = await ac.get("/admin/metrics/summary")
            # 200 or 500 (DB not available) — but NOT 401 or 403
            assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# Admin endpoints exist
# ---------------------------------------------------------------------------

class TestAdminEndpoints:
    @pytest.mark.asyncio
    async def test_all_admin_endpoints_exist(self):
        from clsplusplus.api import create_app
        settings = _admin_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        token = _admin_token()

        endpoints = [
            "/admin/metrics/summary",
            "/admin/metrics/signups",
            "/admin/metrics/revenue",
            "/admin/metrics/operations",
            "/admin/metrics/users",
        ]

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"cls_session": token},
        ) as ac:
            for endpoint in endpoints:
                resp = await ac.get(endpoint)
                # 200 or 500 (DB not available) — but NOT 404
                assert resp.status_code in (200, 500), f"{endpoint} returned {resp.status_code}"


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_compute_cost_basic(self):
        from clsplusplus.cost_model import compute_cost
        metrics = {"write": 100, "read": 200, "embedding": 50}
        cost = compute_cost(metrics)
        expected = 100 * 0.0001 + 200 * 0.00005 + 50 * 0.0002
        assert abs(cost - expected) < 0.0001

    def test_compute_cost_empty(self):
        from clsplusplus.cost_model import compute_cost
        assert compute_cost({}) == 0.0

    def test_compute_cost_unknown_metric(self):
        from clsplusplus.cost_model import compute_cost
        # Unknown metrics should be ignored (cost = 0)
        assert compute_cost({"unknown_metric": 1000}) == 0.0


# ---------------------------------------------------------------------------
# Metrics emitter
# ---------------------------------------------------------------------------

class TestMetricsEmitter:
    @pytest.mark.asyncio
    async def test_emit_and_get(self, in_memory_store):
        from clsplusplus.metrics import MetricsEmitter
        settings = Settings()
        emitter = MetricsEmitter(settings)
        with patch("clsplusplus.metrics._redis_client", return_value=in_memory_store):
            await emitter.emit("user-123", "write", 5)
            await emitter.emit("user-123", "read", 3)
            metrics = await emitter.get_user_metrics("user-123")
            assert metrics["write"] == 5
            assert metrics["read"] == 3

    @pytest.mark.asyncio
    async def test_emit_never_crashes(self):
        """Emit must be fire-and-forget — never raise."""
        from clsplusplus.metrics import MetricsEmitter
        settings = Settings(redis_url="redis://nonexistent:9999")
        emitter = MetricsEmitter(settings)
        # This should not raise even with bad Redis URL
        await emitter.emit("user-123", "write")


# ---------------------------------------------------------------------------
# Tier prices
# ---------------------------------------------------------------------------

class TestTierPrices:
    def test_prices_defined(self):
        from clsplusplus.tiers import Tier, TIER_PRICES
        assert TIER_PRICES[Tier.free] == 0.0
        assert TIER_PRICES[Tier.pro] == 9.0
        assert TIER_PRICES[Tier.business] == 29.0
        assert TIER_PRICES[Tier.enterprise] == 149.0
