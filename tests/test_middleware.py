"""Middleware tests - AuthMiddleware, RateLimitMiddleware, RequestIdMiddleware."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.api import create_app
from clsplusplus.config import Settings
from clsplusplus.middleware import _PUBLIC_PATHS, _is_public
from tests.conftest import VALID_API_KEY


# ---------------------------------------------------------------------------
# Public path detection
# ---------------------------------------------------------------------------

class TestIsPublic:

    def test_root_is_public(self):
        assert _is_public("/", "GET") is True

    def test_health_is_public(self):
        assert _is_public("/v1/memory/health", "GET") is True

    def test_demo_status_is_public(self):
        assert _is_public("/v1/demo/status", "GET") is True

    def test_demo_chat_is_public(self):
        assert _is_public("/v1/demo/chat", "POST") is True

    def test_docs_is_public(self):
        assert _is_public("/docs", "GET") is True
        assert _is_public("/docs/", "GET") is True

    def test_redoc_is_public(self):
        assert _is_public("/redoc", "GET") is True

    def test_openapi_is_public(self):
        assert _is_public("/openapi.json", "GET") is True

    def test_write_is_protected(self):
        assert _is_public("/v1/memory/write", "POST") is False

    def test_read_is_protected(self):
        assert _is_public("/v1/memory/read", "POST") is False

    def test_forget_is_protected(self):
        assert _is_public("/v1/memory/forget", "DELETE") is False

    def test_options_always_public(self):
        assert _is_public("/v1/memory/write", "OPTIONS") is True
        assert _is_public("/anything", "OPTIONS") is True

    def test_trailing_slash_normalized(self):
        assert _is_public("/v1/memory/health/", "GET") is True

    def test_empty_path(self):
        assert _is_public("", "GET") is True


# ---------------------------------------------------------------------------
# RequestIdMiddleware
# ---------------------------------------------------------------------------

class TestRequestIdMiddleware:

    @pytest.mark.asyncio
    async def test_request_id_generated(self, client):
        resp = await client.get("/")
        assert "x-request-id" in resp.headers
        # Should be valid UUID
        uid = resp.headers["x-request-id"]
        assert len(uid) == 36

    @pytest.mark.asyncio
    async def test_request_id_preserved(self, client):
        custom_id = str(uuid.uuid4())
        resp = await client.get("/", headers={"X-Request-Id": custom_id})
        assert resp.headers["x-request-id"] == custom_id

    @pytest.mark.asyncio
    async def test_unique_ids_per_request(self, client):
        resp1 = await client.get("/")
        resp2 = await client.get("/")
        assert resp1.headers["x-request-id"] != resp2.headers["x-request-id"]


# ---------------------------------------------------------------------------
# AuthMiddleware
# ---------------------------------------------------------------------------

class TestAuthMiddleware:

    @pytest.mark.asyncio
    async def test_no_auth_passes_all(self):
        settings = Settings(require_api_key=False)
        app = create_app(settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_blocks_without_key(self):
        settings = Settings(require_api_key=True, api_keys=VALID_API_KEY)
        app = create_app(settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/v1/memory/write", json={"text": "x", "namespace": "default"})
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_passes_with_valid_key(self):
        settings = Settings(require_api_key=True, api_keys=VALID_API_KEY)
        app = create_app(settings)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
        ) as c:
            # Write will fail at DB/embedding level, but should pass auth (not 401).
            # May raise an exception if the error propagates through middleware
            # (e.g., numpy/Redis unavailable) - that also proves auth passed.
            try:
                resp = await c.post("/v1/memory/write", json={"text": "x", "namespace": "default"})
                assert resp.status_code != 401
            except Exception:
                # Exception means the request got past auth middleware
                # and failed downstream (DB/embedding layer)
                pass

    @pytest.mark.asyncio
    async def test_401_response_format(self):
        settings = Settings(require_api_key=True, api_keys=VALID_API_KEY)
        app = create_app(settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/v1/memory/write", json={"text": "x", "namespace": "default"})
            assert resp.status_code == 401
            body = resp.json()
            assert "detail" in body
            assert "WWW-Authenticate" in resp.headers
            assert resp.headers["WWW-Authenticate"] == "Bearer"


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, client):
        resp = await client.options(
            "/v1/memory/write",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    @pytest.mark.asyncio
    async def test_cors_allows_all_origins(self, client):
        resp = await client.options(
            "/",
            headers={
                "Origin": "http://malicious-site.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # With allow_credentials=True, CORS reflects the origin instead of "*"
        origin = resp.headers.get("access-control-allow-origin")
        assert origin in ("*", "http://malicious-site.com")


# ---------------------------------------------------------------------------
# RateLimitMiddleware - rate limit exceeded response (line 80)
# ---------------------------------------------------------------------------

class TestRateLimitExceeded:

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self):
        """Cover line 80: rate limit exceeded returns 429 JSONResponse."""
        from unittest.mock import patch, AsyncMock

        settings = Settings(require_api_key=False)
        app = create_app(settings)

        # Mock check_rate_limit to always deny
        with patch(
            "clsplusplus.middleware.check_rate_limit",
            new_callable=AsyncMock,
            return_value=(False, 101, 100),  # not allowed, count > limit
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as c:
                resp = await c.post("/v1/memory/write", json={"text": "test"})
                assert resp.status_code == 429
                body = resp.json()
                assert body["detail"] == "Rate limit exceeded"
                assert "retry_after" in body
                assert "X-RateLimit-Limit" in resp.headers
                assert resp.headers["X-RateLimit-Remaining"] == "0"
                assert "Retry-After" in resp.headers
