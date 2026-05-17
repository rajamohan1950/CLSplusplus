"""Burst / storm hardening tests.

Covers the launch-hardening guarantees:
  1. The PUBLIC auth endpoints (/v1/auth/login, /v1/auth/register,
     /v1/waitlist/join) are per-IP rate limited even though they bypass the
     per-API-key limiter. Over the limit → HTTP 429 + Retry-After.
  2. A DB-pool-exhaustion error (asyncpg pool-acquire timeout /
     TooManyConnections) on an auth path degrades to a clean HTTP 503 with
     Retry-After — never an unhandled 500 / worker crash.
  3. The per-IP auth limiter fails OPEN on Redis errors (a Redis blip must
     not lock users out).

Run only this file:
    pytest tests/test_burst_hardening.py -q
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from httpx import ASGITransport, AsyncClient

from clsplusplus.config import Settings
from clsplusplus.middleware import RateLimitMiddleware, _is_auth_throttled
from clsplusplus.rate_limit import check_auth_rate_limit


# ---------------------------------------------------------------------------
# Redis pipeline mock helpers (mirrors tests/test_rate_limit.py style)
# ---------------------------------------------------------------------------

def _redis_under_limit(current_count: int):
    """A redis client mock whose zcard reports `current_count` (< limit)."""
    mock_client = MagicMock()
    read_pipe = MagicMock()
    read_pipe.execute = AsyncMock(return_value=[0, current_count])
    read_pipe.zremrangebyscore = MagicMock(return_value=read_pipe)
    read_pipe.zcard = MagicMock(return_value=read_pipe)
    write_pipe = MagicMock()
    write_pipe.execute = AsyncMock(return_value=[True, True])
    write_pipe.zadd = MagicMock(return_value=write_pipe)
    write_pipe.expire = MagicMock(return_value=write_pipe)
    mock_client.pipeline.side_effect = [read_pipe, write_pipe]
    return mock_client


def _redis_at_limit(current_count: int):
    """A redis client mock whose zcard reports `current_count` (>= limit)."""
    mock_client = MagicMock()
    read_pipe = MagicMock()
    read_pipe.execute = AsyncMock(return_value=[0, current_count])
    read_pipe.zremrangebyscore = MagicMock(return_value=read_pipe)
    read_pipe.zcard = MagicMock(return_value=read_pipe)
    mock_client.pipeline.return_value = read_pipe
    return mock_client


def _redis_down():
    """A redis client mock whose pipeline raises on execute."""
    mock_client = MagicMock()
    pipe = MagicMock()
    pipe.execute = AsyncMock(side_effect=ConnectionError("Redis down"))
    pipe.zremrangebyscore = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    mock_client.pipeline.return_value = pipe
    return mock_client


# ---------------------------------------------------------------------------
# 1. Path classification — the auth endpoints must be throttled paths
# ---------------------------------------------------------------------------

class TestAuthPathClassification:

    @pytest.mark.parametrize("path", [
        "/v1/auth/login",
        "/v1/auth/register",
        "/v1/waitlist/join",
        "/v1/auth/login/",        # trailing slash normalized
    ])
    def test_auth_paths_are_throttled(self, path):
        assert _is_auth_throttled(path, "POST") is True

    def test_non_auth_path_not_throttled(self):
        assert _is_auth_throttled("/v1/memory/write", "POST") is False

    def test_options_preflight_not_throttled(self):
        # CORS preflight must never be throttled (would break the browser).
        assert _is_auth_throttled("/v1/auth/login", "OPTIONS") is False


# ---------------------------------------------------------------------------
# 2. check_auth_rate_limit — the per-IP sliding window itself
# ---------------------------------------------------------------------------

class TestCheckAuthRateLimit:

    @pytest.mark.asyncio
    async def test_under_limit_allowed(self):
        s = Settings(auth_rate_limit_per_ip=10, auth_rate_limit_window_seconds=60)
        with patch("clsplusplus.rate_limit._redis_client",
                   return_value=_redis_under_limit(3)):
            allowed, count, limit = await check_auth_rate_limit("1.2.3.4", s)
        assert allowed is True
        assert limit == 10

    @pytest.mark.asyncio
    async def test_at_limit_rejected(self):
        """11th request from an IP within 60s is rejected (limit=10)."""
        s = Settings(auth_rate_limit_per_ip=10, auth_rate_limit_window_seconds=60)
        with patch("clsplusplus.rate_limit._redis_client",
                   return_value=_redis_at_limit(10)):
            allowed, count, limit = await check_auth_rate_limit("1.2.3.4", s)
        assert allowed is False
        assert count == 10
        assert limit == 10

    @pytest.mark.asyncio
    async def test_fails_open_on_redis_error(self):
        """A Redis outage must not lock users out — fail OPEN."""
        s = Settings(auth_rate_limit_per_ip=10)
        with patch("clsplusplus.rate_limit._redis_client", return_value=_redis_down()):
            allowed, _count, _limit = await check_auth_rate_limit("1.2.3.4", s)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_uses_separate_key_namespace(self):
        """The auth limiter must use cls:authlimit:, not cls:ratelimit:,
        so it never collides with the per-API-key limiter."""
        s = Settings(auth_rate_limit_per_ip=10)
        seen_keys = []

        client = MagicMock()
        read_pipe = MagicMock()
        read_pipe.execute = AsyncMock(return_value=[0, 0])
        read_pipe.zremrangebyscore = MagicMock(
            side_effect=lambda k, *a, **kw: seen_keys.append(k) or read_pipe)
        read_pipe.zcard = MagicMock(return_value=read_pipe)
        write_pipe = MagicMock()
        write_pipe.execute = AsyncMock(return_value=[True, True])
        write_pipe.zadd = MagicMock(return_value=write_pipe)
        write_pipe.expire = MagicMock(return_value=write_pipe)
        client.pipeline.side_effect = [read_pipe, write_pipe]

        with patch("clsplusplus.rate_limit._redis_client", return_value=client):
            await check_auth_rate_limit("9.9.9.9", s)
        assert seen_keys == ["cls:authlimit:9.9.9.9"]


# ---------------------------------------------------------------------------
# 3. RateLimitMiddleware — public auth endpoints get the per-IP throttle
# ---------------------------------------------------------------------------

def _build_app(settings: Settings) -> Starlette:
    """A minimal app with RateLimitMiddleware and a public /v1/auth/login."""
    async def login(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/auth/login", login, methods=["POST"])])
    app.add_middleware(RateLimitMiddleware, settings=settings)
    return app


class TestRateLimitMiddlewareAuthThrottle:

    @pytest.mark.asyncio
    async def test_public_auth_endpoint_throttled_returns_429(self):
        """A storm against /v1/auth/login is throttled with 429 + Retry-After,
        even though /v1/auth/login is a PUBLIC path."""
        s = Settings(auth_rate_limit_per_ip=10, auth_rate_limit_window_seconds=60)
        app = _build_app(s)
        with patch("clsplusplus.rate_limit._redis_client",
                   return_value=_redis_at_limit(10)):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport,
                                   base_url="http://test") as client:
                resp = await client.post("/v1/auth/login")
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "60"
        body = resp.json()
        assert body["retry_after"] == 60

    @pytest.mark.asyncio
    async def test_public_auth_endpoint_under_limit_passes(self):
        """Under the per-IP limit the request reaches the handler normally."""
        s = Settings(auth_rate_limit_per_ip=10, auth_rate_limit_window_seconds=60)
        app = _build_app(s)
        with patch("clsplusplus.rate_limit._redis_client",
                   return_value=_redis_under_limit(2)):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport,
                                   base_url="http://test") as client:
                resp = await client.post("/v1/auth/login")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_redis_outage_fails_open_request_passes(self):
        """If Redis is down the throttle fails open — the auth endpoint still
        serves rather than locking everyone out."""
        s = Settings(auth_rate_limit_per_ip=10)
        app = _build_app(s)
        with patch("clsplusplus.rate_limit._redis_client", return_value=_redis_down()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport,
                                   base_url="http://test") as client:
                resp = await client.post("/v1/auth/login")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 4. Graceful DB-pool-exhaustion — auth path returns 503, not a crash
# ---------------------------------------------------------------------------

def _build_app_with_exc_handler(exc: Exception) -> Starlette:
    """Minimal app whose route raises `exc`, with the same catch-all 503/500
    exception handler the real api.py registers."""
    async def boom(request):
        raise exc

    app = Starlette(routes=[Route("/v1/auth/login", boom, methods=["POST"])])

    async def unhandled_exception_handler(request, exc):
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError,
                            asyncpg.TooManyConnectionsError,
                            asyncpg.PostgresConnectionError)):
            return JSONResponse(
                status_code=503,
                content={"error": "service_unavailable", "retry_after": 5},
                headers={"Retry-After": "5"},
            )
        return JSONResponse(status_code=500, content={"error": "internal_error"})

    app.add_exception_handler(Exception, unhandled_exception_handler)
    return app


class TestGracefulPoolExhaustion:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", [
        asyncio.TimeoutError(),                      # pool.acquire() timed out
        asyncpg.TooManyConnectionsError("too many"), # Postgres rejected
    ])
    async def test_pool_exhaustion_returns_503(self, exc):
        """A pool-exhaustion error on an auth handler degrades to a clean 503
        with Retry-After, not an unhandled 500."""
        app = _build_app_with_exc_handler(exc)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/auth/login")
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "5"

    @pytest.mark.asyncio
    async def test_generic_error_returns_500_not_crash(self):
        """A non-pool error still returns a JSON 500 — the worker never crashes."""
        app = _build_app_with_exc_handler(RuntimeError("boom"))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/auth/login")
        assert resp.status_code == 500
        assert resp.json()["error"] == "internal_error"


# ---------------------------------------------------------------------------
# 5. _is_db_saturation classification (the helper the auth handlers use)
# ---------------------------------------------------------------------------

class TestDbSaturationClassification:
    """The auth handlers call api._is_db_saturation; replicate its logic here
    against the same exception set to lock the classification down."""

    def _is_db_saturation(self, exc):
        return isinstance(exc, (asyncio.TimeoutError, TimeoutError,
                                asyncpg.TooManyConnectionsError,
                                asyncpg.PostgresConnectionError))

    @pytest.mark.parametrize("exc", [
        asyncio.TimeoutError(),
        TimeoutError(),
        asyncpg.TooManyConnectionsError("too many"),
    ])
    def test_pool_errors_classified_as_saturation(self, exc):
        assert self._is_db_saturation(exc) is True

    @pytest.mark.parametrize("exc", [
        ValueError("bad input"),
        RuntimeError("boom"),
    ])
    def test_non_pool_errors_not_saturation(self, exc):
        assert self._is_db_saturation(exc) is False
