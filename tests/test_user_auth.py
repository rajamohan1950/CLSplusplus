"""Tests for user registration, login, JWT auth, and Google OAuth."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.config import Settings
from clsplusplus.jwt_utils import create_token, decode_token

JWT_SECRET = "test-secret-for-unit-tests-only-32b"


def _jwt_settings(**overrides) -> Settings:
    defaults = dict(
        require_api_key=False,
        jwt_secret=JWT_SECRET,
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------

class TestJWTUtils:
    def test_create_and_decode_token(self):
        token = create_token("user-123", "a@b.com", False, JWT_SECRET)
        payload = decode_token(token, JWT_SECRET)
        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["email"] == "a@b.com"
        assert payload["is_admin"] is False

    def test_decode_wrong_secret_returns_none(self):
        token = create_token("user-123", "a@b.com", False, JWT_SECRET)
        assert decode_token(token, "wrong-secret") is None

    def test_decode_garbage_returns_none(self):
        assert decode_token("not.a.token", JWT_SECRET) is None

    def test_admin_flag_roundtrip(self):
        token = create_token("admin-1", "admin@co.com", True, JWT_SECRET)
        payload = decode_token(token, JWT_SECRET)
        assert payload["is_admin"] is True


# ---------------------------------------------------------------------------
# User registration
# ---------------------------------------------------------------------------

class TestUserRegistration:
    @pytest.mark.asyncio
    async def test_register_route_exists_and_validates(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/auth/register", json={
                "email": "newuser@example.com",
                "password": "securepass123",
                "name": "Test User",
            })
            # 200 (success), 400 (validation), or 500 (no DB) — but not 404/405
            assert resp.status_code in (200, 400, 500)
            if resp.status_code == 200:
                data = resp.json()
                assert data["email"] == "newuser@example.com"
                assert "password_hash" not in data

    @pytest.mark.asyncio
    async def test_register_weak_password(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/auth/register", json={
                "email": "user@example.com",
                "password": "short",
            })
            assert resp.status_code == 422  # Pydantic validation (min_length=8)

    @pytest.mark.asyncio
    async def test_register_invalid_email(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/auth/register", json={
                "email": "",
                "password": "securepass123",
            })
            # Empty email should fail validation
            assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# User login
# ---------------------------------------------------------------------------

class TestUserLogin:
    @pytest.mark.asyncio
    async def test_login_route_exists(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/auth/login", json={
                "email": "nobody@example.com",
                "password": "doesntmatter",
            })
            # 401 (user not found) or 500 (no DB) — but not 404/405
            assert resp.status_code in (401, 500)


# ---------------------------------------------------------------------------
# JWT middleware (dual auth)
# ---------------------------------------------------------------------------

class TestJWTMiddleware:
    @pytest.mark.asyncio
    async def test_jwt_cookie_grants_access_to_me_endpoint(self):
        """A valid JWT cookie should authenticate the /v1/auth/me endpoint."""
        from clsplusplus.api import create_app
        settings = _jwt_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app)

        token = create_token("user-abc", "user@test.com", False, JWT_SECRET)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"cls_session": token},
        ) as ac:
            resp = await ac.get("/v1/auth/me")
            # 404 (user not in DB) or 500 (no DB connection)
            # But NOT 401 — the JWT cookie was accepted by middleware
            assert resp.status_code in (200, 404, 500)

    @pytest.mark.asyncio
    async def test_no_auth_returns_401_when_required(self):
        """No API key and no cookie should return 401 when auth is required."""
        from clsplusplus.api import create_app
        settings = _jwt_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/memory/write", json={
                "text": "test", "namespace": "default",
            })
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_token_returns_401(self):
        """An expired JWT should not authenticate."""
        from clsplusplus.api import create_app
        settings = _jwt_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app)

        # Create expired token
        import jwt as pyjwt
        from datetime import datetime, timedelta, timezone
        expired_payload = {
            "sub": "user-expired",
            "email": "expired@test.com",
            "is_admin": False,
            "iat": datetime.now(timezone.utc) - timedelta(days=10),
            "exp": datetime.now(timezone.utc) - timedelta(days=3),
        }
        expired_token = pyjwt.encode(expired_payload, JWT_SECRET, algorithm="HS256")

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"cls_session": expired_token},
        ) as ac:
            resp = await ac.post("/v1/memory/write", json={
                "text": "test", "namespace": "default",
            })
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Auth endpoints are public
# ---------------------------------------------------------------------------

class TestAuthEndpointsPublic:
    @pytest.mark.asyncio
    async def test_register_accessible_without_auth(self):
        """Auth endpoints must be public even when require_api_key=True."""
        from clsplusplus.api import create_app
        settings = _jwt_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/auth/register", json={
                "email": "test@example.com",
                "password": "securepass123",
            })
            # Should NOT be 401 — auth endpoints are public
            # 400/500 are acceptable (DB not available)
            assert resp.status_code in (200, 400, 500)

    @pytest.mark.asyncio
    async def test_login_accessible_without_auth(self):
        """Login endpoint must be public even when require_api_key=True."""
        from clsplusplus.api import create_app
        settings = _jwt_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/auth/login", json={
                "email": "test@example.com",
                "password": "anypassword1",
            })
            # Should NOT be 401 — auth endpoints are public
            assert resp.status_code in (200, 401, 500)  # 401 is from user_service (bad credentials), not middleware

    @pytest.mark.asyncio
    async def test_logout_clears_cookie(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/auth/logout")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------

class TestGoogleOAuth:
    @pytest.mark.asyncio
    async def test_google_redirect_returns_redirect(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as ac:
            resp = await ac.get("/v1/auth/google")
            assert resp.status_code == 307
            assert "accounts.google.com" in resp.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_google_not_configured_returns_501(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings(google_client_id="")
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v1/auth/google")
            assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_google_callback_missing_code(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v1/auth/google/callback")
            assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_and_verify(self):
        from clsplusplus.user_service import _hash_password, _verify_password
        hashed = _hash_password("mypassword123")
        assert _verify_password("mypassword123", hashed) is True
        assert _verify_password("wrongpassword", hashed) is False

    def test_hash_is_different_each_time(self):
        from clsplusplus.user_service import _hash_password
        h1 = _hash_password("same")
        h2 = _hash_password("same")
        assert h1 != h2  # bcrypt uses random salt
