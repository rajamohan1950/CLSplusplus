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
        github_client_id="test-gh-client-id",
        github_client_secret="test-gh-client-secret",
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
# GitHub OAuth
# ---------------------------------------------------------------------------

def _extract_state_from_location(location: str) -> str:
    from urllib.parse import urlparse, parse_qs
    return parse_qs(urlparse(location).query)["state"][0]


def _make_github_fake_client(profile: dict, emails: list):
    """Build a stub replacement for `httpx.AsyncClient` used inside
    UserService.github_auth — it supports the `async with` protocol and the
    three calls the service makes (POST token, GET user, GET emails)."""

    class _Resp:
        def __init__(self, status_code: int, payload):
            self.status_code = status_code
            self._payload = payload
        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, data=None, headers=None, **_):
            if url.startswith("https://github.com/login/oauth/access_token"):
                return _Resp(200, {"access_token": "gh_test_token", "token_type": "bearer"})
            return _Resp(404, {})
        async def get(self, url, headers=None, **_):
            if url.endswith("/user"):
                return _Resp(200, profile)
            if url.endswith("/user/emails"):
                return _Resp(200, emails)
            return _Resp(404, {})

    return _FakeClient


class TestGitHubOAuth:
    @pytest.mark.asyncio
    async def test_github_redirect_returns_302_to_github(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as ac:
            resp = await ac.get("/v1/auth/github")
            assert resp.status_code == 307
            location = resp.headers.get("location", "")
            assert location.startswith(
                "https://github.com/login/oauth/authorize?"
            )
            assert "client_id=test-gh-client-id" in location
            assert "scope=read%3Auser+user%3Aemail" in location
            assert "state=" in location
            assert "redirect_uri=" in location

    @pytest.mark.asyncio
    async def test_github_not_configured_returns_501(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings(github_client_id="")
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v1/auth/github")
            assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_github_callback_missing_code(self):
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v1/auth/github/callback")
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_github_callback_rejects_invalid_state(self):
        """State not signed by us → 400, no user action taken."""
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/v1/auth/github/callback",
                params={"code": "anything", "state": "not-a-valid-jwt"},
            )
            assert resp.status_code == 400
            body = resp.json()
            msg = (body.get("detail") or body.get("message") or "").lower()
            assert "state" in msg

    @pytest.mark.asyncio
    async def test_github_callback_rejects_expired_state(self):
        """State JWT signed but expired → 400."""
        import jwt as pyjwt
        from datetime import datetime, timedelta, timezone
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        expired = pyjwt.encode(
            {
                "kind": "gh_oauth",
                "redirect": "/profile",
                "nonce": "abc",
                "iat": datetime.now(timezone.utc) - timedelta(hours=1),
                "exp": datetime.now(timezone.utc) - timedelta(minutes=30),
            },
            JWT_SECRET,
            algorithm="HS256",
        )
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/v1/auth/github/callback",
                params={"code": "anything", "state": expired},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_github_callback_rejects_wrong_kind_state(self):
        """A JWT signed by us but with wrong kind claim → 400."""
        import jwt as pyjwt
        from datetime import datetime, timedelta, timezone
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        wrong = pyjwt.encode(
            {
                "kind": "something_else",
                "redirect": "/profile",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
            },
            JWT_SECRET,
            algorithm="HS256",
        )
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/v1/auth/github/callback",
                params={"code": "anything", "state": wrong},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_github_callback_propagates_user_error(self):
        """OAuth error returned by GitHub (e.g. access_denied) surfaces as 400."""
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/v1/auth/github/callback",
                params={"error": "access_denied"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_github_happy_path_creates_user(self):
        """Mocked GitHub HTTP responses → new user created, session cookie set."""
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)

        # Get a valid state by calling the start endpoint
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as ac:
            start = await ac.get("/v1/auth/github")
            state = _extract_state_from_location(start.headers["location"])

        # Fake an in-memory user store so we don't need a real DB.
        fake_users: dict = {}

        class FakeStore:
            async def get_by_github_id(self, github_id):
                for u in fake_users.values():
                    if u.get("github_id") == github_id:
                        return u
                return None

            async def get_by_email(self, email):
                return fake_users.get(email)

            async def create_user(self, email, github_id=None, name="", avatar_url=None, **_):
                u = {
                    "id": f"user-{len(fake_users)+1}",
                    "email": email,
                    "github_id": github_id,
                    "name": name,
                    "avatar_url": avatar_url,
                    "tier": "free",
                    "is_admin": False,
                    "email_verified": False,
                }
                fake_users[email] = u
                return u

            async def update_github_id(self, user_id, github_id, avatar_url=None):
                for u in fake_users.values():
                    if u["id"] == user_id:
                        u["github_id"] = github_id
                        if avatar_url:
                            u["avatar_url"] = avatar_url
                        return True
                return False

            async def mark_email_verified(self, user_id):
                for u in fake_users.values():
                    if u["id"] == user_id:
                        u["email_verified"] = True

            async def get_by_id(self, user_id):
                for u in fake_users.values():
                    if u["id"] == user_id:
                        return u
                return None

        # user_service is closed over inside create_app(), so we patch the
        # two things it talks to: httpx.AsyncClient (for GitHub API calls)
        # and the UserStore methods (so no DB is required).
        from clsplusplus.stores import user_store as _us_mod

        fake_client = _make_github_fake_client(
            profile={
                "id": 424242,
                "login": "octotester",
                "name": "Octo Tester",
                "email": None,
                "avatar_url": "https://avatars.example/octo.png",
            },
            emails=[
                {"email": "secondary@example.com", "primary": False, "verified": True},
                {"email": "octo@example.com", "primary": True, "verified": True},
            ],
        )

        with patch("clsplusplus.user_service.httpx.AsyncClient", new=fake_client), \
             patch.object(_us_mod.UserStore, "get_by_github_id", FakeStore.get_by_github_id, create=True), \
             patch.object(_us_mod.UserStore, "get_by_email", FakeStore.get_by_email, create=True), \
             patch.object(_us_mod.UserStore, "create_user", FakeStore.create_user, create=True), \
             patch.object(_us_mod.UserStore, "update_github_id", FakeStore.update_github_id, create=True), \
             patch.object(_us_mod.UserStore, "mark_email_verified", FakeStore.mark_email_verified, create=True), \
             patch.object(_us_mod.UserStore, "get_by_id", FakeStore.get_by_id, create=True):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(
                transport=transport, base_url="http://test", follow_redirects=False
            ) as ac:
                resp = await ac.get(
                    "/v1/auth/github/callback",
                    params={"code": "fake_code", "state": state},
                )

        assert resp.status_code == 307, resp.text
        assert resp.headers["location"] == "/profile"
        # Session cookie was set
        cookies = resp.headers.get_list("set-cookie")
        assert any("cls_session=" in c for c in cookies), cookies
        # User was created with the GitHub primary verified email, linked by github_id
        created = fake_users.get("octo@example.com")
        assert created is not None
        assert created["github_id"] == "424242"
        assert created["email_verified"] is True

    @pytest.mark.asyncio
    async def test_github_second_signin_does_not_duplicate(self):
        """Signing in twice with the same GitHub id reuses the same user row."""
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)

        # Build a state
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as ac:
            start = await ac.get("/v1/auth/github")
            state1 = _extract_state_from_location(start.headers["location"])
            start2 = await ac.get("/v1/auth/github")
            state2 = _extract_state_from_location(start2.headers["location"])

        fake_users: dict = {}

        class _FS:
            @staticmethod
            async def get_by_github_id(self, github_id):
                for u in fake_users.values():
                    if u.get("github_id") == github_id:
                        return u
                return None
            @staticmethod
            async def get_by_email(self, email):
                return fake_users.get(email)
            @staticmethod
            async def create_user(self, email, github_id=None, name="", avatar_url=None, **_):
                u = {
                    "id": f"user-{len(fake_users)+1}",
                    "email": email,
                    "github_id": github_id,
                    "name": name,
                    "avatar_url": avatar_url,
                    "tier": "free",
                    "is_admin": False,
                    "email_verified": False,
                }
                fake_users[email] = u
                return u
            @staticmethod
            async def update_github_id(self, user_id, github_id, avatar_url=None):
                for u in fake_users.values():
                    if u["id"] == user_id:
                        u["github_id"] = github_id
                        return True
                return False
            @staticmethod
            async def mark_email_verified(self, user_id):
                for u in fake_users.values():
                    if u["id"] == user_id:
                        u["email_verified"] = True

        fake_client = _make_github_fake_client(
            profile={
                "id": 555,
                "login": "user555",
                "name": "User 555",
                "email": "user555@example.com",
                "avatar_url": "https://avatars.example/u.png",
            },
            emails=[],
        )
        from clsplusplus.stores import user_store as _us_mod
        with patch("clsplusplus.user_service.httpx.AsyncClient", new=fake_client), \
             patch.object(_us_mod.UserStore, "get_by_github_id", _FS.get_by_github_id, create=True), \
             patch.object(_us_mod.UserStore, "get_by_email", _FS.get_by_email, create=True), \
             patch.object(_us_mod.UserStore, "create_user", _FS.create_user, create=True), \
             patch.object(_us_mod.UserStore, "update_github_id", _FS.update_github_id, create=True), \
             patch.object(_us_mod.UserStore, "mark_email_verified", _FS.mark_email_verified, create=True):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(
                transport=transport, base_url="http://test", follow_redirects=False
            ) as ac:
                r1 = await ac.get(
                    "/v1/auth/github/callback",
                    params={"code": "c1", "state": state1},
                )
                r2 = await ac.get(
                    "/v1/auth/github/callback",
                    params={"code": "c2", "state": state2},
                )

        assert r1.status_code == 307
        assert r2.status_code == 307
        assert len(fake_users) == 1, fake_users

    @pytest.mark.asyncio
    async def test_github_links_to_existing_email_account(self):
        """Email-first user later signs in via GitHub → account linked, not duplicated."""
        from clsplusplus.api import create_app
        settings = _jwt_settings()
        app = create_app(settings)

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as ac:
            start = await ac.get("/v1/auth/github")
            state = _extract_state_from_location(start.headers["location"])

        # Pre-populate an email-only user.
        fake_users = {
            "linked@example.com": {
                "id": "user-existing",
                "email": "linked@example.com",
                "password_hash": "$2b$12$abc",
                "github_id": None,
                "google_id": None,
                "name": "Existing",
                "avatar_url": None,
                "tier": "free",
                "is_admin": False,
                "email_verified": False,
            }
        }

        class _FS:
            @staticmethod
            async def get_by_github_id(self, github_id):
                for u in fake_users.values():
                    if u.get("github_id") == github_id:
                        return u
                return None
            @staticmethod
            async def get_by_email(self, email):
                return fake_users.get(email)
            @staticmethod
            async def create_user(self, **kwargs):
                raise AssertionError("create_user should not be called when linking")
            @staticmethod
            async def update_github_id(self, user_id, github_id, avatar_url=None):
                for u in fake_users.values():
                    if u["id"] == user_id:
                        u["github_id"] = github_id
                        if avatar_url:
                            u["avatar_url"] = avatar_url
                        return True
                return False
            @staticmethod
            async def mark_email_verified(self, user_id):
                for u in fake_users.values():
                    if u["id"] == user_id:
                        u["email_verified"] = True

        fake_client = _make_github_fake_client(
            profile={
                "id": 999,
                "login": "linkeduser",
                "name": "Linked User",
                "email": "linked@example.com",
                "avatar_url": None,
            },
            emails=[],
        )
        from clsplusplus.stores import user_store as _us_mod
        with patch("clsplusplus.user_service.httpx.AsyncClient", new=fake_client), \
             patch.object(_us_mod.UserStore, "get_by_github_id", _FS.get_by_github_id, create=True), \
             patch.object(_us_mod.UserStore, "get_by_email", _FS.get_by_email, create=True), \
             patch.object(_us_mod.UserStore, "create_user", _FS.create_user, create=True), \
             patch.object(_us_mod.UserStore, "update_github_id", _FS.update_github_id, create=True), \
             patch.object(_us_mod.UserStore, "mark_email_verified", _FS.mark_email_verified, create=True):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(
                transport=transport, base_url="http://test", follow_redirects=False
            ) as ac:
                resp = await ac.get(
                    "/v1/auth/github/callback",
                    params={"code": "c", "state": state},
                )

        assert resp.status_code == 307
        linked = fake_users["linked@example.com"]
        assert linked["github_id"] == "999"
        assert linked["email_verified"] is True
        assert len(fake_users) == 1  # no duplicate

    @pytest.mark.asyncio
    async def test_github_auth_endpoint_is_public(self):
        """require_api_key=True must NOT block /v1/auth/github."""
        from clsplusplus.api import create_app
        settings = _jwt_settings(require_api_key=True)
        app = create_app(settings)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as ac:
            resp = await ac.get("/v1/auth/github")
            # Should redirect to github, NOT 401
            assert resp.status_code == 307
            assert "github.com" in resp.headers.get("location", "")


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
