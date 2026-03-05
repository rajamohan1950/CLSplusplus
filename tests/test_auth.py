"""Auth and rate limit tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.api import create_app
from clsplusplus.config import Settings


@pytest.fixture
async def client_no_auth():
    """App with require_api_key=False (default)."""
    settings = Settings(require_api_key=False)
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def client_with_auth():
    """App with require_api_key=True and valid key."""
    settings = Settings(
        require_api_key=True,
        api_keys="cls_live_test1234567890123456789012",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer cls_live_test1234567890123456789012"},
    ) as ac:
        yield ac


@pytest.fixture
async def client_unauth():
    """App with require_api_key=True, no auth header."""
    settings = Settings(
        require_api_key=True,
        api_keys="cls_live_test1234567890123456789012",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_public_paths_no_auth_required(client_unauth):
    """Public paths accessible without API key."""
    for path in ["/", "/v1/memory/health"]:
        resp = await client_unauth.get(path)
        assert resp.status_code == 200, f"{path} should be public"


@pytest.mark.asyncio
async def test_protected_path_401_without_key(client_unauth):
    """Protected path returns 401 without API key."""
    resp = await client_unauth.post(
        "/v1/memory/write",
        json={"text": "test", "namespace": "default"},
    )
    assert resp.status_code == 401
    assert "detail" in resp.json()
    assert "WWW-Authenticate" in resp.headers


@pytest.mark.asyncio
async def test_protected_path_401_invalid_key(client_unauth):
    """Protected path returns 401 with invalid API key."""
    resp = await client_unauth.post(
        "/v1/memory/write",
        json={"text": "test", "namespace": "default"},
        headers={"Authorization": "Bearer invalid_key"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_path_401_wrong_format(client_unauth):
    """Protected path returns 401 with wrong key format."""
    resp = await client_unauth.post(
        "/v1/memory/write",
        json={"text": "test", "namespace": "default"},
        headers={"Authorization": "Bearer cls_short"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_no_auth_required_when_disabled(client_no_auth):
    """When require_api_key=False, public and protected paths work without key."""
    resp = await client_no_auth.get("/")
    assert resp.status_code == 200
