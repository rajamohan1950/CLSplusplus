"""Tests for the one-click MCP connect endpoint: GET /v1/mcp/connect.

The endpoint mints a fresh API key for the authenticated user (reusing the
integration-service key-creation path) and returns a ready-to-paste
``mcpServers`` config block plus a one-line ``claude mcp add`` command, both
pre-filled with the new key and pointed at the production API.

Stores are mocked at the service boundary — no real Postgres needed.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.config import Settings
from clsplusplus.jwt_utils import create_token
from clsplusplus.models import ApiKeyResponse, IntegrationResponse


pytestmark = pytest.mark.asyncio

_JWT_SECRET = "test-secret-mcp-connect-padding-0123456789"
_USER_ID = "abcd1234efgh5678"
_USER_EMAIL = "raj@example.com"


def _fake_key(integration_id: str = "intg-1") -> ApiKeyResponse:
    now = datetime.now(timezone.utc)
    return ApiKeyResponse(
        id="key-1",
        integration_id=integration_id,
        key_prefix="cls_live",
        key_hint="ab12",
        scopes=["memories:read", "memories:write"],
        label="MCP one-click connect",
        status="active",
        created_at=now,
        key="cls_live_FAKEKEY1234567890ABCDEF",
    )


def _fake_integration() -> IntegrationResponse:
    now = datetime.now(timezone.utc)
    return IntegrationResponse(
        id="intg-1",
        name="CLS++ MCP",
        description="",
        namespace=f"user-{_USER_ID[:8]}",
        status="active",
        owner_email=_USER_EMAIL,
        created_at=now,
        updated_at=now,
    )


def _build_app():
    from clsplusplus.api import create_app

    settings = Settings(require_api_key=False, jwt_secret=_JWT_SECRET)
    return create_app(settings)


def _auth_cookie() -> dict:
    token = create_token(_USER_ID, _USER_EMAIL, False, _JWT_SECRET)
    return {"cls_session": token}


async def _get(app, cookies=None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.get("/v1/mcp/connect", cookies=cookies or {})


async def test_unauthenticated_request_is_rejected():
    """No session cookie → 401, no key minted."""
    resp = await _get(_build_app())
    assert resp.status_code == 401


async def test_connect_mints_key_for_new_user():
    """A user with no integration gets one created plus a fresh key."""
    app = _build_app()

    with patch(
        "clsplusplus.user_service.UserService.get_user",
        new=AsyncMock(return_value={"id": _USER_ID, "email": _USER_EMAIL}),
    ), patch(
        "clsplusplus.integration_service.IntegrationService.list_all",
        new=AsyncMock(return_value=[]),
    ), patch(
        "clsplusplus.integration_service.IntegrationService.register",
        new=AsyncMock(return_value=(_fake_integration(), _fake_key())),
    ) as mock_register, patch(
        "clsplusplus.integration_service.IntegrationService.create_key",
        new=AsyncMock(),
    ) as mock_create_key:
        resp = await _get(app, cookies=_auth_cookie())

    assert resp.status_code == 200
    # New user → register path used, not create_key.
    mock_register.assert_awaited_once()
    mock_create_key.assert_not_awaited()


async def test_connect_reuses_existing_integration():
    """A user with an active integration gets a fresh key on it (no new integration)."""
    app = _build_app()

    with patch(
        "clsplusplus.user_service.UserService.get_user",
        new=AsyncMock(return_value={"id": _USER_ID, "email": _USER_EMAIL}),
    ), patch(
        "clsplusplus.integration_service.IntegrationService.list_all",
        new=AsyncMock(return_value=[_fake_integration()]),
    ), patch(
        "clsplusplus.integration_service.IntegrationService.register",
        new=AsyncMock(),
    ) as mock_register, patch(
        "clsplusplus.integration_service.IntegrationService.create_key",
        new=AsyncMock(return_value=_fake_key()),
    ) as mock_create_key:
        resp = await _get(app, cookies=_auth_cookie())

    assert resp.status_code == 200
    mock_register.assert_not_awaited()
    mock_create_key.assert_awaited_once()


async def test_connect_returns_production_config_block():
    """Response carries a paste-ready mcpServers block + install command, prod URL, real key."""
    app = _build_app()

    with patch(
        "clsplusplus.user_service.UserService.get_user",
        new=AsyncMock(return_value={"id": _USER_ID, "email": _USER_EMAIL}),
    ), patch(
        "clsplusplus.integration_service.IntegrationService.list_all",
        new=AsyncMock(return_value=[_fake_integration()]),
    ), patch(
        "clsplusplus.integration_service.IntegrationService.create_key",
        new=AsyncMock(return_value=_fake_key()),
    ):
        resp = await _get(app, cookies=_auth_cookie())

    assert resp.status_code == 200
    body = resp.json()

    # Points at production, never the old onrender host.
    assert body["api_url"] == "https://www.clsplusplus.com"

    # The freshly minted key is surfaced (shown once).
    assert body["api_key"] == "cls_live_FAKEKEY1234567890ABCDEF"

    # mcpServers block is ready to paste into .claude/settings.json.
    server = body["mcp_config"]["mcpServers"]["cls-memory"]
    assert server["command"] == "python3"
    assert server["args"] == ["-m", "clsplusplus.mcp_server"]
    assert server["env"]["CLS_API_URL"] == "https://www.clsplusplus.com"
    assert server["env"]["CLS_API_KEY"] == "cls_live_FAKEKEY1234567890ABCDEF"

    # One-line CLI install carries the key and prod URL.
    cmd = body["install_command"]
    assert cmd.startswith("claude mcp add cls-memory")
    assert "CLS_API_URL=https://www.clsplusplus.com" in cmd
    assert "CLS_API_KEY=cls_live_FAKEKEY1234567890ABCDEF" in cmd
    assert cmd.endswith("-- python3 -m clsplusplus.mcp_server")
