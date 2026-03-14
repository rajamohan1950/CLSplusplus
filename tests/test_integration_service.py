"""Tests for CLS++ Integration Service — self-service integration management.

Tests cover: integration CRUD, API key lifecycle, webhook subscriptions,
audit events, scope validation, key rotation/revocation, and edge cases.

All tests mock the PostgreSQL layer (IntegrationStore) to avoid external deps.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clsplusplus.config import Settings
from clsplusplus.integration_service import (
    VALID_EVENTS,
    VALID_SCOPES,
    IntegrationService,
)
from clsplusplus.models import (
    ApiKeyCreate,
    ApiKeyResponse,
    IntegrationCreate,
    IntegrationEventResponse,
    IntegrationResponse,
    WebhookCreate,
    WebhookResponse,
)
from clsplusplus.stores.integration_store import (
    IntegrationStore,
    _generate_api_key,
    _generate_webhook_secret,
    _key_prefix,
    _mask_key,
    _sha256_hex,
)


# ============================================================================
# Utility function tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for store utility functions."""

    def test_sha256_hex_deterministic(self):
        assert _sha256_hex("test") == _sha256_hex("test")
        assert _sha256_hex("a") != _sha256_hex("b")

    def test_sha256_hex_length(self):
        result = _sha256_hex("any string")
        assert len(result) == 64  # SHA-256 hex = 64 chars

    def test_generate_api_key_format(self):
        key = _generate_api_key("live")
        assert key.startswith("cls_live_")
        assert len(key) >= 32

    def test_generate_api_key_test_env(self):
        key = _generate_api_key("test")
        assert key.startswith("cls_test_")

    def test_generate_api_key_uniqueness(self):
        keys = {_generate_api_key() for _ in range(100)}
        assert len(keys) == 100  # All unique

    def test_mask_key_long(self):
        key = "cls_live_abcdefghijklmnopqrstuvwxyz"
        masked = _mask_key(key)
        assert masked.startswith("cls_live_abc")
        assert masked.endswith("wxyz")
        assert "****" in masked

    def test_mask_key_short(self):
        key = "short"
        masked = _mask_key(key)
        assert "****" in masked

    def test_key_prefix(self):
        key = "cls_live_abcdefghijklmnopqrstuvwxyz"
        assert _key_prefix(key) == "cls_live_abc"

    def test_key_prefix_short(self):
        assert _key_prefix("short") == "short"

    def test_generate_webhook_secret_format(self):
        secret = _generate_webhook_secret()
        assert secret.startswith("whsec_")
        assert len(secret) > 20

    def test_generate_webhook_secret_uniqueness(self):
        secrets = {_generate_webhook_secret() for _ in range(100)}
        assert len(secrets) == 100


# ============================================================================
# Fixtures
# ============================================================================

def _now():
    return datetime.now(timezone.utc)


def _make_integration_data(
    name="Test App",
    namespace="default",
    integration_id=None,
    key_data=None,
):
    """Build mock integration result from store."""
    integration_id = integration_id or str(uuid4())
    now = _now()
    return {
        "id": integration_id,
        "name": name,
        "description": "A test integration",
        "namespace": namespace,
        "status": "active",
        "owner_email": "dev@test.com",
        "metadata": {},
        "created_at": now,
        "updated_at": now,
        "key_count": 1,
        "webhook_count": 0,
        "api_key": key_data or {
            "id": str(uuid4()),
            "integration_id": integration_id,
            "key_prefix": "cls_live_abc",
            "key_hint": "cls_live_abc****wxyz",
            "key": "cls_live_abcdefghijklmnopqrstuvwxyz",
            "scopes": ["memories:read", "memories:write"],
            "label": "Default key",
            "status": "active",
            "created_at": now,
            "expires_at": None,
            "last_used_at": None,
        },
    }


def _make_key_data(integration_id=None, scopes=None):
    """Build mock key result from store."""
    now = _now()
    return {
        "id": str(uuid4()),
        "integration_id": integration_id or str(uuid4()),
        "key_prefix": "cls_live_xyz",
        "key_hint": "cls_live_xyz****efgh",
        "key": "cls_live_xyzabcdefghijklmnopqrstuvwxyz",
        "scopes": scopes or ["memories:read", "memories:write"],
        "label": "Test key",
        "status": "active",
        "created_at": now,
        "expires_at": None,
        "last_used_at": None,
    }


def _make_webhook_data(integration_id=None, events=None):
    """Build mock webhook result from store."""
    now = _now()
    return {
        "id": str(uuid4()),
        "integration_id": integration_id or str(uuid4()),
        "url": "https://example.com/webhook",
        "events": events or ["*"],
        "description": "Test webhook",
        "status": "active",
        "failure_count": 0,
        "created_at": now,
        "namespace_filter": None,
        "secret": "whsec_testsecret123456",
    }


def _make_event_data(integration_id=None):
    """Build mock event result from store."""
    now = _now()
    return {
        "id": str(uuid4()),
        "integration_id": integration_id or str(uuid4()),
        "event_type": "integration.created",
        "actor": "system",
        "description": "Integration registered",
        "resource_type": None,
        "resource_id": None,
        "metadata": {},
        "created_at": now,
    }


@pytest.fixture
def mock_store():
    """Mock IntegrationStore with all methods as AsyncMock."""
    store = MagicMock(spec=IntegrationStore)
    store.create_integration = AsyncMock()
    store.get_integration = AsyncMock()
    store.list_integrations = AsyncMock()
    store.delete_integration = AsyncMock()
    store.create_api_key = AsyncMock()
    store.list_api_keys = AsyncMock()
    store.rotate_api_key = AsyncMock()
    store.revoke_api_key = AsyncMock()
    store.create_webhook = AsyncMock()
    store.list_webhooks = AsyncMock()
    store.delete_webhook = AsyncMock()
    store.list_events = AsyncMock()
    store.close = AsyncMock()
    return store


@pytest.fixture
def service(mock_store):
    """IntegrationService with mocked store."""
    svc = IntegrationService.__new__(IntegrationService)
    svc.settings = Settings()
    svc.store = mock_store
    return svc


# ============================================================================
# Integration CRUD
# ============================================================================


class TestIntegrationRegistration:
    """Tests for integration registration."""

    @pytest.mark.asyncio
    async def test_register_returns_integration_and_key(self, service, mock_store):
        integration_id = str(uuid4())
        data = _make_integration_data(integration_id=integration_id)
        mock_store.create_integration.return_value = data

        req = IntegrationCreate(name="Test App", description="Testing")
        integration, api_key = await service.register(req)

        assert isinstance(integration, IntegrationResponse)
        assert integration.id == integration_id
        assert integration.name == "Test App"
        assert integration.status == "active"

        assert isinstance(api_key, ApiKeyResponse)
        assert api_key.key is not None  # Shown once
        assert api_key.status == "active"

    @pytest.mark.asyncio
    async def test_register_with_namespace(self, service, mock_store):
        data = _make_integration_data(namespace="my-ns")
        mock_store.create_integration.return_value = data

        req = IntegrationCreate(name="NS App", namespace="my-ns")
        integration, _ = await service.register(req)
        assert integration.namespace == "my-ns"
        mock_store.create_integration.assert_called_once_with(
            name="NS App", description="", namespace="my-ns",
            owner_email=None, metadata={},
        )

    @pytest.mark.asyncio
    async def test_register_with_owner_email(self, service, mock_store):
        data = _make_integration_data()
        data["owner_email"] = "dev@company.com"
        mock_store.create_integration.return_value = data

        req = IntegrationCreate(name="Email App", owner_email="dev@company.com")
        integration, _ = await service.register(req)
        assert integration.owner_email == "dev@company.com"


class TestIntegrationGet:
    """Tests for getting integrations."""

    @pytest.mark.asyncio
    async def test_get_existing(self, service, mock_store):
        iid = str(uuid4())
        mock_store.get_integration.return_value = _make_integration_data(integration_id=iid)

        result = await service.get(iid)
        assert result is not None
        assert result.id == iid

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, service, mock_store):
        mock_store.get_integration.return_value = None
        result = await service.get("nonexistent-id")
        assert result is None


class TestIntegrationList:
    """Tests for listing integrations."""

    @pytest.mark.asyncio
    async def test_list_returns_all(self, service, mock_store):
        mock_store.list_integrations.return_value = [
            _make_integration_data(name="App 1"),
            _make_integration_data(name="App 2"),
        ]
        results = await service.list_all("default")
        assert len(results) == 2
        assert results[0].name == "App 1"

    @pytest.mark.asyncio
    async def test_list_empty(self, service, mock_store):
        mock_store.list_integrations.return_value = []
        results = await service.list_all("empty-ns")
        assert results == []


class TestIntegrationDelete:
    """Tests for deleting integrations."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, service, mock_store):
        mock_store.delete_integration.return_value = True
        assert await service.delete("some-id") is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, service, mock_store):
        mock_store.delete_integration.return_value = False
        assert await service.delete("nonexistent") is False


# ============================================================================
# API Key Lifecycle
# ============================================================================


class TestApiKeyCreation:
    """Tests for API key creation."""

    @pytest.mark.asyncio
    async def test_create_key_default_scopes(self, service, mock_store):
        iid = str(uuid4())
        mock_store.get_integration.return_value = _make_integration_data(integration_id=iid)
        mock_store.create_api_key.return_value = _make_key_data(integration_id=iid)

        req = ApiKeyCreate()
        result = await service.create_key(iid, req)
        assert result is not None
        assert result.key is not None  # Shown once
        assert "memories:read" in result.scopes

    @pytest.mark.asyncio
    async def test_create_key_custom_scopes(self, service, mock_store):
        iid = str(uuid4())
        mock_store.get_integration.return_value = _make_integration_data(integration_id=iid)
        key_data = _make_key_data(integration_id=iid, scopes=["memories:read", "usage:read"])
        mock_store.create_api_key.return_value = key_data

        req = ApiKeyCreate(scopes=["memories:read", "usage:read"])
        result = await service.create_key(iid, req)
        assert result is not None
        assert "usage:read" in result.scopes

    @pytest.mark.asyncio
    async def test_create_key_invalid_scope_raises(self, service, mock_store):
        iid = str(uuid4())
        req = ApiKeyCreate(scopes=["invalid:scope"])
        with pytest.raises(ValueError, match="Invalid scopes"):
            await service.create_key(iid, req)

    @pytest.mark.asyncio
    async def test_create_key_nonexistent_integration(self, service, mock_store):
        mock_store.get_integration.return_value = None
        req = ApiKeyCreate()
        result = await service.create_key("nonexistent", req)
        assert result is None


class TestApiKeyList:
    """Tests for listing API keys."""

    @pytest.mark.asyncio
    async def test_list_keys_masked(self, service, mock_store):
        keys = [_make_key_data(), _make_key_data()]
        # Remove full key (should not be in list response)
        for k in keys:
            del k["key"]
        mock_store.list_api_keys.return_value = keys

        results = await service.list_keys("some-id")
        assert len(results) == 2
        for r in results:
            assert r.key is None  # Never returned in list


class TestApiKeyRotation:
    """Tests for key rotation."""

    @pytest.mark.asyncio
    async def test_rotate_returns_new_key(self, service, mock_store):
        new_key = _make_key_data()
        mock_store.rotate_api_key.return_value = new_key

        result = await service.rotate_key("old-key-id")
        assert result is not None
        assert result.key is not None  # New key shown once

    @pytest.mark.asyncio
    async def test_rotate_nonexistent_key(self, service, mock_store):
        mock_store.rotate_api_key.return_value = None
        result = await service.rotate_key("nonexistent")
        assert result is None


class TestApiKeyRevocation:
    """Tests for key revocation."""

    @pytest.mark.asyncio
    async def test_revoke_existing(self, service, mock_store):
        mock_store.revoke_api_key.return_value = True
        assert await service.revoke_key("some-key-id") is True

    @pytest.mark.asyncio
    async def test_revoke_nonexistent(self, service, mock_store):
        mock_store.revoke_api_key.return_value = False
        assert await service.revoke_key("nonexistent") is False


# ============================================================================
# Webhook Subscriptions
# ============================================================================


class TestWebhookSubscription:
    """Tests for webhook subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_returns_secret(self, service, mock_store):
        iid = str(uuid4())
        mock_store.get_integration.return_value = _make_integration_data(integration_id=iid)
        mock_store.create_webhook.return_value = _make_webhook_data(integration_id=iid)

        req = WebhookCreate(url="https://example.com/webhook")
        result = await service.subscribe_webhook(iid, req)
        assert result is not None
        assert result.secret is not None  # Shown once
        assert result.url == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_subscribe_custom_events(self, service, mock_store):
        iid = str(uuid4())
        mock_store.get_integration.return_value = _make_integration_data(integration_id=iid)
        webhook_data = _make_webhook_data(
            integration_id=iid,
            events=["memory.created", "memory.promoted"],
        )
        mock_store.create_webhook.return_value = webhook_data

        req = WebhookCreate(
            url="https://example.com/webhook",
            events=["memory.created", "memory.promoted"],
        )
        result = await service.subscribe_webhook(iid, req)
        assert "memory.created" in result.events

    @pytest.mark.asyncio
    async def test_subscribe_invalid_event_raises(self, service, mock_store):
        iid = str(uuid4())
        req = WebhookCreate(url="https://example.com/webhook", events=["invalid.event"])
        with pytest.raises(ValueError, match="Invalid events"):
            await service.subscribe_webhook(iid, req)

    @pytest.mark.asyncio
    async def test_subscribe_nonexistent_integration(self, service, mock_store):
        mock_store.get_integration.return_value = None
        req = WebhookCreate(url="https://example.com/webhook")
        result = await service.subscribe_webhook("nonexistent", req)
        assert result is None


class TestWebhookList:
    """Tests for listing webhooks."""

    @pytest.mark.asyncio
    async def test_list_webhooks(self, service, mock_store):
        webhooks = [_make_webhook_data(), _make_webhook_data()]
        for w in webhooks:
            del w["secret"]
        mock_store.list_webhooks.return_value = webhooks

        results = await service.list_webhooks("some-id")
        assert len(results) == 2
        for r in results:
            assert r.secret is None  # Never in list


class TestWebhookUnsubscribe:
    """Tests for webhook unsubscription."""

    @pytest.mark.asyncio
    async def test_unsubscribe_existing(self, service, mock_store):
        mock_store.delete_webhook.return_value = True
        assert await service.unsubscribe_webhook("some-wh-id") is True

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, service, mock_store):
        mock_store.delete_webhook.return_value = False
        assert await service.unsubscribe_webhook("nonexistent") is False


# ============================================================================
# Audit Events
# ============================================================================


class TestAuditEvents:
    """Tests for audit event log."""

    @pytest.mark.asyncio
    async def test_get_events(self, service, mock_store):
        events = [_make_event_data(), _make_event_data()]
        mock_store.list_events.return_value = events

        results = await service.get_events("some-id", limit=50)
        assert len(results) == 2
        assert all(isinstance(e, IntegrationEventResponse) for e in results)

    @pytest.mark.asyncio
    async def test_get_events_empty(self, service, mock_store):
        mock_store.list_events.return_value = []
        results = await service.get_events("some-id")
        assert results == []


# ============================================================================
# Service Lifecycle
# ============================================================================


class TestServiceLifecycle:
    """Tests for service init and teardown."""

    @pytest.mark.asyncio
    async def test_close(self, service, mock_store):
        await service.close()
        mock_store.close.assert_called_once()


# ============================================================================
# Pydantic Model Validation
# ============================================================================


class TestModelValidation:
    """Tests for integration Pydantic models."""

    def test_integration_create_valid(self):
        req = IntegrationCreate(name="My App")
        assert req.name == "My App"
        assert req.namespace == "default"

    def test_integration_create_invalid_namespace(self):
        with pytest.raises(Exception):
            IntegrationCreate(name="App", namespace="invalid namespace!")

    def test_integration_create_name_too_long(self):
        with pytest.raises(Exception):
            IntegrationCreate(name="x" * 200)

    def test_api_key_create_defaults(self):
        req = ApiKeyCreate()
        assert "memories:read" in req.scopes
        assert req.expires_in_days is None

    def test_api_key_create_with_expiry(self):
        req = ApiKeyCreate(expires_in_days=90)
        assert req.expires_in_days == 90

    def test_api_key_create_expiry_too_long(self):
        with pytest.raises(Exception):
            ApiKeyCreate(expires_in_days=5000)

    def test_webhook_create_valid(self):
        req = WebhookCreate(url="https://example.com/webhook")
        assert req.url == "https://example.com/webhook"
        assert req.events == ["*"]

    def test_webhook_create_url_too_short(self):
        with pytest.raises(Exception):
            WebhookCreate(url="http://x")

    def test_webhook_create_custom_events(self):
        req = WebhookCreate(
            url="https://example.com/webhook",
            events=["memory.created", "memory.deleted"],
        )
        assert len(req.events) == 2


# ============================================================================
# Valid Scopes & Events Constants
# ============================================================================


class TestConstants:
    """Tests for valid scopes and events."""

    def test_valid_scopes_not_empty(self):
        assert len(VALID_SCOPES) > 0

    def test_valid_scopes_contains_basics(self):
        assert "memories:read" in VALID_SCOPES
        assert "memories:write" in VALID_SCOPES
        assert "webhooks:manage" in VALID_SCOPES

    def test_valid_events_not_empty(self):
        assert len(VALID_EVENTS) > 0

    def test_valid_events_contains_wildcard(self):
        assert "*" in VALID_EVENTS

    def test_valid_events_contains_memory_events(self):
        assert "memory.created" in VALID_EVENTS
        assert "memory.promoted" in VALID_EVENTS
        assert "consolidation.complete" in VALID_EVENTS


# ============================================================================
# IntegrationStore unit tests (utility methods, no DB)
# ============================================================================


class TestIntegrationStoreHelpers:
    """Tests for IntegrationStore row conversion methods."""

    def test_row_to_integration(self):
        store = IntegrationStore.__new__(IntegrationStore)
        now = _now()
        row = {
            "id": uuid4(),
            "name": "Test",
            "description": "Desc",
            "namespace": "default",
            "status": "active",
            "owner_email": "test@test.com",
            "metadata": "{}",
            "created_at": now,
            "updated_at": now,
            "key_count": 2,
            "webhook_count": 1,
        }
        result = store._row_to_integration(row)
        assert result["name"] == "Test"
        assert result["key_count"] == 2
        assert result["metadata"] == {}

    def test_row_to_key(self):
        store = IntegrationStore.__new__(IntegrationStore)
        now = _now()
        row = {
            "id": uuid4(),
            "integration_id": uuid4(),
            "key_prefix": "cls_live_abc",
            "key_hint": "cls_live_abc****wxyz",
            "scopes": '["memories:read"]',
            "label": "Test",
            "status": "active",
            "created_at": now,
            "expires_at": None,
            "last_used_at": None,
        }
        result = store._row_to_key(row)
        assert result["scopes"] == ["memories:read"]

    def test_row_to_webhook(self):
        store = IntegrationStore.__new__(IntegrationStore)
        now = _now()
        row = {
            "id": uuid4(),
            "integration_id": uuid4(),
            "url": "https://example.com/wh",
            "events": '["memory.created"]',
            "description": "Test",
            "status": "active",
            "failure_count": 0,
            "created_at": now,
            "namespace_filter": None,
        }
        result = store._row_to_webhook(row)
        assert result["events"] == ["memory.created"]

    def test_row_to_event(self):
        store = IntegrationStore.__new__(IntegrationStore)
        now = _now()
        row = {
            "id": uuid4(),
            "integration_id": uuid4(),
            "event_type": "key.created",
            "actor": "system",
            "description": "Key created",
            "resource_type": "api_key",
            "resource_id": uuid4(),
            "metadata": '{"old": "value"}',
            "created_at": now,
        }
        result = store._row_to_event(row)
        assert result["event_type"] == "key.created"
        assert result["metadata"] == {"old": "value"}

    def test_row_to_integration_json_metadata(self):
        """Test that JSON string metadata is parsed correctly."""
        store = IntegrationStore.__new__(IntegrationStore)
        now = _now()
        row = {
            "id": uuid4(),
            "name": "Test",
            "description": "",
            "namespace": "default",
            "status": "active",
            "owner_email": None,
            "metadata": '{"logo": "https://example.com/logo.png"}',
            "created_at": now,
            "updated_at": now,
            "key_count": 0,
            "webhook_count": 0,
        }
        result = store._row_to_integration(row)
        assert result["metadata"]["logo"] == "https://example.com/logo.png"

    def test_row_to_key_dict_scopes(self):
        """Test that dict-type scopes (from asyncpg JSONB) pass through."""
        store = IntegrationStore.__new__(IntegrationStore)
        now = _now()
        row = {
            "id": uuid4(),
            "integration_id": uuid4(),
            "key_prefix": "cls_live_abc",
            "key_hint": "cls_live_abc****wxyz",
            "scopes": ["memories:read", "memories:write"],  # Already parsed by asyncpg
            "label": "",
            "status": "active",
            "created_at": now,
            "expires_at": None,
            "last_used_at": None,
        }
        result = store._row_to_key(row)
        assert result["scopes"] == ["memories:read", "memories:write"]
