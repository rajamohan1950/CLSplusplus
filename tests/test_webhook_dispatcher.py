"""Tests for CLS++ Webhook Dispatcher — event delivery with HMAC signing."""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clsplusplus.webhook_dispatcher import WebhookDispatcher, _sign_payload


# ============================================================================
# HMAC Signing Tests
# ============================================================================


class TestHMACSigning:
    """Tests for HMAC-SHA256 payload signing."""

    def test_sign_payload_deterministic(self):
        secret = "test_secret"
        payload = b'{"event": "memory.created"}'
        sig1 = _sign_payload(secret, payload)
        sig2 = _sign_payload(secret, payload)
        assert sig1 == sig2

    def test_sign_payload_different_secrets(self):
        payload = b'{"event": "test"}'
        sig1 = _sign_payload("secret_a", payload)
        sig2 = _sign_payload("secret_b", payload)
        assert sig1 != sig2

    def test_sign_payload_different_payloads(self):
        secret = "same_secret"
        sig1 = _sign_payload(secret, b'{"a": 1}')
        sig2 = _sign_payload(secret, b'{"a": 2}')
        assert sig1 != sig2

    def test_sign_payload_matches_manual_hmac(self):
        secret = "my_secret"
        payload = b'hello world'
        expected = hmac.new(
            secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        assert _sign_payload(secret, payload) == expected

    def test_sign_payload_length(self):
        sig = _sign_payload("key", b"data")
        assert len(sig) == 64  # SHA-256 hex = 64 chars


# ============================================================================
# Dispatcher Tests
# ============================================================================


def _make_subscription(
    url="https://example.com/webhook",
    events=None,
    secret_hash="abc123",
    namespace_filter=None,
    failure_count=0,
    max_failures=10,
):
    return {
        "id": str(uuid4()),
        "integration_id": str(uuid4()),
        "url": url,
        "events": events or ["*"],
        "secret_hash": secret_hash,
        "namespace_filter": namespace_filter,
        "failure_count": failure_count,
        "max_failures": max_failures,
    }


class TestDispatcherInit:
    """Tests for dispatcher initialization."""

    def test_init_without_store(self):
        d = WebhookDispatcher()
        assert d.store is None

    def test_init_with_store(self):
        store = MagicMock()
        d = WebhookDispatcher(integration_store=store)
        assert d.store is store


class TestDispatch:
    """Tests for event dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_no_store_noop(self):
        d = WebhookDispatcher(integration_store=None)
        # Should not raise
        await d.dispatch("memory.created", {"id": "123"}, "default")

    @pytest.mark.asyncio
    async def test_dispatch_no_matching_subscriptions(self):
        store = MagicMock()
        store.get_active_webhooks_for_namespace = AsyncMock(return_value=[])
        d = WebhookDispatcher(integration_store=store)
        await d.dispatch("memory.created", {"id": "123"}, "default")
        store.get_active_webhooks_for_namespace.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_dispatch_filters_by_event_type(self):
        sub_match = _make_subscription(events=["memory.created"])
        sub_no_match = _make_subscription(events=["memory.deleted"])
        store = MagicMock()
        store.get_active_webhooks_for_namespace = AsyncMock(
            return_value=[sub_match, sub_no_match]
        )
        store.reset_failures = AsyncMock()
        store.record_delivery = AsyncMock()

        d = WebhookDispatcher(integration_store=store)

        # Mock deliver to track calls
        delivered = []
        original_deliver_safe = d._deliver_safe

        async def mock_deliver_safe(sub, event_type, payload):
            delivered.append(sub["id"])

        d._deliver_safe = mock_deliver_safe
        await d.dispatch("memory.created", {"id": "123"}, "default")
        # Give tasks time to run
        await asyncio.sleep(0.05)
        assert sub_match["id"] in delivered
        assert sub_no_match["id"] not in delivered

    @pytest.mark.asyncio
    async def test_dispatch_wildcard_matches_all(self):
        sub = _make_subscription(events=["*"])
        store = MagicMock()
        store.get_active_webhooks_for_namespace = AsyncMock(return_value=[sub])

        d = WebhookDispatcher(integration_store=store)
        delivered = []

        async def mock_deliver_safe(s, et, p):
            delivered.append(s["id"])

        d._deliver_safe = mock_deliver_safe
        await d.dispatch("memory.promoted", {"id": "123"}, "default")
        await asyncio.sleep(0.05)
        assert sub["id"] in delivered

    @pytest.mark.asyncio
    async def test_dispatch_handles_store_error(self):
        store = MagicMock()
        store.get_active_webhooks_for_namespace = AsyncMock(side_effect=Exception("DB down"))
        d = WebhookDispatcher(integration_store=store)
        # Should not raise
        await d.dispatch("memory.created", {"id": "123"}, "default")


class TestDeliver:
    """Tests for individual webhook delivery."""

    @pytest.mark.asyncio
    async def test_deliver_success(self):
        store = MagicMock()
        store.reset_failures = AsyncMock()
        store.record_delivery = AsyncMock()

        sub = _make_subscription()
        d = WebhookDispatcher(integration_store=store)

        with patch("clsplusplus.webhook_dispatcher.httpx.AsyncClient") as MockClient:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "OK"
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            d._client = mock_client

            result = await d.deliver(sub, "memory.created", {"id": "123"})

            assert result["status"] == "delivered"
            assert result["response_status"] == 200
            store.reset_failures.assert_called_once_with(sub["id"])
            store.record_delivery.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliver_http_error(self):
        store = MagicMock()
        store.increment_failure = AsyncMock()
        store.record_delivery = AsyncMock()

        sub = _make_subscription()
        d = WebhookDispatcher(integration_store=store)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        d._client = mock_client

        result = await d.deliver(sub, "memory.created", {"id": "123"})

        assert result["status"] == "failed"
        assert result["response_status"] == 500
        store.increment_failure.assert_called_once_with(sub["id"])

    @pytest.mark.asyncio
    async def test_deliver_network_error(self):
        store = MagicMock()
        store.increment_failure = AsyncMock()
        store.record_delivery = AsyncMock()

        sub = _make_subscription()
        d = WebhookDispatcher(integration_store=store)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        d._client = mock_client

        result = await d.deliver(sub, "memory.created", {"id": "123"})

        assert result["status"] == "failed"
        assert result["error_message"] == "Connection refused"
        store.increment_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliver_includes_headers(self):
        store = MagicMock()
        store.reset_failures = AsyncMock()
        store.record_delivery = AsyncMock()

        sub = _make_subscription(secret_hash="test_secret_hash")
        d = WebhookDispatcher(integration_store=store)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "OK"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        d._client = mock_client

        await d.deliver(sub, "memory.created", {"id": "123"})

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["X-CLS-Event"] == "memory.created"
        assert "X-CLS-Delivery-Id" in headers
        assert "X-CLS-Timestamp" in headers
        assert headers["X-CLS-Signature-256"].startswith("sha256=")
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_deliver_payload_envelope(self):
        store = MagicMock()
        store.reset_failures = AsyncMock()
        store.record_delivery = AsyncMock()

        sub = _make_subscription()
        d = WebhookDispatcher(integration_store=store)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "OK"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        d._client = mock_client

        await d.deliver(sub, "memory.created", {"id": "item-1", "text": "hello"})

        call_args = mock_client.post.call_args
        payload = json.loads(call_args.kwargs["content"])
        assert payload["event"] == "memory.created"
        assert "event_id" in payload
        assert "delivery_id" in payload
        assert "timestamp" in payload
        assert payload["data"]["id"] == "item-1"
        assert payload["data"]["text"] == "hello"


class TestDeliverSafe:
    """Tests for error-safe delivery wrapper."""

    @pytest.mark.asyncio
    async def test_deliver_safe_catches_exceptions(self):
        d = WebhookDispatcher()

        # Mock deliver to raise
        async def failing_deliver(sub, et, p):
            raise RuntimeError("boom")

        d.deliver = failing_deliver
        sub = _make_subscription()
        # Should not raise
        await d._deliver_safe(sub, "memory.created", {"id": "123"})


class TestDispatcherClose:
    """Tests for dispatcher cleanup."""

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        d = WebhookDispatcher()
        await d.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_with_client(self):
        d = WebhookDispatcher()
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        d._client = mock_client
        await d.close()
        mock_client.aclose.assert_called_once()
        assert d._client is None
