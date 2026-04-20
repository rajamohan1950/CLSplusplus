"""CLS++ Webhook Dispatcher — Fire-and-forget event delivery with HMAC signing.

Dispatches events to webhook subscriptions when memories are created, deleted,
promoted, or pruned. Never blocks memory operations.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import httpx

from clsplusplus.config import Settings
from clsplusplus.tracer import tracer

logger = logging.getLogger(__name__)


def _sign_payload(secret: str, payload_bytes: bytes) -> str:
    """HMAC-SHA256 sign a payload with the webhook secret."""
    return hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WebhookDispatcher:
    """Dispatches webhook events to subscribers.

    Usage:
        dispatcher = WebhookDispatcher(integration_store)
        await dispatcher.dispatch("memory.created", {"id": "...", "text": "..."}, "default")
    """

    def __init__(self, integration_store=None, settings: Optional[Settings] = None):
        self.store = integration_store
        self.settings = settings or Settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def dispatch(
        self,
        event_type: str,
        payload: dict[str, Any],
        namespace: str,
    ) -> None:
        """Find matching subscriptions and fire deliveries (fire-and-forget).

        This method never raises — errors are logged and swallowed
        to prevent webhook failures from blocking memory operations.
        """
        if self.store is None:
            return

        try:
            subscriptions = await self.store.get_active_webhooks_for_namespace(namespace)
        except Exception as e:
            logger.warning("Failed to fetch webhook subscriptions: %s", e)
            return

        for sub in subscriptions:
            events = sub.get("events", ["*"])
            if "*" in events or event_type in events:
                # Fire-and-forget: create task, don't await
                asyncio.create_task(
                    self._deliver_safe(sub, event_type, payload)
                )

    async def _deliver_safe(
        self,
        subscription: dict,
        event_type: str,
        payload: dict,
    ) -> None:
        """Deliver with error handling — never raises."""
        try:
            await self.deliver(subscription, event_type, payload)
        except Exception as e:
            logger.error(
                "Webhook delivery failed for %s to %s: %s",
                event_type, subscription.get("url", "?"), e,
            )

    async def deliver(
        self,
        subscription: dict,
        event_type: str,
        payload: dict,
        trace_id: Optional[str] = None,
        parent_hop_id: Optional[str] = None,
    ) -> dict:
        """Deliver a webhook event to a subscription URL."""
        delivery_id = str(uuid4())
        event_id = str(uuid4())
        timestamp = _now_iso()
        url = subscription.get("url", "?")

        # Create a standalone trace for this delivery (fire-and-forget context)
        tid = trace_id or tracer.new_trace(f"webhook.{event_type}")
        _span_ctx = (
            tracer.span(tid, "webhook.deliver", "webhook_dispatcher",
                        _parent=parent_hop_id,
                        input=f"event={event_type}  url={url[:80]}  item_id={payload.get('id', '?')[:16]}")
            if tid else __import__("contextlib").nullcontext()
        )

        envelope = {
            "event": event_type,
            "event_id": event_id,
            "delivery_id": delivery_id,
            "timestamp": timestamp,
            "data": payload,
        }

        payload_bytes = json.dumps(envelope, default=str).encode("utf-8")

        secret_hash = subscription.get("secret_hash", "")
        signature = _sign_payload(secret_hash, payload_bytes)

        headers = {
            "Content-Type": "application/json",
            "X-CLS-Event": event_type,
            "X-CLS-Delivery-Id": delivery_id,
            "X-CLS-Timestamp": timestamp,
            "X-CLS-Signature-256": f"sha256={signature}",
            "User-Agent": "CLS++/1.0 (Webhook)",
        }

        sub_id = subscription["id"]
        start_time = time.monotonic()
        status = "delivered"
        response_status = None
        response_body = None
        error_message = None

        with _span_ctx as deliver_hop:
            try:
                client = await self._get_client()
                resp = await client.post(url, content=payload_bytes, headers=headers)
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                response_status = resp.status_code
                response_body = resp.text[:4096]

                if 200 <= resp.status_code < 300:
                    status = "delivered"
                    if tid and deliver_hop:
                        tracer.add_metadata(tid, deliver_hop,
                                            output=f"HTTP {response_status}  {elapsed_ms}ms  delivered")
                    if self.store:
                        try:
                            await self.store.reset_failures(sub_id)
                        except Exception:
                            pass
                else:
                    status = "failed"
                    error_message = f"HTTP {resp.status_code}"
                    if tid and deliver_hop:
                        tracer.add_metadata(tid, deliver_hop,
                                            output=f"HTTP {response_status}  {elapsed_ms}ms  FAILED",
                                            error=error_message)
                    if self.store:
                        try:
                            await self.store.increment_failure(sub_id)
                        except Exception:
                            pass

            except Exception as e:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                status = "failed"
                error_message = str(e)
                if tid and deliver_hop:
                    tracer.add_metadata(tid, deliver_hop,
                                        output=f"exception  {elapsed_ms}ms",
                                        error=error_message[:200])
                if self.store:
                    try:
                        await self.store.increment_failure(sub_id)
                    except Exception:
                        pass

        result = {
            "delivery_id": delivery_id,
            "subscription_id": sub_id,
            "event_type": event_type,
            "event_id": event_id,
            "status": status,
            "response_status": response_status,
            "response_body": response_body,
            "response_time_ms": elapsed_ms,
            "error_message": error_message,
        }

        if self.store:
            try:
                await self.store.record_delivery(
                    subscription_id=sub_id,
                    event_type=event_type,
                    event_id=event_id,
                    payload=envelope,
                    status=status,
                    response_status=response_status,
                    response_time_ms=elapsed_ms,
                    error_message=error_message,
                )
            except Exception as e:
                logger.warning("Failed to record webhook delivery: %s", e)

        return result

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
