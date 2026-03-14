"""CLS++ Integration Service — Business logic for self-service integrations.

Thin orchestration layer between API endpoints and IntegrationStore.
Validates business rules, formats responses, handles edge cases.
"""

import logging
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.models import (
    ApiKeyCreate,
    ApiKeyResponse,
    IntegrationCreate,
    IntegrationEventResponse,
    IntegrationResponse,
    WebhookCreate,
    WebhookResponse,
)
from clsplusplus.stores.integration_store import IntegrationStore

logger = logging.getLogger(__name__)

# Valid scopes for API keys
VALID_SCOPES = frozenset({
    "memories:read",
    "memories:write",
    "memories:delete",
    "consolidate",
    "webhooks:manage",
    "integrations:manage",
    "usage:read",
    "admin",
})

# Valid webhook event types
VALID_EVENTS = frozenset({
    "*",
    "memory.created",
    "memory.updated",
    "memory.deleted",
    "memory.promoted",
    "memory.pruned",
    "consolidation.started",
    "consolidation.complete",
    "key.created",
    "key.rotated",
    "key.revoked",
})


class IntegrationService:
    """Orchestrates integration management operations."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.store = IntegrationStore(self.settings)

    async def register(self, req: IntegrationCreate) -> tuple[IntegrationResponse, ApiKeyResponse]:
        """Register a new integration. Returns (integration, first_api_key)."""
        result = await self.store.create_integration(
            name=req.name,
            description=req.description,
            namespace=req.namespace,
            owner_email=req.owner_email,
            metadata=req.metadata,
        )

        integration = IntegrationResponse(
            id=result["id"],
            name=result["name"],
            description=result["description"],
            namespace=result["namespace"],
            status=result["status"],
            owner_email=result["owner_email"],
            metadata=result["metadata"],
            created_at=result["created_at"],
            updated_at=result["updated_at"],
            key_count=result["key_count"],
            webhook_count=result["webhook_count"],
        )

        key_data = result["api_key"]
        api_key = ApiKeyResponse(
            id=key_data["id"],
            integration_id=key_data["integration_id"],
            key_prefix=key_data["key_prefix"],
            key_hint=key_data["key_hint"],
            scopes=key_data["scopes"],
            label=key_data["label"],
            status=key_data["status"],
            created_at=key_data["created_at"],
            expires_at=key_data.get("expires_at"),
            key=key_data["key"],  # Shown once
        )

        return integration, api_key

    async def get(self, integration_id: str) -> Optional[IntegrationResponse]:
        """Get an integration by ID."""
        data = await self.store.get_integration(integration_id)
        if not data:
            return None
        return IntegrationResponse(**data)

    async def list_all(self, namespace: str = "default") -> list[IntegrationResponse]:
        """List all integrations for a namespace."""
        rows = await self.store.list_integrations(namespace)
        return [IntegrationResponse(**r) for r in rows]

    async def delete(self, integration_id: str) -> bool:
        """Deactivate an integration."""
        return await self.store.delete_integration(integration_id)

    async def create_key(
        self, integration_id: str, req: ApiKeyCreate
    ) -> Optional[ApiKeyResponse]:
        """Create a new API key with scope validation."""
        # Validate scopes
        invalid = set(req.scopes) - VALID_SCOPES
        if invalid:
            raise ValueError(f"Invalid scopes: {invalid}. Valid: {sorted(VALID_SCOPES)}")

        # Verify integration exists
        integration = await self.store.get_integration(integration_id)
        if not integration:
            return None

        key_data = await self.store.create_api_key(
            integration_id=integration_id,
            scopes=req.scopes,
            label=req.label,
            expires_in_days=req.expires_in_days,
        )

        return ApiKeyResponse(
            id=key_data["id"],
            integration_id=key_data["integration_id"],
            key_prefix=key_data["key_prefix"],
            key_hint=key_data["key_hint"],
            scopes=key_data["scopes"],
            label=key_data["label"],
            status=key_data["status"],
            created_at=key_data["created_at"],
            expires_at=key_data.get("expires_at"),
            key=key_data["key"],  # Shown once
        )

    async def list_keys(self, integration_id: str) -> list[ApiKeyResponse]:
        """List API keys (masked)."""
        rows = await self.store.list_api_keys(integration_id)
        return [
            ApiKeyResponse(
                id=r["id"],
                integration_id=r["integration_id"],
                key_prefix=r["key_prefix"],
                key_hint=r["key_hint"],
                scopes=r["scopes"],
                label=r["label"],
                status=r["status"],
                created_at=r["created_at"],
                expires_at=r.get("expires_at"),
            )
            for r in rows
        ]

    async def rotate_key(self, key_id: str) -> Optional[ApiKeyResponse]:
        """Rotate an API key (24h grace period on old key)."""
        key_data = await self.store.rotate_api_key(key_id)
        if not key_data:
            return None

        return ApiKeyResponse(
            id=key_data["id"],
            integration_id=key_data["integration_id"],
            key_prefix=key_data["key_prefix"],
            key_hint=key_data["key_hint"],
            scopes=key_data["scopes"],
            label=key_data["label"],
            status=key_data["status"],
            created_at=key_data["created_at"],
            expires_at=key_data.get("expires_at"),
            key=key_data["key"],  # New key shown once
        )

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key immediately."""
        return await self.store.revoke_api_key(key_id)

    async def subscribe_webhook(
        self, integration_id: str, req: WebhookCreate
    ) -> Optional[WebhookResponse]:
        """Subscribe to webhook events with validation."""
        # Validate event types
        invalid = set(req.events) - VALID_EVENTS
        if invalid:
            raise ValueError(f"Invalid events: {invalid}. Valid: {sorted(VALID_EVENTS)}")

        # Verify integration exists
        integration = await self.store.get_integration(integration_id)
        if not integration:
            return None

        data = await self.store.create_webhook(
            integration_id=integration_id,
            url=req.url,
            events=req.events,
            description=req.description,
            namespace_filter=req.namespace_filter,
        )

        return WebhookResponse(
            id=data["id"],
            integration_id=data["integration_id"],
            url=data["url"],
            events=data["events"],
            description=data["description"],
            status=data["status"],
            failure_count=data["failure_count"],
            created_at=data["created_at"],
            namespace_filter=data.get("namespace_filter"),
            secret=data["secret"],  # Shown once
        )

    async def list_webhooks(self, integration_id: str) -> list[WebhookResponse]:
        """List webhook subscriptions."""
        rows = await self.store.list_webhooks(integration_id)
        return [
            WebhookResponse(
                id=r["id"],
                integration_id=r["integration_id"],
                url=r["url"],
                events=r["events"],
                description=r["description"],
                status=r["status"],
                failure_count=r["failure_count"],
                created_at=r["created_at"],
                namespace_filter=r.get("namespace_filter"),
            )
            for r in rows
        ]

    async def unsubscribe_webhook(self, webhook_id: str) -> bool:
        """Unsubscribe from webhook events."""
        return await self.store.delete_webhook(webhook_id)

    async def get_events(
        self, integration_id: str, limit: int = 50
    ) -> list[IntegrationEventResponse]:
        """Get audit log for an integration."""
        rows = await self.store.list_events(integration_id, limit)
        return [IntegrationEventResponse(**r) for r in rows]

    async def close(self) -> None:
        """Close store connections."""
        await self.store.close()
