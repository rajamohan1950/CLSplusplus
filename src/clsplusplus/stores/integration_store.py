"""Integration Store — PostgreSQL-backed integration management.

Follows the same async pool pattern as L1IndexingStore.
Auto-creates tables on first connection using integration_ddl.sql.
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import asyncpg

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

# Key format: cls_live_<32 random chars> or cls_test_<32 random chars>
_KEY_RANDOM_BYTES = 32


def _parse_db_url(url: str) -> str:
    """Convert postgresql:// to postgres:// for asyncpg if needed."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _generate_api_key(environment: str = "live") -> str:
    """Generate a cryptographically secure API key."""
    random_part = secrets.token_urlsafe(_KEY_RANDOM_BYTES)
    return f"cls_{environment}_{random_part}"


def _mask_key(key: str) -> str:
    """Mask key for display: show prefix + last 4 chars."""
    if len(key) <= 16:
        return key[:8] + "****"
    return key[:12] + "****" + key[-4:]


def _key_prefix(key: str) -> str:
    """Extract display-safe prefix (first 12 chars)."""
    return key[:12] if len(key) >= 12 else key


def _generate_webhook_secret() -> str:
    """Generate a cryptographically secure webhook signing secret."""
    return f"whsec_{secrets.token_urlsafe(32)}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class IntegrationStore:
    """PostgreSQL-backed store for integration management."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    async def get_pool(self) -> asyncpg.Pool:
        """Thread-safe lazy pool initialization."""
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        _parse_db_url(self.settings.database_url),
                        min_size=1,
                        max_size=5,
                        command_timeout=60,
                    )
                    async with self._pool.acquire() as conn:
                        await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        """Create integration tables from DDL file."""
        ddl_path = os.path.join(os.path.dirname(__file__), "integration_ddl.sql")
        with open(ddl_path) as f:
            ddl = f.read()
        await conn.execute(ddl)

    # =========================================================================
    # API Key → Namespace Resolution
    # =========================================================================

    async def resolve_namespace_from_key(self, raw_key: str) -> Optional[str]:
        """Resolve an API key to its integration's namespace.

        Looks up the key hash in api_credentials, joins to integrations,
        and returns the namespace. Returns None if key is invalid, expired,
        revoked, or the integration is deleted.
        """
        key_hash = _sha256_hex(raw_key)
        pool = await self.get_pool()
        row = await pool.fetchrow(
            """SELECT i.namespace
               FROM integrations i
               JOIN api_credentials ac ON i.id = ac.integration_id
               WHERE ac.key_hash = $1
                 AND ac.status IN ('active', 'rotated')
                 AND i.status != 'deleted'
                 AND (ac.expires_at IS NULL OR ac.expires_at > now())
                 AND (ac.status = 'active' OR ac.grace_until > now())""",
            key_hash,
        )
        return row["namespace"] if row else None

    async def resolve_tier_from_key(self, raw_key: str) -> Optional[str]:
        """Resolve an API key to its owning user's billing tier.

        Follows the chain api_credentials → integrations → users.email → users.tier.
        Used by the metering pricer to decide whether an event is in-tier
        (free) or over-cap (pay-as-you-go). Returns None when any hop is
        missing (stale key, deleted integration, orphaned integration).
        """
        key_hash = _sha256_hex(raw_key)
        pool = await self.get_pool()
        row = await pool.fetchrow(
            """SELECT u.tier
               FROM integrations i
               JOIN api_credentials ac ON i.id = ac.integration_id
               JOIN users u           ON u.email = i.owner_email
               WHERE ac.key_hash = $1
                 AND ac.status IN ('active', 'rotated')
                 AND i.status != 'deleted'
                 AND (ac.expires_at IS NULL OR ac.expires_at > now())
                 AND (ac.status = 'active' OR ac.grace_until > now())""",
            key_hash,
        )
        return row["tier"] if row else None

    # =========================================================================
    # Integrations CRUD
    # =========================================================================

    async def create_integration(
        self,
        name: str,
        description: str = "",
        namespace: str = "default",
        owner_email: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Register a new integration. Returns integration dict + first API key."""
        pool = await self.get_pool()
        integration_id = str(uuid4())
        now = _now()

        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO integrations (id, name, description, namespace, owner_email, metadata, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                integration_id, name, description, namespace, owner_email,
                json.dumps(metadata or {}), now, now,
            )

            # Auto-generate first API key
            key_data = await self._create_key_internal(
                conn, integration_id,
                scopes=["memories:read", "memories:write"],
                label="Default key",
            )

            # Log event
            await self._log_event(
                conn, integration_id, "integration.created", "system",
                f"Integration '{name}' registered",
            )

        return {
            "id": integration_id,
            "name": name,
            "description": description,
            "namespace": namespace,
            "status": "active",
            "owner_email": owner_email,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "key_count": 1,
            "webhook_count": 0,
            "api_key": key_data,
        }

    async def get_integration(self, integration_id: str) -> Optional[dict]:
        """Get integration by ID with counts."""
        pool = await self.get_pool()
        row = await pool.fetchrow(
            """SELECT i.*,
                      (SELECT COUNT(*) FROM api_credentials c WHERE c.integration_id = i.id AND c.status = 'active') AS key_count,
                      (SELECT COUNT(*) FROM webhook_subscriptions w WHERE w.integration_id = i.id AND w.status = 'active') AS webhook_count
               FROM integrations i
               WHERE i.id = $1 AND i.status != 'deleted'""",
            integration_id,
        )
        if not row:
            return None
        return self._row_to_integration(row)

    async def list_integrations(self, namespace: str = "default") -> list[dict]:
        """List all active integrations for a namespace."""
        pool = await self.get_pool()
        rows = await pool.fetch(
            """SELECT i.*,
                      (SELECT COUNT(*) FROM api_credentials c WHERE c.integration_id = i.id AND c.status = 'active') AS key_count,
                      (SELECT COUNT(*) FROM webhook_subscriptions w WHERE w.integration_id = i.id AND w.status = 'active') AS webhook_count
               FROM integrations i
               WHERE i.namespace = $1 AND i.status != 'deleted'
               ORDER BY i.created_at DESC""",
            namespace,
        )
        return [self._row_to_integration(r) for r in rows]

    async def delete_integration(self, integration_id: str) -> bool:
        """Soft-delete an integration (sets status='deleted')."""
        pool = await self.get_pool()
        now = _now()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE integrations SET status = 'deleted', deleted_at = $2, updated_at = $2
                   WHERE id = $1 AND status != 'deleted'""",
                integration_id, now,
            )
            if "UPDATE 0" in result:
                return False
            # Revoke all active keys
            await conn.execute(
                """UPDATE api_credentials SET status = 'revoked', revoked_at = $2
                   WHERE integration_id = $1 AND status = 'active'""",
                integration_id, now,
            )
            # Disable all webhooks
            await conn.execute(
                """UPDATE webhook_subscriptions SET status = 'deleted', deleted_at = $2
                   WHERE integration_id = $1 AND status = 'active'""",
                integration_id, now,
            )
            await self._log_event(
                conn, integration_id, "integration.deleted", "system",
                "Integration deactivated; all keys revoked, webhooks disabled",
            )
        return True

    # =========================================================================
    # API Keys
    # =========================================================================

    async def create_api_key(
        self,
        integration_id: str,
        scopes: Optional[list[str]] = None,
        label: str = "",
        expires_in_days: Optional[int] = None,
    ) -> dict:
        """Create a new scoped API key for an integration."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            key_data = await self._create_key_internal(
                conn, integration_id, scopes=scopes, label=label,
                expires_in_days=expires_in_days,
            )
            await self._log_event(
                conn, integration_id, "key.created", "system",
                f"API key created: {key_data['key_prefix']}",
                resource_type="api_key", resource_id=key_data["id"],
            )
        return key_data

    async def _create_key_internal(
        self,
        conn: asyncpg.Connection,
        integration_id: str,
        scopes: Optional[list[str]] = None,
        label: str = "",
        expires_in_days: Optional[int] = None,
    ) -> dict:
        """Internal: create key within an existing connection/transaction."""
        scopes = scopes or ["memories:read", "memories:write"]
        raw_key = _generate_api_key()
        key_id = str(uuid4())
        now = _now()
        expires_at = now + timedelta(days=expires_in_days) if expires_in_days else None

        await conn.execute(
            """INSERT INTO api_credentials
               (id, integration_id, key_prefix, key_hash, key_hint, scopes, label, status, expires_at, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', $8, $9)""",
            key_id, integration_id, _key_prefix(raw_key), _sha256_hex(raw_key),
            _mask_key(raw_key), json.dumps(scopes), label, expires_at, now,
        )

        return {
            "id": key_id,
            "integration_id": integration_id,
            "key_prefix": _key_prefix(raw_key),
            "key_hint": _mask_key(raw_key),
            "key": raw_key,  # Only returned once
            "scopes": scopes,
            "label": label,
            "status": "active",
            "created_at": now,
            "expires_at": expires_at,
            "last_used_at": None,
        }

    async def list_api_keys(self, integration_id: str) -> list[dict]:
        """List all API keys for an integration (masked)."""
        pool = await self.get_pool()
        rows = await pool.fetch(
            """SELECT id, integration_id, key_prefix, key_hint, scopes, label,
                      status, created_at, expires_at, last_used_at, revoked_at
               FROM api_credentials
               WHERE integration_id = $1 AND status != 'revoked'
               ORDER BY created_at DESC""",
            integration_id,
        )
        return [self._row_to_key(r) for r in rows]

    async def rotate_api_key(self, key_id: str, grace_hours: int = 24) -> dict:
        """Rotate an API key: create new, mark old as 'rotated' with grace period."""
        pool = await self.get_pool()
        now = _now()
        grace_until = now + timedelta(hours=grace_hours)

        async with pool.acquire() as conn:
            # Get old key details
            old_row = await conn.fetchrow(
                "SELECT * FROM api_credentials WHERE id = $1 AND status = 'active'",
                key_id,
            )
            if not old_row:
                return None

            integration_id = str(old_row["integration_id"])
            old_scopes = old_row["scopes"]
            old_label = old_row["label"] or ""

            # Mark old key as rotated with grace period
            await conn.execute(
                """UPDATE api_credentials
                   SET status = 'rotated', grace_until = $2
                   WHERE id = $1""",
                key_id, grace_until,
            )

            # Parse scopes from JSON if needed
            scopes = json.loads(old_scopes) if isinstance(old_scopes, str) else old_scopes

            # Create new key linked to old
            new_key = await self._create_key_internal(
                conn, integration_id, scopes=scopes,
                label=f"{old_label} (rotated)" if old_label else "Rotated key",
            )

            # Link rotation chain
            await conn.execute(
                "UPDATE api_credentials SET rotated_from_id = $2 WHERE id = $1",
                new_key["id"], key_id,
            )

            await self._log_event(
                conn, integration_id, "key.rotated", "system",
                f"Key rotated: {old_row['key_prefix']} -> {new_key['key_prefix']}",
                resource_type="api_key", resource_id=new_key["id"],
                metadata={"old_key_id": key_id, "grace_until": grace_until.isoformat()},
            )

        return new_key

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key immediately."""
        pool = await self.get_pool()
        now = _now()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT integration_id, key_prefix FROM api_credentials WHERE id = $1 AND status IN ('active', 'rotated')",
                key_id,
            )
            if not row:
                return False

            await conn.execute(
                "UPDATE api_credentials SET status = 'revoked', revoked_at = $2 WHERE id = $1",
                key_id, now,
            )
            await self._log_event(
                conn, str(row["integration_id"]), "key.revoked", "system",
                f"API key revoked: {row['key_prefix']}",
                resource_type="api_key", resource_id=key_id,
            )
        return True

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def create_webhook(
        self,
        integration_id: str,
        url: str,
        events: Optional[list[str]] = None,
        description: str = "",
        namespace_filter: Optional[str] = None,
    ) -> dict:
        """Subscribe to webhook events. Returns subscription with signing secret."""
        pool = await self.get_pool()
        webhook_id = str(uuid4())
        raw_secret = _generate_webhook_secret()
        events = events or ["*"]
        now = _now()

        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO webhook_subscriptions
                   (id, integration_id, url, events, description, secret_hash, secret_hint,
                    namespace_filter, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                webhook_id, integration_id, url, json.dumps(events), description,
                _sha256_hex(raw_secret), _mask_key(raw_secret),
                namespace_filter, now, now,
            )
            await self._log_event(
                conn, integration_id, "webhook.subscribed", "system",
                f"Webhook subscribed: {url} for events {events}",
                resource_type="webhook", resource_id=webhook_id,
            )

        return {
            "id": webhook_id,
            "integration_id": integration_id,
            "url": url,
            "events": events,
            "description": description,
            "status": "active",
            "failure_count": 0,
            "created_at": now,
            "namespace_filter": namespace_filter,
            "secret": raw_secret,  # Only returned once
        }

    async def list_webhooks(self, integration_id: str) -> list[dict]:
        """List webhook subscriptions for an integration."""
        pool = await self.get_pool()
        rows = await pool.fetch(
            """SELECT id, integration_id, url, events, description, status,
                      failure_count, created_at, namespace_filter
               FROM webhook_subscriptions
               WHERE integration_id = $1 AND status != 'deleted'
               ORDER BY created_at DESC""",
            integration_id,
        )
        return [self._row_to_webhook(r) for r in rows]

    async def delete_webhook(self, webhook_id: str) -> bool:
        """Unsubscribe from webhook events."""
        pool = await self.get_pool()
        now = _now()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT integration_id FROM webhook_subscriptions WHERE id = $1 AND status != 'deleted'",
                webhook_id,
            )
            if not row:
                return False
            await conn.execute(
                "UPDATE webhook_subscriptions SET status = 'deleted', deleted_at = $2, updated_at = $2 WHERE id = $1",
                webhook_id, now,
            )
            await self._log_event(
                conn, str(row["integration_id"]), "webhook.unsubscribed", "system",
                "Webhook subscription removed",
                resource_type="webhook", resource_id=webhook_id,
            )
        return True

    # =========================================================================
    # Webhook Dispatch Support (used by WebhookDispatcher)
    # =========================================================================

    async def get_active_webhooks_for_namespace(self, namespace: str) -> list[dict]:
        """Get all active webhook subscriptions matching a namespace (or wildcard)."""
        pool = await self.get_pool()
        rows = await pool.fetch(
            """SELECT ws.id, ws.integration_id, ws.url, ws.events, ws.secret_hash,
                      ws.namespace_filter, ws.failure_count, ws.max_failures
               FROM webhook_subscriptions ws
               JOIN integrations i ON i.id = ws.integration_id
               WHERE ws.status = 'active'
                 AND i.status = 'active'
                 AND (ws.namespace_filter IS NULL OR ws.namespace_filter = $1)""",
            namespace,
        )
        result = []
        for r in rows:
            events = r["events"]
            if isinstance(events, str):
                events = json.loads(events)
            result.append({
                "id": str(r["id"]),
                "integration_id": str(r["integration_id"]),
                "url": r["url"],
                "events": events,
                "secret_hash": r["secret_hash"],
                "namespace_filter": r["namespace_filter"],
                "failure_count": r["failure_count"],
                "max_failures": r["max_failures"],
            })
        return result

    async def record_delivery(
        self,
        subscription_id: str,
        event_type: str,
        event_id: str,
        payload: dict,
        status: str,
        response_status: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a webhook delivery attempt."""
        pool = await self.get_pool()
        await pool.execute(
            """INSERT INTO webhook_deliveries
               (id, subscription_id, event_type, event_id, payload, status,
                response_status, response_time_ms, error_message, delivered_at, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
            str(uuid4()), subscription_id, event_type, event_id,
            json.dumps(payload, default=str), status,
            response_status, response_time_ms, error_message,
            _now() if status == "delivered" else None, _now(),
        )

    async def increment_failure(self, subscription_id: str) -> None:
        """Increment failure count; auto-disable if exceeded max_failures."""
        pool = await self.get_pool()
        await pool.execute(
            """UPDATE webhook_subscriptions
               SET failure_count = failure_count + 1,
                   status = CASE
                       WHEN failure_count + 1 >= max_failures THEN 'disabled'
                       ELSE status
                   END,
                   updated_at = $2
               WHERE id = $1""",
            subscription_id, _now(),
        )

    async def reset_failures(self, subscription_id: str) -> None:
        """Reset failure count on successful delivery."""
        pool = await self.get_pool()
        await pool.execute(
            "UPDATE webhook_subscriptions SET failure_count = 0, updated_at = $2 WHERE id = $1",
            subscription_id, _now(),
        )

    # =========================================================================
    # Events / Audit Log
    # =========================================================================

    async def list_events(
        self,
        integration_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get audit log for an integration."""
        pool = await self.get_pool()
        rows = await pool.fetch(
            """SELECT id, integration_id, event_type, actor, description,
                      resource_type, resource_id, metadata, created_at
               FROM integration_events
               WHERE integration_id = $1
               ORDER BY created_at DESC
               LIMIT $2""",
            integration_id, limit,
        )
        return [self._row_to_event(r) for r in rows]

    async def _log_event(
        self,
        conn: asyncpg.Connection,
        integration_id: str,
        event_type: str,
        actor: str,
        description: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an integration event (within existing connection)."""
        await conn.execute(
            """INSERT INTO integration_events
               (id, integration_id, event_type, actor, description, resource_type, resource_id, metadata, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
            str(uuid4()), integration_id, event_type, actor, description,
            resource_type, resource_id, json.dumps(metadata or {}), _now(),
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_integration(self, row) -> dict:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "description": row["description"] or "",
            "namespace": row["namespace"],
            "status": row["status"],
            "owner_email": row["owner_email"],
            "metadata": metadata or {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "key_count": row.get("key_count", 0),
            "webhook_count": row.get("webhook_count", 0),
        }

    def _row_to_key(self, row) -> dict:
        scopes = row["scopes"]
        if isinstance(scopes, str):
            scopes = json.loads(scopes)
        return {
            "id": str(row["id"]),
            "integration_id": str(row["integration_id"]),
            "key_prefix": row["key_prefix"],
            "key_hint": row["key_hint"],
            "scopes": scopes,
            "label": row["label"] or "",
            "status": row["status"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "last_used_at": row["last_used_at"],
        }

    def _row_to_webhook(self, row) -> dict:
        events = row["events"]
        if isinstance(events, str):
            events = json.loads(events)
        return {
            "id": str(row["id"]),
            "integration_id": str(row["integration_id"]),
            "url": row["url"],
            "events": events,
            "description": row["description"] or "",
            "status": row["status"],
            "failure_count": row["failure_count"],
            "created_at": row["created_at"],
            "namespace_filter": row.get("namespace_filter"),
        }

    def _row_to_event(self, row) -> dict:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return {
            "id": str(row["id"]),
            "integration_id": str(row["integration_id"]),
            "event_type": row["event_type"],
            "actor": row["actor"],
            "description": row["description"],
            "resource_type": row["resource_type"],
            "resource_id": str(row["resource_id"]) if row["resource_id"] else None,
            "metadata": metadata or {},
            "created_at": row["created_at"],
        }

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def health(self) -> dict:
        """Health check."""
        try:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "healthy", "store": "integrations"}
        except Exception as e:
            logger.error("Integration store health check failed: %s", e)
            return {"status": "unhealthy", "store": "integrations", "error": "Connection failed"}
