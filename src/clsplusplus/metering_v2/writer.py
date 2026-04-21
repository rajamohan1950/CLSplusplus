"""Append-only metering writer (ADR 0001 step 2).

This module owns one durable invariant: every metered action either
lands in `usage_events` or lands in `metering_dead_letter`. It never
silently drops. It never raises into the caller.

The writer is fire-and-forget: `record()` schedules an asyncio task and
returns in microseconds so the request latency at the edge is unaffected.
Redis remains primary during step 2 — this writer is strictly additive
and gated by `CLS_METERING_V2_WRITE_ENABLED`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)


# Types accepted in usage_events.actor_kind — mirrors the DDL CHECK.
VALID_ACTOR_KINDS = frozenset({"user", "ext", "ns", "api_key", "system"})


@dataclass
class UsageEvent:
    """A single metered action."""

    idempotency_key: str
    actor_kind: str
    actor_id: str
    event_type: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    namespace: Optional[str] = None
    quantity: int = 1
    unit_cost_cents: int = 0  # pricer lands in a follow-up step; always 0 for now
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actor_kind not in VALID_ACTOR_KINDS:
            raise ValueError(
                f"actor_kind must be one of {sorted(VALID_ACTOR_KINDS)}, "
                f"got {self.actor_kind!r}"
            )
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")
        if self.unit_cost_cents < 0:
            raise ValueError(
                f"unit_cost_cents must be >= 0, got {self.unit_cost_cents}"
            )


PoolGetter = Callable[[], Awaitable[Any]]


class MeteringWriter:
    """Durable append-only writer with dead-letter fallback.

    `record(event)` schedules the insert and returns immediately. The
    background task performs the INSERT; on any error it writes a
    best-effort row into `metering_dead_letter`. The on-call notifier
    (see notifier.py) pages from that table.
    """

    def __init__(self, settings: Settings, pool_getter: PoolGetter):
        self.settings = settings
        self._pool_getter = pool_getter

    @property
    def enabled(self) -> bool:
        return bool(self.settings.metering_v2_write_enabled)

    async def record(self, event: UsageEvent) -> None:
        """Fire-and-forget. Never raises."""
        if not self.enabled:
            return
        try:
            asyncio.create_task(self._record_safely(event))
        except RuntimeError:
            # No running loop (e.g., called from sync context during shutdown).
            # Fall back to sync execution so we still get the durability.
            await self._record_safely(event)

    async def record_sync(self, event: UsageEvent) -> bool:
        """Awaitable variant for tests and offline replay. Returns True if inserted."""
        if not self.enabled:
            return False
        try:
            await self._insert(event)
            return True
        except Exception as exc:
            await self._dead_letter(event, exc)
            return False

    # --- internals ---------------------------------------------------

    async def _record_safely(self, event: UsageEvent) -> None:
        try:
            await self._insert(event)
        except Exception as exc:
            await self._dead_letter(event, exc)

    async def _insert(self, event: UsageEvent) -> None:
        pool = await self._pool_getter()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO usage_events
                    (idempotency_key, actor_kind, actor_id, user_id,
                     api_key_id, namespace, event_type, quantity,
                     unit_cost_cents, occurred_at, raw)
                VALUES ($1, $2, $3, $4::uuid, $5, $6, $7, $8, $9, $10, $11::jsonb)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                event.idempotency_key,
                event.actor_kind,
                event.actor_id,
                event.user_id,
                event.api_key_id,
                event.namespace,
                event.event_type,
                event.quantity,
                event.unit_cost_cents,
                event.occurred_at,
                json.dumps(event.raw),
            )

    async def _dead_letter(self, event: UsageEvent, exc: Exception) -> None:
        """Best-effort enqueue into metering_dead_letter. Can still fail silently
        if the DB is completely unreachable — that's logged, not raised."""
        try:
            pool = await self._pool_getter()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO metering_dead_letter
                        (error_class, error_message, payload)
                    VALUES ($1, $2, $3::jsonb)
                    """,
                    type(exc).__name__,
                    str(exc)[:500],
                    json.dumps({
                        "idempotency_key": event.idempotency_key,
                        "actor_kind": event.actor_kind,
                        "actor_id": event.actor_id,
                        "user_id": event.user_id,
                        "api_key_id": event.api_key_id,
                        "namespace": event.namespace,
                        "event_type": event.event_type,
                        "quantity": event.quantity,
                        "unit_cost_cents": event.unit_cost_cents,
                        "occurred_at": event.occurred_at.isoformat(),
                        "raw": event.raw,
                    }),
                )
        except Exception as final_exc:
            # Last resort: if we can't even dead-letter, we've lost the event.
            # This is an ops emergency — log loudly so monitoring picks it up.
            logger.error(
                "metering: dead-letter insert failed (%s: %s) — event LOST. "
                "original cause: %s: %s. payload: %s",
                type(final_exc).__name__, final_exc,
                type(exc).__name__, exc,
                event.idempotency_key,
            )
