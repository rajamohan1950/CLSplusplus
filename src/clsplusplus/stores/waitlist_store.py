"""Waitlist Store — PostgreSQL-backed launch waitlist.

Mirrors the UserStore pattern: lazy asyncpg pool, auto-DDL on first connect.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)


def _parse_db_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_dict(row: Optional[asyncpg.Record]) -> Optional[dict]:
    if row is None:
        return None
    d = dict(row)
    for k, v in d.items():
        if hasattr(v, "hex") and hasattr(v, "int"):
            d[k] = str(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


class WaitlistStore:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        _parse_db_url(self.settings.database_url),
                        min_size=1,
                        max_size=3,
                        command_timeout=60,
                    )
                    async with self._pool.acquire() as conn:
                        await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        ddl_path = os.path.join(os.path.dirname(__file__), "waitlist_ddl.sql")
        with open(ddl_path) as f:
            await conn.execute(f.read())

    # =========================================================================
    # Pending OTP
    # =========================================================================

    async def upsert_pending_otp(
        self, email: str, otp_code: str, expires_at: datetime, source_variant: str = ""
    ) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO waitlist_pending_otp (email, otp_code, source_variant, expires_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (email) DO UPDATE SET
                    otp_code = EXCLUDED.otp_code,
                    source_variant = EXCLUDED.source_variant,
                    expires_at = EXCLUDED.expires_at,
                    created_at = NOW()
                """,
                email, otp_code, source_variant, expires_at,
            )

    async def get_pending_otp(self, email: str) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM waitlist_pending_otp WHERE email = $1 AND expires_at > $2",
                email, _now(),
            )
            return _row_to_dict(row)

    async def get_pending_otp_any(self, email: str) -> Optional[dict]:
        """Return pending row even if expired (used for cooldown enforcement)."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM waitlist_pending_otp WHERE email = $1",
                email,
            )
            return _row_to_dict(row)

    async def delete_pending_otp(self, email: str) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM waitlist_pending_otp WHERE email = $1", email)

    # =========================================================================
    # Waitlist visitors
    # =========================================================================

    async def get_visitor(self, email: str) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM waitlist_visitors WHERE email = $1", email,
            )
            return _row_to_dict(row)

    async def get_visitor_by_invite_hash(self, token_hash: str) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM waitlist_visitors
                WHERE invite_token_hash = $1
                  AND status = 'invited'
                  AND invite_expires_at > $2
                """,
                token_hash, _now(),
            )
            return _row_to_dict(row)

    async def create_visitor(
        self, email: str, source_variant: str = ""
    ) -> dict:
        """Insert (or return existing) waitlist visitor. Idempotent on email."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO waitlist_visitors (email, source_variant, status, verified_at)
                VALUES ($1, $2, 'waiting', NOW())
                ON CONFLICT (email) DO UPDATE
                    SET verified_at = COALESCE(waitlist_visitors.verified_at, NOW())
                RETURNING *
                """,
                email, source_variant,
            )
            return _row_to_dict(row)

    async def count_waiting(self) -> int:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM waitlist_visitors WHERE status IN ('waiting','invited')"
            ) or 0

    async def get_position(self, email: str) -> Optional[int]:
        """1-indexed position in the waiting queue (oldest = 1). None if not waiting."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) + 1 AS pos
                FROM waitlist_visitors w
                WHERE w.status IN ('waiting','invited')
                  AND w.created_at < (
                    SELECT created_at FROM waitlist_visitors WHERE email = $1
                  )
                """,
                email,
            )
            if not row:
                return None
            v = await self.get_visitor(email)
            if not v or v.get("status") not in ("waiting", "invited"):
                return None
            return int(row["pos"])

    async def get_oldest_waiting(self, limit: int = 1) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM waitlist_visitors
                WHERE status = 'waiting'
                ORDER BY created_at ASC
                LIMIT $1
                """,
                limit,
            )
            return [_row_to_dict(r) for r in rows]

    async def mark_invited(
        self, visitor_id: str, token_hash: str, expires_at: datetime
    ) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE waitlist_visitors
                SET status = 'invited',
                    invited_at = NOW(),
                    invite_token_hash = $1,
                    invite_expires_at = $2
                WHERE id = $3
                """,
                token_hash, expires_at, visitor_id,
            )

    async def mark_activated(self, visitor_id: str) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE waitlist_visitors
                SET status = 'activated',
                    activated_at = NOW(),
                    invite_token_hash = NULL
                WHERE id = $1
                """,
                visitor_id,
            )

    async def expire_stale_invites(self) -> int:
        """Move invited rows whose invite_expires_at has passed back to waiting."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE waitlist_visitors
                SET status = 'waiting',
                    invite_token_hash = NULL,
                    invite_expires_at = NULL
                WHERE status = 'invited'
                  AND invite_expires_at < $1
                """,
                _now(),
            )
            try:
                return int(result.split()[-1])
            except Exception:
                return 0

    async def list_all(self, limit: int = 200) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM waitlist_visitors ORDER BY created_at DESC LIMIT $1",
                limit,
            )
            return [_row_to_dict(r) for r in rows]
