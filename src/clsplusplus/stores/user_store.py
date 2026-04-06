"""User Store — PostgreSQL-backed user management.

Follows the same async pool pattern as IntegrationStore.
Auto-creates tables on first connection using user_ddl.sql.
"""

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


def _row_to_dict(row: asyncpg.Record) -> dict:
    """Convert asyncpg Record to dict, stripping password_hash."""
    d = dict(row)
    d.pop("password_hash", None)
    # Convert UUID and datetime to strings for JSON serialization
    for k, v in d.items():
        if hasattr(v, "hex") and hasattr(v, "int"):  # UUID
            d[k] = str(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


class UserStore:
    """PostgreSQL-backed store for user accounts."""

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
                        max_size=5,
                        command_timeout=60,
                    )
                    async with self._pool.acquire() as conn:
                        await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        ddl_path = os.path.join(os.path.dirname(__file__), "user_ddl.sql")
        with open(ddl_path) as f:
            ddl = f.read()
        await conn.execute(ddl)
        # RBAC tables (depends on users table existing first)
        # Non-fatal: if RBAC DDL fails, user auth still works
        try:
            rbac_path = os.path.join(os.path.dirname(__file__), "rbac_ddl.sql")
            with open(rbac_path) as f:
                rbac_ddl = f.read()
            await conn.execute(rbac_ddl)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("RBAC schema init failed (non-fatal): %s", e)

    # =========================================================================
    # Users CRUD
    # =========================================================================

    async def create_user(
        self,
        email: str,
        password_hash: Optional[str] = None,
        google_id: Optional[str] = None,
        name: str = "",
        avatar_url: Optional[str] = None,
    ) -> dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (email, password_hash, google_id, name, avatar_url)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                email, password_hash, google_id, name, avatar_url,
            )
            return _row_to_dict(row)

    async def get_by_email(self, email: str) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email = $1", email,
            )
            if not row:
                return None
            d = dict(row)
            for k, v in d.items():
                if hasattr(v, "hex") and hasattr(v, "int"):
                    d[k] = str(v)
                elif isinstance(v, datetime):
                    d[k] = v.isoformat()
            return d

    async def get_by_google_id(self, google_id: str) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE google_id = $1", google_id,
            )
            if not row:
                return None
            return _row_to_dict(row)

    async def get_by_id(self, user_id: str) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1", user_id,
            )
            if not row:
                return None
            return _row_to_dict(row)

    async def update_tier(self, user_id: str, tier: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE users SET tier = $1, updated_at = $2 WHERE id = $3",
                tier, _now(), user_id,
            )
            return result == "UPDATE 1"

    async def update_google_id(self, user_id: str, google_id: str, avatar_url: Optional[str] = None) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE users SET google_id = $1, avatar_url = COALESCE($2, avatar_url), updated_at = $3 WHERE id = $4",
                google_id, avatar_url, _now(), user_id,
            )
            return result == "UPDATE 1"

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit, offset,
            )
            return [_row_to_dict(r) for r in rows]

    async def update_user(self, user_id: str, fields: dict) -> Optional[dict]:
        """Update arbitrary user fields (name, email, password_hash)."""
        if not fields:
            return await self.get_by_id(user_id)
        sets = []
        vals = []
        idx = 1
        for key in ("name", "email", "password_hash"):
            if key in fields:
                sets.append(f"{key} = ${idx}")
                vals.append(fields[key])
                idx += 1
        if not sets:
            return await self.get_by_id(user_id)
        sets.append(f"updated_at = ${idx}")
        vals.append(_now())
        idx += 1
        vals.append(user_id)
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE users SET {', '.join(sets)} WHERE id = ${idx} RETURNING *",
                *vals,
            )
            if not row:
                return None
            return _row_to_dict(row)

    async def count_users(self) -> int:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM users")

    async def count_users_by_tier(self) -> dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT tier, COUNT(*) AS cnt FROM users GROUP BY tier"
            )
            return {r["tier"]: r["cnt"] for r in rows}

    async def daily_signups(self, days: int = 30) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DATE(created_at) AS day, COUNT(*) AS count
                FROM users
                WHERE created_at >= NOW() - INTERVAL '1 day' * $1
                GROUP BY DATE(created_at)
                ORDER BY day
                """,
                days,
            )
            return [{"date": str(r["day"]), "count": r["count"]} for r in rows]

    # =========================================================================
    # Revenue events
    # =========================================================================

    async def record_revenue_event(
        self,
        user_id: str,
        event_type: str,
        from_tier: str,
        to_tier: str,
        monthly_revenue: float,
    ) -> dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO revenue_events (user_id, event_type, from_tier, to_tier, monthly_revenue)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                user_id, event_type, from_tier, to_tier, monthly_revenue,
            )
            return _row_to_dict(row)

    async def get_revenue_events(self, limit: int = 100) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM revenue_events ORDER BY created_at DESC LIMIT $1",
                limit,
            )
            return [_row_to_dict(r) for r in rows]
