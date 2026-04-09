"""
CLS++ Namespace Resolver — Canonical Namespace Unification

Maps every authentication path to a single canonical namespace per user:
  - API key auth → integrations.namespace → canonical
  - Cookie/JWT auth → users.id → canonical
  - Extension → linked user.id → canonical

Canonical namespace = "user-{users.id[:8]}"

Redis-cached for sub-millisecond resolution on the hot path.
"""

import json
import logging
from typing import Optional

import asyncpg

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

# Shared Redis client cache (same pattern as rbac_service.py)
_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


def _parse_db_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


class NamespaceResolver:
    """Redis-cached namespace alias resolution.

    resolve(alias) → canonical namespace in <0.1ms (cache hit).
    ensure_alias() creates a new mapping on first encounter.
    """

    CACHE_TTL = 300  # 5 minutes

    def __init__(self, settings: Settings, pool: Optional[asyncpg.Pool] = None):
        self.settings = settings
        self._pool = pool

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                _parse_db_url(self.settings.database_url),
                min_size=1,
                max_size=3,
                command_timeout=15,
            )
        return self._pool

    @staticmethod
    def canonical_for_user(user_id: str) -> str:
        """Derive canonical namespace from user UUID."""
        return f"user-{str(user_id)[:8]}"

    async def resolve(self, alias: str) -> Optional[str]:
        """Resolve alias to canonical namespace.

        1. Check Redis cache (TTL 5min)
        2. Check namespace_aliases table
        3. Return None if no alias exists (use as-is)

        <0.1ms on cache hit. ~5ms on cache miss.
        """
        # 1. Redis cache
        try:
            client = _redis_client(self.settings.redis_url)
            cached = await client.get(f"ns:{alias}")
            if cached:
                return cached
        except Exception:
            pass  # Redis down → fall through to DB

        # 2. Database lookup
        try:
            pool = await self._get_pool()
            row = await pool.fetchrow(
                "SELECT canonical FROM namespace_aliases WHERE alias = $1",
                alias,
            )
            if row:
                canonical = row["canonical"]
                # Populate cache
                try:
                    client = _redis_client(self.settings.redis_url)
                    await client.setex(f"ns:{alias}", self.CACHE_TTL, canonical)
                except Exception:
                    pass
                return canonical
        except Exception as e:
            logger.debug("namespace_aliases lookup failed: %s", e)

        return None

    async def ensure_alias(self, user_id: str, alias: str,
                           canonical: str, source: str = "auto") -> None:
        """Create alias mapping if it doesn't exist.

        Called once per new auth path encounter (first API key use, etc.).
        Idempotent: ON CONFLICT DO NOTHING.
        """
        try:
            pool = await self._get_pool()
            await pool.execute("""
                INSERT INTO namespace_aliases (user_id, alias, canonical, source)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (alias) DO NOTHING
            """, user_id, alias, canonical, source)

            # Populate cache
            try:
                client = _redis_client(self.settings.redis_url)
                await client.setex(f"ns:{alias}", self.CACHE_TTL, canonical)
            except Exception:
                pass
        except Exception as e:
            logger.debug("ensure_alias failed (non-fatal): %s", e)

    async def resolve_or_identity(self, alias: str) -> str:
        """Resolve alias, returning the alias itself if no mapping exists."""
        canonical = await self.resolve(alias)
        return canonical or alias
