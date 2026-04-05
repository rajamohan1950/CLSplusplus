"""RBAC Service — scope resolution with Redis caching."""

from __future__ import annotations

import json
import logging
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.stores.rbac_store import RBACStore

logger = logging.getLogger(__name__)

ALL_SCOPES = frozenset({
    # API scopes
    "memories:read", "memories:write", "memories:delete",
    "consolidate", "webhooks:manage", "integrations:manage",
    "usage:read", "admin", "chat:use", "user:upgrade",
    # Page access scopes
    "page:chat", "page:docs", "page:integrate", "page:getting-started",
    "page:dashboard", "page:memory",
})

_CACHE_TTL = 60  # seconds
_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


class RBACService:
    """Business logic for RBAC — scope resolution and caching."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = RBACStore(settings)

    async def get_effective_scopes(self, user_id: str) -> set[str]:
        """Get effective scopes from DB (no cache)."""
        try:
            return await self.store.get_effective_scopes(user_id)
        except Exception as e:
            logger.warning("RBAC scope resolution failed for %s: %s", user_id, e)
            return set()

    async def get_effective_scopes_cached(self, user_id: str) -> set[str]:
        """Get effective scopes with Redis cache."""
        cache_key = f"rbac:scopes:{user_id}"
        try:
            client = _redis_client(self.settings.redis_url)
            cached = await client.get(cache_key)
            if cached:
                return set(json.loads(cached))
        except Exception:
            pass

        scopes = await self.get_effective_scopes(user_id)

        try:
            client = _redis_client(self.settings.redis_url)
            await client.setex(cache_key, _CACHE_TTL, json.dumps(list(scopes)))
        except Exception:
            pass

        return scopes

    async def invalidate_cache(self, user_id: str) -> None:
        """Clear cached scopes for a user after permission change."""
        try:
            client = _redis_client(self.settings.redis_url)
            await client.delete(f"rbac:scopes:{user_id}")
        except Exception:
            pass

    async def invalidate_group_cache(self, group_id: str) -> None:
        """Clear cached scopes for all members of a group."""
        try:
            members = await self.store.get_group_members(group_id)
            for m in members:
                await self.invalidate_cache(m["id"])
        except Exception:
            pass
