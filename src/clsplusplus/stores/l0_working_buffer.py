"""L0 Working Buffer - Prefrontal Cortex equivalent.

Short-term reasoning, active context. Ring buffer with Redis.
"""

import json
from datetime import datetime
from typing import Optional

import redis.asyncio as redis

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.stores.base import BaseStore


class L0WorkingBuffer(BaseStore):
    """L0: Working Buffer - volatile short-term context."""

    level = StoreLevel.L0
    KEY_PREFIX = "cls:l0"
    LIST_KEY = "cls:l0:ring"
    TTL = 300  # 5 minutes default

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
            )
        return self._client

    def _ns_key(self, namespace: str) -> str:
        return f"{self.KEY_PREFIX}:{namespace}"

    async def write(self, item: MemoryItem) -> MemoryItem:
        """Write to working buffer. Evicts oldest if over capacity."""
        item.store_level = StoreLevel.L0
        key = f"{self._ns_key(item.namespace)}:{item.id}"
        data = json.dumps(item.to_dict(), default=str)
        pipe = self.client.pipeline()
        pipe.set(key, data, ex=self.settings.l0_ttl_seconds)
        pipe.lpush(f"{self.LIST_KEY}:{item.namespace}", item.id)
        pipe.ltrim(f"{self.LIST_KEY}:{item.namespace}", 0, 1000)  # Cap ring size
        await pipe.execute()
        return item

    async def read(
        self,
        query_embedding: list[float],
        namespace: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryItem]:
        """Read from buffer - returns most recent (no semantic search at L0)."""
        ids = await self.client.lrange(f"{self.LIST_KEY}:{namespace}", 0, limit - 1)
        items = []
        for iid in ids:
            data = await self.client.get(f"{self._ns_key(namespace)}:{iid}")
            if data:
                item = MemoryItem.from_dict(json.loads(data))
                if item.confidence >= min_confidence:
                    items.append(item)
        return items[:limit]

    async def get_by_id(self, item_id: str, namespace: str) -> Optional[MemoryItem]:
        """Get item by ID."""
        data = await self.client.get(f"{self._ns_key(namespace)}:{item_id}")
        if data:
            return MemoryItem.from_dict(json.loads(data))
        return None

    async def delete(self, item_id: str, namespace: str) -> bool:
        """Delete from buffer. Returns True only if item existed."""
        key = f"{self._ns_key(namespace)}:{item_id}"
        deleted_count = await self.client.delete(key)
        await self.client.lrem(f"{self.LIST_KEY}:{namespace}", 1, item_id)
        return deleted_count > 0

    async def list_ids(self, namespace: str, limit: int = 100) -> list[str]:
        """List item IDs for sleep cycle."""
        return await self.client.lrange(f"{self.LIST_KEY}:{namespace}", 0, limit - 1)

    async def close(self) -> None:
        """Cleanly shut down the Redis client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health(self) -> dict:
        """Health check."""
        try:
            await self.client.ping()
            return {"status": "healthy", "store": "L0"}
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("L0 health check failed: %s", e)
            return {"status": "unhealthy", "store": "L0", "error": "Connection failed"}
