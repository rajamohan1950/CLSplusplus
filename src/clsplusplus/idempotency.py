"""CLS++ idempotency - prevent duplicate operations from retries."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from clsplusplus.config import Settings

_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


def _cache_key(key: str, method: str, path: str, body: bytes) -> str:
    h = hashlib.sha256(f"{method}:{path}:{body}".encode()).hexdigest()
    return f"cls:idempotency:{key}:{h[:32]}"


async def get_cached_response(
    idempotency_key: str,
    method: str,
    path: str,
    body: bytes,
    settings: Optional[Settings] = None,
) -> Optional[dict[str, Any]]:
    """Return cached response if idempotent request was already processed."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        ckey = _cache_key(idempotency_key, method, path, body)
        raw = await client.get(ckey)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


async def cache_response(
    idempotency_key: str,
    method: str,
    path: str,
    body: bytes,
    status: int,
    response_body: dict,
    settings: Optional[Settings] = None,
) -> None:
    """Cache response for idempotent request."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        ckey = _cache_key(idempotency_key, method, path, body)
        await client.setex(
            ckey,
            settings.idempotency_ttl_seconds,
            json.dumps({"status": status, "body": response_body}),
        )
    except Exception:
        pass
