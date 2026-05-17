"""CLS++ rate limiting - Redis sliding window."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from clsplusplus.config import Settings

_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    """Lazy import; reuse connection per URL."""
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


async def _sliding_window(
    rkey: str,
    limit: int,
    window: int,
    redis_url: str,
) -> tuple[bool, int, int]:
    """Core sliding-window check. Returns (allowed, count, limit).

    Fails OPEN (allowed=True) on any Redis error so a Redis blip never
    locks users out.
    """
    now = time.time()
    window_start = now - window
    try:
        client = _redis_client(redis_url)
        pipe = client.pipeline()
        pipe.zremrangebyscore(rkey, "-inf", window_start)
        pipe.zcard(rkey)
        results = await pipe.execute()
        count = results[1] if len(results) > 1 else 0
        allowed = count < limit
        if allowed:
            pipe2 = client.pipeline()
            pipe2.zadd(rkey, {str(uuid.uuid4()): now})
            pipe2.expire(rkey, window + 60)
            await pipe2.execute()
            count += 1
        return (allowed, count, limit)
    except Exception:
        return (True, 0, limit)  # Fail open on Redis errors


async def check_auth_rate_limit(
    ip: str,
    settings: Optional[Settings] = None,
) -> tuple[bool, int, int]:
    """Tight per-IP throttle for the PUBLIC auth endpoints.

    The unauthenticated auth endpoints (login/register/waitlist) bypass the
    normal per-API-key limiter because they live in `_PUBLIC_PATHS`. An
    unauthenticated storm has no API key, so per-IP is the only handle we
    have. Separate Redis key namespace (`cls:authlimit:`) so it never
    collides with the per-key limiter. Fails OPEN on Redis errors.
    """
    settings = settings or Settings()
    limit = settings.auth_rate_limit_per_ip
    window = settings.auth_rate_limit_window_seconds
    rkey = f"cls:authlimit:{ip}"
    return await _sliding_window(rkey, limit, window, settings.redis_url)


async def check_rate_limit(
    key_id: str,
    settings: Optional[Settings] = None,
) -> tuple[bool, int, int]:
    """
    Sliding window rate limit. Returns (allowed, current_count, limit).
    If allowed, records the request. key_id: tenant/API key identifier.
    """
    settings = settings or Settings()
    limit = settings.rate_limit_requests
    window = settings.rate_limit_window_seconds
    rkey = f"cls:ratelimit:{key_id}"
    now = time.time()
    window_start = now - window

    try:
        client = _redis_client(settings.redis_url)
        pipe = client.pipeline()
        pipe.zremrangebyscore(rkey, "-inf", window_start)
        pipe.zcard(rkey)
        results = await pipe.execute()
        count = results[1] if len(results) > 1 else 0
        allowed = count < limit
        if allowed:
            pipe2 = client.pipeline()
            pipe2.zadd(rkey, {str(uuid.uuid4()): now})
            pipe2.expire(rkey, window + 60)
            await pipe2.execute()
            count += 1
        return (allowed, count, limit)
    except Exception:
        return (True, 0, limit)  # Fail open on Redis errors
