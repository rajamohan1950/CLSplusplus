"""CLS++ usage tracking for marketplace billing."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from clsplusplus.config import Settings

_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


def _period_key() -> str:
    """Current period YYYY-MM."""
    return datetime.utcnow().strftime("%Y-%m")


async def record_usage(
    api_key: str,
    operation: str,
    settings: Optional[Settings] = None,
) -> None:
    """Record a usage event (write, read, etc.)."""
    settings = settings or Settings()
    if not settings.track_usage:
        return
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:usage:{api_key}:{_period_key()}"
        await client.hincrby(key, operation, 1)
        await client.expire(key, 60 * 60 * 24 * 35)  # 35 days
    except Exception:
        pass


async def record_operation(
    api_key: str,
    settings: Optional[Settings] = None,
) -> None:
    """Increment the unified operations counter for billing."""
    settings = settings or Settings()
    if not settings.track_usage:
        return
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:ops:{api_key}:{_period_key()}"
        await client.incr(key)
        await client.expire(key, 60 * 60 * 24 * 35)  # 35 days
    except Exception:
        pass


async def get_operation_count(
    api_key: str,
    settings: Optional[Settings] = None,
) -> int:
    """Get unified operation count for current period."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:ops:{api_key}:{_period_key()}"
        val = await client.get(key)
        return int(val) if val else 0
    except Exception:
        return 0


async def get_usage(
    api_key: str,
    settings: Optional[Settings] = None,
) -> dict:
    """Get usage for current period."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:usage:{api_key}:{_period_key()}"
        data = await client.hgetall(key)
        writes = int(data.get("write", 0)) + int(data.get("encode", 0))
        reads = int(data.get("read", 0)) + int(data.get("retrieve", 0)) + int(data.get("knowledge", 0))
        return {
            "writes": writes,
            "reads": reads,
            "period": _period_key(),
        }
    except Exception:
        return {"writes": 0, "reads": 0, "period": _period_key()}


async def get_usage_history(
    namespace: str,
    months: int = 6,
    settings: Optional[Settings] = None,
) -> list[dict]:
    """Return usage for the last N months."""
    settings = settings or Settings()
    history = []
    now = datetime.utcnow()
    for i in range(months - 1, -1, -1):
        # Calculate past month
        year = now.year
        month = now.month - i
        while month <= 0:
            month += 12
            year -= 1
        period = f"{year:04d}-{month:02d}"
        try:
            client = _redis_client(settings.redis_url)
            ops_key = f"cls:ops:{namespace}:{period}"
            usage_key = f"cls:usage:{namespace}:{period}"
            ops = await client.get(ops_key)
            data = await client.hgetall(usage_key)
            writes = int(data.get("write", 0)) + int(data.get("encode", 0))
            reads = int(data.get("read", 0)) + int(data.get("retrieve", 0)) + int(data.get("knowledge", 0))
            history.append({
                "period": period,
                "operations": int(ops) if ops else 0,
                "writes": writes,
                "reads": reads,
            })
        except Exception:
            history.append({"period": period, "operations": 0, "writes": 0, "reads": 0})
    return history
