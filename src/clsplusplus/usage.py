"""CLS++ usage tracking for marketplace billing.

Counters are keyed on a *billing subject* — a stable id derived from the
api key's owning user — not the raw api key. This is what makes the
monthly quota aggregate per-user: every api key a user owns collapses
onto one `owner:` subject. Legacy keys with no resolvable owner fall
back to a per-key subject, preserving their old isolated behaviour.
See `make_subject` and `clsplusplus.tier_resolver.TierResolver`.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Awaitable, Callable, Optional

from clsplusplus.config import Settings

_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        # Socket + connect timeouts so a dead/slow Redis fails fast instead
        # of hanging every billing call; callers already fail soft on error.
        _redis_client_cache[redis_url] = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            health_check_interval=30,
        )
    return _redis_client_cache[redis_url]


def _period_key() -> str:
    """Current period YYYY-MM."""
    return datetime.utcnow().strftime("%Y-%m")


def make_subject(owner_email: Optional[str], api_key: str) -> str:
    """Stable billing-subject id for usage counters.

    When the api key resolves to an integration owner, usage from *all*
    of that user's keys collapses onto one `owner:` subject so the
    monthly quota aggregates per-user. Legacy keys with no resolvable
    owner fall back to a per-key `key:` subject — unchanged behaviour.
    """
    if owner_email:
        digest = hashlib.sha256(owner_email.encode()).hexdigest()[:16]
        return f"owner:{digest}"
    digest = hashlib.sha256((api_key or "").encode()).hexdigest()[:16]
    return f"key:{digest}"


async def record_usage(
    subject: str,
    operation: str,
    settings: Optional[Settings] = None,
) -> None:
    """Record a per-operation usage event (write, read, etc.).

    This is the detailed breakdown shown on the usage dashboard and is
    opt-in via `track_usage`. The billing-critical unified counter is
    `record_operation`, which is always written.
    """
    settings = settings or Settings()
    if not settings.track_usage:
        return
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:usage:{subject}:{_period_key()}"
        await client.hincrby(key, operation, 1)
        await client.expire(key, 60 * 60 * 24 * 35)  # 35 days
    except Exception:
        pass


async def record_operation(
    subject: str,
    settings: Optional[Settings] = None,
) -> None:
    """Increment the unified operations counter for billing.

    Always written — quota enforcement reads this counter, so gating it
    behind `track_usage` would silently disable the monthly cap.
    """
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:ops:{subject}:{_period_key()}"
        await client.incr(key)
        await client.expire(key, 60 * 60 * 24 * 35)  # 35 days
    except Exception:
        pass


async def get_operation_count(
    subject: str,
    settings: Optional[Settings] = None,
) -> int:
    """Get unified operation count for current period."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:ops:{subject}:{_period_key()}"
        val = await client.get(key)
        return int(val) if val else 0
    except Exception:
        return 0


async def get_usage(
    subject: str,
    settings: Optional[Settings] = None,
) -> dict:
    """Get usage for current period."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        key = f"cls:usage:{subject}:{_period_key()}"
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


def _period_list(months: int) -> list[str]:
    """Return the last N periods (YYYY-MM), oldest first, ending this month."""
    now = datetime.utcnow()
    periods = []
    for i in range(months - 1, -1, -1):
        year, month = now.year, now.month - i
        while month <= 0:
            month += 12
            year -= 1
        periods.append(f"{year:04d}-{month:02d}")
    return periods


async def _usage_history_durable(
    subject: str,
    periods: list[str],
    pool_getter: Callable[[], Awaitable[object]],
) -> list[dict]:
    """Read usage history from the durable usage_events log.

    Redis counters expire after 35 days, so anything older than ~1 month
    is only recoverable from usage_events. Aggregates by month for the
    given billing subject.
    """
    pool = await pool_getter()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT to_char(date_trunc('month', occurred_at), 'YYYY-MM') AS period,
                   event_type,
                   SUM(quantity)::BIGINT AS total
            FROM usage_events
            WHERE billing_subject = $1
              AND occurred_at >= $2::date
            GROUP BY 1, 2
            """,
            subject,
            f"{periods[0]}-01",
        )
    by_period: dict[str, dict[str, int]] = {p: {} for p in periods}
    for r in rows:
        by_period.setdefault(r["period"], {})[r["event_type"]] = int(r["total"])
    history = []
    for period in periods:
        ev = by_period.get(period, {})
        writes = ev.get("write", 0) + ev.get("encode", 0)
        reads = ev.get("read", 0) + ev.get("retrieve", 0) + ev.get("knowledge", 0)
        history.append({
            "period": period,
            "operations": sum(ev.values()),
            "writes": writes,
            "reads": reads,
        })
    return history


async def get_usage_history(
    subject: str,
    months: int = 6,
    settings: Optional[Settings] = None,
    pool_getter: Optional[Callable[[], Awaitable[object]]] = None,
) -> list[dict]:
    """Return usage for the last N months.

    When metering v2 is enabled and a DB pool is available, the full
    window is read from the durable `usage_events` log. Otherwise this
    falls back to Redis, which only retains ~35 days — older months
    will read as zero.
    """
    settings = settings or Settings()
    periods = _period_list(months)

    if pool_getter is not None and settings.metering_v2_write_enabled:
        try:
            return await _usage_history_durable(subject, periods, pool_getter)
        except Exception:
            pass  # fall through to the Redis path

    history = []
    for period in periods:
        try:
            client = _redis_client(settings.redis_url)
            ops_key = f"cls:ops:{subject}:{period}"
            usage_key = f"cls:usage:{subject}:{period}"
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
