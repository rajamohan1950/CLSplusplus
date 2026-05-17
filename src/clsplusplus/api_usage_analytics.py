"""API-key consumption analytics for the admin dashboard.

Query logic lives here, isolated from api.py, so concurrent edits to the
shared router file stay minimal. The single public entry point is
`build_api_usage_report`, called by `GET /admin/metrics/api-usage`.

The durable source is the `usage_events` Postgres table (see
metering_v2_ddl.sql). When metering v2 is off — or the table is empty,
or anything raises — every aggregate degrades to whatever Redis still
holds (current-period op counters) or to zeros. This endpoint must
never crash the admin dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

# Default number of top users returned in the per-user breakdown.
DEFAULT_TOP_N = 25


def _empty_hours() -> list[dict[str, int]]:
    """A zeroed 24-bucket hour-of-day histogram, hour 0..23."""
    return [{"hour": h, "operations": 0} for h in range(24)]


async def _api_key_counts(conn: Any) -> dict[str, int]:
    """Total api keys and how many are *actively consumed*.

    `total` counts every non-revoked credential. `active` counts keys
    whose status is 'active'. `consumed` counts active keys that have a
    `last_used_at` stamp — i.e. a real caller has hit them at least once.
    """
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status != 'revoked')                       AS total,
            COUNT(*) FILTER (WHERE status = 'active')                         AS active,
            COUNT(*) FILTER (WHERE status = 'active' AND last_used_at IS NOT NULL) AS consumed
        FROM api_credentials
        """
    )
    return {
        "total": int(row["total"] or 0),
        "active": int(row["active"] or 0),
        "consumed": int(row["consumed"] or 0),
    }


async def _per_user_consumption(conn: Any, top_n: int) -> list[dict[str, Any]]:
    """Per-user operation consumption from usage_events, top N by ops.

    Aggregates `SUM(quantity)` grouped by `user_id`, joined to `users`
    for a human-readable email/name/tier. Rows with a NULL `user_id`
    (legacy / key-only events) are folded into a single synthetic
    "unattributed" bucket so the totals still reconcile.
    """
    rows = await conn.fetch(
        """
        SELECT
            ue.user_id::text                          AS user_id,
            u.email                                   AS email,
            u.name                                    AS name,
            u.tier                                    AS tier,
            SUM(ue.quantity)::BIGINT                  AS operations,
            COUNT(DISTINCT ue.api_key_id)             AS keys_used,
            MAX(ue.occurred_at)                       AS last_active
        FROM usage_events ue
        LEFT JOIN users u ON u.id = ue.user_id
        WHERE ue.event_type != 'healthcheck'
        GROUP BY ue.user_id, u.email, u.name, u.tier
        ORDER BY operations DESC
        """
    )
    result: list[dict[str, Any]] = []
    for r in rows:
        last_active = r["last_active"]
        result.append(
            {
                "user_id": r["user_id"],
                "email": r["email"] or ("(unattributed)" if not r["user_id"] else None),
                "name": r["name"] or "",
                "tier": r["tier"] or "",
                "operations": int(r["operations"] or 0),
                "keys_used": int(r["keys_used"] or 0),
                "last_active": last_active.isoformat() if last_active else None,
            }
        )
    return result[:top_n]


async def _active_hours(conn: Any) -> list[dict[str, int]]:
    """24-bucket hour-of-day histogram of operations.

    Groups `usage_events` by `EXTRACT(HOUR FROM occurred_at)`. Hours
    with no activity are still emitted as zero so the chart always has
    exactly 24 points. Hours are in UTC — the same clock the durable
    log records `occurred_at` in.
    """
    rows = await conn.fetch(
        """
        SELECT
            EXTRACT(HOUR FROM occurred_at)::int  AS hour,
            SUM(quantity)::BIGINT                AS operations
        FROM usage_events
        WHERE event_type != 'healthcheck'
        GROUP BY 1
        """
    )
    buckets = _empty_hours()
    for r in rows:
        h = int(r["hour"])
        if 0 <= h <= 23:
            buckets[h]["operations"] = int(r["operations"] or 0)
    return buckets


async def _redis_fallback(settings: Settings) -> dict[str, Any]:
    """Best-effort per-subject op counts from Redis when the durable log
    is unavailable.

    Redis only knows the *current* billing period and keys usage by
    billing-subject hash (not user_id), so the breakdown is coarser:
    one row per `cls:ops:{subject}:{period}` counter. Active-hours
    cannot be reconstructed from Redis, so that stays zeroed.
    """
    from datetime import datetime

    from clsplusplus.usage import _redis_client

    period = datetime.utcnow().strftime("%Y-%m")
    try:
        client = _redis_client(settings.redis_url)
        pattern = f"cls:ops:*:{period}"
        subjects: list[dict[str, Any]] = []
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=200)
            for key in keys:
                val = await client.get(key)
                # key shape: cls:ops:{subject}:{YYYY-MM}
                parts = key.split(":")
                subject = ":".join(parts[2:-1]) if len(parts) >= 4 else key
                subjects.append(
                    {
                        "user_id": None,
                        "email": f"subject:{subject}",
                        "name": "",
                        "tier": "",
                        "operations": int(val) if val else 0,
                        "keys_used": 0,
                        "last_active": None,
                    }
                )
            if cursor == 0:
                break
        subjects.sort(key=lambda s: s["operations"], reverse=True)
        return {"per_user": subjects, "total_operations": sum(s["operations"] for s in subjects)}
    except Exception as e:  # noqa: BLE001 - fallback path must never raise
        logger.warning("api-usage redis fallback failed: %s: %s", type(e).__name__, e)
        return {"per_user": [], "total_operations": 0}


async def build_api_usage_report(
    pool_getter: Callable[[], Awaitable[Any]],
    settings: Optional[Settings] = None,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Assemble the full api-usage analytics payload for the admin dashboard.

    `pool_getter` is an awaitable returning an asyncpg pool (the same
    Postgres holding `api_credentials` and `usage_events`).

    The returned dict is stable regardless of metering state:
      - api_keys:    {total, active, consumed}
      - per_user:    list of {user_id, email, name, tier, operations,
                     keys_used, last_active}, top N by operations
      - active_hours: 24 buckets {hour, operations}
      - total_operations: sum across all users
      - source: 'usage_events' | 'redis' | 'empty'
    """
    settings = settings or Settings()
    report: dict[str, Any] = {
        "api_keys": {"total": 0, "active": 0, "consumed": 0},
        "per_user": [],
        "active_hours": _empty_hours(),
        "total_operations": 0,
        "source": "empty",
        "top_n": top_n,
    }

    # --- API key counts: always from api_credentials (independent of metering) ---
    try:
        pool = await pool_getter()
        async with pool.acquire() as conn:
            report["api_keys"] = await _api_key_counts(conn)
    except Exception as e:  # noqa: BLE001
        logger.warning("api-usage key counts failed: %s: %s", type(e).__name__, e)

    # --- Consumption + active hours: durable usage_events when metering v2 is on ---
    if settings.metering_v2_write_enabled:
        try:
            pool = await pool_getter()
            async with pool.acquire() as conn:
                has_rows = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM usage_events "
                    "WHERE event_type != 'healthcheck')"
                )
                if has_rows:
                    per_user = await _per_user_consumption(conn, top_n)
                    active_hours = await _active_hours(conn)
                    report["per_user"] = per_user
                    report["active_hours"] = active_hours
                    report["total_operations"] = sum(
                        h["operations"] for h in active_hours
                    )
                    report["source"] = "usage_events"
                    return report
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "api-usage durable read failed, falling back: %s: %s",
                type(e).__name__, e,
            )

    # --- Fallback: Redis current-period counters (no hour breakdown available) ---
    fallback = await _redis_fallback(settings)
    if fallback["per_user"]:
        report["per_user"] = fallback["per_user"][:top_n]
        report["total_operations"] = fallback["total_operations"]
        report["source"] = "redis"

    return report
