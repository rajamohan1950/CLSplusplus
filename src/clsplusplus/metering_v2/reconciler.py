"""Daily reconciliation of the durable log against the Redis counters
(ADR 0001 step 3).

Why this exists
---------------
Step 2 dual-writes every metered call to both Redis (`cls:ops:{api_key}:{period}`,
fast counter) and Postgres (`usage_events`, durable log). They should
agree to the last op. If they don't — a write dropped, a retry
double-counted, a schema drift ate a field — billing is at risk.

The reconciler compares the two views and pages on-call when drift
crosses a configurable tolerance. Drift findings are written into the
same `metering_dead_letter` table the writer uses, so the existing
notifier emails them without needing a second transport.

Because the ADR's back-dating window is 1 week, we MUST catch drift
within 7 days. This runs daily and covers both the current and the
prior period on every cycle, giving the on-call at most 48 h to react.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)


# ADR 0001 §1: "alerts on > 0.1% drift".
DEFAULT_DRIFT_PERCENT = 0.001
# Ignore small absolute deltas — they're noise from the async write window.
DEFAULT_MIN_ABS_DRIFT = 5
# Daily loop interval.
POLL_INTERVAL_SECONDS = 24 * 60 * 60


PoolGetter = Callable[[], Awaitable[Any]]
RedisGetter = Callable[[], Awaitable[Any]]


def _hash_key(raw_key: str) -> str:
    """Match MeteringWriter's api_key_id column: first 16 hex of SHA-256."""
    return hashlib.sha256(raw_key.encode()).hexdigest()[:16]


def _period_window(period: str) -> tuple[datetime, datetime]:
    """Turn 'YYYY-MM' into [start, end) timestamps (UTC)."""
    year, month = map(int, period.split("-"))
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start, end


def _current_period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _prior_period() -> str:
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month - 1
    if month == 0:
        year, month = year - 1, 12
    return f"{year:04d}-{month:02d}"


@dataclass
class DriftFinding:
    api_key_hash: str
    period: str
    redis_count: int
    postgres_count: int
    drift: int
    drift_pct: float

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass
class ReconciliationResult:
    period: str
    redis_keys_seen: int
    postgres_aggregates_seen: int
    findings: list[DriftFinding]
    elapsed_seconds: float

    def summary(self) -> dict:
        return {
            "period": self.period,
            "redis_keys_seen": self.redis_keys_seen,
            "postgres_aggregates_seen": self.postgres_aggregates_seen,
            "drift_count": len(self.findings),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


class MeteringReconciler:
    """Compares Redis ops counters to usage_events aggregates.

    `reconcile_once(period)` runs a single comparison and returns findings.
    `start()/stop()` runs the daily loop (current + prior period each cycle).
    """

    def __init__(
        self,
        settings: Settings,
        pool_getter: PoolGetter,
        redis_getter: RedisGetter,
        drift_percent: float = DEFAULT_DRIFT_PERCENT,
        min_abs_drift: int = DEFAULT_MIN_ABS_DRIFT,
    ):
        self.settings = settings
        self._pool_getter = pool_getter
        self._redis_getter = redis_getter
        self._drift_pct = drift_percent
        self._min_abs = min_abs_drift
        self._task: Optional[asyncio.Task] = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.metering_v2_write_enabled)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        if not self.enabled:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def reconcile_once(self, period: Optional[str] = None) -> ReconciliationResult:
        """Compare the two views for one period. Writes drift to dead_letter.

        Returns the findings for the caller to log / return from an
        admin endpoint. The caller does NOT need to write dead_letter
        rows — we've already done that.
        """
        t0 = asyncio.get_event_loop().time()
        period = period or _current_period()

        redis_counts = await self._scan_redis(period)
        pg_counts = await self._aggregate_postgres(period)

        findings = self._compare(period, redis_counts, pg_counts)
        if findings:
            await self._enqueue_findings(findings)

        return ReconciliationResult(
            period=period,
            redis_keys_seen=len(redis_counts),
            postgres_aggregates_seen=len(pg_counts),
            findings=findings,
            elapsed_seconds=asyncio.get_event_loop().time() - t0,
        )

    # --- internals ---------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                await self.reconcile_once(_current_period())
                await self.reconcile_once(_prior_period())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("metering reconciler: %s: %s",
                             type(exc).__name__, exc)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _scan_redis(self, period: str) -> dict[str, int]:
        """Return {api_key_hash: ops_count} from Redis for the period.

        Keys in Redis are `cls:ops:{RAW_api_key}:{period}` — we hash the
        raw key on our side to match the writer's api_key_id scheme.
        """
        client = await self._redis_getter()
        if client is None:
            return {}
        counts: dict[str, int] = {}
        pattern = f"cls:ops:*:{period}"
        try:
            async for key in client.scan_iter(match=pattern, count=500):
                # key = "cls:ops:{raw_api_key}:{period}"
                # Strip prefix + suffix to get the raw key.
                suffix = f":{period}"
                if not key.endswith(suffix):
                    continue
                raw_key = key[len("cls:ops:"):-len(suffix)]
                if not raw_key:
                    continue
                val = await client.get(key)
                try:
                    n = int(val) if val else 0
                except (TypeError, ValueError):
                    n = 0
                if n > 0:
                    counts[_hash_key(raw_key)] = counts.get(_hash_key(raw_key), 0) + n
        except Exception as exc:
            logger.error("reconciler: redis scan failed: %s: %s",
                         type(exc).__name__, exc)
        return counts

    async def _aggregate_postgres(self, period: str) -> dict[str, int]:
        """Return {api_key_id: sum(quantity)} from usage_events for the period."""
        start, end = _period_window(period)
        pool = await self._pool_getter()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT api_key_id, SUM(quantity)::BIGINT AS total
                FROM usage_events
                WHERE api_key_id IS NOT NULL
                  AND occurred_at >= $1
                  AND occurred_at <  $2
                GROUP BY api_key_id
                """,
                start, end,
            )
        return {r["api_key_id"]: int(r["total"]) for r in rows if r["api_key_id"]}

    def _compare(
        self,
        period: str,
        redis_counts: dict[str, int],
        pg_counts: dict[str, int],
    ) -> list[DriftFinding]:
        findings: list[DriftFinding] = []
        all_hashes = set(redis_counts) | set(pg_counts)
        for h in all_hashes:
            r = redis_counts.get(h, 0)
            p = pg_counts.get(h, 0)
            drift = abs(r - p)
            if drift <= self._min_abs:
                continue
            denom = max(r, p, 1)
            pct = drift / denom
            if pct <= self._drift_pct:
                continue
            findings.append(DriftFinding(
                api_key_hash=h,
                period=period,
                redis_count=r,
                postgres_count=p,
                drift=drift,
                drift_pct=round(pct, 6),
            ))
        return findings

    async def _enqueue_findings(self, findings: list[DriftFinding]) -> None:
        """Write drift findings into metering_dead_letter. The existing
        notifier picks them up on its next pump cycle."""
        pool = await self._pool_getter()
        async with pool.acquire() as conn:
            for f in findings:
                try:
                    await conn.execute(
                        """
                        INSERT INTO metering_dead_letter
                            (error_class, error_message, payload)
                        VALUES ($1, $2, $3::jsonb)
                        """,
                        "ReconciliationDrift",
                        (
                            f"period={f.period} key={f.api_key_hash} "
                            f"redis={f.redis_count} pg={f.postgres_count} "
                            f"drift={f.drift} ({f.drift_pct * 100:.3f}%)"
                        )[:500],
                        json.dumps(f.to_payload()),
                    )
                except Exception as exc:
                    logger.error(
                        "reconciler: could not enqueue finding: %s: %s",
                        type(exc).__name__, exc,
                    )
