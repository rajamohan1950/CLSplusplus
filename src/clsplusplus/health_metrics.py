"""CLS++ operational-health metrics — live per-request instrumentation.

Records, per HTTP request, the status code + route template + latency into
Redis rolling per-minute buckets. The admin dashboard aggregates these into
error rate, latency percentiles, request volume, top error codes, and the
slowest routes — so the owner can see service health at a glance.

Design goals:
  * O(1) writes — a handful of HINCRBY / LPUSH / EXPIRE per request.
  * Fail-open — any Redis error is swallowed; a request is NEVER slowed
    or failed because health recording broke.
  * Bounded memory — one bucket per UTC minute, each auto-expires after
    the rolling window; the latency reservoir per (minute,route) is capped.

Redis keyspace (all under `cls:health:`):
  cls:health:status:{YYYYMMDDHHMM}      HASH  status_code -> count
  cls:health:vol:{YYYYMMDDHHMM}         STRING (counter) total requests
  cls:health:rtcount:{YYYYMMDDHHMM}     HASH  route -> request count
  cls:health:rtlatsum:{YYYYMMDDHHMM}    HASH  route -> summed latency (ms)
  cls:health:lat:{YYYYMMDDHHMM}         LIST  capped reservoir of latency ms
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_KEY_PREFIX = "cls:health:"

# Rolling window aggregated by the admin endpoint, in minutes.
DEFAULT_WINDOW_MINUTES = 60

# Per-minute latency reservoir cap. Percentiles are estimated from this
# sample; 2000/min is plenty for stable p50/p95/p99 and bounds memory.
_LAT_RESERVOIR_CAP = 2000

# Buckets live a little longer than the longest window we'd query.
_BUCKET_TTL_SECONDS = (DEFAULT_WINDOW_MINUTES + 10) * 60

_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    """Lazy import; reuse one async client per URL (mirrors rate_limit.py)."""
    import redis.asyncio as redis

    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


def _bucket(dt: datetime) -> str:
    """Minute-resolution bucket id for a UTC datetime."""
    return dt.strftime("%Y%m%d%H%M")


def _recent_buckets(window_minutes: int) -> list[str]:
    """Bucket ids covering the last `window_minutes`, newest last."""
    now = datetime.utcnow()
    return [_bucket(now - timedelta(minutes=i)) for i in range(window_minutes - 1, -1, -1)]


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list. pct in [0,100]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac, 2)


async def record_request(
    redis_url: str,
    route: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """Record one completed request. Fire-and-forget; fails open.

    `route` should be the route *template* (e.g. ``/v1/memory/{id}``) so
    cardinality stays bounded — never the raw path with ids inlined.
    """
    try:
        client = _redis_client(redis_url)
        bucket = _bucket(datetime.utcnow())
        latency_ms = max(0.0, float(latency_ms))
        route = (route or "unknown")[:120]

        status_key = f"{_KEY_PREFIX}status:{bucket}"
        vol_key = f"{_KEY_PREFIX}vol:{bucket}"
        rtcount_key = f"{_KEY_PREFIX}rtcount:{bucket}"
        rtlatsum_key = f"{_KEY_PREFIX}rtlatsum:{bucket}"
        lat_key = f"{_KEY_PREFIX}lat:{bucket}"

        pipe = client.pipeline()
        pipe.hincrby(status_key, str(status_code), 1)
        pipe.expire(status_key, _BUCKET_TTL_SECONDS)
        pipe.incr(vol_key)
        pipe.expire(vol_key, _BUCKET_TTL_SECONDS)
        pipe.hincrby(rtcount_key, route, 1)
        pipe.expire(rtcount_key, _BUCKET_TTL_SECONDS)
        # Latency summed in micro-resolution-safe integers (×100) so HINCRBY
        # stays integer; divided back out on read.
        pipe.hincrby(rtlatsum_key, route, int(latency_ms * 100))
        pipe.expire(rtlatsum_key, _BUCKET_TTL_SECONDS)
        pipe.lpush(lat_key, f"{latency_ms:.2f}")
        pipe.ltrim(lat_key, 0, _LAT_RESERVOIR_CAP - 1)
        pipe.expire(lat_key, _BUCKET_TTL_SECONDS)
        await pipe.execute()
    except Exception:
        # Fail-open: health recording must never affect the live request.
        pass


async def aggregate_health(
    redis_url: str,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> dict:
    """Aggregate the rolling window into an operational-health report.

    Returns zeros gracefully when there is no data yet. Fails open: on a
    Redis error it returns an empty report with ``"degraded": true`` so the
    dashboard can show a banner rather than break.
    """
    window_minutes = max(1, min(window_minutes, DEFAULT_WINDOW_MINUTES))
    buckets = _recent_buckets(window_minutes)
    empty = _empty_report(window_minutes)

    try:
        client = _redis_client(redis_url)

        status_totals: dict[int, int] = {}
        route_count: dict[str, int] = {}
        route_latsum: dict[str, int] = {}
        latencies: list[float] = []
        total_requests = 0

        for bucket in buckets:
            pipe = client.pipeline()
            pipe.hgetall(f"{_KEY_PREFIX}status:{bucket}")
            pipe.get(f"{_KEY_PREFIX}vol:{bucket}")
            pipe.hgetall(f"{_KEY_PREFIX}rtcount:{bucket}")
            pipe.hgetall(f"{_KEY_PREFIX}rtlatsum:{bucket}")
            pipe.lrange(f"{_KEY_PREFIX}lat:{bucket}", 0, -1)
            statuses, vol, rtcount, rtlatsum, lats = await pipe.execute()

            for code, cnt in (statuses or {}).items():
                try:
                    status_totals[int(code)] = status_totals.get(int(code), 0) + int(cnt)
                except (ValueError, TypeError):
                    continue
            if vol:
                try:
                    total_requests += int(vol)
                except (ValueError, TypeError):
                    pass
            for rt, cnt in (rtcount or {}).items():
                try:
                    route_count[rt] = route_count.get(rt, 0) + int(cnt)
                except (ValueError, TypeError):
                    continue
            for rt, s in (rtlatsum or {}).items():
                try:
                    route_latsum[rt] = route_latsum.get(rt, 0) + int(s)
                except (ValueError, TypeError):
                    continue
            for v in (lats or []):
                try:
                    latencies.append(float(v))
                except (ValueError, TypeError):
                    continue

        return _build_report(
            window_minutes,
            status_totals,
            route_count,
            route_latsum,
            latencies,
            total_requests,
        )
    except Exception as exc:
        logger.warning("health_metrics aggregate failed: %s: %s", type(exc).__name__, exc)
        empty["degraded"] = True
        return empty


def _empty_report(window_minutes: int) -> dict:
    return {
        "window_minutes": window_minutes,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "degraded": False,
        "total_requests": 0,
        "requests_per_minute": 0.0,
        "error_rate": 0.0,
        "error_rate_4xx": 0.0,
        "error_rate_5xx": 0.0,
        "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "sample_size": 0},
        "status_breakdown": {},
        "top_error_codes": [],
        "guard_counts": {"quota_402": 0, "rate_limit_429": 0, "blocked_403": 0},
        "slowest_routes": [],
    }


def _build_report(
    window_minutes: int,
    status_totals: dict[int, int],
    route_count: dict[str, int],
    route_latsum: dict[str, int],
    latencies: list[float],
    total_requests: int,
) -> dict:
    """Pure aggregation — kept dependency-free so it is trivially testable."""
    report = _empty_report(window_minutes)

    # Volume — prefer the dedicated counter, fall back to summing statuses
    # (the status hash and the volume counter are written together, so they
    # agree; the fallback only matters for very old/partial buckets).
    status_sum = sum(status_totals.values())
    total = total_requests or status_sum
    report["total_requests"] = total
    report["requests_per_minute"] = round(total / window_minutes, 2)

    # Error rate — use status_sum as the denominator for rate math so the
    # numerator (4xx/5xx counts) and denominator come from the same source.
    denom = status_sum or total
    count_4xx = sum(c for code, c in status_totals.items() if 400 <= code < 500)
    count_5xx = sum(c for code, c in status_totals.items() if 500 <= code < 600)
    if denom > 0:
        report["error_rate"] = round((count_4xx + count_5xx) / denom * 100, 2)
        report["error_rate_4xx"] = round(count_4xx / denom * 100, 2)
        report["error_rate_5xx"] = round(count_5xx / denom * 100, 2)

    # Status breakdown + top error codes
    report["status_breakdown"] = {
        str(code): cnt for code, cnt in sorted(status_totals.items())
    }
    errors = sorted(
        ((code, cnt) for code, cnt in status_totals.items() if code >= 400),
        key=lambda kv: kv[1],
        reverse=True,
    )
    report["top_error_codes"] = [
        {"status": code, "count": cnt} for code, cnt in errors[:8]
    ]

    # Guard-signal counts the owner cares about specifically.
    report["guard_counts"] = {
        "quota_402": status_totals.get(402, 0),
        "rate_limit_429": status_totals.get(429, 0),
        "blocked_403": status_totals.get(403, 0),
    }

    # Latency percentiles from the reservoir sample.
    latencies.sort()
    report["latency_ms"] = {
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
        "sample_size": len(latencies),
    }

    # Slowest routes by mean latency (latsum stored ×100). Require >=1 call.
    slowest = []
    for route, cnt in route_count.items():
        if cnt <= 0:
            continue
        mean_ms = (route_latsum.get(route, 0) / 100.0) / cnt
        slowest.append(
            {"route": route, "requests": cnt, "avg_latency_ms": round(mean_ms, 2)}
        )
    slowest.sort(key=lambda r: r["avg_latency_ms"], reverse=True)
    report["slowest_routes"] = slowest[:10]

    return report


def route_template(request) -> str:
    """Best-effort route template for a Starlette request.

    Uses the matched route's path (``/v1/memory/{id}``) when available so
    metrics don't explode in cardinality on path parameters. Falls back to
    ``METHOD /raw/path`` only when no route matched (e.g. a 404).
    """
    method = getattr(request, "method", "?")
    try:
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        if path:
            return f"{method} {path}"
    except Exception:
        pass
    try:
        return f"{method} {request.url.path}"
    except Exception:
        return f"{method} unknown"
