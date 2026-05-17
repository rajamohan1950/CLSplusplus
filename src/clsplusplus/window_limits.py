"""CLS++ multi-window usage caps for the free tier.

The monthly quota in `tiers.check_quota` is the billing-correctness cap.
These window limits are the *cost-safety* cap: a free user (or a runaway
client / leaked free key) must not be able to burn a month's worth of
billable downstream calls in a single hour.

Four windows are enforced, all keyed on the per-user billing subject
(see `usage.make_subject`) so every api key a user owns aggregates onto
one set of counters:

    cls:win:hour:{subject}:{YYYY-MM-DD-HH}
    cls:win:day:{subject}:{YYYY-MM-DD}
    cls:win:week:{subject}:{YYYY-Www}      (ISO week)
    cls:win:month:{subject}:{YYYY-MM}

Each counter is INCR'd and given a TTL slightly longer than its window
so stale buckets self-evict — no sweeper needed.

Design notes:
  * Only the `free` tier is checked. Paid tiers keep the monthly cap only.
  * The window check fails OPEN: a Redis outage must not lock out every
    free user. The monthly cap (`tiers.check_quota`) keeps its existing
    fail-closed behaviour — the two are intentionally different because
    the monthly cap is billing-critical and the window cap is a guard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.usage import _redis_client


# Window definitions: period name → (TTL seconds, slightly > the window).
# The TTL margin absorbs clock skew and ensures a bucket outlives the
# window it represents before Redis evicts it.
_WINDOW_TTL_SECONDS: dict[str, int] = {
    "hour": 60 * 60 + 300,             # 1h + 5m
    "day": 60 * 60 * 24 + 3600,        # 1d + 1h
    "week": 60 * 60 * 24 * 7 + 3600,   # 7d + 1h
    "month": 60 * 60 * 24 * 35,        # 35d (matches usage.record_operation)
}

# Fixed iteration order: report the *shortest* breached window first so the
# 429 Retry-After is the most actionable value for the caller.
_WINDOW_ORDER = ("hour", "day", "week", "month")


def _bucket(period: str, now: Optional[datetime] = None) -> str:
    """Return the current bucket string for a window period."""
    now = now or datetime.utcnow()
    if period == "hour":
        return now.strftime("%Y-%m-%d-%H")
    if period == "day":
        return now.strftime("%Y-%m-%d")
    if period == "week":
        iso_year, iso_week, _ = now.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if period == "month":
        return now.strftime("%Y-%m")
    raise ValueError(f"unknown window period: {period}")


def _window_key(period: str, subject: str, now: Optional[datetime] = None) -> str:
    return f"cls:win:{period}:{subject}:{_bucket(period, now)}"


def _limits(settings: Settings) -> dict[str, int]:
    """Free-tier caps per window, from settings (all env-overridable)."""
    return {
        "hour": settings.free_cap_per_hour,
        "day": settings.free_cap_per_day,
        "week": settings.free_cap_per_week,
        "month": settings.free_cap_per_month,
    }


async def record_window_operation(
    subject: str,
    settings: Optional[Settings] = None,
) -> None:
    """Increment all four window counters for one billable operation.

    Mirrors `usage.record_operation`: best-effort, swallows Redis errors
    so metering never blocks a request. Call this only for free-tier
    subjects — paid tiers don't consult these counters.
    """
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        now = datetime.utcnow()
        for period in _WINDOW_ORDER:
            key = _window_key(period, subject, now)
            await client.incr(key)
            await client.expire(key, _WINDOW_TTL_SECONDS[period])
    except Exception:
        # Fail-open: a counter we couldn't write just means the next
        # check sees a slightly lower number. Never block on this.
        pass


async def check_window_limits(
    subject: str,
    settings: Optional[Settings] = None,
) -> tuple[bool, Optional[str], int, int]:
    """Check a free-tier subject against all four usage windows.

    Returns (allowed, which_window, used, limit):
      * allowed       — False if any window is at/over its cap.
      * which_window  — name of the *shortest* breached window, else None.
      * used          — current count in that window (0 when allowed).
      * limit         — cap for that window (0 when allowed).

    Fails OPEN on any Redis error — returns (True, None, 0, 0) — so an
    outage degrades to "monthly cap only" rather than locking everyone
    out. The monthly billing cap keeps its own fail-closed behaviour.
    """
    settings = settings or Settings()
    limits = _limits(settings)
    try:
        client = _redis_client(settings.redis_url)
        now = datetime.utcnow()
        for period in _WINDOW_ORDER:
            limit = limits[period]
            if limit < 0:
                continue  # negative = unlimited for that window
            raw = await client.get(_window_key(period, subject, now))
            used = int(raw) if raw else 0
            if used >= limit:
                return (False, period, used, limit)
        return (True, None, 0, 0)
    except Exception:
        # Fail-open — see docstring.
        return (True, None, 0, 0)


def retry_after_seconds(which_window: Optional[str]) -> int:
    """Conservative Retry-After (seconds) for a breached window.

    We don't know exactly when the bucket rolls over, so we return the
    full window length — a safe upper bound the client can back off by.
    """
    return {
        "hour": 3600,
        "day": 86400,
        "week": 604800,
        "month": 2592000,
    }.get(which_window or "", 60)
