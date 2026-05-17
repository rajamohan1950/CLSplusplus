"""CLS++ tier definitions and quota enforcement."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union

from clsplusplus.config import Settings


class Tier(str, Enum):
    free = "free"
    pro = "pro"
    business = "business"
    enterprise = "enterprise"


TIER_LIMITS: dict[Tier, dict] = {
    Tier.free: {
        "ops_per_month": 1_000,
        "max_items": 500,
        "max_namespaces": 1,
        "rate_limit_requests": 20,
    },
    Tier.pro: {
        "ops_per_month": 50_000,
        "max_items": 50_000,
        "max_namespaces": 10,
        "rate_limit_requests": 100,
    },
    Tier.business: {
        "ops_per_month": 200_000,
        "max_items": 200_000,
        "max_namespaces": 50,
        "rate_limit_requests": 1_000,
    },
    Tier.enterprise: {
        "ops_per_month": 1_000_000,
        "max_items": 1_000_000,
        "max_namespaces": 500,
        "rate_limit_requests": 5_000,
    },
}


TIER_PRICES: dict[Tier, float] = {
    Tier.free: 0.0,
    Tier.pro: 9.0,
    Tier.business: 29.0,
    Tier.enterprise: 149.0,
}


def get_tier(settings: Settings) -> Tier:
    """Resolve tier from settings, defaulting to free."""
    try:
        return Tier(settings.tier)
    except ValueError:
        return Tier.free


def get_limits(tier: Tier) -> dict:
    """Return limits dict for a tier."""
    return TIER_LIMITS[tier]


def _coerce_expiry(value: Union[datetime, str, None]) -> Optional[datetime]:
    """Normalise a subscription_expires_at to an aware datetime, or None.

    Accepts a datetime (from asyncpg) or an ISO string (from `_row_to_dict`).
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def effective_monthly_cap(
    tier: Tier,
    settings: Settings,
    subscription_expires_at: Union[datetime, str, None] = None,
) -> int:
    """Return the monthly operation cap that actually applies to a user.

    For paid tiers this is just the static `TIER_LIMITS` value. For the
    free tier it is the *launch* cap while the 30-day trial window is open
    (`subscription_expires_at` in the future) and the small *post-trial*
    cap once it has elapsed. This is how a free user's quota shrinks after
    the launch trial without inventing a new tier or downgrading the row.
    """
    if tier is not Tier.free:
        return TIER_LIMITS[tier]["ops_per_month"]
    expiry = _coerce_expiry(subscription_expires_at)
    if expiry is not None and expiry > datetime.now(timezone.utc):
        return settings.free_launch_monthly_cap
    return settings.free_posttrial_monthly_cap


async def check_quota(
    subject: str,
    tier: Tier,
    settings: Settings,
    subscription_expires_at: Union[datetime, str, None] = None,
) -> tuple[bool, int, int]:
    """Check if a billing subject is within its monthly operation quota.

    `subject` is the per-user billing subject (see `usage.make_subject`),
    so all api keys a user owns share one quota. `subscription_expires_at`
    is the owning user's trial expiry — it selects the launch vs post-trial
    free cap (ignored for paid tiers). Returns (allowed, current_usage, limit).
    """
    cap = effective_monthly_cap(tier, settings, subscription_expires_at)
    if cap == -1:
        return (True, 0, -1)

    from clsplusplus.usage import get_operation_count

    current = await get_operation_count(subject, settings)
    return (current < cap, current, cap)


async def get_quota_status(
    subject: str,
    tier: Tier,
    settings: Settings,
    subscription_expires_at: Union[datetime, str, None] = None,
) -> dict:
    """Full quota status for the /v1/usage response.

    `subject` is the per-user billing subject (see `usage.make_subject`).
    `subscription_expires_at` selects the launch vs post-trial free cap.
    """
    from clsplusplus.usage import get_operation_count, get_usage, _period_key

    limits = TIER_LIMITS[tier]
    cap = effective_monthly_cap(tier, settings, subscription_expires_at)
    ops = await get_operation_count(subject, settings)
    legacy = await get_usage(subject, settings)

    return {
        "tier": tier.value,
        "period": _period_key(),
        "operations": ops,
        "operations_limit": cap,
        "writes": legacy["writes"],
        "reads": legacy["reads"],
        "namespaces_limit": limits["max_namespaces"],
        "storage_limit": limits["max_items"],
        "rate_limit": limits["rate_limit_requests"],
    }
