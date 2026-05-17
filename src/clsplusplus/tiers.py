"""CLS++ tier definitions and quota enforcement."""

from __future__ import annotations

from enum import Enum
from typing import Optional

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


async def check_quota(
    subject: str,
    tier: Tier,
    settings: Settings,
) -> tuple[bool, int, int]:
    """Check if a billing subject is within its monthly operation quota.

    `subject` is the per-user billing subject (see `usage.make_subject`),
    so all api keys a user owns share one quota. Returns
    (allowed, current_usage, limit).
    """
    limits = TIER_LIMITS[tier]
    cap = limits["ops_per_month"]
    if cap == -1:
        return (True, 0, -1)

    from clsplusplus.usage import get_operation_count

    current = await get_operation_count(subject, settings)
    return (current < cap, current, cap)


async def get_quota_status(
    subject: str,
    tier: Tier,
    settings: Settings,
) -> dict:
    """Full quota status for the /v1/usage response.

    `subject` is the per-user billing subject (see `usage.make_subject`).
    """
    from clsplusplus.usage import get_operation_count, get_usage, _period_key

    limits = TIER_LIMITS[tier]
    ops = await get_operation_count(subject, settings)
    legacy = await get_usage(subject, settings)

    return {
        "tier": tier.value,
        "period": _period_key(),
        "operations": ops,
        "operations_limit": limits["ops_per_month"],
        "writes": legacy["writes"],
        "reads": legacy["reads"],
        "namespaces_limit": limits["max_namespaces"],
        "storage_limit": limits["max_items"],
        "rate_limit": limits["rate_limit_requests"],
    }
