"""Tests for MeteringPricer and compute_unit_cost_cents (ADR 0001 step 2.5).

Pure-Python tests for the pricing function.
Class-level tests use a FakeTierResolver + ops-counter reader fake.
"""

from __future__ import annotations

from typing import Optional

import pytest

from clsplusplus.config import Settings
from clsplusplus.metering_v2.pricing import MeteringPricer, compute_unit_cost_cents


# --------------------------------------------------------------------------- #
# Pure pricing function
# --------------------------------------------------------------------------- #


def test_zero_when_current_ops_under_cap():
    rates = {"pro": {"_default": 5, "write": 10}}
    assert compute_unit_cost_cents("pro", "write", current_ops=100, cap=1000,
                                   rates=rates) == 0


def test_zero_when_current_ops_equal_to_cap():
    """Cap is inclusive — the last allowed op is still in the flat tier."""
    rates = {"pro": {"_default": 5}}
    assert compute_unit_cost_cents("pro", "write", current_ops=1000, cap=1000,
                                   rates=rates) == 0


def test_over_cap_charges_event_specific_rate_if_present():
    rates = {"pro": {"_default": 5, "write": 10}}
    assert compute_unit_cost_cents("pro", "write", current_ops=1001, cap=1000,
                                   rates=rates) == 10


def test_over_cap_falls_back_to_default_rate():
    rates = {"pro": {"_default": 5, "write": 10}}
    # 'knowledge' not listed → use _default
    assert compute_unit_cost_cents("pro", "knowledge",
                                   current_ops=1001, cap=1000, rates=rates) == 5


def test_over_cap_unknown_tier_is_zero():
    rates = {"pro": {"_default": 5}}
    assert compute_unit_cost_cents("ghost", "write", current_ops=9999, cap=1000,
                                   rates=rates) == 0


def test_unlimited_cap_always_zero():
    """cap <= 0 means unlimited (enterprise-like) — never charge overage."""
    rates = {"enterprise": {"_default": 100}}
    assert compute_unit_cost_cents("enterprise", "write",
                                   current_ops=999_999, cap=-1, rates=rates) == 0
    assert compute_unit_cost_cents("enterprise", "write",
                                   current_ops=999_999, cap=0, rates=rates) == 0


def test_default_rates_are_all_zero_so_flag_on_is_safe():
    """Calling with no rates argument uses the zero default — no surprise bills."""
    assert compute_unit_cost_cents("pro", "write",
                                   current_ops=9999, cap=1000) == 0
    assert compute_unit_cost_cents("business", "read",
                                   current_ops=9999, cap=1000, rates={}) == 0


# --------------------------------------------------------------------------- #
# MeteringPricer — uses the shared TierResolver
# --------------------------------------------------------------------------- #


class FakeTierResolver:
    """Stand-in for `TierResolver` with an explicit mapping."""

    def __init__(self, mapping: dict[str, Optional[str]], raise_for: Optional[str] = None):
        self.mapping = mapping
        self.raise_for = raise_for
        self.calls = 0
        self.invalidated: list[Optional[str]] = []

    async def resolve(self, api_key: str) -> Optional[str]:
        self.calls += 1
        if self.raise_for and api_key == self.raise_for:
            # Real TierResolver swallows exceptions and returns None;
            # mimic that behaviour so pricer sees None, not the raise.
            return None
        return self.mapping.get(api_key)

    def invalidate(self, api_key: Optional[str] = None) -> None:
        self.invalidated.append(api_key)


def _fake_ops_counter(counts: dict[str, int]):
    async def getter(api_key: str, settings: Settings) -> int:
        return counts.get(api_key, 0)
    return getter


@pytest.mark.asyncio
async def test_price_event_no_api_key_returns_zero():
    pricer = MeteringPricer(
        Settings(),
        tier_resolver=FakeTierResolver({}),
        get_ops_counter=_fake_ops_counter({}),
    )
    assert await pricer.price_event(None, "write") == 0
    assert await pricer.price_event("", "write") == 0


@pytest.mark.asyncio
async def test_price_event_under_cap_is_zero():
    settings = Settings(overage_rates_cents={"pro": {"_default": 50}})
    pricer = MeteringPricer(
        settings,
        tier_resolver=FakeTierResolver({"k1": "pro"}),
        get_ops_counter=_fake_ops_counter({"k1": 100}),
    )
    # pro cap is 50_000 — 100 is well under
    assert await pricer.price_event("k1", "write") == 0


@pytest.mark.asyncio
async def test_price_event_over_cap_charges_overage():
    settings = Settings(overage_rates_cents={"pro": {"_default": 50, "write": 75}})
    pricer = MeteringPricer(
        settings,
        tier_resolver=FakeTierResolver({"k1": "pro"}),
        get_ops_counter=_fake_ops_counter({"k1": 60_000}),  # > 50_000 pro cap
    )
    assert await pricer.price_event("k1", "write") == 75    # event-specific
    assert await pricer.price_event("k1", "read") == 50     # _default


@pytest.mark.asyncio
async def test_price_event_unknown_tier_is_zero_and_logs():
    settings = Settings(overage_rates_cents={"pro": {"_default": 999}})
    pricer = MeteringPricer(
        settings,
        tier_resolver=FakeTierResolver({"k1": "platinum-unknown"}),
        get_ops_counter=_fake_ops_counter({"k1": 10_000_000}),
    )
    assert await pricer.price_event("k1", "write") == 0


@pytest.mark.asyncio
async def test_price_event_stale_api_key_is_zero():
    """Key that doesn't resolve to a tier returns 0 — don't bill unknown actors."""
    pricer = MeteringPricer(
        Settings(overage_rates_cents={"pro": {"_default": 999}}),
        tier_resolver=FakeTierResolver({"k1": None}),
        get_ops_counter=_fake_ops_counter({"k1": 10_000_000}),
    )
    assert await pricer.price_event("k1", "write") == 0


@pytest.mark.asyncio
async def test_invalidate_delegates_to_resolver():
    resolver = FakeTierResolver({"k1": "pro"})
    pricer = MeteringPricer(
        Settings(),
        tier_resolver=resolver,
        get_ops_counter=_fake_ops_counter({"k1": 1}),
    )
    pricer.invalidate()
    pricer.invalidate("k1")
    assert resolver.invalidated == [None, "k1"]


@pytest.mark.asyncio
async def test_tier_resolution_error_returns_zero():
    """If the resolver returns None for the key (its error path), price is 0."""
    pricer = MeteringPricer(
        Settings(overage_rates_cents={"pro": {"_default": 999}}),
        tier_resolver=FakeTierResolver({}, raise_for="k1"),
        get_ops_counter=_fake_ops_counter({"k1": 10_000_000}),
    )
    assert await pricer.price_event("k1", "write") == 0
