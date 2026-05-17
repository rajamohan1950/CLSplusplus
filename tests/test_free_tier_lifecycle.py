"""Tests for the free-tier launch lifecycle.

Every new user gets a 30-day trial at the generous launch quota; after the
trial window elapses they stay free but the effective monthly cap drops to
a small permanent number. This is implemented WITHOUT a new tier — the
quota path picks the cap from the user's `subscription_expires_at`.

These tests cover:
  * `effective_monthly_cap` — the cap-selection helper.
  * `check_quota` / `get_quota_status` — the quota path honours the expiry.
  * `TierResolver.resolve_effective` — threads (tier, expiry) to the caller.
  * `UserStore.create_user` — stamps the 30-day trial on every new user.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.tiers import (
    Tier,
    check_quota,
    effective_monthly_cap,
    get_quota_status,
)


def _future(days: int = 5) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days)


def _past(days: int = 5) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


# ---------------------------------------------------------------------------
# effective_monthly_cap — the cap selector
# ---------------------------------------------------------------------------

class TestEffectiveMonthlyCap:
    def test_free_in_trial_window_gets_launch_cap(self):
        s = Settings(free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        assert effective_monthly_cap(Tier.free, s, _future()) == 8000

    def test_free_after_trial_gets_posttrial_cap(self):
        s = Settings(free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        assert effective_monthly_cap(Tier.free, s, _past()) == 800

    def test_free_with_no_expiry_gets_posttrial_cap(self):
        """A free user with NULL expiry (legacy / pre-trial) gets the small cap."""
        s = Settings(free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        assert effective_monthly_cap(Tier.free, s, None) == 800

    def test_iso_string_expiry_is_accepted(self):
        """_row_to_dict serialises datetimes to ISO strings — must still work."""
        s = Settings(free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        assert effective_monthly_cap(Tier.free, s, _future().isoformat()) == 8000
        assert effective_monthly_cap(Tier.free, s, _past().isoformat()) == 800

    def test_naive_datetime_treated_as_utc(self):
        """A tz-naive expiry must not raise on the aware-vs-naive comparison."""
        s = Settings(free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        naive_future = (datetime.now(timezone.utc) + timedelta(days=3)).replace(tzinfo=None)
        assert effective_monthly_cap(Tier.free, s, naive_future) == 8000

    def test_garbage_expiry_string_falls_back_to_posttrial(self):
        s = Settings(free_posttrial_monthly_cap=800)
        assert effective_monthly_cap(Tier.free, s, "not-a-date") == 800

    def test_paid_tiers_ignore_expiry(self):
        """Pro/business/enterprise use the static TIER_LIMITS cap regardless."""
        s = Settings()
        # Even with a past expiry, a pro user keeps the pro cap.
        assert effective_monthly_cap(Tier.pro, s, _past()) == 50_000
        assert effective_monthly_cap(Tier.business, s, _future()) == 200_000
        assert effective_monthly_cap(Tier.enterprise, s, None) == 1_000_000


# ---------------------------------------------------------------------------
# check_quota — the enforced path
# ---------------------------------------------------------------------------

class TestCheckQuotaTrialWindow:
    @pytest.mark.asyncio
    async def test_in_trial_allowed_above_posttrial_cap(self):
        """A trial user at 5000 ops is allowed (launch cap 8000) even though
        that is well past the 800 post-trial cap."""
        s = Settings(track_usage=True,
                     free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        with patch("clsplusplus.usage.get_operation_count",
                   new_callable=AsyncMock, return_value=5000):
            allowed, usage, limit = await check_quota(
                "subj", Tier.free, s, _future(),
            )
            assert allowed is True
            assert limit == 8000

    @pytest.mark.asyncio
    async def test_posttrial_blocked_at_same_usage(self):
        """The SAME 5000 ops is blocked once the trial window has elapsed —
        this is exactly how the post-trial cap takes effect."""
        s = Settings(track_usage=True,
                     free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        with patch("clsplusplus.usage.get_operation_count",
                   new_callable=AsyncMock, return_value=5000):
            allowed, usage, limit = await check_quota(
                "subj", Tier.free, s, _past(),
            )
            assert allowed is False
            assert limit == 800

    @pytest.mark.asyncio
    async def test_posttrial_user_under_small_cap_allowed(self):
        s = Settings(track_usage=True, free_posttrial_monthly_cap=800)
        with patch("clsplusplus.usage.get_operation_count",
                   new_callable=AsyncMock, return_value=500):
            allowed, usage, limit = await check_quota(
                "subj", Tier.free, s, _past(),
            )
            assert allowed is True
            assert limit == 800

    @pytest.mark.asyncio
    async def test_get_quota_status_reports_effective_cap(self):
        """/v1/usage must report the cap that QuotaMiddleware actually enforces."""
        s = Settings(track_usage=True,
                     free_launch_monthly_cap=8000, free_posttrial_monthly_cap=800)
        with patch("clsplusplus.usage.get_operation_count",
                   new_callable=AsyncMock, return_value=10), \
             patch("clsplusplus.usage.get_usage", new_callable=AsyncMock,
                   return_value={"writes": 5, "reads": 5, "period": "2026-05"}):
            in_trial = await get_quota_status("subj", Tier.free, s, _future())
            post_trial = await get_quota_status("subj", Tier.free, s, _past())
        assert in_trial["operations_limit"] == 8000
        assert post_trial["operations_limit"] == 800


# ---------------------------------------------------------------------------
# TierResolver.resolve_effective — threads expiry to the quota path
# ---------------------------------------------------------------------------

class _FakeStore:
    """Integration-store stand-in exposing the expiry-aware lookup."""

    def __init__(self, mapping):
        # api_key -> (tier, expiry)
        self.mapping = mapping
        self.calls = 0

    async def resolve_tier_and_expiry_from_key(self, raw_key):
        self.calls += 1
        return self.mapping.get(raw_key, (None, None))

    async def resolve_tier_from_key(self, raw_key):
        return self.mapping.get(raw_key, (None, None))[0]


class _LegacyStore:
    """Older store WITHOUT the expiry-aware lookup — resolve_effective must
    still degrade gracefully to (tier, None)."""

    def __init__(self, mapping):
        self.mapping = mapping

    async def resolve_tier_from_key(self, raw_key):
        return self.mapping.get(raw_key)


class TestTierResolverEffective:
    @pytest.mark.asyncio
    async def test_resolve_effective_returns_tier_and_expiry(self):
        from clsplusplus.tier_resolver import TierResolver
        expiry = _future()
        r = TierResolver(_FakeStore({"k1": ("free", expiry)}), cache_ttl_seconds=60)
        tier, got = await r.resolve_effective("k1")
        assert tier == "free"
        assert got == expiry

    @pytest.mark.asyncio
    async def test_resolve_effective_caches(self):
        from clsplusplus.tier_resolver import TierResolver
        store = _FakeStore({"k1": ("free", _future())})
        r = TierResolver(store, cache_ttl_seconds=60)
        await r.resolve_effective("k1")
        await r.resolve_effective("k1")
        assert store.calls == 1

    @pytest.mark.asyncio
    async def test_resolve_effective_unknown_key(self):
        from clsplusplus.tier_resolver import TierResolver
        r = TierResolver(_FakeStore({}), cache_ttl_seconds=60)
        assert await r.resolve_effective("missing") == (None, None)

    @pytest.mark.asyncio
    async def test_resolve_effective_falls_back_for_legacy_store(self):
        from clsplusplus.tier_resolver import TierResolver
        r = TierResolver(_LegacyStore({"k1": "pro"}), cache_ttl_seconds=60)
        assert await r.resolve_effective("k1") == ("pro", None)


# ---------------------------------------------------------------------------
# UserStore.create_user — every new user is stamped with a 30-day trial
# ---------------------------------------------------------------------------

class _FakeConn:
    """Captures the INSERT args and echoes back a user row."""

    def __init__(self):
        self.insert_sql = None
        self.insert_args = None

    async def fetchrow(self, sql, *args):
        self.insert_sql = sql
        self.insert_args = args
        # Mirror the columns create_user inserts so _row_to_dict is happy.
        return {
            "id": "00000000-0000-0000-0000-000000000001",
            "email": args[0],
            "password_hash": args[1],
            "tier": "free",
            "is_admin": False,
            "subscription_expires_at": args[6],
            "subscription_status": "trial",
        }


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


class TestCreateUserStampsTrial:
    @pytest.mark.asyncio
    async def test_create_user_stamps_30_day_trial(self):
        from clsplusplus.stores.user_store import UserStore, TRIAL_DAYS

        assert TRIAL_DAYS == 30

        store = UserStore(Settings())
        conn = _FakeConn()
        with patch.object(UserStore, "get_pool",
                          new_callable=AsyncMock, return_value=_FakePool(conn)):
            user = await store.create_user(email="new@example.com",
                                           password_hash="hashed")

        # subscription_expires_at is the 7th positional arg (index 6).
        stamped_expiry = conn.insert_args[6]
        delta = stamped_expiry - datetime.now(timezone.utc)
        # ~30 days from now (allow generous slack for test execution time).
        assert timedelta(days=29, hours=23) < delta <= timedelta(days=30)
        # Tier stays free; status flags the trial; INSERT writes 'trial'.
        assert user["tier"] == "free"
        assert user["subscription_status"] == "trial"
        assert "'trial'" in conn.insert_sql
        assert "subscription_expires_at" in conn.insert_sql
