"""Tests for SubscriptionWatchdog — the daily expiry sweep."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from clsplusplus.config import Settings
from clsplusplus.subscription_watchdog import SubscriptionWatchdog


class FakeStore:
    """Minimal subset of UserStore used by the watchdog."""

    def __init__(self, users: list[dict]):
        self.users = {u["id"]: dict(u) for u in users}
        self.expired: list[str] = []
        self.list_error: Exception | None = None
        self.expire_error_for: set[str] = set()

    async def get_expired_subscriptions(self, as_of=None, limit=500) -> list[dict]:
        if self.list_error:
            raise self.list_error
        as_of = as_of or datetime.now(timezone.utc)
        return [
            dict(u) for u in self.users.values()
            if u.get("tier") != "free"
            and u.get("subscription_expires_at")
            and u["subscription_expires_at"] < as_of
        ][:limit]

    async def expire_subscription(self, user_id: str) -> bool:
        if user_id in self.expire_error_for:
            raise RuntimeError("db error")
        u = self.users.get(user_id)
        if not u or u["tier"] == "free":
            return False
        u["tier"] = "free"
        u["subscription_status"] = "expired"
        self.expired.append(user_id)
        return True


def _mk_user(user_id: str, tier: str, expires_at) -> dict:
    return {
        "id": user_id,
        "email": f"{user_id}@example.com",
        "tier": tier,
        "subscription_expires_at": expires_at,
        "subscription_status": "active",
    }


@pytest.mark.asyncio
async def test_run_once_downgrades_expired_paid_users():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    future = datetime.now(timezone.utc) + timedelta(days=5)
    store = FakeStore([
        _mk_user("u1", "pro", past),          # expired pro
        _mk_user("u2", "business", past),     # expired business
        _mk_user("u3", "pro", future),        # still valid
        _mk_user("u4", "free", past),         # already free — ignored
    ])
    wd = SubscriptionWatchdog(Settings(), store)
    result = await wd.run_once()
    assert result == {"scanned": 2, "downgraded": 2, "errors": 0}
    assert sorted(store.expired) == ["u1", "u2"]
    assert store.users["u3"]["tier"] == "pro"     # untouched
    assert store.users["u4"]["tier"] == "free"    # untouched


@pytest.mark.asyncio
async def test_run_once_skips_users_with_no_expiry():
    """Lifetime-deal users (`subscription_expires_at IS NULL`) are never touched."""
    store = FakeStore([
        _mk_user("u1", "pro", None),          # lifetime pro, no expiry
        _mk_user("u2", "business", None),     # lifetime business
    ])
    wd = SubscriptionWatchdog(Settings(), store)
    result = await wd.run_once()
    assert result == {"scanned": 0, "downgraded": 0, "errors": 0}
    assert store.expired == []


@pytest.mark.asyncio
async def test_run_once_counts_per_user_errors_without_aborting_batch():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    store = FakeStore([
        _mk_user("u1", "pro", past),
        _mk_user("u2", "business", past),
        _mk_user("u3", "enterprise", past),
    ])
    store.expire_error_for = {"u2"}  # this one throws
    wd = SubscriptionWatchdog(Settings(), store)
    result = await wd.run_once()
    # All 3 scanned; u1 + u3 downgraded; u2 errored.
    assert result == {"scanned": 3, "downgraded": 2, "errors": 1}
    assert sorted(store.expired) == ["u1", "u3"]


@pytest.mark.asyncio
async def test_run_once_handles_list_failure():
    store = FakeStore([])
    store.list_error = RuntimeError("db down")
    wd = SubscriptionWatchdog(Settings(), store)
    result = await wd.run_once()
    assert result["scanned"] == 0
    assert result["downgraded"] == 0
    assert result["errors"] == 1
    assert "RuntimeError" in result["error"]


@pytest.mark.asyncio
async def test_tier_resolver_cache_flushed_on_downgrade():
    class FakeResolver:
        def __init__(self):
            self.calls = 0

        def invalidate(self, api_key=None):
            self.calls += 1

    past = datetime.now(timezone.utc) - timedelta(hours=1)
    store = FakeStore([
        _mk_user("u1", "pro", past),
        _mk_user("u2", "business", past),
    ])
    resolver = FakeResolver()
    wd = SubscriptionWatchdog(Settings(), store, tier_resolver=resolver)
    await wd.run_once()
    # Cache invalidated once per downgrade so quota + pricer re-resolve.
    assert resolver.calls == 2


@pytest.mark.asyncio
async def test_notify_called_for_each_downgrade():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    store = FakeStore([
        _mk_user("u1", "pro", past),
        _mk_user("u2", "business", past),
    ])
    notified: list[dict] = []

    async def notify(row: dict) -> None:
        notified.append(row)

    wd = SubscriptionWatchdog(Settings(), store, notify=notify)
    await wd.run_once()
    assert sorted(n["id"] for n in notified) == ["u1", "u2"]


@pytest.mark.asyncio
async def test_start_stop_is_idempotent():
    wd = SubscriptionWatchdog(Settings(), FakeStore([]))
    wd.start()
    wd.start()  # no-op
    assert wd._task is not None
    await wd.stop()
    await wd.stop()  # no-op
    assert wd._task is None
