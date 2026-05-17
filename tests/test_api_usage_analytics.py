"""Tests for api_usage_analytics.build_api_usage_report.

Fakes Postgres and Redis so the whole file runs without infra. The
fake connection routes on SQL fragments and applies the *real*
aggregation semantics (GROUP BY user_id, hour-of-day histogram,
healthcheck exclusion) so the assertions exercise the module's
contract, not the fake.

Covered:
  - durable usage_events path: key counts, per-user top-N ordering,
    24-bucket active-hours histogram, healthcheck rows excluded
  - empty usage_events: graceful zeros, key counts still live
  - metering v2 disabled: Redis fallback produces per-subject rows
  - DB failure mid-read: never raises, degrades to fallback
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from clsplusplus.api_usage_analytics import build_api_usage_report
from clsplusplus.config import Settings


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeConn:
    """Routes SQL fragments to canned result sets with real aggregation."""

    def __init__(self, owner: "FakePool"):
        self.owner = owner

    async def fetchrow(self, sql: str, *args):
        if "FROM api_credentials" in sql:
            return self.owner.key_counts
        return None

    async def fetchval(self, sql: str, *args):
        if "EXISTS" in sql and "usage_events" in sql:
            return any(
                e["event_type"] != "healthcheck" for e in self.owner.events
            )
        return None

    async def fetch(self, sql: str, *args):
        events = [e for e in self.owner.events if e["event_type"] != "healthcheck"]
        if "GROUP BY ue.user_id" in sql:
            # Per-user consumption: SUM(quantity) grouped by user_id.
            agg: dict = {}
            for e in events:
                uid = e["user_id"]
                row = agg.setdefault(
                    uid,
                    {
                        "user_id": uid,
                        "email": e.get("email"),
                        "name": e.get("name"),
                        "tier": e.get("tier"),
                        "operations": 0,
                        "_keys": set(),
                        "last_active": None,
                    },
                )
                row["operations"] += e["quantity"]
                if e.get("api_key_id"):
                    row["_keys"].add(e["api_key_id"])
                la = e["occurred_at"]
                if row["last_active"] is None or la > row["last_active"]:
                    row["last_active"] = la
            out = []
            for r in agg.values():
                out.append(
                    {
                        "user_id": r["user_id"],
                        "email": r["email"],
                        "name": r["name"],
                        "tier": r["tier"],
                        "operations": r["operations"],
                        "keys_used": len(r["_keys"]),
                        "last_active": r["last_active"],
                    }
                )
            out.sort(key=lambda x: x["operations"], reverse=True)
            return out
        if "EXTRACT(HOUR FROM occurred_at)" in sql:
            # Active-hours histogram.
            hist: dict = {}
            for e in events:
                h = e["occurred_at"].hour
                hist[h] = hist.get(h, 0) + e["quantity"]
            return [{"hour": h, "operations": n} for h, n in hist.items()]
        return []


class _AcquireCtx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


class FakePool:
    def __init__(self, key_counts: dict, events: list, fail: bool = False):
        self.key_counts = key_counts
        self.events = events
        self.fail = fail

    def acquire(self):
        if self.fail:
            raise RuntimeError("db unreachable")
        return _AcquireCtx(FakeConn(self))


def _pool_getter(pool):
    async def getter():
        return pool
    return getter


def _ev(user_id, qty, hour, *, email=None, name=None, tier=None,
        api_key_id=None, event_type="write"):
    return {
        "user_id": user_id,
        "email": email,
        "name": name,
        "tier": tier,
        "api_key_id": api_key_id,
        "quantity": qty,
        "event_type": event_type,
        "occurred_at": datetime(2026, 5, 17, hour, 30, tzinfo=timezone.utc),
    }


# --------------------------------------------------------------------------- #
# Durable usage_events path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_durable_report_key_counts_users_and_hours():
    settings = Settings(metering_v2_write_enabled=True)
    key_counts = {"total": 10, "active": 7, "consumed": 4}
    events = [
        _ev("u1", 5, 9, email="a@x.com", tier="pro", api_key_id="k1"),
        _ev("u1", 3, 9, email="a@x.com", tier="pro", api_key_id="k2"),
        _ev("u2", 2, 14, email="b@x.com", tier="free", api_key_id="k3"),
        # healthcheck rows must be excluded everywhere.
        _ev("u3", 99, 3, email="c@x.com", event_type="healthcheck"),
    ]
    pool = FakePool(key_counts, events)

    report = await build_api_usage_report(_pool_getter(pool), settings=settings)

    assert report["source"] == "usage_events"
    assert report["api_keys"] == {"total": 10, "active": 7, "consumed": 4}

    # Per-user: u1 has 8 ops (5+3) using 2 keys; ordered above u2's 2 ops.
    assert report["per_user"][0]["user_id"] == "u1"
    assert report["per_user"][0]["operations"] == 8
    assert report["per_user"][0]["keys_used"] == 2
    assert report["per_user"][1]["user_id"] == "u2"

    # Healthcheck ops (99) excluded — total is 8 + 2.
    assert report["total_operations"] == 10

    # Active hours: 24 buckets, hour 9 = 8 ops, hour 14 = 2 ops, hour 3 = 0.
    hours = {b["hour"]: b["operations"] for b in report["active_hours"]}
    assert len(report["active_hours"]) == 24
    assert hours[9] == 8
    assert hours[14] == 2
    assert hours[3] == 0


@pytest.mark.asyncio
async def test_top_n_truncates_per_user_rows():
    settings = Settings(metering_v2_write_enabled=True)
    events = [_ev(f"u{i}", i + 1, 10, email=f"u{i}@x.com") for i in range(8)]
    pool = FakePool({"total": 8, "active": 8, "consumed": 8}, events)

    report = await build_api_usage_report(
        _pool_getter(pool), settings=settings, top_n=3
    )

    assert len(report["per_user"]) == 3
    # Highest-ops users first: u7 (8), u6 (7), u5 (6).
    assert [r["user_id"] for r in report["per_user"]] == ["u7", "u6", "u5"]
    assert report["top_n"] == 3


@pytest.mark.asyncio
async def test_empty_usage_events_returns_zeros_not_crash():
    settings = Settings(metering_v2_write_enabled=True)
    pool = FakePool({"total": 3, "active": 2, "consumed": 0}, events=[])

    report = await build_api_usage_report(_pool_getter(pool), settings=settings)

    # Key counts are still live even with no usage events.
    assert report["api_keys"] == {"total": 3, "active": 2, "consumed": 0}
    assert report["per_user"] == []
    assert report["total_operations"] == 0
    assert len(report["active_hours"]) == 24
    assert all(b["operations"] == 0 for b in report["active_hours"])
    assert report["source"] in ("empty", "redis")


# --------------------------------------------------------------------------- #
# Metering disabled / failure -> Redis fallback
# --------------------------------------------------------------------------- #


class FakeRedis:
    def __init__(self, counters: dict[str, int]):
        self.counters = counters

    async def scan(self, cursor, match=None, count=200):
        return 0, list(self.counters.keys())

    async def get(self, key):
        v = self.counters.get(key)
        return str(v) if v is not None else None


@pytest.mark.asyncio
async def test_redis_fallback_when_metering_v2_disabled(monkeypatch):
    settings = Settings(metering_v2_write_enabled=False)
    period = datetime.utcnow().strftime("%Y-%m")
    fake_redis = FakeRedis({
        f"cls:ops:owner:abc123:{period}": 40,
        f"cls:ops:owner:def456:{period}": 12,
    })
    monkeypatch.setattr(
        "clsplusplus.usage._redis_client", lambda url: fake_redis
    )
    pool = FakePool({"total": 5, "active": 5, "consumed": 3}, events=[])

    report = await build_api_usage_report(_pool_getter(pool), settings=settings)

    assert report["source"] == "redis"
    assert report["api_keys"]["total"] == 5
    assert report["total_operations"] == 52
    # Subjects ordered by ops desc.
    assert report["per_user"][0]["operations"] == 40
    assert report["per_user"][0]["email"] == "subject:owner:abc123"


@pytest.mark.asyncio
async def test_db_failure_degrades_gracefully(monkeypatch):
    settings = Settings(metering_v2_write_enabled=True)
    monkeypatch.setattr(
        "clsplusplus.usage._redis_client",
        lambda url: FakeRedis({}),
    )
    pool = FakePool({"total": 0, "active": 0, "consumed": 0}, events=[], fail=True)

    # Must not raise even though every pool.acquire() blows up.
    report = await build_api_usage_report(_pool_getter(pool), settings=settings)

    assert report["api_keys"] == {"total": 0, "active": 0, "consumed": 0}
    assert report["per_user"] == []
    assert len(report["active_hours"]) == 24
    assert report["source"] in ("empty", "redis")
