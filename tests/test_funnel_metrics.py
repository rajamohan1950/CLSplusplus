"""Tests for WebEventsStore.funnel_summary — the conversion-funnel aggregation.

Fakes Postgres so the file runs without infra. The fake connection routes on
SQL fragments and applies the *real* aggregation semantics (trailing-window
filter, distinct-session visitor count, click-ranked top pages, the
visitor->signup->active funnel join) so the assertions exercise the module's
contract — conversion-rate math, ranking order, the credential join — not the
fake.

Run ONLY this file:
    python -m pytest tests/test_funnel_metrics.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clsplusplus.config import Settings
from clsplusplus.stores.web_events_store import WebEventsStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Fake Postgres — holds rows, applies the real aggregation in Python
# --------------------------------------------------------------------------- #


class FakeConn:
    def __init__(self, owner: "FakePool"):
        self.owner = owner

    def _window_start(self, days: int) -> datetime:
        return _now() - timedelta(days=days)

    def _events_in_window(self, days: int) -> list[dict]:
        start = self._window_start(days)
        return [e for e in self.owner.events if e["created_at"] >= start]

    async def execute(self, sql: str, *args):
        # DDL on first connect + INSERT from record_event.
        if sql.strip().upper().startswith("INSERT INTO WEB_EVENTS"):
            event, page, ref, session_id = args
            self.owner.events.append({
                "event": event, "page": page, "ref": ref,
                "session_id": session_id, "created_at": _now(),
            })
        return "OK"

    async def fetchrow(self, sql: str, *args):
        if "COUNT(DISTINCT session_id)" in sql:
            days = int(args[0])
            evs = self._events_in_window(days)
            return {
                "unique_visitors": len(
                    {e["session_id"] for e in evs if e["session_id"]}
                ),
                "total_pageviews": sum(1 for e in evs if e["event"] == "pageview"),
                "total_clicks": sum(1 for e in evs if e["event"] == "click"),
            }
        return None

    async def fetchval(self, sql: str, *args):
        days = int(args[0])
        start = self._window_start(days)
        if "FROM users u" in sql and "api_credentials" in sql:
            # Active users: signed up in window AND own a live credential.
            active_emails = {
                i["owner_email"]
                for i in self.owner.integrations
                if any(c["integration_id"] == i["id"] and c["status"] == "active"
                       for c in self.owner.credentials)
            }
            return len({
                u["id"] for u in self.owner.users
                if u["created_at"] >= start and u["email"] in active_emails
            })
        if "FROM users" in sql:
            # Signups in window.
            return sum(1 for u in self.owner.users if u["created_at"] >= start)
        return 0

    async def fetch(self, sql: str, *args):
        days = int(args[0])
        evs = self._events_in_window(days)
        if "FILTER (WHERE event = 'click')    AS clicks" in sql:
            # Top pages ranked by clicks desc, then pageviews desc.
            by_page: dict = {}
            for e in evs:
                if not e["page"]:
                    continue
                row = by_page.setdefault(e["page"], {"clicks": 0, "pageviews": 0})
                if e["event"] == "click":
                    row["clicks"] += 1
                elif e["event"] == "pageview":
                    row["pageviews"] += 1
            rows = [
                {"page": p, "clicks": v["clicks"], "pageviews": v["pageviews"]}
                for p, v in by_page.items()
            ]
            rows.sort(key=lambda r: (-r["clicks"], -r["pageviews"]))
            return rows[:25]
        if "ref AS target" in sql:
            # Top engagement: click targets ranked desc.
            by_ref: dict = {}
            for e in evs:
                if e["event"] == "click" and e["ref"]:
                    by_ref[e["ref"]] = by_ref.get(e["ref"], 0) + 1
            rows = [{"target": r, "clicks": c} for r, c in by_ref.items()]
            rows.sort(key=lambda r: -r["clicks"])
            return rows[:15]
        return []


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self):
        self.events: list[dict] = []
        self.users: list[dict] = []
        self.integrations: list[dict] = []
        self.credentials: list[dict] = []
        self._conn = FakeConn(self)

    def acquire(self):
        return _Acquire(self._conn)


@pytest.fixture
def store(monkeypatch) -> WebEventsStore:
    s = WebEventsStore(Settings(database_url="postgresql://x/y"))
    pool = FakePool()

    async def _get_pool():
        return pool

    monkeypatch.setattr(s, "get_pool", _get_pool)
    s._fake = pool  # test handle
    return s


def _seed_event(store, event, page="", ref="", session_id="", age_days=0):
    store._fake.events.append({
        "event": event, "page": page, "ref": ref, "session_id": session_id,
        "created_at": _now() - timedelta(days=age_days),
    })


def _seed_user(store, uid, email, age_days=0):
    store._fake.users.append({
        "id": uid, "email": email,
        "created_at": _now() - timedelta(days=age_days),
    })


def _seed_active_credential(store, email, status="active"):
    iid = f"int-{email}"
    store._fake.integrations.append({"id": iid, "owner_email": email})
    store._fake.credentials.append({"integration_id": iid, "status": status})


# --------------------------------------------------------------------------- #
# Happy path — the funnel a real owner reads
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_full_funnel_with_conversion_rates(store):
    # 4 unique visitors across 6 pageview events.
    for sid in ("s1", "s2", "s3", "s4"):
        _seed_event(store, "pageview", page="/", session_id=sid)
    _seed_event(store, "pageview", page="/integrate.html", session_id="s1")
    _seed_event(store, "pageview", page="/integrate.html", session_id="s2")
    # 2 of those visitors signed up.
    _seed_user(store, "u1", "a@x.com")
    _seed_user(store, "u2", "b@x.com")
    # 1 signup became active (owns a live credential).
    _seed_active_credential(store, "a@x.com")

    out = await store.funnel_summary(days=30)

    assert out["traffic"]["unique_visitors"] == 4
    assert out["traffic"]["total_pageviews"] == 6
    f = out["funnel"]
    assert f["visitors"] == 4
    assert f["signups"] == 2
    assert f["active_users"] == 1
    # 2/4, 1/2, 1/4 as percentages.
    assert f["visitor_to_signup_pct"] == 50.0
    assert f["signup_to_active_pct"] == 50.0
    assert f["visitor_to_active_pct"] == 25.0


@pytest.mark.asyncio
async def test_top_pages_ranked_by_clicks(store):
    # /integrate gets 3 clicks, / gets 1 click — ranking must order them.
    for _ in range(3):
        _seed_event(store, "click", page="/integrate.html", ref="Get Started")
    _seed_event(store, "click", page="/", ref="Docs")
    _seed_event(store, "pageview", page="/")

    out = await store.funnel_summary(days=30)
    pages = out["top_pages"]
    assert pages[0]["page"] == "/integrate.html"
    assert pages[0]["clicks"] == 3
    assert pages[1]["page"] == "/"
    assert pages[1]["clicks"] == 1

    eng = out["top_engagement"]
    assert eng[0]["target"] == "Get Started"
    assert eng[0]["clicks"] == 3


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_window_excludes_old_events_and_users(store):
    # Inside the 30-day window.
    _seed_event(store, "pageview", session_id="recent", age_days=5)
    _seed_user(store, "u-recent", "recent@x.com", age_days=5)
    # Outside the window — must be ignored.
    _seed_event(store, "pageview", session_id="stale", age_days=90)
    _seed_user(store, "u-stale", "stale@x.com", age_days=90)

    out = await store.funnel_summary(days=30)
    assert out["funnel"]["visitors"] == 1
    assert out["funnel"]["signups"] == 1


@pytest.mark.asyncio
async def test_revoked_credential_does_not_count_as_active(store):
    _seed_user(store, "u1", "a@x.com")
    _seed_active_credential(store, "a@x.com", status="revoked")

    out = await store.funnel_summary(days=30)
    assert out["funnel"]["signups"] == 1
    assert out["funnel"]["active_users"] == 0
    assert out["funnel"]["signup_to_active_pct"] == 0.0


@pytest.mark.asyncio
async def test_empty_dataset_yields_zeroed_rates_not_divide_by_zero(store):
    out = await store.funnel_summary(days=30)
    f = out["funnel"]
    assert f == {
        "visitors": 0, "signups": 0, "active_users": 0,
        "visitor_to_signup_pct": 0.0,
        "signup_to_active_pct": 0.0,
        "visitor_to_active_pct": 0.0,
    }
    assert out["top_pages"] == []
    assert out["top_engagement"] == []


@pytest.mark.asyncio
async def test_blank_session_ids_excluded_from_visitor_count(store):
    _seed_event(store, "pageview", session_id="real")
    _seed_event(store, "pageview", session_id="")  # storage-disabled client
    _seed_event(store, "pageview", session_id="")

    out = await store.funnel_summary(days=30)
    assert out["traffic"]["unique_visitors"] == 1


@pytest.mark.asyncio
async def test_days_param_is_clamped(store):
    # 0 -> clamped to 1; 9999 -> clamped to 365.
    assert (await store.funnel_summary(days=0))["window_days"] == 1
    assert (await store.funnel_summary(days=9999))["window_days"] == 365


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_record_event_appends_and_is_aggregated(store):
    await store.record_event("pageview", page="/", session_id="w1")
    await store.record_event("click", page="/", ref="CTA", session_id="w1")

    out = await store.funnel_summary(days=30)
    assert out["traffic"]["unique_visitors"] == 1
    assert out["traffic"]["total_pageviews"] == 1
    assert out["traffic"]["total_clicks"] == 1


@pytest.mark.asyncio
async def test_record_event_rejects_empty_event(store):
    with pytest.raises(ValueError):
        await store.record_event("   ", page="/")


@pytest.mark.asyncio
async def test_record_event_clips_oversized_fields(store):
    long_page = "/" + ("p" * 1000)
    await store.record_event("pageview", page=long_page, session_id="x")
    stored = store._fake.events[-1]
    assert len(stored["page"]) <= 512
