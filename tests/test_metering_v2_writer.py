"""Tests for MeteringWriter (ADR 0001 step 2).

Unit tests use an in-memory fake pool so they run without Postgres.
One DB-backed test verifies the idempotency ON CONFLICT path against
real Postgres; it skips cleanly when unreachable.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

import pytest

from clsplusplus.config import Settings
from clsplusplus.metering_v2.writer import MeteringWriter, UsageEvent


# --------------------------------------------------------------------------- #
# Fake pool — mimics just enough asyncpg to exercise writer logic
# --------------------------------------------------------------------------- #


class FakeConn:
    def __init__(self, owner: "FakePool"):
        self.owner = owner

    async def execute(self, sql: str, *args):
        if self.owner.fail_on_events and "INTO usage_events" in sql:
            raise RuntimeError(self.owner.fail_on_events)
        if self.owner.fail_on_dead_letter and "INTO metering_dead_letter" in sql:
            raise RuntimeError(self.owner.fail_on_dead_letter)
        if "INTO usage_events" in sql:
            self.owner.events.append(args)
        elif "INTO metering_dead_letter" in sql:
            self.owner.dead_letter.append(args)


class _AcquireCtx:
    def __init__(self, conn: FakeConn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return None


class FakePool:
    def __init__(self) -> None:
        self.events: list = []
        self.dead_letter: list = []
        self.fail_on_events: str | None = None
        self.fail_on_dead_letter: str | None = None

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(FakeConn(self))


@pytest.fixture
def pool() -> FakePool:
    return FakePool()


@pytest.fixture
def settings_on() -> Settings:
    return Settings(metering_v2_write_enabled=True)


@pytest.fixture
def settings_off() -> Settings:
    return Settings(metering_v2_write_enabled=False)


def _event(**overrides: Any) -> UsageEvent:
    defaults = dict(
        idempotency_key="req-1:write",
        actor_kind="user",
        actor_id="u-123",
        event_type="write",
        occurred_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
        user_id="11111111-1111-1111-1111-111111111111",
        raw={"request_id": "req-1"},
    )
    defaults.update(overrides)
    return UsageEvent(**defaults)


# --------------------------------------------------------------------------- #
# UsageEvent validation — pure unit tests
# --------------------------------------------------------------------------- #


def test_usage_event_rejects_bad_actor_kind():
    with pytest.raises(ValueError, match="actor_kind"):
        UsageEvent(
            idempotency_key="k", actor_kind="hacker", actor_id="x", event_type="t",
        )


def test_usage_event_rejects_zero_quantity():
    with pytest.raises(ValueError, match="quantity"):
        UsageEvent(
            idempotency_key="k", actor_kind="user", actor_id="x",
            event_type="t", quantity=0,
        )


def test_usage_event_rejects_negative_cost():
    with pytest.raises(ValueError, match="unit_cost_cents"):
        UsageEvent(
            idempotency_key="k", actor_kind="user", actor_id="x",
            event_type="t", unit_cost_cents=-1,
        )


# --------------------------------------------------------------------------- #
# Writer behaviour — with fake pool
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_writer_flag_off_is_noop(settings_off, pool):
    writer = MeteringWriter(settings_off, _const_getter(pool))
    ok = await writer.record_sync(_event())
    assert ok is False
    assert pool.events == []
    assert pool.dead_letter == []


@pytest.mark.asyncio
async def test_writer_happy_path_inserts_into_usage_events(settings_on, pool):
    writer = MeteringWriter(settings_on, _const_getter(pool))
    ok = await writer.record_sync(_event())
    assert ok is True
    assert len(pool.events) == 1
    assert pool.dead_letter == []
    # Ordered args: (idempotency_key, actor_kind, actor_id, user_id, api_key_id,
    #                namespace, event_type, quantity, unit_cost_cents,
    #                occurred_at, raw_json)
    args = pool.events[0]
    assert args[0] == "req-1:write"
    assert args[1] == "user"
    assert args[6] == "write"
    assert args[7] == 1              # quantity
    assert args[8] == 0              # unit_cost_cents default
    # raw is json-serialised dict
    assert json.loads(args[10])["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_writer_db_error_goes_to_dead_letter(settings_on, pool):
    pool.fail_on_events = "simulated connection refused"
    writer = MeteringWriter(settings_on, _const_getter(pool))
    ok = await writer.record_sync(_event())
    assert ok is False
    assert pool.events == []
    assert len(pool.dead_letter) == 1
    err_class, err_msg, payload_json = pool.dead_letter[0]
    assert err_class == "RuntimeError"
    assert "simulated connection refused" in err_msg
    payload = json.loads(payload_json)
    assert payload["idempotency_key"] == "req-1:write"
    assert payload["event_type"] == "write"


@pytest.mark.asyncio
async def test_writer_dead_letter_failure_does_not_raise(settings_on, pool):
    """If even the dead-letter write fails, we log loudly but don't raise."""
    pool.fail_on_events = "insert-failed"
    pool.fail_on_dead_letter = "dead-letter-also-down"
    writer = MeteringWriter(settings_on, _const_getter(pool))
    ok = await writer.record_sync(_event())
    assert ok is False
    assert pool.events == []
    assert pool.dead_letter == []
    # No raise — test passing is the assertion.


@pytest.mark.asyncio
async def test_writer_record_fire_and_forget_does_not_await(settings_on, pool):
    """record() schedules a task; completion may be asynchronous."""
    writer = MeteringWriter(settings_on, _const_getter(pool))
    await writer.record(_event())
    # Give the scheduled task a tick to run.
    for _ in range(5):
        if pool.events:
            break
        await asyncio.sleep(0.01)
    assert len(pool.events) == 1


def _const_getter(pool):
    async def getter():
        return pool
    return getter


# --------------------------------------------------------------------------- #
# DB-backed idempotency test — ON CONFLICT DO NOTHING
# --------------------------------------------------------------------------- #

_DB_URL = os.environ.get("CLS_DATABASE_URL", "")


@pytest.mark.asyncio
async def test_idempotency_key_deduplicates_against_real_postgres():
    """Same idempotency_key submitted twice results in a single row."""
    if not _DB_URL:
        pytest.skip("CLS_DATABASE_URL unset")
    try:
        import asyncpg
    except ImportError:
        pytest.skip("asyncpg not installed")

    try:
        pool = await asyncpg.create_pool(_DB_URL, min_size=1, max_size=2)
    except Exception as exc:
        pytest.skip(f"Postgres unreachable: {exc}")

    from clsplusplus.metering_v2.schema import apply_schema, drop_schema

    try:
        await drop_schema(pool)
        await apply_schema(pool)

        writer = MeteringWriter(
            Settings(database_url=_DB_URL, metering_v2_write_enabled=True),
            _const_getter(pool),
        )
        ev = _event(idempotency_key="unique-k-42", user_id=None)
        assert await writer.record_sync(ev) is True
        assert await writer.record_sync(ev) is True  # ON CONFLICT DO NOTHING still succeeds

        async with pool.acquire() as conn:
            n = await conn.fetchval(
                "SELECT COUNT(*) FROM usage_events WHERE idempotency_key = $1",
                "unique-k-42",
            )
        assert n == 1
    finally:
        await drop_schema(pool)
        await pool.close()
