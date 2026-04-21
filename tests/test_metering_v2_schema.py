"""Tests for the metering v2 schema bootstrapper (ADR 0001 step 1).

The DB-backed tests skip cleanly when Postgres is unreachable, so this
file is safe to run on developer laptops without infra.
"""

from __future__ import annotations

import os

import pytest

from clsplusplus.config import Settings
from clsplusplus.metering_v2.schema import (
    apply_if_enabled,
    apply_schema,
    drop_schema,
    read_ddl,
)


# --------------------------------------------------------------------------- #
# Pure-Python test — no DB required
# --------------------------------------------------------------------------- #


def test_read_ddl_is_non_empty_and_mentions_both_tables():
    ddl = read_ddl()
    assert "CREATE TABLE" in ddl
    assert "usage_events" in ddl
    assert "metering_dead_letter" in ddl
    # Cost column must exist — pay-as-you-go pricing depends on it.
    assert "unit_cost_cents" in ddl
    # Idempotency key must be a UNIQUE constraint, not just indexed.
    assert "idempotency_key" in ddl and "UNIQUE" in ddl


# --------------------------------------------------------------------------- #
# DB-backed tests — skipped cleanly when Postgres is unreachable
# --------------------------------------------------------------------------- #

_DB_URL = os.environ.get("CLS_DATABASE_URL", "")


def _skip_unless_pg_reachable():
    """Skip if we can't open a connection. Cheap check, runs once per test."""
    if not _DB_URL:
        pytest.skip("CLS_DATABASE_URL unset")
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        pytest.skip("asyncpg not installed")


@pytest.fixture
async def pool():
    _skip_unless_pg_reachable()
    import asyncpg

    try:
        p = await asyncpg.create_pool(_DB_URL, min_size=1, max_size=2)
    except Exception as exc:  # pragma: no cover - environmental
        pytest.skip(f"Postgres unreachable: {exc}")
    try:
        # Clean slate in case a prior run left tables behind.
        await drop_schema(p)
        yield p
    finally:
        await drop_schema(p)
        await p.close()


@pytest.mark.asyncio
async def test_apply_and_drop_roundtrip(pool):
    """Schema applies, is re-appliable (idempotent), then drops cleanly."""
    await apply_schema(pool)
    # Re-apply: everything is IF NOT EXISTS / UNIQUE (no-op expected).
    await apply_schema(pool)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name IN ('usage_events', 'metering_dead_letter')
            """,
        )
        assert sorted(r["table_name"] for r in rows) == [
            "metering_dead_letter",
            "usage_events",
        ]

    await drop_schema(pool)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name IN ('usage_events', 'metering_dead_letter')
            """,
        )
        assert rows == []


@pytest.mark.asyncio
async def test_usage_events_rejects_invalid_actor_kind(pool):
    await apply_schema(pool)
    import asyncpg
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO usage_events
                    (idempotency_key, actor_kind, actor_id, event_type, occurred_at)
                VALUES ('t1', 'bogus-kind', 'x', 'write', NOW())
                """,
            )


@pytest.mark.asyncio
async def test_usage_events_idempotency_key_unique(pool):
    await apply_schema(pool)
    import asyncpg
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO usage_events
                (idempotency_key, actor_kind, actor_id, event_type, occurred_at)
            VALUES ('k1', 'user', 'u1', 'write', NOW())
            """,
        )
        with pytest.raises(asyncpg.exceptions.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO usage_events
                    (idempotency_key, actor_kind, actor_id, event_type, occurred_at)
                VALUES ('k1', 'user', 'u1', 'write', NOW())
                """,
            )


@pytest.mark.asyncio
async def test_usage_events_rejects_negative_unit_cost(pool):
    await apply_schema(pool)
    import asyncpg
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO usage_events
                    (idempotency_key, actor_kind, actor_id, event_type, occurred_at,
                     unit_cost_cents)
                VALUES ('neg', 'user', 'u1', 'write', NOW(), -1)
                """,
            )


@pytest.mark.asyncio
async def test_usage_events_rejects_zero_or_negative_quantity(pool):
    await apply_schema(pool)
    import asyncpg
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO usage_events
                    (idempotency_key, actor_kind, actor_id, event_type, occurred_at, quantity)
                VALUES ('q0', 'user', 'u1', 'write', NOW(), 0)
                """,
            )


@pytest.mark.asyncio
async def test_apply_if_enabled_respects_flag(pool):
    """Flag off = no-op. Flag on = DDL runs."""
    # Flag off — no tables created.
    settings_off = Settings(database_url=_DB_URL, metering_v2_write_enabled=False)
    ran = await apply_if_enabled(settings_off, pool)
    assert ran is False
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'usage_events'
              AND table_schema = current_schema()
            """,
        )
        assert rows == []

    # Flag on — tables created.
    settings_on = Settings(database_url=_DB_URL, metering_v2_write_enabled=True)
    ran = await apply_if_enabled(settings_on, pool)
    assert ran is True
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'usage_events'
              AND table_schema = current_schema()
            """,
        )
        assert rows != []
