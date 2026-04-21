"""Pytest wrapper around MeteringHealthCheck.

Two layers:

1. Unit tests — each sub-check is exercised against fake DB / Redis.
   Runs everywhere, no infra needed.

2. Live suite — one test per sub-check, flipped on by setting
   CLS_METERING_V2_LIVE_TEST=true plus real CLS_DATABASE_URL and
   CLS_REDIS_URL. Each live test fails loudly with the sub-check's
   own remediation string so the output is actionable.

Usage against production:

    CLS_METERING_V2_WRITE_ENABLED=true \\
    CLS_METERING_V2_LIVE_TEST=true    \\
    CLS_DATABASE_URL=<render-pg>      \\
    CLS_REDIS_URL=<render-redis>      \\
    python -m pytest tests/test_metering_v2_healthcheck.py -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from clsplusplus.config import Settings
from clsplusplus.metering_v2.healthcheck import (
    CheckResult,
    HealthReport,
    MeteringHealthCheck,
)


# --------------------------------------------------------------------------- #
# Unit tests — fake infra
# --------------------------------------------------------------------------- #


class FakeConn:
    def __init__(self, owner: "FakePool"):
        self.owner = owner

    async def fetchval(self, sql: str, *args):
        if sql.strip() == "SELECT 1":
            if self.owner.db_broken:
                raise RuntimeError(self.owner.db_broken)
            return 1
        if "COUNT(*)" in sql and "notified_at IS NULL" in sql:
            return self.owner.stuck_dead_letter
        if "COUNT(*)" in sql and "failed_at >= $1" in sql:
            return self.owner.recent_dead_letter
        return 0

    async def fetch(self, sql: str, *args):
        if "information_schema.tables" in sql:
            return [
                {"table_name": t}
                for t in self.owner.tables_present
            ]
        if "FROM usage_events" in sql:
            return self.owner.pg_events
        return []

    async def fetchrow(self, sql: str, *args):
        if "WHERE idempotency_key" in sql:
            return self.owner.roundtrip_row
        return None

    async def execute(self, sql: str, *args):
        if "INTO usage_events" in sql and self.owner.writer_fails_with:
            raise RuntimeError(self.owner.writer_fails_with)
        if "INTO usage_events" in sql:
            self.owner.roundtrip_row = {
                "event_type": args[6],
                "quantity": args[7],
                "unit_cost_cents": args[8],
                "recorded_at": args[9],
            }


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return None


class FakePool:
    def __init__(
        self,
        *,
        db_broken: str | None = None,
        tables_present: list[str] | None = None,
        stuck_dead_letter: int = 0,
        recent_dead_letter: int = 0,
        writer_fails_with: str | None = None,
        pg_events: list | None = None,
    ):
        self.db_broken = db_broken
        self.tables_present = tables_present if tables_present is not None else [
            "usage_events", "metering_dead_letter",
        ]
        self.stuck_dead_letter = stuck_dead_letter
        self.recent_dead_letter = recent_dead_letter
        self.writer_fails_with = writer_fails_with
        self.roundtrip_row: dict | None = None
        self.pg_events = pg_events or []

    def acquire(self):
        return _Acquire(FakeConn(self))


class FakeRedis:
    def __init__(self):
        self.keys: dict[str, str] = {}

    async def scan_iter(self, match: str, count: int = 500):
        for _ in []:
            yield _

    async def get(self, key: str):
        return None


def _make_check(settings: Settings, pool: FakePool) -> MeteringHealthCheck:
    async def pool_getter() -> Any:
        return pool

    async def redis_getter() -> Any:
        return FakeRedis()

    return MeteringHealthCheck(settings, pool_getter, redis_getter)


@pytest.mark.asyncio
async def test_flag_off_short_circuits():
    report = await _make_check(
        Settings(metering_v2_write_enabled=False), FakePool(),
    ).run_all()
    assert report.passed is False
    assert report.checks[0].name == "config.flag_on"
    assert report.checks[0].ok is False
    # Short-circuits — no other checks attempted.
    assert len(report.checks) == 1


@pytest.mark.asyncio
async def test_flag_on_oncall_set_progresses():
    pool = FakePool()
    report = await _make_check(
        Settings(metering_v2_write_enabled=True,
                 oncall_email="ops@example.com"),
        pool,
    ).run_all()
    # First two pass; run_all continues past them.
    assert report.checks[0].ok is True
    assert report.checks[1].ok is True
    assert report.checks[1].detail.startswith("set to")


@pytest.mark.asyncio
async def test_oncall_unset_flags_but_does_not_short_circuit():
    pool = FakePool()
    report = await _make_check(
        Settings(metering_v2_write_enabled=True, oncall_email=""),
        pool,
    ).run_all()
    names = [c.name for c in report.checks]
    assert "config.oncall_email" in names
    oncall = next(c for c in report.checks if c.name == "config.oncall_email")
    assert oncall.ok is False
    # Later checks still ran (oncall being empty is not a short-circuit).
    assert "db.reachable" in names


@pytest.mark.asyncio
async def test_db_unreachable_short_circuits():
    pool = FakePool(db_broken="ECONNREFUSED")
    report = await _make_check(
        Settings(metering_v2_write_enabled=True), pool,
    ).run_all()
    names = [c.name for c in report.checks]
    assert "db.reachable" in names
    db = next(c for c in report.checks if c.name == "db.reachable")
    assert db.ok is False
    # Schema/writer/DL checks NOT attempted.
    assert "db.schema_present" not in names
    assert "writer.roundtrip" not in names


@pytest.mark.asyncio
async def test_schema_missing_short_circuits():
    pool = FakePool(tables_present=["usage_events"])  # missing dead_letter
    report = await _make_check(
        Settings(metering_v2_write_enabled=True), pool,
    ).run_all()
    schema = next(c for c in report.checks if c.name == "db.schema_present")
    assert schema.ok is False
    assert "found only" in schema.detail
    # Writer/dead-letter checks skipped when schema missing.
    names = [c.name for c in report.checks]
    assert "writer.roundtrip" not in names


@pytest.mark.asyncio
async def test_writer_roundtrip_happy_path():
    pool = FakePool()
    report = await _make_check(
        Settings(metering_v2_write_enabled=True), pool,
    ).run_all()
    rt = next(c for c in report.checks if c.name == "writer.roundtrip")
    assert rt.ok is True


@pytest.mark.asyncio
async def test_writer_insert_fails_goes_to_dead_letter_and_reports_fail():
    pool = FakePool(writer_fails_with="simulated connection refused")
    # The writer's own error handling routes to dead-letter, so record_sync
    # returns False and the round-trip reports failure with a clear message.
    report = await _make_check(
        Settings(metering_v2_write_enabled=True), pool,
    ).run_all()
    rt = next(c for c in report.checks if c.name == "writer.roundtrip")
    assert rt.ok is False
    assert "dead_letter" in rt.detail.lower() or "dead-letter" in rt.detail.lower()


@pytest.mark.asyncio
async def test_dead_letter_stuck_rows_flag():
    pool = FakePool(stuck_dead_letter=3, recent_dead_letter=0)
    report = await _make_check(
        Settings(metering_v2_write_enabled=True), pool,
    ).run_all()
    dl = next(c for c in report.checks if c.name == "dead_letter.clean")
    assert dl.ok is False
    assert "3" in dl.detail


@pytest.mark.asyncio
async def test_overall_passes_on_happy_path():
    pool = FakePool()
    report = await _make_check(
        Settings(metering_v2_write_enabled=True,
                 oncall_email="ops@example.com"),
        pool,
    ).run_all()
    assert report.passed is True
    assert all(c.ok for c in report.checks)


# --------------------------------------------------------------------------- #
# Pretty printer
# --------------------------------------------------------------------------- #


def test_report_pretty_marks_passes_and_failures():
    report = HealthReport(
        passed=False,
        ran_at="2026-04-21T07:30:00+00:00",
        checks=[
            CheckResult("config.flag_on", True, "write path enabled"),
            CheckResult("db.reachable", False, "ECONNREFUSED",
                        "Check CLS_DATABASE_URL."),
        ],
    )
    out = report.pretty()
    assert "PASS" not in out  # overall is FAIL
    assert "FAIL" in out
    assert "✓" in out
    assert "✗" in out
    assert "Check CLS_DATABASE_URL." in out


# --------------------------------------------------------------------------- #
# Live suite — opt-in via CLS_METERING_V2_LIVE_TEST=true
# --------------------------------------------------------------------------- #


_LIVE = os.environ.get("CLS_METERING_V2_LIVE_TEST", "").lower() in {"1", "true", "yes"}
pytestmark_live = pytest.mark.skipif(
    not _LIVE,
    reason="Set CLS_METERING_V2_LIVE_TEST=true + CLS_DATABASE_URL to run the "
           "production health check as real pytest cases.",
)


async def _live_report() -> HealthReport:
    """Build a healthcheck against the real infra described by env vars."""
    settings = Settings()

    import asyncpg
    url = settings.database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgres://", 1)
    pool = await asyncpg.create_pool(url, min_size=1, max_size=2)

    async def pool_getter() -> Any:
        return pool

    async def redis_getter() -> Any:
        try:
            import redis.asyncio as _redis
            return _redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            return None

    check = MeteringHealthCheck(settings, pool_getter, redis_getter)
    try:
        return await check.run_all()
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytestmark_live
async def test_live_overall_passes():
    """Headline: the whole thing is healthy."""
    report = await _live_report()
    assert report.passed, "\n" + report.pretty()


@pytest.mark.asyncio
@pytestmark_live
async def test_live_flag_on():
    report = await _live_report()
    c = next(c for c in report.checks if c.name == "config.flag_on")
    assert c.ok, c.remediation or c.detail


@pytest.mark.asyncio
@pytestmark_live
async def test_live_db_reachable():
    report = await _live_report()
    c = next(c for c in report.checks if c.name == "db.reachable")
    assert c.ok, c.remediation or c.detail


@pytest.mark.asyncio
@pytestmark_live
async def test_live_schema_present():
    report = await _live_report()
    c = next(c for c in report.checks if c.name == "db.schema_present")
    assert c.ok, c.remediation or c.detail


@pytest.mark.asyncio
@pytestmark_live
async def test_live_writer_roundtrip():
    """Writes a canary usage_events row and reads it back."""
    report = await _live_report()
    c = next(c for c in report.checks if c.name == "writer.roundtrip")
    assert c.ok, c.remediation or c.detail


@pytest.mark.asyncio
@pytestmark_live
async def test_live_dead_letter_clean():
    report = await _live_report()
    c = next(c for c in report.checks if c.name == "dead_letter.clean")
    assert c.ok, c.remediation or c.detail


@pytest.mark.asyncio
@pytestmark_live
async def test_live_reconciler_no_drift():
    report = await _live_report()
    c = next(c for c in report.checks if c.name == "reconciler.drift")
    assert c.ok, c.remediation or c.detail
