"""End-to-end health check for the metering v2 pipeline.

Turns the "is it working?" question into a single structured report.
Can be run three ways:

1. Python API — `MeteringHealthCheck(settings, pool_getter, redis_getter).run_all()`
2. HTTP       — `GET /admin/metering/health` (wired in api.py)
3. CLI        — `python -m clsplusplus.metering_v2 healthcheck`

Each sub-check returns a structured result: name, ok, detail, remediation.
An overall `passed` is computed from the logical-AND of all sub-checks.

The roundtrip check writes a canary row into `usage_events` with
`event_type="healthcheck"` and `actor_kind="system"`. Those rows are
safe to ignore when aggregating for billing (actor_kind 'system' is
not a billable actor).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from clsplusplus.config import Settings
from clsplusplus.metering_v2.reconciler import MeteringReconciler
from clsplusplus.metering_v2.writer import MeteringWriter, UsageEvent

logger = logging.getLogger(__name__)


PoolGetter = Callable[[], Awaitable[Any]]
RedisGetter = Callable[[], Awaitable[Any]]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    remediation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthReport:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    ran_at: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "ran_at": self.ran_at,
            "checks": [c.to_dict() for c in self.checks],
        }

    def pretty(self) -> str:
        """Fixed-width textual report."""
        lines = [f"METERING V2 HEALTH — {'PASS' if self.passed else 'FAIL'} "
                 f"(ran_at={self.ran_at})"]
        for c in self.checks:
            mark = "✓" if c.ok else "✗"
            lines.append(f"  [{mark}] {c.name:35s} {c.detail}")
            if not c.ok and c.remediation:
                lines.append(f"        → {c.remediation}")
        return "\n".join(lines)


class MeteringHealthCheck:
    """Runs every observable signal and returns a single report."""

    def __init__(
        self,
        settings: Settings,
        pool_getter: PoolGetter,
        redis_getter: RedisGetter,
        *,
        drift_percent: float = 0.001,
        min_abs_drift: int = 5,
        canary_timeout_seconds: float = 5.0,
    ):
        self.settings = settings
        self._pool_getter = pool_getter
        self._redis_getter = redis_getter
        self._drift_pct = drift_percent
        self._min_abs = min_abs_drift
        self._canary_timeout = canary_timeout_seconds

    async def run_all(self) -> HealthReport:
        checks: list[CheckResult] = []

        # Order matters — later checks depend on earlier ones passing.
        flag_check = self._check_flag_on()
        checks.append(flag_check)
        if not flag_check.ok:
            return self._build_report(checks)

        checks.append(self._check_oncall_set())

        db_check = await self._check_database_reachable()
        checks.append(db_check)
        if not db_check.ok:
            return self._build_report(checks)

        schema_check = await self._check_schema_present()
        checks.append(schema_check)
        if not schema_check.ok:
            return self._build_report(checks)

        checks.append(await self._check_writer_roundtrip())
        checks.append(await self._check_dead_letter_clean())
        reconcile = await self._check_reconciler_runs()
        checks.append(reconcile)

        return self._build_report(checks)

    def _build_report(self, checks: list[CheckResult]) -> HealthReport:
        return HealthReport(
            passed=all(c.ok for c in checks),
            checks=checks,
            ran_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # --- individual checks -----------------------------------------

    def _check_flag_on(self) -> CheckResult:
        if self.settings.metering_v2_write_enabled:
            return CheckResult("config.flag_on", True, "write path enabled")
        return CheckResult(
            "config.flag_on", False,
            "CLS_METERING_V2_WRITE_ENABLED is false",
            "Set CLS_METERING_V2_WRITE_ENABLED=true on Render and redeploy.",
        )

    def _check_oncall_set(self) -> CheckResult:
        email = self.settings.oncall_email
        if email:
            return CheckResult("config.oncall_email", True, f"set to {email}")
        return CheckResult(
            "config.oncall_email", False,
            "empty",
            "Set CLS_ONCALL_EMAIL on Render; "
            "dead-letter findings will not page anyone otherwise.",
        )

    async def _check_database_reachable(self) -> CheckResult:
        try:
            pool = await self._pool_getter()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return CheckResult("db.reachable", True, "ok")
        except Exception as exc:
            return CheckResult(
                "db.reachable", False,
                f"{type(exc).__name__}: {exc}",
                "Check CLS_DATABASE_URL and that Postgres is up.",
            )

    async def _check_schema_present(self) -> CheckResult:
        try:
            pool = await self._pool_getter()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT table_name FROM information_schema.tables
                       WHERE table_schema = current_schema()
                         AND table_name IN ('usage_events', 'metering_dead_letter')"""
                )
            found = sorted(r["table_name"] for r in rows)
            if found == ["metering_dead_letter", "usage_events"]:
                return CheckResult("db.schema_present", True,
                                   "usage_events + metering_dead_letter exist")
            return CheckResult(
                "db.schema_present", False,
                f"found only {found}",
                "Redeploy with CLS_METERING_V2_WRITE_ENABLED=true so the "
                "startup hook applies the DDL, OR call "
                "metering_v2.apply_schema() manually.",
            )
        except Exception as exc:
            return CheckResult(
                "db.schema_present", False,
                f"{type(exc).__name__}: {exc}", "",
            )

    async def _check_writer_roundtrip(self) -> CheckResult:
        """Write a canary event synchronously and read it back."""
        canary_key = f"healthcheck-{uuid4()}"
        event = UsageEvent(
            idempotency_key=canary_key,
            actor_kind="system",
            actor_id="healthcheck",
            event_type="healthcheck",
            occurred_at=datetime.now(timezone.utc),
            raw={"source": "MeteringHealthCheck"},
        )
        writer = MeteringWriter(self.settings, self._pool_getter)
        try:
            ok = await asyncio.wait_for(
                writer.record_sync(event), timeout=self._canary_timeout,
            )
        except asyncio.TimeoutError:
            return CheckResult(
                "writer.roundtrip", False,
                "writer timed out after %ss" % self._canary_timeout,
                "Pool is blocking. Check pool size / idle connections / DB latency.",
            )
        except Exception as exc:
            return CheckResult(
                "writer.roundtrip", False,
                f"{type(exc).__name__}: {exc}",
                "Writer raised. Check the dead-letter table — the event may have "
                "been enqueued there.",
            )

        if not ok:
            return CheckResult(
                "writer.roundtrip", False,
                "record_sync returned False (enqueued to dead_letter instead)",
                "Check metering_dead_letter for the canary key.",
            )

        # Verify the row shows up with the expected fields.
        try:
            pool = await self._pool_getter()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT event_type, quantity, unit_cost_cents,
                              recorded_at
                       FROM usage_events
                       WHERE idempotency_key = $1""",
                    canary_key,
                )
        except Exception as exc:
            return CheckResult(
                "writer.roundtrip", False,
                f"read-back failed: {type(exc).__name__}: {exc}", "",
            )

        if row is None:
            return CheckResult(
                "writer.roundtrip", False,
                "canary not found after write",
                "Write silently dropped. This is the money-losing bug — "
                "check logs for a suppressed exception in record_safely.",
            )
        if row["event_type"] != "healthcheck" or row["quantity"] != 1:
            return CheckResult(
                "writer.roundtrip", False,
                f"row shape wrong: {dict(row)}", "",
            )
        return CheckResult(
            "writer.roundtrip", True,
            f"canary written + read back in <{self._canary_timeout}s",
        )

    async def _check_dead_letter_clean(self) -> CheckResult:
        """No unpaged failures older than the notifier's poll cycle + grace."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=3)
        try:
            pool = await self._pool_getter()
            async with pool.acquire() as conn:
                stuck = await conn.fetchval(
                    """SELECT COUNT(*)::INT FROM metering_dead_letter
                       WHERE notified_at IS NULL AND failed_at < $1""",
                    cutoff,
                )
                recent = await conn.fetchval(
                    """SELECT COUNT(*)::INT FROM metering_dead_letter
                       WHERE failed_at >= $1""",
                    cutoff,
                )
        except Exception as exc:
            return CheckResult(
                "dead_letter.clean", False,
                f"{type(exc).__name__}: {exc}", "",
            )

        if stuck == 0:
            return CheckResult(
                "dead_letter.clean", True,
                f"0 stuck rows (recent <3m: {recent})",
            )
        return CheckResult(
            "dead_letter.clean", False,
            f"{stuck} row(s) unpaged > 3 minutes old",
            "Notifier is stuck. Check logs for 'metering notifier:' errors; "
            "likely Resend API down or CLS_ONCALL_EMAIL misconfigured.",
        )

    async def _check_reconciler_runs(self) -> CheckResult:
        """Invoke reconcile_once and check drift."""
        reconciler = MeteringReconciler(
            self.settings, self._pool_getter, self._redis_getter,
            drift_percent=self._drift_pct, min_abs_drift=self._min_abs,
        )
        try:
            result = await reconciler.reconcile_once()
        except Exception as exc:
            return CheckResult(
                "reconciler.runs", False,
                f"{type(exc).__name__}: {exc}",
                "Reconciler raised. Check Redis reachability and "
                "metering_v2.reconciler logs.",
            )
        if result.findings:
            return CheckResult(
                "reconciler.drift", False,
                f"{len(result.findings)} drift finding(s) in {result.period}",
                "Investigate the enqueued ReconciliationDrift rows in "
                "metering_dead_letter. Do NOT auto-fix — pick the correct "
                "source of truth.",
            )
        return CheckResult(
            "reconciler.drift", True,
            f"0 drift (redis={result.redis_keys_seen}, "
            f"pg={result.postgres_aggregates_seen}, "
            f"{result.elapsed_seconds:.2f}s)",
        )


# --------------------------------------------------------------------------- #
# CLI entry point — `python -m clsplusplus.metering_v2 healthcheck`
# --------------------------------------------------------------------------- #


async def _cli(as_json: bool) -> int:
    settings = Settings()

    async def pool_getter():
        import asyncpg
        url = settings.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgres://", 1)
        return await asyncpg.create_pool(url, min_size=1, max_size=2)

    async def redis_getter():
        try:
            import redis.asyncio as _redis
            return _redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            return None

    pool = None
    try:
        # Build one pool for the whole run rather than creating per-call.
        pool = await pool_getter()

        async def reuse_pool():
            return pool

        check = MeteringHealthCheck(settings, reuse_pool, redis_getter)
        report = await check.run_all()
        if as_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.pretty())
        return 0 if report.passed else 1
    finally:
        if pool is not None:
            await pool.close()


def main() -> None:  # pragma: no cover (entry point)
    as_json = "--json" in sys.argv
    try:
        exit_code = asyncio.run(_cli(as_json))
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
