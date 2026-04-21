"""Schema bootstrapper for the metering v2 pipeline.

This module ONLY knows how to apply and drop the DDL. There are no
writers or readers here yet — per ADR 0001 step 1, we land the schema
behind a feature flag first, with zero production writers.

Nothing in the request path imports this. The app factory will call
`apply_if_enabled()` once on startup; when the flag is off (default),
that call is a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clsplusplus.config import Settings


_DDL_PATH = Path(__file__).parent.parent / "stores" / "metering_v2_ddl.sql"


def read_ddl() -> str:
    """Return the DDL text. Separate from apply_schema so tests can inspect it."""
    return _DDL_PATH.read_text()


async def apply_schema(pool: Any) -> None:
    """Idempotently create the metering v2 tables.

    The DDL uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`
    throughout, so calling this repeatedly is safe.
    """
    ddl = read_ddl()
    async with pool.acquire() as conn:
        await conn.execute(ddl)


async def drop_schema(pool: Any) -> None:
    """Drop the tables. Intended for tests and the documented rollback path.

    Keep this as a CASCADE drop so it works even when a test left data behind.
    """
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS usage_events CASCADE;")
        await conn.execute("DROP TABLE IF EXISTS metering_dead_letter CASCADE;")


async def apply_if_enabled(settings: Settings, pool: Any) -> bool:
    """Apply the schema only when the write flag is on.

    Returns whether the DDL actually ran. Safe to call from startup
    regardless of flag state.
    """
    if not settings.metering_v2_write_enabled:
        return False
    await apply_schema(pool)
    return True
