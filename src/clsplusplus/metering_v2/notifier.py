"""Dead-letter notifier (ADR 0001 step 2).

Polls `metering_dead_letter` for rows where `notified_at IS NULL`,
digests them into one email per poll cycle, sends to the on-call
address, and stamps `notified_at = NOW()`. The table IS the queue —
no separate queue infrastructure.

Rate-limit safety: we batch (up to `BATCH_LIMIT` rows per email) so a
storm of failures cannot flood the inbox.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from clsplusplus.config import Settings
from clsplusplus.email_service import EmailService

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 60
BATCH_LIMIT = 50


PoolGetter = Callable[[], Awaitable[Any]]


class MeteringNotifier:
    """Background worker that pages on-call when writes fail."""

    def __init__(
        self,
        settings: Settings,
        pool_getter: PoolGetter,
        email_service: EmailService,
    ):
        self.settings = settings
        self._pool_getter = pool_getter
        self._email = email_service
        self._task: Optional[asyncio.Task] = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.metering_v2_write_enabled)

    def start(self) -> None:
        """Kick off the poll loop. No-op if already running or flag off."""
        if self._task is not None and not self._task.done():
            return
        if not self.enabled:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def pump_once(self) -> int:
        """Run one poll cycle and return the number of rows successfully notified.

        Exposed for tests and manual replay — production uses start()/stop().
        """
        rows = await self._claim_batch()
        if not rows:
            return 0
        if not await self._send_digest(rows):
            return 0
        await self._mark_notified([r["id"] for r in rows])
        return len(rows)

    # --- internals ---------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                count = await self.pump_once()
                if count:
                    logger.info("metering notifier: paged %d events", count)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "metering notifier: %s: %s", type(exc).__name__, exc,
                )
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _claim_batch(self) -> list:
        pool = await self._pool_getter()
        async with pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT id, failed_at, error_class, error_message, payload
                FROM metering_dead_letter
                WHERE notified_at IS NULL
                ORDER BY failed_at
                LIMIT $1
                """,
                BATCH_LIMIT,
            )

    async def _send_digest(self, rows: list) -> bool:
        to = self.settings.oncall_email
        if not to:
            logger.warning(
                "metering notifier: oncall email unset; %d events pending",
                len(rows),
            )
            return False
        subject = f"[CLS++ metering] {len(rows)} dead-letter event(s)"
        html = _render_digest(rows)
        try:
            return await self._email.send_metering_alert(to, subject, html)
        except Exception as exc:
            logger.error("metering notifier: email failed: %s: %s", type(exc).__name__, exc)
            return False

    async def _mark_notified(self, ids: list) -> None:
        pool = await self._pool_getter()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE metering_dead_letter
                SET notified_at = NOW()
                WHERE id = ANY($1::uuid[])
                """,
                ids,
            )


def _render_digest(rows: list) -> str:
    """Compact HTML digest. One <li> per failure, error_class grouped."""
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r["error_class"], []).append(r)

    blocks = []
    for err_class, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        lis = "".join(
            f"<li><time>{r['failed_at'].isoformat(timespec='seconds')}</time> — "
            f"{_escape(r['error_message'][:250])}</li>"
            for r in items[:10]  # limit per bucket so the email stays readable
        )
        overflow = len(items) - 10
        more = f"<li>... and {overflow} more</li>" if overflow > 0 else ""
        blocks.append(
            f"<h4>{_escape(err_class)} "
            f"<span style='color:#888'>({len(items)})</span></h4>"
            f"<ul>{lis}{more}</ul>"
        )
    return (
        "<h3>CLS++ metering dead-letter digest</h3>"
        f"<p>{len(rows)} event(s) failed to write and need investigation or replay. "
        "Check the <code>metering_dead_letter</code> table for full payloads.</p>"
        + "".join(blocks)
    )


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
