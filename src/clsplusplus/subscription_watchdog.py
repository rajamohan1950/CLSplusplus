"""SubscriptionWatchdog — daily downgrade of expired paid tiers.

Scans `users` for rows where `subscription_expires_at < now()` and the
user is still on a paid tier, then calls `expire_subscription` on each.
This is the backstop that makes "Pro license expires" true even when
no webhook arrives (e.g. one-time-payment users who never renew).

Design
------
* Runs once every `POLL_INTERVAL_SECONDS` (default 24h).
* Processes at most `BATCH_LIMIT` rows per cycle so one run never
  holds a long-running lock on the users table.
* Does not itself email anyone — we use the existing
  `metering_dead_letter` → notifier plumbing for oncall-side alerts
  when something goes wrong, and expect a separate "subscription
  expired" email flow for customers (handled elsewhere later).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 24 * 60 * 60
BATCH_LIMIT = 500


class SubscriptionWatchdog:
    """Daily loop + on-demand `run_once()` for admin tooling."""

    def __init__(
        self,
        settings: Settings,
        user_store,
        *,
        tier_resolver=None,
        notify: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        self.settings = settings
        self._store = user_store
        self._tier_resolver = tier_resolver  # optional; used to invalidate cache
        self._notify = notify                 # optional; called after each downgrade
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
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

    async def run_once(self) -> dict:
        """Run one sweep. Returns `{scanned: N, downgraded: M, errors: E}`."""
        try:
            rows = await self._store.get_expired_subscriptions(limit=BATCH_LIMIT)
        except Exception as exc:
            logger.error("watchdog: list failed: %s: %s",
                         type(exc).__name__, exc)
            return {"scanned": 0, "downgraded": 0, "errors": 1,
                    "error": f"{type(exc).__name__}: {exc}"}

        downgraded = 0
        errors = 0
        for row in rows:
            user_id = row["id"]
            try:
                ok = await self._store.expire_subscription(user_id)
                if ok:
                    downgraded += 1
                    if self._tier_resolver is not None:
                        # User's api_keys now have a different tier — flush
                        # the resolver cache so quota + pricer re-resolve.
                        self._tier_resolver.invalidate()
                    if self._notify:
                        try:
                            await self._notify(row)
                        except Exception as exc:  # pragma: no cover - best effort
                            logger.warning(
                                "watchdog: notify failed for %s: %s",
                                user_id, exc,
                            )
            except Exception as exc:
                errors += 1
                logger.error(
                    "watchdog: downgrade failed for %s: %s: %s",
                    user_id, type(exc).__name__, exc,
                )

        if downgraded or errors:
            logger.info(
                "watchdog: scanned=%d downgraded=%d errors=%d",
                len(rows), downgraded, errors,
            )
        return {"scanned": len(rows), "downgraded": downgraded, "errors": errors}

    async def _run(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("watchdog loop: %s: %s",
                             type(exc).__name__, exc)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
