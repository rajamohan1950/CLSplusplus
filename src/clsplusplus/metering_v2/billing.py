"""Overage billing — closes the loop on pay-as-you-go pricing.

The pricer (pricing.py) stamps `unit_cost_cents` on every usage_events
row that crossed the tier cap. This module aggregates those stamps per
user at period close and actually collects the money: it creates a
Razorpay Payment Link for the overage total and records the invoice in
`revenue_events`.

Idempotency is the partial unique index on
`revenue_events(user_id, period) WHERE event_type='overage'` — a re-run
for an already-billed month is a no-op, so the daily loop can safely
re-attempt the prior period every cycle.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Any, Awaitable, Callable, Optional

from clsplusplus.config import Settings
from clsplusplus.metering_v2.reconciler import _period_window, _prior_period

logger = logging.getLogger(__name__)

# Daily loop interval — bills the prior period each cycle (idempotent).
POLL_INTERVAL_SECONDS = 24 * 60 * 60
# Don't invoice trivial amounts — a Razorpay link for 3¢ is not worth it.
DEFAULT_MIN_CHARGE_CENTS = 50


PoolGetter = Callable[[], Awaitable[Any]]


@dataclass
class BillingResult:
    period: str
    candidates: int          # users with a positive overage total
    billed: int              # invoices created this run
    skipped_already: int     # already invoiced for this period
    skipped_below_min: int   # overage below the minimum charge
    errors: int

    def summary(self) -> dict:
        return asdict(self)


class OverageBiller:
    """Aggregates over-cap usage and invoices it via Razorpay payment links.

    `bill_once(period)` runs a single billing pass and returns a result.
    `start()/stop()` runs the daily loop over the prior period.
    """

    def __init__(
        self,
        settings: Settings,
        pool_getter: PoolGetter,
        user_service: Any,
        min_charge_cents: int = DEFAULT_MIN_CHARGE_CENTS,
    ):
        self.settings = settings
        self._pool_getter = pool_getter
        self._user_service = user_service
        self._min_charge = min_charge_cents
        self._task: Optional[asyncio.Task] = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.metering_v2_write_enabled)

    def start(self) -> None:
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

    async def bill_once(self, period: Optional[str] = None) -> BillingResult:
        """Invoice every user's over-cap usage for `period` (default: prior)."""
        period = period or _prior_period()
        totals = await self._aggregate_overage(period)
        already = await self._user_service.store.get_billed_overage_user_ids(period)

        billed = skipped_already = skipped_below_min = errors = 0
        for user_id, cents in totals.items():
            if user_id in already:
                skipped_already += 1
                continue
            if cents < self._min_charge:
                skipped_below_min += 1
                continue
            try:
                await self._invoice_user(user_id, cents, period)
                billed += 1
            except Exception as exc:
                errors += 1
                logger.error(
                    "overage biller: failed to invoice user %s for %s: %s: %s",
                    user_id, period, type(exc).__name__, exc,
                )

        return BillingResult(
            period=period,
            candidates=len(totals),
            billed=billed,
            skipped_already=skipped_already,
            skipped_below_min=skipped_below_min,
            errors=errors,
        )

    # --- internals ---------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                result = await self.bill_once(_prior_period())
                if result.billed:
                    logger.info("overage biller: invoiced %d user(s) for %s",
                                result.billed, result.period)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("overage biller: %s: %s",
                             type(exc).__name__, exc)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _aggregate_overage(self, period: str) -> dict[str, int]:
        """Return {user_id: total overage cents} for the period."""
        start, end = _period_window(period)
        pool = await self._pool_getter()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT user_id::text AS user_id,
                       SUM(unit_cost_cents * quantity)::BIGINT AS cents
                FROM usage_events
                WHERE user_id IS NOT NULL
                  AND unit_cost_cents > 0
                  AND occurred_at >= $1
                  AND occurred_at <  $2
                GROUP BY user_id
                HAVING SUM(unit_cost_cents * quantity) > 0
                """,
                start, end,
            )
        return {r["user_id"]: int(r["cents"]) for r in rows}

    async def _invoice_user(self, user_id: str, cents: int, period: str) -> None:
        """Create a Razorpay payment link, record the invoice, email the user.

        The revenue_events row is recorded before our own email is sent so
        it acts as the idempotency claim. Razorpay also emails the link
        directly (notify.email=True), so our email is a branded extra and
        its failure is non-fatal.
        """
        from clsplusplus import razorpay_service

        user = await self._user_service.get_user(user_id)
        if not user or not user.get("email"):
            raise ValueError(f"user {user_id} has no email on file")
        email = user["email"]

        link = await razorpay_service.create_overage_payment_link(
            user_id, email, cents, period, self.settings,
        )
        await self._user_service.store.record_overage_event(
            user_id, period, cents, link["id"],
        )
        try:
            await self._user_service.email.send_overage_invoice(
                email, period, cents, link.get("short_url", ""),
            )
        except Exception as exc:
            logger.warning(
                "overage biller: branded email failed for user %s "
                "(Razorpay still notified them): %s", user_id, exc,
            )
