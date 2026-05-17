"""Tests for OverageBiller — closes the pay-as-you-go billing loop.

Fakes the DB pool, the user service, and the Razorpay payment-link call
so the whole file runs without infra.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.metering_v2.billing import OverageBiller


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeConn:
    def __init__(self, owner: "FakePool"):
        self.owner = owner

    async def fetch(self, sql: str, *args):
        if "FROM usage_events" in sql:
            return [
                {"user_id": uid, "cents": c}
                for uid, c in self.owner.overage_totals.items()
            ]
        return []


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return None


class FakePool:
    def __init__(self, overage_totals: dict[str, int]):
        self.overage_totals = dict(overage_totals)

    def acquire(self):
        return _Acquire(FakeConn(self))


class FakeStore:
    def __init__(self, billed: Optional[set] = None):
        self.billed: set[str] = set(billed or set())
        self.recorded: list[dict] = []

    async def get_billed_overage_user_ids(self, period: str) -> set:
        return set(self.billed)

    async def record_overage_event(self, user_id, period, amount_cents,
                                   razorpay_order_id=None):
        # Idempotent, like the real partial unique index.
        if user_id in self.billed:
            return None
        self.billed.add(user_id)
        row = {
            "user_id": user_id, "period": period,
            "amount_cents": amount_cents, "razorpay_order_id": razorpay_order_id,
        }
        self.recorded.append(row)
        return row


class FakeEmail:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.sent: list[tuple] = []

    async def send_overage_invoice(self, to, period, amount_cents, pay_url):
        if self.fail:
            raise RuntimeError("resend down")
        self.sent.append((to, period, amount_cents, pay_url))
        return True


class FakeUserService:
    def __init__(self, users: dict[str, dict], store: FakeStore, email: FakeEmail):
        self._users = users
        self.store = store
        self.email = email

    async def get_user(self, user_id: str) -> Optional[dict]:
        return self._users.get(user_id)


def _biller(
    overage_totals: dict[str, int],
    users: dict[str, dict],
    *,
    billed: Optional[set] = None,
    email_fails: bool = False,
    min_charge_cents: int = 50,
) -> tuple[OverageBiller, FakeStore, FakeEmail]:
    pool = FakePool(overage_totals)
    store = FakeStore(billed)
    email = FakeEmail(fail=email_fails)
    user_service = FakeUserService(users, store, email)

    async def pool_getter() -> Any:
        return pool

    settings = Settings(metering_v2_write_enabled=True)
    biller = OverageBiller(
        settings, pool_getter, user_service,
        min_charge_cents=min_charge_cents,
    )
    return biller, store, email


_FAKE_LINK = {
    "id": "plink_test_1",
    "short_url": "https://rzp.io/i/test",
    "amount": 0,
    "currency": "USD",
    "status": "created",
}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_happy_path_invoices_a_user():
    biller, store, email = _biller(
        overage_totals={"u-1": 500},
        users={"u-1": {"id": "u-1", "email": "alice@example.com"}},
    )
    with patch(
        "clsplusplus.razorpay_service.create_overage_payment_link",
        new=AsyncMock(return_value=_FAKE_LINK),
    ) as mock_link:
        result = await biller.bill_once("2026-04")

    assert result.candidates == 1
    assert result.billed == 1
    assert result.errors == 0
    # Razorpay payment link was created for the right amount.
    mock_link.assert_called_once()
    assert mock_link.call_args[0][2] == 500   # amount_cents
    # Invoice recorded + branded email sent.
    assert len(store.recorded) == 1
    assert store.recorded[0]["amount_cents"] == 500
    assert len(email.sent) == 1


@pytest.mark.asyncio
async def test_below_minimum_charge_is_skipped():
    biller, store, email = _biller(
        overage_totals={"u-1": 10},   # 10¢ < 50¢ minimum
        users={"u-1": {"id": "u-1", "email": "alice@example.com"}},
    )
    with patch(
        "clsplusplus.razorpay_service.create_overage_payment_link",
        new=AsyncMock(return_value=_FAKE_LINK),
    ) as mock_link:
        result = await biller.bill_once("2026-04")

    assert result.candidates == 1
    assert result.billed == 0
    assert result.skipped_below_min == 1
    mock_link.assert_not_called()
    assert store.recorded == []


@pytest.mark.asyncio
async def test_already_billed_user_is_skipped():
    biller, store, email = _biller(
        overage_totals={"u-1": 900},
        users={"u-1": {"id": "u-1", "email": "alice@example.com"}},
        billed={"u-1"},
    )
    with patch(
        "clsplusplus.razorpay_service.create_overage_payment_link",
        new=AsyncMock(return_value=_FAKE_LINK),
    ) as mock_link:
        result = await biller.bill_once("2026-04")

    assert result.candidates == 1
    assert result.billed == 0
    assert result.skipped_already == 1
    mock_link.assert_not_called()


@pytest.mark.asyncio
async def test_re_run_is_idempotent():
    """The second pass over the same period invoices nobody new."""
    biller, store, email = _biller(
        overage_totals={"u-1": 900},
        users={"u-1": {"id": "u-1", "email": "alice@example.com"}},
    )
    with patch(
        "clsplusplus.razorpay_service.create_overage_payment_link",
        new=AsyncMock(return_value=_FAKE_LINK),
    ):
        first = await biller.bill_once("2026-04")
        second = await biller.bill_once("2026-04")

    assert first.billed == 1
    assert second.billed == 0
    assert second.skipped_already == 1
    assert len(store.recorded) == 1


@pytest.mark.asyncio
async def test_no_overage_produces_no_invoices():
    """A period with no over-cap usage (e.g. all enterprise/free) bills nobody."""
    biller, store, email = _biller(overage_totals={}, users={})
    with patch(
        "clsplusplus.razorpay_service.create_overage_payment_link",
        new=AsyncMock(return_value=_FAKE_LINK),
    ) as mock_link:
        result = await biller.bill_once("2026-04")

    assert result.candidates == 0
    assert result.billed == 0
    mock_link.assert_not_called()


@pytest.mark.asyncio
async def test_user_without_email_counts_as_error():
    biller, store, email = _biller(
        overage_totals={"u-1": 900},
        users={"u-1": {"id": "u-1", "email": ""}},
    )
    with patch(
        "clsplusplus.razorpay_service.create_overage_payment_link",
        new=AsyncMock(return_value=_FAKE_LINK),
    ):
        result = await biller.bill_once("2026-04")

    assert result.errors == 1
    assert result.billed == 0
    assert store.recorded == []


@pytest.mark.asyncio
async def test_branded_email_failure_is_non_fatal():
    """Razorpay also emails the link, so our email failing must not fail billing."""
    biller, store, email = _biller(
        overage_totals={"u-1": 900},
        users={"u-1": {"id": "u-1", "email": "alice@example.com"}},
        email_fails=True,
    )
    with patch(
        "clsplusplus.razorpay_service.create_overage_payment_link",
        new=AsyncMock(return_value=_FAKE_LINK),
    ):
        result = await biller.bill_once("2026-04")

    assert result.billed == 1
    assert result.errors == 0
    assert len(store.recorded) == 1   # invoice still recorded
