"""Tests for MeteringNotifier (ADR 0001 step 2).

The notifier has a narrow contract: claim a batch of unnotified rows,
email a digest, mark them notified. We test each transition against a
fake pool + fake email service, so the whole file runs without infra.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from clsplusplus.config import Settings
from clsplusplus.metering_v2.notifier import MeteringNotifier, _render_digest


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeEmail:
    def __init__(self, should_raise: str | None = None, return_value: bool = True):
        self.sent: list[tuple[str, str, str]] = []
        self._raise = should_raise
        self._return = return_value

    async def send_metering_alert(self, to: str, subject: str, html: str) -> bool:
        if self._raise:
            raise RuntimeError(self._raise)
        self.sent.append((to, subject, html))
        return self._return


class FakeRow(dict):
    """Minimal mapping that mimics asyncpg Record semantics for dict-access."""


class FakeConn:
    def __init__(self, owner: "FakePool"):
        self.owner = owner

    async def fetch(self, sql: str, *args):
        if "SELECT id, failed_at" in sql:
            limit = args[0] if args else len(self.owner.unnotified)
            taken, self.owner.unnotified = (
                self.owner.unnotified[:limit],
                self.owner.unnotified[limit:],
            )
            # Keep a pending-claim list so mark_notified can reunite them
            self.owner._claimed.extend(taken)
            return taken
        return []

    async def execute(self, sql: str, *args):
        if "UPDATE metering_dead_letter" in sql:
            ids = set(args[0])
            self.owner.notified_ids.update(ids)


class _AcquireCtx:
    def __init__(self, conn: FakeConn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return None


class FakePool:
    def __init__(self, rows: list[FakeRow]):
        self.unnotified = list(rows)
        self._claimed: list[FakeRow] = []
        self.notified_ids: set[UUID] = set()

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(FakeConn(self))


def _row(error_class="RuntimeError", error_message="boom") -> FakeRow:
    return FakeRow(
        id=uuid4(),
        failed_at=datetime(2026, 4, 20, 9, 0, tzinfo=timezone.utc),
        error_class=error_class,
        error_message=error_message,
        payload={},
    )


def _const_getter(pool):
    async def getter():
        return pool
    return getter


# --------------------------------------------------------------------------- #
# Notifier behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pump_once_empty_queue_sends_nothing():
    pool = FakePool([])
    email = FakeEmail()
    n = MeteringNotifier(
        Settings(metering_v2_write_enabled=True), _const_getter(pool), email,
    )
    assert await n.pump_once() == 0
    assert email.sent == []


@pytest.mark.asyncio
async def test_pump_once_sends_digest_and_marks_notified():
    rows = [_row(), _row("ConnectionError"), _row()]
    pool = FakePool(list(rows))
    email = FakeEmail()
    n = MeteringNotifier(
        Settings(metering_v2_write_enabled=True, oncall_email="on@call.com"),
        _const_getter(pool),
        email,
    )
    count = await n.pump_once()
    assert count == 3
    assert len(email.sent) == 1
    to, subject, html = email.sent[0]
    assert to == "on@call.com"
    assert "3" in subject  # "3 dead-letter event(s)"
    # Each row's id was marked notified.
    assert pool.notified_ids == {r["id"] for r in rows}


@pytest.mark.asyncio
async def test_email_failure_leaves_rows_unnotified():
    """If send fails, we must NOT mark notified — next poll retries them."""
    rows = [_row()]
    pool = FakePool(list(rows))
    email = FakeEmail(should_raise="Resend 500")
    n = MeteringNotifier(
        Settings(metering_v2_write_enabled=True, oncall_email="on@call.com"),
        _const_getter(pool),
        email,
    )
    count = await n.pump_once()
    assert count == 0
    assert pool.notified_ids == set()


@pytest.mark.asyncio
async def test_oncall_unset_skips_send_and_leaves_unnotified():
    rows = [_row()]
    pool = FakePool(list(rows))
    email = FakeEmail()
    n = MeteringNotifier(
        Settings(metering_v2_write_enabled=True, oncall_email=""),
        _const_getter(pool),
        email,
    )
    count = await n.pump_once()
    assert count == 0
    assert email.sent == []
    assert pool.notified_ids == set()


# --------------------------------------------------------------------------- #
# Digest rendering
# --------------------------------------------------------------------------- #


def test_render_digest_groups_by_error_class():
    rows = [
        _row("RuntimeError", "first"),
        _row("RuntimeError", "second"),
        _row("ConnectionError", "db down"),
    ]
    html = _render_digest(rows)
    assert "RuntimeError" in html
    assert "ConnectionError" in html
    # Counts in the <h4>s, largest group first.
    assert html.index("RuntimeError") < html.index("ConnectionError")
    assert "(2)" in html  # RuntimeError bucket size
    assert "(1)" in html  # ConnectionError bucket size


def test_render_digest_escapes_html_in_error_message():
    rows = [_row("E", "<script>alert(1)</script>")]
    html = _render_digest(rows)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_digest_truncates_bucket_over_10_rows():
    rows = [_row("Err", f"msg-{i}") for i in range(15)]
    html = _render_digest(rows)
    assert "and 5 more" in html
