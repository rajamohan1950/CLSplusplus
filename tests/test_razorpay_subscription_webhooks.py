"""Tests for the Razorpay webhook handler — subscription lifecycle events.

Each test posts a fake signed payload and asserts the right store method
was called. The real `razorpay` Python client is NOT invoked because
these are webhook *receivers*, not API calls out.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from clsplusplus.config import Settings
from clsplusplus.razorpay_service import handle_webhook, _compute_expiry


WEBHOOK_SECRET = "test-wh-secret"
USER_ID = "user-42"


def _sign(payload: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return hmac.HMAC(secret.encode(), payload, hashlib.sha256).hexdigest()


def _settings() -> Settings:
    return Settings(razorpay_webhook_secret=WEBHOOK_SECRET)


def _fake_user_service():
    """Build a user_service double whose `.store.set_subscription` we inspect."""
    class Store:
        set_subscription = AsyncMock(return_value=True)

    class Svc:
        store = Store()

    return Svc()


@pytest.mark.asyncio
async def test_missing_webhook_secret_raises():
    with pytest.raises(ValueError, match="webhook secret"):
        await handle_webhook(b"{}", "any-sig", Settings(), _fake_user_service())


@pytest.mark.asyncio
async def test_invalid_signature_raises():
    payload = b'{"event": "payment.captured"}'
    with pytest.raises(ValueError, match="Invalid webhook signature"):
        await handle_webhook(payload, "bogus-sig", _settings(), _fake_user_service())


# --------------------------------------------------------------------------- #
# Payment events (one-time Orders)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_payment_captured_upgrades_and_stamps_expiry():
    payload = json.dumps({
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_abc",
                    "notes": {"user_id": USER_ID, "tier": "pro"},
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)

    svc.store.set_subscription.assert_called_once()
    kw = svc.store.set_subscription.call_args.kwargs
    args = svc.store.set_subscription.call_args.args
    # user_id is positional (first arg)
    assert args[0] == USER_ID
    assert kw["tier"] == "pro"
    assert kw["status"] == "active"
    # Expires ~30 days in the future (pro's TIER_DURATION_DAYS).
    expires = kw["expires_at"]
    delta = expires - datetime.now(timezone.utc)
    assert timedelta(days=29) < delta < timedelta(days=31)


@pytest.mark.asyncio
async def test_payment_failed_does_not_upgrade():
    payload = json.dumps({
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "notes": {"user_id": USER_ID},
                    "error_description": "insufficient funds",
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)

    svc.store.set_subscription.assert_not_called()


# --------------------------------------------------------------------------- #
# Subscription events (recurring)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_subscription_charged_extends_expiry_from_current_end():
    """Razorpay gives `current_end` as epoch seconds; we honour it verbatim."""
    future_ts = int((datetime.now(timezone.utc) + timedelta(days=45)).timestamp())
    payload = json.dumps({
        "event": "subscription.charged",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_xyz",
                    "current_end": future_ts,
                    "notes": {"user_id": USER_ID, "tier": "business"},
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)

    kw = svc.store.set_subscription.call_args.kwargs
    assert kw["tier"] == "business"
    assert kw["status"] == "active"
    assert kw["razorpay_subscription_id"] == "sub_xyz"
    assert int(kw["expires_at"].timestamp()) == future_ts


@pytest.mark.asyncio
async def test_subscription_charged_without_current_end_falls_back_to_30d():
    payload = json.dumps({
        "event": "subscription.charged",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_xyz",
                    "notes": {"user_id": USER_ID, "tier": "pro"},
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)

    kw = svc.store.set_subscription.call_args.kwargs
    delta = kw["expires_at"] - datetime.now(timezone.utc)
    assert timedelta(days=29) < delta < timedelta(days=31)


@pytest.mark.asyncio
async def test_subscription_cancelled_sets_status_without_changing_tier():
    """Cancelled means the user opted out, but keeps paid access until expiry.

    The watchdog handles the actual downgrade when expires_at elapses.
    """
    payload = json.dumps({
        "event": "subscription.cancelled",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_xyz",
                    "notes": {"user_id": USER_ID, "tier": "pro"},
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)

    kw = svc.store.set_subscription.call_args.kwargs
    # tier NOT passed → store keeps existing value via COALESCE.
    assert kw.get("tier") is None
    assert kw["status"] == "cancelled"
    assert kw["razorpay_subscription_id"] == "sub_xyz"


@pytest.mark.asyncio
async def test_subscription_halted_downgrades_immediately():
    payload = json.dumps({
        "event": "subscription.halted",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_xyz",
                    "notes": {"user_id": USER_ID, "tier": "pro"},
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)

    kw = svc.store.set_subscription.call_args.kwargs
    assert kw["tier"] == "free"
    assert kw["status"] == "halted"


@pytest.mark.asyncio
async def test_subscription_completed_downgrades_immediately():
    payload = json.dumps({
        "event": "subscription.completed",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_xyz",
                    "notes": {"user_id": USER_ID, "tier": "pro"},
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)

    kw = svc.store.set_subscription.call_args.kwargs
    assert kw["tier"] == "free"
    assert kw["status"] == "expired"


@pytest.mark.asyncio
async def test_unknown_subscription_event_is_noop():
    payload = json.dumps({
        "event": "subscription.activated",   # we don't handle this
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_xyz",
                    "notes": {"user_id": USER_ID, "tier": "pro"},
                },
            },
        },
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)
    # No store call on ignored events.
    svc.store.set_subscription.assert_not_called()


@pytest.mark.asyncio
async def test_subscription_event_without_user_id_is_noop():
    """If notes lack user_id we can't route the update; log + skip."""
    payload = json.dumps({
        "event": "subscription.halted",
        "payload": {"subscription": {"entity": {"id": "sub_xyz", "notes": {}}}},
    }).encode()

    svc = _fake_user_service()
    await handle_webhook(payload, _sign(payload), _settings(), svc)
    svc.store.set_subscription.assert_not_called()


# --------------------------------------------------------------------------- #
# Sanity check the pure helper.
# --------------------------------------------------------------------------- #


def test_compute_expiry_default_tier_is_30_days():
    now = datetime(2026, 4, 21, tzinfo=timezone.utc)
    assert _compute_expiry("pro", now) == now + timedelta(days=30)
    assert _compute_expiry("business", now) == now + timedelta(days=30)
    assert _compute_expiry("enterprise", now) == now + timedelta(days=30)
    # Unknown tier falls back to 30 days too — defensive.
    assert _compute_expiry("ghost", now) == now + timedelta(days=30)
