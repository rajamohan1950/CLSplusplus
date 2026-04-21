"""CLS++ Razorpay Billing Service — Orders, Payment Verification, and Webhooks."""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

import razorpay

if TYPE_CHECKING:
    from clsplusplus.config import Settings
    from clsplusplus.user_service import UserService

logger = logging.getLogger(__name__)

# Authoritative prices in USD cents — mirrors src/clsplusplus/tiers.TIER_PRICES.
# Razorpay accepts currency=USD for merchants with international payments
# enabled. If the merchant is INR-only, the create-order call will surface a
# clear error up to the UI.
TIER_AMOUNT_USD_CENTS: dict[str, int] = {
    "pro": 900,          # $9.00
    "business": 2900,    # $29.00
    "enterprise": 14900, # $149.00
}

# Legacy INR amounts (paise). Kept as fallback for INR-only Razorpay accounts;
# the backend picks USD by default.
TIER_AMOUNT_PAISE: dict[str, int] = {
    "pro": 74900,         # INR 749
    "business": 239900,   # INR 2,399
    "enterprise": 1239900,  # INR 12,399
}

# How long each tier buys when paid via a one-time Razorpay Order. A follow-up
# migration to Razorpay Subscriptions will replace this with actual
# current_period_end pulled from the subscription.charged webhook.
TIER_DURATION_DAYS: dict[str, int] = {
    "pro": 30,
    "business": 30,
    "enterprise": 30,
}


def _compute_expiry(tier: str, now: Optional[datetime] = None) -> datetime:
    """Return the expires_at stamp for a fresh payment at `tier`."""
    now = now or datetime.now(timezone.utc)
    days = TIER_DURATION_DAYS.get(tier, 30)
    return now + timedelta(days=days)


def _get_client(settings: "Settings") -> razorpay.Client:
    """Create a Razorpay client from settings."""
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise ValueError("Razorpay is not configured (CLS_RAZORPAY_KEY_ID or CLS_RAZORPAY_KEY_SECRET missing)")
    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


async def create_order(
    user_id: str,
    tier: str,
    settings: "Settings",
    currency: str = "USD",
) -> dict:
    """Create a Razorpay Order for a tier upgrade.

    Default currency is USD (the prices the UI shows). Callers can override
    with currency="INR" for INR-only merchant accounts.
    """
    if tier == "free":
        raise ValueError("Cannot create order for the free tier")

    currency = (currency or "USD").upper()
    if currency == "USD":
        amount = TIER_AMOUNT_USD_CENTS.get(tier)
    elif currency == "INR":
        amount = TIER_AMOUNT_PAISE.get(tier)
    else:
        raise ValueError(f"Unsupported currency: {currency}")

    if amount is None:
        raise ValueError(f"Invalid tier: {tier}")

    client = _get_client(settings)

    order_data = {
        "amount": amount,
        "currency": currency,
        "notes": {"user_id": user_id, "tier": tier},
    }

    order = client.order.create(data=order_data)

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "key_id": settings.razorpay_key_id,
        "tier": tier,
    }


async def _upgrade_and_stamp_expiry(
    user_service: "UserService",
    user_id: str,
    tier: str,
    razorpay_subscription_id: Optional[str] = None,
) -> None:
    """Set tier=<paid>, status=active, expires_at=now+duration atomically.

    Used by both the synchronous verify-payment path and the
    payment.captured / subscription.charged webhook handlers so the
    "what does a successful payment mean" logic lives in one place.
    """
    expires_at = _compute_expiry(tier)
    await user_service.store.set_subscription(
        user_id,
        tier=tier,
        expires_at=expires_at,
        status="active",
        razorpay_subscription_id=razorpay_subscription_id,
    )


async def verify_payment(
    order_id: str,
    payment_id: str,
    signature: str,
    settings: "Settings",
    user_service: "UserService",
    tier: str,
    user_id: str,
) -> bool:
    """Verify Razorpay payment signature and upgrade user tier.

    Razorpay signature = HMAC-SHA256(order_id + "|" + payment_id, key_secret)
    Returns True if payment is verified and tier upgraded.
    """
    expected = hmac.HMAC(
        settings.razorpay_key_secret.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        logger.warning(
            "Razorpay signature mismatch for order %s, payment %s",
            order_id, payment_id,
        )
        raise ValueError("Payment verification failed — invalid signature")

    await _upgrade_and_stamp_expiry(user_service, user_id, tier)
    logger.info(
        "Razorpay payment verified: upgraded user %s to %s (order=%s, payment=%s)",
        user_id, tier, order_id, payment_id,
    )
    return True


async def handle_webhook(
    payload: bytes,
    sig: str,
    settings: "Settings",
    user_service: "UserService",
) -> None:
    """Process a Razorpay webhook event.

    Handles:
      * payment.captured          — upgrade + stamp expiry (one-time Orders)
      * payment.failed            — log only
      * subscription.charged      — recurring renewal: extend expiry
      * subscription.cancelled    — status=cancelled; downgrade at expiry
      * subscription.halted       — repeated renewal failures: downgrade now
      * subscription.completed    — fixed-term ended: downgrade now
    """
    if not settings.razorpay_webhook_secret:
        raise ValueError("Razorpay webhook secret not configured")

    # Verify webhook signature
    expected = hmac.HMAC(
        settings.razorpay_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, sig):
        raise ValueError("Invalid webhook signature")

    import json
    event = json.loads(payload)
    event_type = event.get("event", "")

    # Razorpay nests entities under payload.{payment|subscription|...}.entity
    pl = event.get("payload") or {}

    # -------- Payment events (one-time Orders) --------
    if event_type in ("payment.captured", "payment.failed"):
        entity = (pl.get("payment") or {}).get("entity") or {}
        notes = entity.get("notes") or {}
        user_id = notes.get("user_id")
        tier = notes.get("tier")

        if event_type == "payment.captured" and user_id and tier:
            try:
                await _upgrade_and_stamp_expiry(user_service, user_id, tier)
                logger.info(
                    "Razorpay: payment captured — user %s → %s (expires in %d days)",
                    user_id, tier, TIER_DURATION_DAYS.get(tier, 30),
                )
            except Exception as exc:
                logger.error("Failed to upgrade user %s after payment.captured: %s",
                             user_id, exc)
                raise
        elif event_type == "payment.failed":
            logger.warning(
                "Razorpay: payment.failed for user %s (reason: %s)",
                user_id, entity.get("error_description", "unknown"),
            )
        return

    # -------- Subscription events (recurring Subscriptions) --------
    subscription_entity = (pl.get("subscription") or {}).get("entity") or {}
    sub_id = subscription_entity.get("id")
    notes = subscription_entity.get("notes") or {}
    user_id = notes.get("user_id")
    tier = notes.get("tier")

    if not user_id:
        logger.warning("Razorpay webhook %s missing user_id in notes (sub_id=%s)",
                       event_type, sub_id)
        return

    try:
        if event_type == "subscription.charged":
            # Extend the window. Razorpay gives current_end (epoch seconds)
            # on the subscription — honour that when present.
            current_end = subscription_entity.get("current_end")
            if current_end:
                expires_at = datetime.fromtimestamp(int(current_end), tz=timezone.utc)
            else:
                expires_at = _compute_expiry(tier or "pro")
            await user_service.store.set_subscription(
                user_id,
                tier=tier,
                expires_at=expires_at,
                status="active",
                razorpay_subscription_id=sub_id,
            )
            logger.info("Razorpay: subscription.charged — user %s expires %s",
                        user_id, expires_at.isoformat())

        elif event_type == "subscription.cancelled":
            # User cancelled. Keep their paid tier until current_end; then
            # the watchdog downgrades them on expiry.
            await user_service.store.set_subscription(
                user_id, status="cancelled",
                razorpay_subscription_id=sub_id,
            )
            logger.info(
                "Razorpay: subscription.cancelled — user %s status=cancelled "
                "(tier preserved until expires_at)", user_id,
            )

        elif event_type == "subscription.halted":
            # Repeated renewal failures — Razorpay halted the subscription.
            # Downgrade immediately: the user's current paid window is
            # effectively forfeit.
            await user_service.store.set_subscription(
                user_id,
                tier="free",
                status="halted",
                razorpay_subscription_id=sub_id,
            )
            logger.warning(
                "Razorpay: subscription.halted — user %s downgraded to free", user_id,
            )

        elif event_type == "subscription.completed":
            # Fixed-term subscription ran its course. Downgrade now.
            await user_service.store.set_subscription(
                user_id,
                tier="free",
                status="expired",
                razorpay_subscription_id=sub_id,
            )
            logger.info(
                "Razorpay: subscription.completed — user %s downgraded to free", user_id,
            )

        else:
            logger.debug("Razorpay webhook: ignoring unhandled event %s", event_type)
    except Exception as exc:
        logger.error("Razorpay webhook %s failed for user %s: %s: %s",
                     event_type, user_id, type(exc).__name__, exc)
        raise
