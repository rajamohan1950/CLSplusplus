"""CLS++ Razorpay Billing Service — Orders, Payment Verification, and Webhooks."""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import TYPE_CHECKING

import razorpay

if TYPE_CHECKING:
    from clsplusplus.config import Settings
    from clsplusplus.user_service import UserService

logger = logging.getLogger(__name__)

# Razorpay amounts are in paise (1/100 of INR)
TIER_AMOUNT_PAISE: dict[str, int] = {
    "pro": 74900,         # INR 749
    "business": 239900,   # INR 2,399
    "enterprise": 1239900,  # INR 12,399
}


def _get_client(settings: "Settings") -> razorpay.Client:
    """Create a Razorpay client from settings."""
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise ValueError("Razorpay is not configured (CLS_RAZORPAY_KEY_ID or CLS_RAZORPAY_KEY_SECRET missing)")
    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


async def create_order(
    user_id: str,
    tier: str,
    settings: "Settings",
) -> dict:
    """Create a Razorpay Order for a tier upgrade.

    Returns dict with order_id, amount, currency, key_id for frontend checkout.
    """
    if tier == "free":
        raise ValueError("Cannot create order for the free tier")

    amount = TIER_AMOUNT_PAISE.get(tier)
    if amount is None:
        raise ValueError(f"Invalid tier: {tier}")

    client = _get_client(settings)

    order_data = {
        "amount": amount,
        "currency": "INR",
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
    expected = hmac.new(
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

    await user_service.update_tier(user_id, tier)
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

    Verifies webhook signature, then handles payment.captured and payment.failed.
    """
    if not settings.razorpay_webhook_secret:
        raise ValueError("Razorpay webhook secret not configured")

    # Verify webhook signature
    expected = hmac.new(
        settings.razorpay_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, sig):
        raise ValueError("Invalid webhook signature")

    import json
    event = json.loads(payload)
    event_type = event.get("event", "")
    entity = (event.get("payload", {}).get("payment", {}).get("entity", {}))

    if event_type == "payment.captured":
        notes = entity.get("notes", {})
        user_id = notes.get("user_id")
        tier = notes.get("tier")
        if user_id and tier:
            try:
                await user_service.update_tier(user_id, tier)
                logger.info(
                    "Razorpay webhook: payment captured, upgraded user %s to %s",
                    user_id, tier,
                )
            except Exception as e:
                logger.error("Failed to upgrade user %s after webhook: %s", user_id, e)
                raise

    elif event_type == "payment.failed":
        notes = entity.get("notes", {})
        user_id = notes.get("user_id")
        logger.warning(
            "Razorpay webhook: payment failed for user %s, reason: %s",
            user_id,
            entity.get("error_description", "unknown"),
        )
