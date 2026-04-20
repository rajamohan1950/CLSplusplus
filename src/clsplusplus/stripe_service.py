"""CLS++ Stripe Billing Service — Checkout, Portal, and Webhook handling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import stripe

if TYPE_CHECKING:
    from clsplusplus.config import Settings
    from clsplusplus.user_service import UserService

logger = logging.getLogger(__name__)

# Map CLS++ tier names to config attribute names for Stripe Price IDs
_TIER_PRICE_ATTR = {
    "pro": "stripe_price_pro",
    "business": "stripe_price_business",
    "enterprise": "stripe_price_enterprise",
}


def _init_stripe(settings: "Settings") -> None:
    """Set the Stripe API key from settings."""
    if not settings.stripe_secret_key:
        raise ValueError("Stripe is not configured (CLS_STRIPE_SECRET_KEY missing)")
    stripe.api_key = settings.stripe_secret_key


async def create_checkout_session(
    user_id: str,
    tier: str,
    settings: "Settings",
) -> str:
    """Create a Stripe Checkout session for a tier upgrade.

    Returns the Checkout session URL to redirect the user to.
    """
    _init_stripe(settings)

    if tier == "free":
        raise ValueError("Cannot checkout for the free tier")

    price_attr = _TIER_PRICE_ATTR.get(tier)
    if not price_attr:
        raise ValueError(f"Invalid tier: {tier}")

    price_id = getattr(settings, price_attr, "")
    if not price_id:
        raise ValueError(f"Stripe Price ID not configured for tier: {tier}")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=settings.stripe_success_url,
        cancel_url=settings.stripe_cancel_url,
        client_reference_id=user_id,
        metadata={"user_id": user_id, "tier": tier},
    )
    return session.url


async def create_portal_session(
    user_id: str,
    settings: "Settings",
) -> str:
    """Create a Stripe Customer Portal session for subscription management.

    Returns the portal URL.
    """
    _init_stripe(settings)

    # Find the Stripe customer by user_id metadata
    customers = stripe.Customer.search(
        query=f'metadata["user_id"]:"{user_id}"',
    )
    if not customers.data:
        raise ValueError("No billing account found. Please subscribe to a plan first.")

    customer_id = customers.data[0].id

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=settings.stripe_success_url,
    )
    return session.url


async def handle_webhook(
    payload: bytes,
    sig: str,
    settings: "Settings",
    user_service: "UserService",
) -> None:
    """Process a Stripe webhook event.

    Handles checkout.session.completed to upgrade user tier.
    """
    _init_stripe(settings)

    if not settings.stripe_webhook_secret:
        raise ValueError("Stripe webhook secret not configured")

    event = stripe.Webhook.construct_event(
        payload=payload,
        sig_header=sig,
        secret=settings.stripe_webhook_secret,
    )

    if event.type == "checkout.session.completed":
        session = event.data.object
        user_id = session.get("client_reference_id") or (session.get("metadata") or {}).get("user_id")
        tier = (session.get("metadata") or {}).get("tier")

        if user_id and tier:
            try:
                await user_service.update_tier(user_id, tier)
                logger.info("Stripe checkout: upgraded user %s to %s", user_id, tier)
            except Exception as e:
                logger.error("Failed to upgrade user %s after checkout: %s", user_id, e)
                raise

    elif event.type == "customer.subscription.deleted":
        subscription = event.data.object
        customer_id = subscription.get("customer")
        if customer_id:
            try:
                customer = stripe.Customer.retrieve(customer_id)
                user_id = (customer.get("metadata") or {}).get("user_id")
                if user_id:
                    await user_service.update_tier(user_id, "free")
                    logger.info("Stripe subscription cancelled: downgraded user %s to free", user_id)
            except Exception as e:
                logger.error("Failed to downgrade user after subscription cancel: %s", e)
                raise
