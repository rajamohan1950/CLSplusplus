"""Razorpay billing tests — order creation, payment verification, webhook handling.

Tests the complete payment flow: order → checkout → verify → tier upgrade.
Uses mocks for Razorpay client since we can't hit the real API in CI.
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clsplusplus.config import Settings


# ── Test Settings ─────────────────────────────────────────────────────────────

TEST_KEY_ID = "rzp_test_ABCDEF1234567890"
TEST_KEY_SECRET = "test_secret_key_for_hmac"
TEST_WEBHOOK_SECRET = "whsec_test_webhook_secret"


@pytest.fixture
def billing_settings():
    return Settings(
        razorpay_key_id=TEST_KEY_ID,
        razorpay_key_secret=TEST_KEY_SECRET,
        razorpay_webhook_secret=TEST_WEBHOOK_SECRET,
    )


@pytest.fixture
def no_billing_settings():
    return Settings(
        razorpay_key_id="",
        razorpay_key_secret="",
        razorpay_webhook_secret="",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Razorpay Service — Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRazorpayService:

    def test_tier_amounts_defined(self):
        from clsplusplus.razorpay_service import TIER_AMOUNT_PAISE
        assert "pro" in TIER_AMOUNT_PAISE
        assert "business" in TIER_AMOUNT_PAISE
        assert "enterprise" in TIER_AMOUNT_PAISE
        # Amounts in paise (INR)
        assert TIER_AMOUNT_PAISE["pro"] == 74900  # INR 749
        assert TIER_AMOUNT_PAISE["business"] == 239900  # INR 2,399
        assert TIER_AMOUNT_PAISE["enterprise"] == 1239900  # INR 12,399

    def test_free_tier_not_billable(self):
        from clsplusplus.razorpay_service import TIER_AMOUNT_PAISE
        assert "free" not in TIER_AMOUNT_PAISE

    @pytest.mark.asyncio
    async def test_create_order_rejects_free(self, billing_settings):
        from clsplusplus.razorpay_service import create_order
        with pytest.raises(ValueError, match="Cannot create order for the free tier"):
            await create_order("user-123", "free", billing_settings)

    @pytest.mark.asyncio
    async def test_create_order_rejects_invalid_tier(self, billing_settings):
        from clsplusplus.razorpay_service import create_order
        with pytest.raises(ValueError, match="Invalid tier"):
            await create_order("user-123", "nonexistent", billing_settings)

    @pytest.mark.asyncio
    async def test_create_order_no_config(self, no_billing_settings):
        from clsplusplus.razorpay_service import create_order
        with pytest.raises(ValueError, match="not configured"):
            await create_order("user-123", "pro", no_billing_settings)

    @pytest.mark.asyncio
    async def test_create_order_success(self, billing_settings):
        from clsplusplus.razorpay_service import create_order

        mock_client = MagicMock()
        mock_client.order.create.return_value = {
            "id": "order_test_123",
            "amount": 74900,
            "currency": "INR",
        }

        with patch("clsplusplus.razorpay_service._get_client", return_value=mock_client):
            result = await create_order("user-123", "pro", billing_settings)

        assert result["order_id"] == "order_test_123"
        assert result["amount"] == 74900
        assert result["currency"] == "INR"
        assert result["key_id"] == TEST_KEY_ID
        assert result["tier"] == "pro"

        # Verify Razorpay client was called correctly
        call_args = mock_client.order.create.call_args[1]["data"]
        assert call_args["amount"] == 74900
        assert call_args["currency"] == "INR"
        assert call_args["notes"]["user_id"] == "user-123"
        assert call_args["notes"]["tier"] == "pro"

    @pytest.mark.asyncio
    async def test_create_order_business(self, billing_settings):
        from clsplusplus.razorpay_service import create_order

        mock_client = MagicMock()
        mock_client.order.create.return_value = {
            "id": "order_biz_456",
            "amount": 239900,
            "currency": "INR",
        }

        with patch("clsplusplus.razorpay_service._get_client", return_value=mock_client):
            result = await create_order("user-456", "business", billing_settings)

        assert result["amount"] == 239900
        assert result["tier"] == "business"


# ═══════════════════════════════════════════════════════════════════════════════
# Payment Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaymentVerification:

    def _make_signature(self, order_id, payment_id, secret=TEST_KEY_SECRET):
        """Generate valid Razorpay HMAC signature."""
        return hmac.new(
            secret.encode(),
            f"{order_id}|{payment_id}".encode(),
            hashlib.sha256,
        ).hexdigest()

    @pytest.mark.asyncio
    async def test_verify_valid_signature(self, billing_settings):
        from clsplusplus.razorpay_service import verify_payment

        order_id = "order_test_123"
        payment_id = "pay_test_456"
        sig = self._make_signature(order_id, payment_id)

        mock_user_service = AsyncMock()
        mock_user_service.update_tier = AsyncMock()

        result = await verify_payment(
            order_id=order_id,
            payment_id=payment_id,
            signature=sig,
            settings=billing_settings,
            user_service=mock_user_service,
            tier="pro",
            user_id="user-123",
        )

        assert result is True
        mock_user_service.update_tier.assert_called_once_with("user-123", "pro")

    @pytest.mark.asyncio
    async def test_verify_invalid_signature(self, billing_settings):
        from clsplusplus.razorpay_service import verify_payment

        mock_user_service = AsyncMock()

        with pytest.raises(ValueError, match="invalid signature"):
            await verify_payment(
                order_id="order_123",
                payment_id="pay_456",
                signature="invalid_sig",
                settings=billing_settings,
                user_service=mock_user_service,
                tier="pro",
                user_id="user-123",
            )

        # User tier should NOT be updated on invalid signature
        mock_user_service.update_tier.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_tampered_order_id(self, billing_settings):
        """Changing order ID after signature was generated should fail."""
        from clsplusplus.razorpay_service import verify_payment

        sig = self._make_signature("order_original", "pay_456")
        mock_user_service = AsyncMock()

        with pytest.raises(ValueError):
            await verify_payment(
                order_id="order_tampered",
                payment_id="pay_456",
                signature=sig,
                settings=billing_settings,
                user_service=mock_user_service,
                tier="pro",
                user_id="user-123",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Webhook Handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebhookHandling:

    def _make_webhook_sig(self, payload_bytes, secret=TEST_WEBHOOK_SECRET):
        return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

    @pytest.mark.asyncio
    async def test_webhook_payment_captured(self, billing_settings):
        from clsplusplus.razorpay_service import handle_webhook

        payload = json.dumps({
            "event": "payment.captured",
            "payload": {
                "payment": {
                    "entity": {
                        "notes": {"user_id": "user-789", "tier": "pro"},
                    }
                }
            }
        }).encode()

        sig = self._make_webhook_sig(payload)
        mock_user_service = AsyncMock()
        mock_user_service.update_tier = AsyncMock()

        await handle_webhook(payload, sig, billing_settings, mock_user_service)
        mock_user_service.update_tier.assert_called_once_with("user-789", "pro")

    @pytest.mark.asyncio
    async def test_webhook_payment_failed(self, billing_settings):
        from clsplusplus.razorpay_service import handle_webhook

        payload = json.dumps({
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "notes": {"user_id": "user-789"},
                        "error_description": "Insufficient funds",
                    }
                }
            }
        }).encode()

        sig = self._make_webhook_sig(payload)
        mock_user_service = AsyncMock()

        # Should not raise, just log
        await handle_webhook(payload, sig, billing_settings, mock_user_service)
        # Tier should NOT be updated on failure
        mock_user_service.update_tier.assert_not_called()

    @pytest.mark.asyncio
    async def test_webhook_invalid_signature(self, billing_settings):
        from clsplusplus.razorpay_service import handle_webhook

        payload = b'{"event":"payment.captured"}'
        mock_user_service = AsyncMock()

        with pytest.raises(ValueError, match="Invalid webhook signature"):
            await handle_webhook(payload, "bad-sig", billing_settings, mock_user_service)

    @pytest.mark.asyncio
    async def test_webhook_no_secret_configured(self, no_billing_settings):
        from clsplusplus.razorpay_service import handle_webhook

        with pytest.raises(ValueError, match="not configured"):
            await handle_webhook(b"{}", "sig", no_billing_settings, AsyncMock())


# ═══════════════════════════════════════════════════════════════════════════════
# Tier Pricing
# ═══════════════════════════════════════════════════════════════════════════════

class TestTierPricing:

    def test_tier_limits_defined(self):
        from clsplusplus.tiers import TIER_LIMITS
        for tier in ["free", "pro", "business", "enterprise"]:
            assert tier in TIER_LIMITS
            limits = TIER_LIMITS[tier]
            assert "ops_per_month" in limits
            assert "max_items" in limits
            assert "max_namespaces" in limits
            assert "rate_limit_requests" in limits

    def test_tier_ordering(self):
        """Higher tiers must have strictly more capacity."""
        from clsplusplus.tiers import TIER_LIMITS
        order = ["free", "pro", "business", "enterprise"]
        for i in range(len(order) - 1):
            lower = TIER_LIMITS[order[i]]
            higher = TIER_LIMITS[order[i + 1]]
            assert higher["ops_per_month"] > lower["ops_per_month"]
            assert higher["max_items"] > lower["max_items"]

    def test_free_tier_has_reasonable_limits(self):
        from clsplusplus.tiers import TIER_LIMITS
        free = TIER_LIMITS["free"]
        assert free["ops_per_month"] >= 100
        assert free["max_items"] >= 100
        assert free["rate_limit_requests"] >= 10
