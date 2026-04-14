"""Email service — sends verification and password reset emails via Resend API."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"


class EmailService:
    """Send transactional emails via Resend."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def _enabled(self) -> bool:
        return bool(self.settings.resend_api_key)

    async def _send(self, to: str, subject: str, html: str) -> bool:
        """Send an email via Resend API. Returns True on success, raises on failure."""
        if not self._enabled:
            raise RuntimeError("Resend not configured (CLS_RESEND_API_KEY not set)")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                RESEND_API_URL,
                headers={
                    "Authorization": f"Bearer {self.settings.resend_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": self.settings.email_from,
                    "to": [to],
                    "subject": subject,
                    "html": html,
                },
                timeout=15.0,
            )
            if resp.status_code in (200, 201):
                logger.info("Email sent to %s: %s", to, subject)
                return True
            else:
                error_body = resp.text
                logger.error("Resend API error %d: %s", resp.status_code, error_body)
                raise RuntimeError(f"Resend API {resp.status_code}: {error_body}")

    async def send_verification_email(
        self, to: str, otp_code: str, verify_link: str
    ) -> bool:
        """Send email verification with 6-digit OTP and magic link."""
        html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:'Inter',-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;background:#fafafa;padding:40px 20px;">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:16px;padding:48px 40px;box-shadow:0 2px 20px rgba(0,0,0,0.06);">
    <div style="text-align:center;margin-bottom:32px;">
      <span style="font-size:24px;font-weight:700;color:#1d1d1f;">CLS</span><span style="font-size:24px;font-weight:700;color:#ff6b35;">++</span>
    </div>
    <h1 style="font-size:22px;font-weight:600;color:#1d1d1f;text-align:center;margin-bottom:8px;">Verify your email</h1>
    <p style="color:#86868b;text-align:center;font-size:15px;margin-bottom:32px;">Enter this code to complete your registration.</p>
    <div style="background:#f5f5f7;border-radius:12px;padding:24px;text-align:center;margin-bottom:24px;">
      <span style="font-size:36px;font-weight:700;letter-spacing:8px;color:#1d1d1f;">{otp_code}</span>
    </div>
    <p style="color:#86868b;text-align:center;font-size:13px;margin-bottom:24px;">This code expires in 15 minutes.</p>
    <div style="text-align:center;margin-bottom:32px;">
      <a href="{verify_link}" style="display:inline-block;background:#ff6b35;color:#fff;padding:14px 32px;border-radius:980px;text-decoration:none;font-weight:600;font-size:15px;">Verify Email</a>
    </div>
    <p style="color:#86868b;text-align:center;font-size:12px;">Or click the button above to verify instantly.</p>
    <hr style="border:none;border-top:1px solid #f0f0f0;margin:32px 0 16px;">
    <p style="color:#c0c0c0;text-align:center;font-size:11px;">If you didn't create a CLS++ account, you can safely ignore this email.</p>
  </div>
</body>
</html>"""
        return await self._send(to, "Verify your CLS++ email", html)

    async def send_password_reset_email(
        self, to: str, otp_code: str, reset_link: str
    ) -> bool:
        """Send password reset email with token."""
        html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:'Inter',-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;background:#fafafa;padding:40px 20px;">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:16px;padding:48px 40px;box-shadow:0 2px 20px rgba(0,0,0,0.06);">
    <div style="text-align:center;margin-bottom:32px;">
      <span style="font-size:24px;font-weight:700;color:#1d1d1f;">CLS</span><span style="font-size:24px;font-weight:700;color:#ff6b35;">++</span>
    </div>
    <h1 style="font-size:22px;font-weight:600;color:#1d1d1f;text-align:center;margin-bottom:8px;">Reset your password</h1>
    <p style="color:#86868b;text-align:center;font-size:15px;margin-bottom:32px;">Use this code to reset your password.</p>
    <div style="background:#f5f5f7;border-radius:12px;padding:24px;text-align:center;margin-bottom:24px;">
      <span style="font-size:36px;font-weight:700;letter-spacing:8px;color:#1d1d1f;">{otp_code}</span>
    </div>
    <p style="color:#86868b;text-align:center;font-size:13px;margin-bottom:24px;">This code expires in 1 hour.</p>
    <div style="text-align:center;margin-bottom:32px;">
      <a href="{reset_link}" style="display:inline-block;background:#ff6b35;color:#fff;padding:14px 32px;border-radius:980px;text-decoration:none;font-weight:600;font-size:15px;">Reset Password</a>
    </div>
    <hr style="border:none;border-top:1px solid #f0f0f0;margin:32px 0 16px;">
    <p style="color:#c0c0c0;text-align:center;font-size:11px;">If you didn't request a password reset, you can safely ignore this email.</p>
  </div>
</body>
</html>"""
        return await self._send(to, "Reset your CLS++ password", html)
