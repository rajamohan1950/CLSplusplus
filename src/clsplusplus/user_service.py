"""User service — registration, login, Google OAuth, tier management, password reset, email verification."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import httpx

from clsplusplus.config import Settings
from clsplusplus.email_service import EmailService
from clsplusplus.jwt_utils import create_token
from clsplusplus.stores.user_store import UserStore

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def _strip_password(user: dict) -> dict:
    """Remove password_hash from user dict (safety net)."""
    user.pop("password_hash", None)
    return user


class UserService:
    """Business logic for user management."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = UserStore(settings)
        self.email = EmailService(settings)

    async def register(
        self, email: str, password: str, name: str = "", base_url: str = ""
    ) -> dict:
        """Step 1: Validate, send OTP, store pending registration.

        Does NOT create the user. Returns {pending: true, email: ...}.
        Raises ValueError on validation failure or duplicate email.
        """
        email = email.strip().lower()
        if not _EMAIL_RE.match(email):
            raise ValueError("Invalid email format")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        existing = await self.store.get_by_email(email)
        if existing:
            raise ValueError("An account with this email already exists")

        password_hash = _hash_password(password)
        otp_code = self._generate_otp()
        raw_token = secrets.token_urlsafe(48)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)

        await self.store.create_pending_registration(
            email=email,
            password_hash=password_hash,
            name=name or email.split("@")[0],
            otp_code=otp_code,
            token_hash=token_hash,
            expires_at=expires_at,
        )

        # Send verification email
        base = base_url.rstrip("/") if base_url else self.settings.site_base_url
        verify_link = f"{base}/v1/auth/verify-register?token={raw_token}"
        email_sent = False
        email_error = None
        try:
            email_sent = await self.email.send_verification_email(
                to=email, otp_code=otp_code, verify_link=verify_link,
            )
        except Exception as e:
            email_error = f"{type(e).__name__}: {e}"
            logger.warning("Verification email failed for %s: %s", email, e)

        result = {"pending": True, "email": email, "email_sent": email_sent}
        if not email_sent:
            result["email_configured"] = self.email._enabled
            if email_error:
                result["email_error"] = email_error
        return result

    async def complete_registration(self, email: str, otp_code: str) -> tuple[dict, str]:
        """Step 2: Verify OTP and create the actual user account.

        Returns (user_dict, jwt_token).
        Raises ValueError on invalid/expired OTP.
        """
        email = email.strip().lower()
        pending = await self.store.get_pending_by_otp(email, otp_code)
        if not pending:
            raise ValueError("Invalid or expired verification code")

        # Check email not taken (race condition guard)
        existing = await self.store.get_by_email(email)
        if existing:
            await self.store.delete_pending(email)
            raise ValueError("An account with this email already exists")

        # Create user
        user = await self.store.create_user(
            email=pending["email"],
            password_hash=pending["password_hash"],
            name=pending["name"],
        )
        # Mark as verified
        await self.store.mark_email_verified(user["id"])
        user["email_verified"] = True

        # Clean up pending
        await self.store.delete_pending(email)

        token = create_token(
            user_id=user["id"],
            email=user["email"],
            is_admin=user["is_admin"],
            secret=self.settings.jwt_secret,
        )
        return _strip_password(user), token

    async def complete_registration_link(self, raw_token: str) -> tuple[dict, str]:
        """Verify via magic link and create the actual user account.

        Returns (user_dict, jwt_token).
        Raises ValueError on invalid/expired token.
        """
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        pending = await self.store.get_pending_by_token(token_hash)
        if not pending:
            raise ValueError("Invalid or expired verification link")

        existing = await self.store.get_by_email(pending["email"])
        if existing:
            await self.store.delete_pending(pending["email"])
            raise ValueError("An account with this email already exists")

        user = await self.store.create_user(
            email=pending["email"],
            password_hash=pending["password_hash"],
            name=pending["name"],
        )
        await self.store.mark_email_verified(user["id"])
        user["email_verified"] = True
        await self.store.delete_pending(pending["email"])

        token = create_token(
            user_id=user["id"],
            email=user["email"],
            is_admin=user["is_admin"],
            secret=self.settings.jwt_secret,
        )
        return _strip_password(user), token

    async def login(self, email: str, password: str) -> tuple[dict, str]:
        """Authenticate with email+password.

        Returns (user_dict, jwt_token).
        Raises ValueError on bad credentials.
        """
        email = email.strip().lower()
        user = await self.store.get_by_email(email)
        if not user:
            raise ValueError("Invalid email or password")

        stored_hash = user.get("password_hash")
        if not stored_hash:
            if user.get("github_id") and not user.get("google_id"):
                raise ValueError("This account uses GitHub sign-in. Please use the GitHub button.")
            raise ValueError("This account uses Google sign-in. Please use the Google button.")

        if not _verify_password(password, stored_hash):
            raise ValueError("Invalid email or password")

        token = create_token(
            user_id=user["id"],
            email=user["email"],
            is_admin=user["is_admin"],
            secret=self.settings.jwt_secret,
        )
        return _strip_password(user), token

    async def google_auth(self, code: str, redirect_uri: str) -> tuple[dict, str]:
        """Exchange Google OAuth code for user session.

        Returns (user_dict, jwt_token).
        Raises ValueError on OAuth failure.
        """
        # Exchange authorization code for access token
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(GOOGLE_TOKEN_URL, data={
                "code": code,
                "client_id": self.settings.google_client_id,
                "client_secret": self.settings.google_client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            })
            if token_resp.status_code != 200:
                raise ValueError("Failed to exchange Google authorization code")
            token_data = token_resp.json()

            # Fetch user profile
            userinfo_resp = await client.get(GOOGLE_USERINFO_URL, headers={
                "Authorization": f"Bearer {token_data['access_token']}",
            })
            if userinfo_resp.status_code != 200:
                raise ValueError("Failed to fetch Google user profile")
            profile = userinfo_resp.json()

        google_id = profile["id"]
        email = profile["email"].strip().lower()
        name = profile.get("name", email.split("@")[0])
        avatar_url = profile.get("picture")

        # Find existing user by google_id or email
        user = await self.store.get_by_google_id(google_id)
        if not user:
            # Check if email exists (link Google to existing account)
            user = await self.store.get_by_email(email)
            if user:
                await self.store.update_google_id(user["id"], google_id, avatar_url)
                user["google_id"] = google_id
                user["avatar_url"] = avatar_url
                # Mark email verified (Google verified it)
                if not user.get("email_verified"):
                    await self.store.mark_email_verified(user["id"])
                    user["email_verified"] = True
            else:
                # Create new user via Google (auto-verified)
                user = await self.store.create_user(
                    email=email,
                    google_id=google_id,
                    name=name,
                    avatar_url=avatar_url,
                )
                # Auto-verify Google users
                await self.store.mark_email_verified(user["id"])
                user["email_verified"] = True

        token = create_token(
            user_id=user["id"],
            email=user["email"],
            is_admin=user["is_admin"],
            secret=self.settings.jwt_secret,
        )
        return _strip_password(user), token

    async def github_auth(self, code: str, redirect_uri: str) -> tuple[dict, str]:
        """Exchange GitHub OAuth code for a user session.

        Fetches the GitHub profile and (if the profile email is private or
        missing) the primary verified email from /user/emails. Links to an
        existing account by github_id first, then by verified email; otherwise
        creates a new auto-verified user.

        Returns (user_dict, jwt_token).
        Raises ValueError on OAuth failure (bad code, missing verified email).
        """
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self.settings.github_client_id,
                    "client_secret": self.settings.github_client_secret,
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )
            if token_resp.status_code != 200:
                raise ValueError("Failed to exchange GitHub authorization code")
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                # GitHub returns 200 with {"error": "..."} on bad code
                raise ValueError("GitHub did not return an access token")

            auth_headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            user_resp = await client.get(GITHUB_USER_URL, headers=auth_headers)
            if user_resp.status_code != 200:
                raise ValueError("Failed to fetch GitHub user profile")
            profile = user_resp.json()

            # GitHub may return null/empty email on the profile when private.
            # /user/emails requires the user:email scope and returns the full
            # list, from which we pick the primary verified one.
            profile_email = (profile.get("email") or "").strip().lower()
            email = profile_email
            if not email:
                emails_resp = await client.get(GITHUB_EMAILS_URL, headers=auth_headers)
                if emails_resp.status_code != 200:
                    raise ValueError("Failed to fetch GitHub user emails")
                emails = emails_resp.json() or []
                primary = next(
                    (e for e in emails if e.get("primary") and e.get("verified")),
                    None,
                )
                if not primary:
                    # Fall back to any verified email
                    primary = next(
                        (e for e in emails if e.get("verified")),
                        None,
                    )
                if not primary:
                    raise ValueError(
                        "No verified email on your GitHub account. "
                        "Verify an email in GitHub settings and try again."
                    )
                email = (primary.get("email") or "").strip().lower()

        if not email:
            raise ValueError("GitHub account has no usable email")

        github_id = str(profile["id"])
        name = (profile.get("name") or profile.get("login") or email.split("@")[0])
        avatar_url = profile.get("avatar_url")

        # Find existing user by github_id, then by email (link), else create.
        user = await self.store.get_by_github_id(github_id)
        if not user:
            user = await self.store.get_by_email(email)
            if user:
                await self.store.update_github_id(user["id"], github_id, avatar_url)
                user["github_id"] = github_id
                user["avatar_url"] = avatar_url
                if not user.get("email_verified"):
                    await self.store.mark_email_verified(user["id"])
                    user["email_verified"] = True
            else:
                user = await self.store.create_user(
                    email=email,
                    github_id=github_id,
                    name=name,
                    avatar_url=avatar_url,
                )
                await self.store.mark_email_verified(user["id"])
                user["email_verified"] = True

        token = create_token(
            user_id=user["id"],
            email=user["email"],
            is_admin=user["is_admin"],
            secret=self.settings.jwt_secret,
        )
        return _strip_password(user), token

    async def get_user(self, user_id: str) -> Optional[dict]:
        user = await self.store.get_by_id(user_id)
        if user:
            return _strip_password(user)
        return None

    async def update_profile(
        self,
        user_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        current_password: Optional[str] = None,
    ) -> dict:
        """Update user profile fields.

        Returns updated user dict.
        Raises ValueError on validation failure.
        """
        user = await self.store.get_by_email(
            (await self.store.get_by_id(user_id) or {}).get("email", "")
        )
        if not user:
            raise ValueError("User not found")

        fields: dict = {}

        if name is not None:
            fields["name"] = name.strip()

        if email is not None:
            email = email.strip().lower()
            if not _EMAIL_RE.match(email):
                raise ValueError("Invalid email format")
            if email != user["email"]:
                existing = await self.store.get_by_email(email)
                if existing:
                    raise ValueError("An account with this email already exists")
                fields["email"] = email

        if password is not None:
            if not current_password:
                raise ValueError("Current password is required to set a new password")
            stored_hash = user.get("password_hash")
            if not stored_hash:
                raise ValueError("Cannot change password for Google-only accounts")
            if not _verify_password(current_password, stored_hash):
                raise ValueError("Current password is incorrect")
            fields["password_hash"] = _hash_password(password)

        if not fields:
            return _strip_password(
                await self.store.get_by_id(user_id) or {}
            )

        updated = await self.store.update_user(user_id, fields)
        if not updated:
            raise ValueError("User not found")
        return _strip_password(updated)

    async def update_tier(self, user_id: str, new_tier: str) -> dict:
        """Change user tier and record revenue event.

        Returns updated user dict.
        Raises ValueError on invalid tier.
        """
        from clsplusplus.tiers import Tier, TIER_PRICES

        if new_tier not in [t.value for t in Tier]:
            raise ValueError(f"Invalid tier: {new_tier}")

        user = await self.store.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        old_tier = user["tier"]
        if old_tier == new_tier:
            # Idempotent — already upgraded (e.g., webhook beat the verify call)
            return _strip_password(user)

        await self.store.update_tier(user_id, new_tier)

        # Record revenue event
        new_price = TIER_PRICES[Tier(new_tier)]
        old_price = TIER_PRICES[Tier(old_tier)]
        event_type = "upgrade" if new_price > old_price else "downgrade"
        await self.store.record_revenue_event(
            user_id=user_id,
            event_type=event_type,
            from_tier=old_tier,
            to_tier=new_tier,
            monthly_revenue=new_price,
        )

        user["tier"] = new_tier
        return _strip_password(user)

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return await self.store.list_users(limit, offset)

    # =========================================================================
    # Password reset
    # =========================================================================

    async def request_password_reset(
        self, email: str, base_url: str = ""
    ) -> Optional[str]:
        """Generate a password reset token for the given email.

        Sends reset email via Resend. Returns the raw token string
        (also sent via email), or None if email not found / Google-only.
        """
        email = email.strip().lower()
        user = await self.store.get_by_email(email)
        if not user:
            return None
        if not user.get("password_hash"):
            return None  # Google-only account

        otp_code = self._generate_otp()
        raw_token = secrets.token_urlsafe(48)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        await self.store.create_reset_token(
            user_id=user["id"],
            token_hash=token_hash,
            expires_at=expires_at,
        )

        # Send email with OTP + reset link
        base = base_url.rstrip("/") if base_url else self.settings.site_base_url
        reset_link = f"{base}/login.html?reset_token={raw_token}"
        try:
            await self.email.send_password_reset_email(
                to=email,
                otp_code=otp_code,
                reset_link=reset_link,
            )
        except Exception as e:
            logger.warning("Reset email failed for %s (non-fatal): %s", email, e)

        return raw_token

    async def reset_password(self, token: str, new_password: str) -> dict:
        """Reset a user's password using a valid reset token.

        Returns updated user dict.
        Raises ValueError on invalid/expired token or bad password.
        """
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters")

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        token_record = await self.store.get_reset_token(token_hash)
        if not token_record:
            raise ValueError("Invalid or expired reset token")

        user_id = token_record["user_id"]
        new_hash = _hash_password(new_password)
        updated = await self.store.update_user(user_id, {"password_hash": new_hash})
        if not updated:
            raise ValueError("User not found")

        await self.store.mark_token_used(token_record["id"])
        return _strip_password(updated)

    # =========================================================================
    # Admin seeding
    # =========================================================================

    async def ensure_admin(self) -> None:
        """Create the default admin user if it doesn't exist."""
        admin_email = "admin@clsplusplus.com"
        admin_password = "admin123"
        admin_hash = _hash_password(admin_password)

        result = await self.store.ensure_admin_user(
            email=admin_email,
            password_hash=admin_hash,
            name="admin",
        )
        if result:
            logger.info("Default admin user created: %s", admin_email)
            # Assign super_admin role via RBAC
            try:
                from clsplusplus.rbac_service import RBACService
                from clsplusplus.stores.rbac_store import RBACStore
                rbac_store = RBACStore(self.settings)
                pool = await self.store.get_pool()
                async with pool.acquire() as conn:
                    role = await conn.fetchrow(
                        "SELECT id FROM roles WHERE name = 'super_admin'"
                    )
                    if role:
                        await conn.execute(
                            """
                            INSERT INTO user_roles (user_id, role_id)
                            VALUES ($1, $2)
                            ON CONFLICT DO NOTHING
                            """,
                            result["id"], str(role["id"]),
                        )
                        logger.info("Assigned super_admin role to admin user")
            except Exception as e:
                logger.warning("Could not assign RBAC role to admin (non-fatal): %s", e)
        else:
            logger.debug("Admin user already exists: %s", admin_email)

    # =========================================================================
    # Email verification
    # =========================================================================

    def _generate_otp(self) -> str:
        """Generate a 6-digit OTP code."""
        return f"{random.randint(100000, 999999)}"

    async def _send_verification_email(
        self, user_id: str, email: str, base_url: str = ""
    ) -> None:
        """Generate OTP + magic link token, store, and send email."""
        otp_code = self._generate_otp()
        raw_token = secrets.token_urlsafe(48)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)

        await self.store.create_verification_token(
            user_id=user_id,
            otp_code=otp_code,
            token_hash=token_hash,
            expires_at=expires_at,
        )

        base = base_url.rstrip("/") if base_url else self.settings.site_base_url
        verify_link = f"{base}/v1/auth/verify-email?token={raw_token}"

        await self.email.send_verification_email(
            to=email,
            otp_code=otp_code,
            verify_link=verify_link,
        )

    async def verify_email_otp(self, user_id: str, otp_code: str) -> bool:
        """Verify email using 6-digit OTP code.

        Returns True if verified.
        Raises ValueError on invalid/expired OTP.
        """
        record = await self.store.get_verification_by_otp(user_id, otp_code)
        if not record:
            raise ValueError("Invalid or expired verification code")

        await self.store.mark_verification_used(record["id"])
        await self.store.mark_email_verified(user_id)
        return True

    async def verify_email_link(self, token: str) -> str:
        """Verify email using magic link token.

        Returns user_id if verified.
        Raises ValueError on invalid/expired token.
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        record = await self.store.get_verification_by_token(token_hash)
        if not record:
            raise ValueError("Invalid or expired verification link")

        await self.store.mark_verification_used(record["id"])
        await self.store.mark_email_verified(record["user_id"])
        return record["user_id"]

    async def resend_verification(self, user_id: str, base_url: str = "") -> None:
        """Resend verification email for an unverified user.

        Raises ValueError if already verified.
        """
        user = await self.store.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        if user.get("email_verified"):
            raise ValueError("Email already verified")
        await self._send_verification_email(user_id, user["email"], base_url)
