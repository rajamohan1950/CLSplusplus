"""User service — registration, login, Google OAuth, tier management."""

from __future__ import annotations

import logging
import re
from typing import Optional

import bcrypt
import httpx

from clsplusplus.config import Settings
from clsplusplus.jwt_utils import create_token
from clsplusplus.stores.user_store import UserStore

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


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

    async def register(self, email: str, password: str, name: str = "") -> tuple[dict, str]:
        """Register a new user with email+password.

        Returns (user_dict, jwt_token).
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
        user = await self.store.create_user(
            email=email,
            password_hash=password_hash,
            name=name or email.split("@")[0],
        )
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
            else:
                # Create new user via Google
                user = await self.store.create_user(
                    email=email,
                    google_id=google_id,
                    name=name,
                    avatar_url=avatar_url,
                )

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
            raise ValueError("Already on this tier")

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
