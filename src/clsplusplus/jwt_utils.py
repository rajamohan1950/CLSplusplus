"""JWT token utilities for CLS++ user sessions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from starlette.requests import Request


def create_token(
    user_id: str,
    email: str,
    is_admin: bool,
    secret: str,
    expiry_days: int = 7,
) -> str:
    """Create a signed JWT token."""
    payload = {
        "sub": user_id,
        "email": email,
        "is_admin": is_admin,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=expiry_days),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> dict | None:
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_token_from_cookie(request: Request) -> str | None:
    """Extract the cls_session cookie value from a request."""
    return request.cookies.get("cls_session")
