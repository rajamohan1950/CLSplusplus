"""CLS++ API authentication - secure, constant-time validation."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Optional

from clsplusplus.config import Settings


# API key format: cls_live_* or cls_test_* (min 32 chars total for entropy)
_API_KEY_PATTERN = re.compile(r"^cls_(?:live|test)_[a-zA-Z0-9]{24,}$")


def _normalize_key(key: Optional[str]) -> Optional[str]:
    """Strip whitespace; return None if empty."""
    if key is None:
        return None
    k = key.strip()
    return k if k else None


def _sha256_hex(s: str) -> str:
    """SHA-256 hex digest of string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _get_key_lookup(settings: Settings) -> dict[str, str]:
    """Build hash -> key lookup for constant-time validation."""
    raw = getattr(settings, "api_keys", None) or ""
    if not raw:
        return {}
    lookup: dict[str, str] = {}
    for k in raw.split(","):
        k = _normalize_key(k)
        if k and _API_KEY_PATTERN.match(k):
            lookup[_sha256_hex(k)] = k
    return lookup


def validate_api_key(key: Optional[str], settings: Optional[Settings] = None) -> bool:
    """
    Validate API key using constant-time comparison.
    Hash lookup + single hmac.compare_digest prevents timing attacks.
    """
    if not key:
        return False
    key = _normalize_key(key)
    if not key or not _API_KEY_PATTERN.match(key):
        return False
    settings = settings or Settings()
    lookup = _get_key_lookup(settings)
    if not lookup:
        return False
    key_hash = _sha256_hex(key)
    stored = lookup.get(key_hash)
    return stored is not None and hmac.compare_digest(key, stored)


def extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    """Extract Bearer token from Authorization header. Returns None if invalid."""
    if not auth_header or not isinstance(auth_header, str):
        return None
    parts = auth_header.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1] if parts[1] else None


def get_api_key_from_request(auth_header: Optional[str]) -> Optional[str]:
    """Extract and return API key from request; None if missing/invalid format."""
    return extract_bearer_token(auth_header)
