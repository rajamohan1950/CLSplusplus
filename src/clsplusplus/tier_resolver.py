"""Shared api_key → owner lookups with a short in-memory cache.

Used by:

  * `QuotaMiddleware` — to enforce the *per-user* monthly quota and to
    resolve the billing subject every counter is keyed on.
  * `MeteringPricer` — to know whether to charge an overage rate.

One resolver instance per process. Cache TTL is 5 minutes; the
`/v1/user/upgrade` handler calls `invalidate()` so a paid upgrade
takes effect immediately. Resolution errors return `None` and the
caller decides what to do (Quota: fall back to free / per-key subject;
Pricer: return 0 cents).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TierResolver:
    """Cached api_key → (tier, owner_email) lookup.

    `integration_store` must expose `resolve_tier_from_key(raw_key)` and
    `resolve_owner_email_from_key(raw_key)`, each returning `Optional[str]`.
    """

    def __init__(self, integration_store: Any, cache_ttl_seconds: int = 300):
        self._store = integration_store
        self._ttl = cache_ttl_seconds
        # api_key → (value, expires_at_monotonic)
        self._tier_cache: dict[str, tuple[Optional[str], float]] = {}
        # api_key → ((tier, subscription_expires_at), expires_at_monotonic)
        self._effective_cache: dict[str, tuple[tuple, float]] = {}
        self._owner_cache: dict[str, tuple[Optional[str], float]] = {}

    def invalidate(self, api_key: Optional[str] = None) -> None:
        """Drop the cache. Call on tier change so the next call re-resolves."""
        if api_key is None:
            self._tier_cache.clear()
            self._effective_cache.clear()
            self._owner_cache.clear()
        else:
            self._tier_cache.pop(api_key, None)
            self._effective_cache.pop(api_key, None)
            self._owner_cache.pop(api_key, None)

    async def resolve(self, api_key: str) -> Optional[str]:
        """Return the owning user's tier, or None when unresolvable."""
        if not api_key:
            return None
        now = time.monotonic()
        cached = self._tier_cache.get(api_key)
        if cached and cached[1] > now:
            return cached[0]
        try:
            tier = await self._store.resolve_tier_from_key(api_key)
        except Exception as exc:
            logger.warning(
                "TierResolver: DB error resolving tier: %s: %s",
                type(exc).__name__, exc,
            )
            return None
        self._tier_cache[api_key] = (tier, now + self._ttl)
        return tier

    async def resolve_effective(self, api_key: str):
        """Return (tier, subscription_expires_at) for the owning user.

        The quota path needs the trial-window expiry to pick the launch vs
        post-trial free cap. Cached under the same TTL as the bare-tier
        cache. Returns (None, None) on unresolvable / DB error. Falls back
        to `resolve()` (expiry None) when the store predates the
        expiry-aware lookup.
        """
        if not api_key:
            return (None, None)
        now = time.monotonic()
        cached = self._effective_cache.get(api_key)
        if cached and cached[1] > now:
            return cached[0]
        resolver = getattr(self._store, "resolve_tier_and_expiry_from_key", None)
        if resolver is None:
            return (await self.resolve(api_key), None)
        try:
            value = await resolver(api_key)
        except Exception as exc:
            logger.warning(
                "TierResolver: DB error resolving tier: %s: %s",
                type(exc).__name__, exc,
            )
            return (None, None)
        self._effective_cache[api_key] = (value, now + self._ttl)
        return value

    async def resolve_owner(self, api_key: str) -> Optional[str]:
        """Return the owning user's email, or None for legacy/unknown keys."""
        if not api_key:
            return None
        now = time.monotonic()
        cached = self._owner_cache.get(api_key)
        if cached and cached[1] > now:
            return cached[0]
        try:
            owner = await self._store.resolve_owner_email_from_key(api_key)
        except Exception as exc:
            logger.warning(
                "TierResolver: DB error resolving owner: %s: %s",
                type(exc).__name__, exc,
            )
            return None
        self._owner_cache[api_key] = (owner, now + self._ttl)
        return owner

    async def resolve_subject(self, api_key: str) -> str:
        """Return the billing subject used to key usage counters.

        `owner:{hash}` when the key resolves to a user; `key:{hash}` for
        legacy keys with no resolvable owner.
        """
        from clsplusplus.usage import make_subject
        owner = await self.resolve_owner(api_key)
        return make_subject(owner, api_key)
