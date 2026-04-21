"""Shared api_key → owner_tier lookup with a short in-memory cache.

Used by:

  * `QuotaMiddleware` — to enforce the *per-user* monthly quota (not a
    server-wide default as it used to).
  * `MeteringPricer` — to know whether to charge an overage rate.

One resolver instance per process. Cache TTL is 5 minutes; the
`/v1/user/upgrade` handler calls `invalidate()` so a paid upgrade
takes effect immediately. Tier-resolution errors return `None` and
the caller decides what to do (Quota: fall back to free; Pricer:
return 0 cents).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TierResolver:
    """Cached api_key → tier string lookup.

    `integration_store` must expose `resolve_tier_from_key(raw_key) -> Optional[str]`.
    """

    def __init__(self, integration_store: Any, cache_ttl_seconds: int = 300):
        self._store = integration_store
        self._ttl = cache_ttl_seconds
        # api_key → (tier, expires_at_monotonic)
        self._cache: dict[str, tuple[Optional[str], float]] = {}

    def invalidate(self, api_key: Optional[str] = None) -> None:
        """Drop the cache. Call on tier change so the next call re-resolves."""
        if api_key is None:
            self._cache.clear()
        else:
            self._cache.pop(api_key, None)

    async def resolve(self, api_key: str) -> Optional[str]:
        if not api_key:
            return None
        now = time.monotonic()
        cached = self._cache.get(api_key)
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
        self._cache[api_key] = (tier, now + self._ttl)
        return tier
