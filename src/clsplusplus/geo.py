"""Geo-resolution + launch-region gating.

The production launch is INDIA-ONLY for active accounts. Visitors from
outside India must still be able to register interest into the waitlist
queue, but they get no API key / no active account.

This module is the single chokepoint for "what country is this request
from" and "is this country allowed an active account". Keeping the logic
here (not in api.py) keeps the api.py hooks to a single localized call,
which matters because other agents are concurrently editing api.py.

Resolution order in `resolve_country`:
  (a) Cloudflare's `CF-IPCountry` request header, if present.
  (b) Fallback — extract the client IP (first hop of `X-Forwarded-For`,
      else `request.client.host`) and call a free GeoIP HTTP API.
  (c) On any failure, or a private/localhost IP, return None.

FAIL OPEN: when `resolve_country` returns None (unknown), `is_region_allowed`
treats the request as allowed. A GeoIP outage must never block Indian users.
"""

from __future__ import annotations

import ipaddress
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# GeoIP fallback endpoint. Returns a bare ISO-2 code as plain text body.
_GEOIP_URL = "https://ipapi.co/{ip}/country/"
_GEOIP_TIMEOUT_SECONDS = 2.0

# Per-IP in-memory cache so a burst of signups from one network does not
# hammer the free GeoIP API. Maps ip -> (country_or_None, expires_epoch).
_CACHE_TTL_SECONDS = 3600.0
_country_cache: dict[str, tuple[Optional[str], float]] = {}


def _client_ip(request) -> Optional[str]:
    """Best-effort client IP. Honors X-Forwarded-For first hop."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    client = getattr(request, "client", None)
    if client and getattr(client, "host", None):
        return client.host
    return None


def _is_public_ip(ip: str) -> bool:
    """True only for routable, non-private, non-loopback addresses."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _normalize_country(raw: Optional[str]) -> Optional[str]:
    """Return an upper-case ISO-2 code, or None if the value isn't one."""
    if not raw:
        return None
    code = raw.strip().upper()
    if len(code) == 2 and code.isalpha():
        return code
    return None


async def _geoip_lookup(ip: str) -> Optional[str]:
    """Single GeoIP HTTP lookup, behind a circuit breaker + retry.

    The breaker means that once ipapi.co is down, requests stop trying
    (and stop waiting the 2s timeout) until it recovers — they just fail
    open instantly. Retry smooths transient blips.
    """
    from clsplusplus.resilience import (
        get_breaker, guarded_call, http_timeout,
    )

    async def _do_lookup() -> Optional[str]:
        async with httpx.AsyncClient(
            timeout=http_timeout(connect=2.0, read=_GEOIP_TIMEOUT_SECONDS)
        ) as client:
            resp = await client.get(_GEOIP_URL.format(ip=ip))
        resp.raise_for_status()
        return _normalize_country(resp.text)

    breaker = get_breaker("geoip", failure_threshold=5, recovery_seconds=30.0)
    return await guarded_call(
        breaker, _do_lookup, attempts=2, base_delay=0.2, max_delay=1.0,
    )


async def resolve_country(request) -> Optional[str]:
    """Resolve the request's origin country as an ISO-2 code, else None.

    None means "unknown" — callers must fail open and treat it as allowed.
    """
    # (a) Cloudflare edge header — authoritative and free when present.
    cf = _normalize_country(request.headers.get("cf-ipcountry"))
    if cf:
        # CF uses "XX" for unknown / Tor; treat that as unresolved.
        return cf if cf != "XX" else None

    # (b) GeoIP fallback keyed on the client IP.
    ip = _client_ip(request)
    if not ip or not _is_public_ip(ip):
        return None

    now = time.monotonic()
    cached = _country_cache.get(ip)
    if cached and cached[1] > now:
        return cached[0]

    country: Optional[str] = None
    try:
        country = await _geoip_lookup(ip)
    except Exception as e:  # noqa: BLE001 — any GeoIP failure must fail open.
        logger.warning("GeoIP lookup failed for %s: %s", ip, e)
        country = None

    _country_cache[ip] = (country, now + _CACHE_TTL_SECONDS)
    return country


def is_region_allowed(country: Optional[str], settings) -> bool:
    """True if a request from `country` may get an active account.

    Fail-open contract:
      - gating disabled            -> allowed
      - country unknown (None)     -> allowed
      - country == launch_country  -> allowed
      - otherwise                  -> NOT allowed (route to waitlist)
    """
    if not getattr(settings, "geo_gating_enabled", True):
        return True
    if country is None:
        return True
    launch = (getattr(settings, "launch_country", "IN") or "IN").upper()
    return country.upper() == launch


# Stable, user-facing payload for an out-of-region signup. Returned with
# HTTP 200 — the caller is on the list, this is not an error.
REGION_QUEUED_RESPONSE = {
    "status": "queued_region",
    "message": (
        "CLS++ isn't open in your region yet — you're on the list, "
        "we'll email you."
    ),
}


async def queue_out_of_region(waitlist_service, email: str, source: str = "geo_gate") -> dict:
    """Route an out-of-region signup into the existing waitlist queue.

    The visitor has already verified intent by submitting a signup form, so
    we drop them straight into the waiting queue rather than sending another
    OTP. Best-effort: a waitlist failure must not turn into a 500 for the
    user — they still get the queued_region response.
    """
    try:
        await waitlist_service.store.create_visitor(email, source)
    except Exception as e:  # noqa: BLE001
        logger.warning("Out-of-region waitlist enqueue failed for %s: %s", email, e)
    return dict(REGION_QUEUED_RESPONSE)
