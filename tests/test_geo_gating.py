"""Tests for the India-only launch region gate (clsplusplus/geo.py).

Scope: the geo module is the single chokepoint for launch gating, so the
contract that matters is tested here directly — country resolution
(CF header / GeoIP fallback / private-IP), the fail-open gating decision,
and the out-of-region waitlist enqueue. The GeoIP HTTP call and the
waitlist store are mocked; no network, no DB.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clsplusplus import geo
from clsplusplus.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request(headers=None, client_host="8.8.8.8"):
    """Minimal Request-like stub: headers (case-insensitive get) + client."""
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    return SimpleNamespace(
        headers=SimpleNamespace(get=lambda k, d=None: hdrs.get(k.lower(), d)),
        client=SimpleNamespace(host=client_host) if client_host else None,
    )


def _httpx_resp(status_code=200, text="IN"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


def _patch_httpx(resp=None, raises=None):
    """Patch httpx.AsyncClient used inside geo.resolve_country."""
    client = MagicMock()
    if raises is not None:
        client.get = AsyncMock(side_effect=raises)
    else:
        client.get = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch.object(geo.httpx, "AsyncClient", return_value=ctx)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Per-IP cache must not leak between tests."""
    geo._country_cache.clear()
    yield
    geo._country_cache.clear()


# ---------------------------------------------------------------------------
# resolve_country — Cloudflare header path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cf_header_takes_precedence():
    """CF-IPCountry is authoritative — no GeoIP call should happen."""
    req = _request(headers={"CF-IPCountry": "in"})
    with _patch_httpx(_httpx_resp(text="US")) as m:
        country = await geo.resolve_country(req)
    assert country == "IN"
    m.return_value.__aenter__.return_value.get.assert_not_called()


@pytest.mark.asyncio
async def test_cf_header_xx_is_unknown():
    """CF returns 'XX' for Tor/unknown — treat as unresolved (None)."""
    req = _request(headers={"CF-IPCountry": "XX"}, client_host="10.0.0.1")
    assert await geo.resolve_country(req) is None


# ---------------------------------------------------------------------------
# resolve_country — GeoIP fallback path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_geoip_fallback_resolves_country():
    req = _request(headers={}, client_host="8.8.8.8")
    with _patch_httpx(_httpx_resp(text="IN\n")):
        assert await geo.resolve_country(req) == "IN"


@pytest.mark.asyncio
async def test_xff_first_hop_is_used():
    """The client IP comes from the first hop of X-Forwarded-For."""
    req = _request(
        headers={"X-Forwarded-For": "1.1.1.1, 70.0.0.1"},
        client_host="10.0.0.1",
    )
    with _patch_httpx(_httpx_resp(text="US")) as m:
        country = await geo.resolve_country(req)
    assert country == "US"
    called_url = m.return_value.__aenter__.return_value.get.call_args[0][0]
    assert "1.1.1.1" in called_url


@pytest.mark.asyncio
async def test_private_ip_returns_none_without_geoip_call():
    req = _request(headers={}, client_host="192.168.1.50")
    with _patch_httpx(_httpx_resp(text="IN")) as m:
        country = await geo.resolve_country(req)
    assert country is None
    m.return_value.__aenter__.return_value.get.assert_not_called()


@pytest.mark.asyncio
async def test_loopback_ip_returns_none():
    req = _request(headers={}, client_host="127.0.0.1")
    assert await geo.resolve_country(req) is None


@pytest.mark.asyncio
async def test_geoip_http_error_returns_none():
    """GeoIP outage must fail soft — return None (caller fails open)."""
    req = _request(headers={}, client_host="8.8.8.8")
    with _patch_httpx(raises=RuntimeError("connection refused")):
        assert await geo.resolve_country(req) is None


@pytest.mark.asyncio
async def test_geoip_non_200_returns_none():
    req = _request(headers={}, client_host="8.8.8.8")
    with _patch_httpx(_httpx_resp(status_code=429, text="rate limited")):
        assert await geo.resolve_country(req) is None


@pytest.mark.asyncio
async def test_geoip_garbage_body_returns_none():
    """A non-ISO-2 body (HTML error page etc.) is not a country."""
    req = _request(headers={}, client_host="8.8.8.8")
    with _patch_httpx(_httpx_resp(text="<html>error</html>")):
        assert await geo.resolve_country(req) is None


@pytest.mark.asyncio
async def test_geoip_result_is_cached_per_ip():
    """A second lookup for the same IP must not hit the GeoIP API again."""
    req = _request(headers={}, client_host="8.8.8.8")
    with _patch_httpx(_httpx_resp(text="IN")) as m:
        assert await geo.resolve_country(req) == "IN"
        assert await geo.resolve_country(req) == "IN"
    assert m.return_value.__aenter__.return_value.get.call_count == 1


# ---------------------------------------------------------------------------
# is_region_allowed — the fail-open gating decision
# ---------------------------------------------------------------------------

def test_allowed_when_country_matches_launch():
    s = Settings(geo_gating_enabled=True, launch_country="IN")
    assert geo.is_region_allowed("IN", s) is True


def test_blocked_when_country_differs():
    s = Settings(geo_gating_enabled=True, launch_country="IN")
    assert geo.is_region_allowed("US", s) is False


def test_unknown_country_fails_open():
    """None (GeoIP outage / private IP) must be treated as ALLOWED."""
    s = Settings(geo_gating_enabled=True, launch_country="IN")
    assert geo.is_region_allowed(None, s) is True


def test_gating_disabled_allows_everyone():
    s = Settings(geo_gating_enabled=False, launch_country="IN")
    assert geo.is_region_allowed("US", s) is True


def test_country_comparison_is_case_insensitive():
    s = Settings(geo_gating_enabled=True, launch_country="IN")
    assert geo.is_region_allowed("in", s) is True


# ---------------------------------------------------------------------------
# queue_out_of_region — routes the email into the existing waitlist queue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queue_out_of_region_enqueues_and_returns_queued_status():
    waitlist = MagicMock()
    waitlist.store = MagicMock()
    waitlist.store.create_visitor = AsyncMock(return_value={"status": "waiting"})

    result = await geo.queue_out_of_region(waitlist, "user@example.com")

    waitlist.store.create_visitor.assert_awaited_once()
    assert waitlist.store.create_visitor.await_args[0][0] == "user@example.com"
    assert result["status"] == "queued_region"
    assert "region" in result["message"].lower()


@pytest.mark.asyncio
async def test_queue_out_of_region_survives_waitlist_failure():
    """A waitlist DB failure must not turn into a 500 — still return queued."""
    waitlist = MagicMock()
    waitlist.store = MagicMock()
    waitlist.store.create_visitor = AsyncMock(side_effect=RuntimeError("db down"))

    result = await geo.queue_out_of_region(waitlist, "user@example.com")
    assert result["status"] == "queued_region"
