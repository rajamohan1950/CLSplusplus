"""Smoke tests: seconds-level liveness checks.

Run against any environment (local stack, staging, prod) by setting
``CLS_TEST_API_URL``. Every smoke test is blackbox, sub-second, and must pass
with *zero* prior state.
"""
from __future__ import annotations

import os
import httpx
import pytest


API_URL = os.environ.get("CLS_TEST_API_URL", "http://localhost:18080")

pytestmark = [pytest.mark.smoke, pytest.mark.blackbox]


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=API_URL, timeout=5.0) as c:
        yield c


def test_root_returns_api_info(client: httpx.Client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    # Post-migration, / must describe the API, not serve an HTML page.
    assert data.get("name", "").startswith("CLS++"), data
    assert "health" in data


def test_health_route_ok(client: httpx.Client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_openapi_json_loads(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "paths" in r.json()


def test_docs_page_loads(client: httpx.Client) -> None:
    r = client.get("/docs")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_waitlist_stats_public(client: httpx.Client) -> None:
    """Waitlist stats must be callable without auth — the landing page uses it."""
    r = client.get("/v1/waitlist/stats")
    assert r.status_code == 200
    data = r.json()
    assert "waiting_count" in data or "count" in data


def test_auth_me_unauthenticated_returns_401(client: httpx.Client) -> None:
    """Sanity: unauthenticated profile fetch must not leak data."""
    r = client.get("/v1/auth/me")
    assert r.status_code in (401, 403)


def test_static_mount_removed(client: httpx.Client) -> None:
    """Regression: the old /index.html static file must no longer be served."""
    r = client.get("/index.html")
    # 404 or similar; what we MUST NOT see is a 200 with <html>…<title>CLS++…
    if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
        body = r.text.lower()
        assert "<title>" not in body or "cls++" not in body, (
            "/index.html still being served statically"
        )
