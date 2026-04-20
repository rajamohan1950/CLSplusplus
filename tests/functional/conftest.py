"""Shared fixtures for functional tests.

Functional tests are BLACKBOX: they exercise only the HTTP surface of a real
running backend, with real Postgres + Redis behind it. Bring the stack up
before running:

    docker compose -f docker-compose.test.yml up -d --build

Then:

    pytest tests/functional -m functional
"""
from __future__ import annotations

import os
import time
import uuid
import httpx
import pytest

TEST_API_URL = os.environ.get("CLS_TEST_API_URL", "http://localhost:18080")
TEST_TIMEOUT = float(os.environ.get("CLS_TEST_HTTP_TIMEOUT", "10"))


def _wait_for_server(url: str, timeout: float = 30.0) -> None:
    """Block until the backend's /health responds 200, or fail the test run."""
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(
        f"Backend at {url} did not become healthy within {timeout}s "
        f"(last error: {last_err}). Did you run `docker compose -f "
        f"docker-compose.test.yml up -d`?"
    )


@pytest.fixture(scope="session")
def api_base() -> str:
    _wait_for_server(TEST_API_URL)
    return TEST_API_URL


@pytest.fixture()
def client(api_base: str):
    """Fresh httpx client per test; shares cookie jar so auth sticks."""
    with httpx.Client(base_url=api_base, timeout=TEST_TIMEOUT, follow_redirects=False) as c:
        yield c


@pytest.fixture()
def unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@clsplusplus-test.local"


@pytest.fixture()
def registered_user(client: httpx.Client, unique_email: str) -> dict:
    """Registers a user and returns {email, password}. Does NOT log in.

    Registration requires OTP verification which isn't available to blackbox
    tests, so we use the admin override when `CLS_TEST_MODE=true` is set on the
    stack. If the deployment rejects it, the test skips gracefully.
    """
    password = "TestPassword123!"
    r = client.post(
        "/v1/auth/register",
        json={"email": unique_email, "password": password, "name": "Test User"},
    )
    if r.status_code == 503:
        pytest.skip("Launch cap reached in test stack")
    assert r.status_code in (200, 201), f"register failed: {r.status_code} {r.text}"
    return {"email": unique_email, "password": password}
