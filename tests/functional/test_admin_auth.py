"""Admin routes must reject API-key callers.

API keys are not admin identities — admin endpoints require a JWT session with
the admin flag. A valid API key must never reach an /admin/* route. This is
enforced at the auth middleware chokepoint so it holds for every admin route
regardless of whether the individual handler also calls `_require_admin`.
"""
from __future__ import annotations

import pytest


pytestmark = [pytest.mark.functional, pytest.mark.regression]


# Admin routes that mutate state or expose operational data. The middleware
# must 403 these for an API-key caller before the handler ever runs.
ADMIN_ROUTES: list[tuple[str, str]] = [
    ("GET",  "/admin/metrics/ops-health"),
    ("GET",  "/admin/metering/health"),
    ("POST", "/admin/subscriptions/expire-due"),
    ("POST", "/admin/metering/reconcile"),
    ("POST", "/admin/metering/bill-overage"),
    ("GET",  "/admin/metrics/summary"),
    ("GET",  "/admin/rbac/roles"),
    ("GET",  "/admin/waitlist"),
]


@pytest.mark.parametrize("method,path", ADMIN_ROUTES, ids=lambda p: str(p))
async def test_admin_route_rejects_api_key(client_with_auth, method: str, path: str) -> None:
    """A valid API key must get 403 on admin routes — never 200/2xx."""
    body = {} if method in ("POST", "PATCH", "PUT") else None
    r = await client_with_auth.request(method, path, json=body)
    assert r.status_code == 403, (
        f"{method} {path} returned {r.status_code} for an API-key caller; "
        f"expected 403. Body: {r.text[:300]}"
    )
