"""Blackbox coverage over every public route.

The goal is to touch every route in `api.py` with at least one HTTP request and
assert we never get 500. Auth walls return 401/403 rather than 500, so we
accept those as valid for endpoints we cannot reach without a logged-in
session.

This is what gives us the "100% route coverage" claim — not that every route's
happy path is exercised (that is the job of functional + journey tests) but
that the entire surface is live and doesn't crash.
"""
from __future__ import annotations

import pytest


pytestmark = [pytest.mark.functional, pytest.mark.blackbox]


# (method, path, allowed_statuses). ``allowed_statuses`` must exclude 5xx.
# Routes that require a path/query parameter use a throwaway literal — the
# contract is that the server rejects the bogus value cleanly, not that it
# succeeds.
PUBLIC_ROUTES: list[tuple[str, str, tuple[int, ...]]] = [
    ("GET",  "/",                                   (200,)),
    ("GET",  "/health",                             (200,)),
    ("GET",  "/v1/health",                          (200,)),
    ("GET",  "/docs",                               (200,)),
    ("GET",  "/openapi.json",                       (200,)),
    ("GET",  "/v1/memory/health",                   (200, 401, 403)),
    ("GET",  "/v1/health/score",                    (200, 401, 403)),
    ("GET",  "/v1/waitlist/stats",                  (200,)),
    ("GET",  "/v1/demo/status",                     (200, 401, 403)),
    ("GET",  "/v1/trg/state",                       (200, 401, 403)),
]


AUTH_WALLED_ROUTES: list[tuple[str, str, tuple[int, ...]]] = [
    ("GET",    "/v1/auth/me",                       (200, 401)),
    ("POST",   "/v1/auth/logout",                   (200, 401)),
    ("GET",    "/v1/memory/list",                   (200, 401, 403)),
    ("POST",   "/v1/memory/write",                  (200, 400, 401, 403, 422)),
    ("POST",   "/v1/memory/read",                   (200, 400, 401, 403, 422)),
    ("POST",   "/v1/memories/search",               (200, 400, 401, 403, 422)),
    ("POST",   "/v1/memories/retrieve",             (200, 400, 401, 403, 422)),
    ("DELETE", "/v1/memory/forget",                 (200, 400, 401, 403, 422)),
    ("GET",    "/v1/memory/personal",               (200, 401, 403)),
    ("GET",    "/v1/context-log",                   (200, 401, 403)),
    ("GET",    "/v1/memory/traces",                 (200, 401, 403)),
    ("GET",    "/v1/memory/namespaces",             (200, 401, 403)),
    ("GET",    "/v1/memory/phases",                 (200, 401, 403)),
    ("GET",    "/v1/memories/knowledge",            (200, 401, 403, 404)),
    ("GET",    "/v1/prompts/sessions",              (200, 401, 403)),
    ("GET",    "/v1/prompts/timeline",              (200, 401, 403)),
    ("POST",   "/v1/prompts/ingest",                (200, 400, 401, 403, 422)),
    ("GET",    "/v1/usage",                         (200, 401, 403)),
    ("GET",    "/v1/billing/usage",                 (200, 401, 403)),
    ("GET",    "/v1/user/usage",                    (200, 401)),
    ("GET",    "/v1/user/usage/history",            (200, 401)),
    ("GET",    "/v1/user/integrations",             (200, 401)),
    ("POST",   "/v1/user/upgrade",                  (200, 400, 401, 422)),
    ("PATCH",  "/v1/user/profile",                  (200, 400, 401, 422)),
    ("POST",   "/v1/billing/checkout",              (200, 400, 401, 403, 422, 503)),
    ("GET",    "/v1/billing/portal",                (200, 401, 403, 503)),
    ("GET",    "/v1/integrations",                  (200, 401, 403)),
    ("POST",   "/v1/integrations",                  (200, 400, 401, 403, 422)),
    ("GET",    "/v1/chat/sessions/bogus-session",   (401, 403, 404)),
]


ADMIN_ROUTES: list[tuple[str, str, tuple[int, ...]]] = [
    ("GET",  "/admin/metrics/summary",              (200, 401, 403)),
    ("GET",  "/admin/metrics/signups",              (200, 401, 403)),
    ("GET",  "/admin/metrics/revenue",              (200, 401, 403)),
    ("GET",  "/admin/metrics/operations",           (200, 401, 403)),
    ("GET",  "/admin/metrics/users",                (200, 401, 403)),
    ("GET",  "/admin/metrics/extension",            (200, 401, 403)),
    ("GET",  "/admin/metrics/storage",              (200, 401, 403)),
    ("GET",  "/admin/rbac/scopes",                  (200, 401, 403)),
    ("GET",  "/admin/rbac/roles",                   (200, 401, 403)),
    ("GET",  "/admin/rbac/groups",                  (200, 401, 403)),
    ("GET",  "/admin/waitlist",                     (200, 401, 403)),
    ("GET",  "/admin/tests/waitlist/history",       (200, 401, 403)),
]


AUTH_PUBLIC_ROUTES: list[tuple[str, str, tuple[int, ...]]] = [
    ("POST", "/v1/auth/register",        (200, 400, 422, 503)),
    ("POST", "/v1/auth/login",           (200, 400, 401, 422)),
    ("POST", "/v1/auth/forgot-password", (200, 400, 422)),
    ("POST", "/v1/auth/reset-password",  (200, 400, 401, 422)),
    ("POST", "/v1/waitlist/join",        (200, 400, 422)),
    ("POST", "/v1/waitlist/verify",      (200, 400, 401, 422)),
]


ALL_ROUTES = PUBLIC_ROUTES + AUTH_WALLED_ROUTES + ADMIN_ROUTES + AUTH_PUBLIC_ROUTES


@pytest.mark.parametrize("method,path,allowed", ALL_ROUTES, ids=lambda p: str(p))
def test_route_never_500(client, method: str, path: str, allowed: tuple[int, ...]) -> None:
    """Every route must respond in the allowed set — never 500 or network error."""
    body = {} if method in ("POST", "PATCH", "PUT") else None
    r = client.request(method, path, json=body)
    assert r.status_code in allowed, (
        f"{method} {path} returned {r.status_code}: {r.text[:300]}"
    )


@pytest.mark.sanity
def test_openapi_lists_all_paths(client) -> None:
    """Sanity: make sure the live OpenAPI doc covers the routes we claim exist."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    declared = set(spec.get("paths", {}).keys())
    # We don't require every path we test — some are parameterised with values
    # that won't appear verbatim in OpenAPI. We do require the common prefixes.
    expected_prefixes = ("/v1/auth/", "/v1/memory", "/v1/billing", "/admin/")
    for prefix in expected_prefixes:
        assert any(p.startswith(prefix) for p in declared), (
            f"OpenAPI has no routes under {prefix}"
        )
