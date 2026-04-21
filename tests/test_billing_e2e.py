"""End-to-end live checks against a running CLS++ API.

What this covers (the full billing spine):

    T1. Admin endpoints reject unauthenticated requests.
    T2. /admin/metering/health returns 200 with all seven sub-checks green.
    T3. /admin/metering/reconcile runs and reports 0 drift.
    T4. /admin/subscriptions/expire-due runs without errors.
    T5. OAuth start endpoints redirect (Google + GitHub).
    T6. Public pages are reachable (signup, homepage).

Opt-in with two env vars:

    CLS_E2E_BASE_URL       — e.g. "https://www.clsplusplus.com"
    CLS_E2E_ADMIN_COOKIE   — the cls_session JWT for an admin user

Run against production:

    CLS_E2E_BASE_URL=https://www.clsplusplus.com \
    CLS_E2E_ADMIN_COOKIE=<paste> \
    python -m pytest tests/test_billing_e2e.py -v

When either env var is unset, the whole module skips cleanly so the
normal test suite never tries to hit a real server.
"""

from __future__ import annotations

import os
import urllib.parse

import pytest

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


_BASE = os.environ.get("CLS_E2E_BASE_URL", "").rstrip("/")
_COOKIE = os.environ.get("CLS_E2E_ADMIN_COOKIE", "")

pytestmark = pytest.mark.skipif(
    not (_BASE and _COOKIE and httpx is not None),
    reason=(
        "Set CLS_E2E_BASE_URL + CLS_E2E_ADMIN_COOKIE (and have httpx installed) "
        "to run the billing E2E suite against a live API."
    ),
)


def _admin_get(path: str) -> httpx.Response:
    with httpx.Client(timeout=15.0, follow_redirects=False) as c:
        return c.get(f"{_BASE}{path}", cookies={"cls_session": _COOKIE})


def _admin_post(path: str) -> httpx.Response:
    with httpx.Client(timeout=30.0, follow_redirects=False) as c:
        return c.post(f"{_BASE}{path}", cookies={"cls_session": _COOKIE})


def _anon_get(path: str) -> httpx.Response:
    with httpx.Client(timeout=15.0, follow_redirects=False) as c:
        return c.get(f"{_BASE}{path}")


def _anon_post(path: str) -> httpx.Response:
    with httpx.Client(timeout=15.0, follow_redirects=False) as c:
        return c.post(f"{_BASE}{path}")


# ---------------------------------------------------------------------------
# T1. Admin endpoints must reject anonymous callers.
# ---------------------------------------------------------------------------


class TestAdminAuth:
    def test_health_endpoint_requires_auth(self):
        resp = _anon_get("/api/admin/metering/health")
        assert resp.status_code == 401, resp.text

    def test_reconcile_endpoint_requires_auth(self):
        resp = _anon_post("/api/admin/metering/reconcile")
        assert resp.status_code == 401, resp.text

    def test_expire_due_requires_auth(self):
        resp = _anon_post("/api/admin/subscriptions/expire-due")
        assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# T2. Metering v2 health — the central "is it working" signal.
# ---------------------------------------------------------------------------


class TestMeteringHealth:
    def test_health_returns_200_and_all_checks_pass(self):
        resp = _admin_get("/api/admin/metering/health")
        assert resp.status_code == 200, (
            f"Health endpoint returned {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body["passed"] is True, body

        failed = [c for c in body["checks"] if not c["ok"]]
        assert failed == [], (
            "One or more health checks failed:\n"
            + "\n".join(
                f"  - {c['name']}: {c['detail']}\n    → {c['remediation']}"
                for c in failed
            )
        )

    def test_health_covers_seven_known_checks(self):
        resp = _admin_get("/api/admin/metering/health")
        body = resp.json()
        names = {c["name"] for c in body["checks"]}
        expected = {
            "config.flag_on",
            "config.oncall_email",
            "db.reachable",
            "db.schema_present",
            "writer.roundtrip",
            "dead_letter.clean",
            "reconciler.drift",
        }
        assert expected.issubset(names), f"missing: {expected - names}"

    def test_writer_roundtrip_proves_durable_write(self):
        """The writer.roundtrip check inserts a canary row and reads it back.
        If this passes, durable metering is operational end-to-end."""
        body = _admin_get("/api/admin/metering/health").json()
        roundtrip = next(c for c in body["checks"] if c["name"] == "writer.roundtrip")
        assert roundtrip["ok"], roundtrip


# ---------------------------------------------------------------------------
# T3. Reconciler — Redis and Postgres agree.
# ---------------------------------------------------------------------------


class TestReconciler:
    def test_reconciler_runs_and_reports_no_drift(self):
        resp = _admin_post("/api/admin/metering/reconcile")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["drift_count"] == 0, (
            f"Reconciler found {body['drift_count']} drift finding(s). "
            f"Investigate metering_dead_letter rows with "
            f"error_class='ReconciliationDrift'. Findings: {body.get('drift_findings')}"
        )

    def test_reconciler_accepts_period_param(self):
        # Current period should be supported.
        from datetime import datetime, timezone
        period = datetime.now(timezone.utc).strftime("%Y-%m")
        encoded = urllib.parse.quote(period)
        resp = _admin_post(f"/api/admin/metering/reconcile?period={encoded}")
        assert resp.status_code == 200
        assert resp.json()["period"] == period


# ---------------------------------------------------------------------------
# T4. Subscription watchdog — auto-downgrade of expired paid users.
# ---------------------------------------------------------------------------


class TestSubscriptionWatchdog:
    def test_expire_due_runs_and_reports_structured_result(self):
        resp = _admin_post("/api/admin/subscriptions/expire-due")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Shape: { scanned, downgraded, errors } — sometimes +error text on failure.
        assert "scanned" in body
        assert "downgraded" in body
        assert "errors" in body
        assert body["errors"] == 0, (
            f"Watchdog errored on {body['errors']} users. See server logs."
        )

    def test_expire_due_no_negative_counters(self):
        body = _admin_post("/api/admin/subscriptions/expire-due").json()
        assert body["scanned"] >= 0
        assert body["downgraded"] >= 0
        assert body["downgraded"] <= body["scanned"]


# ---------------------------------------------------------------------------
# T5. OAuth start endpoints exist and redirect.
# ---------------------------------------------------------------------------


class TestOAuthEndpoints:
    @pytest.mark.parametrize("path,target_host", [
        ("/api/v1/auth/google", "accounts.google.com"),
        ("/api/v1/auth/github", "github.com"),
    ])
    def test_oauth_start_returns_redirect(self, path, target_host):
        resp = _anon_get(path)
        assert resp.status_code in (302, 307), (
            f"{path} should redirect, got {resp.status_code}: {resp.text[:200]}"
        )
        location = resp.headers.get("location", "")
        assert target_host in location, (
            f"{path} redirects to {location!r}, expected to contain {target_host}"
        )


# ---------------------------------------------------------------------------
# T6. Public surfaces are reachable.
# ---------------------------------------------------------------------------


class TestPublicSurfaces:
    @pytest.mark.parametrize("path", ["/", "/signup", "/login", "/pricing"])
    def test_page_loads(self, path):
        resp = _anon_get(path)
        assert resp.status_code == 200, (
            f"{path} returned {resp.status_code} — is Vercel down?"
        )
