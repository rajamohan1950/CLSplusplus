"""Tests for the PostHog-backed weblab rollout system.

Covers the weblab helpers (fail-safe behaviour), the auto-rollback watcher's
green/red evaluation + rollback gating, and the launch-rollout gate on the
registration endpoint.
"""
from __future__ import annotations

import pytest

from clsplusplus import weblab, weblab_watcher
from clsplusplus.config import Settings

pytestmark = [pytest.mark.unit, pytest.mark.regression]


# ── weblab helpers ────────────────────────────────────────────────────────

class _FakeClient:
    """Stand-in PostHog client."""

    def __init__(self, flag_value):
        self._v = flag_value

    def feature_enabled(self, flag, distinct_id):
        return bool(self._v)

    def get_feature_flag(self, flag, distinct_id):
        return self._v

    def get_all_flags(self, distinct_id):
        return {"launch-rollout": self._v}


class TestWeblabHelpers:
    def setup_method(self):
        weblab.reset_client()

    def teardown_method(self):
        weblab.reset_client()

    def test_enabled_fails_safe_to_default_when_unconfigured(self):
        # No posthog_api_key → no client → caller default is returned.
        s = Settings(posthog_api_key="")
        assert weblab.enabled("launch-rollout", "user@x.com", s, default=True) is True
        assert weblab.enabled("launch-rollout", "user@x.com", s, default=False) is False

    def test_treatment_fails_safe_to_default_when_unconfigured(self):
        s = Settings(posthog_api_key="")
        assert weblab.treatment("feat", "ns-1", s, default="control") == "control"

    def test_enabled_reads_from_client(self):
        weblab._client = _FakeClient(True)
        s = Settings(posthog_api_key="phc_test")
        assert weblab.enabled("launch-rollout", "user@x.com", s) is True

    def test_treatment_maps_variant_and_bool(self):
        s = Settings(posthog_api_key="phc_test")
        weblab._client = _FakeClient("T2")
        assert weblab.treatment("feat", "ns-1", s) == "T2"
        weblab._client = _FakeClient(True)
        assert weblab.treatment("feat", "ns-1", s) == "T1"
        weblab._client = _FakeClient(None)
        assert weblab.treatment("feat", "ns-1", s, default="control") == "control"

    def test_helpers_swallow_client_errors(self):
        class _Boom:
            def feature_enabled(self, *a):
                raise RuntimeError("posthog down")

        weblab._client = _Boom()
        s = Settings(posthog_api_key="phc_test")
        # Must not raise — falls back to default.
        assert weblab.enabled("launch-rollout", "u", s, default=True) is True


# ── auto-rollback watcher ─────────────────────────────────────────────────

def _health(total=1000, err_5xx=0.0, p95=200.0):
    return {
        "total_requests": total,
        "error_rate_5xx": err_5xx,
        "latency_ms": {"p95": p95},
    }


class TestWeblabWatcher:
    async def test_evaluate_green_when_healthy(self, monkeypatch):
        async def fake(url, window_minutes=60):
            return _health(total=1000, err_5xx=0.1, p95=200)

        monkeypatch.setattr(weblab_watcher, "aggregate_health", fake)
        w = weblab_watcher.WeblabWatcher(Settings())
        v = await w.evaluate()
        assert v["status"] == "green"
        assert v["breaches"] == []

    async def test_evaluate_red_on_high_5xx(self, monkeypatch):
        async def fake(url, window_minutes=60):
            return _health(total=1000, err_5xx=9.0, p95=200)

        monkeypatch.setattr(weblab_watcher, "aggregate_health", fake)
        w = weblab_watcher.WeblabWatcher(Settings())
        v = await w.evaluate()
        assert v["status"] == "red"
        assert any("5xx" in b for b in v["breaches"])

    async def test_evaluate_red_on_high_latency(self, monkeypatch):
        async def fake(url, window_minutes=60):
            return _health(total=1000, err_5xx=0.0, p95=9000)

        monkeypatch.setattr(weblab_watcher, "aggregate_health", fake)
        w = weblab_watcher.WeblabWatcher(Settings())
        v = await w.evaluate()
        assert v["status"] == "red"
        assert any("p95" in b for b in v["breaches"])

    async def test_evaluate_green_below_min_requests(self, monkeypatch):
        # Bad rates but a tiny sample — must not trip.
        async def fake(url, window_minutes=60):
            return _health(total=3, err_5xx=80.0, p95=9000)

        monkeypatch.setattr(weblab_watcher, "aggregate_health", fake)
        w = weblab_watcher.WeblabWatcher(Settings())
        v = await w.evaluate()
        assert v["status"] == "green"

    async def test_run_once_rolls_back_when_red_and_enabled(self, monkeypatch):
        async def fake(url, window_minutes=60):
            return _health(total=1000, err_5xx=9.0, p95=200)

        monkeypatch.setattr(weblab_watcher, "aggregate_health", fake)
        w = weblab_watcher.WeblabWatcher(
            Settings(weblab_auto_rollback_enabled=True,
                     weblab_watched_flags="launch-rollout")
        )
        called = []

        async def fake_rollback(flag, breaches):
            called.append(flag)
            return True

        monkeypatch.setattr(w, "_rollback", fake_rollback)
        result = await w.run_once()
        assert result["status"] == "red"
        assert result["rolled_back"] == ["launch-rollout"]
        assert called == ["launch-rollout"]

    async def test_run_once_no_rollback_when_disabled(self, monkeypatch):
        async def fake(url, window_minutes=60):
            return _health(total=1000, err_5xx=9.0, p95=200)

        monkeypatch.setattr(weblab_watcher, "aggregate_health", fake)
        w = weblab_watcher.WeblabWatcher(
            Settings(weblab_auto_rollback_enabled=False)
        )
        result = await w.run_once()
        assert result["status"] == "red"
        assert result["rolled_back"] == []

    async def test_run_once_no_rollback_when_green(self, monkeypatch):
        async def fake(url, window_minutes=60):
            return _health(total=1000, err_5xx=0.0, p95=100)

        monkeypatch.setattr(weblab_watcher, "aggregate_health", fake)
        w = weblab_watcher.WeblabWatcher(
            Settings(weblab_auto_rollback_enabled=True)
        )
        result = await w.run_once()
        assert result["status"] == "green"
        assert result["rolled_back"] == []


# ── launch-rollout gate on /v1/auth/register ──────────────────────────────

class TestLaunchRolloutGate:
    async def test_signup_gated_to_waitlist_when_not_rolled_in(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(weblab, "enabled", lambda *a, **k: False)
        r = await client.post(
            "/v1/auth/register",
            json={"email": "wave@acme.com", "password": "supersecret123",
                  "name": "Wave User"},
        )
        assert r.status_code == 503
        body = r.json()
        assert body.get("error") == "launch_wave"
        assert body.get("waitlist") is True

    async def test_signup_passes_gate_when_rolled_in(self, client, monkeypatch):
        monkeypatch.setattr(weblab, "enabled", lambda *a, **k: True)
        r = await client.post(
            "/v1/auth/register",
            json={"email": "rolled@acme.com", "password": "supersecret123",
                  "name": "Rolled User"},
        )
        # Rolled in → past the weblab gate. Downstream may 200 or fail on cap
        # or email, but it must NOT be the launch_wave rejection.
        assert not (r.status_code == 503
                    and r.json().get("error") == "launch_wave")
