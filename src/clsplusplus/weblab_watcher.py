"""Weblab auto-rollback watcher.

Polls the ops-health rolling window. When a watched weblab's success metrics
breach the configured thresholds, it disables the PostHog flag — every user
falls back to control — and records the verdict for the /admin/weblab page.

Modeled on SubscriptionWatchdog: a background asyncio task with start/stop,
plus run_once()/evaluate() for on-demand use by the admin endpoint.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from clsplusplus.config import Settings
from clsplusplus.health_metrics import aggregate_health

logger = logging.getLogger(__name__)

_STARTUP_DELAY_SECONDS = 60


class WeblabWatcher:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._task: Optional[asyncio.Task] = None
        # Last verdict — surfaced read-only by GET /admin/weblab.
        self.last_result: dict = {"status": "pending"}

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def _watched_flags(self) -> list[str]:
        raw = self.settings.weblab_watched_flags or ""
        return [f.strip() for f in raw.split(",") if f.strip()]

    # ── evaluation ────────────────────────────────────────────────────────
    async def evaluate(self) -> dict:
        """Read ops-health and return a green/red verdict. No side effects."""
        s = self.settings
        try:
            health = await aggregate_health(
                s.redis_url, window_minutes=s.weblab_health_window_minutes
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("weblab watcher: health read failed: %s", e)
            return {"status": "unknown", "error": str(e),
                    "watched_flags": self._watched_flags()}

        total = int(health.get("total_requests", 0) or 0)
        err_5xx = float(health.get("error_rate_5xx", 0.0) or 0.0)
        p95 = float((health.get("latency_ms") or {}).get("p95", 0.0) or 0.0)

        breaches: list[str] = []
        # Below the minimum sample we cannot trust the rate — stay green.
        if total >= s.weblab_rollback_min_requests:
            if err_5xx > s.weblab_rollback_5xx_pct:
                breaches.append(
                    f"5xx error rate {err_5xx:.2f}% > {s.weblab_rollback_5xx_pct}%"
                )
            if p95 > s.weblab_rollback_p95_ms:
                breaches.append(
                    f"p95 latency {p95:.0f}ms > {s.weblab_rollback_p95_ms}ms"
                )

        return {
            "status": "red" if breaches else "green",
            "breaches": breaches,
            "metrics": {
                "total_requests": total,
                "error_rate_5xx": round(err_5xx, 3),
                "p95_ms": round(p95, 1),
            },
            "watched_flags": self._watched_flags(),
            "auto_rollback_enabled": s.weblab_auto_rollback_enabled,
        }

    async def run_once(self) -> dict:
        """Evaluate and, when red, roll back every watched flag."""
        result = await self.evaluate()
        result["rolled_back"] = []
        if result.get("status") == "red" and self.settings.weblab_auto_rollback_enabled:
            for flag in result.get("watched_flags", []):
                if await self._rollback(flag, result["breaches"]):
                    result["rolled_back"].append(flag)
        self.last_result = result
        return result

    async def _run(self) -> None:
        await asyncio.sleep(_STARTUP_DELAY_SECONDS)  # let the pool warm up
        while True:
            try:
                result = await self.run_once()
                if result.get("rolled_back"):
                    logger.error(
                        "weblab watcher: ROLLED BACK %s — breaches: %s",
                        result["rolled_back"], result.get("breaches"),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("weblab watcher: tick failed: %s", e)
            await asyncio.sleep(self.settings.weblab_metric_poll_seconds)

    # ── rollback ──────────────────────────────────────────────────────────
    async def _rollback(self, flag_key: str, breaches: list) -> bool:
        """Disable a PostHog flag via the management API. Returns success."""
        s = self.settings
        if not (s.posthog_personal_api_key and s.posthog_project_id):
            logger.error(
                "weblab watcher: cannot roll back %s — PostHog management "
                "API not configured (need CLS_POSTHOG_PERSONAL_API_KEY + "
                "CLS_POSTHOG_PROJECT_ID)", flag_key,
            )
            return False
        headers = {"Authorization": f"Bearer {s.posthog_personal_api_key}"}
        base = (f"{s.posthog_api_host.rstrip('/')}"
                f"/api/projects/{s.posthog_project_id}/feature_flags")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(base + "/", headers=headers,
                                        params={"limit": 300})
                resp.raise_for_status()
                flag_id = next(
                    (f.get("id") for f in resp.json().get("results", [])
                     if f.get("key") == flag_key),
                    None,
                )
                if flag_id is None:
                    logger.error("weblab watcher: flag %s not found in PostHog",
                                 flag_key)
                    return False
                patch = await client.patch(
                    f"{base}/{flag_id}/", headers=headers,
                    json={"active": False},
                )
                patch.raise_for_status()
            logger.error("weblab watcher: disabled PostHog flag %s — %s",
                         flag_key, breaches)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("weblab watcher: rollback of %s failed: %s",
                         flag_key, e)
            return False
