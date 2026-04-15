"""CLS++ per-user metrics emission — fire-and-forget counters in Redis."""

from __future__ import annotations

import logging
import time as _time_mod
from datetime import datetime
from typing import Optional

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


def _period() -> str:
    return datetime.utcnow().strftime("%Y-%m")


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _week() -> str:
    return datetime.utcnow().strftime("%Y-W%W")


class MetricsEmitter:
    """Fire-and-forget per-user metrics. Never blocks user requests."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def emit(self, user_id: str, metric: str, count: int = 1) -> None:
        """Increment a metric counter for a user in the current period."""
        try:
            client = _redis_client(self.settings.redis_url)
            key = f"cls:metrics:{user_id}:{_period()}"
            await client.hincrby(key, metric, count)
            await client.expire(key, 60 * 60 * 24 * 65)  # 65 days (keep ~2 months)
        except Exception:
            pass

    async def get_user_metrics(self, user_id: str, period: str = None) -> dict:
        """Get all metrics for a user in a period."""
        period = period or _period()
        try:
            client = _redis_client(self.settings.redis_url)
            key = f"cls:metrics:{user_id}:{period}"
            data = await client.hgetall(key)
            return {k: int(v) for k, v in data.items()}
        except Exception:
            return {}

    async def get_all_user_ids(self, period: str = None) -> list[str]:
        """Scan for all entity IDs that have metrics in a period.

        IDs can be: user UUIDs, 'ext:{uid}', 'ns:{namespace}', 'system'.
        Key format: cls:metrics:{entity_id}:{period}
        """
        period = period or _period()
        try:
            client = _redis_client(self.settings.redis_url)
            prefix = f"cls:metrics:*:{period}"
            ids = []
            async for key in client.scan_iter(match=prefix, count=500):
                # Strip prefix and suffix to get entity_id
                # key = "cls:metrics:{entity_id}:{period}"
                without_prefix = key[len("cls:metrics:"):]
                # entity_id is everything before the last :{period}
                suffix = f":{period}"
                if without_prefix.endswith(suffix):
                    entity_id = without_prefix[:-len(suffix)]
                    ids.append(entity_id)
            return ids
        except Exception:
            return []

    async def get_aggregate_metrics(self, period: str = None) -> dict:
        """Sum metrics across ALL entities for a period (admin use)."""
        period = period or _period()
        user_ids = await self.get_all_user_ids(period)
        totals: dict[str, int] = {}
        for uid in user_ids:
            user_metrics = await self.get_user_metrics(uid, period)
            for k, v in user_metrics.items():
                totals[k] = totals.get(k, 0) + v
        return totals

    # =========================================================================
    # Unified "active users" — across extension, CLI, plugin, API
    # =========================================================================
    # `cls:active:global` is a Redis ZSET with score = epoch seconds, used as a
    # 15-minute sliding window for the landing-page "active now" counter.
    # `cls:dau:global:{YYYY-MM-DD}` is a Redis SET of unique actor identifiers
    # counted as daily active users across every surface (extension/CLI/API).

    async def record_active_user(self, identity: str) -> None:
        """Mark an authenticated actor as 'active right now'. Fire-and-forget."""
        if not identity:
            return
        try:
            client = _redis_client(self.settings.redis_url)
            now = int(_time_mod.time())
            # Sliding-window set (trimmed on write to keep it cheap)
            await client.zadd("cls:active:global", {identity: now})
            await client.zremrangebyscore("cls:active:global", 0, now - 3600)
            await client.expire("cls:active:global", 3600)
            # DAU set
            dau_key = f"cls:dau:global:{_today()}"
            await client.sadd(dau_key, identity)
            await client.expire(dau_key, 60 * 60 * 24 * 35)
        except Exception:
            pass

    async def get_active_now(self, window_seconds: int = 900) -> int:
        """Count distinct actors seen in the last `window_seconds` (default 15m)."""
        try:
            client = _redis_client(self.settings.redis_url)
            now = int(_time_mod.time())
            return int(
                await client.zcount("cls:active:global", now - window_seconds, now)
                or 0
            )
        except Exception:
            return 0

    async def get_dau(self, date_str: Optional[str] = None) -> int:
        """Daily active users across all surfaces for the given date (default today)."""
        date_str = date_str or _today()
        try:
            client = _redis_client(self.settings.redis_url)
            return int(await client.scard(f"cls:dau:global:{date_str}") or 0)
        except Exception:
            return 0

    # =========================================================================
    # Extension analytics — Redis SET/HASH operations
    # =========================================================================

    async def record_ext_telemetry(self, uid: str, event: str, site: str = "", data: dict = None) -> None:
        """Record an extension telemetry event. Fire-and-forget."""
        try:
            client = _redis_client(self.settings.redis_url)
            today = _today()
            period = _period()
            week = _week()

            if event == "install":
                await client.sadd(f"cls:ext:installs:{today}", uid)
                await client.expire(f"cls:ext:installs:{today}", 60 * 60 * 24 * 95)

            # Track active users (DAU/WAU/MAU)
            await client.sadd(f"cls:ext:active:{today}", uid)
            await client.expire(f"cls:ext:active:{today}", 60 * 60 * 24 * 35)
            await client.sadd(f"cls:ext:active_week:{week}", uid)
            await client.expire(f"cls:ext:active_week:{week}", 60 * 60 * 24 * 10)
            await client.sadd(f"cls:ext:active_month:{period}", uid)
            await client.expire(f"cls:ext:active_month:{period}", 60 * 60 * 24 * 65)

            if event == "context_injected" and site:
                await client.hincrby(f"cls:ext:sites:{period}", site, 1)
                await client.expire(f"cls:ext:sites:{period}", 60 * 60 * 24 * 65)

            if event == "message_captured" and site:
                await client.hincrby(f"cls:ext:messages:{period}", site, 1)
                await client.expire(f"cls:ext:messages:{period}", 60 * 60 * 24 * 65)

            if event == "feature_toggle" and data:
                feature = data.get("feature", "")
                enabled = data.get("enabled", False)
                field = f"{feature}_{'on' if enabled else 'off'}"
                await client.hincrby("cls:ext:settings", field, 1)

            # Also emit as a regular metric for the extension user
            await self.emit(f"ext:{uid}", f"ext_{event}", 1)
            # Extension activity also contributes to the unified active/DAU counter
            await self.record_active_user(f"ext:{uid}")

        except Exception:
            pass

    async def get_extension_analytics(self) -> dict:
        """Read extension analytics for admin dashboard."""
        try:
            client = _redis_client(self.settings.redis_url)
            today = _today()
            period = _period()
            week = _week()

            installs_today = await client.scard(f"cls:ext:installs:{today}") or 0

            # Count monthly installs by scanning daily keys
            installs_month = 0
            async for key in client.scan_iter(match=f"cls:ext:installs:{period[:7]}*", count=100):
                installs_month += await client.scard(key) or 0

            dau = await client.scard(f"cls:ext:active:{today}") or 0
            wau = await client.scard(f"cls:ext:active_week:{week}") or 0
            mau = await client.scard(f"cls:ext:active_month:{period}") or 0

            site_usage = await client.hgetall(f"cls:ext:sites:{period}") or {}
            site_usage = {k: int(v) for k, v in site_usage.items()}

            messages = await client.hgetall(f"cls:ext:messages:{period}") or {}
            messages = {k: int(v) for k, v in messages.items()}

            ext_settings = await client.hgetall("cls:ext:settings") or {}
            ext_settings = {k: int(v) for k, v in ext_settings.items()}

            return {
                "installs_today": installs_today,
                "installs_this_month": installs_month,
                "dau": dau,
                "wau": wau,
                "mau": mau,
                "site_usage": site_usage,
                "messages_captured": messages,
                "settings": ext_settings,
            }
        except Exception:
            return {
                "installs_today": 0, "installs_this_month": 0,
                "dau": 0, "wau": 0, "mau": 0,
                "site_usage": {}, "messages_captured": {}, "settings": {},
            }
