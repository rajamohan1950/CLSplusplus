"""Web Events Store — PostgreSQL-backed first-party website analytics.

Mirrors the WaitlistStore pattern: lazy asyncpg pool, auto-DDL on first
connect. Backs the public POST /v1/events/track capture endpoint and the
admin conversion-funnel dashboard (GET /admin/metrics/funnel).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import asyncpg

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

# Hard caps applied before insert. The DDL CHECK constraints enforce the same
# bounds, but trimming here keeps a malformed client from getting a 500.
_MAX_EVENT = 64
_MAX_PAGE = 512
_MAX_REF = 512
_MAX_SESSION = 128


def _parse_db_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


def _clip(value: object, limit: int) -> str:
    return str(value or "").strip()[:limit]


class WebEventsStore:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        _parse_db_url(self.settings.database_url),
                        min_size=1,
                        max_size=3,
                        command_timeout=60,
                    )
                    async with self._pool.acquire() as conn:
                        await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        ddl_path = os.path.join(os.path.dirname(__file__), "web_events_ddl.sql")
        with open(ddl_path) as f:
            await conn.execute(f.read())

    # =========================================================================
    # Write path — public event capture
    # =========================================================================

    async def record_event(
        self,
        event: str,
        page: str = "",
        ref: str = "",
        session_id: str = "",
    ) -> None:
        """Append one page view / click. Inputs are clipped to DDL bounds."""
        event = _clip(event, _MAX_EVENT)
        if not event:
            raise ValueError("event is required")
        page = _clip(page, _MAX_PAGE)
        ref = _clip(ref, _MAX_REF)
        session_id = _clip(session_id, _MAX_SESSION)
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO web_events (event, page, ref, session_id)
                VALUES ($1, $2, $3, $4)
                """,
                event, page, ref, session_id,
            )

    # =========================================================================
    # Read path — funnel aggregation
    # =========================================================================

    async def funnel_summary(self, days: int = 30) -> dict:
        """Aggregate the conversion funnel over the trailing `days` window.

        Returns unique visitors (distinct session_id), the visitor -> signup
        -> active-user funnel with conversion rates, and the most-clicked /
        most-viewed pages ranked descending.

        - signup count joins web_events to the `users` table by created_at
          window: users who registered inside the same window.
        - active count = users whose email owns an integration with at least
          one active api_credential (chain: users.email = integrations.owner_email
          -> api_credentials.integration_id, status = 'active').
        """
        days = max(1, min(int(days), 365))
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Unique visitors + total events in window.
            traffic = await conn.fetchrow(
                """
                SELECT
                    COUNT(DISTINCT session_id)
                        FILTER (WHERE session_id <> '')        AS unique_visitors,
                    COUNT(*)
                        FILTER (WHERE event = 'pageview')      AS total_pageviews,
                    COUNT(*)
                        FILTER (WHERE event = 'click')         AS total_clicks
                FROM web_events
                WHERE created_at >= NOW() - ($1::text || ' days')::interval
                """,
                str(days),
            )

            # Pages ranked by click count, with a co-aggregated pageview count.
            page_rows = await conn.fetch(
                """
                SELECT
                    page,
                    COUNT(*) FILTER (WHERE event = 'click')    AS clicks,
                    COUNT(*) FILTER (WHERE event = 'pageview') AS pageviews
                FROM web_events
                WHERE created_at >= NOW() - ($1::text || ' days')::interval
                  AND page <> ''
                GROUP BY page
                ORDER BY clicks DESC, pageviews DESC
                LIMIT 25
                """,
                str(days),
            )

            # Engagement: which clicked targets (ref carries the CTA label /
            # section for click events) draw the most interest.
            engagement_rows = await conn.fetch(
                """
                SELECT ref AS target, COUNT(*) AS clicks
                FROM web_events
                WHERE created_at >= NOW() - ($1::text || ' days')::interval
                  AND event = 'click'
                  AND ref <> ''
                GROUP BY ref
                ORDER BY clicks DESC
                LIMIT 15
                """,
                str(days),
            )

            # Funnel stage 2 — signups in the same window.
            signups = await conn.fetchval(
                """
                SELECT COUNT(*) FROM users
                WHERE created_at >= NOW() - ($1::text || ' days')::interval
                """,
                str(days),
            ) or 0

            # Funnel stage 3 — active users: signed up in window AND own an
            # integration with a live api credential.
            active = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT u.id)
                FROM users u
                JOIN integrations i  ON i.owner_email = u.email
                JOIN api_credentials c ON c.integration_id = i.id
                WHERE u.created_at >= NOW() - ($1::text || ' days')::interval
                  AND c.status = 'active'
                """,
                str(days),
            ) or 0

        visitors = int(traffic["unique_visitors"] or 0)
        signups = int(signups)
        active = int(active)

        def _rate(numer: int, denom: int) -> float:
            return round(100.0 * numer / denom, 2) if denom else 0.0

        return {
            "window_days": days,
            "traffic": {
                "unique_visitors": visitors,
                "total_pageviews": int(traffic["total_pageviews"] or 0),
                "total_clicks": int(traffic["total_clicks"] or 0),
            },
            "funnel": {
                "visitors": visitors,
                "signups": signups,
                "active_users": active,
                "visitor_to_signup_pct": _rate(signups, visitors),
                "signup_to_active_pct": _rate(active, signups),
                "visitor_to_active_pct": _rate(active, visitors),
            },
            "top_pages": [
                {
                    "page": r["page"],
                    "clicks": int(r["clicks"] or 0),
                    "pageviews": int(r["pageviews"] or 0),
                }
                for r in page_rows
            ],
            "top_engagement": [
                {"target": r["target"], "clicks": int(r["clicks"] or 0)}
                for r in engagement_rows
            ],
        }
