"""Website-traffic + conversion-funnel analytics routes.

Self-contained module so api.py only needs a single registration call. Adds:

  * POST /v1/events/track   — public first-party event capture (page views,
    clicks). Writes to the `web_events` table.
  * GET  /admin/metrics/funnel — admin-only aggregation: unique visitors,
    the visitor -> signup -> active-user funnel with conversion rates, and
    the most-clicked pages / most-engaged CTAs.

The admin guard is passed in from api.py so this module stays decoupled from
the request-state plumbing there.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import FastAPI, HTTPException, Request

from clsplusplus.config import Settings
from clsplusplus.stores.web_events_store import WebEventsStore

logger = logging.getLogger(__name__)


def register_funnel_routes(
    app: FastAPI,
    settings: Settings,
    require_admin: Callable[[Request], None],
) -> WebEventsStore:
    """Attach the funnel-analytics endpoints to `app`.

    `require_admin` is api.py's `_require_admin` helper — it raises 403 when
    the request is not an authenticated admin.

    Returns the WebEventsStore so callers can reuse the pool if needed.
    """
    store = WebEventsStore(settings)

    @app.post("/v1/events/track")
    async def track_event(request: Request):
        """Public: capture one website page view / click. No auth.

        Body: {event, page, ref, session_id}. `event` is required; the rest
        default to empty strings. Always returns 200 unless the body is
        unparseable or `event` is missing — analytics must never break the
        marketing site.
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        event = body.get("event")
        if not event or not str(event).strip():
            raise HTTPException(status_code=400, detail="event is required")

        try:
            await store.record_event(
                event=str(event),
                page=str(body.get("page", "") or ""),
                ref=str(body.get("ref", "") or ""),
                session_id=str(body.get("session_id", "") or ""),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Capture failure must not surface to the visitor's browser as an
            # error — log and return ok so the page keeps working.
            logger.error("web event capture failed: %s: %s", type(e).__name__, e)
            return {"ok": False}
        return {"ok": True}

    @app.get("/admin/metrics/funnel")
    async def admin_funnel(request: Request):
        """Admin: website traffic + conversion funnel + top pages.

        Optional query param `days` (1-365, default 30) sets the window.
        """
        require_admin(request)
        try:
            days = int(request.query_params.get("days", 30))
        except (TypeError, ValueError):
            days = 30
        try:
            return await store.funnel_summary(days=days)
        except Exception as e:
            logger.error("Admin funnel metrics error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Funnel metrics unavailable")

    return store
