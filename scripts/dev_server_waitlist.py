#!/usr/bin/env python3
"""CLS++ dev server for waitlist UI testing — in-memory, no Postgres/Redis.

Boots the real FastAPI app with the full waitlist feature, but swaps every
external store out for the FakeHarness. Adds dev-only endpoints under /v1/dev/
for UI-test setup (admin login, state reset, email peek, user seeding).

Run:
    python3 scripts/dev_server_waitlist.py           # port 8090
    CLS_DEV_PORT=9000 python3 scripts/dev_server_waitlist.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Install fakes BEFORE importing create_app so stores are patched at import time.
from waitlist_fake_harness import install  # noqa: E402

HARNESS = install()

from clsplusplus.api import create_app  # noqa: E402
from clsplusplus.config import Settings  # noqa: E402
from clsplusplus.jwt_utils import create_token  # noqa: E402

SETTINGS = Settings(
    require_api_key=False,
    jwt_secret="dev-ui-testing-secret-32byteslong00",
    max_active_users=5,
    waitlist_queue_seed_offset=47,
    waitlist_active_floor=3,
    waitlist_dau_healthy_threshold=5,
    waitlist_promote_interval_seconds=86400,
    resend_api_key="x",  # non-empty so EmailService._enabled is True
    track_usage=False,
    enforce_quotas=False,
    cookie_secure=False,
    database_url="postgresql://unused:unused@localhost/unused",
    redis_url="redis://unused:6379",
)

app = create_app(SETTINGS)

# ---------------------------------------------------------------------------
# Dev-only endpoints (UI test helpers)
#
# create_app() ends with a `app.mount("/", StaticFiles(...))` which greedily
# catches every unmatched request. Routes added after create_app returns
# therefore sit BEHIND the mount and 404. Temporarily pop the root mount,
# decorate the dev routes, then re-append it.
# ---------------------------------------------------------------------------

from fastapi import Request as _Req  # noqa: E402
from starlette.responses import RedirectResponse as _Redirect  # noqa: E402
from starlette.routing import Mount as _Mount  # noqa: E402

_static_mount = None
for _i, _r in enumerate(list(app.router.routes)):
    if isinstance(_r, _Mount) and _r.path == "":
        _static_mount = app.router.routes.pop(_i)
        break


@app.get("/v1/dev/login-as-admin")
async def _dev_admin_login():
    """One-click admin session for UI tests.

    Pre-creates the admin user in the fake store so /v1/auth/me finds it.
    """
    user_id = "dev-admin-0"
    HARNESS.users[user_id] = {
        "id": user_id,
        "email": "admin@dev.local",
        "name": "Dev Admin",
        "tier": "enterprise",
        "is_admin": True,
        "password_hash": None,
        "google_id": None,
        "avatar_url": None,
        "email_verified": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    HARNESS.users_by_email["admin@dev.local"] = user_id

    token = create_token(
        user_id=user_id,
        email="admin@dev.local",
        is_admin=True,
        secret=SETTINGS.jwt_secret,
    )
    resp = _Redirect("/admin/dashboard.html")
    resp.set_cookie(
        "cls_session",
        token,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
        max_age=7 * 86400,
    )
    return resp


@app.get("/v1/dev/login-as-user")
async def _dev_user_login():
    """One-click plain-user session for UI tests (non-admin)."""
    user_id = "dev-user-0"
    HARNESS.users[user_id] = {
        "id": user_id,
        "email": "user@dev.local",
        "name": "Dev User",
        "tier": "free",
        "is_admin": False,
        "password_hash": None,
        "google_id": None,
        "avatar_url": None,
        "email_verified": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    HARNESS.users_by_email["user@dev.local"] = user_id
    token = create_token(
        user_id=user_id,
        email="user@dev.local",
        is_admin=False,
        secret=SETTINGS.jwt_secret,
    )
    from fastapi import Query as _Q  # not used, keep import local

    resp = _Redirect("/dashboard.html?welcome=waitlist")
    resp.set_cookie(
        "cls_session",
        token,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
        max_age=7 * 86400,
    )
    return resp


@app.post("/v1/dev/reset-harness")
async def _dev_reset_harness():
    HARNESS.reset()
    return {"ok": True}


@app.get("/v1/dev/emails")
async def _dev_list_emails():
    return {"sent": HARNESS.sent_emails}


@app.post("/v1/dev/seed-users")
async def _dev_seed_users(request: _Req):
    body = await request.json()
    n = int(body.get("count", 0))
    import uuid as _u

    for i in range(n):
        email = f"seed{i}@dev.local"
        if email in HARNESS.users_by_email:
            continue
        uid = str(_u.uuid4())
        HARNESS.users[uid] = {
            "id": uid,
            "email": email,
            "name": f"Seed {i}",
            "is_admin": False,
            "tier": "free",
            "password_hash": None,
            "email_verified": True,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        HARNESS.users_by_email[email] = uid
    return {"count": len(HARNESS.users)}


@app.get("/v1/dev/state")
async def _dev_state():
    return {
        "visitors": len(HARNESS.visitors),
        "users": len(HARNESS.users),
        "pending_otp": len(HARNESS.pending),
        "sent_emails": len(HARNESS.sent_emails),
        "active_now": HARNESS.active_now,
        "dau": HARNESS.dau,
    }


# Re-append the static files mount so it sits AFTER all API routes
if _static_mount is not None:
    app.router.routes.append(_static_mount)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CLS_DEV_PORT", 8090))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
