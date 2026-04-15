"""In-memory fake harness for the launch waitlist — used by the dev UI server.

Monkey-patches WaitlistStore / UserStore / EmailService / MetricsEmitter /
UserService / MemoryService so the full FastAPI app boots without Postgres,
Redis, or Resend. Intended exclusively for local UI testing of the waitlist.

NEVER import this in production code paths.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else dt


class FakeHarness:
    """All external state the waitlist touches, held in a single process."""

    def __init__(self):
        self.visitors: dict = {}
        self.by_email: dict = {}
        self.pending: dict = {}
        self.users: dict = {}
        self.users_by_email: dict = {}
        self.sent_emails: list = []
        self.active_now = 0
        self.dau = 0

    def reset(self):
        self.visitors.clear()
        self.by_email.clear()
        self.pending.clear()
        self.users.clear()
        self.users_by_email.clear()
        self.sent_emails.clear()
        self.active_now = 0
        self.dau = 0


def install(H: "FakeHarness | None" = None) -> "FakeHarness":
    """Monkey-patch all stores to use in-memory state. Returns the harness."""
    if H is None:
        H = FakeHarness()

    from clsplusplus.email_service import EmailService
    from clsplusplus.memory_service import MemoryService
    from clsplusplus.metrics import MetricsEmitter
    from clsplusplus.stores.user_store import UserStore
    from clsplusplus.stores.waitlist_store import WaitlistStore
    from clsplusplus.user_service import UserService

    # -------------------------------------------------------------------------
    # WaitlistStore
    # -------------------------------------------------------------------------
    async def _wl_init(self, conn=None):
        pass

    async def _wl_pool(self):
        return None

    WaitlistStore._init_schema = _wl_init
    WaitlistStore.get_pool = _wl_pool

    async def upsert_pending_otp(self, email, otp_code, expires_at, source_variant=""):
        H.pending[email] = {
            "email": email,
            "otp_code": otp_code,
            "source_variant": source_variant,
            "expires_at": _iso(expires_at),
            "created_at": _iso(_now()),
        }

    async def get_pending_otp(self, email):
        row = H.pending.get(email)
        if not row:
            return None
        if datetime.fromisoformat(row["expires_at"]) <= _now():
            return None
        return dict(row)

    async def get_pending_otp_any(self, email):
        row = H.pending.get(email)
        return dict(row) if row else None

    async def delete_pending_otp(self, email):
        H.pending.pop(email, None)

    async def get_visitor(self, email):
        vid = H.by_email.get(email)
        return dict(H.visitors[vid]) if vid else None

    async def create_visitor(self, email, source_variant=""):
        vid = H.by_email.get(email)
        if vid:
            v = H.visitors[vid]
            v.setdefault("verified_at", _iso(_now()))
            return dict(v)
        vid = str(uuid.uuid4())
        base = _now()
        ticker = len(H.visitors)
        created = base + timedelta(microseconds=ticker)
        v = {
            "id": vid,
            "email": email,
            "status": "waiting",
            "source_variant": source_variant,
            "verified_at": _iso(base),
            "created_at": _iso(created),
            "invited_at": None,
            "invite_token_hash": None,
            "invite_expires_at": None,
            "activated_at": None,
        }
        H.visitors[vid] = v
        H.by_email[email] = vid
        return dict(v)

    async def count_waiting(self):
        return sum(
            1 for v in H.visitors.values() if v["status"] in ("waiting", "invited")
        )

    async def get_position(self, email):
        vid = H.by_email.get(email)
        if not vid:
            return None
        v = H.visitors[vid]
        if v["status"] not in ("waiting", "invited"):
            return None
        mc = v["created_at"]
        return (
            sum(
                1
                for w in H.visitors.values()
                if w["status"] in ("waiting", "invited") and w["created_at"] < mc
            )
            + 1
        )

    async def get_visitor_by_invite_hash(self, token_hash):
        for v in H.visitors.values():
            if (
                v.get("invite_token_hash") == token_hash
                and v["status"] == "invited"
                and v.get("invite_expires_at")
                and datetime.fromisoformat(v["invite_expires_at"]) > _now()
            ):
                return dict(v)
        return None

    async def get_oldest_waiting(self, limit=1):
        waiters = [v for v in H.visitors.values() if v["status"] == "waiting"]
        waiters.sort(key=lambda w: w["created_at"])
        return [dict(x) for x in waiters[:limit]]

    async def mark_invited(self, visitor_id, token_hash, expires_at):
        v = H.visitors.get(visitor_id)
        if v:
            v["status"] = "invited"
            v["invited_at"] = _iso(_now())
            v["invite_token_hash"] = token_hash
            v["invite_expires_at"] = _iso(expires_at)

    async def mark_activated(self, visitor_id):
        v = H.visitors.get(visitor_id)
        if v:
            v["status"] = "activated"
            v["activated_at"] = _iso(_now())
            v["invite_token_hash"] = None

    async def expire_stale_invites(self):
        n = 0
        for v in H.visitors.values():
            if v["status"] == "invited" and v.get("invite_expires_at"):
                if datetime.fromisoformat(v["invite_expires_at"]) < _now():
                    v["status"] = "waiting"
                    v["invite_token_hash"] = None
                    v["invite_expires_at"] = None
                    n += 1
        return n

    async def list_all(self, limit=200):
        rows = sorted(
            H.visitors.values(), key=lambda v: v["created_at"], reverse=True
        )
        return [dict(r) for r in rows[:limit]]

    WaitlistStore.upsert_pending_otp = upsert_pending_otp
    WaitlistStore.get_pending_otp = get_pending_otp
    WaitlistStore.get_pending_otp_any = get_pending_otp_any
    WaitlistStore.delete_pending_otp = delete_pending_otp
    WaitlistStore.get_visitor = get_visitor
    WaitlistStore.create_visitor = create_visitor
    WaitlistStore.count_waiting = count_waiting
    WaitlistStore.get_position = get_position
    WaitlistStore.get_visitor_by_invite_hash = get_visitor_by_invite_hash
    WaitlistStore.get_oldest_waiting = get_oldest_waiting
    WaitlistStore.mark_invited = mark_invited
    WaitlistStore.mark_activated = mark_activated
    WaitlistStore.expire_stale_invites = expire_stale_invites
    WaitlistStore.list_all = list_all

    # -------------------------------------------------------------------------
    # UserStore
    # -------------------------------------------------------------------------
    async def _us_init(self, conn=None):
        pass

    async def _us_pool(self):
        return None

    UserStore._init_schema = _us_init
    UserStore.get_pool = _us_pool

    async def get_by_email(self, email):
        uid = H.users_by_email.get(email)
        return dict(H.users[uid]) if uid else None

    async def get_by_id(self, user_id):
        return dict(H.users[user_id]) if user_id in H.users else None

    async def get_by_google_id(self, google_id):
        return None

    async def count_users(self):
        return len(H.users)

    async def count_users_by_tier(self):
        return {"free": len(H.users)}

    async def create_user(
        self, email, password_hash=None, google_id=None, name="", avatar_url=None
    ):
        uid = str(uuid.uuid4())
        u = {
            "id": uid,
            "email": email,
            "password_hash": password_hash,
            "google_id": google_id,
            "name": name,
            "avatar_url": avatar_url,
            "tier": "free",
            "is_admin": False,
            "created_at": _iso(_now()),
            "updated_at": _iso(_now()),
            "email_verified": False,
        }
        H.users[uid] = u
        H.users_by_email[email] = uid
        return dict(u)

    async def mark_email_verified(self, user_id):
        if user_id in H.users:
            H.users[user_id]["email_verified"] = True

    async def ensure_admin_user(self, email, password_hash, name):
        if email in H.users_by_email:
            uid = H.users_by_email[email]
            H.users[uid]["is_admin"] = True
            H.users[uid]["email_verified"] = True
            return None
        uid = str(uuid.uuid4())
        u = {
            "id": uid,
            "email": email,
            "password_hash": password_hash,
            "google_id": None,
            "name": name,
            "avatar_url": None,
            "tier": "enterprise",
            "is_admin": True,
            "created_at": _iso(_now()),
            "updated_at": _iso(_now()),
            "email_verified": True,
        }
        H.users[uid] = u
        H.users_by_email[email] = uid
        return dict(u)

    async def list_users(self, limit=100, offset=0):
        return list(H.users.values())[offset : offset + limit]

    async def daily_signups(self, days=30):
        return []

    async def update_user(self, user_id, fields):
        if user_id not in H.users:
            return None
        H.users[user_id].update(fields)
        return dict(H.users[user_id])

    async def update_tier(self, user_id, tier):
        if user_id in H.users:
            H.users[user_id]["tier"] = tier
            return True
        return False

    async def delete_user(self, user_id):
        u = H.users.pop(user_id, None)
        if u:
            H.users_by_email.pop(u["email"], None)
            return True
        return False

    async def get_revenue_events(self, limit=100):
        return []

    async def record_revenue_event(
        self, user_id, event_type, from_tier, to_tier, monthly_revenue
    ):
        return {}

    UserStore.get_by_email = get_by_email
    UserStore.get_by_id = get_by_id
    UserStore.get_by_google_id = get_by_google_id
    UserStore.count_users = count_users
    UserStore.count_users_by_tier = count_users_by_tier
    UserStore.create_user = create_user
    UserStore.mark_email_verified = mark_email_verified
    UserStore.ensure_admin_user = ensure_admin_user
    UserStore.list_users = list_users
    UserStore.daily_signups = daily_signups
    UserStore.update_user = update_user
    UserStore.update_tier = update_tier
    UserStore.delete_user = delete_user
    UserStore.get_revenue_events = get_revenue_events
    UserStore.record_revenue_event = record_revenue_event

    # -------------------------------------------------------------------------
    # EmailService — record instead of sending
    # -------------------------------------------------------------------------
    async def send_waitlist_verification(self, to, otp_code):
        H.sent_emails.append({"to": to, "kind": "verification", "otp": otp_code})
        return True

    async def send_waitlist_invite(self, to, accept_link, hours_valid=2):
        H.sent_emails.append(
            {"to": to, "kind": "invite", "link": accept_link, "hours": hours_valid}
        )
        return True

    async def send_verification_email(self, to, otp_code, verify_link):
        H.sent_emails.append(
            {"to": to, "kind": "signup_verification", "otp": otp_code, "link": verify_link}
        )
        return True

    async def send_password_reset_email(self, to, otp_code, reset_link):
        H.sent_emails.append({"to": to, "kind": "password_reset", "otp": otp_code})
        return True

    EmailService.send_waitlist_verification = send_waitlist_verification
    EmailService.send_waitlist_invite = send_waitlist_invite
    EmailService.send_verification_email = send_verification_email
    EmailService.send_password_reset_email = send_password_reset_email

    # -------------------------------------------------------------------------
    # MetricsEmitter
    # -------------------------------------------------------------------------
    async def get_active_now(self, window_seconds=900):
        return H.active_now

    async def get_dau(self, date_str=None):
        return H.dau

    async def record_active_user(self, identity):
        pass

    async def emit(self, user_id, metric, count=1):
        pass

    async def get_aggregate_metrics(self, period=None):
        return {}

    async def get_extension_analytics(self):
        return {"installs_today": 0, "dau": 0, "mau": 0, "wau": 0}

    MetricsEmitter.get_active_now = get_active_now
    MetricsEmitter.get_dau = get_dau
    MetricsEmitter.record_active_user = record_active_user
    MetricsEmitter.emit = emit
    MetricsEmitter.get_aggregate_metrics = get_aggregate_metrics
    MetricsEmitter.get_extension_analytics = get_extension_analytics

    # -------------------------------------------------------------------------
    # Service-level: skip anything that hits real DB at startup
    # -------------------------------------------------------------------------
    async def _ensure_admin_noop(self):
        pass

    async def _preload_noop(self):
        pass

    UserService.ensure_admin = _ensure_admin_noop
    MemoryService.startup_preload = _preload_noop

    return H
