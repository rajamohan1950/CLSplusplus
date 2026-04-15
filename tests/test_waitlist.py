"""End-to-end lifecycle tests for the launch waitlist.

Covers: stats → join → cooldown → verify → position → cap → promote → accept.
Monkey-patches WaitlistStore + UserStore + EmailService + MetricsEmitter so no
DB / Redis / Resend is required. Each test gets a fresh in-memory harness.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from clsplusplus.config import Settings
from clsplusplus.email_service import EmailService
from clsplusplus.metrics import MetricsEmitter
from clsplusplus.stores.user_store import UserStore
from clsplusplus.stores.waitlist_store import WaitlistStore
from clsplusplus.waitlist_service import (
    INVITE_TTL_HOURS,
    OTP_COOLDOWN_SECONDS,
    WaitlistError,
    WaitlistService,
)

JWT_SECRET = "waitlist-test-secret-0123456789abcdef"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else dt


# =============================================================================
# In-memory harness + store monkey-patches
# =============================================================================


class Harness:
    """Fake state for every DB/Redis/email touch the waitlist makes."""

    def __init__(self):
        self.visitors: dict = {}  # id -> visitor dict
        self.by_email: dict = {}  # email -> visitor id
        self.pending: dict = {}  # email -> pending otp row
        self.users: dict = {}  # id -> user dict
        self.users_by_email: dict = {}  # email -> user id
        self.sent_emails: list = []  # [{to, kind, ...}]
        self.active_now = 0  # what MetricsEmitter.get_active_now returns
        self.dau = 0  # what MetricsEmitter.get_dau returns

    def reset(self):
        self.__init__()


@pytest.fixture
def harness(monkeypatch):
    H = Harness()
    _install_patches(monkeypatch, H)
    return H


def _install_patches(monkeypatch, H: Harness) -> None:
    # ---- WaitlistStore: bypass DB pool entirely ----
    async def _wl_init(self, conn=None):
        pass

    async def _wl_pool(self):
        return None

    monkeypatch.setattr(WaitlistStore, "_init_schema", _wl_init)
    monkeypatch.setattr(WaitlistStore, "get_pool", _wl_pool)

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
        exp = datetime.fromisoformat(row["expires_at"])
        if exp <= _now():
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
        # Force strictly-increasing created_at so positions are deterministic
        # even when multiple rows are created in the same microsecond.
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
        my_created = v["created_at"]
        count_before = sum(
            1
            for w in H.visitors.values()
            if w["status"] in ("waiting", "invited") and w["created_at"] < my_created
        )
        return count_before + 1

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
        return [dict(w) for w in waiters[:limit]]

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

    monkeypatch.setattr(WaitlistStore, "upsert_pending_otp", upsert_pending_otp)
    monkeypatch.setattr(WaitlistStore, "get_pending_otp", get_pending_otp)
    monkeypatch.setattr(WaitlistStore, "get_pending_otp_any", get_pending_otp_any)
    monkeypatch.setattr(WaitlistStore, "delete_pending_otp", delete_pending_otp)
    monkeypatch.setattr(WaitlistStore, "get_visitor", get_visitor)
    monkeypatch.setattr(WaitlistStore, "create_visitor", create_visitor)
    monkeypatch.setattr(WaitlistStore, "count_waiting", count_waiting)
    monkeypatch.setattr(WaitlistStore, "get_position", get_position)
    monkeypatch.setattr(
        WaitlistStore, "get_visitor_by_invite_hash", get_visitor_by_invite_hash
    )
    monkeypatch.setattr(WaitlistStore, "get_oldest_waiting", get_oldest_waiting)
    monkeypatch.setattr(WaitlistStore, "mark_invited", mark_invited)
    monkeypatch.setattr(WaitlistStore, "mark_activated", mark_activated)
    monkeypatch.setattr(WaitlistStore, "expire_stale_invites", expire_stale_invites)

    # ---- UserStore: same treatment ----
    async def _us_init(self, conn=None):
        pass

    async def _us_pool(self):
        return None

    monkeypatch.setattr(UserStore, "_init_schema", _us_init)
    monkeypatch.setattr(UserStore, "get_pool", _us_pool)

    async def get_by_email(self, email):
        uid = H.users_by_email.get(email)
        return dict(H.users[uid]) if uid else None

    async def get_by_id(self, user_id):
        return dict(H.users[user_id]) if user_id in H.users else None

    async def count_users(self):
        return len(H.users)

    async def create_user(
        self,
        email,
        password_hash=None,
        google_id=None,
        name="",
        avatar_url=None,
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

    monkeypatch.setattr(UserStore, "get_by_email", get_by_email)
    monkeypatch.setattr(UserStore, "get_by_id", get_by_id)
    monkeypatch.setattr(UserStore, "count_users", count_users)
    monkeypatch.setattr(UserStore, "create_user", create_user)
    monkeypatch.setattr(UserStore, "mark_email_verified", mark_email_verified)

    # ---- EmailService: record, don't send ----
    async def send_waitlist_verification(self, to, otp_code):
        H.sent_emails.append({"to": to, "kind": "verification", "otp": otp_code})
        return True

    async def send_waitlist_invite(self, to, accept_link, hours_valid=2):
        H.sent_emails.append(
            {"to": to, "kind": "invite", "link": accept_link, "hours": hours_valid}
        )
        return True

    monkeypatch.setattr(
        EmailService, "send_waitlist_verification", send_waitlist_verification
    )
    monkeypatch.setattr(EmailService, "send_waitlist_invite", send_waitlist_invite)

    # ---- MetricsEmitter: in-memory read-through ----
    async def get_active_now(self, window_seconds=900):
        return H.active_now

    async def get_dau(self, date_str=None):
        return H.dau

    async def record_active_user(self, identity):
        pass

    monkeypatch.setattr(MetricsEmitter, "get_active_now", get_active_now)
    monkeypatch.setattr(MetricsEmitter, "get_dau", get_dau)
    monkeypatch.setattr(MetricsEmitter, "record_active_user", record_active_user)


def _mk_settings(**overrides) -> Settings:
    defaults = dict(
        require_api_key=False,
        jwt_secret=JWT_SECRET,
        max_active_users=5,
        waitlist_queue_seed_offset=47,
        waitlist_active_floor=3,
        waitlist_dau_healthy_threshold=5,
        waitlist_promote_batch=1,
        resend_api_key="x",  # present so EmailService._enabled is True
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _mk_service(settings: Optional[Settings] = None) -> WaitlistService:
    settings = settings or _mk_settings()
    user_store = UserStore(settings)
    return WaitlistService(
        settings,
        user_store=user_store,
        waitlist_store=WaitlistStore(settings),
        email_service=EmailService(settings),
        metrics=MetricsEmitter(settings),
    )


# =============================================================================
# Stats (displayed numbers obey seed offset + active floor)
# =============================================================================


class TestStats:
    @pytest.mark.asyncio
    async def test_WL_001_stats_empty_returns_seeded_baseline(self, harness):
        svc = _mk_service()
        stats = await svc.stats()
        assert stats["waiting_count"] == 47  # 0 real + 47 offset
        assert stats["active_now"] == 3  # 0 real, clamped to floor

    @pytest.mark.asyncio
    async def test_WL_002_stats_respects_active_floor(self, harness):
        harness.active_now = 1
        svc = _mk_service()
        stats = await svc.stats()
        assert stats["active_now"] == 3

    @pytest.mark.asyncio
    async def test_WL_003_stats_passes_through_when_active_above_floor(
        self, harness
    ):
        harness.active_now = 42
        svc = _mk_service()
        assert (await svc.stats())["active_now"] == 42


# =============================================================================
# Join (email → OTP)
# =============================================================================


class TestJoin:
    @pytest.mark.asyncio
    async def test_WL_010_join_rejects_invalid_email(self, harness):
        svc = _mk_service()
        with pytest.raises(WaitlistError):
            await svc.join("not-an-email")

    @pytest.mark.asyncio
    async def test_WL_011_join_rejects_disposable_domain(self, harness):
        svc = _mk_service()
        with pytest.raises(WaitlistError):
            await svc.join("burner@mailinator.com")

    @pytest.mark.asyncio
    async def test_WL_012_join_valid_email_stores_otp_and_sends(self, harness):
        svc = _mk_service()
        result = await svc.join("real@acme.com")
        assert result["pending"] is True
        assert result["email"] == "real@acme.com"
        assert result["email_sent"] is True
        assert "real@acme.com" in harness.pending
        assert len(harness.sent_emails) == 1
        assert harness.sent_emails[0]["kind"] == "verification"
        assert len(harness.sent_emails[0]["otp"]) == 6

    @pytest.mark.asyncio
    async def test_WL_013_join_cooldown_blocks_rapid_second_call(self, harness):
        svc = _mk_service()
        await svc.join("cool@acme.com")
        result = await svc.join("cool@acme.com")
        assert result["status"] == "otp_cooldown"
        # Only one email sent despite two join calls within cooldown
        assert len([e for e in harness.sent_emails if e["to"] == "cool@acme.com"]) == 1

    @pytest.mark.asyncio
    async def test_WL_014_join_normalises_email_case(self, harness):
        svc = _mk_service()
        result = await svc.join("MixedCase@Acme.COM")
        assert result["email"] == "mixedcase@acme.com"

    @pytest.mark.asyncio
    async def test_WL_015_join_returns_already_member_for_existing_user(
        self, harness
    ):
        harness.users["u1"] = {
            "id": "u1",
            "email": "already@acme.com",
            "password_hash": "x",
            "is_admin": False,
        }
        harness.users_by_email["already@acme.com"] = "u1"
        svc = _mk_service()
        result = await svc.join("already@acme.com")
        assert result["status"] == "already_member"
        assert result["pending"] is False

    @pytest.mark.asyncio
    async def test_WL_016_join_returns_position_for_existing_waiter(self, harness):
        svc = _mk_service()
        await svc.join("dup@acme.com")
        await svc.verify("dup@acme.com", harness.sent_emails[0]["otp"])
        # Clear cooldown so join() reaches the existing-waiter branch
        harness.pending.pop("dup@acme.com", None)
        harness.sent_emails.clear()
        again = await svc.join("dup@acme.com")
        assert again["pending"] is False
        assert again["status"] == "waiting"
        assert again["position"] == 48  # 1 real + 47 offset


# =============================================================================
# Verify (OTP → enter queue)
# =============================================================================


class TestVerify:
    @pytest.mark.asyncio
    async def test_WL_020_verify_wrong_otp_fails(self, harness):
        svc = _mk_service()
        await svc.join("a@acme.com")
        with pytest.raises(WaitlistError):
            await svc.verify("a@acme.com", "000000")

    @pytest.mark.asyncio
    async def test_WL_021_verify_non_numeric_otp_fails(self, harness):
        svc = _mk_service()
        await svc.join("a@acme.com")
        with pytest.raises(WaitlistError):
            await svc.verify("a@acme.com", "abcdef")

    @pytest.mark.asyncio
    async def test_WL_022_verify_correct_otp_creates_visitor(self, harness):
        svc = _mk_service()
        await svc.join("a@acme.com")
        otp = harness.sent_emails[0]["otp"]
        result = await svc.verify("a@acme.com", otp)
        assert result["status"] == "waiting"
        assert result["position"] == 48  # offset-aware
        assert result["waiting_count"] == 48
        assert "a@acme.com" in harness.by_email
        assert "a@acme.com" not in harness.pending  # consumed

    @pytest.mark.asyncio
    async def test_WL_023_verify_second_visitor_gets_next_position(self, harness):
        svc = _mk_service()
        await svc.join("first@acme.com")
        otp1 = harness.sent_emails[-1]["otp"]
        await svc.verify("first@acme.com", otp1)

        await svc.join("second@acme.com")
        otp2 = harness.sent_emails[-1]["otp"]
        result = await svc.verify("second@acme.com", otp2)
        assert result["position"] == 49  # 2 real + 47 offset

    @pytest.mark.asyncio
    async def test_WL_024_verify_stats_with_email_returns_your_position(
        self, harness
    ):
        svc = _mk_service()
        await svc.join("mine@acme.com")
        otp = harness.sent_emails[-1]["otp"]
        await svc.verify("mine@acme.com", otp)

        stats = await svc.stats(email="mine@acme.com")
        assert stats["your_position"] == 48
        assert stats["your_status"] == "waiting"
        assert stats["waiting_count"] == 48


# =============================================================================
# Launch cap
# =============================================================================


class TestLaunchCap:
    @pytest.mark.asyncio
    async def test_WL_030_cap_not_exceeded_when_empty(self, harness):
        svc = _mk_service()
        exceeded, current, cap = await svc.is_launch_cap_exceeded()
        assert exceeded is False
        assert current == 0
        assert cap == 5

    @pytest.mark.asyncio
    async def test_WL_031_cap_exceeded_when_at_threshold(self, harness):
        for i in range(5):
            harness.users[f"u{i}"] = {"id": f"u{i}", "email": f"u{i}@x.com"}
            harness.users_by_email[f"u{i}@x.com"] = f"u{i}"
        svc = _mk_service()
        exceeded, current, cap = await svc.is_launch_cap_exceeded()
        assert exceeded is True
        assert current == 5

    @pytest.mark.asyncio
    async def test_WL_032_cap_zero_means_unlimited(self, harness):
        svc = _mk_service(_mk_settings(max_active_users=0))
        for i in range(100):
            harness.users[f"u{i}"] = {"id": f"u{i}", "email": f"u{i}@x.com"}
            harness.users_by_email[f"u{i}@x.com"] = f"u{i}"
        exceeded, _, _ = await svc.is_launch_cap_exceeded()
        assert exceeded is False


# =============================================================================
# Promote (daily tick → invite)
# =============================================================================


class TestPromote:
    @pytest.mark.asyncio
    async def test_WL_040_promote_empty_queue_is_noop(self, harness):
        svc = _mk_service()
        invited = await svc.promote(batch=1)
        assert invited == []

    @pytest.mark.asyncio
    async def test_WL_041_promote_moves_oldest_waiting_to_invited(self, harness):
        svc = _mk_service()
        # Create 3 waiters, oldest first
        for email in ["first@x.com", "second@x.com", "third@x.com"]:
            await svc.join(email)
            otp = harness.sent_emails[-1]["otp"]
            await svc.verify(email, otp)

        harness.sent_emails.clear()
        invited = await svc.promote(batch=1)
        assert invited == ["first@x.com"]
        # Oldest now has status=invited with a fresh token hash + expiry
        first = harness.visitors[harness.by_email["first@x.com"]]
        assert first["status"] == "invited"
        assert first["invite_token_hash"] is not None
        assert first["invite_expires_at"] is not None
        # Invite email went out
        assert len(harness.sent_emails) == 1
        assert harness.sent_emails[0]["kind"] == "invite"

    @pytest.mark.asyncio
    async def test_WL_042_daily_tick_skips_when_dau_healthy(self, harness):
        svc = _mk_service()
        await svc.join("slow@x.com")
        await svc.verify("slow@x.com", harness.sent_emails[-1]["otp"])
        harness.dau = 10  # above threshold of 5
        result = await svc.daily_promote_tick()
        assert result["invited"] == []

    @pytest.mark.asyncio
    async def test_WL_043_daily_tick_promotes_when_dau_unhealthy(self, harness):
        svc = _mk_service()
        await svc.join("lucky@x.com")
        await svc.verify("lucky@x.com", harness.sent_emails[-1]["otp"])
        harness.dau = 1  # below threshold of 5
        harness.sent_emails.clear()
        result = await svc.daily_promote_tick()
        assert "lucky@x.com" in result["invited"]

    @pytest.mark.asyncio
    async def test_WL_044_stale_invites_expire_back_to_waiting(self, harness):
        svc = _mk_service()
        await svc.join("stale@x.com")
        await svc.verify("stale@x.com", harness.sent_emails[-1]["otp"])
        harness.dau = 1
        await svc.daily_promote_tick()

        # Force the invite into the past
        vid = harness.by_email["stale@x.com"]
        harness.visitors[vid]["invite_expires_at"] = _iso(
            _now() - timedelta(hours=1)
        )

        harness.dau = 10  # healthy → promoter won't re-invite
        result = await svc.daily_promote_tick()
        assert result["stale_invites_reset"] == 1
        assert harness.visitors[vid]["status"] == "waiting"
        assert harness.visitors[vid]["invite_token_hash"] is None


# =============================================================================
# Activate (token → real user + JWT)
# =============================================================================


class TestActivate:
    @pytest.mark.asyncio
    async def test_WL_050_activate_with_missing_token_fails(self, harness):
        svc = _mk_service()
        with pytest.raises(WaitlistError):
            await svc.activate_by_token("")

    @pytest.mark.asyncio
    async def test_WL_051_activate_with_unknown_token_fails(self, harness):
        svc = _mk_service()
        with pytest.raises(WaitlistError):
            await svc.activate_by_token("bogus-token-does-not-exist")

    @pytest.mark.asyncio
    async def test_WL_052_activate_happy_path_creates_user_and_jwt(self, harness):
        svc = _mk_service()
        await svc.join("winner@acme.com")
        await svc.verify("winner@acme.com", harness.sent_emails[-1]["otp"])
        harness.dau = 1
        harness.sent_emails.clear()

        # Run the promoter and capture the raw token from the invite email
        await svc.daily_promote_tick()
        invite = [e for e in harness.sent_emails if e["kind"] == "invite"][0]
        raw_token = invite["link"].rsplit("token=", 1)[1]

        user, jwt_token = await svc.activate_by_token(raw_token)
        assert user["email"] == "winner@acme.com"
        assert user["email_verified"] is True
        assert jwt_token  # opaque JWT string
        # Visitor is now activated
        assert (
            harness.visitors[harness.by_email["winner@acme.com"]]["status"]
            == "activated"
        )
        # User row exists
        assert "winner@acme.com" in harness.users_by_email

    @pytest.mark.asyncio
    async def test_WL_053_activate_bypasses_launch_cap(self, harness):
        """A user at the cap can still activate from the waitlist."""
        # Fill the cap with unrelated users
        for i in range(5):
            harness.users[f"u{i}"] = {"id": f"u{i}", "email": f"u{i}@x.com"}
            harness.users_by_email[f"u{i}@x.com"] = f"u{i}"

        svc = _mk_service()
        await svc.join("lucky@acme.com")
        await svc.verify("lucky@acme.com", harness.sent_emails[-1]["otp"])
        harness.dau = 1
        harness.sent_emails.clear()
        await svc.daily_promote_tick()
        raw_token = [
            e for e in harness.sent_emails if e["kind"] == "invite"
        ][0]["link"].rsplit("token=", 1)[1]

        user, _ = await svc.activate_by_token(raw_token)
        assert user["email"] == "lucky@acme.com"
        assert "lucky@acme.com" in harness.users_by_email
        # Count is now 6 — cap was bypassed on purpose
        assert len(harness.users) == 6

    @pytest.mark.asyncio
    async def test_WL_054_activate_expired_token_fails(self, harness):
        svc = _mk_service()
        await svc.join("slowpoke@acme.com")
        await svc.verify("slowpoke@acme.com", harness.sent_emails[-1]["otp"])
        harness.dau = 1
        harness.sent_emails.clear()
        await svc.daily_promote_tick()
        raw_token = [
            e for e in harness.sent_emails if e["kind"] == "invite"
        ][0]["link"].rsplit("token=", 1)[1]

        # Force expire in storage
        vid = harness.by_email["slowpoke@acme.com"]
        harness.visitors[vid]["invite_expires_at"] = _iso(
            _now() - timedelta(hours=1)
        )
        with pytest.raises(WaitlistError):
            await svc.activate_by_token(raw_token)


# =============================================================================
# HTTP-level smoke tests (via ASGI)
# =============================================================================


class TestHttpEndpoints:
    @pytest.mark.asyncio
    async def test_WL_060_stats_endpoint_returns_seeded_baseline(self, harness):
        from httpx import ASGITransport, AsyncClient
        from clsplusplus.api import create_app

        app = create_app(_mk_settings())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v1/waitlist/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["waiting_count"] == 47
        assert data["active_now"] == 3

    @pytest.mark.asyncio
    async def test_WL_061_join_endpoint_sends_otp(self, harness):
        from httpx import ASGITransport, AsyncClient
        from clsplusplus.api import create_app

        app = create_app(_mk_settings())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/waitlist/join", json={"email": "http@acme.com"}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pending"] is True
        assert body["email"] == "http@acme.com"
        assert any(e["to"] == "http@acme.com" for e in harness.sent_emails)

    @pytest.mark.asyncio
    async def test_WL_062_verify_endpoint_enters_queue(self, harness):
        from httpx import ASGITransport, AsyncClient
        from clsplusplus.api import create_app

        app = create_app(_mk_settings())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/v1/waitlist/join", json={"email": "v@acme.com"})
            otp = [e for e in harness.sent_emails if e["to"] == "v@acme.com"][-1][
                "otp"
            ]
            resp = await ac.post(
                "/v1/waitlist/verify", json={"email": "v@acme.com", "otp_code": otp}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "waiting"
        assert body["position"] == 48

    @pytest.mark.asyncio
    async def test_WL_063_join_invalid_email_returns_400(self, harness):
        from httpx import ASGITransport, AsyncClient
        from clsplusplus.api import create_app

        app = create_app(_mk_settings())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/waitlist/join", json={"email": "nope"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_WL_064_register_blocked_when_cap_reached(self, harness):
        from httpx import ASGITransport, AsyncClient
        from clsplusplus.api import create_app

        # Pre-seed 5 users to hit the cap of 5
        for i in range(5):
            harness.users[f"u{i}"] = {"id": f"u{i}", "email": f"u{i}@x.com"}
            harness.users_by_email[f"u{i}@x.com"] = f"u{i}"

        app = create_app(_mk_settings())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/auth/register",
                json={
                    "email": "walkin@acme.com",
                    "password": "supersecret123",
                    "name": "Walk In",
                },
            )
        assert resp.status_code == 503
        body = resp.json()
        assert body.get("waitlist") is True
        assert body.get("cap") == 5

    @pytest.mark.asyncio
    async def test_WL_065_accept_endpoint_redirects_and_sets_cookie(self, harness):
        from httpx import ASGITransport, AsyncClient
        from clsplusplus.api import create_app

        app = create_app(_mk_settings())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as ac:
            # Full lifecycle: join → verify → promote → accept
            await ac.post("/v1/waitlist/join", json={"email": "e2e@acme.com"})
            otp = [e for e in harness.sent_emails if e["to"] == "e2e@acme.com"][-1][
                "otp"
            ]
            await ac.post(
                "/v1/waitlist/verify",
                json={"email": "e2e@acme.com", "otp_code": otp},
            )
            # Force promote
            harness.dau = 1
            svc = _mk_service(_mk_settings())
            await svc.daily_promote_tick()
            token = [
                e
                for e in harness.sent_emails
                if e["to"] == "e2e@acme.com" and e["kind"] == "invite"
            ][-1]["link"].rsplit("token=", 1)[1]

            resp = await ac.get(f"/v1/waitlist/accept?token={token}")

        assert resp.status_code in (302, 307)
        assert "/dashboard.html" in resp.headers["location"]
        assert "cls_session" in resp.headers.get("set-cookie", "")
