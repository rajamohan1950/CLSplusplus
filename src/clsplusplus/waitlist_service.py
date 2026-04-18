"""Waitlist service — join, verify, promote, activate.

Design:
- join(email)          → validates, stores OTP, sends verification email
- verify(email, otp)   → moves email into waitlist_visitors (status='waiting')
- stats()              → {waiting, active_now, displayed_waiting, displayed_active}
- promote(n)           → oldest N 'waiting' → 'invited' + 2h magic link email
- activate(token, pw)  → creates user, bypasses OTP + launch cap, returns JWT
- daily_promote_tick() → called by background loop; auto-promote if DAU is healthy

The "queue seed offset" gives launch day visitors a plausible-looking position
without actually faking individual rows. The "active floor" clamps the live
counter upward so it never reads 0 at 3am and kills social proof.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.email_service import EmailService
from clsplusplus.jwt_utils import create_token
from clsplusplus.metrics import MetricsEmitter
from clsplusplus.stores.user_store import UserStore
from clsplusplus.stores.waitlist_store import WaitlistStore

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

# Common disposable / throwaway email providers — not exhaustive, just the ones
# abusers reach for first. Kept inline so there's no external dep.
_DISPOSABLE_DOMAINS = frozenset({
    "mailinator.com", "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "10minutemail.com", "10minutemail.net", "yopmail.com", "yopmail.net",
    "throwawaymail.com", "trashmail.com", "fakeinbox.com", "maildrop.cc",
    "temp-mail.org", "tempmail.net", "tempmail.com", "tmpmail.org",
    "dispostable.com", "getnada.com", "sharklasers.com", "spamgourmet.com",
    "mintemail.com", "mohmal.com", "inboxkitten.com", "fakemail.net",
    "mailcatch.com", "spambog.com", "mytemp.email", "emailondeck.com",
    "mailnesia.com", "mvrht.net", "jetable.org", "anonbox.net",
})

OTP_TTL_MINUTES = 15
OTP_COOLDOWN_SECONDS = 60
INVITE_TTL_HOURS = 2


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _generate_otp() -> str:
    return f"{random.randint(100000, 999999)}"


class WaitlistError(ValueError):
    """User-facing validation errors (safe to return in API responses)."""


class WaitlistService:
    def __init__(
        self,
        settings: Settings,
        user_store: UserStore,
        waitlist_store: Optional[WaitlistStore] = None,
        email_service: Optional[EmailService] = None,
        metrics: Optional[MetricsEmitter] = None,
    ):
        self.settings = settings
        self.user_store = user_store
        self.store = waitlist_store or WaitlistStore(settings)
        self.email = email_service or EmailService(settings)
        self.metrics = metrics or MetricsEmitter(settings)

    # =========================================================================
    # Validation helpers
    # =========================================================================

    def _validate_email(self, email: str) -> str:
        email = (email or "").strip().lower()
        if not _EMAIL_RE.match(email):
            raise WaitlistError("Please enter a valid email address")
        domain = email.split("@", 1)[1]
        if domain in _DISPOSABLE_DOMAINS:
            raise WaitlistError(
                "Disposable email addresses aren't accepted. Please use a real inbox."
            )
        return email

    # =========================================================================
    # Join (step 1: email → OTP)
    # =========================================================================

    async def join(
        self, email: str, source_variant: str = "", base_url: str = ""
    ) -> dict:
        """Send a verification OTP. Returns {pending: True, email}.

        Idempotent for already-registered emails (returns status='already_member').
        Enforces a 60s cooldown per email.
        """
        email = self._validate_email(email)

        # Already a full user? Don't re-waitlist them.
        existing_user = await self.user_store.get_by_email(email)
        if existing_user:
            return {"pending": False, "email": email, "status": "already_member"}

        # Already in waitlist? Tell them their position.
        visitor = await self.store.get_visitor(email)
        if visitor and visitor.get("status") in ("waiting", "invited", "activated"):
            pos = await self.store.get_position(email)
            return {
                "pending": False,
                "email": email,
                "status": visitor["status"],
                "position": self._display_position(pos) if pos else None,
            }

        # Hard queue cap — refuse if >= waitlist_queue_limit (default 50).
        queue_limit = int(getattr(self.settings, "waitlist_queue_limit", 50))
        if queue_limit > 0:
            current = int(await self.store.count_waiting())
            if current >= queue_limit:
                raise WaitlistError(
                    f"Waitlist is full ({current}/{queue_limit}). Check back soon."
                )

        # Cooldown: don't spam OTPs if the user hits "Join" repeatedly
        prev = await self.store.get_pending_otp_any(email)
        if prev:
            try:
                prev_created = datetime.fromisoformat(prev["created_at"])
            except Exception:
                prev_created = None
            if prev_created and (_now() - prev_created).total_seconds() < OTP_COOLDOWN_SECONDS:
                return {
                    "pending": True,
                    "email": email,
                    "status": "otp_cooldown",
                    "cooldown_seconds": OTP_COOLDOWN_SECONDS,
                }

        otp = _generate_otp()
        expires = _now() + timedelta(minutes=OTP_TTL_MINUTES)
        await self.store.upsert_pending_otp(email, otp, expires, source_variant or "")

        # Send email (non-fatal on failure so caller sees the problem in logs)
        email_sent = False
        email_error: Optional[str] = None
        try:
            email_sent = await self.email.send_waitlist_verification(email, otp)
        except Exception as e:
            email_error = f"{type(e).__name__}: {e}"
            logger.warning("Waitlist OTP email failed for %s: %s", email, e)

        result = {"pending": True, "email": email, "email_sent": email_sent}
        if not email_sent:
            result["email_configured"] = self.email._enabled
            if email_error:
                result["email_error"] = email_error
        return result

    # =========================================================================
    # Verify (step 2: OTP → enter queue)
    # =========================================================================

    async def verify(self, email: str, otp_code: str) -> dict:
        email = self._validate_email(email)
        otp_code = (otp_code or "").strip()
        if not otp_code or len(otp_code) != 6 or not otp_code.isdigit():
            raise WaitlistError("Enter the 6-digit code from your email")

        pending = await self.store.get_pending_otp(email)
        if not pending or pending["otp_code"] != otp_code:
            raise WaitlistError("Invalid or expired verification code")

        source = pending.get("source_variant") or ""
        visitor = await self.store.create_visitor(email, source)
        await self.store.delete_pending_otp(email)

        raw_position = await self.store.get_position(email) or 1
        waiting_count = await self.store.count_waiting()
        # Fire-and-forget join email + record the notified position so the
        # position-drop watcher knows the baseline.
        try:
            await self.email.send_waitlist_position_update(
                to=email,
                position=raw_position,
                queue_length=waiting_count,
                is_join=True,
            )
            await self._set_last_notified_position(email, raw_position)
        except Exception as e:  # noqa: BLE001 — email failure must not block join
            logger.warning("Waitlist join email failed for %s: %s", email, e)

        return {
            "email": email,
            "status": visitor["status"],
            "position": int(raw_position),
            "waiting_count": int(waiting_count),
        }

    async def _set_last_notified_position(self, email: str, pos: int) -> None:
        try:
            pool = await self.store.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE waitlist_visitors SET last_notified_position = $1 WHERE email = $2",
                    int(pos),
                    email,
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("could not set last_notified_position: %s", e)

    async def notify_position_drops(self, threshold: int = 10) -> int:
        """For every 'waiting' visitor, send a move-up email if their real
        position is at least `threshold` places better than the last one they
        were notified about. Returns the count of emails sent. Intended to
        run from a scheduled job every few minutes.
        """
        sent = 0
        try:
            pool = await self.store.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    WITH ranked AS (
                        SELECT email,
                               last_notified_position,
                               ROW_NUMBER() OVER (ORDER BY created_at) AS pos
                        FROM waitlist_visitors
                        WHERE status IN ('waiting', 'invited')
                    )
                    SELECT email, pos, last_notified_position
                    FROM ranked
                    WHERE last_notified_position IS NOT NULL
                      AND last_notified_position - pos >= $1
                    """,
                    int(threshold),
                )
            waiting_count = await self.store.count_waiting()
            for row in rows:
                try:
                    await self.email.send_waitlist_position_update(
                        to=row["email"],
                        position=int(row["pos"]),
                        queue_length=int(waiting_count),
                        is_join=False,
                    )
                    await self._set_last_notified_position(row["email"], int(row["pos"]))
                    sent += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("position-drop email failed for %s: %s", row["email"], e)
        except Exception as e:  # noqa: BLE001
            logger.warning("notify_position_drops failed: %s", e)
        return sent

    # =========================================================================
    # Public stats (for the landing widget)
    # =========================================================================

    async def stats(self, email: Optional[str] = None) -> dict:
        """Real queue stats — NO seeding, NO floors, NO fake numbers.

        Semantics (agreed with product):
          active_now    = distinct users with at least one api_credentials
                          row with status='active' (= has a working key)
          waiting_count = real waitlist_visitors rows with status='waiting'
          pending_count = pending_registrations with expires_at > NOW()
                          (submitted /auth/register but haven't verified OTP)
          queue_limit   = hard cap on waitlist length (default 50)
          queue_full    = waiting_count >= queue_limit
        """
        waiting_count = int(await self.store.count_waiting())
        active_now = int(await self._count_active_api_users())

        pending_count = 0
        try:
            pool = await self.store.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT COUNT(*)::int AS c FROM pending_registrations "
                    "WHERE expires_at > NOW()"
                )
                pending_count = int(row["c"] if row else 0)
        except Exception as e:  # noqa: BLE001
            logger.debug("pending count unavailable: %s", e)

        queue_limit = int(getattr(self.settings, "waitlist_queue_limit", 50))

        result = {
            "active_now": active_now,
            "waiting_count": waiting_count,
            "pending_count": pending_count,
            "queue_limit": queue_limit,
            "queue_full": waiting_count >= queue_limit,
        }
        if email:
            try:
                email_norm = self._validate_email(email)
                pos = await self.store.get_position(email_norm)
                if pos:
                    result["your_position"] = int(pos)
                    visitor = await self.store.get_visitor(email_norm) or {}
                    result["your_status"] = visitor.get("status")
            except WaitlistError:
                pass
        return result

    async def _count_active_api_users(self) -> int:
        """Count distinct user namespaces that have at least one active key.

        The users table stores profile data; API credentials are owned by
        integrations keyed on namespace=f"user-{uid[:8]}". A user is
        "active" iff any integration of theirs has a live api_credentials
        row. This is the only real-world signal we have for "this account
        is wired up and usable".
        """
        try:
            pool = await self.store.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT COUNT(DISTINCT i.namespace)::int AS c
                    FROM api_credentials c
                    JOIN integrations i ON i.id = c.integration_id
                    WHERE c.status = 'active'
                      AND i.namespace LIKE 'user-%'
                    """
                )
                return int(row["c"] if row else 0)
        except Exception as e:  # noqa: BLE001
            logger.debug("active-api-users count unavailable: %s", e)
            return 0

    def _display_position(self, raw_position: int) -> int:
        return int(raw_position)

    # =========================================================================
    # Promotion (background task triggers this)
    # =========================================================================

    async def promote(
        self, batch: int = 1, base_url: str = ""
    ) -> list[str]:
        """Move oldest N 'waiting' → 'invited', email each a magic link.

        Returns the list of emails that were invited (for logging).
        """
        invited: list[str] = []
        oldest = await self.store.get_oldest_waiting(limit=batch)
        if not oldest:
            return invited

        base = (base_url or self.settings.site_base_url).rstrip("/")
        for v in oldest:
            raw_token = secrets.token_urlsafe(48)
            token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
            expires = _now() + timedelta(hours=INVITE_TTL_HOURS)
            await self.store.mark_invited(v["id"], token_hash, expires)

            accept_link = f"{base}/v1/waitlist/accept?token={raw_token}"
            try:
                await self.email.send_waitlist_invite(
                    to=v["email"],
                    accept_link=accept_link,
                    hours_valid=INVITE_TTL_HOURS,
                )
                invited.append(v["email"])
                logger.info("Waitlist invite sent to %s", v["email"])
            except Exception as e:
                logger.warning("Waitlist invite email failed for %s: %s", v["email"], e)
        return invited

    async def daily_promote_tick(self, base_url: str = "") -> dict:
        """Called by the background loop. Promotes iff DAU is below healthy threshold.

        Also sweeps expired invites back to 'waiting' so nothing gets stuck.
        """
        stale_reset = await self.store.expire_stale_invites()

        dau = await self.metrics.get_dau()
        healthy = int(getattr(self.settings, "waitlist_dau_healthy_threshold", 5))
        batch = int(getattr(self.settings, "waitlist_promote_batch", 1))

        invited: list[str] = []
        if dau < healthy:
            invited = await self.promote(batch=batch, base_url=base_url)

        # Every tick, also sweep for users whose position has dropped by 10
        # or more since we last emailed them.
        position_emails = await self.notify_position_drops(threshold=10)

        return {
            "dau": dau,
            "healthy_threshold": healthy,
            "invited": invited,
            "stale_invites_reset": stale_reset,
            "position_emails_sent": position_emails,
        }

    # =========================================================================
    # Activate (magic link → real user)
    # =========================================================================

    async def activate_by_token(self, raw_token: str) -> tuple[dict, str]:
        """Create the user account from a valid invite token.

        Bypasses:
          - OTP (email already verified at waitlist join)
          - The launch rate-limit cap on /v1/auth/register

        Generates a long random password. The user is signed in via JWT cookie
        and can set their real password from the profile page.

        Returns (user_dict, jwt_token). Raises WaitlistError on bad/expired token.
        """
        if not raw_token:
            raise WaitlistError("Missing invite token")

        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        visitor = await self.store.get_visitor_by_invite_hash(token_hash)
        if not visitor:
            raise WaitlistError("This invite link is invalid or has expired")

        email = visitor["email"]

        # Race: someone already claimed this email via the normal signup
        existing = await self.user_store.get_by_email(email)
        if existing:
            await self.store.mark_activated(visitor["id"])
            token = create_token(
                user_id=existing["id"],
                email=existing["email"],
                is_admin=existing.get("is_admin", False),
                secret=self.settings.jwt_secret,
            )
            return existing, token

        # Generate a random password — user can reset later via profile page.
        import bcrypt
        random_pw = secrets.token_urlsafe(32)
        password_hash = bcrypt.hashpw(
            random_pw.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        user = await self.user_store.create_user(
            email=email,
            password_hash=password_hash,
            name=email.split("@")[0],
        )
        await self.user_store.mark_email_verified(user["id"])
        user["email_verified"] = True

        await self.store.mark_activated(visitor["id"])

        token = create_token(
            user_id=user["id"],
            email=user["email"],
            is_admin=user.get("is_admin", False),
            secret=self.settings.jwt_secret,
        )
        user.pop("password_hash", None)
        return user, token

    # =========================================================================
    # Launch cap check (called from /v1/auth/register)
    # =========================================================================

    async def is_launch_cap_exceeded(self) -> tuple[bool, int, int]:
        """Returns (exceeded, current_active_users, cap).

        "Current" is users with at least one active API key (see
        _count_active_api_users). Signed-up-but-idle users do NOT consume
        a seat; the cap guards capacity, not accounts.
        """
        cap = int(getattr(self.settings, "max_active_users", 0))
        if cap <= 0:
            return False, 0, 0
        current = int(await self._count_active_api_users())
        return current >= cap, current, cap
