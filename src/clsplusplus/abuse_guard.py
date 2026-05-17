"""CLS++ abuse guard — Redis-backed blocklist + cheap heuristic abuse detection.

MVP, not a WAF. Goals:
  * Detect dubious bot/attack traffic and auto-blocklist the offending
    API key or IP fast.
  * Every check is an O(1) Redis op so it can run on the hot path.
  * NEVER raise into a request — every public function fails OPEN
    (allow) on any internal/Redis error, and logs.

Redis keys:
  cls:blocked:{identifier}        -> reason string,    TTL = abuse_block_ttl_seconds
  cls:abuse:authfail:ip:{host}    -> int counter,      TTL = 300s   (5 min)
  cls:abuse:burst:{identifier}    -> int counter,      TTL = 10s
  cls:abuse:scan:ip:{host}        -> int counter,      TTL = 300s   (5 min)
  cls:abuse:bad:ip:{host}         -> int counter,      TTL = 300s   (5 min)

`identifier` is either `ip:{host}` or `key:{sha256(api_key)[:16]}`.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
from typing import Optional

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

# Reuse the cached-client pattern from rate_limit.py.
_redis_client_cache: dict[str, object] = {}


def _redis_client(redis_url: str):
    """Lazy import; reuse one async connection per URL."""
    import redis.asyncio as redis
    if redis_url not in _redis_client_cache:
        _redis_client_cache[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _redis_client_cache[redis_url]


# --- key helpers -----------------------------------------------------------

_BLOCK_PREFIX = "cls:blocked:"
_AUTHFAIL_PREFIX = "cls:abuse:authfail:"
_BURST_PREFIX = "cls:abuse:burst:"
_SCAN_PREFIX = "cls:abuse:scan:"
_BAD_PREFIX = "cls:abuse:bad:"

# Window TTLs (seconds) for the short-lived signal counters.
_AUTHFAIL_WINDOW = 300   # 5 min — credential-stuffing / probing
_BURST_WINDOW = 10       # 10 s  — request burst
_SCAN_WINDOW = 300       # 5 min — path scanning
_BAD_WINDOW = 300        # 5 min — malformed requests

# Thresholds for the signals that have no config knob.
_SCAN_THRESHOLD = 8      # suspicious-path hits per IP per 5 min
_BAD_THRESHOLD = 30      # malformed requests per IP per 5 min

# Obvious attack-probe path fragments. Substring match, lowercased.
# Kept small and deliberate — this is an MVP, not exhaustive.
_SUSPICIOUS_FRAGMENTS = (
    "/wp-login.php",
    "/wp-admin",
    "/xmlrpc.php",
    "/.env",
    "/.git/",
    "/admin.php",
    "/phpmyadmin",
    "/phpunit",
    "/vendor/phpunit",
    "/.aws/credentials",
    "/config.php",
    "/shell.php",
    "/cgi-bin/",
    "/.ssh/",
    "/owa/auth",
    "/solr/",
    "/actuator/env",
    "/eval-stdin.php",
)


def hash_api_key(api_key: str) -> str:
    """Stable short identifier for an API key — never store raw keys."""
    return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]


def is_suspicious_path(path: str) -> bool:
    """True when the path looks like an attack probe."""
    p = path.lower()
    return any(frag in p for frag in _SUSPICIOUS_FRAGMENTS)


def client_ip(request) -> str:
    """Best-effort real client IP, honoring X-Forwarded-For behind a proxy.

    Render (and most PaaS) terminate TLS at a proxy, so `request.client.host`
    is the proxy's PRIVATE IP — identical for every external user. Keying
    abuse/rate-limit counters on that lumps all traffic onto one IP, which
    instantly trips burst detection and blocklists the whole platform. The
    real client is the first hop of X-Forwarded-For.
    """
    try:
        xff = request.headers.get("x-forwarded-for")
        if xff and isinstance(xff, str):
            first = xff.split(",")[0].strip()
            if first:
                return first
    except Exception:  # noqa: BLE001 — never raise into a request
        pass
    try:
        return request.client.host if request.client else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def is_infra_ip(host: str) -> bool:
    """True for private/loopback/link-local IPs and unknown hosts.

    This is platform traffic — load balancers, health checks, internal
    probes — which must never be abuse-gated or counted. Behind Render the
    health checker reaches the app on a private 10.x address.
    """
    if not host or host == "unknown":
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


# Endpoints that must NEVER be abuse-gated — health probes above all.
# A 403 on a health endpoint makes the platform mark the service unhealthy.
_ABUSE_EXEMPT_PREFIXES = (
    "/health",
    "/v1/health",
    "/v1/memory/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def is_abuse_exempt_path(path: str) -> bool:
    """True for health/infra/doc paths that must never be blocked or counted."""
    if path in ("", "/"):
        return True
    return any(path == p or path.startswith(p) for p in _ABUSE_EXEMPT_PREFIXES)


# --- blocklist primitives --------------------------------------------------

async def is_blocked(identifier: str, settings: Optional[Settings] = None) -> bool:
    """True if the identifier is currently blocklisted. Fails OPEN on error."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        return bool(await client.exists(_BLOCK_PREFIX + identifier))
    except Exception as exc:  # noqa: BLE001 — never raise into a request
        logger.warning("abuse_guard.is_blocked failed (%s: %s)", type(exc).__name__, exc)
        return False


async def block(
    identifier: str,
    reason: str,
    ttl: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> bool:
    """Blocklist an identifier with a TTL. Returns True on success."""
    settings = settings or Settings()
    ttl = ttl if ttl is not None else settings.abuse_block_ttl_seconds
    try:
        client = _redis_client(settings.redis_url)
        await client.set(_BLOCK_PREFIX + identifier, reason[:200], ex=ttl)
        logger.warning(
            "abuse_guard: BLOCKED %s for %ss — %s", identifier, ttl, reason,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("abuse_guard.block failed (%s: %s)", type(exc).__name__, exc)
        return False


async def unblock(identifier: str, settings: Optional[Settings] = None) -> bool:
    """Remove an identifier from the blocklist. Returns True if a key was removed."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        removed = await client.delete(_BLOCK_PREFIX + identifier)
        if removed:
            logger.info("abuse_guard: UNBLOCKED %s", identifier)
        return bool(removed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("abuse_guard.unblock failed (%s: %s)", type(exc).__name__, exc)
        return False


async def get_block_reason(
    identifier: str, settings: Optional[Settings] = None,
) -> Optional[str]:
    """Return the stored block reason, or None. Fails OPEN (None) on error."""
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        return await client.get(_BLOCK_PREFIX + identifier)
    except Exception as exc:  # noqa: BLE001
        logger.warning("abuse_guard.get_block_reason failed (%s: %s)",
                       type(exc).__name__, exc)
        return None


# --- signal counter primitive ---------------------------------------------

async def _incr_window(client, key: str, window: int) -> int:
    """Atomically bump a windowed counter and (re)set its TTL. Returns count."""
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, window)
    results = await pipe.execute()
    return int(results[0]) if results else 0


# --- public signal API -----------------------------------------------------

async def record_signal(
    signal: str,
    ip_identifier: str,
    settings: Optional[Settings] = None,
    *,
    key_identifier: Optional[str] = None,
) -> None:
    """Accrue one abuse signal and auto-block when a threshold is crossed.

    `signal` is one of: "auth_fail", "burst", "path_scan", "malformed".
    `ip_identifier` is always the `ip:{host}` form.
    `key_identifier` (optional) is the `key:...` form, used for burst
    accounting so a noisy authenticated key gets blocked by key, not IP.
    Never raises.
    """
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)

        if signal == "auth_fail":
            host = ip_identifier
            count = await _incr_window(
                client, _AUTHFAIL_PREFIX + host, _AUTHFAIL_WINDOW,
            )
            if count >= settings.abuse_auth_fail_threshold:
                await block(
                    host,
                    f"auth-fail flood: {count} in {_AUTHFAIL_WINDOW}s",
                    settings=settings,
                )

        elif signal == "burst":
            # Burst is attributed to the API key when present, else the IP,
            # so one abusive tenant doesn't get a whole NAT'd office blocked.
            target = key_identifier or ip_identifier
            count = await _incr_window(
                client, _BURST_PREFIX + target, _BURST_WINDOW,
            )
            if count >= settings.abuse_burst_threshold:
                await block(
                    target,
                    f"request burst: {count} in {_BURST_WINDOW}s",
                    settings=settings,
                )

        elif signal == "path_scan":
            host = ip_identifier
            count = await _incr_window(
                client, _SCAN_PREFIX + host, _SCAN_WINDOW,
            )
            if count >= _SCAN_THRESHOLD:
                await block(
                    host,
                    f"path scanning: {count} probe paths in {_SCAN_WINDOW}s",
                    settings=settings,
                )

        elif signal == "malformed":
            host = ip_identifier
            count = await _incr_window(
                client, _BAD_PREFIX + host, _BAD_WINDOW,
            )
            if count >= _BAD_THRESHOLD:
                await block(
                    host,
                    f"malformed-request flood: {count} in {_BAD_WINDOW}s",
                    settings=settings,
                )
    except Exception as exc:  # noqa: BLE001 — signal accounting must never break a request
        logger.warning("abuse_guard.record_signal(%s) failed (%s: %s)",
                        signal, type(exc).__name__, exc)


async def evaluate(
    request,
    response_status: int,
    settings: Optional[Settings] = None,
) -> None:
    """Post-response hook: inspect the request/response and accrue signals.

    Called by the middleware after the response is produced. Never raises.
    """
    settings = settings or Settings()
    try:
        host = client_ip(request)
        ip_identifier = f"ip:{host}"

        api_key = getattr(request.state, "api_key", None)
        key_identifier = hash_api_key(api_key) if api_key else None

        path = request.url.path

        # (a) credential-stuffing / probing — 401/403 from one IP.
        if response_status in (401, 403):
            await record_signal("auth_fail", ip_identifier, settings=settings)

        # (b) request burst — every request counts toward a 10s window.
        await record_signal(
            "burst", ip_identifier, settings=settings,
            key_identifier=key_identifier,
        )

        # (c) path scanning — known attack-probe path fragments.
        if is_suspicious_path(path):
            await record_signal("path_scan", ip_identifier, settings=settings)

        # (d) malformed / garbage requests — 400/422 surfaced by the app.
        if response_status in (400, 422):
            await record_signal("malformed", ip_identifier, settings=settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("abuse_guard.evaluate failed (%s: %s)",
                        type(exc).__name__, exc)


def request_identifiers(request) -> tuple[str, Optional[str]]:
    """Return (ip_identifier, key_identifier) for a request. Never raises."""
    try:
        ip_identifier = f"ip:{client_ip(request)}"
    except Exception:  # noqa: BLE001
        ip_identifier = "ip:unknown"
    try:
        api_key = getattr(request.state, "api_key", None)
        key_identifier = hash_api_key(api_key) if api_key else None
    except Exception:  # noqa: BLE001
        key_identifier = None
    return ip_identifier, key_identifier


# --- middleware ------------------------------------------------------------

# Imported here (not at top) to keep the dependency local to the middleware.
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, Response  # noqa: E402


class AbuseGuardMiddleware(BaseHTTPMiddleware):
    """Outermost middleware: blocklist enforcement + abuse-signal accrual.

    On entry: if the request's IP (or, once Auth has run, its API key) is
    blocklisted, return 403 immediately — before any other work.

    On exit: feed `evaluate()` the response status so signals accrue and
    repeat offenders get auto-blocked.

    A Redis failure NEVER raises into the request — every Redis touch is
    wrapped and fails OPEN (allow).
    """

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.settings.abuse_guard_enabled:
            return await call_next(request)

        # Health/doc endpoints and platform (private-IP) traffic are never
        # gated or counted. A health probe must never be 403'd, and the
        # load balancer / health checker all reach the app on ONE private
        # IP — counting it trips burst detection and blocklists the whole
        # platform, failing every health check (→ 502 at the edge).
        if is_abuse_exempt_path(request.url.path) or is_infra_ip(client_ip(request)):
            return await call_next(request)

        ip_identifier, _ = request_identifiers(request)

        # --- entry check: IP blocklist (api_key isn't set yet here) ---
        try:
            if await is_blocked(ip_identifier, self.settings):
                logger.info("abuse_guard: refused blocked %s", ip_identifier)
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden: access temporarily blocked."},
                )
        except Exception as exc:  # noqa: BLE001 — fail OPEN
            logger.warning("abuse_guard: entry check failed (%s: %s)",
                           type(exc).__name__, exc)

        response = await call_next(request)

        # --- post-response: accrue signals (best-effort, never raises) ---
        try:
            await evaluate(request, response.status_code, self.settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("abuse_guard: evaluate hook failed (%s: %s)",
                           type(exc).__name__, exc)

        return response
