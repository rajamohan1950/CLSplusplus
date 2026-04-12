"""CLS++ middleware - auth, rate limiting, request ID, tracing."""

from __future__ import annotations

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from clsplusplus.auth import get_api_key_from_request, validate_api_key
from clsplusplus.config import Settings
from clsplusplus.rate_limit import check_rate_limit


# Paths that never require auth or rate limiting
_PUBLIC_PATHS = frozenset({
    "",
    "/",
    "/health",
    "/v1/memory/health",
    "/v1/demo/status",
    "/v1/demo/chat",
    "/v1/demo/classify",
    "/v1/auth/register",
    "/v1/auth/login",
    "/v1/auth/google",
    "/v1/auth/google/callback",
    "/v1/billing/webhook",
    "/v1/billing/razorpay-webhook",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _is_public(path: str, method: str) -> bool:
    if method == "OPTIONS":
        return True
    normalized = path.rstrip("/") or "/"
    return (normalized in _PUBLIC_PATHS
            or path.startswith("/docs") or path.startswith("/redoc"))


def _requires_admin(path: str) -> bool:
    """Check if path requires admin privileges."""
    return path.startswith("/admin/")


class AuthMiddleware(BaseHTTPMiddleware):
    """Dual auth: API key (Bearer) or JWT cookie (cls_session).

    Sets request.state.api_key when API key is valid.
    Sets request.state.namespace when key resolves to an integration.
    Sets request.state.user_id, user_email, is_admin when JWT is valid.
    """

    def __init__(self, app, settings: Settings, integration_store=None):
        super().__init__(app)
        self.settings = settings
        self.integration_store = integration_store

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_public(request.url.path, request.method):
            return await call_next(request)

        # --- Path 1: API key auth ---
        auth_header = request.headers.get("Authorization")
        key = get_api_key_from_request(auth_header)

        if key:
            # 1a: Check legacy env-var keys
            if validate_api_key(key, self.settings):
                request.state.api_key = key
                request.state.scopes = None  # backward compat
                return await call_next(request)

            # 1b: Check database-stored keys (integration store)
            if self.integration_store:
                try:
                    namespace = await self.integration_store.resolve_namespace_from_key(key)
                    if namespace:
                        request.state.api_key = key
                        request.state.namespace = namespace
                        request.state.scopes = None
                        return await call_next(request)
                except Exception:
                    pass  # DB failure — fall through to JWT or 401

        # --- Path 2: JWT cookie auth ---
        if self.settings.jwt_secret:
            from clsplusplus.jwt_utils import decode_token, get_token_from_cookie
            token = get_token_from_cookie(request)
            if token:
                payload = decode_token(token, self.settings.jwt_secret)
                if payload:
                    request.state.user_id = payload["sub"]
                    request.state.user_email = payload["email"]
                    request.state.is_admin = payload.get("is_admin", False)
                    # Admin route protection
                    if _requires_admin(request.url.path) and not payload.get("is_admin"):
                        return JSONResponse(
                            status_code=403,
                            content={"detail": "Admin access required"},
                        )
                    # Load RBAC scopes (cached in Redis)
                    if not payload.get("is_admin"):
                        try:
                            from clsplusplus.rbac_service import RBACService
                            rbac = RBACService(self.settings)
                            request.state.scopes = await rbac.get_effective_scopes_cached(payload["sub"])
                        except Exception:
                            request.state.scopes = set()  # Fail-closed: no scopes on error
                    else:
                        request.state.scopes = None  # Admin bypasses all scope checks
                    return await call_next(request)

        # --- Neither auth method succeeded ---
        if not self.settings.require_api_key:
            # Auth not required (local/demo mode)
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required. Use API key or sign in."},
            headers={"WWW-Authenticate": "Bearer"},
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limit per API key."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_public(request.url.path, request.method):
            return await call_next(request)

        key_id = getattr(request.state, "api_key", None)
        if not key_id:
            # Rate limit by client IP when no API key is present
            key_id = f"ip:{request.client.host}" if request.client else "ip:unknown"

        allowed, count, limit = await check_rate_limit(key_id, self.settings)
        if not allowed:
            # Emit rate limit hit metric
            try:
                from clsplusplus.metrics import MetricsEmitter
                m = MetricsEmitter(self.settings)
                import asyncio
                asyncio.ensure_future(m.emit("system", "rate_limit_429"))
            except Exception:
                pass
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": self.settings.rate_limit_window_seconds,
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(self.settings.rate_limit_window_seconds),
                },
            )

        # Emit total API request counter
        try:
            from clsplusplus.metrics import MetricsEmitter
            m = MetricsEmitter(self.settings)
            import asyncio
            asyncio.ensure_future(m.emit("system", "total_api_requests"))
        except Exception:
            pass

        response = await call_next(request)
        remaining = limit - count
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        return response


class QuotaMiddleware(BaseHTTPMiddleware):
    """Enforce monthly operation quota based on tier. Returns 402 when exceeded."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.settings.enforce_quotas:
            return await call_next(request)

        if _is_public(request.url.path, request.method):
            return await call_next(request)

        # Only meter mutating/query operations, not GETs on metadata
        if request.method == "GET":
            return await call_next(request)

        api_key = getattr(request.state, "api_key", None)
        if not api_key:
            return await call_next(request)

        try:
            from clsplusplus.tiers import get_tier, check_quota

            tier = get_tier(self.settings)
            allowed, usage, limit = await check_quota(api_key, tier, self.settings)
            if not allowed:
                return JSONResponse(
                    status_code=402,
                    content={
                        "detail": "Monthly operation quota exceeded. Upgrade your plan for more.",
                        "usage": usage,
                        "limit": limit,
                        "tier": tier.value,
                    },
                )
        except Exception:
            pass  # Fail-open: never block on quota check errors

        return await call_next(request)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Add X-Request-Id to every request for distributed tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


# Paths/prefixes that produce too much noise to trace.
# /v1/memory/traces   — trace list/detail polling (memory UI)
# /v1/demo/status     — demo.js warmup ping fires 5× on page load, every 30s
# /v1/memory/phases   — phase-bar polling (every 3s)
# /v1/memory/namespaces — namespace chip polling (every 3s)
# /v1/chat/sessions GET probes are low-value but message POSTs are traced via inner spans
_TRACE_SKIP_PREFIXES = (
    "/docs", "/redoc", "/openapi", "/_",
    "/v1/memory/traces",      # trace list/detail — avoid noise in trace buffer
    "/v1/demo/status",        # warmup ping — not a user action
    "/v1/memory/phases",      # phase-bar polling — high frequency, no value in trace list
    "/v1/memory/namespaces",  # namespace chip polling — same reason
)
_TRACE_SKIP_EXACT = frozenset({"/", "/health", "/v1/health", "/favicon.ico"})

# Map path fragments → short operation names shown in the trace list
_OP_MAP = {
    "/memory/write": "write",
    "/memories/encode": "write",
    "/memory/read": "read",
    "/memories/retrieve": "read",
    "/memories/search": "read",
    "/memories/knowledge": "read",
    "/memory/forget": "delete",
    "/memories/forget": "delete",
    "/memory/traces": "traces",
    "/memory/health": "health",
    "/health/score": "health",
    "/demo/chat": "demo.chat",
    "/demo/status": "demo.status",
    "/demo/memory-cycle": "demo.cycle",
    "/chat/sessions": "chat",
    "/memory/sleep": "sleep",
    "/memories/consolidate": "sleep",
    "/memories/prewarm": "prewarm",
    "/memory/adjudicate_conflict": "adjudicate",
    "/integrations": "integrations",
    "/usage": "usage",
    "/billing": "billing",
}


def _op_name(method: str, path: str) -> str:
    for fragment, name in _OP_MAP.items():
        if fragment in path:
            return name
    if method == "DELETE":
        return "delete"
    # fallback: last non-empty path segment
    parts = [p for p in path.split("/") if p]
    return parts[-1] if parts else "request"


class TracingMiddleware(BaseHTTPMiddleware):
    """Automatically trace every /v1/* API request.

    Creates a root trace at the HTTP layer so the call graph starts from the
    network edge — not buried inside the service.  Every endpoint gets traced:
    demo/chat, health, write, read, delete, integrations, all of them.

    Error information (exception type + message, HTTP status) is recorded on
    the root span so failures are immediately visible in the trace view.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip static assets, docs UI, root
        if (
            path in _TRACE_SKIP_EXACT
            or any(path.startswith(p) for p in _TRACE_SKIP_PREFIXES)
            or not path.startswith("/v1/")
        ):
            return await call_next(request)

        from clsplusplus.tracer import tracer  # local import avoids circular at module load

        # Prefer client-supplied trace/request ID so browser→backend correlation works
        tid = (
            request.headers.get("x-trace-id")
            or request.headers.get("x-request-id")
            or str(uuid.uuid4())
        )

        op = _op_name(request.method, path)
        tracer.new_trace(op, trace_id=tid)
        request.state.trace_id = tid  # handlers read this instead of generating their own

        client_ip = request.client.host if request.client else "?"
        # Label contains everything inline: METHOD /path  — no child meta lines needed
        label = f"{request.method} {path}"

        with tracer.span(tid, label, "http") as http_hop:
            status_code = 500
            try:
                response = await call_next(request)
                status_code = response.status_code
                # Only record status (shown as badge) and client (useful for debugging)
                tracer.add_metadata(tid, http_hop, status=status_code, client=client_ip)
                return response
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {str(exc)[:300]}"
                tracer.add_metadata(tid, http_hop, status=status_code, client=client_ip,
                                    error=err_msg)
                raise
