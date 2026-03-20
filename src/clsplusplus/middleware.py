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
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _is_public(path: str, method: str) -> bool:
    if method == "OPTIONS":
        return True
    normalized = path.rstrip("/") or "/"
    return normalized in _PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc")


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate API key for protected routes. Sets request.state.api_key when valid."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_public(request.url.path, request.method):
            return await call_next(request)

        if not self.settings.require_api_key:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        key = get_api_key_from_request(auth_header)
        if not key or not validate_api_key(key, self.settings):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Use Authorization: Bearer <key>"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        request.state.api_key = key
        return await call_next(request)


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

        response = await call_next(request)
        remaining = limit - count
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        return response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Add X-Request-Id to every request for distributed tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


# Paths/prefixes that produce too much noise to trace.
# /v1/trace*          — trace page auto-polls itself every 5s
# /v1/demo/status     — demo.js warmup ping fires 5× on page load, every 30s
# /v1/memory/phases   — phase-bar polling (every 3s)
# /v1/memory/namespaces — namespace chip polling (every 3s)
# /v1/chat/sessions GET probes are low-value but message POSTs are traced via inner spans
_TRACE_SKIP_PREFIXES = (
    "/docs", "/redoc", "/openapi", "/_",
    "/v1/trace",              # trace list / detail endpoints
    "/v1/demo/status",        # warmup ping — not a user action
    "/v1/memory/phases",      # phase-bar polling — high frequency, no value in trace list
    "/v1/memory/namespaces",  # namespace chip polling — same reason
)
_TRACE_SKIP_EXACT = frozenset({"/", "/health", "/favicon.ico"})

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
