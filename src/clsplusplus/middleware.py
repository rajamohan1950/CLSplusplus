"""CLS++ middleware - auth and rate limiting."""

from __future__ import annotations

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
            return await call_next(request)

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
