"""Scope enforcement for CLS++ endpoints."""

from __future__ import annotations

from functools import wraps

from fastapi import HTTPException
from starlette.requests import Request


def require_scope(*scopes: str):
    """Decorator that checks the request has the required scope(s).

    Usage:
        @app.post("/v1/memory/write")
        @require_scope("memories:write")
        async def write_memory(req: WriteRequest, request: Request):
            ...

    - is_admin=True bypasses all checks (backward compatible)
    - Missing scopes returns 403 with detail of what's missing
    """
    required = set(scopes)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                for a in args:
                    if isinstance(a, Request):
                        request = a
                        break

            if request:
                # is_admin bypasses all scope checks
                if getattr(request.state, "is_admin", False):
                    return await func(*args, **kwargs)

                user_scopes = getattr(request.state, "scopes", None)
                # If scopes not loaded (local/demo mode), allow through
                if user_scopes is None:
                    return await func(*args, **kwargs)

                missing = required - user_scopes
                if missing:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing required scope(s): {', '.join(sorted(missing))}",
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator
