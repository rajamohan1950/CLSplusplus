"""Resilience primitives for outbound / dependency calls.

CLS++ talks to a handful of external dependencies (Postgres, Redis,
GeoIP, Resend, Razorpay, the demo LLM providers, OAuth token endpoints).
A slow or dead dependency must never be allowed to (a) hang a request
forever, or (b) hammer a struggling service into a deeper hole.

This module provides three small, dependency-free building blocks:

  * ``CircuitBreaker`` — an in-process closed/open/half-open breaker.
    After ``failure_threshold`` consecutive failures it OPENS and fails
    fast (raising ``CircuitOpenError``) for ``recovery_seconds``; it then
    moves to HALF-OPEN and lets a single probe through. A probe success
    closes it; a probe failure re-opens it.

  * ``retry_async`` / ``with_retry`` — exponential backoff with full
    jitter, a capped attempt count, and a transient-error predicate so
    only timeouts / connection errors / 5xx are retried (never a 4xx or
    a programming bug).

  * ``make_async_client`` / ``DEFAULT_HTTP_TIMEOUT`` — a helper that
    builds an ``httpx.AsyncClient`` with sane connect/read/write/pool
    timeouts so no outbound HTTP call can hang indefinitely.

Everything here is in-process and per-worker. That is intentional and
sufficient: a dead dependency is dead for every worker, so each worker
opening its own breaker independently still achieves "fail fast".
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, Iterable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# HTTP client helper
# ---------------------------------------------------------------------------

# Default outbound HTTP timeouts. connect is short (a dead host should be
# detected fast); read is the generous bound for a slow-but-alive peer.
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 15.0


def http_timeout(
    *,
    connect: float = DEFAULT_CONNECT_TIMEOUT,
    read: float = DEFAULT_READ_TIMEOUT,
):
    """Return an ``httpx.Timeout`` with explicit per-phase bounds.

    ``write`` and ``pool`` mirror ``read`` so a request can never block
    forever on a slow socket write or on pool-acquisition contention.
    """
    import httpx

    return httpx.Timeout(connect=connect, read=read, write=read, pool=read)


# A ready-made timeout for the common case.
def _default_timeout():
    return http_timeout()


def make_async_client(
    *,
    connect: float = DEFAULT_CONNECT_TIMEOUT,
    read: float = DEFAULT_READ_TIMEOUT,
    **kwargs,
):
    """Build an ``httpx.AsyncClient`` that cannot hang.

    Any explicit ``timeout`` in ``kwargs`` wins; otherwise the sane
    per-phase default is applied.
    """
    import httpx

    kwargs.setdefault("timeout", http_timeout(connect=connect, read=read))
    return httpx.AsyncClient(**kwargs)


# ---------------------------------------------------------------------------
# Transient-error classification
# ---------------------------------------------------------------------------

def is_transient_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient fault worth retrying.

    Retryable: timeouts, connection errors, and HTTP 5xx / 429 responses.
    Not retryable: 4xx (other than 429), programming errors, ValueError,
    auth failures — retrying those just wastes time and amplifies load.
    """
    # asyncio / socket level timeouts.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    # httpx — import lazily so this module has no hard httpx dependency.
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError,
                            httpx.ReadError, httpx.WriteError,
                            httpx.RemoteProtocolError, httpx.PoolTimeout)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status == 429 or status >= 500
    except ImportError:
        pass

    # redis — connection/timeout errors are transient.
    try:
        import redis.exceptions as _redis_exc

        if isinstance(exc, (_redis_exc.ConnectionError, _redis_exc.TimeoutError)):
            return True
    except ImportError:
        pass

    return False


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is OPEN."""


# Breaker states.
_CLOSED = "closed"
_OPEN = "open"
_HALF_OPEN = "half_open"


class CircuitBreaker:
    """In-process circuit breaker for a single flaky dependency.

    Lifecycle::

        CLOSED  --N consecutive failures-->  OPEN
        OPEN    --recovery_seconds elapsed--> HALF_OPEN  (one probe allowed)
        HALF_OPEN --probe ok--> CLOSED
        HALF_OPEN --probe fails--> OPEN

    The breaker is async-safe: state transitions are guarded by an
    ``asyncio.Lock`` so concurrent callers see a consistent view.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds

        self._state = _CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

    # -- introspection ------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state, transparently expiring an OPEN window to HALF_OPEN."""
        if self._state == _OPEN and self._cooldown_elapsed():
            return _HALF_OPEN
        return self._state

    def _cooldown_elapsed(self) -> bool:
        return (time.monotonic() - self._opened_at) >= self.recovery_seconds

    # -- gate ---------------------------------------------------------------

    async def _allow(self) -> None:
        """Raise ``CircuitOpenError`` if the call must be rejected."""
        async with self._lock:
            if self._state == _OPEN:
                if self._cooldown_elapsed():
                    # Cooldown done — promote to HALF_OPEN and let this
                    # single caller through as the probe.
                    self._state = _HALF_OPEN
                    logger.info("circuit %s: open -> half_open (probing)", self.name)
                else:
                    raise CircuitOpenError(
                        f"circuit '{self.name}' is open — failing fast"
                    )

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state != _CLOSED:
                logger.info("circuit %s: %s -> closed", self.name, self._state)
            self._state = _CLOSED
            self._consecutive_failures = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if self._state == _HALF_OPEN:
                # Probe failed — straight back to OPEN, restart cooldown.
                self._state = _OPEN
                self._opened_at = time.monotonic()
                logger.warning("circuit %s: half_open probe failed -> open", self.name)
            elif (
                self._state == _CLOSED
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._state = _OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "circuit %s: %d consecutive failures -> open (cooldown %.0fs)",
                    self.name, self._consecutive_failures, self.recovery_seconds,
                )

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Invoke ``fn`` through the breaker.

        Raises ``CircuitOpenError`` immediately if the breaker is OPEN.
        Any other exception is recorded as a failure and re-raised so the
        caller's existing error handling (fail-soft) still runs.
        """
        await self._allow()
        try:
            result = await fn(*args, **kwargs)
        except CircuitOpenError:
            raise
        except BaseException:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    def reset(self) -> None:
        """Force the breaker back to CLOSED. Intended for tests only."""
        self._state = _CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0


# ---------------------------------------------------------------------------
# Retry with exponential backoff + jitter
# ---------------------------------------------------------------------------

async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args,
    attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
    is_retryable: Callable[[BaseException], bool] = is_transient_error,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs,
) -> T:
    """Call ``fn`` with retry on transient failures.

    Backoff is exponential (``base_delay * 2**n``) capped at ``max_delay``,
    with *full jitter* (the actual sleep is uniform in ``[0, computed]``)
    to spread retries and avoid a thundering herd.

    A non-retryable error, or exhausting ``attempts``, re-raises the last
    exception so the caller's fail-soft path still runs.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return await fn(*args, **kwargs)
        except CircuitOpenError:
            # A breaker rejection is deliberate fast-fail — never retry it.
            raise
        except BaseException as exc:  # noqa: BLE001 — re-raised below.
            last_exc = exc
            if attempt + 1 >= attempts or not is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay = random.uniform(0, delay)  # full jitter
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            else:
                logger.warning(
                    "retry %d/%d after %s: %s — sleeping %.2fs",
                    attempt + 1, attempts, type(exc).__name__, exc, delay,
                )
            await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises. Guard for type checkers.
    assert last_exc is not None
    raise last_exc


def with_retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
    is_retryable: Callable[[BaseException], bool] = is_transient_error,
):
    """Decorator form of :func:`retry_async` for async functions."""

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(
                fn, *args,
                attempts=attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                is_retryable=is_retryable,
                **kwargs,
            )

        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Combined helper: breaker + retry
# ---------------------------------------------------------------------------

async def guarded_call(
    breaker: CircuitBreaker,
    fn: Callable[..., Awaitable[T]],
    *args,
    attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
    **kwargs,
) -> T:
    """Run ``fn`` behind ``breaker`` with retry-with-backoff.

    The retry loop is *inside* the breaker: a single logical operation
    (its retries included) counts as one breaker success or failure, so a
    burst of transient blips does not trip the breaker prematurely while a
    genuinely-dead dependency still trips it after ``failure_threshold``
    logical attempts.
    """

    async def _attempt():
        return await retry_async(
            fn, *args,
            attempts=attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            **kwargs,
        )

    return await breaker.call(_attempt)


# ---------------------------------------------------------------------------
# Shared breaker registry for the genuinely-external flaky dependencies
# ---------------------------------------------------------------------------

_BREAKERS: dict[str, CircuitBreaker] = {}


def get_breaker(
    name: str,
    *,
    failure_threshold: int = 5,
    recovery_seconds: float = 30.0,
) -> CircuitBreaker:
    """Return the process-wide breaker for ``name``, creating it once.

    Call sites share a breaker by name so every request through, e.g.,
    the GeoIP path observes the same open/closed state.
    """
    breaker = _BREAKERS.get(name)
    if breaker is None:
        breaker = CircuitBreaker(
            name,
            failure_threshold=failure_threshold,
            recovery_seconds=recovery_seconds,
        )
        _BREAKERS[name] = breaker
    return breaker


def breaker_from_settings(name: str, settings) -> CircuitBreaker:
    """Build/fetch a breaker using env-overridable thresholds from Settings."""
    return get_breaker(
        name,
        failure_threshold=getattr(settings, "circuit_failure_threshold", 5),
        recovery_seconds=getattr(settings, "circuit_recovery_seconds", 30.0),
    )


def reset_all_breakers() -> None:
    """Reset every registered breaker. Intended for tests only."""
    for breaker in _BREAKERS.values():
        breaker.reset()
    _BREAKERS.clear()


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_READ_TIMEOUT",
    "http_timeout",
    "make_async_client",
    "is_transient_error",
    "retry_async",
    "with_retry",
    "guarded_call",
    "get_breaker",
    "breaker_from_settings",
    "reset_all_breakers",
]
