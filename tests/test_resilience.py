"""Tests for clsplusplus.resilience — circuit breaker + retry-with-backoff.

Focused suite: run with `pytest tests/test_resilience.py`. Does NOT touch
Redis/Postgres/network — every dependency here is a local async stub.
"""

from __future__ import annotations

import asyncio

import pytest

from clsplusplus.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    get_breaker,
    guarded_call,
    is_transient_error,
    reset_all_breakers,
    retry_async,
    with_retry,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Boom(Exception):
    """A non-transient failure (should never be retried)."""


class Flaky:
    """Async callable that fails its first ``fail_n`` calls, then succeeds."""

    def __init__(self, fail_n: int, exc: BaseException | None = None):
        self.fail_n = fail_n
        self.calls = 0
        self.exc = exc or TimeoutError("transient")

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise self.exc
        return "ok"


# ---------------------------------------------------------------------------
# Transient-error classification
# ---------------------------------------------------------------------------

async def test_timeout_and_connection_errors_are_transient():
    assert is_transient_error(TimeoutError())
    assert is_transient_error(asyncio.TimeoutError())
    assert is_transient_error(ConnectionError())


async def test_programming_errors_are_not_transient():
    assert not is_transient_error(ValueError("bad input"))
    assert not is_transient_error(Boom())
    assert not is_transient_error(KeyError("x"))


async def test_httpx_5xx_transient_4xx_not():
    httpx = pytest.importorskip("httpx")
    req = httpx.Request("GET", "https://x.test")

    err_503 = httpx.HTTPStatusError(
        "boom", request=req, response=httpx.Response(503, request=req))
    err_429 = httpx.HTTPStatusError(
        "rate", request=req, response=httpx.Response(429, request=req))
    err_404 = httpx.HTTPStatusError(
        "nope", request=req, response=httpx.Response(404, request=req))

    assert is_transient_error(err_503)
    assert is_transient_error(err_429)
    assert not is_transient_error(err_404)
    assert is_transient_error(httpx.ConnectTimeout("slow"))


# ---------------------------------------------------------------------------
# retry_async — backoff, jitter, retryable predicate
# ---------------------------------------------------------------------------

async def test_retry_succeeds_after_transient_failures():
    flaky = Flaky(fail_n=2)
    result = await retry_async(flaky, attempts=3, base_delay=0.001)
    assert result == "ok"
    assert flaky.calls == 3  # 2 failures + 1 success


async def test_retry_exhausts_and_reraises_last_exception():
    flaky = Flaky(fail_n=10)
    with pytest.raises(TimeoutError):
        await retry_async(flaky, attempts=3, base_delay=0.001)
    assert flaky.calls == 3  # capped at attempts


async def test_retry_does_not_retry_non_transient_error():
    flaky = Flaky(fail_n=10, exc=Boom("permanent"))
    with pytest.raises(Boom):
        await retry_async(flaky, attempts=5, base_delay=0.001)
    assert flaky.calls == 1  # gave up immediately — Boom is not transient


async def test_retry_backoff_grows_and_is_jittered():
    """Delays must be exponential-capped and randomised (full jitter)."""
    sleeps: list[float] = []
    orig_sleep = asyncio.sleep

    async def _record(delay):
        sleeps.append(delay)
        await orig_sleep(0)  # don't actually wait

    flaky = Flaky(fail_n=10)
    import clsplusplus.resilience as resmod
    resmod.asyncio.sleep = _record  # type: ignore[attr-defined]
    try:
        with pytest.raises(TimeoutError):
            await retry_async(flaky, attempts=4, base_delay=1.0, max_delay=100.0)
    finally:
        resmod.asyncio.sleep = orig_sleep  # type: ignore[attr-defined]

    assert len(sleeps) == 3  # 3 retries between 4 attempts
    # Full jitter: each sleep in [0, base*2**n].
    assert 0 <= sleeps[0] <= 1.0
    assert 0 <= sleeps[1] <= 2.0
    assert 0 <= sleeps[2] <= 4.0


async def test_retry_respects_max_delay_cap():
    sleeps: list[float] = []
    orig_sleep = asyncio.sleep

    async def _record(delay):
        sleeps.append(delay)
        await orig_sleep(0)

    flaky = Flaky(fail_n=10)
    import clsplusplus.resilience as resmod
    resmod.asyncio.sleep = _record  # type: ignore[attr-defined]
    try:
        with pytest.raises(TimeoutError):
            await retry_async(flaky, attempts=6, base_delay=10.0, max_delay=2.0)
    finally:
        resmod.asyncio.sleep = orig_sleep  # type: ignore[attr-defined]

    # Every computed delay is capped at max_delay, jitter keeps it <= 2.0.
    assert all(0 <= s <= 2.0 for s in sleeps)


async def test_with_retry_decorator():
    flaky = Flaky(fail_n=1)

    @with_retry(attempts=3, base_delay=0.001)
    async def call():
        return await flaky()

    assert await call() == "ok"
    assert flaky.calls == 2


# ---------------------------------------------------------------------------
# CircuitBreaker — state machine
# ---------------------------------------------------------------------------

async def test_breaker_starts_closed_and_passes_calls_through():
    cb = CircuitBreaker("t", failure_threshold=3)
    assert cb.state == "closed"

    async def ok():
        return 42

    assert await cb.call(ok) == 42
    assert cb.state == "closed"


async def test_breaker_opens_after_threshold_consecutive_failures():
    cb = CircuitBreaker("t", failure_threshold=3, recovery_seconds=60)

    async def fail():
        raise TimeoutError("dep down")

    for _ in range(3):
        with pytest.raises(TimeoutError):
            await cb.call(fail)

    assert cb.state == "open"

    # Once open, calls are rejected fast WITHOUT invoking the function.
    invoked = False

    async def tracked():
        nonlocal invoked
        invoked = True

    with pytest.raises(CircuitOpenError):
        await cb.call(tracked)
    assert invoked is False


async def test_breaker_success_resets_failure_count():
    cb = CircuitBreaker("t", failure_threshold=3)

    async def fail():
        raise TimeoutError()

    async def ok():
        return "ok"

    # Two failures, then a success — counter resets, breaker stays closed.
    for _ in range(2):
        with pytest.raises(TimeoutError):
            await cb.call(fail)
    await cb.call(ok)
    assert cb.state == "closed"

    # Two more failures must NOT open it (counter was reset).
    for _ in range(2):
        with pytest.raises(TimeoutError):
            await cb.call(fail)
    assert cb.state == "closed"


async def test_breaker_half_open_probe_success_closes():
    # Tiny cooldown so the test doesn't actually sleep long.
    cb = CircuitBreaker("t", failure_threshold=2, recovery_seconds=0.05)

    async def fail():
        raise TimeoutError()

    async def ok():
        return "recovered"

    for _ in range(2):
        with pytest.raises(TimeoutError):
            await cb.call(fail)
    assert cb.state == "open"

    await asyncio.sleep(0.06)  # cooldown elapses
    assert cb.state == "half_open"

    # A successful probe closes the breaker.
    assert await cb.call(ok) == "recovered"
    assert cb.state == "closed"


async def test_breaker_half_open_probe_failure_reopens():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_seconds=0.05)

    async def fail():
        raise TimeoutError()

    for _ in range(2):
        with pytest.raises(TimeoutError):
            await cb.call(fail)
    assert cb.state == "open"

    await asyncio.sleep(0.06)
    assert cb.state == "half_open"

    # The probe fails — breaker snaps back to open and restarts cooldown.
    with pytest.raises(TimeoutError):
        await cb.call(fail)
    assert cb.state == "open"


async def test_breaker_reraises_so_caller_can_fail_soft():
    """A breaker failure must re-raise the original exception type."""
    cb = CircuitBreaker("t", failure_threshold=5)

    async def boom():
        raise Boom("explicit")

    with pytest.raises(Boom):
        await cb.call(boom)


# ---------------------------------------------------------------------------
# guarded_call — breaker + retry composed
# ---------------------------------------------------------------------------

async def test_guarded_call_retries_then_succeeds_one_breaker_success():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_seconds=60)
    flaky = Flaky(fail_n=2)

    # 3 attempts: 2 transient fails + 1 success. The whole retried op is
    # ONE breaker success, so the breaker stays closed.
    result = await guarded_call(cb, flaky, attempts=3, base_delay=0.001)
    assert result == "ok"
    assert cb.state == "closed"
    assert flaky.calls == 3


async def test_guarded_call_opens_breaker_after_repeated_logical_failures():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_seconds=60)

    async def always_fail():
        raise TimeoutError("dead")

    # Each guarded_call exhausts its retries -> one logical breaker failure.
    for _ in range(2):
        with pytest.raises(TimeoutError):
            await guarded_call(cb, always_fail, attempts=2, base_delay=0.001)

    assert cb.state == "open"

    # Now the breaker rejects before any retry loop runs.
    with pytest.raises(CircuitOpenError):
        await guarded_call(cb, always_fail, attempts=2, base_delay=0.001)


async def test_guarded_call_does_not_retry_breaker_rejection():
    """CircuitOpenError must propagate immediately, never be retried."""
    cb = CircuitBreaker("t", failure_threshold=1, recovery_seconds=60)

    async def fail():
        raise TimeoutError()

    with pytest.raises(TimeoutError):
        await guarded_call(cb, fail, attempts=1, base_delay=0.001)
    assert cb.state == "open"

    calls = 0

    async def counted():
        nonlocal calls
        calls += 1

    with pytest.raises(CircuitOpenError):
        await guarded_call(cb, counted, attempts=5, base_delay=0.001)
    assert calls == 0  # rejected, retry loop never ran


# ---------------------------------------------------------------------------
# Shared breaker registry
# ---------------------------------------------------------------------------

async def test_get_breaker_returns_same_instance_per_name():
    reset_all_breakers()
    a = get_breaker("svc-x", failure_threshold=2)
    b = get_breaker("svc-x")
    assert a is b
    c = get_breaker("svc-y")
    assert c is not a
    reset_all_breakers()


async def test_concurrent_calls_through_one_breaker_stay_consistent():
    """Concurrent failures must not corrupt the failure counter."""
    cb = CircuitBreaker("t", failure_threshold=10, recovery_seconds=60)

    async def fail():
        await asyncio.sleep(0)
        raise TimeoutError()

    async def attempt():
        try:
            await cb.call(fail)
        except (TimeoutError, CircuitOpenError):
            pass

    # 10 concurrent failures should trip the breaker exactly at threshold.
    await asyncio.gather(*(attempt() for _ in range(10)))
    assert cb.state == "open"
