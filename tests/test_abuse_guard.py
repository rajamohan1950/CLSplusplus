"""Abuse-guard tests — blocklist, heuristic signals, middleware, fail-open.

Uses a small in-memory fake Redis so threshold/auto-block behaviour is
exercised for real (counters, TTLs) without a live Redis. Fail-open paths
use a client that raises.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clsplusplus import abuse_guard
from clsplusplus.abuse_guard import (
    AbuseGuardMiddleware,
    block,
    evaluate,
    get_block_reason,
    hash_api_key,
    is_blocked,
    is_suspicious_path,
    record_signal,
    unblock,
)
from clsplusplus.config import Settings


# ---------------------------------------------------------------------------
# Fake async Redis — just enough surface for abuse_guard.
# ---------------------------------------------------------------------------

class _FakePipe:
    def __init__(self, store):
        self._store = store
        self._ops = []

    def incr(self, key):
        self._ops.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", key, ttl))
        return self

    async def execute(self):
        results = []
        for op in self._ops:
            if op[0] == "incr":
                self._store[op[1]] = int(self._store.get(op[1], 0)) + 1
                results.append(self._store[op[1]])
            elif op[0] == "expire":
                results.append(True)
        self._ops = []
        return results


class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return _FakePipe(self.store)

    async def exists(self, key):
        return 1 if key in self.store else 0

    async def set(self, key, value, ex=None):  # noqa: A003
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key)

    async def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


def _settings(**kw):
    return Settings(**kw)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_hash_api_key_is_stable_and_prefixed(self):
        h1 = hash_api_key("sk-abc123")
        h2 = hash_api_key("sk-abc123")
        assert h1 == h2
        assert h1.startswith("key:")
        assert "sk-abc123" not in h1  # raw key never leaks

    def test_suspicious_paths_detected(self):
        for p in ["/wp-login.php", "/.env", "/admin.php",
                  "/phpmyadmin/index.php", "/.git/config", "/cgi-bin/x"]:
            assert is_suspicious_path(p), p

    def test_suspicious_path_case_insensitive(self):
        assert is_suspicious_path("/WP-Login.PHP")

    def test_legitimate_paths_not_flagged(self):
        for p in ["/v1/memory/write", "/health", "/v1/auth/login", "/docs"]:
            assert not is_suspicious_path(p), p


# ---------------------------------------------------------------------------
# Blocklist primitives
# ---------------------------------------------------------------------------

class TestBlocklist:

    @pytest.mark.asyncio
    async def test_block_then_is_blocked(self):
        fake = FakeRedis()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            assert await is_blocked("ip:1.2.3.4", _settings()) is False
            await block("ip:1.2.3.4", "test reason", ttl=60, settings=_settings())
            assert await is_blocked("ip:1.2.3.4", _settings()) is True

    @pytest.mark.asyncio
    async def test_unblock_removes_entry(self):
        fake = FakeRedis()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await block("ip:9.9.9.9", "r", ttl=60, settings=_settings())
            assert await unblock("ip:9.9.9.9", _settings()) is True
            assert await is_blocked("ip:9.9.9.9", _settings()) is False
            # second unblock is a no-op
            assert await unblock("ip:9.9.9.9", _settings()) is False

    @pytest.mark.asyncio
    async def test_get_block_reason(self):
        fake = FakeRedis()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await block("key:abc", "burst flood", ttl=60, settings=_settings())
            assert await get_block_reason("key:abc", _settings()) == "burst flood"

    @pytest.mark.asyncio
    async def test_block_default_ttl_from_settings(self):
        """ttl=None falls back to settings.abuse_block_ttl_seconds."""
        fake = FakeRedis()
        fake.set = AsyncMock(return_value=True)
        s = _settings(abuse_block_ttl_seconds=3600)
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await block("ip:5.5.5.5", "r", settings=s)
        _, kwargs = fake.set.call_args
        assert kwargs["ex"] == 3600


# ---------------------------------------------------------------------------
# Fail-open: Redis errors must never raise
# ---------------------------------------------------------------------------

class TestFailOpen:

    @pytest.mark.asyncio
    async def test_is_blocked_fails_open(self):
        bad = MagicMock()
        bad.exists = AsyncMock(side_effect=ConnectionError("redis down"))
        with patch("clsplusplus.abuse_guard._redis_client", return_value=bad):
            # Fails OPEN -> not blocked, no exception
            assert await is_blocked("ip:1.1.1.1", _settings()) is False

    @pytest.mark.asyncio
    async def test_block_swallows_errors(self):
        bad = MagicMock()
        bad.set = AsyncMock(side_effect=ConnectionError("redis down"))
        with patch("clsplusplus.abuse_guard._redis_client", return_value=bad):
            assert await block("ip:1.1.1.1", "r", ttl=60, settings=_settings()) is False

    @pytest.mark.asyncio
    async def test_record_signal_swallows_errors(self):
        with patch("clsplusplus.abuse_guard._redis_client",
                   side_effect=ConnectionError("no redis")):
            # Must not raise
            await record_signal("auth_fail", "ip:1.1.1.1", _settings())

    @pytest.mark.asyncio
    async def test_evaluate_swallows_errors(self):
        req = _make_request("/v1/memory/write", host="1.1.1.1")
        with patch("clsplusplus.abuse_guard._redis_client",
                   side_effect=ConnectionError("no redis")):
            await evaluate(req, 200, _settings())  # must not raise


# ---------------------------------------------------------------------------
# Signal accrual -> auto-block at threshold
# ---------------------------------------------------------------------------

class TestSignalThresholds:

    @pytest.mark.asyncio
    async def test_auth_fail_blocks_at_threshold(self):
        fake = FakeRedis()
        s = _settings(abuse_auth_fail_threshold=5)
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            for _ in range(4):
                await record_signal("auth_fail", "ip:2.2.2.2", s)
            assert await is_blocked("ip:2.2.2.2", s) is False
            await record_signal("auth_fail", "ip:2.2.2.2", s)  # 5th -> block
            assert await is_blocked("ip:2.2.2.2", s) is True

    @pytest.mark.asyncio
    async def test_burst_blocks_by_key_when_present(self):
        fake = FakeRedis()
        s = _settings(abuse_burst_threshold=3)
        kid = hash_api_key("sk-noisy")
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            for _ in range(3):
                await record_signal("burst", "ip:3.3.3.3", s, key_identifier=kid)
            # blocked on the key, not the shared IP
            assert await is_blocked(kid, s) is True
            assert await is_blocked("ip:3.3.3.3", s) is False

    @pytest.mark.asyncio
    async def test_burst_blocks_by_ip_when_no_key(self):
        fake = FakeRedis()
        s = _settings(abuse_burst_threshold=3)
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            for _ in range(3):
                await record_signal("burst", "ip:4.4.4.4", s)
            assert await is_blocked("ip:4.4.4.4", s) is True

    @pytest.mark.asyncio
    async def test_path_scan_blocks_at_threshold(self):
        fake = FakeRedis()
        s = _settings()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            for _ in range(abuse_guard._SCAN_THRESHOLD):
                await record_signal("path_scan", "ip:6.6.6.6", s)
            assert await is_blocked("ip:6.6.6.6", s) is True

    @pytest.mark.asyncio
    async def test_malformed_blocks_at_threshold(self):
        fake = FakeRedis()
        s = _settings()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            for _ in range(abuse_guard._BAD_THRESHOLD):
                await record_signal("malformed", "ip:7.7.7.7", s)
            assert await is_blocked("ip:7.7.7.7", s) is True

    @pytest.mark.asyncio
    async def test_unknown_signal_is_noop(self):
        fake = FakeRedis()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await record_signal("bogus", "ip:8.8.8.8", _settings())
        assert fake.store == {}


# ---------------------------------------------------------------------------
# evaluate(): which signals it emits per response
# ---------------------------------------------------------------------------

class TestEvaluate:

    @pytest.mark.asyncio
    async def test_evaluate_401_accrues_auth_fail(self):
        fake = FakeRedis()
        req = _make_request("/v1/memory/write", host="10.0.0.1")
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await evaluate(req, 401, _settings())
        assert fake.store.get("cls:abuse:authfail:ip:10.0.0.1") == 1
        # burst counter always bumps
        assert fake.store.get("cls:abuse:burst:ip:10.0.0.1") == 1

    @pytest.mark.asyncio
    async def test_evaluate_suspicious_path_accrues_scan(self):
        fake = FakeRedis()
        req = _make_request("/wp-login.php", host="10.0.0.2")
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await evaluate(req, 404, _settings())
        assert fake.store.get("cls:abuse:scan:ip:10.0.0.2") == 1

    @pytest.mark.asyncio
    async def test_evaluate_400_accrues_malformed(self):
        fake = FakeRedis()
        req = _make_request("/v1/memory/write", host="10.0.0.3")
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await evaluate(req, 422, _settings())
        assert fake.store.get("cls:abuse:bad:ip:10.0.0.3") == 1

    @pytest.mark.asyncio
    async def test_evaluate_200_only_bumps_burst(self):
        fake = FakeRedis()
        req = _make_request("/v1/memory/write", host="10.0.0.4")
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await evaluate(req, 200, _settings())
        assert fake.store.get("cls:abuse:burst:ip:10.0.0.4") == 1
        assert "cls:abuse:authfail:ip:10.0.0.4" not in fake.store
        assert "cls:abuse:scan:ip:10.0.0.4" not in fake.store


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

def _make_request(path, host="1.2.3.4", api_key=None):
    """Minimal Starlette-ish request stand-in."""
    req = MagicMock()
    req.url.path = path
    req.client.host = host
    state = MagicMock()
    state.api_key = api_key
    # getattr(state, "api_key", None) — MagicMock returns the attr we set
    req.state = state
    return req


class _Resp:
    def __init__(self, status):
        self.status_code = status


class TestMiddleware:

    @pytest.mark.asyncio
    async def test_disabled_passes_through(self):
        s = _settings(abuse_guard_enabled=False)
        mw = AbuseGuardMiddleware(MagicMock(), s)
        called = {}

        async def call_next(req):
            called["yes"] = True
            return _Resp(200)

        req = _make_request("/v1/memory/write")
        resp = await mw.dispatch(req, call_next)
        assert called.get("yes") is True
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_blocked_ip_gets_403_before_call_next(self):
        fake = FakeRedis()
        s = _settings()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            await block("ip:5.6.7.8", "path scanning", ttl=60, settings=s)
            mw = AbuseGuardMiddleware(MagicMock(), s)

            async def call_next(req):
                raise AssertionError("call_next must not run for a blocked IP")

            req = _make_request("/v1/memory/write", host="5.6.7.8")
            resp = await mw.dispatch(req, call_next)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_clean_request_passes_and_accrues(self):
        fake = FakeRedis()
        s = _settings()
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            mw = AbuseGuardMiddleware(MagicMock(), s)

            async def call_next(req):
                return _Resp(200)

            req = _make_request("/v1/memory/write", host="11.11.11.11")
            resp = await mw.dispatch(req, call_next)
        assert resp.status_code == 200
        # post-response evaluate ran -> burst counter incremented
        assert fake.store.get("cls:abuse:burst:ip:11.11.11.11") == 1

    @pytest.mark.asyncio
    async def test_middleware_fails_open_on_redis_error(self):
        """Redis down at entry check -> request still served."""
        s = _settings()
        with patch("clsplusplus.abuse_guard._redis_client",
                   side_effect=ConnectionError("redis down")):
            mw = AbuseGuardMiddleware(MagicMock(), s)

            async def call_next(req):
                return _Resp(200)

            req = _make_request("/v1/memory/write", host="12.12.12.12")
            resp = await mw.dispatch(req, call_next)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_repeated_auth_failures_eventually_blocked(self):
        """End-to-end: probing IP accrues 401s and the next request is 403'd."""
        fake = FakeRedis()
        s = _settings(abuse_auth_fail_threshold=3)
        with patch("clsplusplus.abuse_guard._redis_client", return_value=fake):
            mw = AbuseGuardMiddleware(MagicMock(), s)

            async def call_next(req):
                return _Resp(401)

            req = _make_request("/v1/memory/write", host="13.13.13.13")
            for _ in range(3):
                resp = await mw.dispatch(req, call_next)
                assert resp.status_code == 401  # served, but signals accrue

            # 4th request: IP now blocklisted -> 403 at entry
            resp = await mw.dispatch(req, call_next)
            assert resp.status_code == 403
