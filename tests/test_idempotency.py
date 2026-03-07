"""Idempotency tests - request deduplication, cache key generation, Redis caching."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.idempotency import _cache_key, cache_response, get_cached_response


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------

class TestCacheKey:

    def test_deterministic(self):
        k1 = _cache_key("key1", "POST", "/v1/memory/write", b'{"text":"hello"}')
        k2 = _cache_key("key1", "POST", "/v1/memory/write", b'{"text":"hello"}')
        assert k1 == k2

    def test_different_methods_different_keys(self):
        k1 = _cache_key("key1", "POST", "/path", b"body")
        k2 = _cache_key("key1", "GET", "/path", b"body")
        assert k1 != k2

    def test_different_paths_different_keys(self):
        k1 = _cache_key("key1", "POST", "/path1", b"body")
        k2 = _cache_key("key1", "POST", "/path2", b"body")
        assert k1 != k2

    def test_different_bodies_different_keys(self):
        k1 = _cache_key("key1", "POST", "/path", b"body1")
        k2 = _cache_key("key1", "POST", "/path", b"body2")
        assert k1 != k2

    def test_key_prefix(self):
        k = _cache_key("mykey", "POST", "/path", b"body")
        assert k.startswith("cls:idempotency:mykey:")

    def test_key_hash_length(self):
        k = _cache_key("key", "POST", "/path", b"body")
        # cls:idempotency:key:HASH (hash is 32 chars)
        parts = k.split(":")
        assert len(parts) == 4
        assert len(parts[3]) == 32


# ---------------------------------------------------------------------------
# Get cached response
# ---------------------------------------------------------------------------

class TestGetCachedResponse:

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        s = Settings()
        with patch("clsplusplus.idempotency._redis_client", return_value=mock_client):
            result = await get_cached_response("key", "POST", "/path", b"body", s)
            assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_data(self):
        cached = json.dumps({"status": 200, "body": {"id": "abc"}})
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=cached)

        s = Settings()
        with patch("clsplusplus.idempotency._redis_client", return_value=mock_client):
            result = await get_cached_response("key", "POST", "/path", b"body", s)
            assert result is not None
            assert result["status"] == 200
            assert result["body"]["id"] == "abc"

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("Redis down"))

        s = Settings()
        with patch("clsplusplus.idempotency._redis_client", return_value=mock_client):
            result = await get_cached_response("key", "POST", "/path", b"body", s)
            assert result is None


# ---------------------------------------------------------------------------
# Cache response
# ---------------------------------------------------------------------------

class TestCacheResponse:

    @pytest.mark.asyncio
    async def test_stores_in_redis(self):
        mock_client = AsyncMock()
        mock_client.setex = AsyncMock()

        s = Settings(idempotency_ttl_seconds=3600)
        with patch("clsplusplus.idempotency._redis_client", return_value=mock_client):
            await cache_response("key", "POST", "/path", b"body", 200, {"id": "abc"}, s)
            mock_client.setex.assert_called_once()
            args = mock_client.setex.call_args[0]
            assert args[1] == 3600  # TTL

    @pytest.mark.asyncio
    async def test_redis_error_silently_ignored(self):
        mock_client = AsyncMock()
        mock_client.setex = AsyncMock(side_effect=ConnectionError("Redis down"))

        s = Settings()
        with patch("clsplusplus.idempotency._redis_client", return_value=mock_client):
            # Should not raise
            await cache_response("key", "POST", "/path", b"body", 200, {"id": "abc"}, s)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestIdempotencyRoundTrip:

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        store = {}

        async def mock_get(key):
            return store.get(key)

        async def mock_setex(key, ttl, value):
            store[key] = value

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.setex = mock_setex

        s = Settings()
        with patch("clsplusplus.idempotency._redis_client", return_value=mock_client):
            # Cache a response
            await cache_response("key1", "POST", "/write", b"body1", 200, {"id": "abc"}, s)
            # Retrieve it
            result = await get_cached_response("key1", "POST", "/write", b"body1", s)
            assert result is not None
            assert result["body"]["id"] == "abc"

            # Different request = cache miss
            result2 = await get_cached_response("key1", "POST", "/write", b"body2", s)
            assert result2 is None
