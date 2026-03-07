"""Rate limiting tests - sliding window, Redis interaction, fail-open."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.rate_limit import check_rate_limit


# ---------------------------------------------------------------------------
# Sliding window logic
# ---------------------------------------------------------------------------

class TestCheckRateLimit:

    @pytest.mark.asyncio
    async def test_first_request_allowed(self):
        """First request within window is always allowed."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 0])  # zremrange result, zcard = 0
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_client.pipeline.return_value = mock_pipe

        mock_pipe2 = MagicMock()
        mock_pipe2.execute = AsyncMock(return_value=[True, True])
        mock_pipe2.zadd = MagicMock(return_value=mock_pipe2)
        mock_pipe2.expire = MagicMock(return_value=mock_pipe2)
        mock_client.pipeline.side_effect = [mock_pipe, mock_pipe2]

        s = Settings(rate_limit_requests=100, rate_limit_window_seconds=60)
        with patch("clsplusplus.rate_limit._redis_client", return_value=mock_client):
            allowed, count, limit = await check_rate_limit("test-key", s)
            assert allowed is True
            assert limit == 100

    @pytest.mark.asyncio
    async def test_at_limit_rejected(self):
        """Request at limit is rejected."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 100])  # Already at limit
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_client.pipeline.return_value = mock_pipe

        s = Settings(rate_limit_requests=100)
        with patch("clsplusplus.rate_limit._redis_client", return_value=mock_client):
            allowed, count, limit = await check_rate_limit("test-key", s)
            assert allowed is False
            assert count == 100

    @pytest.mark.asyncio
    async def test_fail_open_on_redis_error(self):
        """When Redis fails, allow the request (fail-open)."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_client.pipeline.return_value = mock_pipe

        s = Settings(rate_limit_requests=100)
        with patch("clsplusplus.rate_limit._redis_client", return_value=mock_client):
            allowed, count, limit = await check_rate_limit("test-key", s)
            assert allowed is True  # Fail open

    @pytest.mark.asyncio
    async def test_custom_rate_limit(self):
        """Custom rate limit settings are respected."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 5])
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_client.pipeline.return_value = mock_pipe

        s = Settings(rate_limit_requests=5)
        with patch("clsplusplus.rate_limit._redis_client", return_value=mock_client):
            allowed, count, limit = await check_rate_limit("test-key", s)
            assert allowed is False
            assert limit == 5

    @pytest.mark.asyncio
    async def test_just_below_limit_allowed(self):
        """Request just below limit is allowed."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 99])  # One below 100
        mock_pipe.zremrangebyscore = MagicMock(return_value=mock_pipe)
        mock_pipe.zcard = MagicMock(return_value=mock_pipe)
        mock_client.pipeline.return_value = mock_pipe

        mock_pipe2 = MagicMock()
        mock_pipe2.execute = AsyncMock(return_value=[True, True])
        mock_pipe2.zadd = MagicMock(return_value=mock_pipe2)
        mock_pipe2.expire = MagicMock(return_value=mock_pipe2)
        mock_client.pipeline.side_effect = [mock_pipe, mock_pipe2]

        s = Settings(rate_limit_requests=100)
        with patch("clsplusplus.rate_limit._redis_client", return_value=mock_client):
            allowed, count, limit = await check_rate_limit("test-key", s)
            assert allowed is True
