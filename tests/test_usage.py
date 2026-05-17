"""Usage tracking tests - record, retrieve, period keys, Redis interaction."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.usage import (
    _period_key,
    get_operation_count,
    get_usage,
    make_subject,
    record_operation,
    record_usage,
)


# ---------------------------------------------------------------------------
# Period key
# ---------------------------------------------------------------------------

class TestPeriodKey:

    def test_format_yyyy_mm(self):
        key = _period_key()
        assert len(key) == 7  # YYYY-MM
        assert key[4] == "-"

    def test_matches_current_month(self):
        now = datetime.utcnow()
        assert _period_key() == now.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Record usage
# ---------------------------------------------------------------------------

class TestRecordUsage:

    @pytest.mark.asyncio
    async def test_tracking_disabled_noop(self):
        s = Settings(track_usage=False)
        # Should not raise even without Redis
        await record_usage("key1", "write", s)

    @pytest.mark.asyncio
    async def test_tracking_enabled_increments(self):
        mock_client = AsyncMock()
        mock_client.hincrby = AsyncMock()
        mock_client.expire = AsyncMock()

        s = Settings(track_usage=True)
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            await record_usage("key1", "write", s)
            mock_client.hincrby.assert_called_once()
            args = mock_client.hincrby.call_args[0]
            assert "write" in args
            assert args[2] == 1

    @pytest.mark.asyncio
    async def test_expire_set(self):
        mock_client = AsyncMock()
        mock_client.hincrby = AsyncMock()
        mock_client.expire = AsyncMock()

        s = Settings(track_usage=True)
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            await record_usage("key1", "read", s)
            mock_client.expire.assert_called_once()
            ttl = mock_client.expire.call_args[0][1]
            assert ttl == 60 * 60 * 24 * 35  # 35 days

    @pytest.mark.asyncio
    async def test_redis_error_silently_ignored(self):
        mock_client = AsyncMock()
        mock_client.hincrby = AsyncMock(side_effect=ConnectionError("Redis down"))

        s = Settings(track_usage=True)
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            await record_usage("key1", "write", s)  # Should not raise

    @pytest.mark.asyncio
    async def test_various_operations(self):
        mock_client = AsyncMock()
        mock_client.hincrby = AsyncMock()
        mock_client.expire = AsyncMock()

        s = Settings(track_usage=True)
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            for op in ["write", "encode", "read", "retrieve", "knowledge", "consolidate", "forget"]:
                await record_usage("key1", op, s)
            assert mock_client.hincrby.call_count == 7


# ---------------------------------------------------------------------------
# Get usage
# ---------------------------------------------------------------------------

class TestGetUsage:

    @pytest.mark.asyncio
    async def test_empty_usage(self):
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(return_value={})

        s = Settings()
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            result = await get_usage("key1", s)
            assert result["writes"] == 0
            assert result["reads"] == 0
            assert "period" in result

    @pytest.mark.asyncio
    async def test_aggregates_writes(self):
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(return_value={"write": "5", "encode": "3"})

        s = Settings()
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            result = await get_usage("key1", s)
            assert result["writes"] == 8  # 5 + 3

    @pytest.mark.asyncio
    async def test_aggregates_reads(self):
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(return_value={"read": "10", "retrieve": "5", "knowledge": "2"})

        s = Settings()
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            result = await get_usage("key1", s)
            assert result["reads"] == 17  # 10 + 5 + 2

    @pytest.mark.asyncio
    async def test_redis_error_returns_zeros(self):
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(side_effect=ConnectionError("Redis down"))

        s = Settings()
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            result = await get_usage("key1", s)
            assert result["writes"] == 0
            assert result["reads"] == 0

    @pytest.mark.asyncio
    async def test_period_in_response(self):
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(return_value={})

        s = Settings()
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            result = await get_usage("key1", s)
            assert result["period"] == _period_key()


# ---------------------------------------------------------------------------
# Billing subject
# ---------------------------------------------------------------------------

class TestMakeSubject:

    def test_owner_email_produces_owner_subject(self):
        s = make_subject("alice@example.com", "cls_live_abc")
        assert s.startswith("owner:")

    def test_same_owner_collapses_regardless_of_key(self):
        """All of a user's api keys must share one billing subject."""
        a = make_subject("alice@example.com", "cls_live_key_one")
        b = make_subject("alice@example.com", "cls_live_key_two")
        assert a == b

    def test_no_owner_falls_back_to_per_key_subject(self):
        s = make_subject(None, "legacy-key")
        assert s.startswith("key:")

    def test_different_legacy_keys_stay_isolated(self):
        assert make_subject(None, "k1") != make_subject(None, "k2")


# ---------------------------------------------------------------------------
# Record operation — billing-critical counter, NOT gated by track_usage
# ---------------------------------------------------------------------------

class TestRecordOperation:

    @pytest.mark.asyncio
    async def test_records_even_when_track_usage_disabled(self):
        """Quota enforcement reads this counter — it must always be written."""
        mock_client = AsyncMock()
        mock_client.incr = AsyncMock()
        mock_client.expire = AsyncMock()

        s = Settings(track_usage=False)
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            await record_operation("owner:abc", s)
            mock_client.incr.assert_called_once()
            key = mock_client.incr.call_args[0][0]
            assert key.startswith("cls:ops:owner:abc:")

    @pytest.mark.asyncio
    async def test_redis_error_silently_ignored(self):
        mock_client = AsyncMock()
        mock_client.incr = AsyncMock(side_effect=ConnectionError("Redis down"))

        s = Settings(track_usage=False)
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            await record_operation("owner:abc", s)  # must not raise

    @pytest.mark.asyncio
    async def test_get_operation_count_reads_subject_key(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value="7")

        s = Settings()
        with patch("clsplusplus.usage._redis_client", return_value=mock_client):
            count = await get_operation_count("owner:abc", s)
            assert count == 7
            key = mock_client.get.call_args[0][0]
            assert key.startswith("cls:ops:owner:abc:")
