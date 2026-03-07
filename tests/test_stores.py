"""Store implementation tests - BaseStore, L0, L1, L2, L3 mock-based tests."""

import pytest

from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.stores.base import BaseStore


# ---------------------------------------------------------------------------
# BaseStore interface
# ---------------------------------------------------------------------------

class TestBaseStore:

    def test_base_is_abstract(self):
        """Cannot instantiate BaseStore directly."""
        with pytest.raises(TypeError):
            BaseStore()

    def test_base_has_required_methods(self):
        assert hasattr(BaseStore, "write")
        assert hasattr(BaseStore, "read")
        assert hasattr(BaseStore, "get_by_id")
        assert hasattr(BaseStore, "delete")
        assert hasattr(BaseStore, "health")


# ---------------------------------------------------------------------------
# MockL0 (tests in-memory L0 behavior)
# ---------------------------------------------------------------------------

class TestMockL0Store:

    @pytest.mark.asyncio
    async def test_write_and_read(self, mock_l0):
        item = MemoryItem(text="hello", namespace="ns1", embedding=[0.1] * 384)
        written = await mock_l0.write(item)
        assert written.store_level == StoreLevel.L0

        items = await mock_l0.read([0.1] * 384, "ns1", limit=10)
        assert len(items) == 1
        assert items[0].text == "hello"

    @pytest.mark.asyncio
    async def test_write_multiple(self, mock_l0):
        for i in range(5):
            await mock_l0.write(MemoryItem(text=f"item-{i}", namespace="ns1"))
        items = await mock_l0.read([], "ns1", limit=10)
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_read_returns_most_recent_first(self, mock_l0):
        await mock_l0.write(MemoryItem(id="first", text="first", namespace="ns1"))
        await mock_l0.write(MemoryItem(id="second", text="second", namespace="ns1"))
        items = await mock_l0.read([], "ns1", limit=10)
        assert items[0].id == "second"  # Most recent first

    @pytest.mark.asyncio
    async def test_read_respects_limit(self, mock_l0):
        for i in range(10):
            await mock_l0.write(MemoryItem(text=f"item-{i}", namespace="ns1"))
        items = await mock_l0.read([], "ns1", limit=3)
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_read_respects_min_confidence(self, mock_l0):
        await mock_l0.write(MemoryItem(text="low", namespace="ns1", confidence=0.2))
        await mock_l0.write(MemoryItem(text="high", namespace="ns1", confidence=0.9))
        items = await mock_l0.read([], "ns1", limit=10, min_confidence=0.5)
        assert all(i.confidence >= 0.5 for i in items)

    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_l0):
        item = MemoryItem(id="find-me", text="hello", namespace="ns1")
        await mock_l0.write(item)
        found = await mock_l0.get_by_id("find-me", "ns1")
        assert found is not None
        assert found.text == "hello"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_l0):
        found = await mock_l0.get_by_id("missing", "ns1")
        assert found is None

    @pytest.mark.asyncio
    async def test_delete(self, mock_l0):
        item = MemoryItem(id="del-me", text="bye", namespace="ns1")
        await mock_l0.write(item)
        deleted = await mock_l0.delete("del-me", "ns1")
        assert deleted is True
        found = await mock_l0.get_by_id("del-me", "ns1")
        assert found is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_l0):
        deleted = await mock_l0.delete("missing", "ns1")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, mock_l0):
        await mock_l0.write(MemoryItem(text="ns1", namespace="ns1"))
        await mock_l0.write(MemoryItem(text="ns2", namespace="ns2"))
        items_ns1 = await mock_l0.read([], "ns1", limit=10)
        items_ns2 = await mock_l0.read([], "ns2", limit=10)
        assert len(items_ns1) == 1
        assert len(items_ns2) == 1

    @pytest.mark.asyncio
    async def test_health(self, mock_l0):
        h = await mock_l0.health()
        assert h["status"] == "healthy"


# ---------------------------------------------------------------------------
# MockPgStore (tests L1/L2/L3 behavior)
# ---------------------------------------------------------------------------

class TestMockPgStore:

    @pytest.mark.asyncio
    async def test_l1_write_and_read(self, mock_l1):
        item = MemoryItem(text="hello", namespace="ns1")
        written = await mock_l1.write(item)
        assert written.store_level == StoreLevel.L1

        items = await mock_l1.read([], "ns1", limit=10)
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_l2_write_sets_level(self, mock_l2):
        item = MemoryItem(text="concept", namespace="ns1")
        written = await mock_l2.write(item)
        assert written.store_level == StoreLevel.L2

    @pytest.mark.asyncio
    async def test_l3_write_sets_level(self, mock_l3):
        item = MemoryItem(text="engram", namespace="ns1")
        written = await mock_l3.write(item)
        assert written.store_level == StoreLevel.L3

    @pytest.mark.asyncio
    async def test_list_for_sleep(self, mock_l1):
        for i in range(5):
            await mock_l1.write(MemoryItem(text=f"item-{i}", namespace="ns1"))
        items = await mock_l1.list_for_sleep("ns1")
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_update_scores(self, mock_l1):
        item = MemoryItem(id="update-me", text="test", namespace="ns1", salience=0.5)
        await mock_l1.write(item)
        await mock_l1.update_scores("update-me", "ns1", salience=0.9)
        found = await mock_l1.get_by_id("update-me", "ns1")
        assert found.salience == 0.9

    @pytest.mark.asyncio
    async def test_delete_from_pg(self, mock_l1):
        item = MemoryItem(id="del-pg", text="test", namespace="ns1")
        await mock_l1.write(item)
        deleted = await mock_l1.delete("del-pg", "ns1")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_health_all_stores(self, mock_l1, mock_l2, mock_l3):
        for store in [mock_l1, mock_l2, mock_l3]:
            h = await store.health()
            assert h["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_l2_decay_edges(self, mock_l2):
        result = await mock_l2.decay_edges("ns1")
        assert result == 0  # No edges in mock

    @pytest.mark.asyncio
    async def test_overwrite_on_conflict(self, mock_l1):
        """Writing same ID should update, not create duplicate."""
        item1 = MemoryItem(id="same-id", text="version1", namespace="ns1")
        item2 = MemoryItem(id="same-id", text="version2", namespace="ns1")
        await mock_l1.write(item1)
        await mock_l1.write(item2)
        found = await mock_l1.get_by_id("same-id", "ns1")
        assert found.text == "version2"

    @pytest.mark.asyncio
    async def test_read_empty_namespace(self, mock_l1):
        items = await mock_l1.read([], "empty-ns", limit=10)
        assert len(items) == 0
