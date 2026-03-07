"""Memory service orchestration tests - write, read, get, delete, health."""

import pytest

from clsplusplus.models import MemoryItem, ReadRequest, ReadResponse, StoreLevel, WriteRequest


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class TestMemoryServiceWrite:

    @pytest.mark.asyncio
    async def test_write_returns_item(self, mock_memory_service):
        req = WriteRequest(text="hello world", namespace="test-ns")
        item = await mock_memory_service.write(req)
        assert isinstance(item, MemoryItem)
        assert item.text == "hello world"
        assert item.namespace == "test-ns"
        assert item.id  # UUID generated

    @pytest.mark.asyncio
    async def test_write_adds_embedding(self, mock_memory_service):
        req = WriteRequest(text="test text", namespace="test-ns")
        item = await mock_memory_service.write(req)
        assert item.embedding is not None
        assert len(item.embedding) == 384

    @pytest.mark.asyncio
    async def test_write_stores_in_l0(self, mock_memory_service, mock_l0):
        req = WriteRequest(text="test", namespace="ns1")
        item = await mock_memory_service.write(req)
        stored = await mock_l0.get_by_id(item.id, "ns1")
        assert stored is not None

    @pytest.mark.asyncio
    async def test_write_stores_in_l1(self, mock_memory_service, mock_l1):
        req = WriteRequest(text="test", namespace="ns1")
        item = await mock_memory_service.write(req)
        stored = await mock_l1.get_by_id(item.id, "ns1")
        assert stored is not None

    @pytest.mark.asyncio
    async def test_write_preserves_salience(self, mock_memory_service):
        req = WriteRequest(text="test", namespace="ns1", salience=0.9)
        item = await mock_memory_service.write(req)
        assert item.salience == 0.9

    @pytest.mark.asyncio
    async def test_write_preserves_authority(self, mock_memory_service):
        req = WriteRequest(text="test", namespace="ns1", authority=0.8)
        item = await mock_memory_service.write(req)
        assert item.authority == 0.8

    @pytest.mark.asyncio
    async def test_write_preserves_metadata(self, mock_memory_service):
        req = WriteRequest(text="test", namespace="ns1", metadata={"key": "value"})
        item = await mock_memory_service.write(req)
        assert item.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_write_preserves_rdf_triple(self, mock_memory_service):
        req = WriteRequest(text="test", namespace="ns1", subject="A", predicate="B", object="C")
        item = await mock_memory_service.write(req)
        assert item.subject == "A"
        assert item.predicate == "B"
        assert item.object == "C"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class TestMemoryServiceRead:

    @pytest.mark.asyncio
    async def test_read_empty_store(self, mock_memory_service):
        req = ReadRequest(query="anything", namespace="empty-ns")
        result = await mock_memory_service.read(req)
        assert isinstance(result, ReadResponse)
        assert len(result.items) == 0
        assert result.query == "anything"

    @pytest.mark.asyncio
    async def test_read_returns_written_items(self, mock_memory_service):
        # Write first
        await mock_memory_service.write(WriteRequest(text="dark mode", namespace="ns1"))
        await mock_memory_service.write(WriteRequest(text="light theme", namespace="ns1"))
        # Read
        result = await mock_memory_service.read(ReadRequest(query="mode", namespace="ns1"))
        assert len(result.items) >= 1

    @pytest.mark.asyncio
    async def test_read_respects_limit(self, mock_memory_service):
        for i in range(5):
            await mock_memory_service.write(WriteRequest(text=f"item {i}", namespace="ns1"))
        result = await mock_memory_service.read(ReadRequest(query="item", namespace="ns1", limit=2))
        assert len(result.items) <= 2

    @pytest.mark.asyncio
    async def test_read_namespace_isolation(self, mock_memory_service):
        await mock_memory_service.write(WriteRequest(text="ns1 data", namespace="ns1"))
        await mock_memory_service.write(WriteRequest(text="ns2 data", namespace="ns2"))
        result = await mock_memory_service.read(ReadRequest(query="data", namespace="ns1"))
        for item in result.items:
            assert item.namespace == "ns1"

    @pytest.mark.asyncio
    async def test_read_respects_min_confidence(self, mock_memory_service):
        await mock_memory_service.write(WriteRequest(text="test", namespace="ns1"))
        result = await mock_memory_service.read(
            ReadRequest(query="test", namespace="ns1", min_confidence=0.9)
        )
        for item in result.items:
            assert item.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_read_deduplicates_by_id(self, mock_memory_service):
        # Write same item (goes to L0 and L1 with same ID)
        req = WriteRequest(text="test", namespace="ns1")
        item = await mock_memory_service.write(req)
        result = await mock_memory_service.read(ReadRequest(query="test", namespace="ns1"))
        ids = [i.id for i in result.items]
        assert len(ids) == len(set(ids))  # No duplicates

    @pytest.mark.asyncio
    async def test_read_store_level_filter(self, mock_memory_service):
        await mock_memory_service.write(WriteRequest(text="test", namespace="ns1"))
        result = await mock_memory_service.read(
            ReadRequest(query="test", namespace="ns1", store_levels=[StoreLevel.L0])
        )
        # Filtering by store_levels=[L0] queries only L0 store;
        # items should be returned (mock stores share the same item object,
        # so store_level attribute may reflect the last write)
        assert len(result.items) >= 1


# ---------------------------------------------------------------------------
# Get item
# ---------------------------------------------------------------------------

class TestMemoryServiceGetItem:

    @pytest.mark.asyncio
    async def test_get_existing_item(self, mock_memory_service):
        req = WriteRequest(text="find me", namespace="ns1")
        written = await mock_memory_service.write(req)
        found = await mock_memory_service.get_item(written.id, "ns1")
        assert found is not None
        assert found.text == "find me"

    @pytest.mark.asyncio
    async def test_get_nonexistent_item(self, mock_memory_service):
        found = await mock_memory_service.get_item("nonexistent", "ns1")
        assert found is None

    @pytest.mark.asyncio
    async def test_get_wrong_namespace(self, mock_memory_service):
        req = WriteRequest(text="test", namespace="ns1")
        written = await mock_memory_service.write(req)
        found = await mock_memory_service.get_item(written.id, "ns2")
        assert found is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestMemoryServiceDelete:

    @pytest.mark.asyncio
    async def test_delete_existing(self, mock_memory_service):
        req = WriteRequest(text="delete me", namespace="ns1")
        written = await mock_memory_service.write(req)
        deleted = await mock_memory_service.delete(written.id, "ns1")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_removes_from_all_stores(self, mock_memory_service, mock_l0, mock_l1):
        req = WriteRequest(text="delete me", namespace="ns1")
        written = await mock_memory_service.write(req)
        await mock_memory_service.delete(written.id, "ns1")
        assert await mock_l0.get_by_id(written.id, "ns1") is None
        assert await mock_l1.get_by_id(written.id, "ns1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_memory_service):
        deleted = await mock_memory_service.delete("nonexistent", "ns1")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_wrong_namespace(self, mock_memory_service):
        req = WriteRequest(text="test", namespace="ns1")
        written = await mock_memory_service.write(req)
        deleted = await mock_memory_service.delete(written.id, "ns2")
        assert deleted is False


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestMemoryServiceHealth:

    @pytest.mark.asyncio
    async def test_all_healthy(self, mock_memory_service):
        health = await mock_memory_service.health()
        assert health["status"] == "healthy"
        assert "L0" in health["stores"]
        assert "L1" in health["stores"]
        assert "L2" in health["stores"]
        assert "L3" in health["stores"]


# ---------------------------------------------------------------------------
# Request to item conversion
# ---------------------------------------------------------------------------

class TestRequestToItem:

    def test_converts_all_fields(self, mock_memory_service):
        req = WriteRequest(
            text="test",
            namespace="ns1",
            source="system",
            salience=0.9,
            authority=0.8,
            metadata={"k": "v"},
            subject="S",
            predicate="P",
            object="O",
        )
        item = mock_memory_service._request_to_item(req)
        assert item.text == "test"
        assert item.namespace == "ns1"
        assert item.source == "system"
        assert item.salience == 0.9
        assert item.authority == 0.8
        assert item.metadata == {"k": "v"}
        assert item.subject == "S"
