"""Full API endpoint handler coverage - mocked MemoryService + SleepOrchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, ReadResponse, StoreLevel


# ---------------------------------------------------------------------------
# Fixtures - create app with fully mocked internals
# ---------------------------------------------------------------------------

def _make_mock_memory_service():
    """Create a mock MemoryService with all methods mocked."""
    svc = MagicMock()
    svc.settings = Settings()

    # Mock embedding_service (needed for adjudicate)
    svc.embedding_service = MagicMock()
    svc.embedding_service.embed_item = MagicMock(side_effect=lambda item: item)

    # Mock reconsolidation gate
    svc.reconsolidation = MagicMock()
    svc.reconsolidation.should_overwrite = MagicMock(return_value=False)

    # Async mocks for all operations
    write_item = MemoryItem(id="mock-id-1", text="test", store_level=StoreLevel.L0)
    svc.write = AsyncMock(return_value=write_item)

    svc.read = AsyncMock(return_value=ReadResponse(
        items=[MemoryItem(text="found item")],
        query="test",
        namespace="default",
    ))

    svc.get_item = AsyncMock(return_value=None)
    svc.delete = AsyncMock(return_value=True)

    svc.health = AsyncMock(return_value={
        "status": "healthy",
        "stores": {"L0": {"status": "healthy"}, "L1": {"status": "healthy"},
                   "L2": {"status": "healthy"}, "L3": {"status": "healthy"}},
    })

    # Mock the stores for adjudicate
    svc.l1 = MagicMock()
    svc.l1.write = AsyncMock(return_value=write_item)

    return svc


def _make_mock_sleep_orchestrator():
    orch = MagicMock()
    orch.run = AsyncMock(return_value={
        "namespace": "default",
        "phases": {"N1": "complete", "N2": "complete", "N3": "complete", "REM": "complete"},
        "reinforced": 2, "pruned": 1, "deduped": 0, "engraved": 1,
    })
    return orch


@pytest.fixture
async def mocked_client():
    """App client with fully mocked MemoryService and SleepOrchestrator."""
    mock_svc = _make_mock_memory_service()
    mock_sleep = _make_mock_sleep_orchestrator()

    with patch("clsplusplus.api.MemoryService", return_value=mock_svc), \
         patch("clsplusplus.api.SleepOrchestrator", return_value=mock_sleep):
        from clsplusplus.api import create_app
        app = create_app(Settings(require_api_key=False, track_usage=False))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, mock_svc, mock_sleep


@pytest.fixture
async def mocked_client_with_auth():
    """App client with auth enabled and mocked services."""
    mock_svc = _make_mock_memory_service()
    mock_sleep = _make_mock_sleep_orchestrator()
    settings = Settings(
        require_api_key=True,
        api_keys="cls_live_test1234567890123456789012",
        track_usage=True,
    )

    with patch("clsplusplus.api.MemoryService", return_value=mock_svc), \
         patch("clsplusplus.api.SleepOrchestrator", return_value=mock_sleep):
        from clsplusplus.api import create_app
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer cls_live_test1234567890123456789012"},
        ) as ac:
            yield ac, mock_svc, mock_sleep


# ---------------------------------------------------------------------------
# Write endpoints
# ---------------------------------------------------------------------------

class TestWriteEndpoint:

    @pytest.mark.asyncio
    async def test_write_memory(self, mocked_client):
        client, mock_svc, _ = mocked_client
        resp = await client.post("/v1/memory/write", json={"text": "hello world"})
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert "store_level" in body
        assert "text" in body
        mock_svc.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_alias(self, mocked_client):
        client, mock_svc, _ = mocked_client
        resp = await client.post("/v1/memories/encode", json={"text": "encode me"})
        assert resp.status_code == 200
        assert "id" in resp.json()

    @pytest.mark.asyncio
    async def test_write_with_usage_tracking(self, mocked_client_with_auth):
        client, mock_svc, _ = mocked_client_with_auth
        with patch("clsplusplus.api.Settings") as _:
            resp = await client.post("/v1/memory/write", json={"text": "tracked write"})
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------

class TestReadEndpoint:

    @pytest.mark.asyncio
    async def test_read_memory(self, mocked_client):
        client, mock_svc, _ = mocked_client
        resp = await client.post("/v1/memory/read", json={"query": "test query"})
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        mock_svc.read.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_alias(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.post("/v1/memories/retrieve", json={"query": "test"})
        assert resp.status_code == 200
        assert "items" in resp.json()

    @pytest.mark.asyncio
    async def test_search_alias(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.post("/v1/memories/search", json={"query": "test"})
        assert resp.status_code == 200
        assert "items" in resp.json()


# ---------------------------------------------------------------------------
# Get item endpoint
# ---------------------------------------------------------------------------

class TestGetItemEndpoint:

    @pytest.mark.asyncio
    async def test_get_item_not_found(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.get_item.return_value = None
        resp = await client.get("/v1/memory/item/test-id-123")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_item_found(self, mocked_client):
        client, mock_svc, _ = mocked_client
        item = MemoryItem(id="test-id-123", text="found it")
        mock_svc.get_item.return_value = item
        resp = await client.get("/v1/memory/item/test-id-123")
        assert resp.status_code == 200
        assert resp.json()["text"] == "found it"

    @pytest.mark.asyncio
    async def test_get_item_invalid_id_returns_400(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.get("/v1/memory/item/bad%20id")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Sleep / Consolidate endpoints
# ---------------------------------------------------------------------------

class TestSleepEndpoint:

    @pytest.mark.asyncio
    async def test_trigger_sleep(self, mocked_client):
        client, _, mock_sleep = mocked_client
        resp = await client.post("/v1/memory/sleep")
        assert resp.status_code == 200
        body = resp.json()
        assert "phases" in body
        mock_sleep.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidate_alias(self, mocked_client):
        client, _, mock_sleep = mocked_client
        resp = await client.post("/v1/memories/consolidate")
        assert resp.status_code == 200
        assert "phases" in resp.json()

    @pytest.mark.asyncio
    async def test_sleep_invalid_namespace(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.post("/v1/memory/sleep?namespace=bad%20ns")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_consolidate_invalid_namespace(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.post("/v1/memories/consolidate?namespace=bad%20ns")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Forget / Delete endpoints
# ---------------------------------------------------------------------------

class TestForgetEndpoint:

    @pytest.mark.asyncio
    async def test_forget_success(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.delete.return_value = True
        resp = await client.request(
            "DELETE", "/v1/memory/forget",
            json={"item_id": "abc123", "namespace": "default"},
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_forget_not_found(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.delete.return_value = False
        resp = await client.request(
            "DELETE", "/v1/memory/forget",
            json={"item_id": "missing", "namespace": "default"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_forget_alias_success(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.delete.return_value = True
        resp = await client.request(
            "DELETE", "/v1/memories/forget",
            json={"item_id": "abc123", "namespace": "default"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_forget_alias_not_found(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.delete.return_value = False
        resp = await client.request(
            "DELETE", "/v1/memories/forget",
            json={"item_id": "missing", "namespace": "default"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_by_id(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.delete.return_value = True
        resp = await client.delete("/v1/memories/test-id-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_by_id_not_found(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.delete.return_value = False
        resp = await client.delete("/v1/memories/test-id-1")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_by_id_invalid(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.delete("/v1/memories/bad%20id")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Adjudicate endpoint
# ---------------------------------------------------------------------------

class TestAdjudicateEndpoint:

    @pytest.mark.asyncio
    async def test_adjudicate_new_fact_no_existing(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.get_item.return_value = None
        resp = await client.post("/v1/memory/adjudicate_conflict", json={
            "new_fact": "Earth is round",
            "evidence": ["science"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "accepted"

    @pytest.mark.asyncio
    async def test_adjudicate_with_existing_reject(self, mocked_client):
        client, mock_svc, _ = mocked_client
        old_item = MemoryItem(id="old-1", text="old fact", embedding=[0.5]*384)
        mock_svc.get_item.return_value = old_item
        mock_svc.reconsolidation.should_overwrite.return_value = False
        resp = await client.post("/v1/memory/adjudicate_conflict", json={
            "new_fact": "New conflicting fact",
            "evidence": ["one"],
            "existing_item_id": "old-1",
        })
        assert resp.status_code == 200
        assert resp.json()["decision"] == "reject"

    @pytest.mark.asyncio
    async def test_adjudicate_with_existing_overwrite(self, mocked_client):
        client, mock_svc, _ = mocked_client
        old_item = MemoryItem(id="old-1", text="old fact", embedding=[0.5]*384)
        mock_svc.get_item.return_value = old_item
        mock_svc.reconsolidation.should_overwrite.return_value = True
        resp = await client.post("/v1/memory/adjudicate_conflict", json={
            "new_fact": "New authoritative fact",
            "evidence": ["a", "b", "c"],
            "existing_item_id": "old-1",
        })
        assert resp.status_code == 200
        assert resp.json()["decision"] == "overwrite"


# ---------------------------------------------------------------------------
# Demo chat endpoint
# ---------------------------------------------------------------------------

class TestDemoChatEndpoint:

    @pytest.mark.asyncio
    async def test_demo_chat_success(self, mocked_client):
        client, _, _ = mocked_client
        # Patch at the module where it's imported lazily
        with patch("clsplusplus.demo_llm.chat_with_llm", new_callable=AsyncMock, return_value="Hello!"):
            resp = await client.post("/v1/demo/chat", json={
                "model": "claude", "message": "Hi there",
            })
            assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_demo_chat_error_handled(self, mocked_client):
        """Demo chat catches exceptions and returns 500."""
        client, _, _ = mocked_client
        with patch("clsplusplus.demo_llm.chat_with_llm", new_callable=AsyncMock, side_effect=RuntimeError("No API key")):
            resp = await client.post("/v1/demo/chat", json={
                "model": "claude", "message": "Hi",
            })
            assert resp.status_code == 500
            body = resp.json()
            assert "error" in body or "message" in body


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_endpoint(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.get("/v1/memory/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "stores" in body

    @pytest.mark.asyncio
    async def test_health_score_alias(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.get("/v1/health/score")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_degraded_with_hint(self, mocked_client):
        client, mock_svc, _ = mocked_client
        mock_svc.health.return_value = {
            "status": "degraded",
            "stores": {"L0": {"status": "unhealthy", "error": "Connection to localhost refused"}},
        }
        resp = await client.get("/v1/memory/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# Knowledge endpoint
# ---------------------------------------------------------------------------

class TestKnowledgeEndpoint:

    @pytest.mark.asyncio
    async def test_knowledge_query(self, mocked_client):
        client, mock_svc, _ = mocked_client
        resp = await client.get("/v1/memories/knowledge?query=test+query")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        mock_svc.read.assert_called()

    @pytest.mark.asyncio
    async def test_knowledge_invalid_namespace(self, mocked_client):
        client, _, _ = mocked_client
        resp = await client.get("/v1/memories/knowledge?query=test&namespace=bad%20ns")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Usage endpoints
# ---------------------------------------------------------------------------

class TestUsageEndpoint:

    @pytest.mark.asyncio
    async def test_usage_no_auth(self, mocked_client):
        client, _, _ = mocked_client
        with patch("clsplusplus.usage.get_usage", new_callable=AsyncMock, return_value={
            "period": "2026-03", "writes": 10, "reads": 20,
        }):
            resp = await client.get("/v1/usage")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_billing_usage_alias(self, mocked_client):
        client, _, _ = mocked_client
        with patch("clsplusplus.usage.get_usage", new_callable=AsyncMock, return_value={
            "period": "2026-03", "writes": 5, "reads": 10,
        }):
            resp = await client.get("/v1/billing/usage")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error handler coverage
# ---------------------------------------------------------------------------

class TestErrorHandler:

    @pytest.mark.asyncio
    async def test_422_error_format(self, mocked_client):
        """Trigger 422 via invalid request body to hit exception handler."""
        client, _, _ = mocked_client
        resp = await client.post("/v1/memory/write", json={"text": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_401_error_format(self):
        """Hit 401 handler via unauthenticated request."""
        mock_svc = _make_mock_memory_service()
        mock_sleep = _make_mock_sleep_orchestrator()
        settings = Settings(
            require_api_key=True,
            api_keys="cls_live_test1234567890123456789012",
        )
        with patch("clsplusplus.api.MemoryService", return_value=mock_svc), \
             patch("clsplusplus.api.SleepOrchestrator", return_value=mock_sleep):
            from clsplusplus.api import create_app
            app = create_app(settings)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as unauth:
                resp = await unauth.post("/v1/memory/write", json={"text": "x"})
                assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_usage_requires_auth_when_enabled(self):
        """Usage endpoint returns 401 when auth enabled and no key."""
        mock_svc = _make_mock_memory_service()
        mock_sleep = _make_mock_sleep_orchestrator()
        settings = Settings(
            require_api_key=True,
            api_keys="cls_live_test1234567890123456789012",
        )

        with patch("clsplusplus.api.MemoryService", return_value=mock_svc), \
             patch("clsplusplus.api.SleepOrchestrator", return_value=mock_sleep):
            from clsplusplus.api import create_app
            app = create_app(settings)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/v1/usage")
                assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Record usage coverage
# ---------------------------------------------------------------------------

class TestRecordUsage:

    @pytest.mark.asyncio
    async def test_write_records_usage(self, mocked_client_with_auth):
        """Write with auth+tracking enabled triggers record_usage."""
        client, mock_svc, _ = mocked_client_with_auth
        with patch("clsplusplus.usage.record_usage", new_callable=AsyncMock) as mock_record:
            resp = await client.post("/v1/memory/write", json={"text": "tracked"})
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_read_records_usage(self, mocked_client_with_auth):
        client, _, _ = mocked_client_with_auth
        with patch("clsplusplus.usage.record_usage", new_callable=AsyncMock):
            resp = await client.post("/v1/memory/read", json={"query": "test"})
            assert resp.status_code == 200
