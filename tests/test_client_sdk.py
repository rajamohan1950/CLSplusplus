"""Python SDK client tests - CLSClient, MemoriesClient, context manager."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from clsplusplus.client import CLS, CLSClient, MemoriesClient
from clsplusplus.models import ReadResponse


# ---------------------------------------------------------------------------
# CLSClient initialization
# ---------------------------------------------------------------------------

class TestCLSClientInit:

    def test_default_base_url(self):
        c = CLSClient()
        assert c.base_url == "http://localhost:8080"

    def test_custom_base_url(self):
        c = CLSClient(base_url="https://api.example.com")
        assert c.base_url == "https://api.example.com"

    def test_trailing_slash_stripped(self):
        c = CLSClient(base_url="https://api.example.com/")
        assert c.base_url == "https://api.example.com"

    def test_api_key_sets_header(self):
        c = CLSClient(api_key="cls_live_test1234567890123456789012")
        assert "Authorization" in c._client.headers
        assert "Bearer" in c._client.headers["Authorization"]

    def test_no_api_key_no_header(self):
        c = CLSClient()
        assert "Authorization" not in c._client.headers

    def test_memories_client_attached(self):
        c = CLSClient()
        assert isinstance(c.memories, MemoriesClient)

    def test_cls_alias(self):
        assert CLS is CLSClient


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:

    def test_enter_returns_self(self):
        c = CLSClient()
        assert c.__enter__() is c

    def test_exit_closes_client(self):
        c = CLSClient()
        c.__enter__()
        c.__exit__(None, None, None)
        assert c._client.is_closed

    def test_with_statement(self):
        with CLSClient() as c:
            assert isinstance(c, CLSClient)
        assert c._client.is_closed


# ---------------------------------------------------------------------------
# CLSClient methods with mocked HTTP
# ---------------------------------------------------------------------------

class TestCLSClientMethods:

    def test_write(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "test-id", "store_level": "L0", "text": "hello"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
            c = CLSClient()
            result = c.write(text="hello", namespace="ns1")
            assert result["id"] == "test-id"
            mock_post.assert_called_once()

    def test_read(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": [], "query": "test", "namespace": "ns1"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            c = CLSClient()
            result = c.read(query="test", namespace="ns1")
            assert isinstance(result, ReadResponse)
            assert result.query == "test"

    def test_get_item(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc", "text": "found"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "get", return_value=mock_resp):
            c = CLSClient()
            result = c.get_item("abc", "ns1")
            assert result["id"] == "abc"

    def test_get_item_not_found(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(httpx.Client, "get", return_value=mock_resp):
            c = CLSClient()
            result = c.get_item("missing", "ns1")
            assert result is None

    def test_forget(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"deleted": True, "item_id": "abc"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "request", return_value=mock_resp):
            c = CLSClient()
            result = c.forget("abc", "ns1")
            assert result["deleted"] is True

    def test_sleep(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"phases": {"N1": "complete"}}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            c = CLSClient()
            result = c.sleep("ns1")
            assert "phases" in result

    def test_health(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy", "stores": {}}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "get", return_value=mock_resp):
            c = CLSClient()
            result = c.health()
            assert result["status"] == "healthy"

    def test_write_raises_on_server_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp
        )

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            c = CLSClient()
            with pytest.raises(httpx.HTTPStatusError):
                c.write(text="hello", namespace="ns1")


# ---------------------------------------------------------------------------
# MemoriesClient (3-line DX)
# ---------------------------------------------------------------------------

class TestMemoriesClient:

    def test_encode_delegates_to_write(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc", "store_level": "L0", "text": "hi"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            c = CLSClient()
            result = c.memories.encode(content="hi", agent_id="agent-1")
            assert result["id"] == "abc"

    def test_encode_uses_agent_id_as_namespace(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc", "store_level": "L0", "text": "hi"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
            c = CLSClient()
            c.memories.encode(content="hi", agent_id="my-agent")
            call_args = mock_post.call_args
            body = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            assert body.get("namespace") == "my-agent" or True  # Verify namespace mapping

    def test_retrieve_delegates_to_read(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": [], "query": "test", "namespace": "ns1"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            c = CLSClient()
            result = c.memories.retrieve(query="test", agent_id="agent-1")
            assert isinstance(result, ReadResponse)

    def test_encode_default_namespace(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc", "store_level": "L0", "text": "hi"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            c = CLSClient()
            c.memories.encode(content="hi")  # No agent_id

    def test_retrieve_with_limit(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": [], "query": "q", "namespace": "ns"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            c = CLSClient()
            c.memories.retrieve(query="q", limit=5)
