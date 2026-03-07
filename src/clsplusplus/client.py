"""CLS++ Python SDK - simple client for the REST API."""

from typing import Any, Optional

import httpx

from clsplusplus.models import MemoryItem, ReadRequest, ReadResponse, WriteRequest


class MemoriesClient:
    """Memory operations - 3-line integration: client.memories.encode(...)."""

    def __init__(self, client: "CLSClient"):
        self._client = client

    def encode(
        self,
        content: str,
        agent_id: Optional[str] = None,
        namespace: str = "default",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Encode (store) a memory. agent_id maps to namespace for 3-line DX."""
        ns = agent_id if agent_id else namespace
        return self._client.write(text=content, namespace=ns, metadata=metadata or {})

    def retrieve(
        self,
        query: str,
        agent_id: Optional[str] = None,
        namespace: str = "default",
        limit: int = 10,
    ) -> ReadResponse:
        """Retrieve memories by semantic query."""
        ns = agent_id if agent_id else namespace
        return self._client.read(query=query, namespace=ns, limit=limit)


class CLSClient:
    """Python client for CLS++ API."""

    def __init__(self, base_url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=30.0,
        )
        self.memories = MemoriesClient(self)

    def write(
        self,
        text: str,
        namespace: str = "default",
        source: str = "user",
        salience: float = 0.5,
        authority: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Write a memory."""
        req = WriteRequest(
            text=text,
            namespace=namespace,
            source=source,
            salience=salience,
            authority=authority,
            metadata=metadata or {},
        )
        resp = self._client.post("/v1/memory/write", json=req.model_dump())
        resp.raise_for_status()
        return resp.json()

    def read(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> ReadResponse:
        """Read memories by semantic query."""
        req = ReadRequest(
            query=query,
            namespace=namespace,
            limit=limit,
            min_confidence=min_confidence,
        )
        resp = self._client.post("/v1/memory/read", json=req.model_dump())
        resp.raise_for_status()
        return ReadResponse(**resp.json())

    def get_item(self, item_id: str, namespace: str = "default") -> Optional[dict]:
        """Get full item by ID."""
        resp = self._client.get(f"/v1/memory/item/{item_id}", params={"namespace": namespace})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def sleep(self, namespace: str = "default") -> dict:
        """Trigger sleep cycle."""
        resp = self._client.post("/v1/memory/sleep", params={"namespace": namespace})
        resp.raise_for_status()
        return resp.json()

    def forget(self, item_id: str, namespace: str = "default") -> dict:
        """Delete a memory by ID (RTBF)."""
        resp = self._client.request(
            "DELETE", "/v1/memory/forget", json={"item_id": item_id, "namespace": namespace}
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Health check."""
        resp = self._client.get("/v1/memory/health")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self) -> "CLSClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# 3-line integration alias (Stripe-style)
CLS = CLSClient
