"""
GCP Vertex AI adapter — CLS++ as memory for Vertex AI Agent Builder.
"""

from typing import Any, Optional
import httpx


class VertexAICLSAdapter:
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def retrieve(self, query: str, namespace: str = "default", limit: int = 10) -> list[dict[str, Any]]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.api_url}/v1/memories/retrieve",
                headers=self._headers,
                json={"query": query, "namespace": namespace, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return [{"snippet": {"content": i["text"]}, "confidence": i.get("confidence", 0)} for i in data.get("items", [])]

    def encode(self, text: str, namespace: str = "default") -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.api_url}/v1/memories/encode",
                headers=self._headers,
                json={"text": text, "namespace": namespace},
            )
            resp.raise_for_status()
            return resp.json()
