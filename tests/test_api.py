"""API tests - require running services (Redis, Postgres, MinIO) or mocks."""

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.api import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    """Health endpoint returns status."""
    resp = await client.get("/v1/memory/health")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "stores" in data


@pytest.mark.asyncio
async def test_product_aliases_exist(client: AsyncClient):
    """Product-aligned routes exist."""
    resp = await client.get("/v1/health/score")
    assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_validation_rejects_invalid_input(client: AsyncClient):
    """Invalid namespace/item_id returns 422."""
    resp = await client.post(
        "/v1/memory/write",
        json={"text": "x", "namespace": "bad name!"},
    )
    assert resp.status_code == 422

    resp = await client.request(
        "DELETE", "/v1/memory/forget",
        json={"item_id": "bad/id", "namespace": "default"},
    )
    assert resp.status_code == 422
