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
    # May be healthy or degraded if services not running
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "stores" in data
