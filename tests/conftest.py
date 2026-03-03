"""Pytest fixtures."""

import asyncio
from typing import AsyncGenerator, Generator

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator:
    from httpx import ASGITransport, AsyncClient
    from clsplusplus.api import create_app
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
