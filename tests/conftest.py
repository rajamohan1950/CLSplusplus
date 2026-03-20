"""Comprehensive test fixtures - mocks for Redis, PostgreSQL, LLM APIs."""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, StoreLevel


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_settings() -> Settings:
    return Settings(
        require_api_key=False,
        redis_url="redis://localhost:6379",
        database_url="postgresql://cls:cls@localhost:5432/cls",
        track_usage=False,
    )


@pytest.fixture
def auth_settings() -> Settings:
    return Settings(
        require_api_key=True,
        api_keys="cls_live_test1234567890123456789012",
        rate_limit_requests=100,
        rate_limit_window_seconds=60,
        track_usage=True,
    )


VALID_API_KEY = "cls_live_test1234567890123456789012"
VALID_TEST_KEY = "cls_test_abcdefghijklmnopqrstuvwx"


# ---------------------------------------------------------------------------
# In-memory store mocks (replaces Redis + PostgreSQL)
# ---------------------------------------------------------------------------

class InMemoryStore:
    """Simulates Redis + PostgreSQL for testing without external deps."""

    def __init__(self):
        self.data: dict[str, Any] = {}
        self.lists: dict[str, list] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.ttls: dict[str, float] = {}

    async def ping(self):
        return True

    async def set(self, key: str, value: str, ex: int = None):
        self.data[key] = value
        if ex:
            self.ttls[key] = time.time() + ex

    async def get(self, key: str) -> Optional[str]:
        if key in self.ttls and time.time() > self.ttls[key]:
            del self.data[key]
            del self.ttls[key]
            return None
        return self.data.get(key)

    async def delete(self, *keys: str):
        for k in keys:
            self.data.pop(k, None)
            self.ttls.pop(k, None)
        return len(keys)

    async def lpush(self, key: str, *values):
        if key not in self.lists:
            self.lists[key] = []
        for v in values:
            self.lists[key].insert(0, v)

    async def ltrim(self, key: str, start: int, end: int):
        if key in self.lists:
            self.lists[key] = self.lists[key][start:end + 1]

    async def lrange(self, key: str, start: int, end: int) -> list:
        if key not in self.lists:
            return []
        return self.lists[key][start:end + 1]

    async def lrem(self, key: str, count: int, value: str):
        if key in self.lists:
            try:
                self.lists[key].remove(value)
            except ValueError:
                pass

    async def setex(self, key: str, ttl: int, value: str):
        self.data[key] = value
        self.ttls[key] = time.time() + ttl

    async def hincrby(self, key: str, field: str, amount: int = 1):
        if key not in self.hashes:
            self.hashes[key] = {}
        current = int(self.hashes[key].get(field, "0"))
        self.hashes[key][field] = str(current + amount)

    async def hgetall(self, key: str) -> dict:
        return self.hashes.get(key, {})

    async def expire(self, key: str, ttl: int):
        self.ttls[key] = time.time() + ttl

    async def zadd(self, key: str, mapping: dict):
        if key not in self.zsets:
            self.zsets[key] = {}
        self.zsets[key].update(mapping)

    async def zremrangebyscore(self, key: str, min_score, max_score):
        if key in self.zsets:
            if isinstance(min_score, str) and min_score == "-inf":
                min_score = float("-inf")
            self.zsets[key] = {
                k: v for k, v in self.zsets[key].items()
                if v > float(max_score)
            }

    async def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, {}))

    def pipeline(self):
        return InMemoryPipeline(self)


class InMemoryPipeline:
    """Pipeline mock that batches commands."""

    def __init__(self, store: InMemoryStore):
        self.store = store
        self.commands: list = []

    def set(self, key, value, ex=None):
        self.commands.append(("set", key, value, ex))
        return self

    def lpush(self, key, *values):
        self.commands.append(("lpush", key, *values))
        return self

    def ltrim(self, key, start, end):
        self.commands.append(("ltrim", key, start, end))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def zremrangebyscore(self, key, min_s, max_s):
        self.commands.append(("zremrangebyscore", key, min_s, max_s))
        return self

    def zcard(self, key):
        self.commands.append(("zcard", key))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    async def execute(self):
        results = []
        for cmd in self.commands:
            op = cmd[0]
            if op == "set":
                await self.store.set(cmd[1], cmd[2], cmd[3])
                results.append(True)
            elif op == "lpush":
                await self.store.lpush(cmd[1], *cmd[2:])
                results.append(True)
            elif op == "ltrim":
                await self.store.ltrim(cmd[1], cmd[2], cmd[3])
                results.append(True)
            elif op == "zadd":
                await self.store.zadd(cmd[1], cmd[2])
                results.append(True)
            elif op == "zremrangebyscore":
                await self.store.zremrangebyscore(cmd[1], cmd[2], cmd[3])
                results.append(True)
            elif op == "zcard":
                count = await self.store.zcard(cmd[1])
                results.append(count)
            elif op == "expire":
                await self.store.expire(cmd[1], cmd[2])
                results.append(True)
        self.commands = []
        return results


@pytest.fixture
def in_memory_store():
    return InMemoryStore()


# ---------------------------------------------------------------------------
# Mock stores that don't need Redis/Postgres
# ---------------------------------------------------------------------------

class MockL0Store:
    """In-memory L0 working buffer mock."""

    def __init__(self):
        self.items: dict[str, dict[str, MemoryItem]] = {}  # ns -> {id -> item}
        self.order: dict[str, list[str]] = {}  # ns -> [ids]

    async def write(self, item: MemoryItem) -> MemoryItem:
        item.store_level = StoreLevel.L0
        ns = item.namespace
        if ns not in self.items:
            self.items[ns] = {}
            self.order[ns] = []
        self.items[ns][item.id] = item
        self.order[ns].insert(0, item.id)
        self.order[ns] = self.order[ns][:1000]
        return item

    async def read(self, query_embedding, namespace, limit=10, min_confidence=0.0):
        ns_items = self.items.get(namespace, {})
        ids = self.order.get(namespace, [])[:limit]
        return [ns_items[iid] for iid in ids if iid in ns_items and ns_items[iid].confidence >= min_confidence]

    async def get_by_id(self, item_id, namespace):
        return self.items.get(namespace, {}).get(item_id)

    async def delete(self, item_id, namespace):
        if namespace in self.items and item_id in self.items[namespace]:
            del self.items[namespace][item_id]
            if item_id in self.order.get(namespace, []):
                self.order[namespace].remove(item_id)
            return True
        return False

    async def health(self):
        return {"status": "healthy", "store": "L0"}


class MockPgStore:
    """In-memory PostgreSQL store mock for L1/L2/L3."""

    def __init__(self, level: StoreLevel):
        self.level = level
        self.items: dict[str, dict[str, MemoryItem]] = {}

    async def write(self, item: MemoryItem, **kwargs) -> MemoryItem:
        item.store_level = self.level
        ns = item.namespace
        if ns not in self.items:
            self.items[ns] = {}
        self.items[ns][item.id] = item
        return item

    async def read(self, query_embedding, namespace, limit=10, min_confidence=0.0):
        ns_items = self.items.get(namespace, {})
        results = [i for i in ns_items.values() if i.confidence >= min_confidence]
        return results[:limit]

    async def get_by_id(self, item_id, namespace):
        return self.items.get(namespace, {}).get(item_id)

    async def delete(self, item_id, namespace):
        if namespace in self.items and item_id in self.items[namespace]:
            del self.items[namespace][item_id]
            return True
        return False

    async def list_for_sleep(self, namespace, limit=20000):
        return list(self.items.get(namespace, {}).values())[:limit]

    async def update_scores(self, item_id, namespace, **kwargs):
        item = self.items.get(namespace, {}).get(item_id)
        if item:
            for k, v in kwargs.items():
                setattr(item, k, v)
        return True

    async def decay_edges(self, namespace, decay_factor=0.95):
        return 0

    async def health(self):
        return {"status": "healthy", "store": self.level.value}


@pytest.fixture
def mock_l0():
    return MockL0Store()


@pytest.fixture
def mock_l1():
    return MockPgStore(StoreLevel.L1)


@pytest.fixture
def mock_l2():
    return MockPgStore(StoreLevel.L2)


@pytest.fixture
def mock_l3():
    return MockPgStore(StoreLevel.L3)


# ---------------------------------------------------------------------------
# Mock embedding service (fast, deterministic)
# ---------------------------------------------------------------------------

class MockEmbeddingService:
    """Deterministic embeddings for testing - no model download needed."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        vec = [float(int(h[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
        return vec + [0.0] * (384 - len(vec))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def embed_item(self, item: MemoryItem) -> MemoryItem:
        if not item.embedding:
            item.embedding = self.embed(item.text)
        return item

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        import numpy as np
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


@pytest.fixture
def mock_embedding_service():
    return MockEmbeddingService()


# ---------------------------------------------------------------------------
# Memory service with all mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_memory_service(mock_l0, mock_l1, mock_l2, mock_l3, mock_embedding_service):
    from clsplusplus.memory_service import MemoryService
    from clsplusplus.memory_phase import PhaseMemoryEngine
    svc = MemoryService.__new__(MemoryService)
    svc.settings = Settings()
    svc.embedding_service = mock_embedding_service
    svc.reconsolidation = __import__("clsplusplus.reconsolidation", fromlist=["ReconsolidationGate"]).ReconsolidationGate()
    # Current architecture: PhaseMemoryEngine is the brain; L1 is persistence
    svc.engine = PhaseMemoryEngine()
    svc.l1 = mock_l1
    svc.l2 = mock_l2
    svc._webhook_dispatcher = None
    # Internal state required by MemoryService methods
    svc._loaded_namespaces = set()
    svc._loading_namespaces = set()
    svc._write_counts = {}
    svc._event_threads = {}
    svc._event_threads_lock = __import__("asyncio").Lock()
    return svc


# ---------------------------------------------------------------------------
# App client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def client() -> AsyncGenerator:
    """Default app (no auth) with mocked stores."""
    from clsplusplus.api import create_app
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def client_no_auth() -> AsyncGenerator:
    """App with auth disabled."""
    from clsplusplus.api import create_app
    settings = Settings(require_api_key=False)
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def client_with_auth() -> AsyncGenerator:
    """App with auth enabled and valid key in headers."""
    from clsplusplus.api import create_app
    settings = Settings(
        require_api_key=True,
        api_keys=VALID_API_KEY,
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    ) as ac:
        yield ac


@pytest.fixture
async def client_unauth() -> AsyncGenerator:
    """App with auth enabled but no key in headers."""
    from clsplusplus.api import create_app
    settings = Settings(
        require_api_key=True,
        api_keys=VALID_API_KEY,
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_memory_item() -> MemoryItem:
    return MemoryItem(
        id="test-item-001",
        text="User prefers dark mode",
        namespace="test-ns",
        store_level=StoreLevel.L0,
        source="user",
        confidence=0.8,
        salience=0.7,
        authority=0.9,
        usage_count=5,
        conflict_score=0.0,
        surprise=0.3,
        embedding=[0.1] * 384,
    )


@pytest.fixture
def sample_memory_item_low() -> MemoryItem:
    return MemoryItem(
        id="test-item-002",
        text="Temporary note",
        namespace="test-ns",
        store_level=StoreLevel.L0,
        source="user",
        confidence=0.2,
        salience=0.1,
        authority=0.2,
        usage_count=0,
    )


@pytest.fixture
def sample_conflicting_items() -> tuple[MemoryItem, MemoryItem]:
    new = MemoryItem(
        id="new-fact",
        text="The capital of France is Berlin",
        namespace="test-ns",
        embedding=[0.5] * 384,
        confidence=0.6,
    )
    old = MemoryItem(
        id="old-fact",
        text="The capital of France is Paris",
        namespace="test-ns",
        embedding=[0.5] * 384,
        confidence=0.9,
    )
    return new, old
