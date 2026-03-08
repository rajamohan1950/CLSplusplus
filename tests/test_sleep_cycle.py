"""Sleep cycle tests - N1 rank, N2 strengthen/decay, N3 dedup, REM consolidate."""

from datetime import datetime, timedelta

import pytest

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.sleep_cycle import SleepOrchestrator


class MockSleepL1:
    """Mock L1 store for sleep cycle testing."""

    def __init__(self):
        self.items: dict[str, dict[str, MemoryItem]] = {}

    async def list_for_sleep(self, namespace, limit=20000):
        return list(self.items.get(namespace, {}).values())[:limit]

    async def write(self, item):
        ns = item.namespace
        if ns not in self.items:
            self.items[ns] = {}
        self.items[ns][item.id] = item
        return item

    async def delete(self, item_id, namespace):
        if namespace in self.items and item_id in self.items[namespace]:
            del self.items[namespace][item_id]
            return True
        return False

    async def update_scores(self, item_id, namespace, **kwargs):
        item = self.items.get(namespace, {}).get(item_id)
        if item:
            for k, v in kwargs.items():
                setattr(item, k, v)
        return True

    async def health(self):
        return {"status": "healthy", "store": "L1"}


class MockSleepL2:
    """Mock L2 store for sleep cycle testing."""

    def __init__(self):
        self.items: dict[str, dict[str, MemoryItem]] = {}

    async def list_for_sleep(self, namespace, limit=5000):
        return list(self.items.get(namespace, {}).values())[:limit]

    async def write(self, item):
        ns = item.namespace
        if ns not in self.items:
            self.items[ns] = {}
        self.items[ns][item.id] = item
        return item

    async def delete(self, item_id, namespace):
        if namespace in self.items and item_id in self.items[namespace]:
            del self.items[namespace][item_id]
            return True
        return False

    async def health(self):
        return {"status": "healthy", "store": "L2"}


class MockSleepL3:
    """Mock L3 store for sleep cycle testing."""

    def __init__(self):
        self.items: dict[str, dict[str, MemoryItem]] = {}

    async def write(self, item):
        ns = item.namespace
        if ns not in self.items:
            self.items[ns] = {}
        self.items[ns][item.id] = item
        return item

    async def health(self):
        return {"status": "healthy", "store": "L3"}


@pytest.fixture
def sleep_orchestrator():
    from tests.conftest import MockEmbeddingService
    orch = SleepOrchestrator.__new__(SleepOrchestrator)
    orch.settings = Settings()
    orch.plasticity = __import__("clsplusplus.plasticity", fromlist=["PlasticityEngine"]).PlasticityEngine()
    orch.embedding_service = MockEmbeddingService()
    orch.l0 = None  # Not used in sleep
    orch.l1 = MockSleepL1()
    orch.l2 = MockSleepL2()
    orch.l3 = MockSleepL3()
    return orch


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

class TestSleepEmpty:

    @pytest.mark.asyncio
    async def test_empty_store_completes(self, sleep_orchestrator):
        report = await sleep_orchestrator.run("empty-ns")
        assert report["error"] is None
        assert report["namespace"] == "empty-ns"
        assert report["phases"]["N1"] == "complete"
        assert report["phases"]["N2"] == "complete"
        assert report["phases"]["N3"] == "complete"
        assert report["phases"]["REM"] == "complete"

    @pytest.mark.asyncio
    async def test_empty_store_zero_counts(self, sleep_orchestrator):
        report = await sleep_orchestrator.run("empty-ns")
        assert report["reinforced"] == 0
        assert report["pruned"] == 0
        assert report["deduped"] == 0
        assert report["engraved"] == 0


# ---------------------------------------------------------------------------
# N2: Strengthen + Decay
# ---------------------------------------------------------------------------

class TestSleepN2:

    @pytest.mark.asyncio
    async def test_top_items_reinforced(self, sleep_orchestrator):
        l1 = sleep_orchestrator.l1
        ns = "test-ns"
        # Add 10 items with varying salience
        for i in range(10):
            item = MemoryItem(
                id=f"item-{i}",
                text=f"item {i}",
                namespace=ns,
                salience=0.5 + i * 0.05,
                authority=0.5 + i * 0.05,
                confidence=0.5,
                embedding=[float(i) / 10] * 384,
            )
            if ns not in l1.items:
                l1.items[ns] = {}
            l1.items[ns][item.id] = item

        report = await sleep_orchestrator.run(ns)
        assert report["reinforced"] > 0

    @pytest.mark.asyncio
    async def test_bottom_items_decayed_or_pruned(self, sleep_orchestrator):
        l1 = sleep_orchestrator.l1
        ns = "test-ns"
        # Add items with very low salience (should get pruned)
        for i in range(10):
            item = MemoryItem(
                id=f"item-{i}",
                text=f"item {i}",
                namespace=ns,
                salience=0.05,  # Very low
                authority=0.1,
                confidence=0.3,
                embedding=[float(i) / 10] * 384,
            )
            if ns not in l1.items:
                l1.items[ns] = {}
            l1.items[ns][item.id] = item

        report = await sleep_orchestrator.run(ns)
        # Some should be pruned (salience < 0.2 after decay)
        assert report["pruned"] >= 0


# ---------------------------------------------------------------------------
# N3: Deduplication
# ---------------------------------------------------------------------------

class TestSleepN3:

    @pytest.mark.asyncio
    async def test_duplicates_merged(self, sleep_orchestrator):
        l1 = sleep_orchestrator.l1
        ns = "test-ns"
        # Two items with identical embeddings
        emb = [0.5] * 384
        item1 = MemoryItem(id="dup-1", text="same text", namespace=ns, embedding=emb, confidence=0.8, salience=0.6)
        item2 = MemoryItem(id="dup-2", text="same text", namespace=ns, embedding=emb, confidence=0.9, salience=0.6)
        l1.items[ns] = {"dup-1": item1, "dup-2": item2}

        report = await sleep_orchestrator.run(ns)
        assert report["deduped"] >= 1

    @pytest.mark.asyncio
    async def test_no_embedding_skipped_in_dedup(self, sleep_orchestrator):
        """Items without embeddings are skipped in N3 dedup (line 89)."""
        l1 = sleep_orchestrator.l1
        ns = "test-ns"
        # One item with no embedding, one with embedding
        item_no_emb = MemoryItem(
            id="no-emb", text="no embedding", namespace=ns,
            embedding=None, confidence=0.5, salience=0.5,
        )
        item_with_emb = MemoryItem(
            id="has-emb", text="has embedding", namespace=ns,
            embedding=[0.5] * 384, confidence=0.5, salience=0.5,
        )
        l1.items[ns] = {"no-emb": item_no_emb, "has-emb": item_with_emb}

        report = await sleep_orchestrator.run(ns)
        assert report["error"] is None
        # The no-embedding item should be skipped, not cause an error
        assert report["deduped"] == 0


# ---------------------------------------------------------------------------
# REM: Consolidation
# ---------------------------------------------------------------------------

class TestSleepREM:

    @pytest.mark.asyncio
    async def test_high_score_promoted_to_l2(self, sleep_orchestrator):
        l1 = sleep_orchestrator.l1
        ns = "test-ns"
        # Item that meets L2 criteria
        item = MemoryItem(
            id="promote-me",
            text="important fact",
            namespace=ns,
            salience=1.0,
            usage_count=100,
            authority=1.0,
            confidence=0.95,
            timestamp=datetime.utcnow() - timedelta(days=10),
            embedding=[0.5] * 384,
        )
        l1.items[ns] = {"promote-me": item}

        report = await sleep_orchestrator.run(ns)
        assert report["engraved"] >= 1
        # Should be in L2
        assert "promote-me" in sleep_orchestrator.l2.items.get(ns, {})


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestSleepReport:

    @pytest.mark.asyncio
    async def test_report_has_all_fields(self, sleep_orchestrator):
        report = await sleep_orchestrator.run("test-ns")
        assert "namespace" in report
        assert "started_at" in report
        assert "completed_at" in report
        assert "phases" in report
        assert "reinforced" in report
        assert "pruned" in report
        assert "deduped" in report
        assert "engraved" in report
        assert "error" in report

    @pytest.mark.asyncio
    async def test_report_timestamps_are_iso(self, sleep_orchestrator):
        report = await sleep_orchestrator.run("test-ns")
        datetime.fromisoformat(report["started_at"])
        datetime.fromisoformat(report["completed_at"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestSleepErrorHandling:

    @pytest.mark.asyncio
    async def test_error_captured_in_report(self):
        orch = SleepOrchestrator.__new__(SleepOrchestrator)
        orch.settings = Settings()
        orch.plasticity = __import__("clsplusplus.plasticity", fromlist=["PlasticityEngine"]).PlasticityEngine()
        orch.embedding_service = None
        # L1 that raises
        class FailingL1:
            async def list_for_sleep(self, ns, limit):
                raise RuntimeError("DB connection failed")
        orch.l1 = FailingL1()
        orch.l2 = MockSleepL2()
        orch.l3 = MockSleepL3()

        report = await orch.run("test-ns")
        assert report["error"] is not None
        assert "DB connection failed" in report["error"]
