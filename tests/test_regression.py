"""Regression tests - edge cases, known bugs, boundary conditions, integration sanity."""

import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from clsplusplus.config import Settings
from clsplusplus.models import (
    AdjudicateRequest,
    DemoChatRequest,
    ForgetRequest,
    MemoryItem,
    ReadRequest,
    ReadResponse,
    StoreLevel,
    WriteRequest,
    _validate_item_id,
    _validate_namespace,
)
from clsplusplus.plasticity import PlasticityEngine
from clsplusplus.reconsolidation import ReconsolidationGate


# ---------------------------------------------------------------------------
# Boundary value analysis
# ---------------------------------------------------------------------------

class TestBoundaryValues:

    def test_namespace_single_char(self):
        assert _validate_namespace("a") == "a"

    def test_namespace_exactly_64_chars(self):
        ns = "a" * 64
        assert _validate_namespace(ns) == ns

    def test_item_id_single_char(self):
        assert _validate_item_id("a") == "a"

    def test_item_id_exactly_64_chars(self):
        iid = "a" * 64
        assert _validate_item_id(iid) == iid

    def test_salience_at_zero(self):
        req = WriteRequest(text="x", salience=0.0)
        assert req.salience == 0.0

    def test_salience_at_one(self):
        req = WriteRequest(text="x", salience=1.0)
        assert req.salience == 1.0

    def test_confidence_at_zero(self):
        item = MemoryItem(text="x", confidence=0.0)
        assert item.confidence == 0.0

    def test_confidence_at_one(self):
        item = MemoryItem(text="x", confidence=1.0)
        assert item.confidence == 1.0

    def test_limit_at_one(self):
        req = ReadRequest(query="x", limit=1)
        assert req.limit == 1

    def test_limit_at_max(self):
        req = ReadRequest(query="x", limit=100)
        assert req.limit == 100

    def test_usage_count_zero(self):
        item = MemoryItem(text="x", usage_count=0)
        engine = PlasticityEngine()
        score = engine.compute_score(item)
        assert score >= 0

    def test_usage_count_very_large(self):
        item = MemoryItem(text="x", usage_count=1_000_000)
        engine = PlasticityEngine()
        score = engine.compute_score(item)
        assert score > 0


# ---------------------------------------------------------------------------
# Edge cases in plasticity
# ---------------------------------------------------------------------------

class TestPlasticityEdgeCases:

    def test_all_zeros_no_crash(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="x",
            salience=0.0,
            usage_count=0,
            authority=0.0,
            conflict_score=0.0,
            surprise=0.0,
        )
        score = engine.compute_score(item)
        assert score == 0.0

    def test_all_max_no_crash(self):
        engine = PlasticityEngine()
        item = MemoryItem(
            text="x",
            salience=1.0,
            usage_count=2**31,
            authority=1.0,
            conflict_score=1.0,
            surprise=1.0,
        )
        score = engine.compute_score(item)
        assert isinstance(score, float)

    def test_repeated_decay_converges_to_zero(self):
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=1.0, confidence=1.0)
        for _ in range(1000):
            engine.apply_decay(item)
        assert item.salience == pytest.approx(0.0, abs=1e-10)
        assert item.confidence == pytest.approx(0.0, abs=1e-10)

    def test_score_idempotent(self):
        """Computing score multiple times should give same result."""
        engine = PlasticityEngine()
        item = MemoryItem(text="x", salience=0.5, authority=0.5, usage_count=5)
        s1 = engine.compute_score(item)
        s2 = engine.compute_score(item)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Edge cases in reconsolidation
# ---------------------------------------------------------------------------

class TestReconsolidationEdgeCases:

    def test_empty_evidence_list(self):
        gate = ReconsolidationGate()
        emb = [0.5] * 384
        a = MemoryItem(text="a b c d e f", embedding=emb)
        b = MemoryItem(text="x y z w v u", embedding=emb)
        # Should not crash with empty evidence
        result = gate.should_overwrite(a, b, evidence=[])
        assert isinstance(result, bool)

    def test_very_long_evidence_list(self):
        gate = ReconsolidationGate()
        emb = [0.5] * 384
        a = MemoryItem(text="a", embedding=emb)
        b = MemoryItem(text="b", embedding=emb)
        evidence = [f"source_{i}" for i in range(1000)]
        result = gate.should_overwrite(a, b, evidence)
        assert isinstance(result, bool)

    def test_identical_items_merge(self):
        gate = ReconsolidationGate()
        emb = [0.5] * 384
        a = MemoryItem(text="exactly same text", embedding=emb)
        b = MemoryItem(text="exactly same text", embedding=emb)
        assert gate.should_merge(a, b) is True

    def test_prepare_empty_lineage(self):
        gate = ReconsolidationGate()
        emb = [0.5] * 384
        old = MemoryItem(id="old", text="old", embedding=emb, version=1, lineage=[])
        new = MemoryItem(id="new", text="new", embedding=emb)
        updated, _, should = gate.prepare_for_reconsolidation(new, old, evidence=["a", "b", "c"])
        if should:
            assert "old" in updated.lineage


# ---------------------------------------------------------------------------
# Serialization edge cases
# ---------------------------------------------------------------------------

class TestSerializationEdgeCases:

    def test_memory_item_with_none_fields(self):
        item = MemoryItem(text="x")
        d = item.to_dict()
        assert d["embedding"] is None
        assert d["subject"] is None

    def test_memory_item_with_special_chars_in_text(self):
        item = MemoryItem(text='He said "hello" & <goodbye>')
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)
        assert restored.text == item.text

    def test_memory_item_with_unicode(self):
        item = MemoryItem(text="日本語テスト 🎉")
        d = item.to_dict()
        json_str = json.dumps(d, default=str)
        restored = MemoryItem.from_dict(json.loads(json_str))
        assert restored.text == item.text

    def test_memory_item_with_newlines(self):
        item = MemoryItem(text="line1\nline2\nline3")
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)
        assert restored.text == item.text

    def test_metadata_with_nested_dict(self):
        item = MemoryItem(text="x", metadata={"nested": {"deep": "value"}})
        d = item.to_dict()
        assert d["metadata"]["nested"]["deep"] == "value"

    def test_empty_metadata(self):
        item = MemoryItem(text="x", metadata={})
        d = item.to_dict()
        assert d["metadata"] == {}

    def test_lineage_with_uuids(self):
        uuids = [str(uuid4()) for _ in range(5)]
        item = MemoryItem(text="x", lineage=uuids)
        d = item.to_dict()
        assert d["lineage"] == uuids


# ---------------------------------------------------------------------------
# Multi-tenant isolation regression
# ---------------------------------------------------------------------------

class TestNamespaceIsolation:

    @pytest.mark.asyncio
    async def test_write_to_different_namespaces(self, mock_memory_service):
        item1 = await mock_memory_service.write(WriteRequest(text="ns1 data", namespace="ns1"))
        item2 = await mock_memory_service.write(WriteRequest(text="ns2 data", namespace="ns2"))
        assert item1.namespace == "ns1"
        assert item2.namespace == "ns2"

    @pytest.mark.asyncio
    async def test_read_isolation(self, mock_memory_service):
        await mock_memory_service.write(WriteRequest(text="secret", namespace="tenant-a"))
        result = await mock_memory_service.read(ReadRequest(query="secret", namespace="tenant-b"))
        for item in result.items:
            assert item.namespace != "tenant-a"

    @pytest.mark.asyncio
    async def test_delete_isolation(self, mock_memory_service):
        item = await mock_memory_service.write(WriteRequest(text="keep me", namespace="ns1"))
        # Try to delete from wrong namespace
        deleted = await mock_memory_service.delete(item.id, "ns2")
        assert deleted is False
        # Original still exists
        found = await mock_memory_service.get_item(item.id, "ns1")
        assert found is not None


# ---------------------------------------------------------------------------
# Store level enumeration
# ---------------------------------------------------------------------------

class TestStoreLevelRegression:

    def test_all_levels_sortable(self):
        levels = [StoreLevel.L3, StoreLevel.L0, StoreLevel.L2, StoreLevel.L1]
        sorted_levels = sorted(levels, key=lambda x: x.value)
        assert sorted_levels == [StoreLevel.L0, StoreLevel.L1, StoreLevel.L2, StoreLevel.L3]

    def test_level_in_dict_keys(self):
        weights = {StoreLevel.L0: 0.5, StoreLevel.L1: 0.7, StoreLevel.L2: 0.9, StoreLevel.L3: 1.0}
        assert weights[StoreLevel.L0] == 0.5

    def test_level_json_serializable(self):
        item = MemoryItem(text="x", store_level=StoreLevel.L2)
        d = item.to_dict()
        assert d["store_level"] == "L2"


# ---------------------------------------------------------------------------
# Health check regression
# ---------------------------------------------------------------------------

class TestHealthRegression:

    @pytest.mark.asyncio
    async def test_health_returns_all_stores(self, mock_memory_service):
        # Current architecture: engine (PhaseMemoryEngine) + L1 + L2 (no L0/L3)
        health = await mock_memory_service.health()
        assert "engine" in health["stores"]
        assert "L1" in health["stores"]
        assert "L2" in health["stores"]

    @pytest.mark.asyncio
    async def test_health_status_field(self, mock_memory_service):
        health = await mock_memory_service.health()
        assert health["status"] in ("healthy", "degraded")


# ---------------------------------------------------------------------------
# API endpoint regression
# ---------------------------------------------------------------------------

class TestAPIRegression:

    @pytest.mark.asyncio
    async def test_root_returns_version(self, client):
        # Root path now returns the CLS++ landing page (HTML).
        # Use /v1/health for version checks.
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        assert "version" in resp.json()

    @pytest.mark.asyncio
    async def test_health_returns_stores(self, client):
        resp = await client.get("/v1/memory/health")
        assert resp.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_concurrent_namespace_validation(self):
        """Multiple rapid validation calls should be thread-safe."""
        import asyncio
        results = await asyncio.gather(*[
            asyncio.to_thread(_validate_namespace, f"ns-{i}")
            for i in range(100)
        ])
        assert all(r.startswith("ns-") for r in results)


# ---------------------------------------------------------------------------
# WriteRequest edge cases
# ---------------------------------------------------------------------------

class TestWriteRequestRegression:

    def test_single_char_text(self):
        req = WriteRequest(text="x")
        assert req.text == "x"

    def test_text_with_only_spaces(self):
        req = WriteRequest(text="   ")
        assert len(req.text) == 3

    def test_text_with_zero_width_chars(self):
        req = WriteRequest(text="test\u200btext")
        assert req.text is not None

    def test_default_source(self):
        req = WriteRequest(text="x")
        assert req.source == "user"

    def test_default_salience(self):
        req = WriteRequest(text="x")
        assert req.salience == 0.5

    def test_default_authority(self):
        req = WriteRequest(text="x")
        assert req.authority == 0.5
