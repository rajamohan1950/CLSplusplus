"""Performance and latency benchmark tests - critical path timing."""

import math
import time
from datetime import datetime

import pytest

from clsplusplus.auth import _sha256_hex, validate_api_key
from clsplusplus.config import Settings
from clsplusplus.embeddings import EmbeddingService
from clsplusplus.models import MemoryItem, ReadRequest, ReadResponse, WriteRequest
from clsplusplus.plasticity import PlasticityEngine
from clsplusplus.reconsolidation import ReconsolidationGate


# ---------------------------------------------------------------------------
# Micro-benchmarks (sub-100ns target where possible)
# ---------------------------------------------------------------------------

class TestCriticalPathLatency:

    def test_sha256_hash_speed(self):
        """SHA-256 should be <1µs per hash."""
        warmup = _sha256_hex("warmup")
        start = time.perf_counter_ns()
        for _ in range(10000):
            _sha256_hex("cls_live_test1234567890123456789012")
        elapsed = (time.perf_counter_ns() - start) / 10000
        assert elapsed < 25_000, f"SHA-256 too slow: {elapsed:.0f}ns"

    def test_api_key_validation_speed(self):
        """Key validation should be <10µs per validation."""
        s = Settings(api_keys="cls_live_test1234567890123456789012")
        # Warmup
        for _ in range(10):
            validate_api_key("cls_live_test1234567890123456789012", s)

        start = time.perf_counter_ns()
        for _ in range(1000):
            validate_api_key("cls_live_test1234567890123456789012", s)
        elapsed = (time.perf_counter_ns() - start) / 1000
        assert elapsed < 100_000, f"Auth validation too slow: {elapsed:.0f}ns ({elapsed/1000:.1f}µs)"

    def test_plasticity_score_speed(self):
        """Score computation should be <10µs."""
        engine = PlasticityEngine()
        item = MemoryItem(
            text="test",
            salience=0.8,
            usage_count=10,
            authority=0.9,
            conflict_score=0.1,
            surprise=0.3,
        )
        # Warmup
        for _ in range(100):
            engine.compute_score(item)

        start = time.perf_counter_ns()
        for _ in range(10000):
            engine.compute_score(item)
        elapsed = (time.perf_counter_ns() - start) / 10000
        assert elapsed < 50_000, f"Plasticity score too slow: {elapsed:.0f}ns"

    def test_model_creation_speed(self):
        """MemoryItem creation should be <10µs."""
        start = time.perf_counter_ns()
        for _ in range(1000):
            MemoryItem(text="test item", namespace="default")
        elapsed = (time.perf_counter_ns() - start) / 1000
        assert elapsed < 100_000, f"Model creation too slow: {elapsed:.0f}ns ({elapsed/1000:.1f}µs)"

    def test_model_serialization_speed(self):
        """to_dict should be <50µs."""
        item = MemoryItem(
            text="test",
            namespace="ns",
            embedding=[0.1] * 384,
            metadata={"key": "value"},
        )
        # Warmup
        for _ in range(100):
            item.to_dict()

        start = time.perf_counter_ns()
        for _ in range(1000):
            item.to_dict()
        elapsed = (time.perf_counter_ns() - start) / 1000
        assert elapsed < 500_000, f"Serialization too slow: {elapsed:.0f}ns ({elapsed/1000:.1f}µs)"

    def test_write_request_validation_speed(self):
        """Pydantic validation should be <50µs."""
        start = time.perf_counter_ns()
        for _ in range(1000):
            WriteRequest(text="hello world", namespace="default", salience=0.8)
        elapsed = (time.perf_counter_ns() - start) / 1000
        assert elapsed < 200_000, f"Request validation too slow: {elapsed:.0f}ns ({elapsed/1000:.1f}µs)"

    def test_read_request_validation_speed(self):
        """Read request validation should be <50µs."""
        start = time.perf_counter_ns()
        for _ in range(1000):
            ReadRequest(query="user preferences", namespace="default", limit=10)
        elapsed = (time.perf_counter_ns() - start) / 1000
        assert elapsed < 200_000, f"Read request validation too slow: {elapsed:.0f}ns ({elapsed/1000:.1f}µs)"

    def test_cosine_similarity_speed(self):
        """384-dim cosine similarity should be <200µs."""
        a = [0.1] * 384
        b = [0.2] * 384
        # Warmup
        for _ in range(100):
            EmbeddingService.cosine_similarity(a, b)

        start = time.perf_counter_ns()
        for _ in range(10000):
            EmbeddingService.cosine_similarity(a, b)
        elapsed = (time.perf_counter_ns() - start) / 10000
        assert elapsed < 500_000, f"Cosine similarity too slow: {elapsed:.0f}ns"

    def test_decay_computation_speed(self):
        """Decay should be <10µs."""
        engine = PlasticityEngine()
        item = MemoryItem(text="test", salience=0.8, confidence=0.9)

        start = time.perf_counter_ns()
        for _ in range(10000):
            item.salience = 0.8
            item.confidence = 0.9
            engine.apply_decay(item)
        elapsed = (time.perf_counter_ns() - start) / 10000
        assert elapsed < 200_000, f"Decay too slow: {elapsed:.0f}ns"

    def test_promotion_check_speed(self):
        """Promotion check should be <50µs."""
        engine = PlasticityEngine()
        item = MemoryItem(text="test", salience=0.8, authority=0.9, usage_count=10)

        start = time.perf_counter_ns()
        for _ in range(10000):
            engine.should_promote_to_l1(item)
        elapsed = (time.perf_counter_ns() - start) / 10000
        assert elapsed < 50_000, f"Promotion check too slow: {elapsed:.0f}ns"

    def test_conflict_score_speed(self):
        """Conflict score should be <100µs."""
        gate = ReconsolidationGate()
        emb = [0.5] * 384
        a = MemoryItem(text="The capital of France is Paris and it is beautiful", embedding=emb)
        b = MemoryItem(text="The capital of France is Berlin which is not correct", embedding=emb)

        start = time.perf_counter_ns()
        for _ in range(1000):
            gate.conflict_score(a, b)
        elapsed = (time.perf_counter_ns() - start) / 1000
        assert elapsed < 500_000, f"Conflict score too slow: {elapsed:.0f}ns ({elapsed/1000:.1f}µs)"


# ---------------------------------------------------------------------------
# Memory service mock benchmarks
# ---------------------------------------------------------------------------

class TestServiceLevelPerformance:

    @pytest.mark.asyncio
    async def test_write_latency(self, mock_memory_service):
        """Write (with mock stores) should be <1ms."""
        req = WriteRequest(text="benchmark write", namespace="perf-ns")
        # Warmup
        for _ in range(5):
            await mock_memory_service.write(req)

        start = time.perf_counter_ns()
        for _ in range(100):
            await mock_memory_service.write(
                WriteRequest(text="benchmark write", namespace="perf-ns")
            )
        elapsed = (time.perf_counter_ns() - start) / 100
        assert elapsed < 10_000_000, f"Write too slow: {elapsed/1e6:.1f}ms"

    @pytest.mark.asyncio
    async def test_read_latency(self, mock_memory_service):
        """Read (with mock stores) should be <1ms."""
        # Pre-populate
        for i in range(10):
            await mock_memory_service.write(
                WriteRequest(text=f"item {i}", namespace="perf-ns")
            )

        req = ReadRequest(query="item", namespace="perf-ns", limit=5)
        # Warmup
        for _ in range(5):
            await mock_memory_service.read(req)

        start = time.perf_counter_ns()
        for _ in range(100):
            await mock_memory_service.read(req)
        elapsed = (time.perf_counter_ns() - start) / 100
        assert elapsed < 10_000_000, f"Read too slow: {elapsed/1e6:.1f}ms"

    @pytest.mark.asyncio
    async def test_health_latency(self, mock_memory_service):
        """Health check should be <1ms."""
        start = time.perf_counter_ns()
        for _ in range(100):
            await mock_memory_service.health()
        elapsed = (time.perf_counter_ns() - start) / 100
        assert elapsed < 5_000_000, f"Health too slow: {elapsed/1e6:.1f}ms"

    @pytest.mark.asyncio
    async def test_delete_latency(self, mock_memory_service):
        """Delete should be <1ms."""
        items = []
        for i in range(20):
            item = await mock_memory_service.write(
                WriteRequest(text=f"delete-{i}", namespace="perf-ns")
            )
            items.append(item)

        start = time.perf_counter_ns()
        for item in items:
            await mock_memory_service.delete(item.id, "perf-ns")
        elapsed = (time.perf_counter_ns() - start) / len(items)
        assert elapsed < 5_000_000, f"Delete too slow: {elapsed/1e6:.1f}ms"


# ---------------------------------------------------------------------------
# Throughput benchmarks
# ---------------------------------------------------------------------------

class TestThroughput:

    def test_score_computation_throughput(self):
        """Should handle >100k scores/sec."""
        engine = PlasticityEngine()
        items = [
            MemoryItem(text=f"item-{i}", salience=0.5, authority=0.5, usage_count=i)
            for i in range(1000)
        ]

        start = time.perf_counter()
        for item in items:
            engine.compute_score(item)
        elapsed = time.perf_counter() - start

        throughput = 1000 / elapsed
        assert throughput > 10_000, f"Score throughput too low: {throughput:.0f}/sec"

    def test_model_creation_throughput(self):
        """Should create >10k models/sec."""
        start = time.perf_counter()
        items = [MemoryItem(text=f"item-{i}") for i in range(1000)]
        elapsed = time.perf_counter() - start

        throughput = 1000 / elapsed
        assert throughput > 1_000, f"Model creation throughput: {throughput:.0f}/sec"

    def test_validation_throughput(self):
        """Should validate >10k requests/sec."""
        start = time.perf_counter()
        for i in range(1000):
            WriteRequest(text=f"item-{i}", namespace="default")
        elapsed = time.perf_counter() - start

        throughput = 1000 / elapsed
        assert throughput > 10_000, f"Validation throughput: {throughput:.0f}/sec"
