"""Comprehensive embedding service tests - dimensions, similarity, edge cases, performance."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clsplusplus.embeddings import EmbeddingService
from clsplusplus.models import MemoryItem


# Check if the sentence-transformers model can actually load (requires numpy compat)
def _model_available():
    try:
        svc = EmbeddingService()
        svc.embed("test")
        return True
    except Exception:
        return False


_skip_no_model = pytest.mark.skipif(
    not _model_available(),
    reason="Sentence-transformers model not available (numpy incompatibility or missing model)",
)


# ---------------------------------------------------------------------------
# Basic embedding
# ---------------------------------------------------------------------------

@_skip_no_model
class TestEmbed:

    def test_embed_returns_384_dims(self):
        svc = EmbeddingService()
        emb = svc.embed("Hello world")
        assert len(emb) == 384

    def test_embed_returns_floats(self):
        svc = EmbeddingService()
        emb = svc.embed("test")
        assert all(isinstance(x, float) for x in emb)

    def test_embed_deterministic(self):
        svc = EmbeddingService()
        e1 = svc.embed("same text")
        e2 = svc.embed("same text")
        assert e1 == e2

    def test_embed_different_texts_different_vectors(self):
        svc = EmbeddingService()
        e1 = svc.embed("cats are great")
        e2 = svc.embed("quantum physics equations")
        assert e1 != e2

    def test_embed_empty_string(self):
        svc = EmbeddingService()
        emb = svc.embed("")
        assert len(emb) == 384

    def test_embed_long_text(self):
        svc = EmbeddingService()
        emb = svc.embed("x " * 10000)
        assert len(emb) == 384

    def test_embed_unicode(self):
        svc = EmbeddingService()
        emb = svc.embed("日本語テスト")
        assert len(emb) == 384

    def test_embed_special_chars(self):
        svc = EmbeddingService()
        emb = svc.embed("!@#$%^&*()[]{}|\\/<>")
        assert len(emb) == 384


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------

@_skip_no_model
class TestEmbedBatch:

    def test_batch_returns_correct_count(self):
        svc = EmbeddingService()
        embs = svc.embed_batch(["a", "b", "c"])
        assert len(embs) == 3

    def test_batch_each_384_dims(self):
        svc = EmbeddingService()
        embs = svc.embed_batch(["hello", "world"])
        for emb in embs:
            assert len(emb) == 384

    def test_batch_empty_list(self):
        svc = EmbeddingService()
        embs = svc.embed_batch([])
        assert len(embs) == 0

    def test_batch_single_item(self):
        svc = EmbeddingService()
        batch = svc.embed_batch(["test"])
        single = svc.embed("test")
        np.testing.assert_allclose(batch[0], single, atol=1e-5)


# ---------------------------------------------------------------------------
# Embed item
# ---------------------------------------------------------------------------

@_skip_no_model
class TestEmbedItem:

    def test_adds_embedding_to_item(self):
        svc = EmbeddingService()
        item = MemoryItem(text="test item")
        result = svc.embed_item(item)
        assert result.embedding is not None
        assert len(result.embedding) == 384

    def test_does_not_overwrite_existing_embedding(self):
        svc = EmbeddingService()
        custom_emb = [0.1] * 384
        item = MemoryItem(text="test", embedding=custom_emb)
        result = svc.embed_item(item)
        assert result.embedding == custom_emb

    def test_returns_same_item(self):
        svc = EmbeddingService()
        item = MemoryItem(text="test")
        result = svc.embed_item(item)
        assert result is item


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        c = [0.0, 1.0, 0.0]
        assert EmbeddingService.cosine_similarity(a, c) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.9, 0.0]
        sim = EmbeddingService.cosine_similarity(a, b)
        assert sim > 0.9

    def test_zero_vector_handling(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = EmbeddingService.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_high_dimensional(self):
        np.random.seed(42)
        a = np.random.randn(384).tolist()
        b = a.copy()
        assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

@_skip_no_model
class TestLazyLoading:

    def test_model_not_loaded_on_init(self):
        svc = EmbeddingService()
        assert svc._model is None

    def test_model_loaded_on_first_use(self):
        svc = EmbeddingService()
        _ = svc.embed("trigger load")
        assert svc._model is not None

    def test_model_reused(self):
        svc = EmbeddingService()
        _ = svc.embed("first")
        model1 = svc._model
        _ = svc.embed("second")
        model2 = svc._model
        assert model1 is model2


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestEmbeddingPerformance:

    @_skip_no_model
    def test_embed_latency(self):
        svc = EmbeddingService()
        svc.embed("warmup")
        start = time.perf_counter_ns()
        for _ in range(10):
            svc.embed("test embedding performance")
        elapsed_ns = (time.perf_counter_ns() - start) / 10
        assert elapsed_ns < 50_000_000, f"Embedding too slow: {elapsed_ns / 1e6:.1f}ms"

    def test_cosine_similarity_latency(self):
        a = [0.1] * 384
        b = [0.2] * 384
        start = time.perf_counter_ns()
        for _ in range(1000):
            EmbeddingService.cosine_similarity(a, b)
        elapsed_ns = (time.perf_counter_ns() - start) / 1000
        assert elapsed_ns < 2_000_000, f"Cosine sim too slow: {elapsed_ns / 1e3:.1f}µs"


# ---------------------------------------------------------------------------
# Mocked model tests (cover embed_batch and embed_item when model is unavailable)
# ---------------------------------------------------------------------------

class TestEmbedWithMockedModel:
    """Tests that use a mocked SentenceTransformer to cover lines 32 and 38."""

    def _make_svc_with_mock_model(self):
        """Create an EmbeddingService with a mocked model."""
        svc = EmbeddingService()
        mock_model = MagicMock()
        # Make encode return a numpy array with tolist() support
        mock_result = MagicMock()
        mock_result.tolist.return_value = [[0.1] * 384, [0.2] * 384]
        mock_model.encode.return_value = mock_result
        svc._model = mock_model
        return svc, mock_model

    def test_embed_batch_calls_model(self):
        """Cover line 32: embed_batch calls model.encode on batch of texts."""
        svc, mock_model = self._make_svc_with_mock_model()
        result = svc.embed_batch(["hello", "world"])
        assert result == [[0.1] * 384, [0.2] * 384]
        mock_model.encode.assert_called_once_with(["hello", "world"], convert_to_numpy=True)

    def test_embed_item_with_existing_embedding(self):
        """Cover line 38: embed_item returns item without re-embedding."""
        svc, mock_model = self._make_svc_with_mock_model()
        existing_emb = [0.5] * 384
        item = MemoryItem(text="already embedded", embedding=existing_emb)
        result = svc.embed_item(item)
        assert result.embedding == existing_emb
        # Model should NOT have been called since item already had embedding
        mock_model.encode.assert_not_called()

    def test_embed_item_without_embedding(self):
        """embed_item embeds when no embedding present."""
        svc, mock_model = self._make_svc_with_mock_model()
        # Override encode for single text
        single_result = MagicMock()
        single_result.tolist.return_value = [0.3] * 384
        mock_model.encode.return_value = single_result
        item = MemoryItem(text="needs embedding")
        result = svc.embed_item(item)
        assert result.embedding == [0.3] * 384
        mock_model.encode.assert_called_once()
