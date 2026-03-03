"""Embedding service tests."""

import pytest

from clsplusplus.embeddings import EmbeddingService


def test_embed():
    svc = EmbeddingService()
    emb = svc.embed("Hello world")
    assert len(emb) == 384
    assert all(isinstance(x, float) for x in emb)


def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(1.0)
    c = [0.0, 1.0, 0.0]
    assert EmbeddingService.cosine_similarity(a, c) == pytest.approx(0.0)
