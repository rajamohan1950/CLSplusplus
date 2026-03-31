"""Embedding service for CLS++."""

from typing import Optional

import numpy as np

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem


class EmbeddingService:
    """Produces embeddings for memory items."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.settings.embedding_model)
            except ImportError:
                self._model = False  # sentinel: unavailable
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns empty list if model unavailable."""
        if self.model is False:
            return []
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns empty lists if model unavailable."""
        if self.model is False:
            return [[] for _ in texts]
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_item(self, item: MemoryItem) -> MemoryItem:
        """Add embedding to a memory item."""
        if not item.embedding:
            item.embedding = self.embed(item.text)
        return item

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        va = np.array(a)
        vb = np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
