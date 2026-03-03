"""Reconsolidation Gate - belief revision with evidence quorum.

Never overwrite without evidence. Archive old, engrave new.
"""

from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.embeddings import EmbeddingService
from clsplusplus.models import MemoryItem


class ReconsolidationGate:
    """Manages belief revision when new info conflicts with existing."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.embedding_service = EmbeddingService(settings)

    def similarity(self, a: MemoryItem, b: MemoryItem) -> float:
        """Cosine similarity between two items."""
        if not a.embedding or not b.embedding:
            return 0.0
        return EmbeddingService.cosine_similarity(a.embedding, b.embedding)

    def conflict_score(self, new: MemoryItem, old: MemoryItem) -> float:
        """Estimate conflict - high similarity + different content suggests contradiction."""
        sim = self.similarity(new, old)
        if sim < self.settings.similarity_threshold:
            return 0.0  # Not about same thing
        # Same topic - check if text is very different (heuristic)
        new_words = set(new.text.lower().split())
        old_words = set(old.text.lower().split())
        overlap = len(new_words & old_words) / max(len(new_words | old_words), 1)
        if overlap < 0.3 and sim > 0.7:
            return 0.5  # Same topic, different content - possible contradiction
        return 0.0

    def should_merge(self, new: MemoryItem, old: MemoryItem) -> bool:
        """Minor merge if similar and low conflict."""
        sim = self.similarity(new, old)
        conflict = self.conflict_score(new, old)
        return sim >= self.settings.similarity_threshold and conflict < self.settings.conflict_threshold

    def should_overwrite(self, new: MemoryItem, old: MemoryItem, evidence: list[str]) -> bool:
        """Overwrite only with evidence quorum."""
        conflict = self.conflict_score(new, old)
        if conflict < self.settings.conflict_threshold:
            return True  # No real conflict
        # Quorum: weighted sum of evidence confidence
        quorum = sum(0.2 for _ in evidence) / max(len(evidence), 1)
        if len(evidence) >= 3:
            quorum = 0.9  # Strong evidence
        return quorum >= self.settings.quorum_threshold

    def prepare_for_reconsolidation(
        self,
        new: MemoryItem,
        old: MemoryItem,
        evidence: list[str],
    ) -> tuple[MemoryItem, MemoryItem, bool]:
        """
        Returns (updated_new, archived_old, should_engrave_new).
        If should_engrave_new: archive old (versioned), engrave new.
        """
        if not self.should_overwrite(new, old, evidence):
            return new, old, False
        # Archive old - add to lineage
        new.lineage = old.lineage + [old.id]
        new.version = old.version + 1
        return new, old, True
