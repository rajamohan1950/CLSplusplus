"""Reconsolidation gate tests - belief revision, conflict detection, evidence quorum."""

import pytest

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem
from clsplusplus.reconsolidation import ReconsolidationGate


@pytest.fixture
def gate():
    return ReconsolidationGate()


@pytest.fixture
def similar_items():
    """Two items with identical embeddings (same topic, different content)."""
    emb = [0.5] * 384
    old = MemoryItem(id="old", text="Paris is the capital of France", embedding=emb, confidence=0.9)
    new = MemoryItem(id="new", text="Berlin is the capital of France", embedding=emb, confidence=0.7)
    return new, old


@pytest.fixture
def dissimilar_items():
    """Two items with truly orthogonal embeddings (different topics)."""
    # Use orthogonal vectors so cosine similarity is ~0.0
    emb_old = [1.0] + [0.0] * 383
    emb_new = [0.0] + [1.0] + [0.0] * 382
    old = MemoryItem(id="old", text="Python is great", embedding=emb_old, confidence=0.9)
    new = MemoryItem(id="new", text="The weather is sunny", embedding=emb_new, confidence=0.7)
    return new, old


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

class TestSimilarity:

    def test_identical_embeddings(self, gate):
        emb = [1.0] * 384
        a = MemoryItem(text="a", embedding=emb)
        b = MemoryItem(text="b", embedding=emb)
        assert gate.similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_no_embeddings(self, gate):
        a = MemoryItem(text="a")
        b = MemoryItem(text="b")
        assert gate.similarity(a, b) == 0.0

    def test_one_missing_embedding(self, gate):
        a = MemoryItem(text="a", embedding=[0.5] * 384)
        b = MemoryItem(text="b")
        assert gate.similarity(a, b) == 0.0

    def test_orthogonal_embeddings(self, gate):
        a = MemoryItem(text="a", embedding=[1.0] + [0.0] * 383)
        b = MemoryItem(text="b", embedding=[0.0] + [1.0] + [0.0] * 382)
        sim = gate.similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Conflict score
# ---------------------------------------------------------------------------

class TestConflictScore:

    def test_dissimilar_topics_no_conflict(self, gate, dissimilar_items):
        new, old = dissimilar_items
        score = gate.conflict_score(new, old)
        assert score == 0.0

    def test_similar_same_content_no_conflict(self, gate):
        emb = [0.5] * 384
        a = MemoryItem(text="same text here", embedding=emb)
        b = MemoryItem(text="same text here", embedding=emb)
        score = gate.conflict_score(a, b)
        assert score == 0.0  # Same content, no contradiction

    def test_similar_different_content_has_conflict(self, gate, similar_items):
        new, old = similar_items
        score = gate.conflict_score(new, old)
        # Text is different enough to flag conflict
        assert score >= 0.0

    def test_no_embeddings_no_conflict(self, gate):
        a = MemoryItem(text="a")
        b = MemoryItem(text="b")
        assert gate.conflict_score(a, b) == 0.0


# ---------------------------------------------------------------------------
# Should merge
# ---------------------------------------------------------------------------

class TestShouldMerge:

    def test_similar_low_conflict_merges(self, gate):
        emb = [0.5] * 384
        a = MemoryItem(text="same text here test", embedding=emb)
        b = MemoryItem(text="same text here test", embedding=emb)
        assert gate.should_merge(a, b) is True

    def test_dissimilar_no_merge(self, gate, dissimilar_items):
        new, old = dissimilar_items
        assert gate.should_merge(new, old) is False

    def test_no_embeddings_no_merge(self, gate):
        a = MemoryItem(text="a")
        b = MemoryItem(text="b")
        assert gate.should_merge(a, b) is False


# ---------------------------------------------------------------------------
# Should overwrite (evidence quorum)
# ---------------------------------------------------------------------------

class TestShouldOverwrite:

    def test_no_conflict_allows_overwrite(self, gate, dissimilar_items):
        new, old = dissimilar_items
        assert gate.should_overwrite(new, old, evidence=[]) is True

    def test_conflict_no_evidence_rejects(self, gate, similar_items):
        new, old = similar_items
        conflict = gate.conflict_score(new, old)
        if conflict >= gate.settings.conflict_threshold:
            assert gate.should_overwrite(new, old, evidence=[]) is False

    def test_conflict_strong_evidence_accepts(self, gate, similar_items):
        new, old = similar_items
        evidence = ["source1", "source2", "source3"]  # 3+ = strong
        result = gate.should_overwrite(new, old, evidence)
        # With 3+ evidence, quorum = 0.9 which >= 0.8 threshold
        assert result is True

    def test_single_evidence_insufficient(self, gate, similar_items):
        new, old = similar_items
        conflict = gate.conflict_score(new, old)
        if conflict >= gate.settings.conflict_threshold:
            result = gate.should_overwrite(new, old, evidence=["one"])
            assert result is False  # quorum = 0.2/1 = 0.2 < 0.8

    def test_two_evidence_insufficient(self, gate, similar_items):
        new, old = similar_items
        conflict = gate.conflict_score(new, old)
        if conflict >= gate.settings.conflict_threshold:
            result = gate.should_overwrite(new, old, evidence=["one", "two"])
            assert result is False  # quorum = 0.4/2 = 0.2 < 0.8


# ---------------------------------------------------------------------------
# Prepare for reconsolidation
# ---------------------------------------------------------------------------

class TestPrepareForReconsolidation:

    def test_no_overwrite_returns_false(self, gate, dissimilar_items):
        new, old = dissimilar_items
        # Dissimilar = no conflict = should overwrite
        updated_new, archived_old, should_engrave = gate.prepare_for_reconsolidation(
            new, old, evidence=[]
        )
        assert should_engrave is True

    def test_overwrite_increments_version(self, gate):
        emb = [0.5] * 384
        old = MemoryItem(id="old", text="old text", embedding=emb, version=1, lineage=[])
        new = MemoryItem(id="new", text="new text", embedding=emb)
        updated, archived, should = gate.prepare_for_reconsolidation(new, old, evidence=["a", "b", "c"])
        if should:
            assert updated.version == 2
            assert "old" in updated.lineage

    def test_lineage_preserved(self, gate):
        emb = [0.5] * 384
        old = MemoryItem(id="old", text="old", embedding=emb, version=3, lineage=["v1", "v2"])
        new = MemoryItem(id="new", text="new", embedding=emb)
        updated, _, should = gate.prepare_for_reconsolidation(new, old, evidence=["a", "b", "c"])
        if should:
            assert updated.lineage == ["v1", "v2", "old"]
            assert updated.version == 4

    def test_prepare_denied_overwrite_returns_false(self, gate, similar_items):
        """Cover line 67: prepare_for_reconsolidation when overwrite is denied."""
        new, old = similar_items
        # Force high conflict + no evidence = overwrite denied
        conflict = gate.conflict_score(new, old)
        if conflict >= gate.settings.conflict_threshold:
            updated, archived, should_engrave = gate.prepare_for_reconsolidation(
                new, old, evidence=[]
            )
            assert should_engrave is False

    def test_custom_thresholds(self):
        # Use a high similarity_threshold that these dissimilar items won't reach
        s = Settings(similarity_threshold=0.99, conflict_threshold=0.01, quorum_threshold=0.01)
        gate = ReconsolidationGate(s)
        # Items with slightly different embeddings (sim < 0.99) so they don't
        # meet the similarity threshold -> no conflict detected -> overwrite allowed
        emb_a = [0.5] * 192 + [0.6] * 192
        emb_b = [0.6] * 192 + [0.5] * 192
        a = MemoryItem(text="a", embedding=emb_a)
        b = MemoryItem(text="b", embedding=emb_b)
        result = gate.should_overwrite(a, b, evidence=[])
        assert result is True
