"""Plasticity Engine - biologically-grounded promotion scoring.

Score = α·S + β·log(1+U) + γ·A − λ·C + δ·Δ
"""

import math
from datetime import datetime, timedelta
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.embeddings import EmbeddingService
from clsplusplus.models import MemoryItem, StoreLevel


class PlasticityEngine:
    """Computes promotion scores and manages memory lifecycle."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.embedding_service = EmbeddingService(settings)

    def compute_score(self, item: MemoryItem) -> float:
        """Compute promotion score from plasticity signals."""
        s = item.salience  # 0-1
        u = item.usage_count
        a = item.authority  # 0-1
        c = item.conflict_score  # 0-1
        d = item.surprise  # 0-1

        score = (
            self.settings.alpha_salience * s
            + self.settings.beta_usage * math.log1p(u)
            + self.settings.gamma_authority * a
            - self.settings.lambda_conflict * c
            + self.settings.delta_surprise * d
        )
        item.promotion_score = max(0.0, score)
        return item.promotion_score

    def should_promote_to_l1(self, item: MemoryItem) -> bool:
        """L0 -> L1: score above threshold."""
        self.compute_score(item)
        return item.promotion_score >= self.settings.l1_promotion_threshold

    def should_promote_to_l2(self, item: MemoryItem) -> bool:
        """L1 -> L2: score, confidence, usage days."""
        self.compute_score(item)
        if item.promotion_score < self.settings.l2_promotion_threshold:
            return False
        if item.confidence < self.settings.l2_min_confidence:
            return False
        age_days = (datetime.utcnow() - item.timestamp.replace(tzinfo=None)).days
        return age_days >= self.settings.l2_min_usage_days

    def should_promote_to_l3(self, item: MemoryItem) -> bool:
        """L2 -> L3: same as L2 threshold + high confidence."""
        return (
            self.should_promote_to_l2(item)
            and item.confidence >= self.settings.l2_min_confidence
        )

    def apply_decay(self, item: MemoryItem, decay_factor: Optional[float] = None) -> MemoryItem:
        """Exponential decay on salience."""
        k = decay_factor if decay_factor is not None else self.settings.decay_constant_k
        item.salience = max(0.0, item.salience * (1 - k))
        item.confidence = max(0.0, item.confidence * (1 - k * 0.5))
        return item

    def should_prune(self, item: MemoryItem) -> bool:
        """Prune if salience below minimum."""
        return item.salience < self.settings.min_salience_prune
