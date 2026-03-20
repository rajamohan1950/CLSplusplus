"""CLS++ Plasticity Engine — memory scoring, promotion, decay, and pruning.

Computes a composite promotion score for each MemoryItem using five signals:

    score = α·salience + β·log(1+usage) + γ·authority − λ·conflict + δ·surprise

All coefficients are tunable via Settings.  score is floored at 0.0 and stored
directly on item.promotion_score for downstream use.

Promotion thresholds (L1/L2/L3) and pruning threshold (min_salience_prune)
are also read from Settings so they can be tuned without code changes.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem


class PlasticityEngine:
    """Score, promote, decay, and prune MemoryItems.

    Zero external dependencies — all arithmetic runs in pure Python.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()

    # =========================================================================
    # Scoring
    # =========================================================================

    def compute_score(self, item: MemoryItem) -> float:
        """Compute and return the composite promotion score for *item*.

        Side-effect: sets item.promotion_score.

        Formula
        -------
        score = α·salience + β·log(1+usage_count) + γ·authority
                − λ·conflict_score + δ·surprise

        All coefficients come from self.settings.  Result is clipped to ≥ 0.0.
        """
        s = self.settings
        raw = (
            s.alpha_salience  * item.salience
            + s.beta_usage    * math.log1p(item.usage_count)
            + s.gamma_authority * item.authority
            - s.lambda_conflict * item.conflict_score
            + s.delta_surprise  * item.surprise
        )
        score = max(0.0, raw)
        item.promotion_score = score
        return score

    # =========================================================================
    # Promotion predicates
    # =========================================================================

    def should_promote_to_l1(self, item: MemoryItem) -> bool:
        """Return True if *item* should be promoted to the L1 indexing store.

        Requires promotion_score ≥ l1_promotion_threshold.  Score is computed
        here if not already set (promotion_score == 0.0 and all signals absent).
        """
        score = item.promotion_score if item.promotion_score > 0.0 else self.compute_score(item)
        return score >= self.settings.l1_promotion_threshold

    def should_promote_to_l2(self, item: MemoryItem) -> bool:
        """Return True if *item* should be crystallised into the L2 schema graph.

        Criteria (all must hold):
        1. promotion_score ≥ l2_promotion_threshold
        2. confidence ≥ l2_min_confidence
        3. age ≥ l2_min_usage_days
        """
        score = item.promotion_score if item.promotion_score > 0.0 else self.compute_score(item)
        if score < self.settings.l2_promotion_threshold:
            return False
        if item.confidence < self.settings.l2_min_confidence:
            return False
        if item.timestamp is None:
            return False
        # Compute age in days — handle both aware and naive timestamps.
        try:
            now = datetime.now(timezone.utc)
            ts = item.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400.0
        except Exception:
            return False
        return age_days >= self.settings.l2_min_usage_days

    def should_promote_to_l3(self, item: MemoryItem) -> bool:
        """Return True if *item* should be archived in the L3 deep recess.

        Uses the same criteria as L2 (an item that qualifies for L2 also
        qualifies for long-term archival in L3).
        """
        return self.should_promote_to_l2(item)

    # =========================================================================
    # Decay
    # =========================================================================

    def apply_decay(self, item: MemoryItem, decay_factor: Optional[float] = None) -> MemoryItem:
        """Apply multiplicative decay to *item*.salience and *item*.confidence.

        Parameters
        ----------
        item
            The MemoryItem to decay in-place.
        decay_factor
            Fraction to remove from salience (0 = no decay, 1 = full decay).
            Defaults to settings.decay_constant_k.

        Salience formula   : new_salience   = salience   × (1 - decay_factor)
        Confidence formula : new_confidence = confidence × (1 - 0.5 × decay_factor)

        Both values are clipped to [0.0, original_value] so they never go
        negative and never accidentally increase.

        Returns *item* for method chaining.
        """
        if decay_factor is None:
            decay_factor = self.settings.decay_constant_k

        item.salience   = max(0.0, item.salience   * (1.0 - decay_factor))
        item.confidence = max(0.0, item.confidence * (1.0 - 0.5 * decay_factor))
        return item

    # =========================================================================
    # Pruning
    # =========================================================================

    def should_prune(self, item: MemoryItem) -> bool:
        """Return True if *item* is below the minimum salience and should be GC'd.

        Threshold: settings.min_salience_prune (default 0.2).
        At-threshold items are kept (strictly less-than).
        """
        return item.salience < self.settings.min_salience_prune
