"""
Gas → Liquid Phase Transition: Thermodynamic Memory Engine.

Memory is a phase of matter. This module implements the Gas → Liquid
phase transition exactly as specified by the free energy formulation:

    F(θ, Σ, ρ, τ) = E_prediction(θ) − Σ · S_model(θ) + λ · L_landauer(θ, τ)

Control parameters (all continuous):
    Σ (Surprise rate)      — KL divergence; effective temperature
    ρ (Memory density)     — active memories / capacity; effective pressure
    τ (Consolidation)      — integration speed / input speed; timescale

Phase boundary: Gas → Liquid occurs when τ > τ_c1.
    Below τ_c1, experiences evaporate (gas phase, s → 0).
    Above τ_c1, experiences persist as structured episodic memories (liquid phase, s → 1).

Time is measured in memory events (birth_order), NOT wall-clock seconds.
Time IS a function of memory dynamics.

References:
    - Landauer (1961): Irreversibility and heat generation in the computing process
    - Friston (2010): The free-energy principle: a unified brain theory?
    - Munoz et al. (2025): Long-range order from memory-induced time non-locality (arXiv)
    - LLM triphasic training dynamics (Emergent Mind, 2025)

Copyright (c) 2026 CLS++. All rights reserved.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional  # noqa: F401 — Optional used in type hints
from uuid import uuid4

# =============================================================================
# Stop Words — Excluded from index lookup to prevent noise coupling
# =============================================================================

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "about", "between", "through", "after", "before", "above",
    "below", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "also", "now", "then", "here", "there", "when",
    "where", "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "tell", "know", "think", "say", "get", "make", "go", "see", "come",
    "take", "give", "find", "let", "put", "keep",
})


# =============================================================================
# Data Structures — The Phases of Information
# =============================================================================


@dataclass(frozen=True)
class Fact:
    """
    Structured episodic memory unit — the LIQUID phase of information.

    A Fact is what remains after raw text (gas) passes through the attention
    gate (LLM extraction). It has internal structure: a subject-relation-value
    triple that enables contradiction detection and belief revision.

    The override flag captures semantic signals ("only", "exclusively",
    "actually", "never") that concentrate probability mass on a single value,
    driving KL divergence toward infinity on the (subject, relation) dimension.

    Immutable (frozen) because a fact, once condensed, does not change.
    Its consolidation_strength changes; its content does not.
    """

    subject: str       # Entity — normalized lowercase, stripped
    relation: str      # Property/relationship — normalized lowercase, stripped
    value: str         # The claim — normalized lowercase, stripped
    override: bool     # True when semantic override signal detected
    raw_text: str      # Original user message, unmodified


@dataclass
class PhaseMemoryItem:
    """
    A memory with continuous thermodynamic state variables.

    Every field maps 1:1 to the theory:

    consolidation_strength (s):
        The ORDER PARAMETER of the gas-liquid phase transition.
        Continuous in [0, 1]. Determines retrievability.
        s → 0: gas phase (volatile, forgotten)
        s → 1: liquid phase (persisted, retrievable)

    surprise_at_birth (Σ_birth):
        KL divergence D_KL(new || existing_model) at the moment this
        memory was created. Measures how much this fact contradicted
        the system's prior beliefs. High Σ = the system learned something.

    tau (τ):
        Consolidation timescale. The number of memory events over which
        this memory's strength halves via natural decay.
        High τ = strong consolidation = slow decay = liquid phase.
        Low τ = weak consolidation = fast decay = gas phase.
        Override statements get τ_override >> τ_default.

    birth_order (t_birth):
        Monotonic event counter at creation time. NOT a wall-clock timestamp.
        Time is a function of memory: Δt = current_event - birth_order
        measures how many memory events have elapsed, making temporal
        dynamics intrinsic to the memory system itself.

    rho_at_birth (ρ_birth):
        Memory density at creation time. The "pressure" under which
        this memory formed. High ρ = system was dense = more competition.

    free_energy (F):
        F(θ) = E_prediction - Σ · S_model + λ · L_landauer
        Per-item free energy. Low F = stable (liquid). High F = unstable (gas).
        The system minimizes total F. Phase transitions occur when
        F's global minimum jumps between s=0 and s=1.

    information_content_bits (H):
        Shannon entropy of the fact content in bits.
        H = -Σ p(c) · log₂(p(c)) over character distribution.
        Determines Landauer erasure cost: erasing H bits costs kT·ln(2)·H.

    landauer_cost (L):
        L = kT · ln(2) · H / τ
        Thermodynamic cost of maintaining this memory per event.
        Third term of F. High τ amortizes cost over longer lifetime.

    accumulated_surprise_damage (D):
        Irreversible damage from contradicting memories.
        When a newer fact contradicts this one on the same (subject, relation)
        dimension, surprise energy damages the old memory's consolidation.
        D = Σ_new · (1/τ_old) · amplifier
        Well-consolidated memories (high τ) resist surprise better.

    retrieval_count (R):
        Number of times this memory has been retrieved.
        Each retrieval reinforces consolidation: s *= (1 + β·ln(1+R)).
        Logarithmic saturation: the 100th recall matters less than the 1st.
    """

    id: str
    fact: Fact
    namespace: str

    # --- Thermodynamic state (all continuous) ---
    consolidation_strength: float       # s ∈ [0, 1] — order parameter
    surprise_at_birth: float            # Σ_birth — KL divergence at creation
    tau: float                          # τ — consolidation timescale (events)
    birth_order: int                    # t_birth — memory-relative time
    rho_at_birth: float                 # ρ at creation
    free_energy: float = 0.0           # F(θ) — current free energy

    # --- Dynamics ---
    retrieval_count: int = 0
    accumulated_surprise_damage: float = 0.0

    # --- Landauer ---
    information_content_bits: float = 0.0    # H in bits
    landauer_cost: float = 0.0               # kT·ln(2)·H / τ

    # --- Thermodynamic Semantic Field (TSF) ---
    # LLM-generated query forms: all ways a human might query this fact
    # Ordered by priority (canonical first, then common, then rare)
    query_field_subject: list[str] = field(default_factory=list)
    query_field_relation: list[str] = field(default_factory=list)
    query_field_value: list[str] = field(default_factory=list)
    _last_field_radius: int = -1  # Last R(s) for lazy index update

    def to_debug_dict(self, strength_floor: float = 0.05) -> dict[str, Any]:
        """Serialize thermodynamic state for the debug panel.

        Args:
            strength_floor: The engine's STRENGTH_FLOOR for phase classification.
                            Passed from the engine to avoid hardcoded magic numbers.
        """
        s = self.consolidation_strength
        return {
            "id": self.id,
            "text": self.fact.raw_text,
            "fact": {
                "subject": self.fact.subject,
                "relation": self.fact.relation,
                "value": self.fact.value,
                "override": self.fact.override,
            },
            "consolidation_strength": round(s, 4),
            "surprise_at_birth": round(self.surprise_at_birth, 4),
            "tau": round(self.tau, 2),
            "birth_order": self.birth_order,
            "rho_at_birth": round(self.rho_at_birth, 6),
            "free_energy": round(self.free_energy, 4),
            "information_content_bits": round(self.information_content_bits, 4),
            "landauer_cost": round(self.landauer_cost, 6),
            "retrieval_count": self.retrieval_count,
            "accumulated_surprise_damage": round(self.accumulated_surprise_damage, 6),
            "phase": "liquid" if s >= strength_floor else "gas",
        }




# =============================================================================
# The Extraction Prompt — Attention Gate (Gas → Liquid Boundary)
# =============================================================================

_EXTRACTION_SYSTEM = (
    "You are a precise fact extraction engine. You extract structured factual "
    "claims from user messages AND generate query fields. You ONLY return valid JSON, nothing else."
)

_EXTRACTION_PROMPT = """Analyze the following message and extract ALL factual claims as a JSON array.

Rules:
1. If the message states FACTS about entities (who/what something is, does, has, eats, likes, etc.), extract EACH fact separately.
2. A single message may contain MULTIPLE facts. "I went to Rome and loved the pasta" → two facts.
3. Normalize subject and relation to lowercase canonical base forms (e.g. "eating" → "eat", "visited" → "visit", "went to" → "go to").
4. Set "override" to true ONLY if the message contains words like: "only", "exclusively", "always", "never", "actually", "not anymore", "switched to", "changed to", "no longer", "just" (meaning exclusively) — any signal that this REPLACES a previous belief entirely.
5. If the message is NOT a factual statement (it's a question, greeting, opinion, or command), return exactly: [{{"extract": false}}]
6. For each fact, generate a "query_field" — all the ways a human might search for this fact:
   - "subject_aliases": other names/spellings for the subject
   - "relation_forms": the canonical relation PLUS all morphological variants (eat/eats/eating/ate/eaten) PLUS semantic neighbors (food/diet/meal/snack/cuisine/prefer). Include 8-15 forms.
   - "value_aliases": other forms of the value (banana/bananas, rome/Roma)

Return ONLY a valid JSON array:

Format A (factual claims with query fields):
[{{"subject": "entity_name", "relation": "relationship", "value": "the_claim", "override": false, "query_field": {{"subject_aliases": ["name1"], "relation_forms": ["form1", "form2", "form3"], "value_aliases": ["val1"]}}}}]

Format B (not a fact):
[{{"extract": false}}]

Message: "{message}"
"""


# =============================================================================
# PhaseMemoryEngine — Thermodynamic Memory System
# =============================================================================


class PhaseMemoryEngine:
    """
    Thermodynamic memory engine implementing the Gas → Liquid phase transition.

    Minimizes the free energy functional:

        F(θ, Σ, ρ, τ) = E_prediction(θ) − Σ · S_model(θ) + λ · L_landauer(θ, τ)

    where:
        E_prediction = 1 − s(t)                           [prediction error]
        S_model      = H(item) · ρ                        [model entropy]
        L_landauer   = kT · ln(2) · H(item) / τ           [Landauer cost]

    Phase transition at τ = τ_c1:
        τ < τ_c1  →  s=0 minimum of F is global  →  gas (memory evaporates)
        τ > τ_c1  →  s=1 minimum of F is global  →  liquid (memory persists)

    The attention gate (LLM fact extraction) IS the phase transition mechanism.
    Raw text (gas) condenses into structured Facts (liquid) when the extraction
    succeeds and τ exceeds τ_c1.
    """

    def __init__(
        self,
        kT: float = 1.0,
        lambda_budget: float = 0.5,
        tau_c1: float = 10.0,
        tau_default: float = 50.0,
        tau_override: float = 200.0,
        strength_floor: float = 0.05,
        capacity: int = 1000,
        beta_retrieval: float = 0.15,
    ) -> None:
        """
        Initialize the thermodynamic memory engine.

        Args:
            kT: Boltzmann constant × temperature analog. Sets the energy scale
                for Landauer cost. Higher kT = more expensive to maintain memories.
            lambda_budget: Energy budget constraint (λ). Scales the Landauer term
                in the free energy equation. Higher λ = stricter energy budget.
            tau_c1: Critical consolidation timescale. The phase boundary.
                τ > τ_c1 → liquid (persistent). τ < τ_c1 → gas (volatile).
            tau_default: Default τ for normal factual statements.
                Must be > τ_c1 for facts to persist (enter liquid phase).
            tau_override: τ for statements with semantic override signals.
                Much larger than τ_default → stronger consolidation, slower decay.
            strength_floor: Consolidation strength below which a memory is
                considered gas-phase and excluded from retrieval.
            capacity: Maximum items per namespace. Denominator for ρ.
            beta_retrieval: Coefficient for retrieval reinforcement.
                s *= (1 + β·ln(1+R)) where R = retrieval count.
        """
        # Physical constants (computational analogs)
        self.kT: float = kT
        self.LAMBDA: float = lambda_budget
        self.TAU_C1: float = tau_c1
        self.TAU_DEFAULT: float = tau_default
        self.TAU_OVERRIDE: float = tau_override
        self.STRENGTH_FLOOR: float = strength_floor
        self.CAPACITY: int = capacity
        self.BETA_RETRIEVAL: float = beta_retrieval

        # State
        self._items: dict[str, list[PhaseMemoryItem]] = {}  # namespace → items
        self._event_counter: int = 0  # Monotonic. Time IS this counter.
        self._dirty_namespaces: set[str] = set()  # Namespaces needing F recomputation

        # Triple Index — Thermodynamic Semantic Field (TSF)
        # Keys: query form strings → lists of PhaseMemoryItems
        # Built at ingest from LLM-generated query fields
        # Entries phase-transition in/out based on R(s) = floor(N × s^(1/3))
        self._subject_index: dict[str, list[PhaseMemoryItem]] = {}
        self._relation_index: dict[str, list[PhaseMemoryItem]] = {}
        self._value_index: dict[str, list[PhaseMemoryItem]] = {}

    # =========================================================================
    # Core Physics — Information Content (Shannon Entropy)
    # =========================================================================

    @staticmethod
    def _information_content(fact: Fact) -> float:
        """
        Compute Shannon entropy H(fact) in bits.

        H = −Σ p(c) · log₂(p(c))

        where p(c) is the empirical probability of character c in the
        normalized fact string "subject relation value".

        This measures the irreducible information content of the fact —
        the minimum number of bits needed to encode it. By Landauer's
        principle, erasing this information costs at least kT·ln(2)·H
        joules of energy (or its computational analog).

        Character-level entropy is used (not word-level) because:
        1. It captures structural regularity in the text itself
        2. It's language-agnostic
        3. It maps directly to Landauer's bit-level formulation
        4. Short texts have enough characters for stable estimates,
           unlike word-level where vocabulary is too small

        Returns:
            H in bits. Minimum 0.0 (constant string), typically 2.0–4.5
            for natural language facts.
        """
        text = f"{fact.subject} {fact.relation} {fact.value}".lower()
        if not text:
            return 0.0

        n = len(text)
        counts = Counter(text)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    # =========================================================================
    # Thermodynamic Semantic Field — Triple Index Architecture
    # =========================================================================

    def _index_item(self, item: PhaseMemoryItem) -> None:
        """
        Add a memory item to the triple index under its active query field keys.

        The number of indexed keys depends on the field radius R(s),
        which is a function of consolidation strength:

            R(s) = floor(N_forms × s^(1/3))

        where 1/3 is the mean-field critical exponent ν from 3D thermodynamics.
        Liquid memories (s→1) have broad fields; gas memories have none.
        """
        s = item.consolidation_strength
        if s < self.STRENGTH_FLOOR:
            # Gas phase: zero correlation length, invisible
            self._deindex_item(item)
            item._last_field_radius = 0
            return

        # Compute field radius R(s) = floor(N × s^(1/3))
        cube_root_s = s ** (1.0 / 3.0)

        for keys, index in [
            (item.query_field_subject, self._subject_index),
            (item.query_field_relation, self._relation_index),
            (item.query_field_value, self._value_index),
        ]:
            n_keys = len(keys)
            radius = max(1, int(n_keys * cube_root_s))  # At least canonical form

            for key in keys[:radius]:
                if key not in index:
                    index[key] = []
                if item not in index[key]:
                    index[key].append(item)

            # De-index keys beyond radius
            for key in keys[radius:]:
                if key in index and item in index[key]:
                    index[key].remove(item)
                    if not index[key]:
                        del index[key]

        item._last_field_radius = int(
            len(item.query_field_relation) * cube_root_s
        )

    def _deindex_item(self, item: PhaseMemoryItem) -> None:
        """Remove a memory item from all indexes."""
        for keys, index in [
            (item.query_field_subject, self._subject_index),
            (item.query_field_relation, self._relation_index),
            (item.query_field_value, self._value_index),
        ]:
            for key in keys:
                if key in index and item in index[key]:
                    index[key].remove(item)
                    if not index[key]:
                        del index[key]

    def _update_field_radius(self, item: PhaseMemoryItem) -> None:
        """
        Lazy field radius update. Only re-indexes if R(s) has changed.

        Called during search for candidate items. O(1) if radius unchanged.
        """
        s = item.consolidation_strength
        if s < self.STRENGTH_FLOOR:
            if item._last_field_radius != 0:
                self._deindex_item(item)
                item._last_field_radius = 0
            return

        cube_root_s = s ** (1.0 / 3.0)
        new_radius = max(1, int(len(item.query_field_relation) * cube_root_s))

        if new_radius != item._last_field_radius:
            self._index_item(item)

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        """
        Tokenize a query into lookup tokens, filtering stop words.

        Returns unigrams + bigrams for multi-word query form matching.
        """
        words = [w for w in query.lower().split() if w not in _STOP_WORDS and len(w) > 1]

        # Generate bigrams for multi-word query forms (e.g. "favorite food")
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

        return words + bigrams

    # =========================================================================
    # Core Physics — Surprise (KL Divergence)
    # =========================================================================

    def _compute_surprise(
        self,
        new_fact: Fact,
        existing_items: list[PhaseMemoryItem],
    ) -> tuple[float, list[PhaseMemoryItem]]:
        """
        Compute surprise Σ as KL divergence D_KL(new || existing_model).

        Information-theoretic surprise measures how much the new observation
        deviates from the system's existing beliefs on the same
        (subject, relation) dimension.

        Mathematical formulation:
            Let P_new be the distribution concentrated on new_fact.value.
            Let P_old be the distribution over existing values for the same
            (subject, relation).

            D_KL(P_new || P_old) = Σ P_new(v) · ln(P_new(v) / P_old(v))

        For discrete facts (single values, not distributions):
            - If P_old has no entry for (subject, relation): Σ = 0
              (new information, no contradiction — nothing to be surprised about)
            - If P_old.value == P_new.value: Σ = 0
              (confirmation, no surprise)
            - If values differ (soft contradiction):
              Σ = 1 − J(new_value, old_value)
              where J is the Jaccard similarity of character bigrams.
              This approximates the overlap between value distributions.
            - If values differ AND override=True (hard override):
              Σ = −ln(ε) where ε = 1e-6, capped at practical maximum.
              The override signal concentrates ALL probability mass on the
              new value, making P_old(new_value) → 0, so D_KL → ∞.
              We cap at −ln(1e-6) ≈ 13.8 for numerical stability.

        Args:
            new_fact: The newly extracted Fact.
            existing_items: All items in the namespace.

        Returns:
            (surprise_value, list_of_contradicted_items)
            surprise_value: Σ ∈ [0, ~13.8]
            contradicted_items: Items that share (subject, relation) with
                different values — these will receive surprise damage.
        """
        contradicted: list[PhaseMemoryItem] = []
        max_surprise = 0.0

        for item in existing_items:
            # Match on (subject, relation) dimension
            if (
                item.fact.subject == new_fact.subject
                and item.fact.relation == new_fact.relation
                and item.consolidation_strength >= self.STRENGTH_FLOOR
            ):
                # Same entity, same relation — check value
                if item.fact.value == new_fact.value:
                    # Confirmation — zero surprise, reinforcement (handled elsewhere)
                    continue

                # Contradiction detected — compute KL divergence
                contradicted.append(item)

                if new_fact.override:
                    # Hard override: D_KL → ∞, capped
                    # P_new concentrates on new_value, P_old has zero mass there
                    # D_KL = -ln(ε) where ε → 0
                    sigma = -math.log(1e-6)  # ≈ 13.8 nats
                else:
                    # Soft contradiction: use Jaccard distance on character bigrams
                    # as proxy for distributional divergence
                    sigma = self._bigram_divergence(new_fact.value, item.fact.value)

                max_surprise = max(max_surprise, sigma)

        return max_surprise, contradicted

    @staticmethod
    def _bigram_divergence(new_value: str, old_value: str) -> float:
        """
        Approximate KL divergence via 1 − Jaccard similarity of character bigrams.

        Bigrams capture local structure. Jaccard similarity J ∈ [0, 1] measures
        set overlap. D_approx = 1 − J gives a [0, 1] divergence measure.

        For identical strings: J = 1, D = 0 (no surprise).
        For completely different strings: J = 0, D = 1 (max soft surprise).

        This is a lower bound on true KL divergence for the soft-contradiction
        case (no override signal). The override case uses the exact -ln(ε)
        formulation instead.
        """
        def _bigrams(s: str) -> set[str]:
            s = s.lower().strip()
            return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

        bg_new = _bigrams(new_value)
        bg_old = _bigrams(old_value)

        intersection = len(bg_new & bg_old)
        union = len(bg_new | bg_old)

        if union == 0:
            return 0.0

        jaccard = intersection / union
        return 1.0 - jaccard

    # =========================================================================
    # Core Physics — Surprise Damage (Irreversible Reconsolidation)
    # =========================================================================

    def _apply_surprise_damage(
        self,
        surprise: float,
        contradicted: list[PhaseMemoryItem],
        new_fact: Fact,
    ) -> None:
        """
        Apply irreversible surprise damage to contradicted memories.

        When a new memory contradicts an existing one, the old memory's
        consolidation is DAMAGED. This models Misanin-Miller-Lewis
        reconsolidation: retrieving (or being reminded of) a memory
        makes it labile — susceptible to modification.

        Damage formula:
            D = σ(Σ_norm) · (τ_new / τ_old) · amplifier

        where:
            Σ_norm = Σ / Σ_max, normalized surprise ∈ [0, 1]
            σ(x) = 1 / (1 + exp(-10·(x - 0.5))), sigmoid sharpening
                Converts normalized surprise into a damage probability.
                Low surprise → near-zero damage. High surprise → near-total damage.
                The sigmoid models the nonlinear nature of belief revision:
                small surprises are absorbed, large surprises are catastrophic.
            τ_new / τ_old = relative consolidation strength.
                A strongly consolidated new memory (high τ_new) does more
                damage to a weakly consolidated old memory (low τ_old).
            amplifier = 1.5 if override, 1.0 otherwise.

        The damage is ACCUMULATED (irreversible). A memory that has been
        contradicted multiple times accumulates damage from each event.
        This models the thermodynamic arrow: you can't un-surprise a system.

        Args:
            surprise: Σ from the new fact (in nats).
            contradicted: Items to damage.
            new_fact: The new fact (used for override amplifier and τ).
        """
        # Maximum possible surprise: -ln(1e-6) ≈ 13.8 nats
        SIGMA_MAX = -math.log(1e-6)
        sigma_norm = min(surprise / SIGMA_MAX, 1.0)

        # Sigmoid sharpening: small surprises → negligible damage,
        # large surprises → near-total damage
        # σ(x) = 1 / (1 + exp(-k·(x - 0.5))) with k=10 for sharp transition
        sigmoid_damage = 1.0 / (1.0 + math.exp(-10.0 * (sigma_norm - 0.5)))

        amplifier = 1.5 if new_fact.override else 1.0
        tau_new = self.TAU_OVERRIDE if new_fact.override else self.TAU_DEFAULT

        for item in contradicted:
            # Relative consolidation: how much stronger is the new memory?
            tau_ratio = tau_new / max(item.tau, 1e-6)
            # Scale tau_ratio to be meaningful: cap at 2x effect
            tau_factor = min(tau_ratio, 4.0) / 4.0 + 0.5  # ∈ [0.5, 1.5]

            damage = sigmoid_damage * tau_factor * amplifier
            item.accumulated_surprise_damage = min(
                item.accumulated_surprise_damage + damage,
                2.0,  # Cap: beyond 2.0, item is irrecoverably dead anyway (s clamped to 0)
            )

    # =========================================================================
    # Core Physics — Consolidation Strength s(t)
    # =========================================================================

    def _compute_consolidation(self, item: PhaseMemoryItem, delta_t: int) -> float:
        """
        Compute the order parameter s(t) — consolidation strength.

        s(t) = s₀ · exp(−Δt / τ) · (1 + β · ln(1 + R)) − D

        where:
            s₀ = 1.0     (initial strength at birth)
            Δt           = current_event_counter − birth_order
                           (memory-relative time, NOT wall-clock)
            τ            = consolidation timescale (events)
            β            = retrieval reinforcement coefficient
            R            = retrieval_count
            D            = accumulated_surprise_damage

        Term-by-term physics:

        exp(−Δt / τ):
            Natural exponential decay. Every memory fades without reinforcement.
            τ sets the timescale: high τ = slow decay = liquid phase.
            At the phase boundary τ = τ_c1, the decay rate exactly balances
            the minimum persistence threshold (strength_floor).

        (1 + β · ln(1 + R)):
            Retrieval reinforcement. Each time a memory is recalled, it
            strengthens. Logarithmic saturation models the psychological
            finding that early retrievals have more impact than later ones
            (spacing effect, Ebbinghaus).

        −D:
            Irreversible surprise damage. Subtracted directly from strength.
            This is the thermodynamic arrow — entropy increases.
            A memory that has been contradicted cannot fully recover.

        Phase transition:
            When τ > τ_c1: exp(−Δt/τ) decays slowly → s stays above floor → LIQUID
            When τ < τ_c1: exp(−Δt/τ) decays fast → s drops below floor → GAS
            The transition is continuous but sharp near τ_c1.

        Returns:
            s ∈ [0, 1]. Clamped.
        """
        natural_decay = math.exp(-delta_t / max(item.tau, 1e-6))
        retrieval_boost = 1.0 + self.BETA_RETRIEVAL * math.log1p(item.retrieval_count)
        s = 1.0 * natural_decay * retrieval_boost - item.accumulated_surprise_damage
        return max(0.0, min(1.0, s))

    # =========================================================================
    # Core Physics — Landauer Cost
    # =========================================================================

    def _compute_landauer_cost(self, item: PhaseMemoryItem) -> float:
        """
        Compute the Landauer erasure cost per memory event.

        L = kT · ln(2) · H(item) / τ

        Landauer's principle (1961): The minimum thermodynamic cost of
        erasing one bit of information is kT · ln(2) joules, where
        k is Boltzmann's constant and T is temperature.

        In our computational analog:
            kT = energy scale parameter (sets how "expensive" memory is)
            H(item) = Shannon entropy of the fact in bits
            τ = consolidation timescale

        The division by τ converts total erasure cost into per-event
        maintenance cost. A strongly consolidated memory (high τ) pays
        its Landauer cost over many events, making the per-event cost low.
        A weakly consolidated memory (low τ) pays the same total cost
        over fewer events, making per-event cost high.

        This creates the thermodynamic incentive structure:
            - Strongly consolidated (high τ): low per-event cost → favored
            - Weakly consolidated (low τ): high per-event cost → disfavored
            - The system naturally prefers consolidated memories.

        Returns:
            L ≥ 0. The Landauer cost contribution to free energy.
        """
        return (self.kT * math.log(2) * item.information_content_bits) / max(item.tau, 1e-6)

    # =========================================================================
    # Core Physics — Free Energy F(θ, Σ, ρ, τ)
    # =========================================================================

    def _compute_free_energy(self, item: PhaseMemoryItem, global_rho: float) -> float:
        """
        Compute per-item free energy.

        F(θ, Σ, ρ, τ) = E_prediction(θ) − Σ · S_model(θ) + λ · L_landauer(θ, τ)

        Term 1: E_prediction = 1 − s(t)
            Prediction error. A strong memory (s → 1) means the system
            "knows" this fact → low prediction error. A weak memory (s → 0)
            means high uncertainty → high prediction error.
            This is Friston's variational free energy: the system minimizes
            prediction error by maintaining accurate world models.

        Term 2: −Σ · S_model = −Σ_birth · H(item) · ρ
            Negative sign: surprise REDUCES free energy for informative memories.
            This is the information-theoretic incentive: surprising facts that
            carry high information (H) in a dense memory system (ρ) are
            VALUABLE — they reduce F and are favored for retention.
            This prevents the system from only keeping unsurprising trivia.

        Term 3: +λ · L_landauer = λ · kT · ln(2) · H / τ
            Positive sign: maintenance has a thermodynamic COST.
            High-information memories cost more to maintain (Landauer).
            But high τ amortizes the cost.
            λ scales the energy budget constraint.

        Phase transition mechanics:
            At s = 0 (gas):  F = 1 − Σ·H·ρ + λ·L  (high E_pred, but no decay)
            At s = 1 (liquid): F = 0 − Σ·H·ρ + λ·L  (low E_pred, decay active)

            When τ > τ_c1, the s=1 state has lower F → liquid is stable.
            When τ < τ_c1, the s=0 state has lower F → gas is stable.
            The transition occurs when these minima exchange global stability.

        Args:
            item: The memory to evaluate.
            global_rho: Current memory density ρ = |active_items| / capacity.

        Returns:
            F(θ). Lower = more stable = better for the system.
        """
        delta_t = self._event_counter - item.birth_order

        # Order parameter
        s = self._compute_consolidation(item, delta_t)
        item.consolidation_strength = s

        # Term 1: Prediction error
        E_pred = 1.0 - s

        # Term 2: Information value (surprise × entropy × density)
        S_model = item.information_content_bits * max(global_rho, 1e-9)

        # Term 3: Landauer maintenance cost
        L_land = self._compute_landauer_cost(item)
        item.landauer_cost = L_land

        # Free energy
        F = E_pred - item.surprise_at_birth * S_model + self.LAMBDA * L_land
        item.free_energy = F

        return F

    # =========================================================================
    # Memory Density ρ
    # =========================================================================

    def _memory_density(self, namespace: str) -> float:
        """
        Compute memory density ρ = |active items| / capacity.

        Active items are those with consolidation_strength ≥ strength_floor
        (i.e., in the liquid phase). Gas-phase items don't contribute to density.

        ρ is the effective "pressure" in the phase diagram:
            Low ρ: sparse memory, low competition between items
            High ρ: dense memory, competition forces consolidation or evaporation
            At critical ρ_c: liquid → solid transition (future implementation)

        Returns:
            ρ ∈ [0, 1]. 0 = empty, 1 = at capacity.
        """
        items = self._items.get(namespace, [])
        active = sum(1 for item in items if item.consolidation_strength >= self.STRENGTH_FLOOR)
        return active / max(self.CAPACITY, 1)

    # =========================================================================
    # Attention Gate — Gas → Liquid Phase Transition
    # =========================================================================

    async def _extract_facts(
        self,
        message: str,
        llm_caller: Callable,
    ) -> list[tuple[Fact, list[str], list[str], list[str]]]:
        """
        The attention gate. The Gas → Liquid phase boundary.

        Raw text (gas phase) is passed through LLM extraction. If the LLM
        identifies factual claims, they condense into structured Facts
        (liquid phase) with LLM-generated query fields (semantic fields).

        Args:
            message: Raw user message (gas phase).
            llm_caller: Async function(system, user_msg) → str.

        Returns:
            List of (Fact, subject_aliases, relation_forms, value_aliases).
            Empty list if the message stays in gas phase.
        """
        prompt = _EXTRACTION_PROMPT.format(message=message)

        try:
            raw_response = await llm_caller(_EXTRACTION_SYSTEM, prompt)
            json_str = raw_response.strip()

            # Try to parse as JSON array first (multi-fact format)
            array_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if array_match:
                data_list = json.loads(array_match.group(0))
                if isinstance(data_list, list):
                    return self._parse_fact_list(data_list, message)

            # Fallback: parse as single JSON object (backward compatibility)
            all_matches = list(re.finditer(r'\{[^{}]*\}', json_str, re.DOTALL))
            if not all_matches:
                return []
            json_str = all_matches[-1].group(0)
            data = json.loads(json_str)
            fact, sa, rf, va = self._parse_single_fact(data, message)
            return [(fact, sa, rf, va)] if fact else []

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return []
        except Exception:
            return []

    def _parse_single_fact(
        self, data: dict, raw_text: str,
    ) -> tuple[Optional[Fact], list[str], list[str], list[str]]:
        """Parse a single fact dict into a Fact object with query field.

        Returns:
            (fact_or_none, subject_aliases, relation_forms, value_aliases)
        """
        empty_field: tuple[None, list[str], list[str], list[str]] = (None, [], [], [])

        if data.get("extract") is False:
            return empty_field

        subject = str(data.get("subject", "")).lower().strip()
        relation = str(data.get("relation", "")).lower().strip()
        value = str(data.get("value", "")).lower().strip()
        override = bool(data.get("override", False))

        if not subject or not relation or not value:
            return empty_field

        # Parse query field (LLM-generated semantic field)
        qf = data.get("query_field", {})
        if not isinstance(qf, dict):
            qf = {}

        subject_aliases = [s.lower().strip() for s in qf.get("subject_aliases", []) if s]
        relation_forms = [r.lower().strip() for r in qf.get("relation_forms", []) if r]
        value_aliases = [v.lower().strip() for v in qf.get("value_aliases", []) if v]

        # Ensure canonical forms are always included
        if subject not in subject_aliases:
            subject_aliases.insert(0, subject)
        if relation not in relation_forms:
            relation_forms.insert(0, relation)
        if value not in value_aliases:
            value_aliases.insert(0, value)

        fact = Fact(
            subject=subject,
            relation=relation,
            value=value,
            override=override,
            raw_text=raw_text,
        )
        return fact, subject_aliases, relation_forms, value_aliases

    def _parse_fact_list(
        self, data_list: list, raw_text: str,
    ) -> list[tuple[Fact, list[str], list[str], list[str]]]:
        """Parse a list of fact dicts into Fact objects with query fields."""
        results = []
        for data in data_list:
            if not isinstance(data, dict):
                continue
            fact, sa, rf, va = self._parse_single_fact(data, raw_text)
            if fact:
                results.append((fact, sa, rf, va))
        return results

    async def ingest(
        self,
        message: str,
        namespace: str,
        llm_caller: Callable,
    ) -> list[PhaseMemoryItem]:
        """
        Full Gas → Liquid ingestion pipeline with multi-fact extraction.

        1. EXTRACTION (attention gate): Raw text → [Fact, ...] (or [])
        2. For each extracted fact:
           a. CONFIRMATION: Deduplicate if exact match exists
           b. SURPRISE: Compute D_KL(new || existing) on (subject, relation)
           c. DAMAGE: Apply surprise to contradicted memories
           d. STORE: Create PhaseMemoryItem with computed thermodynamic state
        3. FREE ENERGY: Update F for all items in namespace

        Args:
            message: Raw user message.
            namespace: Memory namespace (global shared).
            llm_caller: Async function(system, user_msg) → str.

        Returns:
            List of PhaseMemoryItems condensed to liquid. Empty list if all stayed gas.
        """
        # --- Advance memory-relative time FIRST ---
        self._event_counter += 1

        # --- Step 1: Attention gate (multi-fact with query fields) ---
        extracted = await self._extract_facts(message, llm_caller)
        if not extracted:
            return []  # Stayed gas. Evaporated.

        result_items: list[PhaseMemoryItem] = []

        for fact, subject_aliases, relation_forms, value_aliases in extracted:
            existing = self._items.get(namespace, [])

            # --- Step 1b: Check for CONFIRMATION (exact duplicate) ---
            confirmed = False
            for item in existing:
                if (
                    item.fact.subject == fact.subject
                    and item.fact.relation == fact.relation
                    and item.fact.value == fact.value
                    and item.consolidation_strength >= self.STRENGTH_FLOOR
                ):
                    item.retrieval_count += 1
                    result_items.append(item)
                    confirmed = True
                    break

            if confirmed:
                continue

            # --- Step 2: Compute surprise ---
            surprise, contradicted = self._compute_surprise(fact, existing)

            # --- Step 3: Apply surprise damage ---
            if contradicted:
                self._apply_surprise_damage(surprise, contradicted, fact)

            # --- Step 4: Compute thermodynamic state ---
            rho = self._memory_density(namespace)
            tau = self.TAU_OVERRIDE if fact.override else self.TAU_DEFAULT
            H = self._information_content(fact)
            L = (self.kT * math.log(2) * H) / max(tau, 1e-6)

            item = PhaseMemoryItem(
                id=str(uuid4()),
                fact=fact,
                namespace=namespace,
                consolidation_strength=1.0,
                surprise_at_birth=surprise,
                tau=tau,
                birth_order=self._event_counter,
                rho_at_birth=rho,
                free_energy=0.0,
                retrieval_count=0,
                accumulated_surprise_damage=0.0,
                information_content_bits=H,
                landauer_cost=L,
                query_field_subject=subject_aliases,
                query_field_relation=relation_forms,
                query_field_value=value_aliases,
            )

            # --- Step 5: Store + Index ---
            self._items.setdefault(namespace, []).append(item)
            self._index_item(item)
            self._dirty_namespaces.add(namespace)
            result_items.append(item)

        # --- Step 6: Recompute free energy for ALL items ---
        self._recompute_all_free_energies(namespace)

        return result_items

    # =========================================================================
    # Free Energy Recomputation
    # =========================================================================

    def _recompute_all_free_energies(self, namespace: str) -> None:
        """
        Recompute F(θ) for every item in the namespace, then garbage-collect
        gas-phase items that have decayed beyond recovery.

        Must be called after any state change (new item, retrieval, damage)
        because F depends on global ρ which changes when items enter/exit
        the liquid phase.

        Garbage collection: items with s=0 AND accumulated_surprise_damage > 1.0
        are truly dead — they can never return to liquid phase. Removing them
        prevents unbounded memory growth from gas-phase corpses.
        """
        items = self._items.get(namespace, [])
        rho = self._memory_density(namespace)
        for item in items:
            self._compute_free_energy(item, rho)

        # GC: remove items that are irrecoverably dead
        # Condition: s=0 (clamped) AND enough damage that even retrieval can't save them
        alive = []
        for item in items:
            if item.consolidation_strength > 0.0 or item.accumulated_surprise_damage < 1.0:
                alive.append(item)
            else:
                # Fully dead — deindex before removal
                self._deindex_item(item)
        self._items[namespace] = alive

    # =========================================================================
    # Retrieval — Free Energy Ranked Search
    # =========================================================================

    def search(
        self,
        query: str,
        namespace: str,
        limit: int = 10,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Thermodynamic Semantic Field (TSF) retrieval.

        Ranking equation:
            rank(q, i) = (n_matched_slots - 1) - F_i / kT

        where n_matched_slots ∈ {1, 2, 3} is how many (S, R, V) index
        slots the query matched, and F_i is the full free energy.

        Search path (sub-μs, zero ML):
            query → tokenize → filter stopwords → generate bigrams
            → hash lookup against subject/relation/value indexes
            → union candidates → score each → sort → return top-k

        Fallback: if zero candidates from index, return top-k liquid
        items by pure Boltzmann rank -F/kT (maximum entropy response).

        Args:
            query: The search query.
            namespace: Memory namespace.
            limit: Maximum items to return.

        Returns:
            List of (score, item) tuples, sorted by score descending.
        """
        # Ensure free energies are current
        self._recompute_all_free_energies(namespace)

        # Tokenize query: unigrams + bigrams, stopwords removed
        tokens = self._tokenize_query(query)

        # Index lookup: find candidates and track which slots matched
        # item_id → (item, set of matched slot names)
        candidates: dict[str, tuple[PhaseMemoryItem, set[str]]] = {}

        for token in tokens:
            # Subject index
            if token in self._subject_index:
                for item in self._subject_index[token]:
                    if item.namespace != namespace:
                        continue
                    if item.id not in candidates:
                        candidates[item.id] = (item, set())
                    candidates[item.id][1].add("subject")

            # Relation index
            if token in self._relation_index:
                for item in self._relation_index[token]:
                    if item.namespace != namespace:
                        continue
                    if item.id not in candidates:
                        candidates[item.id] = (item, set())
                    candidates[item.id][1].add("relation")

            # Value index
            if token in self._value_index:
                for item in self._value_index[token]:
                    if item.namespace != namespace:
                        continue
                    if item.id not in candidates:
                        candidates[item.id] = (item, set())
                    candidates[item.id][1].add("value")

        scored: list[tuple[float, PhaseMemoryItem]] = []

        if candidates:
            for item, matched_slots in candidates.values():
                # Gas-phase items are invisible
                if item.consolidation_strength < self.STRENGTH_FLOOR:
                    continue

                # Lazy field radius update
                self._update_field_radius(item)

                # TSF ranking: (n_slots - 1) - F/kT
                n_slots = len(matched_slots)
                rank = (n_slots - 1) - item.free_energy / max(self.kT, 1e-9)
                scored.append((rank, item))
        else:
            # Fallback: no index hits → return all liquid items by -F/kT
            items = self._items.get(namespace, [])
            for item in items:
                if item.consolidation_strength < self.STRENGTH_FLOOR:
                    continue
                rank = -item.free_energy / max(self.kT, 1e-9)
                scored.append((rank, item))

        # Sort by rank descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Reinforcement: increment retrieval count for returned items
        for _, item in scored[:limit]:
            item.retrieval_count += 1

        return scored[:limit]

    # =========================================================================
    # Augmented Context Builder
    # =========================================================================

    def build_augmented_context(
        self,
        query: str,
        namespace: str,
        limit: int = 5,
    ) -> tuple[str, list[dict]]:
        """
        Build the memory context string for LLM prompt augmentation.

        Uses free-energy-ranked retrieval to select the most stable,
        relevant memories. Returns a formatted string and debug data.

        The context string includes consolidation strength annotations
        so the LLM can reason about memory confidence:

            "Memory (strongest recall first):
            - [s=0.98] Raj eats banana
            - [s=0.12] Raj eats apple"    ← if above floor; likely excluded

        Memories below strength_floor are already excluded by search().
        The physics handles conflict resolution — no "NEWEST FIRST" hack needed.

        Args:
            query: User's message/question.
            namespace: Memory namespace.
            limit: Max memories to include.

        Returns:
            (context_string, debug_items_list)
        """
        results = self.search(query, namespace, limit=limit)

        if not results:
            return "No prior context yet.", []

        lines: list[str] = []
        debug_items: list[dict] = []

        for score, item in results:
            s = item.consolidation_strength
            lines.append(f"- [strength={s:.2f}] {item.fact.raw_text}")
            debug_items.append({
                "text": item.fact.raw_text,
                "score": round(score, 4),
                **item.to_debug_dict(strength_floor=self.STRENGTH_FLOOR),
            })

        context = "Memory (strongest recall first):\n" + "\n".join(lines)
        return context, debug_items

    # =========================================================================
    # Phase Debug Output — Full Thermodynamic State
    # =========================================================================

    def get_phase_debug(self, namespace: str) -> dict[str, Any]:
        """
        Return the complete thermodynamic state for the debug panel.

        Includes global parameters (ρ, event counter, total F, τ_c1)
        and per-item state (s, Σ, τ, F, H, L, phase).

        This is the "instrument panel" of the thermodynamic engine —
        every number maps to a term in the free energy equation.
        """
        self._recompute_all_free_energies(namespace)

        items = self._items.get(namespace, [])
        rho = self._memory_density(namespace)
        total_F = sum(item.free_energy for item in items)

        return {
            "memory_density_rho": round(rho, 6),
            "global_event_counter": self._event_counter,
            "total_free_energy": round(total_F, 4),
            "tau_c1": self.TAU_C1,
            "kT": self.kT,
            "lambda": self.LAMBDA,
            "strength_floor": self.STRENGTH_FLOOR,
            "item_count": len(items),
            "liquid_count": sum(1 for i in items if i.consolidation_strength >= self.STRENGTH_FLOOR),
            "gas_count": sum(1 for i in items if i.consolidation_strength < self.STRENGTH_FLOOR),
            "items": [item.to_debug_dict(strength_floor=self.STRENGTH_FLOOR) for item in items],
        }
