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

ZERO external dependencies. No LLM calls. No embedding models. No APIs.
The engine is fully self-sufficient: store, retrieve, augment.

References:
    - Landauer (1961): Irreversibility and heat generation in the computing process
    - Friston (2010): The free-energy principle: a unified brain theory?
    - Munoz et al. (2025): Long-range order from memory-induced time non-locality (arXiv)
    - LLM triphasic training dynamics (Emergent Mind, 2025)

Copyright (c) 2026 CLS++. All rights reserved.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional
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

# Override signals — words that indicate a fact REPLACES a previous belief
_OVERRIDE_SIGNALS: frozenset[str] = frozenset({
    "only", "exclusively", "always", "never", "actually",
    "switched", "changed", "anymore", "longer",
})


# =============================================================================
# Token Processing — Engine-Internal, Zero External Intelligence
# =============================================================================


def _normalize_token(token: str) -> str:
    """
    Minimal token normalization: strip trailing 'ing' and 's'.

    NOT a stemmer. Two rules, no exceptions, no conditions beyond length.
    The engine indexes BOTH raw and normalized forms, so imperfect
    normalization is acceptable — raw form provides exact match fallback.

    'eating' → 'eat', 'eats' → 'eat', 'visiting' → 'visit',
    'bananas' → 'banana', 'running' → 'runn' (imperfect but unique)
    """
    t = token.lower()
    if len(t) > 4 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text into index-ready tokens.

    Returns a list of tokens ordered by estimated informativeness
    (longer/rarer tokens first). Both raw and normalized forms included.
    Stop words and single-character tokens filtered.

    The ordering matters for field radius: when R(s) contracts, common
    tokens (at the end) are de-indexed first, preserving discriminating tokens.
    """
    raw_tokens: list[str] = []
    seen: set[str] = set()

    for word in text.lower().split():
        if word in _STOP_WORDS or len(word) <= 1:
            continue
        if word not in seen:
            raw_tokens.append(word)
            seen.add(word)
        normalized = _normalize_token(word)
        if normalized != word and normalized not in seen and len(normalized) > 1:
            raw_tokens.append(normalized)
            seen.add(normalized)

    # Sort by length descending — longer tokens are more specific/informative
    # This is a cheap proxy for IDF before we have corpus statistics
    raw_tokens.sort(key=len, reverse=True)
    return raw_tokens


def _has_override(text: str) -> bool:
    """Detect override signals in raw text. Pure pattern matching."""
    words = set(text.lower().split())
    if words & _OVERRIDE_SIGNALS:
        return True
    # Multi-word signals
    lower = text.lower()
    return "no longer" in lower or "not anymore" in lower or "switched to" in lower


# =============================================================================
# Data Structures — The Phases of Information
# =============================================================================


@dataclass(frozen=True)
class Fact:
    """
    Structured episodic memory unit — the LIQUID phase of information.

    A Fact is what remains after raw text (gas) passes through the attention
    gate. It has internal structure: a subject-relation-value
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


# =============================================================================
# Cross-Entity Resonance (CER) — Kuramoto Coupled Oscillators
# =============================================================================


@dataclass
class EntityNode:
    """
    An entity oscillator in the Kuramoto coupling field.

    Each named entity discovered at write time gets a node. The node
    accumulates an IDF-weighted token frequency spectrum (its 'natural frequency')
    and maintains coupling edges to co-occurring entities.

    When two entities share enough experience (K > K_critical), they synchronize —
    forming a coherent cluster whose shared tokens ARE the answers to
    multi-entity queries.
    """

    name: str                                        # Canonical name (lowercase)
    aliases: set                                     # Alternative forms: {"mel", "melanie"}
    token_spectrum: Counter                          # token → IDF-weighted frequency
    memory_ids: list                                 # PhaseMemoryItem IDs mentioning this entity
    birth_order: int                                 # First appearance (event counter)
    total_mentions: int = 0                          # Number of store() calls
    theta: float = 0.0                               # Oscillator phase (radians)
    omega: float = 0.0                               # Natural frequency (spectrum entropy)
    # cached_magnitude_sq removed — SIC coupling doesn't need magnitude


@dataclass
class EntanglementEdge:
    """
    Weighted coupling between two entity oscillators.

    Computed incrementally at store() time. The coupling_strength K(a,b) is
    the IDF-weighted cosine similarity of token spectra.

    shared_tokens stores the actual overlapping tokens — these ARE the
    resonant frequencies = answers to "what do A and B share?"
    """

    entity_a: str                                    # EntityNode.name
    entity_b: str                                    # EntityNode.name
    coupling_strength: float = 0.0                   # K(a,b) — Kuramoto coupling
    shared_tokens: Counter = field(default_factory=Counter)  # resonant frequencies
    shared_memory_ids: list = field(default_factory=list)
    last_updated: int = 0                            # Event counter at last update
    is_synchronized: bool = False                    # K > K_critical


@dataclass
class ResonanceCluster:
    """
    A synchronized cluster of entities above K_critical.

    Emergent structure: entities sharing enough experience self-organize
    into clusters. Multi-hop queries traverse within-cluster edges.
    The cluster_spectrum is the intersection of all member spectra —
    it represents what ALL members have in common.
    """

    cluster_id: str
    members: set                                     # EntityNode names
    cluster_spectrum: Counter                        # Shared tokens across ALL members
    formation_order: int = 0
    coherence: float = 0.0                           # Mean K within cluster


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

    indexed_tokens:
        Engine-generated tokens from the raw text. Ordered by
        informativeness (longer/rarer first). Used for token index
        and field radius computation. Generated internally — no LLM.

    _last_field_radius:
        Last computed R(s) for lazy index updates. When R changes,
        tokens are indexed/de-indexed to match the new radius.
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

    # --- Token Index (engine-generated, zero external intelligence) ---
    indexed_tokens: list[str] = field(default_factory=list)
    _last_field_radius: int = -1  # Last R(s) for lazy index update

    def to_debug_dict(self, strength_floor: float = 0.05) -> dict[str, Any]:
        """Serialize thermodynamic state for the debug panel."""
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
# PhaseMemoryEngine — Thermodynamic Memory System
# =============================================================================


class PhaseMemoryEngine:
    """
    Thermodynamic memory engine implementing the Gas → Liquid phase transition.

    ZERO external dependencies. No LLM calls. No embedding models. No APIs.
    The engine is fully self-sufficient: store, retrieve, augment.

    Flow: User → store() → search() → build_augmented_context() → User

    Minimizes the free energy functional:

        F(θ, Σ, ρ, τ) = E_prediction(θ) − Σ · S_model(θ) + λ · L_landauer(θ, τ)

    where:
        E_prediction = 1 − s(t)                           [prediction error]
        S_model      = H(item) · ρ                        [model entropy]
        L_landauer   = kT · ln(2) · H(item) / τ           [Landauer cost]

    Phase transition at τ = τ_c1:
        τ < τ_c1  →  s=0 minimum of F is global  →  gas (memory evaporates)
        τ > τ_c1  →  s=1 minimum of F is global  →  liquid (memory persists)
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
        self._items: dict[str, list[PhaseMemoryItem]] = {}
        self._event_counter: int = 0

        # Token Index — Thermodynamic Semantic Field (TSF)
        # Single token index: token → list of PhaseMemoryItems
        # Entries phase-transition in/out based on R(s) = floor(N × s^(1/3))
        self._token_index: dict[str, list[PhaseMemoryItem]] = {}

        # Document frequency for IDF computation (self-computed from corpus)
        self._doc_freq: Counter = Counter()

        # --- Cross-Entity Resonance (CER) — Kuramoto Coupled Oscillators ---
        self._entity_nodes: dict[str, EntityNode] = {}
        self._entity_alias_map: dict[str, str] = {}
        self._entanglement_graph: dict[str, dict[str, EntanglementEdge]] = {}
        self._resonance_clusters: dict[str, ResonanceCluster] = {}
        self._entity_index: dict[str, list[str]] = {}  # token → entity names
        self._K_critical: float = 0.15  # Synchronization phase transition threshold

    # =========================================================================
    # Core Physics — Information Content (Shannon Entropy)
    # =========================================================================

    @staticmethod
    def _information_content(fact: Fact) -> float:
        """
        Compute Shannon entropy H(fact) in bits.

        H = −Σ p(c) · log₂(p(c))

        Character-level entropy of "subject relation value".
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
    # Token Index — Phase-Modulated Field Radius
    # =========================================================================

    def _index_item(self, item: PhaseMemoryItem) -> None:
        """
        Add a memory item to the token index under its active tokens.

        The number of indexed tokens depends on the field radius R(s):

            R(s) = floor(N_tokens × s^(1/3))

        where 1/3 is the mean-field critical exponent ν from 3D thermodynamics.
        Liquid memories (s→1) have broad fields; gas memories have none.

        Tokens are ordered by informativeness (longer first, which correlates
        with rarity). As s decays, common tokens are de-indexed first,
        preserving the most discriminating tokens longest.
        """
        s = item.consolidation_strength
        if s < self.STRENGTH_FLOOR:
            self._deindex_item(item)
            item._last_field_radius = 0
            return

        tokens = item.indexed_tokens
        n_tokens = len(tokens)
        cube_root_s = s ** (1.0 / 3.0)
        radius = max(1, int(n_tokens * cube_root_s))

        # Index tokens within radius
        for token in tokens[:radius]:
            if token not in self._token_index:
                self._token_index[token] = []
            if item not in self._token_index[token]:
                self._token_index[token].append(item)

        # De-index tokens beyond radius
        for token in tokens[radius:]:
            if token in self._token_index and item in self._token_index[token]:
                self._token_index[token].remove(item)
                if not self._token_index[token]:
                    del self._token_index[token]

        item._last_field_radius = radius

    def _deindex_item(self, item: PhaseMemoryItem) -> None:
        """Remove a memory item from the token index entirely."""
        for token in item.indexed_tokens:
            if token in self._token_index and item in self._token_index[token]:
                self._token_index[token].remove(item)
                if not self._token_index[token]:
                    del self._token_index[token]

    def _update_field_radius(self, item: PhaseMemoryItem) -> None:
        """Lazy field radius update. Only re-indexes if R(s) has changed."""
        s = item.consolidation_strength
        if s < self.STRENGTH_FLOOR:
            if item._last_field_radius != 0:
                self._deindex_item(item)
                item._last_field_radius = 0
            return

        cube_root_s = s ** (1.0 / 3.0)
        new_radius = max(1, int(len(item.indexed_tokens) * cube_root_s))

        if new_radius != item._last_field_radius:
            self._index_item(item)

    def _compute_idf(self, token: str) -> float:
        """
        Compute Inverse Document Frequency for a token.

        idf(t) = log(1 + N / (1 + df(t)))

        Self-computed from the engine's own corpus. Zero external knowledge.
        N = total items across all namespaces.
        df(t) = number of items containing token t.
        """
        total_items = sum(len(items) for items in self._items.values())
        df = self._doc_freq.get(token, 0)
        return math.log(1.0 + total_items / (1.0 + df))

    # =========================================================================
    # Contradiction Detection — Token Overlap (No S/R/V Required)
    # =========================================================================

    def _detect_contradiction(
        self,
        new_tokens: set[str],
        existing_item: PhaseMemoryItem,
    ) -> tuple[str, float]:
        """
        Detect contradiction between new text and existing memory via token overlap.

        Returns:
            ('confirmation', 0.0) — high overlap, same content
            ('contradiction', surprise) — partial overlap, different content
            ('unrelated', 0.0) — low overlap, different topics
        """
        old_tokens = set(existing_item.indexed_tokens)
        if not old_tokens or not new_tokens:
            return "unrelated", 0.0

        shared = new_tokens & old_tokens
        union = new_tokens | old_tokens
        shared_ratio = len(shared) / max(len(union), 1)

        if shared_ratio > 0.6:
            return "confirmation", 0.0
        if shared_ratio > 0.25:
            # Jaccard distance as surprise proxy
            surprise = 1.0 - shared_ratio
            return "contradiction", surprise
        return "unrelated", 0.0

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

        For structured facts: matches on (subject, relation) dimension.
        """
        contradicted: list[PhaseMemoryItem] = []
        max_surprise = 0.0

        for item in existing_items:
            if (
                item.fact.subject == new_fact.subject
                and item.fact.relation == new_fact.relation
                and item.consolidation_strength >= self.STRENGTH_FLOOR
            ):
                if item.fact.value == new_fact.value:
                    continue  # Confirmation

                contradicted.append(item)

                if new_fact.override:
                    sigma = -math.log(1e-6)  # ≈ 13.8 nats
                else:
                    sigma = self._bigram_divergence(new_fact.value, item.fact.value)

                max_surprise = max(max_surprise, sigma)

        return max_surprise, contradicted

    def _compute_surprise_from_tokens(
        self,
        new_text: str,
        new_tokens: set[str],
        namespace: str,
    ) -> tuple[float, list[PhaseMemoryItem]]:
        """
        Compute surprise from raw text using token overlap.

        Used when storing raw text without structured (S, R, V) facts.
        Detects contradictions by finding existing memories with high
        token overlap but different content.
        """
        existing = self._items.get(namespace, [])
        contradicted: list[PhaseMemoryItem] = []
        max_surprise = 0.0
        is_override = _has_override(new_text)

        for item in existing:
            if item.consolidation_strength < self.STRENGTH_FLOOR:
                continue

            result, surprise = self._detect_contradiction(new_tokens, item)

            if result == "confirmation":
                # High overlap = likely same content, reinforce
                item.retrieval_count += 1
            elif result == "contradiction":
                contradicted.append(item)
                if is_override:
                    surprise = -math.log(1e-6)  # Override amplification
                max_surprise = max(max_surprise, surprise)

        return max_surprise, contradicted

    @staticmethod
    def _bigram_divergence(new_value: str, old_value: str) -> float:
        """Approximate KL divergence via 1 − Jaccard similarity of character bigrams."""
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

        Damage formula:
            D = σ(Σ_norm) · (τ_new / τ_old) · amplifier

        where σ is a sigmoid sharpening function.
        """
        SIGMA_MAX = -math.log(1e-6)
        sigma_norm = min(surprise / SIGMA_MAX, 1.0)

        sigmoid_damage = 1.0 / (1.0 + math.exp(-10.0 * (sigma_norm - 0.5)))

        amplifier = 1.5 if new_fact.override else 1.0
        tau_new = self.TAU_OVERRIDE if new_fact.override else self.TAU_DEFAULT

        for item in contradicted:
            tau_ratio = tau_new / max(item.tau, 1e-6)
            tau_factor = min(tau_ratio, 4.0) / 4.0 + 0.5
            damage = sigmoid_damage * tau_factor * amplifier
            item.accumulated_surprise_damage = min(
                item.accumulated_surprise_damage + damage,
                2.0,
            )

    def _apply_token_surprise_damage(
        self,
        surprise: float,
        contradicted: list[PhaseMemoryItem],
        is_override: bool,
    ) -> None:
        """Apply surprise damage for token-based contradiction detection."""
        SIGMA_MAX = -math.log(1e-6)
        sigma_norm = min(surprise / SIGMA_MAX, 1.0)

        sigmoid_damage = 1.0 / (1.0 + math.exp(-10.0 * (sigma_norm - 0.5)))
        amplifier = 1.5 if is_override else 1.0

        for item in contradicted:
            damage = sigmoid_damage * amplifier
            item.accumulated_surprise_damage = min(
                item.accumulated_surprise_damage + damage,
                2.0,
            )

    # =========================================================================
    # Core Physics — Consolidation Strength s(t)
    # =========================================================================

    def _compute_consolidation(self, item: PhaseMemoryItem, delta_t: int) -> float:
        """
        Compute the order parameter s(t) — consolidation strength.

        s(t) = s₀ · exp(−Δt / τ) · (1 + β · ln(1 + R)) − D
        """
        natural_decay = math.exp(-delta_t / max(item.tau, 1e-6))
        retrieval_boost = 1.0 + self.BETA_RETRIEVAL * math.log1p(item.retrieval_count)
        s = 1.0 * natural_decay * retrieval_boost - item.accumulated_surprise_damage
        return max(0.0, min(1.0, s))

    # =========================================================================
    # Core Physics — Landauer Cost
    # =========================================================================

    def _compute_landauer_cost(self, item: PhaseMemoryItem) -> float:
        """L = kT · ln(2) · H(item) / τ"""
        return (self.kT * math.log(2) * item.information_content_bits) / max(item.tau, 1e-6)

    # =========================================================================
    # Core Physics — Free Energy F(θ, Σ, ρ, τ)
    # =========================================================================

    def _compute_free_energy(self, item: PhaseMemoryItem, global_rho: float) -> float:
        """
        Compute per-item free energy.

        F(θ, Σ, ρ, τ) = E_prediction(θ) − Σ · S_model(θ) + λ · L_landauer(θ, τ)
        """
        delta_t = self._event_counter - item.birth_order

        s = self._compute_consolidation(item, delta_t)
        item.consolidation_strength = s

        E_pred = 1.0 - s
        S_model = item.information_content_bits * max(global_rho, 1e-9)
        L_land = self._compute_landauer_cost(item)
        item.landauer_cost = L_land

        F = E_pred - item.surprise_at_birth * S_model + self.LAMBDA * L_land
        item.free_energy = F
        return F

    # =========================================================================
    # Memory Density ρ
    # =========================================================================

    def _memory_density(self, namespace: str) -> float:
        """ρ = |active items| / capacity."""
        items = self._items.get(namespace, [])
        active = sum(1 for item in items if item.consolidation_strength >= self.STRENGTH_FLOOR)
        return active / max(self.CAPACITY, 1)

    # =========================================================================
    # Cross-Entity Resonance — Entity Detection (Zero LLM)
    # =========================================================================

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """
        Extract entity candidates from raw text via capitalization heuristic.

        Rules (zero LLM, zero regex):
        1. Words starting with uppercase that are NOT sentence-initial
        2. Multi-word entities via consecutive capitals: "New York" → "new york"

        Returns normalized (lowercase) entity names.
        """
        words = text.split()
        entities: list[str] = []
        i = 0
        while i < len(words):
            word = words[i]
            if not word:
                i += 1
                continue

            # Skip sentence-initial words
            is_sentence_start = (i == 0) or (
                i > 0 and len(words[i - 1]) > 0 and words[i - 1][-1] in ".!?"
            )

            if (
                not is_sentence_start
                and word[0].isupper()
                and word.lower() not in _STOP_WORDS
                and len(word) > 1
            ):
                # Check for multi-word entity (consecutive capitals)
                entity_parts = [word]
                j = i + 1
                while (
                    j < len(words)
                    and words[j]
                    and words[j][0].isupper()
                    and words[j].lower() not in _STOP_WORDS
                ):
                    entity_parts.append(words[j])
                    j += 1
                entity = " ".join(entity_parts).lower().strip()
                if entity not in entities:
                    entities.append(entity)
                i = j
            else:
                i += 1

        return entities

    def _resolve_alias(self, name: str) -> str:
        """
        Resolve an entity name to its canonical form.

        Rules:
        1. Exact match in alias_map → return canonical (pre-registered)
        2. Exact match in _entity_nodes → return self
        3. No match → new entity (NO auto-guessing, NO prefix matching)

        Why no auto-prefix: "art"→"arthur", "city1"→"city10", "al"→"alice"
        are all false positives. Guessing is worse than creating a separate
        entity. Explicit alias registration via register_alias() only.
        """
        if name in self._entity_alias_map:
            return self._entity_alias_map[name]

        if name in self._entity_nodes:
            return name

        return name

    def register_alias(self, alias: str, canonical: str) -> None:
        """
        Explicitly register an alias for an entity.

        Called by orchestration layer when user says 'Mel is Melanie'.
        The engine never guesses aliases — only stores explicit mappings.
        """
        alias = alias.lower()
        canonical = canonical.lower()
        self._entity_alias_map[alias] = canonical
        if canonical in self._entity_nodes:
            self._entity_nodes[canonical].aliases.add(alias)

    # =========================================================================
    # Cross-Entity Resonance — Write-Time Coupling
    # =========================================================================

    def _cer_update(self, item: PhaseMemoryItem, text: str, namespace: str) -> None:
        """
        Cross-Entity Resonance update at write time.

        Relationships between entities are discovered and stored here,
        amortized over N store() calls. Query time just follows
        pre-computed entanglement links — O(1).

        Steps:
        1. Extract entities from text
        2. Create/update EntityNodes with IDF-weighted token spectra
        3. For entity pairs in this item: update EntanglementEdge
        4. For entities sharing tokens with OTHER entities: update edges (top-10)
        5. Check synchronization threshold + update clusters

        Complexity: O(E² × S + 10 × S) per store, E=1-3 typical.
        """
        entities_raw = self._extract_entities(text)

        # Also include fact.subject if available
        if (
            item.fact
            and item.fact.subject
            and item.fact.subject not in _STOP_WORDS
            and len(item.fact.subject) > 1
        ):
            subject = item.fact.subject.lower()
            if subject not in entities_raw:
                entities_raw.append(subject)

        if not entities_raw:
            return

        # Resolve aliases
        entities = list(set(self._resolve_alias(e) for e in entities_raw))

        # Compute IDF for item tokens
        tokens_with_idf: dict[str, float] = {}
        for token in item.indexed_tokens:
            tokens_with_idf[token] = self._compute_idf(token)

        # Create/update EntityNodes
        for entity_name in entities:
            if entity_name not in self._entity_nodes:
                self._entity_nodes[entity_name] = EntityNode(
                    name=entity_name,
                    aliases={entity_name},
                    token_spectrum=Counter(),
                    memory_ids=[],
                    birth_order=self._event_counter,
                )

            node = self._entity_nodes[entity_name]
            node.memory_ids.append(item.id)
            node.total_mentions += 1

            # Update token spectrum (IDF-weighted)
            for token, idf in tokens_with_idf.items():
                if token != entity_name:
                    node.token_spectrum[token] = node.token_spectrum.get(token, 0.0) + idf

            # Update entity index (token → entity names)
            for token in item.indexed_tokens:
                if token not in self._entity_index:
                    self._entity_index[token] = []
                if entity_name not in self._entity_index[token]:
                    self._entity_index[token].append(entity_name)

            # Compute natural frequency (spectrum entropy)
            total_weight = sum(node.token_spectrum.values())
            if total_weight > 0:
                entropy = 0.0
                for count in node.token_spectrum.values():
                    p = count / total_weight
                    if p > 0:
                        entropy -= p * math.log2(p)
                node.omega = entropy

        # Update entanglement edges for entity pairs IN this item
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                self._update_entanglement(entities[i], entities[j], item)

        # Cross-item coupling: entities sharing tokens with OTHER entities.
        # SIC coupling (IDF²) naturally discriminates — common verbs contribute
        # negligible IDF² while rare shared tokens dominate. No artificial filter needed.
        # Only exclude: entity names as tokens (prevent self-referential coupling),
        # and limit to top-5 related entities per store() for O(1) amortized cost.
        for entity_name in entities:
            related: Counter = Counter()
            for token in item.indexed_tokens:
                if token in self._entity_index and token not in self._entity_nodes:
                    idf = tokens_with_idf.get(token, 0.0)
                    for other in self._entity_index[token]:
                        if other != entity_name and other not in entities:
                            related[other] += idf

            for other_entity, _ in related.most_common(5):
                self._update_entanglement(entity_name, other_entity, item)

        # Update clusters
        for entity_name in entities:
            self._update_clusters(entity_name)

    def _update_entanglement(
        self,
        entity_a: str,
        entity_b: str,
        item: PhaseMemoryItem,
    ) -> None:
        """
        Incrementally update the entanglement edge between two entities.

        K(a,b) = Shared Information Content (SIC) — sum of IDF² of shared
        discriminating tokens, normalized by geometric mean of spectrum sizes.

        Why SIC, not cosine:
        1. IDF² naturally suppresses common words (verbs, prepositions)
        2. No magnitude normalization → more shared rare tokens = stronger coupling
        3. No sqrt, no division by magnitude → O(|shared|), numerically stable
        4. Thermodynamically interpretable: shared surprise in bits²
        5. Entity-name tokens are EXCLUDED — prevents self-referential coupling

        Complexity: O(min(|spec_a|, |spec_b|)).
        """
        a, b = (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)

        if a not in self._entanglement_graph:
            self._entanglement_graph[a] = {}

        if b not in self._entanglement_graph[a]:
            self._entanglement_graph[a][b] = EntanglementEdge(
                entity_a=a,
                entity_b=b,
            )

        edge = self._entanglement_graph[a][b]
        edge.last_updated = self._event_counter

        # Track shared memories
        node_a = self._entity_nodes.get(a)
        node_b = self._entity_nodes.get(b)
        if not node_a or not node_b:
            return

        if item.id in node_a.memory_ids and item.id in node_b.memory_ids:
            if item.id not in edge.shared_memory_ids:
                edge.shared_memory_ids.append(item.id)

        # Iterate over the SMALLER spectrum for efficiency
        small, big = (node_a.token_spectrum, node_b.token_spectrum) \
            if len(node_a.token_spectrum) <= len(node_b.token_spectrum) \
            else (node_b.token_spectrum, node_a.token_spectrum)

        # All known entity names — exclude from coupling to prevent
        # self-referential entanglement ("alice" token coupling alice+bob)
        entity_names = set(self._entity_nodes.keys())

        shared = Counter()
        sic_sum = 0.0
        for token in small:
            if token in big and token not in entity_names:
                idf = self._compute_idf(token)
                shared[token] = min(small[token], big[token])
                sic_sum += idf * idf  # IDF² — surprise² in bits²

        edge.shared_tokens = shared

        # Normalize by geometric mean of spectrum sizes (excluding entity names)
        size_a = max(sum(1 for t in node_a.token_spectrum if t not in entity_names), 1)
        size_b = max(sum(1 for t in node_b.token_spectrum if t not in entity_names), 1)
        normalizer = math.sqrt(size_a * size_b)

        edge.coupling_strength = sic_sum / normalizer
        edge.is_synchronized = edge.coupling_strength > self._K_critical

    def _update_clusters(self, entity_name: str) -> None:
        """
        Update resonance clusters after entanglement edges change.

        BFS on synchronized edges → connected components.
        Uses reverse adjacency built on-the-fly for O(degree) per step.
        """
        # Build reverse adjacency index: O(|E|) total, not O(|E|) per BFS step
        reverse_adj: dict[str, list[str]] = {}
        for a, edges in self._entanglement_graph.items():
            for b, edge in edges.items():
                if edge.is_synchronized:
                    reverse_adj.setdefault(b, []).append(a)

        visited: set[str] = set()
        queue = [entity_name]
        cluster_members: set[str] = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            cluster_members.add(current)

            # Forward edges (a < b, so current=a → b is in graph[a])
            if current in self._entanglement_graph:
                for other, edge in self._entanglement_graph[current].items():
                    if edge.is_synchronized and other not in visited:
                        queue.append(other)

            # Reverse edges via pre-built index
            if current in reverse_adj:
                for other in reverse_adj[current]:
                    if other not in visited:
                        queue.append(other)

        if len(cluster_members) < 2:
            return

        # Check if cluster already exists
        for cluster in self._resonance_clusters.values():
            if cluster.members == cluster_members:
                return  # Already exists

        # Remove old clusters that overlap
        to_remove = [
            cid for cid, c in self._resonance_clusters.items()
            if c.members & cluster_members
        ]
        for cid in to_remove:
            del self._resonance_clusters[cid]

        # Create new cluster with spectrum = intersection of all member spectra
        cluster_spectrum = Counter()
        first = True
        for member in cluster_members:
            node = self._entity_nodes.get(member)
            if not node:
                continue
            if first:
                cluster_spectrum = Counter(node.token_spectrum)
                first = False
            else:
                cluster_spectrum = Counter({
                    t: min(cluster_spectrum[t], node.token_spectrum[t])
                    for t in cluster_spectrum
                    if t in node.token_spectrum
                })

        # Compute coherence (mean K within cluster)
        k_sum = 0.0
        k_count = 0
        members_list = sorted(cluster_members)
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                a, b = members_list[i], members_list[j]
                edge = self._entanglement_graph.get(a, {}).get(b)
                if edge:
                    k_sum += edge.coupling_strength
                    k_count += 1

        cluster_id = str(uuid4())
        self._resonance_clusters[cluster_id] = ResonanceCluster(
            cluster_id=cluster_id,
            members=cluster_members,
            cluster_spectrum=cluster_spectrum,
            formation_order=self._event_counter,
            coherence=k_sum / max(k_count, 1),
        )

    # =========================================================================
    # Cross-Entity Resonance — GC for Entity Structures
    # =========================================================================

    def _cer_gc_item(self, item: PhaseMemoryItem) -> None:
        """
        Clean entity structures when a memory item is garbage collected.

        Scans ALL entity nodes for references to this item, not just
        fact.subject — because entities are also extracted from text
        capitalization and may not match the structured subject.
        """
        item_id = item.id
        entities_to_remove: list[str] = []

        for entity_name, node in self._entity_nodes.items():
            if item_id in node.memory_ids:
                node.memory_ids.remove(item_id)
                if not node.memory_ids:
                    entities_to_remove.append(entity_name)

        for entity_name in entities_to_remove:
            del self._entity_nodes[entity_name]
            # Clean alias map
            aliases_to_remove = [
                alias for alias, canonical in self._entity_alias_map.items()
                if canonical == entity_name
            ]
            for alias in aliases_to_remove:
                del self._entity_alias_map[alias]
            # Clean entity index
            for token_entities in self._entity_index.values():
                if entity_name in token_entities:
                    token_entities.remove(entity_name)

    def _prune_entanglement_graph(self) -> None:
        """Prune stale entanglement edges. Called during free energy recomputation."""
        EDGE_DECAY = 0.99
        PRUNE_THRESHOLD = 0.01

        to_remove: list[tuple[str, str]] = []
        for a in self._entanglement_graph:
            for b, edge in self._entanglement_graph[a].items():
                age = self._event_counter - edge.last_updated
                if age > 100:
                    edge.coupling_strength *= EDGE_DECAY ** (age / 100)
                    edge.is_synchronized = edge.coupling_strength > self._K_critical

                if edge.coupling_strength < PRUNE_THRESHOLD:
                    to_remove.append((a, b))

        for a, b in to_remove:
            if a in self._entanglement_graph and b in self._entanglement_graph[a]:
                del self._entanglement_graph[a][b]
                if not self._entanglement_graph[a]:
                    del self._entanglement_graph[a]

    # =========================================================================
    # Store — Self-Sufficient Ingestion (Zero LLM)
    # =========================================================================

    def store(
        self,
        text: str,
        namespace: str,
        fact: Optional[Fact] = None,
    ) -> Optional[PhaseMemoryItem]:
        """
        Store raw text as a memory item. Zero external calls.

        The engine tokenizes, indexes, detects contradictions, and manages
        thermodynamic state entirely on its own.

        If a pre-extracted Fact is provided (from an external orchestration
        layer), it's used for structured contradiction detection. Otherwise,
        token-overlap contradiction detection is used.

        Args:
            text: Raw text to store.
            namespace: Memory namespace.
            fact: Optional pre-extracted Fact for structured surprise computation.

        Returns:
            PhaseMemoryItem if stored, None if confirmed (duplicate reinforced).
        """
        self._event_counter += 1

        # --- Tokenize (engine-internal) ---
        tokens = _tokenize(text)
        token_set = set(tokens)

        # --- Build or use Fact ---
        if fact is None:
            # Auto-create minimal fact from raw text
            # Subject: first non-stop token, relation: second, value: rest
            content_words = [w for w in text.lower().split()
                            if w not in _STOP_WORDS and len(w) > 1]
            if len(content_words) >= 3:
                subject = content_words[0]
                relation = content_words[1]
                value = " ".join(content_words[2:])
            elif len(content_words) == 2:
                subject = content_words[0]
                relation = content_words[1]
                value = content_words[1]
            elif len(content_words) == 1:
                subject = content_words[0]
                relation = ""
                value = ""
            else:
                subject = ""
                relation = ""
                value = ""

            override = _has_override(text)
            fact = Fact(
                subject=subject,
                relation=relation,
                value=value,
                override=override,
                raw_text=text,
            )

        existing = self._items.get(namespace, [])

        # --- Check for confirmation (exact duplicate) ---
        for item in existing:
            if (
                item.fact.subject == fact.subject
                and item.fact.relation == fact.relation
                and item.fact.value == fact.value
                and item.consolidation_strength >= self.STRENGTH_FLOOR
            ):
                item.retrieval_count += 1
                return item  # Confirmation — reinforced existing

        # --- Compute surprise ---
        if fact.subject and fact.relation:
            # Structured surprise (if we have S, R, V)
            surprise, contradicted = self._compute_surprise(fact, existing)
        else:
            # Token-based surprise (raw text)
            surprise, contradicted = self._compute_surprise_from_tokens(
                text, token_set, namespace,
            )

        # --- Apply surprise damage ---
        if contradicted:
            if fact.subject and fact.relation:
                self._apply_surprise_damage(surprise, contradicted, fact)
            else:
                self._apply_token_surprise_damage(
                    surprise, contradicted, fact.override,
                )

        # --- Compute thermodynamic state ---
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
            indexed_tokens=tokens,
        )

        # --- Store + Index ---
        self._items.setdefault(namespace, []).append(item)
        self._index_item(item)

        # Update document frequency
        for token in set(tokens):
            self._doc_freq[token] += 1

        # --- Recompute free energy for ALL items ---
        self._recompute_all_free_energies(namespace)

        # --- Cross-Entity Resonance: Write-Time Coupling ---
        self._cer_update(item, text, namespace)

        return item

    # =========================================================================
    # Free Energy Recomputation
    # =========================================================================

    def _recompute_all_free_energies(self, namespace: str) -> None:
        """Recompute F(θ) for every item in the namespace, then GC dead items."""
        items = self._items.get(namespace, [])
        rho = self._memory_density(namespace)
        for item in items:
            self._compute_free_energy(item, rho)

        alive = []
        for item in items:
            if item.consolidation_strength > 0.0 or item.accumulated_surprise_damage < 1.0:
                alive.append(item)
            else:
                self._deindex_item(item)
                self._cer_gc_item(item)
                # Decrement doc freq for dead item's tokens
                for token in set(item.indexed_tokens):
                    self._doc_freq[token] = max(0, self._doc_freq.get(token, 1) - 1)
        self._items[namespace] = alive

        # Prune stale entanglement edges
        self._prune_entanglement_graph()

    # =========================================================================
    # Retrieval — IDF-Weighted Token Match + Boltzmann Ranking
    # =========================================================================

    def search(
        self,
        query: str,
        namespace: str,
        limit: int = 10,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Unified retrieval: Cross-Entity Resonance (CER) + TSF.

        For multi-entity queries (2+ recognized entities):
            Uses pre-computed entanglement graph — O(1) edge lookup.
            CER results merged with TSF results, CER boosted 2×.

        For single-entity or unrecognized queries:
            Falls back to standard TSF (IDF-weighted Boltzmann ranking).

        Zero external calls. Sub-millisecond.
        """
        self._recompute_all_free_energies(namespace)

        query_tokens = _tokenize(query)
        query_token_set = set(query_tokens)

        # --- CER: Multi-entity detection ---
        query_entities = self._detect_multi_entity_query(query_token_set, namespace)

        if len(query_entities) >= 2:
            cer_results = self._cer_search(
                query_entities, query_token_set, namespace, limit,
            )
            if cer_results:
                tsf_results = self._tsf_search(
                    query_tokens, query_token_set, namespace, limit,
                )
                return self._merge_cer_and_tsf(cer_results, tsf_results, limit)

        return self._tsf_search(query_tokens, query_token_set, namespace, limit)

    # =========================================================================
    # TSF Search — Standard Token-Index Retrieval
    # =========================================================================

    def _tsf_search(
        self,
        query_tokens: list[str],
        query_token_set: set[str],
        namespace: str,
        limit: int,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Standard Thermodynamic Semantic Field retrieval.

        rank(q, i) = Σ idf(matched_tokens) - F_i / kT
        """
        candidates: dict[str, tuple[PhaseMemoryItem, list[str]]] = {}

        for token in query_tokens:
            if token in self._token_index:
                for item in self._token_index[token]:
                    if item.namespace != namespace:
                        continue
                    if item.id not in candidates:
                        candidates[item.id] = (item, [])
                    if token not in candidates[item.id][1]:
                        candidates[item.id][1].append(token)

        scored: list[tuple[float, PhaseMemoryItem]] = []

        if candidates:
            for item, matched_tokens in candidates.values():
                if item.consolidation_strength < self.STRENGTH_FLOOR:
                    continue

                self._update_field_radius(item)

                idf_score = sum(self._compute_idf(t) for t in matched_tokens)
                rank = idf_score - item.free_energy / max(self.kT, 1e-9)
                scored.append((rank, item))
        else:
            items = self._items.get(namespace, [])
            for item in items:
                if item.consolidation_strength < self.STRENGTH_FLOOR:
                    continue
                rank = -item.free_energy / max(self.kT, 1e-9)
                scored.append((rank, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        for _, item in scored[:limit]:
            item.retrieval_count += 1

        return scored[:limit]

    # =========================================================================
    # CER Search — Entanglement Graph Traversal
    # =========================================================================

    def _detect_multi_entity_query(
        self,
        query_token_set: set[str],
        namespace: str,
    ) -> list[str]:
        """
        Detect if query references 2+ known entities.

        Truly O(Q): one hash lookup per query token. Namespace membership
        verified via pre-built set of item IDs — O(1) per check.
        """
        # Build namespace member set ONCE — O(N)
        ns_item_ids = {item.id for item in self._items.get(namespace, [])}

        matched: list[str] = []
        for token in query_token_set:
            canonical = token
            if token in self._entity_alias_map:
                canonical = self._entity_alias_map[token]
            if canonical in self._entity_nodes:
                node = self._entity_nodes[canonical]
                # O(|memory_ids|) with O(1) set lookup — not O(N²)
                has_ns = any(mid in ns_item_ids for mid in node.memory_ids)
                if has_ns and canonical not in matched:
                    matched.append(canonical)
        return matched

    def _cer_search(
        self,
        entities: list[str],
        query_tokens: set[str],
        namespace: str,
        limit: int,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Search via the entanglement graph.

        For 2 entities: O(1) edge lookup → shared_tokens = resonant frequencies.
        For 3+ entities: cluster lookup → cluster_spectrum.

        Score: K × idf × s − F/kT
        """
        scored: list[tuple[float, PhaseMemoryItem]] = []

        if len(entities) == 2:
            a, b = sorted(entities)
            edge = self._entanglement_graph.get(a, {}).get(b)

            if edge and edge.coupling_strength > 0:
                # Use shared_tokens as resonant frequencies
                search_tokens = edge.shared_tokens
                if not search_tokens:
                    return []

                for token, weight in search_tokens.most_common(20):
                    if token in self._token_index:
                        for item in self._token_index[token]:
                            if item.namespace != namespace:
                                continue
                            if item.consolidation_strength < self.STRENGTH_FLOOR:
                                continue
                            idf = self._compute_idf(token)
                            cer_score = (
                                edge.coupling_strength * idf
                                * item.consolidation_strength
                                - item.free_energy / max(self.kT, 1e-9)
                            )
                            scored.append((cer_score, item))

                # Also include shared memories
                for memory_id in edge.shared_memory_ids:
                    for item in self._items.get(namespace, []):
                        if item.id == memory_id and item.consolidation_strength >= self.STRENGTH_FLOOR:
                            rank = (
                                edge.coupling_strength * 10.0
                                - item.free_energy / max(self.kT, 1e-9)
                            )
                            scored.append((rank, item))

        elif len(entities) >= 3:
            # Find cluster containing all entities
            entity_set = set(entities)
            for cluster in self._resonance_clusters.values():
                if entity_set.issubset(cluster.members):
                    for token, weight in cluster.cluster_spectrum.most_common(20):
                        if token in self._token_index:
                            for item in self._token_index[token]:
                                if item.namespace != namespace:
                                    continue
                                if item.consolidation_strength < self.STRENGTH_FLOOR:
                                    continue
                                idf = self._compute_idf(token)
                                cer_score = (
                                    cluster.coherence * idf
                                    - item.free_energy / max(self.kT, 1e-9)
                                )
                                scored.append((cer_score, item))
                    break

        # Deduplicate by item ID, keeping highest score
        seen: dict[str, tuple[float, PhaseMemoryItem]] = {}
        for score, item in scored:
            if item.id not in seen or score > seen[item.id][0]:
                seen[item.id] = (score, item)

        result = sorted(seen.values(), key=lambda x: x[0], reverse=True)
        return result[:limit]

    def _merge_cer_and_tsf(
        self,
        cer_results: list[tuple[float, PhaseMemoryItem]],
        tsf_results: list[tuple[float, PhaseMemoryItem]],
        limit: int,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Merge CER and TSF results. CER gets 2× boost because pre-computed
        cross-entity resonances are more likely correct for multi-entity queries.
        """
        CER_BOOST = 2.0

        merged: dict[str, tuple[float, PhaseMemoryItem]] = {}

        for score, item in cer_results:
            merged[item.id] = (score * CER_BOOST, item)

        for score, item in tsf_results:
            if item.id in merged:
                existing_score = merged[item.id][0]
                merged[item.id] = (max(existing_score, score), item)
            else:
                merged[item.id] = (score, item)

        result = sorted(merged.values(), key=lambda x: x[0], reverse=True)

        for _, item in result[:limit]:
            item.retrieval_count += 1

        return result[:limit]

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
        Build the memory context string for downstream consumption.
        Zero external calls.
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
        """Return the complete thermodynamic state for the debug panel."""
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
            "cer": {
                "entity_count": len(self._entity_nodes),
                "edge_count": sum(len(edges) for edges in self._entanglement_graph.values()),
                "cluster_count": len(self._resonance_clusters),
                "synchronized_edges": sum(
                    1 for edges in self._entanglement_graph.values()
                    for edge in edges.values() if edge.is_synchronized
                ),
            },
        }
