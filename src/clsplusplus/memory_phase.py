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
                # Decrement doc freq for dead item's tokens
                for token in set(item.indexed_tokens):
                    self._doc_freq[token] = max(0, self._doc_freq.get(token, 1) - 1)
        self._items[namespace] = alive

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
        Thermodynamic Semantic Field (TSF) retrieval. Zero external calls.

        Ranking equation:
            rank(q, i) = Σ idf(matched_tokens) - F_i / kT

        Search path (sub-μs):
            query → tokenize + normalize → lookup token_index
            → union candidates → IDF-weighted score → Boltzmann rank → top-k

        Fallback: if zero candidates, return all liquid items by -F/kT.
        """
        self._recompute_all_free_energies(namespace)

        # Tokenize query (same pipeline as store)
        query_tokens = _tokenize(query)
        query_token_set = set(query_tokens)

        # Index lookup → candidates with matched tokens
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

                # IDF-weighted match score
                idf_score = sum(self._compute_idf(t) for t in matched_tokens)

                # Boltzmann rank: IDF score - F/kT
                rank = idf_score - item.free_energy / max(self.kT, 1e-9)
                scored.append((rank, item))
        else:
            # Fallback: return all liquid items by -F/kT
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
        }
