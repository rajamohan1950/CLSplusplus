"""
Gas → Liquid → Solid Phase Transition: Thermodynamic Memory Engine.

Memory is a phase of matter. This module implements the full phase diagram:

    Gas → Liquid:  τ > τ_c1  (attention gate — experiences persist)
    Liquid → Solid: ΔF < 0   (crystallization — episodes compress into schemas)
    Solid → Glass:  H_history converges (over-consolidation — schemas become rigid)

The free energy formulation:

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
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4


# =============================================================================
# Tracer helpers — lazy import so PhaseMemoryEngine stays zero-external-dep
# =============================================================================

def _algo_span(trace_id: Optional[str], label: str, **meta):
    """Return a tracer span context manager, or a nullcontext if no trace_id."""
    if not trace_id:
        from contextlib import nullcontext
        return nullcontext()
    try:
        from clsplusplus.tracer import tracer as _tr  # type: ignore[import]
        return _tr.span(trace_id, label, "phase_engine", **meta)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


def _add_algo_meta(trace_id: Optional[str], hop_id: Optional[str], **kwargs) -> None:
    """Add metadata to a tracer hop — no-op if tracing is unavailable."""
    if not (trace_id and hop_id):
        return
    try:
        from clsplusplus.tracer import tracer as _tr  # type: ignore[import]
        _tr.add_metadata(trace_id, hop_id, **kwargs)
    except Exception:
        pass


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
    "exclusively", "switched", "anymore",
})


# =============================================================================
# Token Processing — Engine-Internal, Zero External Intelligence
#
# Semantic Renormalization Group (SRG): The tokenizer IS the lattice
# discretization. Punctuation is lattice-scale (UV) noise. _strip_punctuation
# removes it without changing the IR physics (critical exponents, field radius).
# =============================================================================

_PUNCTUATION_CHARS: frozenset[str] = frozenset(
    string.punctuation
    + "\u2018\u2019\u201a\u201b"  # Single curly quotes: ' ' ‚ ‛
    + "\u201c\u201d\u201e\u201f"  # Double curly quotes: " " „ ‟
    + "\u2013\u2014\u2015"        # En-dash, em-dash, horizontal bar: – — ―
    + "\u2026"                     # Ellipsis: …
    + "\u00ab\u00bb"               # Guillemets: « »
    + "\u2039\u203a"               # Single guillemets: ‹ ›
    + "\u00b7\u2022\u2023"         # Middle dot, bullet, triangle bullet: · • ‣
)

# Verb/adverb suffixes for auto-fact subject heuristic
_VERB_SUFFIXES: tuple[str, ...] = ("ing", "ed", "ly")


def _strip_punctuation(token: str) -> str:
    """
    RG coarse-graining: strip leading and trailing punctuation.

    "Rome," → "Rome", "(hello)" → "hello", "...wow!!!" → "wow"
    Internal punctuation preserved: "don't" → "don't", "well-known" → "well-known"

    Returns empty string if ALL characters are punctuation.
    O(1) per token.
    """
    start, end = 0, len(token)
    while start < end and token[start] in _PUNCTUATION_CHARS:
        start += 1
    while end > start and token[end - 1] in _PUNCTUATION_CHARS:
        end -= 1
    return token[start:end] if start < end else ""


def _normalize_token(token: str) -> str:
    """
    Minimal token normalization: strip possessives, trailing 'ing' and 's'.

    NOT a stemmer. Rules with minimum result length guard.
    The engine indexes BOTH raw and normalized forms, so imperfect
    normalization is acceptable — raw form provides exact match fallback.

    Result must be >= 4 chars after -ing stripping to prevent destructive
    normalization ("string" → "str" is BLOCKED, "visiting" → "visit" is OK).

    Possessives: "alice's" → "alice", "dogs'" → "dog" (stripped then -s rule)
    'visiting' → 'visit', 'eats' → 'eat', 'bananas' → 'banana',
    'running' → 'runn', 'string' → 'string' (preserved — "str" too short)
    """
    t = token.lower()
    # Possessive stripping: "alice's" → "alice", "dogs'" → "dogs" (then -s rule)
    if t.endswith("'s") and len(t) > 3:
        t = t[:-2]
    elif t.endswith("'") and len(t) > 3:
        t = t[:-1]
    if len(t) > 4 and t.endswith("ing"):
        candidate = t[:-3]
        if len(candidate) >= 4:
            return candidate
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]  # "eats"→"eat", "bananas"→"banana". Known: "bias"→"bia" (raw form fallback)
    return t


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text into index-ready tokens.

    SRG Step 1 (coarse-graining): strip punctuation from each word before
    stop-word check and normalization. "Rome," → "rome", not "rome,".

    Returns a list of tokens ordered by estimated informativeness
    (longer/rarer tokens first). Both raw and normalized forms included.
    Stop words and single-character tokens filtered.

    The ordering matters for field radius: when R(s) contracts, common
    tokens (at the end) are de-indexed first, preserving discriminating tokens.
    """
    raw_tokens: list[str] = []
    seen: set[str] = set()

    for word in text.lower().split():
        # RG coarse-graining: strip punctuation (UV noise)
        clean = _strip_punctuation(word)
        if not clean or clean in _STOP_WORDS or len(clean) <= 1:
            continue
        if clean not in seen:
            raw_tokens.append(clean)
            seen.add(clean)
        normalized = _normalize_token(clean)
        if normalized != clean and normalized not in seen and len(normalized) > 1:
            raw_tokens.append(normalized)
            seen.add(normalized)

    # Sort by length descending — longer tokens are more specific/informative
    # This is a cheap proxy for IDF before we have corpus statistics
    raw_tokens.sort(key=len, reverse=True)
    return raw_tokens


def _has_override(text: str) -> bool:
    """Detect override signals in raw text. Pure pattern matching.
    SRG: strip punctuation before signal check so "only," matches "only".
    """
    words = set(_strip_punctuation(w) for w in text.lower().split())
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
class SchemaMeta:
    """
    Metadata for a crystallized schema (solid/glass phase).

    When ΔF(G) = F_schema − Σ F_liquid(i) + C_abstraction < 0,
    a group of episodic memories crystallize into a schema.

    The schema IS the RG soft fixed point: tokens surviving
    multi-scale coarse-graining across ≥80% of group members.
    """

    member_ids: tuple[str, ...]             # Episodic item IDs that crystallized
    fixed_point_tokens: tuple[str, ...]     # Φ* — RG soft fixed point
    H_schema: float                         # Shannon entropy of schema content
    H_sum_episodes: float                   # Σ Hᵢ at formation
    delta_F: float                          # ΔF at crystallization (negative)
    formation_order: int                    # Event counter at crystallization
    absorption_count: int = 0               # Post-formation absorptions
    H_history: tuple[float, ...] = ()       # H after each absorption (glass detector)


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

    # --- Dense Semantic Embedding (384-dim, optional — set by MemoryService) ---
    # PhaseMemoryEngine itself remains zero-dep. MemoryService attaches this
    # after engine.store() returns, using EmbeddingService (sentence-transformers).
    # Used for post-TRR 384-dim semantic re-ranking (bridges vocabulary gaps).
    embedding_dense: list = field(default_factory=list)

    # --- Conversation context (set by benchmark/API callers, optional) ---
    session_date: str = ""        # Human-readable session date ("8 May 2023")
    conversation_turn: int = 0    # Turn index within its conversation

    # --- Crystallization (Liquid → Solid phase transition) ---
    schema_meta: Optional[SchemaMeta] = None  # Non-None = solid/glass phase

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
            "phase": (
                ("glass" if _is_glass_static(self) else "solid")
                if self.schema_meta is not None
                else ("liquid" if s >= strength_floor else "gas")
            ),
        }


# =============================================================================
# Glass Detection — Static Helper
# =============================================================================


def _is_glass_static(item: PhaseMemoryItem) -> bool:
    """Detect glass phase: schema entropy has converged (H stops changing)."""
    if item.schema_meta is None:
        return False
    history = item.schema_meta.H_history
    if len(history) < 4:  # Need initial + 3 absorptions
        return False
    last_3 = history[-3:]
    mean_h = sum(last_3) / 3.0
    if mean_h < 1e-9:
        return True  # Trivially converged (near-zero entropy)
    variance = sum((h - mean_h) ** 2 for h in last_3) / 3.0
    return math.sqrt(variance) / mean_h < 0.01  # 1% relative std


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
        # INTENTIONAL DESIGN: in-memory cap is 1,000 hot items per namespace.
        # The thermodynamic pressure model (density ρ = active/capacity) is
        # calibrated for this ceiling. Items beyond 1,000 are served via L1
        # kNN vector search (PostgreSQL + IVFFlat) — do NOT raise this limit.
        self.CAPACITY: int = capacity
        self.BETA_RETRIEVAL: float = beta_retrieval

        # --- Crystallization Constants (Liquid → Solid) ---
        self.TAU_SCHEMA: float = self.TAU_OVERRIDE * 2.0   # 400 — crystalline stability
        self.SCHEMA_ABSORPTION_COVERAGE: float = 0.6        # 60% token match to absorb
        self.RG_SOFT_THRESHOLD: float = 0.8                  # 80% member coverage
        self.GLASS_CONVERGENCE: float = 0.01                 # 1% relative std
        self.MIN_FIXED_POINT_TOKENS: int = 2                 # ≥ 2 tokens for schema
        self.MIN_GROUP_SIZE: int = 3                          # ≥ 3 episodes to crystallize

        # --- TRR Constants (Thermodynamic Resonance Retrieval) ---
        self._SVD_RECOMPUTE_INTERVAL: int = 50   # Recompute SVD every N stores
        self._SVD_DIMS: int = 50                  # Embedding dimensionality
        self._MORPH_PREFIX_LEN: int = 4           # Minimum prefix for morphological matching

        # State
        self._items: dict[str, list[PhaseMemoryItem]] = {}
        self._event_counter: int = 0
        self._total_item_count: int = 0  # Cached for O(1) IDF computation
        self._item_by_id: dict[str, PhaseMemoryItem] = {}  # O(1) lookup for PESQD

        # Token Index — Thermodynamic Semantic Field (TSF)
        # Single token index: token → list of PhaseMemoryItems
        # Entries phase-transition in/out based on R(s) = floor(N × s^(1/3))
        self._token_index: dict[str, list[PhaseMemoryItem]] = {}

        # Morphological Kernel — prefix equivalence classes for query-time expansion
        # prefix4 → set of indexed tokens sharing that prefix
        self._prefix_index: dict[str, set[str]] = {}

        # Document frequency for IDF computation (self-computed from corpus)
        self._doc_freq: Counter = Counter()

        # PPMI Co-occurrence — token pairs co-occurring in same item
        self._cooccurrence: Counter = Counter()
        self._svd_store_count: int = 0
        self._svd_dirty: bool = False
        self._token_vectors: dict[str, list[float]] = {}

        # --- Cross-Entity Resonance (CER) — Kuramoto Coupled Oscillators ---
        self._entity_nodes: dict[str, EntityNode] = {}
        self._entity_alias_map: dict[str, str] = {}
        self._entanglement_graph: dict[str, dict[str, EntanglementEdge]] = {}
        self._resonance_clusters: dict[str, ResonanceCluster] = {}
        self._entity_index: dict[str, list[str]] = {}  # token → entity names
        self._compound_entity_index: dict[str, list[tuple[str, int]]] = {}  # first_word → [(full_name, n_parts)]
        self._K_critical: float = 0.15  # Synchronization phase transition threshold

        # --- Benchmark Mode — Suppress crystallization/GC for full recall ---
        self._benchmark_mode: bool = False

        # --- Episode Archive — preserves original episodes after crystallization ---
        # Schemas are the primary search target; archive provides detail drill-down
        self._episode_archive: dict[str, list[PhaseMemoryItem]] = {}  # schema_id → [original episodes]

        # --- Contradiction Log — Chain-of-Thought, never override ---
        self._contradiction_log: list[dict[str, Any]] = []

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
        Liquid memories (s→1) have broad fields; gas memories get radius=1
        (fresh memories are vivid — the kid who just walked in is still visible).

        Tokens are ordered by informativeness (longer first, which correlates
        with rarity). As s decays, common tokens are de-indexed first,
        preserving the most discriminating tokens longest.
        """
        s = item.consolidation_strength
        tokens = item.indexed_tokens
        n_tokens = len(tokens)

        if n_tokens == 0:
            return

        if s < self.STRENGTH_FLOOR:
            # Gas phase: minimal field radius=1 (still searchable, still vivid)
            radius = 1
        else:
            cube_root_s = s ** (1.0 / 3.0)
            radius = max(1, int(n_tokens * cube_root_s))

        # Index tokens within radius + update prefix index
        for token in tokens[:radius]:
            if token not in self._token_index:
                self._token_index[token] = []
            if item not in self._token_index[token]:
                self._token_index[token].append(item)
            # Maintain prefix index for morphological kernel
            if len(token) >= self._MORPH_PREFIX_LEN:
                prefix = token[:self._MORPH_PREFIX_LEN]
                self._prefix_index.setdefault(prefix, set()).add(token)

        # De-index tokens beyond radius
        for token in tokens[radius:]:
            if token in self._token_index and item in self._token_index[token]:
                self._token_index[token].remove(item)
                if not self._token_index[token]:
                    del self._token_index[token]
                    # Clean prefix index if token fully removed
                    if len(token) >= self._MORPH_PREFIX_LEN:
                        prefix = token[:self._MORPH_PREFIX_LEN]
                        if prefix in self._prefix_index:
                            self._prefix_index[prefix].discard(token)
                            if not self._prefix_index[prefix]:
                                del self._prefix_index[prefix]

        item._last_field_radius = radius

    def _deindex_item(self, item: PhaseMemoryItem) -> None:
        """Remove a memory item from the token index entirely."""
        for token in item.indexed_tokens:
            if token in self._token_index and item in self._token_index[token]:
                self._token_index[token].remove(item)
                if not self._token_index[token]:
                    del self._token_index[token]
                    # Clean prefix index if token fully removed
                    if len(token) >= self._MORPH_PREFIX_LEN:
                        prefix = token[:self._MORPH_PREFIX_LEN]
                        if prefix in self._prefix_index:
                            self._prefix_index[prefix].discard(token)
                            if not self._prefix_index[prefix]:
                                del self._prefix_index[prefix]

    def _update_field_radius(self, item: PhaseMemoryItem) -> None:
        """Lazy field radius update. Only re-indexes if R(s) has changed."""
        s = item.consolidation_strength
        n_tokens = len(item.indexed_tokens)
        if n_tokens == 0:
            return

        if s < self.STRENGTH_FLOOR:
            new_radius = 1  # Gas: minimal but present
        else:
            cube_root_s = s ** (1.0 / 3.0)
            new_radius = max(1, int(n_tokens * cube_root_s))

        if new_radius != item._last_field_radius:
            self._index_item(item)

    @staticmethod
    def _safe_fe(item: PhaseMemoryItem) -> float:
        """Guard: return 0.0 for NaN/inf free energy."""
        fe = item.free_energy
        if math.isnan(fe) or math.isinf(fe):
            return 0.0
        return fe

    def _compute_idf(self, token: str) -> float:
        """
        Compute Inverse Document Frequency for a token.

        idf(t) = log(1 + N / (1 + df(t)))

        Self-computed from the engine's own corpus. Zero external knowledge.
        N = total items across all namespaces.
        df(t) = number of items containing token t.
        """
        df = self._doc_freq.get(token, 0)
        return math.log(1.0 + self._total_item_count / (1.0 + df))

    # =========================================================================
    # TRR — Thermodynamic Resonance Retrieval
    #
    # Self-tuning retrieval: all parameters derived from corpus statistics.
    # Zero LLM. Zero external dependencies. Zero manual tuning.
    #
    # Layers:
    #   1. Morphological Kernel — prefix equivalence (query-time)
    #   2. BMX Scoring — entropy-weighted BM25
    #   3. Phase-Dependent Susceptibility — Gas/Liquid/Solid/Glass
    #   4. PPMI Co-occurrence — token coupling constants
    #   5. PPMI-SVD — local embeddings (background recompute)
    #   6. Schema-Aware Query Expansion
    # =========================================================================

    def _morph_expand(self, token: str) -> list[str]:
        """
        Morphological Kernel: expand a query token to all indexed variants
        sharing a ≥4-char prefix.

        "move" → ["move", "moved", "moving", "movement"]

        Uses _prefix_index for O(1) lookup. No suffix rules.
        Universal: works for any language with prefixed morphology.
        """
        if len(token) < self._MORPH_PREFIX_LEN:
            return [token] if token in self._token_index else []
        prefix = token[:self._MORPH_PREFIX_LEN]
        variants = self._prefix_index.get(prefix, set())
        if not variants:
            return [token] if token in self._token_index else []
        # Return variants that are actually in the token index
        result = [v for v in variants if v in self._token_index]
        # Ensure original token is included if indexed
        if token in self._token_index and token not in result:
            result.append(token)
        return result

    def _compute_entropy_weight(self, token: str) -> float:
        """
        BMX entropy weight: penalizes tokens distributed uniformly across items,
        boosts tokens that cluster in specific items.

        H_weight = max(0.1, 1.0 - H_binary(p))
        where p = df(token) / N.

        Clustered token (low entropy) → high weight → more informative.
        Uniform token (high entropy) → low weight → less discriminating.
        """
        df = self._doc_freq.get(token, 0)
        N = max(self._total_item_count, 1)
        if df == 0 or df == N:
            return 1.0
        p = df / N
        # Binary entropy of the token's distribution
        H = -p * math.log2(max(p, 1e-15)) - (1.0 - p) * math.log2(max(1.0 - p, 1e-15))
        return max(0.1, 1.0 - H)

    def _phase_susceptibility(self, item: PhaseMemoryItem) -> float:
        """
        Phase-dependent susceptibility χ(phase).

        Gas:    χ = 0.7  (fresh, vivid, slightly lower trust)
        Liquid: χ = 1.0  (normal)
        Solid:  χ = 1.0 + 0.5 / max(|ΔF|, 0.1)
        Glass:  χ = 1.0 + 1.0 / max(|ΔF|, 0.1)

        Continuous, physics-derived. Replaces hard-coded 1.5× schema boost.
        """
        s = item.consolidation_strength

        if item.schema_meta is not None:
            delta_F = abs(item.schema_meta.delta_F)
            if _is_glass_static(item):
                return 1.0 + 1.0 / max(delta_F, 0.1)
            else:
                return 1.0 + 0.5 / max(delta_F, 0.1)

        if s < self.STRENGTH_FLOOR:
            return 0.7  # Gas

        return 1.0  # Liquid

    def _compute_pmi(self, token_a: str, token_b: str) -> float:
        """
        Pointwise Mutual Information from co-occurrence statistics.

        PMI(a,b) = log(N × co(a,b) / (df(a) × df(b)))
        Clamped to PPMI (≥ 0).

        Self-computed from the engine's own corpus. Zero external knowledge.
        """
        a, b = (token_a, token_b) if token_a < token_b else (token_b, token_a)
        co = self._cooccurrence.get((a, b), 0)
        if co == 0:
            return 0.0
        df_a = self._doc_freq.get(token_a, 0)
        df_b = self._doc_freq.get(token_b, 0)
        if df_a == 0 or df_b == 0:
            return 0.0
        N = max(self._total_item_count, 1)
        pmi = math.log(N * co / (df_a * df_b))
        return max(0.0, pmi)  # PPMI: clamp negative

    def _recompute_svd(self) -> None:
        """
        Recompute PPMI-SVD token vectors from co-occurrence matrix.

        Pure Python. Zero external dependencies (no numpy, no scipy).
        Uses randomized power iteration for truncated SVD.

        Output: self._token_vectors — per-token vectors for semantic matching.
        Mathematically equivalent to word2vec (Levy & Goldberg 2014).
        """
        if not self._cooccurrence:
            return

        # Build vocabulary from tokens with sufficient document frequency
        vocab: set[str] = set()
        for (a, b) in self._cooccurrence:
            if self._doc_freq.get(a, 0) >= 2:
                vocab.add(a)
            if self._doc_freq.get(b, 0) >= 2:
                vocab.add(b)

        # Cap vocabulary at top 5000 by document frequency
        if len(vocab) > 5000:
            vocab_sorted = sorted(vocab, key=lambda t: self._doc_freq.get(t, 0), reverse=True)
            vocab = set(vocab_sorted[:5000])

        vocab_list = sorted(vocab)
        V = len(vocab_list)
        if V < 3:
            return

        tok2idx = {t: i for i, t in enumerate(vocab_list)}
        dims = min(self._SVD_DIMS, V - 1)

        # Build sparse PPMI matrix as dict-of-dicts
        ppmi_rows: dict[int, dict[int, float]] = {}
        N = max(self._total_item_count, 1)
        for (a, b), co in self._cooccurrence.items():
            if a not in tok2idx or b not in tok2idx:
                continue
            df_a = self._doc_freq.get(a, 0)
            df_b = self._doc_freq.get(b, 0)
            if df_a == 0 or df_b == 0:
                continue
            pmi_val = math.log(N * co / (df_a * df_b))
            if pmi_val <= 0:
                continue
            i, j = tok2idx[a], tok2idx[b]
            ppmi_rows.setdefault(i, {})[j] = pmi_val
            ppmi_rows.setdefault(j, {})[i] = pmi_val

        if not ppmi_rows:
            return

        # Sparse matrix-vector multiply: y = PPMI @ x
        def spmv(x: list[float]) -> list[float]:
            y = [0.0] * V
            for i, row in ppmi_rows.items():
                for j, val in row.items():
                    y[i] += val * x[j]
            return y

        # Power iteration for top-k approximate singular vectors
        import random as _rng
        _rng.seed(42)  # Reproducible

        # Initialize random basis
        basis: list[list[float]] = []
        for _ in range(dims):
            vec = [_rng.gauss(0, 1) for _ in range(V)]
            mag = math.sqrt(sum(v * v for v in vec)) or 1.0
            basis.append([v / mag for v in vec])

        # 15 power iterations
        for _iter in range(15):
            # Multiply each basis vector by PPMI
            new_basis = [spmv(vec) for vec in basis]

            # Modified Gram-Schmidt orthogonalization
            for i in range(len(new_basis)):
                for j in range(i):
                    dot = sum(a * b for a, b in zip(new_basis[i], new_basis[j]))
                    new_basis[i] = [a - dot * b for a, b in zip(new_basis[i], new_basis[j])]
                mag = math.sqrt(sum(v * v for v in new_basis[i])) or 1e-12
                new_basis[i] = [v / mag for v in new_basis[i]]

            basis = new_basis

        # Extract token vectors: each token gets a dims-dimensional vector
        self._token_vectors = {}
        for idx, token in enumerate(vocab_list):
            vec = [basis[d][idx] for d in range(dims)]
            # Scale by approximate singular value (Rayleigh quotient)
            Av = [0.0] * V
            for j, val in ppmi_rows.get(idx, {}).items():
                for d in range(dims):
                    Av[d] = Av[d]  # skip full computation for speed
            self._token_vectors[token] = vec

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Cosine similarity between two vectors. Pure Python."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a < 1e-12 or mag_b < 1e-12:
            return 0.0
        return dot / (mag_a * mag_b)

    def _mean_vector(self, tokens: list[str]) -> Optional[list[float]]:
        """Average vector of tokens that have embeddings. None if no vectors."""
        vecs = [self._token_vectors[t] for t in tokens if t in self._token_vectors]
        if not vecs:
            return None
        dims = len(vecs[0])
        avg = [0.0] * dims
        for vec in vecs:
            for d in range(dims):
                avg[d] += vec[d]
        n = len(vecs)
        return [v / n for v in avg]

    def _expand_query_with_schemas(
        self,
        query_tokens: list[str],
        query_token_set: set[str],
        namespace: str,
    ) -> tuple[list[str], set[str], set[str]]:
        """
        Schema-Aware Query Expansion.

        When query matches a known entity with a schema, expand with the
        schema's fixed_point_tokens. The phase engine feeds retrieval.

        Expanded tokens scored at 0.5× weight (inferred, not queried).
        """
        inferred: set[str] = set()

        for token in list(query_token_set):
            # Check if token is a known entity
            canonical = token
            if token in self._entity_alias_map:
                canonical = self._entity_alias_map[token]
            if canonical not in self._entity_nodes:
                continue

            node = self._entity_nodes[canonical]
            # Find schemas this entity participates in
            for mid in node.memory_ids:
                item = self._item_by_id.get(mid)
                if (item and item.namespace == namespace
                        and item.schema_meta is not None):
                    for fp_token in item.schema_meta.fixed_point_tokens[:3]:
                        if (fp_token not in query_token_set
                                and fp_token not in _STOP_WORDS
                                and len(fp_token) > 1):
                            inferred.add(fp_token)

        if inferred:
            expanded_tokens = query_tokens + list(inferred)
            expanded_token_set = query_token_set | inferred
            return expanded_tokens, expanded_token_set, inferred

        return query_tokens, query_token_set, set()

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

        if shared_ratio > 0.8:
            return "confirmation", 0.0
        if shared_ratio > 0.25:
            # Jaccard distance scaled to nats so sigmoid sharpening works
            SIGMA_MAX = -math.log(1e-6)
            surprise = (1.0 - shared_ratio) * SIGMA_MAX
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
                    # Scale bigram divergence [0,1] to nats so sigmoid
                    # sharpening in _apply_surprise_damage works correctly.
                    # Without this, non-override sigma_norm ≈ 0.07 → near-zero damage.
                    SIGMA_MAX = -math.log(1e-6)
                    sigma = self._bigram_divergence(new_fact.value, item.fact.value) * SIGMA_MAX

                max_surprise = max(max_surprise, sigma)

        return max_surprise, contradicted

    def _compute_surprise_from_tokens(
        self,
        new_text: str,
        new_tokens: set[str],
        namespace: str,
    ) -> tuple[float, list[PhaseMemoryItem], PhaseMemoryItem | None]:
        """
        Compute surprise from raw text using token overlap.

        Used when storing raw text without structured (S, R, V) facts.
        Detects contradictions by finding existing memories with high
        token overlap but different content.

        Returns:
            (max_surprise, contradicted_items, confirmed_item_or_None)
            If confirmed_item is not None, the caller should return it
            instead of creating a new item (token-level dedup).
        """
        existing = self._items.get(namespace, [])
        contradicted: list[PhaseMemoryItem] = []
        max_surprise = 0.0
        is_override = _has_override(new_text)
        confirmed: PhaseMemoryItem | None = None

        for item in existing:
            if item.consolidation_strength < self.STRENGTH_FLOOR:
                continue

            result, surprise = self._detect_contradiction(new_tokens, item)

            if result == "confirmation":
                # High overlap = likely same content, reinforce
                item.retrieval_count += 1
                if confirmed is None:
                    confirmed = item
            elif result == "contradiction":
                contradicted.append(item)
                if is_override:
                    surprise = -math.log(1e-6)  # Override amplification
                max_surprise = max(max_surprise, surprise)

        return max_surprise, contradicted, confirmed

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
            # In benchmark mode: log contradiction but DON'T damage.
            # Keep both facts alive — chain-of-thought lets user decide.
            if self._benchmark_mode:
                self._contradiction_log.append({
                    "old_item_id": item.id,
                    "old_text": item.fact.raw_text if item.fact else "",
                    "new_text": new_fact.raw_text if new_fact else "",
                    "surprise": surprise,
                })
                continue

            tau_ratio = tau_new / max(item.tau, 1e-6)
            tau_factor = min(tau_ratio, 4.0) / 4.0 + 0.5
            damage = sigmoid_damage * tau_factor * amplifier
            # Solid/glass phase resistance
            if item.schema_meta is not None:
                resistance = 1.0 / (1.0 + abs(item.schema_meta.delta_F))
                if _is_glass_static(item):
                    resistance *= 0.1  # Glass: 10× more resistant
                damage *= resistance
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
        """Apply surprise damage for token-based contradiction detection.

        Same physics as _apply_surprise_damage: D = σ(Σ_norm) · τ_factor · amplifier.
        The tau_ratio ensures ephemeral notes can't destroy override memories.
        """
        SIGMA_MAX = -math.log(1e-6)
        sigma_norm = min(surprise / SIGMA_MAX, 1.0)

        sigmoid_damage = 1.0 / (1.0 + math.exp(-10.0 * (sigma_norm - 0.5)))
        amplifier = 1.5 if is_override else 1.0
        tau_new = self.TAU_OVERRIDE if is_override else self.TAU_DEFAULT

        for item in contradicted:
            # In benchmark mode: log but don't damage (same as structured path)
            if self._benchmark_mode:
                self._contradiction_log.append({
                    "old_item_id": item.id,
                    "old_text": item.fact.raw_text if item.fact else "",
                    "new_text": "(token-overlap contradiction)",
                    "surprise": surprise,
                })
                continue

            tau_ratio = tau_new / max(item.tau, 1e-6)
            tau_factor = min(tau_ratio, 4.0) / 4.0 + 0.5
            damage = sigmoid_damage * tau_factor * amplifier
            # Solid/glass phase resistance
            if item.schema_meta is not None:
                resistance = 1.0 / (1.0 + abs(item.schema_meta.delta_F))
                if _is_glass_static(item):
                    resistance *= 0.1  # Glass: 10× more resistant
                damage *= resistance
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
        1. Words starting with uppercase that are NOT the very first word
        2. Multi-word entities via consecutive capitals: "New York" → "new york"
        3. Punctuation stripped (RG coarse-graining) before capitalization check

        Post-period capitals ARE entities: "Hello. Jean likes pizza" → "jean".
        Stop words filter out false positives ("The", "He", "She" etc.).

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

            # RG coarse-graining: strip punctuation before capitalization check
            clean = _strip_punctuation(word)
            if not clean or len(clean) <= 1:
                i += 1
                continue

            # Only skip the very first word (sentence-initial).
            # Post-period capitals ARE entities — stop words filter false positives.
            is_sentence_start = (i == 0)

            if (
                not is_sentence_start
                and clean[0].isupper()
                and clean.lower() not in _STOP_WORDS
            ):
                # Check for multi-word entity (consecutive capitals)
                entity_parts = [clean]
                j = i + 1
                while j < len(words):
                    next_clean = _strip_punctuation(words[j])
                    if (
                        next_clean
                        and next_clean[0].isupper()
                        and next_clean.lower() not in _STOP_WORDS
                        and len(next_clean) > 1
                    ):
                        entity_parts.append(next_clean)
                        j += 1
                    else:
                        break
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

        # Build entity token filter: compound entity parts excluded from spectrum
        # e.g. "jean paul" → {"jean paul", "jean", "paul"}
        all_entity_tokens: set[str] = set()
        for e in entities:
            all_entity_tokens.add(e)
            all_entity_tokens.update(e.split())

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
            # Exclude compound entity parts (e.g. "jean" from "jean paul")
            for token, idf in tokens_with_idf.items():
                if token not in all_entity_tokens:
                    node.token_spectrum[token] = node.token_spectrum.get(token, 0.0) + idf

            # Update entity index (token → entity names)
            for token in item.indexed_tokens:
                if token not in self._entity_index:
                    self._entity_index[token] = []
                if entity_name not in self._entity_index[token]:
                    self._entity_index[token].append(entity_name)

            # Update compound entity index for multi-word entities
            parts = entity_name.split()
            if len(parts) > 1:
                first_word = parts[0]
                self._compound_entity_index.setdefault(first_word, [])
                entry = (entity_name, len(parts))
                if entry not in self._compound_entity_index[first_word]:
                    self._compound_entity_index[first_word].append(entry)

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
            # Clean compound entity index
            parts = entity_name.split()
            if len(parts) > 1 and parts[0] in self._compound_entity_index:
                self._compound_entity_index[parts[0]] = [
                    entry for entry in self._compound_entity_index[parts[0]]
                    if entry[0] != entity_name
                ]
                if not self._compound_entity_index[parts[0]]:
                    del self._compound_entity_index[parts[0]]

        # Clean shared_memory_ids from entanglement edges
        for edges in self._entanglement_graph.values():
            for edge in edges.values():
                if item_id in edge.shared_memory_ids:
                    edge.shared_memory_ids.remove(item_id)

        # Clean entanglement graph keys for removed entities
        for entity_name in entities_to_remove:
            # Remove as primary key
            if entity_name in self._entanglement_graph:
                del self._entanglement_graph[entity_name]
            # Remove as secondary key in other entities' edges
            for other_edges in self._entanglement_graph.values():
                if entity_name in other_edges:
                    del other_edges[entity_name]

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
    # Liquid → Solid: Landauer Crystallization Engine
    #
    # ΔF(G) = F_schema − Σ F_liquid(i) + C_abstraction
    # When ΔF < 0: crystallize. ONE threshold. Thermodynamically derived.
    #
    # Hysteresis: melt only when ΔF > F_melt = kT·ln(2)·H_lost
    # Glass: when schema entropy converges (std(H_history)/mean < 1%)
    # =========================================================================

    def _find_crystallization_candidates(
        self, namespace: str,
    ) -> list[list[PhaseMemoryItem]]:
        """
        Find groups of liquid episodic memories that might crystallize.

        Two grouping mechanisms:
        1. Entity nodes — memories sharing a named entity
        2. High-IDF token co-occurrence — catches non-entity patterns

        Returns deduplicated candidate groups. O(E + T).
        """
        seen_group_keys: set[frozenset[str]] = set()
        candidates: list[list[PhaseMemoryItem]] = []

        # 1. Groups from entity nodes
        for _entity_name, node in self._entity_nodes.items():
            members: list[PhaseMemoryItem] = []
            for mid in node.memory_ids:
                item = self._item_by_id.get(mid)
                if (item is not None
                        and item.namespace == namespace
                        and item.consolidation_strength >= self.STRENGTH_FLOOR
                        and item.schema_meta is None):
                    members.append(item)
            if len(members) >= self.MIN_GROUP_SIZE:
                key = frozenset(m.id for m in members)
                if key not in seen_group_keys:
                    seen_group_keys.add(key)
                    candidates.append(members)

        # 2. High-IDF token co-occurrence
        for token, token_items in self._token_index.items():
            idf = self._compute_idf(token)
            if idf < 1.0:
                continue  # Skip common tokens
            ns_items = [
                i for i in token_items
                if i.namespace == namespace
                and i.schema_meta is None
                and i.consolidation_strength >= self.STRENGTH_FLOOR
            ]
            if len(ns_items) >= self.MIN_GROUP_SIZE:
                key = frozenset(i.id for i in ns_items)
                if key not in seen_group_keys:
                    seen_group_keys.add(key)
                    candidates.append(ns_items)

        return candidates

    def _compute_fixed_point(
        self, group: list[PhaseMemoryItem],
    ) -> tuple[list[str], dict[str, float]]:
        """
        RG soft fixed point: tokens in ≥ 80% of group members,
        ordered by MI contribution (IDF² × coverage).

        The fixed point IS the schema content.
        """
        N = len(group)
        token_member_count: Counter = Counter()
        for item in group:
            for token in set(item.indexed_tokens):
                token_member_count[token] += 1

        fixed: dict[str, float] = {}
        for token, count in token_member_count.items():
            coverage = count / N
            if coverage >= self.RG_SOFT_THRESHOLD:
                idf = self._compute_idf(token)
                fixed[token] = idf * idf * coverage  # MI contribution

        ordered = sorted(fixed.keys(), key=lambda t: fixed[t], reverse=True)
        return ordered, fixed

    def _build_schema_fact(
        self,
        fixed_tokens: list[str],
        group: list[PhaseMemoryItem],
        entity_name: str = "",
    ) -> "Fact":
        """Build a Fact from the RG fixed point. Zero LLM."""
        subject = entity_name if entity_name else (
            group[0].fact.subject if group else ""
        )
        value = " ".join(fixed_tokens[:20])
        raw_text = f"[Schema: {subject}] {value}"
        return Fact(
            subject=subject,
            relation="schema",
            value=value,
            override=False,
            raw_text=raw_text,
        )

    def _find_dominant_entity(self, group: list[PhaseMemoryItem]) -> str:
        """Find the entity appearing in the most group members."""
        entity_counts: Counter = Counter()
        for item in group:
            for entity_name, node in self._entity_nodes.items():
                if item.id in node.memory_ids:
                    entity_counts[entity_name] += 1
        if entity_counts:
            return entity_counts.most_common(1)[0][0]
        return group[0].fact.subject if group else ""

    def _compute_delta_F(
        self,
        group: list[PhaseMemoryItem],
        fixed_tokens: list[str],
        token_weights: dict[str, float],
    ) -> tuple[float, float, float]:
        """
        Compute the free energy of crystallization.

        ΔF = F_schema − Σ F_liquid(i) + C_abstraction

        Returns (delta_F, H_schema, H_sum_episodes).
        """
        # Sum of individual Landauer costs
        sum_F_liquid = sum(item.landauer_cost for item in group)

        # Schema entropy
        entity_name = self._find_dominant_entity(group)
        schema_fact = self._build_schema_fact(fixed_tokens, group, entity_name)
        H_schema = self._information_content(schema_fact)

        # Schema Landauer cost
        F_schema = self.kT * math.log(2) * H_schema / max(self.TAU_SCHEMA, 1e-6)

        # Mutual information preserved in fixed point
        MI_shared = sum(token_weights.values())

        # Information lost during coarse-graining
        H_sum = sum(item.information_content_bits for item in group)
        H_lost = max(0.0, H_sum - H_schema - MI_shared)

        # Abstraction cost (surface energy)
        C_abs = self.kT * math.log(2) * H_lost / max(self.TAU_SCHEMA, 1e-6)

        # Density penalty: high density = more conservative crystallization
        # At ρ > 0.5, require increasingly negative ΔF to crystallize
        rho = len(group) / max(self.CAPACITY, 1)
        density_penalty = max(0.0, rho - 0.5) * self.kT * math.log(2)

        delta_F = F_schema - sum_F_liquid + C_abs + density_penalty
        return delta_F, H_schema, H_sum

    def _crystallize(
        self,
        group: list[PhaseMemoryItem],
        fixed_tokens: list[str],
        token_weights: dict[str, float],
        delta_F: float,
        H_schema: float,
        H_sum: float,
        namespace: str,
    ) -> PhaseMemoryItem:
        """
        Create a schema item from a crystallizing group of episodes.

        First-order transition: schema appears discontinuously.
        Constituent episodes are set to sub-critical τ → evaporate naturally.
        """
        entity_name = self._find_dominant_entity(group)
        schema_fact = self._build_schema_fact(fixed_tokens, group, entity_name)

        total_retrievals = sum(item.retrieval_count for item in group)
        max_surprise = max(
            (item.surprise_at_birth for item in group), default=0.0,
        )

        H = self._information_content(schema_fact)
        L = (self.kT * math.log(2) * H) / max(self.TAU_SCHEMA, 1e-6)

        schema_meta = SchemaMeta(
            member_ids=tuple(item.id for item in group),
            fixed_point_tokens=tuple(fixed_tokens),
            H_schema=H_schema,
            H_sum_episodes=H_sum,
            delta_F=delta_F,
            formation_order=self._event_counter,
            absorption_count=0,
            H_history=(H_schema,),
        )

        schema_item = PhaseMemoryItem(
            id=str(uuid4()),
            fact=schema_fact,
            namespace=namespace,
            consolidation_strength=1.0,
            surprise_at_birth=max_surprise,
            tau=self.TAU_SCHEMA,
            birth_order=self._event_counter,
            rho_at_birth=self._memory_density(namespace),
            free_energy=0.0,
            retrieval_count=total_retrievals,
            accumulated_surprise_damage=0.0,
            information_content_bits=H,
            landauer_cost=L,
            indexed_tokens=list(fixed_tokens),
            schema_meta=schema_meta,
        )

        # Index the schema
        self._items.setdefault(namespace, []).append(schema_item)
        self._total_item_count += 1
        self._item_by_id[schema_item.id] = schema_item
        self._index_item(schema_item)

        for token in set(fixed_tokens):
            self._doc_freq[token] += 1

        # Archive constituent episodes for detail retrieval before evaporation
        self._episode_archive[schema_item.id] = list(group)

        # Set constituent episodes to sub-critical τ → evaporate naturally
        for item in group:
            item.tau = self.TAU_C1 * 0.5  # Below critical → gas phase

        return schema_item

    def _check_crystallization(self, namespace: str) -> None:
        """
        Check if any group of liquid memories should crystallize.

        Called from _recompute_all_free_energies after GC.
        ONE threshold: ΔF < 0 (thermodynamically derived).
        """
        candidates = self._find_crystallization_candidates(namespace)
        for group in candidates:
            # Verify group members are still alive (not GC'd in this pass)
            alive_group = [
                item for item in group
                if item.id in self._item_by_id
                and item.consolidation_strength >= self.STRENGTH_FLOOR
                and item.schema_meta is None
            ]
            if len(alive_group) < self.MIN_GROUP_SIZE:
                continue

            # Cap group size to prevent mega-schemas that lose too much detail
            MAX_CRYSTALLIZE_GROUP = 10
            if len(alive_group) > MAX_CRYSTALLIZE_GROUP:
                alive_group = alive_group[:MAX_CRYSTALLIZE_GROUP]

            fixed_tokens, weights = self._compute_fixed_point(alive_group)
            if len(fixed_tokens) < self.MIN_FIXED_POINT_TOKENS:
                continue

            delta_F, H_schema, H_sum = self._compute_delta_F(
                alive_group, fixed_tokens, weights,
            )

            if delta_F < 0:  # THE threshold
                self._crystallize(
                    alive_group, fixed_tokens, weights,
                    delta_F, H_schema, H_sum, namespace,
                )

    def _try_schema_absorption(
        self, new_item: PhaseMemoryItem, namespace: str,
    ) -> bool:
        """
        Check if a new episode should be absorbed by an existing schema.

        Returns True if absorbed (schema reinforced, episode set to evaporate).
        """
        schemas = [
            item for item in self._items.get(namespace, [])
            if item.schema_meta is not None
            and item.consolidation_strength >= self.STRENGTH_FLOOR
        ]

        for schema in schemas:
            fp_tokens = set(schema.schema_meta.fixed_point_tokens)
            new_tokens = set(new_item.indexed_tokens)

            if not fp_tokens:
                continue

            overlap = len(fp_tokens & new_tokens)
            coverage = overlap / len(fp_tokens)

            if coverage >= self.SCHEMA_ABSORPTION_COVERAGE:
                # Archive the absorbed episode
                if schema.id not in self._episode_archive:
                    self._episode_archive[schema.id] = []
                self._episode_archive[schema.id].append(new_item)

                # Reinforce schema
                schema.retrieval_count += 1

                # Update H_history for glass detection
                old_meta = schema.schema_meta
                new_H = schema.information_content_bits  # unchanged content
                new_H_history = old_meta.H_history + (new_H,)

                schema.schema_meta = SchemaMeta(
                    member_ids=old_meta.member_ids + (new_item.id,),
                    fixed_point_tokens=old_meta.fixed_point_tokens,
                    H_schema=old_meta.H_schema,
                    H_sum_episodes=old_meta.H_sum_episodes + new_item.information_content_bits,
                    delta_F=old_meta.delta_F,
                    formation_order=old_meta.formation_order,
                    absorption_count=old_meta.absorption_count + 1,
                    H_history=new_H_history,
                )

                # Accelerate episode decay (absorbed into schema)
                new_item.tau = self.TAU_C1 * 0.5
                return True

        return False

    def _check_schema_melting(self, namespace: str) -> None:
        """
        Check if any schema should melt back to liquid.

        Hysteresis: melting requires ΔF > F_melt = kT·ln(2)·H_lost.
        Forming a schema erases constituent detail; re-creating it costs energy.
        """
        schemas = [
            item for item in self._items.get(namespace, [])
            if item.schema_meta is not None
            and item.consolidation_strength >= self.STRENGTH_FLOOR
        ]

        for schema in schemas:
            meta = schema.schema_meta

            # Count surviving members
            surviving = [
                self._item_by_id[mid]
                for mid in meta.member_ids
                if mid in self._item_by_id
                and self._item_by_id[mid].consolidation_strength >= self.STRENGTH_FLOOR
            ]

            if len(surviving) < 2:
                # Orphan schema — melt without hysteresis
                self._melt_schema(schema)
                continue

            # Recompute ΔF with current state
            fixed_tokens = list(meta.fixed_point_tokens)
            _, weights = self._compute_fixed_point(surviving)
            if not weights:
                self._melt_schema(schema)
                continue

            delta_F, _, _ = self._compute_delta_F(surviving, fixed_tokens, weights)

            # Landauer hysteresis barrier
            H_lost = max(meta.H_sum_episodes - meta.H_schema, 0.0)
            F_melt = self.kT * math.log(2) * H_lost

            if delta_F > F_melt:
                self._melt_schema(schema)

    def _melt_schema(self, schema: PhaseMemoryItem) -> None:
        """Dissolve a schema back into liquid-phase constituents."""
        if schema.schema_meta is None:
            return

        # Restore surviving constituent tau
        for mid in schema.schema_meta.member_ids:
            member = self._item_by_id.get(mid)
            if member is not None and member.consolidation_strength >= self.STRENGTH_FLOOR:
                member.tau = self.TAU_DEFAULT

        # Kill the schema (will be GC'd)
        schema.consolidation_strength = 0.0

    # =========================================================================
    # Store — Self-Sufficient Ingestion (Zero LLM)
    # =========================================================================

    def store(
        self,
        text: str,
        namespace: str,
        fact: Optional[Fact] = None,
        trace_id: Optional[str] = None,
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
        # Skip algo tracing in batch-load mode (startup replay)
        _trace = None if getattr(self, '_batch_mode', False) else trace_id

        # --- Tokenize (engine-internal) ---
        with _algo_span(_trace, "algo.tokenize", input=text[:120]) as _tok_hop:
            tokens = _tokenize(text)
            token_set = set(tokens)
            _add_algo_meta(_trace, _tok_hop, output=f"{len(tokens)} tokens: {' '.join(tokens[:10])}")

        # --- Build or use Fact ---
        _fact_provided = fact is not None
        with _algo_span(_trace, "algo.fact_build",
                        provided=_fact_provided, input=text[:100]) as _fact_hop:
            if fact is None:
                # Auto-create minimal fact from raw text
                # Subject: first non-stop token, relation: second, value: rest
                content_words = [
                    clean for w in text.lower().split()
                    for clean in [_strip_punctuation(w)]  # strip once, reuse
                    if clean and clean not in _STOP_WORDS and len(clean) > 1
                ]
                # Verb-skip heuristic: skip leading verbs/adverbs to find noun subject
                subject_idx = 0
                while subject_idx < len(content_words) - 1:
                    if any(content_words[subject_idx].endswith(sfx) for sfx in _VERB_SUFFIXES):
                        subject_idx += 1
                    else:
                        break
                # Reorder: subject first, then skipped words become relation context
                if subject_idx > 0:
                    content_words = content_words[subject_idx:] + content_words[:subject_idx]
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

            # --- Fact-field indexing: add value tokens for semantic retrieval ---
            # When value field contains content not in raw_text (e.g. LLM-extracted
            # structured facts), tokenize the value and add new tokens.
            # This enables TSF search to match queries against extracted values.
            if fact.value:
                value_tokens = _tokenize(fact.value)
                for ft in value_tokens:
                    if ft not in token_set:
                        tokens.append(ft)
                        token_set.add(ft)
            _add_algo_meta(_trace, _fact_hop,
                           output=f"S={fact.subject!r}  R={fact.relation!r}  V={fact.value[:40]!r}  override={fact.override}")

        existing = self._items.get(namespace, [])

        # --- Check for confirmation (exact duplicate) ---
        # Skip dedup when all fact fields are empty — distinct texts
        # that produce no content words should not collapse into one item.
        with _algo_span(_trace, "algo.dedup_check",
                        existing_count=len(existing)) as _dedup_hop:
            has_fact_fields = bool(fact.subject or fact.relation or fact.value)
            _dedup_result = "new"
            if has_fact_fields:
                for item in existing:
                    if (
                        item.fact.subject == fact.subject
                        and item.fact.relation == fact.relation
                        and item.fact.value == fact.value
                        and item.consolidation_strength >= self.STRENGTH_FLOOR
                    ):
                        item.retrieval_count += 1
                        _add_algo_meta(_trace, _dedup_hop,
                                       output=f"DUPLICATE — reinforced item {item.id[:8]}  rc={item.retrieval_count}")
                        return item  # Confirmation — reinforced existing
            _add_algo_meta(_trace, _dedup_hop,
                           output=f"unique — no duplicate found among {len(existing)} existing")

        # --- Compute surprise ---
        with _algo_span(_trace, "algo.surprise_calc",
                        mode="structured" if (fact.subject and fact.relation) else "token") as _surp_hop:
            if fact.subject and fact.relation:
                # Structured surprise (if we have S, R, V)
                surprise, contradicted = self._compute_surprise(fact, existing)
            else:
                # Token-based surprise (raw text)
                surprise, contradicted, confirmed = self._compute_surprise_from_tokens(
                    text, token_set, namespace,
                )
                if confirmed and not contradicted:
                    _add_algo_meta(_trace, _surp_hop,
                                   output=f"TOKEN DEDUP — reinforced item {confirmed.id[:8]}")
                    return confirmed  # Token-level dedup — no new item needed

            # --- Apply surprise damage ---
            if contradicted:
                if fact.subject and fact.relation:
                    self._apply_surprise_damage(surprise, contradicted, fact)
                else:
                    self._apply_token_surprise_damage(
                        surprise, contradicted, fact.override,
                    )
            _add_algo_meta(_trace, _surp_hop,
                           output=f"surprise={round(surprise, 4)}  contradicted={'yes' if contradicted else 'no'}")

        # --- Compute thermodynamic state ---
        with _algo_span(_trace, "algo.thermo_state") as _thermo_hop:
            rho = self._memory_density(namespace)
            tau = self.TAU_OVERRIDE if fact.override else self.TAU_DEFAULT
            H = self._information_content(fact)
            L = (self.kT * math.log(2) * H) / max(tau, 1e-6)
            _add_algo_meta(_trace, _thermo_hop,
                           output=f"ρ={round(rho, 3)}  τ={round(tau, 3)}  H={round(H, 3)} bits  L={round(L, 6)} J")

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
        with _algo_span(_trace, "algo.index",
                        item_id=item.id[:8], token_count=len(tokens)) as _idx_hop:
            self._items.setdefault(namespace, []).append(item)
            self._total_item_count += 1
            self._item_by_id[item.id] = item
            self._index_item(item)

            # Update document frequency
            for token in set(tokens):
                self._doc_freq[token] += 1

            # Update co-occurrence matrix (PPMI Layer)
            unique_tokens = list(set(tokens))
            for i in range(len(unique_tokens)):
                for j in range(i + 1, len(unique_tokens)):
                    a, b = unique_tokens[i], unique_tokens[j]
                    pair = (a, b) if a < b else (b, a)
                    self._cooccurrence[pair] += 1

            self._svd_store_count += 1
            self._svd_dirty = True

            # Trigger SVD recomputation if enough stores accumulated
            _svd_recomputed = False
            if (self._svd_store_count >= self._SVD_RECOMPUTE_INTERVAL
                    and self._svd_dirty
                    and not getattr(self, '_batch_mode', False)):
                self._recompute_svd()
                self._svd_store_count = 0
                self._svd_dirty = False
                _svd_recomputed = True
            _add_algo_meta(_trace, _idx_hop,
                           output=f"indexed  total_ns={len(self._items.get(namespace,[]))}  svd_recomputed={_svd_recomputed}")

        # --- Recompute free energy for ALL items ---
        # In batch mode, skip per-item recompute (caller handles it)
        if not getattr(self, '_batch_mode', False):
            with _algo_span(_trace, "algo.free_energy",
                            ns_size=len(self._items.get(namespace, []))) as _fe_hop:
                self._recompute_all_free_energies(namespace)
                _add_algo_meta(_trace, _fe_hop,
                               output=f"F={round(item.free_energy, 4)}  s={round(item.consolidation_strength, 3)}")

            # --- Cross-Entity Resonance: Write-Time Coupling ---
            with _algo_span(_trace, "algo.cer_update",
                            item_id=item.id[:8]) as _cer_hop:
                _cer_before = len(getattr(self, '_entity_nodes', {}))
                self._cer_update(item, text, namespace)
                _cer_after = len(getattr(self, '_entity_nodes', {}))
                _add_algo_meta(_trace, _cer_hop,
                               output=f"entity_nodes={_cer_after}  new={_cer_after - _cer_before}")

            # --- Schema Absorption: Check if episode is absorbed by existing schema ---
            with _algo_span(_trace, "algo.schema_absorb",
                            item_id=item.id[:8]) as _schema_hop:
                _was_schema = item.schema_meta is not None
                self._try_schema_absorption(item, namespace)
                _is_schema = item.schema_meta is not None
                _add_algo_meta(_trace, _schema_hop,
                               output=f"crystallized={'yes' if _is_schema else 'no'}  was_schema={_was_schema}")

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

        # --- Benchmark mode: preserve ALL items (no crystallization, no GC) ---
        # Like the brain: memories aren't deleted, just hard to reach.
        # The recall agent keeps pointers alive via retrieval_count boost.
        if self._benchmark_mode:
            return

        # --- Liquid → Solid: Melting + Crystallization (before GC) ---
        self._check_schema_melting(namespace)
        self._check_crystallization(namespace)

        # --- GC: Remove dead items (including melted schemas) ---
        alive = []
        all_items = self._items.get(namespace, [])  # Re-fetch (crystallization may add)
        for item in all_items:
            if item.consolidation_strength >= self.STRENGTH_FLOOR:
                alive.append(item)
            else:
                self._deindex_item(item)
                self._cer_gc_item(item)
                self._item_by_id.pop(item.id, None)
                # Decrement doc freq for dead item's tokens
                for token in set(item.indexed_tokens):
                    self._doc_freq[token] = max(0, self._doc_freq.get(token, 1) - 1)
        self._items[namespace] = alive
        # Recompute cached total item count after GC
        self._total_item_count = sum(len(v) for v in self._items.values())

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
        trace_id: Optional[str] = None,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Unified retrieval: TRR (Thermodynamic Resonance Retrieval).

        Layers applied in order:
            1. Schema-Aware Query Expansion (add fixed-point tokens)
            2. Morphological Kernel (prefix expansion at query time)
            3. BMX Scoring (entropy-weighted BM25)
            4. Semantic Bonus (PPMI-SVD cosine similarity)
            5. Phase-Dependent Susceptibility (Gas/Liquid/Solid/Glass)
            6. Cross-Entity Resonance (CER) for multi-entity queries

        All parameters self-tuned from corpus statistics.
        Zero external calls. Sub-millisecond.
        """
        self._recompute_all_free_energies(namespace)

        query_tokens = _tokenize(query)
        query_token_set = set(query_tokens)

        # --- Layer 6: Schema-Aware Query Expansion ---
        with _algo_span(trace_id, "algo.query_expand",
                        input=query[:120], raw_tokens=len(query_tokens)) as _qe_hop:
            query_tokens, query_token_set, inferred_tokens = self._expand_query_with_schemas(
                query_tokens, query_token_set, namespace,
            )
            _add_algo_meta(trace_id, _qe_hop,
                           output=f"{len(query_tokens)} tokens (inferred={len(inferred_tokens)})  added={len(inferred_tokens)}")

        # --- CER: Multi-entity detection ---
        with _algo_span(trace_id, "algo.cer_detect",
                        input=query[:80]) as _cer_hop:
            query_entities = self._detect_multi_entity_query(query_token_set, namespace, query)
            _add_algo_meta(trace_id, _cer_hop,
                           output=f"{len(query_entities)} entities: {query_entities[:5]}")

        if len(query_entities) >= 2:
            cer_results = self._cer_search(
                query_entities, query_token_set, namespace, limit,
            )
            if cer_results:
                tsf_results = self._tsf_search(
                    query_tokens, query_token_set, namespace, limit,
                    inferred_tokens=inferred_tokens,
                    trace_id=trace_id,
                )
                results = self._merge_cer_and_tsf(cer_results, tsf_results, limit)
                for _, item in results:
                    item.retrieval_count += 1
                return results

        results = self._tsf_search(
            query_tokens, query_token_set, namespace, limit,
            inferred_tokens=inferred_tokens,
            trace_id=trace_id,
        )
        for _, item in results:
            item.retrieval_count += 1
        return results

    # =========================================================================
    # TSF Search — Standard Token-Index Retrieval
    # =========================================================================

    def _tsf_search(
        self,
        query_tokens: list[str],
        query_token_set: set[str],
        namespace: str,
        limit: int,
        inferred_tokens: Optional[set[str]] = None,
        trace_id: Optional[str] = None,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Thermodynamic Resonance Retrieval (TRR).

        Full scoring pipeline:
            1. Morphological Kernel — prefix expansion finds variants
            2. BMX Score — Σ IDF(t) × H_weight(t) × inference_weight
            3. Semantic Bonus — cosine(query_vec, item_vec) × avg_idf
            4. Thermo Component — -F(item) / kT
            5. Phase Susceptibility — χ(phase) multiplier

        rank = (bmx + semantic + thermo) × χ

        All parameters self-tuned. Zero external calls. Sub-millisecond.
        """
        if inferred_tokens is None:
            inferred_tokens = set()

        candidates: dict[str, tuple[PhaseMemoryItem, list[str]]] = {}

        # Layer 1: Morphological Kernel — expand query tokens to prefix variants
        with _algo_span(trace_id, "algo.morph_kernel",
                        query_tokens=len(query_tokens)) as _morph_hop:
            for token in query_tokens:
                variants = self._morph_expand(token)
                if not variants:
                    # Direct lookup fallback for short tokens
                    if token in self._token_index:
                        variants = [token]
                    else:
                        continue
                for variant in variants:
                    if variant not in self._token_index:
                        continue
                    for item in self._token_index[variant]:
                        if item.namespace != namespace:
                            continue
                        if item.id not in candidates:
                            candidates[item.id] = (item, [])
                        if variant not in candidates[item.id][1]:
                            candidates[item.id][1].append(variant)
            _add_algo_meta(trace_id, _morph_hop,
                           output=f"{len(candidates)} candidates from token expansion")

        scored: list[tuple[float, PhaseMemoryItem]] = []

        if candidates:
            # Pre-compute query vector for semantic bonus
            query_vec = self._mean_vector(query_tokens) if self._token_vectors else None
            avg_idf = 0.0
            if query_tokens:
                avg_idf = sum(self._compute_idf(t) for t in query_tokens) / len(query_tokens)

            # Layers 2–5: score all candidates in one pass
            with _algo_span(trace_id, "algo.score",
                            candidates=len(candidates),
                            avg_idf=round(avg_idf, 4),
                            has_svd=query_vec is not None) as _score_hop:
                for item, matched_tokens in candidates.values():
                    self._update_field_radius(item)

                    # Layer 2: BMX Scoring — entropy-weighted BM25
                    bmx_score = 0.0
                    for t in matched_tokens:
                        idf = self._compute_idf(t)
                        h_weight = self._compute_entropy_weight(t)
                        # Inferred tokens (from schema expansion) scored at 0.5×
                        inf_weight = 0.5 if t in inferred_tokens else 1.0
                        bmx_score += idf * h_weight * inf_weight

                    # Layer 3: Semantic Bonus (PPMI-SVD vectors, if available)
                    semantic_bonus = 0.0
                    if query_vec is not None:
                        item_vec = self._mean_vector(item.indexed_tokens)
                        if item_vec is not None:
                            sim = self._cosine_similarity(query_vec, item_vec)
                            semantic_bonus = sim * avg_idf

                    # Layer 4: Thermodynamic component
                    thermo = -self._safe_fe(item) / max(self.kT, 1e-9)

                    # Layer 5: Phase-Dependent Susceptibility
                    chi = self._phase_susceptibility(item)

                    rank = (bmx_score + semantic_bonus + thermo) * chi
                    scored.append((rank, item))

                scored.sort(key=lambda x: x[0], reverse=True)
                _top = scored[0] if scored else None
                _top_info = (f"rank={round(_top[0], 4)}  bmx+sem+thermo×χ  "
                             f"text={(_top[1].fact.raw_text or '')[:60]}") if _top else "no results"
                _add_algo_meta(trace_id, _score_hop,
                               output=f"{len(scored)} scored  top: {_top_info}")
        else:
            # No token matches: fallback to free energy ranking
            with _algo_span(trace_id, "algo.score",
                            candidates=0, fallback="free_energy") as _score_hop:
                items = self._items.get(namespace, [])
                for item in items:
                    chi = self._phase_susceptibility(item)
                    rank = -self._safe_fe(item) / max(self.kT, 1e-9) * chi
                    scored.append((rank, item))
                scored.sort(key=lambda x: x[0], reverse=True)
                _add_algo_meta(trace_id, _score_hop,
                               output=f"fallback FE rank  {len(scored)} items")

        # retrieval_count incremented by search(), not here
        return scored[:limit]

    # =========================================================================
    # CER Search — Entanglement Graph Traversal
    # =========================================================================

    def _detect_multi_entity_query(
        self,
        query_token_set: set[str],
        namespace: str,
        query_text: str = "",
    ) -> list[str]:
        """
        Detect if query references 2+ known entities.

        Two-pass detection:
        1. Single-token entities: O(Q) hash lookups
        2. Compound entities ("New York"): scan query words for consecutive
           sequences matching compound entity index

        Namespace membership verified via pre-built set of item IDs — O(1).
        """
        # Build namespace member set ONCE — O(N)
        ns_item_ids = {item.id for item in self._items.get(namespace, [])}

        matched: list[str] = []

        # Pass 1: single-token entity matching
        for token in query_token_set:
            canonical = token
            if token in self._entity_alias_map:
                canonical = self._entity_alias_map[token]
            if canonical in self._entity_nodes:
                node = self._entity_nodes[canonical]
                has_ns = any(mid in ns_item_ids for mid in node.memory_ids)
                if has_ns and canonical not in matched:
                    matched.append(canonical)

        # Pass 2: compound entity matching (e.g., "New York", "San Francisco")
        if query_text and self._compound_entity_index:
            query_words = [_strip_punctuation(w).lower() for w in query_text.split()]
            i = 0
            while i < len(query_words):
                word = query_words[i]
                if word in self._compound_entity_index:
                    # Try longest match first (sorted by n_parts descending)
                    candidates = sorted(
                        self._compound_entity_index[word],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    found = False
                    for compound_name, n_parts in candidates:
                        if i + n_parts <= len(query_words):
                            candidate = " ".join(query_words[i:i + n_parts])
                            if candidate == compound_name:
                                canonical = compound_name
                                if compound_name in self._entity_alias_map:
                                    canonical = self._entity_alias_map[compound_name]
                                if canonical in self._entity_nodes:
                                    node = self._entity_nodes[canonical]
                                    has_ns = any(mid in ns_item_ids for mid in node.memory_ids)
                                    if has_ns and canonical not in matched:
                                        matched.append(canonical)
                                i += n_parts
                                found = True
                                break
                    if not found:
                        i += 1
                else:
                    i += 1

        # Deduplicate: remove single-token entities that are parts of matched compounds
        # e.g., if "new york" matched, remove "new" if it was a false single-token match
        compound_parts: set[str] = set()
        for entity in matched:
            parts = entity.split()
            if len(parts) > 1:
                compound_parts.update(parts)
        if compound_parts:
            matched = [e for e in matched if " " in e or e not in compound_parts]

        return matched

    def _cer_search(
        self,
        entities: list[str],
        query_tokens: set[str],
        namespace: str,
        limit: int,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        CER dispatcher: PESQD (per-entity decomposition) + legacy shared-token search.

        PESQD decomposes the query into per-entity sub-queries, discovers
        cross-entity resonant tokens, and scores by entity overlap.
        Legacy CER uses pre-computed entanglement graph shared_tokens.

        Results from both are merged (dedup by item ID, keep max score).
        """
        pesqd = self._pesqd_search(entities, query_tokens, namespace, limit)
        if pesqd:
            legacy = self._cer_shared_token_search(
                entities, query_tokens, namespace, limit,
            )
            if legacy:
                # Merge: dedup by item ID, keep max score
                merged: dict[str, tuple[float, PhaseMemoryItem]] = {
                    item.id: (score, item) for score, item in pesqd
                }
                for score, item in legacy:
                    if item.id not in merged or score > merged[item.id][0]:
                        merged[item.id] = (score, item)
                result = sorted(merged.values(), key=lambda x: x[0], reverse=True)
                return result[:limit]
            return pesqd
        return self._cer_shared_token_search(
            entities, query_tokens, namespace, limit,
        )

    # =========================================================================
    # PESQD — Per-Entity Sub-Query Decomposition
    # =========================================================================

    def _get_coupling(self, entities: list[str]) -> float:
        """
        Get coupling strength for a set of entities.

        2 entities → EntanglementEdge coupling_strength.
        3+ entities → cluster coherence, or mean pairwise coupling as fallback.
        """
        if len(entities) == 2:
            a, b = sorted(entities)
            edge = self._entanglement_graph.get(a, {}).get(b)
            return edge.coupling_strength if edge else 0.0

        # 3+ entities: try cluster coherence first
        entity_set = set(entities)
        for cluster in self._resonance_clusters.values():
            if entity_set.issubset(cluster.members):
                return cluster.coherence

        # Fallback: mean pairwise coupling
        pair_sum = 0.0
        pair_count = 0
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = sorted([entities[i], entities[j]])
                edge = self._entanglement_graph.get(a, {}).get(b)
                if edge:
                    pair_sum += edge.coupling_strength
                pair_count += 1
        return pair_sum / max(pair_count, 1)

    def _pesqd_search(
        self,
        entities: list[str],
        query_token_set: set[str],
        namespace: str,
        limit: int,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Per-Entity Sub-Query Decomposition.

        Factorizes the multi-entity response function into single-entity
        propagators with a cross-correlation kernel.

        Phase 1:   Gather each entity's memories independently.
        Phase 1.5: Discover cross-entity resonant tokens (multi-hop bridges).
        Phase 2:   Score by entity overlap × (IDF + filter + cross) × s − F/kT.

        Zero external calls. O(E × M_e + C × T). Sub-millisecond.
        """
        entity_set = set(entities)
        # Build token-level entity filter: includes compound parts
        # e.g. "jean paul" → {"jean paul", "jean", "paul"}
        entity_tokens = set()
        for e in entities:
            entity_tokens.add(e)
            entity_tokens.update(e.split())
        num_entities = len(entity_set)  # Use set size, not list — prevents duplicate inflation

        # --- Phase 1: Per-entity memory gathering ---
        memory_entity_map: dict[str, set[str]] = {}
        for entity_name in entities:
            node = self._entity_nodes.get(entity_name)
            if not node:
                continue
            for mid in node.memory_ids:
                item = self._item_by_id.get(mid)
                if not item or item.namespace != namespace:
                    continue
                # Gas items now searchable (TRR: fresh memories are vivid)
                memory_entity_map.setdefault(mid, set()).add(entity_name)

        if not memory_entity_map:
            return []

        # --- Coupling strength ---
        K_coupling = self._get_coupling(entities)

        # --- Filter tokens: non-entity query tokens ---
        filter_tokens = {
            t for t in query_token_set
            if t not in entity_tokens and t not in self._entity_nodes
        }

        # --- Phase 1.5: Cross-entity token discovery ---
        # Track which entities' memories contain each non-entity token
        token_entity_coverage: dict[str, set[str]] = {}
        for mid, owning in memory_entity_map.items():
            item = self._item_by_id.get(mid)
            if not item:
                continue  # Defensive: stale mid
            for token in item.indexed_tokens:
                if token not in entity_tokens and token not in self._entity_nodes:
                    token_entity_coverage.setdefault(token, set()).update(owning)

        # Tokens covered by ALL queried entities = cross-entity resonant tokens
        cross_entity_tokens: set[str] = {
            t for t, ents in token_entity_coverage.items()
            if len(ents) >= num_entities
        }

        # --- Phase 2: Score each candidate memory ---
        scored: list[tuple[float, PhaseMemoryItem]] = []
        for mid, owning in memory_entity_map.items():
            item = self._item_by_id.get(mid)
            if not item:
                continue  # Defensive: stale mid

            # Entity overlap ratio: fraction of queried entities owning this memory
            overlap_ratio = len(owning) / num_entities
            pesqd_boost = overlap_ratio * (1.0 + K_coupling)

            # IDF from memory's non-entity, non-boosted tokens (base signal)
            item_tokens = set(item.indexed_tokens)
            token_idf = sum(
                self._compute_idf(t) for t in item_tokens
                if t not in entity_tokens
                and t not in cross_entity_tokens
                and t not in filter_tokens
            )

            # Filter bonus: 2× IDF for query content words found in memory
            filter_bonus = sum(
                self._compute_idf(t) * 2.0 for t in filter_tokens
                if t in item_tokens
            )

            # Cross-entity bonus: 3× IDF for multi-hop bridge tokens
            # (highest boost — these are the tokens linking entities)
            cross_bonus = sum(
                self._compute_idf(t) * 3.0 for t in item_tokens
                if t in cross_entity_tokens
            )

            s = item.consolidation_strength
            rank = (
                pesqd_boost * (token_idf + filter_bonus + cross_bonus) * s
                - self._safe_fe(item) / max(self.kT, 1e-9)
            )
            scored.append((rank, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    # =========================================================================
    # Legacy CER — Shared Token Search (pre-PESQD)
    # =========================================================================

    def _cer_shared_token_search(
        self,
        entities: list[str],
        query_tokens: set[str],
        namespace: str,
        limit: int,
    ) -> list[tuple[float, PhaseMemoryItem]]:
        """
        Legacy CER: search via pre-computed entanglement graph shared_tokens.

        For 2 entities: O(1) edge lookup → shared_tokens = resonant frequencies.
        For 3+ entities: cluster lookup → cluster_spectrum.

        Score: K × idf × s − F/kT
        """
        scored: list[tuple[float, PhaseMemoryItem]] = []

        if len(entities) == 2:
            a, b = sorted(entities)
            edge = self._entanglement_graph.get(a, {}).get(b)

            if edge and edge.coupling_strength > 0:
                search_tokens = edge.shared_tokens
                if not search_tokens:
                    return []

                # Accumulate IDF per item across all matching tokens
                # (consistent with TSF/PESQD which sum IDFs)
                item_idf_acc: dict[str, tuple[float, PhaseMemoryItem]] = {}
                for token, weight in search_tokens.most_common(20):
                    if token in self._token_index:
                        for item in self._token_index[token]:
                            if item.namespace != namespace:
                                continue
                            idf = self._compute_idf(token)
                            if item.id in item_idf_acc:
                                prev_idf, _ = item_idf_acc[item.id]
                                item_idf_acc[item.id] = (prev_idf + idf, item)
                            else:
                                item_idf_acc[item.id] = (idf, item)

                for sum_idf, item in item_idf_acc.values():
                    cer_score = (
                        edge.coupling_strength * sum_idf
                        * item.consolidation_strength
                        - self._safe_fe(item) / max(self.kT, 1e-9)
                    )
                    scored.append((cer_score, item))

                # Also include shared memories — score via IDF of shared tokens
                for memory_id in edge.shared_memory_ids:
                    item = self._item_by_id.get(memory_id)
                    if item and item.namespace == namespace:
                        # Use sum of shared token IDFs (consistent with token-match scoring)
                        item_token_set = set(item.indexed_tokens)
                        shared_idf = sum(
                            self._compute_idf(t)
                            for t in search_tokens
                            if t in item_token_set
                        )
                        if shared_idf == 0:
                            # Fallback: mean IDF of item's tokens
                            shared_idf = sum(
                                self._compute_idf(t) for t in item.indexed_tokens
                            ) / max(len(item.indexed_tokens), 1)
                        rank = (
                            edge.coupling_strength * shared_idf
                            * item.consolidation_strength
                            - self._safe_fe(item) / max(self.kT, 1e-9)
                        )
                        scored.append((rank, item))

        elif len(entities) >= 3:
            entity_set = set(entities)
            for cluster in self._resonance_clusters.values():
                if entity_set.issubset(cluster.members):
                    for token, weight in cluster.cluster_spectrum.most_common(20):
                        if token in self._token_index:
                            for item in self._token_index[token]:
                                if item.namespace != namespace:
                                    continue
                                idf = self._compute_idf(token)
                                cer_score = (
                                    cluster.coherence * idf
                                    * item.consolidation_strength
                                    - self._safe_fe(item) / max(self.kT, 1e-9)
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
            # Boost positive CER scores; don't amplify negative penalties
            boosted = score * CER_BOOST if score > 0 else score
            merged[item.id] = (boosted, item)

        for score, item in tsf_results:
            if item.id in merged:
                existing_score = merged[item.id][0]
                merged[item.id] = (max(existing_score, score), item)
            else:
                merged[item.id] = (score, item)

        result = sorted(merged.values(), key=lambda x: x[0], reverse=True)

        # retrieval_count incremented by search(), not here
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
            if item.schema_meta is not None:
                n_members = len(item.schema_meta.member_ids)
                phase = "glass" if _is_glass_static(item) else "schema"
                lines.append(f"- [strength={s:.2f}, {phase}, {n_members} memories] {item.fact.raw_text}")
            else:
                lines.append(f"- [strength={s:.2f}] {item.fact.raw_text}")
            debug_items.append({
                "text": item.fact.raw_text,
                "score": round(score, 4),
                **item.to_debug_dict(strength_floor=self.STRENGTH_FLOOR),
            })

        context = "Memory (strongest recall first):\n" + "\n".join(lines)
        return context, debug_items

    # =========================================================================
    # Two-Tier Retrieval — Schemas + Episode Archive
    # =========================================================================

    def search_with_details(
        self,
        query: str,
        namespace: str,
        limit: int = 10,
        detail_limit: int = 50,
    ) -> tuple[list[tuple[float, "PhaseMemoryItem"]], list["PhaseMemoryItem"]]:
        """
        Two-tier retrieval: schemas for relevance, archive for detail.

        1. Standard search() returns ranked schemas/items
        2. For matched schemas, fetches archived original episodes
        3. Also token-matches query against all archived episodes

        Returns (ranked_results, detail_episodes).
        """
        results = self.search(query, namespace, limit=limit)

        # Collect archived episodes from matched schemas
        detail_episodes: list[PhaseMemoryItem] = []
        seen_ids: set[str] = set()

        for _score, item in results:
            if item.schema_meta is not None and item.id in self._episode_archive:
                for ep in self._episode_archive[item.id]:
                    if ep.id not in seen_ids:
                        detail_episodes.append(ep)
                        seen_ids.add(ep.id)

        # Also do a token-match search against all archived episodes
        query_tokens = _tokenize(query)
        query_set = set(query_tokens)
        for _schema_id, episodes in self._episode_archive.items():
            for ep in episodes:
                if ep.id in seen_ids:
                    continue
                ep_tokens = set(ep.indexed_tokens)
                if query_set & ep_tokens:
                    detail_episodes.append(ep)
                    seen_ids.add(ep.id)

        # Sort by birth_order (chronological)
        detail_episodes.sort(key=lambda e: e.birth_order)
        return results, detail_episodes[:detail_limit]

    def build_augmented_context_with_details(
        self,
        query: str,
        namespace: str,
        limit: int = 10,
        detail_limit: int = 50,
    ) -> tuple[str, list[dict]]:
        """
        Build context with schema summaries + episode details.

        Two sections:
        1. Ranked retrieval results (schemas + liquid items)
        2. Detailed conversation excerpts from episode archive
        """
        results, details = self.search_with_details(
            query, namespace, limit=limit, detail_limit=detail_limit,
        )

        if not results and not details:
            return "No prior context yet.", []

        lines: list[str] = []
        debug_items: list[dict] = []

        # Schema summaries / ranked items
        for score, item in results:
            s = item.consolidation_strength
            if item.schema_meta is not None:
                n_members = len(item.schema_meta.member_ids)
                phase = "glass" if _is_glass_static(item) else "schema"
                lines.append(f"- [strength={s:.2f}, {phase}, {n_members} memories] {item.fact.raw_text}")
            else:
                lines.append(f"- [strength={s:.2f}] {item.fact.raw_text}")
            debug_items.append({
                "text": item.fact.raw_text,
                "score": round(score, 4),
                **item.to_debug_dict(strength_floor=self.STRENGTH_FLOOR),
            })

        # Detail episodes from archive
        if details:
            lines.append("\nDetailed conversation excerpts:")
            for ep in details:
                lines.append(f"  - {ep.fact.raw_text}")
                debug_items.append({
                    "text": ep.fact.raw_text,
                    "score": 0.0,
                    **ep.to_debug_dict(strength_floor=self.STRENGTH_FLOOR),
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
            "liquid_count": sum(
                1 for i in items
                if i.consolidation_strength >= self.STRENGTH_FLOOR
                and i.schema_meta is None
            ),
            "solid_count": sum(
                1 for i in items
                if i.schema_meta is not None
                and i.consolidation_strength >= self.STRENGTH_FLOOR
                and not _is_glass_static(i)
            ),
            "glass_count": sum(
                1 for i in items
                if i.schema_meta is not None
                and _is_glass_static(i)
            ),
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
            "trr": {
                "cooccurrence_pairs": len(self._cooccurrence),
                "token_vectors": len(self._token_vectors),
                "prefix_index_size": len(self._prefix_index),
                "svd_store_count": self._svd_store_count,
            },
        }

    # =========================================================================
    # Cross-User Vector API — Federated Semantic Learning
    # =========================================================================

    def finalize_batch(self, namespace: str | None = None) -> dict[str, Any]:
        """
        Finalize a batch ingest: recompute SVD + free energies.

        MUST be called after setting _batch_mode = False post-ingest.
        Without this, _token_vectors stays empty → semantic bonus = 0.

        Args:
            namespace: If provided, recompute free energies for this namespace.
                       If None, recompute for all namespaces.

        Returns:
            dict with finalization stats (token_vectors, cooccurrence_pairs, etc.)
        """
        self._batch_mode = False

        # 1. Recompute SVD from accumulated co-occurrence
        if self._cooccurrence and self._svd_dirty:
            self._recompute_svd()
            self._svd_store_count = 0
            self._svd_dirty = False

        # 2. Recompute free energies (triggers crystallization, GC, etc.)
        if namespace is not None:
            self._recompute_all_free_energies(namespace)
        else:
            for ns in list(self._items.keys()):
                self._recompute_all_free_energies(ns)

        return {
            "token_vectors": len(self._token_vectors),
            "cooccurrence_pairs": len(self._cooccurrence),
            "total_items": self._total_item_count,
        }

    # =========================================================================
    # Recall Agent — Hippocampal Replay for Long-Tail Memory Preservation
    # =========================================================================

    def recall_long_tail(self, namespace: str, batch_size: int = 50) -> int:
        """
        Memory rehearsal agent. Like hippocampal replay during sleep.

        Human brains don't delete old memories — they lose the retrieval
        pointer. This method "recalls" old, low-retrieval-count items,
        incrementing R so that:

            s(t) = exp(-Δt/τ) · (1 + 0.15·ln(1 + R))

        stays above STRENGTH_FLOOR indefinitely. Mathematically equivalent
        to spaced repetition.

        Finds items with lowest retrieval_count (oldest, least-recalled)
        and touches them. This is the "recall agent" — a background process
        whose ONLY job is to keep long-tail memory pointers alive.

        Args:
            namespace: Memory namespace to rehearse.
            batch_size: Number of items to rehearse per call.

        Returns:
            Number of items rehearsed.
        """
        items = self._items.get(namespace, [])
        if not items:
            return 0

        # Sort by retrieval_count ascending, then birth_order ascending (oldest first)
        candidates = sorted(
            items,
            key=lambda i: (i.retrieval_count, i.birth_order),
        )

        rehearsed = 0
        for item in candidates[:batch_size]:
            item.retrieval_count += 1  # The "recall" — strengthens the pointer
            rehearsed += 1

        return rehearsed

    def export_vectors(self) -> dict[str, list[float]]:
        """
        Export anonymized token vectors for cross-user knowledge sharing.

        Returns a copy of _token_vectors. No user-identifying information.
        Token vectors are derived from co-occurrence statistics only.
        """
        return dict(self._token_vectors)

    def import_vectors(self, external_vectors: dict[str, list[float]]) -> int:
        """
        Import token vectors from another user's memory space.

        Merges external vectors with local via weighted average:
        - Local vectors (user's own data) get 0.8 weight
        - External vectors get 0.2 weight
        - New tokens (not in local space) get external vectors directly

        Returns the number of tokens affected.
        """
        affected = 0
        for token, ext_vec in external_vectors.items():
            if len(ext_vec) != self._SVD_DIMS:
                continue
            if token in self._token_vectors:
                local_vec = self._token_vectors[token]
                merged = [0.8 * l + 0.2 * e for l, e in zip(local_vec, ext_vec)]
                mag = math.sqrt(sum(v * v for v in merged))
                if mag > 1e-12:
                    self._token_vectors[token] = [v / mag for v in merged]
                affected += 1
            else:
                self._token_vectors[token] = ext_vec
                affected += 1
        return affected
