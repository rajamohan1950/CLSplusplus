"""
CLS++ Topical Resonance Graph (TRG) — Cross-LLM Contextual Recall

Sessions as Oscillators in Topic Space
=======================================

Extends the Kuramoto coupled oscillator framework from entities (CER in
memory_phase.py) to **sessions**. Each active LLM session is an oscillator
with a sliding-window topic signature — an IDF-weighted token spectrum that
evolves as the user types.

Sessions with overlapping topic signatures synchronize (high coupling K);
unrelated sessions desynchronize (low K). Only synchronized sessions share
context — preventing injection of irrelevant topics.

Complexity:
    Prompt ingestion:  O(T + S)          T=tokens, S=active sessions
    Coupling update:   O(min(|A|,|B|))   per session pair
    Cross-session:     O(S × P × log P)  S=synced, P=prompts per session
    Cascade recall:    O(S×P + N + kNN)  N=engine items, kNN=L1/L2/L3

All in-process, zero external dependencies.
"""

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from clsplusplus.memory_phase import PhaseMemoryEngine

# Re-use the engine's tokenizer — same vocabulary, same IDF statistics.
from clsplusplus.memory_phase import _tokenize


# =============================================================================
# Constants — Thermodynamic Parameters
# =============================================================================

# Kuramoto critical coupling threshold. Same value as CER entity coupling
# (memory_phase.py line 654). Below this, sessions are independent (different
# topics). Above this, sessions phase-lock (same topic cluster).
K_CRITICAL = 0.15

# Topic signature decay half-life in prompts. After TAU_TOPIC prompts,
# old tokens decay by e^(-1) ≈ 0.37. This handles topic drift:
# 10 prompts about "database" will decay the "auth" signal by 63%.
TAU_TOPIC = 10.0

# Freshness decay half-life in seconds for cross-session recall.
# A prompt from 5 minutes ago scores e^(-300/300) = 0.37 of a fresh prompt.
TAU_FRESH = 300.0

# Maximum age (seconds) for cross-session prompts to be considered.
MAX_PROMPT_AGE = 3600.0  # 1 hour

# Ring buffer size per session — most recent prompts kept for recall.
RING_SIZE = 50

# Maximum active sessions per namespace before pruning stale ones.
MAX_SESSIONS = 20

# Minimum token spectrum weight before pruning (noise floor).
SPECTRUM_FLOOR = 0.01

# Session stale timeout — sessions inactive for this long are pruned.
SESSION_STALE_SECONDS = 7200.0  # 2 hours


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PromptEntry:
    """A single prompt in a session's ring buffer."""
    session_id: str
    role: str             # 'user' or 'assistant'
    content: str          # verbatim text (first 2000 chars)
    llm_provider: str     # 'claude-code', 'chatgpt', 'gemini', 'grok'
    timestamp: float      # time.time()
    token_set: frozenset  # pre-tokenized for O(1) overlap in recall
    sequence_num: int = 0


@dataclass
class SessionOscillator:
    """A session's real-time topic state in the Topical Resonance Graph.

    The topic_spectrum is a sliding-window IDF-weighted token distribution.
    It evolves with each prompt via additive injection + multiplicative decay.
    This IS the session's position in topic space.

    The omega (natural frequency) is the Shannon entropy of the spectrum —
    high entropy = broad topic, low entropy = focused topic.
    """
    session_id: str
    llm_provider: str           # 'claude-code', 'chatgpt', 'gemini', 'grok'
    namespace: str              # canonical user namespace

    # Sliding window topic signature
    topic_spectrum: Counter = field(default_factory=Counter)
    prompt_count: int = 0

    # Kuramoto oscillator state
    theta: float = 0.0          # phase angle (radians)
    omega: float = 0.0          # natural frequency (spectrum entropy)

    # Recent prompts ring (for retrieval after coupling match)
    recent_prompts: deque = field(default_factory=lambda: deque(maxlen=RING_SIZE))

    # Temporal state
    last_active: float = 0.0
    created_at: float = field(default_factory=time.time)

    # Cached magnitude squared for coupling computation
    _mag_sq: float = 0.0


# =============================================================================
# TopicalResonanceGraph
# =============================================================================


class TopicalResonanceGraph:
    """Cross-session topic coupling graph. One per namespace (user).

    The TRG is the pre-filter layer that decides WHICH cross-session context
    to inject. It does NOT replace the PhaseMemoryEngine — it augments it
    with real-time session-level topic awareness.

    Thread safety: this class is designed for single-threaded asyncio.
    The engine reference is read-only (IDF lookups).
    """

    def __init__(self, engine: "PhaseMemoryEngine"):
        self.engine = engine

        # Active session oscillators — session_id → SessionOscillator
        self._sessions: dict[str, SessionOscillator] = {}

        # Coupling matrix — (s1, s2) → K(s1, s2), sparse
        self._coupling: dict[tuple[str, str], float] = {}

        # Session-level document frequency: token → count of sessions containing it.
        # Used to downweight generic tokens ("add", "implement", "use") that appear
        # in every technical session but carry zero topical signal.
        self._session_df: Counter = Counter()

    # ─── Properties ───────────────────────────────────────────────────

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    def get_session(self, session_id: str) -> Optional[SessionOscillator]:
        return self._sessions.get(session_id)

    def get_coupling(self, s1: str, s2: str) -> float:
        return self._coupling.get((s1, s2), 0.0)

    # ─── Prompt Ingestion ─────────────────────────────────────────────

    def on_prompt(self, session_id: str, content: str,
                  llm_provider: str, namespace: str,
                  role: str = "user", sequence_num: int = 0) -> SessionOscillator:
        """Ingest a prompt: update session oscillator + coupling.

        O(T + S × min(|A|,|B|)) where T=tokens, S=active sessions.

        Args:
            session_id: Opaque session identifier from the client.
            content: Verbatim prompt text.
            llm_provider: 'claude-code', 'chatgpt', 'gemini', 'grok', etc.
            namespace: Canonical user namespace.
            role: 'user' or 'assistant'.
            sequence_num: Monotonic ordering within the session.

        Returns:
            The updated SessionOscillator.
        """
        tokens = _tokenize(content[:2000])
        token_set = frozenset(set(tokens))

        # Get or create session oscillator
        osc = self._sessions.get(session_id)
        if osc is None:
            osc = SessionOscillator(
                session_id=session_id,
                llm_provider=llm_provider,
                namespace=namespace,
            )
            self._sessions[session_id] = osc
            # Prune stale sessions if we're at capacity
            if len(self._sessions) > MAX_SESSIONS:
                self._prune_stale()

        # ── Sliding Window Topic Signature ──
        # Add new tokens with IDF weighting
        # Track session-level DF for cross-session IDF normalization
        for token in tokens:
            idf = self.engine._compute_idf(token, namespace)
            if token not in osc.topic_spectrum or osc.topic_spectrum[token] < SPECTRUM_FLOOR:
                # First time this token appears in this session's spectrum
                self._session_df[token] += 1
            osc.topic_spectrum[token] += idf

        # Decay ALL tokens (exponential forgetting per prompt)
        decay = math.exp(-1.0 / TAU_TOPIC)
        dead_tokens = []
        mag_sq = 0.0
        for token in osc.topic_spectrum:
            osc.topic_spectrum[token] *= decay
            w = osc.topic_spectrum[token]
            if w < SPECTRUM_FLOOR:
                dead_tokens.append(token)
            else:
                mag_sq += w * w
        for token in dead_tokens:
            del osc.topic_spectrum[token]

        osc._mag_sq = mag_sq
        osc.prompt_count += 1
        osc.last_active = time.time()

        # Update natural frequency (spectrum entropy)
        osc.omega = self._spectrum_entropy(osc.topic_spectrum)

        # Push to recent prompts ring
        osc.recent_prompts.appendleft(PromptEntry(
            session_id=session_id,
            role=role,
            content=content[:2000],
            llm_provider=llm_provider,
            timestamp=time.time(),
            token_set=token_set,
            sequence_num=sequence_num,
        ))

        # ── Update Coupling with all other active sessions ──
        now = time.time()
        for other_id, other_osc in self._sessions.items():
            if other_id == session_id:
                continue
            if now - other_osc.last_active > SESSION_STALE_SECONDS:
                continue

            K = self._coupling_strength(osc, other_osc)
            self._coupling[(session_id, other_id)] = K
            self._coupling[(other_id, session_id)] = K

        return osc

    # ─── Coupling Strength ────────────────────────────────────────────

    def _coupling_strength(self, a: SessionOscillator,
                           b: SessionOscillator) -> float:
        """Session-IDF-weighted Shared Information Content.

        K(A,B) = SIC_sidf(A,B) / sqrt(|A|²_sidf × |B|²_sidf)

        where SIC_sidf = Σ (min(w_A(t), w_B(t)) × sidf(t))²

        sidf(t) = log(1 + N_sessions / (1 + session_df(t)))

        Tokens that appear in MANY sessions (generic verbs: "add", "implement",
        "use") get sidf ≈ 0 and contribute nothing to coupling. Tokens that
        appear in FEW sessions ("auth", "jwt", "middleware") get high sidf
        and dominate. This is the session-level analog of document-level IDF.

        This extends the CER entanglement formula (memory_phase.py line 655)
        with a session-frequency normalization layer.

        O(min(|A|, |B|)) — iterate the smaller spectrum.
        """
        spec_a, spec_b = a.topic_spectrum, b.topic_spectrum
        n_sessions = max(len(self._sessions), 1)

        # Iterate smaller spectrum for efficiency
        if len(spec_a) > len(spec_b):
            spec_a, spec_b = spec_b, spec_a

        sic = 0.0
        mag_a_sidf = 0.0
        mag_b_sidf = 0.0

        # Compute session-IDF weighted SIC
        for token, weight_s in spec_a.items():
            sdf = self._session_df.get(token, 1)
            sidf = math.log(1.0 + n_sessions / (1.0 + sdf))

            weight_l = spec_b.get(token, 0.0)
            wa_sidf = weight_s * sidf
            mag_a_sidf += wa_sidf * wa_sidf

            if weight_l > 0.0:
                wl_sidf = weight_l * sidf
                sic += min(wa_sidf, wl_sidf) ** 2

        # Compute mag_b with sidf (only tokens in spec_b)
        for token, weight_b in spec_b.items():
            sdf = self._session_df.get(token, 1)
            sidf = math.log(1.0 + n_sessions / (1.0 + sdf))
            wb_sidf = weight_b * sidf
            mag_b_sidf += wb_sidf * wb_sidf

        denominator = math.sqrt(mag_a_sidf * mag_b_sidf)
        if denominator < 1e-9:
            return 0.0

        return sic / denominator

    # ─── Cross-Session Recall ─────────────────────────────────────────

    def recall_cross_session(self, session_id: str, query: str,
                             namespace: str,
                             limit: int = 10) -> list[tuple[float, PromptEntry]]:
        """Retrieve relevant context from topically synchronized sessions.

        Only returns prompts from sessions where K(current, other) > K_CRITICAL.
        This is the topical gate that prevents injecting irrelevant context.

        O(S × P × log P) where S = synchronized sessions, P = prompts per session.
        Typically S < 5, P < 50 → O(250 log 250) ≈ O(2000) operations.

        Args:
            session_id: Current session requesting context.
            query: The user's current prompt text.
            namespace: Canonical namespace for IDF lookups.
            limit: Max results to return.

        Returns:
            List of (score, PromptEntry) sorted by score descending.
        """
        osc = self._sessions.get(session_id)
        if osc is None:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        now = time.time()
        candidates = []

        for other_id, other_osc in self._sessions.items():
            if other_id == session_id:
                continue

            # ── Topical gate: only synchronized sessions ──
            K = self._coupling.get((session_id, other_id), 0.0)
            if K < K_CRITICAL:
                continue

            # ── Score each recent prompt from this synced session ──
            for entry in other_osc.recent_prompts:
                if entry.role != "user":
                    continue

                age = now - entry.timestamp
                if age > MAX_PROMPT_AGE:
                    break  # Ring is ordered by time, so all remaining are older

                # Token relevance: IDF-weighted overlap with query
                overlap_score = 0.0
                for token in query_tokens:
                    if token in entry.token_set:
                        idf = self.engine._compute_idf(token, namespace)
                        overlap_score += idf

                if overlap_score < 0.01:
                    continue

                # Normalize by query length (BM25-style)
                overlap_score /= math.sqrt(len(query_tokens))

                # Coupling boost: stronger coupling = more relevant session
                # At K_CRITICAL: boost = 1.0. At 2×K_CRITICAL: boost = 2.0.
                coupling_boost = K / K_CRITICAL

                # Freshness: exponential decay
                freshness = math.exp(-age / TAU_FRESH)

                # Final score
                score = overlap_score * coupling_boost * freshness

                if score > 0.01:
                    candidates.append((score, entry))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:limit]

    # ─── Promotion Bridge: TRG → Engine Thermodynamic Pipeline ─────

    def reinforce_cross_session(self, session_id: str, query: str,
                                namespace: str) -> int:
        """Bridge cross-session prompts into the engine's consolidation pipeline.

        When cross-session context is recalled and injected into an LLM, the
        relevant facts should get a retrieval boost in the PhaseMemoryEngine.
        This is the thermodynamic equivalent of hippocampal replay — rehearsing
        memories encountered across sessions strengthens their consolidation.

        The existing promotion pipeline (Gas→Liquid→Solid→Glass) then handles
        the rest: frequently-reinforced cross-session facts eventually
        crystallize into schemas (L2) and engrams (L3).

        Returns the number of engine items reinforced.
        """
        cross_results = self.recall_cross_session(session_id, query, namespace, limit=5)
        if not cross_results:
            return 0

        reinforced = 0
        for score, entry in cross_results:
            # Find matching items in the engine by token overlap
            entry_tokens = set(_tokenize(entry.content[:500]))
            for item in self.engine._items.get(namespace, []):
                item_tokens = set(item.indexed_tokens[:10])
                overlap = len(entry_tokens & item_tokens) / max(len(entry_tokens), 1)
                if overlap > 0.3:
                    # Reinforce: increment retrieval count
                    # This feeds into s(t) = exp(-Δt/τ) × (1 + β·ln(1+R))
                    # which drives the promotion pipeline
                    item.retrieval_count += 1
                    reinforced += 1

        return reinforced

    # ─── Session Management ───────────────────────────────────────────

    def remove_session(self, session_id: str) -> None:
        """Remove a session oscillator and all its coupling edges.
        Also decrements session-DF counts for tokens in the removed session."""
        osc = self._sessions.pop(session_id, None)
        if osc:
            for token in osc.topic_spectrum:
                if self._session_df.get(token, 0) > 0:
                    self._session_df[token] -= 1
                    if self._session_df[token] <= 0:
                        del self._session_df[token]
        dead_keys = [k for k in self._coupling if session_id in k]
        for k in dead_keys:
            del self._coupling[k]

    def _prune_stale(self) -> None:
        """Remove sessions inactive for > SESSION_STALE_SECONDS."""
        now = time.time()
        stale = [
            sid for sid, osc in self._sessions.items()
            if now - osc.last_active > SESSION_STALE_SECONDS
        ]
        for sid in stale:
            self.remove_session(sid)

    # ─── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _spectrum_entropy(spectrum: Counter) -> float:
        """Shannon entropy of a token spectrum (natural frequency).

        H = -Σ p(t) × log₂(p(t))

        High H = broad topic (many tokens with similar weight).
        Low H = focused topic (few dominant tokens).
        """
        total = sum(spectrum.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for weight in spectrum.values():
            if weight > 0:
                p = weight / total
                entropy -= p * math.log2(p)
        return entropy

    def debug_state(self) -> dict:
        """Serializable snapshot of TRG state for debugging."""
        sessions = {}
        for sid, osc in self._sessions.items():
            top_tokens = osc.topic_spectrum.most_common(10)
            sessions[sid] = {
                "provider": osc.llm_provider,
                "prompt_count": osc.prompt_count,
                "top_tokens": [(t, round(w, 3)) for t, w in top_tokens],
                "omega": round(osc.omega, 3),
                "recent_count": len(osc.recent_prompts),
                "last_active_ago": round(time.time() - osc.last_active, 1),
            }

        couplings = {}
        for (s1, s2), K in self._coupling.items():
            if s1 < s2:  # Deduplicate symmetric pairs
                couplings[f"{s1[:8]}↔{s2[:8]}"] = {
                    "K": round(K, 4),
                    "synchronized": K >= K_CRITICAL,
                }

        return {"sessions": sessions, "couplings": couplings}
