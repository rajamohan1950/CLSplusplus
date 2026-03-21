"""CLS++ Memory Service - unified single code path.

Architecture:
  - PhaseMemoryEngine: THE brain. All write/search logic. Zero external deps.
  - L1IndexingStore: Persistence layer. Write-through (fire-and-forget).
  - L2SchemaGraph: Crystallized schemas. Written by sleep cycle (REM phase).
  - ReconsolidationGate: Belief revision via adjudicate endpoint.
  - L0/L3: Not in hot path. L0 replaced by PhaseMemoryEngine in-memory buffer.

Write: text → PhaseMemoryEngine.store() → L1.write() [async, persistence]
Read:  query → PhaseMemoryEngine.search() [in-memory, sub-ms]
Startup: L1 → PhaseMemoryEngine (reload persisted items into brain)
"""

import asyncio
import logging
import math
import re as _re
from datetime import datetime, timezone
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.embeddings import EmbeddingService
from clsplusplus.memory_phase import Fact, PhaseMemoryEngine
from clsplusplus.models import MemoryItem, ReadRequest, ReadResponse, StoreLevel, WriteRequest
from clsplusplus.reconsolidation import ReconsolidationGate
from clsplusplus.stores import L1IndexingStore, L2SchemaGraph
from clsplusplus.temporal import (
    annotate_relative_dates,
    extract_event_date,
    parse_temporal_filter,
    resolve_relative_dates,
)
from clsplusplus.tracer import tracer

logger = logging.getLogger(__name__)


class MemoryService:
    """Main service — PhaseMemoryEngine is the brain, L1 is persistence."""

    # Hippocampal replay fires automatically every N writes per namespace
    REPLAY_EVERY_N_WRITES: int = 50

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()

        # THE brain — thermodynamic memory engine, zero external deps
        self.engine = PhaseMemoryEngine()

        # Persistence — write-through, fail-gracefully
        self.embedding_service = EmbeddingService(settings)
        self.l1 = L1IndexingStore(settings)
        self.l2 = L2SchemaGraph(settings)

        # Belief revision (used by adjudicate endpoint)
        self.reconsolidation = ReconsolidationGate(settings)

        self._webhook_dispatcher = None  # Lazy init to avoid circular imports
        self._loaded_namespaces: set[str] = set()
        self._loading_namespaces: set[str] = set()  # currently loading (background)
        self._write_counts: dict[str, int] = {}  # per-namespace write counter
        # ns → {event_key → sorted list[str]} — one date-list per recurring event type
        # Managed entirely outside the PhaseMemoryEngine to avoid crystallization interference.
        self._event_threads: dict[str, dict[str, list[str]]] = {}
        self._event_threads_lock = asyncio.Lock()

    # =========================================================================
    # Conversion — PhaseMemoryItem ↔ MemoryItem
    # =========================================================================

    def _phase_to_item(self, phase_item, req: Optional[WriteRequest] = None) -> MemoryItem:
        """Convert PhaseMemoryItem to MemoryItem for API responses and persistence."""
        s = phase_item.consolidation_strength
        if s < self.engine.STRENGTH_FLOOR:
            store_level = StoreLevel.L0   # gas — volatile
        elif phase_item.schema_meta is not None:
            store_level = StoreLevel.L2   # solid/glass — crystallized schema
        else:
            store_level = StoreLevel.L1   # liquid — episodic

        return MemoryItem(
            id=phase_item.id,
            text=phase_item.fact.raw_text,
            namespace=phase_item.namespace,
            store_level=store_level,
            source=req.source if req else "user",
            timestamp=datetime.utcnow(),
            confidence=min(1.0, s),
            salience=phase_item.surprise_at_birth,
            usage_count=phase_item.retrieval_count,
            surprise=phase_item.surprise_at_birth,
            authority=req.authority if req else 0.5,
            metadata=req.metadata if req else {},
            subject=phase_item.fact.subject or (req.subject if req else None),
            predicate=phase_item.fact.relation or None,
            object=(phase_item.fact.value[:256] if phase_item.fact.value else None) or (req.object if req else None),
            event_at=getattr(phase_item, "event_at", None),
            superseded=getattr(phase_item, "superseded", False),
        )

    # =========================================================================
    # Startup — reload persisted items into PhaseMemoryEngine
    # =========================================================================

    async def ensure_loaded(self, namespace: str) -> None:
        """Load persisted L1 items into PhaseMemoryEngine.

        Non-blocking: if a background pre-load is already in progress for this
        namespace, return immediately — the engine will serve whatever it has
        already loaded.  Only blocks on the very first request if no pre-load
        was started.

        Restores full thermodynamic state from L1:
          • consolidation_strength  ← confidence
          • retrieval_count         ← usage_count
          • surprise_at_birth       ← surprise
          • embedding_dense         ← embedding  (384-dim; enables semantic re-ranking)
        """
        if namespace in self._loaded_namespaces:
            return
        if namespace in self._loading_namespaces:
            # Background load in progress — don't block the request
            return
        self._loading_namespaces.add(namespace)
        try:
            # Top 50k by confidence DESC — strongest memories load first so the
            # engine's 1k hot-item pressure evicts the weakest, not the strongest.
            items = await self.l1.list_for_sleep(namespace, limit=50000)
            if items:
                self.engine._batch_mode = True
                for item in items:
                    phase_item = self.engine.store(item.text, item.namespace)
                    if phase_item is not None:
                        # Restore persisted thermodynamic state so the brain
                        # wakes up with its memory intact, not tabula rasa.
                        phase_item.consolidation_strength = float(item.confidence)
                        phase_item.retrieval_count        = int(item.usage_count)
                        phase_item.surprise_at_birth      = float(item.surprise or 0.0)
                        if item.embedding:
                            phase_item.embedding_dense = list(item.embedding)
                        # Restore temporal provenance
                        if item.event_at is not None:
                            phase_item.event_at = item.event_at
                        phase_item.superseded = bool(item.superseded)
                self.engine.finalize_batch(namespace)
                logger.info(
                    "Loaded %d items from L1 into PhaseMemoryEngine for ns=%s "
                    "(state fully restored: strength, retrieval_count, embeddings)",
                    len(items), namespace,
                )
        except Exception as e:
            logger.warning("Could not load from L1 (continuing with empty brain): %s", e)
        finally:
            self._loading_namespaces.discard(namespace)
            self._loaded_namespaces.add(namespace)

    async def prewarm(self, namespace: str) -> None:
        """Pre-load a namespace in the background so the first user request is instant.

        Call this at application startup for namespaces that are known to be active.
        Safe to call multiple times — idempotent.
        """
        if namespace in self._loaded_namespaces or namespace in self._loading_namespaces:
            return
        asyncio.create_task(self.ensure_loaded(namespace))

    async def startup_preload(self) -> None:
        """Discover every namespace stored in L1 and preload each one into the
        PhaseMemoryEngine in the background.

        Called once from the app startup handler so the brain is warm before
        the first real request arrives.  Two extra things happen here:

        1. IVFFlat index is re-tuned to the current row count so kNN stays
           accurate as the table grows towards 1M+ items.
        2. Each namespace is loaded concurrently (one asyncio.Task per namespace)
           so a deployment with many users doesn't serialize on a single namespace.

        This is safe to call multiple times — idempotent (prewarm is a no-op for
        already-loaded namespaces).
        """
        try:
            # Re-tune vector index before loading (non-blocking rebuild)
            asyncio.create_task(self.l1.ensure_vector_index_scale())

            namespaces = await self.l1.list_namespaces()
            if not namespaces:
                logger.info("startup_preload: L1 is empty — nothing to load")
                return

            logger.info(
                "startup_preload: found %d namespace(s) in L1 — preloading all",
                len(namespaces),
            )
            for ns in namespaces:
                await self.prewarm(ns)   # schedules a background task per namespace

        except Exception as e:
            # Never crash startup — persistence is optional for a running server
            logger.warning("startup_preload failed (server will lazy-load on demand): %s", e)

    # =========================================================================
    # 384-dim Semantic Re-ranking (post-TRR layer)
    # =========================================================================

    def _semantic_rerank(
        self,
        results: list,
        query_emb: list[float],
        alpha: float = 0.4,
    ) -> list:
        """
        Re-rank TRR results using 384-dim SentenceTransformer cosine similarity.

        Final score = (1 - alpha) × ttr_norm + alpha × cosine_384(query, item)

        Bridges vocabulary gaps that the morphological kernel misses
        ("relocated" ↔ "moved", "physician" ↔ "doctor").
        Items without embedding_dense are scored at cosine=0.0 (not penalized).

        Adaptive alpha: when TRR scores are very low (query has no token overlap
        with stored facts — e.g. "what are my hobbies" vs "I went hiking"), alpha
        is raised so semantic similarity dominates over weak TRR noise.
        """
        if not results or not query_emb:
            return results

        scores = [s for s, _ in results]
        max_s = max(scores)
        min_s = min(scores)
        rng = (max_s - min_s) or 1.0

        # Adaptive alpha: poor TRR coverage → let semantics dominate
        effective_alpha = 0.85 if max_s < 0.05 else alpha

        reranked = []
        for ttr_score, item in results:
            ttr_norm = (ttr_score - min_s) / rng  # normalise to [0,1]
            cosine_384 = 0.0
            if item.embedding_dense:
                cosine_384 = max(0.0, EmbeddingService.cosine_similarity(query_emb, item.embedding_dense))
            final = (1.0 - effective_alpha) * ttr_norm + effective_alpha * cosine_384
            reranked.append((final, item))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked

    # =========================================================================
    # Temporal Event Thread — chain recurring episodic events into a timeline
    # =========================================================================

    _THREAD_MAX_SHOWN = 5   # keep first + last N dates; rest summarised
    _DATE_PAT = _re.compile(
        r'\d{1,2} (?:January|February|March|April|May|June|July|'
        r'August|September|October|November|December) \d{4}'
    )
    _MONTH_ORDER = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    @classmethod
    def _parse_thread_date(cls, date_str: str) -> tuple[int, int, int]:
        """Parse a thread date string "7 January 2024" → (2024, 1, 7) for sorting."""
        try:
            parts = date_str.split()
            if len(parts) == 3:
                return (int(parts[2]), cls._MONTH_ORDER.get(parts[1].lower(), 0), int(parts[0]))
        except (ValueError, IndexError):
            pass
        return (0, 0, 0)

    @classmethod
    def _sort_thread_dates(cls, dates: list[str]) -> list[str]:
        """Return dates sorted chronologically."""
        return sorted(dates, key=cls._parse_thread_date)

    def _best_matching_thread_key(self, ns: str, event_tokens: set[str]) -> Optional[str]:
        """Find the thread key in namespace that best matches the event's meaningful tokens.

        Requires ≥1 shared meaningful (non-stopword) token.  When multiple threads
        qualify, the one with the most token overlap wins.

        This enables real-world phrasing variation:
          "I went to the doctor for a checkup"   → both share "doctor"
          "I went to the doctor for a follow-up" → merge into same thread
        while keeping doctor / hiking / luna threads separate (no shared meaningful tokens).
        """
        if not event_tokens:
            return None
        best_key = None
        best_overlap = 0
        for key, _ in self._event_threads.get(ns, {}).items():
            key_tokens = self._meaningful_tokens(key)
            overlap = self._fuzzy_token_overlap(event_tokens, key_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
        return best_key if best_overlap > 0 else None  # any shared meaningful token

    async def _update_event_thread(self, req: WriteRequest, original_text: str, event_date: str) -> None:
        """Maintain one ordered date-list per recurring episodic event type.

        Threads live entirely in self._event_threads (a plain dict) — never in
        the PhaseMemoryEngine.  This avoids crystallisation, GC, and zombie-item
        problems completely.  At read time _matching_threads() injects matching
        threads into results as synthetic MemoryItems.

        Fuzzy key matching (Jaccard on meaningful tokens) allows real-world phrasing
        variation to collapse into a single thread:
          "I went to the doctor for a checkup" and
          "I went to the doctor for a follow-up"
        both share "doctor" → same thread.

        Parameters
        ----------
        req           : original WriteRequest
        original_text : req.text before resolution — the event identity key
        event_date    : absolute date of THIS event (from store_text)
        """
        if not event_date:
            return
        ns = req.namespace
        ns_threads = self._event_threads.setdefault(ns, {})
        event_tokens = self._meaningful_tokens(original_text)

        # Step 1: Check for a fuzzy-matching existing thread (handles phrasing variation)
        existing_key = self._best_matching_thread_key(ns, event_tokens)
        if existing_key is not None:
            dates = ns_threads[existing_key]
            if event_date not in dates:
                dates.append(event_date)
                ns_threads[existing_key] = self._sort_thread_dates(dates)
            return

        # Step 2: No matching thread — search engine for prior occurrences of this event
        # to confirm it is genuinely recurring before creating a thread.
        # Filter results to items that share at least one meaningful token with the
        # event, to avoid harvesting dates from unrelated items that matched on
        # incidental tokens like "went" or "the".
        results = self.engine.search(original_text, ns, limit=20)
        episode_dates: list[str] = []
        for _score, item in results:
            item_tokens = self._meaningful_tokens(item.fact.raw_text or "")
            if self._fuzzy_token_overlap(event_tokens, item_tokens) == 0:
                continue  # unrelated item — skip its dates
            for d in self._DATE_PAT.findall(item.fact.raw_text or ""):
                if d not in episode_dates:
                    episode_dates.append(d)

        if not episode_dates:
            return   # single occurrence so far — don't create a thread yet

        # Seed the thread with previously found dates + current, sorted chronologically
        all_dates = list(dict.fromkeys(episode_dates))
        if event_date not in all_dates:
            all_dates.append(event_date)
        ns_threads[original_text.lower().strip()] = self._sort_thread_dates(all_dates)

    def _thread_display_text(self, original_text: str, dates: list[str]) -> str:
        """Build the human-readable thread string returned to callers."""
        total = len(dates)
        if total > self._THREAD_MAX_SHOWN + 1:
            shown = [dates[0], "…"] + dates[-(self._THREAD_MAX_SHOWN - 1):]
        else:
            shown = list(dates)
        return (
            f"[Thread] {original_text}: "
            f"{' → '.join(shown)} "
            f"({total} time{'s' if total != 1 else ''} total, latest: {dates[-1]})"
        )

    # Common English stopwords + temporal/directional words — excluded from thread token
    # matching to prevent false positives between unrelated events.
    # "yesterday"/"today"/"tomorrow"/"morning"/"evening"/"night" are excluded so that
    # time words don't cause doctor-threads to match hiking-threads.
    _STOPWORDS = frozenset({
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "a", "an", "the", "is", "was", "are", "were",
        "be", "been", "have", "has", "had", "do", "did", "will", "would",
        "could", "should", "may", "might", "to", "of", "in", "on", "at",
        "by", "for", "with", "and", "or", "but", "so", "yet", "if",
        "this", "that", "these", "those", "there", "here", "what", "when",
        "where", "who", "how", "why", "which", "go", "went", "get", "got",
        "s", "t", "re", "ll", "ve", "d",
        # Temporal / directional words (don't discriminate between event types)
        "yesterday", "today", "tomorrow", "morning", "afternoon", "evening",
        "night", "tonight", "now", "just", "ago", "back", "last", "next",
        "some", "all", "up", "out", "over", "new", "like", "very",
    })

    def _meaningful_tokens(self, text: str) -> set[str]:
        """Return non-stopword tokens from text."""
        return {t for t in _re.findall(r'\w+', text.lower()) if t not in self._STOPWORDS}

    @staticmethod
    def _stem_token(t: str) -> str:
        """Minimal English stemmer for thread token matching.

        Strips common suffixes so that "hiking"/"hike"/"hiked"/"hiker" all
        normalise to "hik", and "cooking"/"cook"/"cooked" to "cook".

        NOT a full Porter stemmer — just enough for event-type matching.
        """
        if len(t) > 5 and t.endswith("ing"):
            return t[:-3]            # hiking → hik, cooking → cook
        if len(t) > 4 and t.endswith("ed"):
            base = t[:-2]            # hiked → hik, cooked → cook
            if base.endswith("e"):
                return base[:-1]     # loved → lov (not needed but safe)
            return base
        if len(t) > 4 and t.endswith("er"):
            return t[:-2]            # hiker → hik
        if len(t) > 4 and t.endswith("ers"):
            return t[:-3]            # hikers → hik
        if len(t) > 3 and t.endswith("e"):
            return t[:-1]            # hike → hik, move → mov
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            return t[:-1]            # doctors → doctor
        return t

    @classmethod
    def _tokens_fuzzy_match(cls, a: str, b: str) -> bool:
        """True if two tokens share the same stem.

        Handles common suffix variations so that "hike" matches "hiking",
        "cook" matches "cooking", "doctor" matches "doctors", etc.
        Requires both tokens to be ≥ 3 chars to avoid noise.
        """
        if a == b:
            return True
        if len(a) < 3 or len(b) < 3:
            return False
        return cls._stem_token(a) == cls._stem_token(b)

    def _fuzzy_token_overlap(self, tokens_a: set[str], tokens_b: set[str]) -> int:
        """Count fuzzy-matched token pairs between two token sets."""
        count = 0
        for ta in tokens_a:
            for tb in tokens_b:
                if self._tokens_fuzzy_match(ta, tb):
                    count += 1
                    break  # each token in a matches at most one in b
        return count

    def _matching_threads(
        self,
        query: str,
        namespace: str,
        temporal_filter=None,
    ) -> list[MemoryItem]:
        """Return thread MemoryItems whose event key meaningfully overlaps with the query.

        Uses non-stopword token overlap (with fuzzy suffix matching) to prevent
        common words ("went", "yesterday", "i") from causing unrelated threads to
        inject into results, while handling "hike" vs "hiking" variations.
        Requires at least 1 meaningful shared token.

        When *temporal_filter* has date bounds, thread dates are filtered to
        that window.  Query patterns like "last time", "how many times", and
        "most recent" are handled with specialised display text.
        """
        query_tokens = self._meaningful_tokens(query)
        if not query_tokens:
            return []
        q_lower = query.lower()
        matched = []
        for event_key, dates in self._event_threads.get(namespace, {}).items():
            if len(dates) < 2:
                continue
            key_tokens = self._meaningful_tokens(event_key)
            if self._fuzzy_token_overlap(query_tokens, key_tokens) == 0:
                continue

            # Apply date-range filter if temporal_filter has bounds
            filtered_dates = dates
            if temporal_filter and (temporal_filter.start or temporal_filter.end):
                filtered_dates = [
                    d for d in dates
                    if self._thread_date_in_range(d, temporal_filter)
                ]
            if len(filtered_dates) < 1:
                continue

            # Choose display style based on query intent
            if any(w in q_lower for w in ("last time", "most recent", "latest", "when did i last", "when was the last")):
                text = f"[Last time] {event_key}: {filtered_dates[-1]}"
            elif any(w in q_lower for w in ("how many", "how often", "times have", "how many times", "count")):
                recent = filtered_dates[-3:]
                text = f"[Frequency] {event_key}: {len(filtered_dates)} times. Most recent: {', '.join(recent)}"
            else:
                text = self._thread_display_text(event_key, filtered_dates)

            matched.append(MemoryItem(
                text=text,
                namespace=namespace,
                store_level=StoreLevel.L1,
                source="thread",
                confidence=1.0,
                salience=1.0,
            ))
        return matched

    @staticmethod
    def _thread_date_in_range(date_str: str, tf) -> bool:
        """Return True if a thread date string falls within a TemporalFilter's bounds."""
        # Thread dates are stored as strings like "7 May 2024" — parse them lazily
        try:
            from datetime import datetime as _dt
            # Try "7 May 2024" format
            d = _dt.strptime(date_str.strip(), "%d %B %Y").replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                # Try ISO format fallback
                d = _dt.fromisoformat(date_str.strip())
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
            except ValueError:
                return True  # Can't parse → include conservatively
        if tf.start and d < tf.start:
            return False
        if tf.end and d > tf.end:
            return False
        return True

    # =========================================================================
    # Temporal helpers
    # =========================================================================

    @staticmethod
    def _event_at_in_range(event_at, tf) -> bool:
        """Return True if *event_at* falls within the TemporalFilter bounds.

        If *event_at* is None (unknown), the item is conservatively included.
        """
        if event_at is None:
            return True  # no event_at → unknown time → include conservatively
        dt = event_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if tf.start:
            start = tf.start if tf.start.tzinfo else tf.start.replace(tzinfo=timezone.utc)
            if dt < start:
                return False
        if tf.end:
            end = tf.end if tf.end.tzinfo else tf.end.replace(tzinfo=timezone.utc)
            if dt > end:
                return False
        return True

    def _apply_recency_decay(self, results, now: datetime, tf) -> list:
        """Blend recency score into semantic score using exponential half-life decay.

        Formula: recency = exp(-age_days * ln(2) / half_life_days)
        Final:   score   = (1 - alpha) * semantic + alpha * recency

        Items with unknown event_at are treated as age=0 (brand new, max recency).
        """
        half_life = max(tf.recency_half_life_days, 1.0)
        alpha = tf.recency_alpha
        now_tz = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        reranked = []
        for score, phase_item in results:
            event_at = getattr(phase_item, "event_at", None)
            if event_at is None:
                age_days = 0.0  # unknown age → treat as current
            else:
                dt = event_at if event_at.tzinfo else event_at.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now_tz - dt).total_seconds() / 86400.0)
            recency = math.exp(-age_days * math.log(2) / half_life)
            final = (1.0 - alpha) * score + alpha * recency
            reranked.append((final, phase_item))
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked

    async def _mark_superseded_async(
        self,
        new_item,           # PhaseMemoryItem — the new, winning fact
        namespace: str,
        new_event_at: datetime,
    ) -> None:
        """Mark older same-(subject, relation) facts as superseded.

        Conservative: only supersedes when:
        - exact subject + relation match
        - older event_at < new_event_at
        Never deletes — superseded=True just hides from default reads.
        """
        try:
            subject = new_item.fact.subject
            relation = new_item.fact.relation
            if not subject or not relation:
                return

            new_tz = new_event_at if new_event_at.tzinfo else new_event_at.replace(tzinfo=timezone.utc)

            candidates = self.engine._items.get(namespace, [])
            for item in candidates:
                if item.id == new_item.id:
                    continue
                if item.fact.subject != subject or item.fact.relation != relation:
                    continue
                # Same subject + relation — check if older
                old_event_at = getattr(item, "event_at", None)
                if old_event_at is None:
                    # No event_at on old item → supersede it (new fact wins by default)
                    item.superseded = True
                    asyncio.create_task(self.l1.update_superseded(item.id, namespace))
                    # Record lineage: new item supersedes old
                    if item.id not in new_item.fact.raw_text:
                        pass  # lineage tracked at MemoryItem level, not here
                else:
                    old_tz = old_event_at if old_event_at.tzinfo else old_event_at.replace(tzinfo=timezone.utc)
                    if old_tz < new_tz:
                        item.superseded = True
                        asyncio.create_task(self.l1.update_superseded(item.id, namespace))
        except Exception as e:
            logger.debug("_mark_superseded_async non-fatal: %s", e)

    # =========================================================================
    # Persistence helpers — fire-and-forget so failures never block reads/writes
    # =========================================================================

    async def _persist_to_l1(
        self,
        item: MemoryItem,
        trace_id: Optional[str] = None,
        parent_hop_id: Optional[str] = None,
    ) -> None:
        """Write to L1 PostgreSQL. Called as background task — never blocks."""
        try:
            item = self.embedding_service.embed_item(item)
            await self.l1.write(item, trace_id=trace_id, parent_hop_id=parent_hop_id)
        except Exception as e:
            logger.warning("L1 persist failed (data safe in PhaseMemoryEngine): %s", e)

    def _dispatch_webhook(self, event_type: str, item: MemoryItem) -> None:
        """Fire webhook event (fire-and-forget)."""
        if self._webhook_dispatcher is None:
            return
        try:
            payload = {
                "id": item.id,
                "text": item.text,
                "namespace": item.namespace,
                "store_level": item.store_level.value,
                "confidence": item.confidence,
                "source": item.source,
            }
            asyncio.create_task(
                self._webhook_dispatcher.dispatch(event_type, payload, item.namespace)
            )
        except Exception:
            pass

    # =========================================================================
    # Write
    # =========================================================================

    async def write(self, req: WriteRequest, trace_id: Optional[str] = None) -> MemoryItem:
        """Write memory: PhaseMemoryEngine (brain) + L1 (persistence)."""
        trace_id = tracer.ensure_trace(trace_id or tracer.new_trace("write"), "write")

        with tracer.span(trace_id, "memory_service.write", "memory_service",
                         input=req.text[:200],
                         namespace=req.namespace, source=req.source, text_len=len(req.text)) as ms_write_hop:

            # 1. Ensure namespace data is loaded into PhaseMemoryEngine
            with tracer.span(trace_id, "ensure_loaded", "memory_service",
                             namespace=req.namespace):
                await self.ensure_loaded(req.namespace)

            # 2. Store in PhaseMemoryEngine — THE thermodynamic brain
            #
            # Two transforms when a conversation_date is provided:
            #   store_text   — relative expressions replaced in-place with absolute dates
            #                  ("yesterday" → "on 7 May 2024").  Unique date tokens prevent
            #                  the engine from crystallising two different visits into one
            #                  schema just because the sentence structure is identical.
            #   display_text — original wording kept, resolved date appended in brackets
            #                  ("yesterday [7 May 2024]").  Written to fact.raw_text so the
            #                  LLM sees both natural language AND the exact date.
            store_text = req.text
            display_text = req.text
            if req.conversation_date:
                resolved = resolve_relative_dates(req.text, req.conversation_date)
                if resolved != req.text:
                    store_text = resolved
                    display_text = annotate_relative_dates(req.text, req.conversation_date)

            # When display_text differs from store_text, pass a pre-built Fact so
            # the engine indexes store_text (unique date tokens) while fact.raw_text
            # carries display_text ("yesterday [7 May 2024]") for LLM readability.
            display_fact = None
            if display_text != store_text:
                display_fact = Fact(
                    subject="",
                    relation="said",
                    value=store_text.lower(),
                    override=False,
                    raw_text=display_text,
                )

            with tracer.span(trace_id, "engine.store", "phase_engine",
                             input=store_text[:200],
                             namespace=req.namespace) as store_hop:
                phase_item = self.engine.store(store_text, req.namespace, fact=display_fact, trace_id=trace_id)
                if phase_item is None:
                    # Safety fallback: shouldn't happen but handle gracefully
                    results = self.engine.search(req.text, req.namespace, limit=1)
                    phase_item = results[0][1] if results else None
                if phase_item:
                    tracer.add_metadata(trace_id, store_hop,
                                        output=f"item_id={str(phase_item.id)[:8]}…  phase={'crystallized' if phase_item.schema_meta else 'liquid'}  strength={round(phase_item.consolidation_strength, 3)}",
                                        phase=str(phase_item.schema_meta is not None),
                                        strength=round(phase_item.consolidation_strength, 3))
                else:
                    tracer.add_metadata(trace_id, store_hop, output="none", phase="none", strength=0)

            # ---- Temporal provenance: event_at + session_date + event threads ----
            # Determine when the described event HAPPENED.  Priority:
            #   1. Explicit override from req.event_at
            #   2. Auto-extracted from store_text (which has resolved relative dates)
            #   3. conversation_date fallback (the write happened "now" in the conversation)
            _event_at: Optional[datetime] = (
                req.event_at
                or extract_event_date(store_text, req.conversation_date)
            )
            if phase_item and _event_at is not None:
                phase_item.event_at = _event_at

            if req.conversation_date:
                from clsplusplus.temporal import date_label as _date_label
                _dl = _date_label(req.conversation_date)  # "7 May 2024 (Tuesday)"
                if phase_item:
                    phase_item.session_date = _dl
                if store_text != req.text:
                    # Extract event date from store_text (the resolved text),
                    # not from conversation_date — those can differ by a day.
                    _event_dates = self._DATE_PAT.findall(store_text)
                    if _event_dates:
                        await self._update_event_thread(req, req.text, _event_dates[0])

            # Belief supersession: fire-and-forget, never blocks the write
            if (
                phase_item
                and phase_item.fact.subject
                and phase_item.fact.relation
                and _event_at is not None
            ):
                asyncio.create_task(
                    self._mark_superseded_async(phase_item, req.namespace, _event_at)
                )

            # 2b. Attach 384-dim dense embedding to PhaseMemoryItem (fire-and-forget on item)
            # PhaseMemoryEngine stays zero-dep; we attach post-store so TRR is unaffected.
            with tracer.span(trace_id, "embed_dense", "embedding_service",
                             input=req.text[:120],
                             model=self.settings.embedding_model,
                             namespace=req.namespace) as emb_hop:
                try:
                    if phase_item and not phase_item.embedding_dense:
                        emb = self.embedding_service.embed(req.text)
                        phase_item.embedding_dense = emb
                        tracer.add_metadata(trace_id, emb_hop,
                                            output=f"{len(emb)}-dim vector")
                    elif phase_item and phase_item.embedding_dense:
                        tracer.add_metadata(trace_id, emb_hop,
                                            output=f"{len(phase_item.embedding_dense)}-dim (cached)")
                except Exception as _emb_err:
                    logger.debug("Dense embed failed (non-fatal): %s", _emb_err)
                    tracer.add_metadata(trace_id, emb_hop, output=f"failed: {_emb_err}")

            # 2c. Auto hippocampal replay — every REPLAY_EVERY_N_WRITES writes per namespace
            ns = req.namespace
            self._write_counts[ns] = self._write_counts.get(ns, 0) + 1
            if self._write_counts[ns] % self.REPLAY_EVERY_N_WRITES == 0:
                with tracer.span(trace_id, "engine.recall_long_tail", "phase_engine",
                                 namespace=ns, trigger="auto_n_writes"):
                    rehearsed = self.engine.recall_long_tail(ns, batch_size=50)
                    logger.debug("Auto recall_long_tail ns=%s writes=%d rehearsed=%d",
                                 ns, self._write_counts[ns], rehearsed)

            if phase_item is None:
                raise RuntimeError("PhaseMemoryEngine.store() returned None with no candidates")

            # 3. Convert to MemoryItem for API response
            with tracer.span(trace_id, "convert_item", "memory_service",
                             input=f"phase_item {str(phase_item.id)[:8]}…") as conv_hop:
                item = self._phase_to_item(phase_item, req)
                tracer.add_metadata(trace_id, conv_hop,
                                    output=f"level={item.store_level.value}  confidence={round(item.confidence, 3)}")
            tracer.add_metadata(trace_id, ms_write_hop,
                                output=f"item_id={str(item.id)[:8]}…  level={item.store_level.value}  strength={round(phase_item.consolidation_strength, 3)}")

            # 4. Persist to L1 PostgreSQL (fire-and-forget — never blocks)
            with tracer.span(trace_id, "l1.persist", "stores.l1",
                             item_id=item.id, namespace=req.namespace) as l1_hop:
                asyncio.create_task(self._persist_to_l1(item, trace_id=trace_id, parent_hop_id=l1_hop))

            # 5. Webhook (fire-and-forget)
            with tracer.span(trace_id, "webhook.dispatch", "webhook_dispatcher",
                             event="memory.created", item_id=item.id):
                self._dispatch_webhook("memory.created", item)

        return item

    # =========================================================================
    # Read
    # =========================================================================

    async def read(self, req: ReadRequest, trace_id: Optional[str] = None) -> ReadResponse:
        """Read from PhaseMemoryEngine — full thermodynamic retrieval."""
        trace_id = tracer.ensure_trace(trace_id or tracer.new_trace("read"), "read")

        with tracer.span(trace_id, "memory_service.read", "memory_service",
                         input=req.query[:200],
                         namespace=req.namespace, limit=req.limit) as ms_read_hop:

            # 1. Ensure namespace data is loaded
            with tracer.span(trace_id, "ensure_loaded", "memory_service",
                             namespace=req.namespace):
                await self.ensure_loaded(req.namespace)

            # 1b. Auto-parse temporal filter from query if not provided
            effective_tf = req.temporal_filter
            if effective_tf is None:
                effective_tf = parse_temporal_filter(req.query, datetime.now(timezone.utc))

            # 2. Search via PhaseMemoryEngine (in-memory, sub-ms)
            # Fetch ALL items in the namespace (up to 500) so the semantic re-ranker
            # can find vocabulary-gap matches ("job title" → "software engineer").
            # For temporal queries with date bounds, fetch everything so date-filter works.
            ns_size = len(self.engine._items.get(req.namespace, []))
            if effective_tf and (effective_tf.start or effective_tf.end):
                fetch_limit = max(500, ns_size)
            else:
                fetch_limit = max(req.limit * 2, req.limit + 20, min(500, ns_size))
            with tracer.span(trace_id, "engine.search", "phase_engine",
                             input=req.query[:200],
                             namespace=req.namespace, fetch_limit=fetch_limit) as search_hop:
                results = self.engine.search(req.query, req.namespace, limit=fetch_limit, trace_id=trace_id)
                if results:
                    top_text = (results[0][1].fact.raw_text or "")[:80]
                    tracer.add_metadata(trace_id, search_hop,
                                        output=f"{len(results)} results  top: {top_text}",
                                        results=len(results))
                else:
                    tracer.add_metadata(trace_id, search_hop, output="0 results", results=0)

            # 2a. Apply temporal date filter (before semantic re-rank)
            if effective_tf and (effective_tf.start or effective_tf.end):
                results = [
                    (s, pi) for s, pi in results
                    if self._event_at_in_range(getattr(pi, "event_at", None), effective_tf)
                ]

            # 2b. 384-dim semantic re-ranking (post-TRR layer)
            if len(results) > 1:
                with tracer.span(trace_id, "semantic_rerank", "embedding_service",
                                 input=req.query[:120],
                                 candidates=len(results)) as rerank_hop:
                    try:
                        query_emb = self.embedding_service.embed(req.query)
                        query_meaningful = self._meaningful_tokens(req.query)
                        top_text_tokens = set(_re.findall(r'\w+', (results[0][1].fact.raw_text or "").lower()))
                        ttr_is_relevant = bool(query_meaningful & top_text_tokens)
                        alpha = 0.4 if ttr_is_relevant else 0.85
                        results = self._semantic_rerank(results, query_emb, alpha=alpha)
                        top_text = (results[0][1].fact.raw_text or "")[:80] if results else ""
                        tracer.add_metadata(trace_id, rerank_hop,
                                            output=f"alpha={alpha}  top: {top_text}",
                                            alpha=alpha, ttr_relevant=ttr_is_relevant)
                    except Exception as _rr_err:
                        logger.debug("Semantic re-rank failed (falling back to TRR order): %s", _rr_err)
                        tracer.add_metadata(trace_id, rerank_hop, output=f"failed: {_rr_err}")

            # 2c. Recency decay blend (after semantic re-rank, before slicing)
            if effective_tf and effective_tf.recency_alpha > 0.05:
                results = self._apply_recency_decay(results, datetime.now(timezone.utc), effective_tf)

            results = results[:req.limit]

            # 3. Convert PhaseMemoryItems to MemoryItems
            with tracer.span(trace_id, "convert_results", "memory_service",
                             input=f"{len(results)} candidates  min_confidence={req.min_confidence}") as conv_hop:
                items = []
                for score, phase_item in results:
                    mem_item = self._phase_to_item(phase_item)
                    if mem_item.confidence >= req.min_confidence:
                        # Filter superseded items unless caller explicitly wants them
                        if not req.include_superseded and mem_item.superseded:
                            continue
                        items.append(mem_item)
                preview = " | ".join(i.text[:60] for i in items[:2]) if items else "no results"
                tracer.add_metadata(trace_id, conv_hop,
                                    output=f"{len(items)} items: {preview}")

            # 3b. Inject matching temporal event threads at the top of results.
            thread_items = self._matching_threads(req.query, req.namespace, effective_tf)
            if thread_items:
                items = thread_items + items

            preview = " | ".join(i.text[:70] for i in items[:3]) if items else "no results"
            tracer.add_metadata(trace_id, ms_read_hop,
                                output=f"{len(items)} items returned: {preview}")

        return ReadResponse(
            items=items,
            query=req.query,
            namespace=req.namespace,
            trace_id=trace_id,
        )

    # =========================================================================
    # Get / Delete
    # =========================================================================

    async def get_item(self, item_id: str, namespace: str, trace_id: Optional[str] = None) -> Optional[MemoryItem]:
        """Get single item by ID from PhaseMemoryEngine, fallback to L1."""
        phase_item = self.engine._item_by_id.get(item_id)
        if phase_item and phase_item.namespace == namespace:
            return self._phase_to_item(phase_item)
        # Fallback to L1 persistence
        try:
            return await self.l1.get_by_id(item_id, namespace)
        except Exception:
            return None

    async def delete(self, item_id: str, namespace: str, trace_id: Optional[str] = None) -> bool:
        """Delete from PhaseMemoryEngine + L1 persistence."""
        trace_id = tracer.ensure_trace(trace_id or tracer.new_trace("delete"), "delete")

        deleted = False
        with tracer.span(trace_id, "memory_service.delete", "memory_service",
                         item_id=item_id, namespace=namespace):

            # Delete from PhaseMemoryEngine (set strength to 0 → GC on next recompute)
            with tracer.span(trace_id, "engine.delete", "phase_engine", item_id=item_id) as eng_hop:
                phase_item = self.engine._item_by_id.get(item_id)
                if phase_item and phase_item.namespace == namespace:
                    phase_item.consolidation_strength = 0.0
                    self.engine._recompute_all_free_energies(namespace)
                    deleted = True
                tracer.add_metadata(trace_id, eng_hop, deleted=deleted)

            # Delete from L1 persistence
            with tracer.span(trace_id, "l1.delete", "stores.l1", item_id=item_id) as l1_hop:
                try:
                    l1_ok = await self.l1.delete(item_id, namespace)
                    deleted = deleted or l1_ok
                    tracer.add_metadata(trace_id, l1_hop, deleted=l1_ok)
                except Exception as e:
                    logger.warning("L1 delete failed: %s", e)

            if deleted and self._webhook_dispatcher:
                with tracer.span(trace_id, "webhook.dispatch", "webhook_dispatcher",
                                 event="memory.deleted", item_id=item_id):
                    try:
                        payload = {"id": item_id, "namespace": namespace}
                        asyncio.create_task(
                            self._webhook_dispatcher.dispatch("memory.deleted", payload, namespace)
                        )
                    except Exception:
                        pass

        return deleted

    # =========================================================================
    # Adjudicate — belief revision with evidence quorum (uses ReconsolidationGate)
    # =========================================================================

    async def adjudicate(
        self,
        new_text: str,
        namespace: str,
        evidence: list[str],
        existing_item_id: Optional[str] = None,
    ) -> MemoryItem:
        """
        Belief revision: store new fact only if evidence quorum is met.

        ReconsolidationGate checks: same topic? conflict? quorum of evidence?
        If approved: archive old, store new in PhaseMemoryEngine.
        """
        # Find existing item
        existing = None
        if existing_item_id:
            existing = await self.get_item(existing_item_id, namespace)

        # Build new MemoryItem for reconsolidation check
        new_item = MemoryItem(text=new_text, namespace=namespace)
        new_item = self.embedding_service.embed_item(new_item)

        if existing and existing.embedding:
            updated, archived, should_engrave = self.reconsolidation.prepare_for_reconsolidation(
                new_item, existing, evidence,
            )
            if not should_engrave:
                # Quorum not met — return existing unchanged
                return existing

        # Quorum met (or no existing) — store via PhaseMemoryEngine
        req = WriteRequest(
            text=new_text,
            namespace=namespace,
            source="adjudicate",
            authority=1.0,  # High authority for adjudicated facts
        )
        return await self.write(req)

    # =========================================================================
    # Multi-hop read (Issue #36)
    # =========================================================================

    async def read_multihop(
        self,
        req: ReadRequest,
        trace_id: Optional[str] = None,
    ) -> ReadResponse:
        """Issue #36: 2-pass multi-hop retrieval for chained-entity questions.

        Pass 1: Standard retrieval for req.query.
        Pass 2: Extract key entities from top pass-1 results, re-search with
                augmented query (original question + extracted entities).

        This chains through entity references that the original question doesn't
        mention explicitly — e.g. "What does Bob the doctor's employer do?" first
        retrieves Bob's employer (Stanford Medical), then uses that to retrieve
        facts about Stanford Medical.

        Returns deduplicated combined results, pass-1 first.
        """
        trace_id = tracer.ensure_trace(trace_id or tracer.new_trace("read_multihop"), "read_multihop")

        with tracer.span(trace_id, "memory_service.read_multihop", "memory_service",
                         input=req.query[:200], namespace=req.namespace) as ms_hop:

            # Ensure namespace loaded
            await self.ensure_loaded(req.namespace)

            # Pass 1: standard retrieval
            with tracer.span(trace_id, "engine.search.pass1", "phase_engine",
                             input=req.query[:200], namespace=req.namespace):
                results1 = self.engine.search(req.query, req.namespace, limit=req.limit * 2)

            # Extract key entities from top-3 pass-1 results
            top_texts = [item.fact.raw_text for _, item in results1[:3] if item.fact.raw_text]
            entities = self._extract_multihop_entities(" ".join(top_texts))

            pass2_count = 0
            if entities:
                # Pass 2: augmented query with extracted entities
                augmented = f"{req.query} {entities}"
                with tracer.span(trace_id, "engine.search.pass2", "phase_engine",
                                 input=augmented[:200], namespace=req.namespace):
                    results2 = self.engine.search(augmented, req.namespace, limit=req.limit * 2)

                # Merge: pass-1 first, then unique pass-2 results
                seen_ids = {item.id for _, item in results1}
                combined = list(results1)
                for score, item in results2:
                    if item.id not in seen_ids:
                        combined.append((score * 0.9, item))
                        seen_ids.add(item.id)
                        pass2_count += 1
                combined.sort(key=lambda x: x[0], reverse=True)
                results_final = combined[: req.limit]
            else:
                results_final = results1[: req.limit]

            tracer.add_metadata(trace_id, ms_hop,
                                output=f"pass1={len(results1)} pass2_added={pass2_count} total={len(results_final)}",
                                entities=entities[:100] if entities else "")

            # Semantic re-rank
            if len(results_final) > 1:
                try:
                    query_emb = self.embedding_service.embed(req.query)
                    results_final = self._semantic_rerank(results_final, query_emb, alpha=0.4)
                except Exception:
                    pass

            # Convert to MemoryItems
            items = []
            for _score, phase_item in results_final:
                mem_item = self._phase_to_item(phase_item)
                if mem_item.confidence >= req.min_confidence:
                    items.append(mem_item)

        return ReadResponse(
            items=items,
            query=req.query,
            namespace=req.namespace,
            trace_id=trace_id,
        )

    # Common stopwords for entity extraction
    _MH_STOPWORDS = frozenset({
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "a", "an", "the", "is", "was", "are", "were",
        "be", "been", "have", "has", "had", "do", "did", "will", "would",
        "could", "should", "may", "might", "to", "of", "in", "on", "at",
        "by", "for", "with", "and", "or", "but", "so", "yet", "if",
        "this", "that", "these", "those", "there", "here", "what", "when",
        "where", "who", "how", "why", "which", "go", "went", "get", "got",
        "s", "t", "re", "ll", "ve", "d", "work", "works", "live", "lives",
        "name", "called", "known",
    })

    def _extract_multihop_entities(self, text: str) -> str:
        """Extract key entities from retrieved text for multi-hop pass-2.

        Extracts proper nouns (capitalized mid-sentence tokens) and meaningful
        nouns/verbs (≥4 chars, non-stopword), then returns up to 8 space-joined.
        """
        # Proper nouns: capitalized words not at start of sentence
        proper = _re.findall(r'(?<=[.!?]\s)[A-Z][a-z]{2,}|(?<= )[A-Z][a-z]{2,}', text)
        # Meaningful tokens (≥4 chars, non-stopword)
        tokens = _re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        meaningful = [t for t in tokens if t not in self._MH_STOPWORDS]
        # Combine, deduplicate, limit
        entities = list(dict.fromkeys([p.lower() for p in proper] + meaningful))[:8]
        return " ".join(entities)

    # =========================================================================
    # Health
    # =========================================================================

    async def health(self) -> dict:
        """Aggregate health: PhaseMemoryEngine (always up) + L1/L2 (persistence)."""
        # PhaseMemoryEngine is always healthy (pure Python, no deps)
        engine_stats = {
            "status": "healthy",
            "store": "PhaseMemoryEngine",
            "total_items": self.engine._total_item_count,
            "namespaces": len(self.engine._items),
        }

        l1_h = {"status": "unknown", "store": "L1"}
        l2_h = {"status": "unknown", "store": "L2"}
        try:
            l1_h = await self.l1.health()
        except Exception:
            l1_h = {"status": "unhealthy", "store": "L1", "error": "Connection failed"}
        try:
            l2_h = await self.l2.health()
        except Exception:
            l2_h = {"status": "unhealthy", "store": "L2", "error": "Connection failed"}

        all_healthy = (
            engine_stats["status"] == "healthy"
            and l1_h.get("status") == "healthy"
        )
        return {
            "status": "healthy" if all_healthy else "degraded",
            "stores": {
                "engine": engine_stats,
                "L1": l1_h,
                "L2": l2_h,
            },
        }
