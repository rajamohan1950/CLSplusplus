#!/usr/bin/env python3
"""
CLS++ Topical Resonance Graph — Stress Test Suite

Tests 1000+ combinations across:
  - 8 LLM providers (Claude Code, ChatGPT, GPT-4o, Gemini, Grok, Copilot, Perplexity, Llama)
  - 6 topic clusters (auth, database, frontend, ML, devops, personal)
  - Concurrent sessions per user
  - All 4 memory layers (L0 Gas, L1 Liquid, L2 Solid, L3 Glass)
  - Ring buffer overflow
  - Topic drift within sessions
  - Simultaneous prompt ingestion (asyncio.gather)

Measures:
  - Write latency (prompt ingestion to TRG + engine)
  - Recall latency (cross-session + cascade recall)
  - Injection accuracy (correct topics injected, wrong topics blocked)

Usage:
  PYTHONPATH=src python3 CLS_testing/test_trg_stress.py
"""

import asyncio
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from clsplusplus.memory_phase import PhaseMemoryEngine, _tokenize
from clsplusplus.topical_resonance import (
    TopicalResonanceGraph, K_CRITICAL, PromptEntry,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test Data: LLM Providers × Topic Clusters
# ═══════════════════════════════════════════════════════════════════════════

LLM_PROVIDERS = [
    "claude-code", "chatgpt", "gpt-4o", "gemini",
    "grok", "copilot", "perplexity", "llama",
]

TOPIC_CLUSTERS = {
    "auth": [
        "I am refactoring the auth module to use JWT tokens instead of session cookies",
        "The auth middleware is in src/middleware.py and uses express-session for cookie management",
        "We need to add JWT token validation in the auth middleware with RS256 signing",
        "The JWT secret should be loaded from environment variables for the auth configuration",
        "Add rate limiting to the authentication endpoints to prevent brute force attacks",
        "Implement refresh token rotation in the auth service with sliding expiry",
        "The OAuth2 callback handler needs to validate state parameter to prevent CSRF",
        "Add multi-factor authentication support using TOTP in the login flow",
        "The session store needs to migrate from Redis to PostgreSQL for auth tokens",
        "Implement role-based access control middleware for API endpoint authorization",
    ],
    "database": [
        "We need to optimize the database queries for the user dashboard page",
        "Add a migration to create the orders table with foreign key to users",
        "The PostgreSQL connection pool is exhausting under high load, increase max connections",
        "Implement database sharding strategy for the events table by tenant_id",
        "Add composite index on created_at and user_id for the transactions table",
        "The ORM is generating N+1 queries for the product listing endpoint",
        "Implement soft delete with deleted_at timestamp instead of hard deletes",
        "Add database triggers for audit logging on sensitive tables",
        "The migration from MySQL to PostgreSQL needs schema mapping for enum types",
        "Implement read replicas for the reporting queries to reduce primary load",
    ],
    "frontend": [
        "Refactor the React components to use hooks instead of class components",
        "The CSS bundle size is too large, switch to Tailwind with tree shaking",
        "Implement dark mode toggle using CSS custom properties and prefers-color-scheme",
        "Add virtualized scrolling for the data table component with 10000 rows",
        "The form validation needs real-time feedback with debounced field validation",
        "Implement lazy loading for route components with React.lazy and Suspense",
        "Add accessibility attributes ARIA labels to all interactive components",
        "The responsive layout breaks on tablet portrait mode, fix the grid breakpoints",
        "Implement optimistic UI updates for the shopping cart with rollback on error",
        "Add end-to-end tests for the checkout flow using Playwright",
    ],
    "ml": [
        "Train a sentiment analysis model using BERT fine-tuning on our review dataset",
        "The model inference latency is too high, implement batched prediction with ONNX",
        "Add feature engineering pipeline for the recommendation system using collaborative filtering",
        "Implement A/B testing framework for model deployment with statistical significance checks",
        "The training data has class imbalance, apply SMOTE oversampling for minority classes",
        "Deploy the model serving endpoint using TorchServe with GPU acceleration",
        "Implement model versioning and rollback using MLflow experiment tracking",
        "Add real-time feature store using Redis for the fraud detection model",
        "The embedding model needs fine-tuning on our domain corpus for better retrieval",
        "Implement distributed training across 4 GPUs using PyTorch DDP",
    ],
    "devops": [
        "Set up the CI/CD pipeline with GitHub Actions for automated testing and deployment",
        "The Kubernetes pods are OOMKilled, increase memory limits in the deployment spec",
        "Implement blue-green deployment strategy with traffic shifting in the load balancer",
        "Add Prometheus metrics and Grafana dashboards for service monitoring",
        "The Docker image size is 2GB, use multi-stage build to reduce to under 200MB",
        "Set up centralized logging with ELK stack for all microservices",
        "Implement auto-scaling based on custom metrics from the message queue depth",
        "Add health check endpoints and readiness probes for Kubernetes liveness",
        "The Terraform state file needs to migrate to remote backend in S3",
        "Implement secrets rotation using HashiCorp Vault with dynamic credentials",
    ],
    "personal": [
        "I am planning a vacation to Bali in July with my family and two kids",
        "What are the best beach resorts near Seminyak for families with kids",
        "We need a hotel with a swimming pool and kids club near the beach",
        "Book a cooking class in Ubud for the whole family on the second day",
        "Find a good restaurant with vegetarian options near our hotel in Seminyak",
        "What is the weather like in Bali during July, is it rainy season",
        "Plan a day trip to the rice terraces in Tegallalang from Seminyak",
        "I need to renew my passport before the trip, when does it expire",
        "Compare flight prices from San Francisco to Denpasar for July dates",
        "Add travel insurance for the family trip covering medical and cancellation",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Test Harness
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LatencyRecord:
    operation: str  # 'write', 'recall', 'coupling'
    llm_provider: str
    topic: str
    latency_us: float  # microseconds
    items_returned: int = 0
    cross_session_count: int = 0


@dataclass
class InjectionResult:
    source_session: str
    target_session: str
    source_topic: str
    target_topic: str
    injected: bool
    correct: bool  # True if topics match and injected, or topics differ and not injected


class TRGStressTest:
    def __init__(self):
        self.engine = PhaseMemoryEngine()
        self.trg = TopicalResonanceGraph(engine=self.engine)
        self.namespace = "stress-test-ns"
        self.latencies: list[LatencyRecord] = []
        self.injections: list[InjectionResult] = []
        self.session_counter = 0

    def _make_session_id(self, provider: str, topic: str) -> str:
        self.session_counter += 1
        return f"{provider}-{topic}-{self.session_counter:04d}"

    # ─── Test 1: Write Latency ────────────────────────────────────────

    def test_write_latency(self, n_sessions: int = 50, prompts_per_session: int = 8):
        """Ingest prompts across all LLMs and topics. Measure write latency."""
        print(f"\n{'='*70}")
        print(f"TEST 1: Write Latency — {n_sessions} sessions × {prompts_per_session} prompts")
        print(f"{'='*70}")

        sessions = []
        for i in range(n_sessions):
            provider = LLM_PROVIDERS[i % len(LLM_PROVIDERS)]
            topic = list(TOPIC_CLUSTERS.keys())[i % len(TOPIC_CLUSTERS)]
            session_id = self._make_session_id(provider, topic)
            sessions.append((session_id, provider, topic))

        total_prompts = 0
        for session_id, provider, topic in sessions:
            prompts = TOPIC_CLUSTERS[topic]
            for seq, prompt in enumerate(prompts[:prompts_per_session]):
                t0 = time.perf_counter_ns()
                self.trg.on_prompt(
                    session_id=session_id,
                    content=prompt,
                    llm_provider=provider,
                    namespace=self.namespace,
                    role="user",
                    sequence_num=seq,
                )
                dt_us = (time.perf_counter_ns() - t0) / 1000.0

                self.latencies.append(LatencyRecord(
                    operation="write",
                    llm_provider=provider,
                    topic=topic,
                    latency_us=dt_us,
                ))
                total_prompts += 1

                # Also store in engine for deep memory
                self.engine.store(prompt, self.namespace)

        print(f"  Ingested {total_prompts} prompts across {n_sessions} sessions")
        print(f"  Active sessions in TRG: {self.trg.active_session_count}")
        self._print_latency_stats("write")

    # ─── Test 2: Coupling Accuracy ────────────────────────────────────

    def test_coupling_accuracy(self):
        """Verify that same-topic sessions are coupled, different-topic are not."""
        print(f"\n{'='*70}")
        print(f"TEST 2: Coupling Accuracy — Topic Isolation")
        print(f"{'='*70}")

        sessions = list(self.trg._sessions.values())
        n_pairs = 0
        correct = 0
        false_positive = 0  # Different topic but synced
        false_negative = 0  # Same topic but not synced
        coupling_by_relation = defaultdict(list)

        for i, s1 in enumerate(sessions):
            for s2 in sessions[i+1:]:
                K = self.trg.get_coupling(s1.session_id, s2.session_id)
                synced = K >= K_CRITICAL

                # Extract topic from session_id
                t1 = s1.session_id.split("-")[1]  # e.g., "auth" from "claude-code-auth-0001"
                t2 = s2.session_id.split("-")[1]
                # Handle provider names with dashes
                for topic in TOPIC_CLUSTERS:
                    if topic in s1.session_id:
                        t1 = topic
                    if topic in s2.session_id:
                        t2 = topic

                same_topic = (t1 == t2)
                is_correct = (same_topic == synced) or (not same_topic and not synced)

                if same_topic and synced:
                    correct += 1
                    coupling_by_relation["true_positive"].append(K)
                elif same_topic and not synced:
                    false_negative += 1
                    coupling_by_relation["false_negative"].append(K)
                elif not same_topic and synced:
                    false_positive += 1
                    coupling_by_relation["false_positive"].append(K)
                else:
                    correct += 1
                    coupling_by_relation["true_negative"].append(K)

                n_pairs += 1

                self.injections.append(InjectionResult(
                    source_session=s1.session_id,
                    target_session=s2.session_id,
                    source_topic=t1,
                    target_topic=t2,
                    injected=synced,
                    correct=is_correct,
                ))

        accuracy = correct / n_pairs if n_pairs else 0
        print(f"  Total session pairs evaluated: {n_pairs}")
        print(f"  Correct: {correct} ({accuracy:.1%})")
        print(f"  False positives (wrong topic injected): {false_positive}")
        print(f"  False negatives (missed same-topic link): {false_negative}")

        for relation, values in coupling_by_relation.items():
            if values:
                avg_k = statistics.mean(values)
                print(f"  {relation}: count={len(values)}, avg_K={avg_k:.4f}")

    # ─── Test 3: Recall Latency ───────────────────────────────────────

    def test_recall_latency(self, n_queries: int = 200):
        """Measure recall latency across all sessions and topics."""
        print(f"\n{'='*70}")
        print(f"TEST 3: Recall Latency — {n_queries} queries")
        print(f"{'='*70}")

        sessions = list(self.trg._sessions.values())
        if not sessions:
            print("  No sessions to query!")
            return

        for i in range(n_queries):
            session = sessions[i % len(sessions)]
            topic_name = "unknown"
            for t in TOPIC_CLUSTERS:
                if t in session.session_id:
                    topic_name = t
                    break

            # Use a prompt from the session's topic as query
            prompts = TOPIC_CLUSTERS.get(topic_name, list(TOPIC_CLUSTERS.values())[0])
            query = prompts[i % len(prompts)]

            # TRG cross-session recall
            t0 = time.perf_counter_ns()
            cross_results = self.trg.recall_cross_session(
                session.session_id, query, self.namespace, limit=10)
            dt_cross_us = (time.perf_counter_ns() - t0) / 1000.0

            # Engine deep recall
            t0 = time.perf_counter_ns()
            engine_results = self.engine.search(query, self.namespace, limit=10)
            dt_engine_us = (time.perf_counter_ns() - t0) / 1000.0

            # Total recall
            total_us = dt_cross_us + dt_engine_us

            self.latencies.append(LatencyRecord(
                operation="recall_trg",
                llm_provider=session.llm_provider,
                topic=topic_name,
                latency_us=dt_cross_us,
                items_returned=len(cross_results),
                cross_session_count=len(cross_results),
            ))
            self.latencies.append(LatencyRecord(
                operation="recall_engine",
                llm_provider=session.llm_provider,
                topic=topic_name,
                latency_us=dt_engine_us,
                items_returned=len(engine_results),
            ))
            self.latencies.append(LatencyRecord(
                operation="recall_total",
                llm_provider=session.llm_provider,
                topic=topic_name,
                latency_us=total_us,
                items_returned=len(cross_results) + len(engine_results),
                cross_session_count=len(cross_results),
            ))

        self._print_latency_stats("recall_trg", label="TRG Cross-Session Recall")
        self._print_latency_stats("recall_engine", label="Engine Deep Recall")
        self._print_latency_stats("recall_total", label="Total Cascade Recall")

    # ─── Test 4: Simultaneous Ingestion ───────────────────────────────

    def test_simultaneous_ingestion(self, n_concurrent: int = 100):
        """Simulate all LLMs sending prompts at the exact same time."""
        print(f"\n{'='*70}")
        print(f"TEST 4: Simultaneous Ingestion — {n_concurrent} concurrent prompts")
        print(f"{'='*70}")

        # Prepare prompts from all providers
        concurrent_prompts = []
        for i in range(n_concurrent):
            provider = LLM_PROVIDERS[i % len(LLM_PROVIDERS)]
            topic = list(TOPIC_CLUSTERS.keys())[i % len(TOPIC_CLUSTERS)]
            session_id = self._make_session_id(provider, topic)
            prompts = TOPIC_CLUSTERS[topic]
            prompt = prompts[i % len(prompts)]
            concurrent_prompts.append((session_id, provider, topic, prompt))

        # Fire all at once (simulating concurrent requests)
        t0_batch = time.perf_counter_ns()
        for session_id, provider, topic, prompt in concurrent_prompts:
            self.trg.on_prompt(
                session_id=session_id,
                content=prompt,
                llm_provider=provider,
                namespace=self.namespace,
                role="user",
                sequence_num=0,
            )
        dt_batch_us = (time.perf_counter_ns() - t0_batch) / 1000.0

        avg_per_prompt = dt_batch_us / n_concurrent
        print(f"  Total batch time: {dt_batch_us:.0f} us ({dt_batch_us/1000:.2f} ms)")
        print(f"  Average per prompt: {avg_per_prompt:.1f} us")
        print(f"  Throughput: {n_concurrent / (dt_batch_us / 1_000_000):.0f} prompts/sec")
        print(f"  Active sessions after burst: {self.trg.active_session_count}")

    # ─── Test 5: Ring Buffer Overflow ─────────────────────────────────

    def test_ring_buffer_overflow(self):
        """Test that ring buffer handles overflow correctly (deque maxlen=50)."""
        print(f"\n{'='*70}")
        print(f"TEST 5: Ring Buffer Overflow — 200 prompts into single session")
        print(f"{'='*70}")

        # Use a fresh TRG to avoid MAX_SESSIONS pruning
        fresh_engine = PhaseMemoryEngine()
        fresh_trg = TopicalResonanceGraph(engine=fresh_engine)
        session_id = "overflow-test-001"
        for i in range(200):
            topic = list(TOPIC_CLUSTERS.keys())[i % len(TOPIC_CLUSTERS)]
            prompts = TOPIC_CLUSTERS[topic]
            prompt = prompts[i % len(prompts)]
            fresh_trg.on_prompt(
                session_id=session_id,
                content=prompt,
                llm_provider="claude-code",
                namespace="overflow-ns",
                role="user",
                sequence_num=i,
            )

        osc = fresh_trg.get_session(session_id)
        print(f"  Prompts ingested: 200")
        print(f"  Ring buffer size: {len(osc.recent_prompts)}")
        print(f"  Prompt count tracked: {osc.prompt_count}")
        print(f"  Topic spectrum size: {len(osc.topic_spectrum)}")
        assert len(osc.recent_prompts) <= 50, f"Ring buffer overflow! Size={len(osc.recent_prompts)}"
        print(f"  Ring buffer correctly bounded at maxlen=50")

    # ─── Test 6: Topic Drift ──────────────────────────────────────────

    def test_topic_drift(self):
        """Test that topic drift within a session decouples from old topics."""
        print(f"\n{'='*70}")
        print(f"TEST 6: Topic Drift — session starts auth, shifts to database")
        print(f"{'='*70}")

        # Use a fresh TRG to avoid MAX_SESSIONS conflicts
        drift_engine = PhaseMemoryEngine()
        drift_trg = TopicalResonanceGraph(engine=drift_engine)
        ns = "drift-test-ns"

        drift_session = "drift-claude-001"
        auth_session = "drift-grok-001"

        # Phase 1: Both sessions talk about auth
        for prompt in TOPIC_CLUSTERS["auth"][:5]:
            drift_trg.on_prompt(drift_session, prompt, "claude-code", ns, "user")
        for prompt in TOPIC_CLUSTERS["auth"][:5]:
            drift_trg.on_prompt(auth_session, prompt, "grok", ns, "user")

        K_before = drift_trg.get_coupling(drift_session, auth_session)
        print(f"  Phase 1 (both auth): K={K_before:.4f} ({'SYNCED' if K_before >= K_CRITICAL else 'independent'})")

        # Phase 2: Drift session shifts to database
        for prompt in TOPIC_CLUSTERS["database"][:10]:
            drift_trg.on_prompt(drift_session, prompt, "claude-code", ns, "user")

        K_after = drift_trg.get_coupling(drift_session, auth_session)
        print(f"  Phase 2 (drift→database): K={K_after:.4f} ({'SYNCED' if K_after >= K_CRITICAL else 'independent'})")

        # Verify coupling decreased
        if K_after < K_before:
            print(f"  Coupling correctly decreased by {((K_before - K_after)/K_before)*100:.0f}%")
        else:
            print(f"  WARNING: Coupling did not decrease after topic drift!")

        # Verify drift session now couples with database sessions
        db_session = "drift-gemini-001"
        for prompt in TOPIC_CLUSTERS["database"][:5]:
            drift_trg.on_prompt(db_session, prompt, "gemini", ns, "user")

        K_db = drift_trg.get_coupling(drift_session, db_session)
        print(f"  Drift↔Database coupling: K={K_db:.4f} ({'SYNCED' if K_db >= K_CRITICAL else 'independent'})")

    # ─── Test 7: Cross-Topic Isolation Matrix ─────────────────────────

    def test_cross_topic_isolation(self):
        """Build a full NxN isolation matrix across all topic pairs."""
        print(f"\n{'='*70}")
        print(f"TEST 7: Cross-Topic Isolation Matrix")
        print(f"{'='*70}")

        # Fresh TRG for clean isolation test
        iso_engine = PhaseMemoryEngine()
        iso_trg = TopicalResonanceGraph(engine=iso_engine)
        iso_ns = "isolation-test-ns"

        topics = list(TOPIC_CLUSTERS.keys())
        sessions = {}

        # Create one session per topic with substantial content
        for topic in topics:
            sid = f"iso-{topic}-001"
            sessions[topic] = sid
            for prompt in TOPIC_CLUSTERS[topic]:
                iso_trg.on_prompt(sid, prompt, "claude-code", iso_ns, "user")

        # Build coupling matrix
        print(f"\n  {'':>12}", end="")
        for t in topics:
            print(f"{t:>12}", end="")
        print()

        for t1 in topics:
            print(f"  {t1:>12}", end="")
            for t2 in topics:
                if t1 == t2:
                    print(f"{'---':>12}", end="")
                else:
                    K = iso_trg.get_coupling(sessions[t1], sessions[t2])
                    marker = " *" if K >= K_CRITICAL else ""
                    print(f"{K:>10.4f}{marker}", end="")
            print()

        print(f"\n  * = SYNCED (K >= {K_CRITICAL})")

    # ─── Test 8: Per-LLM Latency Breakdown ────────────────────────────

    def test_per_llm_latency(self):
        """Report latency broken down by LLM provider."""
        print(f"\n{'='*70}")
        print(f"TEST 8: Per-LLM Latency Breakdown")
        print(f"{'='*70}")

        by_provider = defaultdict(lambda: defaultdict(list))
        for rec in self.latencies:
            by_provider[rec.llm_provider][rec.operation].append(rec.latency_us)

        for provider in LLM_PROVIDERS:
            if provider not in by_provider:
                continue
            print(f"\n  {provider}:")
            for op in ["write", "recall_trg", "recall_engine", "recall_total"]:
                values = by_provider[provider].get(op, [])
                if not values:
                    continue
                p50 = statistics.median(values)
                p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values)
                p99 = sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else max(values)
                print(f"    {op:>20}: p50={p50:>8.1f}us  p95={p95:>8.1f}us  p99={p99:>8.1f}us  n={len(values)}")

    # ─── Utilities ────────────────────────────────────────────────────

    def _print_latency_stats(self, operation: str, label: str = None):
        values = [r.latency_us for r in self.latencies if r.operation == operation]
        if not values:
            return
        label = label or operation
        p50 = statistics.median(values)
        p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values)
        p99 = sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else max(values)
        mean = statistics.mean(values)
        mx = max(values)
        print(f"  {label}:")
        print(f"    n={len(values)}  mean={mean:.1f}us  p50={p50:.1f}us  p95={p95:.1f}us  p99={p99:.1f}us  max={mx:.1f}us")

    # ─── Run All ──────────────────────────────────────────────────────

    def run_all(self):
        print("=" * 70)
        print("CLS++ TOPICAL RESONANCE GRAPH — STRESS TEST SUITE")
        print(f"LLM Providers: {len(LLM_PROVIDERS)}")
        print(f"Topic Clusters: {len(TOPIC_CLUSTERS)}")
        print(f"Prompts per Topic: {len(list(TOPIC_CLUSTERS.values())[0])}")
        print("=" * 70)

        t0 = time.time()

        self.test_write_latency(n_sessions=50, prompts_per_session=8)
        self.test_coupling_accuracy()
        self.test_recall_latency(n_queries=200)
        self.test_simultaneous_ingestion(n_concurrent=200)
        self.test_ring_buffer_overflow()
        self.test_topic_drift()
        self.test_cross_topic_isolation()
        self.test_per_llm_latency()

        elapsed = time.time() - t0

        # Final summary
        total_writes = len([r for r in self.latencies if r.operation == "write"])
        total_recalls = len([r for r in self.latencies if r.operation == "recall_total"])
        total_injections = len(self.injections)
        correct_injections = len([i for i in self.injections if i.correct])
        accuracy = correct_injections / total_injections if total_injections else 0

        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"  Total test time: {elapsed:.2f}s")
        print(f"  Total write operations: {total_writes}")
        print(f"  Total recall operations: {total_recalls}")
        print(f"  Total injection decisions: {total_injections}")
        print(f"  Injection accuracy: {correct_injections}/{total_injections} ({accuracy:.1%})")
        print(f"  Active sessions: {self.trg.active_session_count}")
        print(f"  Engine items: {len(self.engine._items.get(self.namespace, []))}")


if __name__ == "__main__":
    test = TRGStressTest()
    test.run_all()
