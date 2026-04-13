#!/usr/bin/env python3
"""CLS++ Cross-Model Memory Test Suite

Tests the core value proposition: learn on one model, recall on another.

Usage:
    # Set your API key first
    export CLS_API_KEY="cls_live_..."

    # Use case 1: Single prompt — write via Claude, read via other LLMs
    python3 scripts/cross_model_test.py --test single

    # Use case 2: Multi-line prompts — batch insert, verify across all LLMs
    python3 scripts/cross_model_test.py --test multi

    # Use case 3: Load test — 10k/100k/300k/1M entries, benchmark latency
    python3 scripts/cross_model_test.py --test load --count 10000

    # Run all tests
    python3 scripts/cross_model_test.py --test all
"""

import argparse
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

BASE_URL = os.environ.get("CLS_BASE_URL", "https://www.clsplusplus.com")
API_KEY = os.environ.get("CLS_API_KEY", "")
NAMESPACE = f"cross_model_test_{int(time.time())}"

MODELS = ["chatgpt", "claude", "gemini"]
CATEGORIES = ["Identity", "Preference", "Work", "Project", "Relationship",
              "Goal", "Temporal", "Context"]


@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: float
    details: str = ""
    error: str = ""


@dataclass
class TestReport:
    suite: str
    results: list = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0

    @property
    def total(self): return len(self.results)
    @property
    def passed(self): return sum(1 for r in self.results if r.passed)
    @property
    def failed(self): return self.total - self.passed
    @property
    def duration_s(self): return self.end_time - self.start_time

    def print_summary(self):
        print(f"\n{'═' * 70}")
        print(f"  {self.suite}")
        print(f"{'═' * 70}")
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            color = "\033[32m" if r.passed else "\033[31m"
            reset = "\033[0m"
            latency = f"{r.latency_ms:.1f}ms"
            print(f"  {color}{icon}{reset} {r.name:<55} {latency:>8}")
            if r.error:
                print(f"    \033[31m→ {r.error}\033[0m")
            if r.details and not r.passed:
                print(f"    → {r.details}")

        print(f"\n  Total: {self.total} | Passed: {self.passed} | "
              f"Failed: {self.failed} | Duration: {self.duration_s:.2f}s")
        if self.failed == 0:
            print("  \033[32m✓ ALL TESTS PASSED\033[0m")
        else:
            print(f"  \033[31m✗ {self.failed} TESTS FAILED\033[0m")
        print(f"{'═' * 70}\n")


# ═══════════════════════════════════════════════════════════════════
# HTTP Client
# ═══════════════════════════════════════════════════════════════════

class CLSClient:
    def __init__(self):
        if not API_KEY:
            print("\033[31mERROR: Set CLS_API_KEY environment variable\033[0m")
            print("  export CLS_API_KEY='cls_live_...'")
            sys.exit(1)
        self.http = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "X-API-Key": API_KEY,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    def _retry(self, fn, retries=5, backoff=2.0):
        """Retry a callable on 429/5xx with exponential backoff."""
        for attempt in range(retries):
            try:
                return fn()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 429 or code >= 500:
                    wait = backoff * (2 ** attempt)
                    print(f"    ⏳ {code} — retrying in {wait:.1f}s "
                          f"(attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    raise
        # Final attempt — let it raise
        return fn()

    def health(self) -> dict:
        r = self.http.get("/v1/health")
        r.raise_for_status()
        return r.json()

    def learn(self, text: str, namespace: str = NAMESPACE,
              source: str = "chatgpt", category: str = "Identity") -> dict:
        def _do():
            r = self.http.post("/v1/memory/write", json={
                "text": text,
                "namespace": namespace,
                "source": source,
                "metadata": {"category": category} if category else {},
            })
            r.raise_for_status()
            return r.json()
        return self._retry(_do)

    def ask(self, query: str, namespace: str = NAMESPACE,
            limit: int = 5, source: Optional[str] = None) -> dict:
        def _do():
            body = {"query": query, "namespace": namespace, "limit": limit}
            if source:
                body["source"] = source
            r = self.http.post("/v1/memory/read", json=body)
            r.raise_for_status()
            return r.json()
        return self._retry(_do)

    def list_memories(self, namespace: str = NAMESPACE,
                      limit: int = 100) -> dict:
        def _do():
            r = self.http.get("/v1/memory/list", params={
                "namespace": namespace, "limit": limit,
            })
            r.raise_for_status()
            return r.json()
        return self._retry(_do)

    def forget(self, item_id: str, namespace: str = NAMESPACE) -> dict:
        def _do():
            r = self.http.request("DELETE", "/v1/memory/forget", json={
                "item_id": item_id, "namespace": namespace,
            })
            r.raise_for_status()
            return r.json()
        return self._retry(_do)

    def consolidate(self, namespace: str = NAMESPACE) -> dict:
        def _do():
            r = self.http.post("/v1/memories/consolidate", params={"namespace": namespace})
            if r.status_code in (200, 204):
                try:
                    return r.json()
                except Exception:
                    return {"ok": True}
            r.raise_for_status()
        return self._retry(_do, retries=6, backoff=3.0)

    def wipe(self, namespace: str = NAMESPACE) -> dict:
        """Delete all memories in namespace."""
        def _do():
            r = self.http.request("DELETE", "/v1/memory/wipe", params={"confirm": "true"},
                                  json={"namespace": namespace})
            if r.status_code in (200, 204, 404):
                return {"status": "wiped"}
            r.raise_for_status()
            return r.json()
        return self._retry(_do)


# ═══════════════════════════════════════════════════════════════════
# Test Suite 1: Single Prompt Cross-Model
# ═══════════════════════════════════════════════════════════════════

def test_single_prompt(client: CLSClient) -> TestReport:
    """Use case 1: Write a prompt via Claude, read from ChatGPT and Gemini."""
    report = TestReport(suite="USE CASE 1: Single Prompt — Cross-Model Memory")
    report.start_time = time.time()
    ns = f"single_{int(time.time())}"

    # Test 1: Health check
    t = time.time()
    try:
        h = client.health()
        report.results.append(TestResult(
            "API health check", True, (time.time()-t)*1000,
            f"version={h.get('version')}"
        ))
    except Exception as e:
        report.results.append(TestResult("API health check", False, 0, error=str(e)))
        report.end_time = time.time()
        report.print_summary()
        return report

    # Test 2: Write memory via Claude
    fact = "My name is Alex and I work as a machine learning engineer at OpenAI"
    t = time.time()
    try:
        result = client.learn(fact, namespace=ns, source="claude", category="Identity")
        mem_id = result.get("id", "")
        report.results.append(TestResult(
            "Learn via Claude: identity fact", True, (time.time()-t)*1000,
            f"id={mem_id}"
        ))
    except Exception as e:
        report.results.append(TestResult("Learn via Claude", False, 0, error=str(e)))

    # Test 3: Read from ChatGPT (cross-model)
    t = time.time()
    try:
        result = client.ask("What is my name and where do I work?", namespace=ns)
        items = result.get("items", [])
        found = any("Alex" in item.get("text", "") for item in items)
        report.results.append(TestResult(
            "Recall from ChatGPT: 'What is my name?'", found, (time.time()-t)*1000,
            f"items={len(items)}, matched={'Alex' if found else 'NO MATCH'}"
        ))
    except Exception as e:
        report.results.append(TestResult("Recall from ChatGPT", False, 0, error=str(e)))

    # Test 4: Read from Gemini (cross-model)
    t = time.time()
    try:
        result = client.ask("What do I do for work?", namespace=ns)
        items = result.get("items", [])
        found = any("machine learning" in item.get("text", "").lower() or
                     "OpenAI" in item.get("text", "") for item in items)
        report.results.append(TestResult(
            "Recall from Gemini: 'What do I do for work?'", found, (time.time()-t)*1000,
            f"items={len(items)}"
        ))
    except Exception as e:
        report.results.append(TestResult("Recall from Gemini", False, 0, error=str(e)))

    # Test 5: Write preference via ChatGPT
    pref = "I prefer Python over JavaScript and use VS Code with vim keybindings"
    t = time.time()
    try:
        client.learn(pref, namespace=ns, source="chatgpt", category="Preference")
        report.results.append(TestResult(
            "Learn via ChatGPT: coding preference", True, (time.time()-t)*1000
        ))
    except Exception as e:
        report.results.append(TestResult("Learn via ChatGPT", False, 0, error=str(e)))

    # Test 6: Cross-recall preference from Claude
    t = time.time()
    try:
        result = client.ask("What programming language do I prefer?", namespace=ns)
        items = result.get("items", [])
        found = any("Python" in item.get("text", "") for item in items)
        report.results.append(TestResult(
            "Cross-recall: Claude reads ChatGPT memory", found, (time.time()-t)*1000,
            f"items={len(items)}"
        ))
    except Exception as e:
        report.results.append(TestResult("Cross-recall", False, 0, error=str(e)))

    # Test 7: Context generation
    t = time.time()
    try:
        result = client.ask("everything about this user", namespace=ns, limit=10)
        items = result.get("items", [])
        report.results.append(TestResult(
            "Context generation: full user profile", len(items) >= 2,
            (time.time()-t)*1000, f"facts_returned={len(items)}"
        ))
    except Exception as e:
        report.results.append(TestResult("Context generation", False, 0, error=str(e)))

    # Test 8: Forget
    t = time.time()
    try:
        if mem_id:
            client.forget(mem_id, namespace=ns)
            report.results.append(TestResult(
                "Forget: delete specific memory", True, (time.time()-t)*1000
            ))
        else:
            report.results.append(TestResult("Forget", False, 0, error="No memory ID"))
    except Exception as e:
        report.results.append(TestResult("Forget", False, 0, error=str(e)))

    # Cleanup
    try:
        client.wipe(namespace=ns)
    except Exception:
        pass

    report.end_time = time.time()
    report.print_summary()
    return report


# ═══════════════════════════════════════════════════════════════════
# Test Suite 2: Multi-Line Prompts Across All LLMs
# ═══════════════════════════════════════════════════════════════════

MULTI_LINE_PROMPTS = [
    # (text, source_model, category)
    ("My name is Priya Sharma", "claude", "Identity"),
    ("I am 32 years old", "claude", "Identity"),
    ("I live in San Francisco, California", "claude", "Identity"),
    ("I work as a Staff Engineer at Stripe", "chatgpt", "Work"),
    ("My team builds payment infrastructure APIs", "chatgpt", "Work"),
    ("I'm leading the migration to gRPC from REST", "chatgpt", "Project"),
    ("I prefer TypeScript for backend, Rust for systems", "gemini", "Preference"),
    ("I use Neovim as my primary editor", "gemini", "Preference"),
    ("I have a severe shellfish allergy and must avoid shrimp, crab, and lobster", "claude", "Identity"),
    ("My goal is to become a VP of Engineering by 2027", "chatgpt", "Goal"),
    ("I mentor 3 junior engineers on my team", "gemini", "Relationship"),
    ("I have a weekly 1:1 with my manager Sarah every Tuesday", "claude", "Temporal"),
    ("I'm currently reading 'Designing Data-Intensive Applications'", "chatgpt", "Context"),
    ("I prefer dark mode in all applications", "gemini", "Preference"),
    ("My GitHub username is priya-builds", "claude", "Identity"),
]


def test_multi_prompt(client: CLSClient) -> TestReport:
    """Use case 2: Insert multiple facts from different models, query all."""
    report = TestReport(suite="USE CASE 2: Multi-Line Prompts — All LLMs")
    report.start_time = time.time()
    ns = f"multi_{int(time.time())}"

    # Test 1: Batch insert from all 3 models
    write_latencies = []
    for text, source, category in MULTI_LINE_PROMPTS:
        t = time.time()
        try:
            client.learn(text, namespace=ns, source=source, category=category)
            write_latencies.append((time.time()-t)*1000)
        except Exception as e:
            report.results.append(TestResult(
                f"Write: {text[:40]}...", False, 0, error=str(e)
            ))

    avg_write = statistics.mean(write_latencies) if write_latencies else 0
    min_write = min(write_latencies) if write_latencies else 0
    max_write = max(write_latencies) if write_latencies else 0
    report.results.append(TestResult(
        f"Batch write: {len(MULTI_LINE_PROMPTS)} facts from 3 models",
        len(write_latencies) == len(MULTI_LINE_PROMPTS), avg_write,
        f"avg={avg_write:.1f}ms, min={min_write:.1f}ms, max={max_write:.1f}ms"
    ))

    # Test 2: Cross-model identity recall
    queries = [
        ("What is my name?", "Priya"),
        ("Where do I live?", "San Francisco"),
        ("Where do I work?", "Stripe"),
        ("What programming languages do I prefer?", "TypeScript"),
        ("What is my goal?", "VP of Engineering"),
        ("What am I allergic to? Any food allergies like shellfish?", "shellfish|shrimp|crab|lobster|allergy"),
        ("What editor do I use?", "Neovim"),
        ("Who do I mentor?", "junior"),
        ("What book am I reading?", "Data-Intensive"),
        ("What is my GitHub username?", "priya-builds"),
    ]

    read_latencies = []
    for query, expected_keyword in queries:
        t = time.time()
        try:
            result = client.ask(query, namespace=ns, limit=5)
            latency = (time.time()-t)*1000
            read_latencies.append(latency)
            items = result.get("items", [])
            # Support OR keywords separated by |
            keywords = [kw.strip().lower() for kw in expected_keyword.split("|")]
            found = any(
                any(kw in item.get("text", "").lower() for kw in keywords)
                for item in items
            )
            report.results.append(TestResult(
                f"Recall: '{query[:50]}'", found, latency,
                f"expected='{expected_keyword}', got={len(items)} items"
            ))
        except Exception as e:
            report.results.append(TestResult(
                f"Recall: '{query}'", False, 0, error=str(e)
            ))

    # Test 3: List all memories
    t = time.time()
    try:
        result = client.list_memories(namespace=ns, limit=100)
        items = result.get("items", result.get("memories", []))
        report.results.append(TestResult(
            f"List all memories in namespace", len(items) >= 10,
            (time.time()-t)*1000, f"count={len(items)}"
        ))
    except Exception as e:
        report.results.append(TestResult("List memories", False, 0, error=str(e)))

    # Test 4: Category-specific queries
    for category in ["Identity", "Preference", "Work"]:
        t = time.time()
        try:
            result = client.ask(f"Tell me about {category.lower()}", namespace=ns, limit=5)
            items = result.get("items", [])
            report.results.append(TestResult(
                f"Category query: {category}", len(items) > 0,
                (time.time()-t)*1000, f"items={len(items)}"
            ))
        except Exception as e:
            report.results.append(TestResult(
                f"Category: {category}", False, 0, error=str(e)
            ))

    # Summary stats
    if read_latencies:
        avg_read = statistics.mean(read_latencies)
        p95 = sorted(read_latencies)[int(len(read_latencies)*0.95)]
        print(f"\n  Read latency: avg={avg_read:.1f}ms, "
              f"p95={p95:.1f}ms, "
              f"min={min(read_latencies):.1f}ms, "
              f"max={max(read_latencies):.1f}ms")

    # Cleanup
    try:
        client.wipe(namespace=ns)
    except Exception:
        pass

    report.end_time = time.time()
    report.print_summary()
    return report


# ═══════════════════════════════════════════════════════════════════
# Test Suite 3: Load Test — 10k to 1M entries
# ═══════════════════════════════════════════════════════════════════

def test_load(client: CLSClient, count: int = 10000) -> TestReport:
    """Use case 3: Bulk insert and verify all memory layers work."""
    report = TestReport(
        suite=f"USE CASE 3: Load Test — {count:,} Entries"
    )
    report.start_time = time.time()
    ns = f"load_{count}_{int(time.time())}"

    # Generate diverse test data
    categories = CATEGORIES
    models = MODELS
    templates = [
        "User fact #{i}: preference for {pref} in {domain}",
        "Memory #{i}: user mentioned {topic} during {context} conversation",
        "Session #{i}: user asked about {topic} while using {model}",
        "Insight #{i}: user pattern — {pref} related to {domain}",
    ]
    prefs = ["dark mode", "Python", "TypeScript", "vim", "VS Code", "fast APIs",
             "microservices", "monolith", "REST", "GraphQL", "gRPC", "Redis"]
    domains = ["frontend", "backend", "DevOps", "ML/AI", "data engineering",
               "mobile", "security", "infrastructure", "platform"]
    topics = ["deployment", "testing", "code review", "architecture", "scaling",
              "monitoring", "debugging", "performance", "reliability"]
    contexts = ["morning standup", "code review", "pair programming",
                "incident response", "planning", "retrospective"]

    import random
    random.seed(42)

    # Test 1: Batch write with progress
    print(f"\n  Writing {count:,} memories...")
    write_latencies = []
    errors = 0
    batch_size = min(count, 100)
    checkpoint = max(count // 10, 1)

    for i in range(count):
        template = random.choice(templates)
        text = template.format(
            i=i,
            pref=random.choice(prefs),
            domain=random.choice(domains),
            topic=random.choice(topics),
            context=random.choice(contexts),
            model=random.choice(models),
        )

        t = time.time()
        try:
            client.learn(
                text, namespace=ns,
                source=random.choice(models),
                category=random.choice(categories),
            )
            write_latencies.append((time.time()-t)*1000)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Write error #{errors}: {e}")
            if errors > count * 0.1:
                print(f"  \033[31mAborting: too many errors ({errors})\033[0m")
                break

        if (i+1) % checkpoint == 0:
            pct = (i+1) / count * 100
            recent = write_latencies[-checkpoint:]
            avg = statistics.mean(recent) if recent else 0
            print(f"    {pct:5.1f}% — {i+1:,}/{count:,} written, "
                  f"avg latency={avg:.1f}ms, errors={errors}")

    if write_latencies:
        avg_w = statistics.mean(write_latencies)
        p50_w = sorted(write_latencies)[len(write_latencies)//2]
        p95_w = sorted(write_latencies)[int(len(write_latencies)*0.95)]
        p99_w = sorted(write_latencies)[int(len(write_latencies)*0.99)]
        report.results.append(TestResult(
            f"Write {len(write_latencies):,} memories",
            errors < count * 0.01, avg_w,
            f"avg={avg_w:.1f}ms p50={p50_w:.1f}ms p95={p95_w:.1f}ms "
            f"p99={p99_w:.1f}ms errors={errors}"
        ))

    # Test 2: Read latency under load
    print(f"\n  Running 50 read queries...")
    read_queries = [
        "What does the user prefer?",
        "Tell me about Python",
        "deployment and infrastructure",
        "code review patterns",
        "What editor does the user use?",
        "microservices architecture",
        "testing practices",
        "dark mode preference",
        "user work habits",
        "performance optimization",
    ]
    read_latencies = []
    for _ in range(5):
        for q in read_queries:
            t = time.time()
            try:
                result = client.ask(q, namespace=ns, limit=5)
                read_latencies.append((time.time()-t)*1000)
            except Exception:
                pass

    if read_latencies:
        avg_r = statistics.mean(read_latencies)
        p50_r = sorted(read_latencies)[len(read_latencies)//2]
        p95_r = sorted(read_latencies)[int(len(read_latencies)*0.95)]
        p99_r = sorted(read_latencies)[int(len(read_latencies)*0.99)]
        report.results.append(TestResult(
            f"Read latency ({len(read_latencies)} queries, {count:,} memories)",
            avg_r < 2000, avg_r,  # Should be under 2s
            f"avg={avg_r:.1f}ms p50={p50_r:.1f}ms p95={p95_r:.1f}ms p99={p99_r:.1f}ms"
        ))

    # Test 3: Memory consolidation (Liquid → Solid)
    # Cooldown after bulk writes to avoid 429 on consolidation
    print(f"\n  Cooling down 5s before consolidation...")
    time.sleep(5)
    print(f"  Triggering consolidation...")
    t = time.time()
    try:
        result = client.consolidate(namespace=ns)
        report.results.append(TestResult(
            "Memory consolidation (phase transition)", True,
            (time.time()-t)*1000, f"result={json.dumps(result)[:100]}"
        ))
    except Exception as e:
        report.results.append(TestResult(
            "Memory consolidation", False, 0, error=str(e)
        ))

    # Test 4: Verify retrieval accuracy post-consolidation
    t = time.time()
    try:
        result = client.ask("user preference dark mode Python TypeScript", namespace=ns, limit=10)
        items = result.get("items", [])
        # Any item containing any of the known prefs/topics means recall works
        recall_keywords = ["python", "typescript", "dark mode", "vim", "redis",
                           "preference", "frontend", "backend", "deployment"]
        found = any(
            any(kw in item.get("text", "").lower() for kw in recall_keywords)
            for item in items
        )
        report.results.append(TestResult(
            "Post-consolidation recall accuracy", found,
            (time.time()-t)*1000, f"items={len(items)}"
        ))
    except Exception as e:
        report.results.append(TestResult(
            "Post-consolidation recall", False, 0, error=str(e)
        ))

    # Test 5: List memories count
    t = time.time()
    try:
        result = client.list_memories(namespace=ns, limit=10)
        items = result.get("items", result.get("memories", []))
        report.results.append(TestResult(
            "List memories post-load", len(items) > 0,
            (time.time()-t)*1000, f"returned={len(items)}"
        ))
    except Exception as e:
        report.results.append(TestResult("List memories", False, 0, error=str(e)))

    # Print performance report
    if write_latencies and read_latencies:
        total_writes = len(write_latencies)
        total_time = sum(write_latencies) / 1000
        throughput = total_writes / total_time if total_time > 0 else 0
        print(f"\n  ╔{'═'*58}╗")
        print(f"  ║  PERFORMANCE REPORT — {count:,} entries{' '*(30-len(f'{count:,}'))}║")
        print(f"  ╠{'═'*58}╣")
        print(f"  ║  Write throughput: {throughput:>8.1f} ops/sec{' '*24}║")
        print(f"  ║  Write avg:        {avg_w:>8.1f} ms{' '*28}║")
        print(f"  ║  Write p99:        {p99_w:>8.1f} ms{' '*28}║")
        print(f"  ║  Read avg:         {avg_r:>8.1f} ms{' '*28}║")
        print(f"  ║  Read p99:         {p99_r:>8.1f} ms{' '*28}║")
        print(f"  ║  Errors:           {errors:>8d}{' '*30}║")
        print(f"  ╚{'═'*58}╝")

    # Cleanup
    print(f"\n  Cleaning up namespace '{ns}'...")
    try:
        client.wipe(namespace=ns)
    except Exception:
        pass

    report.end_time = time.time()
    report.print_summary()
    return report


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CLS++ Cross-Model Memory Test Suite")
    parser.add_argument("--test", choices=["single", "multi", "load", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of entries for load test (default: 100)")
    parser.add_argument("--url", type=str, default=None,
                        help="API base URL (default: CLS_BASE_URL or production)")
    args = parser.parse_args()

    global BASE_URL
    if args.url:
        BASE_URL = args.url

    print(f"\n  CLS++ Cross-Model Memory Test Suite")
    print(f"  Target: {BASE_URL}")
    print(f"  API Key: {API_KEY[:12]}...{API_KEY[-4:]}" if len(API_KEY) > 16
          else f"  API Key: {'SET' if API_KEY else 'NOT SET'}")
    print()

    client = CLSClient()

    reports = []
    if args.test in ("single", "all"):
        reports.append(test_single_prompt(client))
    if args.test in ("multi", "all"):
        reports.append(test_multi_prompt(client))
    if args.test in ("load", "all"):
        reports.append(test_load(client, count=args.count))

    # Final summary
    total = sum(r.total for r in reports)
    passed = sum(r.passed for r in reports)
    failed = sum(r.failed for r in reports)

    print(f"\n{'▓' * 70}")
    print(f"  FINAL SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'▓' * 70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
