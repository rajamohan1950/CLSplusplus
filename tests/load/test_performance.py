"""Performance tests — latency budgets for hot endpoints.

These are standalone pytest runs against the live stack (no locust needed).
They assert p95 latency under a bound across a small warm-up + measurement
window, so they're fast enough to run on every CI pass — separately from
the full load suite in locustfile.py.
"""
from __future__ import annotations

import os
import statistics
import time

import httpx
import pytest


API_URL = os.environ.get("CLS_TEST_API_URL", "http://localhost:18080")
pytestmark = [pytest.mark.performance, pytest.mark.blackbox]


def _sample(client: httpx.Client, method: str, path: str, n: int = 40) -> list[float]:
    latencies: list[float] = []
    # Warm up: 5 requests, drop them
    for _ in range(5):
        client.request(method, path)
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.request(method, path)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if r.status_code < 500:
            latencies.append(elapsed)
    return latencies


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=API_URL, timeout=5.0) as c:
        yield c


@pytest.mark.parametrize(
    "endpoint,budget_ms",
    [
        ("/health", 50),
        ("/", 100),
        ("/v1/waitlist/stats", 300),
    ],
)
def test_hot_endpoint_p95_under_budget(client, endpoint: str, budget_ms: int) -> None:
    samples = _sample(client, "GET", endpoint, n=40)
    assert len(samples) >= 30, f"too few successful samples: {len(samples)}"
    samples.sort()
    p95 = samples[int(0.95 * len(samples)) - 1]
    p50 = statistics.median(samples)
    print(f"{endpoint}: p50={p50:.1f}ms p95={p95:.1f}ms (budget {budget_ms}ms)")
    assert p95 < budget_ms, f"{endpoint} p95 {p95:.1f}ms exceeds budget {budget_ms}ms"
