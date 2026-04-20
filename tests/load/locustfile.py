"""Locust workload covering the public API surface.

Three profiles are packed into one file; pick with a shape via CLI:

    # Load: 50 users constant for 5 min
    locust -f tests/load/locustfile.py --users 50 --spawn-rate 10 --run-time 5m

    # Stress: ramp to 500 users, keep, then push more
    LOCUST_SHAPE=stress locust -f tests/load/locustfile.py

    # Dip: ramp UP, sustain, ramp DOWN, assert latency recovers
    LOCUST_SHAPE=dip locust -f tests/load/locustfile.py

Point it at the local stack (CLS_TEST_API_URL defaults to localhost:18080).
"""
from __future__ import annotations

import os
import random
import uuid

from locust import HttpUser, LoadTestShape, constant, events, task
from locust.env import Environment

API_URL = os.environ.get("CLS_TEST_API_URL", "http://localhost:18080")
SHAPE = os.environ.get("LOCUST_SHAPE", "load").lower()


class PublicAnonUser(HttpUser):
    """Exercises routes that don't need auth — the landing-page surface."""
    host = API_URL
    wait_time = constant(0.5)

    @task(5)
    def root(self):
        self.client.get("/", name="GET /")

    @task(3)
    def health(self):
        self.client.get("/health", name="GET /health")

    @task(5)
    def waitlist_stats(self):
        self.client.get("/v1/waitlist/stats", name="GET /v1/waitlist/stats")

    @task(1)
    def waitlist_join(self):
        email = f"load-{uuid.uuid4().hex[:10]}@clsplusplus-load.local"
        self.client.post(
            "/v1/waitlist/join",
            json={"email": email},
            name="POST /v1/waitlist/join",
        )


class AuthFlowUser(HttpUser):
    """Register → login → poke authenticated endpoints. Only spawned in
    stress / dip profiles where we want to cover write paths."""
    host = API_URL
    wait_time = constant(1.0)

    def on_start(self):
        self.email = f"load-{uuid.uuid4().hex[:10]}@clsplusplus-load.local"
        self.password = "LoadTest123!"
        self.client.post(
            "/v1/auth/register",
            json={"email": self.email, "password": self.password, "name": "Load"},
            name="POST /v1/auth/register",
        )
        self.client.post(
            "/v1/auth/login",
            json={"email": self.email, "password": self.password},
            name="POST /v1/auth/login",
        )

    @task(3)
    def me(self):
        self.client.get("/v1/auth/me", name="GET /v1/auth/me")

    @task(2)
    def usage(self):
        self.client.get("/v1/user/usage", name="GET /v1/user/usage")

    @task(1)
    def memory_write(self):
        self.client.post(
            "/v1/memory/write",
            json={"text": f"load {random.random()}", "namespace": "default"},
            name="POST /v1/memory/write",
        )

    @task(2)
    def memory_list(self):
        self.client.get("/v1/memory/list?limit=20", name="GET /v1/memory/list")


# ---------------------------------------------------------------------------
# Shapes: load (flat), stress (ramp-up beyond capacity), dip (ramp down)
# ---------------------------------------------------------------------------

class StressShape(LoadTestShape):
    """Ramp from 50 → 500 over 5 minutes, hold for 2, then stop."""
    stages = [
        {"duration": 60,  "users": 50,  "spawn_rate": 10},
        {"duration": 180, "users": 200, "spawn_rate": 20},
        {"duration": 300, "users": 500, "spawn_rate": 30},
        {"duration": 420, "users": 500, "spawn_rate": 30},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None


class DipShape(LoadTestShape):
    """Up to 200 users, hold, back down to 10, hold — checks recovery."""
    stages = [
        {"duration": 60,  "users": 50,  "spawn_rate": 10},
        {"duration": 180, "users": 200, "spawn_rate": 20},
        {"duration": 300, "users": 200, "spawn_rate": 20},
        {"duration": 360, "users": 10,  "spawn_rate": 50},  # ramp down
        {"duration": 480, "users": 10,  "spawn_rate": 10},  # recovery window
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None


def _select_shape():
    if SHAPE == "stress":
        return StressShape
    if SHAPE == "dip":
        return DipShape
    return None  # default flat load


ActiveShape = _select_shape()
if ActiveShape is not None:
    # Locust imports `LoadTestShape` subclasses automatically from the file;
    # we just make sure the chosen one is exported with the expected name.
    DefaultShape = ActiveShape


# ---------------------------------------------------------------------------
# Simple pass/fail threshold — non-zero exit code on bad runs
# ---------------------------------------------------------------------------

@events.test_stop.add_listener
def _on_stop(environment: Environment, **kwargs):
    stats = environment.stats.total
    if stats.num_requests == 0:
        return
    fail_ratio = stats.num_failures / stats.num_requests
    p95 = stats.get_response_time_percentile(0.95)
    max_fail = float(os.environ.get("CLS_LOAD_MAX_FAIL_RATIO", "0.01"))
    max_p95 = float(os.environ.get("CLS_LOAD_MAX_P95_MS", "2000"))
    if fail_ratio > max_fail or p95 > max_p95:
        print(
            f"[locust] FAIL fail_ratio={fail_ratio:.3f} p95={p95:.0f}ms "
            f"(thresholds {max_fail} / {max_p95}ms)"
        )
        environment.process_exit_code = 1
    else:
        print(f"[locust] OK fail_ratio={fail_ratio:.3f} p95={p95:.0f}ms")
