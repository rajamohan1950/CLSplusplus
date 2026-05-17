"""Unit tests for the ops-health aggregation in `clsplusplus.health_metrics`.

These exercise the pure aggregation path (`_build_report`, `_percentile`)
which is dependency-free — no Redis needed. The Redis read/write paths are
fail-open thin wrappers and are intentionally not mocked here.

Run only this file:
    python -m pytest tests/test_health_metrics.py -v
"""

from __future__ import annotations

from clsplusplus.health_metrics import (
    _build_report,
    _empty_report,
    _percentile,
)


# --------------------------------------------------------------------------
# _percentile — nearest-rank interpolation
# --------------------------------------------------------------------------

def test_percentile_empty_returns_zero():
    assert _percentile([], 95) == 0.0


def test_percentile_single_value():
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 99) == 42.0


def test_percentile_ordered_distribution():
    vals = [float(i) for i in range(1, 101)]  # 1..100
    assert _percentile(vals, 50) == 50.5
    assert _percentile(vals, 99) == 99.01
    # p0 / p100 hit the bounds exactly.
    assert _percentile(vals, 0) == 1.0
    assert _percentile(vals, 100) == 100.0


# --------------------------------------------------------------------------
# _empty_report — graceful zeros when there is no data
# --------------------------------------------------------------------------

def test_empty_report_is_all_zeros():
    rep = _empty_report(60)
    assert rep["window_minutes"] == 60
    assert rep["total_requests"] == 0
    assert rep["error_rate"] == 0.0
    assert rep["latency_ms"] == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "sample_size": 0}
    assert rep["status_breakdown"] == {}
    assert rep["top_error_codes"] == []
    assert rep["slowest_routes"] == []
    assert rep["guard_counts"] == {"quota_402": 0, "rate_limit_429": 0, "blocked_403": 0}
    assert rep["degraded"] is False


def test_build_report_with_no_data_matches_empty():
    rep = _build_report(60, {}, {}, {}, [], 0)
    assert rep["total_requests"] == 0
    assert rep["error_rate"] == 0.0
    assert rep["requests_per_minute"] == 0.0
    assert rep["slowest_routes"] == []


# --------------------------------------------------------------------------
# _build_report — error rate math
# --------------------------------------------------------------------------

def test_error_rate_split_4xx_5xx():
    # 100 reqs: 80×200, 12×404, 8×500  -> 20% error, 12% 4xx, 8% 5xx
    status_totals = {200: 80, 404: 12, 500: 8}
    rep = _build_report(60, status_totals, {}, {}, [], 100)
    assert rep["total_requests"] == 100
    assert rep["error_rate"] == 20.0
    assert rep["error_rate_4xx"] == 12.0
    assert rep["error_rate_5xx"] == 8.0


def test_error_rate_zero_when_all_success():
    rep = _build_report(60, {200: 50}, {}, {}, [], 50)
    assert rep["error_rate"] == 0.0
    assert rep["error_rate_4xx"] == 0.0
    assert rep["error_rate_5xx"] == 0.0


def test_volume_falls_back_to_status_sum_when_counter_missing():
    # Counter passed as 0 (old/partial bucket) — sum of statuses is used.
    rep = _build_report(60, {200: 30, 500: 10}, {}, {}, [], 0)
    assert rep["total_requests"] == 40
    assert rep["error_rate"] == 25.0


def test_requests_per_minute_divides_by_window():
    rep = _build_report(10, {200: 100}, {}, {}, [], 100)
    assert rep["requests_per_minute"] == 10.0


# --------------------------------------------------------------------------
# _build_report — guard counts (402 / 429 / 403)
# --------------------------------------------------------------------------

def test_guard_counts_extracted():
    status_totals = {200: 500, 402: 7, 429: 13, 403: 4}
    rep = _build_report(60, status_totals, {}, {}, [], 524)
    assert rep["guard_counts"] == {
        "quota_402": 7,
        "rate_limit_429": 13,
        "blocked_403": 4,
    }


def test_guard_counts_zero_when_absent():
    rep = _build_report(60, {200: 10}, {}, {}, [], 10)
    assert rep["guard_counts"] == {"quota_402": 0, "rate_limit_429": 0, "blocked_403": 0}


# --------------------------------------------------------------------------
# _build_report — top error codes
# --------------------------------------------------------------------------

def test_top_error_codes_sorted_by_count_descending():
    status_totals = {200: 1000, 404: 50, 500: 120, 429: 30, 403: 5}
    rep = _build_report(60, status_totals, {}, {}, [], 1205)
    top = rep["top_error_codes"]
    # Only 4xx/5xx, sorted by count desc.
    assert top[0] == {"status": 500, "count": 120}
    assert top[1] == {"status": 404, "count": 50}
    assert top[2] == {"status": 429, "count": 30}
    assert top[3] == {"status": 403, "count": 5}
    # 200 (success) must never appear in error codes.
    assert all(e["status"] >= 400 for e in top)


def test_status_breakdown_includes_all_codes_sorted():
    rep = _build_report(60, {500: 2, 200: 5, 404: 1}, {}, {}, [], 8)
    assert rep["status_breakdown"] == {"200": 5, "404": 1, "500": 2}


# --------------------------------------------------------------------------
# _build_report — latency percentiles
# --------------------------------------------------------------------------

def test_latency_percentiles_from_reservoir():
    latencies = [float(i) for i in range(1, 101)]  # 1..100 ms
    rep = _build_report(60, {200: 100}, {}, {}, latencies, 100)
    lat = rep["latency_ms"]
    assert lat["sample_size"] == 100
    assert lat["p50"] == 50.5
    assert lat["p95"] == 95.05
    assert lat["p99"] == 99.01


def test_latency_handles_unsorted_input():
    # _build_report must sort internally before computing percentiles.
    rep = _build_report(60, {200: 5}, {}, {}, [90.0, 10.0, 50.0, 30.0, 70.0], 5)
    lat = rep["latency_ms"]
    assert lat["p50"] == 50.0
    assert lat["sample_size"] == 5


# --------------------------------------------------------------------------
# _build_report — slowest routes
# --------------------------------------------------------------------------

def test_slowest_routes_ranked_by_mean_latency():
    # latsum is stored ×100 (integer micro-safe). Mean = (latsum/100)/count.
    route_count = {
        "GET /v1/memory/{id}": 10,   # mean 50ms
        "POST /v1/memory": 4,        # mean 500ms  <- slowest
        "GET /health": 100,          # mean 2ms
    }
    route_latsum = {
        "GET /v1/memory/{id}": 50 * 100 * 10,
        "POST /v1/memory": 500 * 100 * 4,
        "GET /health": 2 * 100 * 100,
    }
    rep = _build_report(60, {200: 114}, route_count, route_latsum, [], 114)
    slow = rep["slowest_routes"]
    assert slow[0]["route"] == "POST /v1/memory"
    assert slow[0]["avg_latency_ms"] == 500.0
    assert slow[0]["requests"] == 4
    assert slow[1]["route"] == "GET /v1/memory/{id}"
    assert slow[1]["avg_latency_ms"] == 50.0
    assert slow[2]["route"] == "GET /health"
    assert slow[2]["avg_latency_ms"] == 2.0


def test_slowest_routes_capped_at_ten():
    route_count = {f"GET /r{i}": 1 for i in range(25)}
    route_latsum = {f"GET /r{i}": i * 100 for i in range(25)}
    rep = _build_report(60, {200: 25}, route_count, route_latsum, [], 25)
    assert len(rep["slowest_routes"]) == 10
    # The slowest (highest index -> highest latency) ranks first.
    assert rep["slowest_routes"][0]["route"] == "GET /r24"


def test_slowest_routes_skips_zero_count():
    rep = _build_report(60, {200: 1}, {"GET /x": 0}, {"GET /x": 999}, [], 1)
    assert rep["slowest_routes"] == []


# --------------------------------------------------------------------------
# End-to-end shape — a realistic mixed window
# --------------------------------------------------------------------------

def test_realistic_window_full_report():
    status_totals = {200: 940, 404: 20, 402: 15, 429: 10, 403: 5, 500: 10}
    route_count = {"GET /v1/memory/{id}": 600, "POST /v1/memory": 400}
    route_latsum = {
        "GET /v1/memory/{id}": 20 * 100 * 600,   # mean 20ms
        "POST /v1/memory": 120 * 100 * 400,      # mean 120ms
    }
    latencies = [20.0] * 600 + [120.0] * 400
    rep = _build_report(60, status_totals, route_count, route_latsum, latencies, 1000)

    assert rep["total_requests"] == 1000
    assert rep["requests_per_minute"] == round(1000 / 60, 2)
    assert rep["error_rate"] == 6.0  # (20+15+10+5+10)/1000
    assert rep["guard_counts"]["quota_402"] == 15
    assert rep["guard_counts"]["rate_limit_429"] == 10
    assert rep["guard_counts"]["blocked_403"] == 5
    assert rep["latency_ms"]["sample_size"] == 1000
    assert rep["slowest_routes"][0]["route"] == "POST /v1/memory"
    assert rep["top_error_codes"][0]["count"] == 20  # 404 is the most common error
