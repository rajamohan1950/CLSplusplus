"""Tests for cost_forecast — the Monte Carlo infrastructure-cost forecast.

Covers the things that would quietly break the dashboard:
  * seeded determinism (same seed -> identical paths, different seed -> not)
  * output shape / invariants (7 days, p10 <= mean <= p90, ordered dates)
  * the cost basis is genuinely cost_model, not invented numbers
  * sparse / empty history never crashes and widens the band
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from clsplusplus.cost_forecast import (
    DEFAULT_HISTORY_DAYS,
    FORECAST_HORIZON_DAYS,
    DailyCost,
    build_cost_report,
    build_daily_costs,
    cost_for_event_counts,
    simulate_forecast,
)
from clsplusplus.cost_model import COST_PER_OPERATION


TODAY = date(2026, 5, 17)


def _rows_for(days: int, ops_per_day: int, event_type: str = "write", start_offset: int = 0):
    """Synthetic usage_events rows: `ops_per_day` of `event_type` for `days` days."""
    rows = []
    for i in range(days):
        d = TODAY - timedelta(days=start_offset + i)
        rows.append({"day": d, "event_type": event_type, "quantity": ops_per_day})
    return rows


# --- cost basis --------------------------------------------------------------


def test_cost_basis_uses_cost_model_constants():
    """A day's cost must equal cost_model unit cost x quantity, not a guess."""
    counts = {"write": 100, "read": 200, "adjudication": 5}
    expected = (
        100 * COST_PER_OPERATION["write"]
        + 200 * COST_PER_OPERATION["read"]
        + 5 * COST_PER_OPERATION["adjudication"]
    )
    assert cost_for_event_counts(counts) == pytest.approx(expected)


def test_unknown_event_types_cost_zero():
    """Unknown event types contribute 0 — mirrors cost_model.compute_cost."""
    assert cost_for_event_counts({"not_a_real_event": 9999}) == 0.0


# --- historical aggregation --------------------------------------------------


def test_build_daily_costs_is_dense_and_correct():
    """Missing days are zero-filled; present days carry the right cost."""
    rows = _rows_for(days=3, ops_per_day=10, event_type="write")
    series = build_daily_costs(rows, history_days=DEFAULT_HISTORY_DAYS, today=TODAY)

    assert len(series) == DEFAULT_HISTORY_DAYS
    # Series is contiguous and chronologically ordered.
    days = [d.day for d in series]
    assert days == sorted(days)
    # The 3 most recent days each have 10 write ops.
    recent = series[-3:]
    for d in recent:
        assert d.operations == 10
        assert d.cost_usd == pytest.approx(10 * COST_PER_OPERATION["write"])
    # Older days are zero-filled, not dropped.
    assert series[0].operations == 0
    assert series[0].cost_usd == 0.0


# --- Monte Carlo determinism -------------------------------------------------


def test_forecast_is_deterministic_for_a_fixed_seed():
    history = build_daily_costs(
        _rows_for(days=20, ops_per_day=1000), history_days=DEFAULT_HISTORY_DAYS, today=TODAY
    )
    a = simulate_forecast(history, seed=42, today=TODAY)
    b = simulate_forecast(history, seed=42, today=TODAY)
    assert [(d.day, d.mean_usd, d.p10_usd, d.p90_usd) for d in a] == [
        (d.day, d.mean_usd, d.p10_usd, d.p90_usd) for d in b
    ]


def test_different_seeds_produce_different_paths():
    """A genuine simulation: a different seed must move the numbers."""
    # Non-degenerate history so volatility is non-trivial.
    rows = []
    for i, ops in enumerate([100, 140, 90, 160, 120, 200, 130, 175, 110, 210]):
        rows.append({"day": TODAY - timedelta(days=i), "event_type": "write", "quantity": ops})
    history = build_daily_costs(rows, history_days=DEFAULT_HISTORY_DAYS, today=TODAY)
    a = simulate_forecast(history, seed=1, today=TODAY)
    b = simulate_forecast(history, seed=2, today=TODAY)
    assert [d.mean_usd for d in a] != [d.mean_usd for d in b]


# --- output shape / invariants ----------------------------------------------


def test_forecast_shape_and_band_ordering():
    history = build_daily_costs(
        _rows_for(days=25, ops_per_day=500), history_days=DEFAULT_HISTORY_DAYS, today=TODAY
    )
    forecast = simulate_forecast(history, seed=7, today=TODAY)

    assert len(forecast) == FORECAST_HORIZON_DAYS
    # Dates are the next 7 calendar days, in order, none in the past.
    expected_days = [(TODAY + timedelta(days=i + 1)).isoformat() for i in range(7)]
    assert [d.day for d in forecast] == expected_days
    # The confidence band must be ordered for every day.
    for d in forecast:
        assert d.p10_usd <= d.mean_usd <= d.p90_usd
        assert d.p10_usd >= 0.0


def test_uncertainty_widens_further_into_the_future():
    """A random walk's spread grows with the horizon — sanity-check that."""
    rows = []
    for i, ops in enumerate([100, 130, 95, 150, 115, 175, 125, 160, 105, 190, 120, 200]):
        rows.append({"day": TODAY - timedelta(days=i), "event_type": "write", "quantity": ops})
    history = build_daily_costs(rows, history_days=DEFAULT_HISTORY_DAYS, today=TODAY)
    forecast = simulate_forecast(history, seed=99, today=TODAY)
    first_spread = forecast[0].p90_usd - forecast[0].p10_usd
    last_spread = forecast[-1].p90_usd - forecast[-1].p10_usd
    assert last_spread > first_spread


# --- sparse / empty history safety ------------------------------------------


def test_empty_history_does_not_crash_and_is_flagged_sparse():
    report = build_cost_report([], today=TODAY)
    assert len(report["forecast"]) == FORECAST_HORIZON_DAYS
    assert report["recent_total_usd"] == 0.0
    assert report["model"]["sparse"] is True
    for d in report["forecast"]:
        assert d["p10_usd"] <= d["mean_usd"] <= d["p90_usd"]


def test_single_day_history_is_sparse_with_wide_bounds():
    """One data point cannot fit volatility — must fall back to a wide band."""
    rows = _rows_for(days=1, ops_per_day=1000)
    report = build_cost_report(rows, today=TODAY)
    assert report["model"]["sparse"] is True
    # Wide band: p90 is clearly above p10 on every forecast day.
    for d in report["forecast"]:
        assert d["p90_usd"] > d["p10_usd"]


def test_sparse_history_has_wider_relative_band_than_rich_history():
    sparse = build_cost_report(_rows_for(days=2, ops_per_day=1000), today=TODAY)
    rich_rows = []
    for i, ops in enumerate([1000, 1010, 990, 1005, 995, 1002, 998, 1008, 992, 1006]):
        rich_rows.append({"day": TODAY - timedelta(days=i), "event_type": "write", "quantity": ops})
    rich = build_cost_report(rich_rows, today=TODAY)

    def rel_band(report):
        d = report["forecast"][-1]
        return (d["p90_usd"] - d["p10_usd"]) / max(d["mean_usd"], 1e-12)

    assert rel_band(sparse) > rel_band(rich)


# --- end-to-end report -------------------------------------------------------


def test_build_cost_report_shape():
    rows = _rows_for(days=15, ops_per_day=800, event_type="search")
    report = build_cost_report(rows, today=TODAY)

    assert report["history_days"] == DEFAULT_HISTORY_DAYS
    assert report["horizon_days"] == FORECAST_HORIZON_DAYS
    assert len(report["recent_cost"]) == DEFAULT_HISTORY_DAYS
    assert len(report["forecast"]) == FORECAST_HORIZON_DAYS
    assert report["forecast_total_mean_usd"] == pytest.approx(
        sum(d["mean_usd"] for d in report["forecast"])
    )
    # recent_total reflects 15 days of 800 search ops at the cost_model rate.
    expected_recent = 15 * 800 * COST_PER_OPERATION["search"]
    assert report["recent_total_usd"] == pytest.approx(expected_recent, rel=1e-6)
    # Every recent_cost row carries the documented keys.
    for row in report["recent_cost"]:
        assert set(row.keys()) == {"day", "operations", "cost_usd"}


def test_zero_volume_history_projects_without_crashing():
    """All-zero days: blended unit cost falls back, forecast still produced."""
    rows = [{"day": TODAY - timedelta(days=i), "event_type": "write", "quantity": 0} for i in range(5)]
    report = build_cost_report(rows, today=TODAY)
    assert len(report["forecast"]) == FORECAST_HORIZON_DAYS
    assert report["model"]["sparse"] is True
