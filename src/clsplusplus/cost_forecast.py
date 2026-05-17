"""CLS++ infrastructure-cost dashboard + 7-day Monte Carlo cost forecast.

This module is the single home for all cost-forecasting logic so that the
admin endpoint in ``api.py`` stays a thin wrapper (other agents edit that
file concurrently — keep its surface minimal).

The forecast is a *genuine* simulation, not a stub:

  1. Historical daily operation volume is read from the ``usage_events``
     table and converted to a daily infrastructure-cost series using the
     per-operation constants in :mod:`clsplusplus.cost_model`.
  2. Day-over-day *log growth* of total daily volume is fitted from history
     (drift ``mu`` and volatility ``sigma``).
  3. A geometric random walk with drift — i.e. a Markov process on volume
     where tomorrow depends only on today times a log-normal shock — is
     simulated forward for 7 days across many independent paths.
  4. Per forecast day we report the mean projected cost and the p10/p90
     percentile band across all paths.

All public functions are pure and deterministic given an explicit seed,
so they are straightforward to unit-test.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from clsplusplus.cost_model import COST_PER_OPERATION

# --- tuning constants --------------------------------------------------------

DEFAULT_HISTORY_DAYS = 30
FORECAST_HORIZON_DAYS = 7
DEFAULT_PATHS = 2000
DEFAULT_SEED = 1337

# When history is too sparse to fit a meaningful volatility, fall back to a
# wide, conservative band so the dashboard never implies false precision.
_SPARSE_MIN_DAYS = 4
_SPARSE_SIGMA = 0.45          # ~ +/- 45% daily swing assumption
_MAX_SIGMA = 1.5              # clamp pathological volatility from dirty data
_MAX_DAILY_GROWTH = 4.0       # clamp a single-step multiplier (no >4x/day)


# --- data shapes -------------------------------------------------------------


@dataclass(frozen=True)
class DailyCost:
    """One historical day of infrastructure cost."""

    day: str          # ISO date, e.g. "2026-05-14"
    operations: int   # total billable operations that day
    cost_usd: float   # infra cost in USD for that day


@dataclass(frozen=True)
class ForecastDay:
    """One simulated future day: mean projected cost + p10/p90 band."""

    day: str          # ISO date
    mean_usd: float
    p10_usd: float
    p90_usd: float


# --- cost basis --------------------------------------------------------------


def cost_for_event_counts(event_counts: dict[str, float]) -> float:
    """Total infra cost (USD) for a ``{event_type: count}`` mapping.

    Unknown event types contribute 0 — mirrors ``cost_model.compute_cost``.
    """
    total = 0.0
    for event_type, count in event_counts.items():
        unit = COST_PER_OPERATION.get(event_type, 0.0)
        try:
            total += unit * float(count)
        except (ValueError, TypeError):
            continue
    return total


def _blended_unit_cost(history: list[DailyCost]) -> float:
    """Average cost per operation across history.

    Used to translate a simulated *operation count* back into a cost.
    Falls back to the mean of all known per-op costs when history carries
    no operations yet.
    """
    total_ops = sum(d.operations for d in history)
    total_cost = sum(d.cost_usd for d in history)
    if total_ops > 0:
        return total_cost / total_ops
    known = [c for c in COST_PER_OPERATION.values() if c > 0]
    return (sum(known) / len(known)) if known else 0.0


# --- historical aggregation --------------------------------------------------


def build_daily_costs(
    rows: list[dict],
    history_days: int = DEFAULT_HISTORY_DAYS,
    today: date | None = None,
) -> list[DailyCost]:
    """Turn raw ``usage_events`` rows into a dense per-day cost series.

    ``rows`` items must each carry ``day`` (date or ISO string),
    ``event_type`` (str) and ``quantity`` (number). Days with no events
    are filled with zero so the series is contiguous — the simulation and
    the chart both rely on a gap-free axis.
    """
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=history_days - 1)

    per_day_events: dict[date, dict[str, float]] = {}
    for row in rows:
        raw_day = row.get("day")
        if isinstance(raw_day, datetime):
            d = raw_day.date()
        elif isinstance(raw_day, date):
            d = raw_day
        else:
            try:
                d = date.fromisoformat(str(raw_day)[:10])
            except (ValueError, TypeError):
                continue
        if d < start or d > today:
            continue
        event_type = str(row.get("event_type") or "")
        try:
            qty = float(row.get("quantity") or 0)
        except (ValueError, TypeError):
            qty = 0.0
        bucket = per_day_events.setdefault(d, {})
        bucket[event_type] = bucket.get(event_type, 0.0) + qty

    series: list[DailyCost] = []
    for offset in range(history_days):
        d = start + timedelta(days=offset)
        events = per_day_events.get(d, {})
        ops = int(round(sum(events.values())))
        series.append(
            DailyCost(
                day=d.isoformat(),
                operations=ops,
                cost_usd=round(cost_for_event_counts(events), 6),
            )
        )
    return series


# --- stochastic model fit ----------------------------------------------------


@dataclass(frozen=True)
class _GrowthModel:
    """Fitted geometric-random-walk parameters for daily op volume."""

    start_volume: float   # last observed daily volume (sim start point)
    mu: float             # mean daily log-growth (drift)
    sigma: float          # std-dev of daily log-growth (volatility)
    sparse: bool          # True when history was too thin to trust the fit


def _fit_growth_model(history: list[DailyCost]) -> _GrowthModel:
    """Estimate drift + volatility of day-over-day log volume growth.

    Only days with non-zero volume contribute log-growth samples (a log
    ratio is undefined when either endpoint is zero). With too few usable
    samples we declare the fit *sparse* and substitute a wide default
    volatility instead of crashing or pretending to be precise.
    """
    volumes = [max(0.0, float(d.operations)) for d in history]
    nonzero = [v for v in volumes if v > 0]

    # Sim start: the most recent non-zero volume, else a neutral 1.0 so a
    # brand-new account still produces a (flat, wide) projection.
    start_volume = 0.0
    for v in reversed(volumes):
        if v > 0:
            start_volume = v
            break
    if start_volume <= 0:
        start_volume = nonzero[-1] if nonzero else 1.0

    log_growths: list[float] = []
    for prev, cur in zip(volumes, volumes[1:]):
        if prev > 0 and cur > 0:
            log_growths.append(math.log(cur / prev))

    if len(log_growths) < _SPARSE_MIN_DAYS:
        # Not enough signal: flat drift, deliberately wide band.
        return _GrowthModel(
            start_volume=start_volume,
            mu=0.0,
            sigma=_SPARSE_SIGMA,
            sparse=True,
        )

    mu = sum(log_growths) / len(log_growths)
    var = sum((g - mu) ** 2 for g in log_growths) / (len(log_growths) - 1)
    sigma = min(math.sqrt(max(var, 0.0)), _MAX_SIGMA)
    # Guard against a degenerate zero-variance fit (e.g. perfectly flat
    # synthetic data) — keep a small floor so the band is not a hairline.
    sigma = max(sigma, 0.02)
    return _GrowthModel(start_volume=start_volume, mu=mu, sigma=sigma, sparse=False)


# --- Monte Carlo simulation --------------------------------------------------


def simulate_forecast(
    history: list[DailyCost],
    horizon_days: int = FORECAST_HORIZON_DAYS,
    paths: int = DEFAULT_PATHS,
    seed: int = DEFAULT_SEED,
    today: date | None = None,
) -> list[ForecastDay]:
    """Run the Monte Carlo cost forecast for the next ``horizon_days`` days.

    Each path is a geometric random walk on daily operation volume:

        volume[t] = volume[t-1] * exp(mu + sigma * Z),  Z ~ N(0, 1)

    which is a first-order Markov process (tomorrow depends only on today).
    Volume is converted to cost via the blended per-operation cost fitted
    from history. We report the per-day mean and the p10/p90 percentile
    band across all simulated paths.

    Deterministic for a fixed ``seed`` — same inputs, same output.
    """
    today = today or datetime.now(timezone.utc).date()
    model = _fit_growth_model(history)
    unit_cost = _blended_unit_cost(history)
    rng = random.Random(seed)

    # costs_by_day[t] collects every path's cost for forecast day t.
    costs_by_day: list[list[float]] = [[] for _ in range(horizon_days)]

    for _ in range(max(1, paths)):
        volume = model.start_volume
        for t in range(horizon_days):
            shock = rng.gauss(0.0, 1.0)
            multiplier = math.exp(model.mu + model.sigma * shock)
            # Clamp a single day's multiplier so one fat-tailed draw cannot
            # blow the projection into nonsense.
            multiplier = min(max(multiplier, 1.0 / _MAX_DAILY_GROWTH), _MAX_DAILY_GROWTH)
            volume = max(0.0, volume * multiplier)
            costs_by_day[t].append(volume * unit_cost)

    forecast: list[ForecastDay] = []
    for t in range(horizon_days):
        day_costs = sorted(costs_by_day[t])
        forecast.append(
            ForecastDay(
                day=(today + timedelta(days=t + 1)).isoformat(),
                mean_usd=round(sum(day_costs) / len(day_costs), 6),
                p10_usd=round(_percentile(day_costs, 10), 6),
                p90_usd=round(_percentile(day_costs, 90), 6),
            )
        )
    return forecast


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# --- top-level orchestration -------------------------------------------------


def build_cost_report(
    rows: list[dict],
    history_days: int = DEFAULT_HISTORY_DAYS,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    paths: int = DEFAULT_PATHS,
    seed: int = DEFAULT_SEED,
    today: date | None = None,
) -> dict:
    """End-to-end: raw ``usage_events`` rows -> dashboard-ready report dict.

    Returned shape (JSON-serialisable, consumed by ``GET /admin/metrics/cost``)::

        {
          "history_days": int,
          "horizon_days": int,
          "recent_cost": [{day, operations, cost_usd}, ...],
          "recent_total_usd": float,
          "recent_avg_daily_usd": float,
          "forecast": [{day, mean_usd, p10_usd, p90_usd}, ...],
          "forecast_total_mean_usd": float,
          "model": {"mu", "sigma", "start_volume", "sparse"},
        }
    """
    history = build_daily_costs(rows, history_days=history_days, today=today)
    forecast = simulate_forecast(
        history,
        horizon_days=horizon_days,
        paths=paths,
        seed=seed,
        today=today,
    )
    model = _fit_growth_model(history)

    recent_total = sum(d.cost_usd for d in history)
    recent_avg = recent_total / len(history) if history else 0.0
    forecast_total_mean = sum(f.mean_usd for f in forecast)

    return {
        "history_days": history_days,
        "horizon_days": horizon_days,
        "recent_cost": [
            {"day": d.day, "operations": d.operations, "cost_usd": d.cost_usd}
            for d in history
        ],
        "recent_total_usd": round(recent_total, 6),
        "recent_avg_daily_usd": round(recent_avg, 6),
        "forecast": [
            {
                "day": f.day,
                "mean_usd": f.mean_usd,
                "p10_usd": f.p10_usd,
                "p90_usd": f.p90_usd,
            }
            for f in forecast
        ],
        "forecast_total_mean_usd": round(forecast_total_mean, 6),
        "model": {
            "mu": round(model.mu, 6),
            "sigma": round(model.sigma, 6),
            "start_volume": round(model.start_volume, 4),
            "sparse": model.sparse,
        },
    }


# --- DB query ----------------------------------------------------------------

USAGE_HISTORY_SQL = """
    SELECT DATE(occurred_at) AS day,
           event_type        AS event_type,
           SUM(quantity)     AS quantity
    FROM usage_events
    WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1
    GROUP BY DATE(occurred_at), event_type
    ORDER BY day
"""


async def fetch_usage_history(pool, history_days: int = DEFAULT_HISTORY_DAYS) -> list[dict]:
    """Read per-day, per-event-type operation counts from ``usage_events``.

    ``pool`` is an asyncpg pool. Returns rows shaped for ``build_daily_costs``.
    """
    async with pool.acquire() as conn:
        records = await conn.fetch(USAGE_HISTORY_SQL, history_days)
    return [
        {
            "day": r["day"],
            "event_type": r["event_type"],
            "quantity": r["quantity"],
        }
        for r in records
    ]
