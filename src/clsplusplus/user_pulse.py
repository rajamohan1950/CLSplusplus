"""CLS++ live user pulse — near-real-time frustration / happiness signal.

The CSAT survey (`user_feedback` table) tells us how users felt *after*
they bothered to answer. This module is the *live* signal: it watches
behavioural breadcrumbs and produces a 0-100 frustration score for a
user right now, so an operator can reach out before the user churns.

Signals (all kept in Redis, short rolling windows):

  * error_4xx   — 4xx responses (excluding 402/429, which have their own
                  weight) hitting that user. Bad requests, 404s, 403s.
  * error_5xx   — 5xx responses. Our fault — weighted heaviest.
  * quota_block — 402/429 rejections (quota / rate-limit). The user
                  wanted something and we said no.
  * rage_retry  — the same operation retried in rapid succession. A user
                  hammering a failing button is the clearest live tell.
  * thumbs_down — an explicit thumbs-down from the feedback widget.

Each signal is counted in a Redis key with a TTL so the window self-
evicts — no sweeper. The score is a weighted, saturating sum mapped to
three bands: happy (<25), neutral (25-59), frustrated (>=60).

Design rules (match window_limits.py):
  * Everything fails OPEN and swallows Redis errors. Recording a pulse
    signal must never slow down or crash a real request.
  * Recording is best-effort fire-and-forget; scoring is read-only.
"""

from __future__ import annotations

import time
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.usage import _redis_client

# Rolling window for behavioural signals. Frustration is a *recent* state —
# a 404 from an hour ago should not still count. 15 minutes is long enough
# to catch a bad session, short enough to decay once things recover.
PULSE_WINDOW_SECONDS = 15 * 60

# Signal weights — points added to the raw frustration score per event.
# 5xx is our fault and weighted heaviest; rage-retry is the loudest live
# tell; a plain 4xx is mild on its own. Tuned so a handful of any one
# signal pushes a user out of the "happy" band, and a genuinely bad
# session (mixed errors + retries) saturates near 100.
_SIGNAL_WEIGHTS: dict[str, int] = {
    "error_5xx": 22,
    "rage_retry": 20,
    "thumbs_down": 30,
    "quota_block": 14,
    "error_4xx": 9,
}

# Band thresholds on the final 0-100 score.
_BAND_FRUSTRATED = 60
_BAND_NEUTRAL = 25

# How long a user with no signals stays listed at all. The per-signal
# keys expire on their own; this set just bounds the "who to score" list.
_ACTIVE_SET_TTL = PULSE_WINDOW_SECONDS + 60

# Redis key namespaces (distinct from cls:usage / cls:win / cls:ratelimit).
_KEY_PREFIX = "cls:pulse"
_ACTIVE_SET_KEY = "cls:pulse:active"


def _signal_key(user_id: str, signal: str) -> str:
    return f"{_KEY_PREFIX}:sig:{user_id}:{signal}"


def _retry_key(user_id: str, operation: str) -> str:
    return f"{_KEY_PREFIX}:retry:{user_id}:{operation}"


def _band(score: int) -> str:
    if score >= _BAND_FRUSTRATED:
        return "frustrated"
    if score >= _BAND_NEUTRAL:
        return "neutral"
    return "happy"


async def record_signal(
    user_id: str,
    signal: str,
    settings: Optional[Settings] = None,
    weight: int = 1,
) -> None:
    """Increment a behavioural signal counter for a user.

    Best-effort: any Redis failure is swallowed so this can be called
    inline from middleware without ever blocking or failing a request.
    Unknown signal names are ignored (no key created).
    """
    if not user_id or signal not in _SIGNAL_WEIGHTS:
        return
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        key = _signal_key(user_id, signal)
        pipe = client.pipeline()
        pipe.incrby(key, max(1, weight))
        pipe.expire(key, PULSE_WINDOW_SECONDS)
        # Track the user in the active set so the admin scan is bounded
        # to users who actually did something recently.
        pipe.zadd(_ACTIVE_SET_KEY, {user_id: time.time()})
        pipe.expire(_ACTIVE_SET_KEY, _ACTIVE_SET_TTL)
        await pipe.execute()
    except Exception:
        pass  # Fail-open — see module docstring.


async def record_http_outcome(
    user_id: str,
    status_code: int,
    operation: str,
    settings: Optional[Settings] = None,
) -> None:
    """Classify one HTTP response into pulse signals for a user.

    This is the single hook the middleware calls. It maps the status
    code to error / quota signals and detects rage-retry: the same
    operation completing non-2xx repeatedly inside a short sub-window.
    """
    if not user_id:
        return
    settings = settings or Settings()

    # 2xx/3xx: a healthy response. Nothing to record — recovery is just
    # the absence of new negative signals, and the windows decay on TTL.
    if status_code < 400:
        return

    if status_code in (402, 429):
        await record_signal(user_id, "quota_block", settings)
    elif status_code >= 500:
        await record_signal(user_id, "error_5xx", settings)
    else:
        await record_signal(user_id, "error_4xx", settings)

    # Rage-retry: count failed attempts at the *same* operation. A short
    # 60s sub-window — three+ failures of one operation in a minute is a
    # user hammering a broken button, not normal navigation.
    try:
        client = _redis_client(settings.redis_url)
        rkey = _retry_key(user_id, operation or "unknown")
        attempts = await client.incr(rkey)
        await client.expire(rkey, 60)
        if attempts >= 3:
            await record_signal(user_id, "rage_retry", settings)
    except Exception:
        pass  # Fail-open.


def score_from_counts(counts: dict[str, int]) -> dict:
    """Pure scoring function — counts of each signal → frustration result.

    Separated from Redis I/O so it is trivially unit-testable. Returns a
    dict with the 0-100 `score`, the `band`, and a `reasons` breakdown
    of which signals contributed how many points.
    """
    raw = 0
    reasons: list[dict] = []
    for signal, weight in _SIGNAL_WEIGHTS.items():
        n = int(counts.get(signal, 0) or 0)
        if n <= 0:
            continue
        points = n * weight
        raw += points
        reasons.append({"signal": signal, "count": n, "points": points})
    # Saturate at 100 — a genuinely bad session shouldn't read as 340.
    score = min(100, raw)
    # Loudest contributor first so the admin sees the headline cause.
    reasons.sort(key=lambda r: r["points"], reverse=True)
    return {"score": score, "band": _band(score), "reasons": reasons}


async def get_user_pulse(
    user_id: str,
    settings: Optional[Settings] = None,
) -> dict:
    """Read the live frustration score for a single user.

    Fails OPEN: on any Redis error returns a neutral-empty result
    (`score` 0, band `happy`) rather than raising.
    """
    settings = settings or Settings()
    base = {"user_id": user_id, "score": 0, "band": "happy", "reasons": []}
    try:
        client = _redis_client(settings.redis_url)
        pipe = client.pipeline()
        for signal in _SIGNAL_WEIGHTS:
            pipe.get(_signal_key(user_id, signal))
        raw_values = await pipe.execute()
        counts = {
            signal: int(v) if v else 0
            for signal, v in zip(_SIGNAL_WEIGHTS, raw_values)
        }
        result = score_from_counts(counts)
        result["user_id"] = user_id
        return result
    except Exception:
        return base


async def list_frustrated_users(
    settings: Optional[Settings] = None,
    limit: int = 100,
) -> list[dict]:
    """Return users currently in the 'frustrated' band, worst first.

    Scans only the bounded active set (users with a signal in the last
    window), scores each, and keeps the frustrated ones. Fails OPEN to
    an empty list on Redis errors — the admin page degrades gracefully.
    """
    settings = settings or Settings()
    try:
        client = _redis_client(settings.redis_url)
        # Drop stale members so the scan stays cheap.
        cutoff = time.time() - PULSE_WINDOW_SECONDS
        await client.zremrangebyscore(_ACTIVE_SET_KEY, "-inf", cutoff)
        user_ids = await client.zrange(_ACTIVE_SET_KEY, 0, -1)
    except Exception:
        return []

    frustrated: list[dict] = []
    for user_id in user_ids:
        pulse = await get_user_pulse(user_id, settings)
        if pulse["band"] == "frustrated":
            frustrated.append(pulse)
    frustrated.sort(key=lambda p: p["score"], reverse=True)
    return frustrated[:limit]
