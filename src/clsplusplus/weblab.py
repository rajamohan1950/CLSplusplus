"""Weblab — PostHog-backed staged feature rollouts.

A "weblab" is a PostHog feature flag. A boolean flag gates a feature for a
rollout percentage; a multivariate flag assigns a treatment (control / T1 /
T2 / ...). Assignment is deterministic per `distinct_id` — PostHog hashes it,
so a given user always lands in the same treatment as the dial moves.

Every call fails SAFE: if PostHog is unconfigured or unreachable, the helpers
return the caller-supplied default rather than raising. The launch-rollout
gate must never block signups because of a PostHog outage.
"""
from __future__ import annotations

import logging
from typing import Optional

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

# Module-level singleton — building the client downloads flag definitions for
# local evaluation, so we keep one per process.
_client = None
_client_failed = False


def _get_client(settings: Settings):
    """Lazily build the PostHog client. Returns None when unavailable."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    if not settings.posthog_api_key:
        _client_failed = True
        return None
    try:
        from posthog import Posthog

        _client = Posthog(
            project_api_key=settings.posthog_api_key,
            host=settings.posthog_host,
            # A personal key enables local flag evaluation (no per-call
            # network round-trip). Optional — without it the SDK calls out.
            personal_api_key=settings.posthog_personal_api_key or None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("weblab: PostHog client init failed: %s", e)
        _client_failed = True
        _client = None
    return _client


def reset_client() -> None:
    """Drop the cached client — used by tests."""
    global _client, _client_failed
    _client = None
    _client_failed = False


def enabled(flag: str, distinct_id: str, settings: Settings,
            default: bool = False) -> bool:
    """True if `distinct_id` is rolled into `flag`. Fails safe to `default`."""
    client = _get_client(settings)
    if client is None:
        return default
    try:
        return bool(client.feature_enabled(flag, distinct_id))
    except Exception as e:  # noqa: BLE001
        logger.warning("weblab: feature_enabled(%s) failed: %s", flag, e)
        return default


def treatment(flag: str, distinct_id: str, settings: Settings,
              default: str = "control") -> str:
    """Treatment variant for `distinct_id` on a multivariate `flag`.

    Returns the variant key (e.g. "control", "T1", "T2"). A boolean-True flag
    maps to "T1"; False / missing maps to `default`.
    """
    client = _get_client(settings)
    if client is None:
        return default
    try:
        val = client.get_feature_flag(flag, distinct_id)
        if val is True:
            return "T1"
        if val is False or val is None:
            return default
        return str(val)
    except Exception as e:  # noqa: BLE001
        logger.warning("weblab: get_feature_flag(%s) failed: %s", flag, e)
        return default


def all_flags(distinct_id: str, settings: Settings) -> dict:
    """Every flag value for `distinct_id` — feeds the /v1/config/flags route."""
    client = _get_client(settings)
    if client is None:
        return {}
    try:
        return client.get_all_flags(distinct_id) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("weblab: get_all_flags failed: %s", e)
        return {}
