"""Comprehensive auth tests - format, timing, bypass attempts, edge cases."""

import time

import pytest

from clsplusplus.auth import (
    _API_KEY_PATTERN,
    _get_key_lookup,
    _normalize_key,
    _sha256_hex,
    extract_bearer_token,
    get_api_key_from_request,
    validate_api_key,
)
from clsplusplus.config import Settings

VALID_API_KEY = "cls_live_test1234567890123456789012"
VALID_TEST_KEY = "cls_test_abcdefghijklmnopqrstuvwx"


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------

class TestNormalizeKey:

    def test_none_returns_none(self):
        assert _normalize_key(None) is None

    def test_empty_string_returns_none(self):
        assert _normalize_key("") is None

    def test_whitespace_only_returns_none(self):
        assert _normalize_key("   ") is None

    def test_strips_whitespace(self):
        assert _normalize_key("  key  ") == "key"

    def test_preserves_content(self):
        assert _normalize_key("cls_live_abc") == "cls_live_abc"


# ---------------------------------------------------------------------------
# SHA-256 hashing
# ---------------------------------------------------------------------------

class TestSha256:

    def test_produces_hex_string(self):
        h = _sha256_hex("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert _sha256_hex("hello") == _sha256_hex("hello")

    def test_different_inputs_different_hashes(self):
        assert _sha256_hex("a") != _sha256_hex("b")


# ---------------------------------------------------------------------------
# API key pattern
# ---------------------------------------------------------------------------

class TestApiKeyPattern:

    def test_live_key_valid(self):
        assert _API_KEY_PATTERN.match(VALID_API_KEY)

    def test_test_key_valid(self):
        assert _API_KEY_PATTERN.match(VALID_TEST_KEY)

    def test_short_key_rejected(self):
        assert not _API_KEY_PATTERN.match("cls_live_short")

    def test_wrong_prefix_rejected(self):
        assert not _API_KEY_PATTERN.match("api_live_test1234567890123456789012")

    def test_missing_environment_rejected(self):
        assert not _API_KEY_PATTERN.match("cls_prod_test1234567890123456789012")

    def test_special_chars_rejected(self):
        assert not _API_KEY_PATTERN.match("cls_live_test123456789012345678901!")

    def test_exactly_minimum_length(self):
        key = "cls_live_" + "a" * 24
        assert _API_KEY_PATTERN.match(key)

    def test_one_below_minimum_rejected(self):
        key = "cls_live_" + "a" * 23
        assert not _API_KEY_PATTERN.match(key)

    def test_long_key_accepted(self):
        key = "cls_live_" + "a" * 100
        assert _API_KEY_PATTERN.match(key)


# ---------------------------------------------------------------------------
# Key lookup
# ---------------------------------------------------------------------------

class TestGetKeyLookup:

    def test_empty_keys(self):
        s = Settings(api_keys="")
        assert _get_key_lookup(s) == {}

    def test_single_key(self):
        s = Settings(api_keys=VALID_API_KEY)
        lookup = _get_key_lookup(s)
        assert len(lookup) == 1
        assert _sha256_hex(VALID_API_KEY) in lookup

    def test_multiple_keys(self):
        s = Settings(api_keys=f"{VALID_API_KEY},{VALID_TEST_KEY}")
        lookup = _get_key_lookup(s)
        assert len(lookup) == 2

    def test_invalid_key_filtered_out(self):
        s = Settings(api_keys=f"{VALID_API_KEY},bad_key,{VALID_TEST_KEY}")
        lookup = _get_key_lookup(s)
        assert len(lookup) == 2

    def test_whitespace_in_keys_handled(self):
        s = Settings(api_keys=f"  {VALID_API_KEY}  , {VALID_TEST_KEY} ")
        lookup = _get_key_lookup(s)
        assert len(lookup) == 2


# ---------------------------------------------------------------------------
# Validate API key
# ---------------------------------------------------------------------------

class TestValidateApiKey:

    def test_valid_key(self):
        s = Settings(api_keys=VALID_API_KEY)
        assert validate_api_key(VALID_API_KEY, s) is True

    def test_invalid_key(self):
        s = Settings(api_keys=VALID_API_KEY)
        assert validate_api_key("cls_live_wrong1234567890123456789012", s) is False

    def test_none_key(self):
        assert validate_api_key(None) is False

    def test_empty_key(self):
        assert validate_api_key("") is False

    def test_no_keys_configured(self):
        s = Settings(api_keys="")
        assert validate_api_key(VALID_API_KEY, s) is False

    def test_wrong_format_key(self):
        s = Settings(api_keys=VALID_API_KEY)
        assert validate_api_key("not_a_valid_key", s) is False

    def test_whitespace_key(self):
        s = Settings(api_keys=VALID_API_KEY)
        assert validate_api_key(f"  {VALID_API_KEY}  ", s) is True

    def test_constant_time_comparison(self):
        s = Settings(api_keys=VALID_API_KEY)
        t1 = time.perf_counter_ns()
        for _ in range(100):
            validate_api_key(VALID_API_KEY, s)
        valid_time = time.perf_counter_ns() - t1

        wrong_key = "cls_live_WRONG234567890123456789012"
        t2 = time.perf_counter_ns()
        for _ in range(100):
            validate_api_key(wrong_key, s)
        invalid_time = time.perf_counter_ns() - t2

        ratio = max(valid_time, invalid_time) / max(min(valid_time, invalid_time), 1)
        assert ratio < 10, f"Timing difference too large: {ratio}x"


# ---------------------------------------------------------------------------
# Bearer token extraction
# ---------------------------------------------------------------------------

class TestExtractBearerToken:

    def test_valid_bearer(self):
        assert extract_bearer_token("Bearer mytoken") == "mytoken"

    def test_case_insensitive_bearer(self):
        assert extract_bearer_token("bearer mytoken") == "mytoken"

    def test_no_header(self):
        assert extract_bearer_token(None) is None

    def test_empty_header(self):
        assert extract_bearer_token("") is None

    def test_wrong_scheme(self):
        assert extract_bearer_token("Basic mytoken") is None

    def test_no_token(self):
        assert extract_bearer_token("Bearer") is None

    def test_extra_parts(self):
        assert extract_bearer_token("Bearer token extra") is None

    def test_not_string(self):
        assert extract_bearer_token(123) is None

    def test_whitespace_handling(self):
        assert extract_bearer_token("  Bearer  token  ") == "token"


# ---------------------------------------------------------------------------
# get_api_key_from_request
# ---------------------------------------------------------------------------

class TestGetApiKeyFromRequest:

    def test_delegates_to_extract_bearer(self):
        assert get_api_key_from_request("Bearer mykey") == "mykey"
        assert get_api_key_from_request(None) is None


# ---------------------------------------------------------------------------
# API-level auth tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_public_paths_no_auth_required(client_unauth):
    for path in ["/", "/v1/memory/health", "/v1/demo/status"]:
        resp = await client_unauth.get(path)
        assert resp.status_code in (200, 503), f"{path} should be public"


@pytest.mark.asyncio
async def test_protected_path_401_without_key(client_unauth):
    resp = await client_unauth.post(
        "/v1/memory/write",
        json={"text": "test", "namespace": "default"},
    )
    assert resp.status_code == 401
    assert "detail" in resp.json()
    assert "WWW-Authenticate" in resp.headers


@pytest.mark.asyncio
async def test_protected_path_401_invalid_key(client_unauth):
    resp = await client_unauth.post(
        "/v1/memory/write",
        json={"text": "test", "namespace": "default"},
        headers={"Authorization": "Bearer invalid_key"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_path_401_wrong_format(client_unauth):
    resp = await client_unauth.post(
        "/v1/memory/write",
        json={"text": "test", "namespace": "default"},
        headers={"Authorization": "Bearer cls_short"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_no_auth_required_when_disabled(client_no_auth):
    resp = await client_no_auth.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_header_injection_attempt(client_unauth):
    resp = await client_unauth.post(
        "/v1/memory/write",
        json={"text": "test", "namespace": "default"},
        headers={"Authorization": "Bearer \r\nX-Custom: injected"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_docs_path_always_public(client_unauth):
    resp = await client_unauth.get("/docs")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_options_always_public(client_unauth):
    resp = await client_unauth.options("/v1/memory/write")
    # OPTIONS may return 200 (CORS preflight) or 405 if CORS middleware handles it
    # before auth. The key check: it must NOT be 401 (auth must not block OPTIONS).
    assert resp.status_code in (200, 405)
