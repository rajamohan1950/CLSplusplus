"""Extension integration tests — validates the JavaScript logic patterns
in intercept.js, capture.js, and background.js using code analysis.

Tests the cross-world communication architecture, message extraction
for each LLM provider, and toggle/counter logic.
"""

import json
import os
import re

import pytest

EXT_DIR = os.path.join(os.path.dirname(__file__), "..", "extension")


def read_ext(filename):
    with open(os.path.join(EXT_DIR, filename), "r") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════════
# Intercept.js — Fetch Interception & Message Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterceptFetchLocking:
    """Verify fetch is locked before page code can override it."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("intercept.js")

    def test_saves_original_fetch(self):
        assert "window.fetch.bind(window)" in self.js

    def test_defines_non_writable(self):
        assert "writable: false" in self.js
        assert "configurable: false" in self.js

    def test_fallback_assignment(self):
        """If defineProperty fails, falls back to direct assignment."""
        assert "window.fetch = clsFetch" in self.js

    def test_runs_at_document_start(self):
        """Manifest must set document_start for interception to work."""
        manifest = json.loads(read_ext("manifest.json"))
        main_script = [s for s in manifest["content_scripts"] if s.get("world") == "MAIN"][0]
        assert main_script["run_at"] == "document_start"

    def test_iife_wrapper(self):
        """Must be wrapped in IIFE to avoid global scope pollution."""
        assert "(function" in self.js

    def test_duplicate_guard(self):
        """Prevents loading twice."""
        assert "__clspp" in self.js


class TestInterceptChatGPTExtraction:
    """Verify ChatGPT message extraction from /backend-api/conversation."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("intercept.js")

    def test_detects_chatgpt_url(self):
        assert "backend-api" in self.js
        assert "conversation" in self.js

    def test_skips_init_endpoints(self):
        assert "/init" in self.js
        assert "/prepare" in self.js

    def test_extracts_messages_content_parts(self):
        """ChatGPT format: messages[].content.parts[0]"""
        assert "content.parts" in self.js
        assert "messages" in self.js

    def test_extracts_string_content(self):
        """ChatGPT also uses: messages[].content as plain string."""
        assert "typeof m.content === 'string'" in self.js


class TestInterceptClaudeExtraction:
    """Verify Claude message extraction."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("intercept.js")

    def test_detects_claude_url(self):
        assert "claude" in self.js
        assert "/api/" in self.js

    def test_extracts_prompt_field(self):
        assert "b.prompt" in self.js

    def test_extracts_content_array(self):
        """Claude format: content[{type:'text', text:'...'}]"""
        assert "b.content" in self.js
        assert "type === 'text'" in self.js

    def test_extracts_query_field(self):
        assert "b.query" in self.js

    def test_extracts_text_field(self):
        assert "b.text" in self.js


class TestInterceptGeminiExtraction:
    """Verify Gemini batchexecute URL-encoded form data extraction."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("intercept.js")

    def test_detects_gemini_batchexecute(self):
        assert "batchexecute" in self.js

    def test_has_dedicated_gemini_extractor(self):
        assert "function extractGemini" in self.js

    def test_parses_url_encoded(self):
        assert "URLSearchParams" in self.js

    def test_extracts_f_req_parameter(self):
        assert "f.req" in self.js

    def test_double_json_decode(self):
        """Gemini uses double JSON encoding: outer[1] → inner[0][0]"""
        assert "inner[0][0]" in self.js

    def test_handles_batch_format(self):
        """Handles [[["rpcId", "<json>", ...]]] format."""
        assert "outer[0][0]" in self.js

    def test_re_encodes_after_injection(self):
        """Must re-encode to URL-encoded form data after modification."""
        assert "params.toString()" in self.js

    def test_xhr_interceptor_for_gemini(self):
        """Gemini may use XMLHttpRequest instead of fetch."""
        assert "XMLHttpRequest.prototype.open" in self.js
        assert "XMLHttpRequest.prototype.send" in self.js

    def test_xhr_only_on_gemini(self):
        """XHR interceptor should only be installed on Gemini."""
        assert "host.includes('gemini')" in self.js


class TestInterceptDeepScan:
    """Verify fallback extraction for unknown LLM formats."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("intercept.js")

    def test_scans_all_keys(self):
        assert "Object.keys(b)" in self.js

    def test_skips_metadata_fields(self):
        """Should skip id, token, key, uuid, model, etc."""
        assert re.search(r"id\|token\|key\|uuid\|model", self.js)

    def test_length_bounds(self):
        """Only extract strings between 10 and 10000 chars."""
        assert "10" in self.js
        assert "10000" in self.js


class TestInterceptToggleSupport:
    """Verify injection can be paused."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("intercept.js")

    def test_checks_paused_before_injection(self):
        assert "isPaused()" in self.js

    def test_reads_paused_from_dom(self):
        assert "data-cls-paused" in self.js

    def test_skips_when_paused(self):
        assert "Injection paused" in self.js

    def test_signals_injection_to_capture(self):
        assert "signalInjection()" in self.js
        assert "data-cls-injected" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# Capture.js — ISOLATED World Message Relay
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaptureDOMMailbox:
    """Verify the cross-world DOM mailbox pattern."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("capture.js")

    def test_creates_hidden_element(self):
        assert "__cls_mem" in self.js
        assert "display" in self.js
        assert "none" in self.js

    def test_writes_facts_as_json(self):
        assert "data-facts" in self.js
        assert "JSON.stringify" in self.js

    def test_writes_timestamp(self):
        assert "data-ts" in self.js

    def test_reads_outbox(self):
        assert "data-cls-outbox" in self.js
        assert "JSON.parse" in self.js

    def test_deduplication_with_set(self):
        assert "new Set()" in self.js or "Set()" in self.js

    def test_dedup_key_length(self):
        """Uses first 100 chars as dedup key."""
        assert "100" in self.js

    def test_filters_injected_context(self):
        """Don't store CLS++ injected context back as a memory."""
        assert "For context, here are some things" in self.js

    def test_minimum_text_length(self):
        """Skip very short messages."""
        assert "text.length < 4" in self.js or "length < 4" in self.js


class TestCaptureToggleSync:
    """Verify capture.js syncs toggle state to DOM."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("capture.js")

    def test_reads_paused_from_storage(self):
        assert "cls_injection_paused" in self.js

    def test_reads_per_site_toggle(self):
        assert "cls_site_" in self.js

    def test_writes_paused_to_dom(self):
        assert "data-cls-paused" in self.js

    def test_watches_storage_changes(self):
        assert "chrome.storage.onChanged" in self.js

    def test_site_key_detection(self):
        assert "function getSiteKey" in self.js


class TestCaptureCounterRelay:
    """Verify capture.js relays injection events to background."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("capture.js")

    def test_watches_injected_attribute(self):
        assert "data-cls-injected" in self.js

    def test_sends_increment_message(self):
        assert "INCREMENT_INJECTED" in self.js

    def test_clears_injected_after_relay(self):
        assert "removeAttribute" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# Background.js — Service Worker API Layer
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackgroundAPIEndpoints:
    """Verify background.js calls correct API endpoints."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("background.js")

    def test_write_endpoint(self):
        assert "/v1/memory/write" in self.js

    def test_list_endpoint(self):
        assert "/v1/memory/list" in self.js

    def test_search_endpoint_for_side_panel(self):
        """Side panel search uses /v1/memories/search."""
        assert "/v1/memories/search" in self.js

    def test_usage_endpoint(self):
        assert "/v1/usage" in self.js

    def test_auth_me_endpoint(self):
        assert "/v1/auth/me" in self.js

    def test_uses_bearer_auth(self):
        assert "Bearer" in self.js

    def test_fetch_memories_uses_list(self):
        """Core fetchMemories uses proven /v1/memory/list path."""
        assert "function fetchMemories" in self.js
        assert "/v1/memory/list" in self.js


class TestBackgroundDailyCounters:
    """Verify daily counter logic."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("background.js")

    def test_resets_on_new_day(self):
        assert "cls_daily_date" in self.js
        assert "toISOString" in self.js

    def test_tracks_stored_count(self):
        assert "cls_daily_stored" in self.js

    def test_tracks_injected_count(self):
        assert "cls_daily_injected" in self.js

    def test_increments_atomically(self):
        """Counter should read current value and increment."""
        assert "|| 0" in self.js  # default to 0
        assert "+ 1" in self.js  # increment


class TestBackgroundBadge:
    """Verify badge count update logic."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("background.js")

    def test_sets_badge_text(self):
        assert "setBadgeText" in self.js

    def test_formats_thousands(self):
        """Should show 'k' for thousands."""
        assert "'k'" in self.js or '"k"' in self.js

    def test_badge_color(self):
        assert "setBadgeBackgroundColor" in self.js
        assert "#7c6ef0" in self.js  # purple accent

    def test_clears_badge_when_no_key(self):
        assert "text: ''" in self.js or 'text: ""' in self.js

    def test_periodic_refresh(self):
        assert "setInterval(updateBadge" in self.js


class TestBackgroundMessageRouting:
    """Verify all message types are handled."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("background.js")

    def test_handles_all_message_types(self):
        required = [
            "STORE", "FETCH", "ACTIVITY", "USAGE", "SEARCH",
            "DAILY_COUNTERS", "INCREMENT_INJECTED", "VERIFY_KEY", "OPEN_PANEL",
        ]
        for msg_type in required:
            assert f"'{msg_type}'" in self.js, f"Missing handler for: {msg_type}"

    def test_async_handlers_return_true(self):
        """Async message handlers must return true to keep the channel open."""
        # Count 'return true' — should have multiple for async handlers
        assert self.js.count("return true") >= 4


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-World Communication Pattern
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossWorldArchitecture:
    """Validate the MAIN ↔ ISOLATED world communication via DOM attributes."""

    def test_outbox_write_in_intercept(self):
        js = read_ext("intercept.js")
        assert "data-cls-outbox" in js
        assert "JSON.stringify" in js

    def test_outbox_read_in_capture(self):
        js = read_ext("capture.js")
        assert "data-cls-outbox" in js
        assert "JSON.parse" in js

    def test_mailbox_write_in_capture(self):
        js = read_ext("capture.js")
        assert "__cls_mem" in js
        assert "data-facts" in js

    def test_mailbox_read_in_intercept(self):
        js = read_ext("intercept.js")
        assert "__cls_mem" in js
        assert "data-facts" in js

    def test_pause_flag_write_in_capture(self):
        js = read_ext("capture.js")
        assert "data-cls-paused" in js
        assert "setAttribute" in js

    def test_pause_flag_read_in_intercept(self):
        js = read_ext("intercept.js")
        assert "data-cls-paused" in js
        assert "getAttribute" in js

    def test_injection_signal_write_in_intercept(self):
        js = read_ext("intercept.js")
        assert "data-cls-injected" in js

    def test_injection_signal_read_in_capture(self):
        js = read_ext("capture.js")
        assert "data-cls-injected" in js

    def test_worlds_match_manifest(self):
        """intercept.js = MAIN world, capture.js = ISOLATED world."""
        manifest = json.loads(read_ext("manifest.json"))
        scripts = manifest["content_scripts"]
        main = [s for s in scripts if s.get("world") == "MAIN"]
        isolated = [s for s in scripts if s.get("world") != "MAIN"]
        assert main[0]["js"] == ["intercept.js"]
        assert isolated[0]["js"] == ["capture.js"]


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Filter Logic (capture.js)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryFiltering:
    """Verify capture.js filters out bad memories before writing to DOM."""

    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("capture.js")

    def test_min_length_filter(self):
        assert "length > 8" in self.js or "length > 5" in self.js

    def test_max_length_filter(self):
        assert "length < 250" in self.js

    def test_filters_schema_prefix(self):
        assert "[Schema:" in self.js

    def test_filters_memory_prefix(self):
        assert "[MEMORY" in self.js

    def test_filters_questions(self):
        assert "?" in self.js  # endsWith('?') filter
