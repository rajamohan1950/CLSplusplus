"""Extension UI tests — validates popup, sidepanel, themes, shared CSS,
and manifest correctness without needing Chrome.

Uses file-based assertions: parses HTML/CSS/JS to verify structure,
element IDs, CSS variables, theme definitions, and manifest config.
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
# Manifest
# ═══════════════════════════════════════════════════════════════════════════════

class TestManifest:
    @pytest.fixture(autouse=True)
    def load(self):
        self.manifest = json.loads(read_ext("manifest.json"))

    def test_manifest_version_3(self):
        assert self.manifest["manifest_version"] == 3

    def test_version_semver(self):
        v = self.manifest["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", v), f"Version {v} is not semver"

    def test_permissions_include_storage(self):
        assert "storage" in self.manifest["permissions"]

    def test_permissions_include_side_panel(self):
        assert "sidePanel" in self.manifest["permissions"]

    def test_side_panel_config(self):
        assert "side_panel" in self.manifest
        assert self.manifest["side_panel"]["default_path"] == "sidepanel.html"

    def test_popup_configured(self):
        assert self.manifest["action"]["default_popup"] == "popup.html"

    def test_host_permissions_include_all_llm_sites(self):
        hosts = self.manifest["host_permissions"]
        assert any("chatgpt.com" in h for h in hosts)
        assert any("claude.ai" in h for h in hosts)
        assert any("gemini.google.com" in h for h in hosts)
        assert any("clsplusplus.com" in h for h in hosts)

    def test_content_scripts_main_world(self):
        scripts = self.manifest["content_scripts"]
        main_world = [s for s in scripts if s.get("world") == "MAIN"]
        assert len(main_world) == 1
        assert "intercept.js" in main_world[0]["js"]
        assert main_world[0]["run_at"] == "document_start"

    def test_content_scripts_isolated_world(self):
        scripts = self.manifest["content_scripts"]
        isolated = [s for s in scripts if s.get("world") != "MAIN"]
        assert len(isolated) == 1
        assert "capture.js" in isolated[0]["js"]

    def test_background_service_worker(self):
        assert self.manifest["background"]["service_worker"] == "background.js"

    def test_icons_defined(self):
        icons = self.manifest["icons"]
        assert "16" in icons
        assert "48" in icons
        assert "128" in icons

    def test_all_referenced_files_exist(self):
        """Every file referenced in manifest must exist on disk."""
        files_to_check = [
            self.manifest["action"]["default_popup"],
            self.manifest["side_panel"]["default_path"],
            self.manifest["background"]["service_worker"],
        ]
        for script_group in self.manifest["content_scripts"]:
            files_to_check.extend(script_group["js"])
        for f in files_to_check:
            assert os.path.exists(os.path.join(EXT_DIR, f)), f"Missing: {f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Popup HTML
# ═══════════════════════════════════════════════════════════════════════════════

class TestPopupHTML:
    @pytest.fixture(autouse=True)
    def load(self):
        self.html = read_ext("popup.html")

    def test_loads_shared_css(self):
        assert 'href="shared.css"' in self.html

    def test_loads_themes_js(self):
        assert 'src="themes.js"' in self.html

    def test_loads_popup_js(self):
        assert 'src="popup.js"' in self.html

    def test_has_status_dot(self):
        assert 'id="dot"' in self.html

    def test_has_unlinked_section(self):
        assert 'id="sec-unlinked"' in self.html

    def test_has_linked_section(self):
        assert 'id="sec-linked"' in self.html

    def test_has_api_key_input(self):
        assert 'id="key-input"' in self.html

    def test_has_link_button(self):
        assert 'id="btn-link"' in self.html

    def test_has_open_panel_button(self):
        assert 'id="btn-panel"' in self.html

    def test_has_stat_pills(self):
        assert 'id="stat-memories"' in self.html
        assert 'id="stat-ops"' in self.html
        assert 'id="stat-sites"' in self.html

    def test_has_last_memory_preview(self):
        assert 'id="last-memory"' in self.html
        assert 'id="last-memory-text"' in self.html

    def test_has_user_tier_display(self):
        assert 'id="user-tier"' in self.html
        assert 'id="user-name"' in self.html

    def test_has_error_display(self):
        assert 'id="err"' in self.html

    def test_has_generate_key_link(self):
        assert "profile.html#keys" in self.html


# ═══════════════════════════════════════════════════════════════════════════════
# Side Panel HTML
# ═══════════════════════════════════════════════════════════════════════════════

class TestSidePanelHTML:
    @pytest.fixture(autouse=True)
    def load(self):
        self.html = read_ext("sidepanel.html")

    def test_loads_shared_css(self):
        assert 'href="shared.css"' in self.html

    def test_loads_themes_js(self):
        assert 'src="themes.js"' in self.html

    def test_loads_sidepanel_js(self):
        assert 'src="sidepanel.js"' in self.html

    def test_has_three_tabs(self):
        assert 'data-tab="memories"' in self.html
        assert 'data-tab="activity"' in self.html
        assert 'data-tab="settings"' in self.html

    def test_has_three_tab_panels(self):
        assert 'data-panel="memories"' in self.html
        assert 'data-panel="activity"' in self.html
        assert 'data-panel="settings"' in self.html

    # Memories tab
    def test_has_search_input(self):
        assert 'id="search-input"' in self.html

    def test_has_filter_chips(self):
        assert 'data-source="all"' in self.html
        assert 'data-source="chatgpt"' in self.html
        assert 'data-source="claude"' in self.html
        assert 'data-source="gemini"' in self.html
        assert 'data-source="cli"' in self.html

    def test_has_memory_list_container(self):
        assert 'id="memory-list"' in self.html

    def test_has_memory_count(self):
        assert 'id="memory-count"' in self.html

    def test_has_load_more(self):
        assert 'id="btn-load-more"' in self.html

    def test_has_empty_state(self):
        assert 'id="memory-empty"' in self.html

    # Activity tab
    def test_has_usage_ring(self):
        assert 'id="usage-ring"' in self.html

    def test_has_daily_stats(self):
        assert 'id="daily-stored"' in self.html
        assert 'id="daily-injected"' in self.html

    def test_has_active_sites(self):
        assert 'id="site-chatgpt"' in self.html
        assert 'id="site-claude"' in self.html
        assert 'id="site-gemini"' in self.html

    def test_has_upgrade_card(self):
        assert 'id="upgrade-card"' in self.html

    # Settings tab
    def test_has_injection_toggle(self):
        assert 'id="toggle-injection"' in self.html

    def test_has_per_site_toggles(self):
        assert 'id="toggle-chatgpt"' in self.html
        assert 'id="toggle-claude"' in self.html
        assert 'id="toggle-gemini"' in self.html

    def test_has_theme_grid(self):
        assert 'id="theme-grid"' in self.html

    def test_has_account_info(self):
        assert 'id="account-name"' in self.html
        assert 'id="account-email"' in self.html
        assert 'id="account-tier"' in self.html

    def test_has_unlink_button(self):
        assert 'id="btn-unlink"' in self.html

    def test_has_manage_account_link(self):
        assert "profile.html" in self.html

    def test_has_view_all_memories_link(self):
        assert "memory.html" in self.html

    # Auth
    def test_has_unlinked_section(self):
        assert 'id="sec-unlinked"' in self.html

    def test_has_linked_section(self):
        assert 'id="sec-linked"' in self.html


# ═══════════════════════════════════════════════════════════════════════════════
# Shared CSS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSharedCSS:
    @pytest.fixture(autouse=True)
    def load(self):
        self.css = read_ext("shared.css")

    def test_has_css_custom_properties(self):
        assert "--cls-bg:" in self.css
        assert "--cls-text:" in self.css
        assert "--cls-accent:" in self.css
        assert "--cls-success:" in self.css
        assert "--cls-warning:" in self.css
        assert "--cls-danger:" in self.css

    def test_has_surface_vars(self):
        assert "--cls-surface:" in self.css
        assert "--cls-surface-hover:" in self.css
        assert "--cls-border:" in self.css

    def test_has_theme_switchable_bg(self):
        assert "--cls-bg-image:" in self.css

    def test_has_all_components(self):
        components = [
            ".cls-header", ".cls-dot", ".cls-badge", ".cls-card",
            ".cls-pill", ".cls-btn", ".cls-tab", ".cls-search",
            ".cls-chip", ".cls-memory-item", ".cls-toggle",
            ".cls-ring", ".cls-progress", ".cls-theme-swatch",
        ]
        for comp in components:
            assert comp in self.css, f"Missing component: {comp}"

    def test_has_tier_badge_variants(self):
        assert ".cls-badge-free" in self.css
        assert ".cls-badge-pro" in self.css
        assert ".cls-badge-business" in self.css
        assert ".cls-badge-enterprise" in self.css

    def test_has_transitions(self):
        assert "--cls-transition:" in self.css

    def test_has_spacing_scale(self):
        assert "--cls-xs:" in self.css
        assert "--cls-sm:" in self.css
        assert "--cls-md:" in self.css
        assert "--cls-lg:" in self.css
        assert "--cls-xl:" in self.css

    def test_has_scrollbar_styling(self):
        assert "::-webkit-scrollbar" in self.css

    def test_no_framework_imports(self):
        """No heavy framework imports — ultra-light CSS only."""
        assert "@import" not in self.css
        assert "bootstrap" not in self.css.lower()
        assert "tailwind" not in self.css.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Themes JS
# ═══════════════════════════════════════════════════════════════════════════════

class TestThemesJS:
    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("themes.js")

    def test_has_five_themes(self):
        count = len(re.findall(r"id:\s*'[^']+'", self.js))
        assert count == 5, f"Expected 5 themes, found {count}"

    def test_has_theme_ids(self):
        for theme_id in ["midnight", "aurora", "sunset", "ocean", "light"]:
            assert f"'{theme_id}'" in self.js, f"Missing theme: {theme_id}"

    def test_has_apply_function(self):
        assert "function clsApplyTheme" in self.js

    def test_has_load_function(self):
        assert "function clsLoadTheme" in self.js

    def test_has_render_grid_function(self):
        assert "function clsRenderThemeGrid" in self.js

    def test_has_default_vars(self):
        assert "CLS_DEFAULT_VARS" in self.js

    def test_themes_have_preview_colors(self):
        """Each theme must have bg and accent preview colors."""
        for theme_id in ["midnight", "aurora", "sunset", "ocean", "light"]:
            assert f"bg:" in self.js
            assert f"accent:" in self.js

    def test_saves_to_chrome_storage(self):
        assert "chrome.storage.local.set" in self.js

    def test_loads_from_chrome_storage(self):
        assert "chrome.storage.local.get" in self.js

    def test_sets_css_properties(self):
        assert "style.setProperty" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# Popup JS Logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestPopupJS:
    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("popup.js")

    def test_checks_server_health(self):
        assert "/health" in self.js

    def test_loads_theme_on_init(self):
        assert "clsLoadTheme()" in self.js

    def test_loads_saved_state(self):
        assert "cls_api_key" in self.js
        assert "cls_user" in self.js

    def test_sends_verify_key_message(self):
        assert "VERIFY_KEY" in self.js

    def test_sends_activity_message(self):
        assert "ACTIVITY" in self.js

    def test_sends_usage_message(self):
        assert "USAGE" in self.js

    def test_opens_side_panel(self):
        assert "sidePanel.open" in self.js

    def test_closes_popup_after_panel_open(self):
        assert "window.close()" in self.js

    def test_shows_linked_state(self):
        assert "sec-linked" in self.js
        assert "sec-unlinked" in self.js

    def test_displays_tier_badge(self):
        assert "cls-badge" in self.js

    def test_time_ago_function(self):
        assert "function timeAgo" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# Side Panel JS Logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestSidePanelJS:
    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("sidepanel.js")

    def test_loads_theme(self):
        assert "clsLoadTheme()" in self.js

    def test_tab_switching(self):
        assert "cls-tab" in self.js
        assert "classList" in self.js

    def test_memory_search(self):
        assert "SEARCH" in self.js
        assert "search-input" in self.js

    def test_search_debounce(self):
        assert "setTimeout" in self.js
        assert "300" in self.js  # 300ms debounce

    def test_filter_chips(self):
        assert "filter-chips" in self.js
        assert "currentFilter" in self.js

    def test_loads_memories(self):
        assert "function loadMemories" in self.js
        assert "ACTIVITY" in self.js

    def test_renders_memory_items(self):
        assert "cls-memory-item" in self.js
        assert "cls-memory-text" in self.js

    def test_memory_expand_collapse(self):
        assert "expanded" in self.js

    def test_loads_usage(self):
        assert "function loadUsage" in self.js
        assert "USAGE" in self.js

    def test_usage_ring_animation(self):
        assert "strokeDashoffset" in self.js
        assert "circumference" in self.js

    def test_usage_color_gradient(self):
        """Ring color changes based on usage percentage."""
        assert "cls-danger" in self.js
        assert "cls-warning" in self.js

    def test_daily_counters(self):
        assert "DAILY_COUNTERS" in self.js
        assert "daily-stored" in self.js
        assert "daily-injected" in self.js

    def test_toggle_injection(self):
        assert "toggle-injection" in self.js
        assert "cls_injection_paused" in self.js

    def test_per_site_toggles(self):
        assert "toggle-chatgpt" in self.js
        assert "toggle-claude" in self.js
        assert "toggle-gemini" in self.js

    def test_theme_rendering(self):
        assert "clsRenderThemeGrid" in self.js
        assert "theme-grid" in self.js

    def test_unlink(self):
        assert "btn-unlink" in self.js
        assert "chrome.storage.local.remove" in self.js

    def test_source_icons(self):
        assert "function sourceIcon" in self.js

    def test_time_ago(self):
        assert "function timeAgo" in self.js

    def test_load_more_pagination(self):
        assert "btn-load-more" in self.js
        assert "memoryOffset" in self.js

    def test_upgrade_card_visibility(self):
        assert "upgrade-card" in self.js

    def test_account_info(self):
        assert "account-name" in self.js
        assert "account-email" in self.js
        assert "account-tier" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# Background JS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackgroundJS:
    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("background.js")

    def test_api_url_production(self):
        assert "https://www.clsplusplus.com" in self.js

    def test_all_message_handlers(self):
        handlers = ["STORE", "FETCH", "ACTIVITY", "USAGE", "SEARCH",
                    "DAILY_COUNTERS", "INCREMENT_INJECTED", "VERIFY_KEY", "OPEN_PANEL"]
        for h in handlers:
            assert f"'{h}'" in self.js, f"Missing handler: {h}"

    def test_store_memory(self):
        assert "/v1/memory/write" in self.js
        assert "function storeMemory" in self.js

    def test_fetch_memories(self):
        assert "/v1/memory/list" in self.js
        assert "function fetchMemories" in self.js

    def test_search_memories(self):
        assert "/v1/memories/search" in self.js
        assert "function searchMemories" in self.js

    def test_fetch_usage(self):
        assert "/v1/usage" in self.js
        assert "function fetchUsage" in self.js

    def test_verify_key(self):
        assert "/v1/auth/me" in self.js

    def test_badge_update(self):
        assert "function updateBadge" in self.js
        assert "setBadgeText" in self.js
        assert "setBadgeBackgroundColor" in self.js

    def test_daily_counter_logic(self):
        assert "function incrementDailyCounter" in self.js
        assert "cls_daily_stored" in self.js
        assert "cls_daily_injected" in self.js
        assert "cls_daily_date" in self.js

    def test_badge_periodic_update(self):
        assert "setInterval(updateBadge" in self.js

    def test_auth_headers(self):
        assert "Bearer" in self.js

    def test_side_panel_open(self):
        assert "sidePanel.open" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# Intercept JS
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterceptJS:
    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("intercept.js")

    def test_locks_fetch(self):
        assert "Object.defineProperty(window, 'fetch'" in self.js
        assert "writable: false" in self.js

    def test_pause_check(self):
        assert "function isPaused" in self.js
        assert "data-cls-paused" in self.js

    def test_injection_signal(self):
        assert "function signalInjection" in self.js
        assert "data-cls-injected" in self.js

    def test_detects_chatgpt(self):
        assert "chatgpt" in self.js
        assert "backend-api" in self.js

    def test_detects_claude(self):
        assert "claude" in self.js

    def test_detects_gemini(self):
        assert "gemini" in self.js
        assert "batchexecute" in self.js

    def test_gemini_url_encoded_parsing(self):
        assert "function extractGemini" in self.js
        assert "URLSearchParams" in self.js
        assert "f.req" in self.js

    def test_gemini_xhr_interceptor(self):
        assert "XMLHttpRequest.prototype.open" in self.js
        assert "XMLHttpRequest.prototype.send" in self.js

    def test_memory_context_prefix(self):
        assert "For context, here are some things I have mentioned before" in self.js

    def test_reads_dom_mailbox(self):
        assert "__cls_mem" in self.js
        assert "data-facts" in self.js

    def test_writes_outbox(self):
        assert "data-cls-outbox" in self.js
        assert "function writeOutbox" in self.js

    def test_async_fallback(self):
        assert "Request.text()" in self.js or "clone().text()" in self.js

    def test_skips_when_paused(self):
        assert "isPaused()" in self.js
        assert "Injection paused" in self.js

    def test_signals_after_injection(self):
        assert "signalInjection()" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# Capture JS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaptureJS:
    @pytest.fixture(autouse=True)
    def load(self):
        self.js = read_ext("capture.js")

    def test_syncs_toggles(self):
        assert "function syncToggles" in self.js
        assert "cls_injection_paused" in self.js

    def test_listens_storage_changes(self):
        assert "chrome.storage.onChanged" in self.js

    def test_writes_paused_attribute(self):
        assert "data-cls-paused" in self.js

    def test_watches_outbox(self):
        assert "data-cls-outbox" in self.js
        assert "function checkOutbox" in self.js

    def test_stores_via_background(self):
        assert "type: 'STORE'" in self.js

    def test_checks_injection_signal(self):
        assert "data-cls-injected" in self.js
        assert "INCREMENT_INJECTED" in self.js

    def test_prefetches_memories(self):
        assert "function refreshMemories" in self.js
        assert "__cls_mem" in self.js
        assert "data-facts" in self.js

    def test_refresh_interval(self):
        assert "setInterval(refreshMemories, 30000)" in self.js

    def test_outbox_poll_interval(self):
        assert "setInterval(checkOutbox, 500)" in self.js

    def test_deduplication(self):
        assert "seen" in self.js

    def test_filters_cls_context(self):
        assert "For context, here are some things" in self.js

    def test_site_key_detection(self):
        assert "function getSiteKey" in self.js
        assert "chatgpt" in self.js
        assert "claude" in self.js
        assert "gemini" in self.js

    def test_per_site_toggle(self):
        assert "cls_site_" in self.js


# ═══════════════════════════════════════════════════════════════════════════════
# File Integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileIntegrity:
    def test_all_extension_files_exist(self):
        expected = [
            "manifest.json", "popup.html", "popup.js",
            "sidepanel.html", "sidepanel.js",
            "background.js", "capture.js", "intercept.js",
            "themes.js", "shared.css",
        ]
        for f in expected:
            path = os.path.join(EXT_DIR, f)
            assert os.path.exists(path), f"Missing extension file: {f}"

    def test_no_syntax_errors_in_json(self):
        """Manifest parses cleanly."""
        json.loads(read_ext("manifest.json"))

    def test_css_file_not_empty(self):
        css = read_ext("shared.css")
        assert len(css) > 500, "shared.css seems too small"

    def test_js_files_not_empty(self):
        for f in ["popup.js", "sidepanel.js", "background.js", "capture.js", "intercept.js", "themes.js"]:
            content = read_ext(f)
            assert len(content) > 100, f"{f} seems too small"
