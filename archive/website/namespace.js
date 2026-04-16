/**
 * CLS++ — Canonical User Namespace Manager
 *
 * Single source of truth for the user's persistent namespace.
 * Every page in the product loads this file FIRST before any other script.
 *
 * Design:
 *   namespace = user identity boundary (one brain per user)
 *   session   = conversation history only (does NOT create a new brain)
 *
 * The same namespace is used across:
 *   - Demo page (index.html)
 *   - Chat page (chat.html)
 *   - Trace page (trace.html)
 *   - Any future surface
 *   - All LLM models (Claude, GPT-4, Gemini)
 *
 * Migration: if the old 'cls_demo_namespace' key exists and the new
 * 'cls_user_namespace' key does not, the old value is promoted automatically
 * so existing users keep their memory.
 */
(function () {
  var STORAGE_KEY = 'cls_user_namespace';
  var LEGACY_KEY  = 'cls_demo_namespace';   // promoted on first load

  function _read() {
    try {
      return localStorage.getItem(STORAGE_KEY)
          || localStorage.getItem(LEGACY_KEY)
          || null;
    } catch (e) { return null; }
  }

  function _write(ns) {
    try {
      localStorage.setItem(STORAGE_KEY, ns);
      // Remove legacy key once migrated so we only write one place going forward
      localStorage.removeItem(LEGACY_KEY);
    } catch (e) {}
  }

  function _generate() {
    return 'user-' + Math.random().toString(36).slice(2, 10);
  }

  /** Return the current namespace, creating one if this is a first visit.
   *  If the user is logged in (window._CLS_USER_ID set by auth.js),
   *  use a deterministic namespace derived from their user ID. */
  function get() {
    // Logged-in users get a deterministic namespace tied to their account
    if (window._CLS_USER_ID) {
      var ns = 'user-' + window._CLS_USER_ID.slice(0, 8);
      _write(ns);
      return ns;
    }
    var ns = _read();
    if (!ns) {
      ns = _generate();
    }
    _write(ns);
    return ns;
  }

  /**
   * Reset: clear all stored memory keys and generate a fresh namespace.
   * Call this from the "Reset memory" button.
   * Returns the new namespace string.
   */
  function reset() {
    var ns = _generate();
    try {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(LEGACY_KEY);
      localStorage.setItem(STORAGE_KEY, ns);
    } catch (e) {}
    return ns;
  }

  // ── Public API ──────────────────────────────────────────────────────────
  window.CLSNamespace = {
    get: get,
    reset: reset,
    STORAGE_KEY: STORAGE_KEY,
  };

  // Convenience global — readable by any inline script without calling get()
  window.CLS_USER_NAMESPACE = get();
})();
