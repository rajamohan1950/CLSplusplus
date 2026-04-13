/**
 * CLS++ Extension Analytics (PostHog)
 * Fetches PostHog API key from /v1/config/analytics (server env var).
 * Shared wrapper for popup, sidepanel (full SDK), and background (HTTP POST).
 */
var CLSExtAnalytics = (function () {
  'use strict';

  var PH_HOST = 'https://us.i.posthog.com';
  var API = 'https://www.clsplusplus.com';
  var _phKey = null;
  var _keyPromise = null;

  /**
   * Fetch the PostHog key from the server (cached after first call).
   */
  function _getKey() {
    if (_phKey) return Promise.resolve(_phKey);
    if (_keyPromise) return _keyPromise;
    _keyPromise = fetch(API + '/v1/config/analytics')
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        _phKey = cfg.posthog_key || '';
        return _phKey;
      })
      .catch(function () { _phKey = ''; return ''; });
    return _keyPromise;
  }

  /**
   * Initialize full PostHog SDK (call from popup.js / sidepanel.js).
   * Requires posthog.min.js to be loaded first.
   */
  function initBrowser() {
    _getKey().then(function (key) {
      if (!key) return;
      if (typeof posthog !== 'undefined' && posthog.init) {
        posthog.init(key, {
          api_host: PH_HOST,
          person_profiles: 'identified_only',
          autocapture: true,
          capture_pageview: true,
          persistence: 'localStorage',
          session_recording: { enabled: false },
        });
      }
    });
  }

  /**
   * Lightweight capture via HTTP POST (for background service worker).
   * No SDK dependency — just a fetch call.
   */
  function captureFromBackground(event, properties, distinctId) {
    _getKey().then(function (key) {
      if (!key) return;
      fetch(PH_HOST + '/capture/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: key,
          event: event,
          properties: Object.assign({}, properties || {}, {
            distinct_id: distinctId || 'anonymous_extension_user',
            $lib: 'cls-extension-bg',
          }),
          timestamp: new Date().toISOString(),
        }),
      }).catch(function () { /* silently fail */ });
    });
  }

  /**
   * Identify a user in PostHog (popup/sidepanel only).
   */
  function identifyUser(user) {
    if (typeof posthog !== 'undefined' && user && user.id) {
      posthog.identify(String(user.id), {
        email: user.email,
        name: user.name,
        tier: user.tier,
      });
    }
  }

  /**
   * Track a custom event (popup/sidepanel with full SDK).
   */
  function track(event, props) {
    if (typeof posthog !== 'undefined' && posthog.capture) {
      posthog.capture(event, props || {});
    }
  }

  return {
    initBrowser: initBrowser,
    captureFromBackground: captureFromBackground,
    identifyUser: identifyUser,
    track: track,
  };
})();
