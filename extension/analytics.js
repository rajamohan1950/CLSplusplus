/**
 * CLS++ Extension Analytics (PostHog)
 * Shared wrapper for popup, sidepanel (full SDK), and background (HTTP POST).
 *
 * Replace 'phc_YOUR_PROJECT_API_KEY' with your actual PostHog project API key.
 */
var CLSExtAnalytics = (function () {
  'use strict';

  var PH_KEY = 'phc_YOUR_PROJECT_API_KEY';
  var PH_HOST = 'https://us.i.posthog.com';

  /**
   * Initialize full PostHog SDK (call from popup.js / sidepanel.js).
   * Requires posthog.min.js to be loaded first.
   */
  function initBrowser() {
    if (typeof posthog !== 'undefined' && posthog.init) {
      posthog.init(PH_KEY, {
        api_host: PH_HOST,
        person_profiles: 'identified_only',
        autocapture: true,
        capture_pageview: true,
        persistence: 'localStorage',
        session_recording: { enabled: false },
      });
    }
  }

  /**
   * Lightweight capture via HTTP POST (for background service worker).
   * No SDK dependency — just a fetch call.
   */
  function captureFromBackground(event, properties, distinctId) {
    fetch(PH_HOST + '/capture/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: PH_KEY,
        event: event,
        properties: Object.assign({}, properties || {}, {
          distinct_id: distinctId || 'anonymous_extension_user',
          $lib: 'cls-extension-bg',
        }),
        timestamp: new Date().toISOString(),
      }),
    }).catch(function () { /* silently fail */ });
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
