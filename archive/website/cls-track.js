/**
 * CLS++ First-Party Web Tracker
 *
 * Lightweight, dependency-free page-view + click capture. Posts to
 * POST /v1/events/track (public, no auth) which writes the `web_events`
 * table behind the admin Traffic & Funnel dashboard.
 *
 * This is intentionally separate from posthog.js — it gives the owner a
 * first-party funnel that does not depend on a third-party dashboard and
 * survives ad-blockers that strip PostHog.
 */
(function () {
  'use strict';

  // ── Anonymous session id — stable for the browsing session ───────────────
  function sessionId() {
    try {
      var k = 'cls_sid';
      var v = sessionStorage.getItem(k);
      if (!v) {
        v = (Date.now().toString(36) + Math.random().toString(36).slice(2, 10));
        sessionStorage.setItem(k, v);
      }
      return v;
    } catch (e) {
      // Private mode / storage disabled — fall back to a per-page id.
      return 'nostore-' + Math.random().toString(36).slice(2, 10);
    }
  }

  var SID = sessionId();
  var PAGE = location.pathname || '/';

  function send(event, ref) {
    try {
      var body = JSON.stringify({
        event: event,
        page: PAGE,
        ref: ref || '',
        session_id: SID
      });
      // sendBeacon survives page unloads; fetch is the fallback.
      if (navigator.sendBeacon) {
        navigator.sendBeacon('/v1/events/track', new Blob([body], { type: 'application/json' }));
      } else {
        fetch('/v1/events/track', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: body,
          keepalive: true
        }).catch(function () {});
      }
    } catch (e) { /* tracking must never break the page */ }
  }

  // ── Page view — one per load ─────────────────────────────────────────────
  send('pageview', document.referrer || '');

  // ── Clicks — capture interactive targets and their label ─────────────────
  document.addEventListener('click', function (e) {
    var el = e.target;
    // Walk up to the nearest meaningful interactive ancestor.
    for (var i = 0; i < 4 && el && el !== document.body; i++) {
      var tag = (el.tagName || '').toLowerCase();
      if (tag === 'a' || tag === 'button' ||
          el.getAttribute('role') === 'button' ||
          el.hasAttribute('data-track')) {
        var label =
          el.getAttribute('data-track') ||
          (el.id ? '#' + el.id : '') ||
          (el.textContent || '').trim().slice(0, 60) ||
          (el.getAttribute('href') || '') ||
          tag;
        send('click', label);
        return;
      }
      el = el.parentElement;
    }
  }, true);
})();
