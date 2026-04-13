/**
 * CLS++ PostHog Analytics
 * Loads on every page. Initializes PostHog with autocapture, session recording, web vitals.
 * Exposes CLSAnalytics.track() and CLSAnalytics.identify() for custom events.
 *
 * Replace 'phc_YOUR_PROJECT_API_KEY' with your actual PostHog project API key.
 */
(function () {
  'use strict';

  // PostHog JS SDK snippet (from posthog.com/docs/libraries/js)
  !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url lib get_property getSessionProperty sessionRecordingStarted loadToolbar get_config capture_pageview capture_pageleave createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);

  // --- CONFIGURATION ---
  posthog.init('phc_YOUR_PROJECT_API_KEY', {
    api_host: 'https://us.i.posthog.com',
    person_profiles: 'identified_only',
    autocapture: true,
    capture_pageview: true,
    capture_pageleave: true,
    capture_performance: true,
    enable_recording_console_log: true,
    session_recording: {
      maskAllInputs: false,
      maskTextSelector: '.mask-ph',
    },
    loaded: function (ph) {
      // Auto-identify user if CLSAuth is available
      if (window.CLSAuth) {
        window.CLSAuth.getUser().then(function (user) {
          if (user) {
            ph.identify(String(user.id), {
              email: user.email,
              name: user.name,
              tier: user.tier,
              is_admin: user.is_admin,
            });
          }
        });
      }
    },
  });

  // Expose global helpers for custom event tracking
  window.CLSAnalytics = {
    track: function (event, props) {
      if (window.posthog) posthog.capture(event, props || {});
    },
    identify: function (user) {
      if (window.posthog && user && user.id) {
        posthog.identify(String(user.id), {
          email: user.email,
          name: user.name,
          tier: user.tier,
          is_admin: user.is_admin,
        });
      }
    },
    reset: function () {
      if (window.posthog) posthog.reset();
    },
  };
})();
