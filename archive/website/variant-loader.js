/**
 * CLS++ A/B/C Variant Theme Loader
 *
 * Reads the `cls_variant` cookie (set by the server on first visit)
 * and applies the matching theme CSS to ALL pages.
 *
 *   Variant A = current design (no overrides)
 *   Variant B = Command Center dark theme (theme-b.css)
 *   Variant C = Ambient Focus dark theme  (theme-c.css)
 *
 * Also fires a PostHog event: `page_viewed` with variant metadata.
 *
 * Include this script in EVERY HTML page:
 *   <script src="variant-loader.js"></script>
 */
(function () {
  // --- Read variant from cookie ---
  function getCookie(name) {
    var match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
    return match ? decodeURIComponent(match[1]) : null;
  }

  var variant = getCookie('cls_variant') || 'A';
  window.CLS_VARIANT = variant;

  // --- Load theme CSS for B or C ---
  if (variant === 'B' || variant === 'C') {
    var href = variant === 'B' ? '/theme-b.css' : '/theme-c.css';
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href + '?v=1';
    // Insert after last stylesheet for highest specificity
    var lastLink = document.querySelector('link[rel="stylesheet"]:last-of-type');
    if (lastLink && lastLink.parentNode) {
      lastLink.parentNode.insertBefore(link, lastLink.nextSibling);
    } else {
      document.head.appendChild(link);
    }

    // Add variant class to body for per-variant CSS hooks
    document.addEventListener('DOMContentLoaded', function () {
      document.body.classList.add('variant-' + variant.toLowerCase());
    });
  }

  // --- PostHog tracking ---
  function trackVariant() {
    var page = window.location.pathname || '/';
    if (window.posthog && typeof window.posthog.capture === 'function') {
      window.posthog.capture('page_viewed', {
        variant: variant,
        page: page,
        referrer: document.referrer || '',
      });
    } else if (window.CLSAnalytics && typeof window.CLSAnalytics.track === 'function') {
      window.CLSAnalytics.track('page_viewed', {
        variant: variant,
        page: page,
      });
    }
  }

  // Track after PostHog has a chance to initialize
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      setTimeout(trackVariant, 1500);
    });
  } else {
    setTimeout(trackVariant, 1500);
  }

  // --- Track CTA clicks ---
  document.addEventListener('click', function (e) {
    var el = e.target.closest('a, button');
    if (!el) return;
    var href = el.getAttribute('href') || '';
    var text = (el.textContent || '').trim().substring(0, 50);
    var isGetStarted = /get.?started|sign.?up|subscribe|try.?free/i.test(text + ' ' + href);
    if (isGetStarted) {
      if (window.posthog && typeof window.posthog.capture === 'function') {
        window.posthog.capture('cta_clicked', {
          variant: variant,
          page: window.location.pathname,
          button_text: text,
          href: href,
        });
      }
    }
  });
})();
