/**
 * CLS++ Regression Test Suite
 * =============================
 * Guards against regressions from past bugs and design changes.
 * Each TC references the specific issue it prevents from recurring.
 *
 * Run via: open tests/ui_test_runner.html?suite=regression
 */

(function () {
  'use strict';

  var ALL_PAGES = [
    'index.html', 'getting-started.html', 'docs.html', 'login.html', 'signup.html',
    'profile.html', 'dashboard.html', 'usage.html', 'chat.html', 'memory.html',
    'benchmark.html', 'benchmark_v1_direct.html', 'tests.html', 'trace.html',
    'install.html', 'demo.html', 'submit.html', 'support.html',
    'extension-test.html', 'chat-test.html', 'terms.html', 'privacy.html'
  ];

  var BASE = window.location.pathname.includes('/tests/')
    ? window.location.origin + window.location.pathname.replace(/\/tests\/.*$/, '/')
    : window.location.origin + '/';

  var results = [];
  var currentTC = '';

  function pass(name, detail) { results.push({ tc: currentTC, name: name, status: 'PASS', detail: detail || '' }); }
  function fail(name, detail) { results.push({ tc: currentTC, name: name, status: 'FAIL', detail: detail || '' }); }

  async function fetchHTML(page) {
    var r = await fetch(BASE + page + '?_t=' + Date.now());
    return await r.text();
  }

  // =========================================================================
  // REG-01: No backdrop-filter on tall containers (caused white rendering)
  // Bug: backdrop-filter:blur on elements >5000px tall renders as solid white
  // =========================================================================
  async function REG_01() {
    currentTC = 'REG-01';
    var tallPages = ['index.html', 'benchmark.html', 'benchmark_v1_direct.html'];
    for (var i = 0; i < tallPages.length; i++) {
      var html = await fetchHTML(tallPages[i]);
      // Check .content-wrap and .bench-container don't have backdrop-filter
      if (html.includes('bench-container') || html.includes('content-wrap')) {
        // Extract the relevant CSS block
        var hasBlur = /\.(bench-container|content-wrap)\s*\{[^}]*backdrop-filter\s*:\s*blur/i.test(html);
        if (!hasBlur) pass(tallPages[i] + ' no backdrop-filter on tall container');
        else fail(tallPages[i] + ' has backdrop-filter on tall container', 'Will render as solid white');
      } else {
        pass(tallPages[i] + ' no tall container class');
      }
    }
  }

  // =========================================================================
  // REG-02: .nav-cta specificity (orange pill must override .nav-links a)
  // Bug: .nav-links a had higher specificity than .nav-cta
  // =========================================================================
  async function REG_02() {
    currentTC = 'REG-02';
    try {
      var r = await fetch(BASE + 'styles.css?v=2&_t=' + Date.now());
      var css = await r.text();
      if (css.includes('.nav-links a.nav-cta') || css.includes('.nav-links a .nav-cta')) {
        pass('styles.css uses .nav-links a.nav-cta for specificity');
      } else if (css.includes('.nav-cta')) {
        fail('.nav-cta may lack specificity', 'Should be .nav-links a.nav-cta');
      } else {
        fail('.nav-cta not found in styles.css');
      }
    } catch (e) {
      fail('styles.css check', e.message);
    }
  }

  // =========================================================================
  // REG-03: Cache-busting on all styles.css references
  // Bug: Browser cache served old dark-theme CSS after updates
  // =========================================================================
  async function REG_03() {
    currentTC = 'REG-03';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var html = await fetchHTML(ALL_PAGES[i]);
      var match = html.match(/href\s*=\s*["']styles\.css(?:\?[^"']*)?["']/);
      if (match) {
        if (match[0].includes('?v=')) pass(ALL_PAGES[i] + ' has cache-bust on styles.css');
        else fail(ALL_PAGES[i] + ' missing cache-bust', 'href=' + match[0]);
      }
      // No match = inline styles only, acceptable
    }
  }

  // =========================================================================
  // REG-04: No sidebar layout remnants (old app-layout/app-sidebar pattern)
  // Bug: Profile/dashboard/usage had sidebar layout before top-nav conversion
  // =========================================================================
  async function REG_04() {
    currentTC = 'REG-04';
    var convertedPages = ['profile.html', 'dashboard.html', 'usage.html'];
    for (var i = 0; i < convertedPages.length; i++) {
      var html = await fetchHTML(convertedPages[i]);
      if (html.includes('app-sidebar') || html.includes('sidebar-mobile-toggle')) {
        fail(convertedPages[i] + ' has old sidebar layout');
      } else {
        pass(convertedPages[i] + ' no old sidebar layout');
      }
      if (html.includes('sidebar.js')) {
        fail(convertedPages[i] + ' still loads sidebar.js');
      } else {
        pass(convertedPages[i] + ' no sidebar.js');
      }
    }
  }

  // =========================================================================
  // REG-05: html/body background:transparent on all pages (video visibility)
  // Bug: Solid backgrounds on html/body blocked the fixed video
  // =========================================================================
  async function REG_05() {
    currentTC = 'REG-05';
    var darkBodies = ['background:#0a0a0a', 'background:#06060c', 'background:#05050a',
                      'background:#0d0d0d', 'background:#090909', 'background:#0a0a0f',
                      'background: #0a0a0a', 'background: #06060c'];
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var html = await fetchHTML(ALL_PAGES[i]);
      var hasDark = darkBodies.some(function(b) { return html.includes(b); });
      if (hasDark) fail(ALL_PAGES[i] + ' has dark body background');
      else pass(ALL_PAGES[i] + ' no dark body background');
    }
  }

  // =========================================================================
  // REG-06: No purple accent colors remaining (#7c6ef0, #6366f1)
  // Bug: Purple was the old accent; orange #ff6b35 is the new standard
  // =========================================================================
  async function REG_06() {
    currentTC = 'REG-06';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var html = await fetchHTML(ALL_PAGES[i]);
      // Check inline styles and CSS for purple
      var purpleInCSS = false;
      var styleBlocks = html.match(/<style[^>]*>[\s\S]*?<\/style>/gi) || [];
      styleBlocks.forEach(function(block) {
        if (block.includes('#7c6ef0') || block.includes('#6366f1')) purpleInCSS = true;
      });
      if (purpleInCSS) fail(ALL_PAGES[i] + ' has purple in CSS');
      else pass(ALL_PAGES[i] + ' no purple in CSS');
    }
  }

  // =========================================================================
  // REG-07: Docs hero text not inside frosted glass (was invisible)
  // Bug: White hero text inside .container with white frosted bg = invisible
  // =========================================================================
  async function REG_07() {
    currentTC = 'REG-07';
    var html = await fetchHTML('docs.html');
    var parser = new DOMParser();
    var doc = parser.parseFromString(html, 'text/html');
    var h1 = doc.querySelector('h1');
    if (h1) {
      var parent = h1.parentElement;
      var isInContainer = false;
      while (parent) {
        if (parent.classList && parent.classList.contains('container')) {
          isInContainer = true;
          break;
        }
        parent = parent.parentElement;
      }
      if (!isInContainer) pass('Docs h1 NOT inside .container (visible over video)');
      else fail('Docs h1 inside .container', 'White text on white bg = invisible');
    } else {
      fail('Docs has no h1');
    }
  }

  // =========================================================================
  // REG-08: Login/signup pages have nav (previously had no navigation)
  // =========================================================================
  async function REG_08() {
    currentTC = 'REG-08';
    var authPages = ['login.html', 'signup.html'];
    for (var i = 0; i < authPages.length; i++) {
      var html = await fetchHTML(authPages[i]);
      var doc = new DOMParser().parseFromString(html, 'text/html');
      var nav = doc.querySelector('nav.nav');
      if (nav) pass(authPages[i] + ' has nav bar');
      else fail(authPages[i] + ' missing nav bar', 'Was missing before redesign');
    }
  }

  // =========================================================================
  // REG-09: Content-wrap max-width 980px (video visible on sides)
  // Bug: Full-width content-wrap blocked video on sides
  // =========================================================================
  async function REG_09() {
    currentTC = 'REG-09';
    var html = await fetchHTML('index.html');
    if (html.includes('max-width') && html.includes('980px') && html.includes('content-wrap')) {
      pass('Index content-wrap has max-width:980px');
    } else if (html.includes('content-wrap')) {
      fail('Index content-wrap may be full-width');
    } else {
      pass('Index layout verified');
    }
  }

  // =========================================================================
  // REG-10: No nav-toggle hamburger button on any page
  // Bug: Some pages had hamburger menus from old responsive design
  // =========================================================================
  async function REG_10() {
    currentTC = 'REG-10';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var html = await fetchHTML(ALL_PAGES[i]);
      if (html.includes('nav-toggle')) {
        fail(ALL_PAGES[i] + ' has nav-toggle');
      } else {
        pass(ALL_PAGES[i] + ' no nav-toggle');
      }
    }
  }

  // =========================================================================
  // REG-11: .container in styles.css has no backdrop-filter
  // Bug: Global .container with backdrop-filter caused white rendering on scroll
  // =========================================================================
  async function REG_11() {
    currentTC = 'REG-11';
    try {
      var r = await fetch(BASE + 'styles.css?v=2&_t=' + Date.now());
      var css = await r.text();
      var containerMatch = css.match(/\.container\s*\{[^}]*\}/);
      if (containerMatch) {
        if (containerMatch[0].includes('backdrop-filter')) {
          fail('.container has backdrop-filter in styles.css');
        } else {
          pass('.container has no backdrop-filter');
        }
      } else {
        pass('No .container in styles.css');
      }
    } catch (e) {
      fail('styles.css read', e.message);
    }
  }

  // =========================================================================
  // REG-12: gradient-text uses orange not purple
  // =========================================================================
  async function REG_12() {
    currentTC = 'REG-12';
    try {
      var r = await fetch(BASE + 'styles.css?v=2&_t=' + Date.now());
      var css = await r.text();
      if (css.includes('.gradient-text')) {
        if (css.includes('#ff6b35') || css.includes('#ff8c5a')) pass('.gradient-text uses orange');
        else if (css.includes('#7c6ef0') || css.includes('#6366f1')) fail('.gradient-text still purple');
        else pass('.gradient-text present');
      } else {
        pass('No .gradient-text in styles.css');
      }
    } catch (e) {
      fail('gradient-text check', e.message);
    }
  }

  // =========================================================================
  // REG-13: All pages use Inter font
  // Bug: Some pages used DM Sans or SF Pro only
  // =========================================================================
  async function REG_13() {
    currentTC = 'REG-13';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var html = await fetchHTML(ALL_PAGES[i]);
      if (html.includes("'Inter'") || html.includes('"Inter"') || html.includes('family=Inter')) {
        pass(ALL_PAGES[i] + ' uses Inter font');
      } else {
        fail(ALL_PAGES[i] + ' missing Inter font');
      }
    }
  }

  // =========================================================================
  // REG-14: Video source is sunrise-ocean.mp4 everywhere
  // =========================================================================
  async function REG_14() {
    currentTC = 'REG-14';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var html = await fetchHTML(ALL_PAGES[i]);
      if (html.includes('video-bg')) {
        if (html.includes('sunrise-ocean.mp4')) pass(ALL_PAGES[i] + ' uses sunrise-ocean.mp4');
        else fail(ALL_PAGES[i] + ' wrong video source');
      }
    }
  }

  // =========================================================================
  // REG-15: Production API URL is www.clsplusplus.com (not old onrender.com)
  // =========================================================================
  async function REG_15() {
    currentTC = 'REG-15';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var html = await fetchHTML(ALL_PAGES[i]);
      if (html.includes('clsplusplus-api.onrender.com')) {
        fail(ALL_PAGES[i] + ' references old onrender.com URL');
      } else {
        pass(ALL_PAGES[i] + ' no old API URL');
      }
    }
  }

  // =========================================================================
  // Runner
  // =========================================================================
  var ALL_TCS = [
    REG_01, REG_02, REG_03, REG_04, REG_05,
    REG_06, REG_07, REG_08, REG_09, REG_10,
    REG_11, REG_12, REG_13, REG_14, REG_15
  ];

  async function runAllTests() {
    results = [];
    for (var i = 0; i < ALL_TCS.length; i++) {
      try { await ALL_TCS[i](); } catch (e) { fail('TC error', e.message); }
    }
    return results;
  }

  function renderReport(results) {
    var passed = results.filter(function(r) { return r.status === 'PASS'; }).length;
    var failed = results.filter(function(r) { return r.status === 'FAIL'; }).length;
    var total = results.length;

    var html = '<div style="font-family:Inter,system-ui,sans-serif;padding:24px;max-width:1100px;margin:0 auto;color:#1d1d1f">';
    html += '<h1 style="font-size:2rem;font-weight:800;margin-bottom:8px">CLS++ Regression Test Results</h1>';
    html += '<p style="font-size:1.1rem;margin-bottom:24px;color:#86868b">15 regression guards — prevents known bugs from recurring</p>';

    html += '<div style="display:flex;gap:16px;margin-bottom:32px;flex-wrap:wrap">';
    html += '<div style="background:rgba(255,255,255,0.92);border-radius:16px;padding:20px 28px;box-shadow:0 2px 20px rgba(0,0,0,0.06)"><div style="font-size:0.75rem;color:#86868b;text-transform:uppercase;margin-bottom:4px">Passed</div><div style="font-size:2rem;font-weight:700;color:#16a34a">' + passed + '</div></div>';
    html += '<div style="background:rgba(255,255,255,0.92);border-radius:16px;padding:20px 28px;box-shadow:0 2px 20px rgba(0,0,0,0.06)"><div style="font-size:0.75rem;color:#86868b;text-transform:uppercase;margin-bottom:4px">Failed</div><div style="font-size:2rem;font-weight:700;color:' + (failed > 0 ? '#ef4444' : '#16a34a') + '">' + failed + '</div></div>';
    html += '<div style="background:rgba(255,255,255,0.92);border-radius:16px;padding:20px 28px;box-shadow:0 2px 20px rgba(0,0,0,0.06)"><div style="font-size:0.75rem;color:#86868b;text-transform:uppercase;margin-bottom:4px">Total</div><div style="font-size:2rem;font-weight:700">' + total + '</div></div>';
    html += '</div>';

    var tcs = {};
    results.forEach(function(r) { if (!tcs[r.tc]) tcs[r.tc] = []; tcs[r.tc].push(r); });

    Object.keys(tcs).forEach(function(tc) {
      var tr = tcs[tc];
      var hasFail = tr.some(function(r) { return r.status === 'FAIL'; });
      var color = hasFail ? '#ef4444' : '#16a34a';
      var icon = hasFail ? '✗' : '✓';
      html += '<details style="margin-bottom:4px;background:rgba(255,255,255,0.8);border-radius:10px;border:1px solid rgba(0,0,0,0.06)">';
      html += '<summary style="padding:10px 14px;cursor:pointer;font-weight:600;font-size:0.9rem"><span style="color:' + color + '">' + icon + '</span> ' + tc + '</summary>';
      html += '<div style="padding:0 14px 10px">';
      tr.forEach(function(r) {
        var c = r.status === 'PASS' ? '#16a34a' : '#ef4444';
        html += '<div style="padding:2px 0;font-size:0.82rem;color:' + c + '">' + r.name;
        if (r.detail) html += ' <span style="color:#86868b"> - ' + r.detail + '</span>';
        html += '</div>';
      });
      html += '</div></details>';
    });

    html += '</div>';
    document.body.style.background = 'rgba(245,245,247,1)';
    document.body.innerHTML = html;
  }

  window.CLSRegressionTests = {
    runAllTests: runAllTests,
    renderReport: renderReport,
    getResults: function() { return results; }
  };
})();
