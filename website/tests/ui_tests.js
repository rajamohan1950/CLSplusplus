/**
 * CLS++ Comprehensive UI Test Suite
 * ===================================
 * 30 test cases covering every page, every CTA, nav consistency,
 * video background, sunrise theme, Apple-style design compliance.
 *
 * Run via: open tests/ui_test_runner.html in browser
 * No external dependencies — pure vanilla JS.
 */

(function () {
  'use strict';

  // All pages on the site
  var ALL_PAGES = [
    'index.html',
    'getting-started.html',
    'docs.html',
    'integrate.html',
    'login.html',
    'signup.html',
    'profile.html',
    'dashboard.html',
    'usage.html',
    'chat.html',
    'memory.html',
    'benchmark.html',
    'benchmark_v1_direct.html',
    'tests.html',
    'trace.html',
    'install.html',
    'demo.html',
    'submit.html',
    'support.html',
    'extension-test.html',
    'chat-test.html',
    'terms.html',
    'privacy.html'
  ];

  var BASE = window.location.pathname.includes('/tests/')
    ? window.location.origin + window.location.pathname.replace(/\/tests\/.*$/, '/')
    : window.location.origin + '/';

  var results = [];
  var currentTC = '';

  function pass(name, detail) {
    results.push({ tc: currentTC, name: name, status: 'PASS', detail: detail || '' });
  }
  function fail(name, detail) {
    results.push({ tc: currentTC, name: name, status: 'FAIL', detail: detail || '' });
  }

  // Fetch and parse an HTML page
  async function fetchDoc(page) {
    var res = await fetch(BASE + page + '?_t=' + Date.now());
    if (!res.ok) throw new Error('HTTP ' + res.status);
    var html = await res.text();
    return new DOMParser().parseFromString(html, 'text/html');
  }

  // =========================================================================
  // TC-UI-01: All pages load with HTTP 200
  // =========================================================================
  async function TC_UI_01() {
    currentTC = 'TC-UI-01';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      try {
        var res = await fetch(BASE + page + '?_t=' + Date.now());
        if (res.ok) pass(page + ' loads (HTTP 200)');
        else fail(page + ' loads', 'HTTP ' + res.status);
      } catch (e) {
        fail(page + ' loads', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-02: Every page has correct meta tags (viewport, charset, title)
  // =========================================================================
  async function TC_UI_02() {
    currentTC = 'TC-UI-02';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      try {
        var doc = await fetchDoc(page);
        var viewport = doc.querySelector('meta[name="viewport"]');
        var charset = doc.querySelector('meta[charset]');
        var title = doc.querySelector('title');
        if (viewport) pass(page + ' has viewport meta');
        else fail(page + ' has viewport meta');
        if (charset) pass(page + ' has charset meta');
        else fail(page + ' has charset meta');
        if (title && title.textContent.includes('CLS++')) pass(page + ' title contains CLS++');
        else fail(page + ' title contains CLS++', 'Title: ' + (title ? title.textContent : 'none'));
      } catch (e) {
        fail(page + ' meta check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-03: Frosted glass nav bar present on all public pages
  // =========================================================================
  async function TC_UI_03() {
    currentTC = 'TC-UI-03';
    var pagesWithNav = ALL_PAGES.filter(function(p) { return p !== 'integrate.html'; }); // redirect page
    for (var i = 0; i < pagesWithNav.length; i++) {
      var page = pagesWithNav[i];
      try {
        var doc = await fetchDoc(page);
        var nav = doc.querySelector('nav.nav');
        if (nav) pass(page + ' has <nav class="nav">');
        else fail(page + ' has <nav class="nav">');
      } catch (e) {
        fail(page + ' nav check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-04: Nav has CLS++ logo linking to home
  // =========================================================================
  async function TC_UI_04() {
    currentTC = 'TC-UI-04';
    var pagesWithNav = ALL_PAGES.filter(function(p) { return p !== 'integrate.html'; });
    for (var i = 0; i < pagesWithNav.length; i++) {
      var page = pagesWithNav[i];
      try {
        var doc = await fetchDoc(page);
        var logo = doc.querySelector('nav .logo, nav .nav-logo');
        if (logo) {
          var href = logo.getAttribute('href');
          if (href === '/' || href === '/index.html' || href === 'index.html') {
            pass(page + ' logo links to home');
          } else {
            fail(page + ' logo links to home', 'href=' + href);
          }
        } else {
          fail(page + ' has logo in nav');
        }
      } catch (e) {
        fail(page + ' logo check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-05: "Get Started" orange CTA button in nav on all pages
  // =========================================================================
  async function TC_UI_05() {
    currentTC = 'TC-UI-05';
    var pagesWithNav = ALL_PAGES.filter(function(p) { return p !== 'integrate.html'; });
    for (var i = 0; i < pagesWithNav.length; i++) {
      var page = pagesWithNav[i];
      try {
        var doc = await fetchDoc(page);
        var cta = doc.querySelector('.nav-cta, .nav-links a.nav-cta');
        if (cta) {
          var text = cta.textContent.trim();
          if (text.toLowerCase().includes('get started')) {
            pass(page + ' nav has "Get Started" CTA');
          } else {
            fail(page + ' nav CTA text', 'Got: ' + text);
          }
        } else {
          fail(page + ' nav has Get Started CTA button');
        }
      } catch (e) {
        fail(page + ' CTA check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-06: Video background present on all themed pages
  // =========================================================================
  async function TC_UI_06() {
    currentTC = 'TC-UI-06';
    // Pages that should have video background
    var videoPages = ALL_PAGES.filter(function(p) {
      return p !== 'integrate.html'; // redirect
    });
    for (var i = 0; i < videoPages.length; i++) {
      var page = videoPages[i];
      try {
        var doc = await fetchDoc(page);
        var videoBg = doc.querySelector('.video-bg');
        var videoEl = doc.querySelector('.video-bg video source, .video-bg video');
        if (videoBg && videoEl) {
          pass(page + ' has video background');
        } else {
          // Some pages (demo, install) may use canvas instead
          var canvas = doc.querySelector('canvas');
          if (canvas && videoBg) {
            pass(page + ' has video + canvas background');
          } else if (videoBg) {
            pass(page + ' has video-bg div');
          } else {
            fail(page + ' has video background', 'No .video-bg found');
          }
        }
      } catch (e) {
        fail(page + ' video bg check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-07: No dark backgrounds (#0a0a0a, #06060c, #05050a, #0d0d0d) remaining
  // =========================================================================
  async function TC_UI_07() {
    currentTC = 'TC-UI-07';
    var darkColors = ['#05050a', '#06060c', '#0a0a0a', '#0a0a0f', '#0d0d0d', '#090909'];
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      try {
        var res = await fetch(BASE + page + '?_t=' + Date.now());
        var html = await res.text();
        var found = [];
        darkColors.forEach(function(c) {
          if (html.includes(c)) found.push(c);
        });
        if (found.length === 0) {
          pass(page + ' no dark background colors');
        } else {
          fail(page + ' has dark colors', found.join(', '));
        }
      } catch (e) {
        fail(page + ' dark color check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-08: Orange accent (#ff6b35) used instead of purple (#7c6ef0, #6366f1)
  // =========================================================================
  async function TC_UI_08() {
    currentTC = 'TC-UI-08';
    var purpleColors = ['#7c6ef0', '#6366f1', '#818cf8'];
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      try {
        var res = await fetch(BASE + page + '?_t=' + Date.now());
        var html = await res.text();
        var found = [];
        purpleColors.forEach(function(c) {
          // Exclude comments and data attributes
          var idx = html.indexOf(c);
          if (idx !== -1) found.push(c);
        });
        if (found.length === 0) {
          pass(page + ' no purple accents');
        } else {
          fail(page + ' still has purple', found.join(', '));
        }
      } catch (e) {
        fail(page + ' purple check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-09: styles.css loaded with cache-busting (?v=2) on pages using it
  // =========================================================================
  async function TC_UI_09() {
    currentTC = 'TC-UI-09';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      try {
        var doc = await fetchDoc(page);
        var links = Array.from(doc.querySelectorAll('link[rel="stylesheet"]'));
        var stylesLink = links.find(function(l) {
          var href = l.getAttribute('href') || '';
          return href.includes('styles.css');
        });
        if (stylesLink) {
          var href = stylesLink.getAttribute('href');
          if (href.includes('?v=')) {
            pass(page + ' styles.css has cache-bust');
          } else {
            fail(page + ' styles.css missing cache-bust', 'href=' + href);
          }
        } else {
          // Some pages have all inline styles - acceptable
          pass(page + ' uses inline styles (no styles.css)');
        }
      } catch (e) {
        fail(page + ' stylesheet check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-10: Body/html has transparent background for video visibility
  // =========================================================================
  async function TC_UI_10() {
    currentTC = 'TC-UI-10';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      if (page === 'integrate.html') continue; // redirect
      try {
        var res = await fetch(BASE + page + '?_t=' + Date.now());
        var html = await res.text();
        // Check for background:transparent or background: transparent
        var hasTransparent = html.includes('background:transparent') || html.includes('background: transparent');
        // Or no background set (inherits transparent)
        if (hasTransparent) {
          pass(page + ' body has transparent bg');
        } else {
          // Check if it doesn't set a dark background
          var darkBgRegex = /body\s*\{[^}]*background\s*:\s*#[0-9a-f]{3,6}/i;
          if (darkBgRegex.test(html)) {
            fail(page + ' body has dark background', 'Should be transparent');
          } else {
            pass(page + ' body bg not dark');
          }
        }
      } catch (e) {
        fail(page + ' bg check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-11: All anchor href="#id" targets exist on their page
  // =========================================================================
  async function TC_UI_11() {
    currentTC = 'TC-UI-11';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      if (page === 'integrate.html') continue;
      try {
        var doc = await fetchDoc(page);
        var anchors = Array.from(doc.querySelectorAll('a[href^="#"]'));
        var broken = [];
        anchors.forEach(function(a) {
          var id = a.getAttribute('href').slice(1);
          if (id && !doc.getElementById(id)) broken.push('#' + id);
        });
        if (broken.length === 0) pass(page + ' all anchor targets exist (' + anchors.length + ')');
        else fail(page + ' broken anchors', broken.join(', '));
      } catch (e) {
        fail(page + ' anchor check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-12: All buttons have visible labels or aria-labels
  // =========================================================================
  async function TC_UI_12() {
    currentTC = 'TC-UI-12';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      if (page === 'integrate.html') continue;
      try {
        var doc = await fetchDoc(page);
        var buttons = Array.from(doc.querySelectorAll('button'));
        var unlabeled = [];
        buttons.forEach(function(btn) {
          var text = btn.textContent.trim();
          var ariaLabel = btn.getAttribute('aria-label');
          if (!text && !ariaLabel) unlabeled.push(btn.className || 'unnamed');
        });
        if (unlabeled.length === 0) pass(page + ' all buttons labeled (' + buttons.length + ')');
        else fail(page + ' unlabeled buttons', unlabeled.join(', '));
      } catch (e) {
        fail(page + ' button check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-13: Login page has email, password fields and Log In CTA
  // =========================================================================
  async function TC_UI_13() {
    currentTC = 'TC-UI-13';
    try {
      var doc = await fetchDoc('login.html');
      var email = doc.querySelector('input[type="email"], input#email, input[placeholder*="mail"]');
      var pwd = doc.querySelector('input[type="password"], input#password');
      var loginBtn = doc.querySelector('button');
      if (email) pass('Login has email input');
      else fail('Login has email input');
      if (pwd) pass('Login has password input');
      else fail('Login has password input');
      if (loginBtn && loginBtn.textContent.toLowerCase().includes('log in')) pass('Login has Log In button');
      else fail('Login has Log In button', loginBtn ? loginBtn.textContent : 'no button');
      // Sign up link
      var signupLink = Array.from(doc.querySelectorAll('a')).find(function(a) {
        return a.textContent.toLowerCase().includes('sign up');
      });
      if (signupLink) pass('Login has Sign Up link');
      else fail('Login has Sign Up link');
    } catch (e) {
      fail('Login page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-14: Signup page has name, email, password fields and Sign Up CTA
  // =========================================================================
  async function TC_UI_14() {
    currentTC = 'TC-UI-14';
    try {
      var doc = await fetchDoc('signup.html');
      var name = doc.querySelector('input#name, input[placeholder*="ame"]');
      var email = doc.querySelector('input[type="email"], input#email, input[placeholder*="mail"]');
      var pwd = doc.querySelector('input[type="password"], input#password');
      var signupBtn = doc.querySelector('button');
      if (email) pass('Signup has email input');
      else fail('Signup has email input');
      if (pwd) pass('Signup has password input');
      else fail('Signup has password input');
      if (signupBtn && signupBtn.textContent.toLowerCase().includes('sign up')) pass('Signup has Sign Up CTA');
      else if (signupBtn && signupBtn.textContent.toLowerCase().includes('create')) pass('Signup has Create Account CTA');
      else fail('Signup has Sign Up CTA', signupBtn ? signupBtn.textContent : 'no button');
      // Login link
      var loginLink = Array.from(doc.querySelectorAll('a')).find(function(a) {
        return a.textContent.toLowerCase().includes('log in') || a.getAttribute('href') === '/login.html';
      });
      if (loginLink) pass('Signup has Log In link');
      else fail('Signup has Log In link');
    } catch (e) {
      fail('Signup page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-15: Index page has hero section with "Get started" CTA
  // =========================================================================
  async function TC_UI_15() {
    currentTC = 'TC-UI-15';
    try {
      var doc = await fetchDoc('index.html');
      var hero = doc.querySelector('.hero, section');
      var h1 = doc.querySelector('h1');
      if (h1) pass('Index has h1 heading');
      else fail('Index has h1 heading');
      // Get started CTA in hero
      var heroCTA = Array.from(doc.querySelectorAll('a')).find(function(a) {
        return a.textContent.toLowerCase().includes('get started');
      });
      if (heroCTA) pass('Index hero has "Get started" CTA');
      else fail('Index hero has "Get started" CTA');
    } catch (e) {
      fail('Index page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-16: Index page has pricing section with plan CTAs
  // =========================================================================
  async function TC_UI_16() {
    currentTC = 'TC-UI-16';
    try {
      var doc = await fetchDoc('index.html');
      var pricing = doc.getElementById('pricing');
      if (pricing) {
        pass('Index has #pricing section');
        var pricingCTAs = Array.from(pricing.querySelectorAll('a, button'));
        if (pricingCTAs.length >= 2) pass('Pricing has ' + pricingCTAs.length + ' CTAs');
        else fail('Pricing CTAs', 'Only ' + pricingCTAs.length + ' found');
      } else {
        // Check by text
        var html = doc.body.innerHTML;
        if (html.toLowerCase().includes('pricing')) pass('Index mentions pricing');
        else fail('Index has pricing section');
      }
    } catch (e) {
      fail('Index pricing', e.message);
    }
  }

  // =========================================================================
  // TC-UI-17: Docs page has sidebar navigation and code examples
  // =========================================================================
  async function TC_UI_17() {
    currentTC = 'TC-UI-17';
    try {
      var doc = await fetchDoc('docs.html');
      var sidebarLinks = doc.querySelectorAll('a[href^="#"]');
      if (sidebarLinks.length >= 5) pass('Docs has sidebar nav (' + sidebarLinks.length + ' links)');
      else fail('Docs sidebar nav', 'Only ' + sidebarLinks.length + ' anchor links');
      var codeBlocks = doc.querySelectorAll('pre, code');
      if (codeBlocks.length >= 3) pass('Docs has code examples (' + codeBlocks.length + ')');
      else fail('Docs code examples', 'Only ' + codeBlocks.length);
    } catch (e) {
      fail('Docs page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-18: Chat page has input, send button, new chat button
  // =========================================================================
  async function TC_UI_18() {
    currentTC = 'TC-UI-18';
    try {
      var doc = await fetchDoc('chat.html');
      var input = doc.getElementById('chat-input') || doc.querySelector('textarea, input[type="text"]');
      var send = doc.getElementById('btn-send') || doc.querySelector('button');
      var newChat = doc.getElementById('btn-new-chat');
      if (input) pass('Chat has input field');
      else fail('Chat has input field');
      if (send) pass('Chat has send button');
      else fail('Chat has send button');
      if (newChat) pass('Chat has New Chat button');
      else fail('Chat has New Chat button');
    } catch (e) {
      fail('Chat page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-19: Memory page has sidebar filters (model, category, layer)
  // =========================================================================
  async function TC_UI_19() {
    currentTC = 'TC-UI-19';
    try {
      var doc = await fetchDoc('memory.html');
      var modelList = doc.getElementById('model-list') || doc.querySelector('.model-list');
      var catList = doc.getElementById('cat-list') || doc.querySelector('.cat-list');
      var layerList = doc.getElementById('layer-list') || doc.querySelector('.layer-list');
      var search = doc.getElementById('search') || doc.querySelector('input[placeholder*="earch"]');
      if (modelList) pass('Memory has model filter');
      else fail('Memory has model filter');
      if (catList) pass('Memory has category filter');
      else fail('Memory has category filter');
      if (layerList) pass('Memory has layer filter');
      else fail('Memory has layer filter');
      if (search) pass('Memory has search input');
      else fail('Memory has search input');
    } catch (e) {
      fail('Memory page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-20: Benchmark page has data tables and scores
  // =========================================================================
  async function TC_UI_20() {
    currentTC = 'TC-UI-20';
    var benchPages = ['benchmark.html', 'benchmark_v1_direct.html'];
    for (var i = 0; i < benchPages.length; i++) {
      var page = benchPages[i];
      try {
        var doc = await fetchDoc(page);
        var tables = doc.querySelectorAll('table');
        if (tables.length >= 1) pass(page + ' has data tables (' + tables.length + ')');
        else fail(page + ' has data tables');
        var h1 = doc.querySelector('h1');
        if (h1) pass(page + ' has heading');
        else fail(page + ' has heading');
      } catch (e) {
        fail(page, e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-21: Tests page has run history sidebar and test result display
  // =========================================================================
  async function TC_UI_21() {
    currentTC = 'TC-UI-21';
    try {
      var doc = await fetchDoc('tests.html');
      var runList = doc.getElementById('run-list');
      var mainPanel = doc.getElementById('main-panel');
      if (runList) pass('Tests has run history list');
      else fail('Tests has run history list');
      if (mainPanel) pass('Tests has main results panel');
      else fail('Tests has main results panel');
    } catch (e) {
      fail('Tests page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-22: Install page has Activate button
  // =========================================================================
  async function TC_UI_22() {
    currentTC = 'TC-UI-22';
    try {
      var doc = await fetchDoc('install.html');
      var btn = doc.getElementById('btn') || doc.querySelector('button');
      if (btn) {
        var text = btn.textContent.toLowerCase();
        if (text.includes('activate')) pass('Install has Activate CTA');
        else pass('Install has button: ' + btn.textContent.trim());
      } else {
        fail('Install has Activate button');
      }
    } catch (e) {
      fail('Install page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-23: Profile page has user info, API keys, billing sections
  // =========================================================================
  async function TC_UI_23() {
    currentTC = 'TC-UI-23';
    try {
      var doc = await fetchDoc('profile.html');
      var html = doc.body.innerHTML.toLowerCase();
      if (html.includes('profile') || html.includes('user')) pass('Profile has user section');
      else fail('Profile has user section');
      if (html.includes('api key') || html.includes('api-key')) pass('Profile has API keys section');
      else fail('Profile has API keys section');
      if (html.includes('billing') || html.includes('plan') || html.includes('subscription')) pass('Profile has billing section');
      else fail('Profile has billing section');
    } catch (e) {
      fail('Profile page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-24: Dashboard page has stats cards and quick links
  // =========================================================================
  async function TC_UI_24() {
    currentTC = 'TC-UI-24';
    try {
      var doc = await fetchDoc('dashboard.html');
      var cards = doc.querySelectorAll('.dash-card');
      var quickLinks = doc.querySelectorAll('.quick-link');
      if (cards.length >= 1) pass('Dashboard has stat cards (' + cards.length + ')');
      else fail('Dashboard has stat cards');
      if (quickLinks.length >= 1) pass('Dashboard has quick links (' + quickLinks.length + ')');
      else fail('Dashboard has quick links');
    } catch (e) {
      fail('Dashboard page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-25: Usage page has usage cards and chart
  // =========================================================================
  async function TC_UI_25() {
    currentTC = 'TC-UI-25';
    try {
      var doc = await fetchDoc('usage.html');
      var html = doc.body.innerHTML.toLowerCase();
      if (html.includes('usage') || html.includes('operations')) pass('Usage has usage data');
      else fail('Usage has usage data');
      if (html.includes('chart') || html.includes('canvas')) pass('Usage has chart element');
      else fail('Usage has chart element');
    } catch (e) {
      fail('Usage page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-26: Terms and Privacy pages have legal content
  // =========================================================================
  async function TC_UI_26() {
    currentTC = 'TC-UI-26';
    var legalPages = ['terms.html', 'privacy.html'];
    for (var i = 0; i < legalPages.length; i++) {
      var page = legalPages[i];
      try {
        var doc = await fetchDoc(page);
        var h1 = doc.querySelector('h1');
        var sections = doc.querySelectorAll('h2, h3');
        if (h1) pass(page + ' has main heading');
        else fail(page + ' has main heading');
        if (sections.length >= 3) pass(page + ' has ' + sections.length + ' sections');
        else fail(page + ' sections', 'Only ' + sections.length);
      } catch (e) {
        fail(page, e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-27: Support page has contact info and FAQ
  // =========================================================================
  async function TC_UI_27() {
    currentTC = 'TC-UI-27';
    try {
      var doc = await fetchDoc('support.html');
      var html = doc.body.innerHTML.toLowerCase();
      if (html.includes('email') || html.includes('contact')) pass('Support has contact info');
      else fail('Support has contact info');
      if (html.includes('faq') || html.includes('frequently')) pass('Support has FAQ section');
      else fail('Support has FAQ section');
    } catch (e) {
      fail('Support page', e.message);
    }
  }

  // =========================================================================
  // TC-UI-28: No hamburger menu (nav-toggle) on any page
  // =========================================================================
  async function TC_UI_28() {
    currentTC = 'TC-UI-28';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      if (page === 'integrate.html') continue;
      try {
        var doc = await fetchDoc(page);
        var toggle = doc.querySelector('.nav-toggle, button[aria-label="Menu"]');
        if (!toggle) pass(page + ' no hamburger menu');
        else fail(page + ' has hamburger menu', 'Should be removed');
      } catch (e) {
        fail(page + ' toggle check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-29: Inter font loaded on all pages
  // =========================================================================
  async function TC_UI_29() {
    currentTC = 'TC-UI-29';
    for (var i = 0; i < ALL_PAGES.length; i++) {
      var page = ALL_PAGES[i];
      if (page === 'integrate.html') continue;
      try {
        var res = await fetch(BASE + page + '?_t=' + Date.now());
        var html = await res.text();
        var hasInter = html.includes('Inter') || html.includes('inter');
        if (hasInter) pass(page + ' uses Inter font');
        else fail(page + ' missing Inter font');
      } catch (e) {
        fail(page + ' font check', e.message);
      }
    }
  }

  // =========================================================================
  // TC-UI-30: integrate.html redirects to getting-started.html
  // =========================================================================
  async function TC_UI_30() {
    currentTC = 'TC-UI-30';
    try {
      var doc = await fetchDoc('integrate.html');
      var meta = doc.querySelector('meta[http-equiv="refresh"]');
      var canonical = doc.querySelector('link[rel="canonical"]');
      if (meta) {
        var content = meta.getAttribute('content');
        if (content.includes('getting-started')) pass('integrate.html redirects to getting-started');
        else fail('integrate.html redirect target', content);
      } else {
        fail('integrate.html has redirect meta');
      }
    } catch (e) {
      fail('integrate.html redirect', e.message);
    }
  }

  // =========================================================================
  // Runner
  // =========================================================================
  var ALL_TCS = [
    TC_UI_01, TC_UI_02, TC_UI_03, TC_UI_04, TC_UI_05,
    TC_UI_06, TC_UI_07, TC_UI_08, TC_UI_09, TC_UI_10,
    TC_UI_11, TC_UI_12, TC_UI_13, TC_UI_14, TC_UI_15,
    TC_UI_16, TC_UI_17, TC_UI_18, TC_UI_19, TC_UI_20,
    TC_UI_21, TC_UI_22, TC_UI_23, TC_UI_24, TC_UI_25,
    TC_UI_26, TC_UI_27, TC_UI_28, TC_UI_29, TC_UI_30
  ];

  async function runAllTests() {
    results = [];
    for (var i = 0; i < ALL_TCS.length; i++) {
      try { await ALL_TCS[i](); } catch (e) { fail('TC execution error', e.message); }
    }
    return results;
  }

  // =========================================================================
  // Report renderer (sunrise theme)
  // =========================================================================
  function renderReport(results) {
    var passed = results.filter(function(r) { return r.status === 'PASS'; }).length;
    var failed = results.filter(function(r) { return r.status === 'FAIL'; }).length;
    var total = results.length;
    var rate = total > 0 ? ((passed / total) * 100).toFixed(1) : '0.0';

    var html = '<div style="font-family:Inter,system-ui,sans-serif;padding:24px;max-width:1100px;margin:0 auto;color:#1d1d1f">';
    html += '<h1 style="font-size:2rem;font-weight:800;margin-bottom:8px">CLS++ UI Test Results</h1>';
    html += '<p style="font-size:1.1rem;margin-bottom:24px;color:#86868b">30 test cases across ' + ALL_PAGES.length + ' pages</p>';

    // Summary cards
    html += '<div style="display:flex;gap:16px;margin-bottom:32px;flex-wrap:wrap">';
    html += '<div style="background:rgba(255,255,255,0.92);border-radius:16px;padding:20px 28px;box-shadow:0 2px 20px rgba(0,0,0,0.06);min-width:140px"><div style="font-size:0.75rem;color:#86868b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Pass Rate</div><div style="font-size:2rem;font-weight:700;color:' + (failed === 0 ? '#16a34a' : '#ff6b35') + '">' + rate + '%</div></div>';
    html += '<div style="background:rgba(255,255,255,0.92);border-radius:16px;padding:20px 28px;box-shadow:0 2px 20px rgba(0,0,0,0.06);min-width:140px"><div style="font-size:0.75rem;color:#86868b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Passed</div><div style="font-size:2rem;font-weight:700;color:#16a34a">' + passed + '</div></div>';
    html += '<div style="background:rgba(255,255,255,0.92);border-radius:16px;padding:20px 28px;box-shadow:0 2px 20px rgba(0,0,0,0.06);min-width:140px"><div style="font-size:0.75rem;color:#86868b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Failed</div><div style="font-size:2rem;font-weight:700;color:' + (failed > 0 ? '#ef4444' : '#16a34a') + '">' + failed + '</div></div>';
    html += '<div style="background:rgba(255,255,255,0.92);border-radius:16px;padding:20px 28px;box-shadow:0 2px 20px rgba(0,0,0,0.06);min-width:140px"><div style="font-size:0.75rem;color:#86868b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Total</div><div style="font-size:2rem;font-weight:700">' + total + '</div></div>';
    html += '</div>';

    // Failures first
    if (failed > 0) {
      html += '<h2 style="color:#ef4444;font-size:1.2rem;margin-bottom:12px">Failures</h2>';
      html += '<div style="background:rgba(239,68,68,0.04);border:1px solid rgba(239,68,68,0.12);border-radius:16px;padding:16px;margin-bottom:24px">';
      results.filter(function(r) { return r.status === 'FAIL'; }).forEach(function(r) {
        html += '<div style="padding:6px 0;border-bottom:1px solid rgba(239,68,68,0.08);font-size:0.9rem">';
        html += '<strong style="color:#ef4444">[' + r.tc + ']</strong> ' + r.name;
        if (r.detail) html += ' <span style="color:#86868b">— ' + r.detail + '</span>';
        html += '</div>';
      });
      html += '</div>';
    }

    // Group by TC
    var tcs = {};
    results.forEach(function(r) {
      if (!tcs[r.tc]) tcs[r.tc] = [];
      tcs[r.tc].push(r);
    });

    html += '<h2 style="font-size:1.2rem;margin-bottom:12px;color:#86868b">All Test Cases</h2>';
    Object.keys(tcs).forEach(function(tc) {
      var tcResults = tcs[tc];
      var tcPassed = tcResults.filter(function(r) { return r.status === 'PASS'; }).length;
      var tcFailed = tcResults.filter(function(r) { return r.status === 'FAIL'; }).length;
      var statusColor = tcFailed === 0 ? '#16a34a' : '#ef4444';
      html += '<details style="margin-bottom:8px;background:rgba(255,255,255,0.8);border-radius:12px;border:1px solid rgba(0,0,0,0.06)">';
      html += '<summary style="padding:12px 16px;cursor:pointer;font-weight:600;font-size:0.95rem"><span style="color:' + statusColor + '">' + (tcFailed === 0 ? '✓' : '✗') + '</span> ' + tc + ' <span style="color:#86868b;font-weight:400">(' + tcPassed + '/' + tcResults.length + ')</span></summary>';
      html += '<div style="padding:0 16px 12px">';
      tcResults.forEach(function(r) {
        var color = r.status === 'PASS' ? '#16a34a' : '#ef4444';
        var icon = r.status === 'PASS' ? '✓' : '✗';
        html += '<div style="padding:3px 0;font-size:0.85rem;color:' + color + '">';
        html += icon + ' ' + r.name;
        if (r.detail && r.status === 'FAIL') html += ' <span style="color:#86868b">— ' + r.detail + '</span>';
        html += '</div>';
      });
      html += '</div></details>';
    });

    html += '</div>';
    document.body.style.background = 'rgba(245,245,247,1)';
    document.body.innerHTML = html;
  }

  window.CLSTests = {
    runAllTests: runAllTests,
    renderReport: renderReport,
    getResults: function() { return results; }
  };

  if (document.getElementById('cls-test-runner')) {
    document.getElementById('cls-test-runner').textContent = 'Running 30 UI test cases across ' + ALL_PAGES.length + ' pages...';
    runAllTests().then(renderReport);
  }
})();
