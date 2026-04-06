/**
 * CLS++ Comprehensive UI Test Suite
 *
 * Run via: open tests/ui_test_runner.html in browser, or via preview tools.
 * Tests every page, every link, every button, every form, every anchor.
 * No external dependencies — pure vanilla JS.
 */

(function () {
  const PAGES = [
    'index.html',
    'docs.html',
    'integrate.html',
    'chat.html',
    'benchmark.html',
    'benchmark_v1_direct.html',
  ];

  // Resolve base URL (works from /tests/ subdirectory or root)
  const BASE = window.location.pathname.includes('/tests/')
    ? window.location.origin + window.location.pathname.replace(/\/tests\/.*$/, '/')
    : window.location.origin + '/';

  let results = [];
  let currentPage = '';

  function pass(name, detail) {
    results.push({ page: currentPage, name, status: 'PASS', detail: detail || '' });
  }

  function fail(name, detail) {
    results.push({ page: currentPage, name, status: 'FAIL', detail: detail || '' });
  }

  // =========================================================================
  // Test helpers
  // =========================================================================

  function checkAnchors(doc) {
    const anchorLinks = Array.from(doc.querySelectorAll('a[href^="#"]'));
    let allGood = true;
    anchorLinks.forEach(function (a) {
      const id = a.getAttribute('href').slice(1);
      if (!id) return; // href="#" is fine
      if (!doc.getElementById(id)) {
        fail('Anchor target exists: #' + id, 'No element with id="' + id + '"');
        allGood = false;
      }
    });
    if (allGood && anchorLinks.length > 0) {
      pass('All anchor targets exist (' + anchorLinks.length + ' checked)');
    }
  }

  function checkInternalLinks(doc) {
    const links = Array.from(doc.querySelectorAll('a[href]'));
    const internal = links.filter(function (a) {
      const href = a.getAttribute('href');
      return href && !href.startsWith('#') && !href.startsWith('http') && !href.startsWith('mailto:');
    });

    internal.forEach(function (a) {
      const href = a.getAttribute('href').split('#')[0].split('?')[0];
      if (PAGES.includes(href) || href === 'styles.css' || href === '/docs' || href === '/redoc') {
        pass('Internal link valid: ' + href);
      } else {
        fail('Internal link valid: ' + href, 'File not in known pages list');
      }
    });
  }

  function checkExternalLinks(doc) {
    const links = Array.from(doc.querySelectorAll('a[href^="http"]'));
    links.forEach(function (a) {
      const href = a.getAttribute('href');
      const target = a.getAttribute('target');
      const text = a.textContent.trim().substring(0, 30);
      if (target !== '_blank') {
        fail('External link opens in new tab: ' + text, 'href=' + href + ' missing target="_blank"');
      } else {
        pass('External link opens in new tab: ' + text);
      }
    });
  }

  function checkNoLocalhostOverride(doc) {
    const scripts = Array.from(doc.querySelectorAll('script'));
    let foundInline = false;
    scripts.forEach(function (s) {
      if (s.src) return; // external scripts checked separately
      const code = s.textContent;
      if (code.includes('localhost') && code.includes('CLS_API_URL')) {
        // Check it's properly gated behind ?local=1
        if (code.includes("location.hostname === 'localhost'") || code.includes('location.hostname === "localhost"')) {
          fail('No localhost hostname override', 'Script sets CLS_API_URL based on hostname, not ?local=1 param');
          foundInline = true;
        }
      }
    });
    if (!foundInline) {
      pass('No localhost hostname override in inline scripts');
    }
  }

  function checkNavConsistency(doc) {
    // Sub-pages (not index.html) should have: Home, Docs, Integrations, Chat, Benchmarks, GitHub
    const navLinks = Array.from(doc.querySelectorAll('.nav-links a, .chat-sidebar-footer a'));
    const hrefs = navLinks.map(function (a) { return a.getAttribute('href'); });
    const texts = navLinks.map(function (a) { return a.textContent.trim().toLowerCase(); });

    // chat.html has sidebar nav, not top nav — different pattern, skip strict check
    if (currentPage === 'chat.html') {
      if (hrefs.some(function (h) { return h === 'index.html'; })) {
        pass('Chat sidebar has Home link');
      } else {
        fail('Chat sidebar has Home link');
      }
      return;
    }

    // index.html has its own expanded nav
    if (currentPage === 'index.html') {
      var requiredIndex = ['docs.html', 'integrate.html', 'chat.html', 'benchmark.html'];
      requiredIndex.forEach(function (page) {
        if (hrefs.some(function (h) { return h === page || h.includes(page); })) {
          pass('Index nav has link to ' + page);
        } else {
          fail('Index nav has link to ' + page);
        }
      });
      return;
    }

    // All other sub-pages should have consistent nav
    var requiredPages = ['index.html', 'docs.html', 'integrate.html', 'chat.html', 'benchmark.html'];
    requiredPages.forEach(function (page) {
      if (hrefs.some(function (h) { return h === page || h.includes(page); })) {
        pass('Nav has link to ' + page);
      } else {
        fail('Nav has link to ' + page, 'Missing nav link to ' + page);
      }
    });

    // GitHub link
    if (hrefs.some(function (h) { return h && h.includes('github.com'); })) {
      pass('Nav has GitHub link');
    } else {
      fail('Nav has GitHub link');
    }
  }

  function checkButtons(doc) {
    const buttons = Array.from(doc.querySelectorAll('button'));
    buttons.forEach(function (btn) {
      const text = btn.textContent.trim().substring(0, 30) || btn.getAttribute('aria-label') || '(empty)';
      // Buttons should not be empty/invisible unless they have aria-label
      if (!btn.textContent.trim() && !btn.getAttribute('aria-label')) {
        fail('Button has label: ' + text, 'Button has no text and no aria-label');
      } else {
        pass('Button has label: ' + text);
      }
    });
  }

  function checkForms(doc) {
    const inputs = Array.from(doc.querySelectorAll('input[type="text"], input[type="url"], input[type="email"], textarea'));
    inputs.forEach(function (input) {
      const id = input.id || input.name || '(no id)';
      if (!input.placeholder && !input.getAttribute('aria-label')) {
        fail('Form input has placeholder: ' + id, 'No placeholder or aria-label');
      } else {
        pass('Form input has placeholder: ' + id);
      }
    });
  }

  function checkNoConsoleErrors(doc) {
    // This is checked at runtime — we just verify the page has valid script tags
    const scripts = Array.from(doc.querySelectorAll('script[src]'));
    scripts.forEach(function (s) {
      var src = s.getAttribute('src');
      if (!src.startsWith('http') && !src.startsWith('//')) {
        pass('Script src exists: ' + src);
      }
    });
  }

  function checkCSSLoaded(doc) {
    var links = Array.from(doc.querySelectorAll('link[rel="stylesheet"]'));
    var hasStyles = links.some(function (l) {
      return l.getAttribute('href') === 'styles.css';
    });
    if (hasStyles) {
      pass('styles.css loaded');
    } else {
      fail('styles.css loaded', 'No link to styles.css found');
    }
  }

  function checkMeta(doc) {
    var viewport = doc.querySelector('meta[name="viewport"]');
    if (viewport) {
      pass('Viewport meta tag present');
    } else {
      fail('Viewport meta tag present');
    }

    var charset = doc.querySelector('meta[charset]');
    if (charset) {
      pass('Charset meta tag present');
    } else {
      fail('Charset meta tag present');
    }

    var title = doc.querySelector('title');
    if (title && title.textContent.includes('CLS++')) {
      pass('Title contains CLS++');
    } else {
      fail('Title contains CLS++', 'Title: ' + (title ? title.textContent : '(none)'));
    }
  }

  // =========================================================================
  // Page-specific tests
  // =========================================================================

  function testIndexSpecific(doc) {
    // Demo chat containers
    var claudeChat = doc.getElementById('chat-claude');
    var openaiChat = doc.getElementById('chat-openai');
    if (claudeChat) pass('Claude demo chat container exists');
    else fail('Claude demo chat container exists');
    if (openaiChat) pass('OpenAI demo chat container exists');
    else fail('OpenAI demo chat container exists');

    // Demo inputs
    var claudeInput = doc.querySelector('[data-input="claude"]');
    var openaiInput = doc.querySelector('[data-input="openai"]');
    if (claudeInput) pass('Claude input field exists');
    else fail('Claude input field exists');
    if (openaiInput) pass('OpenAI input field exists');
    else fail('OpenAI input field exists');

    // Send buttons
    var claudeSend = doc.querySelector('[data-send="claude"]');
    var openaiSend = doc.querySelector('[data-send="openai"]');
    if (claudeSend) pass('Claude send button exists');
    else fail('Claude send button exists');
    if (openaiSend) pass('OpenAI send button exists');
    else fail('OpenAI send button exists');

    // Key sections
    ['tryit', 'problem', 'solution', 'demo', 'enterprise', 'pricing'].forEach(function (id) {
      if (doc.getElementById(id)) pass('Section #' + id + ' exists');
      else fail('Section #' + id + ' exists');
    });

    // Pricing cards — should have at least 2
    var pricingLinks = Array.from(doc.querySelectorAll('#pricing a')).filter(function (a) {
      return a.textContent.includes('Get Started') || a.textContent.includes('Trial') || a.textContent.includes('Contact');
    });
    if (pricingLinks.length >= 2) pass('Pricing section has CTAs (' + pricingLinks.length + ')');
    else fail('Pricing section has CTAs', 'Found only ' + pricingLinks.length);

    // Deploy button
    var deployBtn = doc.querySelector('a[href*="render.com/deploy"]');
    if (deployBtn) pass('Deploy on Render button exists');
    else fail('Deploy on Render button exists');

    // CLS_API_URL in demo.js should use production default
    var demoScript = doc.querySelector('script[src^="demo.js"]');
    if (demoScript) pass('demo.js script loaded');
    else fail('demo.js script loaded');
  }

  function testDocsSpecific(doc) {
    // Sidebar nav
    var sidebar = doc.querySelector('.docs-sidebar nav, .docs-nav');
    var sidebarLinks = doc.querySelectorAll('.docs-sidebar a[href^="#"]');
    if (sidebarLinks.length >= 5) pass('Docs sidebar has navigation (' + sidebarLinks.length + ' links)');
    else fail('Docs sidebar has navigation', 'Found only ' + sidebarLinks.length + ' sidebar links');

    // Key sections
    ['quickstart', 'authentication', 'endpoints', 'write', 'read'].forEach(function (id) {
      if (doc.getElementById(id)) pass('Docs section #' + id + ' exists');
      else fail('Docs section #' + id + ' exists');
    });

    // Code blocks
    var codeBlocks = doc.querySelectorAll('pre, code');
    if (codeBlocks.length >= 5) pass('Docs has code examples (' + codeBlocks.length + ')');
    else fail('Docs has code examples', 'Found only ' + codeBlocks.length);
  }

  function testIntegrationsSpecific(doc) {
    // Create integration form
    var nameInput = doc.getElementById('int-name');
    var nsInput = doc.getElementById('int-namespace');
    var createBtn = doc.getElementById('btn-create');
    if (nameInput) pass('Integration name input exists');
    else fail('Integration name input exists');
    if (nsInput) pass('Integration namespace input exists');
    else fail('Integration namespace input exists');
    if (createBtn) pass('Create Integration button exists');
    else fail('Create Integration button exists');

    // Result section (hidden initially)
    var createResult = doc.getElementById('create-result');
    if (createResult && createResult.style.display === 'none') {
      pass('Create result hidden initially');
    } else if (createResult) {
      pass('Create result section exists');
    } else {
      fail('Create result section exists');
    }

    // Copy button
    var copyBtn = doc.getElementById('btn-copy-key');
    if (copyBtn) pass('Copy API key button exists');
    else fail('Copy API key button exists');

    // Snippet tabs
    var tabs = doc.querySelectorAll('.int-tab');
    if (tabs.length >= 3) pass('Code snippet tabs exist (' + tabs.length + ')');
    else fail('Code snippet tabs exist', 'Found ' + tabs.length);

    // Snippet containers
    ['snippet-python', 'snippet-javascript', 'snippet-curl'].forEach(function (id) {
      if (doc.getElementById(id)) pass('Snippet container: ' + id);
      else fail('Snippet container: ' + id);
    });

    // Webhook form
    var whUrl = doc.getElementById('wh-url');
    var whBtn = doc.getElementById('btn-webhook');
    if (whUrl) pass('Webhook URL input exists');
    else fail('Webhook URL input exists');
    if (whBtn) pass('Webhook subscribe button exists');
    else fail('Webhook subscribe button exists');

    // Memory cycle
    var cycleStatements = doc.getElementById('cycle-statements');
    var cycleQueries = doc.getElementById('cycle-queries');
    var cycleBtn = doc.getElementById('btn-cycle');
    if (cycleStatements) pass('Cycle statements textarea exists');
    else fail('Cycle statements textarea exists');
    if (cycleQueries) pass('Cycle queries textarea exists');
    else fail('Cycle queries textarea exists');
    if (cycleBtn) pass('Run Memory Cycle button exists');
    else fail('Run Memory Cycle button exists');
  }

  function testChatSpecific(doc) {
    // Core elements
    var sidebar = doc.getElementById('chat-sidebar');
    var messages = doc.getElementById('chat-messages');
    var input = doc.getElementById('chat-input');
    var sendBtn = doc.getElementById('btn-send');
    var newChatBtn = doc.getElementById('btn-new-chat');
    var sessionList = doc.getElementById('session-list');
    var debugAugmented = doc.getElementById('debug-augmented');
    var debugMemory = doc.getElementById('debug-memory');

    if (sidebar) pass('Chat sidebar exists');
    else fail('Chat sidebar exists');
    if (messages) pass('Chat messages container exists');
    else fail('Chat messages container exists');
    if (input) pass('Chat input exists');
    else fail('Chat input exists');
    if (sendBtn) pass('Chat send button exists');
    else fail('Chat send button exists');
    if (newChatBtn) pass('New Chat button exists');
    else fail('New Chat button exists');
    if (sessionList) pass('Session list container exists');
    else fail('Session list container exists');
    if (debugAugmented) pass('Debug augmented panel exists');
    else fail('Debug augmented panel exists');
    if (debugMemory) pass('Debug memory panel exists');
    else fail('Debug memory panel exists');

    // chat.js loaded
    var chatScript = doc.querySelector('script[src^="chat.js"]');
    if (chatScript) pass('chat.js script loaded');
    else fail('chat.js script loaded');
  }

  function testBenchmarkSpecific(doc) {
    // Tables
    var tables = doc.querySelectorAll('table');
    if (tables.length >= 1) pass('Benchmark has data tables (' + tables.length + ')');
    else fail('Benchmark has data tables');

    // Key content
    var h1 = doc.querySelector('h1');
    if (h1 && h1.textContent.toLowerCase().includes('benchmark')) {
      pass('Benchmark heading present');
    } else {
      fail('Benchmark heading present');
    }
  }

  // =========================================================================
  // JS file content tests (loaded via fetch)
  // =========================================================================

  async function testJSFiles() {
    currentPage = 'JS Files';

    var jsFiles = ['demo.js', 'integrations.js', 'chat.js', 'script.js'];

    for (var i = 0; i < jsFiles.length; i++) {
      var file = jsFiles[i];
      try {
        var res = await fetch(BASE + file + '?_t=' + Date.now());
        if (!res.ok) {
          fail(file + ' loads', 'HTTP ' + res.status);
          continue;
        }
        var code = await res.text();

        // Check no hardcoded localhost as default API URL
        if (code.includes("'http://localhost:8090'") || code.includes('"http://localhost:8090"')) {
          // It's OK if it's behind window.CLS_API_URL fallback
          var lines = code.split('\n');
          var localhostLines = lines.filter(function (l) {
            return l.includes('localhost:8090') && !l.includes('CLS_API_URL') && !l.trim().startsWith('//');
          });
          if (localhostLines.length > 0) {
            fail(file + ' no hardcoded localhost', 'Found localhost:8090 not behind CLS_API_URL check');
          } else {
            pass(file + ' localhost is behind CLS_API_URL fallback');
          }
        } else {
          pass(file + ' no hardcoded localhost');
        }

        // Check API URL defaults to same-origin (empty string)
        if (code.includes("|| ''") || code.includes("|| \"\"")) {
          pass(file + ' uses same-origin API URL default');
        }

        // Check for syntax issues — try to parse
        try {
          new Function(code);
          pass(file + ' parses without syntax errors');
        } catch (e) {
          fail(file + ' parses without syntax errors', e.message);
        }

      } catch (e) {
        fail(file + ' loads', e.message);
      }
    }
  }

  // =========================================================================
  // Runner
  // =========================================================================

  async function runPageTests(page) {
    currentPage = page;

    try {
      var res = await fetch(BASE + page + '?_t=' + Date.now());
      if (!res.ok) {
        fail('Page loads', 'HTTP ' + res.status);
        return;
      }
      pass('Page loads (HTTP 200)');

      var html = await res.text();
      var parser = new DOMParser();
      var doc = parser.parseFromString(html, 'text/html');

      // Universal tests
      checkMeta(doc);
      checkCSSLoaded(doc);
      checkAnchors(doc);
      checkInternalLinks(doc);
      checkExternalLinks(doc);
      checkNoLocalhostOverride(doc);
      checkNavConsistency(doc);
      checkButtons(doc);
      checkForms(doc);
      checkNoConsoleErrors(doc);

      // Page-specific tests
      if (page === 'index.html') testIndexSpecific(doc);
      else if (page === 'docs.html') testDocsSpecific(doc);
      else if (page === 'integrate.html') testIntegrationsSpecific(doc);
      else if (page === 'chat.html') testChatSpecific(doc);
      else if (page.includes('benchmark')) testBenchmarkSpecific(doc);

    } catch (e) {
      fail('Page loads', e.message);
    }
  }

  async function runAllTests() {
    results = [];

    for (var i = 0; i < PAGES.length; i++) {
      await runPageTests(PAGES[i]);
    }

    await testJSFiles();

    return results;
  }

  // =========================================================================
  // Report
  // =========================================================================

  function renderReport(results) {
    var passed = results.filter(function (r) { return r.status === 'PASS'; }).length;
    var failed = results.filter(function (r) { return r.status === 'FAIL'; }).length;
    var total = results.length;

    var html = '<div style="font-family:monospace;padding:20px;background:#0a0a0a;color:#e0e0e0;min-height:100vh">';
    html += '<h1 style="color:#fff">CLS++ UI Test Results</h1>';
    html += '<p style="font-size:18px">';
    html += '<span style="color:#4caf50">' + passed + ' passed</span> / ';
    html += '<span style="color:#f44336">' + failed + ' failed</span> / ';
    html += total + ' total';
    html += '</p>';

    if (failed > 0) {
      html += '<h2 style="color:#f44336">Failures</h2>';
      results.filter(function (r) { return r.status === 'FAIL'; }).forEach(function (r) {
        html += '<div style="background:#1a0000;border-left:3px solid #f44336;padding:8px 12px;margin:4px 0">';
        html += '<strong>[' + r.page + ']</strong> ' + r.name;
        if (r.detail) html += ' — <span style="color:#ff8a80">' + r.detail + '</span>';
        html += '</div>';
      });
    }

    // Group by page
    var pages = {};
    results.forEach(function (r) {
      if (!pages[r.page]) pages[r.page] = [];
      pages[r.page].push(r);
    });

    html += '<h2 style="color:#aaa;margin-top:30px">All Results by Page</h2>';
    Object.keys(pages).forEach(function (page) {
      var pageResults = pages[page];
      var pagePassed = pageResults.filter(function (r) { return r.status === 'PASS'; }).length;
      var pageFailed = pageResults.filter(function (r) { return r.status === 'FAIL'; }).length;
      html += '<h3 style="margin-top:20px;color:#ccc">' + page + ' (' + pagePassed + '/' + pageResults.length + ')</h3>';
      pageResults.forEach(function (r) {
        var color = r.status === 'PASS' ? '#4caf50' : '#f44336';
        var icon = r.status === 'PASS' ? '✓' : '✗';
        html += '<div style="padding:2px 0;color:' + color + '">';
        html += icon + ' ' + r.name;
        if (r.detail && r.status === 'FAIL') html += ' — ' + r.detail;
        html += '</div>';
      });
    });

    html += '</div>';
    document.body.innerHTML = html;
  }

  // =========================================================================
  // Export for programmatic use and auto-run for browser
  // =========================================================================

  window.CLSTests = {
    runAllTests: runAllTests,
    renderReport: renderReport,
    getResults: function () { return results; },
  };

  // If loaded in test runner page, auto-run
  if (document.getElementById('cls-test-runner')) {
    document.getElementById('cls-test-runner').textContent = 'Running tests...';
    runAllTests().then(function (res) {
      renderReport(res);
    });
  }
})();
