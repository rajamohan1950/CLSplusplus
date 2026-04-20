/**
 * CLS++ Getting Started Page
 * Handles integration creation, code snippet generation, and live testing.
 */
(function () {
  'use strict';

  var _apiKey = null;

  // Track onboarding page view
  if (window.CLSAnalytics) CLSAnalytics.track('onboarding_started');

  // ── Init ──────────────────────────────────────────────────────────────────

  async function init() {
    // Check if user is logged in
    if (typeof CLSAuth !== 'undefined') {
      var user = await CLSAuth.getUser();
      if (user) {
        document.getElementById('step2-auth').style.display = 'block';
        document.getElementById('step2-signin').style.display = 'none';
      } else {
        document.getElementById('step2-auth').style.display = 'none';
        document.getElementById('step2-signin').style.display = 'block';
      }
    }
  }

  // ── Language tabs ─────────────────────────────────────────────────────────

  window.showLang = function (lang) {
    ['openai', 'anthropic', 'langchain'].forEach(function (l) {
      var el = document.getElementById('install-' + l);
      if (el) el.style.display = (l === lang) ? 'block' : 'none';
    });
    document.querySelectorAll('.gs-tab').forEach(function (t) {
      t.classList.toggle('active', t.textContent.toLowerCase().includes(lang));
    });
  };

  // ── Playground learn/ask for Step 3 ───────────────────────────────────────

  window.pgLearnGS = async function () {
    var fact = document.getElementById('test-fact').value.trim();
    var res = document.getElementById('test-results');
    var err = document.getElementById('step3-error');
    if (!fact) return;
    err.style.display = 'none';
    try {
      var r = await fetch('/v1/memory/write', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({text: fact, namespace: 'playground-user'})});
      if (r.ok) {
        res.innerHTML = '<div class="gs-test-phase"><span class="phase-name">Learned</span><span class="phase-status pass">"' + fact + '"</span></div>';
        res.style.display = 'block';
        document.getElementById('test-fact').value = '';
      }
    } catch(e) { err.textContent = 'Server not reachable'; err.style.display = 'block'; }
  };

  window.pgAskGS = async function () {
    var query = document.getElementById('test-query').value.trim();
    var res = document.getElementById('test-results');
    var err = document.getElementById('step3-error');
    if (!query) return;
    err.style.display = 'none';
    try {
      var r = await fetch('/v1/memory/read', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({query: query, namespace: 'playground-user', limit: 5})});
      if (r.ok) {
        var data = await r.json();
        var items = (data.items || []);
        if (items.length) {
          var html = '';
          items.forEach(function(i) { html += '<div class="gs-test-phase"><span class="phase-name">Recalled</span><span class="phase-status pass">' + i.text + '</span></div>'; });
          res.innerHTML = html;
        } else {
          res.innerHTML = '<div class="gs-test-phase"><span class="phase-name">Result</span><span class="phase-status fail">No memories yet. Click Learn first!</span></div>';
        }
        res.style.display = 'block';
      }
    } catch(e) { err.textContent = 'Server not reachable'; err.style.display = 'block'; }
  };

  // ── Copy to clipboard ────────────────────────────────────────────────────

  window.copyText = function (btn) {
    var block = btn.closest('.gs-code') || btn.closest('.key-display');
    var text = '';
    if (block) {
      // Get text content excluding the button
      var clone = block.cloneNode(true);
      var buttons = clone.querySelectorAll('.copy-btn');
      buttons.forEach(function (b) { b.remove(); });
      text = clone.textContent.trim();
    }
    navigator.clipboard.writeText(text).then(function () {
      var orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(function () { btn.textContent = orig; }, 1500);
    });
  };

  // ── Create integration ────────────────────────────────────────────────────

  window.createIntegration = async function () {
    var nameInput = document.getElementById('app-name');
    var btn = document.getElementById('create-btn');
    var errEl = document.getElementById('step2-error');
    var resEl = document.getElementById('step2-result');
    var name = nameInput.value.trim();

    if (!name) { nameInput.focus(); return; }
    errEl.style.display = 'none';
    resEl.style.display = 'none';
    btn.disabled = true;
    btn.textContent = 'Creating...';

    try {
      // Get user namespace
      var user = await CLSAuth.getUser();
      var ns = user ? 'user-' + user.id.slice(0, 8) : 'default';

      var resp = await fetch('/v1/integrations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          name: name,
          namespace: ns,
          owner_email: user ? user.email : '',
        }),
      });

      if (resp.ok) {
        var data = await resp.json();
        var key = data.api_key && data.api_key.key ? data.api_key.key : 'Key created (check dashboard)';
        _apiKey = key;
        document.getElementById('api-key-display').textContent = key;
        resEl.style.display = 'block';

        // Update code snippet with real key
        var codeKey = document.getElementById('code-key');
        if (codeKey) codeKey.textContent = key;
      } else {
        var err = await resp.json();
        errEl.textContent = err.message || err.detail || 'Failed to create integration';
        errEl.style.display = 'block';
      }
    } catch (e) {
      errEl.textContent = 'Network error. Please try again.';
      errEl.style.display = 'block';
    }

    btn.disabled = false;
    btn.textContent = 'Create API Key';
  };

  // ── Live test ─────────────────────────────────────────────────────────────

  window.runTest = async function () {
    var fact = document.getElementById('test-fact').value.trim();
    var query = document.getElementById('test-query').value.trim();
    var btn = document.getElementById('test-btn');
    var errEl = document.getElementById('step3-error');
    var resEl = document.getElementById('test-results');

    if (!fact || !query) {
      errEl.textContent = 'Enter both a fact and a query to test.';
      errEl.style.display = 'block';
      return;
    }

    errEl.style.display = 'none';
    resEl.style.display = 'none';
    btn.disabled = true;
    btn.textContent = 'Testing...';

    try {
      var resp = await fetch('/v1/demo/memory-cycle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          statements: [fact],
          queries: [query],
          models: ['claude'],
          namespace: 'gs-test-' + Math.random().toString(36).slice(2, 8),
        }),
      });

      if (resp.ok) {
        var data = await resp.json();
        resEl.innerHTML = '';

        // Encode phase
        var enc = data.encode || {};
        addPhase(resEl, 'Encode', enc.stored + '/' + enc.total + ' stored', enc.stored > 0);

        // Retrieve phase
        var ret = data.retrieve || {};
        var retCount = 0;
        if (ret.results) {
          for (var q in ret.results) { retCount += (ret.results[q].length || 0); }
        }
        addPhase(resEl, 'Retrieve', retCount + ' memories found', retCount > 0);

        // Augment phase
        var aug = data.augment || {};
        var augOk = false;
        if (aug.results) {
          for (var m in aug.results) {
            var r = aug.results[m];
            if (r && r.length > 0 && r[0].reply) {
              addPhase(resEl, 'LLM Response', r[0].reply.slice(0, 150) + '...', true);
              augOk = true;
            }
          }
        }
        if (!augOk) addPhase(resEl, 'LLM Response', 'No LLM configured (set API keys to test)', false);

        // Cross-session
        var cross = data.cross_session || {};
        addPhase(resEl, 'Cross-Session', cross.persisted ? 'Memory persists across sessions' : 'Persistence check', cross.persisted);

        resEl.style.display = 'block';
      } else {
        var err = await resp.json();
        errEl.textContent = err.message || err.detail || 'Test failed';
        errEl.style.display = 'block';
      }
    } catch (e) {
      errEl.textContent = 'Test failed: ' + e.message;
      errEl.style.display = 'block';
    }

    btn.disabled = false;
    btn.textContent = 'Run Test';
  };

  function addPhase(container, name, detail, ok) {
    var div = document.createElement('div');
    div.className = 'gs-test-phase';
    div.innerHTML = '<span class="phase-name">' + name + '</span><span class="phase-status ' + (ok ? 'pass' : 'fail') + '">' + detail + '</span>';
    container.appendChild(div);
  }

  // ── Start ─────────────────────────────────────────────────────────────────

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
