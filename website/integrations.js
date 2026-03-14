/**
 * CLS++ Integration Dashboard — Zero-code integration management.
 * Create integrations, get API keys, subscribe to webhooks, run memory cycles.
 */
(function () {
  const API_URL = (typeof window !== 'undefined' && window.CLS_API_URL) || 'https://clsplusplus-api.onrender.com';

  let currentIntegrationId = null;

  // =========================================================================
  // Utility
  // =========================================================================

  function copyToClipboard(text, btn) {
    navigator.clipboard.writeText(text).then(function () {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('btn-success');
      setTimeout(function () {
        btn.textContent = orig;
        btn.classList.remove('btn-success');
      }, 2000);
    });
  }

  function setLoading(btn, loading) {
    btn.disabled = loading;
    if (loading) {
      btn.dataset.origText = btn.textContent;
      btn.textContent = 'Working...';
    } else {
      btn.textContent = btn.dataset.origText || btn.textContent;
    }
  }

  async function apiCall(path, method, body) {
    const opts = {
      method: method || 'GET',
      headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(API_URL + path, opts);
    if (!res.ok) {
      const err = await res.json().catch(function () { return {}; });
      throw new Error(err.message || err.detail || res.statusText);
    }
    return res.json();
  }

  // =========================================================================
  // Create Integration
  // =========================================================================

  const btnCreate = document.getElementById('btn-create');
  if (btnCreate) {
    btnCreate.addEventListener('click', async function () {
      const name = document.getElementById('int-name').value.trim();
      const namespace = document.getElementById('int-namespace').value.trim() || 'default';

      if (!name) {
        alert('Please enter an app name.');
        return;
      }

      setLoading(btnCreate, true);
      try {
        const data = await apiCall('/v1/integrations', 'POST', {
          name: name,
          namespace: namespace,
        });

        currentIntegrationId = data.integration.id;
        const apiKey = data.api_key.key;

        // Show result
        document.getElementById('result-name').textContent = name;
        document.getElementById('result-key').textContent = apiKey;

        // Update snippets with actual key
        document.querySelectorAll('.snippet-key').forEach(function (el) {
          el.textContent = apiKey;
        });

        document.getElementById('create-result').style.display = 'block';
        document.getElementById('webhook-form').style.display = 'block';

        // Scroll to result
        document.getElementById('create-result').scrollIntoView({ behavior: 'smooth', block: 'center' });
      } catch (e) {
        alert('Error: ' + e.message);
      } finally {
        setLoading(btnCreate, false);
      }
    });
  }

  // Copy API key
  const btnCopyKey = document.getElementById('btn-copy-key');
  if (btnCopyKey) {
    btnCopyKey.addEventListener('click', function () {
      const key = document.getElementById('result-key').textContent;
      copyToClipboard(key, btnCopyKey);
    });
  }

  // =========================================================================
  // Code Snippet Tabs
  // =========================================================================

  document.querySelectorAll('.int-tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      document.querySelectorAll('.int-tab').forEach(function (t) { t.classList.remove('active'); });
      tab.classList.add('active');

      const lang = tab.dataset.lang;
      document.querySelectorAll('.int-code').forEach(function (el) { el.style.display = 'none'; });
      const target = document.getElementById('snippet-' + lang);
      if (target) target.style.display = 'block';
    });
  });

  // =========================================================================
  // Webhook Subscription
  // =========================================================================

  const btnWebhook = document.getElementById('btn-webhook');
  if (btnWebhook) {
    btnWebhook.addEventListener('click', async function () {
      if (!currentIntegrationId) {
        alert('Create an integration first.');
        return;
      }

      const url = document.getElementById('wh-url').value.trim();
      if (!url) {
        alert('Please enter a webhook URL.');
        return;
      }

      // Collect checked events
      const events = [];
      document.querySelectorAll('#webhook-form input[type="checkbox"]:checked').forEach(function (cb) {
        events.push(cb.value);
      });
      if (events.length === 0) events.push('*');
      // If "all" is selected, just send "*"
      if (events.includes('*')) {
        events.length = 0;
        events.push('*');
      }

      setLoading(btnWebhook, true);
      try {
        const data = await apiCall(
          '/v1/integrations/' + currentIntegrationId + '/webhooks',
          'POST',
          { url: url, events: events }
        );

        document.getElementById('wh-secret').textContent = data.webhook.secret;
        document.getElementById('webhook-result').style.display = 'block';
        document.getElementById('webhook-result').scrollIntoView({ behavior: 'smooth', block: 'center' });
      } catch (e) {
        alert('Error: ' + e.message);
      } finally {
        setLoading(btnWebhook, false);
      }
    });
  }

  // Copy webhook secret
  const btnCopySecret = document.getElementById('btn-copy-secret');
  if (btnCopySecret) {
    btnCopySecret.addEventListener('click', function () {
      const secret = document.getElementById('wh-secret').textContent;
      copyToClipboard(secret, btnCopySecret);
    });
  }

  // =========================================================================
  // Memory Cycle Test
  // =========================================================================

  const btnCycle = document.getElementById('btn-cycle');
  if (btnCycle) {
    btnCycle.addEventListener('click', async function () {
      const statementsRaw = document.getElementById('cycle-statements').value.trim();
      const queriesRaw = document.getElementById('cycle-queries').value.trim();

      const statements = statementsRaw.split('\n').map(function (s) { return s.trim(); }).filter(Boolean);
      const queries = queriesRaw.split('\n').map(function (s) { return s.trim(); }).filter(Boolean);

      if (statements.length === 0) {
        alert('Enter at least one statement.');
        return;
      }
      if (queries.length === 0) {
        alert('Enter at least one query.');
        return;
      }

      setLoading(btnCycle, true);
      document.getElementById('cycle-result').style.display = 'block';
      document.getElementById('cycle-phases').innerHTML = '<p class="int-loading">Running memory cycle...</p>';
      document.getElementById('cycle-verdict').textContent = '';

      try {
        const namespace = 'cycle-' + Math.random().toString(36).slice(2, 10);
        const data = await apiCall('/v1/demo/memory-cycle', 'POST', {
          statements: statements,
          queries: queries,
          models: ['claude', 'openai'],
          namespace: namespace,
        });

        renderCycleResult(data);
      } catch (e) {
        document.getElementById('cycle-phases').innerHTML =
          '<div class="int-error">Error: ' + e.message + '</div>';
      } finally {
        setLoading(btnCycle, false);
      }
    });
  }

  function renderCycleResult(data) {
    const phases = data.phases;
    let html = '';

    // Encode
    html += '<div class="int-phase">';
    html += '<h4>Phase 1: Encode</h4>';
    html += '<p>Stored <strong>' + phases.encode.stored + '/' + phases.encode.total + '</strong> memories</p>';
    html += '</div>';

    // Retrieve
    html += '<div class="int-phase">';
    html += '<h4>Phase 2: Retrieve</h4>';
    html += '<p>Found <strong>' + phases.retrieve.total_found + '</strong> items (avg confidence: ' + phases.retrieve.confidence_avg + ')</p>';
    html += '</div>';

    // Augment
    html += '<div class="int-phase">';
    html += '<h4>Phase 3: Augment (LLM with Memory)</h4>';
    for (const model in phases.augment) {
      phases.augment[model].forEach(function (r) {
        html += '<div class="int-augment-result">';
        html += '<span class="int-badge">' + model + '</span> ';
        html += '<strong>' + r.query + '</strong><br>';
        html += '<span class="int-response">' + (r.response || r.error || '') + '</span>';
        html += '<span class="int-meta"> (' + (r.memory_context_items || 0) + ' memory items used)</span>';
        html += '</div>';
      });
    }
    html += '</div>';

    // Cross-session
    html += '<div class="int-phase">';
    html += '<h4>Phase 4: Cross-Session Persistence</h4>';
    html += '<p>Memories persisted: <strong>' + (phases.cross_session.memories_persisted ? 'Yes' : 'No') + '</strong>';
    html += ' (' + phases.cross_session.items_found + ' items found)</p>';
    html += '</div>';

    document.getElementById('cycle-phases').innerHTML = html;

    // Verdict
    const verdictEl = document.getElementById('cycle-verdict');
    verdictEl.textContent = data.verdict;
    verdictEl.className = 'int-verdict ' + (data.verdict === 'PASS' ? 'int-verdict-pass' : 'int-verdict-fail');
  }
})();
