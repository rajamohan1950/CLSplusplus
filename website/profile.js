/**
 * CLS++ Profile Page Logic
 * Handles user info, API key CRUD, instant connect, and billing.
 */
(function () {
  'use strict';

  var _user = null;
  var _integrations = [];
  var _selectedIntegration = null;
  var _instantKey = null; // key from instant connect

  var API_URL = 'https://www.clsplusplus.com';

  async function loadProfile() {
    if (typeof CLSAuth !== 'undefined') {
      await CLSAuth.requireAuth();
    }

    _user = (typeof CLSAuth !== 'undefined') ? await CLSAuth.getUser() : null;
    if (!_user) return;

    // Render sidebar
    if (typeof renderSidebar === 'function') {
      await renderSidebar('profile');
    }

    // Tier badge
    var badge = document.getElementById('tier-badge');
    if (badge) {
      var tierLabel = _user.tier.charAt(0).toUpperCase() + _user.tier.slice(1);
      badge.textContent = tierLabel;
      badge.className = 'tier-badge tier-' + _user.tier;
    }

    // User info
    renderUserInfo();

    // Show change password button
    var pwBtn = document.getElementById('btn-change-pw');
    if (pwBtn) pwBtn.style.display = 'inline-block';

    // Preferences
    var ns = document.getElementById('pref-namespace');
    if (ns) ns.textContent = 'user-' + _user.id.slice(0, 8);

    // Detect extension
    if (window.__CLS_EXTENSION_INSTALLED__ || document.querySelector('[data-cls-extension]')) {
      var extOpt = document.getElementById('ic-ext-option');
      if (extOpt) extOpt.style.display = 'block';
    }

    // Load integrations (auto-creates if none)
    await loadIntegrations();

    // Load billing info
    await loadBilling();

    // Update plan cards
    updatePlanCards(_user.tier);

    // Handle hash navigation
    handleHash();

    // Check URL params for billing result
    var params = new URLSearchParams(window.location.search);
    if (params.get('billing') === 'success') {
      showInlineMsg('billing', 'Subscription updated successfully!', 'var(--success)');
    }
  }

  function renderUserInfo() {
    if (!_user) return;

    var avatarEl = document.getElementById('profile-avatar');
    if (avatarEl) {
      if (_user.avatar_url) {
        avatarEl.innerHTML = '<img src="' + _user.avatar_url + '" style="width:40px;height:40px;border-radius:50%;object-fit:cover;" alt="">';
      } else {
        avatarEl.innerHTML = '<span style="width:40px;height:40px;border-radius:50%;background:var(--accent);color:#fff;display:inline-flex;align-items:center;justify-content:center;font-weight:600;">' + (_user.name || _user.email)[0].toUpperCase() + '</span>';
      }
    }

    setText('profile-name', _user.name || '--');
    setText('profile-email', _user.email);
    setText('profile-tier', _user.tier.charAt(0).toUpperCase() + _user.tier.slice(1));
    setText('profile-since', _user.created_at ? new Date(_user.created_at).toLocaleDateString() : '--');

    var nameInput = document.getElementById('edit-name');
    var emailInput = document.getElementById('edit-email');
    if (nameInput) nameInput.value = _user.name || '';
    if (emailInput) emailInput.value = _user.email || '';
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function showInlineMsg(section, msg, color) {
    // Generic inline message near a section
    var el = document.getElementById(section + '-msg') || document.getElementById('edit-profile-msg');
    if (el) { el.textContent = msg; el.style.color = color || 'var(--text-muted)'; }
  }

  // ── Hash Navigation ────────────────────────────────────────────────────────

  function handleHash() {
    var hash = window.location.hash.replace('#', '');
    if (hash) {
      var target = document.getElementById(hash);
      if (target) {
        setTimeout(function () {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
      }
    }
  }

  window.addEventListener('hashchange', handleHash);

  // ── Edit Profile ───────────────────────────────────────────────────────────

  window.toggleEditProfile = function () {
    var form = document.getElementById('edit-profile-form');
    if (form) form.classList.toggle('visible');
    var pwForm = document.getElementById('change-pw-form');
    if (pwForm) pwForm.classList.remove('visible');
  };

  window.toggleChangePw = function () {
    var form = document.getElementById('change-pw-form');
    if (form) form.classList.toggle('visible');
    var editForm = document.getElementById('edit-profile-form');
    if (editForm) editForm.classList.remove('visible');
  };

  window.saveProfile = async function () {
    var name = document.getElementById('edit-name').value.trim();
    var email = document.getElementById('edit-email').value.trim();
    var body = {};
    if (name && name !== _user.name) body.name = name;
    if (email && email !== _user.email) body.email = email;
    if (!Object.keys(body).length) {
      showInlineMsg('edit-profile', 'No changes to save.', 'var(--text-muted)');
      return;
    }

    try {
      var resp = await fetch('/v1/user/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(body),
      });
      if (resp.ok) {
        _user = await resp.json();
        renderUserInfo();
        document.getElementById('edit-profile-form').classList.remove('visible');
        showInlineMsg('edit-profile', '', '');
      } else {
        var err = await resp.json();
        showInlineMsg('edit-profile', err.detail || 'Update failed', '#ef4444');
      }
    } catch (e) {
      showInlineMsg('edit-profile', 'Network error', '#ef4444');
    }
  };

  window.changePassword = async function () {
    var current = document.getElementById('pw-current').value;
    var newPw = document.getElementById('pw-new').value;
    if (!current || !newPw) {
      showInlineMsg('change-pw', 'Both fields are required.', '#ef4444');
      return;
    }
    if (newPw.length < 8) {
      showInlineMsg('change-pw', 'Password must be at least 8 characters.', '#ef4444');
      return;
    }

    try {
      var resp = await fetch('/v1/user/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ password: newPw, current_password: current }),
      });
      if (resp.ok) {
        var msgEl = document.getElementById('change-pw-msg');
        if (msgEl) { msgEl.textContent = 'Password updated.'; msgEl.style.color = 'var(--success)'; }
        document.getElementById('pw-current').value = '';
        document.getElementById('pw-new').value = '';
        setTimeout(function () {
          document.getElementById('change-pw-form').classList.remove('visible');
          if (msgEl) { msgEl.textContent = ''; }
        }, 1500);
      } else {
        var err = await resp.json();
        var msgEl = document.getElementById('change-pw-msg');
        if (msgEl) { msgEl.textContent = err.detail || 'Failed'; msgEl.style.color = '#ef4444'; }
      }
    } catch (e) {
      var msgEl = document.getElementById('change-pw-msg');
      if (msgEl) { msgEl.textContent = 'Network error'; msgEl.style.color = '#ef4444'; }
    }
  };

  // ── Integrations & API Keys ────────────────────────────────────────────────

  async function loadIntegrations() {
    var select = document.getElementById('key-integration');
    try {
      var resp = await fetch('/v1/user/integrations', { credentials: 'same-origin' });
      if (resp.ok) {
        var data = await resp.json();
        _integrations = data.integrations || data || [];
      }
    } catch (e) {
      _integrations = [];
    }

    // Auto-create default integration if none exist
    if (!_integrations.length && _user) {
      try {
        var ns = 'user-' + _user.id.slice(0, 8);
        var createResp = await fetch('/v1/integrations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ name: 'My App', namespace: ns, owner_email: _user.email }),
        });
        if (createResp.ok) {
          var createData = await createResp.json();
          // Store the first key from auto-creation
          if (createData.api_key && createData.api_key.key) {
            _instantKey = createData.api_key.key;
          }
          // Reload integrations
          var reloadResp = await fetch('/v1/user/integrations', { credentials: 'same-origin' });
          if (reloadResp.ok) {
            var reloadData = await reloadResp.json();
            _integrations = reloadData.integrations || [];
          }
        }
      } catch (e) { /* silent */ }
    }

    if (!select) return;
    select.innerHTML = '';

    if (!_integrations.length) {
      select.innerHTML = '<option value="">No integrations yet</option>';
      return;
    }

    _integrations.forEach(function (int) {
      var opt = document.createElement('option');
      opt.value = int.id;
      opt.textContent = int.name + ' (' + int.namespace + ')';
      select.appendChild(opt);
    });

    _selectedIntegration = _integrations[0].id;
    select.value = _selectedIntegration;
    select.addEventListener('change', function () {
      _selectedIntegration = this.value;
      loadKeys();
      loadKeyHistory();
    });

    await loadKeys();
    await loadKeyHistory();
  }

  window.createNewIntegration = async function () {
    var name = prompt('Integration name:');
    if (!name || !name.trim()) return;

    try {
      var ns = 'user-' + _user.id.slice(0, 8);
      var resp = await fetch('/v1/integrations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ name: name.trim(), namespace: ns, owner_email: _user.email }),
      });
      if (resp.ok) {
        var data = await resp.json();
        if (data.api_key && data.api_key.key) {
          document.getElementById('new-key-value').textContent = data.api_key.key;
          document.getElementById('new-key-box').classList.add('visible');
        }
        await loadIntegrations();
      } else {
        var err = await resp.json();
        alert(err.detail || 'Failed to create integration');
      }
    } catch (e) {
      alert('Network error');
    }
  };

  async function loadKeys() {
    var container = document.getElementById('keys-table-container');
    if (!_selectedIntegration) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">No integration selected.</p>';
      return;
    }

    try {
      var resp = await fetch('/v1/integrations/' + _selectedIntegration + '/keys', { credentials: 'same-origin' });
      if (!resp.ok) throw new Error();
      var keys = await resp.json();
      if (!Array.isArray(keys)) keys = keys.keys || [];
      renderKeysTable(keys);
    } catch (e) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">Failed to load keys.</p>';
    }
  }

  function renderKeysTable(keys) {
    var container = document.getElementById('keys-table-container');
    if (!keys.length) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">No API keys yet. Generate one above or use Instant Connect.</p>';
      return;
    }

    var html = '<table class="keys-table">';
    html += '<tr><th>Label</th><th>Key</th><th>Scopes</th><th>Status</th><th>Created</th><th>Expires</th><th>Actions</th></tr>';
    keys.forEach(function (k) {
      var statusClass = 'status-' + k.status;
      var created = k.created_at ? new Date(k.created_at).toLocaleDateString() : '--';
      var expires = k.expires_at ? new Date(k.expires_at).toLocaleDateString() : 'Never';
      var scopes = (k.scopes || []).map(function (s) { return '<span class="scope-badge">' + s + '</span>'; }).join(' ');
      var actions = '';
      if (k.status === 'active') {
        actions += '<button class="key-action-btn" onclick="rotateKey(\'' + k.id + '\')">Rotate</button>';
        actions += '<button class="key-action-btn danger" onclick="revokeKey(\'' + k.id + '\')">Revoke</button>';
      }
      html += '<tr>';
      html += '<td>' + (k.label || '<em>unlabeled</em>') + '</td>';
      html += '<td><span class="key-hint">' + k.key_prefix + '...' + k.key_hint + '</span></td>';
      html += '<td>' + scopes + '</td>';
      html += '<td><span class="' + statusClass + '">' + k.status + '</span></td>';
      html += '<td>' + created + '</td>';
      html += '<td>' + expires + '</td>';
      html += '<td>' + actions + '</td>';
      html += '</tr>';
    });
    html += '</table>';
    container.innerHTML = html;
  }

  window.generateKey = async function () {
    if (!_selectedIntegration) {
      alert('Please create an integration first using the "+ New" button.');
      return;
    }

    var label = document.getElementById('key-label').value.trim();
    var expiryVal = document.getElementById('key-expiry').value;
    var scopeBoxes = document.querySelectorAll('#scope-checkboxes input:checked');
    var scopes = [];
    scopeBoxes.forEach(function (cb) { scopes.push(cb.value); });
    if (!scopes.length) {
      alert('Please select at least one scope.');
      return;
    }

    var body = { scopes: scopes, label: label };
    if (expiryVal) body.expires_in_days = parseInt(expiryVal, 10);

    try {
      var resp = await fetch('/v1/integrations/' + _selectedIntegration + '/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(body),
      });
      if (resp.ok) {
        var data = await resp.json();
        var fullKey = data.key || (data.api_key && data.api_key.key) || 'Created';
        document.getElementById('new-key-value').textContent = fullKey;
        document.getElementById('new-key-box').classList.add('visible');
        document.getElementById('key-label').value = '';
        await loadKeys();
        await loadKeyHistory();
      } else {
        var err = await resp.json();
        alert(err.detail || 'Failed to create key');
      }
    } catch (e) {
      alert('Network error');
    }
  };

  window.copyNewKey = function () {
    var keyText = document.getElementById('new-key-value').textContent;
    navigator.clipboard.writeText(keyText).then(function () {
      var btn = document.querySelector('#new-key-box .btn');
      if (btn) { btn.textContent = 'Copied!'; setTimeout(function () { btn.textContent = 'Copy'; }, 1500); }
    });
  };

  window.rotateKey = async function (keyId) {
    if (!confirm('Rotate this key? The old key will have a 24-hour grace period.')) return;

    try {
      var resp = await fetch('/v1/integrations/' + _selectedIntegration + '/keys/' + keyId + '/rotate', {
        method: 'POST',
        credentials: 'same-origin',
      });
      if (resp.ok) {
        var data = await resp.json();
        var newKey = data.key || 'Rotated';
        document.getElementById('new-key-value').textContent = newKey;
        document.getElementById('new-key-box').classList.add('visible');
        await loadKeys();
        await loadKeyHistory();
      } else {
        var err = await resp.json();
        alert(err.detail || 'Rotation failed');
      }
    } catch (e) {
      alert('Network error');
    }
  };

  window.revokeKey = async function (keyId) {
    if (!confirm('Revoke this key? This action is immediate and cannot be undone.')) return;

    try {
      var resp = await fetch('/v1/integrations/' + _selectedIntegration + '/keys/' + keyId, {
        method: 'DELETE',
        credentials: 'same-origin',
      });
      if (resp.ok) {
        await loadKeys();
        await loadKeyHistory();
      } else {
        var err = await resp.json();
        alert(err.detail || 'Revocation failed');
      }
    } catch (e) {
      alert('Network error');
    }
  };

  async function loadKeyHistory() {
    var container = document.getElementById('key-audit-log');
    if (!_selectedIntegration) return;

    try {
      var resp = await fetch('/v1/integrations/' + _selectedIntegration + '/events?limit=20', { credentials: 'same-origin' });
      if (!resp.ok) throw new Error();
      var events = await resp.json();
      if (!Array.isArray(events)) events = events.events || [];
      renderAuditLog(events);
    } catch (e) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;">No events yet.</p>';
    }
  }

  function renderAuditLog(events) {
    var container = document.getElementById('key-audit-log');
    if (!events.length) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;">No events yet.</p>';
      return;
    }

    var html = '';
    events.forEach(function (ev) {
      var time = ev.created_at ? new Date(ev.created_at).toLocaleString() : '--';
      html += '<div class="audit-entry">';
      html += '  <div><span class="audit-event">' + ev.event_type + '</span> &mdash; ' + (ev.description || '') + '</div>';
      html += '  <div class="audit-time">' + time + '</div>';
      html += '</div>';
    });
    container.innerHTML = html;
  }

  // ── Instant Connect — One button, everything automatic ──────────────────

  window.instantConnect = async function () {
    var panel = document.getElementById('ic-panel');
    var btn = document.getElementById('btn-instant-connect');
    var statusEl = document.getElementById('ic-status');

    btn.textContent = 'Connecting...';
    btn.disabled = true;
    panel.classList.add('visible');
    statusEl.innerHTML = '';

    function addStatus(icon, text) {
      statusEl.innerHTML += '<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:0.85rem;"><span>' + icon + '</span><span>' + text + '</span></div>';
    }

    try {
      // Step 1: Get or create API key
      addStatus('\u23F3', 'Creating API key...');

      if (!_instantKey) {
        if (!_selectedIntegration && _integrations.length > 0) {
          _selectedIntegration = _integrations[0].id;
        }

        if (!_selectedIntegration) {
          var ns = 'user-' + _user.id.slice(0, 8);
          var intResp = await fetch('/v1/integrations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ name: 'My App', namespace: ns, owner_email: _user.email }),
          });
          if (intResp.ok) {
            var intData = await intResp.json();
            _instantKey = intData.api_key.key;
            _selectedIntegration = intData.integration.id;
            await loadIntegrations();
          }
        } else {
          var keyResp = await fetch('/v1/integrations/' + _selectedIntegration + '/keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ scopes: ['memories:read', 'memories:write'], label: 'Instant Connect' }),
          });
          if (keyResp.ok) {
            var keyData = await keyResp.json();
            _instantKey = keyData.key || (keyData.api_key && keyData.api_key.key);
            await loadKeys();
            await loadKeyHistory();
          }
        }
      }

      if (!_instantKey) {
        addStatus('\u274C', 'Failed to create API key.');
        btn.textContent = 'Instant Connect';
        btn.disabled = false;
        return;
      }

      addStatus('\u2705', 'API key created.');

      // Step 2: Auto-copy key to clipboard
      try {
        await navigator.clipboard.writeText(_instantKey);
        addStatus('\u2705', 'API key copied to clipboard.');
      } catch (e) {
        addStatus('\u26A0\uFE0F', 'Could not copy to clipboard (browser blocked it).');
      }

      // Step 3: Auto-link extension if detected
      if (window.__CLS_EXTENSION_INSTALLED__ || document.querySelector('[data-cls-extension]')) {
        window.postMessage({ type: 'CLS_LINK_ACCOUNT', apiKey: _instantKey }, '*');
        addStatus('\u2705', 'Chrome extension auto-linked.');
      }

      // Step 4: Auto-download .env AND quickstart.py
      addStatus('\u23F3', 'Downloading project files...');
      try {
        // Download .env
        var envContent = '# CLS++ Configuration\n# Generated ' + new Date().toISOString() + '\n\nCLS_API_KEY=' + _instantKey + '\nCLS_API_URL=' + API_URL + '\n';
        _downloadFile('.env', envContent);

        // Download cls_quickstart.py
        var pyContent = '#!/usr/bin/env python3\n'
          + '"""CLS++ Quickstart — auto-generated by Instant Connect."""\n\n'
          + 'from clsplusplus import CLS\n\n'
          + 'client = CLS(api_key="' + _instantKey + '")\n\n'
          + '# Store a memory\n'
          + 'client.memories.encode(content="User prefers dark mode")\n'
          + 'print("Memory stored.")\n\n'
          + '# Retrieve memories\n'
          + 'results = client.memories.retrieve(query="user preferences")\n'
          + 'for item in results.items:\n'
          + '    print(f"  {item.text} (confidence: {item.confidence})")\n\n'
          + 'print("\\nCLS++ is working! Your memories persist across every AI model.")\n';
        _downloadFile('cls_quickstart.py', pyContent);

        addStatus('\u2705', 'Downloaded .env and cls_quickstart.py to your Downloads folder.');
      } catch (e) {
        addStatus('\u26A0\uFE0F', 'Could not auto-download files.');
      }

      // Step 6: Show code snippets (reference)
      _buildSnippets(_instantKey);
      document.getElementById('ic-snippets').style.display = 'block';
      _icShowTab('python');

      // Step 7: Inline prompt — try storing a memory, right in the flow
      statusEl.innerHTML += '<div style="margin-top:12px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:8px;">'
        + '<div style="font-size:0.85rem;margin-bottom:8px;">Now try it \u2014 type anything to remember:</div>'
        + '<div style="display:flex;gap:8px;">'
        + '  <input type="text" id="ic-prompt" placeholder="e.g. I prefer Python and dark mode" maxlength="500" style="flex:1;padding:8px 12px;background:var(--bg-elevated);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;font-family:inherit;outline:none;">'
        + '  <button class="btn btn-primary btn-sm" id="ic-try-btn" onclick="icTryMemory()">Remember</button>'
        + '</div>'
        + '<div id="ic-try-result"></div>'
        + '</div>';
      var promptEl = document.getElementById('ic-prompt');
      promptEl.focus();
      promptEl.addEventListener('keydown', function (e) { if (e.key === 'Enter') icTryMemory(); });

      // Done
      btn.textContent = 'Connected';
      document.getElementById('ic-done').style.display = 'block';

    } catch (e) {
      addStatus('\u274C', 'Network error. Please try again.');
      btn.textContent = 'Instant Connect';
      btn.disabled = false;
    }

    // Always show done if we have a key, even if some steps had issues
    if (_instantKey && btn.textContent === 'Connected') {
      document.getElementById('ic-done').style.display = 'block';
    }
  };

  var _snippets = {};
  function _buildSnippets(key) {
    _snippets.python = 'from clsplusplus import CLS\n\nclient = CLS(api_key="' + key + '")\n\n# Store a memory\nclient.memories.encode(content="User prefers dark mode")\n\n# Retrieve memories\nresults = client.memories.retrieve(query="user preferences")\nfor item in results.items:\n    print(item.text, item.confidence)';
    _snippets.javascript = 'const res = await fetch("' + API_URL + '/v1/memories/encode", {\n  method: "POST",\n  headers: {\n    "Content-Type": "application/json",\n    "Authorization": "Bearer ' + key + '"\n  },\n  body: JSON.stringify({ text: "User prefers dark mode", namespace: "default" })\n});\n\nconst read = await fetch("' + API_URL + '/v1/memories/retrieve", {\n  method: "POST",\n  headers: {\n    "Content-Type": "application/json",\n    "Authorization": "Bearer ' + key + '"\n  },\n  body: JSON.stringify({ query: "user preferences", namespace: "default" })\n});';
    _snippets.curl = '# Store a memory\ncurl -X POST ' + API_URL + '/v1/memories/encode \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer ' + key + '" \\\n  -d \'{"text": "User prefers dark mode", "namespace": "default"}\'\n\n# Retrieve memories\ncurl -X POST ' + API_URL + '/v1/memories/retrieve \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer ' + key + '" \\\n  -d \'{"query": "user preferences", "namespace": "default"}\'';
  }

  function _icShowTab(lang) {
    var code = document.getElementById('ic-snippet-code');
    if (code) code.textContent = _snippets[lang] || '';
    document.querySelectorAll('.ic-tab').forEach(function (t) {
      t.classList.toggle('active', t.getAttribute('data-lang') === lang);
    });
  }

  window.icShowSnippet = function (lang, tabEl) {
    if (_instantKey && !_snippets.python) _buildSnippets(_instantKey);
    _icShowTab(lang);
  };

  window.icTryMemory = async function () {
    var input = document.getElementById('ic-prompt');
    var text = input.value.trim();
    if (!text) { input.focus(); return; }

    var btn = document.getElementById('ic-try-btn');
    var resultEl = document.getElementById('ic-try-result');
    btn.disabled = true;
    btn.textContent = 'Storing...';
    resultEl.innerHTML = '<p style="font-size:0.85rem;color:var(--text-muted);">\u23F3 Writing memory...</p>';

    var testNs = 'user-' + _user.id.slice(0, 8);

    try {
      // Write
      var writeResp = await fetch(API_URL + '/v1/memory/write', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + _instantKey },
        body: JSON.stringify({ text: text, namespace: testNs, source: 'instant-connect' }),
      });

      if (!writeResp.ok) {
        resultEl.innerHTML = '<p style="font-size:0.85rem;color:#ef4444;">\u274C Write failed (' + writeResp.status + ')</p>';
        btn.disabled = false;
        btn.textContent = 'Remember';
        return;
      }

      resultEl.innerHTML = '<p style="font-size:0.85rem;color:var(--text-muted);">\u2705 Stored. \u23F3 Reading it back...</p>';

      // Read back
      var readResp = await fetch(API_URL + '/v1/memory/read', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + _instantKey },
        body: JSON.stringify({ query: text, namespace: testNs, limit: 1 }),
      });

      if (readResp.ok) {
        var readData = await readResp.json();
        var items = readData.items || [];
        if (items.length > 0) {
          var mem = items[0];
          resultEl.innerHTML = '<div style="padding:12px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.15);border-radius:8px;margin-top:8px;">'
            + '<p style="margin:0 0 6px;font-weight:600;color:var(--success);font-size:0.9rem;">\u2705 Memory stored and retrieved!</p>'
            + '<p style="margin:0;font-size:0.85rem;"><strong>Text:</strong> ' + escapeHtml(mem.text) + '</p>'
            + '<p style="margin:4px 0 0;font-size:0.8rem;color:var(--text-muted);">Confidence: ' + (mem.confidence || 0).toFixed(2) + ' &bull; Store: ' + (mem.store_level || 'L1') + '</p>'
            + '<a href="/memory.html" style="display:inline-block;margin-top:10px;color:var(--accent-light);text-decoration:none;font-size:0.85rem;font-weight:600;">See all your memories in Memory Viewer \u2192</a>'
            + '</div>';
        } else {
          resultEl.innerHTML = '<p style="font-size:0.85rem;color:var(--success);">\u2705 Stored! Memory is being indexed. <a href="/memory.html" style="color:var(--accent-light);">Check Memory Viewer \u2192</a></p>';
        }
      } else {
        resultEl.innerHTML = '<p style="font-size:0.85rem;color:var(--success);">\u2705 Stored! <a href="/memory.html" style="color:var(--accent-light);">Check Memory Viewer \u2192</a></p>';
      }

      // Reset for another try
      input.value = '';
      btn.disabled = false;
      btn.textContent = 'Remember';

    } catch (e) {
      resultEl.innerHTML = '<p style="font-size:0.85rem;color:#ef4444;">\u274C Network error. Try again.</p>';
      btn.disabled = false;
      btn.textContent = 'Remember';
    }
  };

  function _downloadFile(filename, content) {
    var blob = new Blob([content], { type: 'text/plain' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ── Billing ────────────────────────────────────────────────────────────────

  var TIER_PRICES = { free: '$0', pro: '$9', business: '$29', enterprise: '$149' };

  async function loadBilling() {
    try {
      var resp = await fetch('/v1/user/usage', { credentials: 'same-origin' });
      if (resp.ok) {
        var usage = await resp.json();
        var tier = usage.tier || _user.tier;
        setText('billing-plan', tier.charAt(0).toUpperCase() + tier.slice(1));
        setText('billing-price', (TIER_PRICES[tier] || '$0') + '/month');
        setText('billing-period', usage.period || '--');
      }
    } catch (e) { /* ignore */ }

    if (_user.tier !== 'free') {
      var btn = document.getElementById('btn-manage-billing');
      if (btn) btn.style.display = 'inline-block';
    }
  }

  function updatePlanCards(currentTier) {
    var tiers = ['free', 'pro', 'business', 'enterprise'];
    var currentIdx = tiers.indexOf(currentTier);

    document.querySelectorAll('.plan-card').forEach(function (card) {
      var cardTier = card.getAttribute('data-tier');
      var cardIdx = tiers.indexOf(cardTier);
      var btn = card.querySelector('.plan-btn');

      if (cardIdx === currentIdx) {
        card.classList.add('current');
        if (btn) { btn.disabled = true; btn.textContent = 'Current Plan'; btn.className = 'btn btn-outline btn-sm plan-btn'; }
      } else if (cardIdx < currentIdx) {
        if (btn) { btn.disabled = true; btn.textContent = cardTier.charAt(0).toUpperCase() + cardTier.slice(1); }
      }
    });
  }

  window.startCheckout = async function (tier) {
    var btn = document.querySelector('.plan-btn[data-tier="' + tier + '"]');
    if (btn) { btn.disabled = true; btn.textContent = 'Loading...'; }

    try {
      var resp = await fetch('/v1/billing/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ tier: tier }),
      });
      if (resp.ok) {
        var data = await resp.json();
        if (data.url) { window.location.href = data.url; return; }
      }
      var err = await resp.json().catch(function () { return {}; });
      alert(err.detail || 'Billing service unavailable. Please try again later.');
    } catch (e) {
      alert('Network error. Please try again.');
    }
    if (btn) { btn.disabled = false; btn.textContent = 'Upgrade'; }
  };

  window.openBillingPortal = async function () {
    try {
      var resp = await fetch('/v1/billing/portal', { credentials: 'same-origin' });
      if (resp.ok) {
        var data = await resp.json();
        if (data.url) { window.location.href = data.url; return; }
      }
      alert('Could not open billing portal. Please try again.');
    } catch (e) {
      alert('Network error');
    }
  };

  // ── Init ───────────────────────────────────────────────────────────────────

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadProfile);
  } else {
    loadProfile();
  }
})();
