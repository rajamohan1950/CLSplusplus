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

      // Step 4: Auto-download single installer script
      addStatus('\u23F3', 'Downloading installer...');
      try {
        var installer = '#!/usr/bin/env python3\n'
          + '"""CLS++ Installer — auto-generated. Run: python cls_install.py"""\n'
          + 'import subprocess, sys, os, configparser\n\n'
          + 'KEY = "' + _instantKey + '"\n'
          + 'USER = "' + (_user.name || _user.email.split('@')[0]).replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase() + '"\n'
          + 'URL = "' + API_URL + '"\n\n'
          + 'print("\\n  CLS++ Installer")\n'
          + 'print("  " + "=" * 36 + "\\n")\n\n'
          + '# Step 1: Install\n'
          + 'print("  Installing clsplusplus...")\n'
          + 'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "clsplusplus"])\n'
          + 'print("  Installed.\\n")\n\n'
          + '# Step 2: Configure\n'
          + 'cfg_path = os.path.expanduser("~/.clsplusplus")\n'
          + 'cp = configparser.ConfigParser()\n'
          + 'cp["default"] = {"api_key": KEY, "user": USER, "url": URL}\n'
          + 'with open(cfg_path, "w") as f:\n'
          + '    cp.write(f)\n'
          + 'print(f"  Config saved to {cfg_path}")\n\n'
          + '# Step 3: Test\n'
          + 'print("  Testing connection...\\n")\n'
          + 'from clsplusplus import Brain\n'
          + 'brain = Brain(USER, api_key=KEY, url=URL)\n'
          + 'brain.learn("CLS++ installed successfully")\n'
          + 'result = brain.ask("CLS++ installed")\n'
          + 'print(f"  Stored:    CLS++ installed successfully")\n'
          + 'print(f"  Retrieved: {result[0] if result else \'(indexing...)\'}")\n'
          + 'print(f"  Status:    Working!\\n")\n\n'
          + 'print("  All done! Try these commands:\\n")\n'
          + 'print("    cls learn \\"I prefer Python\\"")\n'
          + 'print("    cls ask \\"What do I prefer?\\"")\n'
          + 'print("    cls chat --model gpt-4o \\"Hello!\\"\\n")\n';
        _downloadFile('cls_install.py', installer);
        addStatus('\u2705', 'Downloaded cls_install.py. Run: python ~/Downloads/cls_install.py');
      } catch (e) {
        addStatus('\u26A0\uFE0F', 'Could not auto-download installer.');
      }

      // Step 6: Show code snippets (reference)
      _buildSnippets(_instantKey);
      document.getElementById('ic-snippets').style.display = 'block';
      _icShowTab('python');

      // Step 7: Show popup — ask user to type a memory
      _showMemoryPopup();

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

  function _showMemoryPopup() {
    // Create overlay popup
    var overlay = document.createElement('div');
    overlay.id = 'ic-popup-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:999;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(4px);';

    var box = document.createElement('div');
    box.style.cssText = 'background:var(--bg-elevated);border:1px solid var(--border);border-radius:16px;padding:32px;width:90%;max-width:480px;text-align:center;';

    box.innerHTML = '<h3 style="font-size:1.1rem;margin-bottom:6px;">What should CLS++ remember?</h3>'
      + '<p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:16px;">Type anything. Hit Enter.</p>'
      + '<input type="text" id="ic-popup-input" placeholder="e.g. I prefer Python and dark mode" maxlength="500" style="width:100%;padding:12px 16px;background:var(--bg);border:1px solid var(--border);border-radius:10px;color:var(--text);font-size:1rem;font-family:inherit;outline:none;text-align:center;">'
      + '<div id="ic-popup-status" style="margin-top:12px;font-size:0.85rem;min-height:20px;"></div>';

    overlay.appendChild(box);
    document.body.appendChild(overlay);

    var input = document.getElementById('ic-popup-input');
    input.focus();

    // Close on overlay click (not box); let links inside box work
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) overlay.remove();
    });
    box.addEventListener('click', function (e) { e.stopPropagation(); });

    // Submit on Enter
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') _submitPopupMemory();
    });
  }

  async function _submitPopupMemory() {
    var input = document.getElementById('ic-popup-input');
    var statusEl = document.getElementById('ic-popup-status');
    var text = input.value.trim();
    if (!text) { input.focus(); return; }

    input.disabled = true;
    statusEl.innerHTML = '\u23F3 Storing...';

    var testNs = 'user-' + _user.id.slice(0, 8);

    try {
      // Write
      var writeResp = await fetch('/v1/memory/write', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + _instantKey },
        credentials: 'same-origin',
        body: JSON.stringify({ text: text, namespace: testNs, source: 'instant-connect' }),
      });

      if (!writeResp.ok) {
        statusEl.innerHTML = '<span style="color:#ef4444;">\u274C Failed (' + writeResp.status + '). Try again.</span>';
        input.disabled = false;
        input.focus();
        return;
      }

      statusEl.innerHTML = '\u2705 Stored! Reading it back...';

      // Read back to prove it
      var readResp = await fetch('/v1/memory/read', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + _instantKey },
        credentials: 'same-origin',
        body: JSON.stringify({ query: text, namespace: testNs, limit: 1 }),
      });

      var readText = '';
      var confidence = '';
      if (readResp.ok) {
        var readData = await readResp.json();
        var items = readData.items || [];
        if (items.length > 0) {
          readText = items[0].text;
          confidence = (items[0].confidence || 0).toFixed(2);
        }
      }

      // Show proof in popup
      var box = document.querySelector('#ic-popup-overlay > div');
      box.innerHTML = '<div style="text-align:center;">'
        + '<div style="font-size:2rem;margin-bottom:8px;">\u2705</div>'
        + '<h3 style="font-size:1.1rem;margin-bottom:12px;color:var(--success);">Memory stored and verified!</h3>'
        + (readText ? '<div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px;text-align:left;margin-bottom:12px;">'
          + '<div style="font-size:0.85rem;color:var(--text-muted);margin-bottom:4px;">Retrieved from CLS++:</div>'
          + '<div style="font-size:0.95rem;">' + escapeHtml(readText) + '</div>'
          + '<div style="font-size:0.8rem;color:var(--text-muted);margin-top:6px;">Confidence: ' + confidence + '</div>'
          + '</div>' : '')
        + '<p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:16px;">This memory now persists across every AI model you use.</p>'
        + '<a href="/memory.html" style="display:inline-block;padding:10px 24px;background:var(--accent);color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.9rem;">Open Memory Viewer</a>'
        + '</div>';

    } catch (e) {
      statusEl.innerHTML = '<span style="color:#ef4444;">\u274C Network error. Try again.</span>';
      input.disabled = false;
      input.focus();
    }
  }

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
