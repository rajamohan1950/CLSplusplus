/**
 * CLS++ Profile Page Logic
 * Handles user info, API key CRUD, and billing.
 */
(function () {
  'use strict';

  var _user = null;
  var _integrations = [];
  var _selectedIntegration = null;

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

    // Show change password button only for password-based accounts
    // (Google-only accounts don't have a password)
    var pwBtn = document.getElementById('btn-change-pw');
    if (pwBtn) pwBtn.style.display = 'inline-block';

    // Preferences
    var ns = document.getElementById('pref-namespace');
    if (ns) ns.textContent = 'user-' + _user.id.slice(0, 8);

    // Load integrations for API keys
    await loadIntegrations();

    // Load billing info
    await loadBilling();

    // Update plan cards
    updatePlanCards(_user.tier);

    // Check URL params for billing result
    var params = new URLSearchParams(window.location.search);
    if (params.get('billing') === 'success') {
      showMsg('billing-msg', 'Subscription updated successfully!', 'var(--success)');
    }
  }

  function renderUserInfo() {
    if (!_user) return;

    // Avatar
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

    // Pre-fill edit form
    var nameInput = document.getElementById('edit-name');
    var emailInput = document.getElementById('edit-email');
    if (nameInput) nameInput.value = _user.name || '';
    if (emailInput) emailInput.value = _user.email || '';
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function showMsg(id, msg, color) {
    var el = document.getElementById(id);
    if (el) {
      el.textContent = msg;
      el.style.color = color || 'var(--text-muted)';
    }
  }

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
      showMsg('edit-profile-msg', 'No changes to save.', 'var(--text-muted)');
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
        showMsg('edit-profile-msg', '', '');
      } else {
        var err = await resp.json();
        showMsg('edit-profile-msg', err.detail || 'Update failed', '#ef4444');
      }
    } catch (e) {
      showMsg('edit-profile-msg', 'Network error', '#ef4444');
    }
  };

  window.changePassword = async function () {
    var current = document.getElementById('pw-current').value;
    var newPw = document.getElementById('pw-new').value;
    if (!current || !newPw) {
      showMsg('change-pw-msg', 'Both fields are required.', '#ef4444');
      return;
    }
    if (newPw.length < 8) {
      showMsg('change-pw-msg', 'Password must be at least 8 characters.', '#ef4444');
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
        showMsg('change-pw-msg', 'Password updated.', 'var(--success)');
        document.getElementById('pw-current').value = '';
        document.getElementById('pw-new').value = '';
        setTimeout(function () {
          document.getElementById('change-pw-form').classList.remove('visible');
          showMsg('change-pw-msg', '', '');
        }, 1500);
      } else {
        var err = await resp.json();
        showMsg('change-pw-msg', err.detail || 'Failed', '#ef4444');
      }
    } catch (e) {
      showMsg('change-pw-msg', 'Network error', '#ef4444');
    }
  };

  // ── API Keys ───────────────────────────────────────────────────────────────

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

    // Auto-select first
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

  async function loadKeys() {
    var container = document.getElementById('keys-table-container');
    if (!_selectedIntegration) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">Select an integration to view keys.</p>';
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
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">No API keys yet. Generate one above.</p>';
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
      alert('Please select or create an integration first.');
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

    // Show manage subscription button for paid tiers
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
        if (data.url) {
          window.location.href = data.url;
          return;
        }
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
        if (data.url) {
          window.location.href = data.url;
          return;
        }
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
