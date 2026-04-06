/**
 * CLS++ User Dashboard (Simplified)
 * Overview stats + integrations. Detailed usage/billing/account moved to profile & usage pages.
 */
(function () {
  'use strict';

  async function loadDashboard() {
    // Require login
    if (typeof CLSAuth !== 'undefined') {
      await CLSAuth.requireAuth();
    }

    var user = (typeof CLSAuth !== 'undefined') ? await CLSAuth.getUser() : null;
    if (!user) return;

    // Admin users go straight to admin dashboard
    if (user.is_admin) {
      window.location.href = '/admin/dashboard.html';
      return;
    }

    // Render sidebar
    if (typeof renderSidebar === 'function') {
      await renderSidebar('dashboard');
    }

    // Tier badge
    var badge = document.getElementById('tier-badge');
    if (badge) {
      badge.textContent = user.tier.charAt(0).toUpperCase() + user.tier.slice(1);
      badge.className = 'tier-badge tier-' + user.tier;
    }

    // Fetch usage summary
    try {
      var resp = await fetch('/v1/user/usage', { credentials: 'same-origin' });
      if (resp.ok) {
        var usage = await resp.json();
        renderUsage(usage);
      }
    } catch (e) { /* usage unavailable */ }

    // Fetch integrations
    try {
      var intResp = await fetch('/v1/user/integrations', { credentials: 'same-origin' });
      if (intResp.ok) {
        var intData = await intResp.json();
        renderIntegrations(intData);
      } else {
        document.getElementById('integrations-list').innerHTML = '<p style="color:var(--text-muted);">Create your first integration to get an API key.</p>';
      }
    } catch (e) {
      document.getElementById('integrations-list').innerHTML = '<p style="color:var(--text-muted);">Create your first integration to get an API key.</p>';
    }
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function renderUsage(usage) {
    var ops = usage.operations || 0;
    var limit = usage.operations_limit || 0;
    var limitStr = limit === -1 ? 'Unlimited' : limit.toLocaleString();

    setText('ops-value', ops.toLocaleString());
    setText('ops-sub', 'of ' + limitStr + ' limit');
    setText('writes-value', (usage.writes || 0).toLocaleString());
    setText('reads-value', (usage.reads || 0).toLocaleString());

    // Progress bar
    var bar = document.getElementById('ops-bar');
    if (bar && limit > 0) {
      var pct = Math.min((ops / limit) * 100, 100);
      bar.style.width = pct + '%';
      bar.className = 'progress-fill' + (pct > 90 ? ' danger' : pct > 70 ? ' warning' : '');
    } else if (bar && limit === -1) {
      bar.style.width = '30%';
    }
  }

  function renderIntegrations(data) {
    var list = document.getElementById('integrations-list');
    var integrations = data.integrations || data || [];
    if (!Array.isArray(integrations)) integrations = [];
    if (!integrations.length) {
      list.innerHTML = '<p style="color:var(--text-muted);">Create your first integration to get an API key.</p>';
      return;
    }
    var html = '<table class="int-table">';
    html += '<tr><th>App Name</th><th>Status</th><th>Keys</th><th>Created</th></tr>';
    integrations.forEach(function (int) {
      var created = int.created_at ? new Date(int.created_at).toLocaleDateString() : '--';
      html += '<tr>';
      html += '<td>' + (int.name || '--') + '</td>';
      html += '<td><span style="color:var(--success);">' + (int.status || 'active') + '</span></td>';
      html += '<td>' + (int.key_count || 0) + ' key(s)</td>';
      html += '<td>' + created + '</td>';
      html += '</tr>';
    });
    html += '</table>';
    list.innerHTML = html;
  }

  window.createDashIntegration = async function () {
    var nameInput = document.getElementById('new-int-name');
    var name = nameInput.value.trim();
    if (!name) { nameInput.focus(); return; }

    try {
      var user = await CLSAuth.getUser();
      var ns = user ? 'user-' + user.id.slice(0, 8) : 'default';
      var resp = await fetch('/v1/integrations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ name: name, namespace: ns, owner_email: user ? user.email : '' }),
      });
      if (resp.ok) {
        var data = await resp.json();
        var key = data.api_key && data.api_key.key ? data.api_key.key : 'Created';
        document.getElementById('new-key-value').textContent = key;
        document.getElementById('new-key-result').style.display = 'block';
        nameInput.value = '';
        // Reload integrations list
        var intResp = await fetch('/v1/user/integrations', { credentials: 'same-origin' });
        if (intResp.ok) renderIntegrations(await intResp.json());
      } else {
        var err = await resp.json();
        alert(err.message || err.detail || 'Failed to create integration');
      }
    } catch (e) {
      alert('Network error');
    }
  };

  // Load on ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadDashboard);
  } else {
    loadDashboard();
  }
})();
