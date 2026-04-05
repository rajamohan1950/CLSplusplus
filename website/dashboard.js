/**
 * CLS++ User Dashboard
 * Fetches usage data and renders tier info, progress bars, and chart.
 */
(function () {
  'use strict';

  var usageChart = null;

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

    // Populate account section
    setText('acct-email', user.email);
    setText('acct-name', user.name || '--');
    setText('acct-tier', user.tier.charAt(0).toUpperCase() + user.tier.slice(1));
    setText('acct-since', user.created_at ? new Date(user.created_at).toLocaleDateString() : '--');

    // Tier badge
    var badge = document.getElementById('tier-badge');
    if (badge) {
      badge.textContent = user.tier.charAt(0).toUpperCase() + user.tier.slice(1);
      badge.className = 'tier-badge tier-' + user.tier;
    }

    // Upgrade buttons
    updateUpgradeButtons(user.tier);

    // Fetch usage
    try {
      var resp = await fetch('/v1/user/usage', { credentials: 'same-origin' });
      if (resp.ok) {
        var usage = await resp.json();
        renderUsage(usage);
      }
    } catch (e) { /* usage unavailable */ }

    // Fetch history
    try {
      var histResp = await fetch('/v1/user/usage/history', { credentials: 'same-origin' });
      if (histResp.ok) {
        var history = await histResp.json();
        renderChart(history);
      }
    } catch (e) { /* history unavailable */ }

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

    // Billing & Limits
    var prices = { free: '$0/month', pro: '$9/month', business: '$29/month', enterprise: '$149/month' };
    setText('billing-period', usage.period || '--');
    setText('billing-price', prices[usage.tier] || '--');
    var storageLimit = usage.storage_limit || 0;
    setText('storage-limit', storageLimit === -1 ? 'Unlimited' : storageLimit.toLocaleString());
    var nsLimit = usage.namespaces_limit || 0;
    setText('ns-limit', nsLimit === -1 ? 'Unlimited' : nsLimit.toLocaleString());

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

  function renderChart(history) {
    var canvas = document.getElementById('usage-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    var labels = history.map(function (h) { return h.period; });
    var opsData = history.map(function (h) { return h.operations; });
    var writesData = history.map(function (h) { return h.writes; });
    var readsData = history.map(function (h) { return h.reads; });

    if (usageChart) usageChart.destroy();

    usageChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Writes',
            data: writesData,
            backgroundColor: 'rgba(99, 102, 241, 0.7)',
            borderRadius: 4,
          },
          {
            label: 'Reads',
            data: readsData,
            backgroundColor: 'rgba(129, 140, 248, 0.4)',
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8b8b9a' } },
        },
        scales: {
          x: { ticks: { color: '#8b8b9a' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#8b8b9a' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
        },
      },
    });
  }

  function updateUpgradeButtons(currentTier) {
    var tiers = ['free', 'pro', 'business', 'enterprise'];
    var currentIdx = tiers.indexOf(currentTier);
    tiers.forEach(function(t, i) {
      var btn = document.getElementById('btn-' + t);
      if (!btn) return;
      if (i <= currentIdx) {
        btn.disabled = true;
        btn.textContent = (i === currentIdx) ? 'Current Plan' : t.charAt(0).toUpperCase() + t.slice(1);
      }
    });
    if (currentTier === 'enterprise') {
      var section = document.getElementById('upgrade-section');
      if (section) section.style.display = 'none';
    }
  }

  window.upgradeTier = async function (tier) {
    var btn = document.getElementById('btn-' + tier);
    if (btn) { btn.disabled = true; btn.textContent = 'Upgrading...'; }

    try {
      var resp = await fetch('/v1/user/upgrade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ tier: tier }),
      });

      if (resp.ok) {
        window.location.reload();
      } else {
        var data = await resp.json();
        alert(data.message || data.detail || 'Upgrade failed');
        if (btn) { btn.disabled = false; }
      }
    } catch (e) {
      alert('Network error. Please try again.');
      if (btn) { btn.disabled = false; }
    }
  };

  // ── Integrations ───────────────────────────────────────────────────────────

  function renderIntegrations(data) {
    var list = document.getElementById('integrations-list');
    var integrations = data.integrations || data || [];
    if (!Array.isArray(integrations)) integrations = [];
    if (!integrations.length) {
      list.innerHTML = '<p style="color:var(--text-muted);">Create your first integration to get an API key.</p>';
      return;
    }
    var html = '<table style="width:100%;border-collapse:collapse;font-size:0.9rem;">';
    html += '<tr style="border-bottom:1px solid var(--border);"><th style="text-align:left;padding:8px 0;color:var(--text-muted);font-size:0.8rem;">App Name</th><th style="text-align:left;padding:8px 0;color:var(--text-muted);font-size:0.8rem;">Status</th><th style="text-align:left;padding:8px 0;color:var(--text-muted);font-size:0.8rem;">Keys</th><th style="text-align:left;padding:8px 0;color:var(--text-muted);font-size:0.8rem;">Created</th></tr>';
    integrations.forEach(function(int) {
      var created = int.created_at ? new Date(int.created_at).toLocaleDateString() : '--';
      html += '<tr style="border-bottom:1px solid var(--border);">';
      html += '<td style="padding:10px 0;">' + (int.name || '--') + '</td>';
      html += '<td style="padding:10px 0;"><span style="color:#22c55e;">' + (int.status || 'active') + '</span></td>';
      html += '<td style="padding:10px 0;">' + (int.key_count || 0) + ' key(s)</td>';
      html += '<td style="padding:10px 0;">' + created + '</td>';
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
