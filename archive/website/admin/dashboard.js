/**
 * CLS++ Admin Dashboard
 * Fetches all admin endpoints and renders KPIs, charts, categorized metrics,
 * extension analytics, and per-user table.
 */
(function () {
  'use strict';

  var chartInstances = {};

  async function loadAdmin() {
    if (typeof CLSAuth !== 'undefined') {
      await CLSAuth.requireAdmin();
    }

    var [summary, signups, revenue, operations, users, extension, storage] = await Promise.all([
      fetchJSON('/admin/metrics/summary'),
      fetchJSON('/admin/metrics/signups'),
      fetchJSON('/admin/metrics/revenue'),
      fetchJSON('/admin/metrics/operations'),
      fetchJSON('/admin/metrics/users'),
      fetchJSON('/admin/metrics/extension'),
      fetchJSON('/admin/metrics/storage'),
    ]);

    if (summary) renderKPIs(summary);
    if (signups) renderSignupsChart(signups.signups || []);
    if (revenue) renderRevenueCharts(revenue);
    if (operations) renderCategorizedMetrics(operations);
    if (users) renderUserTable(users.users || []);
    if (extension) renderExtensionAnalytics(extension);
    if (storage) renderStorageMetrics(storage);
    if (summary && operations) renderProfitForecast(summary, operations);
  }

  async function fetchJSON(url) {
    try {
      var resp = await fetch(url, { credentials: 'same-origin' });
      if (resp.ok) return await resp.json();
    } catch (e) {}
    return null;
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // ── KPIs ──────────────────────────────────────────────────────────────────

  function renderKPIs(s) {
    setText('kpi-users', s.total_users.toLocaleString());
    var tc = s.tier_counts || {};
    setText('kpi-users-sub', (tc.free||0)+' Free / '+(tc.pro||0)+' Pro / '+(tc.business||0)+' Business / '+(tc.enterprise||0)+' Enterprise');
    setText('kpi-revenue', '$' + s.monthly_revenue.toLocaleString());
    var revParts = [];
    if (tc.pro) revParts.push(tc.pro + ' Pro ($' + (tc.pro*9) + ')');
    if (tc.business) revParts.push(tc.business + ' Business ($' + (tc.business*29) + ')');
    if (tc.enterprise) revParts.push(tc.enterprise + ' Enterprise ($' + (tc.enterprise*149) + ')');
    setText('kpi-revenue-sub', revParts.join(' + ') || 'No paying users');
    setText('kpi-cost', '$' + s.monthly_cost.toFixed(2));
    setText('kpi-cost-sub', 'Infrastructure this month');

    // User buckets
    setText('bucket-free-count', (tc.free || 0).toLocaleString());
    setText('bucket-free-rev', '$0 revenue');
    setText('bucket-pro-count', (tc.pro || 0).toLocaleString());
    setText('bucket-pro-rev', '$' + ((tc.pro || 0) * 9) + ' revenue');
    setText('bucket-business-count', (tc.business || 0).toLocaleString());
    setText('bucket-business-rev', '$' + ((tc.business || 0) * 29) + ' revenue');
    setText('bucket-enterprise-count', (tc.enterprise || 0).toLocaleString());
    setText('bucket-enterprise-rev', '$' + ((tc.enterprise || 0) * 149) + ' revenue');
    var marginEl = document.getElementById('kpi-margin');
    if (marginEl) {
      marginEl.textContent = s.margin_percent.toFixed(1) + '%';
      marginEl.className = 'kpi-value ' + (s.margin_percent >= 0 ? 'kpi-green' : 'kpi-red');
    }
  }

  // ── Charts ────────────────────────────────────────────────────────────────

  var chartDefaults = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8b8b9a', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { ticks: { color: '#8b8b9a' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
    },
  };

  function makeChart(canvasId, type, labels, datasets, extraOpts) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;
    if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
    var opts = JSON.parse(JSON.stringify(chartDefaults));
    if (extraOpts) { for (var k in extraOpts) opts[k] = extraOpts[k]; }
    chartInstances[canvasId] = new Chart(canvas, { type: type, data: { labels: labels, datasets: datasets }, options: opts });
  }

  function renderSignupsChart(signups) {
    makeChart('chart-signups', 'bar', signups.map(function(s){return s.date;}), [{
      label: 'Signups', data: signups.map(function(s){return s.count;}),
      backgroundColor: 'rgba(255,107,53,0.7)', borderRadius: 3,
    }]);
  }

  function renderRevenueCharts(rev) {
    makeChart('chart-mrr', 'bar', ['Current MRR'], [{ label: 'MRR', data: [rev.mrr], backgroundColor: 'rgba(34,197,94,0.7)', borderRadius: 4 }]);
    makeChart('chart-arr', 'bar', ['Current ARR'], [{ label: 'ARR', data: [rev.arr], backgroundColor: 'rgba(34,197,94,0.5)', borderRadius: 4 }]);
    var now = new Date(), fLabels = [], fData = [];
    for (var m = now.getMonth(); m < 12; m++) {
      fLabels.push(new Date(now.getFullYear(), m).toLocaleString('default', { month: 'short' }));
      fData.push(rev.mrr);
    }
    makeChart('chart-forecast', 'line', fLabels, [{
      label: 'Projected MRR', data: fData,
      borderColor: 'rgba(255,107,53,0.8)', backgroundColor: 'rgba(255,107,53,0.1)',
      fill: true, tension: 0.3, pointRadius: 4,
    }], { plugins: { legend: { display: true, labels: { color: '#8b8b9a' } } } });
  }

  // ── Extension Analytics ───────────────────────────────────────────────────

  function renderExtensionAnalytics(ext) {
    setText('ext-installs-today', ext.installs_today.toLocaleString());
    setText('ext-installs-month', ext.installs_this_month.toLocaleString());
    setText('ext-dau', ext.dau.toLocaleString());
    setText('ext-wau', ext.wau.toLocaleString());
    setText('ext-mau', ext.mau.toLocaleString());

    var s = ext.settings || {};
    setText('ext-autoinject', (s.autoInject_on || 0).toLocaleString());
    setText('ext-localmode', (s.cls_local_on || 0).toLocaleString());
    setText('ext-cloudmode', (s.cls_local_off || 0).toLocaleString());

    // Site usage chart
    var sites = ext.site_usage || {};
    var siteLabels = Object.keys(sites);
    var siteData = siteLabels.map(function(k) { return sites[k]; });
    var siteColors = { chatgpt: '#19c37d', claude: '#cc785c', gemini: '#4285f4', copilot: '#2564cf' };
    var colors = siteLabels.map(function(k) { return siteColors[k] || '#8b8b9a'; });

    makeChart('chart-sites', 'bar', siteLabels, [{
      label: 'Injections', data: siteData, backgroundColor: colors, borderRadius: 4,
    }], { indexAxis: 'y' });

    // Messages chart
    var msgs = ext.messages_captured || {};
    var msgLabels = Object.keys(msgs);
    var msgData = msgLabels.map(function(k) { return msgs[k]; });
    var msgColors = msgLabels.map(function(k) { return siteColors[k] || '#8b8b9a'; });

    makeChart('chart-messages', 'bar', msgLabels, [{
      label: 'Messages', data: msgData, backgroundColor: msgColors, borderRadius: 4,
    }], { indexAxis: 'y' });
  }

  // ── Categorized Metrics ───────────────────────────────────────────────────

  var CATEGORIES = {
    'metrics-api': ['write', 'encode', 'read', 'retrieve', 'search', 'knowledge', 'delete', 'context_injection', 'adjudication', 'consolidation', 'prewarm', 'chat_message'],
    'metrics-compute': ['embedding', 'hippocampal_replay', 'l2_promotion'],
    'metrics-external': ['llm_token_in', 'llm_token_out', 'ext_llm_proxy_openai', 'ext_llm_proxy_anthropic'],
    'metrics-ext-backend': ['ext_write', 'ext_search', 'ext_context_injection', 'ext_memory_fetch', 'ext_delete', 'ext_download', 'ext_ws_connect',
      'ext_write_chatgpt', 'ext_write_claude', 'ext_write_gemini'],
    'metrics-platform': ['total_api_requests', 'rate_limit_429'],
  };

  function renderCategorizedMetrics(ops) {
    var metrics = ops.metrics || {};
    var usedKeys = {};

    for (var gridId in CATEGORIES) {
      var grid = document.getElementById(gridId);
      if (!grid) continue;
      grid.innerHTML = '';
      var fields = CATEGORIES[gridId];
      var hasAny = false;
      for (var i = 0; i < fields.length; i++) {
        var f = fields[i];
        var val = metrics[f] || 0;
        grid.appendChild(createMetricItem(f, val));
        usedKeys[f] = true;
        if (val > 0) hasAny = true;
      }
      if (!hasAny && grid.children.length === 0) {
        grid.innerHTML = '<div class="metric-placeholder">No data yet.</div>';
      }
    }

    // Total cost
    setText('total-cost-value', '$' + (ops.total_cost || 0).toFixed(4));
  }

  function createMetricItem(name, value) {
    var div = document.createElement('div');
    div.className = 'metric-item';
    var displayName = name.replace(/_/g, ' ').replace(/^ext /, '');
    var displayValue = (typeof value === 'number') ? value.toLocaleString() : value;
    div.innerHTML = '<div class="metric-name">' + displayName + '</div><div class="metric-val">' + displayValue + '</div>';
    return div;
  }

  // ── Storage Metrics ────────────────────────────────────────────────────────

  // ── Profit & 30-Day Forecast ────────────────────────────────────────────

  function renderProfitForecast(summary, operations) {
    var now = new Date();
    var daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
    var dayOfMonth = now.getDate();

    var monthlyRevenue = summary.monthly_revenue || 0;
    var monthlyCost = operations.total_cost || 0;

    // Daily rates
    var dailyRevenue = monthlyRevenue / daysInMonth;
    var dailyCost = dayOfMonth > 0 ? monthlyCost / dayOfMonth : 0;
    var dailyNet = dailyRevenue - dailyCost;
    var projected30d = dailyNet * 30;

    setText('profit-daily-rev', '$' + dailyRevenue.toFixed(2));
    setText('profit-daily-cost', '$' + dailyCost.toFixed(4));

    var netEl = document.getElementById('profit-daily-net');
    if (netEl) {
      netEl.textContent = '$' + dailyNet.toFixed(2);
      netEl.className = 'kpi-value ' + (dailyNet >= 0 ? 'kpi-green' : 'kpi-red');
    }
    var p30El = document.getElementById('profit-30d');
    if (p30El) {
      p30El.textContent = '$' + projected30d.toFixed(2);
      p30El.className = 'kpi-value ' + (projected30d >= 0 ? 'kpi-green' : 'kpi-red');
    }

    // 30-day forecast chart: revenue line, cost line, net profit area
    var labels = [];
    var revData = [];
    var costData = [];
    var netData = [];
    for (var d = 1; d <= 30; d++) {
      var date = new Date(now);
      date.setDate(now.getDate() + d - dayOfMonth);
      labels.push(date.toLocaleDateString('default', { month: 'short', day: 'numeric' }));

      var cumRev = dailyRevenue * d;
      var cumCost = dailyCost * d;
      revData.push(Math.round(cumRev * 100) / 100);
      costData.push(Math.round(cumCost * 10000) / 10000);
      netData.push(Math.round((cumRev - cumCost) * 100) / 100);
    }

    makeChart('chart-profit-forecast', 'line', labels, [
      {
        label: 'Cumulative Revenue',
        data: revData,
        borderColor: 'rgba(34, 197, 94, 0.8)',
        backgroundColor: 'rgba(34, 197, 94, 0.05)',
        fill: false,
        tension: 0.1,
        pointRadius: 2,
      },
      {
        label: 'Cumulative Cost',
        data: costData,
        borderColor: 'rgba(239, 68, 68, 0.8)',
        backgroundColor: 'rgba(239, 68, 68, 0.05)',
        fill: false,
        tension: 0.1,
        pointRadius: 2,
      },
      {
        label: 'Net Profit',
        data: netData,
        borderColor: 'rgba(99, 102, 241, 0.9)',
        backgroundColor: 'rgba(99, 102, 241, 0.15)',
        fill: true,
        tension: 0.1,
        pointRadius: 2,
      },
    ], {
      plugins: { legend: { display: true, labels: { color: '#8b8b9a' } } },
    });
  }

  function renderStorageMetrics(storage) {
    var grid = document.getElementById('metrics-storage');
    if (!grid) return;
    grid.innerHTML = '';
    var fields = [
      ['l0_items', 'L0 In-Memory Items'],
      ['l0_namespaces', 'L0 Namespaces'],
      ['l1_items', 'L1 PostgreSQL Items'],
      ['l1_namespaces', 'L1 Namespaces'],
      ['loaded_namespaces', 'Loaded Namespaces'],
    ];
    fields.forEach(function(f) {
      var div = document.createElement('div');
      div.className = 'metric-item';
      div.innerHTML = '<div class="metric-name">' + f[1] + '</div><div class="metric-val">' + (storage[f[0]] || 0).toLocaleString() + '</div>';
      grid.appendChild(div);
    });
  }

  // ── User Table ────────────────────────────────────────────────────────────

  function renderUserTable(users) {
    var tbody = document.getElementById('users-tbody');
    if (!tbody) return;
    if (!users.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);">No users yet.</td></tr>';
      return;
    }
    tbody.innerHTML = '';
    users.forEach(function (u) {
      var margin = u.revenue > 0 ? ((u.revenue - u.cost) / u.revenue * 100) : 0;
      var mc = margin >= 0 ? 'margin-positive' : 'margin-negative';
      var joined = u.created_at ? new Date(u.created_at).toLocaleDateString() : '--';
      var tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.onclick = function() { showUserDetail(u.id, u.email); };
      tr.innerHTML =
        '<td>' + escHtml(u.email) + '</td>' +
        '<td>' + escHtml(u.name || '--') + '</td>' +
        '<td><span class="tier-pill ' + u.tier + '">' + u.tier + '</span></td>' +
        '<td>' + (u.operations || 0).toLocaleString() + '</td>' +
        '<td>$' + u.cost.toFixed(4) + '</td>' +
        '<td>$' + u.revenue.toFixed(2) + '</td>' +
        '<td class="' + mc + '">' + margin.toFixed(1) + '%</td>' +
        '<td>' + joined + '</td>';
      tbody.appendChild(tr);
    });
  }

  // ── User Detail Modal ─────────────────────────────────────────────────────

  async function showUserDetail(userId, email) {
    var modal = document.getElementById('user-modal');
    if (!modal) return;
    setText('modal-title', email);
    setText('modal-tier', '--');
    setText('modal-ops', '--');
    setText('modal-cost', '--');
    setText('modal-rev', '--');
    document.getElementById('modal-metrics').innerHTML = '<div class="metric-placeholder">Loading...</div>';
    modal.style.display = 'block';

    var data = await fetchJSON('/admin/metrics/user/' + userId);
    if (!data) {
      document.getElementById('modal-metrics').innerHTML = '<div class="metric-placeholder">Failed to load.</div>';
      return;
    }

    var u = data.user || {};
    setText('modal-title', (u.name || email) + ' (' + u.email + ')');
    setText('modal-tier', (u.tier || '--').toUpperCase());
    var totalOps = 0;
    var metrics = data.metrics || {};
    for (var k in metrics) totalOps += metrics[k];
    setText('modal-ops', totalOps.toLocaleString());
    setText('modal-cost', '$' + (data.cost || 0).toFixed(4));
    setText('modal-rev', '$' + (data.revenue || 0).toFixed(2));

    var grid = document.getElementById('modal-metrics');
    grid.innerHTML = '';
    var keys = Object.keys(metrics);
    if (keys.length === 0) {
      grid.innerHTML = '<div class="metric-placeholder">No activity yet.</div>';
      return;
    }
    keys.forEach(function(k) {
      grid.appendChild(createMetricItem(k, metrics[k]));
    });
  }

  function escHtml(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadAdmin);
  } else {
    loadAdmin();
  }
})();
