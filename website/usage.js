/**
 * CLS++ Usage Page Logic
 * Displays usage stats and history chart (moved from dashboard).
 */
(function () {
  'use strict';

  var usageChart = null;

  async function loadUsage() {
    if (typeof CLSAuth !== 'undefined') {
      await CLSAuth.requireAuth();
    }

    var user = (typeof CLSAuth !== 'undefined') ? await CLSAuth.getUser() : null;
    if (!user) return;
    if (window.CLSAnalytics) CLSAnalytics.track('usage_page_viewed', { tier: user.tier });

    // Render sidebar
    if (typeof renderSidebar === 'function') {
      await renderSidebar('usage');
    }

    // Tier badge
    var badge = document.getElementById('tier-badge');
    if (badge) {
      badge.textContent = user.tier.charAt(0).toUpperCase() + user.tier.slice(1);
      badge.className = 'tier-badge tier-' + user.tier;
    }

    // Fetch usage
    try {
      var resp = await fetch('/v1/user/usage', { credentials: 'same-origin' });
      if (resp.ok) {
        var usage = await resp.json();
        renderUsageCards(usage);
      }
    } catch (e) { /* unavailable */ }

    // Fetch history
    try {
      var histResp = await fetch('/v1/user/usage/history', { credentials: 'same-origin' });
      if (histResp.ok) {
        var history = await histResp.json();
        renderChart(history);
      }
    } catch (e) { /* unavailable */ }
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function renderUsageCards(usage) {
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
    var rateLimit = usage.rate_limit || 0;
    setText('rate-limit', rateLimit.toLocaleString());

    // Progress bar
    var bar = document.getElementById('ops-bar');
    if (bar && limit > 0) {
      var pct = Math.min((ops / limit) * 100, 100);
      bar.style.width = pct + '%';
      bar.className = 'usage-progress-fill' + (pct > 90 ? ' danger' : pct > 70 ? ' warning' : '');
    } else if (bar && limit === -1) {
      bar.style.width = '30%';
    }
  }

  function renderChart(history) {
    var canvas = document.getElementById('usage-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    var labels = history.map(function (h) { return h.period; });
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

  // Init
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadUsage);
  } else {
    loadUsage();
  }
})();
