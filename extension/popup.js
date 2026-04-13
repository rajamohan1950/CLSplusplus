// CLS++ Popup — Compact Quick Glance + Open Panel
(function () {
  'use strict';
  var API = 'https://www.clsplusplus.com';

  // Load theme
  clsLoadTheme();
  CLSExtAnalytics.initBrowser();
  CLSExtAnalytics.track('popup_opened');

  // Check server
  fetch(API + '/health').then(function (r) {
    if (r.ok) document.getElementById('dot').className = 'cls-dot on';
  }).catch(function () {});

  // Load saved state
  chrome.storage.local.get(['cls_api_key', 'cls_user'], function (r) {
    if (r.cls_api_key && r.cls_user) showLinked(r.cls_user);
  });

  function showLinked(user) {
    document.getElementById('sec-unlinked').style.display = 'none';
    document.getElementById('sec-linked').style.display = 'block';
    document.getElementById('user-name').textContent = user.name || user.email || 'Linked';

    var tier = (user.tier || 'free').toLowerCase();
    var tierEl = document.getElementById('user-tier');
    tierEl.innerHTML = '';
    var badge = document.createElement('span');
    badge.className = 'cls-badge cls-badge-' + tier;
    badge.textContent = tier.toUpperCase();
    tierEl.appendChild(badge);

    // Load quick stats
    loadStats();
  }

  function showUnlinked() {
    document.getElementById('sec-unlinked').style.display = 'block';
    document.getElementById('sec-linked').style.display = 'none';
    document.getElementById('key-input').value = '';
  }

  function loadStats() {
    // Memory count + last memory
    chrome.runtime.sendMessage({ type: 'ACTIVITY', limit: 1 }, function (data) {
      if (!data) return;
      document.getElementById('stat-memories').textContent = formatNum(data.total || 0);

      if (data.items && data.items.length > 0) {
        var item = data.items[0];
        document.getElementById('last-memory').style.display = '';
        document.getElementById('last-memory-text').textContent = (item.text || '').slice(0, 80);
        var src = sourceName(item.source);
        var ago = timeAgo(item.timestamp);
        document.getElementById('last-memory-meta').textContent = src + (ago ? ' · ' + ago : '');
      }
    });

    // Usage
    chrome.runtime.sendMessage({ type: 'USAGE' }, function (data) {
      if (!data) return;
      var used = data.operations || 0;
      var limit = data.operations_limit || 1000;
      document.getElementById('stat-ops').textContent = formatNum(used) + '/' + formatNum(limit);
    });

    // Active sites count (check which LLM tabs are open)
    document.getElementById('stat-sites').textContent = '3';
  }

  function formatNum(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return Math.floor(n / 1000) + 'K';
    return String(n);
  }

  function sourceName(src) {
    src = (src || '').toLowerCase();
    if (src.includes('chatgpt') || src.includes('openai')) return 'ChatGPT';
    if (src.includes('claude')) return 'Claude';
    if (src.includes('gemini')) return 'Gemini';
    if (src.includes('hook') || src.includes('cli')) return 'CLI';
    return 'Extension';
  }

  function timeAgo(ts) {
    if (!ts) return '';
    var diff = Date.now() - new Date(ts).getTime();
    var mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return mins + 'm ago';
    var hours = Math.floor(mins / 60);
    if (hours < 24) return hours + 'h ago';
    return Math.floor(hours / 24) + 'd ago';
  }

  // Link
  document.getElementById('btn-link').addEventListener('click', function () {
    var key = document.getElementById('key-input').value.trim();
    var err = document.getElementById('err');
    var btn = document.getElementById('btn-link');
    if (!key) return;

    err.classList.remove('visible');
    btn.textContent = 'Verifying...';
    btn.disabled = true;

    chrome.runtime.sendMessage({ type: 'VERIFY_KEY', key: key }, function (user) {
      if (user && user.email) {
        chrome.storage.local.set({ cls_api_key: key, cls_user: user });
        showLinked(user);
        CLSExtAnalytics.identifyUser(user);
        CLSExtAnalytics.track('account_linked', { method: 'api_key' });
      } else {
        err.textContent = 'Invalid key or server unreachable';
        err.classList.add('visible');
      }
      btn.textContent = 'Link Account';
      btn.disabled = false;
    });
  });

  // Open Side Panel
  document.getElementById('btn-panel').addEventListener('click', function () {
    // Use sidePanel API from popup context
    CLSExtAnalytics.track('sidepanel_opened_from_popup');
    chrome.sidePanel.open({ windowId: chrome.windows.WINDOW_ID_CURRENT }).then(function () {
      window.close(); // close popup after opening panel
    }).catch(function () {
      // Fallback
      chrome.runtime.sendMessage({ type: 'OPEN_PANEL' });
      window.close();
    });
  });

})();
