// CLS++ Side Panel — Full Hub
(function () {
  'use strict';

  var memoryOffset = 0;
  var memoryLimit = 15;
  var allMemories = [];
  var currentFilter = 'all';
  var searchTimeout = null;

  // ── Init ──
  clsLoadTheme();
  CLSExtAnalytics.initBrowser();
  CLSExtAnalytics.track('sidepanel_opened');
  checkServerAndLoad();

  // ── Server check + load state ──
  function checkServerAndLoad() {
    fetch('https://www.clsplusplus.com/health').then(function (r) {
      if (r.ok) {
        document.getElementById('dot').className = 'cls-dot on';
      }
    }).catch(function () {});

    chrome.storage.local.get(['cls_api_key', 'cls_user'], function (r) {
      if (r.cls_api_key && r.cls_user) {
        showLinked(r.cls_user);
      } else {
        showUnlinked();
      }
    });
  }

  function showLinked(user) {
    document.getElementById('sec-unlinked').style.display = 'none';
    var linked = document.getElementById('sec-linked');
    linked.style.display = 'flex';
    document.getElementById('user-name').textContent = user.name || user.email || 'Linked';

    var tier = (user.tier || 'free').toLowerCase();
    var tierEl = document.getElementById('user-tier');
    tierEl.innerHTML = '';
    var badge = document.createElement('span');
    badge.className = 'cls-badge cls-badge-' + tier;
    badge.textContent = tier.toUpperCase();
    tierEl.appendChild(badge);

    // Account info
    document.getElementById('account-name').textContent = user.name || '—';
    document.getElementById('account-email').textContent = user.email || '—';
    document.getElementById('account-tier').textContent = (user.tier || 'free').toUpperCase();

    // Load data
    loadMemories();
    loadUsage();
    loadDailyCounters();
    loadToggles();
    renderThemes();
  }

  function showUnlinked() {
    document.getElementById('sec-unlinked').style.display = 'block';
    document.getElementById('sec-linked').style.display = 'none';
  }

  // ── Link / Unlink ──
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
      } else {
        err.textContent = 'Invalid key or server unreachable';
        err.classList.add('visible');
      }
      btn.textContent = 'Link Account';
      btn.disabled = false;
    });
  });

  document.getElementById('btn-unlink').addEventListener('click', function () {
    CLSExtAnalytics.track('account_unlinked');
    chrome.storage.local.remove(['cls_api_key', 'cls_user']);
    showUnlinked();
  });

  // ── Tabs ──
  document.querySelectorAll('.cls-tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      var target = tab.dataset.tab;
      document.querySelectorAll('.cls-tab').forEach(function (t) { t.classList.remove('active'); });
      document.querySelectorAll('.cls-tab-content').forEach(function (p) { p.classList.remove('active'); });
      tab.classList.add('active');
      document.querySelector('[data-panel="' + target + '"]').classList.add('active');
      CLSExtAnalytics.track('tab_switched', { tab: target });

      // Refresh data when switching tabs
      if (target === 'activity') {
        loadUsage();
        loadDailyCounters();
      }
    });
  });

  // ── Memories ──
  function loadMemories() {
    memoryOffset = 0;
    allMemories = [];
    chrome.runtime.sendMessage({ type: 'ACTIVITY', limit: memoryLimit }, function (data) {
      if (!data) return;
      allMemories = data.items || [];
      renderMemories();
      var count = data.total || 0;
      document.getElementById('memory-count').innerHTML = '<strong>' + count + '</strong> memories';

      document.getElementById('load-more').style.display = allMemories.length < (data.total || 0) ? '' : 'none';
      document.getElementById('memory-empty').style.display = allMemories.length === 0 ? '' : 'none';
    });
  }

  function renderMemories() {
    var list = document.getElementById('memory-list');
    list.innerHTML = '';
    var filtered = filterMemories(allMemories);
    filtered.forEach(function (item) {
      list.appendChild(createMemoryEl(item));
    });
    document.getElementById('memory-empty').style.display = filtered.length === 0 ? '' : 'none';
  }

  function filterMemories(items) {
    if (currentFilter === 'all') return items;
    return items.filter(function (item) {
      var src = (item.source || '').toLowerCase();
      if (currentFilter === 'chatgpt') return src.includes('chatgpt') || src.includes('openai');
      if (currentFilter === 'claude') return src.includes('claude');
      if (currentFilter === 'gemini') return src.includes('gemini');
      if (currentFilter === 'cli') return src.includes('cli') || src.includes('hook') || src.includes('claude-code');
      return true;
    });
  }

  function createMemoryEl(item) {
    var el = document.createElement('div');
    el.className = 'cls-memory-item';

    var text = document.createElement('div');
    text.className = 'cls-memory-text';
    text.textContent = item.text || '';

    var meta = document.createElement('div');
    meta.className = 'cls-memory-meta';

    var source = document.createElement('span');
    source.className = 'cls-memory-source';
    source.textContent = sourceIcon(item.source) + ' ' + sourceName(item.source);

    var time = document.createElement('span');
    time.textContent = timeAgo(item.timestamp);

    var conf = document.createElement('div');
    conf.className = 'cls-memory-confidence';
    var confFill = document.createElement('div');
    confFill.className = 'cls-memory-confidence-fill';
    confFill.style.width = Math.round((item.confidence || 0.5) * 100) + '%';
    conf.appendChild(confFill);

    meta.appendChild(source);
    meta.appendChild(time);
    meta.appendChild(conf);
    el.appendChild(text);
    el.appendChild(meta);

    el.addEventListener('click', function () { el.classList.toggle('expanded'); });
    return el;
  }

  function sourceIcon(src) {
    src = (src || '').toLowerCase();
    if (src.includes('chatgpt') || src.includes('openai')) return '💬';
    if (src.includes('claude')) return '🟠';
    if (src.includes('gemini')) return '✨';
    return '⌨️';
  }

  function sourceName(src) {
    src = (src || '').toLowerCase();
    if (src.includes('chatgpt') || src.includes('openai')) return 'ChatGPT';
    if (src.includes('claude')) return 'Claude';
    if (src.includes('gemini')) return 'Gemini';
    if (src.includes('hook') || src.includes('cli')) return 'CLI';
    return src || 'Extension';
  }

  function timeAgo(ts) {
    if (!ts) return '';
    var diff = Date.now() - new Date(ts).getTime();
    var mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return mins + 'm ago';
    var hours = Math.floor(mins / 60);
    if (hours < 24) return hours + 'h ago';
    var days = Math.floor(hours / 24);
    return days + 'd ago';
  }

  // ── Search ──
  document.getElementById('search-input').addEventListener('input', function (e) {
    var query = e.target.value.trim();
    clearTimeout(searchTimeout);
    if (query.length < 2) {
      loadMemories();
      return;
    }
    searchTimeout = setTimeout(function () {
      chrome.runtime.sendMessage({ type: 'SEARCH', query: query, limit: 20 }, function (data) {
        if (!data) return;
        allMemories = data.items || [];
        renderMemories();
        document.getElementById('memory-count').innerHTML = '<strong>' + (data.total || 0) + '</strong> results';
        document.getElementById('load-more').style.display = 'none';
        CLSExtAnalytics.track('memories_searched', { query_length: query.length, result_count: data.total || 0 });
      });
    }, 300);
  });

  // ── Filters ──
  document.getElementById('filter-chips').addEventListener('click', function (e) {
    var chip = e.target.closest('.cls-chip');
    if (!chip) return;
    currentFilter = chip.dataset.source;
    document.querySelectorAll('.cls-chip').forEach(function (c) { c.classList.remove('active'); });
    chip.classList.add('active');
    renderMemories();
    CLSExtAnalytics.track('filter_changed', { filter: currentFilter });
  });

  // ── Load more ──
  document.getElementById('btn-load-more').addEventListener('click', function () {
    memoryOffset += memoryLimit;
    chrome.runtime.sendMessage({ type: 'ACTIVITY', limit: memoryLimit + memoryOffset }, function (data) {
      if (!data) return;
      allMemories = data.items || [];
      renderMemories();
      document.getElementById('load-more').style.display = allMemories.length < (data.total || 0) ? '' : 'none';
    });
  });

  // ── Usage ──
  function loadUsage() {
    chrome.runtime.sendMessage({ type: 'USAGE' }, function (data) {
      if (!data) return;
      var used = data.operations || 0;
      var limit = data.operations_limit || 1000;
      var pct = Math.min(used / limit, 1);

      // Ring animation
      var ring = document.getElementById('usage-ring');
      var circumference = 2 * Math.PI * 16; // r=16
      ring.style.strokeDasharray = circumference;
      ring.style.strokeDashoffset = circumference * (1 - pct);

      // Color based on usage
      if (pct > 0.9) ring.style.stroke = 'var(--cls-danger)';
      else if (pct > 0.7) ring.style.stroke = 'var(--cls-warning)';
      else ring.style.stroke = 'var(--cls-accent)';

      document.getElementById('usage-text').textContent = formatNumber(used) + ' / ' + formatNumber(limit);
      document.getElementById('usage-label').textContent = 'operations this month';

      // Upgrade card
      var tier = (data.tier || 'free').toLowerCase();
      if (tier === 'free' && pct > 0.5) {
        document.getElementById('upgrade-card').style.display = '';
      }
    });
  }

  function formatNumber(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'K';
    return String(n);
  }

  // ── Daily Counters ──
  function loadDailyCounters() {
    chrome.runtime.sendMessage({ type: 'DAILY_COUNTERS' }, function (data) {
      if (!data) return;
      document.getElementById('daily-stored').textContent = data.stored || 0;
      document.getElementById('daily-injected').textContent = data.injected || 0;
    });
  }

  // ── Toggles ──
  function loadToggles() {
    chrome.storage.local.get([
      'cls_injection_paused',
      'cls_site_chatgpt', 'cls_site_claude', 'cls_site_gemini'
    ], function (r) {
      document.getElementById('toggle-injection').checked = !r.cls_injection_paused;
      document.getElementById('toggle-chatgpt').checked = r.cls_site_chatgpt !== false;
      document.getElementById('toggle-claude').checked = r.cls_site_claude !== false;
      document.getElementById('toggle-gemini').checked = r.cls_site_gemini !== false;
    });
  }

  document.getElementById('toggle-injection').addEventListener('change', function (e) {
    chrome.storage.local.set({ cls_injection_paused: !e.target.checked });
    CLSExtAnalytics.track('toggle_changed', { toggle: 'injection', enabled: e.target.checked });
  });

  ['chatgpt', 'claude', 'gemini'].forEach(function (site) {
    document.getElementById('toggle-' + site).addEventListener('change', function (e) {
      var obj = {};
      obj['cls_site_' + site] = e.target.checked;
      chrome.storage.local.set(obj);
      CLSExtAnalytics.track('toggle_changed', { toggle: 'site_' + site, enabled: e.target.checked });
    });
  });

  // ── Themes ──
  function renderThemes() {
    chrome.storage.local.get('cls_theme', function (r) {
      clsRenderThemeGrid(document.getElementById('theme-grid'), r.cls_theme || 'midnight');
    });
  }

})();
