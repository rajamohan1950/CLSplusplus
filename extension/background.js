importScripts('analytics.js');

// CLS++ Background Service Worker — v6.0.2
// Core store/fetch path restored from v5.1.0 (proven working).
// Added: side panel handlers (ACTIVITY, USAGE, SEARCH, etc.)

let API = 'https://www.clsplusplus.com';

// ── Get API key from storage ──
async function getKey() {
  const { cls_api_key } = await chrome.storage.local.get('cls_api_key');
  return cls_api_key || '';
}

// ── Store a memory ──
async function storeMemory(text) {
  if (text.length < 4) return;
  const key = await getKey();
  if (!key) {
    console.error('[CLS++] No API key linked — memory NOT stored. Click the 🧠 extension icon to link your account.');
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#f05d9a' });
    return;
  }
  try {
    console.log('[CLS++] BG: Storing:', text.slice(0, 60));
    const r = await fetch(`${API}/v1/memory/write`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
      body: JSON.stringify({ text, source: 'extension' }),
    });
    console.log('[CLS++] BG: Store response:', r.status);
  } catch (e) { console.error('[CLS++] BG: Store error:', e.message); }
}

// ── Fetch memories — returns array of text strings ──
async function fetchMemories(limit) {
  const key = await getKey();
  if (!key) return [];
  try {
    const r = await fetch(`${API}/v1/memory/list?limit=${limit || 15}`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (r.ok) {
      const d = await r.json();
      return (d.items || []).map(i => i.text || '').filter(t => t.length > 5);
    }
  } catch (_) {}
  return [];
}

// ── Fetch recent memories with full metadata (side panel) ──
async function fetchActivity(limit) {
  const key = await getKey();
  if (!key) return { items: [], total: 0 };
  try {
    const r = await fetch(`${API}/v1/memory/list?limit=${limit || 5}`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (r.ok) {
      const d = await r.json();
      return { items: d.items || [], total: d.total || 0 };
    }
  } catch (_) {}
  return { items: [], total: 0 };
}

// ── Fetch usage stats (side panel) ──
async function fetchUsage() {
  const key = await getKey();
  if (!key) return null;
  try {
    const r = await fetch(`${API}/v1/usage`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (r.ok) return await r.json();
  } catch (_) {}
  return null;
}

// ── Search memories (side panel search bar) ──
async function searchMemories(query, limit) {
  const key = await getKey();
  if (!key) return { items: [], total: 0 };
  try {
    const r = await fetch(`${API}/v1/memories/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
      body: JSON.stringify({ query, limit: limit || 10 }),
    });
    if (r.ok) {
      const d = await r.json();
      return { items: d.items || [], total: d.total || 0 };
    }
  } catch (_) {}
  return { items: [], total: 0 };
}

// ── Daily counters (side panel activity tab) ──
async function incrementDailyCounter(key) {
  const today = new Date().toISOString().slice(0, 10);
  const data = await chrome.storage.local.get(['cls_daily_date', key]);
  if (data.cls_daily_date !== today) {
    await chrome.storage.local.set({
      cls_daily_date: today,
      cls_daily_stored: key === 'cls_daily_stored' ? 1 : 0,
      cls_daily_injected: key === 'cls_daily_injected' ? 1 : 0,
    });
  } else {
    var update = {};
    update[key] = (data[key] || 0) + 1;
    await chrome.storage.local.set(update);
  }
}

async function getDailyCounters() {
  const today = new Date().toISOString().slice(0, 10);
  const data = await chrome.storage.local.get(['cls_daily_date', 'cls_daily_stored', 'cls_daily_injected']);
  if (data.cls_daily_date !== today) return { stored: 0, injected: 0 };
  return { stored: data.cls_daily_stored || 0, injected: data.cls_daily_injected || 0 };
}

// ── Badge ──
async function updateBadge() {
  const key = await getKey();
  if (!key) { chrome.action.setBadgeText({ text: '' }); return; }
  try {
    const r = await fetch(`${API}/v1/memory/list?limit=1`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (r.ok) {
      const d = await r.json();
      const count = d.total || 0;
      var text = count > 999 ? Math.floor(count / 1000) + 'k' : String(count);
      chrome.action.setBadgeText({ text });
      chrome.action.setBadgeBackgroundColor({ color: '#7c6ef0' });
      chrome.action.setBadgeTextColor({ color: '#ffffff' });
    }
  } catch (_) {}
}

updateBadge();
setInterval(updateBadge, 60000);

// ── Message handler ──
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // CORE — store memory (proven path from v5.1.0)
  if (msg.type === 'STORE') {
    storeMemory(msg.text);
    incrementDailyCounter('cls_daily_stored');
    chrome.storage.local.get('cls_user', function(r) {
      var uid = (r.cls_user && r.cls_user.id) ? String(r.cls_user.id) : 'anonymous';
      CLSExtAnalytics.captureFromBackground('memory_stored', { text_length: msg.text.length }, uid);
    });
    sendResponse({ ok: true });
    return false;
  }

  // CORE — fetch memories for injection (proven path from v5.1.0)
  if (msg.type === 'FETCH') {
    fetchMemories(msg.limit || 15).then(facts => {
      chrome.storage.local.get('cls_user', function(r) {
        var uid = (r.cls_user && r.cls_user.id) ? String(r.cls_user.id) : 'anonymous';
        CLSExtAnalytics.captureFromBackground('memories_fetched', { count: facts.length }, uid);
      });
      sendResponse({ facts });
    });
    return true;
  }

  // Side panel — recent items with metadata
  if (msg.type === 'ACTIVITY') {
    fetchActivity(msg.limit || 5).then(data => sendResponse(data));
    return true;
  }

  // Side panel — usage stats
  if (msg.type === 'USAGE') {
    fetchUsage().then(data => sendResponse(data));
    return true;
  }

  // Side panel — search
  if (msg.type === 'SEARCH') {
    searchMemories(msg.query, msg.limit || 10).then(data => {
      chrome.storage.local.get('cls_user', function(r) {
        var uid = (r.cls_user && r.cls_user.id) ? String(r.cls_user.id) : 'anonymous';
        CLSExtAnalytics.captureFromBackground('memories_searched', { query_length: (msg.query || '').length, result_count: data.total || 0 }, uid);
      });
      sendResponse(data);
    });
    return true;
  }

  // Side panel — daily counters
  if (msg.type === 'DAILY_COUNTERS') {
    getDailyCounters().then(data => sendResponse(data));
    return true;
  }

  // Side panel — injection counter
  if (msg.type === 'INCREMENT_INJECTED') {
    incrementDailyCounter('cls_daily_injected');
    sendResponse({ ok: true });
    return false;
  }

  // Popup — verify API key
  if (msg.type === 'VERIFY_KEY') {
    (async () => {
      try {
        const r = await fetch(`${API}/v1/auth/me`, {
          headers: { 'Authorization': `Bearer ${msg.key}` },
        });
        if (r.ok) {
          var userData = await r.json();
          CLSExtAnalytics.captureFromBackground('api_key_verified', { success: true }, userData.id ? String(userData.id) : 'anonymous');
          sendResponse(userData);
          setTimeout(updateBadge, 500);
        } else {
          CLSExtAnalytics.captureFromBackground('api_key_verified', { success: false }, 'anonymous');
          sendResponse(null);
        }
      } catch (_) {
        CLSExtAnalytics.captureFromBackground('api_key_verified', { success: false }, 'anonymous');
        sendResponse(null);
      }
    })();
    return true;
  }

  // Popup — open side panel
  if (msg.type === 'OPEN_PANEL') {
    chrome.sidePanel.open({ windowId: sender.tab ? sender.tab.windowId : undefined }).catch(() => {
      chrome.windows.getCurrent(win => {
        chrome.sidePanel.open({ windowId: win.id }).catch(() => {});
      });
    });
    sendResponse({ ok: true });
    return false;
  }
});
