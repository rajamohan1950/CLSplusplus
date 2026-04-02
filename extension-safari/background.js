// CLS++ Background Script — Safari
// Handles identity, memory API calls, and cross-tab state
// Safari Web Extensions use background scripts with MV3

// Polyfill: Safari uses browser.* but also supports chrome.* compat
const api = typeof browser !== 'undefined' ? browser : chrome;

// Use local server when CLS_LOCAL is set in storage, otherwise cloud
let API = 'https://clsplusplus.onrender.com';
api.storage.local.get('cls_local', (r) => {
  if (r && r.cls_local) API = 'http://localhost:8080';
});

// ── User identity ──────────────────────────────────────────────────────────
let _cachedUID = null;
async function getUID() {
  if (_cachedUID) return _cachedUID;
  return new Promise((resolve) => {
    api.storage.local.get('uid', (result) => {
      if (result && result.uid) {
        _cachedUID = result.uid;
        resolve(result.uid);
        return;
      }
      const newUID = 'u_' + Math.random().toString(36).slice(2, 14) + Date.now().toString(36);
      api.storage.local.set({ uid: newUID }, () => {
        _cachedUID = newUID;
        resolve(newUID);
      });
    });
  });
}

// ── API helpers ────────────────────────────────────────────────────────────
async function searchMemories(query, limit = 5) {
  const uid = await getUID();
  try {
    const r = await fetch(`${API}/api/search/${uid}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit }),
    });
    const d = await r.json();
    return d.memories || [];
  } catch (_) { return []; }
}

async function storeMessage(text, source, model) {
  const uid = await getUID();
  try {
    await fetch(`${API}/api/store/${uid}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, source, model }),
    });
  } catch (_) {}
}

async function getMemoryCount() {
  const uid = await getUID();
  try {
    const r = await fetch(`${API}/api/memories/${uid}?limit=1`);
    const d = await r.json();
    return d.count || 0;
  } catch (_) { return 0; }
}

// ── Message handler from content scripts ──────────────────────────────────
api.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'SEARCH_MEMORIES') {
    searchMemories(msg.query, msg.limit || 5).then(sendResponse);
    return true;
  }
  if (msg.type === 'STORE_MESSAGE') {
    storeMessage(msg.text, msg.source, msg.model).then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === 'GET_COUNT') {
    getMemoryCount().then(count => sendResponse({ count }));
    return true;
  }
  if (msg.type === 'GET_UID') {
    getUID().then(uid => sendResponse({ uid }));
    return true;
  }
});

// ── Update badge count every 10s ──────────────────────────────────────────
async function updateBadge() {
  const count = await getMemoryCount();
  // Safari uses setBadgeText on the action API
  try {
    api.action.setBadgeText({ text: count > 0 ? String(count) : '' });
    api.action.setBadgeBackgroundColor({ color: '#7c6ef0' });
  } catch (_) {
    // Badge API may not be fully supported in all Safari versions
  }
}

updateBadge();
setInterval(updateBadge, 10000);

// ── On install: open welcome page ──────────────────────────────────────────
api.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    fetch('http://localhost:8080/health').then(r => {
      if (r.ok) api.tabs.create({ url: 'http://localhost:8080/ui/memory.html' });
      else api.tabs.create({ url: 'https://clsplusplus.onrender.com/install.html' });
    }).catch(() => {
      api.tabs.create({ url: 'https://clsplusplus.onrender.com/install.html' });
    });
  }
});
