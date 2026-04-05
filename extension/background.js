// CLS++ Background Service Worker
// Handles identity, memory API calls, and cross-tab state

// Use local server when CLS_LOCAL is set in storage, otherwise cloud
let API = 'https://clsplusplus.onrender.com';
chrome.storage.local.get('cls_local', (r) => {
  if (r.cls_local) API = 'http://localhost:8080';
});

// ── User identity ──────────────────────────────────────────────────────────
// Stable per-browser UID stored in chrome.storage.local (no local server).
let _cachedUID = null;
async function getUID() {
  if (_cachedUID) return _cachedUID;
  const { uid } = await chrome.storage.local.get('uid');
  if (uid) { _cachedUID = uid; return uid; }
  const newUID = 'u_' + Math.random().toString(36).slice(2, 14) + Date.now().toString(36);
  await chrome.storage.local.set({ uid: newUID });
  _cachedUID = newUID;
  return newUID;
}

// ── Telemetry ─────────────────────────────────────────────────────────────
async function logTelemetry(event, data = {}) {
  const uid = await getUID();
  try {
    await fetch(`${API}/api/telemetry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Client': 'extension' },
      body: JSON.stringify({ uid, event, site: data.site || '', data, ts: new Date().toISOString() }),
    });
  } catch (_) {}
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
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'SEARCH_MEMORIES') {
    searchMemories(msg.query, msg.limit || 5).then(sendResponse);
    return true; // async
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
  if (msg.type === 'TELEMETRY') {
    logTelemetry(msg.event, msg.data || {});
    sendResponse({ ok: true });
    return false;
  }
});

// ── Update badge count every 10s ──────────────────────────────────────────
async function updateBadge() {
  const count = await getMemoryCount();
  chrome.action.setBadgeText({ text: count > 0 ? String(count) : '' });
  chrome.action.setBadgeBackgroundColor({ color: '#7c6ef0' });
}

updateBadge();
setInterval(updateBadge, 10000);

// ── On install: open welcome page ──────────────────────────────────────────
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    logTelemetry('install');
    // Check if local server is running first, prefer local
    fetch('http://localhost:8080/health').then(r => {
      if (r.ok) chrome.tabs.create({ url: 'http://localhost:8080/ui/memory.html' });
      else chrome.tabs.create({ url: 'https://clsplusplus.onrender.com/install.html' });
    }).catch(() => {
      chrome.tabs.create({ url: 'https://clsplusplus.onrender.com/install.html' });
    });
  }
  // Set up daily ping alarm for DAU tracking
  chrome.alarms.create('daily_ping', { periodInMinutes: 1440 }); // once per day
});

// ── Daily ping for DAU tracking ───────────────────────────────────────────
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'daily_ping') {
    logTelemetry('ping');
  }
});

// ── Allow index.html to detect extension is present ────────────────────────
chrome.runtime.onMessageExternal.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'PING') sendResponse({ ok: true, version: '2.1.0' });
});
