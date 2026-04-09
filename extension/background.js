// CLS++ Background Service Worker
// Handles identity, memory API calls, and cross-tab state

// Use local server when CLS_LOCAL is set in storage, otherwise cloud
let API = 'https://clsplusplus.onrender.com';
try {
  chrome.storage.local.get(['cls_local', 'cls_api_url'], (r) => {
    if (!r) return;
    if (r.cls_api_url) API = r.cls_api_url;
    else if (r.cls_local) API = 'http://localhost:8181';
  });
} catch (e) { /* storage not ready yet */ }

// ── User identity ──────────────────────────────────────────────────────────
// If linked to account: use user-{id[:8]} namespace
// If anonymous: use random u_ UID per browser
let _cachedUID = null;
let _cachedApiKey = null;

async function getUID() {
  if (_cachedUID) return _cachedUID;

  // Check for linked account first
  const { cls_api_key, cls_user } = await chrome.storage.local.get(['cls_api_key', 'cls_user']);
  if (cls_api_key && cls_user && cls_user.id) {
    _cachedApiKey = cls_api_key;
    _cachedUID = 'user-' + cls_user.id.slice(0, 8);
    return _cachedUID;
  }

  // Fallback: anonymous UID
  const { uid } = await chrome.storage.local.get('uid');
  if (uid) { _cachedUID = uid; return uid; }
  const newUID = 'u_' + Math.random().toString(36).slice(2, 14) + Date.now().toString(36);
  await chrome.storage.local.set({ uid: newUID });
  _cachedUID = newUID;
  return newUID;
}

async function getAuthHeaders() {
  if (!_cachedApiKey) {
    const { cls_api_key } = await chrome.storage.local.get('cls_api_key');
    _cachedApiKey = cls_api_key || null;
  }
  const headers = { 'Content-Type': 'application/json', 'X-Client': 'extension' };
  if (_cachedApiKey) {
    headers['Authorization'] = `Bearer ${_cachedApiKey}`;
  }
  return headers;
}

// ── Telemetry ─────────────────────────────────────────────────────────────
async function logTelemetry(event, data = {}) {
  const uid = await getUID();
  try {
    await fetch(`${API}/api/telemetry`, {
      method: 'POST',
      headers: await getAuthHeaders(),
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
      headers: await getAuthHeaders(),
      body: JSON.stringify({ query, limit }),
    });
    const d = await r.json();
    return d.memories || [];
  } catch (_) { return []; }
}

// Session tracking for TRG — one session_id per browser tab per site
const _tabSessions = {};
function getSessionId(model, tabId) {
  const key = `${model}-${tabId || 'default'}`;
  if (!_tabSessions[key]) {
    _tabSessions[key] = `ext-${model}-${Date.now().toString(36)}`;
  }
  return _tabSessions[key];
}

let _seqCounter = 0;

async function storeMessage(text, source, model, tabId) {
  const uid = await getUID();
  const headers = await getAuthHeaders();

  // 1. Store via local API (existing extension path — backward compatible)
  try {
    await fetch(`${API}/api/store/${uid}`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ text, source, model }),
    });
  } catch (_) {}

  // 2. Also feed into TRG + PhaseMemoryEngine via authenticated path
  //    This bridges ChatGPT/Gemini/Claude browser sessions into cross-LLM recall
  if (_cachedApiKey) {
    const sessionId = getSessionId(model, tabId);
    _seqCounter++;
    try {
      await fetch(`${API}/v1/prompts/ingest`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          session_id: sessionId,
          llm_provider: model || 'unknown',
          llm_model: model || 'unknown',
          client_type: 'extension',
          entries: [{
            role: source === 'user' ? 'user' : 'assistant',
            content: text.slice(0, 2000),
            sequence_num: _seqCounter,
          }],
        }),
      });
    } catch (_) {}
  }
}

async function getMemoryCount() {
  const uid = await getUID();
  try {
    const r = await fetch(`${API}/api/memories/${uid}?limit=1`, {
      headers: await getAuthHeaders(),
    });
    const d = await r.json();
    return d.count || 0;
  } catch (_) { return 0; }
}

// ── Message handler from content scripts ──────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'SEARCH_MEMORIES') {
    searchMemories(msg.query, msg.limit || 5).then(sendResponse);
    return true;
  }
  if (msg.type === 'STORE_MESSAGE') {
    const tabId = sender && sender.tab ? sender.tab.id : null;
    storeMessage(msg.text, msg.source, msg.model, tabId).then(() => sendResponse({ ok: true }));
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
  // Account link/unlink — clear cached identity so it refreshes
  if (msg.type === 'ACCOUNT_LINKED' || msg.type === 'ACCOUNT_UNLINKED') {
    _cachedUID = null;
    _cachedApiKey = null;
    sendResponse({ ok: true });
    updateBadge();
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
    fetch('http://localhost:8181/health').then(r => {
      if (r.ok) chrome.tabs.create({ url: 'http://localhost:8181/memory.html' });
      else chrome.tabs.create({ url: 'https://clsplusplus.onrender.com/install.html' });
    }).catch(() => {
      chrome.tabs.create({ url: 'https://clsplusplus.onrender.com/install.html' });
    });
  }
  chrome.alarms.create('daily_ping', { periodInMinutes: 1440 });
});

// ── Daily ping for DAU tracking ───────────────────────────────────────────
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'daily_ping') {
    logTelemetry('ping');
  }
});

// ── Allow index.html to detect extension is present ────────────────────────
chrome.runtime.onMessageExternal.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'PING') sendResponse({ ok: true, version: '3.0.0' });
});
