// CLS++ Background Service Worker
// Handles identity, memory API calls, cross-tab state, and demo seeding

// Use local server when CLS_LOCAL is set in storage, otherwise cloud
let API = 'https://www.clsplusplus.com';  // Production default
try {
  chrome.storage.local.get(['cls_local', 'cls_api_url'], (r) => {
    if (!r) return;
    if (r.cls_api_url) API = r.cls_api_url;
    else if (r.cls_local) API = 'http://localhost:8181';
  });
} catch (e) { /* storage not ready yet */ }

// ── User identity ──────────────────────────────────────────────────────────
let _cachedUID = null;
let _cachedApiKey = null;

async function getUID() {
  if (_cachedUID) return _cachedUID;
  const { cls_api_key, cls_user } = await chrome.storage.local.get(['cls_api_key', 'cls_user']);
  if (cls_api_key && cls_user && cls_user.id) {
    _cachedApiKey = cls_api_key;
    _cachedUID = 'user-' + cls_user.id.slice(0, 8);
    return _cachedUID;
  }
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

// Session tracking for TRG
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

  // 1. Store via local API
  try {
    await fetch(`${API}/api/store/${uid}`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ text, source, model }),
    });
  } catch (_) {}

  // 2. Feed into TRG + PhaseMemoryEngine
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

  // 3. Flash badge to confirm memory saved
  flashBadge();
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

// ── Badge flash — visible confirmation of "Memory saved" ────────────────
let _flashTimer = null;
function flashBadge() {
  chrome.action.setBadgeText({ text: '✓' });
  chrome.action.setBadgeBackgroundColor({ color: '#5de0c5' });
  clearTimeout(_flashTimer);
  _flashTimer = setTimeout(() => updateBadge(), 1500);
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
  if (msg.type === 'ACCOUNT_LINKED' || msg.type === 'ACCOUNT_UNLINKED') {
    _cachedUID = null;
    _cachedApiKey = null;
    sendResponse({ ok: true });
    updateBadge();
    return false;
  }
});

// ── Badge count ──────────────────────────────────────────────────────────
async function updateBadge() {
  const count = await getMemoryCount();
  chrome.action.setBadgeText({ text: count > 0 ? String(count) : '' });
  chrome.action.setBadgeBackgroundColor({ color: '#7c6ef0' });
}

updateBadge();
setInterval(updateBadge, 10000);

// ══════════════════════════════════════════════════════════════════════════
// FIRST INSTALL: Seed demo memories so reviewer sees content immediately
// ══════════════════════════════════════════════════════════════════════════

const DEMO_MEMORIES = [
  // Universal truths — global knowledge baseline
  'The sun rises in the east and sets in the west.',
  'Water boils at 100 degrees Celsius at sea level.',
  'The Earth orbits the Sun once every 365.25 days.',
  'Light travels at approximately 300,000 kilometers per second.',
  'There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.',
  // Demo user preferences — shows how personal memory works
  'This user prefers dark mode in all applications.',
  'This user likes concise answers without unnecessary filler.',
  'This user is evaluating CLS++ for cross-model memory.',
];

async function seedDemoMemories() {
  const uid = await getUID();
  const headers = await getAuthHeaders();

  for (const text of DEMO_MEMORIES) {
    try {
      await fetch(`${API}/api/store/${uid}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ text, source: 'user', model: 'demo' }),
      });
    } catch (_) {}
  }
  // Mark as seeded so we don't re-seed
  await chrome.storage.local.set({ cls_demo_seeded: true });
  updateBadge();
}

chrome.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === 'install') {
    logTelemetry('install');

    // Seed demo memories after short delay (server may need cold start)
    setTimeout(async () => {
      const { cls_demo_seeded } = await chrome.storage.local.get('cls_demo_seeded');
      if (!cls_demo_seeded) {
        await seedDemoMemories();
      }
    }, 3000);

    // Open memory viewer
    try {
      const r = await fetch(`${API}/health`);
      if (r.ok) chrome.tabs.create({ url: `${API}/memory.html` });
      else chrome.tabs.create({ url: 'https://www.clsplusplus.com/integrate.html' });
    } catch (_) {
      chrome.tabs.create({ url: 'https://www.clsplusplus.com/integrate.html' });
    }
  }

  // Also seed on update if never seeded
  if (details.reason === 'update') {
    const { cls_demo_seeded } = await chrome.storage.local.get('cls_demo_seeded');
    if (!cls_demo_seeded) {
      setTimeout(seedDemoMemories, 3000);
    }
  }

  chrome.alarms.create('daily_ping', { periodInMinutes: 1440 });
});

// ── Daily ping ───────────────────────────────────────────────────────────
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'daily_ping') logTelemetry('ping');
});

// ── External extension detection ─────────────────────────────────────────
chrome.runtime.onMessageExternal.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'PING') sendResponse({ ok: true, version: '3.0.0' });
});
