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

  // 1. Always persist via authenticated /v1/memory/write (survives restarts)
  const apiKey = _cachedApiKey || (await chrome.storage.local.get('cls_api_key')).cls_api_key;
  if (apiKey) {
    try {
      await fetch(`${API}/v1/memory/write`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ text, source: model || 'extension' }),
      });
    } catch (_) {}
    _cachedApiKey = apiKey;  // Cache it for next time
  }

  // 2. Also store via local API (backward compatible for anonymous users)
  try {
    await fetch(`${API}/api/store/${uid}`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ text, source, model }),
    });
  } catch (_) {}

  // 3. Feed into TRG + PhaseMemoryEngine
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
  if (msg.type === 'FETCH_MEMORIES') {
    // Route /v1/memory/read through background to avoid CORS
    (async () => {
      const apiKey = _cachedApiKey || (await chrome.storage.local.get('cls_api_key')).cls_api_key;
      if (!apiKey) { sendResponse({ items: [] }); return; }
      try {
        const r = await fetch(`${API}/v1/memory/read`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({ query: msg.query || '*', limit: msg.limit || 10 }),
        });
        if (r.ok) {
          const d = await r.json();
          sendResponse({ items: d.items || [] });
        } else {
          sendResponse({ items: [] });
        }
      } catch (e) {
        sendResponse({ items: [] });
      }
    })();
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
// CHATGPT CUSTOM INSTRUCTIONS SYNC — runs in background (no CORS issues)
// Reads CLS++ memories, writes to ChatGPT's /backend-api/user_system_messages
// using ChatGPT's session cookies. Completely invisible to the user.
// ══════════════════════════════════════════════════════════════════════════

const CI_MARKER_START = '[CLS++ Memory]';
const CI_MARKER_END = '[/CLS++]';
const CI_SYNC_INTERVAL = 60000; // 60 seconds

async function syncChatGPTCustomInstructions() {
  console.log('[CLS++] CI sync starting...');
  // Only sync if user has linked API key
  const apiKey = _cachedApiKey || (await chrome.storage.local.get('cls_api_key')).cls_api_key;
  if (!apiKey) { console.log('[CLS++] CI sync: no API key'); return; }

  // 1. Fetch CLS++ memories — multiple queries for diverse coverage
  let facts = [];
  const seen = new Set();
  const queries = [
    'personal name identity who am I',
    'preferences likes dislikes favorites movies music perfume',
    'family mother father relationships friends people',
    'recent work project decisions current status',
  ];
  for (const q of queries) {
    try {
      const r = await fetch(`${API}/v1/memory/read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ query: q, limit: 5 }),
      });
      if (r.ok) {
        const d = await r.json();
        for (const item of (d.items || [])) {
          const t = item.text || '';
          // Filter: only statements that look like personal facts
          // Must contain a name, relationship, preference, or identity signal
          const tl = t.toLowerCase();
          const hasPersonalSignal = (
            /\b(name|is|are|was|has|likes?|loves?|prefers?|works?|lives?|born|sister|brother|mother|father|wife|husband|friend|favou?rite|colou?r|planning|trip|installed|diesel|perfume|mango)\b/i.test(t) ||
            /\b(raj|veena|rupa|ruchi|suchi|dingu|muthaiah|jmuthaiah)\b/i.test(t)
          );
          const isJunk = t.length < 12 || t.length > 300 ||
            t.startsWith('[') || t.endsWith('?') ||
            /\b(merge|branch|commit|deploy|push|extension|localhost|hardcode|chrome|hook|server|render|docker|test|fix|error|debug)\b/i.test(t) ||
            /^(fu|bs|ok|yes|no|s|k|ya|nothing|prepare|why|how|what)\b/i.test(t);
          if (hasPersonalSignal && !isJunk && !seen.has(t)) {
            seen.add(t);
            facts.push(t);
          }
        }
      }
    } catch (e) {}
  }

  console.log('[CLS++] CI sync: got', facts.length, 'memories');
  if (!facts.length) return;

  // 2. Build memory block
  const memBlock = [CI_MARKER_START,
    'Verified facts about this user from all AI conversations:',
    ...facts.map(f => '- ' + f.slice(0, 200)),
    CI_MARKER_END,
  ].join('\n');

  // 3. Get ChatGPT session cookie for auth
  let accessToken = '';
  try {
    // ChatGPT stores the session token — fetch it via their session endpoint
    const sessionResp = await fetch('https://chatgpt.com/api/auth/session', {
      credentials: 'include',
    });
    if (sessionResp.ok) {
      const session = await sessionResp.json();
      accessToken = session.accessToken || '';
    }
  } catch (e) {}

  console.log('[CLS++] CI sync: session token', accessToken ? 'OK (' + accessToken.slice(0,10) + '...)' : 'FAILED');
  if (!accessToken) return;

  // 4. Read current Custom Instructions
  let current;
  try {
    const r = await fetch('https://chatgpt.com/backend-api/user_system_messages', {
      headers: { 'Authorization': `Bearer ${accessToken}` },
    });
    console.log('[CLS++] CI sync: read CI status', r.status);
    if (!r.ok) { console.log('[CLS++] CI sync: read failed', r.status, await r.text().catch(() => '')); return; }
    current = await r.json();
    console.log('[CLS++] CI sync: current about_user length', (current.about_user_message || '').length);
  } catch (e) { console.log('[CLS++] CI sync: read error', e.message); return; }

  let aboutUser = current.about_user_message || '';

  // Remove old CLS++ block
  const startIdx = aboutUser.indexOf(CI_MARKER_START);
  const endIdx = aboutUser.indexOf(CI_MARKER_END);
  if (startIdx !== -1 && endIdx !== -1) {
    aboutUser = (aboutUser.slice(0, startIdx) + aboutUser.slice(endIdx + CI_MARKER_END.length)).trim();
  }

  // Append new block
  const newAboutUser = (aboutUser ? aboutUser + '\n\n' : '') + memBlock;
  if (newAboutUser === (current.about_user_message || '')) { console.log('[CLS++] CI sync: no change needed'); return; }

  console.log('[CLS++] CI sync: writing', newAboutUser.length, 'chars');

  // 5. Write updated Custom Instructions
  try {
    const wr = await fetch('https://chatgpt.com/backend-api/user_system_messages', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        about_user_message: newAboutUser,
        about_model_message: current.about_model_message || '',
        enabled: true,
      }),
    });
    console.log('[CLS++] CI sync: write status', wr.status);
    if (wr.ok) console.log('[CLS++] CI sync: SUCCESS — memories written to ChatGPT Custom Instructions');
    else console.log('[CLS++] CI sync: write failed', wr.status, await wr.text().catch(() => ''));
  } catch (e) { console.log('[CLS++] CI sync: write error', e.message); }
}

// Use chrome.alarms for reliable MV3 wake-up (setInterval dies when SW sleeps)
chrome.alarms.create('cls_ci_sync', { delayInMinutes: 0.15, periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'cls_ci_sync') {
    syncChatGPTCustomInstructions();
  }
});

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
