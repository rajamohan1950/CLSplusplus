// CLS++ Background Service Worker — Minimal
// Handles: API calls (no CORS), memory store/fetch, badge

let API = 'https://www.clsplusplus.com';

// ── Get API key from storage ──
async function getKey() {
  const { cls_api_key } = await chrome.storage.local.get('cls_api_key');
  return cls_api_key || '';
}

// ── Store a memory ──
async function storeMemory(text) {
  const key = await getKey();
  if (!key || text.length < 4) return;
  try {
    await fetch(`${API}/v1/memory/write`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
      body: JSON.stringify({ text, source: 'extension' }),
    });
  } catch (_) {}
}

// ── Fetch memories (returns array of text strings) ──
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

// ── Message handler ──
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'STORE') {
    storeMemory(msg.text);
    sendResponse({ ok: true });
    return false;
  }
  if (msg.type === 'FETCH') {
    fetchMemories(msg.limit || 15).then(facts => sendResponse({ facts }));
    return true; // async
  }
  if (msg.type === 'VERIFY_KEY') {
    (async () => {
      try {
        const r = await fetch(`${API}/v1/auth/me`, {
          headers: { 'Authorization': `Bearer ${msg.key}` },
        });
        if (r.ok) sendResponse(await r.json());
        else sendResponse(null);
      } catch (_) { sendResponse(null); }
    })();
    return true;
  }
});
