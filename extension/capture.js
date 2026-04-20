// CLS++ Capture — ISOLATED world
// Core store/prefetch path restored from v5.1.0 (proven working).
// Added: toggle sync for side panel settings.

console.log('[CLS++] capture.js loading on', location.hostname);
const seen = new Set();
const host = location.hostname;

// ── Detect which site we're on ──
function getSiteKey() {
  if (host.includes('chatgpt') || host.includes('openai')) return 'chatgpt';
  if (host.includes('claude')) return 'claude';
  if (host.includes('gemini')) return 'gemini';
  return 'unknown';
}
const siteKey = getSiteKey();

// ── Write toggle states to DOM for intercept.js (MAIN world) ──
function syncToggles() {
  chrome.storage.local.get(['cls_injection_paused', 'cls_site_' + siteKey], function (r) {
    var paused = !!r.cls_injection_paused;
    var siteDisabled = r['cls_site_' + siteKey] === false;
    var shouldPause = paused || siteDisabled;
    document.body.setAttribute('data-cls-paused', shouldPause ? 'true' : 'false');
  });
}

syncToggles();
chrome.storage.onChanged.addListener(function (changes) {
  if (changes.cls_injection_paused || changes['cls_site_' + siteKey]) {
    syncToggles();
  }
});

// ── Watch outbox on document.body — intercept.js (MAIN) writes user messages here ──
// PROVEN PATH from v5.1.0 — do not add async calls or refreshMemories here
let lastOutboxTs = 0;
function checkOutbox() {
  try {
    const raw = document.body.getAttribute('data-cls-outbox');
    if (!raw) return;
    const data = JSON.parse(raw);
    if (!data || data.ts <= lastOutboxTs) return;
    lastOutboxTs = data.ts;
    const text = (data.t || '').trim();
    if (text.length < 4) return;
    if (text.includes('For context, here are some things')) return;
    const key = text.slice(0, 100);
    if (seen.has(key)) return;
    seen.add(key);
    console.log('[CLS++] Storing from outbox:', text.slice(0, 80));
    chrome.runtime.sendMessage({ type: 'STORE', text: text.slice(0, 2000) });
  } catch (_) {}
}

// Poll outbox every 500ms (lightweight — just reads one body attribute)
setInterval(checkOutbox, 500);

// ── Watch for injection events from intercept.js ──
function checkInjected() {
  try {
    var val = document.body.getAttribute('data-cls-injected');
    if (val) {
      document.body.removeAttribute('data-cls-injected');
      chrome.runtime.sendMessage({ type: 'INCREMENT_INJECTED' });
    }
  } catch (_) {}
}
setInterval(checkInjected, 500);

// ── DOM Mailbox: prefetch memories and write to hidden element ──
// PROVEN PATH from v5.1.0 — simple /v1/memory/list, always works
async function refreshMemories() {
  try {
    const resp = await new Promise(resolve => {
      chrome.runtime.sendMessage({ type: 'FETCH', limit: 10 }, r => resolve(r || { facts: [] }));
    });
    const facts = (resp.facts || []).filter(f =>
      f.length > 8 && f.length < 250 &&
      !f.startsWith('[Schema:') && !f.startsWith('[MEMORY') &&
      !f.includes('VERIFIED USER') && !f.includes('END MEMORY') &&
      !f.endsWith('?')
    );
    console.log('[CLS++] Prefetched facts:', facts.length, 'of', (resp.facts || []).length);

    // Write to hidden DOM element — readable by MAIN world
    let el = document.getElementById('__cls_mem');
    if (!el) {
      el = document.createElement('div');
      el.id = '__cls_mem';
      el.style.display = 'none';
      document.documentElement.appendChild(el);
    }
    el.setAttribute('data-facts', JSON.stringify(facts));
    el.setAttribute('data-ts', Date.now().toString());
  } catch (e) {
    console.log('[CLS++] Memory prefetch error:', e.message);
  }
}

// Refresh on load and every 30 seconds
refreshMemories();
setInterval(refreshMemories, 30000);

console.log('[CLS++] Capture ready on', host, '(site:', siteKey + ')');
