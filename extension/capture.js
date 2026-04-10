// CLS++ Capture — ISOLATED world
// 1. Captures user messages from DOM → stores via background.js
// 2. Prefetches memories into a hidden DOM element for intercept.js to read

console.log('[CLS++] capture.js loading on', location.hostname);
const seen = new Set();
const host = location.hostname;

// ── Watch outbox on document.body — intercept.js (MAIN) writes user messages here ──
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

// ── DOM Mailbox: prefetch memories and write to hidden element ──
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

console.log('[CLS++] Capture ready on', host);
