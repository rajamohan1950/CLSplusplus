// CLS++ — shared logic injected into every AI site
// This file is loaded by each site-specific content script

const MEMORY_PREFIX = '[MEMORY — VERIFIED USER FACTS]';
// Detect API endpoint: localStorage pages use localhost, AI sites check storage flag.
let CLSPP_API = 'https://www.clsplusplus.com';
if (typeof location !== 'undefined' && /^(localhost|127\.0\.0\.1)$/i.test(location.hostname)) {
  CLSPP_API = `http://${location.hostname}:${location.port || '8181'}`;
}
// Override to localhost if user toggled "Local mode" in popup
try {
  chrome.storage.local.get(['cls_local', 'cls_api_url'], (r) => {
    if (!r) return;
    if (r.cls_api_url) CLSPP_API = r.cls_api_url;
    else if (r.cls_local) CLSPP_API = 'http://localhost:8181';
  });
} catch (e) { /* extension context not available */ }

// ── Share UID with MAIN world (intercept.js) ─────────────────────────────
let _clsppUID = null;
chrome.runtime.sendMessage({ type: 'GET_UID' }, (resp) => {
  if (resp && resp.uid) {
    _clsppUID = resp.uid;
    window.dispatchEvent(new CustomEvent('__clspp_uid', { detail: resp.uid }));
  }
});

// ── Context bridge: MAIN world can't fetch localhost due to page CSP ─────
// intercept.js dispatches __clspp_context_request, we fetch and respond.
window.addEventListener('__clspp_context_request', async (e) => {
  const { id, query } = e.detail;
  let context = '';
  // Respect Auto-inject toggle
  const { autoInject, cls_api_key } = await chrome.storage.local.get(['autoInject', 'cls_api_key']);
  if (autoInject === false) {
    window.dispatchEvent(new CustomEvent('__clspp_context_response', { detail: { id, context: '' } }));
    return;
  }
  try {
    // If user has linked API key, use authenticated TRG + engine recall
    if (cls_api_key) {
      const r = await fetch(`${CLSPP_API}/v1/memory/read`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${cls_api_key}`,
        },
        body: JSON.stringify({ query: query, limit: 5 }),
      });
      if (r.ok) {
        const d = await r.json();
        const items = d.items || [];
        if (items.length > 0) {
          const lines = [
            '[MEMORY — VERIFIED USER FACTS]',
            'These are confirmed facts about this user from prior conversations across all AI models.',
            'Treat them as ground truth:',
          ];
          items.forEach(m => lines.push('- ' + (m.text || '')));
          context = lines.join('\n') + '\n';
        }
      }
    } else {
      // Fallback: local anonymous API
      const payload = { query };
      if (_clsppUID) payload.uid = _clsppUID;
      const r = await fetch(`${CLSPP_API}/api/context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (r.ok) {
        const d = await r.json();
        context = d.context || '';
      }
    }
  } catch (err) {
    // Server not running — silently fail
  }
  window.dispatchEvent(new CustomEvent('__clspp_context_response', {
    detail: { id, context }
  }));
});

// Signal to intercept.js (MAIN world) that the bridge is ready
window.dispatchEvent(new CustomEvent('__clspp_bridge_ready'));
console.log('[CLS++] context bridge ready');

// ── Forward telemetry events from MAIN world to background ───────────────
window.addEventListener('__clspp_telemetry', (e) => {
  if (e.detail && e.detail.event) {
    chrome.runtime.sendMessage({ type: 'TELEMETRY', event: e.detail.event, data: e.detail });
  }
});

// ── Query memories from background ────────────────────────────────────────
async function queryMemories(query) {
  return new Promise(resolve => {
    chrome.runtime.sendMessage({ type: 'SEARCH_MEMORIES', query, limit: 5 }, resolve);
  });
}

// ── Store a message via background ────────────────────────────────────────
function storeMessage(text, source, model) {
  if (!text || text.trim().length < 6) return;
  if (text.startsWith(MEMORY_PREFIX)) return; // don't store our own injections
  chrome.runtime.sendMessage({ type: 'STORE_MESSAGE', text: text.trim(), source, model });
  chrome.runtime.sendMessage({ type: 'TELEMETRY', event: 'message_captured', data: { site: model, source } });
}

// ── Format memory context block ────────────────────────────────────────────
function buildContext(memories) {
  if (!memories || !memories.length) return '';
  const lines = [
    '[MEMORY — VERIFIED USER FACTS]',
    'These are confirmed facts about this user from their own prior statements.',
    'Treat them as ground truth. If the user\'s current message contradicts a stored fact,',
    'gently remind them of what they previously said. Always prefer these facts over assumptions:'
  ];
  memories.forEach(m => lines.push('- ' + m.text));
  lines.push('');
  return lines.join('\n');
}

// ── Set text in a contenteditable element (works with React/ProseMirror) ──
function setEditableText(el, text) {
  el.focus();
  const sel = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(el);
  sel.removeAllRanges();
  sel.addRange(range);
  document.execCommand('insertText', false, text);
}

// ── Set text in a plain textarea ──────────────────────────────────────────
function setTextareaValue(el, text) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
  setter.call(el, text);
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

// ── Watch DOM for new messages ─────────────────────────────────────────────
function watchMessages(getMessages, siteName) {
  let seen = new Set();

  function checkNew() {
    const msgs = getMessages();
    msgs.forEach(({ role, text }) => {
      if (!text || text.length < 6) return;
      if (text.startsWith(MEMORY_PREFIX)) return;
      const key = role + ':' + text.slice(0, 80);
      if (seen.has(key)) return;
      seen.add(key);
      const source = role === 'user' ? 'user' : 'assistant';
      storeMessage(text, source, siteName);
    });
  }

  setTimeout(checkNew, 2000);
  new MutationObserver(checkNew).observe(document.body, { childList: true, subtree: true });
}
