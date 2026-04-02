// CLS++ — shared logic injected into every AI site (Safari)
// This file is loaded by each site-specific content script

// Polyfill: Safari uses browser.* but also supports chrome.* compat
const _api = typeof browser !== 'undefined' ? browser : chrome;

const MEMORY_PREFIX = '[MEMORY — VERIFIED USER FACTS]';
// Detect API endpoint: localStorage pages use localhost, AI sites check storage flag.
let CLSPP_API = 'https://clsplusplus.onrender.com';
if (typeof location !== 'undefined' && /^(localhost|127\.0\.0\.1)$/i.test(location.hostname)) {
  CLSPP_API = `http://${location.hostname}:${location.port || '8080'}`;
}
// Override to localhost if user toggled "Local mode" in popup
_api.storage.local.get('cls_local', (r) => {
  if (r && r.cls_local) CLSPP_API = 'http://localhost:8080';
});

// ── Share UID with MAIN world (intercept.js) ─────────────────────────────
let _clsppUID = null;
_api.runtime.sendMessage({ type: 'GET_UID' }, (resp) => {
  if (resp && resp.uid) {
    _clsppUID = resp.uid;
    window.dispatchEvent(new CustomEvent('__clspp_uid', { detail: resp.uid }));
  }
});

// ── Context bridge: MAIN world can't fetch localhost due to page CSP ─────
window.addEventListener('__clspp_context_request', async (e) => {
  const { id, query } = e.detail;
  let context = '';
  // Respect Auto-inject toggle
  const result = await new Promise(resolve => _api.storage.local.get('autoInject', resolve));
  if (result && result.autoInject === false) {
    window.dispatchEvent(new CustomEvent('__clspp_context_response', { detail: { id, context: '' } }));
    return;
  }
  try {
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
  } catch (e) {
    // Server not running — silently fail
  }
  window.dispatchEvent(new CustomEvent('__clspp_context_response', {
    detail: { id, context }
  }));
});

// Signal to intercept.js (MAIN world) that the bridge is ready
window.dispatchEvent(new CustomEvent('__clspp_bridge_ready'));
console.log('[CLS++] context bridge ready');

// ── Query memories from background ────────────────────────────────────────
async function queryMemories(query) {
  return new Promise(resolve => {
    _api.runtime.sendMessage({ type: 'SEARCH_MEMORIES', query, limit: 5 }, resolve);
  });
}

// ── Store a message via background ────────────────────────────────────────
function storeMessage(text, source, model) {
  if (!text || text.trim().length < 6) return;
  if (text.startsWith(MEMORY_PREFIX)) return;
  _api.runtime.sendMessage({ type: 'STORE_MESSAGE', text: text.trim(), source, model });
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
