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
  // Route ALL API calls through background.js service worker (no CORS issues)
  try {
    const resp = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: 'FETCH_MEMORIES', query: query, limit: 5 },
        (r) => resolve(r || { items: [] })
      );
    });
    const items = resp.items || [];
    if (items.length > 0) {
      const lines = [
        '[MEMORY — VERIFIED USER FACTS]',
        'These are confirmed facts about this user from prior conversations across all AI models.',
        'Treat them as ground truth:',
      ];
      items.forEach(m => lines.push('- ' + (m.text || '')));
      context = lines.join('\n') + '\n';
    }
  } catch (err) {
    // Background service worker not available
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

// ══════════════════════════════════════════════════════════════════════════
// INVISIBLE DOM PREPEND — Foolproof cross-LLM memory injection
//
// Pre-fetches memories every 30s into a cache. When user presses Enter
// or clicks Send, synchronously prepends cached context into the input
// field BEFORE the page reads it. Works on ChatGPT, Claude, Gemini.
//
// Uses capture phase (fires BEFORE page handlers) — no fetch hooks needed.
// ══════════════════════════════════════════════════════════════════════════

let _cachedContext = '';
let _lastCacheTime = 0;
const CACHE_INTERVAL = 30000; // 30 seconds
const CONTEXT_PREFIX = '[MEMORY — VERIFIED USER FACTS]\nAnswer using these confirmed facts about this user:\n';
const CONTEXT_SUFFIX = '\n[END MEMORY]\n\n';

// Periodic cache refresh — memories always ready when user presses Enter
async function refreshMemoryCache() {
  try {
    const { autoInject } = await chrome.storage.local.get('autoInject');
    if (autoInject === false) { _cachedContext = ''; return; }

    const resp = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: 'FETCH_MEMORIES', query: 'user personal identity preferences facts relationships', limit: 8 },
        (r) => resolve(r || { items: [] })
      );
    });
    const items = (resp.items || []).filter(i => {
      const t = i.text || '';
      const hasSignal = /\b(name|is|are|was|has|likes?|loves?|prefers?|works?|lives?|born|sister|brother|mother|father|wife|husband|friend|favou?rite|colou?r|planning|trip|installed|diesel|perfume|mango)\b/i.test(t);
      const isJunk = t.length < 12 || t.length > 300 || t.startsWith('[') || t.endsWith('?') ||
        /\b(merge|branch|commit|deploy|push|extension|localhost|hardcode|chrome|hook|server|render|docker|test|fix|error|debug)\b/i.test(t) ||
        /^(fu|bs|ok|yes|no|s|k|ya|nothing|prepare|why|how|what)\b/i.test(t);
      return hasSignal && !isJunk;
    });
    if (items.length > 0) {
      _cachedContext = CONTEXT_PREFIX + items.map(i => '- ' + (i.text || '').slice(0, 200)).join('\n') + CONTEXT_SUFFIX;
    } else {
      _cachedContext = '';
    }
    _lastCacheTime = Date.now();
    console.log('[CLS++] Memory cache refreshed:', items.length, 'facts');
  } catch (e) {
    // Background not available
  }
}

// Refresh immediately, then every 30s
refreshMemoryCache();
setInterval(refreshMemoryCache, CACHE_INTERVAL);

// ── Find the active input field on any LLM site ──────────────────────────
function findActiveInput() {
  // ChatGPT
  const chatgpt = document.querySelector('#prompt-textarea, div[contenteditable="true"][id="prompt-textarea"]');
  if (chatgpt) return chatgpt;

  // Claude.ai
  const claude = document.querySelector(
    'div[contenteditable="true"].ProseMirror, ' +
    'div[contenteditable="true"][data-placeholder], ' +
    'div[contenteditable="true"][role="textbox"]'
  );
  if (claude) return claude;

  // Gemini
  const gemini = document.querySelector(
    'div[contenteditable="true"][aria-label], ' +
    'rich-textarea div[contenteditable="true"]'
  );
  if (gemini) return gemini;

  // Generic fallback
  return document.querySelector(
    'textarea[placeholder], div[contenteditable="true"]'
  );
}

// ── Prepend context into input field (works with contenteditable + textarea) ──
function prependToInput(input) {
  if (!_cachedContext) return false;
  if (!input) return false;

  const currentText = input.innerText || input.value || '';
  if (currentText.length < 3) return false;
  if (currentText.includes('[MEMORY')) return false; // Already injected

  const newText = _cachedContext + currentText;

  if (input.contentEditable === 'true') {
    // contenteditable (ChatGPT, Claude, Gemini)
    input.focus();
    document.execCommand('selectAll', false, null);
    document.execCommand('insertText', false, newText);
  } else if (input.tagName === 'TEXTAREA') {
    // textarea fallback
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
    if (setter) {
      setter.call(input, newText);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }

  console.log('[CLS++] Memory prepended to input (' + _cachedContext.length + ' chars)');
  return true;
}

// ── Capture-phase listeners — fire BEFORE page handlers ──────────────────
// When user presses Enter or clicks Send, prepend cached context SYNCHRONOUSLY

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && _cachedContext) {
    const input = findActiveInput();
    if (input) prependToInput(input);
  }
}, { capture: true }); // capture: true = fires BEFORE page's event handlers

// Also watch for Send button clicks
function hookSendButton() {
  const btns = document.querySelectorAll(
    'button[data-testid="send-button"], ' +
    'button[aria-label="Send prompt"], ' +
    'button[aria-label="Send Message"], ' +
    'button[aria-label="Send message"], ' +
    'form button[type="submit"]'
  );
  btns.forEach(btn => {
    if (btn._clsppHooked) return;
    btn._clsppHooked = true;
    btn.addEventListener('click', () => {
      if (_cachedContext) {
        const input = findActiveInput();
        if (input) prependToInput(input);
      }
    }, { capture: true });
  });
}

// Re-check for Send buttons (SPA navigation creates new DOM)
setInterval(hookSendButton, 3000);
hookSendButton();

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
