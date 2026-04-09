// CLS++ content script — Claude (claude.ai)
// 1. CAPTURES messages from DOM and stores them
// 2. AUTO-SYNCS memories into Claude's system prompt
//    Uses Claude.ai's own API to set project/conversation instructions.

const SITE = 'claude';
const _seen = new Set();
const SYNC_INTERVAL = 30000;
const CLS_MARKER = '— CLS++ Cross-Model Memory (auto-updated) —';

// ── CAPTURE: Watch DOM for new messages ──────────────────────────────────

function getAllMessages() {
  const msgs = [];
  // User messages
  document.querySelectorAll(
    '[data-testid="user-message"], .font-user-message, .human-turn'
  ).forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'user', text });
  });
  if (msgs.filter(m => m.role === 'user').length === 0) {
    document.querySelectorAll('[role="group"][aria-label="Message actions"]').forEach(group => {
      const hasFeedback = group.querySelector('button[aria-label="Give positive feedback"]');
      if (!hasFeedback) {
        const container = group.closest('[data-testid]') || group.parentElement?.parentElement;
        if (container) {
          const text = (container.innerText || '').trim();
          if (text && text.length >= 6) msgs.push({ role: 'user', text });
        }
      }
    });
  }
  // Assistant messages
  document.querySelectorAll(
    '.font-claude-message, .ai-turn, [data-is-streaming="false"]'
  ).forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'assistant', text });
  });
  if (msgs.filter(m => m.role === 'assistant').length === 0) {
    document.querySelectorAll('[role="group"][aria-label="Message actions"]').forEach(group => {
      const hasFeedback = group.querySelector('button[aria-label="Give positive feedback"]');
      if (hasFeedback) {
        const container = group.closest('[data-testid]') || group.parentElement?.parentElement;
        if (container) {
          if (container.querySelector('[data-is-streaming="true"]')) return;
          const text = (container.innerText || '').trim();
          if (text && text.length >= 6) msgs.push({ role: 'assistant', text });
        }
      }
    });
  }
  return msgs;
}

function captureNewMessages() {
  const msgs = getAllMessages();
  msgs.forEach(({ role, text }) => {
    const key = role + ':' + text.slice(0, 120);
    if (_seen.has(key)) return;
    _seen.add(key);
    const source = role === 'user' ? 'user' : 'assistant';
    storeMessage(text, source, SITE);
  });
}

const observer = new MutationObserver(() => {
  clearTimeout(observer._timer);
  observer._timer = setTimeout(captureNewMessages, 1500);
});

function startWatching() {
  const container = document.querySelector('main') || document.body;
  observer.observe(container, { childList: true, subtree: true, characterData: true });
  setTimeout(captureNewMessages, 2000);
}

if (document.readyState === 'complete') startWatching();
else window.addEventListener('load', startWatching);

// ── MEMORY SYNC FOR CLAUDE.AI ────────────────────────────────────────────
// Claude.ai doesn't have a public "Custom Instructions" API like ChatGPT.
// Instead, we use the direct input injection approach as a fallback,
// AND prepend context via the fetch intercept (intercept.js handles this).
//
// For Claude.ai, the most reliable path is:
// 1. Extension captures messages → stores to CLS++ production
// 2. User uses the CLS++ proxy (base_url) for API-based Claude usage
// 3. For browser: intercept.js handles fetch-level injection
//
// Additionally, we show a floating memory panel so the user can SEE
// what CLS++ knows, even if injection fails.

async function fetchMemories() {
  const { cls_api_key } = await chrome.storage.local.get('cls_api_key');
  if (!cls_api_key) return null;
  try {
    const r = await fetch(`${CLSPP_API}/v1/memory/read`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${cls_api_key}`,
      },
      body: JSON.stringify({
        query: 'user identity preferences facts relationships context',
        limit: 8,
      }),
    });
    if (r.ok) {
      const d = await r.json();
      return (d.items || [])
        .map(i => i.text || '')
        .filter(t => t.length > 3 && !t.startsWith('[Schema:'));
    }
  } catch (e) {}
  return null;
}

// ── Floating memory panel (visible context for Claude.ai) ────────────────
let _panelCreated = false;

function showMemoryPanel(facts) {
  if (!facts || !facts.length) return;

  let panel = document.getElementById('__clspp_panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = '__clspp_panel';
    panel.style.cssText = 'position:fixed;bottom:80px;right:16px;z-index:999999;' +
      'width:280px;max-height:300px;overflow-y:auto;' +
      'background:rgba(10,10,20,0.95);border:1px solid rgba(124,110,240,0.3);' +
      'border-radius:12px;padding:12px;font-family:system-ui;font-size:11px;' +
      'color:#e0e0f0;box-shadow:0 8px 32px rgba(0,0,0,0.5);' +
      'transition:opacity 0.3s;';

    // Close button
    const close = document.createElement('div');
    close.textContent = '✕';
    close.style.cssText = 'position:absolute;top:6px;right:8px;cursor:pointer;' +
      'color:rgba(255,255,255,0.4);font-size:14px;';
    close.addEventListener('click', () => { panel.style.display = 'none'; });
    panel.appendChild(close);

    // Minimize toggle
    const header = document.createElement('div');
    header.style.cssText = 'font-size:10px;font-weight:700;color:#7c6ef0;' +
      'letter-spacing:0.05em;text-transform:uppercase;margin-bottom:8px;cursor:pointer;';
    header.textContent = '🧠 CLS++ Memory';
    header.addEventListener('click', () => {
      const body = panel.querySelector('.clspp-panel-body');
      body.style.display = body.style.display === 'none' ? '' : 'none';
    });
    panel.appendChild(header);

    const body = document.createElement('div');
    body.className = 'clspp-panel-body';
    panel.appendChild(body);

    document.body.appendChild(panel);
    _panelCreated = true;
  }

  const body = panel.querySelector('.clspp-panel-body');
  body.innerHTML = facts.map(f =>
    '<div style="padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.06);color:#c0c0d0;">• ' +
    f.slice(0, 150).replace(/</g, '&lt;') + '</div>'
  ).join('');
  panel.style.display = '';
}

async function syncMemoryPanel() {
  const { autoInject } = await chrome.storage.local.get('autoInject');
  if (autoInject === false) return;
  const facts = await fetchMemories();
  if (facts && facts.length) {
    showMemoryPanel(facts);
  }
}

// Sync on load, then every 30s
setTimeout(syncMemoryPanel, 5000);
setInterval(syncMemoryPanel, SYNC_INTERVAL);

console.log('[CLS++] Claude content script loaded (capture + memory panel)');
