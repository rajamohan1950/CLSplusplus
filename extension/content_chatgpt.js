// CLS++ content script — ChatGPT (chat.openai.com, chatgpt.com)
// TWO JOBS:
// 1. CAPTURE messages from DOM → store to CLS++ production
// 2. SYNC memories into ChatGPT Custom Instructions (silent, zero friction)
//    Uses ChatGPT's /backend-api/user_system_messages — system-level, deterministic.

const SITE = 'chatgpt';
const _seen = new Set();
const SYNC_INTERVAL = 60000; // 60 seconds
const CLS_MARKER_START = '[CLS++ Memory]';
const CLS_MARKER_END = '[/CLS++]';

// ── JOB 1: CAPTURE ──────────────────────────────────────────────────────

function getAllMessages() {
  const msgs = [];
  document.querySelectorAll('[data-message-author-role]').forEach(el => {
    const role = el.dataset.messageAuthorRole;
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role, text });
  });
  return msgs;
}

function captureNewMessages() {
  const msgs = getAllMessages();
  msgs.forEach(({ role, text }) => {
    const key = role + ':' + text.slice(0, 120);
    if (_seen.has(key)) return;
    _seen.add(key);
    storeMessage(text, role === 'user' ? 'user' : 'assistant', SITE);
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

// ── JOB 2: CUSTOM INSTRUCTIONS SYNC (silent, zero friction) ─────────────
// ChatGPT sees Custom Instructions as SYSTEM context on every conversation.
// We write CLS++ memories there. The user never has to do anything.

async function fetchCLSMemories() {
  const { cls_api_key } = await chrome.storage.local.get('cls_api_key');
  if (!cls_api_key) return null;
  try {
    const r = await fetch(CLSPP_API + '/v1/memory/read', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + cls_api_key },
      body: JSON.stringify({ query: 'user identity preferences facts relationships', limit: 10 }),
    });
    if (r.ok) {
      const d = await r.json();
      return (d.items || []).map(i => i.text || '').filter(t => t.length > 3 && !t.startsWith('[Schema:'));
    }
  } catch (e) {}
  return null;
}

function buildMemoryBlock(facts) {
  if (!facts || !facts.length) return '';
  const lines = [CLS_MARKER_START];
  lines.push('Verified facts about this user from all AI conversations:');
  facts.forEach(f => lines.push('- ' + f.slice(0, 200)));
  lines.push(CLS_MARKER_END);
  return lines.join('\n');
}

async function syncCustomInstructions() {
  const { autoInject } = await chrome.storage.local.get('autoInject');
  if (autoInject === false) return;

  const facts = await fetchCLSMemories();
  if (!facts || !facts.length) return;

  const memBlock = buildMemoryBlock(facts);

  // Read current Custom Instructions
  let current;
  try {
    const r = await fetch('/backend-api/user_system_messages', { credentials: 'include' });
    if (!r.ok) return;
    current = await r.json();
  } catch (e) { return; }

  let aboutUser = current.about_user_message || '';

  // Remove old CLS++ block
  const startIdx = aboutUser.indexOf(CLS_MARKER_START);
  const endIdx = aboutUser.indexOf(CLS_MARKER_END);
  if (startIdx !== -1 && endIdx !== -1) {
    aboutUser = (aboutUser.slice(0, startIdx) + aboutUser.slice(endIdx + CLS_MARKER_END.length)).trim();
  }

  // Append new block
  const newAboutUser = (aboutUser ? aboutUser + '\n\n' : '') + memBlock;

  // Skip if unchanged
  if (newAboutUser === (current.about_user_message || '')) return;

  // Write
  try {
    await fetch('/backend-api/user_system_messages', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        about_user_message: newAboutUser,
        about_model_message: current.about_model_message || '',
        enabled: true,
      }),
    });
    console.log('[CLS++] Custom Instructions synced: ' + facts.length + ' memories');
  } catch (e) {}
}

// Sync after page loads, then every 60s
setTimeout(syncCustomInstructions, 5000);
setInterval(syncCustomInstructions, SYNC_INTERVAL);

console.log('[CLS++] ChatGPT loaded (capture + Custom Instructions sync)');
