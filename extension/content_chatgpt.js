// CLS++ content script — ChatGPT (chat.openai.com, chatgpt.com)
// 1. CAPTURES messages from DOM and stores them
// 2. AUTO-SYNCS Custom Instructions with user's CLS++ memories
//    This is deterministic — ChatGPT's own system-level injection, no DOM hacking.

const SITE = 'chatgpt';
const _seen = new Set();
const SYNC_INTERVAL = 30000; // 30 seconds
const CLS_MARKER = '— CLS++ Cross-Model Memory (auto-updated) —';

// ── CAPTURE: Watch DOM for new messages ──────────────────────────────────

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

// ── CUSTOM INSTRUCTIONS SYNC ─────────────────────────────────────────────
// Uses ChatGPT's own /backend-api/user_system_messages endpoint.
// This injects memories as a system-level instruction that ChatGPT ALWAYS sees.
// No fetch interception, no DOM manipulation. 100% deterministic.

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
        limit: 10,
      }),
    });
    if (r.ok) {
      const d = await r.json();
      return (d.items || [])
        .map(i => i.text || '')
        .filter(t => t.length > 3 && !t.startsWith('[Schema:'));
    }
  } catch (e) {
    console.log('[CLS++] Memory fetch failed:', e);
  }
  return null;
}

function buildCustomInstructions(facts) {
  if (!facts || !facts.length) return null;
  const lines = [
    CLS_MARKER,
    'These are verified facts about this user from their conversations across ALL AI models (Claude, Gemini, Grok, etc.).',
    'Treat them as ground truth. Answer based on these facts when relevant:',
    '',
  ];
  facts.forEach(f => lines.push('- ' + f.slice(0, 200)));
  lines.push('', CLS_MARKER);
  return lines.join('\n');
}

async function getCurrentCustomInstructions() {
  try {
    const r = await fetch('/backend-api/user_system_messages', {
      credentials: 'include',
    });
    if (r.ok) {
      const d = await r.json();
      return {
        about_user: d.about_user_message || '',
        about_model: d.about_model_message || '',
        enabled: d.enabled !== false,
      };
    }
  } catch (e) {
    console.log('[CLS++] Failed to read custom instructions:', e);
  }
  return null;
}

async function setCustomInstructions(aboutUser, aboutModel) {
  try {
    const r = await fetch('/backend-api/user_system_messages', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        about_user_message: aboutUser,
        about_model_message: aboutModel,
        enabled: true,
      }),
    });
    return r.ok;
  } catch (e) {
    console.log('[CLS++] Failed to set custom instructions:', e);
    return false;
  }
}

async function syncCustomInstructions() {
  const { autoInject } = await chrome.storage.local.get('autoInject');
  if (autoInject === false) return;

  const facts = await fetchMemories();
  if (!facts || !facts.length) return;

  const memoryBlock = buildCustomInstructions(facts);
  if (!memoryBlock) return;

  const current = await getCurrentCustomInstructions();
  if (!current) return;

  // Merge: preserve user's existing custom instructions, update CLS++ block
  let aboutUser = current.about_user;

  // Remove old CLS++ block if present
  const markerRegex = new RegExp(
    CLS_MARKER.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') +
    '[\\s\\S]*?' +
    CLS_MARKER.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
    'g'
  );
  aboutUser = aboutUser.replace(markerRegex, '').trim();

  // Append new CLS++ block
  aboutUser = (aboutUser ? aboutUser + '\n\n' : '') + memoryBlock;

  // Only update if changed
  if (aboutUser !== current.about_user) {
    const ok = await setCustomInstructions(aboutUser, current.about_model);
    if (ok) {
      console.log('[CLS++] Custom Instructions synced (' + facts.length + ' memories)');
    }
  }
}

// Sync on page load, then every 30s
setTimeout(syncCustomInstructions, 5000);
setInterval(syncCustomInstructions, SYNC_INTERVAL);

console.log('[CLS++] ChatGPT content script loaded (Custom Instructions sync)');
