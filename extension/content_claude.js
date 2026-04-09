// CLS++ content script — Claude (claude.ai)
// CAPTURE: watches DOM for messages and stores them to CLS++ production.
// INJECT: handled by intercept.js (fetch hook, payload-based detection).

const SITE = 'claude';
const _seen = new Set();

function getAllMessages() {
  const msgs = [];
  document.querySelectorAll(
    '[data-testid="user-message"], .font-user-message, .human-turn'
  ).forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'user', text });
  });
  if (msgs.filter(m => m.role === 'user').length === 0) {
    document.querySelectorAll('[role="group"][aria-label="Message actions"]').forEach(group => {
      if (!group.querySelector('button[aria-label="Give positive feedback"]')) {
        const container = group.closest('[data-testid]') || group.parentElement?.parentElement;
        if (container) {
          const text = (container.innerText || '').trim();
          if (text && text.length >= 6) msgs.push({ role: 'user', text });
        }
      }
    });
  }
  document.querySelectorAll(
    '.font-claude-message, .ai-turn, [data-is-streaming="false"]'
  ).forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'assistant', text });
  });
  if (msgs.filter(m => m.role === 'assistant').length === 0) {
    document.querySelectorAll('[role="group"][aria-label="Message actions"]').forEach(group => {
      if (group.querySelector('button[aria-label="Give positive feedback"]')) {
        const container = group.closest('[data-testid]') || group.parentElement?.parentElement;
        if (container && !container.querySelector('[data-is-streaming="true"]')) {
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

console.log('[CLS++] Claude capture loaded');
