// CLS++ content script — Claude (claude.ai)
// CAPTURE ONLY — watches DOM for new messages and stores them

const SITE = 'claude';
const _seen = new Set();

function getAllMessages() {
  const msgs = [];

  // User messages — try multiple selectors for resilience
  document.querySelectorAll(
    '[data-testid="user-message"], .font-user-message, .human-turn'
  ).forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'user', text });
  });

  // If no user messages found via selectors, use action-bar heuristic:
  // Message groups WITHOUT a feedback button are user messages
  if (msgs.filter(m => m.role === 'user').length === 0) {
    document.querySelectorAll('[role="group"][aria-label="Message actions"]').forEach(group => {
      const hasFeedback = group.querySelector('button[aria-label="Give positive feedback"]');
      if (!hasFeedback) {
        // This is a user message action bar — find the message content above it
        const container = group.closest('[data-testid]') || group.parentElement?.parentElement;
        if (container) {
          const text = (container.innerText || '').trim();
          if (text && text.length >= 6) msgs.push({ role: 'user', text });
        }
      }
    });
  }

  // Assistant messages — non-streaming only
  document.querySelectorAll(
    '.font-claude-message, .ai-turn, [data-is-streaming="false"]'
  ).forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'assistant', text });
  });

  // If no assistant messages found, use feedback button heuristic
  if (msgs.filter(m => m.role === 'assistant').length === 0) {
    document.querySelectorAll('[role="group"][aria-label="Message actions"]').forEach(group => {
      const hasFeedback = group.querySelector('button[aria-label="Give positive feedback"]');
      if (hasFeedback) {
        const container = group.closest('[data-testid]') || group.parentElement?.parentElement;
        if (container) {
          // Skip if still streaming
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
    console.log('[CLS++] captured', source, text.slice(0, 60));
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

if (document.readyState === 'complete') {
  startWatching();
} else {
  window.addEventListener('load', startWatching);
}

console.log('[CLS++] Claude content script loaded');
