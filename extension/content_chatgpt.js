// CLS++ content script — ChatGPT (chat.openai.com, chatgpt.com)
// CAPTURE: watches DOM for messages and stores them to CLS++ production.
// INJECT: handled by intercept.js (fetch hook, payload-based detection).

const SITE = 'chatgpt';
const _seen = new Set();

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

console.log('[CLS++] ChatGPT capture loaded');
