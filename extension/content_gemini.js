// CLS++ content script — Gemini (gemini.google.com)
// CAPTURE ONLY — watches DOM for new messages and stores them

const SITE = 'gemini';
const _seen = new Set();

function getAllMessages() {
  const msgs = [];
  document.querySelectorAll('user-query').forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'user', text });
  });
  document.querySelectorAll('model-response').forEach(el => {
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) msgs.push({ role: 'assistant', text });
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

console.log('[CLS++] Gemini content script loaded');
