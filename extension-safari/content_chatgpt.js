// CLS++ content script — ChatGPT (chat.openai.com, chatgpt.com)
// CAPTURE ONLY — watches DOM for new messages and stores them

const SITE = 'chatgpt';
const _seen = new Set();

function getAllMessages() {
  const msgs = [];
  // ChatGPT uses [data-message-author-role] on message containers
  document.querySelectorAll('[data-message-author-role]').forEach(el => {
    const role = el.dataset.messageAuthorRole; // "user" or "assistant"
    const text = (el.innerText || '').trim();
    if (text && text.length >= 6) {
      msgs.push({ role, text });
    }
  });
  return msgs;
}

function captureNewMessages() {
  const msgs = getAllMessages();
  msgs.forEach(({ role, text }) => {
    // Use first 120 chars as dedup key
    const key = role + ':' + text.slice(0, 120);
    if (_seen.has(key)) return;
    _seen.add(key);

    const source = role === 'user' ? 'user' : 'assistant';
    storeMessage(text, source, SITE);
    console.log('[CLS++] captured', source, text.slice(0, 60));
  });
}

// Watch for DOM changes (new messages appearing)
const observer = new MutationObserver(() => {
  // Debounce — ChatGPT streams tokens, wait for it to settle
  clearTimeout(observer._timer);
  observer._timer = setTimeout(captureNewMessages, 1500);
});

// Start observing once the conversation container exists
function startWatching() {
  const container = document.querySelector('main') || document.body;
  observer.observe(container, { childList: true, subtree: true, characterData: true });
  // Initial capture of any existing messages
  setTimeout(captureNewMessages, 2000);
}

if (document.readyState === 'complete') {
  startWatching();
} else {
  window.addEventListener('load', startWatching);
}

console.log('[CLS++] ChatGPT content script loaded');
