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

// ── INJECT: Prepend memory context when user sends a message ──────────
// Direct input injection — no fetch hooking, no URL pattern matching.
// Works regardless of ChatGPT API endpoint changes.

function findChatGPTInput() {
  return document.querySelector(
    '#prompt-textarea, ' +
    'textarea[data-id="root"], ' +
    'div[contenteditable="true"][id="prompt-textarea"], ' +
    'div[contenteditable="true"][data-placeholder]'
  );
}

let _injecting = false;

async function injectBeforeSend() {
  if (_injecting) return;
  _injecting = true;

  try {
    const input = findChatGPTInput();
    if (!input) { _injecting = false; return; }

    const text = (input.innerText || input.value || '').trim();
    if (!text || text.length < 3) { _injecting = false; return; }
    if (text.includes('[MEMORY')) { _injecting = false; return; }

    const { autoInject, cls_api_key } = await chrome.storage.local.get(['autoInject', 'cls_api_key']);
    if (autoInject === false || !cls_api_key) { _injecting = false; return; }

    let context = '';
    try {
      const r = await fetch(CLSPP_API + '/v1/memory/read', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + cls_api_key,
        },
        body: JSON.stringify({ query: text, limit: 5 }),
      });
      if (r.ok) {
        const d = await r.json();
        const items = d.items || [];
        if (items.length > 0) {
          const lines = [
            '[MEMORY — VERIFIED USER FACTS]',
            'Answer based on these confirmed facts about the user:',
          ];
          items.forEach(m => lines.push('- ' + (m.text || '').slice(0, 200)));
          context = lines.join('\n') + '\n\n';
        }
      }
    } catch (err) {
      console.log('[CLS++] Memory fetch failed:', err);
    }

    if (context) {
      if (input.contentEditable === 'true') {
        const existingText = input.innerText || '';
        input.focus();
        document.execCommand('selectAll', false, null);
        document.execCommand('insertText', false, context + existingText);
      } else if (input.tagName === 'TEXTAREA') {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
        setter.call(input, context + input.value);
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
      console.log('[CLS++] Memory context injected into ChatGPT input (' + context.length + ' chars)');
    }
  } catch (e) {
    console.error('[CLS++] Inject error:', e);
  }

  _injecting = false;
}

function watchChatGPTSend() {
  const input = findChatGPTInput();
  if (input && !input._clsppHooked) {
    input._clsppHooked = true;
    input.addEventListener('keydown', async (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        await injectBeforeSend();
      }
    }, { capture: true });
  }

  // Also hook send button
  const btn = document.querySelector(
    'button[data-testid="send-button"], button[aria-label="Send prompt"], ' +
    'form button[type="submit"]'
  );
  if (btn && !btn._clsppHooked) {
    btn._clsppHooked = true;
    btn.addEventListener('click', () => injectBeforeSend(), { capture: true });
  }
}

setInterval(watchChatGPTSend, 2000);
watchChatGPTSend();

console.log('[CLS++] ChatGPT content script loaded (with direct input injection)');
