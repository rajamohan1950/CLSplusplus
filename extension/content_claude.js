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

// ── INJECT: Prepend memory context when user sends a message ──────────
// Instead of hooking fetch (which breaks when Claude changes API endpoints),
// we watch for the send button click and prepend context to the input.
// This is foolproof — works regardless of API format changes.

function findSendButton() {
  // Claude.ai send button — try multiple selectors
  return document.querySelector(
    'button[aria-label="Send Message"], button[data-testid="send-button"], ' +
    'button[aria-label="Send message"], button.send-button, ' +
    'form button[type="submit"]'
  );
}

function findInputField() {
  // Claude.ai input — contenteditable div or textarea
  return document.querySelector(
    '[contenteditable="true"].ProseMirror, ' +
    'div[contenteditable="true"][data-placeholder], ' +
    'div[contenteditable="true"][role="textbox"], ' +
    'textarea[placeholder]'
  );
}

let _injecting = false;

async function injectBeforeSend() {
  if (_injecting) return;
  _injecting = true;

  try {
    const input = findInputField();
    if (!input) { _injecting = false; return; }

    const text = (input.innerText || input.value || '').trim();
    if (!text || text.length < 3) { _injecting = false; return; }

    // Don't re-inject into already-injected messages
    if (text.includes('[MEMORY')) { _injecting = false; return; }

    // Get context from bridge
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
      // Prepend context to the input field
      if (input.contentEditable === 'true') {
        // ProseMirror / contenteditable
        const existingText = input.innerText || '';
        input.focus();
        // Set text with context prepended (invisible to user since send is immediate)
        document.execCommand('selectAll', false, null);
        document.execCommand('insertText', false, context + existingText);
      } else if (input.tagName === 'TEXTAREA') {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
        setter.call(input, context + input.value);
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
      console.log('[CLS++] Memory context injected into Claude input (' + context.length + ' chars)');
    }
  } catch (e) {
    console.error('[CLS++] Inject error:', e);
  }

  _injecting = false;
}

// Watch for send button and intercept click
function watchSendButton() {
  const btn = findSendButton();
  if (btn && !btn._clsppHooked) {
    btn._clsppHooked = true;
    btn.addEventListener('click', (e) => {
      // Don't block — inject then let the click proceed
      // The injection happens synchronously enough for the form submit
    }, { capture: true });
  }

  // Also watch for Enter key in the input
  const input = findInputField();
  if (input && !input._clsppHooked) {
    input._clsppHooked = true;
    input.addEventListener('keydown', async (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        // User is sending — inject before the message goes
        await injectBeforeSend();
      }
    }, { capture: true });
  }
}

// Re-check for send button periodically (SPA navigation changes DOM)
setInterval(watchSendButton, 2000);
watchSendButton();

console.log('[CLS++] Claude content script loaded (with direct input injection)');
