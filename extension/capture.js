// CLS++ Capture — ISOLATED world
// 1. Captures user messages from DOM → stores via background.js
// 2. Context bridge: responds to intercept.js memory requests

const seen = new Set();
const host = location.hostname;

// ── DOM Selectors per site ──
function getMessages() {
  const msgs = [];
  if (host.includes('chatgpt') || host.includes('openai')) {
    document.querySelectorAll('[data-message-author-role="user"]').forEach(el => {
      const t = (el.innerText || '').trim();
      if (t.length > 5) msgs.push(t);
    });
  } else if (host.includes('claude')) {
    document.querySelectorAll('[data-testid="user-message"], .font-user-message, .human-turn').forEach(el => {
      const t = (el.innerText || '').trim();
      if (t.length > 5) msgs.push(t);
    });
  } else if (host.includes('gemini')) {
    document.querySelectorAll('user-query').forEach(el => {
      const t = (el.innerText || '').trim();
      if (t.length > 5) msgs.push(t);
    });
  }
  return msgs;
}

// ── Capture new user messages ──
function capture() {
  getMessages().forEach(text => {
    const key = text.slice(0, 100);
    if (seen.has(key)) return;
    seen.add(key);
    // Skip our own injected context
    if (text.includes('For context, here are some things')) return;
    chrome.runtime.sendMessage({ type: 'STORE', text: text.slice(0, 2000) });
  });
}

new MutationObserver(() => { clearTimeout(capture._t); capture._t = setTimeout(capture, 1500); })
  .observe(document.body, { childList: true, subtree: true });
setTimeout(capture, 2000);

// ── Context bridge: intercept.js (MAIN) asks for memories ──
window.addEventListener('__cls_req', async (e) => {
  const { id } = e.detail || {};
  let ctx = '';
  try {
    const resp = await new Promise(resolve => {
      chrome.runtime.sendMessage({ type: 'FETCH', limit: 10 }, r => resolve(r || { facts: [] }));
    });
    const facts = (resp.facts || []).filter(f =>
      f.length > 8 && f.length < 250 &&
      !f.startsWith('[Schema:') && !f.startsWith('[MEMORY') &&
      !f.includes('VERIFIED USER') && !f.includes('END MEMORY') &&
      !f.endsWith('?')
    );
    if (facts.length > 0) {
      ctx = 'For context, here are some things I have mentioned before in other conversations:\n'
        + facts.map(f => '- ' + f.slice(0, 200)).join('\n')
        + '\n\nNow, my actual question:\n';
    }
  } catch (_) {}
  window.dispatchEvent(new CustomEvent('__cls_ctx', { detail: { id, ctx } }));
});

window.dispatchEvent(new CustomEvent('__cls_ready'));
console.log('[CLS++] Capture ready on', host);
