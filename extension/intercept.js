// CLS++ Fetch Interceptor v5 — Non-overridable API intercept
// Uses Object.defineProperty to lock the fetch hook so no page code can replace it.
// Detects LLM API calls by payload shape, injects memory context.

(function () {
  'use strict';
  if (window.__clspp_intercepted) return;
  window.__clspp_intercepted = true;

  const _origFetch = window.fetch.bind(window);

  const host = location.hostname;
  let site = 'unknown';
  if (host.includes('chatgpt') || host.includes('chat.openai')) site = 'chatgpt';
  else if (host.includes('claude.ai')) site = 'claude';
  else if (host.includes('gemini.google')) site = 'gemini';
  else if (host === 'localhost' || host === '127.0.0.1') site = 'local';

  if (site === 'unknown') return;

  // ── Context bridge to content script ────────────────────────────────
  let _ctxId = 0;
  const _ctxCallbacks = {};
  let _bridgeReady = false;

  window.addEventListener('__clspp_bridge_ready', () => { _bridgeReady = true; });
  window.addEventListener('__clspp_context_response', (e) => {
    if (!e.detail) return;
    const { id, context } = e.detail;
    if (_ctxCallbacks[id]) {
      _ctxCallbacks[id](context || '');
      delete _ctxCallbacks[id];
    }
  });

  function getContext(query) {
    return new Promise((resolve) => {
      const id = ++_ctxId;
      const timer = setTimeout(() => { delete _ctxCallbacks[id]; resolve(''); }, 3000);
      _ctxCallbacks[id] = (ctx) => { clearTimeout(timer); resolve(ctx); };
      const dispatch = () => {
        window.dispatchEvent(new CustomEvent('__clspp_context_request', { detail: { id, query } }));
      };
      if (_bridgeReady) dispatch();
      else {
        const bt = setTimeout(dispatch, 1500);
        window.addEventListener('__clspp_bridge_ready', () => { clearTimeout(bt); dispatch(); }, { once: true });
      }
    });
  }

  // ── Visual banner ───────────────────────────────────────────────────
  function showBanner(n) {
    let b = document.getElementById('__clspp_banner');
    if (!b) {
      b = document.createElement('div');
      b.id = '__clspp_banner';
      b.style.cssText = 'position:fixed;top:8px;right:8px;z-index:999999;' +
        'background:rgba(16,24,16,0.95);border:1px solid rgba(93,224,197,0.4);' +
        'border-radius:8px;padding:6px 12px;font-size:12px;font-family:system-ui;' +
        'color:#5de0c5;pointer-events:none;transition:opacity 0.3s;opacity:0;' +
        'box-shadow:0 4px 12px rgba(0,0,0,0.3);display:flex;align-items:center;gap:6px;';
      document.body.appendChild(b);
    }
    b.textContent = '🧠 CLS++ injected ' + n + ' memories';
    b.style.opacity = '1';
    clearTimeout(b._t);
    b._t = setTimeout(() => { b.style.opacity = '0'; }, 3000);
  }

  // ── Extract user query from request body ────────────────────────────
  function extractAndInject(bodyStr) {
    try {
      const b = JSON.parse(bodyStr);

      // ChatGPT: messages[].content.parts[]
      if (b.messages && Array.isArray(b.messages)) {
        const last = b.messages[b.messages.length - 1];
        if (last && last.content && last.content.parts && typeof last.content.parts[0] === 'string') {
          return { query: last.content.parts[0], inject: (ctx) => {
            last.content.parts[0] = ctx + last.content.parts[0];
            return JSON.stringify(b);
          }};
        }
        if (last && typeof last.content === 'string' && last.content.length > 3) {
          return { query: last.content, inject: (ctx) => {
            last.content = ctx + last.content;
            return JSON.stringify(b);
          }};
        }
        if (last && Array.isArray(last.content)) {
          const tb = last.content.find(c => c.type === 'text' && c.text);
          if (tb && tb.text.length > 3) {
            return { query: tb.text, inject: (ctx) => {
              tb.text = ctx + tb.text;
              return JSON.stringify(b);
            }};
          }
        }
      }

      // Claude: prompt field
      if (typeof b.prompt === 'string' && b.prompt.length > 3) {
        return { query: b.prompt, inject: (ctx) => {
          b.prompt = ctx + b.prompt;
          return JSON.stringify(b);
        }};
      }

      // Claude: content array
      if (Array.isArray(b.content)) {
        const tb = b.content.find(c => c.type === 'text' && c.text && c.text.length > 3);
        if (tb) {
          return { query: tb.text, inject: (ctx) => {
            tb.text = ctx + tb.text;
            return JSON.stringify(b);
          }};
        }
      }
    } catch (e) {}
    return null;
  }

  // ── Should we intercept? ────────────────────────────────────────────
  function isLLMPost(url, method) {
    if (method !== 'POST') return false;
    const u = url.toLowerCase();
    if (u.includes('/analytics') || u.includes('/telemetry') || u.includes('.js') ||
        u.includes('.css') || u.includes('/auth') || u.includes('/token')) return false;
    if (site === 'chatgpt') return u.includes('/backend-api/') || u.includes('/api/');
    if (site === 'claude') return u.includes('/api/');
    if (site === 'gemini') return u.includes('batchexecute') || u.includes('generate');
    return false;
  }

  // ── The fetch wrapper ───────────────────────────────────────────────
  function clsFetch(...args) {
    let url = '', method = 'GET';
    if (args[0] instanceof Request) { url = args[0].url; method = args[0].method; }
    else { url = String(args[0] || ''); method = (args[1] && args[1].method) || 'GET'; }

    if (!isLLMPost(url, method)) {
      return _origFetch(...args);
    }

    return (async () => {
      try {
        let body = null;
        if (args[0] instanceof Request) {
          try { body = await args[0].clone().text(); } catch (e) {}
          if (args[1] && typeof args[1].body === 'string') body = args[1].body;
        } else {
          body = (args[1] && typeof args[1].body === 'string') ? args[1].body : null;
        }

        if (body && body.length > 10) {
          const ex = extractAndInject(body);
          if (ex) {
            const ctx = await getContext(ex.query);
            if (ctx) {
              const newBody = ex.inject(ctx);
              if (args[0] instanceof Request) {
                args = [new Request(args[0], { body: newBody })];
              } else {
                args = [args[0], { ...(args[1] || {}), body: newBody }];
              }
              const n = (ctx.match(/\n- /g) || []).length;
              showBanner(n);
              console.log('[CLS++] Injected', n, 'memories into', site);
            }
          }
        }
      } catch (e) {}

      return _origFetch(...args);
    })();
  }

  // ── LOCK the fetch hook — non-writable, non-configurable ────────────
  // This prevents ChatGPT/Claude/Gemini from replacing our hook
  try {
    Object.defineProperty(window, 'fetch', {
      value: clsFetch,
      writable: false,
      configurable: false,
    });
    console.log('[CLS++] Fetch hook LOCKED on', site, '(non-writable)');
  } catch (e) {
    // Fallback: simple assignment if defineProperty fails
    window.fetch = clsFetch;
    console.log('[CLS++] Fetch hook set on', site, '(writable fallback)');
  }
})();
