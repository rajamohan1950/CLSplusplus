// CLS++ Fetch Interceptor v4 — Zero Friction
// Hooks window.fetch. Intercepts ALL POST requests on LLM sites.
// Instead of hardcoded URL patterns, detects LLM API calls by payload shape.
// If the payload has messages/prompt/content → it's an LLM call → inject.

(function () {
  'use strict';
  if (window.__clspp_intercepted) return;
  window.__clspp_intercepted = true;

  const _origFetch = window.fetch;
  const host = location.hostname;

  let site = 'unknown';
  if (host.includes('chatgpt') || host.includes('chat.openai')) site = 'chatgpt';
  else if (host.includes('claude.ai')) site = 'claude';
  else if (host.includes('gemini.google')) site = 'gemini';
  else if (host === 'localhost' || host === '127.0.0.1') site = 'local';

  if (site === 'unknown') return; // Not an LLM site

  // ── Context bridge to content script (ISOLATED world) ────────────────
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

  // ── Visual banner ────────────────────────────────────────────────────
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
    b.innerHTML = '<span style="font-size:14px">🧠</span> CLS++ injected ' + n + ' memories';
    b.style.opacity = '1';
    clearTimeout(b._t);
    b._t = setTimeout(() => { b.style.opacity = '0'; }, 3000);
  }

  // ── Payload-based LLM detection (no URL guessing) ────────────────────
  // Instead of matching URLs, parse the request body and detect LLM patterns.

  function extractUserQuery(bodyStr) {
    try {
      const b = JSON.parse(bodyStr);

      // ChatGPT: messages[].content.parts[]
      if (b.messages && Array.isArray(b.messages)) {
        const last = b.messages[b.messages.length - 1];
        if (last) {
          // ChatGPT web: {content: {parts: ["text"]}}
          if (last.content && last.content.parts && typeof last.content.parts[0] === 'string') {
            return { parsed: b, query: last.content.parts[0], inject: (ctx) => {
              last.content.parts[0] = ctx + '\n\n' + last.content.parts[0];
              return JSON.stringify(b);
            }};
          }
          // OpenAI API format: {content: "text"}
          if (typeof last.content === 'string' && last.content.length > 3) {
            return { parsed: b, query: last.content, inject: (ctx) => {
              last.content = ctx + '\n\n' + last.content;
              return JSON.stringify(b);
            }};
          }
          // content as array of blocks
          if (Array.isArray(last.content)) {
            const tb = last.content.find(c => c.type === 'text' && c.text);
            if (tb && tb.text.length > 3) {
              return { parsed: b, query: tb.text, inject: (ctx) => {
                tb.text = ctx + '\n\n' + tb.text;
                return JSON.stringify(b);
              }};
            }
          }
        }
      }

      // Claude.ai: prompt field
      if (typeof b.prompt === 'string' && b.prompt.length > 3) {
        return { parsed: b, query: b.prompt, inject: (ctx) => {
          b.prompt = ctx + '\n\n' + b.prompt;
          return JSON.stringify(b);
        }};
      }

      // Claude.ai: content array
      if (Array.isArray(b.content)) {
        const tb = b.content.find(c => c.type === 'text' && c.text && c.text.length > 3);
        if (tb) {
          return { parsed: b, query: tb.text, inject: (ctx) => {
            tb.text = ctx + '\n\n' + tb.text;
            return JSON.stringify(b);
          }};
        }
      }

    } catch (e) {}
    return null;
  }

  // For Gemini: different body format (not JSON, protobuf-like)
  function extractGeminiQuery(bodyStr) {
    if (typeof bodyStr !== 'string' || bodyStr.length < 20) return null;
    // Find user text in the nested array structure
    const matches = bodyStr.match(/\[\["([^"]{6,})"/g);
    if (matches) {
      for (const m of matches) {
        const inner = m.match(/\[\["([^"]{6,})"/);
        if (inner && inner[1].length > 5 && !/^[a-zA-Z0-9_]+$/.test(inner[1])) {
          return { query: inner[1], inject: (ctx) => {
            return bodyStr.replace(inner[1], ctx + '\n\n' + inner[1]);
          }};
        }
      }
    }
    return null;
  }

  // ── Should we intercept this request? ────────────────────────────────
  // Quick checks to avoid async overhead on non-LLM requests
  function mightBeLLMCall(url, method) {
    if (method !== 'POST') return false;
    const u = url.toLowerCase();
    // Skip known non-LLM endpoints
    if (u.includes('/analytics') || u.includes('/telemetry') || u.includes('/log') ||
        u.includes('.js') || u.includes('.css') || u.includes('.png') ||
        u.includes('/auth') || u.includes('/token') || u.includes('/session')) {
      return false;
    }
    // For known sites, check broad patterns
    if (site === 'chatgpt') return u.includes('/backend-api/') || u.includes('/api/');
    if (site === 'claude') return u.includes('/api/');
    if (site === 'gemini') return u.includes('batchexecute') || u.includes('generate');
    return u.includes('/api/') || u.includes('/v1/');
  }

  // ── Main fetch hook ──────────────────────────────────────────────────
  window.fetch = function (...args) {
    let url = '';
    let method = 'GET';

    if (args[0] instanceof Request) {
      url = args[0].url;
      method = args[0].method;
    } else {
      url = typeof args[0] === 'string' ? args[0] : '';
      method = (args[1] && args[1].method) || 'GET';
    }

    if (!mightBeLLMCall(url, method)) {
      return _origFetch.apply(this, args);
    }

    return (async () => {
      try {
        // Read body
        let body = null;
        if (args[0] instanceof Request) {
          try { body = await args[0].clone().text(); } catch (e) {}
          if (args[1] && args[1].body) body = typeof args[1].body === 'string' ? args[1].body : null;
        } else {
          body = (args[1] && typeof args[1].body === 'string') ? args[1].body : null;
        }

        if (body && body.length > 10) {
          // Try JSON-based extraction (ChatGPT, Claude)
          let ex = extractUserQuery(body);

          // Try Gemini extraction
          if (!ex && site === 'gemini') ex = extractGeminiQuery(body);

          if (ex) {
            const ctx = await getContext(ex.query);
            if (ctx) {
              const newBody = ex.inject(ctx);
              // Rebuild args with new body
              if (args[0] instanceof Request) {
                args = [new Request(args[0], { body: newBody })];
              } else {
                args = [args[0], { ...(args[1] || {}), body: newBody }];
              }
              const n = (ctx.match(/^- /gm) || []).length;
              showBanner(n);
              console.log('[CLS++] Injected ' + n + ' memories into ' + site);
            }
          }
        }
      } catch (e) {
        // Never block the user's message
      }

      return _origFetch.apply(this, args);
    })();
  };

  console.log('[CLS++] Interceptor v4 active on ' + site + ' (payload-based detection)');
})();
