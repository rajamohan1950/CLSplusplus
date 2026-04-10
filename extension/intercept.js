// CLS++ Fetch Interceptor — MAIN world, document_start
// Locks window.fetch so no page code can replace it.
// On LLM API POST: gets context from capture.js bridge, injects into body.

(function () {
  'use strict';
  if (window.__clspp) return;
  window.__clspp = true;

  const _fetch = window.fetch.bind(window);
  const host = location.hostname;

  // Context bridge to ISOLATED world (capture.js)
  let _id = 0, _cbs = {}, _ready = false;
  window.addEventListener('__cls_ready', () => { _ready = true; });
  window.addEventListener('__cls_ctx', e => {
    if (e.detail && _cbs[e.detail.id]) { _cbs[e.detail.id](e.detail.ctx || ''); delete _cbs[e.detail.id]; }
  });

  function getCtx(query) {
    return new Promise(resolve => {
      const id = ++_id;
      const t = setTimeout(() => { delete _cbs[id]; resolve(''); }, 3000);
      _cbs[id] = ctx => { clearTimeout(t); resolve(ctx); };
      const go = () => window.dispatchEvent(new CustomEvent('__cls_req', { detail: { id, query } }));
      if (_ready) go(); else { setTimeout(go, 1500); window.addEventListener('__cls_ready', go, { once: true }); }
    });
  }

  // Extract user message from JSON body
  function extract(body) {
    try {
      const b = JSON.parse(body);
      // ChatGPT
      if (b.messages && b.messages.length) {
        const m = b.messages[b.messages.length - 1];
        if (m.content && m.content.parts && typeof m.content.parts[0] === 'string')
          return { q: m.content.parts[0], set: ctx => { m.content.parts[0] = ctx + m.content.parts[0]; return JSON.stringify(b); } };
        if (typeof m.content === 'string')
          return { q: m.content, set: ctx => { m.content = ctx + m.content; return JSON.stringify(b); } };
      }
      // Claude
      if (typeof b.prompt === 'string' && b.prompt.length > 3)
        return { q: b.prompt, set: ctx => { b.prompt = ctx + b.prompt; return JSON.stringify(b); } };
      if (Array.isArray(b.content)) {
        const t = b.content.find(c => c.type === 'text' && c.text);
        if (t) return { q: t.text, set: ctx => { t.text = ctx + t.text; return JSON.stringify(b); } };
      }
    } catch (_) {}
    return null;
  }

  // Is this an LLM API call?
  function isLLM(url) {
    const u = url.toLowerCase();
    if (host.includes('chatgpt') || host.includes('openai')) return u.includes('/backend-api/') && u.includes('conversation');
    if (host.includes('claude')) return u.includes('/api/') && (u.includes('chat') || u.includes('completion') || u.includes('message'));
    if (host.includes('gemini')) return u.includes('batchexecute');
    return false;
  }

  function clsFetch(...args) {
    let url = '', method = 'GET';
    if (args[0] instanceof Request) { url = args[0].url; method = args[0].method; }
    else { url = String(args[0] || ''); method = (args[1] && args[1].method) || 'GET'; }

    if (method !== 'POST' || !isLLM(url)) return _fetch(...args);

    return (async () => {
      try {
        let body = null;
        if (args[0] instanceof Request) { try { body = await args[0].clone().text(); } catch (_) {} }
        else body = (args[1] && typeof args[1].body === 'string') ? args[1].body : null;

        if (body) {
          const ex = extract(body);
          if (ex) {
            const ctx = await getCtx(ex.q);
            if (ctx) {
              const nb = ex.set(ctx);
              args = args[0] instanceof Request ? [new Request(args[0], { body: nb })] : [args[0], { ...(args[1] || {}), body: nb }];
              console.log('[CLS++] Memory injected');
            }
          }
        }
      } catch (_) {}
      return _fetch(...args);
    })();
  }

  // Lock fetch — non-writable
  try {
    Object.defineProperty(window, 'fetch', { value: clsFetch, writable: false, configurable: false });
    console.log('[CLS++] Fetch locked on', host);
  } catch (_) {
    window.fetch = clsFetch;
  }
})();
