// CLS++ Fetch Interceptor — MAIN world, document_start
// Locks window.fetch so no page code can replace it.
// On LLM API POST: reads prefetched memories from DOM, injects into body.

(function () {
  'use strict';
  console.log('[CLS++] intercept.js loading on', location.hostname);
  if (window.__clspp) { console.log('[CLS++] already loaded, skipping'); return; }
  window.__clspp = true;

  const _fetch = window.fetch.bind(window);
  const host = location.hostname;

  // Read memories from DOM mailbox (written by capture.js in ISOLATED world)
  function getCtx() {
    const el = document.getElementById('__cls_mem');
    if (!el) { console.log('[CLS++] No memory element found'); return ''; }
    try {
      const facts = JSON.parse(el.getAttribute('data-facts') || '[]');
      if (facts.length === 0) { console.log('[CLS++] Memory element empty'); return ''; }
      console.log('[CLS++] Read', facts.length, 'facts from DOM');
      return 'For context, here are some things I have mentioned before in other conversations:\n'
        + facts.map(f => '- ' + f.slice(0, 200)).join('\n')
        + '\n\nNow, my actual question:\n';
    } catch (e) { console.log('[CLS++] Failed to parse memory element:', e.message); return ''; }
  }

  // Write user message to body attribute — capture.js (ISOLATED) watches and stores via API
  function writeOutbox(text) {
    try {
      const payload = JSON.stringify({ t: text.slice(0, 2000), ts: Date.now() });
      document.body.setAttribute('data-cls-outbox', payload);
      console.log('[CLS++] Wrote to outbox:', text.slice(0, 80));
    } catch (_) {}
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
      // Claude — b.prompt (legacy)
      if (typeof b.prompt === 'string' && b.prompt.length > 3)
        return { q: b.prompt, set: ctx => { b.prompt = ctx + b.prompt; return JSON.stringify(b); } };
      // Claude — b.content array [{type:'text', text:'...'}]
      if (Array.isArray(b.content)) {
        const t = b.content.find(c => c.type === 'text' && c.text);
        if (t) return { q: t.text, set: ctx => { t.text = ctx + t.text; return JSON.stringify(b); } };
      }
      // Claude web — other possible shapes
      if (typeof b.query === 'string' && b.query.length > 3)
        return { q: b.query, set: ctx => { b.query = ctx + b.query; return JSON.stringify(b); } };
      if (typeof b.text === 'string' && b.text.length > 3)
        return { q: b.text, set: ctx => { b.text = ctx + b.text; return JSON.stringify(b); } };
      // Deep scan: find first user-like text field (skip action/type/role/status fields)
      for (const key of Object.keys(b)) {
        if (typeof b[key] === 'string' && b[key].length > 10 && b[key].length < 10000 && !key.match(/id|token|key|uuid|model|org|action|type|role|status|mode|method/i)) {
          console.log('[CLS++] Fallback extract on key:', key);
          return { q: b[key], set: ctx => { b[key] = ctx + b[key]; return JSON.stringify(b); } };
        }
      }
      console.log('[CLS++] Body keys:', Object.keys(b).join(', '));
    } catch (e) { console.log('[CLS++] JSON parse failed:', e.message); }
    return null;
  }

  // Is this an LLM API call?
  function isLLM(url) {
    const u = url.toLowerCase();
    if (host.includes('chatgpt') || host.includes('openai')) {
      if (!u.includes('/backend-api/') || !u.includes('conversation')) return false;
      // Skip setup/validation endpoints — only inject into actual message sends
      if (u.includes('/init') || u.includes('/prepare')) return false;
      return true;
    }
    if (host.includes('claude')) return u.includes('/api/') && (u.includes('chat') || u.includes('completion') || u.includes('message'));
    if (host.includes('gemini')) return u.includes('batchexecute');
    return false;
  }

  function clsFetch(...args) {
    let url = '', method = 'GET';
    if (args[0] instanceof Request) { url = args[0].url; method = args[0].method; }
    else { url = String(args[0] || ''); method = (args[1] && args[1].method) || 'GET'; }

    if (method !== 'POST' || !isLLM(url)) return _fetch(...args);
    console.log('[CLS++] LLM API detected:', url.slice(0, 120));

    try {
      let body = null;
      if (args[0] instanceof Request) {
        // Request body — need async, fall through to async path
      } else {
        const raw = args[1] && args[1].body;
        if (typeof raw === 'string') body = raw;
      }

      if (body) {
        const ex = extract(body);
        if (ex) {
          writeOutbox(ex.q);
          const ctx = getCtx();
          if (ctx) {
            const nb = ex.set(ctx);
            args = [args[0], { ...(args[1] || {}), body: nb }];
            console.log('[CLS++] Memory injected');
          } else {
            console.log('[CLS++] No memories available');
          }
        } else {
          console.log('[CLS++] Could not extract user message from body');
        }
        return _fetch(...args);
      }
    } catch (e) { console.log('[CLS++] Sync injection error:', e.message); }

    // Async fallback for Request objects or non-string bodies
    return (async () => {
      try {
        let body = null;
        if (args[0] instanceof Request) {
          try { body = await args[0].clone().text(); } catch (e) { console.log('[CLS++] Request.text() failed:', e.message); }
        } else {
          const raw = args[1] && args[1].body;
          if (raw instanceof Blob) body = await raw.text();
          else if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) body = new TextDecoder().decode(raw);
        }
        if (body) {
          const ex = extract(body);
          if (ex) {
            writeOutbox(ex.q);
            const ctx = getCtx();
            if (ctx) {
              const nb = ex.set(ctx);
              args = args[0] instanceof Request ? [new Request(args[0], { body: nb })] : [args[0], { ...(args[1] || {}), body: nb }];
              console.log('[CLS++] Memory injected (async)');
            }
          }
        }
      } catch (e) { console.log('[CLS++] Async injection error:', e.message); }
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
