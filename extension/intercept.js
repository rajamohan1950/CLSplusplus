// CLS++ Fetch Interceptor — MAIN world, document_start
// Locks window.fetch so no page code can replace it.
// On LLM API POST: reads prefetched memories from DOM, injects into body.
// Core injection path restored from v5.1.0 (proven working).
// Added: Gemini batchexecute support, pause toggle, injection signal.

(function () {
  'use strict';
  console.log('[CLS++] intercept.js loading on', location.hostname);
  if (window.__clspp) { console.log('[CLS++] already loaded, skipping'); return; }
  window.__clspp = true;

  const _fetch = window.fetch.bind(window);
  const host = location.hostname;

  // Check if injection is paused (set by capture.js via DOM attribute)
  function isPaused() {
    return document.body && document.body.getAttribute('data-cls-paused') === 'true';
  }

  // Signal that injection happened (capture.js picks this up and increments counter)
  function signalInjection() {
    try { document.body.setAttribute('data-cls-injected', '1'); } catch (_) {}
  }

  // Read memories from DOM mailbox (written by capture.js in ISOLATED world)
  // PROVEN PATH from v5.1.0 — synchronous DOM read, always works
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

  // Extract user message from Gemini's URL-encoded batchexecute body
  function extractGemini(body) {
    try {
      const params = new URLSearchParams(body);
      const fReq = params.get('f.req');
      if (!fReq) { console.log('[CLS++] Gemini: no f.req param'); return null; }
      const outer = JSON.parse(fReq);
      let innerStr = null;
      if (Array.isArray(outer) && outer[1] && typeof outer[1] === 'string') {
        innerStr = outer[1];
      } else if (Array.isArray(outer) && Array.isArray(outer[0]) && Array.isArray(outer[0][0])) {
        const rpc = outer[0][0];
        if (rpc[1] && typeof rpc[1] === 'string') innerStr = rpc[1];
      }
      if (!innerStr) { console.log('[CLS++] Gemini: could not find inner JSON'); return null; }
      const inner = JSON.parse(innerStr);
      if (Array.isArray(inner) && Array.isArray(inner[0]) && typeof inner[0][0] === 'string' && inner[0][0].length > 0) {
        const msg = inner[0][0];
        console.log('[CLS++] Gemini extract:', msg.slice(0, 80));
        return {
          q: msg,
          set: ctx => {
            inner[0][0] = ctx + inner[0][0];
            if (Array.isArray(outer) && typeof outer[1] === 'string') {
              outer[1] = JSON.stringify(inner);
            } else {
              outer[0][0][1] = JSON.stringify(inner);
            }
            params.set('f.req', JSON.stringify(outer));
            return params.toString();
          }
        };
      }
      console.log('[CLS++] Gemini: inner[0][0] not a string');
    } catch (e) { console.log('[CLS++] Gemini extract error:', e.message); }
    return null;
  }

  // Extract user message from JSON body (ChatGPT, Claude)
  function extract(body) {
    // Gemini sends URL-encoded form data, not JSON
    if (host.includes('gemini')) return extractGemini(body);
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
      if (u.includes('/init') || u.includes('/prepare')) return false;
      return true;
    }
    if (host.includes('claude')) return u.includes('/api/') && (u.includes('chat') || u.includes('completion') || u.includes('message'));
    if (host.includes('gemini')) return u.includes('batchexecute');
    return false;
  }

  // PROVEN injection path from v5.1.0 — sync first, async fallback
  // ALL fetch calls pass through here — log everything for debugging
  function clsFetch(...args) {
    let url = '', method = 'GET';
    if (args[0] instanceof Request) { url = args[0].url; method = args[0].method; }
    else { url = String(args[0] || ''); method = (args[1] && args[1].method) || 'GET'; }

    if (method !== 'POST' || !isLLM(url)) return _fetch(...args);
    console.log('[CLS++] LLM API detected:', url.slice(0, 120));
    console.log('[CLS++] arg[0] type:', args[0] instanceof Request ? 'Request' : typeof args[0],
                '| body type:', args[1] && args[1].body ? typeof args[1].body : (args[0] instanceof Request ? 'in-Request' : 'none'));

    if (isPaused()) {
      console.log('[CLS++] Injection paused by user');
      return _fetch(...args);
    }

    try {
      let body = null;
      if (args[0] instanceof Request) {
        console.log('[CLS++] Request object — going async path');
        // Request body — need async, fall through to async path
      } else {
        const raw = args[1] && args[1].body;
        if (typeof raw === 'string') {
          body = raw;
          console.log('[CLS++] Got string body, length:', body.length);
        } else if (raw) {
          console.log('[CLS++] Non-string body type:', Object.prototype.toString.call(raw));
        }
      }

      if (body) {
        const ex = extract(body);
        if (ex) {
          console.log('[CLS++] Extracted message:', ex.q.slice(0, 80));
          writeOutbox(ex.q);
          const ctx = getCtx();
          if (ctx) {
            const nb = ex.set(ctx);
            args = [args[0], { ...(args[1] || {}), body: nb }];
            console.log('[CLS++] Memory injected');
            signalInjection();
          } else {
            console.log('[CLS++] No memories available (DOM empty or missing)');
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
          try {
            body = await args[0].clone().text();
            console.log('[CLS++] Async: read Request body, length:', body ? body.length : 0);
          } catch (e) { console.log('[CLS++] Request.text() failed:', e.message); }
        } else {
          const raw = args[1] && args[1].body;
          if (raw instanceof Blob) { body = await raw.text(); console.log('[CLS++] Async: read Blob body'); }
          else if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) { body = new TextDecoder().decode(raw); console.log('[CLS++] Async: read ArrayBuffer body'); }
          else if (raw && typeof raw.getReader === 'function') {
            // ReadableStream — read it
            try {
              const reader = raw.getReader();
              const chunks = [];
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
              }
              body = new TextDecoder().decode(new Uint8Array(chunks.reduce((a, c) => [...a, ...c], [])));
              console.log('[CLS++] Async: read ReadableStream body, length:', body.length);
            } catch (e) { console.log('[CLS++] ReadableStream read failed:', e.message); }
          }
        }
        if (body) {
          const ex = extract(body);
          if (ex) {
            console.log('[CLS++] Async extracted:', ex.q.slice(0, 80));
            writeOutbox(ex.q);
            const ctx = getCtx();
            if (ctx) {
              const nb = ex.set(ctx);
              args = args[0] instanceof Request ? [new Request(args[0], { body: nb })] : [args[0], { ...(args[1] || {}), body: nb }];
              console.log('[CLS++] Memory injected (async)');
              signalInjection();
            }
          }
        } else {
          console.log('[CLS++] Async: no body could be read');
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

  // Gemini may use XMLHttpRequest instead of fetch for batchexecute
  if (host.includes('gemini')) {
    const _xhrOpen = XMLHttpRequest.prototype.open;
    const _xhrSend = XMLHttpRequest.prototype.send;

    XMLHttpRequest.prototype.open = function (method, url, ...rest) {
      this.__clsMethod = method;
      this.__clsUrl = String(url || '');
      return _xhrOpen.call(this, method, url, ...rest);
    };

    XMLHttpRequest.prototype.send = function (body) {
      if (this.__clsMethod === 'POST' && this.__clsUrl.includes('batchexecute') && typeof body === 'string' && !isPaused()) {
        console.log('[CLS++] Gemini XHR detected:', this.__clsUrl.slice(0, 120));
        const ex = extract(body);
        if (ex) {
          writeOutbox(ex.q);
          const ctx = getCtx();
          if (ctx) {
            body = ex.set(ctx);
            console.log('[CLS++] Memory injected (XHR)');
            signalInjection();
          }
        }
      }
      return _xhrSend.call(this, body);
    };
    console.log('[CLS++] XHR interceptor installed for Gemini');
  }
})();
