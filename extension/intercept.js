// CLS++ Fetch Interceptor — runs in MAIN world at document_start
// Hooks window.fetch BEFORE any page code can capture a reference.
// When the page sends a message to the LLM API, this intercept:
// 1. Extracts the user's query from the request body
// 2. Requests memory context via CustomEvent bridge (CSP-safe)
// 3. Prepends the memory context to the user's message
// The user sees NOTHING — injection happens at the API payload level.

(function () {
  'use strict';
  if (window.__clspp_intercepted) return;
  window.__clspp_intercepted = true;

  const _origFetch = window.fetch;

  // Detect which site we're on
  const host = location.hostname;
  let site = 'unknown';
  if (host.includes('chatgpt') || host.includes('chat.openai')) site = 'chatgpt';
  else if (host.includes('claude.ai')) site = 'claude';
  else if (host.includes('gemini.google')) site = 'gemini';
  // Local E2E harness (prototype/extension-e2e.html + extension/e2e/)
  else if (
    (host === 'localhost' || host === '127.0.0.1') &&
    location.pathname.includes('extension-e2e')
  ) {
    site = 'e2e';
  }

  // ── Context bridge: request from content script (bypasses page CSP) ─────
  // MAIN world can't fetch localhost on sites with strict CSP (ChatGPT).
  // So we ask the ISOLATED world content script to fetch for us.
  let _ctxId = 0;
  const _ctxCallbacks = {};
  let _bridgeReady = false;

  window.addEventListener('__clspp_bridge_ready', () => { _bridgeReady = true; });
  window.addEventListener('__clspp_context_response', (e) => {
    const { id, context } = e.detail;
    if (_ctxCallbacks[id]) {
      _ctxCallbacks[id](context || '');
      delete _ctxCallbacks[id];
    }
  });

  // Wait for bridge to be ready (content_common.js loaded), then dispatch
  function _dispatchContextRequest(id, query) {
    window.dispatchEvent(new CustomEvent('__clspp_context_request', {
      detail: { id, query }
    }));
  }

  function getContext(query) {
    return new Promise((resolve) => {
      const id = ++_ctxId;
      const timer = setTimeout(() => {
        delete _ctxCallbacks[id];
        resolve('');
      }, 4000);
      _ctxCallbacks[id] = (ctx) => {
        clearTimeout(timer);
        resolve(ctx);
      };
      if (_bridgeReady) {
        _dispatchContextRequest(id, query);
      } else {
        // Bridge not ready yet — wait up to 2s for it, then dispatch
        const waitForBridge = () => {
          _dispatchContextRequest(id, query);
        };
        const bridgeListener = () => {
          clearTimeout(bridgeTimer);
          waitForBridge();
        };
        const bridgeTimer = setTimeout(waitForBridge, 2000);
        window.addEventListener('__clspp_bridge_ready', bridgeListener, { once: true });
      }
    });
  }

  // ── Normalize fetch arguments ──────────────────────────────────────────
  // Handles both fetch(url, opts) and fetch(Request) patterns
  async function normalizeFetchArgs(args) {
    let url = '';
    let method = 'GET';
    let body = null;

    if (args[0] instanceof Request) {
      const req = args[0];
      url = req.url;
      method = req.method;
      try {
        const clone = req.clone();
        body = await clone.text();
      } catch (e) {}
      if (args[1] && args[1].body) {
        body = typeof args[1].body === 'string' ? args[1].body : null;
        method = args[1].method || method;
      }
    } else {
      url = typeof args[0] === 'string' ? args[0] : '';
      const opts = args[1] || {};
      method = opts.method || 'GET';
      body = typeof opts.body === 'string' ? opts.body : null;
    }

    return { url, method, body };
  }

  // ── Rebuild fetch arguments with new body ─────────────────────────────
  function rebuildFetchArgs(args, newBody) {
    if (args[0] instanceof Request) {
      const orig = args[0];
      const newReq = new Request(orig, { body: newBody });
      return [newReq];
    } else {
      const opts = { ...(args[1] || {}), body: newBody };
      return [args[0], opts];
    }
  }

  // ── ChatGPT interceptor ──────────────────────────────────────────────────
  function isChatGPTConversation(url) {
    return (
      typeof url === 'string' &&
      (url.includes('/backend-api/conversation') ||
        url.includes('/backend-api/f/conversation'))
    );
  }

  function extractChatGPTQuery(body) {
    try {
      const b = JSON.parse(body);
      const msgs = b.messages;
      if (!msgs || !msgs.length) return null;
      const last = msgs[msgs.length - 1];
      const parts = last && last.content && last.content.parts;
      if (parts && typeof parts[0] === 'string' && parts[0].length > 3) {
        return { parsed: b, query: parts[0], parts };
      }
    } catch (e) {}
    return null;
  }

  // ── Claude interceptor ───────────────────────────────────────────────────
  function isClaudeConversation(url) {
    return (
      typeof url === 'string' &&
      url.includes('/api/') &&
      (url.includes('/chat_conversations/') || url.includes('/completion'))
    );
  }

  function extractClaudeQuery(body) {
    try {
      const b = JSON.parse(body);

      if (typeof b.prompt === 'string' && b.prompt.length > 3) {
        return { parsed: b, query: b.prompt, field: 'prompt' };
      }
      if (Array.isArray(b.content)) {
        const textBlock = b.content.find(c => c.type === 'text' && c.text);
        if (textBlock && textBlock.text.length > 3) {
          return { parsed: b, query: textBlock.text, field: 'content', block: textBlock };
        }
      }
      if (Array.isArray(b.messages) && b.messages.length > 0) {
        const last = b.messages[b.messages.length - 1];
        let q = '';
        if (typeof last === 'string') q = last;
        else if (typeof last.content === 'string') q = last.content;
        else if (Array.isArray(last.content)) {
          const tb = last.content.find(c => c.type === 'text' && c.text);
          if (tb) q = tb.text;
        }
        if (q && q.length > 3) return { parsed: b, query: q, field: 'messages', lastMsg: last };
      }
    } catch (e) {}
    return null;
  }

  function injectClaudeContext(ex, ctx) {
    const b = ex.parsed;
    if (ex.field === 'prompt') {
      b.prompt = ctx + '\n\n' + b.prompt;
    } else if (ex.field === 'content' && ex.block) {
      ex.block.text = ctx + '\n\n' + ex.block.text;
    } else if (ex.field === 'messages') {
      const last = ex.lastMsg;
      if (typeof last === 'string') {
        b.messages[b.messages.length - 1] = ctx + '\n\n' + last;
      } else if (typeof last.content === 'string') {
        last.content = ctx + '\n\n' + last.content;
      } else if (Array.isArray(last.content)) {
        const tb = last.content.find(c => c.type === 'text' && c.text);
        if (tb) tb.text = ctx + '\n\n' + tb.text;
      }
    }
    return JSON.stringify(b);
  }

  // ── Gemini interceptor ───────────────────────────────────────────────────
  function isGeminiConversation(url) {
    return (
      typeof url === 'string' &&
      (url.includes('batchexecute') || url.includes('BardFrontendService'))
    );
  }

  function extractGeminiQuery(body) {
    if (typeof body !== 'string' || body.length < 20) return null;
    const matches = body.match(/\[\["([^"]{6,})"/g);
    if (matches) {
      for (const m of matches) {
        const inner = m.match(/\[\["([^"]{6,})"/);
        if (inner && inner[1].length > 5 && !/^[a-zA-Z0-9_]+$/.test(inner[1])) {
          return { query: inner[1], body };
        }
      }
    }
    const alt = body.match(/"([^"]{10,200})"/g);
    if (alt) {
      for (const a of alt) {
        const t = a.slice(1, -1);
        if (t.includes(' ') && !/^(at|en|boq|http|rpc|source)/.test(t)) {
          return { query: t, body };
        }
      }
    }
    return null;
  }

  // ── Main fetch hook ──────────────────────────────────────────────────────
  window.fetch = async function (...args) {
    const { url: urlStr, method, body } = await normalizeFetchArgs(args);

    if (method === 'POST' && body) {
      try {
        // ChatGPT (+ local E2E uses same payload shape)
        if ((site === 'chatgpt' || site === 'e2e') && isChatGPTConversation(urlStr)) {
          const ex = extractChatGPTQuery(body);
          if (ex) {
            const ctx = await getContext(ex.query);
            if (ctx) {
              ex.parts[0] = ctx + '\n\n' + ex.query;
              args = rebuildFetchArgs(args, JSON.stringify(ex.parsed));
              console.log('[CLS++] context injected into ChatGPT');
            }
          }
        }

        // Claude (+ local E2E)
        if ((site === 'claude' || site === 'e2e') && isClaudeConversation(urlStr)) {
          const ex = extractClaudeQuery(body);
          if (ex) {
            const ctx = await getContext(ex.query);
            if (ctx) {
              const newBody = injectClaudeContext(ex, ctx);
              args = rebuildFetchArgs(args, newBody);
              console.log('[CLS++] context injected into Claude (' + ex.field + ')');
            }
          }
        }

        // Gemini
        if (site === 'gemini' && isGeminiConversation(urlStr)) {
          const ex = extractGeminiQuery(body);
          if (ex) {
            const ctx = await getContext(ex.query);
            if (ctx) {
              const newBody = ex.body.replace(ex.query, ctx + '\n\n' + ex.query);
              args = rebuildFetchArgs(args, newBody);
              console.log('[CLS++] context injected into Gemini');
            }
          }
        }
      } catch (e) {
        console.log('[CLS++] intercept error:', e);
      }
    }

    return _origFetch.apply(this, args);
  };

  console.log('[CLS++] fetch interceptor active on', site);
})();
