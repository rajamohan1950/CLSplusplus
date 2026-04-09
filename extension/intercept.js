// CLS++ Fetch Interceptor v3 — Hardened
// Runs in MAIN world at document_start.
// Hooks window.fetch AND XMLHttpRequest for maximum coverage.
// Visual indicator shows when memory is injected.

(function () {
  'use strict';
  if (window.__clspp_intercepted) return;
  window.__clspp_intercepted = true;

  const _origFetch = window.fetch;

  // Detect site
  const host = location.hostname;
  let site = 'unknown';
  if (host.includes('chatgpt') || host.includes('chat.openai')) site = 'chatgpt';
  else if (host.includes('claude.ai')) site = 'claude';
  else if (host.includes('gemini.google')) site = 'gemini';
  else if ((host === 'localhost' || host === '127.0.0.1') &&
    (location.pathname.includes('extension-e2e') || location.pathname.includes('chat-test'))) {
    site = 'e2e';
  }

  // ── Visual injection indicator ─────────────────────────────────────────
  // Shows a small green banner when memory is successfully injected
  function showInjectionBanner(count) {
    let banner = document.getElementById('__clspp_banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = '__clspp_banner';
      banner.style.cssText = 'position:fixed;top:8px;right:8px;z-index:999999;' +
        'background:rgba(16,24,16,0.95);border:1px solid rgba(93,224,197,0.4);' +
        'border-radius:8px;padding:6px 12px;font-size:12px;font-family:system-ui;' +
        'color:#5de0c5;pointer-events:none;transition:opacity 0.3s;opacity:0;' +
        'box-shadow:0 4px 12px rgba(0,0,0,0.3);display:flex;align-items:center;gap:6px;';
      document.body.appendChild(banner);
    }
    banner.innerHTML = '<span style="font-size:14px">🧠</span> CLS++ injected ' + count + ' memories';
    banner.style.opacity = '1';
    clearTimeout(banner._timer);
    banner._timer = setTimeout(() => { banner.style.opacity = '0'; }, 3000);
  }

  // ── Context bridge ─────────────────────────────────────────────────────
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
      // Shorter timeout: 3s. Don't delay user's message too long.
      const timer = setTimeout(() => {
        delete _ctxCallbacks[id];
        console.log('[CLS++] Context bridge timeout — sending without memory');
        resolve('');
      }, 3000);
      _ctxCallbacks[id] = (ctx) => {
        clearTimeout(timer);
        resolve(ctx);
      };
      const dispatch = () => {
        window.dispatchEvent(new CustomEvent('__clspp_context_request', {
          detail: { id, query }
        }));
      };
      if (_bridgeReady) {
        dispatch();
      } else {
        const bridgeTimer = setTimeout(dispatch, 1500);
        window.addEventListener('__clspp_bridge_ready', () => {
          clearTimeout(bridgeTimer);
          dispatch();
        }, { once: true });
      }
    });
  }

  // ── Normalize fetch args ───────────────────────────────────────────────
  async function normalizeFetchArgs(args) {
    let url = '', method = 'GET', body = null;
    if (args[0] instanceof Request) {
      const req = args[0];
      url = req.url;
      method = req.method;
      try { body = await req.clone().text(); } catch (e) {}
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

  function rebuildFetchArgs(args, newBody) {
    if (args[0] instanceof Request) {
      return [new Request(args[0], { body: newBody })];
    }
    return [args[0], { ...(args[1] || {}), body: newBody }];
  }

  // ── URL matchers — broadened for 2026 API changes ─────────────────────

  function isChatGPTUrl(url) {
    return typeof url === 'string' && (
      url.includes('/backend-api/conversation') ||
      url.includes('/backend-api/f/conversation') ||
      url.includes('/backend-api/ce/conversation') ||
      // Catch any future /backend-api/ conversation variants
      (url.includes('/backend-api/') && url.includes('conversation'))
    );
  }

  function isClaudeUrl(url) {
    return typeof url === 'string' && url.includes('/api/') && (
      url.includes('/chat_conversations/') ||
      url.includes('/completion') ||
      url.includes('/messages') ||
      url.includes('/chat/') ||
      // Claude.ai uses organization-scoped endpoints
      url.includes('/organizations/') && (url.includes('/chat') || url.includes('/completion'))
    );
  }

  function isGeminiUrl(url) {
    return typeof url === 'string' && (
      url.includes('batchexecute') ||
      url.includes('BardFrontendService') ||
      url.includes('StreamGenerate')
    );
  }

  function isTargetUrl(rawUrl) {
    const u = typeof rawUrl === 'string' ? rawUrl : (rawUrl instanceof Request ? rawUrl.url : '');
    if (site === 'chatgpt') return isChatGPTUrl(u);
    if (site === 'claude') return isClaudeUrl(u);
    if (site === 'gemini') return isGeminiUrl(u);
    if (site === 'e2e') return isChatGPTUrl(u) || isClaudeUrl(u);
    return false;
  }

  // ── Query extractors ──────────────────────────────────────────────────

  function extractChatGPTQuery(body) {
    try {
      const b = JSON.parse(body);
      // Standard: messages[].content.parts[]
      const msgs = b.messages;
      if (msgs && msgs.length) {
        const last = msgs[msgs.length - 1];
        const parts = last && last.content && last.content.parts;
        if (parts && typeof parts[0] === 'string' && parts[0].length > 3) {
          return { parsed: b, query: parts[0], parts, type: 'parts' };
        }
        // Alternative: messages[].content as string
        if (last && typeof last.content === 'string' && last.content.length > 3) {
          return { parsed: b, query: last.content, msg: last, type: 'string' };
        }
      }
      // Fallback: prompt field
      if (typeof b.prompt === 'string' && b.prompt.length > 3) {
        return { parsed: b, query: b.prompt, type: 'prompt' };
      }
    } catch (e) {}
    return null;
  }

  function extractClaudeQuery(body) {
    try {
      const b = JSON.parse(body);
      // claude.ai: prompt field
      if (typeof b.prompt === 'string' && b.prompt.length > 3) {
        return { parsed: b, query: b.prompt, field: 'prompt' };
      }
      // claude.ai: content array with text blocks
      if (Array.isArray(b.content)) {
        const tb = b.content.find(c => c.type === 'text' && c.text);
        if (tb && tb.text.length > 3) {
          return { parsed: b, query: tb.text, field: 'content', block: tb };
        }
      }
      // API format: messages array
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

  // ── Injectors ─────────────────────────────────────────────────────────

  function injectChatGPT(ex, ctx) {
    if (ex.type === 'parts') {
      ex.parts[0] = ctx + '\n\n' + ex.query;
    } else if (ex.type === 'string') {
      ex.msg.content = ctx + '\n\n' + ex.query;
    } else if (ex.type === 'prompt') {
      ex.parsed.prompt = ctx + '\n\n' + ex.query;
    }
    return JSON.stringify(ex.parsed);
  }

  function injectClaude(ex, ctx) {
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

  // ── Count memories in context block ────────────────────────────────────
  function countMemories(ctx) {
    if (!ctx) return 0;
    return (ctx.match(/^- /gm) || []).length;
  }

  // ── Main fetch hook ────────────────────────────────────────────────────
  window.fetch = function (...args) {
    if (!isTargetUrl(args[0])) {
      return _origFetch.apply(this, args);
    }

    return (async () => {
      const { url: urlStr, method, body } = await normalizeFetchArgs(args);

      if (method === 'POST' && body) {
        try {
          let ctx = '';
          let injected = false;

          // ChatGPT
          if ((site === 'chatgpt' || site === 'e2e') && isChatGPTUrl(urlStr)) {
            const ex = extractChatGPTQuery(body);
            if (ex) {
              ctx = await getContext(ex.query);
              if (ctx) {
                args = rebuildFetchArgs(args, injectChatGPT(ex, ctx));
                injected = true;
              }
            }
          }

          // Claude
          if ((site === 'claude' || site === 'e2e') && isClaudeUrl(urlStr)) {
            const ex = extractClaudeQuery(body);
            if (ex) {
              ctx = await getContext(ex.query);
              if (ctx) {
                args = rebuildFetchArgs(args, injectClaude(ex, ctx));
                injected = true;
              }
            }
          }

          // Gemini
          if (site === 'gemini' && isGeminiUrl(urlStr)) {
            const ex = extractGeminiQuery(body);
            if (ex) {
              ctx = await getContext(ex.query);
              if (ctx) {
                args = rebuildFetchArgs(args, ex.body.replace(ex.query, ctx + '\n\n' + ex.query));
                injected = true;
              }
            }
          }

          if (injected) {
            const n = countMemories(ctx);
            showInjectionBanner(n);
            console.log('[CLS++] Memory injected into ' + site + ': ' + n + ' facts');
            window.dispatchEvent(new CustomEvent('__clspp_telemetry', {
              detail: { event: 'context_injected', site }
            }));
          }
        } catch (e) {
          console.error('[CLS++] Intercept error:', e);
        }
      }

      return _origFetch.apply(this, args);
    })();
  };

  // ── Also hook XMLHttpRequest for sites that don't use fetch ────────────
  const _origXHROpen = XMLHttpRequest.prototype.open;
  const _origXHRSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this._clsppUrl = url;
    this._clsppMethod = method;
    return _origXHROpen.call(this, method, url, ...rest);
  };

  XMLHttpRequest.prototype.send = function (body) {
    // Only intercept POST to target URLs
    if (this._clsppMethod === 'POST' && typeof body === 'string' && isTargetUrl(this._clsppUrl)) {
      console.log('[CLS++ XHR] Target POST detected:', (this._clsppUrl || '').slice(0, 80));
      // For XHR, we can't easily make it async.
      // Log it so we know it's happening, but delegate to fetch hook.
    }
    return _origXHRSend.call(this, body);
  };

  console.log('[CLS++] Interceptor v3 active on ' + site + ' (fetch + XHR)');
})();
