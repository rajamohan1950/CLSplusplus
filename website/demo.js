/**
 * CLS++ Try It Demo - Real Claude | Gemini | OpenAI with shared memory
 * Tell one, ask another. Memory lives in CLS++, all three use real LLMs.
 */
(function () {
  function init() {
  const API_URL = (typeof window !== 'undefined' && window.CLS_API_URL) || 'https://clsplusplus-api.onrender.com';
  const NAMESPACE = 'demo-' + Math.random().toString(36).slice(2, 10);
  const FETCH_TIMEOUT_MS = 90000;  // Render cold start can take 60s

  const MODELS = ['claude', 'openai'];

  // Warm up API on page load and keep pinging (Render cold start ~60–90s)
  for (var w = 0; w < 5; w++) {
    (function (delay) {
      setTimeout(function () {
        fetch(API_URL + '/v1/demo/status', { method: 'GET' }).catch(function () {});
      }, delay);
    })(w * 30000);
  }

  function addMsg(container, text, isUser, isPreserved) {
    const div = document.createElement('div');
    div.className = 'demo-msg demo-msg-' + (isUser ? 'user' : 'ai');
    if (isPreserved) div.classList.add('demo-msg-preserved');
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  function setLoading(loading) {
    document.querySelectorAll('[data-send]').forEach((btn) => (btn.disabled = loading));
  }

  async function chatWithLLM(model, message) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    const res = await fetch(API_URL + '/v1/demo/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model,
        message: message,
        namespace: NAMESPACE,
      }),
      signal: ctrl.signal,
    });
    clearTimeout(to);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText || 'Request failed');
    }
    const data = await res.json();
    return data.reply || '';
  }

  function sleep(ms) {
    return new Promise(function (r) { setTimeout(r, ms); });
  }

  var lastSentByModel = {};

  async function onSend(model, messageOverride) {
    const input = document.querySelector(`[data-input="${model}"]`);
    const container = document.getElementById(`chat-${model}`);
    if (!container || !input) return;
    const text = (messageOverride !== undefined ? messageOverride : input.value.trim());
    if (!text) return;

    if (messageOverride === undefined) {
      input.value = '';
      addMsg(container, text, true);
    }
    lastSentByModel[model] = text;
    const loadingEl = document.createElement('div');
    loadingEl.className = 'demo-msg demo-msg-ai demo-msg-loading';
    loadingEl.textContent = '...';
    container.appendChild(loadingEl);
    container.scrollTop = container.scrollHeight;

    setLoading(true);
    var maxAttempts = 5;
    var retryDelayMs = 25000;

    for (var attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        if (attempt > 1) {
          loadingEl.textContent = 'Waking up API... attempt ' + attempt + '/' + maxAttempts + ' (retry in ' + (retryDelayMs / 1000) + 's)';
          await sleep(retryDelayMs);
        }
        var reply = await chatWithLLM(model, text);
        loadingEl.remove();
        addMsg(container, reply, false, true);
        setLoading(false);
        return;
      } catch (e) {
        if (e.name === 'AbortError') {
          loadingEl.remove();
          addMsg(container, 'Timed out. Click Send to retry.', false);
          setLoading(false);
          return;
        }
        if (e.message !== 'Failed to fetch' && !e.message.includes('NetworkError')) {
          loadingEl.remove();
          var msg = e.message || 'Request failed';
          if (msg.includes('API') || msg.includes('key') || msg.includes('env') || msg.includes('401')) {
            addMsg(container, msg + ' (Add CLS_ANTHROPIC_API_KEY, CLS_OPENAI_API_KEY in Render → Environment)', false);
          } else {
            addMsg(container, 'Error: ' + msg, false);
          }
          setLoading(false);
          return;
        }
      }
    }

    loadingEl.remove();
    var errEl = document.createElement('div');
    errEl.className = 'demo-msg demo-msg-ai';
    errEl.innerHTML = 'API still unreachable after ' + maxAttempts + ' attempts. <button type="button" class="btn btn-primary" style="margin-top:8px">Retry</button>';
    errEl.querySelector('button').onclick = function () { onSend(model, lastSentByModel[model]); };
    container.appendChild(errEl);
    container.scrollTop = container.scrollHeight;
    setLoading(false);
  }

  MODELS.forEach((model) => {
    const container = document.getElementById(`chat-${model}`);
    if (container) addMsg(container, "Tell me something or ask me anything. Memory is shared.", false);
  });

  MODELS.forEach((model) => {
    const btn = document.querySelector(`[data-send="${model}"]`);
    const input = document.querySelector(`[data-input="${model}"]`);
    if (btn) btn.addEventListener('click', (e) => { e.preventDefault(); onSend(model); });
    if (input) input.addEventListener('keypress', (e) => { if (e.key === 'Enter') { e.preventDefault(); onSend(model); } });
  });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
