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

  async function onSend(model) {
    const input = document.querySelector(`[data-input="${model}"]`);
    const container = document.getElementById(`chat-${model}`);
    if (!container || !input) return;
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    addMsg(container, text, true);
    const loadingEl = document.createElement('div');
    loadingEl.className = 'demo-msg demo-msg-ai demo-msg-loading';
    loadingEl.textContent = '...';
    container.appendChild(loadingEl);
    container.scrollTop = container.scrollHeight;

    setLoading(true);
    try {
      const reply = await chatWithLLM(model, text);
      loadingEl.remove();
      addMsg(container, reply, false, true);
    } catch (e) {
      loadingEl.remove();
      const msg = e.message || 'Request failed';
      if (e.name === 'AbortError') {
        addMsg(container, 'Timed out. API may be cold-starting (Render free tier). Wait 1–2 min, then try again.', false);
      } else if (msg === 'Failed to fetch' || msg.includes('NetworkError')) {
        addMsg(container, 'API unreachable (cold start or CORS). Wait 1–2 min and retry, or try locally with ?local=1', false);
      } else if (msg.includes('API') || msg.includes('key') || msg.includes('env') || msg.includes('401')) {
        addMsg(container, msg + ' (Add CLS_ANTHROPIC_API_KEY, CLS_OPENAI_API_KEY in Render → Environment)', false);
      } else {
        addMsg(container, 'Error: ' + msg, false);
      }
    }
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
