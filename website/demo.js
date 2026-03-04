/**
 * CLS++ Try It Demo - Real Claude | Gemini | OpenAI with shared memory
 * Tell one, ask another. Memory lives in CLS++, all three use real LLMs.
 */
(function () {
  const API_URL = 'https://clsplusplus-api.onrender.com';
  const NAMESPACE = 'demo-' + Math.random().toString(36).slice(2, 10);

  const MODELS = ['claude', 'gemini', 'openai'];

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
    const res = await fetch(API_URL + '/v1/demo/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model,
        message: message,
        namespace: NAMESPACE,
      }),
    });
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
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    addMsg(container, text, true);

    setLoading(true);
    try {
      const reply = await chatWithLLM(model, text);
      addMsg(container, reply, false, true);
    } catch (e) {
      const msg = e.message || 'Request failed';
      if (msg.includes('API') || msg.includes('key') || msg.includes('env')) {
        addMsg(container, msg + ' (Add keys in Render env.)', false);
      } else {
        addMsg(container, 'Error: ' + msg, false);
      }
    }
    setLoading(false);
  }

  MODELS.forEach((model) => {
    const container = document.getElementById(`chat-${model}`);
    if (container) addMsg(container, "Tell me something or ask me anything. Memory is shared across all three.", false);
  });

  MODELS.forEach((model) => {
    const btn = document.querySelector(`[data-send="${model}"]`);
    const input = document.querySelector(`[data-input="${model}"]`);
    if (btn) btn.addEventListener('click', () => onSend(model));
    if (input) input.addEventListener('keypress', (e) => e.key === 'Enter' && onSend(model));
  });
})();
