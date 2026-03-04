/**
 * CLS++ Try It Demo - Claude | Gemini | OpenAI side-by-side, shared memory
 * Tell one, ask another. Memory lives in CLS++, not in the model.
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

  async function writeMemory(text) {
    const res = await fetch(API_URL + '/v1/memory/write', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        namespace: NAMESPACE,
        source: 'demo',
        salience: 0.8,
      }),
    });
    if (!res.ok) throw new Error('Failed to save');
    return res.json();
  }

  async function readMemory(query) {
    const res = await fetch(API_URL + '/v1/memory/read', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query || 'everything we talked about',
        namespace: NAMESPACE,
        limit: 10,
      }),
    });
    if (!res.ok) throw new Error('Failed to read');
    return res.json();
  }

  function isQuestion(text) {
    const t = text.trim().toLowerCase();
    return t.includes('?') || /^(what|who|where|when|how|which|is my|do you remember)/.test(t);
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
      if (isQuestion(text)) {
        const data = await readMemory(text);
        const items = data.items || [];
        if (items.length > 0) {
          const answer = items.map((i) => i.text).join('. ');
          addMsg(container, answer, false, true);
        } else {
          addMsg(container, "I don't have that yet. Tell one of us first.", false);
        }
      } else {
        await writeMemory(text);
        addMsg(container, "Got it! Ask any of us — we all remember.", false);
      }
    } catch (e) {
      addMsg(container, "API sleeping. Wait ~30 sec and try again.", false);
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
    if (btn) btn.addEventListener('click', () => onSend(model));
    if (input) input.addEventListener('keypress', (e) => e.key === 'Enter' && onSend(model));
  });
})();
