/**
 * CLS++ Try It Demo - Super simple: talk, switch, see memory persist
 */
(function () {
  const API_URL = 'https://clsplusplus-api.onrender.com';
  const NAMESPACE = 'demo-' + Math.random().toString(36).slice(2, 10);

  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const switchBtn = document.getElementById('switch-btn');
  const currentModel = document.getElementById('current-model');
  const demoHint = document.getElementById('demo-hint');
  const welcomeMsg = document.getElementById('welcome-msg');

  let isClaude = true;

  function addMsg(text, isUser, isPreserved) {
    const div = document.createElement('div');
    div.className = 'demo-msg demo-msg-' + (isUser ? 'user' : 'ai');
    if (isPreserved) div.classList.add('demo-msg-preserved');
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function setLoading(loading) {
    sendBtn.disabled = loading;
    switchBtn.disabled = loading;
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

  async function readMemory() {
    const res = await fetch(API_URL + '/v1/memory/read', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'everything we talked about',
        namespace: NAMESPACE,
        limit: 10,
      }),
    });
    if (!res.ok) throw new Error('Failed to read');
    return res.json();
  }

  async function onSend() {
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';
    addMsg(text, true);

    setLoading(true);
    try {
      await writeMemory(text);
      const friendly = isClaude
        ? "Got it! I'll remember that. Try switching to OpenAI now."
        : "I remember that too! See? Same memory, different AI.";
      addMsg(friendly, false);
      demoHint.textContent = 'Now click "Switch to OpenAI" (or back to Claude) — your words follow you!';
    } catch (e) {
      addMsg("Oops! API might be sleeping. Wait 30 seconds and try again.", false);
    }
    setLoading(false);
  }

  async function onSwitch() {
    setLoading(true);
    try {
      const data = await readMemory();
      const items = data.items || [];
      isClaude = !isClaude;

      currentModel.textContent = isClaude ? 'Claude' : 'OpenAI';
      currentModel.className = 'demo-model-badge ' + (isClaude ? 'claude' : 'openai');
      switchBtn.textContent = isClaude ? 'Switch to OpenAI' : 'Switch to Claude';

      if (items.length > 0) {
        const memories = items.map((i) => i.text).join(' • ');
        addMsg(`I remember: ${memories}. What else?`, false, true);
        demoHint.textContent = "That's CLS++ — your memory lives outside the AI. Switch anytime.";
      } else {
        addMsg("Hi! I'm " + (isClaude ? "Claude" : "OpenAI") + ". Type something and I'll remember it.", false);
        demoHint.textContent = 'Step 1: Type a message. Step 2: Click Switch. Step 3: See your memory preserved.';
      }
    } catch (e) {
      addMsg("API not ready. Try again in a minute.", false);
    }
    setLoading(false);
  }

  sendBtn.addEventListener('click', onSend);
  chatInput.addEventListener('keypress', (e) => e.key === 'Enter' && onSend());
  switchBtn.addEventListener('click', onSwitch);
})();
