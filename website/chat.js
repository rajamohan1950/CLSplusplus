/**
 * CLS++ Chat — Multi-session chat with persistent memory and automatic LLM routing.
 * User just types. Backend handles everything: memory, LLM selection, failover.
 */
(function () {
  const API_URL = (typeof window !== 'undefined' && window.CLS_API_URL) || 'https://clsplusplus-api.onrender.com';

  let activeSessionId = null;
  let sessions = []; // [{session_id, name}]
  let sending = false;

  // =========================================================================
  // DOM refs
  // =========================================================================
  const sessionListEl = document.getElementById('session-list');
  const messagesEl = document.getElementById('chat-messages');
  const welcomeEl = document.getElementById('chat-welcome');
  const inputEl = document.getElementById('chat-input');
  const sendBtn = document.getElementById('btn-send');
  const newChatBtn = document.getElementById('btn-new-chat');
  const headerName = document.getElementById('chat-header-name');
  const menuBtn = document.getElementById('btn-menu');
  const sidebar = document.getElementById('chat-sidebar');
  const debugAugmented = document.getElementById('debug-augmented');
  const debugMemory = document.getElementById('debug-memory');

  // =========================================================================
  // API helpers
  // =========================================================================
  async function api(path, method, body) {
    var opts = { method: method || 'GET', headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    var res = await fetch(API_URL + path, opts);
    if (!res.ok) {
      var err = await res.json().catch(function () { return {}; });
      throw new Error(err.detail || err.message || res.statusText);
    }
    return res.json();
  }

  // =========================================================================
  // localStorage sync
  // =========================================================================
  function saveSessions() {
    localStorage.setItem('cls_chat_sessions', JSON.stringify(sessions));
    localStorage.setItem('cls_chat_active', activeSessionId || '');
  }

  function loadSessions() {
    try {
      sessions = JSON.parse(localStorage.getItem('cls_chat_sessions') || '[]');
      activeSessionId = localStorage.getItem('cls_chat_active') || null;
    } catch (e) {
      sessions = [];
      activeSessionId = null;
    }
  }

  // =========================================================================
  // Session list rendering
  // =========================================================================
  function renderSessionList() {
    sessionListEl.innerHTML = '';
    sessions.forEach(function (s) {
      var item = document.createElement('div');
      item.className = 'chat-session-item' + (s.session_id === activeSessionId ? ' active' : '');
      item.dataset.id = s.session_id;

      var nameSpan = document.createElement('span');
      nameSpan.className = 'chat-session-name';
      nameSpan.textContent = s.name;

      var delBtn = document.createElement('button');
      delBtn.className = 'chat-session-del';
      delBtn.textContent = '\u00d7';
      delBtn.title = 'Delete';
      delBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        deleteSession(s.session_id);
      });

      item.appendChild(nameSpan);
      item.appendChild(delBtn);
      item.addEventListener('click', function () {
        switchSession(s.session_id);
      });
      sessionListEl.appendChild(item);
    });
  }

  // =========================================================================
  // Message rendering
  // =========================================================================
  function renderMessages(messages) {
    // Clear except welcome
    messagesEl.innerHTML = '';
    if (!messages || messages.length === 0) {
      messagesEl.appendChild(welcomeEl.cloneNode(true));
      return;
    }

    messages.forEach(function (msg) {
      appendMessage(msg.role, msg.content, msg.memory_used, msg.memory_count);
    });
    scrollToBottom();
  }

  function appendMessage(role, content, memoryUsed, memoryCount) {
    // Hide welcome on first message
    var welcome = messagesEl.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    var row = document.createElement('div');
    row.className = 'chat-msg-row chat-msg-' + role;

    var bubble = document.createElement('div');
    bubble.className = 'chat-msg-bubble';
    bubble.textContent = content;

    row.appendChild(bubble);

    if (role === 'assistant' && memoryUsed) {
      var dot = document.createElement('span');
      dot.className = 'chat-memory-dot';
      dot.title = memoryCount + ' memory items used';
      row.appendChild(dot);
    }

    messagesEl.appendChild(row);
  }

  function showTyping() {
    var row = document.createElement('div');
    row.className = 'chat-msg-row chat-msg-assistant';
    row.id = 'typing-indicator';

    var bubble = document.createElement('div');
    bubble.className = 'chat-msg-bubble chat-typing';
    bubble.innerHTML = '<span class="chat-dot"></span><span class="chat-dot"></span><span class="chat-dot"></span>';

    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function hideTyping() {
    var el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // =========================================================================
  // Session actions
  // =========================================================================
  async function createSession() {
    try {
      var data = await api('/v1/chat/sessions', 'POST', {});
      sessions.unshift({ session_id: data.session_id, name: data.name });
      activeSessionId = data.session_id;
      saveSessions();
      renderSessionList();
      renderMessages([]);
      headerName.textContent = data.name;
      inputEl.focus();
    } catch (e) {
      console.error('Failed to create session:', e);
    }
  }

  async function switchSession(sessionId) {
    activeSessionId = sessionId;
    saveSessions();
    renderSessionList();

    var s = sessions.find(function (x) { return x.session_id === sessionId; });
    headerName.textContent = s ? s.name : 'Chat';

    try {
      var data = await api('/v1/chat/sessions/' + sessionId, 'GET');
      renderMessages(data.messages);
    } catch (e) {
      renderMessages([]);
    }
  }

  async function deleteSession(sessionId) {
    try {
      await api('/v1/chat/sessions/' + sessionId, 'DELETE');
    } catch (e) {
      // Ignore — may already be gone on server restart
    }
    sessions = sessions.filter(function (s) { return s.session_id !== sessionId; });
    if (activeSessionId === sessionId) {
      if (sessions.length > 0) {
        switchSession(sessions[0].session_id);
      } else {
        activeSessionId = null;
        saveSessions();
        createSession();
      }
    } else {
      saveSessions();
      renderSessionList();
    }
  }

  // =========================================================================
  // Debug panel
  // =========================================================================
  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function updateDebugPanel(data) {
    if (!data || !data.debug) return;
    var d = data.debug;

    // Upper box: augmented prompt + model used + LLM response
    var html = '';
    html += '<div class="debug-label">Model Used</div>';
    html += '<div class="debug-value debug-model">' + escHtml(d.model_used) + '</div>';

    html += '<div class="debug-label">User Message</div>';
    html += '<div class="debug-value">' + escHtml(d.user_message) + '</div>';

    html += '<div class="debug-label">Memory Searched (' + (d.memory_searched ? d.memory_searched.length : 0) + ' hits)</div>';
    if (d.memory_searched && d.memory_searched.length > 0) {
      html += '<div class="debug-value debug-list">';
      d.memory_searched.forEach(function (t) { html += '<div class="debug-item">' + escHtml(t) + '</div>'; });
      html += '</div>';
    } else {
      html += '<div class="debug-value debug-empty">No memory matches</div>';
    }

    html += '<div class="debug-label">Conversation History Lines: ' + d.conversation_history_lines + '</div>';

    html += '<div class="debug-label">Full Augmented System Prompt</div>';
    html += '<pre class="debug-prompt">' + escHtml(d.augmented_prompt) + '</pre>';

    html += '<div class="debug-label">LLM Response</div>';
    html += '<div class="debug-value debug-response">' + escHtml(data.reply) + '</div>';

    debugAugmented.innerHTML = html;

    // Lower box: Phase Memory Dynamics — thermodynamic state
    var memHtml = '';
    var pd = d.phase_dynamics;
    if (pd) {
      // Global thermodynamic parameters
      memHtml += '<div class="debug-phase-global">';
      memHtml += '<span class="debug-phase-param">\u03C1=' + pd.memory_density_rho.toFixed(4) + '</span>';
      memHtml += '<span class="debug-phase-param">events=' + pd.global_event_counter + '</span>';
      memHtml += '<span class="debug-phase-param">\u03A3F=' + pd.total_free_energy.toFixed(2) + '</span>';
      memHtml += '<span class="debug-phase-param">\u03C4\u2081=' + pd.tau_c1 + '</span>';
      memHtml += '<span class="debug-phase-param">liquid=' + pd.liquid_count + '</span>';
      memHtml += '<span class="debug-phase-param">gas=' + pd.gas_count + '</span>';
      memHtml += '</div>';

      // Per-item thermodynamic state with consolidation strength bars
      if (pd.items && pd.items.length > 0) {
        pd.items.forEach(function (item, i) {
          var pct = Math.round(item.consolidation_strength * 100);
          var color = pct > 50 ? '#4caf50' : pct > 20 ? '#ff9800' : '#f44336';
          var phaseTag = item.phase === 'liquid'
            ? '<span class="debug-phase-tag debug-phase-liquid">liquid</span>'
            : '<span class="debug-phase-tag debug-phase-gas">gas</span>';

          memHtml += '<div class="debug-mem-item">';
          // Strength bar
          memHtml += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">';
          memHtml += '<div style="width:80px;height:8px;background:#222;border-radius:4px;overflow:hidden;flex-shrink:0">';
          memHtml += '<div style="width:' + pct + '%;height:100%;background:' + color + ';border-radius:4px"></div></div>';
          memHtml += '<span style="font-size:11px;color:' + color + ';min-width:32px">' + pct + '%</span>';
          memHtml += phaseTag;
          memHtml += '</div>';
          // Fact text
          memHtml += '<div class="debug-mem-text">' + escHtml(item.text) + '</div>';
          // Thermodynamic parameters
          memHtml += '<div class="debug-phase-meta">';
          memHtml += 's=' + item.consolidation_strength.toFixed(3);
          memHtml += ' | \u03A3=' + item.surprise_at_birth.toFixed(3);
          memHtml += ' | \u03C4=' + item.tau.toFixed(1);
          memHtml += ' | F=' + item.free_energy.toFixed(3);
          memHtml += ' | H=' + item.information_content_bits.toFixed(2) + 'b';
          memHtml += ' | L=' + item.landauer_cost.toFixed(4);
          memHtml += ' | R=' + item.retrieval_count;
          memHtml += '</div>';
          // Fact structure
          if (item.fact) {
            memHtml += '<div class="debug-phase-fact">';
            memHtml += '(' + escHtml(item.fact.subject) + ', ' + escHtml(item.fact.relation) + ', ' + escHtml(item.fact.value) + ')';
            if (item.fact.override) memHtml += ' <span style="color:#f44336;font-weight:600">OVERRIDE</span>';
            memHtml += '</div>';
          }
          memHtml += '</div>';
        });
      } else {
        memHtml += '<span class="debug-placeholder">No memories stored yet</span>';
      }
    } else if (d.memory_store && d.memory_store.length > 0) {
      // Fallback to legacy format
      memHtml += '<div class="debug-label">Total items: ' + d.memory_store.length + '</div>';
      d.memory_store.forEach(function (item, i) {
        memHtml += '<div class="debug-mem-item">#' + (i+1) + ' ' + escHtml(item.text || '') + '</div>';
      });
    } else {
      memHtml = '<span class="debug-placeholder">No memories stored yet</span>';
    }
    debugMemory.innerHTML = memHtml;

    // Auto-scroll debug panels to bottom
    debugAugmented.scrollTop = debugAugmented.scrollHeight;
    debugMemory.scrollTop = debugMemory.scrollHeight;
  }

  // =========================================================================
  // Send message
  // =========================================================================
  async function sendMessage() {
    if (sending) return;
    var text = inputEl.value.trim();
    if (!text) return;
    if (!activeSessionId) return;

    sending = true;
    inputEl.value = '';
    autoGrow();
    sendBtn.disabled = true;

    // Optimistic: show user message immediately
    appendMessage('user', text);
    scrollToBottom();
    showTyping();

    try {
      var data = await api('/v1/chat/sessions/' + activeSessionId + '/message', 'POST', { message: text });
      hideTyping();
      appendMessage('assistant', data.reply, data.memory_used, data.memory_count);
      scrollToBottom();
      updateDebugPanel(data);

      // Auto-rename session after first exchange
      var s = sessions.find(function (x) { return x.session_id === activeSessionId; });
      if (s && s.name.startsWith('Chat ') && s.name.length <= 10) {
        // Use first 30 chars of user's first message as name
        var newName = text.length > 30 ? text.substring(0, 30) + '...' : text;
        s.name = newName;
        saveSessions();
        renderSessionList();
        headerName.textContent = newName;
      }
    } catch (e) {
      hideTyping();
      appendMessage('assistant', 'Error: ' + e.message);
      scrollToBottom();
    } finally {
      sending = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  // =========================================================================
  // Input auto-grow
  // =========================================================================
  function autoGrow() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  }

  // =========================================================================
  // Event listeners
  // =========================================================================
  sendBtn.addEventListener('click', sendMessage);

  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputEl.addEventListener('input', autoGrow);

  newChatBtn.addEventListener('click', createSession);

  // Mobile sidebar toggle
  menuBtn.addEventListener('click', function () {
    sidebar.classList.toggle('open');
  });

  // Close sidebar on session click (mobile)
  sessionListEl.addEventListener('click', function () {
    if (window.innerWidth <= 768) {
      sidebar.classList.remove('open');
    }
  });

  // =========================================================================
  // Init
  // =========================================================================
  async function init() {
    loadSessions();

    if (sessions.length === 0) {
      // First visit: auto-create a session
      await createSession();
    } else {
      renderSessionList();
      // Load active session or first one
      var targetId = activeSessionId || sessions[0].session_id;
      await switchSession(targetId);
    }
  }

  init();
})();
