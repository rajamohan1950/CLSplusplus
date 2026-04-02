// Polyfill: Safari uses browser.* but also supports chrome.* compat
const api = typeof browser !== 'undefined' ? browser : chrome;

// Use local server when cls_local is set in storage, otherwise cloud
let API = 'https://clsplusplus.onrender.com';

// Initialize everything after loading storage state
api.storage.local.get(['cls_local', 'autoInject'], (r) => {
  if (!r) r = {};
  if (r.cls_local) API = 'http://localhost:8080';

  // Update "View My Memories" link (local prototype serves under /ui/)
  const memPath = r.cls_local ? '/ui/memory.html' : '/memory.html';
  document.getElementById('view-btn').href = `${API}${memPath}`;

  // Load local mode toggle
  document.getElementById('toggle-local').checked = !!r.cls_local;

  // Load auto-inject toggle
  document.getElementById('toggle-inject').checked = r.autoInject !== false;

  // Load memory count
  api.runtime.sendMessage({ type: 'GET_COUNT' }, (resp) => {
    if (resp) document.getElementById('count').textContent = resp.count || 0;
  });

  // Check server status
  fetch(`${API}/health`).then(res => {
    if (res.ok) {
      document.getElementById('status-dot').style.background = '#5de0c5';
      document.getElementById('status-text').textContent = 'Memory engine online';
    }
  }).catch(() => {
    document.getElementById('status-dot').style.background = '#f05d9a';
    document.getElementById('status-text').textContent = 'Server unreachable';
  });
});

// Save auto-inject toggle
document.getElementById('toggle-inject').addEventListener('change', e => {
  api.storage.local.set({ autoInject: e.target.checked });
});

// Save local mode toggle — close popup to take effect
document.getElementById('toggle-local').addEventListener('change', e => {
  api.storage.local.set({ cls_local: e.target.checked }, () => {
    window.close();
  });
});
