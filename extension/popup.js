const API = 'https://clsplusplus.onrender.com';

// Load count
chrome.runtime.sendMessage({ type: 'GET_COUNT' }, (resp) => {
  if (resp) document.getElementById('count').textContent = resp.count || 0;
});

// Load toggle state
chrome.storage.local.get('autoInject', ({ autoInject }) => {
  document.getElementById('toggle-inject').checked = autoInject !== false;
});

// Save toggle
document.getElementById('toggle-inject').addEventListener('change', e => {
  chrome.storage.local.set({ autoInject: e.target.checked });
});

// Check server status
fetch(`${API}/health`).then(r => {
  if (r.ok) {
    document.getElementById('status-dot').style.background = '#5de0c5';
    document.getElementById('status-text').textContent = 'Memory engine online';
  }
}).catch(() => {
  document.getElementById('status-dot').style.background = '#f05d9a';
  document.getElementById('status-text').textContent = 'Server unreachable';
});
