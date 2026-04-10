// CLS++ Popup — Minimal
const API = 'https://www.clsplusplus.com';

// Check server
fetch(API + '/health').then(r => {
  if (r.ok) { document.getElementById('dot').className = 'dot on'; document.getElementById('srv').textContent = 'Connected'; }
}).catch(() => { document.getElementById('srv').textContent = 'Server offline'; });

// Load saved state
chrome.storage.local.get(['cls_api_key', 'cls_user'], r => {
  if (r.cls_api_key && r.cls_user) showLinked(r.cls_user);
});

function showLinked(user) {
  document.getElementById('sec-unlinked').style.display = 'none';
  document.getElementById('sec-linked').style.display = 'block';
  document.getElementById('user-name').textContent = user.name || user.email || 'Linked';
  document.getElementById('user-tier').textContent = (user.tier || 'free').toUpperCase();
}

function showUnlinked() {
  document.getElementById('sec-unlinked').style.display = 'block';
  document.getElementById('sec-linked').style.display = 'none';
  document.getElementById('key-input').value = '';
}

// Link
document.getElementById('btn-link').addEventListener('click', async () => {
  const key = document.getElementById('key-input').value.trim();
  const err = document.getElementById('err');
  const btn = document.getElementById('btn-link');
  if (!key) return;

  err.style.display = 'none';
  btn.textContent = 'Verifying...';
  btn.disabled = true;

  chrome.runtime.sendMessage({ type: 'VERIFY_KEY', key }, user => {
    if (user && user.email) {
      chrome.storage.local.set({ cls_api_key: key, cls_user: user });
      showLinked(user);
    } else {
      err.textContent = 'Invalid key or server unreachable';
      err.style.display = 'block';
    }
    btn.textContent = 'Link Account';
    btn.disabled = false;
  });
});

// Unlink
document.getElementById('btn-unlink').addEventListener('click', () => {
  chrome.storage.local.remove(['cls_api_key', 'cls_user']);
  showUnlinked();
});
