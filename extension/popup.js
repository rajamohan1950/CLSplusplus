// Use local server when cls_local is set in storage, otherwise cloud
let API = 'https://clsplusplus.onrender.com';

// Initialize everything after loading storage state
chrome.storage.local.get(['cls_local', 'autoInject', 'cls_api_key', 'cls_user'], (r) => {
  if (r.cls_local) API = 'http://localhost:8080';

  // Dashboard link
  document.getElementById('dashboard-link').href = `${API}/dashboard.html`;

  // Update "View My Memories" link
  const memPath = r.cls_local ? '/ui/memory.html' : '/memory.html';
  document.getElementById('view-btn').href = `${API}${memPath}`;

  // Load toggles
  document.getElementById('toggle-local').checked = !!r.cls_local;
  document.getElementById('toggle-inject').checked = r.autoInject !== false;

  // Load memory count
  chrome.runtime.sendMessage({ type: 'GET_COUNT' }, (resp) => {
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

  // Show account state
  if (r.cls_api_key && r.cls_user) {
    showLinked(r.cls_user);
  } else if (r.cls_api_key) {
    // Key stored but no cached user — verify it
    verifyKey(r.cls_api_key);
  }
});

// ── Account linking ─────────────────────────────────────────────────────

function showLinked(user) {
  document.getElementById('account-unlinked').style.display = 'none';
  document.getElementById('account-linked').style.display = 'block';
  document.getElementById('linked-email').textContent = user.email || user.name || 'Linked';
  document.getElementById('linked-tier').textContent = (user.tier || 'free').toUpperCase() + ' tier';
}

function showUnlinked() {
  document.getElementById('account-unlinked').style.display = 'block';
  document.getElementById('account-linked').style.display = 'none';
  document.getElementById('link-error').style.display = 'none';
  document.getElementById('api-key-input').value = '';
}

async function verifyKey(key) {
  try {
    const r = await fetch(`${API}/v1/auth/me`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (r.ok) {
      const user = await r.json();
      chrome.storage.local.set({ cls_user: user });
      showLinked(user);
      // Tell background to refresh
      chrome.runtime.sendMessage({ type: 'ACCOUNT_LINKED' });
      return user;
    } else {
      // Key invalid — clear it
      chrome.storage.local.remove(['cls_api_key', 'cls_user']);
      showUnlinked();
      return null;
    }
  } catch (e) {
    return null;
  }
}

window.linkAccount = async function () {
  const key = document.getElementById('api-key-input').value.trim();
  const errEl = document.getElementById('link-error');
  if (!key) return;

  errEl.style.display = 'none';
  document.getElementById('link-btn').textContent = 'Verifying...';

  // Save key first
  chrome.storage.local.set({ cls_api_key: key });

  const user = await verifyKey(key);
  if (user) {
    // Success — background will pick up the key
    document.getElementById('link-btn').textContent = 'Link Account';
  } else {
    errEl.textContent = 'Invalid API key. Check your dashboard.';
    errEl.style.display = 'block';
    document.getElementById('link-btn').textContent = 'Link Account';
    chrome.storage.local.remove(['cls_api_key']);
  }
};

window.unlinkAccount = function () {
  chrome.storage.local.remove(['cls_api_key', 'cls_user'], () => {
    showUnlinked();
    chrome.runtime.sendMessage({ type: 'ACCOUNT_UNLINKED' });
  });
};

// Save auto-inject toggle
document.getElementById('toggle-inject').addEventListener('change', e => {
  chrome.storage.local.set({ autoInject: e.target.checked });
});

// Save local mode toggle — reload extension to take effect
document.getElementById('toggle-local').addEventListener('change', e => {
  chrome.storage.local.set({ cls_local: e.target.checked }, () => {
    chrome.runtime.reload();
  });
});
