// CLS++ Extension Popup
// API is resolved from storage BEFORE any network calls

let API = null;

function resolveAPI(callback) {
  if (API) return callback(API);
  chrome.storage.local.get(['cls_local', 'cls_api_url'], (r) => {
    r = r || {};
    if (r.cls_api_url) API = r.cls_api_url;
    else if (r.cls_local) API = 'http://localhost:8181';
    else API = 'https://www.clsplusplus.com';
    callback(API);
  });
}

// ── Toast notifications ─────────────────────────────────────────────────

function showToast(msg, type, duration) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast ' + (type || 'info') + ' show';
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.classList.remove('show'); }, duration || 3000);
}

// ── Initialize ──────────────────────────────────────────────────────────

resolveAPI((api) => {
  chrome.storage.local.get(['autoInject', 'cls_local', 'cls_api_key', 'cls_user'], (r) => {
    r = r || {};
    document.getElementById('generate-key-link').href = `${api}/profile.html#keys`;
    document.getElementById('view-btn').href = `${api}/memory.html`;
    document.getElementById('toggle-local').checked = !!r.cls_local;
    document.getElementById('toggle-inject').checked = r.autoInject !== false;

    // Memory count
    chrome.runtime.sendMessage({ type: 'GET_COUNT' }, (resp) => {
      if (resp && resp.count !== undefined) {
        document.getElementById('count').textContent = resp.count || 0;
      }
    });

    // Server status with cold-start handling
    checkServerWithRetry(api, 0);

    // Account state
    if (r.cls_api_key && r.cls_user) {
      showLinked(r.cls_user);
    } else if (r.cls_api_key) {
      verifyKey(r.cls_api_key);
    }
  });
});

// ── Server status with cold-start retry ─────────────────────────────────

function checkServerWithRetry(api, attempt) {
  const bar = document.getElementById('top-bar');
  const server = document.getElementById('top-server');
  const user = document.getElementById('top-user');
  const display = api.replace('http://', '').replace('https://', '');

  if (attempt === 0) {
    bar.className = 'top-bar connecting';
    server.textContent = 'Connecting to ' + display + '...';
    user.textContent = 'First load may take 1-2 minutes';
  }

  fetch(`${api}/health`, { signal: AbortSignal.timeout(10000) }).then(res => {
    if (res.ok) {
      bar.className = 'top-bar online';
      server.textContent = display;
      user.textContent = 'Memory engine online';
      if (attempt > 0) showToast('Server is ready!', 'success');
    } else {
      throw new Error('not ok');
    }
  }).catch(() => {
    if (attempt < 5) {
      // Cold start — retry with backoff
      const wait = Math.min(10, 3 + attempt * 2);
      server.textContent = display;
      user.textContent = `Server warming up... retry in ${wait}s`;
      bar.className = 'top-bar connecting';
      setTimeout(() => checkServerWithRetry(api, attempt + 1), wait * 1000);
    } else {
      // Give up
      bar.className = 'top-bar offline';
      server.textContent = display;
      user.textContent = 'Server unreachable — try again in 60s';
      showToast('Server is warming up. Please try again in 60 seconds.', 'error', 5000);
    }
  });
}

// ── Top user display ────────────────────────────────────────────────────

function updateTopUser(userData) {
  const bar = document.getElementById('top-bar');
  const user = document.getElementById('top-user');
  const tier = document.getElementById('top-tier');
  if (userData) {
    user.textContent = userData.name || userData.email || 'Linked';
    tier.textContent = (userData.tier || 'free').toUpperCase();
    tier.style.display = 'block';
    // Keep online/offline class from server status, just update text
  } else {
    user.textContent = bar.classList.contains('online') ? 'Memory engine online' : 'Not linked';
    tier.style.display = 'none';
  }
}

// ── Account ─────────────────────────────────────────────────────────────

function showLinked(userData) {
  document.getElementById('account-unlinked').style.display = 'none';
  document.getElementById('account-linked').style.display = 'block';
  document.getElementById('account-box').classList.add('linked');
  document.getElementById('linked-email').textContent = userData.email || userData.name || 'Linked';
  document.getElementById('linked-tier-text').textContent = (userData.tier || 'free').toUpperCase() + ' tier';
  document.getElementById('api-key-display').value = '************************';
  updateTopUser(userData);
}

function showUnlinked() {
  document.getElementById('account-unlinked').style.display = 'block';
  document.getElementById('account-linked').style.display = 'none';
  document.getElementById('account-box').classList.remove('linked');
  document.getElementById('link-error').style.display = 'none';
  document.getElementById('api-key-input').value = '';
  document.getElementById('api-key-input').disabled = false;
  const btn = document.getElementById('link-btn');
  btn.classList.remove('disabled');
  btn.classList.add('active');
  btn.disabled = false;
  btn.textContent = 'Link Account';
  updateTopUser(null);
}

async function verifyKey(key) {
  if (!API) await new Promise(resolve => resolveAPI(() => resolve()));
  try {
    const r = await fetch(`${API}/v1/auth/me`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (r.ok) {
      const userData = await r.json();
      chrome.storage.local.set({ cls_user: userData });
      showLinked(userData);
      showToast('Account linked!', 'success');
      chrome.runtime.sendMessage({ type: 'ACCOUNT_LINKED' });
      return userData;
    } else {
      chrome.storage.local.remove(['cls_api_key', 'cls_user']);
      showUnlinked();
      return null;
    }
  } catch (e) {
    showToast('Cannot reach server — it may be warming up', 'error', 4000);
    return null;
  }
}

async function linkAccount() {
  const key = document.getElementById('api-key-input').value.trim();
  const errEl = document.getElementById('link-error');
  const btnEl = document.getElementById('link-btn');
  const inputEl = document.getElementById('api-key-input');
  if (!key) return;

  errEl.style.display = 'none';
  btnEl.textContent = 'Verifying...';
  btnEl.disabled = true;
  inputEl.disabled = true;

  chrome.storage.local.set({ cls_api_key: key });

  const userData = await verifyKey(key);
  if (!userData) {
    errEl.textContent = `Cannot verify key. Server may be warming up — try again in 60s.`;
    errEl.style.display = 'block';
    btnEl.textContent = 'Link Account';
    btnEl.disabled = false;
    inputEl.disabled = false;
    chrome.storage.local.remove(['cls_api_key']);
  }
}

function unlinkAccount() {
  chrome.storage.local.remove(['cls_api_key', 'cls_user'], () => {
    showUnlinked();
    showToast('Account unlinked', 'info');
    chrome.runtime.sendMessage({ type: 'ACCOUNT_UNLINKED' });
  });
}

// ── Event listeners (MV3 — no inline handlers) ─────────────────────────

document.getElementById('link-btn').addEventListener('click', linkAccount);
document.getElementById('unlink-btn').addEventListener('click', unlinkAccount);

document.getElementById('toggle-inject').addEventListener('change', e => {
  chrome.storage.local.set({ autoInject: e.target.checked });
  showToast(e.target.checked ? 'Auto-inject enabled' : 'Auto-inject disabled', 'info', 2000);
});

document.getElementById('toggle-local').addEventListener('change', e => {
  chrome.storage.local.set({ cls_local: e.target.checked }, () => {
    chrome.runtime.reload();
  });
});
