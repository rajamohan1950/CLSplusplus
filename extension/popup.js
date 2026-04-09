// CLS++ Extension Popup
// API is resolved from storage BEFORE any network calls

let API = null; // Set once storage loads

function resolveAPI(callback) {
  if (API) return callback(API);
  chrome.storage.local.get(['cls_local', 'cls_api_url'], (r) => {
    r = r || {};
    if (r.cls_api_url) API = r.cls_api_url;
    else if (r.cls_local) API = 'http://localhost:8181';
    else API = 'https://clsplusplus.onrender.com';
    callback(API);
  });
}

// ── Initialize ──────────────────────────────────────────────────────────

resolveAPI((api) => {
  chrome.storage.local.get(['autoInject', 'cls_local', 'cls_api_key', 'cls_user'], (r) => {
    r = r || {};
    // Links
    document.getElementById('dashboard-link').href = `${api}/dashboard.html`;
    document.getElementById('view-btn').href = `${api}/memory.html`;

    // Toggles
    document.getElementById('toggle-local').checked = !!r.cls_local;
    document.getElementById('toggle-inject').checked = r.autoInject !== false;

    // Memory count
    chrome.runtime.sendMessage({ type: 'GET_COUNT' }, (resp) => {
      if (resp) document.getElementById('count').textContent = resp.count || 0;
    });

    // Server status
    updateServerStatus(api);

    // Account state
    if (r.cls_api_key && r.cls_user) {
      showLinked(r.cls_user);
    } else if (r.cls_api_key) {
      verifyKey(r.cls_api_key);
    }
  });
});

// ── Top bar ─────────────────────────────────────────────────────────────

function updateServerStatus(api) {
  const dot = document.getElementById('top-dot');
  const server = document.getElementById('top-server');
  const display = api.replace('http://', '').replace('https://', '');
  server.textContent = display;

  fetch(`${api}/health`).then(res => {
    if (res.ok) {
      dot.classList.add('online');
    }
  }).catch(() => {
    dot.classList.remove('online');
    server.textContent = display + ' (offline)';
  });
}

function updateTopUser(user) {
  const el = document.getElementById('top-user');
  const tier = document.getElementById('top-tier');
  if (user) {
    el.textContent = user.name || user.email || 'Linked';
    el.classList.remove('offline');
    tier.textContent = (user.tier || 'free').toUpperCase();
    tier.style.display = 'block';
  } else {
    el.textContent = 'Not linked';
    el.classList.add('offline');
    tier.style.display = 'none';
  }
}

// ── Account ─────────────────────────────────────────────────────────────

function showLinked(user) {
  document.getElementById('account-unlinked').style.display = 'none';
  document.getElementById('account-linked').style.display = 'block';
  document.getElementById('account-box').classList.add('linked');
  document.getElementById('linked-email').textContent = user.email || user.name || 'Linked';
  document.getElementById('linked-tier-text').textContent = (user.tier || 'free').toUpperCase() + ' tier';
  document.getElementById('api-key-display').value = '************************';
  updateTopUser(user);
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
  // Ensure API is resolved
  if (!API) {
    await new Promise(resolve => resolveAPI(() => resolve()));
  }
  try {
    const r = await fetch(`${API}/v1/auth/me`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (r.ok) {
      const user = await r.json();
      chrome.storage.local.set({ cls_user: user });
      showLinked(user);
      chrome.runtime.sendMessage({ type: 'ACCOUNT_LINKED' });
      return user;
    } else {
      chrome.storage.local.remove(['cls_api_key', 'cls_user']);
      showUnlinked();
      return null;
    }
  } catch (e) {
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

  const user = await verifyKey(key);
  if (!user) {
    errEl.textContent = `Cannot verify key. Server: ${API || 'unknown'}`;
    errEl.style.display = 'block';
    btnEl.textContent = 'Link Account';
    btnEl.disabled = false;
    inputEl.disabled = false;
    chrome.storage.local.remove(['cls_api_key']);
  }
};

function unlinkAccount() {
  chrome.storage.local.remove(['cls_api_key', 'cls_user'], () => {
    showUnlinked();
    chrome.runtime.sendMessage({ type: 'ACCOUNT_UNLINKED' });
  });
}

// ── Toggles ─────────────────────────────────────────────────────────────

// ── All event listeners (MV3 bans inline onclick) ───────────────────────

document.getElementById('link-btn').addEventListener('click', linkAccount);
document.getElementById('unlink-btn').addEventListener('click', unlinkAccount);

document.getElementById('toggle-inject').addEventListener('change', e => {
  chrome.storage.local.set({ autoInject: e.target.checked });
});

document.getElementById('toggle-local').addEventListener('change', e => {
  chrome.storage.local.set({ cls_local: e.target.checked }, () => {
    chrome.runtime.reload();
  });
});
