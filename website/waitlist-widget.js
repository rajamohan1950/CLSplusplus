/* CLS++ Waitlist Widget — self-contained, zero-dep.
 *
 * Renders a fixed bottom-right panel on the landing page showing:
 *   • total waiting (seeded for social proof)
 *   • animated funnel with top-of-queue avatars (decorative)
 *   • live "active now" counter (clamped to floor)
 *   • email + OTP form → joins waitlist → shows your position
 *
 * Re-fetches /v1/waitlist/stats every 20s.
 * Persists the user's email in localStorage so returning visitors see their position.
 *
 * Drop-in: <script src="waitlist-widget.js"></script> at the bottom of <body>.
 */
(function () {
  'use strict';

  if (window.__clsWaitlistMounted) return;
  window.__clsWaitlistMounted = true;

  var API = ''; // same-origin
  var STATS_POLL_MS = 20000;
  var STORAGE_KEY = 'cls_waitlist_email';

  // ── Styles ───────────────────────────────────────────────────────────────
  var css = '' +
    '.cls-wl-root{position:fixed;right:20px;bottom:20px;z-index:9999;font-family:-apple-system,BlinkMacSystemFont,"Inter","Helvetica Neue",Arial,sans-serif;color:#1d1d1f}' +
    '.cls-wl-panel{width:280px;background:rgba(255,255,255,0.96);backdrop-filter:blur(24px) saturate(1.6);-webkit-backdrop-filter:blur(24px) saturate(1.6);border:1px solid rgba(0,0,0,0.08);border-radius:20px;box-shadow:0 12px 48px rgba(0,0,0,0.18);overflow:hidden;transition:transform .3s ease,opacity .3s ease;transform-origin:bottom right}' +
    '.cls-wl-panel.cls-wl-collapsed{width:auto;border-radius:999px;padding:0}' +
    '.cls-wl-head{display:flex;align-items:center;justify-content:space-between;padding:14px 18px 6px}' +
    '.cls-wl-title{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#86868b}' +
    '.cls-wl-close{background:none;border:none;color:#86868b;font-size:18px;cursor:pointer;padding:0;width:22px;height:22px;line-height:1;border-radius:50%;display:flex;align-items:center;justify-content:center}' +
    '.cls-wl-close:hover{background:#f0f0f0;color:#1d1d1f}' +
    '.cls-wl-waiting{padding:4px 18px 10px;display:flex;align-items:baseline;gap:8px}' +
    '.cls-wl-waiting-num{font-size:38px;font-weight:800;letter-spacing:-1px;color:#1d1d1f;line-height:1;font-variant-numeric:tabular-nums}' +
    '.cls-wl-waiting-lbl{font-size:12px;color:#86868b;font-weight:600}' +
    '.cls-wl-funnel{position:relative;margin:4px 18px 12px;padding:10px 0;background:linear-gradient(180deg,#faf9f7 0%,#f3f0ea 100%);border-radius:14px;border:1px solid rgba(0,0,0,0.04)}' +
    '.cls-wl-funnel::before{content:"";position:absolute;top:-1px;left:50%;transform:translateX(-50%);width:80%;height:2px;background:#ff6b35;border-radius:2px;opacity:0.6}' +
    '.cls-wl-funnel-inner{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;padding:0 16px;clip-path:polygon(0 0,100% 0,85% 100%,15% 100%)}' +
    '.cls-wl-avatar{height:26px;display:flex;align-items:center;justify-content:center;border-radius:8px;font-size:10px;font-weight:700;color:#fff;animation:clsWlPulse 2.4s ease-in-out infinite}' +
    '.cls-wl-avatar:nth-child(1){background:#ff6b35;animation-delay:0s}' +
    '.cls-wl-avatar:nth-child(2){background:#f59e0b;animation-delay:.2s}' +
    '.cls-wl-avatar:nth-child(3){background:#10b981;animation-delay:.4s}' +
    '.cls-wl-avatar:nth-child(4){background:#3b82f6;animation-delay:.6s}' +
    '.cls-wl-avatar:nth-child(5){background:#8b5cf6;animation-delay:.8s}' +
    '.cls-wl-avatar:nth-child(6){background:#ec4899;animation-delay:1s}' +
    '@keyframes clsWlPulse{0%,100%{opacity:.85;transform:translateY(0)}50%{opacity:1;transform:translateY(-1px)}}' +
    '.cls-wl-active{display:flex;align-items:center;gap:8px;padding:0 18px 14px;font-size:13px;color:#1d1d1f}' +
    '.cls-wl-dot{width:8px;height:8px;border-radius:50%;background:#10b981;box-shadow:0 0 0 0 rgba(16,185,129,.7);animation:clsWlDot 1.8s ease-out infinite}' +
    '@keyframes clsWlDot{0%{box-shadow:0 0 0 0 rgba(16,185,129,.6)}70%{box-shadow:0 0 0 8px rgba(16,185,129,0)}100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}}' +
    '.cls-wl-active strong{font-weight:700;font-variant-numeric:tabular-nums}' +
    '.cls-wl-cta{padding:12px 18px 16px;background:#fafafa;border-top:1px solid rgba(0,0,0,0.05)}' +
    '.cls-wl-cta-title{font-size:13px;font-weight:600;color:#1d1d1f;margin:0 0 2px}' +
    '.cls-wl-cta-sub{font-size:11px;color:#86868b;margin:0 0 10px}' +
    '.cls-wl-input{width:100%;padding:10px 12px;border:1px solid rgba(0,0,0,0.12);border-radius:10px;font-size:13px;box-sizing:border-box;font-family:inherit;outline:none;transition:border-color .15s}' +
    '.cls-wl-input:focus{border-color:#ff6b35}' +
    '.cls-wl-btn{width:100%;margin-top:8px;padding:10px 16px;background:#1d1d1f;color:#fff;border:none;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;transition:background .15s}' +
    '.cls-wl-btn:hover{background:#000}' +
    '.cls-wl-btn:disabled{background:#c0c0c0;cursor:not-allowed}' +
    '.cls-wl-err{color:#dc2626;font-size:11px;margin-top:6px;min-height:14px}' +
    '.cls-wl-ok{background:#fff;border-top:1px solid rgba(0,0,0,0.05);padding:16px 18px;text-align:center}' +
    '.cls-wl-ok-badge{display:inline-block;padding:4px 12px;background:#ff6b35;color:#fff;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.5px;margin-bottom:8px}' +
    '.cls-wl-ok-pos{font-size:32px;font-weight:800;color:#1d1d1f;line-height:1;font-variant-numeric:tabular-nums}' +
    '.cls-wl-ok-sub{font-size:12px;color:#86868b;margin-top:6px}' +
    '.cls-wl-reopen{padding:10px 18px;background:#1d1d1f;color:#fff;border:none;border-radius:999px;font-size:12px;font-weight:600;cursor:pointer;box-shadow:0 8px 24px rgba(0,0,0,0.2)}' +
    '.cls-wl-reopen:hover{background:#000}' +
    '@media (max-width:640px){.cls-wl-root{right:12px;bottom:12px}.cls-wl-panel{width:260px}}';

  var style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  // ── DOM ──────────────────────────────────────────────────────────────────
  var root = document.createElement('div');
  root.className = 'cls-wl-root';
  root.innerHTML =
    '<div class="cls-wl-panel" id="clsWlPanel">' +
    '  <div class="cls-wl-head">' +
    '    <span class="cls-wl-title">Launching slowly</span>' +
    '    <button class="cls-wl-close" id="clsWlClose" aria-label="Minimize">×</button>' +
    '  </div>' +
    '  <div class="cls-wl-waiting">' +
    '    <span class="cls-wl-waiting-num" id="clsWlWaiting">—</span>' +
    '    <span class="cls-wl-waiting-lbl">waiting</span>' +
    '  </div>' +
    '  <div class="cls-wl-funnel"><div class="cls-wl-funnel-inner">' +
    '    <div class="cls-wl-avatar">1</div>' +
    '    <div class="cls-wl-avatar">2</div>' +
    '    <div class="cls-wl-avatar">3</div>' +
    '    <div class="cls-wl-avatar">4</div>' +
    '    <div class="cls-wl-avatar">5</div>' +
    '    <div class="cls-wl-avatar">6</div>' +
    '  </div></div>' +
    '  <div class="cls-wl-active">' +
    '    <span class="cls-wl-dot"></span>' +
    '    <span><strong id="clsWlActive">—</strong> active right now</span>' +
    '  </div>' +
    '  <div class="cls-wl-cta" id="clsWlCta">' +
    '    <p class="cls-wl-cta-title">Want in?</p>' +
    '    <p class="cls-wl-cta-sub">We\'re releasing access in small waves. Get on the list — we\'ll email you when your turn comes up.</p>' +
    '    <form id="clsWlFormEmail" novalidate>' +
    '      <input type="email" class="cls-wl-input" id="clsWlEmail" placeholder="you@work.com" required autocomplete="email">' +
    '      <button type="submit" class="cls-wl-btn" id="clsWlBtnEmail">Get early access</button>' +
    '      <div class="cls-wl-err" id="clsWlErrEmail"></div>' +
    '    </form>' +
    '    <form id="clsWlFormOtp" novalidate style="display:none">' +
    '      <input type="text" class="cls-wl-input" id="clsWlOtp" placeholder="6-digit code from email" maxlength="6" inputmode="numeric" pattern="[0-9]{6}" required>' +
    '      <button type="submit" class="cls-wl-btn" id="clsWlBtnOtp">Verify</button>' +
    '      <div class="cls-wl-err" id="clsWlErrOtp"></div>' +
    '    </form>' +
    '  </div>' +
    '  <div class="cls-wl-ok" id="clsWlOk" style="display:none">' +
    '    <div class="cls-wl-ok-badge">YOU\'RE IN LINE</div>' +
    '    <div class="cls-wl-ok-pos">#<span id="clsWlOkPos">—</span></div>' +
    '    <div class="cls-wl-ok-sub">We\'ll email <span id="clsWlOkEmail"></span> the moment it\'s your turn.</div>' +
    '  </div>' +
    '</div>';
  document.body.appendChild(root);

  // Reopen button (hidden initially)
  var reopen = document.createElement('button');
  reopen.className = 'cls-wl-reopen';
  reopen.textContent = '👀 See the queue';
  reopen.style.display = 'none';
  reopen.addEventListener('click', function () {
    panel.style.display = '';
    reopen.style.display = 'none';
  });
  root.appendChild(reopen);

  var panel = document.getElementById('clsWlPanel');
  var waitingEl = document.getElementById('clsWlWaiting');
  var activeEl = document.getElementById('clsWlActive');
  var formEmail = document.getElementById('clsWlFormEmail');
  var formOtp = document.getElementById('clsWlFormOtp');
  var emailInput = document.getElementById('clsWlEmail');
  var otpInput = document.getElementById('clsWlOtp');
  var btnEmail = document.getElementById('clsWlBtnEmail');
  var btnOtp = document.getElementById('clsWlBtnOtp');
  var errEmail = document.getElementById('clsWlErrEmail');
  var errOtp = document.getElementById('clsWlErrOtp');
  var cta = document.getElementById('clsWlCta');
  var ok = document.getElementById('clsWlOk');
  var okPos = document.getElementById('clsWlOkPos');
  var okEmail = document.getElementById('clsWlOkEmail');

  document.getElementById('clsWlClose').addEventListener('click', function () {
    panel.style.display = 'none';
    reopen.style.display = 'inline-block';
  });

  var pendingEmail = '';

  function showOk(email, position) {
    cta.style.display = 'none';
    ok.style.display = '';
    okPos.textContent = position;
    okEmail.textContent = email;
    try { localStorage.setItem(STORAGE_KEY, email); } catch (e) {}
  }

  function showEmailForm() {
    cta.style.display = '';
    ok.style.display = 'none';
    formEmail.style.display = '';
    formOtp.style.display = 'none';
  }

  function showOtpForm() {
    formEmail.style.display = 'none';
    formOtp.style.display = '';
    setTimeout(function () { otpInput.focus(); }, 50);
  }

  async function fetchJson(url, opts) {
    var resp = await fetch(url, Object.assign({
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin'
    }, opts || {}));
    var data = null;
    try { data = await resp.json(); } catch (e) {}
    return { ok: resp.ok, status: resp.status, data: data || {} };
  }

  async function refreshStats() {
    var saved;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
    var url = '/v1/waitlist/stats' + (saved ? '?email=' + encodeURIComponent(saved) : '');
    var res = await fetchJson(url);
    if (!res.ok) return;
    var d = res.data;
    if (typeof d.waiting_count === 'number') waitingEl.textContent = d.waiting_count.toLocaleString();
    if (typeof d.active_now === 'number') activeEl.textContent = d.active_now.toLocaleString();
    if (d.your_position && d.your_status !== 'activated') {
      showOk(saved || '', d.your_position);
    }
  }

  formEmail.addEventListener('submit', async function (e) {
    e.preventDefault();
    errEmail.textContent = '';
    var email = (emailInput.value || '').trim();
    if (!email || email.indexOf('@') < 0) {
      errEmail.textContent = 'Please enter a valid email';
      return;
    }
    btnEmail.disabled = true;
    btnEmail.textContent = 'Sending code…';
    var res = await fetchJson('/v1/waitlist/join', {
      method: 'POST',
      body: JSON.stringify({ email: email })
    });
    btnEmail.disabled = false;
    btnEmail.textContent = 'Get early access';
    if (!res.ok) {
      errEmail.textContent = (res.data && (res.data.message || res.data.detail)) || 'Something went wrong. Try again?';
      return;
    }
    if (res.data.status === 'already_member') {
      errEmail.textContent = 'You already have an account. Sign in above.';
      return;
    }
    if (res.data.status === 'waiting' || res.data.status === 'invited') {
      showOk(email, res.data.position || '?');
      return;
    }
    pendingEmail = email;
    showOtpForm();
  });

  formOtp.addEventListener('submit', async function (e) {
    e.preventDefault();
    errOtp.textContent = '';
    var code = (otpInput.value || '').trim();
    if (!/^[0-9]{6}$/.test(code)) {
      errOtp.textContent = 'Enter the 6-digit code';
      return;
    }
    btnOtp.disabled = true;
    btnOtp.textContent = 'Verifying…';
    var res = await fetchJson('/v1/waitlist/verify', {
      method: 'POST',
      body: JSON.stringify({ email: pendingEmail, otp_code: code })
    });
    btnOtp.disabled = false;
    btnOtp.textContent = 'Verify';
    if (!res.ok) {
      errOtp.textContent = (res.data && (res.data.message || res.data.detail)) || 'Invalid code';
      return;
    }
    showOk(pendingEmail, res.data.position || '?');
    if (typeof res.data.waiting_count === 'number') {
      waitingEl.textContent = res.data.waiting_count.toLocaleString();
    }
  });

  // Init
  refreshStats();
  setInterval(refreshStats, STATS_POLL_MS);
})();
