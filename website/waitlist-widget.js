/* CLS++ Waitlist — "The Queue"
 *
 * Renders a live funnel graphic on the right side of the page. Not a panel,
 * not a card — a bona-fide queue shape that blends with the page background.
 * Visitors land at the wide top ("47 waiting"), flow down the tapering body
 * past real-time position tickers, and emerge at the narrow bottom
 * ("3 active") with a pulsing green dot. Underneath: a one-line email form
 * that reveals the visitor's own position when they verify.
 *
 * Zero dep. Fixed right-side anchor. Polls /v1/waitlist/stats every 20s.
 * Persists verified email in localStorage so returning visitors see
 * their slot without re-entering anything.
 *
 * Drop-in: <script src="waitlist-widget.js"></script>
 */
(function () {
  'use strict';

  if (window.__clsWaitlistMounted) return;
  window.__clsWaitlistMounted = true;

  var STATS_POLL_MS = 20000;
  var STORAGE_KEY = 'cls_waitlist_email';
  var ACCENT = '#ff6b35';
  var INK = '#1d1d1f';
  var DIM = '#86868b';
  var GREEN = '#10b981';

  // ── Styles ───────────────────────────────────────────────────────────────
  // Intentional: no card, no box-shadow, no background. The funnel IS the UI.
  //
  // Challenge: the hero chat dock on the landing pages stretches to ~95% of
  // viewport width, so there's no free right margin for a fixed right-middle
  // widget. Solution: when the widget mounts, we add a body class that pads
  // the <body> on the right, physically pushing hero content away. Because
  // the widget itself is position:fixed, it sits OUTSIDE the padded body and
  // lands in the cleared gap. On narrower viewports we fall back to the
  // bottom-right corner and skip the body padding entirely.
  var GUTTER = 280;
  var css =
    'body.cls-q-active{transition:padding-right .25s ease}' +
    '@media (min-width:1280px){body.cls-q-active{padding-right:' + GUTTER + 'px}}' +
    '.cls-q-root{position:fixed;right:20px;top:50%;transform:translateY(-50%);z-index:9999;font-family:-apple-system,BlinkMacSystemFont,"Inter","Helvetica Neue",Arial,sans-serif;color:' + INK + ';width:240px;pointer-events:none}' +
    '.cls-q-root *{pointer-events:auto}' +
    '.cls-q-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;padding:0 6px}' +
    // Label uses currentColor-like mix-blend so it reads on both themes.
    '.cls-q-label{font-size:10px;font-weight:700;letter-spacing:2.4px;text-transform:uppercase;color:' + DIM + ';opacity:0.9;mix-blend-mode:difference;color:#999}' +
    '.cls-q-close{background:none;border:none;color:' + DIM + ';font-size:14px;cursor:pointer;padding:2px 6px;line-height:1;border-radius:10px;opacity:0.6;transition:opacity .15s;mix-blend-mode:difference}' +
    '.cls-q-close:hover{opacity:1}' +
    '.cls-q-svg{display:block;width:100%;height:auto;filter:drop-shadow(0 4px 16px rgba(255,107,53,0.08))}' +
    '.cls-q-foot{margin-top:10px;padding:0 6px}' +
    // Teaser uses difference-blend so it's readable on any background
    '.cls-q-teaser{font-size:12px;text-align:center;margin-bottom:8px;line-height:1.4;color:#999;mix-blend-mode:difference}' +
    '.cls-q-teaser strong{color:' + ACCENT + ';font-weight:800;mix-blend-mode:normal}' +
    '.cls-q-row{display:flex;gap:6px;align-items:stretch}' +
    '.cls-q-input{flex:1;min-width:0;padding:9px 12px;border:1px solid rgba(29,29,31,0.15);border-radius:12px;font-size:12px;font-family:inherit;outline:none;background:rgba(255,255,255,0.7);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);transition:border-color .15s;color:' + INK + '}' +
    '.cls-q-input::placeholder{color:' + DIM + ';opacity:0.8}' +
    '.cls-q-input:focus{border-color:' + ACCENT + '}' +
    '.cls-q-btn{padding:9px 14px;border:none;border-radius:12px;background:' + INK + ';color:#fff;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit;transition:background .15s,transform .1s;white-space:nowrap}' +
    '.cls-q-btn:hover{background:#000}' +
    '.cls-q-btn:active{transform:scale(0.97)}' +
    '.cls-q-btn:disabled{background:#c0c0c0;cursor:not-allowed}' +
    '.cls-q-err{color:#dc2626;font-size:10.5px;text-align:center;margin-top:6px;min-height:13px;font-weight:500}' +
    '.cls-q-ok{margin-top:6px;padding:10px 12px;background:rgba(255,107,53,0.08);border:1px dashed rgba(255,107,53,0.35);border-radius:12px;text-align:center}' +
    '.cls-q-ok-badge{display:inline-block;padding:2px 10px;background:' + ACCENT + ';color:#fff;border-radius:999px;font-size:9px;font-weight:700;letter-spacing:1px;margin-bottom:4px}' +
    '.cls-q-ok-pos{font-family:"JetBrains Mono","SF Mono",Menlo,monospace;font-size:22px;font-weight:800;color:' + INK + ';line-height:1}' +
    '.cls-q-ok-sub{font-size:10px;color:' + DIM + ';margin-top:4px}' +
    '.cls-q-reopen{pointer-events:auto;padding:8px 14px;background:' + INK + ';color:#fff;border:none;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:0.5px;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,0.15);font-family:inherit;text-transform:uppercase}' +
    '.cls-q-reopen:hover{background:#000}' +
    '.cls-q-reopen::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:' + GREEN + ';margin-right:8px;vertical-align:1px;box-shadow:0 0 0 0 rgba(16,185,129,0.6);animation:cls-q-pulse 1.8s ease-out infinite}' +
    '@keyframes cls-q-pulse{0%{box-shadow:0 0 0 0 rgba(16,185,129,0.6)}70%{box-shadow:0 0 0 6px rgba(16,185,129,0)}100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}}' +
    '@keyframes cls-q-flow{0%{transform:translateY(-6px);opacity:0}15%{opacity:0.7}85%{opacity:0.7}100%{transform:translateY(14px);opacity:0}}' +
    '@keyframes cls-q-blink{0%,45%,100%{opacity:1}50%,95%{opacity:0.25}}' +
    '.cls-q-ticker-row{animation:cls-q-flow 4.5s ease-in-out infinite}' +
    '.cls-q-ticker-row:nth-child(2){animation-delay:1.5s}' +
    '.cls-q-ticker-row:nth-child(3){animation-delay:3s}' +
    '.cls-q-dot-live{animation:cls-q-pulse 1.8s ease-out infinite}' +
    '.cls-q-waiting-num.big{font-size:30px}' +
    '.cls-q-waiting-num.mid{font-size:26px}' +
    '.cls-q-waiting-num.small{font-size:22px}' +
    // Below 1280: fall back to bottom-right corner to avoid the hero column.
    '@media (max-width:1280px){.cls-q-root{right:16px;top:auto;bottom:16px;transform:none;width:230px}}' +
    '@media (max-width:640px){.cls-q-root{width:220px;right:12px;bottom:12px}}';

  var style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  // ── Funnel SVG (viewBox is fixed; content scales) ────────────────────────
  // Width 300, the funnel path is a trapezoid: wide top (280) tapering to
  // narrow bottom (100). Height 340 to give room for two data zones + flow.
  var svgMarkup =
    '<svg class="cls-q-svg" viewBox="0 0 300 340" xmlns="http://www.w3.org/2000/svg" aria-label="CLS++ waitlist">' +
      '<defs>' +
        '<linearGradient id="clsQGrad" x1="0" y1="0" x2="0" y2="1">' +
          '<stop offset="0" stop-color="' + ACCENT + '" stop-opacity="0.06"/>' +
          '<stop offset="0.5" stop-color="' + ACCENT + '" stop-opacity="0.14"/>' +
          '<stop offset="1" stop-color="' + ACCENT + '" stop-opacity="0.22"/>' +
        '</linearGradient>' +
        '<linearGradient id="clsQStroke" x1="0" y1="0" x2="0" y2="1">' +
          '<stop offset="0" stop-color="' + ACCENT + '" stop-opacity="0.55"/>' +
          '<stop offset="1" stop-color="' + ACCENT + '" stop-opacity="0.9"/>' +
        '</linearGradient>' +
      '</defs>' +

      // The funnel itself
      '<path d="M 10 18 L 290 18 L 200 322 L 100 322 Z" fill="url(#clsQGrad)" stroke="url(#clsQStroke)" stroke-width="1.5" stroke-dasharray="3 4" stroke-linejoin="round"/>' +

      // Top waiting section — orange brand color so it pops on light AND dark themes
      '<text id="clsQWaitingNum" class="cls-q-waiting-num big" x="150" y="66" text-anchor="middle" font-weight="800" fill="' + ACCENT + '" font-family="-apple-system,BlinkMacSystemFont,Inter,sans-serif">—</text>' +
      '<text x="150" y="86" text-anchor="middle" font-size="10" font-weight="700" letter-spacing="2.4" fill="' + ACCENT + '" fill-opacity="0.8" font-family="-apple-system,BlinkMacSystemFont,Inter,sans-serif">IN THE QUEUE</text>' +

      // Divider
      '<line x1="55" y1="100" x2="245" y2="100" stroke="' + ACCENT + '" stroke-width="0.6" stroke-dasharray="2 3" opacity="0.4"/>' +

      // Position ticker rows (monospace, animated flow). All orange so visible
      // on any page theme. Varying opacity builds depth.
      '<g transform="translate(0,120)" font-family="JetBrains Mono,SF Mono,Menlo,monospace" font-size="11">' +
        '<g class="cls-q-ticker-row">' +
          '<text id="clsQPosA" x="90" y="0" fill="' + ACCENT + '" fill-opacity="0.55">#45</text>' +
          '<text x="210" y="0" text-anchor="end" fill="' + ACCENT + '" fill-opacity="0.55">◂ in</text>' +
        '</g>' +
        '<g class="cls-q-ticker-row">' +
          '<text id="clsQPosB" x="95" y="22" fill="' + ACCENT + '" fill-opacity="0.75">#46</text>' +
          '<text x="205" y="22" text-anchor="end" fill="' + ACCENT + '" fill-opacity="0.75">◂ in</text>' +
        '</g>' +
        '<g class="cls-q-ticker-row">' +
          '<text id="clsQPosC" x="100" y="44" fill="' + ACCENT + '" font-weight="700">#47</text>' +
          '<text x="200" y="44" text-anchor="end" fill="' + ACCENT + '" font-weight="700">◂ next</text>' +
        '</g>' +
      '</g>' +

      // Flow arrow
      '<g opacity="0.55">' +
        '<line x1="150" y1="198" x2="150" y2="232" stroke="' + ACCENT + '" stroke-width="1.3" stroke-dasharray="2 3"/>' +
        '<polygon points="144,228 156,228 150,238" fill="' + ACCENT + '"/>' +
      '</g>' +

      // Active section (narrow end)
      '<line x1="110" y1="252" x2="190" y2="252" stroke="' + ACCENT + '" stroke-width="0.6" stroke-dasharray="2 3" opacity="0.4"/>' +
      '<circle id="clsQDot" cx="115" cy="282" r="4" fill="' + GREEN + '" class="cls-q-dot-live"/>' +
      // Green brand color so it reads on light AND dark themes alike
      '<text id="clsQActiveNum" x="150" y="290" text-anchor="middle" font-size="22" font-weight="800" fill="' + GREEN + '" font-family="-apple-system,BlinkMacSystemFont,Inter,sans-serif">—</text>' +
      '<text x="150" y="308" text-anchor="middle" font-size="9" font-weight="700" letter-spacing="1.8" fill="' + GREEN + '" font-family="-apple-system,BlinkMacSystemFont,Inter,sans-serif">ACTIVE RIGHT NOW</text>' +
    '</svg>';

  // ── Mount ────────────────────────────────────────────────────────────────
  document.body.classList.add('cls-q-active');

  var root = document.createElement('div');
  root.className = 'cls-q-root';
  root.innerHTML =
    '<div class="cls-q-header">' +
    '  <span class="cls-q-label">Launching slowly</span>' +
    '  <button class="cls-q-close" id="clsQClose" aria-label="Minimize">×</button>' +
    '</div>' +
    svgMarkup +
    '<div class="cls-q-foot" id="clsQFoot">' +
    '  <div class="cls-q-teaser" id="clsQTeaser">You could be <strong id="clsQNextPos">#48</strong></div>' +
    '  <form class="cls-q-row" id="clsQFormEmail" novalidate>' +
    '    <input type="email" class="cls-q-input" id="clsQEmail" placeholder="your@email" required autocomplete="email">' +
    '    <button type="submit" class="cls-q-btn" id="clsQBtnEmail">Join →</button>' +
    '  </form>' +
    '  <form class="cls-q-row" id="clsQFormOtp" novalidate style="display:none">' +
    '    <input type="text" class="cls-q-input" id="clsQOtp" placeholder="6-digit code" maxlength="6" inputmode="numeric" pattern="[0-9]{6}" required>' +
    '    <button type="submit" class="cls-q-btn" id="clsQBtnOtp">Verify</button>' +
    '  </form>' +
    '  <div class="cls-q-err" id="clsQErr"></div>' +
    '  <div class="cls-q-ok" id="clsQOk" style="display:none">' +
    '    <div class="cls-q-ok-badge">YOU\'RE IN</div>' +
    '    <div class="cls-q-ok-pos">#<span id="clsQOkPos">—</span></div>' +
    '    <div class="cls-q-ok-sub" id="clsQOkSub">We\'ll email you the moment it\'s your turn.</div>' +
    '  </div>' +
    '</div>';
  document.body.appendChild(root);

  // Reopen chip — hidden until closed
  var reopen = document.createElement('button');
  reopen.className = 'cls-q-reopen';
  reopen.textContent = 'The Queue';
  reopen.style.display = 'none';
  reopen.style.position = 'fixed';
  reopen.style.right = '20px';
  reopen.style.bottom = '20px';
  reopen.style.zIndex = '9998';
  document.body.appendChild(reopen);

  // Element refs
  var waitingNum = document.getElementById('clsQWaitingNum');
  var activeNum = document.getElementById('clsQActiveNum');
  var posA = document.getElementById('clsQPosA');
  var posB = document.getElementById('clsQPosB');
  var posC = document.getElementById('clsQPosC');
  var nextPos = document.getElementById('clsQNextPos');
  var teaser = document.getElementById('clsQTeaser');
  var formEmail = document.getElementById('clsQFormEmail');
  var formOtp = document.getElementById('clsQFormOtp');
  var emailInput = document.getElementById('clsQEmail');
  var otpInput = document.getElementById('clsQOtp');
  var btnEmail = document.getElementById('clsQBtnEmail');
  var btnOtp = document.getElementById('clsQBtnOtp');
  var err = document.getElementById('clsQErr');
  var ok = document.getElementById('clsQOk');
  var okPos = document.getElementById('clsQOkPos');
  var okSub = document.getElementById('clsQOkSub');
  var foot = document.getElementById('clsQFoot');
  var closeBtn = document.getElementById('clsQClose');

  var pendingEmail = '';

  // ── Helpers ──────────────────────────────────────────────────────────────
  function fmtBig(n) {
    if (n == null || isNaN(n)) return '—';
    n = Math.max(0, Math.floor(n));
    if (n >= 10000) return (Math.floor(n / 100) / 10).toFixed(1).replace(/\.0$/, '') + 'k';
    return n.toLocaleString();
  }

  function fitWaitingFont(text) {
    // Shrink-to-fit when the number gets wide
    waitingNum.classList.remove('big', 'mid', 'small');
    var len = String(text).length;
    if (len <= 3) waitingNum.classList.add('big');
    else if (len <= 5) waitingNum.classList.add('mid');
    else waitingNum.classList.add('small');
    waitingNum.setAttribute('font-size', len <= 3 ? '30' : len <= 5 ? '24' : '20');
  }

  function setWaiting(n) {
    var s = fmtBig(n);
    waitingNum.textContent = s;
    fitWaitingFont(s);
    // Also update the three ticker positions at the top of the queue.
    // Show the 3 most recent slots (n-2, n-1, n), which is where "your next"
    // slot would slot in. If fewer than 3 real positions exist, fall back
    // to whatever makes sense.
    var safe = Math.max(1, Math.floor(n));
    posA.textContent = '#' + Math.max(1, safe - 2);
    posB.textContent = '#' + Math.max(1, safe - 1);
    posC.textContent = '#' + safe;
    nextPos.textContent = '#' + (safe + 1);
  }

  function setActive(n) {
    var s = fmtBig(n);
    activeNum.textContent = s;
    // Shrink active number when >= 4 digits
    activeNum.setAttribute('font-size', String(s).length >= 4 ? '16' : '22');
  }

  function showOk(email, position) {
    foot.querySelectorAll('form').forEach(function (f) { f.style.display = 'none'; });
    teaser.style.display = 'none';
    err.textContent = '';
    ok.style.display = 'block';
    okPos.textContent = position;
    okSub.textContent = 'We\'ll email ' + email + ' the moment it\'s your turn.';
    try { localStorage.setItem(STORAGE_KEY, email); } catch (e) {}
  }

  function showEmailForm() {
    ok.style.display = 'none';
    teaser.style.display = '';
    formEmail.style.display = 'flex';
    formOtp.style.display = 'none';
    err.textContent = '';
  }

  function showOtpForm() {
    formEmail.style.display = 'none';
    formOtp.style.display = 'flex';
    err.textContent = '';
    setTimeout(function () { otpInput.focus(); }, 40);
  }

  async function fetchJson(url, opts) {
    var r = await fetch(url, Object.assign({
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin'
    }, opts || {}));
    var data = null;
    try { data = await r.json(); } catch (e) {}
    return { ok: r.ok, status: r.status, data: data || {} };
  }

  async function refreshStats() {
    var saved;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
    var url = '/v1/waitlist/stats' + (saved ? '?email=' + encodeURIComponent(saved) : '');
    var res = await fetchJson(url);
    if (!res.ok) return;
    var d = res.data;
    if (typeof d.waiting_count === 'number') setWaiting(d.waiting_count);
    if (typeof d.active_now === 'number') setActive(d.active_now);
    if (d.your_position && d.your_status !== 'activated') {
      showOk(saved || '', d.your_position);
    }
  }

  // ── Event handlers ───────────────────────────────────────────────────────
  closeBtn.addEventListener('click', function () {
    root.style.display = 'none';
    reopen.style.display = 'inline-flex';
    // Release the body padding when the widget is minimized so the page
    // breathes out to full width.
    document.body.classList.remove('cls-q-active');
  });

  reopen.addEventListener('click', function () {
    root.style.display = '';
    reopen.style.display = 'none';
    document.body.classList.add('cls-q-active');
  });

  formEmail.addEventListener('submit', async function (e) {
    e.preventDefault();
    err.textContent = '';
    var email = (emailInput.value || '').trim();
    if (!email || email.indexOf('@') < 0) {
      err.textContent = 'Please enter a valid email';
      return;
    }
    btnEmail.disabled = true;
    btnEmail.textContent = '…';
    var res = await fetchJson('/v1/waitlist/join', {
      method: 'POST',
      body: JSON.stringify({ email: email })
    });
    btnEmail.disabled = false;
    btnEmail.textContent = 'Join →';
    if (!res.ok) {
      err.textContent = (res.data && (res.data.message || res.data.detail)) || 'Something went wrong';
      return;
    }
    if (res.data.status === 'already_member') {
      err.textContent = 'You already have an account';
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
    err.textContent = '';
    var code = (otpInput.value || '').trim();
    if (!/^[0-9]{6}$/.test(code)) {
      err.textContent = 'Enter the 6-digit code';
      return;
    }
    btnOtp.disabled = true;
    btnOtp.textContent = '…';
    var res = await fetchJson('/v1/waitlist/verify', {
      method: 'POST',
      body: JSON.stringify({ email: pendingEmail, otp_code: code })
    });
    btnOtp.disabled = false;
    btnOtp.textContent = 'Verify';
    if (!res.ok) {
      err.textContent = (res.data && (res.data.message || res.data.detail)) || 'Invalid code';
      return;
    }
    showOk(pendingEmail, res.data.position || '?');
    if (typeof res.data.waiting_count === 'number') setWaiting(res.data.waiting_count);
  });

  // Init
  refreshStats();
  setInterval(refreshStats, STATS_POLL_MS);
})();
