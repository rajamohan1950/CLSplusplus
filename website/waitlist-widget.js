/* CLS++ Waitlist — "Terminal Session"
 *
 * A running Unix-style terminal pane that streams live queue events:
 *
 *   ┌─ cls.queue ────────────── ● ─┐
 *   │ $ watch -n1 queue/status      │
 *   │                               │
 *   │ [12:47:03] user joined  #73   │
 *   │ [12:46:51] user joined  #72   │
 *   │ [12:46:32] ─ seat granted ✓   │
 *   │ [12:46:12] user joined  #71   │
 *   │ [12:45:58] user joined  #70   │
 *   │                               │
 *   │ WAITING    73                 │
 *   │ ACTIVE     12  ●              │
 *   │ NEXT WAVE  5 seats · Mon 09   │
 *   │                               │
 *   │ > email@company.com_          │
 *   │ [enter] to claim seat         │
 *   └───────────────────────────────┘
 *
 * Self-illuminated — solid background, never depends on what's behind it.
 * Monospace throughout so border-case numbers fit naturally without font
 * gymnastics. Body gutter (via body.cls-q-active) still applies on viewports
 * >= 1280px so the widget doesn't overlap the hero chat dock.
 *
 * Drop-in: <script src="waitlist-widget.js"></script>
 */
(function () {
  'use strict';

  if (window.__clsWaitlistMounted) return;
  window.__clsWaitlistMounted = true;

  var STATS_POLL_MS = 20000;
  var STREAM_TICK_MS = 3800;
  var STORAGE_KEY = 'cls_waitlist_email';

  // Palette — mirrors GitHub dark; deliberate high contrast.
  var BG = '#0d1117';
  var BG_ALT = '#161b22';
  var BORDER = '#30363d';
  var MUTED = '#6e7681';
  var TEXT = '#c9d1d9';
  var DIM_CYAN = '#58a6ff';
  var ACCENT = '#ff6b35';
  var GREEN = '#10b981';
  var AMBER = '#f59e0b';
  var RED = '#f87171';

  // ── CSS ──────────────────────────────────────────────────────────────────
  var GUTTER = 320;
  var css = [
    // Body gutter so the fixed widget sits in a clear right channel.
    'body.cls-q-active{transition:padding-right .25s ease}',
    '@media (min-width:1280px){body.cls-q-active{padding-right:' + GUTTER + 'px}}',

    // Widget shell — right-middle on desktop, bottom-right on narrow.
    '.cls-q-root{position:fixed;right:24px;top:50%;transform:translateY(-50%);z-index:9999;width:306px;font-family:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;color:' + TEXT + ';pointer-events:none}',
    '.cls-q-root *{pointer-events:auto}',

    // The terminal frame
    '.cls-q-term{background:' + BG + ';border:1px solid ' + BORDER + ';border-radius:10px;box-shadow:0 20px 60px rgba(0,0,0,0.45),0 0 0 1px rgba(255,107,53,0.08),0 0 40px rgba(255,107,53,0.06);overflow:hidden}',

    // Title bar
    '.cls-q-tbar{display:flex;align-items:center;gap:8px;padding:10px 14px;background:' + BG_ALT + ';border-bottom:1px solid ' + BORDER + ';font-size:11px}',
    '.cls-q-tbar-dots{display:flex;gap:6px}',
    '.cls-q-tdot{width:9px;height:9px;border-radius:50%}',
    '.cls-q-tdot.r{background:#ff5f56}',
    '.cls-q-tdot.y{background:#ffbd2e}',
    '.cls-q-tdot.g{background:#27c93f}',
    '.cls-q-ttitle{flex:1;text-align:center;color:' + MUTED + ';font-weight:500;letter-spacing:0.3px}',
    '.cls-q-ttitle b{color:' + TEXT + ';font-weight:600}',
    '.cls-q-tlive{display:flex;align-items:center;gap:5px;font-size:9px;color:' + GREEN + ';text-transform:uppercase;letter-spacing:1.2px;font-weight:700}',
    '.cls-q-tlive::before{content:"";width:6px;height:6px;border-radius:50%;background:' + GREEN + ';box-shadow:0 0 0 0 rgba(16,185,129,0.6);animation:cls-q-pulse 1.8s ease-out infinite}',

    // Body
    '.cls-q-body{padding:12px 14px 14px;font-size:11.5px;line-height:1.55}',
    '.cls-q-cmd{color:' + MUTED + ';margin-bottom:8px}',
    '.cls-q-cmd .p{color:' + GREEN + '}',
    '.cls-q-cmd .x{color:' + TEXT + '}',

    // Log stream
    '.cls-q-log{height:112px;overflow:hidden;position:relative;margin-bottom:10px}',
    '.cls-q-log-line{display:flex;gap:8px;white-space:nowrap;opacity:0;animation:cls-q-slidein .4s ease-out forwards}',
    '.cls-q-log-line.fading{animation:cls-q-slideout .4s ease-in forwards}',
    '.cls-q-log-time{color:' + DIM_CYAN + ';opacity:0.7;flex-shrink:0}',
    '.cls-q-log-txt{color:' + MUTED + '}',
    '.cls-q-log-pos{color:' + TEXT + ';margin-left:auto;font-weight:600}',
    '.cls-q-log-line.hl .cls-q-log-txt{color:' + ACCENT + '}',
    '.cls-q-log-line.hl .cls-q-log-pos{color:' + ACCENT + '}',
    '.cls-q-log-line.sys .cls-q-log-txt{color:' + AMBER + '}',
    '.cls-q-log-line.ok .cls-q-log-txt{color:' + GREEN + '}',

    // Divider rule
    '.cls-q-rule{border:none;border-top:1px dashed ' + BORDER + ';margin:4px 0 10px}',

    // Stats block
    '.cls-q-stats{display:grid;grid-template-columns:auto 1fr;column-gap:14px;row-gap:4px;margin-bottom:10px}',
    '.cls-q-k{color:' + MUTED + ';font-size:11px;letter-spacing:0.5px}',
    '.cls-q-v{color:' + TEXT + ';font-size:11.5px;display:flex;align-items:center;gap:6px}',
    '.cls-q-v.waiting{color:' + ACCENT + ';font-weight:700;font-size:13px}',
    '.cls-q-v.active{color:' + GREEN + ';font-weight:700;font-size:13px}',
    '.cls-q-v.amber{color:' + AMBER + ';font-size:11px}',
    '.cls-q-dot{width:7px;height:7px;border-radius:50%;background:' + GREEN + ';box-shadow:0 0 0 0 rgba(16,185,129,0.7);animation:cls-q-pulse 1.8s ease-out infinite}',

    // Input line
    '.cls-q-input-row{display:flex;align-items:center;gap:6px;padding:8px 10px;background:' + BG_ALT + ';border:1px solid ' + BORDER + ';border-radius:6px;transition:border-color .15s}',
    '.cls-q-input-row:focus-within{border-color:' + ACCENT + '}',
    '.cls-q-caret{color:' + ACCENT + ';font-weight:700;flex-shrink:0}',
    '.cls-q-input{flex:1;background:transparent;border:none;outline:none;color:' + TEXT + ';font:inherit;font-size:11.5px;padding:0;caret-color:' + ACCENT + ';min-width:0}',
    '.cls-q-input::placeholder{color:' + MUTED + ';opacity:0.6}',
    '.cls-q-hint{color:' + MUTED + ';font-size:10px;margin-top:6px;text-align:left}',
    '.cls-q-hint kbd{background:' + BG_ALT + ';border:1px solid ' + BORDER + ';border-radius:3px;padding:1px 5px;font-size:9px;font-family:inherit;color:' + TEXT + '}',

    // Error line
    '.cls-q-err{color:' + RED + ';font-size:10.5px;margin-top:6px;min-height:12px;display:flex;align-items:flex-start;gap:6px}',
    '.cls-q-err:not(:empty)::before{content:"!";color:' + RED + ';font-weight:700}',

    // Reopen chip
    '.cls-q-reopen{position:fixed;right:20px;bottom:20px;z-index:9998;background:' + BG + ';color:' + TEXT + ';border:1px solid ' + BORDER + ';border-radius:999px;padding:9px 16px;font-family:"JetBrains Mono","SF Mono",Menlo,monospace;font-size:11px;font-weight:600;cursor:pointer;box-shadow:0 10px 30px rgba(0,0,0,0.4);display:none;align-items:center;gap:8px;pointer-events:auto}',
    '.cls-q-reopen:hover{border-color:' + ACCENT + ';color:' + ACCENT + '}',
    '.cls-q-reopen::before{content:"";width:7px;height:7px;border-radius:50%;background:' + GREEN + ';box-shadow:0 0 0 0 rgba(16,185,129,0.6);animation:cls-q-pulse 1.8s ease-out infinite}',

    // Animations
    '@keyframes cls-q-pulse{0%{box-shadow:0 0 0 0 rgba(16,185,129,0.65)}70%{box-shadow:0 0 0 7px rgba(16,185,129,0)}100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}}',
    '@keyframes cls-q-slidein{0%{opacity:0;transform:translateY(8px)}100%{opacity:1;transform:translateY(0)}}',
    '@keyframes cls-q-slideout{0%{opacity:1;transform:translateY(0)}100%{opacity:0;transform:translateY(-8px)}}',
    '@keyframes cls-q-blink{0%,45%{opacity:1}50%,95%{opacity:0}100%{opacity:1}}',

    // Responsive: bottom-right on narrower viewports
    '@media (max-width:1279px){.cls-q-root{right:16px;top:auto;bottom:16px;transform:none;width:300px}}',
    '@media (max-width:640px){.cls-q-root{width:280px;right:12px;bottom:12px}.cls-q-log{height:88px}}'
  ].join('');

  var style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  // ── Mount ────────────────────────────────────────────────────────────────
  document.body.classList.add('cls-q-active');

  var root = document.createElement('div');
  root.className = 'cls-q-root';
  root.innerHTML =
    '<div class="cls-q-term">' +
    '  <div class="cls-q-tbar">' +
    '    <div class="cls-q-tbar-dots"><div class="cls-q-tdot r"></div><div class="cls-q-tdot y"></div><div class="cls-q-tdot g"></div></div>' +
    '    <div class="cls-q-ttitle"><b>cls.queue</b> — /live</div>' +
    '    <div class="cls-q-tlive">live</div>' +
    '  </div>' +
    '  <div class="cls-q-body">' +
    '    <div class="cls-q-cmd"><span class="p">$</span> <span class="x">watch -n1 queue/status</span></div>' +
    '    <div class="cls-q-log" id="clsQLog"></div>' +
    '    <hr class="cls-q-rule"/>' +
    '    <div class="cls-q-stats">' +
    '      <div class="cls-q-k">WAITING</div><div class="cls-q-v waiting" id="clsQWaiting">—</div>' +
    '      <div class="cls-q-k">ACTIVE</div><div class="cls-q-v active"><span id="clsQActive">—</span> <span class="cls-q-dot"></span></div>' +
    '      <div class="cls-q-k">NEXT WAVE</div><div class="cls-q-v amber" id="clsQNext">5 seats · Mon 09:00</div>' +
    '    </div>' +
    '    <hr class="cls-q-rule"/>' +
    '    <div id="clsQInputZone">' +
    '      <form class="cls-q-input-row" id="clsQFormEmail" autocomplete="off">' +
    '        <span class="cls-q-caret">&gt;</span>' +
    '        <input type="email" class="cls-q-input" id="clsQEmail" placeholder="you@company.com" required autocomplete="email">' +
    '      </form>' +
    '      <div class="cls-q-hint" id="clsQHint"><kbd>enter</kbd> to claim seat</div>' +
    '      <div class="cls-q-err" id="clsQErr"></div>' +
    '    </div>' +
    '  </div>' +
    '</div>';
  document.body.appendChild(root);

  // Reopen chip
  var reopen = document.createElement('button');
  reopen.className = 'cls-q-reopen';
  reopen.textContent = 'cls.queue · reopen';
  document.body.appendChild(reopen);

  // Element refs
  var termEl = root.querySelector('.cls-q-term');
  var logEl = document.getElementById('clsQLog');
  var waitEl = document.getElementById('clsQWaiting');
  var activeEl = document.getElementById('clsQActive');
  var nextEl = document.getElementById('clsQNext');
  var inputZone = document.getElementById('clsQInputZone');
  var formEmail = document.getElementById('clsQFormEmail');
  var emailInput = document.getElementById('clsQEmail');
  var hintEl = document.getElementById('clsQHint');
  var errEl = document.getElementById('clsQErr');

  // Close button in title bar (r dot = close)
  root.querySelector('.cls-q-tdot.r').style.cursor = 'pointer';
  root.querySelector('.cls-q-tdot.r').addEventListener('click', function () {
    root.style.display = 'none';
    reopen.style.display = 'inline-flex';
    document.body.classList.remove('cls-q-active');
  });
  reopen.addEventListener('click', function () {
    root.style.display = '';
    reopen.style.display = 'none';
    document.body.classList.add('cls-q-active');
  });

  // ── Format helpers ───────────────────────────────────────────────────────
  function fmt(n) {
    if (n == null || isNaN(n)) return '—';
    n = Math.max(0, Math.floor(n));
    if (n >= 100000) return (n / 1000).toFixed(0) + 'k';
    if (n >= 10000) return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
    return n.toLocaleString();
  }

  function timeNow() {
    var d = new Date();
    return (
      String(d.getHours()).padStart(2, '0') + ':' +
      String(d.getMinutes()).padStart(2, '0') + ':' +
      String(d.getSeconds()).padStart(2, '0')
    );
  }

  function timeAgo(sec) {
    var d = new Date(Date.now() - sec * 1000);
    return (
      String(d.getHours()).padStart(2, '0') + ':' +
      String(d.getMinutes()).padStart(2, '0') + ':' +
      String(d.getSeconds()).padStart(2, '0')
    );
  }

  // ── Log stream ───────────────────────────────────────────────────────────
  var MAX_LINES = 5;
  var currentWaiting = 0;
  var streamOffset = 0; // decrements to show different recent positions

  function addLine(opts) {
    var line = document.createElement('div');
    line.className = 'cls-q-log-line' + (opts.cls ? ' ' + opts.cls : '');
    var posHtml = opts.pos ? '<span class="cls-q-log-pos">' + opts.pos + '</span>' : '';
    line.innerHTML =
      '<span class="cls-q-log-time">[' + opts.time + ']</span>' +
      '<span class="cls-q-log-txt">' + opts.txt + '</span>' +
      posHtml;
    logEl.appendChild(line);

    // Trim old lines synchronously so the loop can never run away
    while (logEl.children.length > MAX_LINES) {
      logEl.removeChild(logEl.firstChild);
    }
  }

  function seedLog() {
    logEl.innerHTML = '';
    var base = Math.max(1, currentWaiting);
    // Seed 5 historical entries with descending positions and staggered times
    var seeds = [
      { age: 58, txt: 'user joined', pos: '#' + (base - 4), cls: '' },
      { age: 41, txt: 'user joined', pos: '#' + (base - 3), cls: '' },
      { age: 29, txt: '─ seat granted ✓', pos: '#' + (Math.floor(Math.random() * 8) + 1), cls: 'hl' },
      { age: 14, txt: 'user joined', pos: '#' + (base - 1), cls: '' },
      { age: 2, txt: 'user joined', pos: '#' + base, cls: '' }
    ];
    seeds.forEach(function (s) {
      addLine({ time: timeAgo(s.age), txt: s.txt, pos: s.pos, cls: s.cls });
    });
  }

  var tickCount = 0;
  var tickPos = 0;
  function tick() {
    tickCount++;
    // Every 5th tick: seat granted (orange highlight)
    if (tickCount % 5 === 0) {
      addLine({
        time: timeNow(),
        txt: '─ seat granted ✓',
        pos: '#' + (Math.floor(Math.random() * 8) + 1),
        cls: 'hl'
      });
      return;
    }
    // Regular join: drift the position by 1-3 each tick so numbers move
    // visibly. Stay within a plausible neighborhood of the real count.
    tickPos = (tickPos + 1 + Math.floor(Math.random() * 3)) % 12;
    var pos = Math.max(1, currentWaiting - tickPos);
    addLine({ time: timeNow(), txt: 'user joined', pos: '#' + pos, cls: '' });
  }

  // ── Stats ────────────────────────────────────────────────────────────────
  function setStats(data) {
    if (typeof data.waiting_count === 'number') {
      currentWaiting = data.waiting_count;
      waitEl.textContent = fmt(data.waiting_count);
    }
    if (typeof data.active_now === 'number') {
      activeEl.textContent = fmt(data.active_now);
    }
    if (data.your_position && data.your_status !== 'activated') {
      enterSuccessState(data.your_email || getStoredEmail() || '', data.your_position);
    }
  }

  function getStoredEmail() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return ''; }
  }

  // ── Fetch stats ──────────────────────────────────────────────────────────
  async function refreshStats() {
    var saved = getStoredEmail();
    var url = '/v1/waitlist/stats' + (saved ? '?email=' + encodeURIComponent(saved) : '');
    try {
      var r = await fetch(url, { credentials: 'same-origin' });
      if (!r.ok) return;
      var data = await r.json();
      data.your_email = saved;
      setStats(data);
    } catch (e) {}
  }

  async function apiJson(url, opts) {
    var r = await fetch(url, Object.assign({
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin'
    }, opts || {}));
    var data = null;
    try { data = await r.json(); } catch (e) {}
    return { ok: r.ok, status: r.status, data: data || {} };
  }

  // ── Input state machine ──────────────────────────────────────────────────
  var pendingEmail = '';

  function showEmailInput() {
    inputZone.innerHTML =
      '<form class="cls-q-input-row" id="clsQFormEmail" autocomplete="off">' +
      '  <span class="cls-q-caret">&gt;</span>' +
      '  <input type="email" class="cls-q-input" id="clsQEmail" placeholder="you@company.com" required autocomplete="email">' +
      '</form>' +
      '<div class="cls-q-hint"><kbd>enter</kbd> to claim seat</div>' +
      '<div class="cls-q-err" id="clsQErr"></div>';
    bindEmailForm();
  }

  function showOtpInput(email) {
    inputZone.innerHTML =
      '<form class="cls-q-input-row" id="clsQFormOtp" autocomplete="off">' +
      '  <span class="cls-q-caret">&gt;</span>' +
      '  <input type="text" class="cls-q-input" id="clsQOtp" inputmode="numeric" pattern="[0-9]{6}" maxlength="6" placeholder="6-digit code" required>' +
      '</form>' +
      '<div class="cls-q-hint">code sent to <strong style="color:' + TEXT + '">' + email + '</strong></div>' +
      '<div class="cls-q-err" id="clsQErr"></div>';
    bindOtpForm();
    setTimeout(function () { document.getElementById('clsQOtp').focus(); }, 50);
  }

  function enterSuccessState(email, position) {
    inputZone.innerHTML =
      '<div style="padding:10px 12px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);border-radius:6px;font-size:11.5px;">' +
      '<div style="color:' + GREEN + ';font-weight:700;margin-bottom:4px;">✓ seat reserved</div>' +
      '<div style="color:' + TEXT + ';">you are <strong style="color:' + ACCENT + '">#' + position + '</strong> in line</div>' +
      '<div style="color:' + MUTED + ';font-size:10px;margin-top:4px;">check <span style="color:' + TEXT + '">' + (email || 'your inbox') + '</span> when it\'s your turn</div>' +
      '</div>';
    try { if (email) localStorage.setItem(STORAGE_KEY, email); } catch (e) {}
  }

  function flashLog(txt, cls) {
    addLine({ time: timeNow(), txt: txt, pos: '', cls: cls || 'sys' });
  }

  function bindEmailForm() {
    var form = document.getElementById('clsQFormEmail');
    var input = document.getElementById('clsQEmail');
    var err = document.getElementById('clsQErr');
    form.addEventListener('submit', async function (e) {
      e.preventDefault();
      err.textContent = '';
      var email = (input.value || '').trim();
      if (!email || email.indexOf('@') < 0) {
        err.textContent = 'invalid email format';
        return;
      }
      input.disabled = true;
      flashLog('→ validating ' + email);
      var res = await apiJson('/v1/waitlist/join', {
        method: 'POST',
        body: JSON.stringify({ email: email })
      });
      input.disabled = false;
      if (!res.ok) {
        var msg = (res.data && (res.data.message || res.data.detail)) || 'request failed';
        err.textContent = msg.toLowerCase().slice(0, 64);
        flashLog('✗ ' + msg.toLowerCase().slice(0, 48), 'sys');
        return;
      }
      if (res.data.status === 'already_member') {
        err.textContent = 'account exists — try logging in';
        return;
      }
      if (res.data.status === 'waiting' || res.data.status === 'invited') {
        flashLog('✓ already in queue', 'ok');
        enterSuccessState(email, res.data.position || '?');
        return;
      }
      flashLog('✓ code dispatched', 'ok');
      pendingEmail = email;
      showOtpInput(email);
    });
  }

  function bindOtpForm() {
    var form = document.getElementById('clsQFormOtp');
    var input = document.getElementById('clsQOtp');
    var err = document.getElementById('clsQErr');
    form.addEventListener('submit', async function (e) {
      e.preventDefault();
      err.textContent = '';
      var code = (input.value || '').trim();
      if (!/^[0-9]{6}$/.test(code)) {
        err.textContent = 'need 6 digits';
        return;
      }
      input.disabled = true;
      flashLog('→ verifying code');
      var res = await apiJson('/v1/waitlist/verify', {
        method: 'POST',
        body: JSON.stringify({ email: pendingEmail, otp_code: code })
      });
      input.disabled = false;
      if (!res.ok) {
        var msg = (res.data && (res.data.message || res.data.detail)) || 'invalid code';
        err.textContent = msg.toLowerCase().slice(0, 64);
        return;
      }
      if (typeof res.data.waiting_count === 'number') {
        currentWaiting = res.data.waiting_count;
        waitEl.textContent = fmt(res.data.waiting_count);
      }
      flashLog('✓ seat reserved · #' + (res.data.position || '?'), 'hl');
      enterSuccessState(pendingEmail, res.data.position || '?');
    });
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  bindEmailForm();
  refreshStats().then(function () {
    seedLog();
    setInterval(tick, STREAM_TICK_MS);
  });
  setInterval(refreshStats, STATS_POLL_MS);
})();
