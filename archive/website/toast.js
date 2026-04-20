/* CLS++ Toast Notification System — replaces all alert() calls */
(function () {
  'use strict';

  var container = document.createElement('div');
  container.id = 'toast-container';
  container.setAttribute('aria-live', 'assertive');
  container.style.cssText = 'position:fixed;top:20px;right:20px;z-index:10000;display:flex;flex-direction:column;gap:10px;pointer-events:none;max-width:420px;width:calc(100% - 40px);';

  function ensureContainer() {
    if (!container.parentNode) document.body.appendChild(container);
  }

  var ICONS = {
    success: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M6 10.5l2.5 2.5 5.5-5.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    error: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M7 7l6 6M13 7l-6 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    warning: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 2L1 18h18L10 2z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M10 8v4M10 14.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    info: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M10 9v5M10 6.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
  };

  var COLORS = {
    success: { bg: 'rgba(34,197,94,0.10)', border: 'rgba(34,197,94,0.25)', icon: '#22c55e' },
    error:   { bg: 'rgba(239,68,68,0.10)', border: 'rgba(239,68,68,0.25)', icon: '#ef4444' },
    warning: { bg: 'rgba(245,158,11,0.10)', border: 'rgba(245,158,11,0.25)', icon: '#f59e0b' },
    info:    { bg: 'rgba(99,102,241,0.10)', border: 'rgba(99,102,241,0.25)', icon: '#818cf8' },
  };

  window.showToast = function (message, type, duration) {
    ensureContainer();
    type = type || 'info';
    duration = duration || (type === 'error' ? 6000 : 4000);
    var c = COLORS[type] || COLORS.info;

    var toast = document.createElement('div');
    toast.setAttribute('role', 'alert');
    toast.style.cssText = 'pointer-events:auto;display:flex;align-items:flex-start;gap:12px;'
      + 'padding:14px 18px;background:' + c.bg + ';border:1px solid ' + c.border + ';'
      + 'border-radius:12px;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);'
      + 'color:var(--text,#e8e8ed);font-size:0.875rem;line-height:1.45;font-family:inherit;'
      + 'box-shadow:0 8px 32px rgba(0,0,0,0.3);'
      + 'transform:translateX(120%);transition:transform 0.35s cubic-bezier(0.22,1,0.36,1),opacity 0.35s ease;opacity:0;';

    var iconSpan = document.createElement('span');
    iconSpan.style.cssText = 'color:' + c.icon + ';flex-shrink:0;margin-top:1px;';
    iconSpan.innerHTML = ICONS[type] || ICONS.info;

    var textSpan = document.createElement('span');
    textSpan.textContent = message;

    toast.appendChild(iconSpan);
    toast.appendChild(textSpan);
    container.appendChild(toast);

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        toast.style.transform = 'translateX(0)';
        toast.style.opacity = '1';
      });
    });

    var timer = setTimeout(function () { dismiss(); }, duration);

    toast.addEventListener('click', function () {
      clearTimeout(timer);
      dismiss();
    });

    function dismiss() {
      toast.style.transform = 'translateX(120%)';
      toast.style.opacity = '0';
      setTimeout(function () { if (toast.parentNode) toast.remove(); }, 400);
    }
  };
})();
