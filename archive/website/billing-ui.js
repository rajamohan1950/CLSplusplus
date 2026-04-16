/* CLS++ Billing UI — Payment overlay, success celebration, failure handling */
(function () {
  'use strict';

  // ── Payment Loading Overlay ──────────────────────────────────────────────

  window.showPaymentOverlay = function () {
    if (document.getElementById('payment-overlay')) return;
    var overlay = document.createElement('div');
    overlay.id = 'payment-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;z-index:9998;background:rgba(10,10,15,0.92);'
      + 'display:flex;flex-direction:column;align-items:center;justify-content:center;'
      + 'backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);'
      + 'opacity:0;transition:opacity 0.3s ease;';

    overlay.innerHTML = '<div style="text-align:center;">'
      + '<div class="payment-spinner"></div>'
      + '<p style="font-size:1.05rem;margin-top:20px;font-weight:600;color:var(--text,#e8e8ed);">Preparing secure checkout...</p>'
      + '<p style="font-size:0.82rem;color:var(--text-muted,#8b8b9a);margin-top:8px;">This will only take a moment.</p>'
      + '</div>';

    document.body.appendChild(overlay);
    requestAnimationFrame(function () { overlay.style.opacity = '1'; });
  };

  window.hidePaymentOverlay = function () {
    var el = document.getElementById('payment-overlay');
    if (el) {
      el.style.opacity = '0';
      setTimeout(function () { if (el.parentNode) el.remove(); }, 300);
    }
  };

  // ── Success Celebration ──────────────────────────────────────────────────

  window.showBillingSuccess = function () {
    if (document.getElementById('success-celebration')) return;
    var overlay = document.createElement('div');
    overlay.id = 'success-celebration';
    overlay.style.cssText = 'position:fixed;inset:0;z-index:9999;background:rgba(10,10,15,0.95);'
      + 'display:flex;flex-direction:column;align-items:center;justify-content:center;'
      + 'backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);'
      + 'opacity:0;transition:opacity 0.4s ease;overflow:hidden;';

    overlay.innerHTML = '<div style="text-align:center;position:relative;z-index:2;">'
      + '<div class="success-check-anim">'
      + '<svg width="88" height="88" viewBox="0 0 88 88">'
      + '<circle cx="44" cy="44" r="40" fill="none" stroke="#22c55e" stroke-width="2.5" class="check-circle"/>'
      + '<path d="M26 46 L38 58 L62 34" fill="none" stroke="#22c55e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="check-mark"/>'
      + '</svg>'
      + '</div>'
      + '<h2 style="font-size:1.5rem;margin-top:24px;color:#22c55e;font-weight:700;">Welcome to your new plan!</h2>'
      + '<p style="font-size:0.95rem;color:var(--text-muted,#8b8b9a);margin-top:10px;max-width:360px;">Your subscription is now active. Enjoy your upgraded limits and features.</p>'
      + '<button class="btn btn-primary" id="btn-celebration-continue" style="margin-top:28px;padding:10px 32px;font-size:0.95rem;">Continue</button>'
      + '</div>';

    // Confetti particles
    var confettiColors = ['#22c55e', '#6366f1', '#f59e0b', '#818cf8', '#ec4899', '#14b8a6', '#f97316'];
    for (var i = 0; i < 40; i++) {
      var p = document.createElement('div');
      p.className = 'confetti-particle';
      p.style.left = (Math.random() * 100) + '%';
      p.style.animationDelay = (Math.random() * 2.5) + 's';
      p.style.animationDuration = (2.5 + Math.random() * 2) + 's';
      p.style.background = confettiColors[Math.floor(Math.random() * confettiColors.length)];
      p.style.width = (5 + Math.random() * 6) + 'px';
      p.style.height = (5 + Math.random() * 6) + 'px';
      overlay.appendChild(p);
    }

    document.body.appendChild(overlay);
    requestAnimationFrame(function () { overlay.style.opacity = '1'; });

    document.getElementById('btn-celebration-continue').addEventListener('click', function () {
      overlay.style.opacity = '0';
      setTimeout(function () { if (overlay.parentNode) overlay.remove(); }, 400);
    });
  };

  // ── Cancel / Failure Toasts ──────────────────────────────────────────────

  window.showBillingCancel = function () {
    if (typeof showToast === 'function') {
      showToast('Checkout was cancelled. No charges were made.', 'warning', 5000);
    }
  };

  window.showBillingFailure = function (detail) {
    if (typeof showToast === 'function') {
      showToast(detail || 'Payment could not be processed. Please try again or use a different payment method.', 'error', 8000);
    }
  };
})();
