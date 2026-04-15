/**
 * CLS++ Feedback Widget
 *
 * Floating bottom-right widget that appears after 30s on landing pages.
 * Collects 1-click emoji rating + optional comment.
 * Fires PostHog event: `landing_feedback` with variant, rating, comment.
 * Shows only once per user (cls_feedback_sent cookie).
 *
 * Include on any page:
 *   <script src="feedback-widget.js" defer></script>
 */
(function () {
  // Only show on landing / index pages
  var path = window.location.pathname;
  var isLanding = path === '/' || path === '/index.html'
    || path === '/landing-d.html' || path === '/landing-e.html';
  if (!isLanding) return;

  // Skip if already submitted
  if (document.cookie.indexOf('cls_feedback_sent=1') !== -1) return;

  var variant = window.CLS_VARIANT || 'A';

  // --- Inject widget after delay ---
  setTimeout(function () {
    var widget = document.createElement('div');
    widget.id = 'cls-feedback';
    widget.innerHTML = ''
      + '<div class="fb-card">'
      + '  <button class="fb-close" id="fb-close">&times;</button>'
      + '  <div class="fb-title">How does this page feel?</div>'
      + '  <div class="fb-ratings" id="fb-ratings">'
      + '    <button class="fb-rate" data-r="5" title="Amazing">😍</button>'
      + '    <button class="fb-rate" data-r="4" title="Good">😊</button>'
      + '    <button class="fb-rate" data-r="3" title="Okay">😐</button>'
      + '    <button class="fb-rate" data-r="2" title="Not great">😕</button>'
      + '    <button class="fb-rate" data-r="1" title="Bad">😞</button>'
      + '  </div>'
      + '  <div class="fb-step2" id="fb-step2" style="display:none">'
      + '    <textarea class="fb-comment" id="fb-comment" placeholder="Any thoughts? (optional)" rows="2" maxlength="500"></textarea>'
      + '    <button class="fb-submit" id="fb-submit">Send Feedback</button>'
      + '  </div>'
      + '  <div class="fb-thanks" id="fb-thanks" style="display:none">'
      + '    <span>Thank you! 🙏</span>'
      + '  </div>'
      + '</div>';

    // --- Styles ---
    var style = document.createElement('style');
    style.textContent = ''
      + '#cls-feedback{'
      + '  position:fixed;bottom:24px;right:24px;z-index:9999;'
      + '  animation:fb-in 0.4s ease;'
      + '}'
      + '@keyframes fb-in{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}'
      + '.fb-card{'
      + '  background:#18181b;border:1px solid rgba(255,255,255,0.08);'
      + '  border-radius:16px;padding:20px 24px;min-width:260px;max-width:320px;'
      + '  box-shadow:0 12px 40px rgba(0,0,0,0.5);color:#f4f4f5;'
      + '  font-family:"Inter",-apple-system,sans-serif;position:relative;'
      + '}'
      + '.fb-close{'
      + '  position:absolute;top:8px;right:12px;background:none;border:none;'
      + '  color:#71717a;font-size:18px;cursor:pointer;padding:4px;'
      + '}'
      + '.fb-close:hover{color:#fff}'
      + '.fb-title{font-size:0.88rem;font-weight:600;margin-bottom:14px}'
      + '.fb-ratings{display:flex;gap:8px;justify-content:center}'
      + '.fb-rate{'
      + '  width:42px;height:42px;border-radius:10px;border:1px solid rgba(255,255,255,0.06);'
      + '  background:#27272a;cursor:pointer;font-size:20px;display:flex;'
      + '  align-items:center;justify-content:center;transition:all 0.15s;'
      + '}'
      + '.fb-rate:hover{transform:scale(1.15);border-color:#ff6b35;background:#ff6b3515}'
      + '.fb-rate.selected{border-color:#ff6b35;background:#ff6b3520;transform:scale(1.1)}'
      + '.fb-comment{'
      + '  width:100%;padding:10px 12px;background:#27272a;border:1px solid rgba(255,255,255,0.06);'
      + '  border-radius:10px;color:#f4f4f5;font-family:inherit;font-size:0.82rem;'
      + '  resize:none;outline:none;margin-bottom:10px;margin-top:14px;'
      + '}'
      + '.fb-comment::placeholder{color:#71717a}'
      + '.fb-comment:focus{border-color:#ff6b35}'
      + '.fb-submit{'
      + '  width:100%;padding:10px;background:#ff6b35;color:#fff;border:none;'
      + '  border-radius:10px;font-family:inherit;font-size:0.85rem;font-weight:600;'
      + '  cursor:pointer;transition:background 0.15s;'
      + '}'
      + '.fb-submit:hover{background:#ff8c5a}'
      + '.fb-thanks{text-align:center;padding:8px 0;font-size:0.92rem}'
      + '@media(max-width:500px){'
      + '  #cls-feedback{bottom:12px;right:12px;left:12px}'
      + '  .fb-card{min-width:auto;max-width:none}'
      + '}';

    document.head.appendChild(style);
    document.body.appendChild(widget);

    // --- Interactions ---
    var selectedRating = 0;

    document.getElementById('fb-ratings').addEventListener('click', function (e) {
      var btn = e.target.closest('.fb-rate');
      if (!btn) return;
      selectedRating = parseInt(btn.dataset.r, 10);
      // Highlight selected
      document.querySelectorAll('.fb-rate').forEach(function (b) { b.classList.remove('selected'); });
      btn.classList.add('selected');
      // Show step 2
      document.getElementById('fb-step2').style.display = 'block';
    });

    document.getElementById('fb-submit').addEventListener('click', function () {
      var comment = (document.getElementById('fb-comment').value || '').trim();

      // Fire PostHog event
      if (window.posthog && typeof window.posthog.capture === 'function') {
        window.posthog.capture('landing_feedback', {
          variant: variant,
          rating: selectedRating,
          comment: comment,
          page: window.location.pathname,
        });
      }

      // Set cookie so we don't show again
      document.cookie = 'cls_feedback_sent=1;path=/;max-age=' + (365 * 24 * 60 * 60) + ';samesite=lax';

      // Show thanks
      document.getElementById('fb-ratings').style.display = 'none';
      document.getElementById('fb-step2').style.display = 'none';
      document.getElementById('fb-thanks').style.display = 'block';

      // Auto-hide after 2s
      setTimeout(function () {
        var el = document.getElementById('cls-feedback');
        if (el) { el.style.transition = 'opacity 0.4s'; el.style.opacity = '0'; setTimeout(function () { el.remove(); }, 400); }
      }, 2000);
    });

    document.getElementById('fb-close').addEventListener('click', function () {
      // Dismiss without feedback — set cookie so it doesn't reappear
      document.cookie = 'cls_feedback_sent=1;path=/;max-age=' + (7 * 24 * 60 * 60) + ';samesite=lax';
      var el = document.getElementById('cls-feedback');
      if (el) { el.style.transition = 'opacity 0.3s'; el.style.opacity = '0'; setTimeout(function () { el.remove(); }, 300); }
    });

  }, 30000); // 30 second delay
})();
