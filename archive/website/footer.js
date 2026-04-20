/* CLS++ Shared Footer Sitemap — injected on every page */
(function() {
  'use strict';

  var footer = document.createElement('footer');
  footer.className = 'site-footer';
  footer.innerHTML =
    '<div class="sf-inner">' +

      /* ── Column 1: Brand ── */
      '<div class="sf-col sf-brand-col">' +
        '<div class="sf-brand">CLS<span class="sf-plus">++</span></div>' +
        '<p class="sf-tagline">Every AI remembers you.</p>' +
        '<p class="sf-corp">AlphaForge AI Labs</p>' +
        '<p class="sf-patent">Provisional patent filed Oct 2025</p>' +
      '</div>' +

      /* ── Column 2: Product ── */
      '<div class="sf-col">' +
        '<h4 class="sf-heading">Product</h4>' +
        '<a href="/">Home</a>' +
        '<a href="/getting-started.html">Get Started</a>' +
        '<a href="/integrate.html">Integrate</a>' +
        '<a href="/install.html">Activate Memory</a>' +
        '<a href="/#pricing">Pricing</a>' +
        '<a href="/demo.html">Live Demo</a>' +
        '<a href="/benchmark.html">Benchmarks</a>' +
      '</div>' +

      /* ── Column 3: Your Account ── */
      '<div class="sf-col">' +
        '<h4 class="sf-heading">Your Account</h4>' +
        '<a href="/dashboard.html">Dashboard</a>' +
        '<a href="/profile.html">Profile</a>' +
        '<a href="/profile.html#keys">API Keys</a>' +
        '<a href="/profile.html#billing">Billing</a>' +
        '<a href="/usage.html">Usage</a>' +
        '<a href="/memory.html">Memory Viewer</a>' +
      '</div>' +

      /* ── Column 4: Developer ── */
      '<div class="sf-col">' +
        '<h4 class="sf-heading">Developer</h4>' +
        '<a href="/docs.html">API Documentation</a>' +
        '<a href="/docs">Swagger UI</a>' +
        '<a href="/chat.html">Chat Playground</a>' +
        '<a href="/trace.html">Request Traces</a>' +
        '<a href="/tests.html">Test Suite</a>' +
        '<a href="https://github.com/rajamohan1950/CLSplusplus" target="_blank" rel="noopener">GitHub</a>' +
      '</div>' +

      /* ── Column 5: Company ── */
      '<div class="sf-col">' +
        '<h4 class="sf-heading">Company</h4>' +
        '<a href="/support.html">Support</a>' +
        '<a href="mailto:contact@alphaforge.ai">Contact</a>' +
        '<a href="/terms.html">Terms of Service</a>' +
        '<a href="/privacy.html">Privacy Policy</a>' +
      '</div>' +

    '</div>' +

    /* ── Bottom bar ── */
    '<div class="sf-bottom">' +
      '<span>&copy; ' + new Date().getFullYear() + ' AlphaForge AI Labs. All rights reserved.</span>' +
    '</div>';

  /* ── Inject styles ── */
  var style = document.createElement('style');
  style.textContent =
    '.site-footer{' +
      'position:relative;z-index:2;' +
      'max-width:980px;margin:0 auto;' +
      'background:rgba(29,29,31,0.92);' +
      'border-radius:24px;' +
      'box-shadow:0 4px 40px rgba(0,0,0,0.12);' +
      'padding:48px 40px 24px;' +
      'margin-bottom:24px;' +
      'color:rgba(255,255,255,0.7);' +
      'font-family:"Inter",-apple-system,BlinkMacSystemFont,sans-serif;' +
      '-webkit-font-smoothing:antialiased;' +
    '}' +
    '.sf-inner{' +
      'display:grid;' +
      'grid-template-columns:1.5fr repeat(4,1fr);' +
      'gap:32px;' +
    '}' +
    '.sf-col{display:flex;flex-direction:column;gap:8px;}' +
    '.sf-brand{font-size:20px;font-weight:800;color:#fff;letter-spacing:-0.02em;}' +
    '.sf-plus{color:#ff6b35;}' +
    '.sf-tagline{font-size:13px;color:rgba(255,255,255,0.5);margin:4px 0 8px;line-height:1.4;}' +
    '.sf-corp{font-size:11px;color:rgba(255,255,255,0.3);margin:0;}' +
    '.sf-patent{font-size:11px;color:rgba(255,255,255,0.2);margin:2px 0 0;}' +
    '.sf-heading{' +
      'font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;' +
      'color:rgba(255,255,255,0.4);margin:0 0 4px;' +
    '}' +
    '.sf-col a{' +
      'font-size:13px;color:rgba(255,255,255,0.55);text-decoration:none;' +
      'transition:color 0.15s;line-height:1.4;' +
    '}' +
    '.sf-col a:hover{color:#ff6b35;}' +
    '.sf-bottom{' +
      'margin-top:32px;padding-top:16px;' +
      'border-top:1px solid rgba(255,255,255,0.08);' +
      'font-size:11px;color:rgba(255,255,255,0.2);' +
      'text-align:center;' +
    '}' +

    /* Responsive */
    '@media(max-width:768px){' +
      '.sf-inner{grid-template-columns:1fr 1fr;gap:24px;}' +
      '.sf-brand-col{grid-column:1/-1;}' +
    '}' +
    '@media(max-width:480px){' +
      '.sf-inner{grid-template-columns:1fr;}' +
      '.site-footer{padding:32px 20px 16px;}' +
    '}';

  document.head.appendChild(style);

  /* ── Insert at very end of body ── */
  document.body.appendChild(footer);
})();
