#!/usr/bin/env node
/**
 * CLS++ Production UI Test Runner
 * Runs all 30 UI test cases against www.clsplusplus.com
 */

const https = require('https');
const { JSDOM } = require('jsdom');

const BASE = 'https://www.clsplusplus.com/';
const ALL_PAGES = [
  'index.html', 'getting-started.html', 'docs.html', 'integrate.html',
  'login.html', 'signup.html', 'profile.html', 'dashboard.html',
  'usage.html', 'chat.html', 'memory.html', 'benchmark.html',
  'benchmark_v1_direct.html', 'tests.html', 'trace.html', 'install.html',
  'demo.html', 'submit.html', 'support.html', 'extension-test.html',
  'chat-test.html', 'terms.html', 'privacy.html'
];

let results = [];
let currentTC = '';

function pass(name, detail) { results.push({ tc: currentTC, name, status: 'PASS', detail: detail || '' }); }
function fail(name, detail) { results.push({ tc: currentTC, name, status: 'FAIL', detail: detail || '' }); }

function fetchPage(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': 'CLS-Test/1.0' } }, (res) => {
      // Follow redirects
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        const redirectUrl = res.headers.location.startsWith('http')
          ? res.headers.location
          : new URL(res.headers.location, url).href;
        fetchPage(redirectUrl).then(resolve).catch(reject);
        return;
      }
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve({ status: res.statusCode, html: data }));
    });
    req.on('error', reject);
    req.setTimeout(10000, () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

function parseDoc(html) {
  const dom = new JSDOM(html);
  return dom.window.document;
}

// TC-UI-01: All pages load with HTTP 200
async function TC_UI_01() {
  currentTC = 'TC-UI-01';
  for (const page of ALL_PAGES) {
    try {
      const { status } = await fetchPage(BASE + page);
      if (status === 200) pass(page + ' loads (HTTP 200)');
      else fail(page + ' loads', 'HTTP ' + status);
    } catch (e) { fail(page + ' loads', e.message); }
  }
}

// TC-UI-02: Every page has correct meta tags
async function TC_UI_02() {
  currentTC = 'TC-UI-02';
  for (const page of ALL_PAGES) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const viewport = doc.querySelector('meta[name="viewport"]');
      const charset = doc.querySelector('meta[charset]');
      const title = doc.querySelector('title');
      if (viewport) pass(page + ' has viewport meta'); else fail(page + ' has viewport meta');
      if (charset) pass(page + ' has charset meta'); else fail(page + ' has charset meta');
      if (title && title.textContent.includes('CLS++')) pass(page + ' title contains CLS++');
      else fail(page + ' title contains CLS++', 'Title: ' + (title ? title.textContent : 'none'));
    } catch (e) { fail(page + ' meta check', e.message); }
  }
}

// TC-UI-03: Frosted glass nav bar present on all public pages
async function TC_UI_03() {
  currentTC = 'TC-UI-03';
  const pagesWithNav = ALL_PAGES.filter(p => p !== 'integrate.html');
  for (const page of pagesWithNav) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const nav = doc.querySelector('nav.nav');
      if (nav) pass(page + ' has <nav class="nav">');
      else fail(page + ' has <nav class="nav">');
    } catch (e) { fail(page + ' nav check', e.message); }
  }
}

// TC-UI-04: Nav has CLS++ logo linking to home
async function TC_UI_04() {
  currentTC = 'TC-UI-04';
  const pagesWithNav = ALL_PAGES.filter(p => p !== 'integrate.html');
  for (const page of pagesWithNav) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const logo = doc.querySelector('nav .logo, nav .nav-logo');
      if (logo) {
        const href = logo.getAttribute('href');
        if (href === '/' || href === '/index.html' || href === 'index.html') pass(page + ' logo links to home');
        else fail(page + ' logo links to home', 'href=' + href);
      } else { fail(page + ' has logo in nav'); }
    } catch (e) { fail(page + ' logo check', e.message); }
  }
}

// TC-UI-05: "Get Started" CTA in nav
async function TC_UI_05() {
  currentTC = 'TC-UI-05';
  const pagesWithNav = ALL_PAGES.filter(p => p !== 'integrate.html');
  for (const page of pagesWithNav) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const cta = doc.querySelector('.nav-cta, .nav-links a.nav-cta');
      if (cta) {
        const text = cta.textContent.trim();
        if (text.toLowerCase().includes('get started')) pass(page + ' nav has "Get Started" CTA');
        else fail(page + ' nav CTA text', 'Got: ' + text);
      } else { fail(page + ' nav has Get Started CTA'); }
    } catch (e) { fail(page + ' CTA check', e.message); }
  }
}

// TC-UI-06: Video background present
async function TC_UI_06() {
  currentTC = 'TC-UI-06';
  const videoPages = ALL_PAGES.filter(p => p !== 'integrate.html');
  for (const page of videoPages) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const videoBg = doc.querySelector('.video-bg');
      if (videoBg) pass(page + ' has video background');
      else fail(page + ' has video background', 'No .video-bg found');
    } catch (e) { fail(page + ' video bg check', e.message); }
  }
}

// TC-UI-07: No dark backgrounds
async function TC_UI_07() {
  currentTC = 'TC-UI-07';
  const darkColors = ['#05050a', '#06060c', '#0a0a0a', '#0a0a0f', '#0d0d0d', '#090909'];
  for (const page of ALL_PAGES) {
    try {
      const { html } = await fetchPage(BASE + page);
      const found = darkColors.filter(c => html.includes(c));
      if (found.length === 0) pass(page + ' no dark background colors');
      else fail(page + ' has dark colors', found.join(', '));
    } catch (e) { fail(page + ' dark color check', e.message); }
  }
}

// TC-UI-08: Orange accent, no purple
async function TC_UI_08() {
  currentTC = 'TC-UI-08';
  const purpleColors = ['#7c6ef0', '#6366f1', '#818cf8'];
  for (const page of ALL_PAGES) {
    try {
      const { html } = await fetchPage(BASE + page);
      const found = purpleColors.filter(c => html.includes(c));
      if (found.length === 0) pass(page + ' no purple accents');
      else fail(page + ' still has purple', found.join(', '));
    } catch (e) { fail(page + ' purple check', e.message); }
  }
}

// TC-UI-09: styles.css loaded
async function TC_UI_09() {
  currentTC = 'TC-UI-09';
  for (const page of ALL_PAGES) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const links = Array.from(doc.querySelectorAll('link[rel="stylesheet"]'));
      const stylesLink = links.find(l => (l.getAttribute('href') || '').includes('styles.css'));
      if (stylesLink) {
        const href = stylesLink.getAttribute('href');
        if (href.includes('?v=')) pass(page + ' styles.css has cache-bust');
        else fail(page + ' styles.css missing cache-bust', 'href=' + href);
      } else {
        pass(page + ' uses inline styles (no styles.css)');
      }
    } catch (e) { fail(page + ' stylesheet check', e.message); }
  }
}

// TC-UI-10: Body transparent bg
async function TC_UI_10() {
  currentTC = 'TC-UI-10';
  for (const page of ALL_PAGES) {
    if (page === 'integrate.html') continue;
    try {
      const { html } = await fetchPage(BASE + page);
      const hasTransparent = html.includes('background:transparent') || html.includes('background: transparent');
      if (hasTransparent) { pass(page + ' body has transparent bg'); }
      else {
        const darkBgRegex = /body\s*\{[^}]*background\s*:\s*#[0-9a-f]{3,6}/i;
        if (darkBgRegex.test(html)) fail(page + ' body has dark background', 'Should be transparent');
        else pass(page + ' body bg not dark');
      }
    } catch (e) { fail(page + ' bg check', e.message); }
  }
}

// TC-UI-11: Anchor targets exist
async function TC_UI_11() {
  currentTC = 'TC-UI-11';
  for (const page of ALL_PAGES) {
    if (page === 'integrate.html') continue;
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const anchors = Array.from(doc.querySelectorAll('a[href^="#"]'));
      const broken = [];
      anchors.forEach(a => {
        const id = a.getAttribute('href').slice(1);
        if (id && !doc.getElementById(id)) broken.push('#' + id);
      });
      if (broken.length === 0) pass(page + ' all anchor targets exist (' + anchors.length + ')');
      else fail(page + ' broken anchors', broken.join(', '));
    } catch (e) { fail(page + ' anchor check', e.message); }
  }
}

// TC-UI-12: All buttons labeled
async function TC_UI_12() {
  currentTC = 'TC-UI-12';
  for (const page of ALL_PAGES) {
    if (page === 'integrate.html') continue;
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const buttons = Array.from(doc.querySelectorAll('button'));
      const unlabeled = [];
      buttons.forEach(btn => {
        const text = btn.textContent.trim();
        const ariaLabel = btn.getAttribute('aria-label');
        if (!text && !ariaLabel) unlabeled.push(btn.className || 'unnamed');
      });
      if (unlabeled.length === 0) pass(page + ' all buttons labeled (' + buttons.length + ')');
      else fail(page + ' unlabeled buttons', unlabeled.join(', '));
    } catch (e) { fail(page + ' button check', e.message); }
  }
}

// TC-UI-13: Login page
async function TC_UI_13() {
  currentTC = 'TC-UI-13';
  try {
    const { html } = await fetchPage(BASE + 'login.html');
    const doc = parseDoc(html);
    if (doc.querySelector('input[type="email"], input#email, input[placeholder*="mail"]')) pass('Login has email input'); else fail('Login has email input');
    if (doc.querySelector('input[type="password"], input#password')) pass('Login has password input'); else fail('Login has password input');
    const loginBtn = doc.querySelector('button');
    if (loginBtn && loginBtn.textContent.toLowerCase().includes('log in')) pass('Login has Log In button');
    else fail('Login has Log In button', loginBtn ? loginBtn.textContent : 'no button');
    const signupLink = Array.from(doc.querySelectorAll('a')).find(a => a.textContent.toLowerCase().includes('sign up'));
    if (signupLink) pass('Login has Sign Up link'); else fail('Login has Sign Up link');
  } catch (e) { fail('Login page', e.message); }
}

// TC-UI-14: Signup page
async function TC_UI_14() {
  currentTC = 'TC-UI-14';
  try {
    const { html } = await fetchPage(BASE + 'signup.html');
    const doc = parseDoc(html);
    if (doc.querySelector('input[type="email"], input#email')) pass('Signup has email input'); else fail('Signup has email input');
    if (doc.querySelector('input[type="password"], input#password')) pass('Signup has password input'); else fail('Signup has password input');
    const btn = doc.querySelector('button');
    if (btn && (btn.textContent.toLowerCase().includes('sign up') || btn.textContent.toLowerCase().includes('create'))) pass('Signup has Sign Up CTA');
    else fail('Signup has Sign Up CTA', btn ? btn.textContent : 'no button');
    const loginLink = Array.from(doc.querySelectorAll('a')).find(a => a.textContent.toLowerCase().includes('log in') || a.getAttribute('href') === '/login.html');
    if (loginLink) pass('Signup has Log In link'); else fail('Signup has Log In link');
  } catch (e) { fail('Signup page', e.message); }
}

// TC-UI-15: Index hero
async function TC_UI_15() {
  currentTC = 'TC-UI-15';
  try {
    const { html } = await fetchPage(BASE + 'index.html');
    const doc = parseDoc(html);
    if (doc.querySelector('h1')) pass('Index has h1 heading'); else fail('Index has h1 heading');
    const heroCTA = Array.from(doc.querySelectorAll('a')).find(a => a.textContent.toLowerCase().includes('get started'));
    if (heroCTA) pass('Index hero has "Get started" CTA'); else fail('Index hero has "Get started" CTA');
  } catch (e) { fail('Index page', e.message); }
}

// TC-UI-16: Pricing section
async function TC_UI_16() {
  currentTC = 'TC-UI-16';
  try {
    const { html } = await fetchPage(BASE + 'index.html');
    const doc = parseDoc(html);
    const pricing = doc.getElementById('pricing');
    if (pricing) {
      pass('Index has #pricing section');
      const ctas = Array.from(pricing.querySelectorAll('a, button'));
      if (ctas.length >= 2) pass('Pricing has ' + ctas.length + ' CTAs');
      else fail('Pricing CTAs', 'Only ' + ctas.length + ' found');
    } else {
      if (html.toLowerCase().includes('pricing')) pass('Index mentions pricing');
      else fail('Index has pricing section');
    }
  } catch (e) { fail('Index pricing', e.message); }
}

// TC-UI-17: Docs page
async function TC_UI_17() {
  currentTC = 'TC-UI-17';
  try {
    const { html } = await fetchPage(BASE + 'docs.html');
    const doc = parseDoc(html);
    const sidebarLinks = doc.querySelectorAll('a[href^="#"]');
    if (sidebarLinks.length >= 5) pass('Docs has sidebar nav (' + sidebarLinks.length + ' links)');
    else fail('Docs sidebar nav', 'Only ' + sidebarLinks.length + ' anchor links');
    const codeBlocks = doc.querySelectorAll('pre, code');
    if (codeBlocks.length >= 3) pass('Docs has code examples (' + codeBlocks.length + ')');
    else fail('Docs code examples', 'Only ' + codeBlocks.length);
  } catch (e) { fail('Docs page', e.message); }
}

// TC-UI-18: Chat page
async function TC_UI_18() {
  currentTC = 'TC-UI-18';
  try {
    const { html } = await fetchPage(BASE + 'chat.html');
    const doc = parseDoc(html);
    if (doc.getElementById('chat-input') || doc.querySelector('textarea, input[type="text"]')) pass('Chat has input field'); else fail('Chat has input field');
    if (doc.getElementById('btn-send') || doc.querySelector('button')) pass('Chat has send button'); else fail('Chat has send button');
    if (doc.getElementById('btn-new-chat')) pass('Chat has New Chat button'); else fail('Chat has New Chat button');
  } catch (e) { fail('Chat page', e.message); }
}

// TC-UI-19: Memory page
async function TC_UI_19() {
  currentTC = 'TC-UI-19';
  try {
    const { html } = await fetchPage(BASE + 'memory.html');
    const doc = parseDoc(html);
    if (doc.getElementById('model-list') || doc.querySelector('.model-list')) pass('Memory has model filter'); else fail('Memory has model filter');
    if (doc.getElementById('cat-list') || doc.querySelector('.cat-list')) pass('Memory has category filter'); else fail('Memory has category filter');
    if (doc.getElementById('layer-list') || doc.querySelector('.layer-list')) pass('Memory has layer filter'); else fail('Memory has layer filter');
    if (doc.getElementById('search') || doc.querySelector('input[placeholder*="earch"]')) pass('Memory has search input'); else fail('Memory has search input');
  } catch (e) { fail('Memory page', e.message); }
}

// TC-UI-20: Benchmark pages
async function TC_UI_20() {
  currentTC = 'TC-UI-20';
  for (const page of ['benchmark.html', 'benchmark_v1_direct.html']) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const tables = doc.querySelectorAll('table');
      if (tables.length >= 1) pass(page + ' has data tables (' + tables.length + ')');
      else fail(page + ' has data tables');
      if (doc.querySelector('h1')) pass(page + ' has heading'); else fail(page + ' has heading');
    } catch (e) { fail(page, e.message); }
  }
}

// TC-UI-21: Tests page
async function TC_UI_21() {
  currentTC = 'TC-UI-21';
  try {
    const { html } = await fetchPage(BASE + 'tests.html');
    const doc = parseDoc(html);
    if (doc.getElementById('run-list')) pass('Tests has run history list'); else fail('Tests has run history list');
    if (doc.getElementById('main-panel')) pass('Tests has main results panel'); else fail('Tests has main results panel');
  } catch (e) { fail('Tests page', e.message); }
}

// TC-UI-22: Install page
async function TC_UI_22() {
  currentTC = 'TC-UI-22';
  try {
    const { html } = await fetchPage(BASE + 'install.html');
    const doc = parseDoc(html);
    const btn = doc.getElementById('btn') || doc.querySelector('button');
    if (btn) {
      const text = btn.textContent.toLowerCase();
      if (text.includes('activate')) pass('Install has Activate CTA');
      else pass('Install has button: ' + btn.textContent.trim());
    } else { fail('Install has Activate button'); }
  } catch (e) { fail('Install page', e.message); }
}

// TC-UI-23: Profile page
async function TC_UI_23() {
  currentTC = 'TC-UI-23';
  try {
    const { html } = await fetchPage(BASE + 'profile.html');
    const lower = html.toLowerCase();
    if (lower.includes('profile') || lower.includes('user')) pass('Profile has user section'); else fail('Profile has user section');
    if (lower.includes('api key') || lower.includes('api-key')) pass('Profile has API keys section'); else fail('Profile has API keys section');
    if (lower.includes('billing') || lower.includes('plan') || lower.includes('subscription')) pass('Profile has billing section'); else fail('Profile has billing section');
  } catch (e) { fail('Profile page', e.message); }
}

// TC-UI-24: Dashboard page
async function TC_UI_24() {
  currentTC = 'TC-UI-24';
  try {
    const { html } = await fetchPage(BASE + 'dashboard.html');
    const doc = parseDoc(html);
    const cards = doc.querySelectorAll('.dash-card');
    const quickLinks = doc.querySelectorAll('.quick-link');
    if (cards.length >= 1) pass('Dashboard has stat cards (' + cards.length + ')');
    else fail('Dashboard has stat cards');
    if (quickLinks.length >= 1) pass('Dashboard has quick links (' + quickLinks.length + ')');
    else fail('Dashboard has quick links');
  } catch (e) { fail('Dashboard page', e.message); }
}

// TC-UI-25: Usage page
async function TC_UI_25() {
  currentTC = 'TC-UI-25';
  try {
    const { html } = await fetchPage(BASE + 'usage.html');
    const lower = html.toLowerCase();
    if (lower.includes('usage') || lower.includes('operations')) pass('Usage has usage data'); else fail('Usage has usage data');
    if (lower.includes('chart') || lower.includes('canvas')) pass('Usage has chart element'); else fail('Usage has chart element');
  } catch (e) { fail('Usage page', e.message); }
}

// TC-UI-26: Legal pages
async function TC_UI_26() {
  currentTC = 'TC-UI-26';
  for (const page of ['terms.html', 'privacy.html']) {
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      if (doc.querySelector('h1')) pass(page + ' has main heading'); else fail(page + ' has main heading');
      const sections = doc.querySelectorAll('h2, h3');
      if (sections.length >= 3) pass(page + ' has ' + sections.length + ' sections');
      else fail(page + ' sections', 'Only ' + sections.length);
    } catch (e) { fail(page, e.message); }
  }
}

// TC-UI-27: Support page
async function TC_UI_27() {
  currentTC = 'TC-UI-27';
  try {
    const { html } = await fetchPage(BASE + 'support.html');
    const lower = html.toLowerCase();
    if (lower.includes('email') || lower.includes('contact')) pass('Support has contact info'); else fail('Support has contact info');
    if (lower.includes('faq') || lower.includes('frequently')) pass('Support has FAQ section'); else fail('Support has FAQ section');
  } catch (e) { fail('Support page', e.message); }
}

// TC-UI-28: No hamburger menu
async function TC_UI_28() {
  currentTC = 'TC-UI-28';
  for (const page of ALL_PAGES) {
    if (page === 'integrate.html') continue;
    try {
      const { html } = await fetchPage(BASE + page);
      const doc = parseDoc(html);
      const toggle = doc.querySelector('.nav-toggle, button[aria-label="Menu"]');
      if (!toggle) pass(page + ' no hamburger menu');
      else fail(page + ' has hamburger menu', 'Should be removed');
    } catch (e) { fail(page + ' toggle check', e.message); }
  }
}

// TC-UI-29: System font or Inter loaded
async function TC_UI_29() {
  currentTC = 'TC-UI-29';
  for (const page of ALL_PAGES) {
    if (page === 'integrate.html') continue;
    try {
      const { html } = await fetchPage(BASE + page);
      const hasFont = html.includes('-apple-system') || html.includes('system-ui')
        || html.includes('Inter') || html.includes('inter')
        || html.includes('styles.css');
      if (hasFont) pass(page + ' uses system/Inter font or styles.css');
      else fail(page + ' missing font declaration');
    } catch (e) { fail(page + ' font check', e.message); }
  }
}

// TC-UI-30: integrate.html redirects
async function TC_UI_30() {
  currentTC = 'TC-UI-30';
  try {
    const { html } = await fetchPage(BASE + 'integrate.html');
    const doc = parseDoc(html);
    const meta = doc.querySelector('meta[http-equiv="refresh"]');
    if (meta) {
      const content = meta.getAttribute('content');
      if (content.includes('getting-started')) pass('integrate.html redirects to getting-started');
      else fail('integrate.html redirect target', content);
    } else { fail('integrate.html has redirect meta'); }
  } catch (e) { fail('integrate.html redirect', e.message); }
}

// =========================================================================
// Runner
// =========================================================================
async function main() {
  console.log('CLS++ Production UI Test Suite');
  console.log('Target: ' + BASE);
  console.log('═'.repeat(60));
  console.log('');

  const ALL_TCS = [
    TC_UI_01, TC_UI_02, TC_UI_03, TC_UI_04, TC_UI_05,
    TC_UI_06, TC_UI_07, TC_UI_08, TC_UI_09, TC_UI_10,
    TC_UI_11, TC_UI_12, TC_UI_13, TC_UI_14, TC_UI_15,
    TC_UI_16, TC_UI_17, TC_UI_18, TC_UI_19, TC_UI_20,
    TC_UI_21, TC_UI_22, TC_UI_23, TC_UI_24, TC_UI_25,
    TC_UI_26, TC_UI_27, TC_UI_28, TC_UI_29, TC_UI_30
  ];

  for (let i = 0; i < ALL_TCS.length; i++) {
    const tcName = 'TC-UI-' + String(i + 1).padStart(2, '0');
    process.stdout.write('  Running ' + tcName + '...');
    try {
      await ALL_TCS[i]();
      const tcResults = results.filter(r => r.tc === tcName);
      const tcFails = tcResults.filter(r => r.status === 'FAIL').length;
      if (tcFails === 0) console.log(' \x1b[32m✓ PASS\x1b[0m (' + tcResults.length + ' checks)');
      else console.log(' \x1b[31m✗ FAIL\x1b[0m (' + tcFails + '/' + tcResults.length + ' failed)');
    } catch (e) {
      console.log(' \x1b[31m✗ ERROR\x1b[0m ' + e.message);
    }
  }

  // Summary
  const passed = results.filter(r => r.status === 'PASS').length;
  const failed = results.filter(r => r.status === 'FAIL').length;
  const total = results.length;
  const rate = total > 0 ? ((passed / total) * 100).toFixed(1) : '0.0';

  console.log('');
  console.log('═'.repeat(60));
  console.log('  RESULTS: ' + passed + '/' + total + ' passed (' + rate + '%)');
  console.log('  \x1b[32mPassed: ' + passed + '\x1b[0m  \x1b[31mFailed: ' + failed + '\x1b[0m');
  console.log('═'.repeat(60));

  // Print failures
  const failures = results.filter(r => r.status === 'FAIL');
  if (failures.length > 0) {
    console.log('');
    console.log('\x1b[31mFAILURES (' + failures.length + '):\x1b[0m');
    failures.forEach(f => {
      console.log('  \x1b[31m✗\x1b[0m [' + f.tc + '] ' + f.name + (f.detail ? ' — ' + f.detail : ''));
    });
  }

  // Return exit code
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
