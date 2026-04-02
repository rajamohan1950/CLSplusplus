/**
 * Cross-browser E2E tests for CLS++ extension variants.
 * Tests Edge (Chromium) and Firefox extensions alongside Chrome.
 *
 * Edge: Uses same Chromium --load-extension mechanism as Chrome.
 * Firefox: Uses Playwright's Firefox extension support (web-ext).
 *
 * Run:
 *   cd extension && npm run test:e2e
 */
const { test, expect, chromium, firefox } = require('@playwright/test');
const path = require('path');
const os = require('os');
const fs = require('fs');

const BASE = `http://127.0.0.1:${process.env.CLSPP_E2E_PORT || '9876'}`;
const ROOT = path.resolve(__dirname, '../..');
const CHROME_EXT = path.resolve(ROOT, 'extension');
const EDGE_EXT = path.resolve(ROOT, 'extension-edge');
const FIREFOX_EXT = path.resolve(ROOT, 'extension-firefox');

// Each describe block can run independently

// ── Shared helpers ─────────────────────────────────────────────────────────

async function seedMemory(uid, text) {
  await fetch(`${BASE}/api/memories/${uid}`, { method: 'DELETE' });
  const r = await fetch(`${BASE}/api/store/${uid}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source: 'user', model: 'e2e' }),
  });
  if (!r.ok) throw new Error(`seed failed: ${r.status}`);
}

// ── Edge extension test (Chromium-based) ───────────────────────────────────

test.describe('Edge extension', () => {
  test('loads and has correct manifest', async () => {
    // Verify Edge manifest exists and is valid
    const manifest = JSON.parse(fs.readFileSync(path.join(EDGE_EXT, 'manifest.json'), 'utf8'));
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.name).toBe('CLS++ Memory');
    expect(manifest.background.service_worker).toBe('background.js');
    expect(manifest.permissions).toContain('storage');
  });

  test('Edge extension files are complete', async () => {
    const requiredFiles = [
      'manifest.json', 'background.js', 'popup.html', 'popup.js',
      'intercept.js', 'content_common.js', 'content_chatgpt.js',
      'content_claude.js', 'content_gemini.js', 'content_marker.js',
      'icons/icon16.png', 'icons/icon48.png', 'icons/icon128.png',
    ];
    for (const f of requiredFiles) {
      const p = path.join(EDGE_EXT, f);
      expect(fs.existsSync(p), `Missing: ${f}`).toBe(true);
    }
  });

  test('Edge extension injects memory (Chromium engine)', async () => {
    const userDataDir = path.join(os.tmpdir(), `clspp-edge-e2e-${Date.now()}`);
    const context = await chromium.launchPersistentContext(userDataDir, {
      channel: 'chromium',
      args: [`--disable-extensions-except=${EDGE_EXT}`, `--load-extension=${EDGE_EXT}`],
      headless: false,
    });

    try {
      await new Promise(r => setTimeout(r, 5000));

      let page = context.pages()[0];
      if (!page) page = await context.newPage();

      await page.addInitScript(() => {
        window.__clspp_uid_value = null;
        window.addEventListener('__clspp_uid', (e) => {
          window.__clspp_uid_value = e.detail;
        });
      });

      await page.goto(`${BASE}/ui/extension-e2e.html`, { waitUntil: 'load' });
      await page.waitForSelector('#btn-chatgpt', { state: 'attached', timeout: 10_000 });

      let uid = null;
      for (let attempt = 0; attempt < 4; attempt++) {
        await new Promise(r => setTimeout(r, 2000));
        uid = await page.evaluate(() => window.__clspp_uid_value);
        if (uid) break;
        if (attempt < 3) await page.reload({ waitUntil: 'load' });
      }
      expect(uid).toBeTruthy();

      await seedMemory(uid, 'Edge test memory ORANGEFOX');
      await page.locator('#btn-chatgpt').click({ force: true });
      await expect(page.locator('#result')).toContainText('"injection_ok": true', { timeout: 30_000 });
    } finally {
      await context.close();
    }
  });
});

// ── Firefox extension test ──────────────────────────────────────────────────

test.describe('Firefox extension', () => {
  test('has correct manifest with gecko settings', async () => {
    const manifest = JSON.parse(fs.readFileSync(path.join(FIREFOX_EXT, 'manifest.json'), 'utf8'));
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.name).toBe('CLS++ Memory');
    expect(manifest.browser_specific_settings.gecko.id).toBe('clsplusplus@clsplusplus.com');
    expect(manifest.background.scripts).toContain('background.js');
    // Firefox MV3 uses background.scripts, not service_worker
    expect(manifest.background.service_worker).toBeUndefined();
  });

  test('Firefox extension files are complete', async () => {
    const requiredFiles = [
      'manifest.json', 'background.js', 'popup.html', 'popup.js',
      'intercept.js', 'content_common.js', 'content_chatgpt.js',
      'content_claude.js', 'content_gemini.js', 'content_marker.js',
      'icons/icon16.png', 'icons/icon48.png', 'icons/icon128.png',
    ];
    for (const f of requiredFiles) {
      const p = path.join(FIREFOX_EXT, f);
      expect(fs.existsSync(p), `Missing: ${f}`).toBe(true);
    }
  });

  test('Firefox popup.js does not call chrome.runtime.reload()', async () => {
    // Firefox does not support chrome.runtime.reload() from popup
    const popupJs = fs.readFileSync(path.join(FIREFOX_EXT, 'popup.js'), 'utf8');
    // Strip comments before checking — mentions in comments are fine
    const codeOnly = popupJs.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(codeOnly).not.toContain('chrome.runtime.reload');
    expect(popupJs).toContain('window.close()');
  });

  test('Firefox background.js omits onMessageExternal', async () => {
    // Firefox does not support onMessageExternal in MV3
    const bgJs = fs.readFileSync(path.join(FIREFOX_EXT, 'background.js'), 'utf8');
    expect(bgJs).not.toContain('onMessageExternal');
  });
});

// ── Safari extension test ──────────────────────────────────────────────────

test.describe('Safari extension', () => {
  const SAFARI_EXT = path.resolve(ROOT, 'extension-safari');

  test('has correct manifest for Safari', async () => {
    const manifest = JSON.parse(fs.readFileSync(path.join(SAFARI_EXT, 'manifest.json'), 'utf8'));
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.name).toBe('CLS++ Memory');
    // Safari uses background.scripts, not service_worker
    expect(manifest.background.scripts).toContain('background.js');
  });

  test('Safari extension files are complete', async () => {
    const requiredFiles = [
      'manifest.json', 'background.js', 'popup.html', 'popup.js',
      'intercept.js', 'content_common.js', 'content_chatgpt.js',
      'content_claude.js', 'content_gemini.js', 'content_marker.js',
      'icons/icon16.png', 'icons/icon48.png', 'icons/icon128.png',
    ];
    for (const f of requiredFiles) {
      const p = path.join(SAFARI_EXT, f);
      expect(fs.existsSync(p), `Missing: ${f}`).toBe(true);
    }
  });

  test('Safari files use browser API polyfill', async () => {
    const bgJs = fs.readFileSync(path.join(SAFARI_EXT, 'background.js'), 'utf8');
    const commonJs = fs.readFileSync(path.join(SAFARI_EXT, 'content_common.js'), 'utf8');
    const popupJs = fs.readFileSync(path.join(SAFARI_EXT, 'popup.js'), 'utf8');

    // Safari files should use browser/chrome polyfill pattern
    expect(bgJs).toContain("typeof browser !== 'undefined'");
    expect(commonJs).toContain("typeof browser !== 'undefined'");
    expect(popupJs).toContain("typeof browser !== 'undefined'");
  });

  test('Safari content_marker.js uses browser API polyfill', async () => {
    const markerJs = fs.readFileSync(path.join(SAFARI_EXT, 'content_marker.js'), 'utf8');
    expect(markerJs).toContain("typeof browser !== 'undefined'");
  });

  test('Safari manifest has no E2E test URLs', async () => {
    // Safari distribution should not include localhost test URLs in content scripts
    const manifest = JSON.parse(fs.readFileSync(path.join(SAFARI_EXT, 'manifest.json'), 'utf8'));
    const allMatches = manifest.content_scripts.flatMap(cs => cs.matches);
    const e2eMatches = allMatches.filter(m => m.includes('extension-e2e'));
    expect(e2eMatches).toHaveLength(0);
  });
});

// ── Popup HTML consistency tests ───────────────────────────────────────────

test.describe('Popup UI consistency', () => {
  const browsers = [
    { name: 'Chrome', dir: CHROME_EXT },
    { name: 'Edge', dir: EDGE_EXT },
    { name: 'Firefox', dir: FIREFOX_EXT },
    { name: 'Safari', dir: path.resolve(ROOT, 'extension-safari') },
  ];

  for (const { name, dir } of browsers) {
    test(`${name} popup.html has all UI elements`, async () => {
      const html = fs.readFileSync(path.join(dir, 'popup.html'), 'utf8');

      // Brain emoji logo
      expect(html).toContain('CLS++ Memory');

      // Memory count display
      expect(html).toContain('id="count"');
      expect(html).toContain('memories stored');

      // Auto-inject toggle
      expect(html).toContain('id="toggle-inject"');
      expect(html).toContain('Auto-inject');

      // Local mode toggle
      expect(html).toContain('id="toggle-local"');
      expect(html).toContain('Local mode');

      // View memories button
      expect(html).toContain('id="view-btn"');
      expect(html).toContain('View My Memories');

      // Status indicator
      expect(html).toContain('id="status-dot"');
      expect(html).toContain('id="status-text"');

      // Dark theme colors
      expect(html).toContain('#06060c');  // background
      expect(html).toContain('#7c6ef0');  // primary purple
      expect(html).toContain('#5de0c5');  // status green
    });
  }
});
