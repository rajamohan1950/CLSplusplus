/**
 * E2E: Chrome extension prepends CLS++ context into ChatGPT- and Claude-shaped fetch bodies.
 * Requires Chromium + unpacked extension (headed; MV3 + headless often skips page HTML).
 *
 * Run from repo:
 *   cd extension && npm ci && npx playwright install chromium && npm run test:e2e
 */
const { test, expect, chromium } = require('@playwright/test');
const path = require('path');
const os = require('os');

const BASE = `http://127.0.0.1:${process.env.CLSPP_E2E_PORT || '9876'}`;
const EXTENSION_DIR = path.resolve(__dirname, '..');

test.describe.configure({ mode: 'serial' });

async function clearAndStore(uid, text) {
  const delR = await fetch(`${BASE}/api/memories/${uid}`, { method: 'DELETE' });
  if (!delR.ok) throw new Error(`clear failed: ${delR.status}`);
  const r = await fetch(`${BASE}/api/store/${uid}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source: 'user', model: 'e2e' }),
  });
  if (!r.ok) throw new Error(`seed store failed: ${r.status} ${await r.text()}`);
}

test('memory injection: ChatGPT + Claude shaped requests', async () => {
  const userDataDir = path.join(os.tmpdir(), `clspp-ext-e2e-${Date.now()}`);
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: 'chromium',
    args: [`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`],
    headless: false,
  });

  try {
    // Wait for extension's onInstalled handler to settle
    await new Promise(r => setTimeout(r, 5000));

    // Use first existing page
    let page = context.pages()[0];
    if (!page) page = await context.newPage();

    // Register UID capture BEFORE navigation so we don't miss the event.
    // The content script dispatches __clspp_uid on window at document_idle.
    await page.addInitScript(() => {
      window.__clspp_uid_value = null;
      window.addEventListener('__clspp_uid', (e) => {
        window.__clspp_uid_value = e.detail;
      });
    });

    // Navigate to the E2E harness
    await page.goto(`${BASE}/ui/extension-e2e.html`, { waitUntil: 'load' });
    await page.waitForSelector('#btn-chatgpt', { state: 'attached', timeout: 10_000 });

    // Wait for content script UID (with retry via reload)
    let uid = null;
    for (let attempt = 0; attempt < 4; attempt++) {
      await new Promise(r => setTimeout(r, 2000));
      uid = await page.evaluate(() => window.__clspp_uid_value);
      if (uid) break;
      if (attempt < 3) await page.reload({ waitUntil: 'load' });
    }
    if (!uid) throw new Error('extension did not provide UID after 4 attempts');

    // --- ChatGPT-shaped injection test ---
    await clearAndStore(uid, 'My secret codename is BLUEBADGER');
    await page.locator('#btn-chatgpt').click({ force: true });
    await expect(page.locator('#result')).toContainText('"injection_ok": true', { timeout: 30_000 });
    await expect(page.locator('#status')).toContainText('PASS');

    // --- Claude-shaped injection test ---
    await clearAndStore(uid, 'I live in Springfield');
    await page.locator('#btn-claude').click({ force: true });
    await expect(page.locator('#result')).toContainText('"injection_ok": true', { timeout: 30_000 });
    await expect(page.locator('#status')).toContainText('PASS');
  } finally {
    await context.close();
  }
});
