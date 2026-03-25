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

async function getUid() {
  const uidRes = await fetch(`${BASE}/api/uid`);
  const { uid } = await uidRes.json();
  if (!uid) throw new Error('no uid from server');
  return uid;
}

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

// Two facts in the same profile push engine scores below /api/context MIN_RELEVANCE; seed one at a time.
test('memory injection: ChatGPT + Claude shaped requests', async () => {
  // MV3 extensions are unreliable in headless Chromium; use a real display or xvfb-run in CI.
  const userDataDir = path.join(os.tmpdir(), `clspp-ext-e2e-${Date.now()}`);
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: 'chromium',
    args: [`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`],
    headless: false,
  });

  try {
    const uid = await getUid();
    const page = await context.newPage();
    await page.goto(`${BASE}/ui/extension-e2e.html`, { waitUntil: 'load' });
    await page.waitForSelector('#btn-chatgpt', { state: 'attached', timeout: 20_000 });
    await clearAndStore(uid, 'My secret codename is BLUEBADGER');
    await page.locator('#btn-chatgpt').click({ force: true });
    await expect(page.locator('#result')).toContainText('"injection_ok": true', { timeout: 30_000 });
    await expect(page.locator('#status')).toContainText('PASS');

    await clearAndStore(uid, 'I live in Springfield');
    await page.locator('#btn-claude').click({ force: true });
    await expect(page.locator('#result')).toContainText('"injection_ok": true', { timeout: 30_000 });
    await expect(page.locator('#status')).toContainText('PASS');
  } finally {
    await context.close();
  }
});
