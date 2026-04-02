/**
 * UI E2E: prototype home, install page, and extension download.
 * Run: cd extension && npm ci && npx playwright install chromium && npx playwright test e2e/prototype-home-install.spec.cjs
 *
 * Start the prototype server:
 *   CLSPP_E2E_PORT=9876 PYTHONPATH=.. uvicorn server:app --host 127.0.0.1 --port 9876
 */
const { test, expect } = require('@playwright/test');

const BASE = `http://127.0.0.1:${process.env.CLSPP_E2E_PORT || '9876'}`;

test.describe('prototype home (new user)', () => {
  test('preset connection → Memories lists seeded fact', async ({ browser }) => {
    const context = await browser.newContext();
    await context.addInitScript((base) => {
      window.__CLSPP_CONNECTION__ = {
        apiBase: base,
        backend: 'prototype',
        armed: true,
      };
    }, BASE);
    const page = await context.newPage();
    await page.goto('about:blank');
    await page.evaluate(() => {
      try {
        localStorage.clear();
        sessionStorage.clear();
      } catch (_) {}
    });

    await page.goto(`${BASE}/ui/index.html`, { waitUntil: 'domcontentloaded' });

    await expect(page.locator('#btn-arm')).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem('clspp_armed'))).toBe('1');
    expect(await page.evaluate(() => localStorage.getItem('clspp_ui_backend'))).toBe('prototype');
    expect(await page.evaluate(() => localStorage.getItem('clspp_api_base'))).toBe(BASE);

    const uid = await page.evaluate(() => localStorage.getItem('clspp_uid'));
    expect(uid).toBeTruthy();

    const store = await page.request.post(`${BASE}/api/store/${encodeURIComponent(uid)}`, {
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify({
        text: 'E2E install flow remembers dark mode preference for testing UI',
        source: 'user',
        model: 'e2e-ui',
      }),
    });
    if (!store.ok()) throw new Error(`store failed: ${store.status()} ${await store.text()}`);

    const memApi = await page.request.get(`${BASE}/api/memories/${encodeURIComponent(uid)}`);
    expect(memApi.ok()).toBeTruthy();
    const memJson = await memApi.json();
    expect(memJson.count).toBeGreaterThan(0);

    await page.goto(`${BASE}/ui/memory.html`, { waitUntil: 'domcontentloaded' });
    // Memory grid may contain data from prior runs; look for our text anywhere in the grid
    await expect(page.locator('#mem-grid')).toContainText('E2E install flow', { timeout: 20_000 });

    await context.close();
  });

  test('install.html renders the four-step guide', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(`${BASE}/ui/install.html`, { waitUntil: 'domcontentloaded' });

    await expect(page.locator('h1')).toContainText('Install the Extension');
    await expect(page.locator('.step')).toHaveCount(4);
    await expect(page.locator('.step-title').nth(0)).toContainText('Download started');
    await expect(page.locator('.step-title').nth(1)).toContainText('Unzip');
    await expect(page.locator('.step-title').nth(2)).toContainText('Load into Chrome');
    await expect(page.locator('.step-title').nth(3)).toContainText("You're done");

    await context.close();
  });

  test('GET /extension/download returns a zip', async ({ browser }) => {
    const context = await browser.newContext();
    const resp = await context.request.get(`${BASE}/extension/download`);
    expect(resp.ok()).toBeTruthy();
    expect(resp.headers()['content-type']).toContain('application/zip');
    const body = await resp.body();
    expect(body.length).toBeGreaterThan(100);
    await context.close();
  });
});
