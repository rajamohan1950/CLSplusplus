const { defineConfig } = require('@playwright/test');
const path = require('path');

// Dedicated port so tests never attach to an unrelated process on 8080.
const E2E_PORT = process.env.CLSPP_E2E_PORT || '9876';

/** @see https://playwright.dev/docs/chrome-extensions */
module.exports = defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 25_000 },
  fullyParallel: false,
  workers: 1,
  webServer: {
    command: `PYTHONPATH=.. uvicorn server:app --host 127.0.0.1 --port ${E2E_PORT}`,
    cwd: path.join(__dirname, '../prototype'),
    url: `http://127.0.0.1:${E2E_PORT}/health`,
    reuseExistingServer: !!process.env.CLSPP_REUSE_SERVER,
    timeout: 120_000,
  },
});
