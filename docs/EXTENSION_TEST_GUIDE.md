# CLS++ Chrome Extension — Test Guide

**Memory across ChatGPT, Claude, and Gemini.** The extension silently injects relevant memories when you send messages and captures new facts for future sessions.

---

## Prerequisites

1. **Chrome** (or Chromium-based: Edge, Brave, Arc)
2. **Prototype server** running on `localhost:8080`
3. **Extension** loaded in developer mode

---

## 1. Start the Server

```bash
cd /path/to/CLSplusplus
PYTHONPATH=src python prototype/server.py
```

Server runs at `http://localhost:8080`. Verify:

```bash
curl http://localhost:8080/health
# {"status":"ok","namespaces":0,"total_memories":0}

curl http://localhost:8080/api/uid
# {"uid":"u_xxxx"}
```

---

## 2. Load the Extension

1. Open Chrome → `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `extension` folder in the CLSplusplus repo

You should see **CLS++ Memory** in the extensions list. Pin it to the toolbar.

---

## 3. Automated E2E (Playwright)

Verifies **fetch interception** + **context bridge** against mock ChatGPT/Claude endpoints (no real OpenAI/Anthropic login).

```bash
cd extension
npm install
npx playwright install chromium
npm run test:e2e
```

**Requirements:** Prototype server on `127.0.0.1:8080` (Playwright config starts it if not already running). **Headed Chromium** opens briefly — not supported in headless CI; set `CI=1` to skip the test in pipelines.

**Harness page:** [prototype/extension-e2e.html](../prototype/extension-e2e.html) — also test manually with the extension loaded.

---

## 4. Test Memory Across LLMs

### Flow

1. **Tell ChatGPT something:** e.g., "My name is Raj and I prefer dark mode"
2. **Ask Claude:** e.g., "What's my name and what do I prefer?"
3. **Claude should answer** using the memory (if injection worked)
4. **Ask Gemini:** "Do you remember anything about me?"
5. **Gemini should also use the same memory**

### What to Check

| Step | Expected |
|------|----------|
| Popup icon | Badge shows memory count |
| Popup status | "Memory engine running" (green dot) |
| ChatGPT send | Console: `[CLS++] context injected into ChatGPT` |
| Claude send | Console: `[CLS++] context injected into Claude` |
| Gemini send | Console: `[CLS++] context injected into Gemini` |
| New message | Console: `[CLS++] captured user/assistant ...` |
| View Memories | `http://localhost:8080/ui/memory.html` shows stored items |

### Debug: Open DevTools

- **ChatGPT:** F12 → Console tab
- **Claude:** F12 → Console tab
- **Gemini:** F12 → Console tab

Look for `[CLS++]` messages on send. If you see "context injected" — injection worked. If the LLM response reflects prior facts — memory is shared.

---

## 5. Known Limitations

| Site | Notes |
|------|-------|
| **ChatGPT** | URL patterns: `/backend-api/conversation`, `/backend-api/f/conversation`. OpenAI may change paths. |
| **Claude** | URL patterns: `/api/` + `chat_conversations/` or `completion` |
| **Gemini** | Uses `batchexecute` or `BardFrontendService`; regex extracts query from request body |
| **CSP** | ChatGPT has strict CSP; extension uses content-script bridge to bypass (MAIN world can't fetch localhost) |

---

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| Popup shows "Start server.py to activate" | Run `python prototype/server.py` |
| Badge shows 0 | Normal at start; send messages to capture |
| No "context injected" in console | Check Auto-inject toggle in popup (must be on) |
| Extension not loading | Ensure manifest.json is valid; check chrome://extensions for errors |
| Claude/Gemini URL changed | Update `intercept.js` URL patterns |

---

## 7. Architecture Summary

```
User sends message on ChatGPT/Claude/Gemini
    → intercept.js (MAIN world) hooks fetch
    → Extracts user query from request body
    → Dispatches __clspp_context_request
    → content_common.js (ISOLATED) fetches /api/context from localhost
    → Returns relevant memories via __clspp_context_response
    → intercept.js prepends memory block to user message
    → Request sent to LLM API with memory context
    → content_*.js watches DOM, stores new messages via /api/store/{uid}
```

---

**AlphaForge AI Labs** | [CLS++](https://github.com/rajamohan1950/CLSplusplus)
