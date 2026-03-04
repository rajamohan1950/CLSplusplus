# Demo Testing Guide

## Prerequisites

1. **API keys** in Render → clsplusplus-api → Environment:
   - `CLS_ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com/)
   - `CLS_OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com/api-keys)
   - `CLS_GOOGLE_API_KEY` — [aistudio.google.com](https://aistudio.google.com/app/apikey)

2. **Redis + Postgres** (for memory) — see `RENDER_ENV_SETUP.md`

## 5-Minute Chat + Criss-Cross Test (Manual)

**From the UI** at [clsplusplus-website.onrender.com](https://clsplusplus-website.onrender.com/#tryit):

1. **Phase 1 — Chat with each model (~5 min)**
   - **Claude**: "My name is Alex and I have a dog named Max"
   - **Claude**: "My favorite color is blue"
   - **Gemini**: "I live in Boston and love pizza"
   - **OpenAI**: "My hobby is chess"
   - **OpenAI**: "I work at Acme Corp"

2. **Phase 2 — Criss-cross questions**
   - **OpenAI**: "What is my name?" → should say Alex
   - **Claude**: "What city do I live in?" → should say Boston
   - **Gemini**: "What is my dog's name?" → should say Max
   - **Claude**: "What is my hobby?" → should say chess
   - **OpenAI**: "Where do I work?" → should say Acme Corp

3. **Pass criteria**: Each answer contains the fact told to a *different* model.

## 5 Concurrent Users Load Test (Automated)

```bash
pip install httpx
python scripts/load_test_demo.py          # 5 concurrent users
python scripts/load_test_demo.py 1 seq    # 1 user sequential (smoke test)
python scripts/load_test_demo.py 5 seq    # 5 users sequential
```

Each user: tells 6 facts across Claude/Gemini/OpenAI, then asks 5 criss-cross questions. Pass = all answers contain expected facts.

## Quick Checks

### Demo status (which keys are set)

```bash
curl https://clsplusplus-api.onrender.com/v1/demo/status
```

### Test single LLM

```bash
curl --max-time 120 -X POST https://clsplusplus-api.onrender.com/v1/demo/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"claude","message":"Say hi","namespace":"test"}'
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CORS blocked | Ensure API is deployed with latest CORS config; 502 from proxy = backend down |
| "Add CLS_*_API_KEY to env" | Add the key in Render Environment |
| "Request timed out" | Wait 60s (cold start), try again |
| 500 / 502 | Check Render logs; add Redis/Postgres + API keys |
| "API sleeping" | Render free tier spins down; first request wakes it |
