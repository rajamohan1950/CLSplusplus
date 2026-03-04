# Local Testing — 100% Before Deploy

## 1. Add your API keys

```bash
cp .env.example .env
# Edit .env — replace placeholders with REAL keys:
#   CLS_ANTHROPIC_API_KEY  from console.anthropic.com
#   CLS_OPENAI_API_KEY     from platform.openai.com/api-keys
#   CLS_GOOGLE_API_KEY     from aistudio.google.com/app/apikey
```

## 2. Run local demo (no Redis/Postgres)

```bash
./scripts/run_local_demo.sh
```

Opens:
- **API**: http://localhost:8080
- **Website**: http://localhost:3000?local=1

## 3. Test in browser

1. Open http://localhost:3000?local=1
2. Scroll to **Try It Yourself**
3. **Claude**: type "My name is Alex" → Send → Claude should reply
4. **OpenAI**: type "What is my name?" → Send → should say Alex
5. **Gemini**: type "My favorite color is blue" → Send → Gemini should reply
6. **Claude**: type "What is my favorite color?" → should say blue

## 4. Run load test locally

```bash
CLS_API_URL=http://localhost:8080 python scripts/load_test_demo.py 1 seq
```

(Start the API first with `./scripts/run_local_demo.sh` in another terminal, or run API only: `PYTHONPATH=src uvicorn clsplusplus.demo_local:app --port 8080`)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Add CLS_*_API_KEY" | Put real keys in .env |
| "401 invalid" | Key is wrong or expired |
| Port 8080 in use | Use `--port 8081` and update CLS_API_URL |
