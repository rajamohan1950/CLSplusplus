# CLS++ Wiki

Welcome to the **CLS++ (Continuous Learning System++)** wiki — brain-inspired, model-agnostic persistent memory for LLMs.

---

## Quick Links

| Page | Description |
|------|-------------|
| [Architecture](Architecture) | Four-store hierarchy, plasticity, sleep cycle |
| [API Reference](API-Reference) | Endpoints, auth, examples |
| [Deployment Guide](Deployment-Guide) | Render, AWS, Azure |
| [Integration Examples](Integration-Examples) | Python, JavaScript, LangChain |
| [SaaS & Pricing](SaaS-and-Pricing) | Memory-as-a-Service, tiers |
| [Contributing](Contributing) | How to contribute |

---

## What is CLS++?

CLS++ gives LLMs **persistent memory** that works like a human brain. Your AI remembers what matters, forgets what doesn't, and stays consistent across sessions and model switches.

**Key features:**
- **Model-agnostic** — Switch GPT-4 → Claude → Gemini without losing context
- **Brain-inspired** — Memory strengthens with use, decays when unused
- **Reconsolidation gate** — Belief revision only with evidence
- **REST API** — Any LLM plugs in

---

## Get Started

```bash
pip install clsplusplus
docker compose up -d redis postgres minio
uvicorn clsplusplus.api:app --host 0.0.0.0 --port 8080
```

[Try the live demo](https://clsplusplus.onrender.com) • [Deploy free on Render](https://render.com/deploy?repo=https://github.com/rajamohan1950/CLSplusplus)

---

**AlphaForge AI Labs** | [GitHub](https://github.com/rajamohan1950/CLSplusplus) | [API Docs](https://clsplusplus-api.onrender.com/docs)
