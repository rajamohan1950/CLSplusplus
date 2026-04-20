<p align="center">
  <img src="https://img.shields.io/badge/CLS%2B%2B-Memory%20for%20LLMs-6366f1?style=for-the-badge&logo=github" alt="CLS++" />
</p>

<h1 align="center">CLS++ — Continuous Learning System++</h1>
<p align="center">
  <strong>Switch AI models. Never lose context.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/clsplusplus/"><img src="https://img.shields.io/pypi/v/clsplusplus?style=flat-square&label=PyPI&color=6366f1" alt="PyPI" /></a>
  <a href="https://www.npmjs.com/package/clsplusplus"><img src="https://img.shields.io/npm/v/clsplusplus?style=flat-square&label=npm&color=cb3837" alt="npm" /></a>
  <a href="https://pypi.org/project/clsplusplus/"><img src="https://img.shields.io/pypi/pyversions/clsplusplus?style=flat-square" alt="Python" /></a>
  <a href="https://www.clsplusplus.com/docs"><img src="https://img.shields.io/badge/API-Live-22c55e?style=flat-square" alt="API" /></a>
  <a href="https://github.com/rajamohan1950/CLSplusplus/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License" /></a>
  <a href="https://github.com/rajamohan1950/CLSplusplus"><img src="https://img.shields.io/badge/Patent-Oct%202025-blue?style=flat-square" alt="Patent" /></a>
</p>

---

## What is CLS++?

Every LLM in production today operates with **amnesia**. Sessions end, context windows clear, and the model forgets everything—preferences, corrections, facts established over months.

**CLS++** is an external memory substrate that solves this at its root. Drawing from neuroscientific [Complementary Learning Systems (CLS)](https://en.wikipedia.org/wiki/Complementary_learning_systems) theory, it implements:

| Feature | Description |
|---------|-------------|
| **Four-store hierarchy** | L0 (Working Buffer) → L1 (Indexing) → L2 (Schema Graph) → L3 (Deep Recess) |
| **Biological consolidation** | Salience, Usage, Authority, Conflict, Surprise signals |
| **Sleep cycle** | Nightly maintenance: rank, decay, deduplicate, consolidate |
| **Reconsolidation gate** | Belief revision only with evidence quorum |
| **Model-agnostic** | Any LLM plugs in via REST API—Claude, GPT-4, Gemini, Llama |

Memory is **external** to the model. **Switch models anytime.** No reset.

---

## Quick Start

### Install

```bash
pip install clsplusplus          # Python (lightweight: only httpx + pydantic)
npm install clsplusplus          # JavaScript / TypeScript (zero dependencies)
```

### Python SDK

```python
from clsplusplus import Brain

brain = Brain("alice")

# Teach it anything in natural language
brain.learn("I work at Google as a senior engineer")
brain.learn("I prefer Python over JavaScript")

# Ask it anything — semantic recall, not keyword matching
brain.ask("What's my job?")           # ["I work at Google as a senior engineer"]

# Get LLM-ready context for any prompt
brain.context("coding help")
# "Known facts about this user:\n- I work at Google..."

# Forget (GDPR right to be forgotten)
brain.forget("I work at Google as a senior engineer")
```

### JavaScript / TypeScript SDK

```typescript
import { Brain } from "clsplusplus";

const brain = new Brain("alice");

await brain.learn("I work at Google as a senior engineer");
const facts = await brain.ask("What's my job?");
const context = await brain.context("coding help");
await brain.forget("I work at Google as a senior engineer");
```

### Use with OpenAI

```python
from clsplusplus import Brain

brain = Brain("alice")

# Wrap any LLM function — auto-injects memory, auto-learns
@brain.wrap
def chat(system_prompt, user_message):
    return openai.chat(system=system_prompt, user=user_message)

response = chat("You are a helpful assistant", "Help me with Python")
# Brain auto-recalls relevant memory, injects into prompt,
# calls your LLM, learns from the exchange, returns response.
```

### Full API

| Method | Description |
|--------|-------------|
| `brain.learn(fact)` | Teach a fact. Returns memory ID. |
| `brain.ask(question)` | Query for relevant facts. Returns list. |
| `brain.context(topic)` | Get LLM-ready context string. |
| `brain.forget(fact)` | Forget by text or ID. |
| `brain.absorb(text)` | Bulk-learn from document or conversation. |
| `brain.who()` | Auto-generated user profile. |
| `brain.correct(old, new)` | Update a belief. |
| `brain.chat(message, llm)` | Full conversation handler with memory. |
| `brain.teach(dict)` | Learn from structured data. |
| `brain.watch(messages)` | Learn from chat message history. |
| `brain.wrap(fn)` | Wrap any LLM function with auto-memory. |

### Run the Full Server Locally

```bash
git clone https://github.com/rajamohan1950/CLSplusplus.git
cd CLSplusplus
pip install -e ".[server]"

# Start infrastructure (Redis + PostgreSQL)
docker compose up -d redis postgres

# Start the API server
uvicorn clsplusplus.api:create_app --factory --host 0.0.0.0 --port 8080
```

---

## Try It Live

**[Try the demo](https://clsplusplus.onrender.com)** — Tell Claude something, ask OpenAI. Same memory. No sign-up.

---

## Architecture

```
Client (any LLM) → POST /v1/memory/read (before inference)
                        ↓
              ┌─────────────────────────┐
              │   CLS++ Core Service    │
              │   L0: Working Buffer    │   ← Prefrontal Cortex
              │   L1: Indexing Store    │   ← Hippocampus
              │   L2: Schema Graph      │   ← Neocortex
              │   L3: Deep Recess       │   ← Thalamus
              │   Plasticity Engine     │
              │   Sleep Orchestrator    │
              └─────────────────────────┘
                        ↓
              POST /v1/memory/write (after inference)
```

---

## SaaS Mode (Memory-as-a-Service)

Enable API key auth and rate limiting for production:

```bash
export CLS_API_KEYS=cls_live_xxxxxxxxxxxxxxxxxxxxxxxx
export CLS_REQUIRE_API_KEY=true
export CLS_RATE_LIMIT_REQUESTS=100
export CLS_RATE_LIMIT_WINDOW_SECONDS=60
```

Product endpoints: `POST /v1/memories/encode`, `POST /v1/memories/retrieve`, `DELETE /v1/memories/forget`, `GET /v1/health/score`. See [SaaS docs](docs/SAAS_MEMORY_AS_SERVICE.md).

---

## Deployment

| Platform | Guide |
|----------|-------|
| **Render** (free tier) | [Deploy in 1 click](https://render.com/deploy?repo=https://github.com/rajamohan1950/CLSplusplus) • [Setup guide](docs/DEPLOY_RENDER.md) |
| **AWS Free Tier** | [CloudFormation](infrastructure/aws/cloudformation-free-tier.yaml) • [Step-by-step](infrastructure/aws/FREE_TIER_GUIDE.md) |
| **AWS** | [CloudFormation](infrastructure/aws/cloudformation.yaml) |
| **Azure** | [ARM template](infrastructure/azure/arm-template.json) |

---

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API_DOCUMENTATION.md) | Endpoints, auth, examples |
| [API Blueprint](docs/API_BLUEPRINT.md) | SaaS API playbook (DX, security, billing) |
| [SaaS Strategy](docs/SAAS_MEMORY_AS_SERVICE.md) | Memory-as-a-Service, pricing |
| [Marketplace Integration](docs/MARKETPLACE_INTEGRATION.md) | AWS, Azure, GCP, OCI |
| [Productionization](docs/PRODUCTIONIZATION_ROADMAP.md) | Deployment, security, compliance |
| [Commercialization](docs/COMMERCIALIZATION_STRATEGY.md) | Go-to-market, licensing |

---

## Status

**Phase 1 (Foundation)** — Complete

- [x] Four stores (L0–L3) + Plasticity Engine
- [x] Write/Read API + Python SDK
- [x] Docker Compose + Render deploy
- [x] Sleep cycle orchestrator
- [x] Reconsolidation gate
- [x] API key auth + rate limiting
- [x] SaaS product endpoints

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](.github/CONTRIBUTING.md) and the [Wiki](https://github.com/rajamohan1950/CLSplusplus/wiki) for details.

---

## License

Provisional patent filed October 2025. Apache 2.0 (see [LICENSE](LICENSE)).

---

<p align="center">
  <strong>AlphaForge AI Labs</strong> • <a href="https://github.com/rajamohan1950">Rajamohan Jabbala</a> • 2026
</p>
