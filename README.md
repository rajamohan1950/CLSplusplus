# CLS++ — Continuous Learning System++

**Switch AI models. Never lose context.**

[![Provisional Patent Filed](https://img.shields.io/badge/Patent-Oct%202025-blue)]()
[![AlphaForge AI Labs](https://img.shields.io/badge/AlphaForge-AI%20Labs-orange)]()

**Website:** [Deploy on Render](docs/DEPLOY_RENDER.md) | **GitHub:** [rajamohan1950/CLSplusplus](https://github.com/rajamohan1950/CLSplusplus)

---

## What is CLS++?

Every LLM in production today operates with **amnesia**. Sessions end, context windows clear, and the model forgets everything—preferences, corrections, facts established over months.

**CLS++** is an external memory substrate that solves this at its root. Drawing from neuroscientific Complementary Learning Systems (CLS) theory, it implements:

- **Four-store hierarchy** — L0 (Working Buffer) → L1 (Indexing) → L2 (Schema Graph) → L3 (Deep Recess)
- **Biological consolidation signals** — Salience, Usage, Authority, Conflict, Surprise
- **Sleep cycle** — Nightly maintenance: rank, decay, deduplicate, consolidate
- **Reconsolidation gate** — Belief revision only with evidence quorum
- **Social Graph** — PII-free collective intelligence from anonymized behavioral patterns

Memory is **external** to the model. **Model-agnostic** by design. Any LLM plugs in via REST API.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Start infrastructure (Redis, PostgreSQL, MinIO)
docker compose up -d redis postgres minio

# 3. Wait for services, then start the API
uvicorn clsplusplus.api:app --host 0.0.0.0 --port 8080

# 4. Or run full stack with Docker
docker compose up -d

# Write a memory
curl -X POST http://localhost:8080/v1/memory/write \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode", "namespace": "user:123"}'

# Read memories
curl -X POST http://localhost:8080/v1/memory/read \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "namespace": "user:123"}'
```

### Python SDK

```python
from clsplusplus.client import CLSClient

with CLSClient("http://localhost:8080") as client:
    client.write("User prefers dark mode", namespace="user:123")
    results = client.read("user preferences", namespace="user:123")
    for item in results.items:
        print(item.text, item.confidence)
```

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

## Documentation

| Document | Description |
|----------|-------------|
| [HLD](docs/CLS_Plus_Plus_HLD.docx) | High-Level Design (full specification) |
| [Productionization Roadmap](docs/PRODUCTIONIZATION_ROADMAP.md) | Deployment, security, compliance |
| [Commercialization Strategy](docs/COMMERCIALIZATION_STRATEGY.md) | Go-to-market, pricing, licensing |

---

## Status

**Phase 1 (Foundation)** — Complete

- [x] Four stores (L0–L3) + Plasticity Engine
- [x] Write/Read API
- [x] Docker Compose
- [x] Python SDK
- [x] Sleep cycle orchestrator
- [x] Reconsolidation gate

---

## License

Provisional patent filed October 2025. License TBD (Apache 2.0 / Commercial dual).

---

**AlphaForge AI Labs** | [Rajamohan Jabbala](https://github.com/rajamohan1950) | February 2026
