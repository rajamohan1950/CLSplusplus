# CLS++ as Memory-as-a-Service (SaaS)

**Goal:** Package the current CLS++ feature set as a sellable SaaS API. Use existing capabilities as-is; add SaaS essentials (auth, billing, DX).

---

## 1. Current API → Product Strategy Mapping

| Product Endpoint | Brain Analog | Current CLS++ | Status |
|------------------|--------------|---------------|--------|
| `POST /memories/encode` | Hippocampal encoding | `POST /v1/memory/write` | ✅ Direct map |
| `GET /memories/retrieve` | Recall + pattern matching | `POST /v1/memory/read` | ✅ Same logic; consider GET with query params for REST purity |
| `POST /memories/consolidate` | Sleep consolidation | `POST /v1/memory/sleep` | ✅ Direct map |
| `GET /knowledge/query` | Neocortical knowledge | Read with `store_levels: [L2, L3]` | ✅ Exists via read; add thin alias |
| `DELETE /memories/forget` | Adaptive forgetting | Sleep cycle prunes (N2) | ⚠️ Implicit; add explicit DELETE for RTBF |
| `POST /goals/set` | Prefrontal goal context | — | 🔲 Future: goal biases retrieval |
| `GET /health/score` | Metacognition | `GET /v1/memory/health` | ✅ Direct map |

**Differentiators already in place:**

| Feature | Implementation |
|---------|----------------|
| **Consolidation Engine** | `SleepOrchestrator` — N1 rank, N2 strengthen+decay, N3 dedupe, REM L1→L2→L3 |
| **Adaptive Forgetting** | `PlasticityEngine.apply_decay`, `should_prune` in sleep cycle |
| **Cross-Model Portability** | Namespace-based; demo shows Claude ↔ OpenAI shared memory |
| **Goal-Directed Retrieval** | Not yet — would bias `read()` by goal embedding |

---

## 2. SaaS Build-Out (Using Current Features)

### 2.1 API Surface (Minimal Changes)

Add a **product-aligned route layer** that proxies to existing handlers. Keeps `/v1/memory/*` for backward compatibility; adds `/v1/memories/*` for SaaS branding.

```
POST /v1/memories/encode     → /v1/memory/write
POST /v1/memories/retrieve   → /v1/memory/read  (or GET with ?query=)
POST /v1/memories/consolidate → /v1/memory/sleep
GET  /v1/memories/knowledge  → /v1/memory/read?store_levels=L2,L3
DELETE /v1/memories/forget   → new: explicit prune by id or query
GET  /v1/health/score        → /v1/memory/health
```

### 2.2 SaaS Essentials

| Layer | What to Add | Effort |
|-------|-------------|--------|
| **Auth** | API key in `Authorization: Bearer <key>`; validate per request | 1–2 days |
| **Tenant isolation** | `namespace` = tenant ID (or derive from API key) | Already supported |
| **Rate limits** | Per-key limits (e.g. 100 req/min free, 1000 paid) | 1–2 days |
| **Usage tracking** | Log writes/reads per key for billing | 1 day |
| **Billing hook** | Webhook or Stripe metering when limits exceeded | 2–3 days |

### 2.3 Developer Experience (Beat Mem0’s 3-Line Integration)

**Target: 3 lines or fewer**

```python
# Python
from clsplusplus import CLSClient
client = CLSClient(api_key="cls_xxx")
client.write("User prefers dark mode")
```

```javascript
// JavaScript / Vercel AI SDK
import { CLSClient } from "@clsplusplus/sdk";
const cls = new CLSClient({ apiKey: process.env.CLS_API_KEY });
await cls.encode("User prefers dark mode");
```

**Current client:** `CLSClient` exists; add `api_key` to constructor (already supported). Publish `pip install clsplusplus` and `npm install @clsplusplus/sdk`.

---

## 3. Integration Points (Roadmap)

| Integration | Priority | Notes |
|-------------|----------|-------|
| **LangChain** | P0 | `LangChainMemory` backend; 1 adapter class |
| **LangGraph** | P0 | Same as LangChain; checkpointer + memory |
| **Vercel AI SDK** | P1 | `useChat` + memory context injection |
| **CrewAI / AutoGen** | P1 | Shared memory layer for multi-agent |
| **AWS Bedrock Agent** | P2 | Alternative to Mem0’s exclusive deal |
| **Azure AI Agent** | P2 | Microsoft partnership |
| **gRPC** | P2 | For high-throughput; REST first |

---

## 4. Pricing Model (Draft)

| Tier | Writes/mo | Reads/mo | Consolidations | Price |
|------|-----------|----------|----------------|-------|
| Free | 1,000 | 5,000 | 1/day | $0 |
| Pro | 50,000 | 250,000 | 10/day | $49/mo |
| Team | 500,000 | 2.5M | Unlimited | $199/mo |
| Enterprise | Custom | Custom | Custom | Contact |

**Metering:** Count `write` and `read` calls per API key. Consolidation = `sleep` trigger (or cron).

---

## 5. Implementation Status

**Phase A: API Aliases** ✅  
- `/v1/memories/encode`, `/v1/memories/retrieve`, `/v1/memories/consolidate`, `/v1/memories/forget`, `/v1/health/score`, `/v1/memories/knowledge`

**Phase B: API Key Auth** ✅  
- `AuthMiddleware`: `Authorization: Bearer <key>`, constant-time validation (SHA-256 + hmac.compare_digest)  
- `CLS_REQUIRE_API_KEY`, `CLS_API_KEYS` (comma-separated)  
- Key format: `cls_live_*` or `cls_test_*` (min 32 chars)

**Phase C: Rate Limits** ✅  
- `RateLimitMiddleware`: Redis sliding window per API key  
- `CLS_RATE_LIMIT_REQUESTS`, `CLS_RATE_LIMIT_WINDOW_SECONDS`  
- 429 with `Retry-After`, `X-RateLimit-*` headers  

**Phase D: Security** ✅  
- Input validation: namespace, item_id, text length, limit bounds  
- L2 delete SQL fix (parameterized)

**Phase E: DX + Billing**  
- Python client has `forget()`, API key support  
- LangChain adapter, Stripe: future

---

## 6. What Stays As-Is

- Memory service, stores (L0–L3), plasticity, sleep cycle  
- Embeddings, reconsolidation gate  
- Demo (Claude/OpenAI shared memory)  
- Health endpoint  

---

## 7. Next Step

**Recommended:** Start with Phase A (API aliases) + Phase B (API key auth). That gives you a sellable API surface with minimal risk. Add usage/billing once you have early customers.
