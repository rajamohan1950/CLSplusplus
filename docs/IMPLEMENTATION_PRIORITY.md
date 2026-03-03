# CLS++ Implementation Priority — From HLD to Production

**Tie-in document** connecting the HLD, productionization roadmap, and commercialization strategy into actionable sprints.

---

## Phase 0: Foundation (Weeks 1–2)

**Goal:** Repo structure, CI, and minimal "hello world" API

| Task | Owner | Deliverable |
|------|-------|-------------|
| Create repo structure | Dev | `src/`, `tests/`, `docker/`, `docs/` |
| FastAPI skeleton | Dev | `POST /v1/memory/write`, `POST /v1/memory/read` (stub) |
| Docker Compose | Dev | `docker-compose.yml` with Redis, PostgreSQL |
| GitHub Actions | Dev | Lint, test, build on push |
| OpenAPI spec | Dev | Auto-generated from FastAPI |

**Exit:** `curl` to write/read returns 200 with stub response

---

## Phase 1: Four Stores (Weeks 3–8)

**Goal:** All stores operational; plasticity at mathematical spec

| Task | Store | Deliverable |
|------|-------|-------------|
| L0 Working Buffer | L0 | In-process deque + Redis snapshot; TTL eviction |
| L1 Indexing Store | L1 | pgvector + FAISS; embeddings via all-MiniLM-L6-v2 |
| L2 Schema Graph | L2 | Neo4j or custom KV-graph; edge weight decay |
| L3 Deep Recess | L3 | MinIO + Parquet; VQ compression |
| Plasticity Engine | All | Score = α·S + β·log(1+U) + γ·A − λ·C + δ·Δ |
| Promotion pipeline | All | L0→L1→L2→L3 thresholds |

**Exit:** 60 unit tests; CI green; all stores operational

---

## Phase 2: Deep Memory (Weeks 9–12)

**Goal:** Engram formation, reconsolidation gate, adjudication

| Task | Deliverable |
|------|-------------|
| Reconsolidation gate | Quorum-based belief revision; archive old, engrave new |
| Engram formation | L2→L3 with confidence ≥ 0.85, usage_days ≥ 5 |
| Adjudication API | `POST /v1/memory/adjudicate_conflict` |
| Conflict scoring | Cosine similarity + semantic contradiction |

**Exit:** Day 5 simulation; wrong-update rate = 0

---

## Phase 3: Sleep + Eval (Weeks 13–16)

**Goal:** Sleep cycle, dream replay, quality metrics

| Task | Deliverable |
|------|-------------|
| Sleep cycle (4 phases) | N1–N2–N3–REM; 60-min budget; idempotent |
| Dream replay | Re-embedding of high-salience items |
| LHRA/DR/CP eval harnesses | Benchmark scripts |
| Grafana dashboard | Per-store metrics, health score |

**Exit:** LHRA 14d ≥ 0.90; DR ≥ 0.98; sleep completes in 60 min

---

## Phase 4: Multi-Tenant + Privacy (Weeks 17–20)

**Goal:** Enterprise-ready; zero cross-tenant leakage

| Task | Deliverable |
|------|-------------|
| RLS (Row-Level Security) | Per-tenant isolation in PostgreSQL |
| Per-tenant encryption | Namespace-scoped keys |
| RTBF | `DELETE /v1/memory/forget`; tombstone + compaction |
| Social Graph | PII scanner, Leiden community detection |
| Consent enforcement | Consent tags on memory items |

**Exit:** Zero cross-tenant leakage; RTBF SLA met

---

## Phase 5: Ship v1 (Weeks 21–24)

**Goal:** Production-ready, sellable

| Task | Deliverable |
|------|-------------|
| Load test | 200 RPS sustained |
| Chaos tests | Kill Redis, DB failover, network partition |
| Python SDK | `pip install clsplusplus` |
| Landing page | clsplusplus.ai |
| Stripe integration | Team/Enterprise products |
| Terms, Privacy, SLA | Legal docs |

**Exit:** All SLOs met; P95 read < 120ms; LHRA 14d ≥ 0.90

---

## Parallel Tracks (Ongoing)

| Track | Tasks |
|-------|-------|
| **Documentation** | API docs, integration guides, architecture diagrams |
| **Security** | Pen test, secret management, audit logging |
| **Compliance** | GDPR checklist, HIPAA BAA template, SOC 2 prep |
| **Sales** | Landing page, case study template, pricing page |

---

## Recommended First Sprint (2 weeks)

1. **Day 1–2:** Repo scaffold, FastAPI + Docker Compose
2. **Day 3–5:** L0 + L1 (Redis + pgvector)
3. **Day 6–8:** Embeddings pipeline, write flow end-to-end
4. **Day 9–10:** Read flow, basic plasticity scoring
5. **Day 11–12:** Unit tests, CI, README

**Outcome:** Working prototype that can store and retrieve memories with basic promotion logic.
