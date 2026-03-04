# Deploy CLS++ on Render

## One-Click Deploy

1. **Click:** [Deploy to Render](https://render.com/deploy?repo=https://github.com/rajamohan1950/CLSplusplus)
2. **Sign in** to Render (or create a free account)
3. **Approve** the blueprint — Render creates all 5 services
4. **Wait** ~10 min for the API build (first deploy)
5. **Enable pgvector:** Postgres → Connect → run `CREATE EXTENSION IF NOT EXISTS vector;`
6. **Done.** Website and API will be live.

---

## Full Stack (Website + Backend)

The `render.yaml` blueprint deploys:

| Service | Type | URL |
|---------|------|-----|
| **Website** | Static | `https://clsplusplus-website.onrender.com` |
| **API** | Web (Docker) | `https://clsplusplus-api.onrender.com` |
| **Redis** | Key Value | Internal |
| **PostgreSQL** | Database | Internal |
| **MinIO** | Private Service | Internal |

### Deploy Steps

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. **New** → **Blueprint**
3. Connect GitHub repo: `rajamohan1950/CLSplusplus`
4. Render will detect `render.yaml` and create all services
5. **Enable pgvector** (one-time): After first deploy, open your Postgres in Render Dashboard → **Connect** → run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
6. If the API fails on first deploy (MinIO still starting), click **Manual Deploy** to retry

### First Deploy Notes

- **Build time:** API build takes ~5–10 min (sentence-transformers download)
- **Cold start:** Free/starter services spin down after inactivity; first request may take 30–60s
- **pgvector:** Must run `CREATE EXTENSION vector` in Postgres before L1/L2 work

### Environment Variables (auto-configured)

The blueprint wires:

- `CLS_REDIS_URL` ← from Redis
- `CLS_DATABASE_URL` ← from PostgreSQL
- `CLS_MINIO_ENDPOINT`, `CLS_MINIO_ACCESS_KEY`, `CLS_MINIO_SECRET_KEY` ← from MinIO

### Test the API

```bash
# Health
curl https://clsplusplus-api.onrender.com/v1/memory/health

# Write
curl -X POST https://clsplusplus-api.onrender.com/v1/memory/write \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode", "namespace": "user:123"}'

# Read
curl -X POST https://clsplusplus-api.onrender.com/v1/memory/read \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "namespace": "user:123"}'
```

---

## Website Only (Static)

To deploy just the marketing site:

1. **New** → **Static Site**
2. Connect repo, set **Publish Directory:** `website`
3. **Build Command:** `true`

---

## Backend Only (Manual)

To add the backend without the blueprint:

1. Create **Redis** (Key Value), **PostgreSQL**, and optionally **MinIO** (Docker pserv)
2. Create **Web Service** (Docker), point to repo
3. Set env vars manually (see above)
