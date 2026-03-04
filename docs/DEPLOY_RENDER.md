# Deploy CLS++ on Render — 100% Free, No Credit Card

Render allows **only 1 free Postgres** and **1 free Redis** per account. This blueprint uses your existing ones.

## Before Deploy

1. **Create free Postgres** (if you don't have one): Dashboard → New → PostgreSQL → Free
2. **Create free Redis** (if you don't have one): Dashboard → New → Redis → Free
3. **Enable pgvector** in Postgres: Connect → run `CREATE EXTENSION IF NOT EXISTS vector;`
4. Copy the **Internal Connection String** from each (Dashboard → your service → Info)

## Deploy

1. **Click:** [Deploy to Render](https://render.com/deploy?repo=https://github.com/rajamohan1950/CLSplusplus)
2. **When prompted**, paste:
   - `CLS_REDIS_URL` → your Redis internal connection string
   - `CLS_DATABASE_URL` → your Postgres internal connection string
3. **Wait** ~10 min for the API build
4. **Done.** Website and API are live.

---

## Free Tier Limits

- **API cold start:** First request after 15 min idle takes ~1 min to wake
- **Postgres:** Free DB expires after 30 days — export data and recreate if needed
- **750 hours:** Shared across all free web services; usually enough for one API

---

## URLs After Deploy

- **Website:** `https://clsplusplus-website.onrender.com`
- **API:** `https://clsplusplus-api.onrender.com`
- **API docs:** `https://clsplusplus-api.onrender.com/docs`

---

## Test the API

```bash
curl https://clsplusplus-api.onrender.com/v1/memory/health
```
