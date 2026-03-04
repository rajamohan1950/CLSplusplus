# Deploy CLS++ on Render — 100% Free, No Credit Card

Render's free tier requires **no credit card**. Deploy the full stack at zero cost.

## What's Free

| Service | Free Tier |
|---------|-----------|
| **Static site** | Unlimited |
| **Web service** | 750 hrs/month (spins down after 15 min idle) |
| **Redis** | Free |
| **PostgreSQL** | Free (expires after 30 days — recreate if needed) |

**Note:** L3 storage uses PostgreSQL instead of MinIO (no persistent disk = no cost).

---

## One-Click Deploy

1. **Click:** [Deploy to Render](https://render.com/deploy?repo=https://github.com/rajamohan1950/CLSplusplus)
2. **Sign up** with GitHub (no credit card)
3. **Approve** the blueprint
4. **Wait** ~10 min for the API build
5. **Enable pgvector:** Postgres → Connect → run `CREATE EXTENSION IF NOT EXISTS vector;`
6. **Done.** Website and API are live.

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
