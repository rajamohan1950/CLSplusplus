# Fix "Connection refused" / "degraded" on Render

The API returns `"status": "degraded"` when Redis and Postgres env vars are missing.

## Add Environment Variables

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **clsplusplus-api**
3. Click **Environment** (left sidebar)
4. Add these variables:

### CLS_REDIS_URL

- Click **Add Environment Variable**
- Key: `CLS_REDIS_URL`
- Value: From your Redis service → **Info** tab → **Internal Connection String**
  - Format: `redis://red-xxxxx:6379` (use **Internal**, not External)

### CLS_DATABASE_URL

- Click **Add Environment Variable**
- Key: `CLS_DATABASE_URL`
- Value: From your Postgres service → **Info** tab → **Internal Connection String**
  - Format: `postgresql://user:password@dpg-xxxxx/dbname` (use **Internal**)

5. Click **Save Changes** — Render will redeploy automatically
6. Wait 2–3 minutes, then check: https://clsplusplus-api.onrender.com/v1/memory/health

## Enable pgvector (one-time)

In your Postgres shell (Render → Postgres → Connect):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
