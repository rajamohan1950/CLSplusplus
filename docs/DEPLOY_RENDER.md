# Deploy CLS++ on Render

## Website (Static Site)

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. **New** → **Static Site**
3. Connect your GitHub repo: `rajamohan1950/CLSplusplus`
4. Configure:
   - **Build Command:** `true`
   - **Publish Directory:** `website`
   - **Branch:** `main`
5. Click **Create Static Site**
6. Your site will be live at `https://clsplusplus-website.onrender.com` (or your custom domain)

### Using Blueprint (render.yaml)

1. **New** → **Blueprint**
2. Connect repo `rajamohan1950/CLSplusplus`
3. Render will detect `render.yaml` and create the static site automatically

---

## Backend API (Optional)

To deploy the CLS++ API on Render:

1. **Create Redis:** New → Redis (Key Value) → Free plan
2. **Create PostgreSQL:** New → PostgreSQL → Free plan
3. Run in Postgres shell: `CREATE EXTENSION IF NOT EXISTS vector;`
4. **Create MinIO:** Use a Docker private service or external S3-compatible storage (e.g., Cloudflare R2)
5. **Create Web Service:** New → Web Service
   - Connect repo
   - **Runtime:** Docker
   - **Dockerfile Path:** `./Dockerfile`
   - **Environment Variables:**
     - `CLS_REDIS_URL` → from Redis connection string
     - `CLS_DATABASE_URL` → from Postgres connection string
     - `CLS_MINIO_ENDPOINT`, `CLS_MINIO_ACCESS_KEY`, `CLS_MINIO_SECRET_KEY` → from MinIO or S3

---

## Video Embed

After recording your demo video:

1. Upload to YouTube, Vimeo, or Loom
2. Get the embed URL (e.g., `https://www.youtube.com/embed/VIDEO_ID`)
3. Edit `website/index.html` and set the iframe `src` attribute
4. Remove or hide the `.video-overlay` div
5. Push to GitHub — Render will auto-redeploy
