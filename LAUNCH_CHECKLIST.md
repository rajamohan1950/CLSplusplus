# CLS++ Free-Tier Production Launch Checklist

Operator checklist for a safe India-only, rate-limited, free-tier launch on Render.
All env vars use the `CLS_` prefix (pydantic-settings, see `src/clsplusplus/config.py`).

For incident response after launch, see [docs/RUNBOOKS.md](docs/RUNBOOKS.md).

---

## 1. Required Secrets

Launch fails, is insecure, or loses core functionality without every one of these.

| Env Var | What it's for |
|---|---|
| `CLS_JWT_SECRET` | Signs every session JWT — must be a long random string; empty = no auth |
| `CLS_DATABASE_URL` | PostgreSQL connection string (Render Postgres internal URL) |
| `CLS_REDIS_URL` | Redis connection string (Render Redis internal URL) |
| `CLS_RAZORPAY_KEY_ID` | Razorpay API key identifier (needed for checkout even if paid plans are off) |
| `CLS_RAZORPAY_KEY_SECRET` | Razorpay API secret for server-side order creation |
| `CLS_RAZORPAY_WEBHOOK_SECRET` | Razorpay webhook HMAC secret — protects the payment callback endpoint |
| `CLS_RESEND_API_KEY` | Resend transactional email key — required for waitlist invite and signup emails |
| `CLS_GOOGLE_CLIENT_ID` | Google OAuth app client ID — required for Google Sign-In |
| `CLS_GOOGLE_CLIENT_SECRET` | Google OAuth app client secret |
| `CLS_GITHUB_CLIENT_ID` | GitHub OAuth app client ID — required for GitHub Sign-In |
| `CLS_GITHUB_CLIENT_SECRET` | GitHub OAuth app client secret |

**Optional but strongly recommended:**

| Env Var | What it's for |
|---|---|
| `CLS_GOOGLE_REDIRECT_URI` | Explicit Google OAuth callback URL; set if API host differs from frontend host |
| `CLS_GITHUB_REDIRECT_URI` | Explicit GitHub OAuth callback URL; same reason as above |
| `CLS_FRONTEND_URL` | Frontend origin (`https://www.clsplusplus.com`) for post-auth redirects |
| `CLS_COOKIE_DOMAIN` | Set to `.clsplusplus.com` so session cookie works across subdomains |
| `CLS_EMAIL_FROM` | Sender address for Resend emails (default: `CLS++ <noreply@clsplusplus.com>`) |

---

## 2. Launch-Control Settings

Set these explicitly on Render — do not rely on code defaults for production-critical knobs.

### Quota enforcement

| Env Var | Value for launch | Notes |
|---|---|---|
| `CLS_ENFORCE_QUOTAS` | `true` | Block over-cap users with HTTP 402 |
| `CLS_QUOTA_FAIL_CLOSED` | `true` | On Redis outage, return 503 instead of bypassing quota |

### Free-tier multi-window caps (HTTP 429 when exceeded)

| Env Var | Recommended launch value | Code default |
|---|---|---|
| `CLS_FREE_CAP_PER_HOUR` | `120` | 120 |
| `CLS_FREE_CAP_PER_DAY` | `1000` | 1000 |
| `CLS_FREE_CAP_PER_WEEK` | `4000` | 4000 |
| `CLS_FREE_CAP_PER_MONTH` | `8000` | 8000 |

### Free-trial lifecycle caps

| Env Var | Recommended launch value | Notes |
|---|---|---|
| `CLS_FREE_LAUNCH_MONTHLY_CAP` | `8000` | Generous cap for first 30 days |
| `CLS_FREE_POSTTRIAL_MONTHLY_CAP` | `800` | Permanent hard cap after trial expires |

### Waitlist and active-user gating

| Env Var | Recommended launch value | Notes |
|---|---|---|
| `CLS_MAX_ACTIVE_USERS` | `100` | Walk-in signups beyond this go to waitlist; raise as you gain confidence |
| `CLS_WAITLIST_QUEUE_LIMIT` | `500` | Max entries in the waiting queue before new joins are refused |
| `CLS_WAITLIST_PROMOTE_BATCH` | `5` | Promote 5 waitlist users per daily promotion run |

### Geo-gating (India-only launch)

| Env Var | Value for launch | Notes |
|---|---|---|
| `CLS_GEO_GATING_ENABLED` | `true` | Route non-IN signups to waitlist instead of active status |
| `CLS_LAUNCH_COUNTRY` | `IN` | ISO-3166-1 alpha-2 country code; flip `CLS_GEO_GATING_ENABLED` to open globally |

### Abuse guard

| Env Var | Value for launch | Notes |
|---|---|---|
| `CLS_ABUSE_GUARD_ENABLED` | `true` | Enable abuse detection at launch |

### Auth endpoint per-IP throttle

| Env Var | Recommended launch value | Notes |
|---|---|---|
| `CLS_AUTH_RATE_LIMIT_PER_IP` | `10` | Max auth requests per IP per window (login/register/waitlist) |
| `CLS_AUTH_RATE_LIMIT_WINDOW_SECONDS` | `60` | Window length in seconds for per-IP throttle |

---

## 3. Things to Keep OFF for Launch

| Env Var | Launch setting | Why |
|---|---|---|
| `CLS_STRIPE_SECRET_KEY` | Leave unset (empty) | Stripe is parked — not the active payment gateway |
| `CLS_STRIPE_WEBHOOK_SECRET` | Leave unset | Same; Razorpay is active |
| `CLS_METERING_V2_WRITE_ENABLED` | `false` (or leave unset) | Schema migration runs on startup when true — safe but not needed at launch |
| `CLS_METERING_V2_READ_ENABLED` | `false` (or leave unset) | Read path not wired to production endpoints yet |
| `CLS_OVERAGE_RATES_CENTS` | Leave unset | Default config already sets free tier to 0¢ overage; free tier hard-blocks at cap rather than billing overages |

> Note on `CLS_OVERAGE_RATES_CENTS`: the code default charges 0¢ overage for free users (hard-block via 429/402). This is correct for launch. Unset = use defaults. Do NOT set a non-zero value for the free tier.

---

## 4. Pre-Launch Smoke Checklist

Run through each of these manually after deploying:

- [ ] **Signup flow** — sign up with a new Google or GitHub account; confirm email/invite lands in inbox via Resend; confirm API key is issued on the profile page
- [ ] **MCP connect** — configure the issued API key in a local MCP client; confirm `list_tools` returns the CLS++ memory tools without auth errors
- [ ] **Extension** — load the Chrome extension with the prod API URL (`https://www.clsplusplus.com`); confirm it connects and the memory icon is active
- [ ] **Metered request** — make a memory write call; confirm the usage counter in the profile page increments
- [ ] **Cap fires (HTTP 429)** — temporarily lower `CLS_FREE_CAP_PER_HOUR` to `1` on a test key, make two requests, confirm the second returns `429`; restore value
- [ ] **Quota block (HTTP 402)** — advance a test user's usage counter past their monthly cap via Redis directly and confirm the API returns `402`
- [ ] **Waitlist cap** — set `CLS_MAX_ACTIVE_USERS` below current active count on a staging deploy; confirm a new signup lands on the waitlist, not active
- [ ] **Geo-gate** — with a VPN set to a non-IN country, attempt signup; confirm new user lands on waitlist rather than getting active key (geo fails open so Indian users are never blocked)
- [ ] **Razorpay** — paid plans are disabled for launch; confirm no upgrade button is visible to free-tier users OR that clicking it surfaces a "coming soon" message rather than a broken checkout
- [ ] **Auth throttle** — hammer `/v1/auth/login` more than 10 times in 60 s from one IP; confirm `429` with `Retry-After` header
