"""CLS++ configuration."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="CLS_", env_file=".env")

    # API
    host: str = "0.0.0.0"
    port: int = 8080

    # SaaS: API key auth (comma-separated for multiple keys)
    api_keys: str = ""

    # SaaS: Require auth for memory endpoints (true = secure default; set CLS_REQUIRE_API_KEY=false for local/demo)
    require_api_key: bool = True

    # SaaS: Rate limit - requests per window per key
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # SaaS: Usage tracking for billing (marketplace)
    track_usage: bool = False

    # SaaS: Tier and quota enforcement
    tier: str = "free"                # DEPRECATED fallback. Real per-user tier
                                      # comes from the DB via TierResolver.
                                      # Used only when resolution returns None.
    enforce_quotas: bool = True       # Block over-cap users with 402. Safe
                                      # default now that tier resolution is
                                      # per-user (set CLS_ENFORCE_QUOTAS=false
                                      # for local/demo stacks).
    # When Redis is unreachable, the quota check cannot tell whether a user
    # is over their cap. Default: fail-CLOSED (return 503) so we never give
    # away unlimited billable usage during an outage. Flip to false only if
    # availability matters more than billing accuracy during Redis outages.
    quota_fail_closed: bool = True    # CLS_QUOTA_FAIL_CLOSED

    # User auth (JWT + Google OAuth + GitHub OAuth)
    jwt_secret: str = ""                  # CLS_JWT_SECRET (required for user auth)
    google_client_id: str = ""            # CLS_GOOGLE_CLIENT_ID
    google_client_secret: str = ""        # CLS_GOOGLE_CLIENT_SECRET
    github_client_id: str = ""            # CLS_GITHUB_CLIENT_ID
    github_client_secret: str = ""        # CLS_GITHUB_CLIENT_SECRET
    # Optional explicit redirect URIs; if empty, computed from request.base_url.
    # Set these when the OAuth callback must go through a frontend proxy (e.g.
    # Vercel rewrites) so Set-Cookie lands on the UI host, not the API host.
    google_redirect_uri: str = ""         # CLS_GOOGLE_REDIRECT_URI
    github_redirect_uri: str = ""         # CLS_GITHUB_REDIRECT_URI

    # Frontend origin for post-auth redirects. Needed when the API host
    # (e.g. api.clsplusplus.com) differs from the UI host (www.clsplusplus.com).
    # Leave empty to use same-origin relative redirects.
    frontend_url: str = ""                # CLS_FRONTEND_URL

    # Idempotency: cache window (seconds)
    idempotency_ttl_seconds: int = 86400  # 24 hours

    # Redis (L0)
    redis_url: str = "redis://localhost:6379"

    # PostgreSQL (L1)
    database_url: str = "postgresql://cls:cls@localhost:5432/cls"

    # MinIO (L3)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket: str = "cls-l3"

    # Embeddings (used by L1/L2/L3 tiers, not by phase engine)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Plasticity coefficients (α, β, γ, λ, δ)
    alpha_salience: float = 1.0
    beta_usage: float = 0.3
    gamma_authority: float = 0.5
    lambda_conflict: float = 0.7
    delta_surprise: float = 0.4

    # Promotion thresholds
    l1_promotion_threshold: float = 1.5
    l2_promotion_threshold: float = 2.2
    l2_min_confidence: float = 0.85
    l2_min_usage_days: int = 5

    # Decay
    decay_constant_k: float = 0.1
    min_salience_prune: float = 0.2

    # Reconsolidation
    similarity_threshold: float = 0.7
    conflict_threshold: float = 0.3
    quorum_threshold: float = 0.8

    # Working buffer
    l0_capacity_tokens: int = 4096
    l0_ttl_seconds: int = 300

    # Phase dynamics — Gas → Liquid transition
    # Maps 1:1 to: F(θ, Σ, ρ, τ) = E_pred − Σ·S_model + λ·L_landauer
    phase_kT: float = 1.0                # Boltzmann analog (energy scale for Landauer cost)
    phase_lambda: float = 0.5            # Energy budget constraint λ (scales Landauer term in F)
    phase_tau_c1: float = 10.0           # Critical τ for gas→liquid phase boundary
    phase_tau_default: float = 50.0      # τ for normal factual statements
    phase_tau_override: float = 200.0    # τ for override statements ("only", "exclusively")
    phase_strength_floor: float = 0.05   # s < floor → gas phase (not retrievable)
    phase_capacity: int = 1000           # Max items per namespace (denominator for ρ)
    phase_beta_retrieval: float = 0.15   # Retrieval reinforcement: s *= (1 + β·ln(1+R))

    # Temporal recency decay
    # half-life used when query has no temporal signal (fallback)
    temporal_recency_half_life_days: float = 90.0
    # blend weight for queries with no temporal signal ("what are my hobbies?")
    temporal_recency_alpha_default: float = 0.1
    # blend weight for strong recency signals ("recently", "last week", "yesterday")
    temporal_recency_alpha_strong: float = 0.5

    # Site
    site_base_url: str = "https://www.clsplusplus.com"  # CLS_SITE_BASE_URL
    cookie_secure: bool = True  # CLS_COOKIE_SECURE (False for local dev only)
    # Cookie Domain attribute. Set to ".clsplusplus.com" in prod so the session
    # cookie set by api.clsplusplus.com is also sent on www.clsplusplus.com.
    # Leave empty for same-host dev (cookie stays host-only).
    cookie_domain: str = ""  # CLS_COOKIE_DOMAIN

    # Metering v2 — append-only event log pipeline. See docs/adr/0001-metering-data-lake.md.
    # Rollout is in small, reversible steps. Both flags default to False; no
    # production path consults these yet. Flipping the write flag causes the
    # new schema to be applied on startup; it does NOT enable any writer.
    metering_v2_write_enabled: bool = False  # CLS_METERING_V2_WRITE_ENABLED
    metering_v2_read_enabled: bool = False   # CLS_METERING_V2_READ_ENABLED

    # Oncall address that receives metering_dead_letter paging digests.
    # Default is the project owner; override per-deployment via env var.
    oncall_email: str = "rjabbala@gmail.com"  # CLS_ONCALL_EMAIL

    # Per-tier per-event overage rates in cents, applied to operations that
    # cross `TIER_LIMITS[tier]["ops_per_month"]`.
    #
    # Defaults are a conservative starting point — adjust via CLS_OVERAGE_RATES_CENTS
    # when you want different numbers. Rationale:
    #   free       0¢  — hard-block is better than surprise bill.
    #   pro        2¢  — $9/mo gives 50k ops ≈ 0.018¢/op. Overage @ 2¢ is ~11×
    #                   markup → strong upgrade signal toward Business.
    #   business   1¢  — $29/mo for 200k ≈ 0.0145¢/op. Overage @ 1¢ is ~7× markup.
    #   enterprise 0¢  — flat price includes elastic usage (these customers want
    #                   predictability; charge-on-overage would be a surprise).
    #
    # Override with JSON env var, e.g.
    # CLS_OVERAGE_RATES_CENTS='{"pro":{"_default":3,"write":5}}'
    overage_rates_cents: dict = {
        "free":       {"_default": 0},
        "pro":        {"_default": 2},
        "business":   {"_default": 1},
        "enterprise": {"_default": 0},
    }  # CLS_OVERAGE_RATES_CENTS (JSON)

    # Razorpay billing (active payment gateway)
    razorpay_key_id: str = ""               # CLS_RAZORPAY_KEY_ID
    razorpay_key_secret: str = ""           # CLS_RAZORPAY_KEY_SECRET
    razorpay_webhook_secret: str = ""       # CLS_RAZORPAY_WEBHOOK_SECRET

    # Stripe billing (parked — not active)
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro: str = ""            # Stripe Price ID for Pro tier
    stripe_price_business: str = ""       # Stripe Price ID for Business tier
    stripe_price_enterprise: str = ""     # Stripe Price ID for Enterprise tier
    stripe_success_url: str = "/profile.html?billing=success"
    stripe_cancel_url: str = "/profile.html?billing=cancel"

    # Email (Resend)
    resend_api_key: str = ""                  # CLS_RESEND_API_KEY
    email_from: str = "CLS++ <noreply@clsplusplus.com>"  # CLS_EMAIL_FROM

    # ── Launch waitlist / rate-limited rollout ────────────────────────────
    # Hard cap on users with an active API key. Walk-in signups above this
    # are redirected to the waitlist. Waitlist-invited users bypass the cap.
    # Set to 0 to disable the cap entirely.
    max_active_users: int = 50
    # Hard upper bound on the waiting queue. When waiting_count >= this,
    # /v1/waitlist/join refuses new entries and the widget shows "queue full".
    waitlist_queue_limit: int = 50
    # DEPRECATED — kept to avoid config crashes on old deploys. The public
    # stats endpoint no longer applies any seeding or floor to the live
    # counters, so these values are ignored.
    waitlist_queue_seed_offset: int = 0
    waitlist_active_floor: int = 0
    # Daily promotion loop: promote the oldest N waiting visitors iff DAU is
    # below the healthy threshold.
    waitlist_promote_batch: int = 1
    waitlist_dau_healthy_threshold: int = 5
    waitlist_promote_interval_seconds: int = 86400  # 24h

    # Demo LLM keys (optional; demo uses these for real Claude/OpenAI/Gemini)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
