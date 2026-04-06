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

    # SaaS: Require auth for memory endpoints (false = open for local/demo)
    require_api_key: bool = False

    # SaaS: Rate limit - requests per window per key
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # SaaS: Usage tracking for billing (marketplace)
    track_usage: bool = False

    # SaaS: Tier and quota enforcement
    tier: str = "free"                # free | pro | unlimited
    enforce_quotas: bool = False      # Enable quota enforcement (off for local/demo)

    # User auth (JWT + Google OAuth)
    jwt_secret: str = ""                  # CLS_JWT_SECRET (required for user auth)
    google_client_id: str = ""            # CLS_GOOGLE_CLIENT_ID
    google_client_secret: str = ""        # CLS_GOOGLE_CLIENT_SECRET

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

    # Stripe billing
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro: str = ""            # Stripe Price ID for Pro tier
    stripe_price_business: str = ""       # Stripe Price ID for Business tier
    stripe_price_enterprise: str = ""     # Stripe Price ID for Enterprise tier
    stripe_success_url: str = "/profile.html?billing=success"
    stripe_cancel_url: str = "/profile.html?billing=cancel"

    # Demo LLM keys (optional; demo uses these for real Claude/OpenAI/Gemini)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
