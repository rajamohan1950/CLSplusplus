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

    # Embeddings
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

    # Demo LLM keys (optional; demo uses these for real Claude/OpenAI/Gemini)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
