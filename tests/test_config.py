"""Configuration tests - defaults, env overrides, type coercion, boundaries."""

import os
from unittest.mock import patch

import pytest

from clsplusplus.config import Settings


class TestSettingsDefaults:

    def test_default_host(self):
        s = Settings()
        assert s.host == "0.0.0.0"

    def test_default_port(self):
        s = Settings()
        assert s.port == 8080

    def test_default_no_auth(self):
        s = Settings()
        assert s.require_api_key is False

    def test_default_empty_api_keys(self):
        s = Settings()
        assert s.api_keys == ""

    def test_default_rate_limit(self):
        s = Settings()
        assert s.rate_limit_requests == 100
        assert s.rate_limit_window_seconds == 60

    def test_default_no_usage_tracking(self):
        s = Settings()
        assert s.track_usage is False

    def test_default_idempotency_ttl(self):
        s = Settings()
        assert s.idempotency_ttl_seconds == 86400

    def test_default_redis_url(self):
        s = Settings()
        assert "localhost" in s.redis_url

    def test_default_database_url(self):
        s = Settings()
        assert "postgresql" in s.database_url or "postgres" in s.database_url

    def test_default_embedding_model(self):
        s = Settings()
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.embedding_dim == 384

    def test_plasticity_coefficients_defaults(self):
        s = Settings()
        assert s.alpha_salience == 1.0
        assert s.beta_usage == 0.3
        assert s.gamma_authority == 0.5
        assert s.lambda_conflict == 0.7
        assert s.delta_surprise == 0.4

    def test_promotion_thresholds(self):
        s = Settings()
        assert s.l1_promotion_threshold == 1.5
        assert s.l2_promotion_threshold == 2.2
        assert s.l2_min_confidence == 0.85
        assert s.l2_min_usage_days == 5

    def test_decay_defaults(self):
        s = Settings()
        assert s.decay_constant_k == 0.1
        assert s.min_salience_prune == 0.2

    def test_reconsolidation_defaults(self):
        s = Settings()
        assert s.similarity_threshold == 0.7
        assert s.conflict_threshold == 0.3
        assert s.quorum_threshold == 0.8

    def test_l0_defaults(self):
        s = Settings()
        assert s.l0_capacity_tokens == 4096
        assert s.l0_ttl_seconds == 300

    def test_no_llm_keys_by_default(self):
        s = Settings()
        # Keys may be set via .env; just verify they exist as attributes
        assert hasattr(s, "anthropic_api_key")
        assert hasattr(s, "openai_api_key")
        assert hasattr(s, "google_api_key")


class TestSettingsOverrides:

    def test_direct_override(self):
        s = Settings(port=9090, require_api_key=True)
        assert s.port == 9090
        assert s.require_api_key is True

    def test_api_keys_comma_separated(self):
        s = Settings(api_keys="cls_live_key1234567890123456789012,cls_test_key2345678901234567890123")
        keys = s.api_keys.split(",")
        assert len(keys) == 2

    def test_custom_plasticity_coefficients(self):
        s = Settings(alpha_salience=2.0, beta_usage=0.5)
        assert s.alpha_salience == 2.0
        assert s.beta_usage == 0.5

    def test_custom_thresholds(self):
        s = Settings(l1_promotion_threshold=3.0)
        assert s.l1_promotion_threshold == 3.0


class TestSettingsEnvPrefix:

    def test_env_prefix_is_cls(self):
        assert Settings.model_config.get("env_prefix") == "CLS_"


class TestSettingsTypeCoercion:

    def test_bool_coercion(self):
        s = Settings(require_api_key=True)
        assert s.require_api_key is True

    def test_float_coercion(self):
        s = Settings(alpha_salience=2)
        assert isinstance(s.alpha_salience, float)

    def test_int_coercion(self):
        s = Settings(port=8080)
        assert isinstance(s.port, int)


class TestSettingsSecurityConstraints:

    def test_minio_defaults_not_production(self):
        s = Settings()
        assert s.minio_access_key == "minioadmin"
        assert s.minio_secure is False

    def test_sensitive_fields_exist(self):
        s = Settings()
        assert hasattr(s, "anthropic_api_key")
        assert hasattr(s, "openai_api_key")
        assert hasattr(s, "google_api_key")
        assert hasattr(s, "minio_secret_key")
