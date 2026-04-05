-- CLS++ Monthly Metrics Snapshots
-- Flushed from Redis at end of each billing period.

CREATE TABLE IF NOT EXISTS monthly_metrics (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    period          TEXT NOT NULL,
    operations      INT NOT NULL DEFAULT 0,
    reads           INT NOT NULL DEFAULT 0,
    writes          INT NOT NULL DEFAULT 0,
    searches        INT NOT NULL DEFAULT 0,
    deletes         INT NOT NULL DEFAULT 0,
    llm_tokens_in   BIGINT NOT NULL DEFAULT 0,
    llm_tokens_out  BIGINT NOT NULL DEFAULT 0,
    embeddings      INT NOT NULL DEFAULT 0,
    consolidations  INT NOT NULL DEFAULT 0,
    total_cost      NUMERIC(10,4) NOT NULL DEFAULT 0,
    revenue         NUMERIC(10,2) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, period)
);

CREATE INDEX IF NOT EXISTS idx_monthly_metrics_period ON monthly_metrics(period);
CREATE INDEX IF NOT EXISTS idx_monthly_metrics_user ON monthly_metrics(user_id);
