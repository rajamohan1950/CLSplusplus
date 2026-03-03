-- Enable pgvector for CLS++ L1/L2 stores
-- Run once: psql $DATABASE_URL -f scripts/init-pgvector.sql
-- Or in Render Dashboard: Postgres → Connect → run this
CREATE EXTENSION IF NOT EXISTS vector;
