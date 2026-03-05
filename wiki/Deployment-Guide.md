# Deployment Guide

---

## Render (Recommended for Quick Start)

**1-click deploy:** [Deploy to Render](https://render.com/deploy?repo=https://github.com/rajamohan1950/CLSplusplus)

1. Sign in with GitHub
2. Approve the deploy
3. Add environment variables (Redis, PostgreSQL, etc.) — see [DEPLOY_RENDER.md](../docs/DEPLOY_RENDER.md)

---

## AWS Free Tier

**100% free for 12 months.** Uses t3.micro + db.t3.micro. Redis runs on EC2.

1. [Download the template](../infrastructure/aws/cloudformation-free-tier.yaml)
2. AWS Console → CloudFormation → Create stack → Upload template
3. Enter `DbPassword` (required)
4. Follow [FREE_TIER_GUIDE.md](../infrastructure/aws/FREE_TIER_GUIDE.md)

---

## AWS CloudFormation (Paid)

```bash
aws cloudformation create-stack \
  --stack-name clsplusplus \
  --template-body file://infrastructure/aws/cloudformation.yaml \
  --parameters \
    ParameterKey=DbPassword,ParameterValue='YourSecureP@ss1' \
    ParameterKey=AnthropicApiKey,ParameterValue='sk-ant-...' \
    ParameterKey=OpenAIApiKey,ParameterValue='sk-...' \
  --capabilities CAPABILITY_IAM
```

---

## Azure ARM

```bash
az group create --name clsplusplus-rg --location eastus

az deployment group create \
  --resource-group clsplusplus-rg \
  --template-file infrastructure/azure/arm-template.json \
  --parameters infrastructure/azure/parameters.json
```

---

## Docker Compose (Local)

```bash
docker compose up -d
```

Starts Redis, PostgreSQL, MinIO, and the API.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CLS_REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `CLS_POSTGRES_URL` | PostgreSQL + pgvector | — |
| `CLS_MINIO_*` | MinIO (optional) | — |
| `CLS_REQUIRE_API_KEY` | Enforce API key | `false` |
| `CLS_API_KEYS` | Comma-separated keys | — |
| `CLS_RATE_LIMIT_REQUESTS` | Per-window limit | `100` |
| `CLS_RATE_LIMIT_WINDOW_SECONDS` | Window size | `60` |

---

[← API Reference](API-Reference) | [Integration Examples →](Integration-Examples)
