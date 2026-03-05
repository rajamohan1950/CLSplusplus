# CLS++ Infrastructure

One-template deployment for AWS and Azure. Provisions Redis, PostgreSQL (pgvector), and the API (Docker on VM).

## AWS Free Tier (Recommended for Trying Out)

**100% free for 12 months.** Uses t3.micro + db.t3.micro. Redis runs on the same EC2 (no ElastiCache).

- **Template:** `aws/cloudformation-free-tier.yaml`
- **Guide:** [aws/FREE_TIER_GUIDE.md](aws/FREE_TIER_GUIDE.md) — step-by-step for beginners

Upload the template in AWS Console → CloudFormation → Create stack. Enter a password. Done.

---

## AWS CloudFormation (Paid)

```bash
aws cloudformation create-stack \
  --stack-name clsplusplus \
  --template-body file://aws/cloudformation.yaml \
  --parameters \
    ParameterKey=DbPassword,ParameterValue='YourSecureP@ss1' \
    ParameterKey=AnthropicApiKey,ParameterValue='sk-ant-...' \
    ParameterKey=OpenAIApiKey,ParameterValue='sk-...' \
  --capabilities CAPABILITY_IAM
```

Or upload `aws/cloudformation.yaml` in the AWS Console → CloudFormation → Create stack.

**Parameters:**
- `DbPassword` (required): PostgreSQL master password
- `GitHubRepo` (optional): Default `https://github.com/rajamohan1950/CLSplusplus`
- `AnthropicApiKey` (optional): For demo Claude
- `OpenAIApiKey` (optional): For demo OpenAI

**Outputs:** `ApiUrl`, `HealthUrl` — allow 10–15 min for EC2 UserData (Docker build).

---

## Azure ARM

```bash
az group create --name clsplusplus-rg --location eastus

az deployment group create \
  --resource-group clsplusplus-rg \
  --template-file azure/arm-template.json \
  --parameters azure/parameters.json
```

Or deploy via Portal: Resource group → Deploy a custom template → upload `arm-template.json`.

**Parameters:**
- `dbPassword` (required): PostgreSQL admin password
- `projectName` (optional): Base name, max 12 chars
- `anthropicApiKey` (optional): For demo
- `openaiApiKey` (optional): For demo

**Outputs:** `apiUrl`, `healthUrl` — allow 10–15 min for VM extension (Docker build).
