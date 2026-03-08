# CLS++ Marketplace Integration Guide

**Memory-as-a-Service** — Deploy and distribute CLS++ on AWS, Azure, GCP, OCI, and other cloud marketplaces.

---

## Overview

| Marketplace | Listing Type | Requirements |
|-------------|-------------|--------------|
| **AWS Marketplace** | SaaS (API) | APN, MCP, usage metering |
| **Azure Marketplace** | SaaS (API) | Partner Center, transactable offer |
| **GCP Marketplace** | SaaS (API) | Partner profile, billing integration |
| **OCI Marketplace** | SaaS (API) | Oracle Partner Network, listing |

---

## 1. AWS Integration

### 1.1 Deployment

- **CloudFormation:** [infrastructure/aws/cloudformation.yaml](../infrastructure/aws/cloudformation.yaml)
- **Free Tier:** [infrastructure/aws/cloudformation-free-tier.yaml](../infrastructure/aws/cloudformation-free-tier.yaml)

### 1.2 AWS Bedrock Agent (Roadmap)

CLS++ can serve as a **knowledge base** or **memory backend** for Bedrock Agents:

- Implement `Retrieve` and `RetrieveAndGenerate` API compatibility
- Adapter: [integrations/aws/bedrock_adapter.py](../integrations/aws/bedrock_adapter.py)

### 1.3 AWS Marketplace Listing

1. **APN (AWS Partner Network)** — Join APN, complete MCP (Marketplace Customer Program)
2. **Product** — Create SaaS product, define dimensions (writes, reads)
3. **Metering** — Use `GET /v1/usage` or AWS Marketplace Metering API
4. **Documentation** — Provide deployment guide, API docs

---

## 2. Azure Integration

### 2.1 Deployment

- **ARM Template:** [infrastructure/azure/arm-template.json](../infrastructure/azure/arm-template.json)
- **Parameters:** [infrastructure/azure/parameters.json](../infrastructure/azure/parameters.json)

### 2.2 Azure AI Agent (Roadmap)

CLS++ as memory for Azure AI Agent / Copilot Studio:

- REST API compatible with Azure AI connectors
- Adapter: [integrations/azure/ai_agent_adapter.py](../integrations/azure/ai_agent_adapter.py)

### 2.3 Azure Marketplace Listing

1. **Partner Center** — Create account, verify
2. **Transactable Offer** — SaaS offer type
3. **Metering** — Azure Marketplace Metering API (usage events)
4. **Landing Page** — Deployment instructions, support

---

## 3. GCP Integration

### 3.1 Deployment

- **Cloud Run:** [infrastructure/gcp/cloud-run.yaml](../infrastructure/gcp/cloud-run.yaml)
- **Terraform:** [infrastructure/gcp/terraform/](../infrastructure/gcp/terraform/)

### 3.2 Vertex AI (Roadmap)

CLS++ as memory for Vertex AI Agent Builder:

- REST API compatible
- Adapter: [integrations/gcp/vertex_adapter.py](../integrations/gcp/vertex_adapter.py)

### 3.3 GCP Marketplace Listing

1. **Google Cloud Partner Advantage** — Enroll
2. **Private / Public Listing** — Define product, pricing
3. **Billing** — Usage-based via Cloud Billing
4. **Documentation** — Deployment, API reference

---

## 4. OCI (Oracle Cloud) Integration

### 4.1 Deployment

- **Terraform:** [infrastructure/oci/terraform/](../infrastructure/oci/terraform/)
- **Resource Manager:** Stack for one-click deploy

### 4.2 OCI Marketplace Listing

1. **Oracle Partner Network** — OPN membership
2. **Marketplace Listing** — Submit stack, documentation
3. **Metering** — OCI usage reporting
4. **Support** — Define support tiers

---

## 5. Integration Adapters

Adapters provide cloud-specific interfaces while calling the same CLS++ API:

```
integrations/
├── aws/
│   ├── __init__.py
│   ├── bedrock_adapter.py    # Bedrock Agent knowledge base
│   └── lambda_handler.py     # Lambda → CLS++ proxy
├── azure/
│   ├── __init__.py
│   └── ai_agent_adapter.py   # Azure AI Agent connector
├── gcp/
│   ├── __init__.py
│   └── vertex_adapter.py     # Vertex AI Agent
└── oci/
    ├── __init__.py
    └── oci_adapter.py        # OCI Functions / API Gateway
```

---

## 6. Usage Metering for Marketplaces

All marketplaces require **usage metering** for pay-per-use billing.

### CLS++ Metering

- **Endpoint:** `GET /v1/usage`
- **Response:** `{ "writes": N, "reads": N, "period": "YYYY-MM" }`
- **Enable:** `CLS_TRACK_USAGE=true`

### Marketplace-Specific Hooks

| Marketplace | Integration |
|-------------|-------------|
| AWS | Call `meter_usage` (Metering API) from usage middleware |
| Azure | Call Marketplace Metering API (usage event) |
| GCP | Report to Cloud Billing |
| OCI | OCI usage reporting |

---

## 7. Checklist: Marketplace Readiness

- [ ] API key auth (`CLS_REQUIRE_API_KEY=true`)
- [ ] Rate limiting
- [ ] Usage tracking (`CLS_TRACK_USAGE=true`)
- [ ] Health endpoint (`/v1/health/score`)
- [ ] Documentation (API, deployment)
- [ ] Terms of Service, Privacy Policy
- [ ] Support channel (email, ticket)
- [ ] Partner program enrollment (APN, Partner Center, etc.)

---

**AlphaForge AI Labs** | [CLS++ GitHub](https://github.com/rajamohan1950/CLSplusplus)
