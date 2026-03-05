# CLS++ API Documentation

Brain-inspired, model-agnostic persistent memory for LLMs. Store, retrieve, and manage context across any model.

**Base URL:** `https://clsplusplus-api.onrender.com` (or `http://localhost:8080` for local)

---

## Quick Start

### 1. Install the Python SDK

```bash
pip install clsplusplus
```

### 2. Write and read in 3 lines

```python
from clsplusplus.client import CLSClient

with CLSClient("https://clsplusplus-api.onrender.com", api_key="cls_live_xxx") as client:
    client.write("User prefers dark mode", namespace="user:123")
    results = client.read("user preferences", namespace="user:123")
    for item in results.items:
        print(item.text, item.confidence)
```

### 3. Or use curl

```bash
# Write
curl -X POST https://clsplusplus-api.onrender.com/v1/memory/write \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cls_live_xxx" \
  -d '{"text": "User prefers dark mode", "namespace": "user:123"}'

# Read
curl -X POST https://clsplusplus-api.onrender.com/v1/memory/read \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cls_live_xxx" \
  -d '{"query": "user preferences", "namespace": "user:123"}'
```

---

## Authentication

When `CLS_REQUIRE_API_KEY=true`, include your API key in every request:

```
Authorization: Bearer cls_live_xxxxxxxxxxxxxxxxxxxxxxxx
```

**Key format:** `cls_live_*` or `cls_test_*` (min 32 chars). Generate with:

```bash
echo "cls_live_$(openssl rand -hex 24)"
```

**Public paths** (no auth): `/`, `/health`, `/v1/memory/health`, `/docs`, `/redoc`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/memory/write` | Store a memory |
| POST | `/v1/memories/encode` | Alias: encode memory |
| POST | `/v1/memory/read` | Retrieve by semantic query |
| POST | `/v1/memories/retrieve` | Alias: retrieve memories |
| GET | `/v1/memory/item/{item_id}` | Get item by ID |
| DELETE | `/v1/memory/forget` | Delete memory (RTBF) |
| DELETE | `/v1/memories/forget` | Alias: forget |
| POST | `/v1/memory/sleep` | Trigger consolidation |
| POST | `/v1/memories/consolidate` | Alias: consolidate |
| GET | `/v1/memories/knowledge` | Query L2/L3 knowledge only |
| GET | `/v1/memory/health` | Health check |
| GET | `/v1/health/score` | Alias: health |

---

## Write / Encode

Store a new memory. Flows to L0 (working buffer) and L1 (episodic store).

**POST** `/v1/memory/write`

### Request body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | Memory content (max 64KB) |
| namespace | string | No | Tenant/scope (default: "default") |
| source | string | No | Source identifier (default: "user") |
| salience | float | No | 0–1 (default: 0.5) |
| authority | float | No | 0–1 (default: 0.5) |
| metadata | object | No | Custom key-value pairs |

### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "store_level": "L1",
  "text": "User prefers dark mode"
}
```

---

## Read / Retrieve

Retrieve memories by semantic similarity. Searches across L0–L3 stores.

**POST** `/v1/memory/read`

### Request body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Semantic query (max 4KB) |
| namespace | string | No | Scope (default: "default") |
| limit | int | No | Max results (1–100, default: 10) |
| min_confidence | float | No | Filter by confidence (0–1) |

### Response

```json
{
  "items": [
    {
      "id": "...",
      "text": "User prefers dark mode",
      "confidence": 0.92,
      "store_level": "L1",
      "timestamp": "2026-03-04T12:00:00Z"
    }
  ],
  "query": "user preferences",
  "namespace": "user:123"
}
```

**Knowledge query (L2/L3 only):** `GET /v1/memories/knowledge?query=...&namespace=...`

---

## Forget

Delete a memory by ID (Right to be Forgotten).

**DELETE** `/v1/memory/forget`

### Request body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| item_id | string | Yes | Memory ID |
| namespace | string | No | Scope (default: "default") |

### Response

```json
{"deleted": true, "item_id": "550e8400-e29b-41d4-a716-446655440000"}
```

---

## Integration Examples

### Python (sync)

```python
from clsplusplus.client import CLSClient

client = CLSClient("https://clsplusplus-api.onrender.com", api_key="cls_live_xxx")
client.write("User prefers dark mode", namespace="user:123")
results = client.read("preferences", namespace="user:123")
client.forget("item-id", namespace="user:123")
client.close()
```

### Python (context manager)

```python
with CLSClient("https://clsplusplus-api.onrender.com", api_key="cls_live_xxx") as c:
    c.write("Fact: project X uses Python")
    items = c.read("project tech stack")
```

### JavaScript / fetch

```javascript
const API = "https://clsplusplus-api.onrender.com";
const KEY = "cls_live_xxx";

// Write
await fetch(`${API}/v1/memory/write`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${KEY}`,
  },
  body: JSON.stringify({ text: "User prefers dark mode", namespace: "user:123" }),
});

// Read
const res = await fetch(`${API}/v1/memory/read`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${KEY}`,
  },
  body: JSON.stringify({ query: "preferences", namespace: "user:123" }),
});
const data = await res.json();
console.log(data.items);
```

### Augmenting LLM prompts

```python
# Before calling your LLM:
memories = client.read("user preferences and context", namespace="user:123")
context = "\n".join(m.text for m in memories.items)
prompt = f"Context from memory:\n{context}\n\nUser: {user_message}"
response = llm.complete(prompt)
```

---

## Error Handling

| Code | Meaning |
|------|---------|
| 400 | Bad request — invalid input |
| 401 | Unauthorized — missing or invalid API key |
| 404 | Not found — item doesn't exist |
| 422 | Validation error — request body fails schema |
| 429 | Rate limit exceeded — retry after `Retry-After` seconds |
| 500 | Server error — check health endpoint |

---

## Interactive Docs

- **Swagger UI:** https://clsplusplus-api.onrender.com/docs
- **ReDoc:** https://clsplusplus-api.onrender.com/redoc
