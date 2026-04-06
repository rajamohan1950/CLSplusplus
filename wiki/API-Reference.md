# API Reference

Base URL: `https://www.clsplusplus.com` (or `http://localhost:8080` for local)

---

## Authentication

When `CLS_REQUIRE_API_KEY=true`:

```
Authorization: Bearer cls_live_xxxxxxxxxxxxxxxxxxxxxxxx
```

**Key format:** `cls_live_*` or `cls_test_*` (min 32 chars)

**Public paths:** `/`, `/health`, `/v1/memory/health`, `/docs`, `/redoc`

---

## Endpoints

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

## Write

**POST** `/v1/memory/write`

```json
{
  "text": "User prefers dark mode",
  "namespace": "user:123",
  "source": "user",
  "salience": 0.5,
  "authority": 0.5,
  "metadata": {}
}
```

**Response:** `{ "id": "...", "store_level": "L1", "text": "..." }`

---

## Read

**POST** `/v1/memory/read`

```json
{
  "query": "user preferences",
  "namespace": "user:123",
  "limit": 10,
  "min_confidence": 0.5
}
```

**Response:** `{ "items": [...], "query": "...", "namespace": "..." }`

---

## Forget (RTBF)

**DELETE** `/v1/memory/forget`

```json
{
  "item_id": "550e8400-e29b-41d4-a716-446655440000",
  "namespace": "user:123"
}
```

---

## Interactive Docs

- **Swagger UI:** https://www.clsplusplus.com/docs
- **ReDoc:** https://www.clsplusplus.com/redoc

---

[← Architecture](Architecture) | [Deployment Guide →](Deployment-Guide)
