# SaaS & Pricing

CLS++ can run as **Memory-as-a-Service** with API key auth and rate limiting.

---

## Enabling SaaS Mode

```bash
export CLS_API_KEYS=cls_live_xxxxxxxxxxxxxxxxxxxxxxxx
export CLS_REQUIRE_API_KEY=true
export CLS_RATE_LIMIT_REQUESTS=100
export CLS_RATE_LIMIT_WINDOW_SECONDS=60
```

---

## Product Endpoints

| Endpoint | Maps to |
|----------|---------|
| `POST /v1/memories/encode` | `POST /v1/memory/write` |
| `POST /v1/memories/retrieve` | `POST /v1/memory/read` |
| `DELETE /v1/memories/forget` | `DELETE /v1/memory/forget` |
| `POST /v1/memories/consolidate` | `POST /v1/memory/sleep` |
| `GET /v1/memories/knowledge` | Read L2/L3 only |
| `GET /v1/health/score` | `GET /v1/memory/health` |

---

## Pricing Model (Draft)

| Tier | Writes/mo | Reads/mo | Price |
|------|-----------|----------|-------|
| Free | 1,000 | 5,000 | $0 |
| Pro | 50,000 | 250,000 | $49/mo |
| Team | 500,000 | 2.5M | $199/mo |
| Enterprise | Custom | Custom | Contact |

---

## Rate Limits

- **429** when exceeded
- `Retry-After` header
- `X-RateLimit-Remaining`, `X-RateLimit-Limit` headers

---

## Right to be Forgotten (RTBF)

`DELETE /v1/memories/forget` with `item_id` and `namespace` removes the memory from all stores.

---

[← Integration Examples](Integration-Examples) | [Contributing →](Contributing)
