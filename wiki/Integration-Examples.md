# Integration Examples

---

## Python (SDK)

```python
from clsplusplus.client import CLSClient

with CLSClient("https://www.clsplusplus.com", api_key="cls_live_xxx") as client:
    client.write("User prefers dark mode", namespace="user:123")
    results = client.read("user preferences", namespace="user:123")
    for item in results.items:
        print(item.text, item.confidence)
    client.forget("item-id", namespace="user:123")
```

---

## Python (Augmenting LLM Prompts)

```python
# Before calling your LLM:
memories = client.read("user preferences and context", namespace="user:123")
context = "\n".join(m.text for m in memories.items)
prompt = f"Context from memory:\n{context}\n\nUser: {user_message}"
response = llm.complete(prompt)
```

---

## JavaScript / fetch

```javascript
const API = "https://www.clsplusplus.com";
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

---

## curl

```bash
# Write
curl -X POST https://www.clsplusplus.com/v1/memory/write \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cls_live_xxx" \
  -d '{"text": "User prefers dark mode", "namespace": "user:123"}'

# Read
curl -X POST https://www.clsplusplus.com/v1/memory/read \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cls_live_xxx" \
  -d '{"query": "user preferences", "namespace": "user:123"}'
```

---

## Future Integrations (Roadmap)

| Integration | Priority |
|-------------|----------|
| LangChain | P0 |
| LangGraph | P0 |
| Vercel AI SDK | P1 |
| CrewAI / AutoGen | P1 |
| AWS Bedrock Agent | P2 |
| Azure AI Agent | P2 |

---

[← Deployment Guide](Deployment-Guide) | [SaaS & Pricing →](SaaS-and-Pricing)
