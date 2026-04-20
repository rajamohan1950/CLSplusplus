#!/bin/bash
# CLS++ Memory Recall — SessionStart hook
# Fetches relevant memories from previous sessions and injects them as context.

set -euo pipefail

if [ -z "${CLS_API_KEY:-}" ]; then
  exit 0
fi

# Consume stdin
cat > /dev/null

python3 << 'PYEOF'
import json, os, sys, urllib.request

API_URL = "https://www.clsplusplus.com"
API_KEY = os.environ["CLS_API_KEY"]

payload = json.dumps({
    "query": "Claude Code development session context, recent work, decisions, and status",
    "namespace": "claude-code",
    "limit": 15,
}).encode()

req = urllib.request.Request(
    f"{API_URL}/v1/memory/read",
    data=payload,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    resp = urllib.request.urlopen(req, timeout=8)
    data = json.loads(resp.read())
except Exception:
    sys.exit(0)

items = data.get("items", [])
if not items:
    sys.exit(0)

print("## CLS++ Memory Recall (from previous sessions)")
print()
for item in items:
    text = item.get("text", "").strip()
    ts = item.get("timestamp", "")[:10]
    if text:
        prefix = f"[{ts}]" if ts else ""
        print(f"- {prefix} {text}")
print()
print("Use these memories as context. They may be outdated -- verify against current code if acting on them.")

PYEOF

exit 0
