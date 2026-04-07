#!/bin/bash
# CLS++ Memory Recall Hook — SessionStart
# Fetches relevant memories from the CLS++ API and injects them into the session.
# Requires: CLS_API_KEY environment variable set to a valid API key.

set -euo pipefail

# Skip if no API key configured
if [ -z "${CLS_API_KEY:-}" ]; then
  exit 0
fi

API_URL="${CLS_API_URL:-https://www.clsplusplus.com}"

# Read stdin (session info JSON) — we don't need it but must consume it
cat > /dev/null

# Fetch recent development memories
RESPONSE=$(curl -s --max-time 8 \
  -X POST "${API_URL}/v1/memory/read" \
  -H "Authorization: Bearer ${CLS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Claude Code development session context, recent work, decisions, and status",
    "namespace": "claude-code",
    "limit": 15
  }' 2>/dev/null) || exit 0

# Extract memory texts — skip if empty or error
ITEMS=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    items = data.get('items', [])
    if not items:
        sys.exit(0)
    print('## CLS++ Memory Recall (from previous sessions)')
    print()
    for item in items:
        text = item.get('text', '').strip()
        ts = item.get('timestamp', '')[:10]
        source = item.get('source', '')
        if text:
            prefix = f'[{ts}]' if ts else ''
            print(f'- {prefix} {text}')
    print()
    print('Use these memories as context. They may be outdated — verify against current code if acting on them.')
except Exception:
    sys.exit(0)
" 2>/dev/null) || exit 0

if [ -n "$ITEMS" ]; then
  echo "$ITEMS"
fi

exit 0
