---
name: cls-memory
description: Check CLS++ memory connection status and manually recall or store memories
user_invocable: true
---

# CLS++ Memory Status and Control

Check the connection to CLS++ persistent memory and perform manual memory operations.

## Instructions

1. Check if the `CLS_API_KEY` environment variable is set by running: `echo $CLS_API_KEY | head -c 15`
   - If empty, tell the user: "Set your CLS++ API key: `export CLS_API_KEY=cls_live_xxxxx` — get your key at clsplusplus.com/profile.html"
   - If set, continue to step 2

2. Test the connection by running:
   ```bash
   curl -s --max-time 5 -X POST "https://www.clsplusplus.com/v1/memory/read" \
     -H "Authorization: Bearer $CLS_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "connection test", "namespace": "claude-code", "limit": 1}'
   ```

3. Report the status:
   - If the response contains `"items"`, the connection is working. Show the number of memories stored.
   - If 401, the API key is invalid. Tell the user to check their key at clsplusplus.com/profile.html
   - If connection fails, the API may be down. Suggest trying again in a moment.

4. If the user asked to recall memories, fetch them:
   ```bash
   curl -s --max-time 8 -X POST "https://www.clsplusplus.com/v1/memory/read" \
     -H "Authorization: Bearer $CLS_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "Claude Code development session context", "namespace": "claude-code", "limit": 15}'
   ```
   Display the recalled memories as a bulleted list.

5. If the user asked to store something, write it:
   ```bash
   curl -s --max-time 8 -X POST "https://www.clsplusplus.com/v1/memory/write" \
     -H "Authorization: Bearer $CLS_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"text": "<user's text>", "namespace": "claude-code", "source": "manual", "salience": 0.8}'
   ```
   Confirm the memory was stored.
