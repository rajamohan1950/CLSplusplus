#!/bin/bash
# Quick verify: all 3 LLMs respond. Requires .env with real keys.
set -e
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "Create .env with API keys (see .env.example)"; exit 1; }
export $(grep -v '^#' .env | xargs)

echo "Starting API..."
PYTHONPATH=src python -m uvicorn clsplusplus.demo_local:app --host 127.0.0.1 --port 8080 &
PID=$!
trap "kill $PID 2>/dev/null" EXIT
sleep 3

echo ""
echo "Testing Claude..."
R=$(curl -s -X POST http://localhost:8080/v1/demo/chat -H "Content-Type: application/json" -d '{"model":"claude","message":"Say hi in 3 words","namespace":"v"}')
echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  OK' if 'reply' in d and 'Add ' not in d.get('reply','') and 'error' not in d.get('reply','').lower() else '  FAIL: '+d.get('reply','')[:80])"

echo "Testing OpenAI..."
R=$(curl -s -X POST http://localhost:8080/v1/demo/chat -H "Content-Type: application/json" -d '{"model":"openai","message":"Say hi in 3 words","namespace":"v"}')
echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  OK' if 'reply' in d and 'Add ' not in d.get('reply','') and 'error' not in d.get('reply','').lower() else '  FAIL: '+d.get('reply','')[:80])"

echo "Testing Gemini..."
R=$(curl -s -X POST http://localhost:8080/v1/demo/chat -H "Content-Type: application/json" -d '{"model":"gemini","message":"Say hi in 3 words","namespace":"v"}')
echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  OK' if 'reply' in d and 'Add ' not in d.get('reply','') and 'error' not in d.get('reply','').lower() else '  FAIL: '+d.get('reply','')[:80])"

echo ""
echo "If all OK: run ./scripts/run_local_demo.sh and test criss-cross in browser."
