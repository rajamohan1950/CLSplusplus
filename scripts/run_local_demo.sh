#!/bin/bash
# Run demo locally - real Claude, OpenAI, Gemini. No Redis/Postgres.
# 1. cp .env.example .env and add your API keys
# 2. ./scripts/run_local_demo.sh

set -e
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "Create .env from .env.example and add your REAL API keys:"
  echo "  cp .env.example .env"
  echo "  # Edit .env with real keys from Anthropic, OpenAI, Google AI Studio"
  exit 1
fi

echo "Installing deps if needed..."
pip install -q anthropic openai google-generativeai uvicorn pydantic-settings 2>/dev/null || true

export $(grep -v '^#' .env | xargs)

echo "Demo API: http://localhost:8080"
echo "Website:  http://localhost:3000?local=1"
echo ""

cleanup() { kill $API_PID $WEB_PID 2>/dev/null; exit 0; }
trap cleanup SIGINT SIGTERM

echo "Starting API..."
PYTHONPATH=src uvicorn clsplusplus.demo_local:app --host 0.0.0.0 --port 8080 &
API_PID=$!
sleep 2
echo "Starting website..."
python3 -m http.server 3000 --directory website &
WEB_PID=$!
echo ""
echo ">>> Open: http://localhost:3000?local=1 <<<"
echo "Press Ctrl+C to stop"
wait
